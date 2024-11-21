/** \file   interface_nemo.cpp
    \author Eugene Vasiliev
    \date   2014-2024
    \brief  Wrapper for AGAMA potentials to be used in NEMO

    The shared library agama.so can be used as a plugin in the NEMO
    stellar-dynamical toolbox, providing external potential that can
    be used, in particular, in the N-body code gyrfalcON.
    The potential must be specified as an INI file with all
    parameters (possibly for a multi-component potential),
    possibly including potential expansion coefficients.
    There are two optional additional parameters:
    - Pattern speed Omega (default is 0);
      a nonzero value makes the potential rotate about the z axis,
      but a more powerful way of introducing rotation (e.g. with
      a non-uniform pattern speed or nonzero initial angle)
      is through the "rotation" modifier in the INI file.
    - Gravitational constant G (default is 1);
      if gyrfalcON was invoked with a non-default value Grav!=1,
      then one needs to duplicate it in the parameters of the external
      potential, since the plugin has no access to gyrfalcON's G.
      The mass- or density-related parameters of the potential specified
      in the INI file are then multiplied by G.
    The usage is as follows: in gyrfalcON, for instance, one adds
    `accname=agama accfile=params.ini [accpars=Omega[,G]]`
    to the command line arguments, and provides all necessary parameters.
    in the params.ini text file (see the reference documentation for
    explanation of the options).
    The ini file may contain multiple components and/or time-dependent
    features (such as center offset, acceleration, or an evolving sequence
    of potentials), which are fully supported by the NEMO interface.

    This file exports a few routines that make possible to load
    the shared library agama.so without any further modifications;
    NEMO framework is not needed for compilation (if it's not present,
    these few extra routines just add a little ballast to the library).
**/
#include "potential_factory.h"
#include "math_core.h"
#include <cstdio>
#include <cstdlib>
//#include <stdexcept>

namespace{

/// pointer to the function computing acceleration
typedef void(*acc_pter)(
    int NDIM,           /* input:  number of dimensions            */
    double t,           /* input:  simulation time                 */
    int nbody,          /* input:  number bodies = size of arrays  */
    const void* m,      /* input:  masses:         m[i]            */
    const void* pos,    /* input:  positions       (x,y,z)[i]      */
    const void* vel,    /* input:  velocities      (u,v,w)[i]      */
    const int * flags,  /* input:  flags           f[i]            */
    void* pot,          /* output: potentials      p[i]            */
    void* acc,          /* output: accelerations   (ax,ay,az)[i]   */
    int indicator,      /* input:  indicator                       */
    char numtype);      /* input:  type: 'f' or 'd'                */


/// pointer to the actual potential (a single instance)
static potential::PtrPotential mypot;
static double Omega = 0;

/// add or assign a number
template<typename NumT, int Add> void op(NumT x, NumT& y);
template<> inline void op<float, 0>(float  x, float& y) { y = x; }
template<> inline void op<float, 1>(float  x, float& y) { y+= x; }
template<> inline void op<double,0>(double x, double&y) { y = x; }
template<> inline void op<double,1>(double x, double&y) { y+= x; }

/// loop over active particles and compute potential and acceleration
template<typename NumT, int NDIM, int AddPot, int AddAcc>
inline void getAcc(double time, int nbody, const int* active, const void* _pos, void* _pot, void* _acc)
{
    const NumT* pos = static_cast<const NumT*>(_pos);
    NumT* pot = static_cast<NumT*>(_pot);
    NumT* acc = static_cast<NumT*>(_acc);
    double cosphi=1, sinphi=0;  // rotation angle of the potential reference frame at the current time
    math::sincos(Omega*time, sinphi, cosphi);

    for(int i=0; i<nbody; i++) {
        if(active && (active[i] & 1) != 1)
            continue;   // skip particles which are not marked as active
        coord::PosCar point(
            pos[i*NDIM + 0] * cosphi + pos[i*NDIM + 1] * sinphi,
            pos[i*NDIM + 1] * cosphi - pos[i*NDIM + 0] * sinphi,
            NDIM==3 ? pos[i*NDIM + 2] : 0);
        coord::GradCar grad;
        double Phi;
        mypot->eval(point, &Phi, &grad, NULL, time);
        // potential and acceleration are either added or assigned, depending on the flags
        op<NumT,AddPot>( static_cast<NumT>(Phi), pot[i]);
        op<NumT,AddAcc>(-static_cast<NumT>(grad.dx * cosphi - grad.dy * sinphi), acc[i*NDIM + 0]);
        op<NumT,AddAcc>(-static_cast<NumT>(grad.dy * cosphi + grad.dx * sinphi), acc[i*NDIM + 1]);
        if(NDIM==3) op<NumT,AddAcc>(-static_cast<NumT>(grad.dz), acc[i*NDIM + 2]);
    }
}

/// partial specialization depending on the runtime value of flag
template<typename NumT, int NDIM>
inline void getAcc(double time, int nbody, const int* active, const void* pos, void* pot, void* acc, int flag)
{
    switch(flag & 3) {
        case 0: getAcc<NumT, NDIM, 0, 0>(time, nbody, active, pos, pot, acc); break;
        case 1: getAcc<NumT, NDIM, 1, 0>(time, nbody, active, pos, pot, acc); break;
        case 2: getAcc<NumT, NDIM, 0, 1>(time, nbody, active, pos, pot, acc); break;
        case 3: getAcc<NumT, NDIM, 1, 1>(time, nbody, active, pos, pot, acc); break;
        default: ;
    }
}

/// the actual function called from NEMO
static void myacc(
    int NDIM,           /* input:  number of dimensions (2 or 3)   */
    double time,        /* input:  simulation time                 */
    int nbody,          /* input:  number bodies = size of arrays  */
    const void* /*mas*/,/* input:  masses:         m[i]            */
    const void* pos,    /* input:  positions       (x,y,z)[i]      */
    const void* /*vel*/,/* input:  velocities      (u,v,w)[i]      */
    const int * active, /* input:  active flag     f[i]            */
    void* pot,          /* output: potentials      p[i]            */
    void* acc,          /* output: accelerations   (ax,ay,az)[i]   */
    int flag,           /* input:  operation flag:
    If bit 0 is set, the potential    is added, otherwise assigned,
    If bit 1 is set, the acceleration is added, otherwise assigned.*/
    char numtype)       /* input:  type: 'f' or 'd'                */
{
    // choose an appropriate statically compiled template instantiation depending on runtime parameters
    if(numtype=='f') {
        if(NDIM==2)
            getAcc<float, 2>(time, nbody, active, pos, pot, acc, flag);
        else if(NDIM==3)
            getAcc<float, 3>(time, nbody, active, pos, pot, acc, flag);
        //else throw std::runtime_error("Invalid NDIM");
    } else if(numtype=='d') {
        if(NDIM==2)
            getAcc<double, 2>(time, nbody, active, pos, pot, acc, flag);
        else if(NDIM==3)
            getAcc<double, 3>(time, nbody, active, pos, pot, acc, flag);
        //else throw std::runtime_error("Invalid NDIM");
    } //else   throw std::runtime_error("Invalid NumT");
}

}  // end internal namespace

// external interface: routines that can be called from NEMO

/// compute potential and acceleration at a single point (double version)
extern "C" void potential_double(
    const int   *ndim,
    const double*pos,
    double      *acc,
    double      *pot,
    const double*time)
{
    if(*ndim==2)
        getAcc<double, 2, 0, 0>(*time, 1, NULL, pos, pot, acc);
    else if(*ndim==3)
        getAcc<double, 3, 0, 0>(*time, 1, NULL, pos, pot, acc);
    //else throw std::runtime_error("Invalid NDIM");
}

/// compute potential and acceleration at a single point (float version)
extern "C" void potential_float(
    const int   *ndim,
    const float *pos,
    float       *acc,
    float       *pot,
    const float *time)
{
    if(*ndim==2)
        getAcc<float, 2, 0, 0>(*time, 1, NULL, pos, pot, acc);
    else if(*ndim==3)
        getAcc<float, 3, 0, 0>(*time, 1, NULL, pos, pot, acc);
    //else throw std::runtime_error("Invalid NDIM");
}

/// install the potential (older interface)
extern "C" void inipotential(
    const int   *npar,
    const double*pars,
    const char  *file)
{
    if(file==NULL || file[0]==0) {
        fprintf(stderr, "Agama plugin for NEMO: "
            "Should provide the name of INI or potential coefficients file in accfile=...");
        std::exit(1);
    }
    Omega = *npar>=1 ? pars[0] : 0;
    double G = *npar>=2 ? pars[1] : 1;
    if(G==1)
        mypot = potential::readPotential(file);
    else {
        const units::InternalUnits unit(units::Kpc, units::Kpc/units::kms);
        mypot = potential::readPotential(file,
            units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun * unit.to_Msun * G));
    }
    fprintf(stderr, "Agama plugin for NEMO: Created an instance of %s potential", mypot->name().c_str());
    if(Omega != 0)
        fprintf(stderr, " with pattern speed %g", Omega);
    if(G != 1)
        fprintf(stderr, " with G=%g", G);
    fprintf(stderr, "\n");
}

/// install the potential (newer interface)
extern "C" void iniacceleration(
    const double*pars,
    int          npar,
    const char  *file,
    acc_pter    *accel,
    bool        *need_m,
    bool        *need_v)
{
    inipotential(&npar, pars, file);
    *accel = myacc;
    if(need_m) *need_m = false;
    if(need_v) *need_v = false;
}
