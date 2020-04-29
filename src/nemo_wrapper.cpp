/** \file   nemo_wrapper.cpp
    \author Eugene Vasiliev
    \date   2014-2020
    \brief  Wrapper for AGAMA potentials to be used in NEMO

    The shared library agama.so can be used as a plugin in the NEMO
    stellar-dynamical toolbox, providing external potential that can
    be used, in particular, in the N-body code gyrfalcON.
    The potential must be specified either as an INI file with all
    parameters (possibly for a multi-component potential),
    or as a file with potential expansion coefficients.
    In both cases, the only free parameter is the pattern speed
    (Omega, optional, default is 0).
    The usage is as follows: in gyrfalcON, for instance, one adds
    `accname=agama accfile=params.ini [accpars=Omega]`
    to the command line arguments, and provides all necessary parameters.
    in the params.ini text file (see the reference documentation for
    explanation of the options); alternatively one may use
    `accname=agama accfile=my_potential_coefs [accpars=Omega]`
    The choice is determined by the file extesion (.ini vs any other).

    This file exports a few routines that make possible to load
    the shared library agama.so without any further modifications;
    NEMO framework is not needed for compilation (if it's not present,
    these few extra routines just add a little ballast to the library).
**/
#include "potential_factory.h"
#include "math_core.h"
#include "utils.h"
#include <cstdio>
#include <fstream>
#include <stdexcept>

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
/// pattern speed
static double Omega = 0;
/// time-dependent offsets of the potential center from origin
static math::CubicSpline centerx, centery, centerz;

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
    if(!mypot)
        throw std::runtime_error("Agama plugin for NEMO: potential is not initialized");
    const NumT* pos = static_cast<const NumT*>(_pos);
    NumT* pot = static_cast<NumT*>(_pot);
    NumT* acc = static_cast<NumT*>(_acc);
    double cosphi=1, sinphi=0,   // rotation angle of the potential reference frame at the current time
        offx=0, offy=0, offz=0,  // offset of the potential center
        accx=0, accy=0, accz=0;  // accelerations due to the motion of the potential center
    math::sincos(Omega*time, cosphi, sinphi);
    if(!centerx.empty()) {
        centerx.evalDeriv(time, &offx, NULL, &accx);
        centery.evalDeriv(time, &offy, NULL, &accy);
        centerz.evalDeriv(time, &offz, NULL, &accz);
    }
    for(int i=0; i<nbody; i++) {
        if(active && (active[i] & 1) != 1)
            continue;   // skip particles which are not marked as active
        coord::PosCar point(
            (pos[i*NDIM + 0] - offx) * cosphi + (pos[i*NDIM + 1] - offy) * sinphi,
            (pos[i*NDIM + 1] - offy) * cosphi - (pos[i*NDIM + 0] - offx) * sinphi,
            (NDIM==3 ? pos[i*NDIM + 2] : 0) - offz);
        coord::GradCar grad;
        double Phi;
        mypot->eval(point, &Phi, &grad);
        // potential and acceleration are either added or assigned, depending on the flags
        op<NumT,AddPot>( static_cast<NumT>(Phi), pot[i]);
        op<NumT,AddAcc>(-static_cast<NumT>(grad.dx * cosphi - grad.dy * sinphi - accx), acc[i*NDIM + 0]);
        op<NumT,AddAcc>(-static_cast<NumT>(grad.dy * cosphi + grad.dx * sinphi - accy), acc[i*NDIM + 1]);
        if(NDIM==3) op<NumT,AddAcc>(-static_cast<NumT>(grad.dz - accz), acc[i*NDIM + 2]);
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
        else throw std::runtime_error("Invalid NDIM");
    } else if(numtype=='d') {
        if(NDIM==2)
            getAcc<double, 2>(time, nbody, active, pos, pot, acc, flag);
        else if(NDIM==3)
            getAcc<double, 3>(time, nbody, active, pos, pot, acc, flag);
        else throw std::runtime_error("Invalid NDIM");
    } else   throw std::runtime_error("Invalid NumT");
}

static void readOrbit(const std::string& filename)
{
    std::ifstream strm(filename.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("Agama plugin for NEMO: cannot read from file "+filename);
    std::string buffer;
    std::vector<std::string> fields;
    std::vector<double> time, posx, posy, posz, velx, vely, velz;
    while(std::getline(strm, buffer) && !strm.eof()) {
        if(!buffer.empty() && utils::isComment(buffer[0]))  // commented line
            continue;
        fields = utils::splitString(buffer, "#;, \t");
        size_t numFields = fields.size();
        if(numFields < 4 ||
            !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
            continue;
        time.push_back(utils::toDouble(fields[0]));
        posx.push_back(utils::toDouble(fields[1]));
        posy.push_back(utils::toDouble(fields[2]));
        posz.push_back(utils::toDouble(fields[3]));
        if(numFields >= 7) {
            velx.push_back(utils::toDouble(fields[4]));
            vely.push_back(utils::toDouble(fields[5]));
            velz.push_back(utils::toDouble(fields[6]));
        }
    }
    if(velx.size() == posx.size()) {
        // create Hermite splines from values and derivatives at each moment of time
        centerx = math::CubicSpline(time, posx, velx);
        centery = math::CubicSpline(time, posy, vely);
        centerz = math::CubicSpline(time, posz, velz);
    } else {
        // create natural cubic splines from just the values
        centerx = math::CubicSpline(time, posx);
        centery = math::CubicSpline(time, posy);
        centerz = math::CubicSpline(time, posz);
    }
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
    else throw std::runtime_error("Invalid NDIM");
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
    else throw std::runtime_error("Invalid NDIM");
}

/// install the potential (older interface)
extern "C" void inipotential(
    const int   *npar,
    const double*pars,
    const char  *file)
{
    if(file==NULL || file[0]==0)
        throw std::runtime_error("Agama plugin for NEMO: "
            "Should provide the name of INI or potential coefficients file in accfile=...");
    const std::string filename(file);
    if(filename.size()>4 && filename.substr(filename.size()-4)==".ini") {
        mypot = potential::createPotential(filename);
        std::string centerfile = utils::ConfigFile(filename).findSection("center").getString("file");
        if(!centerfile.empty())
            readOrbit(centerfile);
    } else
        mypot = potential::readPotential(filename);
    Omega = *npar>=1 ? pars[0] : 0;
    fprintf(stderr, "Agama plugin for NEMO: "
        "Created an instance of %s potential with pattern speed=%g\n", mypot->name(), Omega);
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
