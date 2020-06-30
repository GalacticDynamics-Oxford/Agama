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

    There are a number of features that make it possible to use
    a time-dependent potential. They are specified as additional sections
    in the .ini file.
    First, the potential itself may be shifted from origin by a constant
    or a time-dependent offset vector. This is specified by the section
    [center], with either three numbers x=..., y=..., z=... for a constant
    offset, or a text file given by file=... with four columns: time,
    x, y, z  for a time-dependent offset vector.
    Second, one may add a spatially uniform time-dependent acceleration 
    field given in a text file, specified by file=... parameter in
    a section [acceleration]. (Again the file should have 4 columns).
    Finally, the potential itself needs not be the same at all times.
    To initialize a time-dependent potential, add a section [time-dependent]
    to the ini file, with a single parameter file=...;
    this index file should have two columns - time and the name of a file 
    with potential coefficients at the given time.
    Alternatively, one may provide the name of the index file as the
    argument accfile=... in the command line, bypassing the creation of
    a separate ini file.
    Forces from a time-dependent potential are linearly interpolated in time,
    while the offsets and accelerations are cubically interpolated.

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

/// array of time stamps for a time-dependent potential
static std::vector<double> times;
/// array of potentials corresponding to each moment of time (possibly just one)
static std::vector<potential::PtrPotential> mypot;
/// pattern speed
static double Omega = 0;
/// time-dependent offsets of the potential center from origin
static math::CubicSpline centerx, centery, centerz;
/// time-dependent uniform accelerations
static math::CubicSpline accelx, accely, accelz;

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
    if(times.empty() || mypot.empty())
        throw std::runtime_error("Agama plugin for NEMO: potential is not initialized");
    const NumT* pos = static_cast<const NumT*>(_pos);
    NumT* pot = static_cast<NumT*>(_pot);
    NumT* acc = static_cast<NumT*>(_acc);
    // coordinate offset and rotation
    double cosphi=1, sinphi=0,   // rotation angle of the potential reference frame at the current time
        offx=0, offy=0, offz=0,  // offset of the potential center
        accx=0, accy=0, accz=0;  // external uniform acceleration field
    math::sincos(Omega*time, sinphi, cosphi);
    if(!centerx.empty()) {
        offx = centerx(time);
        offy = centery(time);
        offz = centerz(time);
    }
    if(!accelx.empty()) {
        accx = accelx(time);
        accy = accely(time);
        accz = accelz(time);
    }
    // determine which potential(s) to use in the time-dependent case,
    // linearly interpolating between the two timestamps enclosing the current time
    int index = 0;      // index of the leftmost (smaller of the two) timestamp
    double weight = 1.; // weigth of the potential associated with the leftmost timestamp
    if(times.size()>=2) {
        index = math::binSearch(time, &times.front(), times.size());
        if(index<0) {
            index = 0;
            weight = 1;
        } else if(index>=(int)times.size()-1) {
            index = times.size()-1;
            weight = 1;
        } else {
            weight = (times[index+1] - time) / (times[index+1] - times[index]);
        }
    }
    for(int i=0; i<nbody; i++) {
        if(active && (active[i] & 1) != 1)
            continue;   // skip particles which are not marked as active
        coord::PosCar point(
            (pos[i*NDIM + 0] - offx) * cosphi + (pos[i*NDIM + 1] - offy) * sinphi,
            (pos[i*NDIM + 1] - offy) * cosphi - (pos[i*NDIM + 0] - offx) * sinphi,
            (NDIM==3 ? pos[i*NDIM + 2] : 0) - offz);
        coord::GradCar grad, grad1;
        double Phi, Phi1;
        mypot[index]->eval(point, &Phi, &grad);
        // if needed, interpolate between two potentials
        if(weight<1) {
            mypot[index+1]->eval(point, &Phi1, &grad1);
            Phi = Phi * weight + Phi1 * (1-weight);
            grad.dx = grad.dx * weight + grad1.dx * (1-weight);
            grad.dy = grad.dy * weight + grad1.dy * (1-weight);
            grad.dz = grad.dz * weight + grad1.dz * (1-weight);
        }
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

static void readTimeDep(const std::string& filename,
    /*output*/ math::CubicSpline& splx, math::CubicSpline& sply, math::CubicSpline& splz)
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
        splx = math::CubicSpline(time, posx, velx);
        sply = math::CubicSpline(time, posy, vely);
        splz = math::CubicSpline(time, posz, velz);
    } else {
        // create natural cubic splines from just the values (and regularize)
        splx = math::CubicSpline(time, posx, true);
        sply = math::CubicSpline(time, posy, true);
        splz = math::CubicSpline(time, posz, true);
    }
}

static void readPotentials(const std::string& filename)
{
    std::ifstream strm(filename.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("Agama plugin for NEMO: cannot read from file "+filename);
    std::string buffer;
    std::vector<std::string> fields;
    // attempt to parse the file as if it contained time stamps and names of corresponding potential files
    try{
        while(std::getline(strm, buffer) && !strm.eof()) {
            if(!buffer.empty() && utils::isComment(buffer[0]))  // commented line
                continue;
            fields = utils::splitString(buffer, "#;, \t");
            size_t numFields = fields.size();
            if(numFields < 2 ||
               !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
                continue;
            if(!utils::fileExists(fields[1])) {
                fprintf(stderr, "file %s missing\n", fields[1].c_str());
                continue;
            }
            times.push_back(utils::toDouble(fields[0]));
            mypot.push_back(potential::readPotential(fields[1]));
            if(times.size()>=2 && !(times[times.size()-1] > times[times.size()-2]))
                throw std::runtime_error("Times are not monotonic");
        }
    }
    catch(std::exception&) {
        times.clear();
        mypot.clear();
    }
    strm.close();
    // if this didn't work, assume that the file contains a single potential expansion
    if(times.empty()) {
        times.assign(1, 0.);
        mypot.assign(1, potential::readPotential(filename));
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
        // parse the ini file and find out if it has any auxiliary sections apart from [Potential***]
        utils::ConfigFile cfg(filename);
        // [center]: provides the offset of the potential expansion from origin
        const utils::KeyValueMap& sec_center = cfg.findSection("center");
        if(sec_center.contains("file")) {
            // time-dependent offset
            readTimeDep(sec_center.getString("file"), /*output*/ centerx, centery, centerz);
        } else {
            // no file but perhaps values of x,y,z are provided (if not, just use zeros)
            std::vector<double> times(2); times[1] = 1;
            centerx = math::CubicSpline(times, std::vector<double>(2, sec_center.getDouble("x", 0)));
            centery = math::CubicSpline(times, std::vector<double>(2, sec_center.getDouble("y", 0)));
            centerz = math::CubicSpline(times, std::vector<double>(2, sec_center.getDouble("z", 0)));
        }
        // [acceleration]: provides an additional time-dependent uniform acceleration field
        std::string accelfile = cfg.findSection("acceleration").getString("file");
        if(!accelfile.empty())
            readTimeDep(accelfile, /*output*/ accelx, accely, accelz);
        // [time-dependent]: provides a list of files with time-dependent potentials
        std::string timedepfile = cfg.findSection("time-dependent").getString("file");
        if(!timedepfile.empty())
            readPotentials(timedepfile);
        else {
            // non-time-dependent case, just a single potential: use all [Potential***] sections in the ini file
            times.assign(1, 0.);
            mypot.assign(1, potential::createPotential(filename));
        }
    } else {
        readPotentials(filename);
    }
    // optional additional parameter: pattern speed
    if(*npar>=1) {
        Omega = pars[0];
        fprintf(stderr, "Agama plugin for NEMO: "
        "Created an instance of %s potential with pattern speed=%g\n", mypot[0]->name(), Omega);
    } else {
        Omega = 0;
        fprintf(stderr, "Agama plugin for NEMO: Created an instance of %s potential\n", mypot[0]->name());
    }
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
