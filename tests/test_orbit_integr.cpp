/** \file    test_orbit_integr.cpp
    \date    2015-2018
    \author  Eugene Vasiliev

    Test orbit integration in various potentials and coordinate systems.
*/
#include "orbit.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_dehnen.h"
#include "potential_factory.h"
#include "potential_ferrers.h"
#include "potential_utils.h"
#include "units.h"
#include "utils.h"
#include "debug_utils.h"
#include "math_random.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double eps=1e-5;     // accuracy of energy conservation
const double epsrot=5e-4;  // accuracy of comparison between inertial and rotating frames
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;
const double Omega=2.718;  // rotation frequency (arbitrary)

template<typename CoordT> bool isCartesian() { return false; }
template<> bool isCartesian<coord::Car>() { return true; }

template<typename CoordT>
bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelT<CoordT>& initial_conditions)
{
    double total_time = 10.0 * T_circ(potential, totalEnergy(potential, initial_conditions));
    if(!isFinite(total_time))
        total_time = 100.0;
    double timestep = 0.01999999999999 * total_time;
    // whether compare the result of orbit integration in rotating and inertial frames
    bool checkRot = isCartesian<CoordT>() && isAxisymmetric(potential);
    std::cout << potential.name()<<"  "<<CoordT::name()<<" ("<<initial_conditions<<
        "Torb: "<<T_circ(potential, totalEnergy(potential, initial_conditions))<<")\n";
    double ic[6];
    initial_conditions.unpack_to(ic);
    std::vector< std::pair<coord::PosVelT<CoordT>, double> > traj, trajFull;
    std::vector< std::pair<coord::PosVelCar, double> > trajRot;
    orbit::RuntimeFncArray fncs(2);
    // record the orbit at regular intervals of time
    fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<CoordT>(timestep, /*output*/ traj));
    // record the orbit at each timestep of the ODE integrator
    fncs[1] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<CoordT>(0, trajFull));
    orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegrator<CoordT>(potential), fncs,
        orbit::OrbitIntParams(/*accuracy*/ 1e-8, /*maxNumSteps*/10000));
    if(checkRot) {
        fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<coord::Car>(timestep, trajRot));
        fncs.resize(1);
        orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegratorRot(potential, Omega), fncs);
        if(trajRot.size() != traj.size()) {
            std::cout << "\033[1;34mRotating frame inconsistent\033[0m\n";
            return false;
        }
    }
    math::Averager avgE, avgL;
    bool okrot = true;
    for(size_t i=0; i<traj.size(); i++) {
        avgE.add(totalEnergy(potential, traj[i].first));
        avgL.add(Lz(traj[i].first));
        if(checkRot) {
            double angle = traj[i].second*Omega, cosa = cos(angle), sina = sin(angle);
            double rxv[6];
            trajRot[i].first.unpack_to(rxv);
            double pointRot[6] = {
                rxv[0] * cosa - rxv[1] * sina, rxv[1] * cosa + rxv[0] * sina, rxv[2],
                rxv[3] * cosa - rxv[4] * sina, rxv[4] * cosa + rxv[3] * sina, rxv[5] };
            if(Lz(traj[i].first)!=0)
                okrot &= equalPosVel(coord::PosVelT<CoordT>(pointRot), traj[i].first, epsrot);
        }
    }
    if(output) {
        std::ostringstream s;
        double x[6];
        initial_conditions.unpack_to(x);
        s<<"Orbit_"<<potential.name()<<'_'<<CoordT::name()<<'_'<<
            x[0]<<'_'<<x[1]<<'_'<<x[2]<<'_'<<x[3]<<'_'<<x[4]<<'_'<<x[5];
        std::ofstream strm(s.str().c_str());
        for(size_t i=0; i<trajFull.size(); i++) {
            double xv[6];
            coord::toPosVelCar(trajFull[i].first).unpack_to(xv);
            strm << utils::pp(trajFull[i].second, 18) <<"  " <<
                utils::pp(xv[0], 18) <<' '<< utils::pp(xv[1], 18) <<' '<< utils::pp(xv[2], 18) <<"  "<<
                utils::pp(xv[3], 18) <<' '<< utils::pp(xv[4], 18) <<' '<< utils::pp(xv[5], 18) <<"   "<<
                utils::pp(totalEnergy(potential, trajFull[i].first), 18) << ' '<<
                utils::pp(Lz(trajFull[i].first), 18) << '\n';
        }
    }
    bool completed = traj.back().second > 0.999999*total_time;
    // if Lz==0 initially and the potential is axisymmetric, it must stay so for the entire orbit
    bool Lzok = isAxisymmetric(potential) && Lz(initial_conditions)==0 ? avgL.mean()==0 : true;
    bool ok = avgE.disp()<eps*eps && (avgL.disp()<eps*eps || !isAxisymmetric(potential));
    if(completed)
        std::cout << trajFull.size()-1 << " steps,  ";
    else {
        std::cout << "\033[1;31mCRASHED\033[0m after " << trajFull.size()-1 <<
            " steps at time " << trajFull.back().second << ",  ";
        ok=false;
    }
    if(!okrot) {
        std::cout << "\033[1;34mROTATING FRAME FAILED\033[0m,  ";
        ok = false;
    }
    std::cout << "E=" <<avgE.mean() <<" +- "<<sqrt(avgE.disp())<<
        ",  Lz="<<avgL.mean()<<" +- "<<sqrt(avgL.disp())<<
        (Lzok ? "" : " \033[1;35m!!\033[0m") <<
        (ok? "" : " \033[1;31m**\033[0m") << "\n";
    return ok && Lzok;
}

potential::PtrPotential make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    potential::PtrPotential gp = potential::readGalaxyPotential(params_file, units::galactic_Myr);
    std::remove(params_file);
    if(!gp.get())
        std::cout<<"Potential not created\n";
    return gp;
}

const char* galpot_params[] = {
"2\n"
"1e10 3 0.3 0 0\n"
"2e9 3.5 0.9 0 0\n"
"2\n"
"1e12 0.5 0 1.8 0.075 2.1\n"
"1e8 1 1 3 17 0\n", 
"3\n"  // test various types of discs
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n"  };// log density profile with cutoff

const int NUMPOT=8;
const char* potential_params[NUMPOT] = {
    "type=CylSpline, density=Plummer, mass=10.0, scaleRadius=5.0",
    "type=Plummer, mass=10.0, scaleRadius=5.0",
    "type=Isochrone, mass=10.0, scaleRadius=3.0",
    "type=NFW, mass=20.0, scaleRadius=2.5",
    "type=MiyamotoNagai, mass=5.0, scaleRadius=2.0, scaleHeight=0.2",
    "type=Logarithmic, v0=1.0, scaleRadius=0.2, axisRatioY=0.8, axisRatioZ=0.5",
    "type=Ferrers, mass=0.4, scaleRadius=0.5, axisRatioY=0.7, axisRatioZ=0.5",
    "type=Dehnen, mass=2.0, gamma=0.8, scaleRadius=1.0, axisRatioY=0.7, axisRatioZ=0.5" };

/// define test suite in terms of points for various coord systems
const int numtestpoints=5;
const double posvel_car[numtestpoints][6] = {
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {0,-1, 2, 0,   0.2,-0.5},   // point in y-z plane 
    {2, 0,-1, 0,  -0.3, 0.4},   // point in x-z plane
    {0, 0, 1, 0,   0,   0.5},   // point along z axis
    {0, 0, 0,-0.4,-0.2,-0.1}};  // point at origin with nonzero velocity
const double posvel_cyl[numtestpoints][6] = {   // order: R, z, phi
    {1, 2, 3, 0.4, 0.2, 0.3},   // ordinary point
    {2,-1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {0, 2, 0, 0,  -0.5, 0  },   // point along z axis, vphi must be zero
    {0,-1, 2, 0.5, 0.3, 0  },   // point along z axis, vphi must be zero, but vR is non-zero
    {0, 0, 0, 0.3,-0.5, 0  }};  // point at origin with nonzero velocity in R and z
const double posvel_sph[numtestpoints][6] = {   // order: R, theta, phi
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {2, 1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {1, 0, 0,-0.5, 0,   0  },   // point along z axis, vphi must be zero
    {1,3.14159, 2, 0.5, 0.3, 1e-4},   // point almost along z axis, vphi must be small, but vtheta is non-zero
    {0, 2,-1, 0.5, 0,   0  }};  // point at origin with nonzero velocity in R


// test the routines identical to those in orbit.cpp, which normalize the coordinates/velocities
// in cylindrical and spherical systems to the standard range (r>0, 0<theta<pi)
inline coord::PosVelCyl getPosVelCyl(const double data[6])
{
    if(data[0] >= 0)
        return coord::PosVelCyl(data);
    else
        return coord::PosVelCyl(-data[0], data[1], data[2]+M_PI, -data[3], data[4], -data[5]);
}

inline coord::PosVelSph getPosVelSph(const double data[6])
{
    double r = data[0];
    double phi = data[2];
    int signr = 1, signt = 1;
    double theta = fmod(data[1], 2*M_PI);
    // normalize theta to the range 0..pi, and r to >=0
    if(theta<-M_PI) {
        theta += 2*M_PI;
    } else if(theta<0) {
        theta = -theta;
        signt = -1;
    } else if(theta>M_PI) {
        theta = 2*M_PI-theta;
        signt = -1;
    }
    if(r<0) {
        r = -r;
        theta = M_PI-theta;
        signr = -1;
    }
    if((signr == -1) ^ (signt == -1))
        phi += M_PI;
    phi = math::wrapAngle(phi);
    return coord::PosVelSph(r, theta, phi, data[3] * signr, data[4] * signt, data[5] * signr * signt);
}

bool test_normalize_range()
{
    bool ok = true;
    double data[6];
    for(int a=0; a<30; a++) {
        for(int d=0; d<6; d++) data[d] = math::random()*10-5;
        coord::PosVelCyl c0(data), c1(getPosVelCyl(data));
        coord::PosVelSph s0(data), s1(getPosVelSph(data));
        coord::PosVelCar C0(toPosVelCar(c0)), C1(toPosVelCar(c1));
        coord::PosVelCar S0(toPosVelCar(s0)), S1(toPosVelCar(s1));
        bool okc = equalPosVel(C0,C1,1e-14), oks = equalPosVel(S0,S1,1e-14);
        if(output) {
            std::cout <<
                c0 << '\n' << c1 << '\n' << C0 << '\n' << C1 << '\n' <<
                (okc ? "OK\n\n" : "!!!!!WRONG!!!!!\n\n") <<
                s0 << '\n' << s1 << '\n' << S0 << '\n' << S1 << '\n' <<
                (oks ? "OK\n\n" : "!!!!!WRONG!!!!!\n\n");
        }
        ok &= okc && oks;
    }
    if(!ok)
        std::cout << "test_normalize_range \033[1;31mFAILED\033[0m\n";
    return ok;
}

int main() {
    std::vector<potential::PtrPotential> pots;
    for(int p=0; p<NUMPOT; p++)
        pots.push_back(potential::createPotential(utils::KeyValueMap(potential_params[p])));
    pots.push_back(make_galpot(galpot_params[0]));
    pots.push_back(make_galpot(galpot_params[1]));
    bool allok = true;
    allok &= test_normalize_range();
    for(unsigned int ip=0; ip<pots.size(); ip++) {
        for(int ic=0; ic<numtestpoints; ic++) {
            allok &= test_potential(*pots[ip], coord::PosVelCar(posvel_car[ic]));
            allok &= test_potential(*pots[ip], coord::PosVelCyl(posvel_cyl[ic]));
            allok &= test_potential(*pots[ip], coord::PosVelSph(posvel_sph[ic]));
        }
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}