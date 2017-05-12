/** \file    test_orbit_integr.cpp
    \date    2015-2016
    \author  Eugene Vasiliev

    Test orbit integration in various potentials and coordinate systems.
    Note: not all tests pass at the moment.
*/
#include "orbit.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_dehnen.h"
#include "potential_factory.h"
#include "potential_ferrers.h"
#include "units.h"
#include "utils.h"
#include "debug_utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const double eps=1e-6;     // accuracy of conservation
const double epsrot=1e-4;  // accuracy of comparison between inertial and rotating frames
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;
const double Omega=2.718;  // rotation frequency (arbitrary)

// count the number of internal steps of the ODE integrator
class RuntimeCountSteps: public orbit::BaseRuntimeFnc {
public:
    size_t& count;

    RuntimeCountSteps(size_t& _count) : count(_count) {}

    virtual orbit::StepResult processTimestep(
        const math::BaseOdeSolver&, const double, const double, double[])
    {
        ++count;
        return orbit::SR_CONTINUE;
    }
};

template<typename coordSysT> bool isCartesian() { return false; }
template<> bool isCartesian<coord::Car>() { return true; }

template<typename coordSysT>
bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep)
{
    // whether compare the result of orbit integration in rotating and inertial frames
    bool checkRot = isCartesian<coordSysT>() && isAxisymmetric(potential);
    std::cout << potential.name()<<"  "<<coordSysT::name()<<" ("<<initial_conditions<<")\n";
    double ic[6];
    initial_conditions.unpack_to(ic);
    std::vector<coord::PosVelT<coordSysT> > traj;
    std::vector<coord::PosVelCar> trajRot;
    size_t numsteps=0, numstepsRot=0;
    orbit::RuntimeFncArray fncs(2);
    fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<coordSysT>(traj, timestep));
    fncs[1] = orbit::PtrRuntimeFnc(new RuntimeCountSteps(numsteps));
    orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegrator<coordSysT>(potential), fncs);
    if(checkRot) {
        fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<coord::Car>(trajRot, timestep));
        fncs[1] = orbit::PtrRuntimeFnc(new RuntimeCountSteps(numstepsRot));
        orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegratorRot(potential, Omega), fncs);
        if(trajRot.size() != traj.size()) {
            std::cout << "\033[1;34mRotating frame inconsistent\033[0m\n";
            return false;
        }
    }
    double avgH=0, avgLz=0, dispH=0, dispLz=0;
    bool okrot = true;
    for(size_t i=0; i<traj.size(); i++) {
        double H =totalEnergy(potential, traj[i]);
        double Lz=coord::Lz(traj[i]);
        avgH +=H;  dispH +=pow_2(H);
        avgLz+=Lz; dispLz+=pow_2(Lz);
        double xv[6];
        coord::toPosVelCar(traj[i]).unpack_to(xv);
        if(checkRot) {
            double angle = i*timestep*Omega, cosa = cos(angle), sina = sin(angle);
            coord::PosVelCar pointRot(
                xv[0] * cosa + xv[1] * sina, xv[1] * cosa - xv[0] * sina, xv[2],
                xv[3] * cosa + xv[4] * sina, xv[4] * cosa - xv[3] * sina, xv[5]);
            if(!equalPosVel(pointRot, trajRot[i], epsrot)) {
                okrot = false;
            }
        }
        if(output) {
            std::cout << i*timestep<<"   " <<
                xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   ";
            if(checkRot) std::cout <<
                trajRot[i].x << ' ' << trajRot[i].y << ' ' << trajRot[i].z << "   ";
            std::cout << H << ' ' << Lz << ' ' << potential.density(traj[i]) << '\n';
        }
    }
    avgH/=traj.size();
    avgLz/=traj.size();
    dispH/=traj.size();
    dispLz/=traj.size();
    dispH= sqrt(std::max<double>(0, dispH -pow_2(avgH)));
    dispLz=sqrt(std::max<double>(0, dispLz-pow_2(avgLz)));
    bool completed = traj.size()>0.99*total_time/timestep;
    bool ok = dispH<eps && (dispLz<eps || !isAxisymmetric(potential));
    if(completed)
        std::cout <<numsteps<<" steps,  ";
    else if(numsteps>0) {
        std::cout <<"\033[1;33mCRASHED\033[0m after "<<numsteps<<" steps,  ";
        // this may naturally happen in the degenerate case when an orbit with zero angular momentum
        // approaches the origin -- not all combinations of potential and coordinate system
        // can cope with this. If that happens for a non-degenerate case, this is an error.
        if(avgLz!=0) ok=false;
    }
    else {
        // this may happen for an orbit started at r==0 in some potentials where the force is singular,
        // otherwise it's an error
        if(toPosSph(initial_conditions).r != 0) {
            std::cout <<"\033[1;31mFAILED\033[0m,  ";
            ok = false;
        }
    }
    if(!okrot) {
        std::cout << "\033[1;34mROTATING FRAME FAILED\033[0m,  ";
        ok = false;
    }
    std::cout << "E=" <<avgH <<" +- "<<dispH<<",  Lz="<<avgLz<<" +- "<<dispLz<<
        (ok? "" : " \033[1;31m**\033[0m") << "\n";
    return ok;
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

const char* test_galpot_params[] = {
"2\n"  // stanard galaxy model
"7.52975e+08 3 0.3 0 0\n"
"1.81982e+08 3.5 0.9 0 0\n"
"2\n"
"9.41496e+10 0.5 0 1.8 0.075 2.1\n"
"1.25339e+07 1 1 3 17 0\n", 
"3\n"  // test various types of discs
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n"  };// log density profile with cutoff

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


int main() {
    std::vector<potential::PtrPotential> pots;
    pots.push_back(potential::PtrPotential(new potential::Plummer(10.,5.)));
    pots.push_back(potential::PtrPotential(new potential::Isochrone(6.,3.)));
    pots.push_back(potential::PtrPotential(new potential::NFW(10.,10.)));
    pots.push_back(potential::PtrPotential(new potential::MiyamotoNagai(5.,2.,0.2)));
    pots.push_back(potential::PtrPotential(new potential::Logarithmic(1.,0.01,.8,.5)));
    pots.push_back(potential::PtrPotential(new potential::Logarithmic(1.,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Ferrers(1.,0.9,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Dehnen(2.,1.,1.5,.7,.5)));
    pots.push_back(make_galpot(test_galpot_params[0]));
    pots.push_back(make_galpot(test_galpot_params[1]));
    std::cout<<std::setprecision(10);
    const double total_time=100.;
    const double timestep=0.8;
    bool allok = true;
    for(unsigned int ip=0; ip<pots.size(); ip++) {
        for(int ic=0; ic<numtestpoints; ic++) {
            allok &= test_potential(*pots[ip], coord::PosVelCar(posvel_car[ic]), total_time, timestep);
            allok &= test_potential(*pots[ip], coord::PosVelCyl(posvel_cyl[ic]), total_time, timestep);
            allok &= test_potential(*pots[ip], coord::PosVelSph(posvel_sph[ic]), total_time, timestep);
        }
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}