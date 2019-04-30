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
#include <iostream>
#include <fstream>
#include <cmath>

const double eps=1e-5;     // accuracy of energy conservation
const double epsrot=1e-4;  // accuracy of comparison between inertial and rotating frames
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;
const double Omega=2.718;  // rotation frequency (arbitrary)

// count the number of internal steps of the ODE integrator
class RuntimeCountSteps: public orbit::BaseRuntimeFnc {
public:
    size_t& count;  // store the number of steps
    double& time;   // store the maximum achieved time

    RuntimeCountSteps(size_t& _count, double& _time) :
        count(_count), time(_time) {}

    virtual orbit::StepResult processTimestep(
        const math::BaseOdeSolver&, const double, const double t, double [])
    {
        ++count;
        time = t;
        return orbit::SR_CONTINUE;
    }
};

template<typename CoordT> bool isCartesian() { return false; }
template<> bool isCartesian<coord::Car>() { return true; }

template<typename CoordT>
bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelT<CoordT>& initial_conditions,
    const double total_time, const double timestep)
{
    // whether compare the result of orbit integration in rotating and inertial frames
    bool checkRot = isCartesian<CoordT>() && isAxisymmetric(potential);
    std::cout << potential.name()<<"  "<<CoordT::name()<<" ("<<initial_conditions<<
        "Torb: "<<T_circ(potential, totalEnergy(potential, initial_conditions))<<")\n";
    double ic[6];
    initial_conditions.unpack_to(ic);
    std::vector<coord::PosVelT<CoordT> > traj;
    std::vector<coord::PosVelCar> trajRot;
    size_t numSteps=0, numStepsRot=0;
    double time=0, timeRot=0;
    orbit::RuntimeFncArray fncs(2);
    fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<CoordT>(timestep, /*output*/ traj));
    fncs[1] = orbit::PtrRuntimeFnc(new RuntimeCountSteps(/*output*/ numSteps, time));
    orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegrator<CoordT>(potential), fncs,
        orbit::OrbitIntParams(/*accuracy*/ 1e-8, /*maxNumSteps*/10000));
    if(checkRot) {
        fncs[0] = orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory<coord::Car>(timestep, trajRot));
        fncs[1] = orbit::PtrRuntimeFnc(new RuntimeCountSteps(numStepsRot, timeRot));
        orbit::integrate(initial_conditions, total_time, orbit::OrbitIntegratorRot(potential, Omega), fncs);
        if(trajRot.size() != traj.size()) {
            std::cout << "\033[1;34mRotating frame inconsistent\033[0m\n";
            return false;
        }
    }
    math::Averager avgE, avgL;
    bool okrot = true;
    for(size_t i=0; i<traj.size(); i++) {
        double E =totalEnergy(potential, traj[i]);
        double Lz=coord::Lz(traj[i]);
        avgE.add(E);
        avgL.add(Lz);
        double xv[6];
        coord::toPosVelCar(traj[i]).unpack_to(xv);
        if(output)
            std::cout << utils::pp(i*timestep,5) <<"   " <<
                utils::pp(xv[0], 9) <<' '<< utils::pp(xv[1], 9) <<' '<< utils::pp(xv[2], 9) <<"  "<<
                utils::pp(xv[3], 9) <<' '<< utils::pp(xv[4], 9) <<' '<< utils::pp(xv[5], 9) <<"   ";
        if(checkRot) {
            double angle = i*timestep*Omega, cosa = cos(angle), sina = sin(angle);
            double rxv[6];
            trajRot[i].unpack_to(rxv);
            double pointRot[6] = {
                rxv[0] * cosa - rxv[1] * sina, rxv[1] * cosa + rxv[0] * sina, rxv[2],
                rxv[3] * cosa - rxv[4] * sina, rxv[4] * cosa + rxv[3] * sina, rxv[5] };
            if(!equalPosVel(coord::PosVelT<CoordT>(pointRot), traj[i], epsrot) && coord::Lz(traj[i])!=0)
                okrot = false;
            if(output) std::cout <<
                utils::pp(pointRot[0], 9) << ' ' <<
                utils::pp(pointRot[1], 9) << ' ' <<
                utils::pp(pointRot[2], 9) << "   " <<
                utils::pp(totalEnergy(potential, coord::PosVelT<CoordT>(pointRot)), 15) << "  ";
        }
        if(output)
            std::cout << utils::pp(E, 15) << ' '<<
            utils::pp(totalEnergy(potential, coord::PosVelCar(xv)), 15) << ' '<<
            utils::pp(Lz, 15) << '\n';
    }
    bool completed = time>0.999999*total_time;
    // if Lz==0 initially and the potential is axisymmetric, it must stay so for the entire orbit
    bool Lzok = isAxisymmetric(potential) && Lz(initial_conditions)==0 ? avgL.mean()==0 : true;
    bool ok = avgE.disp()<eps*eps && (avgL.disp()<eps*eps || !isAxisymmetric(potential));
    if(completed)
        std::cout <<numSteps<<" steps,  ";
    else {
        std::cout << "\033[1;31mCRASHED\033[0m after "<<numSteps<<" steps at time "<<time<<",  ";
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

const char* test_galpot_params[] = {
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
    pots.push_back(potential::PtrPotential(new potential::Plummer(10.0, 5.0)));
    pots.push_back(potential::PtrPotential(new potential::Isochrone(10.0, 3.0)));
    pots.push_back(potential::PtrPotential(new potential::NFW(20.0, 2.5)));
    pots.push_back(potential::PtrPotential(new potential::MiyamotoNagai(5.0, 2.0, 0.2)));
    pots.push_back(potential::PtrPotential(new potential::Logarithmic(1.0, 0.2, 0.8, 0.5)));
    pots.push_back(potential::PtrPotential(new potential::Ferrers(0.4, 0.5, 0.7, 0.5)));
    pots.push_back(potential::PtrPotential(new potential::Dehnen(2.0, 1.0, 0.8, 0.7, 0.5)));
    pots.push_back(make_galpot(test_galpot_params[0]));
    pots.push_back(make_galpot(test_galpot_params[1]));
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