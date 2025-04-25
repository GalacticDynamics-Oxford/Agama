/** \file    test_orbit_integr.cpp
    \date    2015-2023
    \author  Eugene Vasiliev

    Test orbit integration in various potentials and coordinate systems.
*/
#include "orbit.h"
#include "orbit_variational.h"
#include "potential_factory.h"
#include "potential_utils.h"
#include "utils.h"
#include "debug_utils.h"
#include "math_random.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double epsE   = 3e-6;  // accuracy of energy conservation
const double epsL   = 1e-7;  // accuracy of energy conservation
const double epsCS  = 2e-3;  // accuracy of comparison of orbits in different coordinate systems
const double epsrot = 1e-3;  // accuracy of comparison between inertial and rotating frames
const bool output   = utils::verbosityLevel >= utils::VL_VERBOSE;
const double Omega  = 2.718; // rotation frequency (arbitrary)

inline double difposvel(const coord::PosVelCar& a, const coord::PosVelCar& b) {
    return sqrt(
        pow_2(a.x -b.x ) + pow_2(a.y -b.y ) + pow_2(a.z -b.z ) +
        pow_2(a.vx-b.vx) + pow_2(a.vy-b.vy) + pow_2(a.vz-b.vz));
}

void convertOrbitToSpline(const std::vector< std::pair<coord::PosVelCar, double> > &traj,
    math::QuinticSpline& splx, math::QuinticSpline& sply, math::QuinticSpline& splz)
{
    size_t n=traj.size();
    std::vector<double> t(n), x(n), y(n), z(n), vx(n), vy(n), vz(n);
    for(size_t i=0; i<n; i++) {
        t[i] = traj[i].second;
        x[i] = traj[i].first.x;
        vx[i]= traj[i].first.vx;
        y[i] = traj[i].first.y;
        vy[i]= traj[i].first.vy;
        z[i] = traj[i].first.z;
        vz[i]= traj[i].first.vz;
    }
    splx = math::QuinticSpline(t, x, vx);
    sply = math::QuinticSpline(t, y, vy);
    splz = math::QuinticSpline(t, z, vz);
}

class DistanceFromPointToOrbit: public math::IFunctionNoDeriv {
    const double x1, y1, z1;
    const math::IFunction &x2, &y2, &z2;
public:
    DistanceFromPointToOrbit(double _x1, double _y1, double _z1,
        const math::IFunction &_x2, const math::IFunction &_y2, const math::IFunction &_z2) :
        x1(_x1), y1(_y1), z1(_z1), x2(_x2), y2(_y2), z2(_z2) {}
    virtual double value(double t) const {
        return sqrt(pow_2(x1-x2(t)) + pow_2(y1-y2(t)) + pow_2(z1-z2(t)));
    }
};

// To compare two orbits, we may look at the distance between points recorded at the same time,
// but this is not very robust: because the orbit energy is not exactly conserved during integration,
// the two orbits at a fixed time gradually drift apart due to accumulated phase differences.
// Instead we look at how close the orbits are in space after accounting for a possible time
// difference, i.e., for each point on orbit1 find the closest distance to point on orbit2,
// even if it is achieved at a slightly different time.
double maxDistanceBetweenOrbits(
    const std::vector< std::pair<coord::PosVelCar, double> > &traj1,
    const std::vector< std::pair<coord::PosVelCar, double> > &traj2)
{
    math::QuinticSpline x2, y2, z2;
    convertOrbitToSpline(traj2, x2, y2, z2);
    const std::vector<double> &t2 = x2.xvalues();
    // loop over all timestamps on orbit 1 except the boundary points,
    // and for each timestamp, find the minimum distance from orbit2 to the point in orbit1
    double maxdist = 0;
    for(size_t i1=1, i2=0; i1<traj1.size()-1; i1++) {
        while(i2 < t2.size() && t2[i2] < traj1[i1].second)
            i2++;
        // determine the interval for searching the minimum distance: +- one timestep on either orbit
        double tlow = fmin(traj1[i1-1].second, t2[i2>0 ? i2-1 : 0]);
        double tupp = fmax(traj1[i1+1].second, t2[i2]);
        double tnearest = math::findMin(DistanceFromPointToOrbit(
            traj1[i1].first.x, traj1[i1].first.y, traj1[i1].first.z, x2, y2, z2),
            tlow, tupp, traj1[i1].second, /*tolerance*/ 1e-12);
        maxdist = fmax(maxdist, sqrt(
            pow_2(traj1[i1].first.x - x2(tnearest)) +
            pow_2(traj1[i1].first.y - y2(tnearest)) +
            pow_2(traj1[i1].first.z - z2(tnearest)) ));
    }
    return maxdist;
}

template<typename CoordT>
bool test_coordsys(const potential::BasePotential& potential,
    const coord::PosVelCar& initial_conditions, double total_time,
    std::vector< std::pair<coord::PosVelCar, double> > &trajFull,
    orbit::OrbitIntParams::Method method = orbit::OrbitIntParams::DOP853)
{
    std::string name = CoordT::name();
    name.resize(12, ' ');
    std::cout << name;
    double timestep = 0.002 * total_time;
    double init_time = -0.25 * total_time;
    std::vector< std::pair<coord::PosVelCar, double> > trajReg, trajRot;
    orbit::OrbitIntParams params(/*accuracy*/ 1e-8, /*maxNumSteps*/10000, method);
    orbit::OrbitIntegrator<CoordT> orbint(potential, /*Omega*/0, params);
    // record the orbit at regular intervals of time
    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(
        orbint, timestep, /*output*/ trajReg)));
    // record the orbit at each timestep of the ODE integrator
    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(
        orbint, /*sampling interval - every timestep*/ 0, /*output*/ trajFull)));
    // run the orbit
    orbint.init(initial_conditions, init_time);
    orbint.run(total_time);

    // whether compare the result of orbit integration in rotating and inertial frames
    bool checkRot = isAxisymmetric(potential);
    if(checkRot) {
        orbit::OrbitIntegrator<CoordT> orbrot(potential, Omega, params);
        orbrot.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(
            orbrot, timestep, trajRot)));
        orbrot.init(initial_conditions, init_time);
        // check that a continuation of the orbit integration gives the same result
        // as a single run for a longer time
        orbrot.run(total_time*0.6);
        orbrot.run(total_time*0.4);
        if(trajRot.size() != trajReg.size()) {
            std::cout << "\033[1;34mOrbit in the rotating frame has different length\033[0m\n";
            return false;
        } else {
            // convert the orbit back into non-rotating frame
            for(size_t i=0; i<trajReg.size(); i++) {
                double angle = (trajReg[i].second-init_time)*Omega, cosa = cos(angle), sina = sin(angle);
                coord::PosVelCar& xv = trajRot[i].first;  // cartesian coords in the rotating frame
                xv = coord::PosVelCar(  // converted to the inertial frame
                    xv.x  * cosa - xv.y  * sina, xv.y  * cosa + xv.x  * sina, xv.z,
                    xv.vx * cosa - xv.vy * sina, xv.vy * cosa + xv.vx * sina, xv.vz);
            }
        }
    }
    math::Averager avgE, avgL;
    for(size_t i=0; i<trajFull.size(); i++) {
        avgE.add(totalEnergy(potential, trajFull[i].first));
        avgL.add(Lz(trajFull[i].first));
    }
    if(output) {
        std::ofstream strm((std::string("Orbit_") + potential.name() + '_' + CoordT::name() + '_' +
            utils::toString(initial_conditions.x) + '_' + utils::toString(initial_conditions.y) + '_' +
            utils::toString(initial_conditions.z) + '_' + utils::toString(initial_conditions.vx)+ '_' +
            utils::toString(initial_conditions.vy)+ '_' + utils::toString(initial_conditions.vz) ).
            c_str());
        for(size_t i=0; i<trajFull.size(); i++) {
            const coord::PosVelCar& xv = trajFull[i].first;
            strm << utils::pp(trajFull[i].second, 18) << "  " <<
            utils::pp(xv.x , 18) <<' '<< utils::pp(xv.y , 18) <<' '<< utils::pp(xv.z , 18) <<"  " <<
            utils::pp(xv.vx, 18) <<' '<< utils::pp(xv.vy, 18) <<' '<< utils::pp(xv.vz, 18) <<"   "<<
            utils::pp(totalEnergy(potential, xv), 18) << ' '<< utils::pp(Lz(xv), 18) << '\n';
        }
    }
    bool ok = trajReg.back().second > 0.999999 * (total_time + init_time);
    if(ok)
        std::cout << trajFull.size()-1 << " steps";
    else {
        std::cout << "\033[1;31mCRASHED\033[0m after " << trajFull.size()-1 <<
            " steps at time " << trajFull.back().second;
    }
    std::cout << ", E=" << avgE.mean() << " +- " << sqrt(avgE.disp());
    if(avgE.disp() > epsE*epsE) {
        std::cout << " \033[1;31m**\033[0m";
        ok = false;
    }
    if(isAxisymmetric(potential)) {
        std::cout << ", Lz=" << avgL.mean() << " +- " << sqrt(avgL.disp());
        if(avgL.disp() > epsL*epsL || (Lz(initial_conditions)==0 && fabs(avgL.mean()) > 4e-15)) {
            std::cout << " \033[1;35m**\033[0m";
            ok = false;
        }
    }
    if(checkRot) {
        double difrot = maxDistanceBetweenOrbits(trajReg, trajRot);
        std::cout << ", |rot-nonrot|=" << difrot;
        if(difrot > epsrot) {
            std::cout << " \033[1;34m**\033[0m";
            ok = false;
        }
    }
    if(trajReg.size() != round(total_time / timestep) + 1) {
        std::cout << "\033[1;33m[incorrect length]\033[0m\n";
        ok = false;
    }
    std::cout << "\n";
    return ok;
}

bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelCar& initial_conditions)
{
    double total_time = 10.0 * T_circ(potential, totalEnergy(potential, initial_conditions));
    if(!isFinite(total_time))
        total_time = 100.0;
    std::cout << "\033[1;37m" << potential.name() << "\033[0m, ic=(" << initial_conditions <<
        "), Torb=" << T_circ(potential, totalEnergy(potential, initial_conditions)) << "\n";

    std::vector< std::pair<coord::PosVelCar, double> > trajCar, trajCyl, trajSph, trajRkn;
    bool ok = true;
    ok &= test_coordsys<coord::Car>(potential, initial_conditions, total_time, trajRkn,
        orbit::OrbitIntParams::DPRKN8);
    ok &= test_coordsys<coord::Car>(potential, initial_conditions, total_time, trajCar);
    ok &= test_coordsys<coord::Cyl>(potential, initial_conditions, total_time, trajCyl);
    ok &= test_coordsys<coord::Sph>(potential, initial_conditions, total_time, trajSph);
    double maxdifcyl = maxDistanceBetweenOrbits(trajCar, trajCyl);
    double maxdifsph = maxDistanceBetweenOrbits(trajCar, trajSph);
    std::cout << "|Car-Cyl|=" << maxdifcyl;
    if(maxdifcyl > epsCS) {
        std::cout << " \033[1;31m**\033[0m";
        ok = false;
    }
    std::cout << ", |Car-Sph|=" << maxdifsph;
    if(maxdifsph > epsCS) {
        std::cout << " \033[1;31m**\033[0m";
        ok = false;
    }
    std::cout << "\n";
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

const char* galpot_params =
"3\n"  // test various types of discs
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n";

const int NUMPOT=8;
const char* potential_params[NUMPOT] = {
    "type=CylSpline, density=Plummer, mass=10.0, scaleRadius=5.0",
    "type=Plummer, mass=10.0, scaleRadius=5.0",
    "type=Isochrone, mass=10.0, scaleRadius=3.0",
    "type=NFW, mass=20.0, scaleRadius=2.5",
    "type=MiyamotoNagai, mass=5.0, scaleRadius=2.0, scaleHeight=0.2",
    "type=Logarithmic, v0=1.0, scaleRadius=0.2, axisRatioY=0.8, axisRatioZ=0.5",
    "type=Ferrers, mass=0.7, scaleRadius=1.4, axisRatioY=0.7, axisRatioZ=0.5",
    "type=Dehnen, mass=2.0, gamma=0.8, scaleRadius=1.0, axisRatioY=0.7, axisRatioZ=0.5" };

/// define test suite in terms of points for various coord systems
const int NUMPOINTS=7;
const double posvel_car[NUMPOINTS][6] = {
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {0,-1, 2, 0,   0.2,-0.5},   // point in y-z plane
    {2, 0,-1, 0,  -0.3, 0.4},   // point in x-z plane
    {0, 0, 1, 0,   0,   0.5},   // point along z axis and only vz!=0 (1d orbit)
    {0, 0, 1, 0.2,-0.3, 0.4},   // point along z axis, but with all three velocity components !=0
    {1e-6,0,-1,0.3,1e-4,0.5},   // point nearly on z axis but with nonzero Lz
    {0, 0, 0, 0.2,-0.3, 0.6}};  // point at origin with nonzero velocity

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

bool test_zero_length_orbit(const potential::BasePotential& pot)
{
    const double timeStart = 1.23, Omega = 4.56;
    const coord::PosVelCar ic(0.1, 0.2, 0.3, 0.4, 0.5, 0.6);
    orbit::Trajectory traj, dev[6];
    {   // sampling interval set to 0 - store two (identical) endpoints of the zero-length time interval
        orbit::OrbitIntegrator<coord::Car> orbint(pot, Omega);
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(orbint, 0, traj)));
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeVariational(orbint, 0, dev)));
        orbint.init(ic, timeStart);
        orbint.run(0);
    }
    bool ok = traj.size() == 2 && dev[0].size() == 2 &&
        equalPosVel(traj[0].first, ic, /*eps*/0) && traj[0].second == timeStart &&
        equalPosVel(traj[1].first, ic, /*eps*/0) && traj[1].second == timeStart &&
        equalPosVel(dev[0][0].first, coord::PosVelCar(1, 0, 0, 0, 0, 0), DBL_EPSILON) &&
        equalPosVel(dev[0][1].first, dev[0][0].first, /*eps*/0);
    {   // sampling interval set to INFINITY - store only the final point
        orbit::OrbitIntegrator<coord::Car> orbint(pot, Omega);
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(orbint, INFINITY, traj)));
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeVariational(orbint, INFINITY, dev)));
        orbint.init(ic, timeStart);
        orbint.run(0);
    }
    ok = traj.size() == 1 && dev[0].size() == 1 &&
        equalPosVel(traj[0].first, ic, /*eps*/0) && traj[0].second == timeStart &&
        equalPosVel(dev[0][0].first, coord::PosVelCar(1, 0, 0, 0, 0, 0), DBL_EPSILON);
    if(!ok)
        std::cout << "test_zero_length_orbit \033[1;31mFAILED\033[0m\n";
    return ok;
}

int main() {
    std::vector<potential::PtrPotential> pots;
    for(int p=0; p<NUMPOT; p++)
        pots.push_back(potential::createPotential(utils::KeyValueMap(potential_params[p])));
    pots.push_back(make_galpot(galpot_params));
    bool allok = true;
    allok &= test_normalize_range();
    allok &= test_zero_length_orbit(*pots[1]);
    for(unsigned int ip=0; ip<pots.size(); ip++) {
        for(int ic=0; ic<NUMPOINTS; ic++)
            allok &= test_potential(*pots[ip], coord::PosVelCar(posvel_car[ic]));
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}