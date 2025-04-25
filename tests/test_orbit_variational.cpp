/** \file    test_orbit_variational.cpp
    \author  Eugene Vasiliev
    \date    2008-2024

    This example demonstrates how to augment the orbit integration in AGAMA with the simultaneous
    integration of the variational equation, which describes the evolution of deviation vectors
    (infinitesimally small perturbations of the initial conditions of the orbit).
    It enables the computation of the (largest) Lyapunov exponent, which measures the orbit chaoticity.
    The task is divided into three routines:
    - the class that provides the right-hand side of the differential equation;
    - the routine that computes and stores the orbit and the magnitude of the deviation vector;
    - the routine that estimates the Lyapunov exponent from the time evolution of the dev.vec.
*/
#include "orbit.h"
#include "orbit_variational.h"
#include "potential_factory.h"
#include "potential_utils.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <cassert>

const char* err = " \033[1;31m**\033[0m";

//---------------------------------//
// PART 1: testing the ODE solvers //
//---------------------------------//

/** Test function for the second-order linear ODE system:  d2x(t)/dt2 = c(t) x(t),
    with N-dimensional vectors x and c.
    This class provides both the IOdeSystem interface,
    representing the second-order ODE system as a first-order system of twice as many equations,
    and the IOde2System interface, allowing to use it in the specialized solvers for this type of ODEs.
*/
template<int NDIM>
class RHS: public math::IOdeSystem2ndOrderLinear {
    const double& tbegin;  // reference to the current moment of time at the beginning of this timestep
public:
    RHS(const double& _tbegin) : tbegin(_tbegin) {}
    virtual void evalMat(const double deltat, double a[], double b[]) const;
    virtual unsigned int size() const { return NDIM*2; }
    // the analytic solution
    static void trueSolution(const double t, double sol[]);
};

// NDIM=1: trivial system x'' = -3.25*x + x', analytic solution in terms of trig and exp
template<> void RHS<1>::evalMat(const double, double a[], double b[]) const {
    a[0] = -3.25;
    b[0] = 1.0;
}

template<> void RHS<1>::trueSolution(const double t, double sol[]) {
    sol[0] = (cos(M_SQRT3*t) - 0.5/M_SQRT3 * sin(M_SQRT3*t)) * exp(0.5*t);
}

// NDIM=2: more complicated system with time-dependent coefs, analytical solution in terms of trig fncs
template<> void RHS<2>::evalMat(const double deltat, double a[], double b[]) const {
    double t=tbegin+deltat, t2=t*t, t3=t*t2, t4=t*t3, den=1 / (t4 + 4*t + 4);
    a[0] = den * (-0.25*t4 - 2*t2 - 8*t - 4);
    a[1] = den * (2.75*t2 + 1.5*t - 2);
    a[2] = den * (1.5*t3 + 8*t + 8);
    a[3] = den * (-t4 - 2*t2 - 6.5*t - 1);
    b[0] = den * (2*t3 + 1.5*t2 + 4);
    b[1] = den * (-0.75*t3 + 2*t + 4);
    b[2] = den * (-4*t2 + 3*t);
    b[3] = den * (2*t3 - 1.5*t2);
}

template<> void RHS<2>::trueSolution(const double t, double sol[]) {
    sol[0] =   cos(t) + t*sin(t/2);
    sol[1] = t*sin(t) + 2*sin(t/2);
}


template<class Stepper, int NDIM, typename Param>
bool testStepper(double dt, double& errStep, double& errInt, Param param)
{
    double t = 0.;
    double x[] = {1., 0., 0., 1.};
    RHS<NDIM> rhs(/*reference to the time variable, modified as the integration progresses*/ t);
    Stepper stepper(rhs, param);
    stepper.init(x);
    int ind=0;
    errStep = 0, errInt = 0;
    const int numPoints = 65;  // points where to compare the numerical solution with the analytic one
    double testTimes[numPoints];
    for(int i=0; i<numPoints; i++)
        testTimes[i] = 2 + 0.015625*i;
    while(t<testTimes[numPoints-1]) {
        double stepTaken = stepper.doStep(dt);
        if(stepTaken != dt) {
            std::cout << "Prescribed stepsize not respected: " << dt << "!=" << stepTaken << err << "\n";
            return false;
        }
        t+=dt;
        while(ind<numPoints && testTimes[ind] <= t) {
            double tcheck = testTimes[ind], err = 0;
            double trueSolution[NDIM];
            rhs.trueSolution(tcheck, trueSolution);
            for(int d=0; d<NDIM; d++)
                err = fmax(err, fabs(stepper.getSol(tcheck-t+dt, d) - trueSolution[d]));
            if(t == tcheck)
                errStep = fmax(errStep, err);
            else
                errInt  = fmax(errInt,  err);
            ind++;
        }
    }
    return true;
}

// test the accuracy of ODE solvers for systems with known analytic solutions:
// check the error as a function of timestep and compare to the expected scalings
bool testOde()
{
    std::cout << "Testing the accuracy of ODE integrators: GL3, GL4, DOP853.\n"
    "Print the error in numerical solution as a function of timestep: \n"
    "(s) evaluated at the end of each step (the accuracy of solution itself), and \n"
    "(i) evaluated at some points inside the step (the accuracy of interpolation / dense output).\n"
    "timestep GL3(s)  GL3(i)  GL4(s)  GL4(i)  DOP(s)  DOP(i)\n";
    bool ok = true;
    double err3s, err3i, err4s, err4i, err8s, err8i;
    std::cout << "1d system\n";
    for(double dt=1; dt>=0.0625; dt*=0.5) {
        ok &= testStepper<math::Ode2StepperGL3<1>, 1>(dt, err3s, err3i, /*num vectors*/ 1);
        ok &= testStepper<math::Ode2StepperGL4<1>, 1>(dt, err4s, err4i, /*num vectors*/ 1);
        ok &= testStepper<math::OdeStepperDOP853 , 1>(dt, err8s, err8i, /*accuracy*/ INFINITY);
        std::cout << utils::pp(dt,6) + "  " +
        utils::pp(err3s,7) + ' ' + utils::pp(err3i,7) + ' ' +
        utils::pp(err4s,7) + ' ' + utils::pp(err4i,7) + ' ' +
        utils::pp(err8s,7) + ' ' + utils::pp(err8i,7);
        if( !(err3s < 4e-3 * pow(dt, 6) && err3i < 6e-3 * pow(dt, 5) &&
              err4s < 1e-4 * pow(dt, 8) && err4i < 5e-4 * pow(dt, 6) &&
              err8s < 2e-4 * pow(dt, 8) && err8i < 3e-4 * pow(dt, 7) ))
        {
            ok = false;
            std::cout << err << '\n';
        } else
            std::cout << '\n';
    }
    std::cout << "2d system\n";
    for(double dt=1; dt>=0.0625; dt*=0.5) {
        ok &= testStepper<math::Ode2StepperGL3<2>, 2>(dt, err3s, err3i, 1);
        ok &= testStepper<math::Ode2StepperGL4<2>, 2>(dt, err4s, err4i, 1);
        ok &= testStepper<math::OdeStepperDOP853 , 2>(dt, err8s, err8i, INFINITY);
        std::cout << utils::pp(dt,6) + "  " +
        utils::pp(err3s,7) + ' ' + utils::pp(err3i,7) + ' ' +
        utils::pp(err4s,7) + ' ' + utils::pp(err4i,7) + ' ' +
        utils::pp(err8s,7) + ' ' + utils::pp(err8i,7);
        if( !(err3s < 2e-4 * pow(dt, 6) && err3i < 3e-4 * pow(dt, 5) &&
              err4s < 1e-5 * pow(dt, 8) && err4i < 3e-5 * pow(dt, 6) &&
              err8s < 2e-5 * pow(dt, 8) && err8i < 6e-6 * pow(dt, 7) ))
        {
            ok = false;
            std::cout << err << '\n';
        } else
            std::cout << '\n';
    }
    std::cout << "Expected:  dt^6    dt^5    dt^8    dt^6    dt^8    dt^7\n\n";
    return ok;
}


//----------------------------------------------//
// PART 2: testing the Lyapunov chaos estimator //
//----------------------------------------------//

/** The function providing the RHS of the differential equation dX/dt = f(X)
    for the orbit computed in the given potential, optionally rotating about the z axis
    with a constant pattern speed Omega, and one deviation vector computed along with the orbit.
    X is a 12-dimensional vector:
    first 6 components are the ordinary position and velocity in cartesian or cylindrical
    coordinates, and the rest are the corresponding components of the deviation vector,
    which measures the distance between the orbit and its infinitesimally close counterpart.
    The evolution of the deviation vector is described by the variational equation,
    which contains second derivatives of the potential computed along the original orbit.
*/
template<typename CoordT>
class OrbitIntegratorVarEq: public math::IOdeSystem {
public:
    /// gravitational potential in which the orbit is computed
    const potential::BasePotential& potential;
    
    /// angular frequency (pattern speed) of the rotating frame
    const double Omega;

    // reference to the current moment of time at the beginning of this timestep
    const double& tbegin;

    /// initialize the object for the given potential
    OrbitIntegratorVarEq(const potential::BasePotential& _potential, double _Omega, double &_tbegin) :
        potential(_potential), Omega(_Omega), tbegin(_tbegin) {}

    virtual unsigned int size() const { return 12; }

    virtual void eval(const double timeOffset, const double x[], double dxdt[], double*) const;
};

template<>
void OrbitIntegratorVarEq<coord::Car>::eval(const double timeOffset, const double x[], double dxdt[],
    double* /*accFac - ignored*/) const
{
    coord::GradCar grad;
    coord::HessCar hess;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad, &hess, tbegin + timeOffset);
    // time derivative of position
    dxdt[0] = x[3] + Omega * x[1];
    dxdt[1] = x[4] - Omega * x[0];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx + Omega * x[4];
    dxdt[4] = -grad.dy - Omega * x[3];
    dxdt[5] = -grad.dz;
    // remaining part follows the evolution of the deviation vector for RuntimeVariational
    // time derivative of position part of the deviation vector
    dxdt[6] = x[9]  + Omega * x[7];
    dxdt[7] = x[10] - Omega * x[6];
    dxdt[8] = x[11];
    // time derivative of velocity part of the deviation vector
    dxdt[9] = -hess.dx2  * x[6] - hess.dxdy * x[7] - hess.dxdz * x[8] + Omega * x[10];
    dxdt[10]= -hess.dxdy * x[6] - hess.dy2  * x[7] - hess.dydz * x[8] - Omega * x[9];
    dxdt[11]= -hess.dxdz * x[6] - hess.dydz * x[7] - hess.dz2  * x[8];
}

template<>
void OrbitIntegratorVarEq<coord::Cyl>::eval(const double timeOffset, const double x[], double dxdt[],
    double* /*accFac - ignored*/) const
{
    coord::PosVelCyl p(x);
    if(x[0]<0) {    // R<0
        p.R = -p.R; // apply reflection
        p.phi += M_PI;
    }
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(p, NULL, &grad, &hess, tbegin + timeOffset);
    double Rinv = p.R!=0 ? 1/p.R : 0;  // avoid NAN in degenerate cases
    if(x[0]<0)
        grad.dR = -grad.dR;
    dxdt[0] = p.vR;
    dxdt[1] = p.vz;
    dxdt[2] = p.vphi * Rinv - Omega;
    dxdt[3] = -grad.dR + pow_2(p.vphi) * Rinv;
    dxdt[4] = -grad.dz;
    dxdt[5] = -(grad.dphi + p.vR*p.vphi) * Rinv;
    if(x[0]<0) {
        hess.dRdphi = -hess.dRdphi;
        hess.dRdz   = -hess.dRdz;
    }
    dxdt[6] = x[9];
    dxdt[7] = x[10];
    dxdt[8] = (-p.vphi * Rinv * x[6] + x[11]) * Rinv;
    dxdt[9] = (-hess.dR2 - pow_2(p.vphi * Rinv)) * x[6] - hess.dRdz * x[7] - hess.dRdphi * x[8]
        + 2*p.vphi * Rinv * x[11];
    dxdt[10]= -hess.dRdz * x[6] - hess.dz2 * x[7] - hess.dzdphi * x[8];
    dxdt[11]= -((hess.dRdphi + dxdt[5]) * x[6] + hess.dzdphi * x[7] + hess.dphi2 * x[8]
        + p.vphi * x[9] + p.vR * x[11]) * Rinv;
}

// the routine for integrating the orbit + variational equation in the given coordinate system,
// largely replicating the implementation of the methods orbit::OrbitIntegrator::run() and
// processTimestep() of corresponding runtime functions
template<typename CoordT>
void integrateOrbitVarEq(
    const coord::PosVelCar& initialConditions,
    const double totalTime,
    const double samplingInterval,
    const potential::BasePotential& potential,
    const double Omega,
    orbit::Trajectory& trajectory,
    std::vector< std::pair<double, double> >& logDevVec)
{
    double timeCurr = 0;
    OrbitIntegratorVarEq<CoordT> orbitIntegrator(potential, Omega, /*reference to*/ timeCurr);
    math::OdeStepperDOP853 stepper(orbitIntegrator/*, accuracy*/);
    // first 6 variables are always position/velocity, the rest is the deviation vector
    double vars[12] = { 0, 0, 0, 0, 0, 0,
        sqrt(7./90), sqrt(11./90), sqrt(13./90), sqrt(17./90), sqrt(19./90), sqrt(23./90) };
    coord::toPosVel<coord::Car, CoordT>(initialConditions).unpack_to(vars);
    stepper.init(vars);
    double addLogDevVec = 0;
    while(timeCurr < totalTime) {
        double timeStep = stepper.doStep(totalTime - timeCurr);
        double timeNext = timeCurr + timeStep;  // time at the end of this timestep
        const double ROUNDOFF = 1e-15;
        ptrdiff_t ibegin = static_cast<ptrdiff_t>(timeCurr / samplingInterval);
        ptrdiff_t iend   = static_cast<ptrdiff_t>(timeNext / samplingInterval * (1 + ROUNDOFF));
        double dtroundoff = ROUNDOFF * timeNext;
        trajectory.resize(iend + 1);
        logDevVec. resize(iend + 1);
        bool needToRescale = false;
        for(ptrdiff_t iout=ibegin; iout<=iend; iout++) {
            double tout = samplingInterval * iout, toffset = tout-timeCurr;
            if(tout >= timeCurr && tout <= timeNext + dtroundoff) {
                double devVec2 = 0;
                for(int d=0; d<6; d++) {
                    vars[d] = stepper.getSol(toffset, d);
                    devVec2 += pow_2(stepper.getSol(toffset, d+6));
                }
                trajectory[iout].first = toPosVelCar(coord::PosVelT<CoordT>(vars));
                trajectory[iout].second= tout;
                double logDV = log(devVec2) * 0.5;
                logDevVec [iout].first = tout;
                logDevVec [iout].second= logDV + addLogDevVec;
                needToRescale |= logDV > 300;
            }
        }
        if(needToRescale) {
            double devVec2 = 0;
            for(int d=0; d<12; d++) {
                vars[d] = stepper.getSol(timeStep, d);
                if(d>=6) devVec2 += pow_2(vars[d]);
            }
            double logDV = log(devVec2) * 0.5, mult = exp(-logDV);
            addLogDevVec += logDV;
            for(int d=0; d<6; d++)
                vars[d+6] *= mult;
            stepper.init(vars);
        }
        timeCurr = timeNext;
    }
}

// finally, the test suite itself
bool testLyapunov(const char* potParams, double Omega, bool expectChaotic)
{
    potential::PtrPotential pot = potential::createPotential(utils::KeyValueMap(potParams));
    coord::PosVelCar initCond(1., 1., 1., 0., 0., 0.);
    double orbitalPeriod = potential::T_circ(*pot, totalEnergy(*pot, initCond));  // characteristic time
    unsigned int trajSize = 20001;  // number of points in recorded trajectory
    double timeTotal = (trajSize/10) * orbitalPeriod;
    double samplingInterval = timeTotal / (trajSize-1);
    orbit::Trajectory
        trajectory6,              // from standard 6-dimensional ODE integrator
        trajectory6rev,           // same but reversing the time direction
        trajectory12car,          // from 12-dimensional ODE integrator (orbit+var.eq.)
        trajectory12cyl,          // same in cylindrical coords
        deviationVectors[6];      // deviation vectors reported by the 6d var.eq.solver
    std::vector< std::pair<double, double> >
        logDeviationVector12car,  // from the var.eq. solved by the 12d orbit integrator
        logDeviationVector12cyl;  // same in cylindrical coords
    // Lyapunov exponents and chaos onset times estimated from these arrays
    double lyap6[2], lyap6rev[2], lyap12car[2], lyap12cyl[2];
    // accuracy requirements are different for 6d and 12d, chosen so that the number of timesteps
    // taken by the ODE integrators are approximately equal (hence 6d has a tighter tolerance)
    orbit::OrbitIntParams par6(1e-9);

    // 6d ODE system with the variational equation solved by another specialized solver attached
    // to the RuntimeVariational function, which also outputs the evolution of all 6 deviation vectors
    // (this is the standard way of computing orbit derivatives or the Lyapunov exponent)
    {
        orbit::OrbitIntegrator<coord::Car> orbint(*pot, Omega, par6);
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeVariational(
            orbint, samplingInterval, /*output*/ deviationVectors, lyap6)));
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(
            orbint, samplingInterval, trajectory6)));
        orbint.init(initCond);
        orbint.run(timeTotal);
        // finalize the runtime functions (compute Lyapunov exponent from stored data) once they go out of scope
    }

    // same as above, but starting at nonzero initial time and integrating backward;
    // the chaotic properties of the orbit should be the same regardless of the direction of time
    {
        orbit::OrbitIntegrator<coord::Car> orbint(*pot, Omega, par6);
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeVariational(
            orbint, samplingInterval, /*no deviation vectors*/ NULL, lyap6rev)));
        orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new orbit::RuntimeTrajectory(
            orbint, samplingInterval, trajectory6rev)));
        orbint.init(initCond, timeTotal * 0.3);
        orbint.run(-timeTotal);
    }
    if(trajectory6.size() != trajSize || trajectory6rev.size() != trajSize) {
        std::cout << "incorrect length of trajectory" << err << '\n';
        return false;
    }

    // compare the evolution of deviation vectors (i.e. infinitesimally small perturbations
    // of the original trajectory) to six orbits with slightly perturbed initial conditions;
    // this comparison makes sense only on intervals shorter than the time it takes to grow
    // the small initial perturbations to a macroscopically large amplitude, which is determined below
    const double DEV_INIT = 1e-10, DEV_MAX = 1e-5, TOLERANCE = 1e-4;
    double timeFollow = timeTotal;
    for(int vec=0; vec<6; vec++) {
        if(deviationVectors[vec].size() != trajSize) {
            std::cout << "incorrect length of deviation vectors" << err << '\n';
            return false;
        }
        for(size_t s=0; s<deviationVectors[vec].size(); s++) {
            const coord::PosVelCar& pv = deviationVectors[vec][s].first;
            double mag = sqrt(pow_2(pv.x) + pow_2(pv.y) + pow_2(pv.z) +
                pow_2(pv.vx) + pow_2(pv.vy) + pow_2(pv.vz));
            if(mag > DEV_MAX / DEV_INIT) {
                timeFollow = fmin(timeFollow, deviationVectors[vec][s].second);
                break;
            }
        }
    }
    // now follow 6 slightly perturbed orbits for a timeFollow interval
    double maxDifference = 0;
    for(int vec=0; vec<6; vec++) {
        coord::PosVelCar initCondPerturb = initCond;
        combine(initCondPerturb, deviationVectors[vec][0].first, 1, DEV_INIT);
        orbit::Trajectory trajectory6perturb = orbit::integrateTraj(
            initCondPerturb, timeFollow, samplingInterval, *pot, Omega, par6);
        for(size_t i=0; i<trajectory6perturb.size(); i++) {
            // expected pos/vel from the evolution of this deviation vector
            coord::PosVelCar pointExpected = trajectory6[i].first;
            combine(pointExpected, deviationVectors[vec][i].first, 1, DEV_INIT);
            const coord::PosVelCar& pointActual = trajectory6perturb[i].first;
            double difference = sqrt(
                pow_2(pointExpected.x  - pointActual.x)  +
                pow_2(pointExpected.y  - pointActual.y)  +
                pow_2(pointExpected.z  - pointActual.z)  +
                pow_2(pointExpected.vx - pointActual.vx) +
                pow_2(pointExpected.vy - pointActual.vy) +
                pow_2(pointExpected.vz - pointActual.vz));
            maxDifference = fmax(maxDifference, difference);
        }
    }

    // 12d ODE system with the variational equation solved directly by the ODE integrator
    integrateOrbitVarEq<coord::Car>(
        initCond, timeTotal, samplingInterval, *pot, Omega, trajectory12car, logDeviationVector12car);
    orbit::calcLyapunovExponent(logDeviationVector12car, orbitalPeriod, /*output*/ lyap12car);
    integrateOrbitVarEq<coord::Cyl>(
        initCond, timeTotal, samplingInterval, *pot, Omega, trajectory12cyl, logDeviationVector12cyl);
    orbit::calcLyapunovExponent(logDeviationVector12cyl, orbitalPeriod, /*output*/ lyap12cyl);

    // print out log(devvec) as a function of time
    if( trajSize != trajectory12car.size() || trajSize != trajectory12cyl.size() ||
        trajSize != logDeviationVector12car.size() || trajSize != logDeviationVector12cyl.size() )
    {
        std::cout << "incorrect length of output arrays" << err << '\n';
        return false;
    }
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        static int testIndex = 0;
        std::ofstream strm(("test_orbit_lyapunov"+utils::toString(testIndex++)+".dat").c_str());
        strm << "#time\tx(6) y(6) z(6)\txcar(12) ycar(12) zcar(12)\txcyl(12) ycyl(12) zcyl(12)\t"
            "lnw6 lnw12car lnw12cyl\n";
        for(size_t i=0; i<trajSize; i++) {
            double logDeviationVector6 = -INFINITY;
            for(int vec=0; vec<6; vec++) {
                const coord::PosVelCar& p = deviationVectors[vec][i].first;
                // avoid (or rather, delay) overflow in computing the norm
                const double mul = 1e-150;
                double norm = 0.5 * log(
                    pow_2(p.x  * mul) + pow_2(p.y  * mul) + pow_2(p.z  * mul) +
                    pow_2(p.vx * mul) + pow_2(p.vy * mul) + pow_2(p.vz * mul)) - log(mul);
                logDeviationVector6 = fmax(logDeviationVector6, norm);
            }
            strm << /*time*/ trajectory6[i].second << '\t' <<
            trajectory6    [i].first.x << ' ' <<
            trajectory6    [i].first.y << ' ' <<
            trajectory6    [i].first.z << '\t'<<
            trajectory12car[i].first.x << ' ' <<
            trajectory12car[i].first.y << ' ' <<
            trajectory12car[i].first.z << '\t'<<
            trajectory12cyl[i].first.x << ' ' <<
            trajectory12cyl[i].first.y << ' ' <<
            trajectory12cyl[i].first.z << '\t'<<
            logDeviationVector6        << ' ' <<
            logDeviationVector12car[i].second << ' ' <<
            logDeviationVector12cyl[i].second << '\n';
        }
    }
    double
    Einit    = totalEnergy(*pot, initCond) - Omega * Lz(initCond),   // Jacobi energy
    Eend6    = totalEnergy(*pot, trajectory6.    back().first) - Omega * Lz(trajectory6.    back().first),
    Eend12car= totalEnergy(*pot, trajectory12car.back().first) - Omega * Lz(trajectory12car.back().first),
    Eend12cyl= totalEnergy(*pot, trajectory12cyl.back().first) - Omega * Lz(trajectory12cyl.back().first);
    std::cout << "Potential: " << potParams <<
        ";  pattern speed: "   << Omega << 
        ";  orbital period: "  << orbitalPeriod <<
        ";  energy error: 6d=" << Eend6-Einit <<
        " 12d(car)=" << Eend12car-Einit << " 12d(cyl)=" << Eend12cyl-Einit <<
        ";  max.deviation from linearized orbit during time " << timeFollow << " is " << maxDifference <<
        ";  Lyapunov exponent:  6d=" << lyap6[0] << " 6d(reverse)=" << lyap6rev[0] <<
        " 12d(car)=" << lyap12car[0] << " 12d(cyl)=" << lyap12cyl[0];

    // check if various estimators of the Lyapunov exponent are in agreement
    bool ok = expectChaotic ?
        (lyap6[0] >  0 && lyap6rev[0] >  0 && lyap12car[0] >  0 && lyap12cyl[0] >  0) :
        (lyap6[0] == 0 && lyap6rev[0] == 0 && lyap12car[0] == 0 && lyap12cyl[0] == 0);
    ok &= maxDifference < TOLERANCE;
    if(ok)
        std::cout << '\n';
    else
        std::cout << err << '\n';
    return ok;
}

int main()
{
    bool ok = testOde();
    ok &= testLyapunov("type=Logarithmic scaleRadius=0 axisRatioY=0.9 axisRatioZ=0.8", 0.0, true);
    ok &= testLyapunov("type=Logarithmic scaleRadius=1 axisRatioY=0.9 axisRatioZ=0.8", 0.0, false);
    ok &= testLyapunov("type=Logarithmic scaleRadius=1 axisRatioY=0.9 axisRatioZ=0.8", 0.1, true);
    ok &= testLyapunov("type=Logarithmic scaleRadius=1 axisRatioY=0.9 axisRatioZ=0.8", 0.2, false);
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}