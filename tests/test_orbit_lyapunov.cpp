/** \file    example_lyapunov.cpp
    \author  Eugene Vasiliev
    \date    2008-2017

    This example demonstrates how to augment the orbit integration in AGAMA with
    the computation of the (largest) Lyapunov exponent, which is a measure of orbit chaoticity.
    The task is divided into three routines:
    - the class that provides the right-hand side of the differential equation;
    - the routine that computes and stores the orbit and the magnitude of the deviation vector;
    - the routine that estimates the Lyapunov exponent from the time evolution of the dev.vec.
*/
#include "orbit.h"
#include "orbit_lyapunov.h"
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
class RHS: public math::IOdeSystem, public math::IOde2System {
    void rhs(const double t, double mat[]) const;
public:
    // IOdeSystem interface
    virtual void eval(const double t, const double x[], double dxdt[]) const;
    // IOde2System interface
    virtual void eval(const double t, double mat[]) const { rhs(t, mat); }
    virtual unsigned int size() const { return NDIM*2; }
};

// NDIM=1: trivial system x''=-3*x, analytic solution: x=cos(sqrt(3)*t)
template<> void RHS<1>::rhs(const double, double mat[]) const {
    mat[0] = -3;
}
template<> void RHS<1>::eval(const double, const double x[], double dxdt[]) const {
    dxdt[0] = x[1];
    dxdt[1] =-x[0]*3;
}
// NDIM=2: more complicated system with time-dependent coefs, analytical solution in terms of Bessel fncs
template<> void RHS<2>::rhs(const double t, double mat[]) const {
    mat[0] = mat[3] = cos(t);
    mat[2] = sin(t);
    mat[1] = -mat[2];
}
template<> void RHS<2>::eval(const double t, const double x[], double dxdt[]) const {
    double mat[4];
    rhs(t, mat);
    dxdt[0] = x[2];
    dxdt[1] = x[3];
    dxdt[2] = mat[0] * x[0] + mat[1] * x[1];
    dxdt[3] = mat[2] * x[0] + mat[3] * x[1];
}


const int numPoints = 36;
const double exactSolution2d[numPoints*3] = {   // exact analytic solution for the 2d case
    // time,  first variable,      second variable
 // 1.00000,  1.40299965153760788, 1.33523553025283437,
    2.00000,  1.12844004088278717, 4.21807704493242670,
    2.03125,  1.06864419748133139, 4.32229607353448196,
    2.06250,  1.00460433059309503, 4.42557386137590566,
    2.09375,  0.93629257838877154, 4.52767554672774388,
    2.12500,  0.86369474689976034, 4.62836066994521750,
    2.15625,  0.78681092667170627, 4.72738401238024595,
    2.18750,  0.70565605578340031, 4.82449649782909292,
    2.21875,  0.62026042290958394, 4.91944615305437966,
    2.25000,  0.53067010431638711, 5.01197912326816993,
    2.28125,  0.43694732895782223, 5.10184073781988986,
    2.31250,  0.33917076619113891, 5.18877662070718437,
    2.34375,  0.23743573104723507, 5.27253383992725643,
    2.37500,  0.13185430247810225, 5.35286208911867567,
    2.40625,  0.02255535055392235, 5.42951489441688942,
    2.43750, -0.09031553080555468, 5.50225083896837516,
    2.46875, -0.20659618831175633, 5.57083479712589193,
    2.50000, -0.32610817302596892, 5.63503916998754944,
    2.53125, -0.44865710457575111, 5.69464511365183921,
    2.56250, -0.57403313632445597, 5.74944375134512297,
    2.59375, -0.70201152111418268, 5.79923736044236389,
    2.62500, -0.83235327634522169, 5.84384052535028942,
    2.65625, -0.96480594628459452, 5.88308124725790336,
    2.68750, -1.09910445862207776, 5.91680200188453305,
    2.71875, -1.23497207142392417, 5.94486073657152096,
    2.75000, -1.37212140578234411, 5.96713179837023649,
    2.78125, -1.51025555863273454, 5.98350678517511831,
    2.81250, -1.64906928942061884, 5.99389531243360129,
    2.84375, -1.78825027355603060, 5.99822568853150140,
    2.87500, -1.92748041490401247, 5.99644549259804547,
    2.90625, -2.06643720893486780, 5.98852204919343792,
    2.921875,-2.13571145764725492, 5.98225189030521965,
    2.93750, -2.20479514760500057, 5.97444279512681623,
    2.953125,-2.27364735682480398, 5.96509637764853464,
    2.96875, -2.34222715656602641, 5.95421553449583807,
    2.984375,-2.41049365731975413, 5.94180444658386952,
    3.00000, -2.47840605491285111, 5.92786857893227925 };

template<class Solver, int NDIM>
void testSolver(double dt, const double exactSolution[], double& errStep, double& errInt)
{
    RHS<NDIM> rhs;
    double t = 0.;
    double x[] = {1., 0., 0., 1.};
    Solver solver(rhs);
    solver.init(x);
    int ind=0;
    errStep = 0, errInt = 0;
    while(t<exactSolution[(numPoints-1)*(NDIM+1)]) {
        solver.doStep(dt);
        t+=dt;
        while(ind<numPoints && exactSolution[ind*(NDIM+1)] <= t) {
            double tcheck = exactSolution[ind*(NDIM+1)], err = 0;
            for(int d=0; d<NDIM; d++)
                err = fmax(err, fabs(solver.getSol(tcheck, d) - exactSolution[ind*(NDIM+1)+1+d]));
            if(t == tcheck)
                errStep = fmax(errStep, err);
            else
                errInt  = fmax(errInt,  err);
            ind++;
        }
    }
}

// test the accuracy of ODE solvers for systems with known analytic solutions:
// check the error as a function of timestep and compare to the expected scalings
bool testOde()
{
    double exactSolution1d[2*numPoints];
    for(int i=0; i<numPoints; i++) {
        exactSolution1d[i*2] = exactSolution2d[i*3];  // time
        exactSolution1d[i*2+1] = cos(M_SQRT3*exactSolution1d[i*2]);
    }
    std::cout << "Testing the accuracy of ODE integrators: GL3, GL4, DOP853.\n"
    "Print the error in numerical solution as a function of timestep: \n"
    "(s) evaluated at the end of each step (the accuracy of solution itself), and \n"
    "(i) evaluated at some points inside the step (the accuracy of interpolation / dense output).\n"
    "timestep GL3(s)  GL3(i)  GL4(s)  GL4(i)  DOP(s)  DOP(i)\n";
    bool ok = true;
    std::cout << "1d system\n";
    for(double dt=1; dt>=0.0625; dt*=0.5) {
        double err3s, err3i, err4s, err4i, err8s, err8i;
        testSolver<math::Ode2SolverGL3<1>, 1>(dt, exactSolution1d, err3s, err3i);
        testSolver<math::Ode2SolverGL4<1>, 1>(dt, exactSolution1d, err4s, err4i);
        testSolver<math::OdeSolverDOP853 , 1>(dt, exactSolution1d, err8s, err8i);
        std::cout << utils::pp(dt,6) + "  " +
        utils::pp(err3s,7) + ' ' + utils::pp(err3i,7) + ' ' +
        utils::pp(err4s,7) + ' ' + utils::pp(err4i,7) + ' ' +
        utils::pp(err8s,7) + ' ' + utils::pp(err8i,7);
        if( !(err3s < 3e-4 * pow(dt, 6) && err3i < 2e-3 * pow(dt, 5) &&
              err4s < 1e-4 * pow(dt, 8) && err4i < 1e-4 * pow(dt, 6) &&
              err8s < 5e-5 * pow(dt, 8) && err8i < 1e-4 * pow(dt, 7) ))
        {
            ok = false;
            std::cout << err;
        } else
            std::cout << '\n';
    }
    std::cout << "2d system\n";
    for(double dt=1; dt>=0.0625; dt*=0.5) {
        double err3s, err3i, err4s, err4i, err8s, err8i;
        testSolver<math::Ode2SolverGL3<2>, 2>(dt, exactSolution2d, err3s, err3i);
        testSolver<math::Ode2SolverGL4<2>, 2>(dt, exactSolution2d, err4s, err4i);
        testSolver<math::OdeSolverDOP853 , 2>(dt, exactSolution2d, err8s, err8i);
        std::cout << utils::pp(dt,6) + "  " +
        utils::pp(err3s,7) + ' ' + utils::pp(err3i,7) + ' ' +
        utils::pp(err4s,7) + ' ' + utils::pp(err4i,7) + ' ' +
        utils::pp(err8s,7) + ' ' + utils::pp(err8i,7);
        if( !(err3s < 1e-3 * pow(dt, 6) && err3i < 5e-3 * pow(dt, 5) &&
              err4s < 3e-5 * pow(dt, 8) && err4i < 5e-4 * pow(dt, 6) &&
              err8s < 3e-5 * pow(dt, 8) && err8i < 2e-4 * pow(dt, 7) ))
        {
            ok = false;
            std::cout << err;
        } else
            std::cout << '\n';
    }
    std::cout << "Expected:  dt^6    dt^5    dt^8    dt^6    dt^8    dt^7\n\n";
    return ok;
}


//----------------------------------------------//
// PART 2: testing the Lyapunov chaos estimator //
//----------------------------------------------//

bool testLyapunov(const char* potParams, double Omega, bool expectChaotic)
{
    potential::PtrPotential pot = potential::createPotential(utils::KeyValueMap(potParams));

    coord::PosVelCar initCond(1., 1., 1., 0., 0., 0.);
    double orbitalPeriod = potential::T_circ(*pot, totalEnergy(*pot, initCond));  // characteristic time
    double timeTotal = 1000. * orbitalPeriod;       // integration time in orbital periods
    double samplingInterval = 0.1 * orbitalPeriod;  // store output with sufficient temporal resolution
    std::vector< std::pair<coord::PosVelCar, double> >
        trajectory6,                                // from standard 6-dimensional ODE integrator
        trajectory12;                               // from 12-dimensional ODE integrator (orbit+var.eq.)
    std::vector<double> logDeviationVector6i;       // from the internal var.eq.solver in the 6d case
    std::vector<double> logDeviationVector12i;      // same for the 12d case
    std::vector<double> logDeviationVector12o;      // from the var.eq. solved by the 12d orbit integrator
    double lyap6i, lyap12i, lyap12o;                // Lyapunov exp. estimated from these three arrays
    orbit::RuntimeFncArray rfnc;                    // list of runtime functions
    orbit::OrbitIntParams par6(1e-10), par12(1e-9); // accuracy requirements are different for 6d and 12d,
    // chosen so that the number of timesteps taken by the ODE integrators are approximately equal

    // 6d ODE system with the variational equation solved by another solver
    // (internal to the RuntimeLyapunov function with the template parameter UseInternalVarEqSolver=true)
    rfnc.push_back(orbit::PtrRuntimeFnc(new orbit::RuntimeLyapunov<true>(
        *pot, samplingInterval, /*output*/ lyap6i,  &logDeviationVector6i)));
    rfnc.push_back(orbit::PtrRuntimeFnc(
        new orbit::RuntimeTrajectory<coord::Car>(samplingInterval, trajectory6)));
    orbit::integrate(initCond, timeTotal, orbit::OrbitIntegratorRot(*pot, Omega), rfnc, par6);
    rfnc.clear();  // finalize the runtime functions (compute Lyapunov exponent from stored data)

    // 12d ODE system with the variational equation solved directly by the orbit integrator
    // and analyzed by the RuntimeLyapunov function with UseInternalVarEqSolver=false;
    // for comparison, we also attach another RuntimeLyapunov function with an internal var.eq.solver,
    // similar to the 6d case, so that we have two different estimates of the deviation vector evolution.
    // The latter will be somewhat different from the 6d case because the adaptive timesteps chosen by
    // the orbit integrator depend on the ODE system (6d and 12d have different error estimates).
    rfnc.push_back(orbit::PtrRuntimeFnc(new orbit::RuntimeLyapunov<true>(
        *pot, samplingInterval, /*output*/ lyap12i, &logDeviationVector12i)));
    rfnc.push_back(orbit::PtrRuntimeFnc(new orbit::RuntimeLyapunov<false>(
        *pot, samplingInterval, /*output*/ lyap12o, &logDeviationVector12o)));
    rfnc.push_back(orbit::PtrRuntimeFnc(
        new orbit::RuntimeTrajectory<coord::Car>(samplingInterval, trajectory12)));
    orbit::integrate(initCond, timeTotal, orbit::OrbitIntegratorVarEq(*pot, Omega), rfnc, par12);
    rfnc.clear();

    // print out log(devvec) as a function of time
    size_t size =  trajectory6.size();
    assert(size == trajectory12.size() && size == logDeviationVector6i.size() &&
        size == logDeviationVector12i.size() && size == logDeviationVector12o.size());
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        static int testIndex = 0;
        std::ofstream strm(("test_orbit_lyapunov"+utils::toString(testIndex++)+".dat").c_str());
        strm << "#time\tx(6) y(6) z(6)\tx(12) y(12) z(12)\tlnw6i lnw12i lnw12o\n";
        for(size_t i=0; i<size; i++)
            strm << /*time*/ trajectory6[i].second << '\t' <<
            trajectory6 [i].first.x << ' ' <<
            trajectory6 [i].first.y << ' ' <<
            trajectory6 [i].first.z << '\t'<<
            trajectory12[i].first.x << ' ' <<
            trajectory12[i].first.y << ' ' <<
            trajectory12[i].first.z << '\t'<<
            logDeviationVector6i[i] << ' ' <<
            logDeviationVector12i[i]<< ' ' <<
            logDeviationVector12o[i]<< '\n';
    }
    double Einit = totalEnergy(*pot, initCond) - Omega * Lz(initCond);   // Jacobi energy
    double Eend6 = totalEnergy(*pot, trajectory6. back().first) - Omega * Lz(trajectory6. back().first);
    double Eend12= totalEnergy(*pot, trajectory12.back().first) - Omega * Lz(trajectory12.back().first);
    std::cout << "Potential: " << potParams <<
        ";  pattern speed: "   << Omega << 
        ";  orbital period: "  << orbitalPeriod <<
        ";  energy error: 6d=" << Eend6-Einit << " 12d=" << Eend12-Einit <<
        ";  Lyapunov exponent: ";
    if(Omega==0)  // the following two are incorrect in a rotating frame
        std::cout << " 6d,int=" << lyap6i << " 12d,int=" << lyap12i;
    std::cout << " 12d,orb=" << lyap12o;

    // check if various estimators of the Lyapunov exponent are in agreement
    // (the ones based on the internally evolved var.eq. are not expected to work in rotating frame,
    // so are ignored if Omega!=0)
    bool ok = expectChaotic ?
        (Omega!=0 || (lyap6i >  0 && lyap12i >  0)) && lyap12o >  0 :
        (Omega!=0 || (lyap6i == 0 && lyap12i == 0)) && lyap12o == 0;
    if(ok)
        std::cout << '\n';
    else
        std::cout << err;
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