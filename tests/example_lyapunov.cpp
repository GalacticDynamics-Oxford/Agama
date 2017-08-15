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
#include "potential_analytic.h"
#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>

/** The right-hand side (rhs) of the differential equation dX/dt = f(X).
    In this example, X is a 12-dimensional vector:
    first 6 components are the ordinary position and velocity in cartesian coordinates,
    and the rest are the corresponding components of the deviation vector,
    which measures the distance between the orbit and its infinitesimally close counterpart.
    The evolution of the deviation vector is described by the variational equation,
    which contains second derivatives of the potential computed along the original orbit.
    This class provides the interface used by the ODE solver to evolve the first-order ODE.
*/
class OrbitIntegratorVarEq: public math::IOdeSystem {
    const potential::BasePotential& pot;  ///< reference to the potential used in orbit integration
public:
    OrbitIntegratorVarEq(const potential::BasePotential& _pot) : pot(_pot) {}

    /// compute the time derivative of the position/velocity vector and the deviation vector at time t
    virtual void eval(const double /*t*/, const math::OdeStateType& x, math::OdeStateType& dxdt) const
    {
        coord::GradCar grad;
        coord::HessCar hess;
        pot.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad, &hess);
        // time derivative of position
        dxdt[0] = x[3];
        dxdt[1] = x[4];
        dxdt[2] = x[5];
        // time derivative of velocity
        dxdt[3] = -grad.dx;
        dxdt[4] = -grad.dy;
        dxdt[5] = -grad.dz;
        // time derivative of position part of the deviation vector
        dxdt[6] = x[9];
        dxdt[7] = x[10];
        dxdt[8] = x[11];
        // time derivative of velocity part of the deviation vector
        dxdt[9] = -hess.dx2  * x[6] - hess.dxdy * x[7] - hess.dxdz * x[8];
        dxdt[10]= -hess.dxdy * x[6] - hess.dy2  * x[7] - hess.dydz * x[8];
        dxdt[11]= -hess.dxdz * x[6] - hess.dydz * x[7] - hess.dz2  * x[8];
    }

    /** The size of the position/velocity vector plus the deviation vector */
    virtual unsigned int size() const { return 12; }

    /** not presently used */
    virtual bool isStdHamiltonian() const { return false; }
};

/// compute the logarithm of the L2-norm of a 6d vector
double logMagnitude(const double x[6])
{
    double norm = 0.;
    for(int i=0; i<6; i++)
        norm += pow_2(x[i]);
    return 0.5 * log(norm);
}

/** Replacement of the orbit integration routine that stores both the orbit and
    the logarithm of the deviation vector magnitude.
    It uses the standard ODE solver DOP853, feeds it with the OrbitIntegratorVarEq class
    providing the equation to integrate, and stores the data at regular intervals of time.
    In addition, since the deviation vector may grow exponentially, it is necessary to
    renormalize it in order to stay within the range of floating-point variable.
    \param[in]  initialConditions  is a 6d position/velocity vector;
    \param[in]  totalTime  is the integration time;
    \param[in]  samplingInterval  is the time between successive outputs of trajectory
    (has no relation to the internal timestep of the ODE solver);
    \param[in]  pot  is the instance of potential in which the orbit is computed;
    \param[out] trajectory  will contain the position/velocity sampled at regular intervals
    of time, starting from the initial conditions; will be resized as needed.
    \param[out] logDeviationVector  will contain the logarithm of the L2-norm of the deviation
    vector, which is normalized to unity at t=0.
*/
void integrateVarEq(const coord::PosVelCar& initialConditions, const double totalTime,
    const double samplingInterval, const potential::BasePotential& pot,
    /*output*/ std::vector<coord::PosVelCar>& trajectory, std::vector<double>& logDeviationVector)
{
    OrbitIntegratorVarEq orbint(pot);       // the instance of ODE system (provides the rhs of diff.eq.)
    math::OdeSolverDOP853 odesolver(orbint, /*accuracy*/ 1e-8);  // the instance of ODE integrator
    math::OdeStateType vars(orbint.size()); // storage for the pos/vel vector and the deviation vector
    initialConditions.unpack_to(&vars[0]);  // initialize the position/velocity values
    for(int i=6; i<12; i++)                 // initialize the deviation vector by random values
        vars[i] = math::random();           // (maybe better to do it deterministically?)
    odesolver.init(vars);                   // set up solver internal state
    trajectory.clear();
    logDeviationVector.clear();
    double timeCurr = 0.;
    double logDevVec = logMagnitude(&vars[6]);  // current magnitude of the (rescaled) dev.vec.
    double offsetLogDevVec = -logDevVec;    // offset added to the output values of log(dev.vec.)
    while(timeCurr < totalTime) {
        // perform one internal step of ODE integrator
        if(odesolver.doStep() <= 0.) {  // signal of error
            std::cout << "Warning: timestep is zero at t=" << timeCurr << std::endl;
            break;
        }
        timeCurr = fmin(odesolver.getTime(), totalTime);
        // store trajectory and magnitude of the deviation vector at regular intervals of time
        while(samplingInterval * trajectory.size() <= timeCurr) {
            odesolver.getSol(samplingInterval * trajectory.size(), &vars[0]);
            trajectory.push_back(coord::PosVelCar(&vars[0]));
            logDevVec = logMagnitude(&vars[6]);
            logDeviationVector.push_back(logDevVec + offsetLogDevVec);
        }
        // if necessary, renormalize the deviation vector and reinit the ODE solver
        if(logDevVec > 50.) {
            odesolver.getSol(timeCurr, &vars[0]); // obtain the variables at the end of timestep
            logDevVec = logMagnitude(&vars[6]);   // current log-magnitude of the deviation vector
            for(int i=6; i<12; i++)               // rescale the deviation vector back to unit magnitude
                vars[i] *= exp(-logDevVec);
            offsetLogDevVec += logDevVec;         // compensate the rescaling by increasing the offset
            logDevVec = 0.;                       // reset the current magnitude to unity
            odesolver.init(vars);                 // reinit the internal state of the ODE solver
            //std::cout << "At time "<<timeCurr<<" rescaling dev.vec.\n";
        }
    }
}

/** Estimate the (largest) Lyapunov exponent from the time series of the (log) of deviation vector 'w'
    computed along the orbit.
    If the orbit is regular, |w| grows at most linearly with time, whereas if it is chaotic,
    then eventually |w| starts to grow exponentially, after some initial 'hidden' period.
    We keep track of the running average of |w|/t, and when it exceeds some threshold,
    this is taken to be the transition to the exponential growth regime.
    From that point on, we collect the average of ln(|w|)/t as a proxy for the Lyapunov exponent,
    and return the value normalized to the characteristic time (orbital period).
    If no such transition was detected, then the return value is zero.
    \param[in]  logDeviationVector  is the array of collected values of ln(|w|) during orbit integration;
    \param[in]  samplingInterval  is the time between consecutive values;
    \param[in]  orbitalPeriod  is the characteristic time of the orbit, needed both to define
    a suitable interval for the running average of ln(|w|), because it exhibits large fluctuations
    over the timescale of orbital period, and for normalizing the output value of the Lyapunov exponent
    to the characteristic orbital time (it may then be taken as a direct measure of chaoticity -
    values around unity indicate strongly chaotic orbits, and much less than unity - weakly chaotic).
    \return  the estimate of the normalized Lyapunov exponent, or zero if the orbit seems to be regular.
*/
double calcLargestLyapunovExponent(const std::vector<double>& logDeviationVector,
    const double samplingInterval, const double orbitalPeriod)
{
    double sumavg=0;     // to calculate average value of ln(|w|/t) over the linear-growth interval
    int cntavg=0;
    double sumlambda=0;  // to calculate average value of ln(|w|)/t over the exponential growth interval
    int cntlambda=0;
    // number of samples in the averaging interval (roughly orbital period * 2)
    int timeblock=static_cast<int>(2 * orbitalPeriod / samplingInterval);
    std::vector<double> interval(timeblock);
    bool chaotic=false;
    for(unsigned int t=timeblock, size=logDeviationVector.size(); t<size; t+=timeblock) {
        int cnt = std::min<int>(timeblock, size-t);
        if(!chaotic) {
            // compute the median value of ln(|w|/t) over 'timeblock' points to avoid outliers
            interval.resize(cnt);
            for(int i=0; i<cnt; i++)
                interval[i] = (logDeviationVector[i+t] - log((i+t) * samplingInterval));
            std::nth_element(interval.begin(), interval.begin() + cnt/2, interval.begin() + cnt);
            double median = interval[cnt/2];
            sumavg += median;
            cntavg++;
            if((cntavg>4) && (median > sumavg/cntavg + 2))
                // from here on, assume that we got into the asymptotic regime (t-->infinity)
                // and start collecting data for estimating lambda
                chaotic=true;
        }
        if(chaotic) {
            for(int i=0; i<cnt; i++) {
                sumlambda += logDeviationVector[i+t]/(i+t);
                cntlambda++;
            }
        }
    }
    if(chaotic)
        return (sumlambda / cntlambda * orbitalPeriod / samplingInterval);
    else
        return 0.;  // chaotic behavior not detected, return zero estimate for the Lyapunov exponent
}

int main()
{
    potential::Logarithmic pot(/*amplitude*/ 1., /*core radius*/ 0., /*y/x*/ 0.9, /*z/x*/ 0.8);
    coord::PosVelCar initCond(1., 1., 1., 0., 0., 0.);
    double orbitalPeriod = potential::T_circ(pot, totalEnergy(pot, initCond));  // characteristic time
    double timeTotal = 1000. * orbitalPeriod;        // integrate for 1000 orbital times
    double samplingInterval = 0.05 * orbitalPeriod;  // store output with sufficient temporal resolution
    std::vector<coord::PosVelCar> trajectory;
    std::vector<double> logDeviationVector;
    integrateVarEq(initCond, timeTotal, samplingInterval, pot, /*output*/ trajectory, logDeviationVector);
    double lyap = calcLargestLyapunovExponent(logDeviationVector, samplingInterval, orbitalPeriod);
    // print out log(devvec) as a function of time
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("example_lyapunov.dat");
        strm << "#time  log(devVec)\n";
        for(size_t i=0; i<logDeviationVector.size(); i++)
            strm << (i*samplingInterval/orbitalPeriod) << "\t" << logDeviationVector[i] << "\n";
    }
    std::cout << "Orbital period: " << orbitalPeriod << ", Lyapunov exponent: " << lyap << "\n";
}
