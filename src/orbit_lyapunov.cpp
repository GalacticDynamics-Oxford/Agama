#include "orbit_lyapunov.h"
#include "potential_utils.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace orbit{

namespace{
/// initialize the deviation vector always with the same sequence (not sure if it's optimal);
/// the choice below produces comparable and incommensurable values of components, and has a unit norm
static const double DEV_VEC_INIT[6] =
{ sqrt(7./90), sqrt(11./90), sqrt(13./90), sqrt(17./90), sqrt(19./90), sqrt(23./90) };

/// compute the logarithm of the L2-norm of a 6d vector
double logMagnitude(const double x[6])
{
    double norm = 0.;
    for(int i=0; i<6; i++)
        norm += pow_2(x[i]);
    return 0.5 * log(norm);
}
}  // internal ns


void OrbitIntegratorVarEq::eval(const double /*t*/, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    coord::HessCar hess;
    potential.eval(coord::PosCar(x[0], x[1], x[2]), NULL, &grad, &hess);
    // time derivative of position
    dxdt[0] = x[3] + Omega * x[1];
    dxdt[1] = x[4] - Omega * x[0];
    dxdt[2] = x[5];
    // time derivative of velocity
    dxdt[3] = -grad.dx + Omega * x[4];
    dxdt[4] = -grad.dy - Omega * x[3];
    dxdt[5] = -grad.dz;
    // time derivative of position part of the deviation vector
    dxdt[6] = x[9]  + Omega * x[7];
    dxdt[7] = x[10] - Omega * x[6];
    dxdt[8] = x[11];
    // time derivative of velocity part of the deviation vector
    dxdt[9] = -hess.dx2  * x[6] - hess.dxdy * x[7] - hess.dxdz * x[8] + Omega * x[10];
    dxdt[10]= -hess.dxdy * x[6] - hess.dy2  * x[7] - hess.dydz * x[8] - Omega * x[9];
    dxdt[11]= -hess.dxdz * x[6] - hess.dydz * x[7] - hess.dz2  * x[8];
}


template<bool UseInternalVarEqSolver>
RuntimeLyapunov<UseInternalVarEqSolver>::RuntimeLyapunov(
    const potential::BasePotential& _potential, double _samplingInterval,
    double& _outputLyapunovExponent, std::vector<double>* outputLogDeviationVector)
:
    varEqSolver(*this),
    orbitSolver(NULL),   // not available yet - will be assigned during each timestep
    potential(_potential),
    samplingInterval(_samplingInterval),
    outputLyapunovExponent(_outputLyapunovExponent),
    logDeviationVector(  // alias to either the external or internal arrays that store the deviation vector
        outputLogDeviationVector==NULL ? logDeviationVectorInternal : *outputLogDeviationVector),
    addLogDevVec(0.),    // initially the deviation vector is normalized to unity
    orbitalPeriod(NAN)   // not known yet - will be assigned on the first timestep
{
    logDeviationVector.clear();
}

template<bool UseInternalVarEqSolver>
void RuntimeLyapunov<UseInternalVarEqSolver>::eval(const double t, double mat[]) const
{
    assert(orbitSolver != NULL && UseInternalVarEqSolver &&
        "RuntimeLyapunov::eval must be called internally from processTimestep");
    coord::PosCar pos(orbitSolver->getSol(t, 0), orbitSolver->getSol(t, 1), orbitSolver->getSol(t, 2));
    coord::HessCar hess;
    potential.eval(pos, NULL, NULL, &hess);
    mat[0] = -hess.dx2;
    mat[1] = -hess.dxdy;
    mat[2] = -hess.dxdz;
    mat[3] = -hess.dxdy;
    mat[4] = -hess.dy2;
    mat[5] = -hess.dydz;
    mat[6] = -hess.dxdz;
    mat[7] = -hess.dydz;
    mat[8] = -hess.dz2;
}

template<bool UseInternalVarEqSolver>
StepResult RuntimeLyapunov<UseInternalVarEqSolver>::processTimestep(
    const math::BaseOdeSolver& solver, const double tbegin, const double tend, double vars[])
{
    // assign the orbital period on the first timestep
    if(!isFinite(orbitalPeriod) /*initially it was set to NAN*/) {
        orbitalPeriod = T_circ(potential, totalEnergy(potential, coord::PosVelCar(vars)));
        if(!isFinite(orbitalPeriod))  // e.g. the orbit is unbound,
            orbitalPeriod = 0;        // so no meaningful Lyapunov exponent can be computed anyway
        if(UseInternalVarEqSolver) {
            varEqSolver.init(DEV_VEC_INIT);
            return SR_CONTINUE;
        } else {
            // initialize the deviation vector that is stored in the ODE variables of
            // the orbit integrator, starting from index 6
            for(int d=0; d<6; d++)
                vars[d+6] = DEV_VEC_INIT[d];
            return SR_REINIT;
        }
    }

    if(UseInternalVarEqSolver) {
        // perform one step of the internal variational equation solver
        // to compute the evolution of the deviation vector during the current timestep.
        // varEqSolver calls our 'eval()' method, which needs the access to the orbitSolver
        // to retrieve the orbit trajectory during this timestep; hence we temporarily initialize
        // the internal pointer to that solver and then clear it again (not a very satisfactory design...)
        orbitSolver = &solver;
        varEqSolver.doStep(tend-tbegin);
        orbitSolver = NULL;
    } else {
        // otherwise the variational equation is solved by the orbit integrator,
        // with the r.h.s. provided by OrbitIntegratorVarEq or a similar class;
        // the number of variables must be [at least] 12
        assert(solver.size() >= 12 && "RuntimeLyapunov: orbit integrator must handle >=12 variables");
    }

    // store the deviation vector at regular intervals of time and check its magnitude
    bool needToRescale = false;
    while(samplingInterval * logDeviationVector.size() <= tend) {
        double tsamp = samplingInterval * logDeviationVector.size();
        double devVec[6];   // retrieved either from the internal varEqSolver or from the orbit integrator
        for(int d=0; d<6; d++)
            devVec[d] = UseInternalVarEqSolver ? varEqSolver.getSol(tsamp, d) : solver.getSol(tsamp, d+6);
        double logDevVec = logMagnitude(devVec);
        logDeviationVector.push_back(logDevVec + addLogDevVec);
        needToRescale |= logDevVec > 100;
    }

    // if the orbit is chaotic, the magnitude of the deviation vector grows exponentially,
    // and we need to renormalize it every now and then to avoid floating-point overflows
    if(needToRescale) {
        double devVec[6];
        for(int d=0; d<6; d++)
            devVec[d] = UseInternalVarEqSolver ? varEqSolver.getSol(tend, d) : vars[d+6];
        double logDevVec = logMagnitude(devVec), mult = exp(-logDevVec);
        addLogDevVec += logDevVec;
        if(UseInternalVarEqSolver) {
            for(int d=0; d<6; d++)
                devVec[d] *= mult;     // bring it back to unit norm
            varEqSolver.init(devVec);  // and reinit the internal variational equation solver
        } else {
            for(int d=0; d<6; d++)
                vars[d+6] *= mult;     // bring it back to unit norm
            return SR_REINIT;          // and reinit the orbit integrator itself
        }
    }

    return SR_CONTINUE;
}


double calcLyapunovExponent(const std::vector<double>& logDeviationVector,
    const double samplingInterval, const double orbitalPeriod)
{
    if(samplingInterval<=0 || orbitalPeriod<=0)
        return NAN;
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

// compile the two template instantiations
template class RuntimeLyapunov<true>;
template class RuntimeLyapunov<false>;

}  // namespace orbit
