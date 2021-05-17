#include "orbit_lyapunov.h"
#include "potential_utils.h"
#include <cassert>
#include <cmath>
#include <algorithm>

namespace orbit{

namespace{

/// roundoff tolerance in the sampling interval calculation
const double ROUNDOFF = 10*DBL_EPSILON;

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


RuntimeLyapunov::RuntimeLyapunov(
    BaseOrbitIntegrator& orbint,
    double _samplingInterval,
    double& _outputLyapunovExponent, std::vector<double>* outputLogDeviationVector)
:
    BaseRuntimeFnc(orbint),
    varEqSolver(*this),
    samplingInterval(_samplingInterval),
    outputLyapunovExponent(_outputLyapunovExponent),
    logDeviationVector(  // alias to either the external or internal arrays that store the deviation vector
        outputLogDeviationVector==NULL ? logDeviationVectorInternal : *outputLogDeviationVector),
    addLogDevVec(0.),    // initially the deviation vector is normalized to unity
    orbitalPeriod(NAN),  // not known yet - will be assigned on the first timestep
    t0(NAN)              // same
{
    logDeviationVector.clear();
}

void RuntimeLyapunov::eval(const double t, double a[], double b[]) const
{
    coord::PosCar pos = orbint.getSol(t);
    coord::HessCar hess;
    orbint.potential.eval(pos, NULL, NULL, &hess, t);
    a[0] = -hess.dx2 + pow_2(orbint.Omega);
    a[1] = -hess.dxdy;
    a[2] = -hess.dxdz;
    a[3] = -hess.dxdy;
    a[4] = -hess.dy2 + pow_2(orbint.Omega);
    a[5] = -hess.dydz;
    a[6] = -hess.dxdz;
    a[7] = -hess.dydz;
    a[8] = -hess.dz2;
    std::fill(b, b+9, 0);
    b[1] =  2*orbint.Omega;
    b[3] = -2*orbint.Omega;
}

bool RuntimeLyapunov::processTimestep(double tbegin, double tend)
{
    if(t0 != t0)
        t0 = tbegin;   // record the first ever moment of time

    // 0. assign the orbital period on the first timestep
    if(orbitalPeriod != orbitalPeriod /*initially it was set to NAN*/) {
        orbitalPeriod = T_circ(orbint.potential, totalEnergy(orbint.potential, orbint.getSol(tbegin)));
        if(!isFinite(orbitalPeriod))  // e.g. the orbit is unbound,
            orbitalPeriod = 0;        // so no meaningful Lyapunov exponent can be computed anyway
        varEqSolver.init(DEV_VEC_INIT, tbegin);
    }

    // 1. perform one step of the internal variational equation solver
    // to compute the evolution of the deviation vector during the current timestep
    varEqSolver.doStep(tend-tbegin);

    // 2. store the deviation vector at regular intervals of time and check its magnitude
    bool needToRescale = false;
    double sign = tend>=tbegin ? +1 : -1;  // integrating forward or backward in time
    ptrdiff_t ibegin = static_cast<ptrdiff_t>(sign * (tbegin-t0) / samplingInterval);
    ptrdiff_t iend   = static_cast<ptrdiff_t>(sign * (tend-t0)   / samplingInterval * (1 + ROUNDOFF));
    double dtroundoff = ROUNDOFF * fmax(fabs(tend), fabs(tbegin));
    logDeviationVector.resize(iend + 1);
    for(ptrdiff_t iout=ibegin; iout<=iend; iout++) {
        double tout = sign * samplingInterval * iout + t0;
        if(sign * tout >= sign * tbegin - dtroundoff && sign * tout <= sign * tend + dtroundoff) {
            double devVec[6];
            for(int d=0; d<6; d++)
                devVec[d] = varEqSolver.getSol(tout, d);
            double logDevVec = logMagnitude(devVec);
            logDeviationVector[iout] = logDevVec + addLogDevVec;
            needToRescale |= logDevVec > 300;
        }
    }

    // 3. if the orbit is chaotic, the magnitude of the deviation vector grows exponentially,
    // and we need to renormalize it every now and then to avoid floating-point overflows
    if(needToRescale) {
        double devVec[6];
        for(int d=0; d<6; d++)
            devVec[d] = varEqSolver.getSol(tend, d);
        double logDevVec = logMagnitude(devVec), mult = exp(-logDevVec);
        addLogDevVec += logDevVec;
        for(int d=0; d<6; d++)
            devVec[d] *= mult;     // bring it back to unit norm
        varEqSolver.init(devVec);  // and reinit the internal variational equation solver
    }

    return true;
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

}  // namespace orbit
