#include "orbit_variational.h"
#include "potential_utils.h"
#include "math_core.h"
#include "math_fit.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace orbit{

namespace{

/// whether to perform successive orthogonalizations if the vectors grow exponentially:
/// this adds extra computational cost, and does not affect the values of vectors that are output
/// to the user; however, it makes possible to keep track of the entire set of Lyapunov exponents
/// (both positive and negative), whereas without orthogonalization one can reliably estimate
/// only the largest exponent. For the moment, there is no use in this feature, so it is disabled.
//#define ORTHOGONALIZE

/// there are two ways of integrating the variational equation in the rotating frame (Omega!=0):
/// (1) integration is performed in the rotating frame for the position and velocity components
/// of the deviation vector (Dx,Dv) and the hessian of the potential evaluated at the point x
/// of the original trajectory in the rotating frame. The equations for (Dx,Dv) are
///     d(Dx)/dt = Dv - [Omega cross Dx]
///     d(Dv)/dt = -hess(x) Dx - [Omega cross Dv]
/// and after combining them into one 2nd order ODE for Dx, we have
///     d^2(Dx)/dt^2 = (-hess(x) + Omega^2 * (1,1,0)) Dx - 2 [Omega cross d(Dx)/dt].
/// In this case, we need to reconstruct the velocity part of the deviation vector Dv from
/// the time derivative of Dx when storing the output.
/// (2) integration is performed in the inertial frame for the components of the deviation vector
/// (Dx',Dv') in the inertial frame. The hessian of the potential is evaluated at the point x
/// of the original trajectory, which is still provided in the rotating frame; it is then
/// transformed into the inertial frame (hess'), in which the equations of motion are simply
///     d(Dx')/dt = Dv'
///     d(Dv')/dt = -hess'(x) Dx'
/// In this case, we need to rotate (Dx',Dv') back to the rotated frame when storing the output.
/// The first variant is selected by the macro below, but it it less efficient than the second one,
/// so is disabled.
//#define ROTATING_FRAME

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

RuntimeVariational::RuntimeVariational(
    BaseOrbitIntegrator& orbint,
    double _samplingInterval,
    Trajectory* _outputDeviationVectors,
    double* _outputLyapunovExponentAndChaosOnsetTime)
:
    BaseRuntimeFnc(orbint),
    numVectors(_outputDeviationVectors ? 6 : 1),
    varEqSolver(*this, numVectors),
    samplingInterval(_samplingInterval),
    outputDeviationVectors(_outputDeviationVectors),
    outputLyapunovExponentAndChaosOnsetTime(_outputLyapunovExponentAndChaosOnsetTime),
    addLogDevVec(0.),    // initially the deviation vector is normalized to unity
    orbitalPeriod(NAN),  // not known yet - will be assigned on the first timestep
    t0(NAN),             // same
    timeBegin(NAN),      // store the beginning time and the length of the currently processed
    timeStep(NAN),       // timestep: assigned in processTimestep, read in eval and other methods
    matQ(6, 6, 0.0),
    matR(6, 6, 0.0),
    matS(6, 6, 0.0)
{
    if(outputDeviationVectors) {
        for(unsigned int vec=0; vec<numVectors; vec++)
            outputDeviationVectors[vec].clear();
    } else if(!outputLyapunovExponentAndChaosOnsetTime)
        throw std::runtime_error("RuntimeVariational: no output requested");
    for(int d=0; d<6; d++)
        matR(d,d) = 1.0;
}

void RuntimeVariational::evalMat(double timeOffset, double a[], double b[]) const
{
    // request the solution at the time offset from the beginning of the current timestep
    coord::PosCar pos = orbint.getSol(timeOffset);
    coord::HessCar hess;
    orbint.potential.eval(pos, NULL, NULL, &hess, timeBegin + timeOffset);
    std::fill(b, b+9, 0);
#ifdef ROTATING_FRAME
    a[0] = -hess.dx2 + pow_2(orbint.Omega);
    a[1] = -hess.dxdy;
    a[2] = -hess.dxdz;
    a[3] = -hess.dxdy;
    a[4] = -hess.dy2 + pow_2(orbint.Omega);
    a[5] = -hess.dydz;
    a[6] = -hess.dxdz;
    a[7] = -hess.dydz;
    a[8] = -hess.dz2;
    b[1] =  2*orbint.Omega;
    b[3] = -2*orbint.Omega;
#else
    double ca=1, sa=0;
    if(orbint.Omega)
        math::sincos(orbint.Omega * (timeBegin + timeOffset), sa, ca);
    a[0] = -hess.dx2  * (ca * ca) - hess.dy2 * (sa * sa) + 2 * hess.dxdy * (ca * sa);
    a[1] = -hess.dxdy * (ca * ca - sa * sa) - (hess.dx2 - hess.dy2) * (ca * sa);
    a[2] = -hess.dxdz *  ca + hess.dydz *  sa;
    a[3] = a[1];
    a[4] = -hess.dy2  * (ca * ca) - hess.dx2 * (sa * sa) - 2 * hess.dxdy * (ca * sa);
    a[5] = -hess.dydz *  ca - hess.dxdz *  sa;
    a[6] = a[2];
    a[7] = a[5];
    a[8] = -hess.dz2;
#endif
}

bool RuntimeVariational::processTimestep(double tbegin, double timestep)
{
    timeBegin = tbegin;   // record the (absolute) time at the beginning of the current timestep
    timeStep = timestep;  // and the length of this timestep
    double tend = tbegin + timestep;

    // 0. assign the orbital period on the first timestep
    if(t0 != t0) {
        t0 = tbegin;   // record the first ever moment of time
        orbitalPeriod = T_circ(orbint.potential, totalEnergy(orbint.potential,
            orbint.getSol(/*solution at the beginning of the first timestep*/ 0)));
        if(!isFinite(orbitalPeriod))  // e.g. the orbit is unbound,
            orbitalPeriod = 0;        // so no meaningful Lyapunov exponent can be computed anyway
        if(numVectors==6) {
            // initialize a full set of deviation vector by an identity matrix
            double devVec[6*6] = {0};
#ifdef ROTATING_FRAME
            for(int d=0; d<6; d++)
                devVec[d*7] = 1.;
            // in case of rotating frame, the deviation vectors contain position and velocity {x,v}
            // but the variables followed by the Ode2Solver contain {x,dx/dt};
            // dx/dt = v - [Omega cross x], so we subtract the extra term to the dx/dt variables
            // and add it again when outputting the deviation vectors.
            devVec[4] = -devVec[0] * orbint.Omega;
            devVec[9] =  devVec[7] * orbint.Omega;
#else
            double ca, sa;
            math::sincos(orbint.Omega * tbegin, sa, ca);
            devVec[ 0] = devVec[ 7] = devVec[21] = devVec[28] = ca;
            devVec[ 1] = devVec[22] =  sa;
            devVec[ 6] = devVec[27] = -sa;
            devVec[14] = devVec[35] = 1;
#endif
            varEqSolver.init(devVec);
            storeSolution(/*timeOffset*/ 0, /*ind*/ 0);
        } else {
            // follow just one deviation vector,
            // initialized with a hardcoded sequence of incommensurable numbers
            varEqSolver.init(DEV_VEC_INIT);
        }
    }

    // 1. perform one step of the internal variational equation solver
    // to compute the evolution of the deviation vector(s) during the current timestep
    varEqSolver.doStep(timestep);

    // 2. if needed, store the deviation vector(s) at regular intervals of time
    if(outputDeviationVectors && samplingInterval > 0 && samplingInterval < INFINITY) {
        double sign = timestep>=0 ? +1 : -1;  // integrating forward or backward in time
        double dtroundoff = ROUNDOFF * fmax(fmax(fabs(tend), fabs(tbegin)), fabs(t0));
        ptrdiff_t iout = static_cast<ptrdiff_t>(ceil( sign * (tbegin-t0) / samplingInterval));
        ptrdiff_t iend = static_cast<ptrdiff_t>((sign * (tend-t0) + dtroundoff) / samplingInterval);
        iout = std::max<ptrdiff_t>(1, iout);  // 0th point (IC) is already stored at first step
        for(; iout <= iend; iout++) {
            double timeOffset = math::clip(
                (sign * iout * samplingInterval - (tbegin-t0)) * sign,
                0.0, sign * timestep) * sign;
            storeSolution(timeOffset, iout);
        }
    }

    // 3. irrespective of the above, also compute the deviation vector(s) at the end of the timestep,
    // store all vectors if needed (i.e., if the output sampling interval is set to every timestep),
    // construct the fiducial vector as a linear combination of six vectors, store its logarithm,
    // and check for the need to renormalize
    ptrdiff_t iout = -1;
    if(outputDeviationVectors) {
        if(samplingInterval == 0)
            iout = outputDeviationVectors[0].size();
        else if(samplingInterval == INFINITY)
            iout = 0;
        // otherwise keep it at -1, meaning that no deviation vectors should be stored at this time
    }
    double logDevVec = storeSolution(timeStep, iout);
    logDeviationVector.push_back(std::pair<double, double>(fabs(tend - t0), logDevVec + addLogDevVec));

    // 4. if the orbit is chaotic, the magnitude of the deviation vectors grows exponentially,
    // and we need to renormalize them every now and then to avoid floating-point overflows
    if(logDevVec > 10)
        renormalize();

    return true;
}

double RuntimeVariational::storeSolution(double timeOffset, ptrdiff_t ind)
{
    double* matQflattened = matQ.data(), *devVec = matQflattened;
    for(unsigned int i=0; i<numVectors*6; i++)
        matQflattened[i] = varEqSolver.getSol(timeOffset, i);
    // first 6 elements of Q store 0th vector, next 6 - 1st vector, etc.
#ifdef ORTHOGONALIZE
    if(numVectors==6) {
        // if the orthogonalization procedure was previously applied, Q contains the current
        // set of deviation vectors, which is not the same as the dynamically evolved set
        // of initial dev vectors. The latter can be recovered by multiplying Q on the left
        // by the transposed rotation matrix R;
        // the result is stored in matS in the same form (one vector in each row)
        math::blas_dgemm(math::CblasTrans, math::CblasNoTrans, 1, matR, matQ, 0, matS);
        devVec = matS.data();
    }
    // otherwise devVec already contains the current set of deviation vectors in the right order
#endif
    double mul = exp(addLogDevVec);  // note: may overflow, the solution will be reported as infinity
    double fiducialVector[6] = {0};
    for(unsigned int vec=0; vec<numVectors; vec++) {
        if(outputDeviationVectors && ind>=0) {
            outputDeviationVectors[vec].resize(ind + 1);
#ifdef ROTATING_FRAME
            outputDeviationVectors[vec][ind] = Trajectory::value_type( /*std::pair*/
                /*first*/ coord::PosVelCar(
                devVec[vec*6+0] * mul, devVec[vec*6+1] * mul, devVec[vec*6+2] * mul,
                // components of devVec are dx/dt, while velocity stored in the output array
                // is v = dx/dt + [Omega cross x]
                (devVec[vec*6+3] - orbint.Omega * devVec[vec*6+1]) * mul,
                (devVec[vec*6+4] + orbint.Omega * devVec[vec*6+0]) * mul,
                devVec[vec*6+5] * mul),
                /*second*/ timeBegin + timeOffset);
#else
            double ca=1, sa=0;
            if(orbint.Omega)
                math::sincos(orbint.Omega * (timeBegin + timeOffset), sa, ca);
            outputDeviationVectors[vec][ind] = Trajectory::value_type( /*std::pair*/
                /*first*/ coord::PosVelCar(
                (devVec[vec*6+0] * ca + devVec[vec*6+1] * sa) * mul,
                (devVec[vec*6+1] * ca - devVec[vec*6+0] * sa) * mul,
                (devVec[vec*6+2] * mul),
                (devVec[vec*6+3] * ca + devVec[vec*6+4] * sa) * mul,
                (devVec[vec*6+4] * ca - devVec[vec*6+3] * sa) * mul,
                (devVec[vec*6+5] * mul)),
                /*second*/ timeBegin + timeOffset);
#endif
        }
        // construct a linear combination of all six vectors, equivalent to
        // the current state of a vector initialized with DEV_VEC_INIT components
        if(numVectors==6) {
            for(int i=0; i<6; i++)
                fiducialVector[i] += devVec[vec*6+i] * DEV_VEC_INIT[vec];
        }
    }
    return logMagnitude(numVectors == 1 ? devVec : fiducialVector);
}

void RuntimeVariational::renormalize()
{
#ifdef ORTHOGONALIZE
    if(numVectors==6) {
        // orthogonalize the full set of deviation vectors:
        // first assemble the matrix S containing vectors in its columns
        for(unsigned int vec=0; vec<numVectors; vec++)
            for(int d=0; d<6; d++)
                matS(d, vec) = varEqSolver.getSol(/*end of this step*/ timeStep, vec * 6 + d);
        // then perform the factorization S = Q R, where Q is orthogonal and R is upper triangular
        math::Matrix<double> Q, R;
        math::QRDecomp(matS).QR(Q, R);
        // the new set of orthogonal vectors are in the columns of Q, and will be stored
        // in the array devVec in a transposed form (i.e. each vector contiguously in its row);
        // make sure that R has nonnegative elements on the diagonal:
        // if not, flip signs of the offending row in R and the corresponding column in Q
        double devVec[6*6];
        double maxDiag = 0;
        for(unsigned int vec=0; vec<numVectors; vec++) {
            double sign = R(vec, vec) >= 0 ? 1 : -1;
            for(int d=0; d<6; d++) {
                devVec[vec * 6 + d] = Q(d, vec) * sign;
                R(vec, d) *= sign;
            }
            maxDiag = fmax(maxDiag, R(vec, vec) * matR(vec, vec));
        }
        varEqSolver.init(devVec);
        // rescale the R matrix so that its largest diagonal element is unity,
        // and record the cumulative scaling factor (or, rather, its logarithm to avoid overflow)
        addLogDevVec += log(maxDiag);
        // the matrix matR is the product of all previous rotation matrices R,
        // i.e. matR = R(last) R(next-to-last) ... R(first);
        // its diagonal elements may grow very large (the ratio of max/min may well exceed 1e16)
        math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1/maxDiag, R, matR, 0, matS);
        matR = matS;
    } else
#endif
    {   // just rescale the deviation vector(s), but do not orthogonalize
        double devVec[6*6];
        double logDevVec = -INFINITY;  // record the log-magnitude of the largest dev.vector
        for(unsigned int vec=0; vec<numVectors; vec++) {
            for(int d=0; d<6; d++)
                devVec[vec * 6 + d] = varEqSolver.getSol(/*end of this step*/ timeStep, vec * 6 + d);
            logDevVec = fmax(logDevVec, logMagnitude(devVec + vec*6));
        }
        double mult = exp(-logDevVec);
        addLogDevVec += logDevVec;
        // rescale all dev.vectors so that the largest of them has a unit norm
        for(unsigned int i=0; i<numVectors*6; i++)
            devVec[i] *= mult;
        // and reinit the internal variational equation solver
        varEqSolver.init(devVec);
    }
}

namespace {

// compute F(x) = ln[(exp(x)-1)/x] while avoiding over/underflows
inline double logexpm1overx(double x)
{
    return fabs(x) < 1e-2 ?  x * (0.5 + x * (1./24 - (1./2880) * x*x))  :
        x < 33 ?  log(expm1(x) / x)  :  x - log(x);
}

// compute the derivative of the above function F while avoiding over/underflows
inline double derlogexpm1overx(double x)
{
    return fabs(x) < 1e-2 ?  0.5 + x * (1./12 - (1./720) * x*x)  :  -1/x - 1/expm1(-x);
}

// fit the time evolution of ln(w(t)) by the following function:
// A + ln(t)                                     if t<=tch  (linear growth of w(t)),
// A + ln(t) + F(lambda * t) - F(lambda * tch)   if t> tch  (exponential growth of w(t)).
// here the function F = ln[(exp(x) - 1) / x] has the following properties:
// it is monotonically increasing, crosses zero at x=0, and its asymptotic behaviour is
// F(x --> -inf) = -ln(-x) - exp(x);  F(x --> 0) = x/2;  F(x --> +inf) = x-ln(x) - exp(-x).
// It is easy to see that when lambda>0 and t>>tch, the fitting function approaches lambda*t+const.
// This class is used in the Levenberg-Marquardt fitting to determine the parameters lambda, A, tch;
// with the above parameterisation, the function is well-defined for any combination of parameters,
// including negative lambda and/or tch, but these will be treated at a post-processing step.
class LyapunovFitter: public math::IFunctionNdimDeriv {
    const std::vector<double> t, lnw;
public:
    LyapunovFitter(const std::vector<double>& _t, const std::vector<double>& _lnw) :
        t(_t), lnw(_lnw)
    {}
    virtual void evalDeriv(const double params[], double values[], double *derivs=NULL) const
    {
        double lambda = params[0], A = params[1], tch = params[2];
        for(size_t k=0, size=t.size(); k<size; k++) {
            if(values) {
                values[k] = A + log(t[k]) + (t[k] < tch ? 0 :
                    logexpm1overx(lambda * t[k]) - logexpm1overx(lambda * tch)) - lnw[k];
            }
            if(derivs) {
                derivs[k*3+0] = t[k] < tch ? 0 : (t[k] - tch) * derlogexpm1overx(lambda * t[k]);
                derivs[k*3+1] = 1;
                derivs[k*3+2] = t[k] < tch ? 0 : -lambda * derlogexpm1overx(lambda * tch);
            }
        }
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return t.size(); };
};

}  // internal ns

void calcLyapunovExponent(
    const std::vector< std::pair<double, double> >& logDeviationVector,
    const double orbitalPeriod, double outputLyapunovExponentAndChaosOnsetTime[2])
{
    outputLyapunovExponentAndChaosOnsetTime[0] = outputLyapunovExponentAndChaosOnsetTime[1] = NAN;
    if(logDeviationVector.empty())
        return;
    double maxTime = logDeviationVector.back().first;
    // Instead of using all recorded values of log(devVec) at each timestep,
    // replace them by a much smaller number of bins
    const int MIN_BIN_SIZE = 5;  // minimum bin size for coarse-graining the input values
    double minBinTime = fmax(maxTime*0.01, orbitalPeriod);  // minimum time interval in one bin
    std::vector<double>
        t,   // mean time of each bin
        lnw; // log of the mean value of devVec across the time interval of each bin
    double
        binStart = NAN,    // start time of the current bin (NAN if the bin has not started yet)
        logOffset = 0,     // offset for averaging devVec to prevent overflow
        sumDevVecBin = 0;  // running sum of devVec in each bin, weighted by the time interval
    int binSize = 0;
    for(size_t i=0, size=logDeviationVector.size()-1; i<size; i++) {
        double ti = logDeviationVector[i  ].first, vi = logDeviationVector[i  ].second;
        double tp = logDeviationVector[i+1].first, vp = logDeviationVector[i+1].second;
        if(binStart != binStart) {  // start a new bin
            binStart = ti;
            logOffset = vi;
            sumDevVecBin = 0;
            binSize = 0;
        }
        sumDevVecBin += (tp-ti) * exp(0.5 * (vi + vp) - logOffset);
        binSize++;
        if(binSize >= MIN_BIN_SIZE && (tp >= binStart + minBinTime || i+1 >= size)) {
            // finish the current bin
            t.push_back(0.5 * (binStart + tp));
            lnw.push_back(log(sumDevVecBin / (tp - binStart)) + logOffset);
            binStart = NAN;  // will start a new bin at the next point
        }
    }
    if(t.size() < 5)
        return;  // orbit is too short
    outputLyapunovExponentAndChaosOnsetTime[0] = 0;  // default return values mean no chaos detected
    outputLyapunovExponentAndChaosOnsetTime[1] = INFINITY;
    // fit the time evolution of ln(w(t)) by the following function:
    // A + ln(t)                                if t<=tch  (linear growth of w(t)),
    // A + ln(t) + F(lambda*t) - F(lambda*tch)  if t> tch  (exponential growth of w(t)),
    // where F(x) = ln[ (exp(x)-1) / x ] resembles a softened ramp function
    // To begin with, perform a linear fit to ln(w(t)) by the function lambda * t + A
    double params[3];  // lambda, A, tch
    math::linearFit(t, lnw, NULL, params[0], params[1], NULL);
    // now perform the full nonlinear fit, setting the initial guess for the changeover time
    // to half of the total orbit integration time
    params[2] = maxTime*0.5;
    math::nonlinearMultiFit(LyapunovFitter(t, lnw),
        /*init*/ params, /*accuracy*/ 1e-6, /*max.num.fnc.eval.*/ 100, /*output*/ params);
    double lambda = params[0], /*A = params[1],*/ tch = params[2];
    // lambda is not constrained and may turn out to be negative, but then should be replaced by zero;
    // likewise, if the changeover time is longer than the orbit duration, lambda should be zero too.
    // as a precaution against spurious chaos detections, we also exclude cases when the exp growth
    // only manifests itself in the last bin
    if(lambda < 0 || tch >= t[t.size()-2])
        return;
    // tch is not constrained to be positive, but values of tch < t[0] are nearly equivalent
    // to setting tch=0 and simultaneously adjusting the offset term, which is unused anyway
    if(tch < t.front())
        tch = 0;
    // to assess the significance of the detected exponential growth at times t>tch,
    // we compare the increase in ln(w) on this interval (tch..end) due to exponential growth
    // with the increase in ln(w) on the same interval expected from the linear growth alone
    double delta_lnw = logexpm1overx(lambda * t.back()) - logexpm1overx(lambda * tch);
    if(delta_lnw < M_LN2)  // exp growth is not significant
        return;
    outputLyapunovExponentAndChaosOnsetTime[0] = lambda * orbitalPeriod;
    outputLyapunovExponentAndChaosOnsetTime[1] = tch / orbitalPeriod;
}

}  // namespace orbit
