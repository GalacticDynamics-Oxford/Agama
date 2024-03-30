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
    double* _outputLyapunovExponent)
:
    BaseRuntimeFnc(orbint),
    numVectors(_outputDeviationVectors ? 6 : 1),
    varEqSolver(*this, numVectors),
    samplingInterval(_samplingInterval),
    outputDeviationVectors(_outputDeviationVectors),
    outputLyapunovExponent(_outputLyapunovExponent),
    addLogDevVec(0.),    // initially the deviation vector is normalized to unity
    orbitalPeriod(NAN),  // not known yet - will be assigned on the first timestep
    t0(NAN),             // same
    matQ(6, 6, 0.0),
    matR(6, 6, 0.0),
    matS(6, 6, 0.0)
{
    if(outputDeviationVectors) {
        for(unsigned int vec=0; vec<numVectors; vec++)
            outputDeviationVectors[vec].clear();
    } else if(!outputLyapunovExponent)
        throw std::runtime_error("RuntimeVariational: no output requested");
    for(int d=0; d<6; d++)
        matR(d,d) = 1.0;
}

void RuntimeVariational::eval(const double t, double a[], double b[]) const
{
    coord::PosCar pos = orbint.getSol(t);
    coord::HessCar hess;
    orbint.potential.eval(pos, NULL, NULL, &hess, t);
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
    std::fill(b, b+9, 0);
    b[1] =  2*orbint.Omega;
    b[3] = -2*orbint.Omega;
#else
    double ca=1, sa=0;
    if(orbint.Omega)
        math::sincos(orbint.Omega * t, sa, ca);
    a[0] = -hess.dx2  * (ca * ca) - hess.dy2 * (sa * sa) + 2 * hess.dxdy * (ca * sa);
    a[1] = -hess.dxdy * (ca * ca - sa * sa) - (hess.dx2 - hess.dy2) * (ca * sa);
    a[2] = -hess.dxdz *  ca + hess.dydz *  sa;
    a[3] = a[1];
    a[4] = -hess.dy2  * (ca * ca) - hess.dx2 * (sa * sa) - 2 * hess.dxdy * (ca * sa);
    a[5] = -hess.dydz *  ca - hess.dxdz *  sa;
    a[6] = a[2];
    a[7] = a[5];
    a[8] = -hess.dz2;
    std::fill(b, b+9, 0);
#endif
}

bool RuntimeVariational::processTimestep(double tbegin, double tend)
{
    if(tbegin == tend)
        return true;   // nothing to do (timestep is likely too small)

    // 0. assign the orbital period on the first timestep
    if(t0 != t0) {
        t0 = tbegin;   // record the first ever moment of time
        orbitalPeriod = T_circ(orbint.potential, totalEnergy(orbint.potential, orbint.getSol(tbegin)));
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
            varEqSolver.init(devVec, tbegin);
            storeSolution(tbegin, 0);
        } else {
            varEqSolver.init(DEV_VEC_INIT, tbegin);
        }
    }

    // 1. perform one step of the internal variational equation solver
    // to compute the evolution of the deviation vector(s) during the current timestep
    varEqSolver.doStep(tend-tbegin);

    // 2. if needed, store the deviation vector(s) at regular intervals of time
    if(outputDeviationVectors && samplingInterval > 0 && samplingInterval < INFINITY) {
        double sign = tend>=tbegin ? +1 : -1;  // integrating forward or backward in time
        ptrdiff_t ibegin = static_cast<ptrdiff_t>(sign * (tbegin-t0) / samplingInterval);
        ptrdiff_t iend   = static_cast<ptrdiff_t>(sign * (tend-t0)   / samplingInterval * (1 + ROUNDOFF));
        double dtroundoff = ROUNDOFF * fmax(fabs(tend), fabs(tbegin));
        for(ptrdiff_t iout=std::max<ptrdiff_t>(ibegin, 1); iout<=iend; iout++) {
            double tout = sign * samplingInterval * iout + t0;
            if(sign * tout >= sign * tbegin - dtroundoff && sign * tout <= sign * tend + dtroundoff)
                storeSolution(tout, iout);
        }
    }

    // 3. irrespective of the above, also compute the deviation vector(s) at the end of the timestep,
    // store all vectors if needed (i.e., if the output sampling interval is set to every timestep),
    // compute the maximum magnitude and store its logarithm, and check for the need to renormalize
    ptrdiff_t iout = -1;
    if(outputDeviationVectors) {
        if(samplingInterval == 0)
            iout = outputDeviationVectors[0].size();
        else if(samplingInterval == INFINITY)
            iout = 0;
        // otherwise keep it at -1, meaning that no deviation vectors should be stored at this time
    }
    double logDevVec = storeSolution(tend, iout);
    logDeviationVector.push_back(std::pair<double, double>(fabs(tend - t0), logDevVec + addLogDevVec));

    // 4. if the orbit is chaotic, the magnitude of the deviation vectors grows exponentially,
    // and we need to renormalize them every now and then to avoid floating-point overflows
    if(logDevVec > 10)
        renormalize(tend);

    return true;
}

double RuntimeVariational::storeSolution(double time, ptrdiff_t ind)
{
    double* matQflattened = matQ.data(), *devVec = matQflattened;
    for(unsigned int i=0; i<numVectors*6; i++)
        matQflattened[i] = varEqSolver.getSol(time, i);
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
    double logDevVec = -INFINITY;
    double mul = exp(addLogDevVec);
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
                /*second*/ time);
#else
            double ca=1, sa=0;
            if(orbint.Omega)
                math::sincos(orbint.Omega * time, sa, ca);
            outputDeviationVectors[vec][ind] = Trajectory::value_type( /*std::pair*/
                /*first*/ coord::PosVelCar(
                (devVec[vec*6+0] * ca + devVec[vec*6+1] * sa) * mul,
                (devVec[vec*6+1] * ca - devVec[vec*6+0] * sa) * mul,
                (devVec[vec*6+2] * mul),
                (devVec[vec*6+3] * ca + devVec[vec*6+4] * sa) * mul,
                (devVec[vec*6+4] * ca - devVec[vec*6+3] * sa) * mul,
                (devVec[vec*6+5] * mul)),
                /*second*/ time);
#endif
        }
        logDevVec = fmax(logDevVec, logMagnitude(devVec + vec*6));
    }
    return logDevVec;
}

void RuntimeVariational::renormalize(double time)
{
#ifdef ORTHOGONALIZE
    if(numVectors==6) {
        // orthogonalize the full set of deviation vectors:
        // first assemble the matrix S containing vectors in its columns
        for(unsigned int vec=0; vec<numVectors; vec++)
            for(int d=0; d<6; d++)
                matS(d, vec) = varEqSolver.getSol(time, vec * 6 + d);
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
                devVec[vec * 6 + d] = varEqSolver.getSol(time, vec * 6 + d);
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


double calcLyapunovExponent(
    const std::vector< std::pair<double, double> >& logDeviationVector,
    const double orbitalPeriod)
{
    if(logDeviationVector.empty())
        return NAN;
    double maxTime = logDeviationVector.back().first;
    const int MIN_BIN_SIZE = 5;     // minimum bin size for coarse-graining the input values
    const double minBinTime = maxTime * 0.01;  // minimum length of time interval in one bin
    const double skipTime   = maxTime * 0.2;   // fraction of time to skip at the beginning
    // assemble the bins starting from the end (where the deviation vector is at its largest);
    // in each bin, take the median value of log(devVec) and subtract the contribution of
    // the regular phase mixing, which grows as log(t).
    std::vector<double>
        t,     // mean timestamp of each bin
        v,     // median value of log(devVec/t) in each bin
        binv;  // collection of all values of log(devVec/t) in the current bin
    double te = NAN;  // endtime of the current bin (NAN if the bin has not started yet)
    for(size_t i=logDeviationVector.size()-1; i>0; i--) {
        double ti = logDeviationVector[i].first, vi = logDeviationVector[i].second;
        if(ti < skipTime)
            break;
        if(te!=te)
            te = ti;
        binv.push_back(vi - log(ti));
        size_t binsize = binv.size();
        if((ti < te - minBinTime) && (binsize >= MIN_BIN_SIZE)) {
            // finish the current bin
            std::nth_element(binv.begin(), binv.begin() + binsize/2, binv.end());
            t.push_back(0.5 * (ti + te));
            v.push_back(binv[binsize/2]);
            te = NAN;
            binv.clear();
        }
    }
    if(t.size() < 4)
        return NAN;   // too few bins for a meaningful fit
    double slope, intercept;
    math::linearFit(t, v, NULL, slope, intercept, NULL);
    double logterm = intercept + log(t[0]);
    double linterm = slope * t[0];
    if(linterm > fmax(0.1 * logterm, 2.0))
        return slope * orbitalPeriod;  // Lyapunov exponent normalized by orbital period
    else
        return 0;  // chaotic behavior not detected, return zero estimate for the Lyapunov exponent
}

}  // namespace orbit
