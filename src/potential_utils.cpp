#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <cmath>
#include <cfloat>
#include <cassert>
#include <stdexcept>
#include <fstream>   // for writing debug info
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace potential{

namespace{  // internal routines

/// if defined, use the integrateNdim routine for computing the projected density;
/// it takes a somewhat larger number of evaluations, but performs vectorized calls to density()
#define PROJ_DENSITY_VECTORIZED

/// relative accuracy of integration of projected density/potential
static const double EPSREL_DENSITY_INT = 1e-4;

/// max. number of density evaluations for multidimensional integration
static const size_t MAX_NUM_EVAL_INT = 10000;

/// relative accuracy of root-finders for radius
static const double ACCURACY_ROOT = 1e-10;

/// required tolerance on the 2nd deriv to declare the asymptotic limit
static const double ACCURACY_INTERP = 1e-6;
static const double ACCURACY_INTERP2= 1e-4;

/// size of the interpolation grid in the dimension corresponding to relative angular momentum
static const unsigned int GRID_SIZE_L = 40;

/// a number that is considered nearly infinity in log-scaled root-finders
static const double HUGE_NUMBER = 1e100;

/// safety factor to avoid roundoff errors in estimating the inner/outer asymptotic slopes
static const double ROUNDOFF_THRESHOLD = DBL_EPSILON / ROOT3_DBL_EPSILON;  // eps^(2/3) ~ 4e-11

/// minimum relative difference between two adjacent values of potential (to reduce roundoff errors)
static const double MIN_REL_DIFFERENCE = ROUNDOFF_THRESHOLD;

/// maximum value for L/Lcirc above which approximate the peri/apocenter radii analytically
static const double LREL_NEARLY_CIRCULAR = 0.999999;

/// fixed order of Gauss-Legendre integration of PhaseVolume on each segment of a log-grid
static const int GLORDER1 = 6;   // for shorter segments
static const int GLORDER2 = 10;  // for larger segments
/// the choice between short and long segments is determined by the ratio between consecutive nodes
static const double GLRATIO = 2.0;

// --------- computation of projected density, potential and its derivatives --------- //

/// helper class for integrating the density along the line of sight
class ProjectedDensityIntegrand: public math::IFunctionNoDeriv, public math::IFunctionNdim {
    const BaseDensity& dens;  ///< the density model
    const double X, Y, R;     ///< coordinates in the image plane
    const coord::Orientation& orientation; ///< converion between intrinsic and observed coords
    double time;              ///< time at which the density is evaluated
public:
    ProjectedDensityIntegrand(const BaseDensity& _dens, const coord::PosProj& pos,
        const coord::Orientation& _orientation, double _time)
    :
        dens(_dens), X(pos.X), Y(pos.Y), R(sqrt(X*X+Y*Y)), orientation(_orientation), time(_time) {}
    virtual double value(double s) const
    {
        // unscale the input scaled coordinate, which lies in the range (0..1);
        double dZds, Z = unscale(math::ScalingDoubleInf(R), s, &dZds);
        return nan2num(dens.density(orientation.fromRotated(coord::PosCar(X, Y, Z)), time) * dZds);
    }
    virtual void eval(const double vars[], double values[]) const {
        values[0] = value(vars[0]);
    }
    // vectorized version of the integrand
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        if(npoints==0)
            return;
        coord::PosCar* points = static_cast<coord::PosCar*>(alloca(npoints * sizeof(coord::PosCar)));
        double* dZds = static_cast<double*>(alloca(npoints * sizeof(double)));
        for(size_t i=0; i<npoints; i++) {
            double Z = unscale(math::ScalingDoubleInf(R), vars[i], &dZds[i]);
            points[i] = orientation.fromRotated(coord::PosCar(X, Y, Z));
        }
        dens.evalmanyDensityCar(npoints, points, values, time);
        for(size_t i=0; i<npoints; i++)
            values[i] = nan2num(values[i] * dZds[i]);
    }
    virtual unsigned int numVars()   const { return 1; }
    virtual unsigned int numValues() const { return 1; }
};

/// helper class for integrating the potential, acceleration and its derivatives along the line of sight
class ProjectedEvalIntegrand: public math::IFunctionNdim {
    const BasePotential& pot; ///< the potential
    const double X, Y, R;     ///< coordinates in the image plane
    const coord::Orientation& orientation;  ///< converion between intrinsic and observed coords
    const bool needPhi, needGrad, needHess; ///< which quantities are needed
    double time;              ///< time at which the density is evaluated
public:
    ProjectedEvalIntegrand(const BasePotential& _pot, const coord::PosProj& pos,
        const coord::Orientation& _orientation,
        bool _needPhi, bool _needGrad, bool _needHess, double _time)
    :
        pot(_pot), X(pos.X), Y(pos.Y), R(sqrt(X*X+Y*Y)), orientation(_orientation),
        needPhi(_needPhi), needGrad(_needGrad), needHess(_needHess), time(_time)
    {
        if(needPhi && !isFinite(pot.value(coord::PosCar(0, 0, 0))) )
            throw std::runtime_error("Potential must be finite at origin");
    }

    virtual unsigned int numVars()   const { return 1; }
    virtual unsigned int numValues() const { return int(needPhi) + 2*int(needGrad) + 3*int(needHess); }

    virtual void eval(const double vars[], double values[]) const
    {
        // unscale the input scaled coordinate, which lies in the range (0..1);
        double dZds, Z = unscale(math::ScalingDoubleInf(R), vars[0], &dZds), Phi, Phi0;
        coord::GradCar grad_int, grad_obs;  // gradient in the intrinsic and observed systems
        coord::HessCar hess_int, hess_obs;  // same for hessian
        pot.eval(orientation.fromRotated(coord::PosCar(X, Y, Z)),
            needPhi? &Phi : NULL, needGrad? &grad_int : NULL, needHess? &hess_int : NULL, time);
        int numOutputs = 0;
        if(needPhi) {
            pot.eval(coord::PosCar(0, 0, Z), &Phi0, NULL, NULL, time);
            values[numOutputs++] = nan2num((Phi-Phi0) * dZds);
        }
        if(needGrad) {
            grad_obs = orientation.toRotated(grad_int);
            values[numOutputs++] = nan2num(grad_obs.dx * dZds);
            values[numOutputs++] = nan2num(grad_obs.dy * dZds);
        }
        if(needHess) {
            hess_obs = orientation.toRotated(hess_int);
            values[numOutputs++] = nan2num(hess_obs.dx2  * dZds);
            values[numOutputs++] = nan2num(hess_obs.dy2  * dZds);
            values[numOutputs++] = nan2num(hess_obs.dxdy * dZds);
        }
    }
};

/// helper class for computing the principal axes of the ellipsoidally-weighted inertia tensor
class PrincipalAxesIntegrand: public math::IFunctionNdim {
    const BaseDensity& dens;        ///< the density model
    const coord::SymmetryType sym;  ///< cached value of its symmetry
    const double ax, ay, az;        ///< current estimates of principal axes
    const coord::Orientation& orientation;  ///< orientation of principal axes
public:
    PrincipalAxesIntegrand(const BaseDensity& _dens, double axes[],
        const coord::Orientation& _orientation)
    :
        dens(_dens), sym(_dens.symmetry()),
        ax(axes[0]), ay(axes[1]), az(axes[2]), orientation(_orientation)
    {}

    virtual void eval(const double vars[], double values[]) const {  // unused
        evalmany(1, vars, values);
    }

    // vectorized version of the integrand
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        // The input point(s) are always taken from one octant in the xyz space,
        // but depending on the symmetry of the density profile, we may need to add mirrored points.
        // For simplicity, consider only two cases: triaxial (no need to mirror) and anything else.
        int ncopies = isTriaxial(sym) ? 1 : 8;

        // 0. allocate various temporary arrays on the stack - no need to delete them manually
        // positions in cylindrical coords (unscaled from the input variables)
        coord::PosCar* pos = static_cast<coord::PosCar*>(
             alloca(npoints * ncopies * sizeof(coord::PosCar)));
        // jacobian of coordinate transformation at each point (all copies of one point share one jac)
        double* jac = static_cast<double*>(alloca(npoints * sizeof(double)));
        // values of density at each point
        double* val = static_cast<double*>(alloca(npoints * ncopies * sizeof(double)));

        // 1. unscale the input variables and create mirrored copies of each point, if necessary
        for(size_t i=0; i<npoints; i++) {
            coord::PosCar postmp = toPosCar(unscaleCoords(&vars[i*3], /*output*/ &jac[i]));
            postmp.x *= ax;
            postmp.y *= ay;
            postmp.z *= az;
            pos[i*ncopies] = orientation.fromRotated(postmp);
            if(ncopies == 8) {
                postmp.z *= -1;
                pos[i*8+1] = orientation.fromRotated(postmp);
                postmp.y *= -1;
                pos[i*8+2] = orientation.fromRotated(postmp);
                postmp.z *= -1;
                pos[i*8+3] = orientation.fromRotated(postmp);
                postmp.x *= -1;
                pos[i*8+4] = orientation.fromRotated(postmp);
                postmp.z *= -1;
                pos[i*8+5] = orientation.fromRotated(postmp);
                postmp.y *= -1;
                pos[i*8+6] = orientation.fromRotated(postmp);
                postmp.z *= -1;
                pos[i*8+7] = orientation.fromRotated(postmp);
            }
        }

        // 2. compute the density for all these points at once
        dens.evalmanyDensityCar(npoints * ncopies, pos, val);

        // 3. multiply by jacobian and output the density weighted by x_i x_j,
        // accounting for all mirrored copies of the input point
        const unsigned int numvals = numValues();
        std::fill(values, values + npoints * numvals, 0);
        for(size_t i=0; i < npoints * ncopies; i++) {
            size_t o = i / ncopies;
            val[i] *= jac[o];
            values[o*numvals  ] += nan2num(val[i] * pow_2(pos[i].x));
            values[o*numvals+1] += nan2num(val[i] * pow_2(pos[i].y));
            values[o*numvals+2] += nan2num(val[i] * pow_2(pos[i].z));
            if(numvals==6) {
                values[o*numvals+3] += nan2num(val[i] * pos[i].x * pos[i].y);
                values[o*numvals+4] += nan2num(val[i] * pos[i].x * pos[i].z);
                values[o*numvals+5] += nan2num(val[i] * pos[i].y * pos[i].z);
            }
        }
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return isTriaxial(sym) ? 3 : 6; }
};

// -------- routines for conversion between energy, radius and angular momentum --------- //

/** helper class to find the root of  Phi(R) = E */
class RmaxRootFinder: public math::IFunction {
    const Axisymmetrized<BasePotential> pot;
    const double E;
public:
    RmaxRootFinder(const BasePotential& _pot, double _E) : pot(_pot), E(_E) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* =0) const
    {
        double Phi, R = exp(logR);
        coord::GradCyl grad;
        pot.eval(coord::PosCyl(R, 0, 0), &Phi, deriv? &grad : NULL);
        if(val) {
            *val = Phi - E;
            if(!isFinite(*val)) {  // take special measures
                if(E==-INFINITY)
                    *val = Phi==E ? 0 : +1.0;
                else if(Phi==-INFINITY)
                    *val = -1.0;  // safely negative value
                else if(R>=HUGE_NUMBER)
                    *val = +1.0;
            }
        }
        if(deriv)
            *deriv = grad.dR * R;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  Phi(R) + 1/2 R dPhi/dR = E
    (i.e. the radius R of a circular orbit with the given energy E).
*/
class RcircRootFinder: public math::IFunction {
    const Axisymmetrized<BasePotential> pot;
    const double E;
public:
    RcircRootFinder(const BasePotential& _pot, double _E) : pot(_pot), E(_E) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* =0) const
    {
        double Phi, R = exp(logR);
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(R, 0, 0), &Phi, &grad, deriv? &hess : NULL);
        if(val) {
            double v = 0.5 * R * grad.dR;
            *val = Phi - E + (isFinite(v) ? v : 0);
            if(!isFinite(*val)) {  // special cases
                if(E==-INFINITY)
                    *val = Phi==E ? 0 : +1.0;
                else if(Phi==-INFINITY)
                    *val = -1.0;  // safely negative value
                else if(R>=HUGE_NUMBER)
                    *val = +1.0;  // safely positive value
            }
        }
        if(deriv)
            *deriv = (1.5 * grad.dR + 0.5 * R * hess.dR2) * R;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  L^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given angular momentum L).
*/
class RfromLRootFinder: public math::IFunction {
    const Axisymmetrized<BasePotential> pot;
    const double L2;
public:
    RfromLRootFinder(const BasePotential& _pot, double _L) : pot(_pot), L2(_L*_L) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* =0) const
    {
        double R = exp(logR);
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(R, 0, 0), NULL, &grad, deriv? &hess : NULL);
        double F = pow_3(R) * grad.dR; // Lz^2(R)
        if(val)
            *val = isFinite(F) ? F - L2 :
            // this may fail if R --> 0 or R --> infinity,
            // in these cases replace it with a finite number of a correct sign
            R < 1 ? -L2 : +L2;
        if(deriv)
            *deriv = pow_3(R) * (3*grad.dR + R*hess.dR2);
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** Helper function for finding the roots of (effective) potential in R direction */
class RPeriApoRootFinder: public math::IFunction {
    const BasePotential& pot;
    const double E, halfL2;
public:
    RPeriApoRootFinder(const BasePotential& _pot, double _E, double L) :
        pot(_pot), E(_E), halfL2(L*L/2) {};
    virtual unsigned int numDerivs() const { return 1; }
    virtual void evalDeriv(const double R, double* val=0, double* der=0, double* =0) const {
        double Phi=0;
        coord::GradCyl grad;
        pot.eval(coord::PosCyl(R,0,0), &Phi, der? &grad : NULL);
        if(val)
            *val = (R>0 ? (E-Phi)*R*R : 0) - halfL2;
        if(der)
            *der = 2*R*(E-Phi) - R*R*grad.dR;
    }
};

/** Helper function for finding the roots of (effective) potential in R direction,
    for a power-law asymptotic form potential at small radii */
class RPeriApoRootFinderPowerLaw: public math::IFunction {
    const double s, v2;  // potential slope and squared relative ang.mom.(normalized to Lcirc)
public:
    RPeriApoRootFinderPowerLaw(double slope, double Lrel2) : s(slope), v2(Lrel2)
    { assert(s>=-1 && Lrel2>=0 && Lrel2<=1); }
    virtual unsigned int numDerivs() const { return 1; }
    virtual void evalDeriv(const double x, double* val=0, double* der=0, double* =0) const {
        if(val)
            *val = s!=0 ?  (2+s)*x*x - 2*std::pow(x, 2+s) - v2*s  :  (x==0 ? 0 : x*x * (1 - 2*log(x)) - v2);
        if(der)
            *der = s!=0 ?  (4+2*s) * (x - std::pow(x, 1+s))  :  -4*x*log(x);
    }
};

/// root polishing routine to improve the accuracy of peri/apocenter radii determination
inline double refineRoot(const math::IFunction& pot, double R, double E, double L)
{
    double val, der, der2;
    pot.evalDeriv(R, &val, &der, &der2);
    // F = E - Phi(r) - L^2/(2r^2), refine the root of F=0 using Halley's method with two derivatives
    double F  = E - val - 0.5*pow_2(L/R);
    double Fp = pow_2(L/R)/R - der;
    double Fpp= -3*pow_2(L/(R*R)) - der2;
    double dR = -F / (Fp - 0.5 * F * Fpp / Fp);
    return fabs(dR) < 0.25*R ? R+dR : R;  // precaution to avoid unpredictably large corrections
}

/// helper class for finding the minimum or a given value of the potential along the line of sight
class PotentialFinder: public math::IFunctionNoDeriv {
    const BasePotential& pot; ///< the potential
    const double E;           ///< required value of the potential when solving for Phi(X,Y,Z)=E
    const double X, Y;        ///< coordinates in the image plane
    const coord::Orientation& orientation; ///< converion between intrinsic and observed coords
    const math::ScalingDoubleInf scaling;  ///< scaling transformation for Z
public:
    PotentialFinder(const BasePotential& _pot, double _E,
        double _X, double _Y, const coord::Orientation& _orientation)
    :
        pot(_pot), E(_E), X(_X), Y(_Y), orientation(_orientation), scaling(sqrt(X*X+Y*Y)) {}
    virtual double value(double s) const
    {
        double Z=unscale(scaling, s);
        if(fabs(Z)==INFINITY)
            return pot.value(coord::PosCar(INFINITY,0,0)) - E;
        return pot.value(orientation.fromRotated(coord::PosCar(X, Y, Z))) - E;
    }
    void findRoots(double& Zm, double& Z1, double& Z2) const
    {
        double s = math::findMin(*this, 0, 1, 0.5, ACCURACY_ROOT);
        Zm = unscale(scaling, s);
        if(value(s) > 0) {   // even the minimum value exceeds the target - no roots
            Z1 = Z2 = NAN;
        } else {   // two roots on the intervals -infinity..Z(s) and Z(s)..+infinity
            Z1 = unscale(scaling, math::findRoot(*this, 0, s, ACCURACY_ROOT));
            Z2 = unscale(scaling, math::findRoot(*this, s, 1, ACCURACY_ROOT));
        }
    }
};

/// Scaling transformations for energy: the input energy ranges from Phi0 to 0,
/// the output scaled variable - from -inf to +inf. Here Phi0=Phi(0) may be finite or -inf.
/// The goal is to avoid cancellation errors when Phi0 is finite and E --> Phi0 --
/// in this case the scaled variable may achieve any value down to -inf, instead of
/// cramping into a few remaining bits of precision when E is almost equal to Phi0,
/// so that any function that depends on scaledE may be comfortably evaluated with full precision.
/// Additionally, this transformation is intended to achieve an asymptotic power-law behaviour
/// for any quantity whose logarithm is interpolated as a function of scaledE and linearly
/// extrapolated as its argument tends to +- infinity.

/// return scaledE and dE/d(scaledE) as functions of E and invPhi0 = 1/Phi(0)
inline void scaleE(const double E, const double invPhi0,
    /*output*/ double& scaledE, double& dEdscaledE)
{
    double expE = invPhi0 - 1/E;
    scaledE     = log(expE);
    dEdscaledE  = E * E * expE;
}

/// return E and dE/d(scaledE) as functions of scaledE
inline void unscaleE(const double scaledE, const double invPhi0,
    /*output*/ double& E, double& dEdscaledE, double& d2EdscaledE2)
{
    double expE = exp(scaledE);
    E           = 1 / (invPhi0 - expE);
    dEdscaledE  = E * E * expE;
    d2EdscaledE2= E * dEdscaledE * (invPhi0 + expE);
    // d3EdscaledE3 = dEdscaledE * (1 + 6 * dEdscaledE * invPhi0)
}

/// same as above, but for two separate values of E1 and E2;
/// in addition, compute the difference between E1 and E2 in a way that is not prone
/// to cancellation when both E1 and E2 are close to Phi0 and the latter is finite.
inline void unscaleDeltaE(const double scaledE1, const double scaledE2, const double invPhi0,
    /*output*/ double& E1, double& E2, double& E1minusE2, double& dE1dscaledE1)
{
    double exp1  = exp(scaledE1);
    double exp2  = exp(scaledE2);
    E1           = 1 / (invPhi0 - exp1);
    E2           = 1 / (invPhi0 - exp2);
    E1minusE2    = (exp1 - exp2) * E1 * E2;
    dE1dscaledE1 = E1 * E1 * exp1;
}

/** A specially designed function whose second derivative indicates the local variation of potential,
    used to determine the range and spacing between radial grid nodes for interpolation.
    Its second derivative is identically zero if the potential is a power-law in radius (e.g., -M/r).
*/
class ScalePhi: public math::IFunction {
    const BasePotential& pot;
    const double invPhi0;
public:
    ScalePhi(const BasePotential& _pot) : pot(_pot), invPhi0(1/pot.value(coord::PosCyl(0,0,0))) {}
    virtual void evalDeriv(const double logr, double* val, double* der, double* der2) const {
        double r=exp(logr), Phi;
        coord::GradCyl dPhi;
        coord::HessCyl d2Phi;
        pot.eval(coord::PosCyl(r,0,0), &Phi, &dPhi, &d2Phi);
        double expE  = invPhi0 - 1/Phi;
        // only the 2nd derivative is needed, use a carefully concocted combination of derivatives
        if(val)
            *val = NAN; //log(expE) + 2 * log(-Phi);  // never used
        if(der)
            *der = 0;  // unused but should be finite
        if(der2) {
            if(invPhi0!=0 && expE < -invPhi0 * MIN_REL_DIFFERENCE)
                // in case of a finite potential at r=0,
                // we avoid approaching too close to 0 to avoid roundoff errors in Phi
                *der2 = 0;
            else
                *der2 = pow_2(r/Phi) / expE * (dPhi.dR * (1/r - dPhi.dR/Phi * (2 + 1/Phi/expE)) + d2Phi.dR2)
                    + 2 * r*r/Phi * (d2Phi.dR2 + d2Phi.dz2 - pow_2(dPhi.dR)/Phi);
        }
    }
    virtual unsigned int numDerivs() const { return 2; }
};

}  // internal namespace


double projectedDensity(const BaseDensity& dens, const coord::PosProj& pos,
    const coord::Orientation& orientation, double time)
{
#ifndef PROJ_DENSITY_VECTORIZED
    return math::integrateAdaptive(ProjectedDensityIntegrand(dens, X, Y, orientation),
        0, 1, EPSREL_DENSITY_INT);
#else
    // use integrateNdim as the adaptive integration engine with vectorization
    double xlower[1] = {0}, xupper[1] = {1}, result;
    math::integrateNdim(ProjectedDensityIntegrand(dens, pos, orientation, time),
        xlower, xupper, EPSREL_DENSITY_INT, /*maxNumEval*/ MAX_NUM_EVAL_INT, &result);
    return result;
#endif
}

void projectedEval(
    const BasePotential& pot, const coord::PosProj& pos, const coord::Orientation& orientation,
    double *value, coord::GradCar* grad, coord::HessCar* hess, double time)
{
    double xlower[1] = {0}, xupper[1] = {1}, result[6], error[6]; int neval;
    math::integrateNdim(
        ProjectedEvalIntegrand(pot, pos, orientation, value!=NULL, grad!=NULL, hess!=NULL, time),
        xlower, xupper, EPSREL_DENSITY_INT, /*maxNumEval*/ MAX_NUM_EVAL_INT, result, error, &neval);
    int numOutputs = 0;
    if(value) {
        *value = result[numOutputs++];
    }
    if(grad) {
        grad->dx = result[numOutputs++];
        grad->dy = result[numOutputs++];
        grad->dz = 0;
    }
    if(hess) {
        hess->dx2  = result[numOutputs++];
        hess->dy2  = result[numOutputs++];
        hess->dxdy = result[numOutputs++];
        hess->dz2  = hess->dxdz = hess->dydz = 0;
    }
}

void principalAxes(const BaseDensity& dens, double radius,
    /*output*/ double axes[3], double angles[3])
{
    if(isUnknown(dens.symmetry()))
        throw std::runtime_error("symmetry is not provided");
    if(isSpherical(dens)) {
        axes[0] = axes[1] = axes[2] = 1;
        if(angles)
            angles[0] = angles[1] = angles[2] = 0;
        return;
    }
    // TODO: need a special treatment of axisymmetric systems
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {math::scale(math::ScalingSemiInf(), radius), 0.5, 0.25};
    double result[6] = {0};
    coord::Orientation orientation;
    axes[0] = axes[1] = axes[2] = 1.0;
    double prev2[3], prev[3]={0};
    for(int numiter=0, numprev=0; numiter<20; numiter++, numprev++) {
        for(int i=0; i<3; i++) {
            prev2[i] = prev[i];
            prev [i] = axes[i];
        }
        math::integrateNdim(PrincipalAxesIntegrand(dens, axes, orientation),
            xlower, xupper, EPSREL_DENSITY_INT, MAX_NUM_EVAL_INT, result);
        double norm = 0;
        for(int i=0; i<6; i++)
            norm = fmax(norm, fabs(result[i]));
        math::Matrix<double> inertia(3, 3);
        inertia(0, 0) = result[0] / norm;
        inertia(1, 1) = result[1] / norm;
        inertia(2, 2) = result[2] / norm;
        inertia(0, 1) = inertia(1, 0) = result[3] / norm;
        inertia(0, 2) = inertia(2, 0) = result[4] / norm;
        inertia(1, 2) = inertia(2, 1) = result[5] / norm;
        math::SVDecomp svd(inertia);  // equivalent to eigendecomposition for symmetric matrices
        std::vector<double> S = svd.S();
        math::Matrix<double> U = svd.U();
        // U may turn out to have determinant of -1 instead of 1,
        // in which case we need to flip the sign of one column
        double detU =
            U(0,0) * (U(1,1) * U(2,2) - U(1,2) * U(2,1)) +
            U(0,1) * (U(1,2) * U(2,0) - U(1,0) * U(2,2)) +
            U(0,2) * (U(1,0) * U(2,1) - U(1,1) * U(2,0));
        if(detU < 0) {
            U(0,0) *= -1;
            U(1,0) *= -1;
            U(2,0) *= -1;
        }
        for(int i=0; i<3; i++) {
            axes[i] = sqrt(S[i]);
            for(int j=0; j<3; j++)
                orientation.mat[j*3+i] = U(i, j);
        }
        norm = cbrt(axes[0] * axes[1] * axes[2]);
        for(int i=0; i<3; i++)
            axes[i] /= norm;
        if(fabs(prev[0]-axes[0]) + fabs(prev[1]-axes[1]) + fabs(prev[2]-axes[2]) < EPSREL_DENSITY_INT)
            break;
        else if(numprev>=2) {
            // Aitken's acceleration trick - extrapolate the slowly convergent series
            double extr[3];
            for(int i=0; i<3; i++)
                extr[i] = axes[i] - pow_2(axes[i] - prev[i]) / (axes[i] - 2*prev[i] + prev2[i]);
            // accept the result only if it keeps axes in the same order (precaution)
            if(extr[0] >= extr[1] && extr[1] >= extr[2] && extr[2] > 0) {
                norm = cbrt(extr[0] * extr[1] * extr[2]);
                for(int i=0; i<3; i++)
                    axes[i] = extr[i] /= norm;
                numprev = 0;
            }
        }
    }

    // convert the rotation matrix into Euler angles
    if(angles) {
        // this provides angles in the range -pi..pi for alpha & gamma, and 0..pi for beta
        orientation.toEulerAngles(angles[0], angles[1], angles[2]);
        // further restrict beta to the range 0..pi/2
        if(angles[1] > 0.5*M_PI) {
            angles[1] = M_PI-angles[1];
            angles[0] = M_PI+angles[0];
            angles[2] = M_PI-angles[2];
        }
        angles[0] = fmod(angles[0]+M_PI, M_PI*2) - M_PI;  // restrict alpha to -pi..pi
        angles[2] = fmod(angles[2]+M_PI, M_PI);           // restrict gamma to 0..pi
    }
}

double v_circ(const BasePotential& potential, double R)
{
    if(R==0)
        return isFinite(potential.value(coord::PosCyl(0, 0, 0))) ? 0 : INFINITY;
    coord::GradCyl grad;
    Axisymmetrized<BasePotential>(potential).eval(coord::PosCyl(R, 0, 0), NULL, &grad);
    return sqrt(R * grad.dR);
}

double R_circ(const BasePotential& potential, double E) {
    return exp(math::findRoot(RcircRootFinder(potential, E), math::ScalingInf(), ACCURACY_ROOT));
}

double R_from_Lz(const BasePotential& potential, double L) {
    if(L==0)
        return 0;
    if(fabs(L) == INFINITY)
        return INFINITY;
    return exp(math::findRoot(RfromLRootFinder(potential, L), math::ScalingInf(), ACCURACY_ROOT));
}

double R_max(const BasePotential& potential, double E) {
    return exp(math::findRoot(RmaxRootFinder(potential, E), math::ScalingInf(), ACCURACY_ROOT));
}

void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega)
{
    coord::GradCyl grad;
    coord::HessCyl hess;
    Axisymmetrized<BasePotential>(potential).eval(coord::PosCyl(R, 0, 0), NULL, &grad, &hess);
    double gradR_over_R = (R==0 && grad.dR==0) ? hess.dR2 : grad.dR/R;
    // no attempt to check if the expressions under sqrt are non-negative - 
    // they could well be for a physically plausible potential of a flat disk with an inner hole
    kappa = sqrt(hess.dR2 + 3*gradR_over_R);
    nu    = sqrt(hess.dz2);
    Omega = sqrt(gradR_over_R);
}

double innerSlope(const math::IFunction& potential, double* Phi0, double* coef)
{
    // this routine shouldn't suffer from cancellation errors, provided that
    // the potential and its derivatives are computed accurately,
    // thus we may use a fixed tiny radius at which the slope is estimated.
    double r = 1e-10;  // TODO: try making it more scale-invariant?
    double Phi, dPhidR, d2PhidR2;
    potential.evalDeriv(r, &Phi, &dPhidR, &d2PhidR2);
    double  s = 1 + r * d2PhidR2 / dPhidR;
    if(coef)
        *coef = s==0 ?  dPhidR * r  :  dPhidR / s * std::pow(r, 1-s);
    if(Phi0)
        *Phi0 = s==0 ?  Phi - r * dPhidR * log(r)  :  Phi - r * dPhidR / s;
    return s;
}

void findPlanarOrbitExtent(const BasePotential& potential, double E, double L, double& R1, double& R2)
{
    Axisymmetrized<BasePotential> axipot(potential);
    double Phi0, coef, slope = innerSlope(axipot, &Phi0, &coef);

    if(slope>0  &&  E >= Phi0  &&  E < Phi0 * (1-MIN_REL_DIFFERENCE)) {
        // accurate treatment at origin to avoid roundoff errors when Phi -> Phi(r=0),
        // assuming a power-law asymptotic behavior of potential at r->0
        double Rcirc = slope==0 ?  exp((E-Phi0) / coef - 0.5)  :
            std::pow((E-Phi0) / (coef * (1+0.5*slope)), 1/slope);
        if(!isFinite(Rcirc)) {
            R1 = R2 = NAN;
            return;
        }
        if(Rcirc == 0) {  // energy exactly equals the potential at origin
            R1 = R2 = 0;
            return;
        }
        double Lcirc = Rcirc * (slope==0 ? sqrt(coef) : sqrt(coef*slope) * std::pow(Rcirc, 0.5*slope));
        if(L < Lcirc) {
            RPeriApoRootFinderPowerLaw fnc(slope, pow_2(L / Lcirc));
            R1 = Rcirc * fmin(1., math::findRoot(fnc, 0, 1, ACCURACY_ROOT));
            R2 = Rcirc * fmax(1., math::findRoot(fnc, 1, 2, ACCURACY_ROOT));
        } else {
            R1 = R2 = Rcirc;
        }
    } else {  // normal scenario when we don't suffer from roundoff errors 
        double Rcirc = R_circ(potential, E);
        if(!isFinite(Rcirc)) {
            R1 = R2 = NAN;
            return;
        }
        if(Rcirc == 0) {  // energy exactly equals the potential at origin
            R1 = R2 = 0;
            return;
        }
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        axipot.eval(coord::PosCyl(Rcirc, 0, 0), &Phi, &grad, &hess);
        double Lcirc = Rcirc * sqrt(Rcirc * grad.dR);
        if(L >= Lcirc || (!isFinite(Lcirc) && Rcirc <= 1./HUGE_NUMBER))
        {
            // assume an exactly circular orbit (to within roundoff error),
            // i.e., don't panic if the input E and L were incompatible
            R1 = R2 = Rcirc;
        } else if(L > Lcirc * LREL_NEARLY_CIRCULAR ||
            (E-Phi)*pow_2(Rcirc) <= 0.5*L*L /*in this case the root-finder would fail due to roundoff*/)
        {   // asymptotic expressions for nearly circular orbits, when the ordinary method is inefficient
            double offset = sqrt( (1 - pow_2(L/Lcirc)) * grad.dR / (3 * grad.dR + Rcirc * hess.dR2) );
            R1 = Rcirc * (1-offset);
            R2 = Rcirc * (1+offset);
            // root polishing to improve the accuracy of peri/apocenter radii determination
            R1 = fmin(Rcirc, refineRoot(axipot, R1, E, L));
            R2 = fmax(Rcirc, refineRoot(axipot, R2, E, L));
        } else {
            // normal case
            RPeriApoRootFinder fnc(axipot, E, L);
            R1 = math::findRoot(fnc, 0, Rcirc, ACCURACY_ROOT);
            R2 = math::findRoot(fnc, Rcirc, 3*Rcirc, ACCURACY_ROOT);
            // for a reasonable potential, 2*Rcirc is actually an upper limit,
            // but in case of trouble, repeat with a safely larger value (+extra cost of computing Rmax)
            if(!isFinite(R2))
                R2 = math::findRoot(fnc, Rcirc, (1+ACCURACY_ROOT) * R_max(potential, E), ACCURACY_ROOT);
        }
    }
}

void findRoots(const BasePotential& pot, double E,
    double X, double Y, const coord::Orientation& orientation,
    /*output*/ double &Zm, double &Z1, double &Z2)
{
    PotentialFinder(pot, E, X, Y, orientation).findRoots(Zm, Z1, Z2);
}

std::vector<double> createInterpolationGrid(const BasePotential& potential, double accuracy)
{
    // create a grid in log-radius with spacing depending on the local variation of the potential
    std::vector<double> grid = math::createInterpolationGrid(ScalePhi(potential), accuracy);

    // convert to grid in radius
    for(size_t i=0, size=grid.size(); i<size; i++)
        grid[i] = exp(grid[i]);

    // erase innermost grid nodes where the value of potential is too close to Phi(0) (within roundoff)
    double Phi0 = potential.value(coord::PosCyl(0,0,0));
    while(grid.size() > 2  &&  potential.value(coord::PosCyl(grid[0],0,0)) < Phi0 * (1-MIN_REL_DIFFERENCE))
        grid.erase(grid.begin());

    return grid;
}

// -------- Same tasks implemented as an interpolation interface -------- //

Interpolator::Interpolator(const BasePotential& potential) :
    invPhi0(1./potential.value(coord::PosCyl(0,0,0)))
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Interpolator: can only work with axisymmetric potentials");
    double Phiinf = potential.value(coord::PosCyl(INFINITY,1.,1.));
    // not every potential returns a valid value at infinity, but if it does, make sure that it's zero
    if(Phiinf==Phiinf && Phiinf!=0)
        throw std::runtime_error("Interpolator: can only work with potentials "
            "that tend to zero as r->infinity");   // otherwise assume Phiinf==0
    // well-behaved potential must be -INFINITY <= Phi0 < 0
    if(invPhi0 > 0 || !isFinite(invPhi0))
        throw std::runtime_error("Interpolator: potential must be negative at r=0");

    std::vector<double> gridR = createInterpolationGrid(potential, ACCURACY_INTERP);
    unsigned int gridsize = gridR.size();
    std::vector<double>   // various arrays:
    gridLogR(gridsize),   // ln(r)
    gridPhi(gridsize),    // scaled Phi(r)
    gridE(gridsize),      // scaled Ecirc(Rcirc) where Rcirc=r
    gridL(gridsize),      // log(Lcirc)
    gridNu(gridsize),     // ratio of squared epicyclic frequencies nu^2/Omega^2
    gridPhider(gridsize), // d(scaled Phi)/ d(log r)
    gridRder(gridsize),   // d(log Rcirc) / d(log Lcirc)
    gridLder(gridsize);   // d(log Lcirc) / d(scaled Ecirc)

    std::ofstream strm;   // debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open("PotentialInterpolator.log");
        strm << "#R      \tPhi(R)  \tdPhi/dR \td2Phi/dR2\td2Phi/dz2\tEcirc   \tLcirc\n";
    }

    for(unsigned int i=0; i<gridsize; i++) {
        double R = gridR[i];
        gridLogR[i] = log(R);
        double Phival;
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(R, 0, 0), &Phival, &grad, &hess);
        // epicyclic frequencies
        double kappa2= hess.dR2 + 3*grad.dR/R;  // kappa^2
        double Omega = sqrt(grad.dR/R);         // Omega, always exists if potential is monotonic with R
        double nu2Om = hess.dz2 / grad.dR * R;  // ratio of nu^2/Omega^2 - allowed to be negative
        double Ecirc = Phival + 0.5*R*grad.dR;  // energy of a circular orbit at this radius
        double Lcirc = Omega * R*R;             // angular momentum of a circular orbit
        double scaledPhi, dPhidscaledPhi, scaledEcirc, dEcircdscaledEcirc;
        scaleE(Phival, invPhi0, scaledPhi,   dPhidscaledPhi);
        scaleE(Ecirc,  invPhi0, scaledEcirc, dEcircdscaledEcirc);
        gridPhi[i] = scaledPhi;    // log-scaled potential at the radius
        gridE  [i] = scaledEcirc;  // log-scaled energy of a circular orbit at the radius
        gridL  [i] = log(Lcirc);   // log-scaled ang.mom. of a circular orbit
        gridNu [i] = nu2Om;        // ratio of nu^2/Omega^2 
        // also compute the scaled derivatives for the quintic splines
        double dRdL = 2*Omega / (kappa2 * R);
        double dLdE = 1/Omega;
        gridRder  [i] = dRdL * Lcirc / R;  // extra factors are from conversion to log-derivatives
        gridLder  [i] = dLdE * dEcircdscaledEcirc / Lcirc;
        gridPhider[i] = grad.dR * R / dPhidscaledPhi;

        // debugging printout
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm << utils::pp(R, 15) + '\t' +
            utils::pp(Phival,    15) + '\t' +
            utils::pp(grad.dR,   15) + '\t' +
            utils::pp(hess.dR2,  15) + '\t' +
            utils::pp(hess.dz2,  15) + '\t' +
            utils::pp(Ecirc,     15) + '\t' +
            utils::pp(Lcirc,     15) + '\n' << std::flush;
        }

        // guard against weird behaviour of potential
        if(!(Phival<0 && grad.dR>=0 && (i==0 || gridPhi[i]>gridPhi[i-1])))
            throw std::runtime_error(
                "Interpolator: potential is not monotonically increasing with radius at R=" +
                utils::toString(R) + '\n' + utils::stacktrace());
        if(!(Ecirc<0 && Lcirc>=0 && (i==0 || (gridE[i]>gridE[i-1] && gridL[i]>gridL[i-1])) && dRdL>=0))
            throw std::runtime_error(
                "Interpolator: energy or angular momentum of a circular orbit are not monotonic "
                "with radius at R=" + utils::toString(R) + '\n' + utils::stacktrace());
        if(!(nu2Om>=0))  // not a critical error, but possibly a sign of problems
            FILTERMSG(utils::VL_WARNING, "Interpolator",
                "Vertical epicyclic frequency is negative at R=" + utils::toString(R));

        // estimate the outer asymptotic behaviour
        if(i==gridsize-1) {
            double num1 = 2*grad.dR, num2 = -R*hess.dR2, den1 = grad.dR, den2 = -Phival/R;
            slopeOut    = (num1 - num2) / (den1 - den2);
            bool roundoff =    // check if the value of slope is dominated by roundoff errors
                fabs(num1-num2) < fmax(fabs(num1), fabs(num2)) * ROUNDOFF_THRESHOLD ||
                fabs(den1-den2) < fmax(fabs(den1), fabs(den2)) * ROUNDOFF_THRESHOLD;
            if(roundoff || slopeOut>=0) {    // not successful - use the total mass only
                slopeOut= -1;
                coefOut = 0;
                massOut = -Phival * R;
            } else {
                if(fabs(slopeOut+1) < ROUNDOFF_THRESHOLD)
                    slopeOut = -1;   // value for a logarithmically-growing M(r), as in NFW
                coefOut = (Phival + R*grad.dR) * std::pow(R, -slopeOut);
                massOut = -R*Phival + coefOut *
                    (slopeOut==-1 ? log(R) : (std::pow(R, slopeOut+1) - 1) / (slopeOut+1));
            }
        }
    }

    // init various 1d splines
    freqNu = math::CubicSpline  (gridLogR, gridNu,   0, 0);  // set endpoint derivatives to zero
    LofE   = math::QuinticSpline(gridE,    gridL,    gridLder);
    RofL   = math::QuinticSpline(gridL,    gridLogR, gridRder);
    PhiofR = math::QuinticSpline(gridLogR, gridPhi,  gridPhider);
    // inverse relation between R and Phi - the derivative is reciprocal
    for(unsigned int i=0; i<gridsize; i++)
        gridPhider[i] = 1/gridPhider[i];
    RofPhi = math::QuinticSpline(gridPhi, gridLogR, gridPhider);
}

void Interpolator::evalDeriv(
    const double R, double* val, double* deriv, double* deriv2, double* deriv3) const
{
    double logR = log(R);
    if(logR > PhiofR.xvalues().back() && coefOut!=0)
    {  // special care for extrapolation at large r
        double Rs = exp(logR * slopeOut);   // R^slopeOut
        double Phi= (-massOut + (slopeOut==-1 ? logR : (R*Rs-1) / (slopeOut+1)) * coefOut ) / R;
        if(val)
            *val = Phi;
        if(deriv)
            *deriv = (-Phi + coefOut * Rs) / R;
        if(deriv2)
            *deriv2 = (2 * Phi + coefOut * Rs * (slopeOut-2) ) / pow_2(R);
        if(deriv3)
            *deriv3 = (-6 * Phi + coefOut * Rs * (pow_2(slopeOut-2) + 2)) / pow_3(R);
        return;
    }
    double scaledPhi, dscaledPhi_dlogR, d2scaledPhi_dlogR2, d3scaledPhi_dlogR3;
    PhiofR.evalDeriv(logR, &scaledPhi,
        deriv3 || deriv2 || deriv ? &dscaledPhi_dlogR : NULL,
        deriv3 || deriv2 ? &d2scaledPhi_dlogR2 : NULL,
        deriv3 ? &d3scaledPhi_dlogR3 : NULL);
    double Phival, dPhi_dscaledPhi, d2Phi_dscaledPhi2;
    unscaleE(scaledPhi, invPhi0, Phival, dPhi_dscaledPhi, d2Phi_dscaledPhi2);
    if(val)
        *val    = Phival;
    if(deriv)
        *deriv  = dPhi_dscaledPhi * dscaledPhi_dlogR / R;
    if(deriv2)
        *deriv2 = ( dPhi_dscaledPhi * (d2scaledPhi_dlogR2 - dscaledPhi_dlogR) +
            d2Phi_dscaledPhi2 * pow_2(dscaledPhi_dlogR) ) / pow_2(R);
    if(deriv3) {
        double d3Phi_dscaledPhi3 = dPhi_dscaledPhi * (1 + 6 * dPhi_dscaledPhi * invPhi0);
        *deriv3 =
            (dPhi_dscaledPhi  * (d3scaledPhi_dlogR3 - d2scaledPhi_dlogR2 * 3 + dscaledPhi_dlogR * 2) +
            d2Phi_dscaledPhi2 * (d2scaledPhi_dlogR2 -  dscaledPhi_dlogR) * 3 * dscaledPhi_dlogR +
            d3Phi_dscaledPhi3 * pow_3(dscaledPhi_dlogR) ) / pow_3(R);
    }
}

double Interpolator::innerSlope(double* Phi0, double* coef) const
{
    double val, der, logr = PhiofR.xvalues().front();
    PhiofR.evalDeriv(logr, &val, &der);
    double Phival, dummy1, dummy2;
    unscaleE(val, invPhi0, Phival, dummy1, dummy2);
    if(invPhi0!=0) {
        double slope = der * Phival * invPhi0;
        if(Phi0)
            *Phi0 = 1/invPhi0;
        if(coef)
            *coef = (Phival - 1/invPhi0) * exp(-logr * slope);
        return slope;
    } else {
        if(Phi0)
            *Phi0 = 0;  // we don't have a more accurate approximation in this case
        if(coef)
            *coef = Phival * exp(logr * der);
        return -der;
    }
}

double Interpolator::R_max(const double E, double* deriv) const
{
    double scaledE, dEdscaledE, logR;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    RofPhi.evalDeriv(scaledE, &logR, deriv);
    double R = exp(logR);
    if(logR > PhiofR.xvalues().back()) {
        // extra correction step at large r because of non-trivial extrapolation of potential
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(R, &Phi, &dPhidR, &d2PhidR2);
        R -= math::clip(   // cautionary measure to avoid too large corrections
            (Phi-E) / (dPhidR - 0.5 * (Phi-E) * d2PhidR2 / dPhidR),   // Halley correction
            -0.25*R, 0.25*R);
    }
    if(deriv)
        *deriv *= R / dEdscaledE;
    return R;
}

double Interpolator::L_circ(const double E, double* deriv) const
{
    if(!(E>=1./invPhi0 && E<=0)) {
        if(deriv)
            *deriv = NAN;
        return NAN;
    }
    double scaledE, dEdscaledE, logL, logLder;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &logL, deriv!=NULL ? &logLder : NULL);
    double Lcirc = exp(logL);
    if(scaledE > LofE.xvalues().back()) {
        // extra correction step at large radii
        double Rcirc = exp(RofL(logL));  // first get an approximation for Rcirc
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        double Ecirc = Phi + 0.5 * Rcirc * dPhidR;
        double denom = 1 - 0.5 * (Ecirc-E) * (Rcirc * d2PhidR2 - dPhidR) /
            ((Rcirc * d2PhidR2 + 3 * dPhidR) * Rcirc * dPhidR);
        Lcirc = math::clip(   // cautionary measure to avoid too large corrections
            sqrt(Rcirc * dPhidR) * (Rcirc - (Ecirc-E) / (dPhidR * denom)),   // Halley correction
            0.75*Lcirc, 1.25*Lcirc);
    }
    if(deriv)
        *deriv = logLder / dEdscaledE * Lcirc;
    return Lcirc;
}

double Interpolator::R_from_Lz(const double Lz, double* deriv) const
{
    double logL = log(fabs(Lz)), logR, logRder;
    RofL.evalDeriv(logL, &logR, deriv!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(logL > RofL.xvalues().back()) {
        // extra correction step at large radii
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        Rcirc -= math::clip(   // cautionary measure to avoid too large corrections
            (Rcirc * dPhidR - pow_2(Lz/Rcirc)) / (3 * dPhidR + Rcirc * d2PhidR2),   // Newton correction
            -0.25*Rcirc, 0.25*Rcirc);
        // even though this is Newton (1st order), not Halley (2nd order) correction,
        // it seems to be fairly accurate
    }
    if(deriv)
        *deriv = logRder * Rcirc / Lz;
    return Rcirc;
}

double Interpolator::R_circ(const double E, double* deriv) const
{
    if(!(E>=1./invPhi0 && E<=0)) {
        if(deriv)
            *deriv = NAN;
        return NAN;
    }
    double scaledE, dEdscaledE, logL, logLder, logR, logRder;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &logL, deriv!=NULL ? &logLder : NULL);
    RofL.evalDeriv(logL,    &logR, deriv!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(logL > RofL.xvalues().back()) {
        // extra correction step at large radii
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        Rcirc -= math::clip(   // cautionary measure to avoid too large corrections
            ( 2*(Phi-E) + Rcirc * dPhidR ) / (3 * dPhidR + Rcirc * d2PhidR2),   // Newton correction
            -0.25*Rcirc, 0.25*Rcirc);
        // this is only 1st order correction, and could be improved by another iteration,
        // but we leave it as it is
    }
    if(deriv)
        *deriv = logLder * logRder / dEdscaledE * Rcirc;
    return Rcirc;
}

void Interpolator::epicycleFreqs(double R,
    double& kappa, double& nu, double& Omega, double derivs[3]) const
{
    double dPhi_dR, d2Phi_dR2, d3Phi_dR3, nu2_over_Omega2, dnu2_over_Omega2_dlogR;
    evalDeriv(R, NULL, &dPhi_dR, &d2Phi_dR2, derivs ? &d3Phi_dR3 : NULL);
    freqNu.evalDeriv(log(R), &nu2_over_Omega2, derivs ? &dnu2_over_Omega2_dlogR : NULL);
    // correct limit at r->0 if dPhi/dr->0 too
    double dPhi_dR_over_R = R>0 || dPhi_dR!=0 ? dPhi_dR/R : d2Phi_dR2;
    kappa = sqrt(d2Phi_dR2 + 3 * dPhi_dR_over_R);
    nu    = sqrt(nu2_over_Omega2 * dPhi_dR_over_R);  // nu^2 = Omega^2 * spline-interpolated fnc
    Omega = sqrt(dPhi_dR_over_R);
    if(derivs) {
        derivs[0] = 0.5 * ( d3Phi_dR3 + 3 * (d2Phi_dR2 - dPhi_dR_over_R) / R) / kappa;
        derivs[1] = 0.5 * ((d2Phi_dR2 - dPhi_dR_over_R) * nu2_over_Omega2 +
            dPhi_dR_over_R * dnu2_over_Omega2_dlogR) / nu / R;
        derivs[2] = 0.5 * ( d2Phi_dR2 - dPhi_dR_over_R) / Omega / R;
    }
}


// --------- 2d interpolation of peri/apocenter radii in equatorial plane --------- //

Interpolator2d::Interpolator2d(const BasePotential& potential) :
    Interpolator(potential),
    invPhi0(1./potential.value(coord::PosCyl(0,0,0)))  // -infinity <= Phi(0) < 0
{
    std::vector<double> gridR = createInterpolationGrid(potential, ACCURACY_INTERP2);

    // interpolation grid in scaled variables: X = scaledE = log(1/Phi(0)-1/E), Y = L / Lcirc(E)
    const int sizeE = gridR.size();
    const int sizeL = GRID_SIZE_L;
    std::vector<double> gridX(sizeE), gridY(sizeL);

    // create a non-uniform grid in Y = L/Lcirc(E), using a transformation of interval [0:1]
    // onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at Y=0 and Y=1
    math::ScalingQui scaling(0, 1);
    for(int i=0; i<sizeL; i++)
        gridY[i] = math::unscale(scaling, 1. * i / (sizeL-1));

    // 2d grids for scaled peri/apocenter radii W1, W2 and their derivatives in {X,Y}:
    // W1 = (R1 / Rc - 1)^2, same for W2, where
    // R1 and R2 are the peri/apocenter radii, and Rc is the radius of a circular orbit;
    // W1 and W2 are exactly zero when L=Lcirc (equivalently Y=1), and vary linearly as Y -> 1
    math::Matrix<double> gridW1  (sizeE, sizeL), gridW2  (sizeE, sizeL);
    math::Matrix<double> gridW1dX(sizeE, sizeL), gridW1dY(sizeE, sizeL);
    math::Matrix<double> gridW2dX(sizeE, sizeL), gridW2dY(sizeE, sizeL);

    std::string errorMessage;  // store the error text in case of an exception in the openmp block
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iE=0; iE<sizeE; iE++) {
        try{
            double Rc = gridR[iE];
            double Phi;
            coord::GradCyl grad;
            coord::HessCyl hess;
            potential.eval(coord::PosCyl(Rc, 0, 0), &Phi, &grad, &hess);
            double E  = Phi + 0.5*Rc*grad.dR;   // energy of a circular orbit at this radius
            double Lc = Rc * sqrt(Rc*grad.dR);  // angular momentum of a circular orbit
            double dEdX;                        // dE / d scaledE
            scaleE(E, invPhi0, /*output*/gridX[iE], dEdX);
            double Om2kap2= grad.dR / (3*grad.dR + Rc*hess.dR2);  // ratio of epi.freqs (Omega / kappa)^2
            double dRcdE  = 2/grad.dR * Om2kap2;
            double dLcdE  = Rc*Rc/Lc;
            for(int iL=0; iL<sizeL-1; iL++) {
                double L = Lc * gridY[iL];
                double R1, R2, Phi;
                if(iL==0) {  // exact values for a radial orbit
                    R1=0;
                    R2=potential::R_max(potential, E);
                } else
                    potential::findPlanarOrbitExtent(potential, E, L, R1, R2);
                gridW1(iE, iL) = pow_2(R1 / Rc - 1);
                gridW2(iE, iL) = pow_2(R2 / Rc - 1);
                // compute derivatives of Rperi/apo w.r.t. E and L/Lcirc
                potential.eval(coord::PosCyl(R1,0,0), &Phi, &grad);
                if(R1==0) grad.dR=0;   // it won't be used anyway, but prevents a possible NaN
                double dW1dE = (1 / (E-Phi) - 2 * dLcdE / Lc) / (grad.dR / (E-Phi) - 2 / R1);
                double dW1dY = -Lc * M_SQRT2 / (grad.dR * R1 / sqrt(E-Phi) - 2*sqrt(E-Phi));
                potential.eval(coord::PosCyl(R2,0,0), &Phi, &grad);
                double dW2dE = (1 - 2*(E-Phi) * dLcdE / Lc) / (grad.dR - 2*(E-Phi) / R2);
                double dW2dY = -Lc * L / (grad.dR * pow_2(R2) - 2*(E-Phi) * R2);
                gridW1dX(iE, iL) = 2*(R1-Rc) / pow_2(Rc) * (dW1dE - R1/Rc * dRcdE) * dEdX;
                gridW1dY(iE, iL) = 2*(R1-Rc) / pow_2(Rc) *  dW1dY;
                gridW2dX(iE, iL) = 2*(R2-Rc) / pow_2(Rc) * (dW2dE - R2/Rc * dRcdE) * dEdX;
                gridW2dY(iE, iL) = 2*(R2-Rc) / pow_2(Rc) *  dW2dY;
            }
            // limiting values for a nearly circular orbit:
            // R{1,2} = Rcirc * (1 +- Omega/kappa * ecc),
            // where ecc = sqrt(1 - (L/Lcirc)^2) = sqrt(1-Y^2)
            gridW1  (iE, sizeL-1) = gridW2  (iE, sizeL-1) = 0;
            gridW1dX(iE, sizeL-1) = gridW2dX(iE, sizeL-1) = 0;
            gridW1dY(iE, sizeL-1) = gridW2dY(iE, sizeL-1) = -2*Om2kap2;
        }
        catch(std::exception& e) {
            errorMessage = e.what();
        }
    }

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {   // debugging output
        std::ofstream strm("PotentialInterpolator2d.log");
        strm << "# X=scaledE    \tY=L/Lcirc      \t"
            "W1=scaledRperi \tdW1/dX         \tdW1/dY         \t"
            "W2=scaledRapo  \tdW2/dX         \tdW2/dY         \n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                strm <<
                utils::pp(gridX[iE], 15) + "\t" +
                utils::pp(gridY[iL], 15) + "\t" +
                utils::pp(gridW1  (iE, iL), 15) + "\t" +
                utils::pp(gridW1dX(iE, iL), 15) + "\t" +
                utils::pp(gridW1dY(iE, iL), 15) + "\t" +
                utils::pp(gridW2  (iE, iL), 15) + "\t" +
                utils::pp(gridW2dX(iE, iL), 15) + "\t" +
                utils::pp(gridW2dY(iE, iL), 15) + "\n";
            }
            strm << "\n";
        }
    }

    if(!errorMessage.empty())
        throw std::runtime_error("Interpolator2d: "+errorMessage);

    // create 2d interpolators
    intR1 = math::QuinticSpline2d(gridX, gridY, gridW1, gridW1dX, gridW1dY);
    intR2 = math::QuinticSpline2d(gridX, gridY, gridW2, gridW2dX, gridW2dY);
}

void Interpolator2d::findPlanarOrbitExtent(double E, double L,
    double &R1, double &R2) const
{
    double Lc   = L_circ(E);
    double Rc   = R_from_Lz(Lc);
    double Lrel = Lc>0 ? math::clip(fabs(L/Lc), 0., 1.) : 0;
    double scaledE, dEdscaledE;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    scaledE = math::clip(scaledE, intR1.xmin(), intR1.xmax());
    R1 = (1 - sqrt(intR1.value(scaledE, Lrel))) * Rc;
    R2 = (1 + sqrt(intR2.value(scaledE, Lrel))) * Rc;
    // one iteration of root polishing
    R1 = fmin(refineRoot(*this, R1, E, L), Rc);
    R2 = fmax(refineRoot(*this, R2, E, L), Rc);
}


//---- Correspondence between h and E ----//

PhaseVolume::PhaseVolume(const math::IFunction& pot)
{
    double Phi0 = pot(0);
    if(!(Phi0<0))
        throw std::invalid_argument("PhaseVolume: invalid value of Phi(r=0)");
    invPhi0 = 1/Phi0;

    // create grid in log(r)
    std::vector<double> gridr = math::createInterpolationGrid(
        ScalePhi(FunctionToPotentialWrapper(pot)), ACCURACY_INTERP);
    for(size_t i=0; i<gridr.size(); i++)
        gridr[i] = exp(gridr[i]);  // convert to grid in r
    std::vector<double> gridE;
    gridE.reserve(gridr.size());

    // compute the potential at each node of the radial grid, throwing away nodes that are
    // too closely spaced, such that the difference between adjacent potential values suffers from
    // roundoff/cancellation errors due to finite precision of floating-point arithmetic
    double prevPhi = Phi0;
    for(size_t i=0; i<gridr.size(); ) {
        double E = pot.value(gridr[i]);
        if(i>0 && !(E>=gridE[i-1]))
            throw std::invalid_argument(
                "PhaseVolume: potential is non-monotonic at r="+utils::toString(gridr[i]));
        if(E > prevPhi * (1-MIN_REL_DIFFERENCE)) {
            gridE.push_back(E);
            i++;
            prevPhi = E;
        } else {
            gridr.erase(gridr.begin()+i);
        }
    }
    size_t gridsize = gridr.size();
    if(gridsize == 0)
        throw std::runtime_error("PhaseVolume: cannot construct a suitable grid in radius");

    std::vector<double> gridH(gridsize), gridG(gridsize);
    const double *glnodes1 = math::GLPOINTS[GLORDER1], *glweights1 = math::GLWEIGHTS[GLORDER1];
    const double *glnodes2 = math::GLPOINTS[GLORDER2], *glweights2 = math::GLWEIGHTS[GLORDER2];

    // loop through all grid segments, and in each segment add the contribution to integrals
    // in all other segments leftmost of the current one (thus the complexity is Ngrid^2,
    // but the number of potential evaluations is only Ngrid * GLORDER).
    for(size_t i=0; i<gridsize; i++) {
        double deltar = gridr[i] - (i>0 ? gridr[i-1] : 0);
        // choose a higher-order quadrature rule for longer grid segments
        int glorder = i>0 && gridr[i] < gridr[i-1]*GLRATIO ? GLORDER1 : GLORDER2;
        const double *glnodes   = glorder == GLORDER1 ? glnodes1   : glnodes2;
        const double *glweights = glorder == GLORDER1 ? glweights1 : glweights2;
        for(int k=0; k<glorder; k++) {
            // node of Gauss-Legendre quadrature within the current segment (r[i-1] .. r[i]);
            // the integration variable y ranges from 0 to 1, and r(y) is defined below
            double y = glnodes[k];
            double r = gridr[i] - pow_2(1-y) * deltar;
            double E = pot.value(r);
            // contribution of this point to each integral on the current segment, taking into account
            // the transformation of variable y -> r  and the common weight factor r^2
            double weight = glweights[k] * 2*(1-y) * deltar * pow_2(r);
            // add a contribution to the integrals expressing g(E_j) and h(E_j) for all E_j > Phi(r[i])
            for(size_t j=i; j<gridsize; j++) {
                double v  = sqrt(fmax(0, gridE[j] - E));
                gridG[j] += weight * v * 1.5;
                gridH[j] += weight * pow_3(v);
            }
        }
    }

    // debugging output: asymptotic slopes
    if(utils::verbosityLevel >= utils::VL_DEBUG) {
        double inner = gridH.front() / gridG.front() / (gridE.front() - (isFinite(Phi0) ? Phi0 : 0));
        double outer = gridH.back()  / gridG.back()  / gridE.back();
        utils::msg(utils::VL_DEBUG, "PhaseVolume", "Potential asymptotes: "
            "Phi(r) ~ r^" + utils::toString( 6 * inner / (2 - 3 * inner), 8) + " at small r, "
            "Phi(r) ~ r^" + utils::toString( 6 * outer / (2 - 3 * outer), 8) + " at large r.");
    }

    std::ofstream strm;   // debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open("PhaseVolume.log");
        strm << "#r      \tE       \th       \tg=dh/dE\n";
    }

    // convert h, g and E to scaled coordinates
    for(size_t i=0; i<gridsize; i++) {
        double E = gridE[i], H = gridH[i], G = gridG[i], scaledE, dEdscaledE;
        scaleE(E, invPhi0, scaledE, dEdscaledE);
        gridE[i] = scaledE;
        gridH[i] = log(H) + log(16*M_PI*M_PI/3*2*M_SQRT2);
        gridG[i] = G / H * dEdscaledE;
        // debugging printout
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm <<
                utils::pp(gridr[i], 15) + '\t' +
                utils::pp(E, 15) + '\t' +
                utils::pp(H, 15) + '\t' +
                utils::pp(G, 15) + '\n' << std::flush;
        }
    }

    HofE = math::QuinticSpline(gridE, gridH, gridG);
    // inverse relation between E and H - the derivative is reciprocal
    for(size_t i=0; i<gridG.size(); i++)
        gridG[i] = 1/gridG[i];
    EofH = math::QuinticSpline(gridH, gridE, gridG);
}

void PhaseVolume::evalDeriv(const double E, double* h, double* g, double*) const
{
    // out-of-bounds value of energy returns 0 or infinity, but not NAN
    if(!(E * invPhi0 < 1)) {
        if(h) *h=0;
        if(g) *g=0;
        return;
    }
    if(E>=0) {
        if(h) *h=INFINITY;
        if(g) *g=INFINITY;
        return;
    }
    double scaledE, dEdscaledE, val;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    HofE.evalDeriv(scaledE, &val, g);
    val = exp(val);
    if(h)
        *h = val;
    if(g)
        *g *= val / dEdscaledE;
}

double PhaseVolume::E(const double h, double* g, double* dgdh) const
{
    if(h==0) {
        if(g) *g=0;
        return invPhi0 == 0 ? -INFINITY : 1/invPhi0;
    }
    if(h==INFINITY) {
        if(g) *g=INFINITY;
        return 0;
    }
    double scaledE, dEdscaledE, d2EdscaledE2, realE, dscaledEdlogh, d2scaledEdlogh2;
    EofH.evalDeriv(log(h), &scaledE,
        g!=NULL || dgdh!=NULL ? &dscaledEdlogh : NULL,
        dgdh!=NULL ? &d2scaledEdlogh2 : NULL);
    unscaleE(scaledE, invPhi0, realE, dEdscaledE, d2EdscaledE2);
    if(g)
        *g = h / ( dEdscaledE * dscaledEdlogh );
    if(dgdh)
        *dgdh = ( (1 - d2scaledEdlogh2 / dscaledEdlogh) / dscaledEdlogh -
            d2EdscaledE2 / dEdscaledE ) / dEdscaledE;
    return realE;
}

double PhaseVolume::deltaE(const double logh1, const double logh2, double* g1) const
{
    //return E(exp(logh1), g1) - E(exp(logh2)); //<-- naive implementation
    double scaledE1, scaledE2, E1, E2, E1minusE2, scaledE1deriv;
    EofH.evalDeriv(logh1, &scaledE1, g1);
    EofH.evalDeriv(logh2, &scaledE2);
    unscaleDeltaE(scaledE1, scaledE2, invPhi0, E1, E2, E1minusE2, scaledE1deriv);
    if(g1)
        *g1 = exp(logh1) / *g1 / scaledE1deriv;
    return E1minusE2;
}

}  // namespace potential
