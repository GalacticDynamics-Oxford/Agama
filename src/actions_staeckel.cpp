#include "actions_staeckel.h"
#include "potential_perfect_ellipsoid.h"
#include "math_core.h"
#include "math_fit.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

// debugging output
#include <fstream>

namespace actions{

namespace {  // internal routines

/** Accuracy of integrals for computing actions and angles
    is determined by the number of points in fixed-order Gauss-Legendre scheme with an order
    that depends on the orbit eccentricity, varying from the value below up to MAX_GL_ORDER */
static const unsigned int INTEGR_ORDER = 10;

/** order of Gauss-Legendre quadrature for actions, frequencies and angles:
    use a higher order for more eccentric orbits, as indicated by the ratio
    of pericenter to apocenter radii (R1/R2) */
inline unsigned int integrOrder(double R1overR2) {
    if(R1overR2==0)
        return math::MAX_GL_ORDER;
    int log2;  // base-2 logarithm of R1/R2
    frexp(R1overR2, &log2);
    return std::min<int>(math::MAX_GL_ORDER, INTEGR_ORDER - log2);
}

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over */
static const double ACCURACY_RANGE = 1e-6;

/** minimum range of variation of nu, lambda that is considered to be non-zero */
static const double MINIMUM_RANGE = 1e-10;

/// accuracy parameter determining the spacing of the interpolation grid along the energy axis
static const double ACCURACY_INTERP2 = 1e-4;

// ------ Data structures for both Axisymmetric Staeckel and Fudge action-angle finders ------

/** integration intervals for actions and angles
    (shared between Staeckel and Fudge action finders). */
struct AxisymIntLimits {
    double lambda_min, lambda_max, nu_min, nu_max;
    int integrOrder;   ///< order of GL quadrature that depends on the eccentricity
};

/** Derivatives of actions by integrals of motion */
struct AxisymActionDerivatives {
    double dJrdE, dJrdI3, dJrdLz, dJzdE, dJzdI3, dJzdLz;
};

/** Derivatives of integrals of motion over actions (do not depend on angles).
    Note that dE/dJ are the frequencies of oscillations in three directions, 
    so that we reuse the `Frequencies` struct members (Omega***), 
    and add other derivatives with a somewhat different naming convention. */
struct AxisymIntDerivatives: public Frequencies {
    double dI3dJr, dI3dJz, dI3dJphi, dLzdJr, dLzdJz, dLzdJphi;
};

/** Derivatives of generating function over integrals of motion (depend on angles) */
struct AxisymGenFuncDerivatives {
    double dSdE, dSdI3, dSdLz;
};

/** aggregate class that contains the point in prolate spheroidal coordinates,
    integrals of motion, and reference to the potential,
    shared between Axisymmetric Staeckel  and Axisymmetric Fudge action finders;
    only the coordinates and the two classical integrals are in common between them.
    It also implements the IFunction interface, providing the "auxiliary function" 
    that is used in finding the integration limits and computing actions and angles;
    this function F(tau) is related to the canonical momentum p(tau)  as 
    \f$  p(tau)^2 = F(tau) / (2*(tau+alpha)^2*(tau+gamma))  \f$,
    and the actual implementation of this function is specific to each descendant class.
*/
class AxisymFunctionBase: public math::IFunction {
public:
    coord::PosVelProlSph point; ///< position/derivative in prolate spheroidal coordinates
    const double E;             ///< total energy
    const double Lz;            ///< z-component of angular momentum
    const double I3;            ///< third integral
    AxisymFunctionBase(const coord::PosVelProlSph& _point, double _E, double _Lz, double _I3) :
        point(_point), E(_E), Lz(_Lz), I3(_I3) {};
};

// ------ SPECIALIZED functions for Staeckel action finder -------

/** parameters of potential, integrals of motion, and prolate spheroidal coordinates 
    SPECIALIZED for the Axisymmetric Staeckel action finder */
class AxisymFunctionStaeckel: public AxisymFunctionBase {
public:
    const math::IFunction& fncG;  ///< single-variable function of a Staeckel potential
    AxisymFunctionStaeckel(const coord::PosVelProlSph& _point, double _E, double _Lz, double _I3,
        const math::IFunction& _fncG) :
        AxisymFunctionBase(_point, _E, _Lz, _I3), fncG(_fncG) {};

    /** auxiliary function that enters the definition of canonical momentum for 
        for the Staeckel potential: it is the numerator of eq.50 in de Zeeuw(1985);
        the argument tau is replaced by tau+gamma >= 0. */
    virtual void evalDeriv(const double tauplusgamma, 
        double* value=0, double* deriv=0, double* deriv2=0) const;
    virtual unsigned int numDerivs() const { return 2; }
};

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system 
*/
AxisymFunctionStaeckel findIntegralsOfMotionOblatePerfectEllipsoid(
    const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    double E = totalEnergy(potential, point);
    double Lz= coord::Lz(point);
    const coord::ProlSph& coordsys = potential.coordsys();
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double Glambda;
    potential.evalDeriv(pprol.lambda, &Glambda);
    double I3;
    if(point.z==0)   // special case: nu=0
        I3 = 0.5 * pow_2(point.vz) * (pow_2(point.R) + coordsys.Delta2);
    else   // general case: eq.3 in Sanders(2012)
        I3 = fmax(0,
            pprol.lambda * (E - pow_2(Lz) / 2 / (pprol.lambda - coordsys.Delta2) + Glambda) -
            pow_2( pprol.lambdadot * (pprol.lambda - fabs(pprol.nu)) ) / 
            (8 * (pprol.lambda - coordsys.Delta2) * pprol.lambda) );
    return AxisymFunctionStaeckel(pprol, E, Lz, I3, potential);
}

/** auxiliary function that enters the definition of canonical momentum for 
    for the Staeckel potential: it is the numerator of eq.50 in de Zeeuw(1985);
    except that in our convention `tau` >= 0 is equivalent to `tau+gamma` from that paper. */
void AxisymFunctionStaeckel::evalDeriv(const double tau, 
    double* val, double* der, double* der2) const
{
    assert(tau>=0);
    double G, dG, d2G;
    fncG.evalDeriv(tau, &G, der || der2 ? &dG : NULL, der2 ? &d2G : NULL);
    const double tauminusdelta = tau - point.coordsys.Delta2;
    if(val)
        *val = ( (E + G) * tau - I3 ) * tauminusdelta - Lz*Lz/2 * tau;
    if(der)
        *der = (E + G) * (tau + tauminusdelta) + dG * tau * tauminusdelta - I3 - Lz*Lz/2;
    if(der2)
        *der2 = 2 * (E + G) + 2 * dG * (tau + tauminusdelta) + d2G * tau * tauminusdelta;
}

// -------- SPECIALIZED functions for the Axisymmetric Fudge action finder --------

/** parameters of potential, integrals of motion, and prolate spheroidal coordinates 
    SPECIALIZED for the Axisymmetric Fudge action finder */
class AxisymFunctionFudge: public AxisymFunctionBase {
public:
    const double flambda, fnu;  ///< values of separable potential functions at the given point
    const potential::BasePotential& poten;  ///< gravitational potential
    AxisymFunctionFudge(const coord::PosVelProlSph& _point, double _E, double _Lz, double _I3,
        double _flambda, double _fnu, const potential::BasePotential& _poten) :
        AxisymFunctionBase(_point, _E, _Lz, _I3), flambda(_flambda), fnu(_fnu), poten(_poten) {};

    /** Auxiliary function F analogous to that of Staeckel action finder:
        namely, the momentum is given by  p_tau^2 = F(tau) / [2 tau (tau-delta)^2],
        where    0 <= tau < delta    for the  nu-component of momentum, 
        and  delta <= tau < infinity for the  lambda-component of momentum.
        For numerical convenience, tau is replaced by x=tau+gamma. */
    virtual void evalDeriv(const double tauplusgamma, 
        double* value=0, double* deriv=0, double* deriv2=0) const;
    virtual unsigned int numDerivs() const { return 2; }
};

/** compute true (E, Lz) and approximate (Ilambda, Inu) integrals of motion in an arbitrary 
    potential used for the Staeckel Fudge, 
    together with the coordinates in its prolate spheroidal coordinate system 
*/
AxisymFunctionFudge findIntegralsOfMotionAxisymFudge(
    const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, 
    const coord::ProlSph& coordsys)
{
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    const double
    Phi  = potential.value(point),
    Ekin = 0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi)),
    E    = Phi + Ekin,
    Lz   = coord::Lz(point),
    Phi0 = potential.value(coord::PosCyl(sqrt(pprol.lambda-coordsys.Delta2), 0, point.phi)),
    fla  = -pprol.lambda * Phi0,
    fnu  = fla + Phi * (pprol.lambda - fabs(pprol.nu)),
    I3   = pprol.lambda * (Phi - Phi0) + 0.5 * (
        pow_2(point.z * point.vphi) +
        pow_2(point.R * point.vz - point.z * point.vR) +
        pow_2(point.vz) * coordsys.Delta2 );
    return AxisymFunctionFudge(pprol, E, Lz, I3, fla, fnu, potential);
}

/** Auxiliary function F analogous to that of Staeckel action finder:
    namely, the momentum is given by  p_tau^2 = F(tau) / (2*(tau-delta)^2*tau),
    where    0 <= tau <= delta    for the  nu-component of momentum, 
    and  delta <= tau < infinity  for the  lambda-component of momentum.
*/
void AxisymFunctionFudge::evalDeriv(const double tau, 
    double* val, double* der, double* der2) const
{
    assert(tau>=0);
    double lambda, nu, I, mult;
    if(tau >= point.coordsys.Delta2) {  // evaluating J_lambda
        lambda= tau;
        nu    = fabs(point.nu);
        mult  = lambda - nu;
        I     = I3 - fnu;
    } else {    // evaluating J_nu
        lambda= point.lambda;
        nu    = tau;
        mult  = nu - lambda;
        I     = I3 - flambda;
    }
    // compute the potential in coordinates transformed from prol.sph. to cylindrical
    coord::PosDerivT  <coord::ProlSph, coord::Cyl> coordDeriv;
    coord::PosDeriv2T <coord::ProlSph, coord::Cyl> coordDeriv2;
    const coord::PosProlSph posProl(lambda, nu, point.phi, point.coordsys);
    const coord::PosCyl posCyl = der || der2? 
        coord::toPosDeriv<coord::ProlSph, coord::Cyl>(posProl, &coordDeriv, der2? &coordDeriv2 : NULL) :
        coord::toPosCyl(posProl);
    double Phi;
    coord::GradCyl gradCyl;
    coord::HessCyl hessCyl;
    poten.eval(posCyl, &Phi, der || der2? &gradCyl : NULL, der2? &hessCyl : NULL);
    const double tauminusdelta = tau - point.coordsys.Delta2;
    if(val)
        *val = (tauminusdelta!=0 ? (E * tau- I - Phi * mult) * tauminusdelta : 0) - pow_2(Lz)/2 * tau;
    if(der || der2) {
        coord::GradProlSph gradProl = coord::toGrad<coord::Cyl, coord::ProlSph> (gradCyl, coordDeriv);
        double dPhidtau = (tau >= point.coordsys.Delta2) ? gradProl.dlambda : gradProl.dnu;
        if(der)
            *der = E * (tau+tauminusdelta) - pow_2(Lz)/2 - I 
                 - (mult+tauminusdelta) * Phi - (tauminusdelta!=0 ? tauminusdelta * mult * dPhidtau : 0);
        if(der2) {
            double d2Phidtau2 = (tau >= point.coordsys.Delta2) ?
                // d2Phi/dlambda^2
                hessCyl.dR2 * pow_2(coordDeriv.dRdlambda) + hessCyl.dz2 * pow_2(coordDeriv.dzdlambda) + 
                2*hessCyl.dRdz * coordDeriv.dRdlambda*coordDeriv.dzdlambda +
                gradCyl.dR * coordDeriv2.d2Rdlambda2 + gradCyl.dz * coordDeriv2.d2zdlambda2
            :   // d2Phi/dnu^2
                hessCyl.dR2 * pow_2(coordDeriv.dRdnu) + hessCyl.dz2 * pow_2(coordDeriv.dzdnu) + 
                2*hessCyl.dRdz * coordDeriv.dRdnu*coordDeriv.dzdnu +
                gradCyl.dR * coordDeriv2.d2Rdnu2 + gradCyl.dz * coordDeriv2.d2zdnu2;
            *der2 = 2 * (E - Phi - (mult+tauminusdelta) * dPhidtau)
                  - tauminusdelta * mult * d2Phidtau2;
        }
    }
}

// -------- COMMON routines for Staeckel and Fudge action finders --------
/** parameters for the function that computes actions and angles
    by integrating an auxiliary function "fnc" as follows:
    the canonical momentum is   p^2(tau) = fnc(tau) / [2 (tau-delta)^2 tau ];
    the integrand is given by   p^n * (tau-delta)^a * tau^c  if p^2>0, otherwise 0.
*/
class AxisymIntegrand: public math::IFunctionNoDeriv {
public:
    const AxisymFunctionBase& fnc;      ///< parameters of aux.fnc. (Staeckel or Fudge)
    enum { nplus1, nminus1 } n;         ///< power of p: +1 or -1
    enum { azero, aminus1, aminus2 } a; ///< power of (tau-delta): 0, -1, -2
    enum { czero, cminus1 } c;          ///< power of tau: 0 or -1
    explicit AxisymIntegrand(const AxisymFunctionBase& d) : fnc(d) {};

    /** integrand for the expressions for actions and their derivatives 
        (e.g.Sanders 2012, eqs. A1, A4-A12).  It uses the auxiliary function to compute momentum,
        and multiplies it by some powers of (tau-delta) and tau.
    */
    virtual double value(const double tau) const {
        assert(tau>=0);
        const coord::ProlSph& CS = fnc.point.coordsys;
        const double tauminusdelta = tau - CS.Delta2;
        const double p2 = fnc(tau) / 
            (2*pow_2(tauminusdelta)*tau);
        if(p2<0)
            return 0;
        double result = sqrt(p2);
        if(n==nminus1)
            result = 1/result;
        if(a==aminus1)
            result /= tauminusdelta;
        else if(a==aminus2)
            result /= pow_2(tauminusdelta);
        if(c==cminus1)
            result /= tau;
        if(!isFinite(result))
            result=0;  // ad hoc fix to avoid problems at the boundaries of integration interval
        return result;
    }

    /** limiting case of the integration interval collapsing to a single point tau,
        i.e. f(tau)~=0, f'(tau)~=0, and f''(tau)<0 (downward-curving parabola).
        In this case, if f(tau) is in the numerator, the integral is assumed to be zero,
        while if the integrand contains f(tau)^(-1/2), then the limiting value of the integral
        is computed from the second derivative of f(tau) at the (single) point.
    */
    double limitingIntegralValue(const double tau) const {
        if(n==nplus1)
            return 0;
        assert(tau>=0);
        const coord::ProlSph& CS = fnc.point.coordsys;
        const double tauminusdelta = tau - CS.Delta2;
        double fncder2;
        fnc.evalDeriv(tau, NULL, NULL, &fncder2);  // ignore f(tau) and f'(tau), only take f''(tau)
        double result = 2*M_PI * sqrt(-tau/fncder2) * fabs(tauminusdelta);
        if(a==aminus1)
            result /= tauminusdelta;
        else if(a==aminus2)
            result /= pow_2(tauminusdelta);
        if(c==cminus1)
            result /= tau;
        return result;
    }
};

/** A simple function that facilitates locating the root of auxiliary function 
    on a semi-infinite interval for lambda: instead of F(tau) we consider 
    F1(tau)=F(tau)/tau^2, which tends to a finite negative limit as tau tends to infinity. */
class AxisymScaledForRootfinder: public math::IFunction {
public:
    const AxisymFunctionBase& fnc;
    explicit AxisymScaledForRootfinder(const AxisymFunctionBase& d) : fnc(d) {};
    virtual unsigned int numDerivs() const { return fnc.numDerivs(); }
    virtual void evalDeriv(const double tau, 
        double* val=0, double* der=0, double* der2=0) const
    {
        assert(tau>=0);
        if(der2)
            *der2 = NAN;
        if(/*!isFinite(tau)*/tau>1e100) {
            if(val)
                *val = fnc.E; // the asymptotic value
            if(der)
                *der = NAN;    // we don't know it
            return;
        }
        double fval, fder;
        fnc.evalDeriv(tau, &fval, der? &fder : NULL);
        if(val)
            *val = fval / pow_2(tau);
        if(der)
            *der = (fder - 2*fval/tau) / pow_2(tau);
    }
};

/** Compute the intervals of tau for which p^2(tau)>=0, 
    where  0    = nu_min     <= tau <= nu_max     <= delta    is the interval for the "nu" branch,
    and  delta <= lambda_min <= tau <= lambda_max < infinity  is the interval for "lambda".
*/
AxisymIntLimits findIntegrationLimitsAxisym(const AxisymFunctionBase& fnc)
{
    AxisymIntLimits lim;
    const double delta=fnc.point.coordsys.Delta2;

    // figure out the value of function at and around some important points
    double f_zero = fnc(0);
    double f_lambda, df_lambda, d2f_lambda;
    fnc.evalDeriv(fnc.point.lambda, &f_lambda, &df_lambda, &d2f_lambda);
    lim.nu_min = 0;
    lim.nu_max = lim.lambda_min = lim.lambda_max = NAN;  // means not yet determined
    double nu_upper = delta;     // upper bound on the interval to locate the root for nu_max
    double lambda_lower = delta; // lower bound on the interval for lambda_min

    if(fnc.Lz==0) {
        // special case: f(delta) = -0.5 Lz^2 = 0, may have either tube or box orbit in the meridional plane
        double deltaminus = delta*(1-1e-15), deltaplus = delta*(1+1e-15);
        if(fnc(deltaminus)<0) {  // box orbit: f<0 at some interval left of delta
            if(f_zero>0)         // there must be a range of nu where the function is positive
                nu_upper = deltaminus;
            else 
                lim.nu_max = lim.nu_min;
        } else
            lim.nu_max = delta;
        if(fnc(deltaplus)<0)     // tube orbit: f must be negative on some interval right of delta
            lambda_lower = deltaplus;
        else
            lim.lambda_min = delta;
    }

    if(!isFinite(lim.nu_max)) 
    {   // find range for J_nu (i.e. J_z) if it has not been determined at the previous stage
        if(f_zero>0)
            lim.nu_max = math::findRoot(fnc, fabs(fnc.point.nu), nu_upper, ACCURACY_RANGE);
        if(!isFinite(lim.nu_max))
            // means that the value f(nu) was just very slightly negative, or that f(0)<=0
            // i.e. this is a clear upper boundary of the range of allowed nu
            lim.nu_max = fabs(fnc.point.nu);
    }

    // find the range for J_lambda (i.e. J_r).
    // We assume that the point lambda is inside or at the edge of the interval where f(lambda)>=0,
    // so that we will search for roots on the intervals (delta, lambda) and (lambda, infinity).
    // However, due to roundoff errors, it may actually happen that f(lambda) is negative,
    // or even positive but very small, or simply zero. In this case at least one or both intervals
    // must be modified so as to robustly bracket the point where f passes through zero.

    // this will be the point that is guaranteed to lie "well inside" the interval of positive f,
    // or, in other words, that the intervals [delta, lambda_pos] and [lambda_pos, infinity)
    // both firmly bracket the roots (respectively, lambda_min and lambda_max).
    double lambda_pos = fnc.point.lambda;
    // linear extrapolation to estimate the location of the nearest root for lambda
    double dxToRoot   = -f_lambda / df_lambda;

    if(f_lambda<=0 || fabs(dxToRoot) < fnc.point.lambda * MINIMUM_RANGE)
    {   // we are at the endpoint of the interval where f is positive,
        // so at least one of the endpoints may be assigned immediately
        if(df_lambda>=0) {
            lim.lambda_min = fnc.point.lambda;
        } 
        if(df_lambda<=0) {
            lim.lambda_max = fnc.point.lambda;
            if(fnc.point.lambda == delta)  // can't be lower than that! means that the range is zero
                lim.lambda_min = delta;
        }

        // now it may also happen that we are at or very near the shell orbit,
        // i.e. both lambda_min and lambda_max are very close (or equal) to lambda.
        // This happens when d^2 f / d lambda^2 < 0, i.e. the function is a downward parabola,
        // and the distance between its roots is very small, or even it does not cross zero at all
        // (of course, this could only happen due to roundoff errors).
        // the second derivative must be negative, and the determinant either small or negative.
        bool nearShell = d2f_lambda < 0 &&
            pow_2(df_lambda) - 2*f_lambda*d2f_lambda -
            pow_2(fnc.point.lambda * MINIMUM_RANGE * d2f_lambda) < 0;

        // However, the complication is that when lambda = delta, the second derivative is not defined.
        // Therefore, we first shift the point (lambda_pos) by a small amount in the direction of
        // increasing f, then recompute the second derivative, and then again test the condition.
        double safeOffset = fmin(0.5*fabs(df_lambda / d2f_lambda), fnc.point.lambda * MINIMUM_RANGE);
        lambda_pos += fmax(dxToRoot, safeOffset) * (df_lambda>=0?1:-1);
        double f_lampos, df_lampos, d2f_lampos;
        fnc.evalDeriv(lambda_pos, &f_lampos, &df_lampos, &d2f_lampos);
        nearShell |= d2f_lampos < 0 &&
            pow_2(df_lampos) - 2*f_lampos*d2f_lampos -
            pow_2(fnc.point.lambda * MINIMUM_RANGE * d2f_lampos) < 0;

        // unfortunately, f(lambda) is subject to such a severe cancellation error
        // that the above procedure does not always correctly identify a near-shell orbit.
        // therefore, we declare this to be the case even when the distance-between-roots condition
        // is not met, but the function is still negative, which would fail the root-finder anyway.
        if(nearShell || f_lampos < 0) {
            lim.lambda_min = lim.lambda_max = fnc.point.lambda;
        }
    }
    if(!isFinite(lim.lambda_min)) {  // not yet determined 
        lim.lambda_min = math::findRoot(fnc, lambda_lower, lambda_pos, ACCURACY_RANGE);
    }
    if(!isFinite(lim.lambda_max)) {
        lim.lambda_max = math::findRoot(AxisymScaledForRootfinder(fnc),
            math::ScalingSemiInf(lambda_pos) /* find root on [lambda_pos..+inf) */, ACCURACY_RANGE);
    }

    // sanity check
    if(utils::verbosityLevel >= utils::VL_WARNING &&
        (!isFinite(lim.lambda_min+lim.lambda_max+lim.nu_max+lim.nu_min)
        || fabs(fnc.point.nu) > lim.nu_max
        || fnc.point.lambda   < lim.lambda_min
        || fnc.point.lambda   > lim.lambda_max))
        utils::msg(utils::VL_WARNING, "findIntegrationLimitsAxisym", "failed at lambda="+
            utils::toString(fnc.point.lambda)+", nu="+utils::toString(fnc.point.nu)+", E="+
            utils::toString(fnc.E)+", Lz="+utils::toString(fnc.Lz)+", I3="+utils::toString(fnc.I3));

    // ignore extremely small intervals
    if(!(lim.nu_max >= delta * MINIMUM_RANGE))
        lim.nu_max = 0;
    if(!(lim.lambda_max-lim.lambda_min >= delta * MINIMUM_RANGE))
        lim.lambda_min = lim.lambda_max = fnc.point.lambda;

    // choose the order of Gauss-Legendre integration depending on the approximate eccentricity
    double RperiOverRapo = sqrt((lim.lambda_min-delta) / (lim.lambda_max-delta));
    lim.integrOrder = integrOrder(RperiOverRapo);
    return lim;
}

/** Compute the derivatives of actions (Jr, Jz, Jphi) over integrals of motion (E, Lz, I3),
    using the expressions A4-A9 in Sanders(2012).
    A possible improvement in efficiency may be obtained by saving the values of potential
    taken along the same integration path when computing the actions, and re-using them in this routine.
*/
AxisymActionDerivatives computeActionDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    AxisymActionDerivatives der;
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrand<math::ScalingCub>
        transf_l(math::ScalingCub(lim.lambda_min, lim.lambda_max), integrand),
        transf_n(math::ScalingCub(lim.nu_min,     lim.nu_max),     integrand);
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    der.dJrdE = (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, lim.integrOrder) ) / (4*M_PI);
    der.dJzdE = math::integrateGL(transf_n, 0, 1, lim.integrOrder) / (2*M_PI);
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    der.dJrdI3 = - (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, lim.integrOrder) ) / (4*M_PI);
    der.dJzdI3 = -math::integrateGL(transf_n, 0, 1, lim.integrOrder) / (2*M_PI);
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    der.dJrdLz = -fnc.Lz * (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, lim.integrOrder) ) / (4*M_PI);
    der.dJzdLz = -fnc.Lz * math::integrateGL(transf_n, 0, 1, lim.integrOrder) / (2*M_PI);
    return der;
}

/** Compute the derivatives of integrals of motion (E, Lz, I3) over actions (Jr, Jz, Jphi), inverting
    the matrix of action derivatives by integrals.  These quantities are independent of angles,
    and in particular, the derivatives of energy w.r.t. the three actions are the frequencies. */
AxisymIntDerivatives computeIntDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    AxisymActionDerivatives dJ = computeActionDerivatives(fnc, lim);
    AxisymIntDerivatives der;
    // invert the matrix of derivatives
    double det  = dJ.dJrdE * dJ.dJzdI3 - dJ.dJrdI3 * dJ.dJzdE;
    if(lim.nu_min==lim.nu_max || det==0) {
        // special case z==0: motion in z is irrelevant, but we could not compute dJzdI3 which is not zero
        der.Omegar   = 1 / dJ.dJrdE;
        der.Omegaphi =-dJ.dJrdLz / dJ.dJrdE;
        der.dI3dJr   = der.dI3dJz = der.dI3dJphi = der.Omegaz = 0;
    } else {  // everything as normal
        der.Omegar   = dJ.dJzdI3 / det;  // dE/dJr
        der.Omegaz   =-dJ.dJrdI3 / det;  // dE/dJz
        der.Omegaphi = (dJ.dJrdI3 * dJ.dJzdLz - dJ.dJrdLz * dJ.dJzdI3) / det;  // dE/dJphi
        der.dI3dJr   =-dJ.dJzdE / det;
        der.dI3dJz   = dJ.dJrdE / det;
        der.dI3dJphi =-(dJ.dJrdE * dJ.dJzdLz - dJ.dJrdLz * dJ.dJzdE) / det;
    }
    der.dLzdJr  = 0;
    der.dLzdJz  = 0;
    der.dLzdJphi= 1;
    return der;
}

/** Compute the derivatives of generating function S over integrals of motion (E, Lz, I3),
    using the expressions A10-A12 in Sanders(2012).  These quantities do depend on angles.
    A possible improvement is to compute all three integrals for each of two directions at once,
    saving on repetitive potential evaluations along the same paths.
*/
AxisymGenFuncDerivatives computeGenFuncDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    const double signldot = fnc.point.lambdadot >= 0 ? +1 : -1;
    const double signndot = fnc.point.nudot * fnc.point.nu >= 0 ? +1 : -1;
    AxisymGenFuncDerivatives der;
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrand<math::ScalingCub>
        transf_l(math::ScalingCub(lim.lambda_min, lim.lambda_max), integrand),
        transf_n(math::ScalingCub(lim.nu_min,     lim.nu_max),     integrand);
    const double yl = math::scale(transf_l.scaling, fnc.point.lambda);
    const double yn = lim.nu_min==lim.nu_max ? 0 : math::scale(transf_n.scaling, fabs(fnc.point.nu));
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    der.dSdE =
        signldot * math::integrateGL(transf_l, 0, yl, lim.integrOrder) / 4
      + signndot * math::integrateGL(transf_n, 0, yn, lim.integrOrder) / 4;
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    der.dSdI3 = 
        signldot * -math::integrateGL(transf_l, 0, yl, lim.integrOrder) / 4
      + signndot * -math::integrateGL(transf_n, 0, yn, lim.integrOrder) / 4;
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    der.dSdLz = fnc.Lz==0 ? 0 : fnc.point.phi +
        signldot * -fnc.Lz * math::integrateGL(transf_l, 0, yl, lim.integrOrder) / 4
      + signndot * -fnc.Lz * math::integrateGL(transf_n, 0, yn, lim.integrOrder) / 4;
    return der;
}

/** Compute actions by integrating the momentum over the range of tau on which it is positive,
    separately for the "nu" and "lambda" branches (equation A1 in Sanders 2012). */
Actions computeActions(const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    Actions acts;
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrand<math::ScalingCub>
        transf_l(math::ScalingCub(lim.lambda_min, lim.lambda_max), integrand),
        transf_n(math::ScalingCub(lim.nu_min,     lim.nu_max),     integrand);
    integrand.n = AxisymIntegrand::nplus1;  // momentum goes into the numerator
    integrand.a = AxisymIntegrand::azero;
    integrand.c = AxisymIntegrand::czero;
    acts.Jr = math::integrateGL(transf_l, 0, 1, lim.integrOrder) / M_PI;
    // factor of 2 in Jz because we only integrate over half of the orbit (z>=0)
    acts.Jz = math::integrateGL(transf_n, 0, 1, lim.integrOrder) / M_PI * 2;
    acts.Jphi = fnc.Lz;
    return acts;
}

/** Compute angles from the derivatives of integrals of motion and the generating function
    (equation A3 in Sanders 2012). */
Angles computeAngles(const AxisymIntDerivatives& derI, const AxisymGenFuncDerivatives& derS,
    bool addPiToThetaZ)
{
    Angles angs;
    angs.thetar   = derS.dSdE*derI.Omegar   + derS.dSdI3*derI.dI3dJr   + derS.dSdLz*derI.dLzdJr;
    angs.thetaz   = derS.dSdE*derI.Omegaz   + derS.dSdI3*derI.dI3dJz   + derS.dSdLz*derI.dLzdJz;
    angs.thetaphi = derS.dSdE*derI.Omegaphi + derS.dSdI3*derI.dI3dJphi + derS.dSdLz*derI.dLzdJphi;
    angs.thetar   = math::wrapAngle(angs.thetar);
    angs.thetaz   = math::wrapAngle(angs.thetaz + M_PI*addPiToThetaZ);
    angs.thetaphi = math::wrapAngle(angs.thetaphi);
    return angs;
}

/** The sequence of operations needed to compute both actions and angles.
    Note that for a given orbit, only the derivatives of the generating function depend
    on the angles (assuming that the actions are constant); in principle, this may be used
    to skip the computation of the matrix of integral derivatives (not presently implemented).
    \param[in]  fnc  is the instance of `AxisymFunctionStaeckel` or `AxisymFunctionFudge`;
    \param[in]  lim  are the limits of motion in auxiliary coordinate system;
    \param[out] freq if not NULL, store frequencies of motion in this variable
*/
ActionAngles computeActionAngles(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim, Frequencies* freq)
{
    Actions acts = computeActions(fnc, lim);
    AxisymIntDerivatives derI = computeIntDerivatives(fnc, lim);
    AxisymGenFuncDerivatives derS = computeGenFuncDerivatives(fnc, lim);
    bool addPiToThetaZ = fnc.point.nudot<0 && acts.Jz!=0;
    Angles angs = computeAngles(derI, derS, addPiToThetaZ);
    if(freq!=NULL)
        *freq = derI;  // store frequencies which are the first row of the derivatives matrix
    return ActionAngles(acts, angs);
}

}  // internal namespace

// -------- THE DRIVER ROUTINES --------

Actions actionsAxisymStaeckel(const potential::OblatePerfectEllipsoid& potential,
    const coord::PosVelCyl& point)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    if(!isFinite(fnc.E+fnc.I3+fnc.Lz) || fnc.E>=0)
        return Actions(NAN, NAN, fnc.Lz);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles actionAnglesAxisymStaeckel(const potential::OblatePerfectEllipsoid& potential,
    const coord::PosVelCyl& point, Frequencies* freq)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    if(!isFinite(fnc.E+fnc.I3+fnc.Lz) || fnc.E>=0)
        return ActionAngles(Actions(NAN, NAN, fnc.Lz), Angles(NAN, NAN, NAN));
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim, freq);
}

Actions actionsAxisymFudge(const potential::BasePotential& potential,
    const coord::PosVelCyl& point, double focalDistance)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(focalDistance<=0)
        focalDistance = fmax(point.R, 1.) * 1e-4;   // this is a temporary workaround!
    const coord::ProlSph coordsys(focalDistance);
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    if(!isFinite(fnc.E+fnc.I3+fnc.Lz) || fnc.E>=0)
        return Actions(NAN, NAN, fnc.Lz);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles actionAnglesAxisymFudge(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double focalDistance, Frequencies* freq)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(focalDistance<=0)
        focalDistance = fmax(point.R, 1.) * 1e-4;   // this is a temporary workaround!
    const coord::ProlSph coordsys(focalDistance);
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    if(!isFinite(fnc.E+fnc.I3+fnc.Lz) || fnc.E>=0) {
        if(freq) freq->Omegar = freq->Omegaz = freq->Omegaphi = NAN;
        return ActionAngles(Actions(NAN, NAN, fnc.Lz), Angles(NAN, NAN, NAN));
    }
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim, freq);
}


// ----------- INTERPOLATOR ----------- //
namespace {

/** compute the best-suitable focal distance at a 2d grid in E, L/Lcirc(E) */
void createGridFocalDistance(
    const potential::BasePotential& pot,
    const std::vector<double>& gridE, const std::vector<double>& gridL,
    /*output: focal distance*/ math::Matrix<double>& grid2dD,
    /*output: radius of shell orbit normalized to R_circ(E) */ math::Matrix<double>& grid2dR)
{
    int sizeE = gridE.size(), sizeL = gridL.size(), sizeEL = (sizeE-1) * (sizeL-1);
    grid2dD = math::Matrix<double>(sizeE, sizeL);
    grid2dR = math::Matrix<double>(sizeE, sizeL);
    std::string errorMessage;  // store the error text in case of an exception in the openmp block
    // loop over the grid in E and L (combined index for better load balancing)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int iEL = 0; iEL < sizeEL; iEL++) {
        try{
            int iE    = iEL / (sizeL-1);
            int iL    = iEL % (sizeL-1);
            double E  = gridE[iE];
            double Rc = R_circ(pot, E);
            double vc = v_circ(pot, Rc);
            double Lc = Rc * vc;
            double Lz = Lc * gridL[iL];
            double Rsh, FD  = estimateFocalDistanceShellOrbit(pot, E, Lz, &Rsh);
            grid2dD(iE, iL) = FD;
            grid2dR(iE, iL) = Rsh / Rc;
        }
        catch(std::exception& ex) {
            errorMessage = ex.what();
        }
    }
    // limiting cases of Lz=0 or Lz=Lcirc - copy from the adjacent column of the table
    for(int iE=0; iE<sizeE-1; iE++) {
        grid2dD(iE, 0)       = grid2dD(iE, 1);  // for Lz=0 don't trust the focal distance from shell orbit
        grid2dD(iE, sizeL-1) = grid2dD(iE, sizeL-2);
        grid2dR(iE, sizeL-1) = 1.;  // Rshell = Rcirc for the planar circular orbit
    }
    // limiting case of E=0 - assume a Keplerian potential at large radii
    for(int iL=0; iL<sizeL; iL++) {
        grid2dD(sizeE-1, iL) = 0.;
        grid2dR(sizeE-1, iL) = 1.;  // Rshell = Rcirc
    }
    if(!errorMessage.empty())
        throw std::runtime_error(errorMessage);
}

/// return scaledE as a function of E and invPhi0 = 1/Phi(0)
inline double scaleE(const double E, const double invPhi0) { return log(invPhi0 - 1/E); }

}  // internal namespace

ActionFinderAxisymFudge::ActionFinderAxisymFudge(
    const potential::PtrPotential& _pot, const bool interpolate) :
    invPhi0(1./_pot->value(coord::PosCyl(0,0,0))), pot(_pot), interp(*pot)
{
    // construct a grid in radius with unequal spacing depending on the variation of the potential
    std::vector<double> gridR = potential::createInterpolationGrid(*pot, ACCURACY_INTERP2);

    const int sizeE = gridR.size();
    const int sizeL = 25;
    const int sizeI = 25;

    // convert the grid in radius into the grid in energy and xi=scaledE
    std::vector<double> gridE(sizeE), gridEscaled(sizeE);
    for(int i=0; i<sizeE; i++) {
        gridE[i] = interp.value(gridR[i]);
        gridEscaled[i] = scaleE(gridE[i], invPhi0);
    }

    // transformation s <-> u of the interval 0<=s<=1 onto 0<=u<=1, which stretches the regions
    // near boundaries: a cubic function u(s) with zero derivatives at s=0 and s=1
    math::ScalingCub scaling(0, 1);
    std::vector<double> gridL(sizeL);  // grid in Lzrel = Lz/Lcirc(E)    = u(chi)
    std::vector<double> gridI(sizeI);  // grid in I3rel = I3/I3max(E,Lz) = u(psi)
    // for the 2d/3d interpolators we use non-uniformly spaced grids in scaled variables,
    // where the grid spacing is also denser towards the endpoints and is determined by
    // applying the same scaling transformation to a uniform grid.
    // consequently, the un-scaled vars (Lzrel and I3rel) are doubly stretched (clustered near endpoints)
    std::vector<double> gridLscaled(sizeL);   // chi = s(Lzrel)
    std::vector<double> gridIscaled(sizeI);   // psi = s(I3rel)
    for(int i=0; i<sizeL; i++) {
        gridLscaled[i] = math::unscale(scaling, i/(sizeL-1.));
        gridL[i] = math::unscale(scaling, gridLscaled[i]);  // Lzrel = u(chi)
    }
    for(int i=0; i<sizeI; i++) {
        gridIscaled[i] = math::unscale(scaling, i/(sizeI-1.));
        gridI[i] = math::unscale(scaling, gridIscaled[i]);  // and I3rel = u(psi)
    }

    // initialize the interpolator for the focal distance as a function of E and Lzrel
    math::Matrix<double> grid2dD;  // focal distance
    math::Matrix<double> grid2dR;  // Rshell / Rcirc(E)
    createGridFocalDistance(*pot, gridE, gridL, /*output*/ grid2dD, grid2dR);
    interpD = math::LinearInterpolator2d(gridEscaled, gridLscaled, grid2dD); //, /*regularize*/true);

    if(!interpolate) {
        // nothing more to do, except perhaps writing the debug information
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            std::ofstream strm("ActionFinderAxisymFudge.log");
            strm << "#xi_E    chi_Lz unused\tEnergy   Lzrel  unused\tFocalD\tRsh/Rc\n";
            for(int iE=0; iE<sizeE; iE++) {
                for(int iL=0; iL<sizeL; iL++) {
                    strm <<
                    utils::pp(gridEscaled[iE], 8) +' ' +
                    utils::pp(gridLscaled[iL], 6) +' ' +
                    "0     \t"+
                    utils::pp(gridE[iE],       8) +' ' +
                    utils::pp(gridL[iL],       6) +' ' +
                    "0     \t"+
                    utils::pp(grid2dD(iE, iL), 7) +'\t'+
                    utils::pp(grid2dR(iE, iL), 7) +'\n';
                }
                strm << '\n';
            }
        }
        return;
    }

    // we're constructing an interpolation grid for Jr and Jz in (E,Lz,I3), suitably scaled
    std::vector<double> grid3dJr(sizeE * sizeL * sizeI);  // Jr(E, Lz, I3)
    std::vector<double> grid3dJz(sizeE * sizeL * sizeI);  // same for Jz

    int sizeEL = (sizeE-1) * (sizeL-1);
    std::string errorMessage;  // store the error text in case of an exception in the openmp block
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int iEL=0; iEL<sizeEL; iEL++) {
        try{
            int iE      = iEL / (sizeL-1);
            int iL      = iEL % (sizeL-1);
            double E    = gridE[iE];
            double Rc   = R_circ(*pot, E);
            double vc   = v_circ(*pot, Rc);
            double Lc   = Rc * vc;
            double Lz   = Lc * gridL[iL];
            double Rsh  = grid2dR(iE, iL) * Rc;
            double Phi0 = pot->value(coord::PosCyl(Rsh,0,0));
            double vphi = Lz>0 ? Lz / Rsh : 0;
            double vmer = sqrt(fmax( 2 * (E - Phi0) - pow_2(vphi), 0));

            // focal distance: presently can't work when fd=0 exactly
            double fd   = fmax(grid2dD(iE, iL), Rc * 1e-4);
            double lambda  = pow_2(Rsh) + pow_2(fd);
            double flambda = -lambda * Phi0;
            double I3max   = 0.5 * lambda * pow_2(vmer);
            const coord::ProlSph coordsys(fd);

            // explore the range of I3
            for(int iI=0; iI<sizeI; iI++) {
                int index = (iE * sizeL + iL) * sizeI + iI;
                const double I3 = I3max * gridI[iI];

                const coord::PosProlSph pprol(lambda, 0, 0, coordsys);
                const AxisymFunctionFudge fnc(coord::PosVelProlSph(pprol, 0, 0, 0),
                    E, Lz, I3, flambda, 0, *pot);
                AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
                if(iI==0)        // no vertical oscillation for a planar orbit
                    lim.nu_max = lim.nu_min = 0;
                if(iI==sizeI-1)  // no radial oscillation for a shell orbit
                    lim.lambda_min = lim.lambda_max = fnc.point.lambda;
                Actions acts = computeActions(fnc, lim);

                // sanity check
                if(!isFinite(acts.Jr+acts.Jz))
                    throw std::runtime_error("cannot compute actions for"
                        " R="+utils::toString(Rsh)+", z=0"
                        ", vR="+utils::toString(vmer * sqrt(1-gridI[iI]))+
                        ", vz="+utils::toString(vmer * sqrt(  gridI[iI]))+
                        ", vphi="+utils::toString(vphi));

                // scaled values passed to the interpolator
                grid3dJr[index] = acts.Jr / (Lc-Lz);
                grid3dJz[index] = acts.Jz / (Lc-Lz);
            }
        }
        catch(std::exception& ex) {
            errorMessage = ex.what();
        }
    }
    if(!errorMessage.empty())
        throw std::runtime_error("ActionFinderAxisymFudge: " + errorMessage);

    // limiting cases are considered separately
    // 1. limiting case of a circular orbit
    for(int iE=0; iE<sizeE-1; iE++) {
        int iL = sizeL-1;
        grid2dR(iE, iL) = 1.;  // shell orbit coincides with the circular orbit in the equatorial plane
        double kappa, nu, Omega, Rc = R_circ(*pot, gridE[iE]);
        epicycleFreqs(*pot, Rc, kappa, nu, Omega);
        if(kappa>0 && nu>0 && Omega>0) {
            for(int iI=0; iI<sizeI; iI++) {
                int index = (iE * sizeL + iL) * sizeI + iI;
                grid3dJr[index] = Omega / kappa * (1-gridI[iI]);
                grid3dJz[index] = Omega / nu    *    gridI[iI];
            }
        } else {
            utils::msg(utils::VL_WARNING, "ActionFinderAxisymFudge",
                "cannot compute epicyclic frequencies at R="+utils::toString(Rc));
            // simply repeat the values from the previous row
            for(int iI=0; iI<sizeI; iI++) {
                int index = (iE * sizeL + iL) * sizeI + iI;
                grid3dJr[index] = grid3dJr[index - sizeI];
                grid3dJz[index] = grid3dJz[index - sizeI];
            }
        }
    }
    // 2. limiting case of E --> 0, assuming a Keplerian potential at large radii
    for(int iL=0; iL<sizeL; iL++) {
        int iE = sizeE-1;
        grid2dR(iE, iL) = 1.;  // shell orbit coincides with the circular orbit
        for(int iI=0; iI<sizeI; iI++) {
            int index = (iE * sizeL + iL) * sizeI + iI;
            // in the Keplerian regime, Lcirc = Jr + L = Jr + Jz + Jphi,
            // and L = sqrt( Lz^2 + (I3/I3max) * (Lcirc^2-Lz^2) ).
            // in our scaled units gridL[iL] = Lz/Lcirc, and gridI[iI] = I3/I3max,
            // thus Jr,rel = (Lcirc - L) / (Lcirc - Lz)  and  Jz,rel = (L - Lz) / (Lcirc - Lz).
            double Lzrel = gridL[iL];
            double I3rel = gridI[iI];
            double Lrel  = sqrt( pow_2(Lzrel) * (1 - I3rel) + I3rel);  // L/Lcirc
            grid3dJr[index] = (1 + Lzrel) * (1 - I3rel) / (1 + Lrel);
            grid3dJz[index] = iI+iL>0 ? (1 + Lzrel) * I3rel / (Lzrel + Lrel) : 0;
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("ActionFinderAxisymFudge.log");
        strm << "#xi_E    chi_Lz psi_I3\tEnergy   Lzrel  I3rel\tFocalD\tRsh/Rc\tJrrel\tJzrel\n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                for(int iI=0; iI<sizeI; iI++) {
                    strm <<
                    utils::pp(gridEscaled[iE], 8) +' ' +
                    utils::pp(gridLscaled[iL], 6) +' ' +
                    utils::pp(gridIscaled[iI], 6) +'\t'+
                    utils::pp(gridE[iE],       8) +' ' +
                    utils::pp(gridL[iL],       6) +' ' +
                    utils::pp(gridI[iI],       6) +'\t'+
                    utils::pp(grid2dD(iE, iL), 7) +'\t'+
                    utils::pp(grid2dR(iE, iL), 7) +'\t'+
                    utils::pp(grid3dJr[ (iE * sizeL + iL) * sizeI + iI ], 7) +'\t'+
                    utils::pp(grid3dJz[ (iE * sizeL + iL) * sizeI + iI ], 7) +'\n';
                }
            }
            strm << '\n';
        }
    }

    interpR = math::CubicSpline2d(gridEscaled, gridLscaled, grid2dR, /*regularize*/true);
    intJr   = math::CubicSpline3d(gridEscaled, gridLscaled, gridIscaled, grid3dJr, true);
    intJz   = math::CubicSpline3d(gridEscaled, gridLscaled, gridIscaled, grid3dJz, true);
}

Actions ActionFinderAxisymFudge::actions(const coord::PosVelCyl& point) const
{
    // step 0. find the two classical integrals of motion
    double Phi   = pot->value(point);
    double E     = Phi + 0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    double Lz    = coord::Lz(point);
    if(E>=0)
        return Actions(NAN, NAN, Lz);

    // step 1. find the focal distance d from the interpolator
    double Lcirc = interp.L_circ(E);
    // interpolator works in scaled variables:
    // xi = scaledE (restricted to a suitable range) and chi = s( Lz/Lcirc(E) ),
    // where s is the cubic scaling transformation
    math::ScalingCub scaling(0, 1);
    double xi    = math::clip(scaleE(E, invPhi0), interpD.xmin(), interpD.xmax());
    double Lzrel = math::clip(fabs(Lz) / Lcirc, 0., 1.);
    double chi   = math::scale(scaling, Lzrel);
    double fd    = fmax(0, interpD.value(xi, chi));   // focal distance

    // if we are not using the 3d interpolation, then compute the actions by the direct method
    if(intJr.empty())
        return actionsAxisymFudge(*pot, point, fd);

    // step 2. find the third (approximate) integral of motion
    double Rcirc = interp.R_from_Lz(Lcirc);   // radius of a circular orbit with the given E
    if(Rcirc == 0)  // degenerate case
        return Actions(0, 0, 0);
    if(fd==0) fd = Rcirc*1e-4;
    coord::ProlSph coordsys(fd);
    const coord::PosProlSph pprol = coord::toPos<coord::Cyl, coord::ProlSph>(point, coordsys);

    // the third coordinate in the 3d interpolation grid is I3/I3max,
    // where I3max(E, Lz) is the maximum possible value of I3, computed from the radius of a shell orbit
    double Rshell= fmax(0, interpR.value(xi, chi)) * Rcirc;
    double PhiS  = interp.value(Rshell);
    double lamS  = pow_2(Rshell) + fd*fd;  // lambda(Rshell,z=0)
    double I3max = fmax(0, E - PhiS - (Rshell>0 ? 0.5 * pow_2(Lz/Rshell) : 0) ) * lamS;
#if 0   // method L: take the potential at point (lambda,0)
    double PhiL  = interp.value(sqrt(pprol.lambda - coordsys.Delta2));
    double add   = pprol.lambda * (Phi - PhiL);
#else   // method N: take the potential at point (lambda_shell,nu) - generally more accurate
    double PhiN  = pot->value(coord::toPosCyl(coord::PosProlSph(lamS, pprol.nu, 0, coordsys)));
    double add   = lamS * (PhiN - PhiS) + fabs(pprol.nu) * (Phi - PhiN);
#endif
    // Y is (L^2 - L_z^2 + Delta^2 v_z^2)
    double Y  = pow_2(point.z*point.vphi) + pow_2(point.R*point.vz-point.z*point.vR) + pow_2(fd*point.vz);
    double I3 = 0.5*Y + add;

    // step 3. obtain the interpolated values of (suitably scaled) Jr and Jz
    // as functions of three scaled variables:  E, chi, psi = s( I3/I3max )
    double psi   = math::scale(scaling, math::clip(I3 / I3max, 0., 1.));
    double Jrrel = fmax(0, intJr.value(xi, chi, psi));
    double Jzrel = fmax(0, intJz.value(xi, chi, psi));
    return Actions(Lcirc * (1-Lzrel) * Jrrel, Lcirc * (1-Lzrel) * Jzrel, Lz);
}

double ActionFinderAxisymFudge::focalDistance(const coord::PosVelCyl& point) const
{
    double E     = totalEnergy(*pot, point);
    double Lz    = coord::Lz(point);
    double Lc    = interp.L_circ(E);
    double Lzrel = math::clip(fabs(Lz) / Lc, 0., 1.);
    double xi    = math::clip(scaleE(E, invPhi0), interpD.xmin(), interpD.xmax());
    double chi   = math::scale(math::ScalingCub(0, 1), Lzrel);
    return fmax(0, interpD.value(xi, chi));
}

}  // namespace actions
