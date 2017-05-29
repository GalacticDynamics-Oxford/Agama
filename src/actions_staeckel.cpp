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
    is determined by the number of points in fixed-order Gauss-Legendre scheme */
static const unsigned int INTEGR_ORDER = 10;  // good enough

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over */
static const double ACCURACY_RANGE = 1e-6;

/** minimum range of variation of nu, lambda that is considered to be non-zero */
static const double MINIMUM_RANGE = 1e-10;

// ------ Data structures for both Axisymmetric Staeckel and Fudge action-angle finders ------

/** integration intervals for actions and angles
    (shared between Staeckel and Fudge action finders). */
struct AxisymIntLimits {
    double lambda_min, lambda_max, nu_min, nu_max;
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
        *val = ( E * tauminusdelta - pow_2(Lz)/2 ) * tau
             - (I + Phi * mult) * tauminusdelta;
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
        if(!isFinite(tau)) {
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
        lambda_pos += fmax(dxToRoot, safeOffset) * math::sign(df_lambda);
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
            lambda_pos, INFINITY, ACCURACY_RANGE);
    }

    // sanity check
    if(!isFinite(lim.lambda_min+lim.lambda_max+lim.nu_max+lim.nu_min)
        || fabs(fnc.point.nu) > lim.nu_max
        || fnc.point.lambda   < lim.lambda_min
        || fnc.point.lambda   > lim.lambda_max)
        utils::msg(utils::VL_WARNING, "findIntegrationLimitsAxisym", "failed");

    // ignore extremely small intervals
    if(!(lim.nu_max >= delta * MINIMUM_RANGE))
        lim.nu_max = 0;
    if(!(lim.lambda_max-lim.lambda_min >= delta * MINIMUM_RANGE))
        lim.lambda_min = lim.lambda_max = fnc.point.lambda;

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
    math::ScaledIntegrandEndpointSing transf_l(integrand, lim.lambda_min, lim.lambda_max);
    math::ScaledIntegrandEndpointSing transf_n(integrand, lim.nu_min, lim.nu_max);
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    der.dJrdE = (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) ) / (4*M_PI);
    der.dJzdE = math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    der.dJrdI3 = - (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) ) / (4*M_PI);
    der.dJzdI3 = -math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    der.dJrdLz = -fnc.Lz * (lim.lambda_min==lim.lambda_max ? 
        integrand.limitingIntegralValue(lim.lambda_min) :
        math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) ) / (4*M_PI);
    der.dJzdLz = -fnc.Lz * math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);
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
    math::ScaledIntegrandEndpointSing transf_l(integrand, lim.lambda_min, lim.lambda_max);
    math::ScaledIntegrandEndpointSing transf_n(integrand, lim.nu_min, lim.nu_max);
    const double yl = transf_l.y_from_x(fnc.point.lambda);
    const double yn = lim.nu_min==lim.nu_max ? 0 : transf_n.y_from_x(fabs(fnc.point.nu));
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    der.dSdE =
        signldot * math::integrateGL(transf_l, 0, yl, INTEGR_ORDER) / 4
      + signndot * math::integrateGL(transf_n, 0, yn, INTEGR_ORDER) / 4;
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    der.dSdI3 = 
        signldot * -math::integrateGL(transf_l, 0, yl, INTEGR_ORDER) / 4
      + signndot * -math::integrateGL(transf_n, 0, yn, INTEGR_ORDER) / 4;
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    der.dSdLz = fnc.Lz==0 ? 0 : fnc.point.phi +
        signldot * -fnc.Lz * math::integrateGL(transf_l, 0, yl, INTEGR_ORDER) / 4
      + signndot * -fnc.Lz * math::integrateGL(transf_n, 0, yn, INTEGR_ORDER) / 4;
    return der;
}

/** Compute actions by integrating the momentum over the range of tau on which it is positive,
    separately for the "nu" and "lambda" branches (equation A1 in Sanders 2012). */
Actions computeActions(const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    Actions acts;
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrandEndpointSing transf_l(integrand, lim.lambda_min, lim.lambda_max);
    math::ScaledIntegrandEndpointSing transf_n(integrand, lim.nu_min, lim.nu_max);
    integrand.n = AxisymIntegrand::nplus1;  // momentum goes into the numerator
    integrand.a = AxisymIntegrand::azero;
    integrand.c = AxisymIntegrand::czero;
    acts.Jr = math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) / M_PI;
    // factor of 2 in Jz because we only integrate over half of the orbit (z>=0)
    acts.Jz = math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / M_PI * 2;
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

/** create a hand-crafted non-uniform grid on the interval [0,1] that is denser towards the endpoints */
inline std::vector<double> createGrid(int size)
{
    assert(size>10);
    std::vector<double> grid(size);
    for(int i=0; i<size-4; i++) {
        double x = i/(size-5.);
        // transformation of interval [0:1] onto itself that places more grid points near the edges:
        // a function with zero 1st and 2nd derivs at x=0 and x=1
        grid[i+2] = pow_3(x) * (10+x*(-15+x*6));
    }
    // manual tuning for the first/last few nodes
    grid[4] = grid[5]/2.5;
    grid[3] = grid[4]/3.0;
    grid[2] = grid[3]/3.0;
    grid[1] = grid[2]/3.0;
    grid[0] = 0;
    grid[size-1] = 1;
    grid[size-2] = 1-grid[1];
    grid[size-3] = 1-grid[2];
    grid[size-4] = 1-grid[3];
    grid[size-5] = 1-grid[4];
    return grid;
}

/** compute the best-suitable focal distance at a 2d grid in E, L/Lcirc(E) */
math::Matrix<double> createGridFocalDistance(
    const potential::BasePotential& pot,
    const std::vector<double>& gridE, const std::vector<double>& gridL)
{
    int sizeE = gridE.size(), sizeL = gridL.size(), sizeEL = (sizeE-1) * (sizeL-2);
    math::Matrix<double> grid2dD(sizeE, sizeL);
    std::string errorMessage;  // store the error text in case of an exception in the openmp block
    // loop over the grid in E and L (combined index for better load balancing)
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int iEL = 0; iEL < sizeEL; iEL++) {
        try{
        int iE    = iEL / (sizeL-2);
        int iL    = iEL % (sizeL-2)+1;
        double E  = gridE[iE];
        double Lc = L_circ(pot, E);
        double Lz = gridL[iL] * Lc;
#if 1
        // estimate the focal distance from fitting a line lambda=const to a shell orbit (with Jr=0)
        double fd = estimateFocalDistanceShellOrbit(pot, E, Lz);
        //if(fd > 0 && fd < Rthin * 1e-2)
        //    fd = 0;   // probably dominated by inaccuracies
#else
        // estimate the focal distance from the mixed derivatives of potential
        // averaged over the region accessible to the ensemble of orbits with the given E and Lz
        double R1, R2;
        potential::findPlanarOrbitExtent(pot, E, Lz, R1, R2);
        const int NP=32;
        std::vector<double> x, y, P;
        for(int iR=0; iR<NP; iR++) {
            double R = (iR + 0.5) / NP * (R2-R1) + R1;
            for(int iz=0; iz<NP; iz++) {
                double z = pow_2((iz + 0.5) / NP) * R2;
                double Phi;
                coord::GradCyl grad;
                coord::HessCyl hess;
                pot.eval(coord::PosCyl(R, z, 0), &Phi, &grad, &hess);
                if(Phi + 0.5*pow_2(Lz/R) < E) {
                    P.push_back(Phi);
                    x.push_back(hess.dRdz);
                    y.push_back(3*z * grad.dR - 3*R * grad.dz +
                        R*z * (hess.dR2-hess.dz2) + (z*z - R*R) * hess.dRdz);
                } else
                    break;
            }
        }
        double fd = sqrt( fmax( math::linearFitZero(x, y, NULL), 0) );
#endif
        grid2dD(iE, iL) = fd;
        }
        catch(std::exception& ex) {
            errorMessage = ex.what();
        }
    }
    // limiting cases of Lz=0 and Lz=Lcirc - copy from the adjacent columns of the table
    for(int iE=0; iE<sizeE-1; iE++) {
        grid2dD(iE, 0)       = grid2dD(iE, 1);
        grid2dD(iE, sizeL-1) = grid2dD(iE, sizeL-2);
    }
    // limiting case of E=0 - copy from the penultimate row
    for(int iL=0; iL<sizeL; iL++)
        grid2dD(sizeE-1, iL) = grid2dD(sizeE-2, iL);
    if(!errorMessage.empty())
        throw std::runtime_error(errorMessage);
    return grid2dD;
}

// helper class to find the minimum of -I3
class MaxI3finder: public math::IFunctionNoDeriv {
    const potential::BasePotential& pot;
    double E, Lz2, d2;
public:
    MaxI3finder(const potential::BasePotential& _pot, double _E, double _Lz, double _ifd) :
        pot(_pot), E(_E), Lz2(_Lz*_Lz), d2(_ifd*_ifd) {}
    virtual double value(const double R) const
    {
        double Phi = pot.value(coord::PosCyl(R, 0, 0));
        return (R*R + d2) * (Phi - E + 0.5 * (R>0 ? Lz2 / pow_2(R) : 0));
    }
};

// find the radius of a shell orbit, i.e. the one that maximizes I3:
// I3 = (R^2 + Delta^2) (E - Phi(R) - Lz^2/(2R^2) ), where Delta is the focal distance (fd).
// we do not use the shell orbits obtained during the construction of an interpolator for Delta(E,Lz),
// because they are valid for the _real_ potential, but the _approximate_ radial action needs not
// be zero when evaluated for such an orbit. Instead we explicitly locate the point on the R axis
// that corresponds to the minimum of effective potential for the given value of Delta.
double findRshell(const potential::BasePotential& pot, double E, double Lz, double fd)
{
    MaxI3finder fnc(pot, E, Lz, fd);
    // unfortunately, the function to minimize may have more than one local minimum,
    // therefore we first do a brute-force grid search to locate the global minimum, and then polish it
    double R1, R2, Rshell=NAN;
    potential::findPlanarOrbitExtent(pot, E, Lz, R1, R2);
    double minval = INFINITY;
    for(int k=1; k<100; k++) {
        double R = pow_2(k/100.)*(3-2*k/100.) * (R2-R1) + R1;  // search between R1 and R2
        double I = fnc(R);
        if(I<minval) {
            Rshell = R;
            minval = I;
        }
    }
    return math::findMin(fnc, R1, R2, Rshell, ACCURACY_RANGE);
}

}  // internal namespace

ActionFinderAxisymFudge::ActionFinderAxisymFudge(
    const potential::PtrPotential& _pot, const bool interpolate) :
    pot(_pot), interp(*pot)
{
    double Phi0 = pot->value(coord::PosCyl(0,0,0));
    if(!isFinite(Phi0))
        throw std::runtime_error(
            "ActionFinderAxisymFudge: can only deal with potentials that are finite at r->0");
    const int sizeE = 50;
    const int sizeL = 25;
    const int sizeI = 25;

    std::vector<double> gridE = createGrid(sizeE+1);      // grid in energy
    for(int i=0; i<=sizeE; i++)
        gridE[i] = Phi0 * (1-gridE[i]);
    gridE.erase(gridE.begin());   // the very first node exactly at origin is not used
    std::vector<double> gridL = createGrid(sizeL);        // grid in L/Lcirc(E)
    std::vector<double> gridI = createGrid(sizeI);        // grid in I3/I3max(E,L)

    // initialize the interpolator for the focal distance as a function of E and Lz/Lcirc
    math::Matrix<double> grid2dD = createGridFocalDistance(*pot, gridE, gridL);
    interpD = math::LinearInterpolator2d(gridE, gridL, grid2dD);
    if(!interpolate)
        return;

    // we're constructing an interpolation grid for Jr and Jz in (E,L,I3)
    math::Matrix<double> grid2dR(sizeE, sizeL);           // Rshell(E, Lz/Lc)
    math::Matrix<double> grid2dI(sizeE, sizeL);           // I3max(E, Lz/Lc)
    std::vector<double> grid3dJr(sizeE * sizeL * sizeI);  // Jr(E, Lz/Lc, I3/I3max)
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
        double Lz   = gridL[iL] * Lc;
        // focal distance: presently can't work when fd=0 exactly
        double fd   = fmax(grid2dD(iE, iL), Rc * 1e-4);
        double Rsh  = findRshell(*pot, E, Lz, fd);  // radius of an orbit with Jr=0, i.e. I3=I3max
        if(!isFinite(Rsh))
            throw std::runtime_error("cannot find a shell orbit for "
                "E="+utils::toString(E)+", Lz="+utils::toString(Lz));
        if(Rsh == 0 && utils::verbosityLevel >= utils::VL_WARNING)  // this may be a legitimate situation?
            utils::msg(utils::VL_WARNING, "ActionFinderAxisymFudge", "Rthin=0 for E="+utils::toString(E));
        grid2dR(iE, iL) = Rsh / Rc;

        double Phi0 = pot->value(coord::PosCyl(Rsh,0,0));
        double vphi = Lz>0 ? Lz / Rsh : 0;
        double vmer = sqrt(fmax( 2 * (E - Phi0) - pow_2(vphi), 0));
        double lambda  = pow_2(Rsh) + pow_2(fd);
        double flambda = -lambda * Phi0;
        double I3max   = 0.5 * lambda * pow_2(vmer);
        double I3norm  = (0.5 * vc*vc * (1-pow_2(gridL[iL])) * (Rc*Rc + fd*fd));
        grid2dI(iE,iL) = I3max / I3norm;
        const coord::ProlSph coordsys(fd);

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
            actions::Actions acts = computeActions(fnc, lim);

            // sanity check
            if(!isFinite(acts.Jr+acts.Jz))
                throw std::runtime_error("cannot compute actions for "
                    "R="+utils::toString(Rsh)+", z=0, vR="+utils::toString(vmer * sqrt(1-gridI[iI]))+
                    ", vz="+utils::toString(vmer * sqrt(gridI[iI]))+", vphi="+utils::toString(vphi));

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
        grid2dR(iE, iL) = 1.;
        grid2dI(iE, iL) = 1.;
        double kappa, nu, Omega, Rc = R_circ(*pot, gridE[iE]);
        epicycleFreqs(*pot, Rc, kappa, nu, Omega);
        if(kappa>0 && nu>0 && Omega>0) {
            for(int iI=0; iI<sizeI; iI++) {
                int index = (iE * sizeL + iL) * sizeI + iI;
                grid3dJr[index] = Omega / kappa * (1-gridI[iI]);
                grid3dJz[index] = Omega / nu * gridI[iI];
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
        grid2dR(iE, iL) = 1.;
        grid2dI(iE, iL) = 1.;
        for(int iI=0; iI<sizeI; iI++) {
            int index = (iE * sizeL + iL) * sizeI + iI;
            // in the Keplerian regime, Lcirc = Jr + L = Jr + Jz + Jphi,
            // and L = sqrt( Lz^2 + (I3/I3max) * (Lcirc^2-Lz^2) ).
            // in our scaled units gridL[iL] = Lz/Lcirc, and gridI[iI] = I3/I3max.
            // thus Jr,rel = (Lcirc - L) / (Lcirc - Lz)  and  Jz,rel = (L - Lz) / (Lcirc - Lz)
            double L = sqrt( pow_2(gridL[iL]) * (1 - gridI[iI]) + gridI[iI]);  // normalized to Lcirc
            grid3dJr[index] = (1 + gridL[iL]) * (1 - gridI[iI]) / (1 + L);
            grid3dJz[index] = iI+iL>0 ? (1 + gridL[iL]) * gridI[iI] / (gridL[iL] + L) : 0;
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("ActionFinderAxisymFudge.log");
        strm << "#E L/Lcirc I3rel\tIFD\tRthin/Rcirc\tJrrel\tJzrel\n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                for(int iI=0; iI<sizeI; iI++) {
                    strm <<
                    utils::pp(gridE[iE], 8) +' '+
                    utils::pp(gridL[iL], 6) +' '+
                    utils::pp(gridI[iI], 6) +'\t'+
                    utils::pp(grid2dD(iE, iL), 7) +'\t'+
                    utils::pp(grid2dR(iE, iL), 7) +'\t'+
                    utils::pp(grid2dI(iE, iL), 7) +'\t'+
                    utils::pp(grid3dJr[ (iE * sizeL + iL) * sizeI + iI ], 7) +'\t'+
                    utils::pp(grid3dJz[ (iE * sizeL + iL) * sizeI + iI ], 7) +'\n';
                }
            }
            strm << '\n';
        }
    }

    interpI = math::CubicSpline2d(gridE, gridL, grid2dI);
    intJr   = math::CubicSpline3d(gridE, gridL, gridI, grid3dJr);
    intJz   = math::CubicSpline3d(gridE, gridL, gridI, grid3dJz);
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
    // interpolator works in scaled variables: E (restricted to a suitable range) and Lz/Lcirc(E)
    double Lzrel = fmin(fmax(fabs(Lz) / Lcirc, 0), 1);
    double Eint  = fmin(fmax(E, interpD.xmin()), interpD.xmax());
    double fd    = fmax(0, interpD.value(Eint, Lzrel));   // focal distance

    // if we are not using the 3d interpolation, then compute the actions by the direct method
    if(intJr.empty())
        return actionsAxisymFudge(*pot, point, fd);

    // step 2. find the third (approximate) integral of motion
    double Rcirc = interp.R_from_Lz(Lcirc);   // radius of a circular orbit with the given E
    if(Rcirc == 0)  // degenerate case
        return Actions(0, 0, 0);
    if(fd==0) fd = Rcirc*1e-4;
    coord::ProlSph coordsys(fd);
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double lmd   = pprol.lambda - coordsys.Delta2;
    double Phi0  = interp.value(sqrt(lmd));   // potential at R=sqrt(lambda), z=0
    double I3    = point.z==0 && point.vz==0 ? 0 :
        pprol.lambda * (E - Phi0 - 0.5 / lmd *
        (pow_2(Lz) + pow_2(point.R * point.vR + point.z * point.vz * lmd / pprol.lambda)) );

    // the third coordinate in the 3d interpolation grid is I3/I3max,
    // where I3max(E, Lz) is the maximum possible value of I3,
    // and itself is found from a 2d interpolator and multiplied by a dimensional scaling factor
    // dimensional normalization factor for I3max
    double I3norm= 0.5 * (1 - pow_2(Lzrel)) * pow_2(Lcirc) * (1 + pow_2(fd/Rcirc));
    double I3max = fmax(0, interpI.value(Eint, Lzrel)) * I3norm;

    // step 3. obtain the interpolated values of (suitably scaled) Jr and Jz
    // as functions of three scaled variables:  E, Lz/Lcirc(E), I3/I3max(E,Lz)
    double I3rel = fmax(0, fmin(1, I3 / I3max));
    double Jrrel = fmax(0, intJr.value(Eint, Lzrel, I3rel));
    double Jzrel = fmax(0, intJz.value(Eint, Lzrel, I3rel));
    return Actions(Lcirc * (1-Lzrel) * Jrrel, Lcirc * (1-Lzrel) * Jzrel, Lz);
}

double ActionFinderAxisymFudge::focalDistance(const coord::PosVelCyl& point) const
{
    double E    = totalEnergy(*pot, point);
    double Lz   = coord::Lz(point);
    double Lc   = interp.L_circ(E);
    double Lzrel= fmin(fmax(fabs(Lz) / Lc, 0), 1);
    double Eint = fmin(fmax(E, interpD.xmin()), interpD.xmax());
    return fmax(0, interpD.value(Eint, Lzrel));
}

}  // namespace actions
