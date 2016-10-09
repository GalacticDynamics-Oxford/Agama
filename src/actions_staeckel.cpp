#include "actions_staeckel.h"
#include "math_core.h"
#include "math_fit.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

namespace {  // internal routines

/** Accuracy of integrals for computing actions and angles
    is determined by the number of points in fixed-order Gauss-Legendre scheme */
const unsigned int INTEGR_ORDER = 10;  // good enough

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over */
const double ACCURACY_RANGE = 1e-6;

/** minimum range of variation of nu, lambda that is considered to be non-zero */
const double MINIMUM_RANGE = 1e-12;

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
    AxisymFunctionBase(const coord::PosVelProlSph& _point, double _E, double _Lz) :
        point(_point), E(_E), Lz(_Lz) {};
};

// ------ SPECIALIZED functions for Staeckel action finder -------

/** parameters of potential, integrals of motion, and prolate spheroidal coordinates 
    SPECIALIZED for the Axisymmetric Staeckel action finder */
class AxisymFunctionStaeckel: public AxisymFunctionBase {
public:
    const double I3;              ///< third integral
    const math::IFunction& fncG;  ///< single-variable function of a Staeckel potential
    AxisymFunctionStaeckel(const coord::PosVelProlSph& _point, double _E, double _Lz,
        double _I3, const math::IFunction& _fncG) :
        AxisymFunctionBase(_point, _E, _Lz), I3(_I3), fncG(_fncG) {};

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
        I3 = 0.5 * pow_2(point.vz) * (pow_2(point.R) + coordsys.delta);
    else   // general case: eq.3 in Sanders(2012)
        I3 = fmax(0,
            pprol.lambda * (E - pow_2(Lz) / 2 / (pprol.lambda - coordsys.delta) + Glambda) -
            pow_2( pprol.lambdadot * (pprol.lambda - fabs(pprol.nu)) ) / 
            (8 * (pprol.lambda - coordsys.delta) * pprol.lambda) );
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
    const double tauminusdelta = tau - point.coordsys.delta;
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
    const double Ilambda, Inu;  ///< approximate integrals of motion for two quasi-separable directions
    const potential::BasePotential& poten;  ///< gravitational potential
    AxisymFunctionFudge(const coord::PosVelProlSph& _point, double _E, double _Lz,
        double _Ilambda, double _Inu, const potential::BasePotential& _poten) :
        AxisymFunctionBase(_point, _E, _Lz), Ilambda(_Ilambda), Inu(_Inu), poten(_poten) {};

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
    double Phi;
    potential.eval(point, &Phi);
    double E = Phi+(pow_2(point.vR)+pow_2(point.vz)+pow_2(point.vphi))/2;
    double Lz= coord::Lz(point);
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    const double absnu = fabs(pprol.nu);

    // check for various extreme cases, and provide asymptotically valid expressions if necessary
    // if R is nearly zero and z^2 < delta, then lambda-delta ~= 0
    const bool lambda_near_min = pprol.lambda - coordsys.delta <= MINIMUM_RANGE*coordsys.delta;
    // if R is nearly zero and z^2 > delta, then nu-delta ~= 0
    const bool nu_near_max = coordsys.delta-absnu <= MINIMUM_RANGE*coordsys.delta;
    // if z is nearly zero, then nu ~= 0
    const bool nu_near_min = absnu <= MINIMUM_RANGE*coordsys.delta;

    const double El = lambda_near_min ? 
        pow_2(point.vphi) * absnu / coordsys.delta / 2 :
        pow_2(Lz) / 2 / (pprol.lambda - coordsys.delta);
    const double addIlambda = lambda_near_min ?
        -pow_2(point.vR) * (pprol.lambda - absnu) / 2 :
        -pow_2( pprol.lambdadot * (pprol.lambda - absnu) ) /
            (8 * (pprol.lambda-coordsys.delta) * pprol.lambda);
    const double Ilambda  = pprol.lambda * (E - El) - (pprol.lambda - absnu) * Phi + addIlambda;

    const double En = nu_near_max ? 
        pow_2(point.vphi) * (1 - pprol.lambda / coordsys.delta) / 2 :
        pow_2(Lz) / (absnu - coordsys.delta) / 2;
    const double addInu = nu_near_min ?
        pow_2(point.vz) * (pprol.lambda - absnu) / 2 :
    nu_near_max ?
        pow_2(point.vR) * (pprol.lambda - absnu) / 2 :
        pow_2( pprol.nudot * (pprol.lambda - absnu) ) /
            (8 * (coordsys.delta - absnu) * absnu );
    const double Inu = absnu * (E - En) + (pprol.lambda - absnu) * Phi + addInu;

    return AxisymFunctionFudge(pprol, E, Lz, Ilambda, Inu, potential);
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
    if(tau >= point.coordsys.delta) {  // evaluating J_lambda
        lambda= tau;
        nu    = fabs(point.nu);
        mult  = lambda-nu;
        I     = Ilambda;
    } else {    // evaluating J_nu
        lambda= point.lambda;
        nu    = tau;
        mult  = nu-lambda;
        I     = Inu;
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
    const double tauminusdelta = tau - point.coordsys.delta;
    if(val)
        *val = ( E * tauminusdelta - pow_2(Lz)/2 ) * tau
             - (I + Phi * mult) * tauminusdelta;
    if(der || der2) {
        coord::GradProlSph gradProl = coord::toGrad<coord::Cyl, coord::ProlSph> (gradCyl, coordDeriv);
        double dPhidtau = (tau >= point.coordsys.delta) ? gradProl.dlambda : gradProl.dnu;
        if(der)
            *der = E * (tau+tauminusdelta) - pow_2(Lz)/2 - I 
                 - (mult+tauminusdelta) * Phi - (tauminusdelta!=0 ? tauminusdelta * mult * dPhidtau : 0);
        if(der2) {
            double d2Phidtau2 = (tau >= point.coordsys.delta) ?
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
    const AxisymFunctionBase& fnc;      ///< parameters of aux.fnc. (AxisymFunctionStaeckel or AxisymFunctionFudge)
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
        const double tauminusdelta = tau - CS.delta;
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
        const double tauminusdelta = tau - CS.delta;
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
    const double delta=fnc.point.coordsys.delta;

    // figure out the value of function at and around some important points
    double fnc_zero = fnc(0);
    lim.nu_min = 0;
    lim.nu_max = lim.lambda_min = lim.lambda_max = NAN;  // means not yet determined
    double nu_upper = delta;     // upper bound on the interval to locate the root for nu_max
    double lambda_lower = delta; // lower bound on the interval for lambda_min

    if(fnc.Lz==0) {
        // special case: f(delta) = -0.5 Lz^2 = 0, may have either tube or box orbit in the meridional plane
        double deltaminus = delta*(1-1e-15), deltaplus = delta*(1+1e-15);
        if(fnc(deltaminus)<0) {  // box orbit: f<0 at some interval left of delta
            if(fnc_zero>0)       // there must be a range of nu where the function is positive
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
    {   // find range for J_nu = J_z if it has not been determined at the previous stage
        if(fnc_zero>0)
            lim.nu_max = math::findRoot(fnc, fabs(fnc.point.nu), nu_upper, ACCURACY_RANGE);
        if(!isFinite(lim.nu_max))
            // means that the value f(nu) was just very slightly negative, or that f(0)<=0
            lim.nu_max = fabs(fnc.point.nu);  // i.e. this is a clear upper boundary of the range of allowed nu
    }

    // range for J_lambda = J_r.
    // due to roundoff errors, it may actually happen that f(lambda) is a very small negative number
    // in this case we need to estimate the value of lambda at which it is strictly positive (for root-finder)
    double lambda_pos = fnc.point.lambda;
    const math::PointNeighborhood pn_lambda(fnc, fnc.point.lambda);
    if(pn_lambda.f0<=0) {   // it could be slightly smaller than zero due to roundoff errors
        lambda_pos += pn_lambda.dxToPositive();
        if(pn_lambda.fder>=0) {
            lim.lambda_min = fnc.point.lambda;
            // it may still happen that lambda is large and dx is small, i.e. lambda+dx == lambda due to roundoff
            if(lambda_pos == fnc.point.lambda)
                lambda_pos *= 1+1e-15;     // add a tiny bit, in the hope that it fixes the problem...
        } 
        if(pn_lambda.fder<=0){
            lim.lambda_max = fnc.point.lambda;
            if(fnc.point.lambda == delta)  // can't be lower than that! means that the range is zero
                lim.lambda_min = delta;
            else if(lambda_pos == fnc.point.lambda)
                lambda_pos *= 1-1e-15;     // subtract a tiny bit
        }
    }
    /*  now two more problems may occur:
        1. lambda_pos = NaN means that it could not be found, i.e., 
        f(lambda)<0, f'(lambda) is very small and f''(lambda)<0, we are near the top of
        inverse parabola that does not cross zero. This means that within roundoff errors 
        the range of lambda with positive f() is negligibly small, so we discard it.
        2. we are still near the top of parabola that crosses zero in two points which are
        very close to each other. Then again the range of lambda with positive f() is too small
        to be tracked accurately. Note that dxBetweenRoots() may be NaN in a perfectly legitimate
        case that f(lambda)>0 and f''>0, so that the parabola is curved upward and never crosses zero; 
        thus we test an inverse condition which is valid for NaN.
    */
    if(isFinite(lambda_pos) && !(pn_lambda.dxBetweenRoots() < fnc.point.lambda * ACCURACY_RANGE)) {
        if(!isFinite(lim.lambda_min)) {  // not yet determined 
            lim.lambda_min = math::findRoot(fnc, lambda_lower, lambda_pos, ACCURACY_RANGE);
        }
        if(!isFinite(lim.lambda_max)) {
            lim.lambda_max = math::findRoot(AxisymScaledForRootfinder(fnc), 
                lambda_pos, INFINITY, ACCURACY_RANGE);
        }
    } else {  // can't find a value of lambda with positive p^2(lambda) -- dominated by roundoff errors
        lim.lambda_min = lim.lambda_max = fnc.point.lambda;  // means that we are on a (nearly)-circular orbit
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
    using the expressions A4-A9 in Sanders(2012). */
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
    using the expressions A10-A12 in Sanders(2012).  These quantities do depend on angles. */
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
Angles computeAngles(const AxisymIntDerivatives& derI, const AxisymGenFuncDerivatives& derS, bool addPiToThetaZ)
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
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    const coord::ProlSph coordsys(pow_2(interfocalDistance));
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    if(!isFinite(fnc.E+fnc.Ilambda+fnc.Inu+fnc.Lz) || fnc.E>=0)
        return Actions(NAN, NAN, fnc.Lz);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles actionAnglesAxisymFudge(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance, Frequencies* freq)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    const coord::ProlSph coordsys(pow_2(interfocalDistance));
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    if(!isFinite(fnc.E+fnc.Ilambda+fnc.Inu+fnc.Lz) || fnc.E>=0)
        return ActionAngles(Actions(NAN, NAN, fnc.Lz), Angles(NAN, NAN, NAN));
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim, freq);
}

#if 0
/// Inverse transformation: obtain integrals of motion (E,I3,Lz) from actions (Jr,Jz,Jphi)
namespace {

class OrbitCenterFinder: public math::IFunctionNoDeriv {
public:
    OrbitCenterFinder(const potential::OblatePerfectEllipsoid& p,
        double _Lz, double _I3) : potential(p), Lz2(_Lz*_Lz), I3(_I3) {}
    virtual double value(const double R) const
    {
        coord::GradCyl grad;
        potential.eval(coord::PosCyl(R, 0, 0), NULL, &grad);
        return -grad.dR + Lz2/pow_3(R) + 2*R*I3/pow_2(R*R+potential.coordsys().delta);
    }
private:
    const potential::OblatePerfectEllipsoid& potential;
    const double Lz2, I3;
};

class IntegralsOfMotionFinder: public math::IFunctionNdimDeriv {
public:
    IntegralsOfMotionFinder(
        const potential::OblatePerfectEllipsoid& p,
        const Actions& a, AxisymIntLimits &l) :
        potential(p), acts(a), lim(l) {}

    AxisymFunctionStaeckel makefnc(const double vars[]) const
    {
        double E = vars[0];
        double I3= vars[1];
        // find a plausible cylindrical radius in the equatorial plane lying inside the orbit
        OrbitCenterFinder f(potential, acts.Jphi, I3);
        double R0 = math::findRoot(f, 1e-5, 1e5, 1e-10);  ///!!!!!! boundaries????
        coord::PosVelProlSph pos = coord::toPosVel<coord::Cyl, coord::ProlSph>(
            coord::PosVelCyl(R0, 0, 0, 0, 0, acts.Jphi!=0 ? acts.Jphi/R0 : 0), potential.coordsys());
        return AxisymFunctionStaeckel(pos, E, acts.Jphi, I3, potential);
    }
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        const AxisymFunctionStaeckel fnc = makefnc(vars);
        // store the limits in the external variable,
        // so that the best-match values will be available after findRootNdim finishes
        lim = findIntegrationLimitsAxisym(fnc);
        if(values) {
            Actions trialActs = computeActions(fnc, lim);
            values[0] = trialActs.Jr - acts.Jr;
            if(numValues()==2)
                values[1] = trialActs.Jz - acts.Jz;
        }
        if(derivs) {
            AxisymActionDerivatives der = computeActionDerivatives(fnc, lim);
            derivs[0] = der.dJrdE;
            if(numValues()==2) {
                derivs[1] = der.dJrdI3;
                derivs[2] = der.dJzdE;
                derivs[3] = der.dJzdI3;
            }
        }
    }
    virtual unsigned int numVars() const { return acts.Jz==0 ? 1 : 2; }
    virtual unsigned int numValues() const { return acts.Jz==0 ? 1 : 2; }
private:
    const potential::OblatePerfectEllipsoid& potential;
    const Actions acts;
    AxisymIntLimits &lim;
};

class ToyEtaFinder: public math::IFunctionNoDeriv {
public:
    ToyEtaFinder(const double _ecc, const double _rhs) :
        ecc(_ecc), rhs(_rhs) {}
    virtual double value(const double eta) const {
        double lhs = eta + ecc * sin(eta)
            - 2 * sqrt(1-ecc*ecc) * atan( sqrt( (1+ecc) / (1-ecc) ) * tan(eta*0.5) );
        return lhs - rhs;
    }
private:
    const double ecc, rhs;
};

class ToyTauFinder: public math::IFunctionNoDeriv {
public:
    ToyTauFinder(const double _cosi, const double _rhs) :
    cosi(_cosi), rhs(_rhs) {}
    virtual double value(const double t) const {
        double xtmp = 2*t / sqrt(fmax( pow_2(1-t*t) - pow_2(cosi * (1+t*t)), 0) );
        double lhs  = atan(xtmp) - cosi * atan(xtmp * cosi);
        return lhs  - rhs;
    }
private:
    const double cosi, rhs;
};

math::PtrFunction createRadialCoordMapping(const AxisymFunctionBase& fnc, const AxisymIntLimits& lim, double L) 
{
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrandEndpointSing transf(integrand, lim.lambda_min, lim.lambda_max);
    integrand.n = AxisymIntegrand::nplus1;
    integrand.a = AxisymIntegrand::azero;
    integrand.c = AxisymIntegrand::czero;
    const int NSUB = 48;
    std::vector<double> rho(NSUB+1), subint(NSUB+1), toyr(NSUB+1);
    rho[0] = 0.5 * (sqrt(lim.lambda_min) + sqrt(lim.lambda_min - fnc.point.coordsys.delta) - sqrt(fnc.point.coordsys.delta));
    for(int s=1; s<=NSUB; s++) {
        double la1 = (lim.lambda_max-lim.lambda_min) * ((s-1.)/NSUB) + lim.lambda_min;
        double la2 = (lim.lambda_max-lim.lambda_min) * ((s+0.)/NSUB) + lim.lambda_min;
        subint[s]  = subint[s-1] +
            math::integrateGL(transf, transf.y_from_x(la1), transf.y_from_x(la2), INTEGR_ORDER);
        rho[s] = 0.5 * (sqrt(la2) + sqrt(la2 - fnc.point.coordsys.delta) - sqrt(fnc.point.coordsys.delta));
    }
    // now the last element of subint is the value of Jr times 1/Pi
    double J   = subint.back() / M_PI + L;  // J = Jr+L = sqrt(a),  where a is semimajor axis
    double ecc = sqrt(1 - pow_2(L/J));
    // for each value of rho of the real coord.sys.,
    // we find the corresponding r_toy of the toy coordinate system from the equation
    //   eta + ecc sin(eta) - 2 sqrt(1-ecc^2) arctan( sqrt((1+ecc)/(1-ecc)) tan(eta/2) ) = subint(rho) / J,
    // where  r_toy = a (1 - ecc cos(eta) )
    for(int s=0; s<=NSUB; s++) {
        ToyEtaFinder f(ecc, subint[s] / J);
        double eta = s==0 ? 0 : s==NSUB ? M_PI : math::findRoot(f, 0, M_PI, ACCURACY_RANGE);
        toyr[s] = J*J * (1 - ecc * cos(eta));
    }
    // add point at zero
    toyr.insert(toyr.begin(), 0);
    rho.insert(rho.begin(), 0);
    // create an interpolator
    return math::PtrFunction(new math::CubicSpline(toyr, rho));
}

math::PtrFunction createVerticalCoordMapping(const AxisymFunctionBase& fnc, const AxisymIntLimits& lim) 
{
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrandEndpointSing transf(integrand, lim.nu_min, lim.nu_max);
    integrand.n = AxisymIntegrand::nplus1;
    integrand.a = AxisymIntegrand::azero;
    integrand.c = AxisymIntegrand::czero;
    const int NSUB = 48;
    std::vector<double> tau(NSUB+1), subint(NSUB+1), toytau(NSUB+1);
    for(int s=1; s<=NSUB; s++) {
        double nu1 = lim.nu_max * ((s-1.)/NSUB);  // limits of integration over sub-interval
        double nu2 = lim.nu_max * ((s+0.)/NSUB);
        subint[s]  = subint[s-1] +
        math::integrateGL(transf, transf.y_from_x(nu1), transf.y_from_x(nu2), INTEGR_ORDER);
        double cosv2 = sqrt(nu2 / fnc.point.coordsys.delta);
        tau[s] = cosv2 / (sqrt(1-pow_2(cosv2)) + 1);
    }
    // now the last element of subint is the value of Jz times 2/Pi
    double L = subint.back() * (2/M_PI) + fabs(fnc.Lz);
    double cosi = fabs(fnc.Lz) / L;
    // for each value of tau of the real coord.sys.,
    // we find the corresponding tau_toy of the toy coordinate system from the equation
    //   arctan(x) - cos(i) arctan(x cos(i)) = subint(tau) / L,  where
    //   x = cos(theta) / sqrt(sin(i)^2 - cos(theta)^2),  and  tau_toy = cos(theta) / (1+sin(theta))
    toytau.back() = sqrt(1 - pow_2(cosi)) / (1 + cosi);
    for(int s=1; s<NSUB; s++) {
        ToyTauFinder f(cosi, subint[s] / L);
        toytau[s] = math::findRoot(f, toytau.front(), toytau.back(), ACCURACY_RANGE);
    }
    // add the end point for extrapolation
    if(cosi>0) {
        toytau.push_back(1.);
        tau.push_back(1.);
    }
    // add a symmetric branch at negative tau
    assert(toytau.size() == tau.size());
    toytau.insert(toytau.begin(), toytau.size()-1, 0);
    tau.insert(tau.begin(), tau.size()-1, 0);
    for(unsigned int s=0; s<tau.size()/2; s++) {
        tau[s] = -tau[tau.size()-1-s];
        toytau[s] = -toytau[toytau.size()-1-s];
    }
    // create an interpolator
    return math::PtrFunction(new math::CubicSpline(toytau, tau));
}

} // namespace

void computeIntegralsStaeckel(
    const potential::OblatePerfectEllipsoid& potential, 
    const Actions& acts,
    math::PtrFunction &rad, math::PtrFunction &ver)
{
    // initial guess on the point lying within the orbit extent in the meridional plane
    double Rcirc = R_from_Lz(potential, fabs(acts.Jphi)+acts.Jr+acts.Jz);
    double kappa, nu, Omega;
    epicycleFreqs(potential, Rcirc, kappa, nu, Omega);
    // initial guess for total energy E and third integral I3
    double initVars[2] = {0,0};
    potential.eval(coord::PosCyl(Rcirc, 0, 0), initVars);
    if(acts.Jphi!=0)
        initVars[0] += 0.5 * pow_2(acts.Jphi/Rcirc);
    initVars[0] += kappa * acts.Jr + nu * acts.Jz;
    initVars[1] = nu * acts.Jz * (pow_2(Rcirc) + potential.coordsys().delta);
    double results[2] = {0,0};
    // find the integrals of motion that result in the required actions,
    // and as a by-product store the limits of oscillation in both lambda and nu directions
    AxisymIntLimits lim;
    IntegralsOfMotionFinder f(potential, acts, lim);
    math::findRootNdimDeriv(f, initVars, 1e-6, 32, results);
    const AxisymFunctionStaeckel fnc = f.makefnc(results);
    rad = createRadialCoordMapping(fnc, lim, acts.Jz+fabs(acts.Jphi));
    ver = createVerticalCoordMapping(fnc, lim);
}
#endif
}  // namespace actions
