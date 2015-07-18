#include "actions_staeckel.h"
#include "mathutils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

/** relative accuracy in integrals for computing actions and angles */
const double ACCURACY_ACTION = 0e-3;  // this is more than enough

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over,
    also determined the minimum range that will be considered non-zero (relative to gamma-alpha) */
const double ACCURACY_RANGE = 1e-6;

// ------ SPECIALIZED functions for Staeckel action finder -------

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymFunctionStaeckel findIntegralsOfMotionOblatePerfectEllipsoid
    (const potential::StaeckelOblatePerfectEllipsoid& poten, const coord::PosVelCyl& point)
{
    double E = potential::totalEnergy(poten, point);
    double Lz= coord::Lz(point);
    const coord::ProlSph& coordsys=poten.coordsys();
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double Glambda;
    poten.eval_deriv(pprol.lambda, &Glambda);
    double I3;
    if(point.z==0)   // special case: nu=0
        I3 = 0.5 * pow_2(point.vz) * (pow_2(point.R)+coordsys.gamma-coordsys.alpha);
    else   // general case: eq.3 in Sanders(2012)
        I3 = fmax(0,
            (pprol.lambda+coordsys.gamma) * 
            (E - pow_2(Lz)/2/(pprol.lambda+coordsys.alpha) + Glambda) -
            pow_2(pprol.lambdadot*(pprol.lambda-pprol.nu)) / 
            (8*(pprol.lambda+coordsys.alpha)*(pprol.lambda+coordsys.gamma)) );
    return AxisymFunctionStaeckel(pprol, mathutils::sign(point.z), E, Lz, I3, poten);
}

/** auxiliary function that enters the definition of canonical momentum for 
    for the Staeckel potential: it is the numerator of eq.50 in de Zeeuw(1985);
    the argument tau is replaced by tau+gamma >= 0. */
void AxisymFunctionStaeckel::eval_deriv(const double tauplusgamma, 
    double* val, double* der, double* der2) const
{
    assert(tauplusgamma>=0);
    if(!mathutils::isFinite(tauplusgamma)) {
        if(val) { // used in the root-finder on an infinite interval
            // need to return a negative number of a reasonable magnitude
            double bigtau = 10*(point.coordsys.gamma-point.coordsys.alpha);
            *val = - (fabs(E) * bigtau + fabs(I3) + Lz*Lz) * bigtau;
        }
        if(der)
            *der = NAN;
        if(der2)
            *der2 = NAN;
        return;
    }
    double G, dG, d2G;
    fncG.eval_deriv(tauplusgamma-point.coordsys.gamma, &G, der? &dG : NULL, der2? &d2G : NULL);
    const double tauplusalpha = tauplusgamma+point.coordsys.alpha-point.coordsys.gamma;
    if(val)
        *val = ( (E + G) * tauplusgamma - I3 ) * tauplusalpha - Lz*Lz/2 * tauplusgamma;
    if(der)
        *der = (E+G)*(tauplusgamma+tauplusalpha) + dG*tauplusgamma*tauplusalpha - I3 - Lz*Lz/2;
    if(der2)
        *der2 = 2*(E+G) + 2*dG*(tauplusgamma+tauplusalpha) + d2G*tauplusgamma*tauplusalpha;
}

// -------- SPECIALIZED functions for the Axisymmetric Fudge action finder --------

/** compute true (E, Lz) and approximate (Ilambda, Inu) integrals of motion in an arbitrary 
    potential used for the Staeckel Fudge, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymFunctionFudge findIntegralsOfMotionAxisymFudge
    (const potential::BasePotential& poten, const coord::PosVelCyl& point, const coord::ProlSph& coordsys)
{
    double Phi;
    poten.eval(point, &Phi);
    double E=Phi+(pow_2(point.vR)+pow_2(point.vz)+pow_2(point.vphi))/2;
    double Lz= coord::Lz(point);
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double Ilambda, Inu;
    Ilambda = (pprol.lambda+coordsys.gamma) * (E - pow_2(Lz)/2/(pprol.lambda+coordsys.alpha))
            - (pprol.lambda-pprol.nu)*Phi
            - pow_2(pprol.lambdadot*(pprol.lambda-pprol.nu)) /
              (8*(pprol.lambda+coordsys.alpha)*(pprol.lambda+coordsys.gamma));
    Inu     = (pprol.nu+coordsys.gamma) * (E - pow_2(Lz)/2/(pprol.nu+coordsys.alpha))
            + (pprol.lambda-pprol.nu)*Phi;
    if(pprol.nu+coordsys.gamma<=1e-12*(coordsys.gamma-coordsys.alpha))  // z==0, nearly
        Inu+= pow_2(point.vz)*(pprol.lambda-pprol.nu)/2;
    else
        Inu-= pow_2(pprol.nudot*(pprol.lambda-pprol.nu)) /
              (8*(pprol.nu+coordsys.alpha)*(pprol.nu+coordsys.gamma));
    return AxisymFunctionFudge(pprol, mathutils::sign(point.z), E, Lz, Ilambda, Inu, poten);
}

/** Auxiliary function F analogous to that of Staeckel action finder:
    namely, the momentum is given by  p_tau^2 = F(tau) / (2*(tau+alpha)^2*(tau+gamma)),
    where  -gamma<=tau<=-alpha  for the  nu-component of momentum, 
    and   -alpha<=tau<infinity  for the  lambda-component of momentum.
    For numerical convenience, tau is replaced by x=tau+gamma.
*/
void AxisymFunctionFudge::eval_deriv(const double tauplusgamma, 
    double* val, double* der, double* der2) const
{
    assert(tauplusgamma>=0);
    const double gamma=point.coordsys.gamma, alpha=point.coordsys.alpha;
    if(der2)
        *der2 = NAN;    // shouldn't be used
    if(!mathutils::isFinite(tauplusgamma)) {
        if(val) { // used in the root-finder on an infinite interval:
            // need to return a negative number of a reasonable magnitude
            double bigtau = 10*(gamma-alpha);
            *val = - (fabs(E) * bigtau + fabs(Ilambda) + Lz*Lz/2) * bigtau;
        }
        if(der)
            *der = NAN;
        return;
    }
    double lambda, nu, I, mult;
    if(tauplusgamma >= gamma-alpha) {  // evaluating J_lambda
        lambda= tauplusgamma-gamma;
        nu    = point.nu;
        mult  = lambda-nu;
        I     = Ilambda;
    } else {    // evaluating J_nu
        lambda= point.lambda;
        nu    = tauplusgamma-gamma;
        mult  = nu-lambda;
        I     = Inu;
    }
    // compute the potential in coordinates transformed from prol.sph. to cylindrical
    coord::PosDerivT <coord::ProlSph, coord::Cyl> coordDeriv;
    const coord::PosProlSph posProl(lambda, nu, point.phi, point.coordsys);
    const coord::PosCyl posCyl = der? 
        coord::toPosDeriv<coord::ProlSph, coord::Cyl>(posProl, &coordDeriv) :
        coord::toPosCyl(posProl);
    double Phi;
    coord::GradCyl gradCyl;
    poten.eval(posCyl, &Phi, der? &gradCyl : NULL);
    const double tauplusalpha = tauplusgamma+alpha-gamma;
    if(val)
        *val = ( E * tauplusalpha - pow_2(Lz)/2 ) * tauplusgamma
             - (I + Phi * mult) * tauplusalpha;
    if(der) {
        coord::GradProlSph gradProl = coord::toGrad<coord::Cyl, coord::ProlSph> (gradCyl, coordDeriv);
        double dPhidtau = (tauplusgamma >= gamma-alpha) ? gradProl.dlambda : gradProl.dnu;
        *der = E * (tauplusgamma+tauplusalpha) - pow_2(Lz)/2 - I 
             - (mult+tauplusalpha) * Phi - tauplusalpha * mult * dPhidtau;
    }
}

// -------- COMMON routines for Staeckel and Fudge action finders --------
/** parameters for the function that computes actions and angles
    by integrating an auxiliary function "fnc" as follows:
    the canonical momentum is   p^2(tau) = fnc(tau) / [2 (tau+alpha)^2 (tau+gamma)];
    the integrand is given by   p^n * (tau+alpha)^a * (tau+gamma)^c  if p^2>0, otherwise 0.
*/
class AxisymIntegrand: public mathutils::IFunction {
public:
    const AxisymFunctionBase& fnc;          ///< parameters of aux.fnc.: may be either AxisymFunctionStaeckel or AxisymFunctionFudge
    enum { nplus1, nminus1 } n;             ///< power of p: +1 or -1
    enum { azero, aminus1, aminus2 } a;     ///< power of (tau+alpha): 0, -1, -2
    enum { czero, cminus1 } c;              ///< power of (tau+gamma): 0 or -1
    explicit AxisymIntegrand(const AxisymFunctionBase& d) : fnc(d) {};

    virtual int numDerivs() const { return 0; }    
    /** integrand for the expressions for actions and their derivatives 
        (e.g.Sanders 2012, eqs. A1, A4-A12).  It uses the auxiliary function to compute momentum,
        and multiplies it by some powers of (tau+alpha) and (tau+gamma).
    */
    virtual void eval_deriv(const double tauplusgamma, 
        double* val=0, double* =0, double* =0) const
    {
        assert(val!=NULL);
        assert(tauplusgamma>=0);
        const coord::ProlSph& CS = fnc.point.coordsys;
        const double tauplusalpha = tauplusgamma+CS.alpha-CS.gamma;
        const double p2 = fnc.value(tauplusgamma) / 
            (2*pow_2(tauplusalpha)*tauplusgamma);
        if(p2<0) {
            *val = 0;
            return;
        }
        double result = sqrt(p2);
        if(n==nminus1)
            result = 1/result;
        if(a==aminus1)
            result /= tauplusalpha;
        else if(a==aminus2)
            result /= pow_2(tauplusalpha);
        if(c==cminus1)
            result /= tauplusgamma;
        if(!mathutils::isFinite(result))
            result=0;  // ad hoc fix to avoid problems at the boundaries of integration interval
        *val = result;
    }
};

/** A simple function that facilitates locating the root of auxiliary function 
    on a semi-infinite interval for lambda: instead of F(tau) we consider 
    F1(tau)=F(tau)/tau^2, which tends to a finite negative limit as tau tends to infinity. */
class AxisymScaledForRootfinder: public mathutils::IFunction {
public:
    const AxisymFunctionBase& fnc;
    explicit AxisymScaledForRootfinder(const AxisymFunctionBase& d) : fnc(d) {};
    virtual int numDerivs() const { return fnc.numDerivs(); }    
    virtual void eval_deriv(const double tauplusgamma, 
        double* val=0, double* der=0, double* =0) const
    {
        assert(tauplusgamma>=0);
        if(!mathutils::isFinite(tauplusgamma)) {
            if(val)
                *val = fnc.E; // the asymptotic value
            if(der)
                *der = NAN;    // we don't know it
            return;
        }
        double fval, fder;
        fnc.eval_deriv(tauplusgamma, &fval, der? &fder : NULL);
        if(val)
            *val = fval / pow_2(tauplusgamma);
        if(der)
            *der = (fder - 2*fval/tauplusgamma) / pow_2(tauplusgamma);
    }
};

/** Compute the intervals of tau for which p^2(tau)>=0, 
    where  -gamma = tau_nu_min <= tau <= tau_nu_max <= -alpha  is the interval for the "nu" branch,
    and  -alpha <= tau_lambda_min <= tau <= tau_lambda_max < infinity  is the interval for "lambda".
    For numerical convenience, we replace tau with  x=tau+gamma.
*/
AxisymIntLimits findIntegrationLimitsAxisym(const AxisymFunctionBase& fnc)
{
    if(fnc.E>=0)
        throw std::invalid_argument("Error in Axisymmetric Staeckel/Fudge action finder: E>=0");
    AxisymIntLimits lim;
    const double gamma=fnc.point.coordsys.gamma, alpha=fnc.point.coordsys.alpha;

    // figure out the value of function at and around some important points
    double fnc_gamma = fnc.value(0);
    const mathutils::PointNeighborhood pn_alpha (fnc, gamma-alpha);
    assert(pn_alpha.f0<=0);  // this is -0.5 Lz^2 so may not be positive
    const mathutils::PointNeighborhood pn_lambda(fnc, gamma+fnc.point.lambda);
    lim.xnu_min = 0;
    lim.xnu_max = lim.xlambda_min = lim.xlambda_max = NAN;  // means not yet determined
    double xnu_upper = gamma-alpha;      // upper bound on the interval to locate the root for nu_max
    double xlambda_lower = gamma-alpha;  // lower bound on the interval for lambda_min
    if(pn_alpha.f0==0) {        // special case: L_z==0, may have either tube or box orbit in the meridional plane
        if(pn_alpha.fder>0) {   // box orbit: f(alpha)=0 and f'(alpha)>0, so f<0 at some interval left of alpha
            if(fnc_gamma>0)     // there must be a range of nu where the function is positive
                xnu_upper = fmax((gamma-alpha)*0.9, gamma-alpha+pn_alpha.dx_to_negative());
            else 
                lim.xnu_max = lim.xnu_min;
            lim.xlambda_min = gamma-alpha;
        } else {      // tube orbit: f(alpha)=0, f'(alpha)<0, f must be negative on some interval right of alpha
            xlambda_lower = gamma-alpha + fmin((alpha+fnc.point.lambda)*0.1, pn_alpha.dx_to_negative());
            lim.xnu_max = gamma-alpha;
        }
    }
    if(!mathutils::isFinite(lim.xnu_max)) 
    {   // find range for J_nu = J_z if it has not been determined at the previous stage
        if(fnc_gamma>0)
            lim.xnu_max = mathutils::findRoot(fnc, fnc.point.nu+gamma, xnu_upper, ACCURACY_RANGE);
        if(!mathutils::isFinite(lim.xnu_max))   // means that the value f(nu) was just very slightly negative, or f(gamma)<=0
            lim.xnu_max = fnc.point.nu+gamma;   // i.e. this is a clear upper boundary of the range of allowed nu
    }

    // range for J_lambda = J_r
    if(pn_lambda.f0<=0) {   // it could be slightly smaller than zero due to roundoff errors
        if(pn_lambda.fder>=0) {
            lim.xlambda_min = gamma+fnc.point.lambda;
        } 
        if(pn_lambda.fder<=0){
            lim.xlambda_max = gamma+fnc.point.lambda;
        }
    }
    // due to roundoff errors, it may actually happen that f(lambda) is a very small negative number
    // in this case we need to estimate the value of lambda at which it is strictly positive (for root-finder)
    double xlambda_pos = gamma+fnc.point.lambda + pn_lambda.dx_to_positive();
    if(lim.xlambda_min!=lim.xlambda_min) {  // not yet determined 
        lim.xlambda_min = mathutils::findRoot(fnc, xlambda_lower, xlambda_pos, ACCURACY_RANGE);
    }
    if(lim.xlambda_max!=lim.xlambda_max)
        lim.xlambda_max = mathutils::findRoot(AxisymScaledForRootfinder(fnc), 
            xlambda_pos, HUGE_VAL, ACCURACY_RANGE);

    if(!mathutils::isFinite(lim.xlambda_min+lim.xlambda_max+lim.xnu_max+lim.xnu_min)
        || fnc.point.nu+gamma>lim.xnu_max
        || fnc.point.lambda+gamma<lim.xlambda_min
        || fnc.point.lambda+gamma>lim.xlambda_max)
        throw std::invalid_argument("findLimits: something wrong with the data");
    
    return lim;
}

/** Compute the derivatives of integrals of motion (E, Lz, I3) over actions (Jr, Jz, Jphi),
    using the expressions A4-A9 in Sanders(2012).  These quantities are independent of angles,
    and in particular, the derivatives of energy w.r.t. the three actions are the frequencies. */
AxisymIntDerivatives computeIntDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    AxisymIntegrand integrand(fnc);
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    double dJrdE = mathutils::integrateScaled(integrand, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdE = mathutils::integrateScaled(integrand, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    double dJrdI3 = -mathutils::integrateScaled(integrand, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdI3 = -mathutils::integrateScaled(integrand, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    double dJrdLz = -fnc.Lz * mathutils::integrateScaled(integrand, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdLz = -fnc.Lz * mathutils::integrateScaled(integrand, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);

    double signLz  = mathutils::sign(fnc.Lz);
    AxisymIntDerivatives der;
    // invert the matrix of derivatives
    if(lim.xnu_min==lim.xnu_max) {
        // special case z==0: motion in z is irrelevant, but we could not compute dJzdI3 which is not zero
        der.dEdJr   = 1/dJrdE;
        der.dEdJphi =-dJrdLz/dJrdE;
        der.dI3dJr  = der.dI3dJz = der.dI3dJphi = der.dEdJz = 0;
    } else {  // everything as normal
        double det  = dJrdE*dJzdI3-dJrdI3*dJzdE;
        der.dEdJr   = dJzdI3/det;
        der.dEdJz   =-dJrdI3/det;
        der.dEdJphi = (dJrdI3*dJzdLz-dJrdLz*dJzdI3)/det * signLz;
        der.dI3dJr  =-dJzdE/det;
        der.dI3dJz  = dJrdE/det;
        der.dI3dJphi=-(dJrdE*dJzdLz-dJrdLz*dJzdE)/det * signLz;
    }
    der.dLzdJr  = 0;
    der.dLzdJz  = 0;
    der.dLzdJphi= signLz;
    return der;
}

/** Compute the derivatives of generating function S over integrals of motion (E, Lz, I3),
    using the expressions A10-A12 in Sanders(2012).  These quantities do depend on angles. */
AxisymGenFuncDerivatives computeGenFuncDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    const double signldot = fnc.point.lambdadot>=0?+1:-1;
    const double signndot = fnc.point.nudot>=0?+1:-1;
    const double gamma    = fnc.point.coordsys.gamma;
    AxisymGenFuncDerivatives der;
    AxisymIntegrand integrand(fnc);
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    der.dSdE =
        signldot * mathutils::integrateScaled(integrand, 
            lim.xlambda_min, fnc.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * mathutils::integrateScaled(integrand, 
            lim.xnu_min, fnc.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    der.dSdI3 = 
        signldot * -mathutils::integrateScaled(integrand, 
            lim.xlambda_min, fnc.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * -mathutils::integrateScaled(integrand, 
            lim.xnu_min, fnc.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    der.dSdLz = fnc.point.phi +
        signldot * -fnc.Lz * mathutils::integrateScaled(integrand, 
            lim.xlambda_min, fnc.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * -fnc.Lz * mathutils::integrateScaled(integrand, 
            lim.xnu_min, fnc.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    return der;
}

/** Compute actions by integrating the momentum over the range of tau on which it is positive,
    separately for the "nu" and "lambda" branches (equation A1 in Sanders 2012). */
Actions computeActions(const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    Actions acts;
    AxisymIntegrand integrand(fnc);
    integrand.n = AxisymIntegrand::nplus1;  // momentum goes into the numerator
    integrand.a = AxisymIntegrand::azero;
    integrand.c = AxisymIntegrand::czero;
    acts.Jr = mathutils::integrateScaled(integrand, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / M_PI;
    acts.Jz = mathutils::integrateScaled(integrand, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) * 2/M_PI;
    acts.Jphi = fabs(fnc.Lz);
    return acts;
}

/** Compute angles from the derivatives of integrals of motion and the generating function
    (equation A3 in Sanders 2012). */
Angles computeAngles(const AxisymIntDerivatives& derI, const AxisymGenFuncDerivatives& derS, bool addPiToThetaZ)
{
    Angles angs;
    angs.thetar   = derS.dSdE*derI.dEdJr   + derS.dSdI3*derI.dI3dJr   + derS.dSdLz*derI.dLzdJr;
    angs.thetaz   = derS.dSdE*derI.dEdJz   + derS.dSdI3*derI.dI3dJz   + derS.dSdLz*derI.dLzdJz;
    angs.thetaphi = derS.dSdE*derI.dEdJphi + derS.dSdI3*derI.dI3dJphi + derS.dSdLz*derI.dLzdJphi;
    angs.thetar   = mathutils::wrapAngle(angs.thetar);
    angs.thetaz   = mathutils::wrapAngle(angs.thetaz + M_PI*addPiToThetaZ);
    angs.thetaphi = mathutils::wrapAngle(angs.thetaphi);
    return angs;
}

/** The sequence of operations needed to compute both actions and angles.
    Note that for a given orbit, only the derivatives of the generating function depend 
    on the angles (assuming that the actions are constant); in principle, this may be used 
    to skip the computation of the derivatives matrix of integrals (not presently implemented). */
ActionAngles computeActionAngles(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    Actions acts = computeActions(fnc, lim);
    AxisymIntDerivatives derI = computeIntDerivatives(fnc, lim);
    AxisymGenFuncDerivatives derS = computeGenFuncDerivatives(fnc, lim);
    bool addPiToThetaZ = ((fnc.signz<0)^(fnc.point.nudot<0)) && acts.Jz!=0;
    Angles angs = computeAngles(derI, derS, addPiToThetaZ);
    return ActionAngles(acts, angs);
}

// -------- THE DRIVER ROUTINES --------

Actions axisymStaeckelActions(const potential::StaeckelOblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles axisymStaeckelActionAngles(const potential::StaeckelOblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim);
}

Actions axisymFudgeActions(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if((potential.symmetry() & potential.ST_AXISYMMETRIC) != potential.ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(interfocalDistance==0)
        interfocalDistance = estimateInterfocalDistance(potential, point);
    const coord::ProlSph coordsys(-pow_2(interfocalDistance)-1., -1.);
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles axisymFudgeActionAngles(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if((potential.symmetry() & potential.ST_AXISYMMETRIC) != potential.ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(interfocalDistance==0)
        interfocalDistance = estimateInterfocalDistance(potential, point);
    const coord::ProlSph coordsys(-pow_2(interfocalDistance)-1., -1.);
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);    
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim);
}

// ------ Estimation of interfocal distance -------

/** Helper function for finding the roots of (effective) potential in either R or z direction */
class OrbitSizeFunction: public mathutils::IFunction {
public:
    const potential::BasePotential& potential;
    double R;
    double phi;
    double E;
    double Lz2;
    enum { FIND_RMIN, FIND_RMAX, FIND_ZMAX } mode;
    explicit OrbitSizeFunction(const potential::BasePotential& p) : potential(p), mode(FIND_RMAX) {};
    virtual int numDerivs() const { return 2; }
    /** This function is used in the root-finder to determine the turnaround points of an orbit:
        in the radial direction, it returns -(1/2) v_R^2, and in the vertical direction -(1/2) v_z^2 .
        Moreover, to compute the location of pericenter this is multiplied by R^2 to curb the sharp rise 
        of effective potential at zero, which is problematic for root-finder. */
    virtual void eval_deriv(const double x, 
        double* val=0, double* deriv=0, double* deriv2=0) const
    {
        double Phi=0;
        coord::GradCyl grad;
        coord::HessCyl hess;
        if(mathutils::isFinite(x)) {
            if(mode == FIND_ZMAX)
                potential.eval(coord::PosCyl(R, x, phi), &Phi, deriv? &grad : NULL, deriv2? &hess: NULL);
            else
                potential.eval(coord::PosCyl(x, 0, phi), &Phi, deriv? &grad : NULL, deriv2? &hess: NULL);
        } else {
            if(deriv) 
                grad.dR = NAN;
            if(deriv2)
                hess.dR2 = NAN;
        }
        double result=Phi-E;
        if(mode == FIND_RMIN) {    // f(R) = -(1/2) v_R^2 * R^2
            result = result*x*x + Lz2/2;
            if(deriv) 
                *deriv = 2*x*(Phi-E) + x*x*grad.dR;
            if(deriv2)
                *deriv2 = 2*(Phi-E) + 4*x*grad.dR + x*x*hess.dR2;
        } else if(mode == FIND_RMAX) {  // f(R) = -(1/2) v_R^2 = Phi(R) - E + Lz^2/(2 R^2)
            if(Lz2>0)
                result += Lz2/(2*x*x);
            if(deriv)
                *deriv = grad.dR - (Lz2>0 ? Lz2/(x*x*x) : 0);
            if(deriv2)
                *deriv2 = hess.dR2 + (Lz2>0 ? 3*Lz2/(x*x*x*x) : 0);
        } else {  // FIND_ZMAX:   f(z) = -(1/2) v_z^2
            if(deriv)
                *deriv = grad.dz;
            if(deriv2)
                *deriv2= hess.dz2;
        }
        if(val)
            *val = result;
    }
};

bool estimateOrbitExtent(const potential::BasePotential& potential, const coord::PosVelCyl& point,
    double& Rmin, double& Rmax, double& zmaxRmin, double& zmaxRmax)
{
    const double toler = 1e-2;  // relative tolerance in root-finder, don't need high accuracy here
    double absz = fabs(point.z), absR = fabs(point.R);
    OrbitSizeFunction fnc(potential);
    fnc.Lz2 = pow_2(point.R*point.vphi);
    fnc.R   = absR;
    fnc.phi = point.phi;

    // examine the behavior of effective potential near R_0:
    // call the effective potential function to store the potential and its derivatives,
    // then compute the total energy and subtract it from the preliminary value of the function at R_0.
    // in this way we call the potential evaluation function only once.
    fnc.E   = 0;                                       // assign a temporary value
    mathutils::PointNeighborhood nh(fnc, absR);        // compute the potential and its derivatives at point
    double Phi_R_0 = nh.f0 - 0.5*pow_2(point.vphi);    // nh.f0 contained  Phi(R_0) + Lz^2 / (2 R_0^2)
    fnc.E   = nh.f0 + 0.5*pow_2(point.vR);             // compute the correct value of energy of in-plane motion (excluding v_z)
    nh.f0   = Phi_R_0 - fnc.E + 0.5*pow_2(point.vphi); // and store the correct value of Phi_eff(R_0) - E
    
    // estimate radial extent
    Rmin = Rmax = absR;
    double dR_to_zero = nh.dx_to_nearest_root();
    double maxPeri = absR, minApo = absR;  // endpoints of interval for locating peri/apocenter radii
    if(fabs(dR_to_zero) < absR*toler) {    // we are already near peri- or apocenter radius
        if(dR_to_zero > 0) {
            minApo  = NAN;
            maxPeri = absR + nh.dx_to_negative();
        } else {
            maxPeri = NAN;
            minApo  = absR + nh.dx_to_negative();
        }
    }
    if(fnc.Lz2>0) {
        if(mathutils::isFinite(maxPeri)) {
            fnc.mode = OrbitSizeFunction::FIND_RMIN;
            Rmin = mathutils::findRoot(fnc, 0., maxPeri, toler);
        }
    } else  // angular momentum is zero
        Rmin = 0;
    if(mathutils::isFinite(minApo)) {
        fnc.mode = OrbitSizeFunction::FIND_RMAX;
        Rmax = mathutils::findRoot(fnc, minApo, HUGE_VAL, toler);
    }   // else Rmax=absR

    if(!mathutils::isFinite(Rmin+Rmax))
        return false;  // likely reason: energy is positive

    // estimate vertical extent at R=R_0
    double zmax = absz;
    double Phi_R_z;  // potential at the initial position
    if(point.z != 0)
        potential.eval(point, &Phi_R_z);
    else
        Phi_R_z = Phi_R_0;
    fnc.E = Phi_R_z + pow_2(point.vz)/2;  // "vertical energy"
    if(point.vz != 0) {
        fnc.mode = OrbitSizeFunction::FIND_ZMAX;
        zmax = mathutils::findRoot(fnc, absz, HUGE_VAL, toler);
        if(!mathutils::isFinite(zmax))
            return false;
    }
    zmaxRmin=zmaxRmax=zmax;
    if(zmax>0 && absR>Rmin*1.2) {
        // a first-order correction for vertical extent
        fnc.E -= Phi_R_0;  // energy in vertical oscillation at R_0, equals to Phi(R_0,zmax)-Phi(R_0,0)
        double Phi_Rmin_0, Phi_Rmin_zmax;
        potential.eval(coord::PosCyl(Rmin, 0, point.phi), &Phi_Rmin_0);
        potential.eval(coord::PosCyl(Rmin, zmax, point.phi), &Phi_Rmin_zmax);
        // assuming that the potential varies quadratically with z, estimate corrected zmax at Rmin
        double corr=fnc.E/(Phi_Rmin_zmax-Phi_Rmin_0);
        if(corr>0.1 && corr<10)
            zmaxRmin = zmax*sqrt(corr);
    }
    if(zmax>0 && absR<Rmax*0.8) {
        // same at Rmax
        double Phi_Rmax_0, Phi_Rmax_zmax;
        potential.eval(coord::PosCyl(Rmax, 0, point.phi), &Phi_Rmax_0);
        potential.eval(coord::PosCyl(Rmax, zmax, point.phi), &Phi_Rmax_zmax);
        double corr = fnc.E/(Phi_Rmax_zmax-Phi_Rmax_0);
        if(corr>0.1 && corr<10)
            zmaxRmax = zmax*sqrt(corr);
    }
    return true;
}

double estimateInterfocalDistanceBox(const potential::BasePotential& potential, 
    double R1, double R2, double z1, double z2)
{
    if(z1+z2<=(R1+R2)*1e-8)   // orbit in x-y plane, any (non-zero) result will go
        return (R1+R2)/2;
    const int nR=4, nz=2, numpoints=nR*nz;
    double x[numpoints], y[numpoints];
    const double r1=sqrt(R1*R1+z1*z1), r2=sqrt(R2*R2+z2*z2);
    const double a1=atan2(z1, R1), a2=atan2(z2, R2);
    double sumsq=0;
    for(int iR=0; iR<nR; iR++) {
        double r=r1+(r2-r1)*iR/(nR-1);
        for(int iz=0; iz<nz; iz++) {
            const int ind=iR*nz+iz;
            double a=(iz+1.)/nz * (a1+(a2-a1)*iR/(nR-1));
            coord::GradCyl grad;
            coord::HessCyl hess;
            coord::PosCyl pos(r*cos(a), r*sin(a), 0);
            potential.eval(pos, NULL, &grad, &hess);
            x[ind] = hess.dRdz;
            y[ind] = 3*pos.z*grad.dR - 3*pos.R*grad.dz + pos.R*pos.z*(hess.dR2-hess.dz2)
                   + (pos.z*pos.z-pos.R*pos.R) * hess.dRdz;
            sumsq += pow_2(x[ind]);
        }
    }
    double coef = sumsq>0 ? mathutils::linearFitZero(numpoints, x, y) : 0;
    coef = fmax(coef, fmin(R1*R1+z1*z1,R2*R2+z2*z2)*0.0001);  // prevent it from going below or around zero
    return sqrt(coef);
}

double estimateInterfocalDistance(
    const potential::BasePotential& potential, const coord::PosVelCyl& point)
{
    double R1, R2, z1, z2;
    if(!estimateOrbitExtent(potential, point, R1, R2, z1, z2)) {
        R1=R2=point.R; z1=z2=point.z;
    }
    return estimateInterfocalDistanceBox(potential, R1, R2, z1, z2);
}

}  // namespace actions
