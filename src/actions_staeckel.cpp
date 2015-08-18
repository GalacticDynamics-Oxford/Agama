#include "actions_staeckel.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

/** Accuracy of integrals for computing actions and angles
    is determined by the number of points in fixed-order Gauss-Legendre scheme */
const unsigned int INTEGR_ORDER = 10;  // good enough

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over */
const double ACCURACY_RANGE = 1e-6;

/** minimum range of variation of nu, lambda that is considered to be non-zero */
const double MINIMUM_RANGE = 1e-12;

// ------ SPECIALIZED functions for Staeckel action finder -------

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymFunctionStaeckel findIntegralsOfMotionOblatePerfectEllipsoid
    (const potential::OblatePerfectEllipsoid& poten, const coord::PosVelCyl& point)
{
    double E = totalEnergy(poten, point);
    double Lz= coord::Lz(point);
    const coord::ProlSph& coordsys = poten.coordsys();
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double Glambda;
    poten.evalDeriv(pprol.lambda, &Glambda);
    double I3;
    if(point.z==0)   // special case: nu=0
        I3 = 0.5 * pow_2(point.vz) * (pow_2(point.R) + coordsys.delta);
    else   // general case: eq.3 in Sanders(2012)
        I3 = fmax(0,
            pprol.lambda * 
            (E - pow_2(Lz) / 2 / (pprol.lambda - coordsys.delta) + Glambda) -
            pow_2( pprol.lambdadot * (pprol.lambda - fabs(pprol.nu)) ) / 
            (8 * (pprol.lambda - coordsys.delta) * pprol.lambda) );
    if(!math::isFinite(E+I3+Lz))
        throw std::invalid_argument("Error in Axisymmetric Staeckel action finder: "
            "cannot compute integrals of motion");
    return AxisymFunctionStaeckel(pprol, E, Lz, I3, poten);
}

/** auxiliary function that enters the definition of canonical momentum for 
    for the Staeckel potential: it is the numerator of eq.50 in de Zeeuw(1985);
    except that in our convention `tau` >= 0 is equivalent to `tau+gamma` from that paper. */
void AxisymFunctionStaeckel::evalDeriv(const double tau, 
    double* val, double* der, double* der2) const
{
    assert(tau>=0);
    double G, dG, d2G;
    fncG.evalDeriv(tau, &G, der? &dG : NULL, der2? &d2G : NULL);
    const double tauminusdelta = tau - point.coordsys.delta;
    if(val)
        *val = ( (E + G) * tau - I3 ) * tauminusdelta - Lz*Lz/2 * tau;
    if(der)
        *der = (E+G)*(tau+tauminusdelta) + dG*tau*tauminusdelta - I3 - Lz*Lz/2;
    if(der2)
        *der2 = 2*(E+G) + 2*dG*(tau+tauminusdelta) + d2G*tau*tauminusdelta;
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
    double E = Phi+(pow_2(point.vR)+pow_2(point.vz)+pow_2(point.vphi))/2;
    double Lz= coord::Lz(point);
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double absnu = fabs(pprol.nu);
    double Ilambda, Inu;
    Ilambda = pprol.lambda * (E - pow_2(Lz) / 2 / (pprol.lambda - coordsys.delta) )
            - (pprol.lambda - absnu) * Phi
            - pow_2( pprol.lambdadot * (pprol.lambda - absnu) ) /
              (8 * (pprol.lambda-coordsys.delta) * pprol.lambda);
    Inu     = absnu * (E - pow_2(Lz) / 2 / (absnu - coordsys.delta))
            + (pprol.lambda - absnu) * Phi;
    if(absnu <= 1e-12*coordsys.delta)  // z==0, nearly
        Inu+= pow_2(point.vz) * (pprol.lambda - absnu) / 2;
    else
        Inu+= pow_2(pprol.nudot * (pprol.lambda - absnu)) /
              (8 * (coordsys.delta - absnu) * absnu );
    if(!math::isFinite(E+Ilambda+Inu+Lz))
        throw std::invalid_argument("Error in Axisymmetric Fudge action finder: "
            "cannot compute integrals of motion");
    return AxisymFunctionFudge(pprol, E, Lz, Ilambda, Inu, poten);
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
    if(der2)
        *der2 = NAN;    // shouldn't be used
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
    coord::PosDerivT <coord::ProlSph, coord::Cyl> coordDeriv;
    const coord::PosProlSph posProl(lambda, nu, point.phi, point.coordsys);
    const coord::PosCyl posCyl = der? 
        coord::toPosDeriv<coord::ProlSph, coord::Cyl>(posProl, &coordDeriv) :
        coord::toPosCyl(posProl);
    double Phi;
    coord::GradCyl gradCyl;
    poten.eval(posCyl, &Phi, der? &gradCyl : NULL);
    const double tauminusdelta = tau - point.coordsys.delta;
    if(val)
        *val = ( E * tauminusdelta - pow_2(Lz)/2 ) * tau
             - (I + Phi * mult) * tauminusdelta;
    if(der) {
        coord::GradProlSph gradProl = coord::toGrad<coord::Cyl, coord::ProlSph> (gradCyl, coordDeriv);
        double dPhidtau = (tau >= point.coordsys.delta) ? gradProl.dlambda : gradProl.dnu;
        *der = E * (tau+tauminusdelta) - pow_2(Lz)/2 - I 
             - (mult+tauminusdelta) * Phi - tauminusdelta * mult * dPhidtau;
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
    const AxisymFunctionBase& fnc;          ///< parameters of aux.fnc.: may be either AxisymFunctionStaeckel or AxisymFunctionFudge
    enum { nplus1, nminus1 } n;             ///< power of p: +1 or -1
    enum { azero, aminus1, aminus2 } a;     ///< power of (tau-delta): 0, -1, -2
    enum { czero, cminus1 } c;              ///< power of tau: 0 or -1
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
        if(!math::isFinite(result))
            result=0;  // ad hoc fix to avoid problems at the boundaries of integration interval
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
        if(!math::isFinite(tau)) {
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
    if(fnc.E>=0)
        throw std::invalid_argument("Error in Axisymmetric Staeckel/Fudge action finder: E>=0");
    AxisymIntLimits lim;
    const double delta=fnc.point.coordsys.delta;

    // figure out the value of function at and around some important points
    double fnc_zero = fnc(0);
    const math::PointNeighborhood pn_delta (fnc, delta);
    assert(pn_delta.f0<=0);  // this is -0.5 Lz^2 so may not be positive
    const math::PointNeighborhood pn_lambda(fnc, fnc.point.lambda);
    lim.nu_min = 0;
    lim.nu_max = lim.lambda_min = lim.lambda_max = NAN;  // means not yet determined
    double nu_upper = delta;     // upper bound on the interval to locate the root for nu_max
    double lambda_lower = delta; // lower bound on the interval for lambda_min

    if(pn_delta.f0==0) {         // special case: L_z==0, may have either tube or box orbit in the meridional plane
        if(pn_delta.fder>0) {    // box orbit: f(delta)=0 and f'(delta)>0, so f<0 at some interval left of delta
            if(fnc_zero>0)       // there must be a range of nu where the function is positive
                nu_upper = fmax(delta*0.9, delta + pn_delta.dxToNegative());
            else 
                lim.nu_max = lim.nu_min;
            lim.lambda_min = delta;
        } else {      // tube orbit: f(delta)=0, f'(delta)<0, f must be negative on some interval right of delta
            lambda_lower = delta + fmin((fnc.point.lambda-delta)*0.1, pn_delta.dxToNegative());
            lim.nu_max = delta;
        }
    }

    if(!math::isFinite(lim.nu_max)) 
    {   // find range for J_nu = J_z if it has not been determined at the previous stage
        if(fnc_zero>0)
            lim.nu_max = math::findRoot(fnc, fabs(fnc.point.nu), nu_upper, ACCURACY_RANGE);
        if(!math::isFinite(lim.nu_max))       // means that the value f(nu) was just very slightly negative, or that f(0)<=0
            lim.nu_max = fabs(fnc.point.nu);  // i.e. this is a clear upper boundary of the range of allowed nu
    }

    // range for J_lambda = J_r
    if(pn_lambda.f0<=0) {   // it could be slightly smaller than zero due to roundoff errors
        if(pn_lambda.fder>=0) {
            lim.lambda_min = fnc.point.lambda;
        } 
        if(pn_lambda.fder<=0){
            lim.lambda_max = fnc.point.lambda;
        }
    }

    // due to roundoff errors, it may actually happen that f(lambda) is a very small negative number
    // in this case we need to estimate the value of lambda at which it is strictly positive (for root-finder)
    double lambda_pos = fnc.point.lambda + pn_lambda.dxToPositive();
    if(math::isFinite(lambda_pos)) {
        if(!math::isFinite(lim.lambda_min)) {  // not yet determined 
            lim.lambda_min = math::findRoot(fnc, lambda_lower, lambda_pos, ACCURACY_RANGE);
        }
        if(!math::isFinite(lim.lambda_max)) {
            lim.lambda_max = math::findRoot(AxisymScaledForRootfinder(fnc), 
                lambda_pos, INFINITY, ACCURACY_RANGE);
        }
    } else {  // can't find a value of lambda with positive p^2(lambda) -- dominated by roundoff errors
        lim.lambda_min = lim.lambda_max = fnc.point.lambda;
    }

    // sanity check
    if(!math::isFinite(lim.lambda_min+lim.lambda_max+lim.nu_max+lim.nu_min)
        || fabs(fnc.point.nu) > lim.nu_max
        || fnc.point.lambda < lim.lambda_min
        || fnc.point.lambda > lim.lambda_max)
        throw std::invalid_argument("findLimits: something wrong with the data");

    // ignore extremely small intervals
    if(lim.nu_max < delta * MINIMUM_RANGE)
        lim.nu_max = 0;
    if(lim.lambda_max-lim.lambda_min < delta * MINIMUM_RANGE)
        lim.lambda_min = lim.lambda_max = fnc.point.lambda;

    return lim;
}

/** Compute the derivatives of integrals of motion (E, Lz, I3) over actions (Jr, Jz, Jphi),
    using the expressions A4-A9 in Sanders(2012).  These quantities are independent of angles,
    and in particular, the derivatives of energy w.r.t. the three actions are the frequencies. */
AxisymIntDerivatives computeIntDerivatives(
    const AxisymFunctionBase& fnc, const AxisymIntLimits& lim)
{
    AxisymIntegrand integrand(fnc);
    math::ScaledIntegrandEndpointSing transf_l(integrand, lim.lambda_min, lim.lambda_max);
    math::ScaledIntegrandEndpointSing transf_n(integrand, lim.nu_min, lim.nu_max);
    integrand.n = AxisymIntegrand::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::czero;
    double dJrdE = math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) / (4*M_PI);
    double dJzdE = math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);
    // derivatives w.r.t. I3
    integrand.a = AxisymIntegrand::aminus1;
    integrand.c = AxisymIntegrand::cminus1;
    double dJrdI3 = -math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) / (4*M_PI);
    double dJzdI3 = -math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);
    // derivatives w.r.t. Lz
    integrand.a = AxisymIntegrand::aminus2;
    integrand.c = AxisymIntegrand::czero;
    double dJrdLz = -fnc.Lz * math::integrateGL(transf_l, 0, 1, INTEGR_ORDER) / (4*M_PI);
    double dJzdLz = -fnc.Lz * math::integrateGL(transf_n, 0, 1, INTEGR_ORDER) / (2*M_PI);

    AxisymIntDerivatives der;
    // invert the matrix of derivatives
    double det  = dJrdE*dJzdI3-dJrdI3*dJzdE;
    if(lim.nu_min==lim.nu_max || det==0) {
        // special case z==0: motion in z is irrelevant, but we could not compute dJzdI3 which is not zero
        der.Omegar   = 1/dJrdE;
        der.Omegaphi =-dJrdLz/dJrdE;
        der.dI3dJr   = der.dI3dJz = der.dI3dJphi = der.Omegaz = 0;
    } else {  // everything as normal
        der.Omegar   = dJzdI3/det;  // dE/dJr
        der.Omegaz   =-dJrdI3/det;  // dE/dJz
        der.Omegaphi = (dJrdI3*dJzdLz-dJrdLz*dJzdI3)/det;  // dE/dJphi
        der.dI3dJr   =-dJzdE/det;
        der.dI3dJz   = dJrdE/det;
        der.dI3dJphi =-(dJrdE*dJzdLz-dJrdLz*dJzdE)/det;
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
    to skip the computation of the derivatives matrix of integrals (not presently implemented). */
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

// -------- THE DRIVER ROUTINES --------

// a special case of spherical potential does not require any fudge
static Actions sphericalActions(const potential::BasePotential& potential,
    const coord::PosVelCyl& point)
{
    if((potential.symmetry() & potential::ST_SPHERICAL) != potential::ST_SPHERICAL)
        throw std::invalid_argument("This routine only can deal with actions in a spherical potential");
    Actions acts;
    double Ltot = Ltotal(point);
    acts.Jphi = Lz(point);
    acts.Jz = Ltot - fabs(acts.Jphi);
    double R1, R2;
    findPlanarOrbitExtent(potential, totalEnergy(potential, point), Ltot, R1, R2, &acts.Jr);
    return acts;
}

Actions axisymStaeckelActions(const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles axisymStaeckelActionAngles(const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point, Frequencies* freq)
{
    const AxisymFunctionStaeckel fnc = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim, freq);
}

Actions axisymFudgeActions(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if((potential.symmetry() & potential::ST_AXISYMMETRIC) != potential::ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if((potential.symmetry() & potential::ST_SPHERICAL) == potential::ST_SPHERICAL)
        return sphericalActions(potential, point);
    const coord::ProlSph coordsys(pow_2(interfocalDistance));
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActions(fnc, lim);
}

ActionAngles axisymFudgeActionAngles(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance, Frequencies* freq)
{
    if((potential.symmetry() & potential::ST_AXISYMMETRIC) != potential::ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    const coord::ProlSph coordsys(pow_2(interfocalDistance));
    const AxisymFunctionFudge fnc = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);    
    const AxisymIntLimits lim = findIntegrationLimitsAxisym(fnc);
    return computeActionAngles(fnc, lim, freq);
}

}  // namespace actions
