#include "actions_staeckel.h"
#include "mathutils.h"
#include <stdexcept>
#include <cmath>

int numeval_eoe=0, numeval_eidb=0, numeval_fil=0, numeval_ca=0, numeval_cid=0, numeval_cgfd=0, numeval_other=0;
int* numeval=&numeval_other;

namespace actions{

/** relative accuracy in integrals for computing actions and angles */
const double ACCURACY_ACTION = 0e-3;  // this is more than enough

/** relative tolerance in determining the range of variables (nu,lambda) to integrate over,
    also determined the minimum range that will be considered non-zero (relative to gamma-alpha) */
const double ACCURACY_RANGE = 1e-5;

/** parameters for the function that computes actions and angles
    by integrating an auxiliary function "fnc" as follows:
    the canonical momentum is   p^2(tau) = fnc(tau) / [2 (tau+alpha)^2 (tau+gamma)];
    the integrand is given by   p^n * (tau+alpha)^a * (tau+gamma)^c  if p^2>0, otherwise 0.
*/
struct AxisymIntegrandParam {
    mathutils::function fnc;                ///< auxilary function: either axisymStaeckelFnc or axisymFudgeFnc
    const AxisymData* data;                 ///< parameters of aux.fnc.: may be either AxisymStaeckelData or AxisymFudgeData
    enum { nplus1, nminus1 } n;             ///< power of p: +1 or -1
    enum { azero, aminus1, aminus2 } a;     ///< power of (tau+alpha): 0, -1, -2
    enum { czero, cminus1 } c;              ///< power of (tau+gamma): 0 or -1
};

// ------ SPECIALIZED functions for Staeckel action finder -------

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymStaeckelData findIntegralsOfMotionOblatePerfectEllipsoid
    (const potential::StaeckelOblatePerfectEllipsoid& poten, const coord::PosVelCyl& point)
{
    double E = potential::totalEnergy(poten, point);
    double Lz= coord::Lz(point);
    const coord::ProlSph& coordsys=poten.coordsys();
    const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
    double Glambda;
    poten.eval_simple(pprol.lambda, &Glambda);
    double I3;
    if(point.z==0)   // special case: nu=0
        I3 = 0.5 * pow_2(point.vz) * (pow_2(point.R)+coordsys.gamma-coordsys.alpha);
    else   // general case: eq.3 in Sanders(2012)
        I3 = fmax(0,
            (pprol.lambda+coordsys.gamma) * 
            (E - pow_2(Lz)/2/(pprol.lambda+coordsys.alpha) + Glambda) -
            pow_2(pprol.lambdadot*(pprol.lambda-pprol.nu)) / 
            (8*(pprol.lambda+coordsys.alpha)*(pprol.lambda+coordsys.gamma)) );
    return AxisymStaeckelData(pprol, mathutils::sign(point.z), E, Lz, I3, poten);
}

/** auxiliary function that enters the definition of canonical momentum for 
    for the Staeckel potential: it is the numerator of eq.50 in de Zeeuw(1985);
    the argument tau is replaced by tau+gamma >= 0. */
double axisymStaeckelFnc(double tauplusgamma, void* v_param)
{
    if(!mathutils::isFinite(tauplusgamma))
        return -1;  // used in the root-finder on an infinite interval
    AxisymStaeckelData* param=static_cast<AxisymStaeckelData*>(v_param);
    double G;
    param->fncG.eval_simple(tauplusgamma-param->point.coordsys.gamma, &G);
    const double tauplusalpha = tauplusgamma+param->point.coordsys.alpha-param->point.coordsys.gamma;
    return ( (param->E + G) * tauplusgamma - param->I3 ) * tauplusalpha
          - param->Lz*param->Lz/2 * tauplusgamma;
}

// -------- SPECIALIZED functions for the Axisymmetric Fudge action finder --------

/** compute true (E, Lz) and approximate (Ilambda, Inu) integrals of motion in an arbitrary 
    potential used for the Staeckel Fudge, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymFudgeData findIntegralsOfMotionAxisymFudge
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
    return AxisymFudgeData(pprol, mathutils::sign(point.z), E, Lz, Ilambda, Inu, poten);
}

/** Auxiliary function F analogous to that of Staeckel action finder:
    namely, the momentum is given by  p_tau^2 = F(tau) / (2*(tau+alpha)^2*(tau+gamma)),
    where  -gamma<=tau<=-alpha  for the  nu-component of momentum, 
    and   -alpha<=tau<infinity  for the  lambda-component of momentum.
    For numerical convenience, tau is replaced by x=tau+gamma.
*/
double axisymFudgeFnc(double tauplusgamma, void* v_data)
{
    if(!mathutils::isFinite(tauplusgamma))
        return -1;  // used in the root-finder on an infinite interval
    const AxisymFudgeData* data=static_cast<AxisymFudgeData*>(v_data);
    const double gamma=data->point.coordsys.gamma, alpha=data->point.coordsys.alpha;
    double lambda, nu, I, mult;
    if(tauplusgamma >= gamma-alpha) {  // evaluating J_lambda
        lambda= tauplusgamma-gamma;
        nu    = data->point.nu;
        mult  = lambda-nu;
        I     = data->Ilambda;
    } else {    // evaluating J_nu
        lambda= data->point.lambda;
        nu    = tauplusgamma-gamma;
        mult  = nu-lambda;
        I     = data->Inu;
    }
    double Phi;
    data->poten.eval(coord::toPosCyl(coord::PosProlSph(lambda, nu, data->point.phi, data->point.coordsys)), &Phi);
    const double tauplusalpha = tauplusgamma+alpha-gamma;
    return ( data->E * tauplusalpha - pow_2(data->Lz)/2 ) * tauplusgamma
         - (I + Phi * mult) * tauplusalpha;
}

// -------- COMMON routines for Staeckel and Fudge action finders --------

/** integrand for the expressions for actions and their derivatives 
    (e.g.Sanders 2012, eqs. A1, A4-A12).  It uses the auxiliary function to compute momentum,
    and multiplies it by some powers of (tau+alpha) and (tau+gamma).
*/
static double axisymIntegrand(double tauplusgamma, void* i_param)
{
    const AxisymIntegrandParam* param = static_cast<AxisymIntegrandParam*>(i_param);
    const coord::ProlSph& CS = param->data->point.coordsys;
    const double tauplusalpha = tauplusgamma+CS.alpha-CS.gamma;
    const double p2 = param->fnc(tauplusgamma, const_cast<AxisymData*>(param->data)) / 
        (2*pow_2(tauplusalpha)*tauplusgamma);
    if(p2<0) return 0;
    double result = sqrt(p2);
    if(param->n==AxisymIntegrandParam::nminus1)
        result = 1/result;
    if(param->a==AxisymIntegrandParam::aminus1)
        result /= tauplusalpha;
    else if(param->a==AxisymIntegrandParam::aminus2)
        result /= pow_2(tauplusalpha);
    if(param->c==AxisymIntegrandParam::cminus1)
        result /= tauplusgamma;
    if(!mathutils::isFinite(result))
        result=0;  // ad hoc fix to avoid problems at the boundaries of integration interval
    return result;
}

/** Compute the intervals of tau for which p^2(tau)>=0, 
    where  -gamma = tau_nu_min <= tau <= tau_nu_max <= -alpha  is the interval for the "nu" branch,
    and  -alpha <= tau_lambda_min <= tau <= tau_lambda_max < infinity  is the interval for "lambda".
    For numerical convenience, we replace tau with  x=tau+gamma.
*/
AxisymActionIntLimits findIntegrationLimitsAxisym(
    mathutils::function fnc, AxisymData& data)
{
    numeval=&numeval_fil;

    if(data.E>=0)
        throw std::invalid_argument("Error in Axisymmetric Staeckel/Fudge action finder: E>=0");
    AxisymActionIntLimits lim;
    const double gamma=data.point.coordsys.gamma, alpha=data.point.coordsys.alpha;
    const double toler=(gamma-alpha)*ACCURACY_RANGE;  // if an interval is smaller than this, discard it altogether

    // precautionary measures: need to find a point x_0, close to lambda+gamma, 
    // at which p^2>0 (or equivalently a_0>0); 
    // this condition may not hold at precisely x=lambda+gamma due to numerical inaccuracies
    double a_lambda, a_0;
    double x_0 = mathutils::findPositiveValue(fnc, &data, 
        gamma+data.point.lambda, &a_lambda, &a_0);
    double a_alpha = fnc(gamma-alpha, &data);
    double a_gamma = fnc(0, &data);

    lim.xnu_min = lim.xnu_max = lim.xlambda_min = lim.xlambda_max = 0;  // 0 means not set
    if(a_alpha==0) {  // special case: L_z==0, may have either tube or box orbit in the meridional plane
        double der = mathutils::deriv(fnc, &data, gamma-alpha, toler, 0);
        if(der>0) {   // box orbit 
            if(a_gamma>0)
                lim.xnu_max = mathutils::findRootGuess(fnc, &data, 
                    0, gamma-alpha, (gamma-alpha)/2, false, ACCURACY_RANGE);
            else 
                lim.xnu_max = lim.xnu_min;
            lim.xlambda_min = gamma-alpha;
        } else {      // tube orbit 
            lim.xnu_max = gamma-alpha;
            lim.xlambda_min = a_0>0 ? mathutils::findRootGuess(fnc, &data, 
                gamma-alpha, x_0, (gamma-alpha+x_0)/2, true, ACCURACY_RANGE) : 0;
        }
    }

    // find range for J_nu = J_z if it has not been determined at the previous stage
    if(a_gamma>0 && lim.xnu_max==0) {  // otherwise assume that motion is confined to x-y plane, and J_z=0
        lim.xnu_max = mathutils::findRoot(fnc, &data, data.point.nu+gamma, gamma-alpha, ACCURACY_RANGE);
        if(!mathutils::isFinite(lim.xnu_max) || lim.xnu_max<toler) 
            lim.xnu_max = lim.xnu_min;
    }
    if(lim.xnu_max==lim.xnu_min && data.point.nu+gamma>lim.xnu_max)
        data.point.nu=-gamma;

    // range for J_lambda = J_r
    if(a_0>0) {
        if(a_lambda==0) {
            if(x_0>gamma+data.point.lambda) {
                lim.xlambda_min = gamma+data.point.lambda;
            } else {
                lim.xlambda_max = gamma+data.point.lambda;
            }
        }
        if(lim.xlambda_min==0)  // not yet determined 
            lim.xlambda_min = mathutils::findRoot(fnc, &data, gamma-alpha, x_0, ACCURACY_RANGE);
        if(lim.xlambda_max==0)
            lim.xlambda_max = mathutils::findRoot(fnc, &data, x_0, HUGE_VAL, ACCURACY_RANGE);
        if(!mathutils::isFinite(lim.xlambda_min+lim.xlambda_max))
           a_0=0;
    }
    if(a_0<=0) {
        lim.xlambda_min = lim.xlambda_max = data.point.lambda+gamma;
    }
    return lim;
}

/** Compute the derivatives of integrals of motion (E, Lz, I3) over actions (Jr, Jz, Jphi),
    using the expressions A4-A9 in Sanders(2012).  These quantities are independent of angles,
    and in particular, the derivatives of energy w.r.t. the three actions are the frequencies. */
AxisymIntDerivatives computeIntDerivatives(mathutils::function fnc, 
    const AxisymData& data, const AxisymActionIntLimits& lim)
{
    numeval=&numeval_cid;

    AxisymIntegrandParam param;
    param.fnc = fnc;
    param.data = &data;
    param.n = AxisymIntegrandParam::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    param.a = AxisymIntegrandParam::aminus1;
    param.c = AxisymIntegrandParam::czero;
    double dJrdE = mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdE = mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);
    // derivatives w.r.t. I3
    param.a = AxisymIntegrandParam::aminus1;
    param.c = AxisymIntegrandParam::cminus1;
    double dJrdI3 = -mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdI3 = -mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);
    // derivatives w.r.t. Lz
    param.a = AxisymIntegrandParam::aminus2;
    param.c = AxisymIntegrandParam::czero;
    double dJrdLz = -data.Lz * mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / (4*M_PI);
    double dJzdLz = -data.Lz * mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / (2*M_PI);

    double signLz  = mathutils::sign(data.Lz);
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
AxisymGenFuncDerivatives computeGenFuncDerivatives(mathutils::function fnc, 
    const AxisymData& data, const AxisymActionIntLimits& lim)
{
    numeval=&numeval_cgfd;

    const double signldot = data.point.lambdadot>=0?+1:-1;
    const double signndot = data.point.nudot>=0?+1:-1;
    const double gamma=data.point.coordsys.gamma;
    AxisymGenFuncDerivatives der;
    AxisymIntegrandParam param;
    param.fnc = fnc;
    param.data = &data;
    param.n = AxisymIntegrandParam::nminus1;  // momentum goes into the denominator
    // derivatives w.r.t. E
    param.a = AxisymIntegrandParam::aminus1;
    param.c = AxisymIntegrandParam::czero;
    der.dSdE =
        signldot * mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xlambda_min, data.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xnu_min, data.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    // derivatives w.r.t. I3
    param.a = AxisymIntegrandParam::aminus1;
    param.c = AxisymIntegrandParam::cminus1;
    der.dSdI3 = 
        signldot * -mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xlambda_min, data.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * -mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xnu_min, data.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    // derivatives w.r.t. Lz
    param.a = AxisymIntegrandParam::aminus2;
    param.c = AxisymIntegrandParam::czero;
    der.dSdLz = data.point.phi +
        signldot * -data.Lz * mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xlambda_min, data.point.lambda+gamma, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / 4
      + signndot * -data.Lz * mathutils::integrateScaled(axisymIntegrand, &param, 
            lim.xnu_min, data.point.nu+gamma, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) / 4;
    return der;
}

/** Compute actions by integrating the momentum over the range of tau on which it is positive,
    separately for the "nu" and "lambda" branches (equation A1 in Sanders 2012). */
Actions computeActions(mathutils::function fnc, 
    const AxisymData& data, const AxisymActionIntLimits& lim)
{
    numeval=&numeval_ca;

    Actions acts;
    AxisymIntegrandParam param;
    param.fnc = fnc;
    param.data = &data;
    param.n = AxisymIntegrandParam::nplus1;  // momentum goes into the numerator
    param.a = AxisymIntegrandParam::azero;
    param.c = AxisymIntegrandParam::czero;
    acts.Jr = mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xlambda_min, lim.xlambda_max, lim.xlambda_min, lim.xlambda_max, ACCURACY_ACTION) / M_PI;
    acts.Jz = mathutils::integrateScaled(axisymIntegrand, &param, 
        lim.xnu_min, lim.xnu_max, lim.xnu_min, lim.xnu_max, ACCURACY_ACTION) * 2/M_PI;
    acts.Jphi = fabs(data.Lz);
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
ActionAngles computeActionAngles(mathutils::function fnc, 
    const AxisymData& data, const AxisymActionIntLimits& lim)
{
    Actions acts = computeActions(fnc, data, lim);
    AxisymIntDerivatives derI = computeIntDerivatives(fnc, data, lim);
    AxisymGenFuncDerivatives derS = computeGenFuncDerivatives(fnc, data, lim);
    bool addPiToThetaZ = ((data.signz<0)^(data.point.nudot<0)) && acts.Jz!=0;
    Angles angs = computeAngles(derI, derS, addPiToThetaZ);
    return ActionAngles(acts, angs);
}

// -------- THE DRIVER ROUTINES --------

Actions axisymStaeckelActions(const potential::StaeckelOblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    AxisymStaeckelData data = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymStaeckelFnc, data);
    return computeActions(axisymStaeckelFnc, data, lim);
}

ActionAngles axisymStaeckelActionAngles(const potential::StaeckelOblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point)
{
    AxisymStaeckelData data = findIntegralsOfMotionOblatePerfectEllipsoid(potential, point);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymStaeckelFnc, data);
    return computeActionAngles(axisymStaeckelFnc, data, lim);
}

Actions axisymFudgeActions(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if((potential.symmetry() & potential.ST_AXISYMMETRIC) != potential.ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(interfocalDistance==0)
        interfocalDistance = estimateInterfocalDistance(potential, point);
    const coord::ProlSph coordsys(-pow_2(interfocalDistance)-1., -1.);
    AxisymFudgeData data = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymFudgeFnc, data);
    return computeActions(axisymFudgeFnc, data, lim);
}

ActionAngles axisymFudgeActionAngles(const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, double interfocalDistance)
{
    if((potential.symmetry() & potential.ST_AXISYMMETRIC) != potential.ST_AXISYMMETRIC)
        throw std::invalid_argument("Fudge approximation only works for axisymmetric potentials");
    if(interfocalDistance==0)
        interfocalDistance = estimateInterfocalDistance(potential, point);
    const coord::ProlSph coordsys(-pow_2(interfocalDistance)-1., -1.);
    AxisymFudgeData data = findIntegralsOfMotionAxisymFudge(potential, point, coordsys);    
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymFudgeFnc, data);
    return computeActionAngles(axisymFudgeFnc, data, lim);
}

// ------ Estimation of interfocal distance -------

struct OrbitSizeParam {
    const potential::BasePotential& potential;
    double R;
    double phi;
    double E;
    double Lz2;
    explicit OrbitSizeParam(const potential::BasePotential& p) : potential(p) {};
};

static double findOrbitVerticalExtent(double z, void* v_param)
{
    double Phi=0;
    const OrbitSizeParam* param = static_cast<OrbitSizeParam*>(v_param);
    if(mathutils::isFinite(z))
        param->potential.eval(coord::PosCyl(param->R, z, param->phi), &Phi);
    return Phi-param->E;
}

static double findOrbitRadialExtentPeri(double R, void* v_param)
{
    double Phi;
    const OrbitSizeParam* param = static_cast<OrbitSizeParam*>(v_param);
    param->potential.eval(coord::PosCyl(R, 0, param->phi), &Phi);
    return (Phi-param->E)*R*R + param->Lz2/2;
}

static double findOrbitRadialExtentApo(double R, void* v_param)
{
    double Phi=0;
    const OrbitSizeParam* param = static_cast<OrbitSizeParam*>(v_param);
    if(mathutils::isFinite(R))   // otherwise we are at infinity, Phi=0 (presumably?)
        param->potential.eval(coord::PosCyl(R, 0, param->phi), &Phi);
    Phi -= param->E;
    if(param->Lz2>0)  // means that R cannot be zero
        Phi += param->Lz2/(2*R*R);
    return Phi;
}

bool estimateOrbitExtent(const potential::BasePotential& potential, const coord::PosVelCyl& point,
    double& Rmin, double& Rmax, double& zmaxRmin, double& zmaxRmax)
{
    numeval=&numeval_eoe;
    
    const double toler=1e-2;  // relative tolerance in root-finder, don't need high accuracy here
    double Phi_R_z, Phi_R_0;  // potential at the initial position, and at the same radius and z=0
    double absz=fabs(point.z), absR=fabs(point.R);
    OrbitSizeParam param(potential);
    param.Lz2=pow_2(point.R*point.vphi);
    param.R=absR;
    param.phi=point.phi;
    Rmin=Rmax=absR;

    // estimate radial extent
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(absR, 0, point.phi), &Phi_R_0, &grad, &hess);
    param.E = Phi_R_0 + (pow_2(point.vR)+pow_2(point.vphi))/2;
    // we seek the roots of the function f(R) = -(1/2)V_R^2 = Phi(R)-E+Lz^2/2/R^2
    // but we need to be careful if f(R_0) is close to zero
    double der = grad.dR - (point.vphi!=0 ? pow_2(point.vphi)/absR : 0);
    double deltaR = pow_2(point.vR)/2/der;  // rough estimate of the offset from R_0 to the root
    bool needPeri=true, needApo=true;
    double maxPeri=absR, minApo=absR;  // endpoints of interval for locating peri/apocenter radii
    // check if we are at (or near) either peri- or apocenter
    if(fabs(deltaR) < absR*toler || fabs(point.vR) < fabs(point.vphi)*toler) {
        double der2 = hess.dR2 + (point.vphi!=0 ? 3*pow_2(point.vphi/absR) : 0);
        double discr = der2>0 ? sqrt(pow_2(der) + der2*pow_2(point.vR)) : 0;
        double deltaR1 = -deltaR;  // offset to the other root
        if(der>=0) {
            needApo = false;  // we are already at apocenter
            deltaR1 = -(der+discr)/fmax(der2, 0);
        }
        if(der<=0) {
            needPeri= false; // we are at pericenter
            deltaR1 = (discr-der)/fmax(der2, 0);
        }
        if(needPeri) {
            maxPeri = fmax(absR+deltaR1*0.5, absR*(1-toler));  // 0.5 is a safety factor
            if(maxPeri > absR*(1-toler))
                needPeri = false;  // don't need it either (we are on a near-circular orbit)
        }
        if(needApo) {
            minApo = fmin(absR+deltaR1*0.5, absR*(1+toler));
            if(minApo < absR*(1+toler))
                needApo = false;
        }
    }
    if(param.Lz2>0) {
        if(needPeri)
            Rmin = mathutils::findRoot(findOrbitRadialExtentPeri, &param, 0., maxPeri, toler);
    } else  // angular momentum is zero
        Rmin = 0;
    if(needApo)  // else Rmax is already equal to R
        Rmax = mathutils::findRoot(findOrbitRadialExtentApo, &param, minApo, HUGE_VAL, toler);
    if(!mathutils::isFinite(Rmin+Rmax))
        return false;  // likely reason: energy is positive

    // estimate vertical extent at R=R_0
    double zmax=absz;
    if(point.z!=0)
        potential.eval(point, &Phi_R_z);
    else
        Phi_R_z=Phi_R_0;
    param.E = Phi_R_z + pow_2(point.vz)/2;  // "vertical energy"
    if(point.vz!=0) {
        zmax = mathutils::findRoot(findOrbitVerticalExtent, &param, absz, HUGE_VAL, toler);
        if(!mathutils::isFinite(zmax))
            return false;
    }
    zmaxRmin=zmaxRmax=zmax;
    if(zmax>0 && Rmin<Rmax*0.9) {
        // a first-order correction for vertical extent
        param.E -= Phi_R_0;  // energy in vertical oscillation at R_0, equals to Phi(R_0,zmax)-Phi(R_0,0)
        double Phi_Rmin_0, Phi_Rmin_zmax;
        potential.eval(coord::PosCyl(Rmin, 0, point.phi), &Phi_Rmin_0);
        potential.eval(coord::PosCyl(Rmin, zmax, point.phi), &Phi_Rmin_zmax);
        // assuming that the potential varies quadratically with z, estimate corrected zmax at Rmin
        double corr=param.E/(Phi_Rmin_zmax-Phi_Rmin_0);
        if(corr>0.1 && corr<10)
            zmaxRmin = zmax*sqrt(corr);
        // same at Rmax
        double Phi_Rmax_0, Phi_Rmax_zmax;
        potential.eval(coord::PosCyl(Rmax, 0, point.phi), &Phi_Rmax_0);
        potential.eval(coord::PosCyl(Rmax, zmax, point.phi), &Phi_Rmax_zmax);
        corr = param.E/(Phi_Rmax_zmax-Phi_Rmax_0);
        if(corr>0.1 && corr<10)
            zmaxRmax = zmax*sqrt(corr);
    }
    return true;
}

double estimateInterfocalDistanceBox(const potential::BasePotential& potential, 
    double R1, double R2, double z1, double z2)
{
    numeval = &numeval_eidb;

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
