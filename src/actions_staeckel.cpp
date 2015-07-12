#include "actions_staeckel.h"
#include "mathutils.h"
#include <stdexcept>
#include <cmath>

namespace actions{

/// \name ------ Data structures for both Axisymmetric Staeckel and Fudge action-angle finders ------
///@{ 

/** relative accuracy in integrals for computing actions and angles */
const double ACCURACY_ACTION=1e-3;  // this is more than enough

/** integration intervals for actions and angles: x=tau+gamma, and tau is lambda or nu 
    (shared between Staeckel and Fudge action finders). */
struct AxisymActionIntLimits {
    double xlambda_min, xlambda_max, xnu_min, xnu_max;
};

/** aggregate structure that contains the point in prolate spheroidal coordinates,
    integrals of motion, and reference to the potential,
    shared between Axisymmetric Staeckel  and Axisymmetric Fudge action finders;
    only the coordinates and the two classical integrals are in common between them.
*/
struct AxisymData {
    coord::PosVelProlSph point;             ///< position/derivative in prolate spheroidal coordinates
    const double signz;                     ///< sign of z coordinate (needed because it is lost in prolate spheroidal coords)
    const double E;                         ///< total energy
    const double Lz;                        ///< z-component of angular momentum
    AxisymData(const coord::PosVelProlSph& _point, double _signz, double _E, double _Lz) :
        point(_point), signz(_signz), E(_E), Lz(_Lz) {};
};

/** parameters of potential, integrals of motion, and prolate spheroidal coordinates 
    SPECIALIZED for the Axisymmetric Staeckel action finder */
struct AxisymStaeckelData: public AxisymData {
    const double I3;                        ///< third integral
    const coord::ISimpleFunction& fncG;     ///< single-variable function of a Staeckel potential
    AxisymStaeckelData(const coord::PosVelProlSph& _point, double _signz, double _E, double _Lz,
        double _I3, const coord::ISimpleFunction& _fncG) :
        AxisymData(_point, _signz, _E, _Lz), I3(_I3), fncG(_fncG) {};
};

/** parameters of potential, integrals of motion, and prolate spheroidal coordinates 
    SPECIALIZED for the Axisymmetric Fudge action finder */
struct AxisymFudgeData: public AxisymData {
    const double Ilambda, Inu;              ///< approximate integrals of motion for two quasi-separable directions
    const potential::BasePotential& poten;  ///< gravitational potential
    AxisymFudgeData(const coord::PosVelProlSph& _point, double _signz, double _E, double _Lz,
        double _Ilambda, double _Inu, const potential::BasePotential& _poten) :
        AxisymData(_point, _signz, _E, _Lz), Ilambda(_Ilambda), Inu(_Inu), poten(_poten) {};
};

/** parameters for the function that computes actions and angles
    by integrating an auxiliary function "fnc" as follows:
    the canonical momentum is   p^2(tau) = fnc(tau) / [2 (tau+alpha)^2 (tau+gamma)];
    the integrand is given by   p^n * (tau+alpha)^a * (tau+gamma)^c  if p^2>0, otherwise 0.
*/
struct AxisymIntegrandParam {
    mathutils::function fnc;                ///< auxilary function
    const AxisymData* data;                 ///< parameters of aux.fnc.
    enum { nplus1, nminus1 } n;             ///< power of p: +1 or -1
    enum { azero, aminus1, aminus2 } a;     ///< power of (tau+alpha): 0, -1, -2
    enum { czero, cminus1 } c;              ///< power of (tau+gamma): 0 or -1
};

/** Derivatives of integrals of motion over actions (do not depend on angles) */
struct AxisymIntDerivatives {
    double dEdJr, dEdJz, dEdJphi, dI3dJr, dI3dJz, dI3dJphi, dLzdJr, dLzdJz, dLzdJphi;
};

/** Derivatives of generating function over integrals of motion (depend on angles) */
struct AxisymGenFuncDerivatives {
    double dSdE, dSdI3, dSdLz;
};

///@}
/// \name ------ SPECIALIZED functions for Staeckel action finder -------
///@{

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system */
static AxisymStaeckelData findIntegralsOfMotionOblatePerfectEllipsoid
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
static double axisymStaeckelFnc(double tauplusgamma, void* v_param)
{
    AxisymStaeckelData* param=static_cast<AxisymStaeckelData*>(v_param);
    double G;
    param->fncG.eval_simple(tauplusgamma-param->point.coordsys.gamma, &G);
    const double tauplusalpha = tauplusgamma+param->point.coordsys.alpha-param->point.coordsys.gamma;
    return ( (param->E + G) * tauplusgamma - param->I3 ) * tauplusalpha
          - param->Lz*param->Lz/2 * tauplusgamma;
}

///@}
/// \name -------- SPECIALIZED functions for the Axisymmetric Fudge action finder --------
///@{

/** compute true (E, Lz) and approximate (Ilambda, Inu) integrals of motion in an arbitrary 
    potential used for the Staeckel Fudge, 
    together with the coordinates in its prolate spheroidal coordinate system */
static AxisymFudgeData findIntegralsOfMotionAxisymFudge
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
    if(pprol.nu+coordsys.gamma<=0)  // z==0
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
static double axisymFudgeFnc(double tauplusgamma, void* v_data)
{
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

///@}
/// \name -------- COMMON routines for Staeckel and Fudge action finders --------
///@{

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
    if(data.E>=0)
        throw std::invalid_argument("Error in Axisymmetric Staeckel/Fudge action finder: E>=0");
    AxisymActionIntLimits lim;
    const double gamma=data.point.coordsys.gamma, alpha=data.point.coordsys.alpha;
    const double tol=(gamma-alpha)*1e-10;  // if an interval is smaller than this, discard it altogether
    
    // precautionary measures: need to find a point x_0, close to lambda+gamma, 
    // at which p^2>0 (or equivalently a_0>0); 
    // this condition may not hold at precisely x=lambda+gamma due to numerical inaccuracies
    double a_0;
    double x_0 = mathutils::findPositiveValue(fnc, &data, 
        gamma+data.point.lambda, &a_0);
    double a_alpha = fnc(gamma-alpha, &data);
    double a_gamma = fnc(0, &data);

    lim.xnu_min = lim.xnu_max = lim.xlambda_min = lim.xlambda_max = 0;  // 0 means not set
    if(a_alpha==0) {  // special case: L_z==0, may have either tube or box orbit in the meridional plane
        double der = mathutils::deriv(fnc, &data, gamma-alpha, tol, 0);
        if(der>0) {   // box orbit 
            if(a_gamma>0)
                lim.xnu_max = mathutils::findRootGuess(fnc, &data, 
                    0, gamma-alpha, (gamma-alpha)/2, false);
            else 
                lim.xnu_max = lim.xnu_min;
            lim.xlambda_min = gamma-alpha;
        } else {      // tube orbit 
            lim.xnu_max = gamma-alpha;
            lim.xlambda_min = a_0>0 ? mathutils::findRootGuess(fnc, &data, 
                gamma-alpha, x_0, (gamma-alpha+x_0)/2, true) : 0;
        }
    }

    // find range for J_nu = J_z if it has not been determined at the previous stage
    if(a_gamma>0 && lim.xnu_max==0) {  // otherwise assume that motion is confined to x-y plane, and J_z=0
        lim.xnu_max = mathutils::findRoot(fnc, &data, 0, gamma-alpha);
        if(!mathutils::isFinite(lim.xnu_max) || lim.xnu_max<(gamma-alpha)*1e-10) 
            lim.xnu_max = lim.xnu_min;
    }
    if(lim.xnu_max==lim.xnu_min && data.point.nu>-gamma)
        data.point.nu=-gamma;
    
    // range for J_lambda = J_r
    if(a_0>0) {
        double a_lambda = fnc(gamma+data.point.lambda, &data);
        if(a_lambda==0) {
            if(x_0>gamma+data.point.lambda) {
                lim.xlambda_min = gamma+data.point.lambda;
            } else {
                lim.xlambda_max = gamma+data.point.lambda;
            }
        }
        if(lim.xlambda_min==0)  // not yet determined 
            lim.xlambda_min = mathutils::findRoot(fnc, &data, gamma-alpha, x_0);
        if(lim.xlambda_max==0)
            lim.xlambda_max = mathutils::findRootGuess(fnc, &data,
                lim.xlambda_min, HUGE_VAL, x_0, false);
    } else {
        lim.xlambda_min = lim.xlambda_max = data.point.lambda+gamma;
    }
    return lim;
}

/** Compute the derivatives of integrals of motion (E, Lz, I3) over actions (Jr, Jz, Jphi),
    using the expressions A4-A9 in Sanders(2012).  These quantities are independent of angles. */
AxisymIntDerivatives computeIntDerivatives(mathutils::function fnc, 
    const AxisymData& data, const AxisymActionIntLimits& lim)
{
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

///@}
/// \name -------- CLASS MEMBER FUNCTIONS --------
///@{

Actions ActionFinderAxisymmetricStaeckel::actions(const coord::PosVelCyl& point) const
{
    AxisymStaeckelData data = findIntegralsOfMotionOblatePerfectEllipsoid(poten, point);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymStaeckelFnc, data);
    return computeActions(axisymStaeckelFnc, data, lim);
}
    
ActionAngles ActionFinderAxisymmetricStaeckel::actionAngles(const coord::PosVelCyl& point) const
{
    AxisymStaeckelData data = findIntegralsOfMotionOblatePerfectEllipsoid(poten, point);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymStaeckelFnc, data);
    return computeActionAngles(axisymStaeckelFnc, data, lim);
}        

Actions ActionFinderAxisymmetricFudgeJS::actions(const coord::PosVelCyl& point) const
{
    const coord::ProlSph coordsys(alpha, gamma);
    AxisymFudgeData data = findIntegralsOfMotionAxisymFudge(poten, point, coordsys);
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymFudgeFnc, data);
    return computeActions(axisymFudgeFnc, data, lim);
}
    
ActionAngles ActionFinderAxisymmetricFudgeJS::actionAngles(const coord::PosVelCyl& point) const
{
    const coord::ProlSph coordsys(alpha, gamma);
    AxisymFudgeData data = findIntegralsOfMotionAxisymFudge(poten, point, coordsys);    
    AxisymActionIntLimits lim = findIntegrationLimitsAxisym(axisymFudgeFnc, data);
    return computeActionAngles(axisymFudgeFnc, data, lim);
}

///@}
}  // namespace actions
