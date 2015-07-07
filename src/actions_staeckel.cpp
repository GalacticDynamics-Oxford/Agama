#include "actions_staeckel.h"
#include "mathutils.h"
#include <stdexcept>
#include <cmath>

namespace actions{
    
const double ACCURACY_ACTION=1e-6;

struct AxisymStaeckelParam {
    const coord::ProlSph& coordsys;
    const coord::ISimpleFunction& fncG;
    double E;     ///< total energy
    double Lz;    ///< z-component of angular momentum
    double I3;    ///< third integral
    double lambda, nu;  ///< coordinates in prolate spheroidal coord.sys.
    AxisymStaeckelParam(const coord::ProlSph& cs, const coord::ISimpleFunction& G,
        double _E, double _Lz, double _I3, double _lambda, double _nu) :
        coordsys(cs), fncG(G), E(_E), Lz(_Lz), I3(_I3), lambda(_lambda), nu(_nu) {};
};

static double axisymStaeckelMomentumSq(double tauplusgamma, void* v_param)
{
    AxisymStaeckelParam* param=static_cast<AxisymStaeckelParam*>(v_param);
    double G;
    param->fncG.eval_simple(tauplusgamma-param->coordsys.gamma, &G);
    double tauplusalpha = tauplusgamma+param->coordsys.alpha-param->coordsys.gamma;
    return (param->E
          - param->Lz*param->Lz / (2*tauplusalpha)
          - param->I3 / tauplusgamma
          + G) / (2*tauplusalpha);
}

struct ActionIntParam {
    double(*fncMomentumSq)(double,void*);
    void* param;
    double xmin, xmax;
};

static double fncMomentum(double y, void* aiparam) {
    ActionIntParam* param=static_cast<ActionIntParam*>(aiparam);
    const double x = param->xmin + (param->xmax-param->xmin) * y*y*(3-2*y);
    const double dx = (param->xmax-param->xmin) * 6*y*(1-y);
    double val=(*(param->fncMomentumSq))(x, param->param);
    if(val<=0 || !mathutils::is_finite(val)) return 0;
    return sqrt(val) * dx;
}

/** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
    together with the coordinates in its prolate spheroidal coordinate system */
AxisymStaeckelParam findIntegralsOfMotionOblatePerfectEllipsoid
    (const potential::StaeckelOblatePerfectEllipsoid& poten, const coord::PosVelCyl& point)
{
    double E = potential::totalEnergy(poten, point);
    if(E>=0)
        throw std::invalid_argument("Error in Axisymmetric Staeckel action finder: E>=0");
    double Lz= coord::Lz(point);
    const coord::ProlSph& coordsys=poten.coordsys();
    coord::PosDerivT<coord::Cyl, coord::ProlSph> derivs;
    const coord::PosProlSph pprol = coord::toPosDeriv<coord::Cyl, coord::ProlSph>
        (point, coordsys, &derivs);
    double lambdadot = derivs.dlambdadR*point.vR + derivs.dlambdadz*point.vz;
    double Glambda;
    poten.eval_simple(pprol.lambda, &Glambda);
    double I3 = point.z==0 ? 
        0.5 * pow_2(point.vz) * (pow_2(point.R)+coordsys.gamma-coordsys.alpha) : // special case nu=0
        (pprol.lambda+coordsys.gamma) * 
        (E - pow_2(Lz)/2/(pprol.lambda+coordsys.alpha) + Glambda) -
        pow_2(lambdadot*(pprol.lambda-pprol.nu)) / 
        (8*(pprol.lambda+coordsys.alpha)*(pprol.lambda+coordsys.gamma));
    I3=fmax(I3, 0);
    return AxisymStaeckelParam(coordsys, poten, E, Lz, I3, pprol.lambda, pprol.nu);
}

Actions ActionFinderAxisymmetricStaeckel::actions(const coord::PosVelCar& point) const
{
    AxisymStaeckelParam data = findIntegralsOfMotionOblatePerfectEllipsoid(
        poten, coord::toPosVelCyl(point));
    
    Actions acts;
    ActionIntParam aiparam;
    const coord::ProlSph& coordsys=poten.coordsys();
    aiparam.fncMomentumSq = &axisymStaeckelMomentumSq;
    aiparam.param = &data;
    
    // to find the actions, we integrate p(tau) over tau in two different intervals (for Jz and for Jr);
    // to avoid roundoff errors when tau is close to -gamma we replace tau with x=tau+gamma>=0
    double gminusa = coordsys.gamma-coordsys.alpha;
    
    // Jz:  0 <= x <= xmax < -alpha+gamma
    if(data.I3>0) {
        aiparam.xmin = 0;
        double guess = fmax(data.nu+coordsys.gamma, gminusa*1e-3);
        aiparam.xmax = mathutils::findroot_guess(&axisymStaeckelMomentumSq, &data, 
            0, gminusa, guess, false);
        acts.Jz = mathutils::integrate(fncMomentum, &aiparam, 0, 1, ACCURACY_ACTION) * 2/M_PI;
    } else 
        acts.Jz=0;
    
    // Jr:  -alpha+gamma < xmin <= x <= xmax < infinity
    aiparam.xmin = mathutils::findroot_guess(&axisymStaeckelMomentumSq, &data, 
        gminusa, data.lambda+coordsys.gamma, data.lambda+coordsys.gamma, true);
    aiparam.xmax = mathutils::findroot_guess(&axisymStaeckelMomentumSq, &data, 
        data.lambda+coordsys.gamma, HUGE_VAL, data.lambda+coordsys.gamma, false);
    acts.Jr = mathutils::integrate(fncMomentum, &aiparam, 0, 1, ACCURACY_ACTION) / M_PI;
    
    // Jphi:  simply Lz
    acts.Jphi = data.Lz;
    return acts;
}

}  // namespace actions
