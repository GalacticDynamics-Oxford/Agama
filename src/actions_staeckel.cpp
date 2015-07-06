#include "actions_staeckel.h"
#include <gsl/gsl_integration.h>
#include <stdexcept>
#include "GSLInterface.h"

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
    
    static double axisymStaeckelMomentumSq(double tau, void* v_param)
    {
        AxisymStaeckelParam* param=static_cast<AxisymStaeckelParam*>(v_param);
        double G;
        param->fncG.eval_simple(tau, &G);
        return (param->E
              - pow_2(param->Lz) / (2*(tau+param->coordsys.alpha))
              - param->I3 / (tau+param->coordsys.gamma)
              + G) / (2*(tau+param->coordsys.alpha));
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
        if(val<=0) return 0;
        return sqrt(val) * dx;
    }
    
    static double computeAction(ActionIntParam& param) {
        gsl_function F;
        F.function=&fncMomentum;
        F.params=&param;
        double result, error;
        size_t neval;
        gsl_integration_qng(&F, 0, 1, 0, ACCURACY_ACTION, &result, &error, &neval);
        return result;
    }
        
    /** compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid, 
        together with the coordinates in its prolate spheroidal coordinate system */
    AxisymStaeckelParam findIntegralsOfMotionOblatePerfectEllipsoid
        (const potential::StaeckelOblatePerfectEllipsoid& poten, const coord::PosVelCyl& point)
    {
        double E = potential::totalEnergy(poten, point);
        if(E>=0)
            throw std::runtime_error("Error in Axisymmetric Staeckel action finder: E>=0");
        double Lz= coord::Lz(point);
        const coord::ProlSph& coordsys=poten.coordsys();
        coord::PosDerivT<coord::Cyl, coord::ProlSph> derivs;
        const coord::PosProlSph pprol = coord::toPosDeriv<coord::Cyl, coord::ProlSph>
            (point, coordsys, &derivs);
        double lambdadot = derivs.dlambdadR*point.vR + derivs.dlambdadz*point.vz;
        double Glambda;
        poten.eval_simple(pprol.lambda, &Glambda);
        double I3 = (pprol.lambda+coordsys.gamma) * 
            (E - pow_2(Lz)/2/(pprol.lambda+coordsys.alpha) + Glambda) -
            pow_2(lambdadot*(pprol.lambda-pprol.nu)) / 
            (8*(pprol.lambda+coordsys.alpha)*(pprol.lambda+coordsys.gamma));
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
        GSLmath::root_find root_finder(0, 42);
        
        // Jz
        aiparam.xmin = -coordsys.gamma;
        aiparam.xmax = data.nu;
        while(axisymStaeckelMomentumSq(aiparam.xmax, &data)>0 && aiparam.xmax<-coordsys.alpha)
            aiparam.xmax = (aiparam.xmax-coordsys.alpha)/2;
        aiparam.xmax = root_finder.findroot(&axisymStaeckelMomentumSq, data.nu, aiparam.xmax, &data);
        acts.Jz = computeAction(aiparam) * 2/M_PI;
        
        // Jr
        aiparam.xmin = data.lambda;
        while(axisymStaeckelMomentumSq(aiparam.xmin, &data)>0 && aiparam.xmin>-coordsys.alpha)
            aiparam.xmin = (aiparam.xmin-coordsys.alpha)/2;
        aiparam.xmin = root_finder.findroot(&axisymStaeckelMomentumSq, aiparam.xmin, data.lambda, &data);
        aiparam.xmax = data.lambda;
        while(axisymStaeckelMomentumSq(aiparam.xmax, &data)>0 && isfinite(aiparam.xmax))
            aiparam.xmax = 2*aiparam.xmax-coordsys.alpha;
        aiparam.xmax = root_finder.findroot(&axisymStaeckelMomentumSq, data.lambda, aiparam.xmax, &data);
        acts.Jr = computeAction(aiparam) / M_PI;
        
        acts.Jphi = data.Lz;
        return acts;
    }

}  // namespace actions
