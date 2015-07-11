#pragma once
#include "potential_base.h"

namespace potential{

/** Axisymmetric Stackel potential with oblate perfect ellipsoidal density.
    Potential is computed in prolate spheroidal coordinate system
    through an auxiliary function  \f$  G(\tau)  \f$  as
    \f$  \Phi = -[ (\lambda+\gamma)G(\lambda) - (\nu+\gamma)G(\nu) ] / (\lambda-\nu)  \f$.
    The parameters of the internal prolate spheroidal coordinate system are 
    \f$  \alpha=-a^2, \gamma=-c^2  \f$, where a and c are the major and minor axes, 
    and the coordinates  \f$  \lambda, \nu  \f$  in this system satisfy
    \f$  -\gamma \le \nu \le -\alpha \le \lambda < \infty  \f$.
*/
class StaeckelOblatePerfectEllipsoid: public BasePotential, 
    public coord::IScalarFunction<coord::ProlSph>, public coord::ISimpleFunction {
public:
    StaeckelOblatePerfectEllipsoid(double _mass, double major_axis, double minor_axis);

    virtual SYMMETRYTYPE symmetry() const { return ST_AXISYMMETRIC; }

    const coord::ProlSph& coordsys() const { return coordSys; }
    
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "OblatePerfectEllipsoid"; };

    /** evaluates the function G(tau) and up to two its derivatives,
        if the supplied output arguments are not NULL 
        (implements the coord::ISimpleFunction interface) */
    virtual void eval_simple(double tau, double* G=0, double* Gderiv=0, double* Gderiv2=0) const;

private:
    const double mass;
    /** prolate spheroidal coordinate system corresponding to the oblate density profile */
    const coord::ProlSph coordSys;

    /** implementations of the standard triad of coordinate transformations */
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::eval_and_convert_twostep<coord::ProlSph, coord::Cyl, coord::Car>
            (*this, pos, coordSys, potential, deriv, deriv2);  // no direct conversion exists, use two-step
    }
    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::eval_and_convert<coord::ProlSph, coord::Cyl>
            (*this, pos, coordSys, potential, deriv, deriv2);
    }
    virtual void eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::eval_and_convert_twostep<coord::ProlSph, coord::Cyl, coord::Sph>
            (*this, pos, coordSys, potential, deriv, deriv2);  // use two-step conversion
    }

    /** the function that does the actual computation in prolate spheroidal coordinates 
        (implements the coord::IScalarFunction<ProlSph> interface) */
    virtual void eval_scalar(const coord::PosProlSph& pos,
        double* value=0, coord::GradProlSph* deriv=0, coord::HessProlSph* deriv2=0) const;
};

}  // namespace potential