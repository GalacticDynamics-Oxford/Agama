/** \file    potential_perfect_ellipsoid.h
    \brief   Potential for Oblate Perfect Ellipsoid model
    \author  Eugene Vasiliev
    \date    2015
*/
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
class OblatePerfectEllipsoid: public BasePotential, 
    public coord::IScalarFunction<coord::ProlSph>, public math::IFunction {
public:
    OblatePerfectEllipsoid(double _mass, double major_axis, double minor_axis);

    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }

    const coord::ProlSph& coordsys() const { return coordSys; }

    virtual std::string name() const { return myName(); }
    static std::string myName() { return "PerfectEllipsoid"; }
    virtual double totalMass() const { return mass; }

    /** evaluates the function G(tau) and up to two its derivatives,
        if the supplied output arguments are not NULL 
        (implements the math::IFunction interface) */
    virtual void evalDeriv(double tau, double* G=NULL, double* Gderiv=NULL, double* Gderiv2=NULL) const;

private:
    const double mass;
    /** prolate spheroidal coordinate system corresponding to the oblate density profile */
    const coord::ProlSph coordSys;
    const double minorAxis;

    /** implementations of the standard triad of coordinate transformations */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const {
        // no direct conversion exists, use two-step
        coord::evalAndConvertTwoStep<coord::ProlSph, coord::Cyl, coord::Car>
            (*this, pos, coordSys, potential, deriv, deriv2, time);
    }
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const {
        coord::evalAndConvert<coord::ProlSph, coord::Cyl>
            (*this, pos, coordSys, potential, deriv, deriv2, time);
    }
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const {
        coord::evalAndConvertTwoStep<coord::ProlSph, coord::Cyl, coord::Sph>
            (*this, pos, coordSys, potential, deriv, deriv2, time);  // use two-step conversion
    }

    /** the function that does the actual computation in prolate spheroidal coordinates 
        (implements the coord::IScalarFunction<ProlSph> interface) */
    virtual void evalScalar(const coord::PosProlSph& pos,
        double* value=NULL, coord::GradProlSph* deriv=NULL, coord::HessProlSph* deriv2=NULL,
        double time=0) const;

    virtual unsigned int numDerivs() const { return 2; }
};

}  // namespace potential
