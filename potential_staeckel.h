#pragma once
#include "potential_base.h"

namespace potential{

    /** Axisymmetric Stackel potential with oblate perfect ellipsoidal density.
        Potential is computed in prolate spheroidal coordinate system
        through an auxiliary function G(tau) as
        Phi = -[ (lambda+gamma)G(lambda) - (nu+gamma)G(nu) ] / (lambda-nu). */
    class StaeckelOblatePerfectEllipsoid: public BasePotential, coord::ScalarFunction<coord::ProlSph>{
    private:
        const double mass;
        /** prolate spheroidal coordinate system corresponding to the oblate density profile */
        const coord::ProlSph coordSys;

        /** implementations of the standard triad of coordinate transformations */
        virtual void eval_car(const coord::PosCar &pos,
            double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
            coord::eval_and_convert_twostep<coord::ProlSph, coord::Cyl, coord::Car>
                (*this, pos, coordSys, potential, deriv, deriv2);
        }
        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
            coord::eval_and_convert<coord::ProlSph, coord::Cyl>
                (*this, pos, coordSys, potential, deriv, deriv2);
        }
        virtual void eval_sph(const coord::PosSph &pos,
            double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
            coord::eval_and_convert_twostep<coord::ProlSph, coord::Cyl, coord::Sph>
                (*this, pos, coordSys, potential, deriv, deriv2);
        }

        /** the function that does the actual computation in prolate spheroidal coordinates  */
        virtual void evaluate(const coord::PosProlSph& pos,
            double* value=0, coord::GradProlSph* deriv=0, coord::HessProlSph* deriv2=0) const;

    public:
        StaeckelOblatePerfectEllipsoid(double _mass, double major_axis, double minor_axis);

        virtual SYMMETRYTYPE symmetry() const { return ST_AXISYMMETRIC; }

        const coord::ProlSph& coordsys() const { return coordSys; }

        /** evaluates the function G(tau) and up to two its derivatives,
            if the supplied output arguments are not NULL */
        double eval_G(double tau, double* deriv=0, double* deriv2=0) const;
    };

}  // namespace potential