#pragma once
#include "potential_base.h"

namespace potential{

    /** Axisymmetric Stackel potential with oblate perfect ellipsoidal density
        (adapted from Jason Sanders).
        Potential is expressed through an auxiliary function G(tau) as
        Phi = -[ (lambda+gamma)G(lambda) - (nu+gamma)G(nu) ] / (lambda-nu). */
    class StaeckelOblatePerfectEllipsoid: public BasePotentialCyl{
    private:
        const double mass;
        /** prolate spheroidal coordinate system corresponding to the oblate density profile */
        const coord::ProlSph CS;

        virtual void eval_cyl(const coord::PosCyl &pos,
            double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;

    public:
        StaeckelOblatePerfectEllipsoid(double _mass, double major_axis, double minor_axis);

        virtual SYMMETRYTYPE symmetry() const { return ST_AXISYMMETRIC; }

        const coord::ProlSph& coordsys() const { return CS; }

        /** evaluates the function G(tau) and up to two its derivatives,
            if the supplied output arguments are not NULL */
        double eval_G(double tau, double* deriv=0, double* deriv2=0) const;
    };

}  // namespace potential