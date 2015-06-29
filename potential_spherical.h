#pragma once
#include "potential_base.h"

namespace potential{

    class PlummerPotential: public BasePotentialSphericallySymmetric{
    public:
        PlummerPotential(double _mass, double _scale_radius) :
            BasePotentialSphericallySymmetric(),
            mass(_mass),
            scale_radius(_scale_radius)
            {};
    private:
        double mass;
        double scale_radius;

        /** Evaluate potential and up to two its derivatives by spherical radius. */
        virtual void eval_sph_rad(const coord::PosSph &pos,
            double* potential, double* deriv, double* deriv2) const;

        /** Evaluate density at the position specified in spherical coordinates. */
    //    virtual double density_sph(const coord::PosSph &pos) const;

    };

}