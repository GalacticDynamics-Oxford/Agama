#include "potential_base.h"
#include <cmath>
#include <stdexcept>

namespace potential{

    double BasePotential::density_car(const coord::PosCar &pos) const
    {
        coord::HessCar deriv2;
        eval(pos, NULL, (coord::GradCar*)NULL, &deriv2);
        return (deriv2.dx2 + deriv2.dy2 + deriv2.dz2) / (4*M_PI);
    }

    double BasePotential::density_cyl(const coord::PosCyl &pos) const
    {
        coord::GradCyl deriv;
        coord::HessCyl deriv2;
        eval(pos, NULL, &deriv, &deriv2);
        if(pos.R==0)
            throw std::runtime_error("density_cyl: R=0 not implemented");
        return (deriv2.dR2 + deriv.dR/pos.R + deriv2.dz2 + deriv2.dphi2/pow_2(pos.R)) / (4*M_PI);
    }

    double BasePotential::density_sph(const coord::PosSph &pos) const
    {
        coord::GradSph deriv;
        coord::HessSph deriv2;
        eval(pos, NULL, &deriv, &deriv2);
        double sintheta=sin(pos.theta), cottheta=cos(pos.theta)/sintheta;
        if(pos.r==0 || sintheta==0)
            throw std::runtime_error("density_cyl: r=0 or theta=0 not implemented");
        return (deriv2.dr2 + deriv.dr*2/pos.r +
            (deriv2.dtheta2 + deriv.dtheta*cottheta + deriv2.dphi2/pow_2(sintheta))/pow_2(pos.r) ) / (4*M_PI);
    }


    void BasePotentialSphericallySymmetric::eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const
    {
        double der, der2;
        eval_sph_rad(pos, potential, deriv!=NULL ? &der : NULL, deriv2!=NULL ? &der2 : NULL);  // implemented in derived classes
        if(deriv) {
            deriv->dr = der;
            deriv->dtheta = deriv->dphi = 0;
        }
        if(deriv2) {
            deriv2->dr2 = der2;
            deriv2->dtheta2 = deriv2->dphi2 = deriv2->drdtheta = deriv2->drdphi = deriv2->dthetadphi = 0;
        }
    }

    // convenience function
    double v_circ(const BasePotential& potential, double radius)
    {
        if(potential.symmetry() & BasePotential::ST_ZROTSYM != BasePotential::ST_ZROTSYM)
            throw std::invalid_argument("Potential is not axisymmetric, "
                "no meaningful definition of circular velocity is possible");
        coord::GradCyl deriv;
        potential.eval(coord::PosCyl(radius, 0, 0), NULL, &deriv);
        return sqrt(radius*deriv.dR);
    }

}  // namespace potential
