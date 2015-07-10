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
    double derivR_over_R = deriv.dR/pos.R;
    double deriv2phi_over_R2 = deriv2.dphi2/pow_2(pos.R);
    if(pos.R==0) {
        if(deriv.dR==0)  // otherwise should remain infinite
            derivR_over_R = deriv2.dR2;
        deriv2phi_over_R2 = 0;
    }
    return (deriv2.dR2 + derivR_over_R + deriv2.dz2 + deriv2phi_over_R2) / (4*M_PI);
}

double BasePotential::density_sph(const coord::PosSph &pos) const
{
    coord::GradSph deriv;
    coord::HessSph deriv2;
    eval(pos, NULL, &deriv, &deriv2);
    double sintheta=sin(pos.theta);
    double derivr_over_r = deriv.dr/pos.r;
    double derivtheta_cottheta = deriv.dtheta*cos(pos.theta)/sintheta;
    if(sintheta==0)
        derivtheta_cottheta = deriv2.dtheta2;
    double angular_part = (deriv2.dtheta2 + derivtheta_cottheta + 
        deriv2.dphi2/pow_2(sintheta))/pow_2(pos.r);
    if(pos.r==0) {
        if(derivr_over_r==0)  // otherwise should remain infinite
            derivr_over_r = deriv2.dr2;
        angular_part=0; ///??? is this correct assumption?
    }
    return (deriv2.dr2 + 2*derivr_over_r + angular_part) / (4*M_PI);
}


// convenience function
double v_circ(const BasePotential& potential, double radius)
{
    if((potential.symmetry() & BasePotential::ST_ZROTSYM) != BasePotential::ST_ZROTSYM)
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular velocity is possible");
    coord::GradCyl deriv;
    potential.eval(coord::PosCyl(radius, 0, 0), NULL, &deriv);
    return sqrt(radius*deriv.dR);
}

}  // namespace potential
