#include "potential_spherical.h"
#include <cmath>

namespace potential{

    void PlummerPotential::eval_sph_rad(const coord::PosSph &pos,
        double* potential, double* deriv, double* deriv2) const
    {
        double rsq = pow_2(pos.r) + pow_2(scale_radius);
        double pot = -mass/sqrt(rsq);
        if(potential)
            *potential = pot;
        if(deriv)
            *deriv = -pot*pos.r/rsq;
        if(deriv2)
            *deriv2 = pot*(2*pow_2(pos.r)-pow_2(scale_radius))/pow_2(rsq);
    }

//    double PlummerPotential::density_sph(const coord::PosSph &pos) const
//    {
//        return 3./(4*M_PI)*mass/pow_3(scale_radius)*pow(1+pow_2(pos.r/scale_radius), -2.5); 
//    }

}  // namespace potential