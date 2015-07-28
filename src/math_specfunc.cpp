#include "math_specfunc.h"
#include <cmath>
#include <cassert>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>

namespace math {

double legendrePoly(const int l, const int m, const double theta)
{
    return gsl_sf_legendre_sphPlm(l, m, cos(theta));
}

void legendrePolyArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array, double* deriv2_array)
{
    assert(result_array!=NULL);
    double costheta = cos(theta), sintheta;
    // compute unnormalized polynomials and then normalize manually, which is faster than computing normalized ones.
    // This is not suitable for large l,m (when overflow may occur), but in our application we aren't going to have such large values.
    if(deriv_array) {
        gsl_sf_legendre_Plm_deriv_array(lmax, m, costheta, result_array, deriv_array);
        sintheta = sin(theta);
    }
    else
        gsl_sf_legendre_Plm_array(lmax, m, costheta, result_array);
    double prefact = 0.5/sqrt(M_PI*gsl_sf_fact(2*m));
    for(int l=m; l<=lmax; l++) {
        double prefactl=sqrt(2*l+1.0)*prefact;
        result_array[l-m] *= prefactl;
        if(deriv_array)
            deriv_array[l-m] *= prefactl;
        prefact *= sqrt((l+1.0-m)/(l+1.0+m));
    }
    if(deriv2_array) {
        assert(deriv_array!=NULL);
        for(int l=m; l<=lmax; l++) {
            // accurate treatment of asymptotic values to avoid NaN
            if(m==0)
                deriv2_array[l-m] = costheta * deriv_array[l-m] - l*(l+1) * result_array[l-m];
            else if(costheta>=1-1e-6)
                deriv2_array[l-m] = deriv_array[l-m] * (costheta - 2*(l*(l+1)*(costheta-1)/m + m/(costheta+1)) );
            else if(costheta<=-1+1e-6)
                deriv2_array[l-m] = deriv_array[l-m] * (costheta - 2*(l*(l+1)*(costheta+1)/m + m/(costheta-1)) );
            else
                deriv2_array[l-m] = costheta * deriv_array[l-m] - (l*(l+1)-pow_2(m/sintheta)) * result_array[l-m];
        }
    }
    if(deriv_array) {
        for(int l=0; l<=lmax-m; l++)
            deriv_array[l] *= -sintheta;
    }
}
}  // namespace