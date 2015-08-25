#include "math_specfunc.h"
#include "math_spline.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_sf_gegenbauer.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_ellint.h>
#include <gsl/gsl_sf_bessel.h>

/* Most of the functions here are implemented by calling corresponding routines from GSL,
   but having library-independent wrappers makes it possible to switch the back-end if necessary */

namespace math {

double legendrePoly(const int l, const int m, const double theta) {
    return gsl_sf_legendre_sphPlm(l, m, cos(theta));
}

void legendrePolyArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array, double* deriv2_array)
{
    assert(result_array!=NULL);
    double costheta = cos(theta), sintheta=0;
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

double gegenbauer(const int n, double lambda, double x) {
    return gsl_sf_gegenpoly_n(n, lambda, x);
}

void gegenbauerArray(const int nmax, double lambda, double x, double* result_array) {
    gsl_sf_gegenpoly_array(nmax, lambda, x, result_array);
}

double hypergeom2F1(const double a, const double b, const double c, const double x)
{
    if (-1.<=x and x<1.)
        return gsl_sf_hyperg_2F1(a, b, c, x);
    // extension for 2F1 into the range x<-1 which is not provided by GSL; code from Heiko Bauke
    if (x<-1.) {
        if (c-a<0)
            return pow(1.-x, -a) * gsl_sf_hyperg_2F1(a, c-b, c, x/(x-1.));
        if (c-b<0)
            return pow(1.-x, -b) * gsl_sf_hyperg_2F1(c-a, c, c, x/(x-1.));
        // choose one of two equivalent formulas which is expected to be more accurate
        if (a*(c-b)<(c-a)*b)
            return pow(1.-x, -a) * gsl_sf_hyperg_2F1(a, c-b, c, x/(x-1.));
        else
            return pow(1.-x, -b) * gsl_sf_hyperg_2F1(c-a, b, c, x/(x-1.));
    }
    return NAN;  // not defined for x>=1
}

double factorial(const unsigned int n) {
    return gsl_sf_fact(n);
}

double lnfactorial(const unsigned int n) {
    return gsl_sf_lnfact(n);
}

double gamma(const double x) {
    return gsl_sf_gamma(x);
}

double lngamma(const double x) {
    return gsl_sf_lngamma(x);
}

double digamma(const double x) {
    return gsl_sf_psi(x);
}

double digamma(const int x) {
    return gsl_sf_psi_int(x);
}

double ellintK(const double k) {
    return gsl_sf_ellint_Kcomp(k, GSL_PREC_SINGLE);
}

double ellintE(const double k) {
    return gsl_sf_ellint_Ecomp(k, GSL_PREC_SINGLE);
}

double ellintF(const double phi, const double k) {
    return gsl_sf_ellint_F(phi, k, GSL_PREC_SINGLE);
}

double ellintE(const double phi, const double k) {
    return gsl_sf_ellint_E(phi, k, GSL_PREC_SINGLE);
}

double ellintP(const double phi, const double k, const double n) {
    return gsl_sf_ellint_P(phi, k, n, GSL_PREC_SINGLE);
}

double besselJ(const int n, const double x) {
    return gsl_sf_bessel_Jn(n, x);
}

double besselY(const int n, const double x) {
    return gsl_sf_bessel_Yn(n, x);
}

double besselI(const int n, const double x) {
    return gsl_sf_bessel_In(n, x);
}

double besselK(const int n, const double x) {
    return gsl_sf_bessel_Kn(n, x);
}

}  // namespace