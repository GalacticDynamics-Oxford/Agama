/** \file   math_specfunc.h
    \brief  various special functions
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"

namespace math{
/** Normalized associated Legendre polynomial (or, rather, function) of the first kind:
    \f$  P_l^m(\cos(\theta))  \f$.
    These functions are used in spherical-harmonic expansion as follows: 
    \f$  Y_l^m = P_l^m(\cos(\theta)) * \{\sin,\cos\}(m\phi)  \f$.
*/
double legendrePoly(const int l, const int m, const double theta);

/** Array of normalized associated Legendre polynomials for l=m..lmax.
    The output arrays contain values of P, dP/dtheta, d^2P/dtheta^2  for l=m,m+1,...,lmax;
    if either deriv_array or deriv2_array = NULL, the corresponding thing is not computed
    (note that if deriv2_array is not NULL, deriv_array must not be NULL too).
*/
void legendrePolyArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array=0, double* deriv2_array=0);

/** Gauss's hypergeometric function 2F1(a, b; c; x) */
double hypergeom2F1(const double a, const double b, const double c, const double x);

/** Factorial of an integer number */
double factorial(const unsigned int n);

/** Gamma function */
double gamma(const double x);

/** Psi (digamma) function */
double digamma(const double x);
double digamma(const int x);

}  // namespace
