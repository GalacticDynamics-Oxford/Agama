/** \file   math_specfunc.h
    \brief  various special functions
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"

namespace math{
/** Compute normalized associated Legendre polynomial (or, rather, function) of the first kind:
    \f$  P_l^m(\cos(\theta))  \f$.
    These functions are used in spherical-harmonic expansion as follows: 
    \f$  Y_l^m = P_l^m(\cos(\theta)) * \{\sin,\cos\}(m\phi)  \f$.
*/
double legendrePoly(const int l, const int m, const double theta);

/** Compute array of normalized associated Legendre polynomials for l=m..lmax.
    The output arrays contain values of P, dP/dtheta, d^2P/dtheta^2  for l=m,m+1,...,lmax;
    if either deriv_array or deriv2_array = NULL, the corresponding thing is not computed
    (note that if deriv2_array is not NULL, deriv_array must not be NULL too).
*/
void legendrePolyArray(const int lmax, const int m, const double theta,
    double* result_array, double* deriv_array=0, double* deriv2_array=0);

}  // namespace
