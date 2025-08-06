/** \file   math_specfunc.h
    \brief  various special functions
    \date   2015-2016
    \author Eugene Vasiliev
*/
#pragma once

// most of the functions here return NAN in case of invalid arguments or out-of-range results
namespace math {

/** Error function */
double erf(const double x);

/** Inverse error function (defined for -1<x<1) */
double erfinv(const double x);

/** Gauss's hypergeometric function 2F1(a, b; c; x) */
double hypergeom2F1(const double a, const double b, const double c, const double x);

/** Associate Legendre function of the second kind, together with its derivative if necessary */
double legendreQ(const double m, const double x, double* deriv=0/*NULL*/);

/** Factorial of an integer number */
double factorial(const unsigned int n);

/** Logarithm of factorial of an integer number (doesn't overflow quite that easy) */
double lnfactorial(const unsigned int n);

/** Double-factorial n!! of an integer number */
double dfactorial(const unsigned int n);

/** Gamma function */
double gamma(const double x);

/** Logarithm of gamma function (doesn't overflow quite that easy) */
double lngamma(const double x);

/** Upper incomplete Gamma function: the ordinary Gamma(x) = gammainc(x, y=0) */
double gammainc(const double x, const double y);

/** Psi (digamma) function */
double digamma(const double x);

/** Psi (digamma) function for integer argument */
double digamma(const int x);

/** Complete elliptic integrals of the first kind K(k) = F(pi/2, k) */
double ellintK(const double k);

/** Complete elliptic integrals of the second kind E(k) = E(pi/2, k) */
double ellintE(const double k);

/** Incomplete elliptic integrals of the first kind:
    \f$  F(\phi,k) = \int_0^\phi d t \, \frac{1}{\sqrt{1-k^2\sin^2 t}}  \f$  */
double ellintF(const double phi, const double k);

/** Incomplete elliptic integrals of the second kind:
    \f$  E(\phi,k) = \int_0^\phi d t \, \sqrt{1-k^2\sin^2 t}  \f$  */
double ellintE(const double phi, const double k);

/** Incomplete elliptic integrals of the third kind:
    \f$  \Pi(\phi,k,n) = \int_0^\phi d t \, \frac{1}{(1+n\sin^2 t)\sqrt{1-k^2\sin^2 t}}  \f$  */
double ellintP(const double phi, const double k, const double n);

/** Bessel J_n(x) function (regular at x=0) */
double besselJ(const int n, const double x);

/** Bessel Y_n(x) function (singular at x=0) */
double besselY(const int n, const double x);

/** Modified Bessel function I_n(x) (regular) */
double besselI(const int n, const double x);

/** Modified Bessel function K_n(x) (singular) */
double besselK(const int n, const double x);

/** Lambert W function W_0,W_{-1}(x), satisfying the equation W exp(W) = x;
    for -exp(-1)<=x<0 there are two branches, W_0>=-1, and W_{-1}<=-1;
    the second one is selected by setting the second argument to true */
double lambertW(const double x, bool Wminus1branch=false);

/** The "generalized exponential" function (1 + q*x)^(1/q), which tends to exp(x) in the limit q->0 */
double qexp(const double x, const double q=0);

/** solve the Kepler equation:  phase = eta - ecc * sin(eta)  for eta (eccentric anomaly) */
double solveKepler(double ecc, double phase);

}  // namespace
