/**
\file WD_Math.h
\brief Useful mathematic functions & values

*                                                                              *
* WDMath.h                                                                     *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1Keble Road, Oxford, OX1 3NP, United Kingdom.                       *
* e-mail:  dehnen@thphys.ox.ac.uk                                              *
*                                                                              *
********************************************************************************
*                                                                              *
* header file for some special mathematical functions.                         *
*                                                                              *
*******************************************************************************/

#ifndef _Math_def_
#define _Math_def_ 1

#include <cmath>

////////////////////////////////////////////////////////////////////////////////
// (1) constants
////////////////////////////////////////////////////////////////////////////////

namespace WD{

const double EulerGamma  = 0.577215664901532860606512090082;
const double LogofTwo    = 0.693147180559945309417232121458;
const double LogofTwoInv = 1.442695040888963407359924681002;
const double LogofTen    = 2.302585092994045684017991454684;
const double LogofTenInv = 0.434294481903251827651128918917;

////////////////////////////////////////////////////////////////////////////////
// (2) functions
////////////////////////////////////////////////////////////////////////////////

void   MathError        (const char*);  // writes out error message & exits
void   MathWarning      (const char*);  // writes out warning message

//==============================================================================
// miscellaneous
//==============================================================================

double SphVol(const int);       // volume of the unit sphere in n dimensions

//==============================================================================
// Log's and Exp's (inlines)
//==============================================================================

inline double ln(const double x)
        { if(x<=0) MathError("ln() of argument <= 0");
          return log(x); }
inline double ld(const double x)
        { if(x<=0) MathError("ld() of argument <= 0");
          return LogofTwoInv*log(x); }
inline double lg(const double x)
        { if(x<=0) MathError("lg() of argument <= 0");
          return log10(x); }
inline double Tento(const double x) { return exp(LogofTen*x); }
inline double Twoto(const double x) { return exp(LogofTwo*x); }

//==============================================================================
// logarithms of complex trigonometric and hyperbolic functions
//==============================================================================

#ifdef __COMPLEX__
complex<double> lnsin (const complex<double>&);
complex<double> lncos (const complex<double>&);
complex<double> lnsinh(const complex<double>&);
complex<double> lncosh(const complex<double>&);
#endif

//==============================================================================
// Gamma functions
//==============================================================================

double LogGamma(const double);
        // returns Log[Gamma[x]]
double GammaP(const double, const double);
        // returns P[a,x] = Int[Exp[-t] t^(a-1),{t,0,x}]
double LogGamma(const double, const double);
        // returns Log[Gamma[a,x]] = Log[ Gamma[x] * Q[a,x] ] 
        // where Q[a,x] = Int[Exp[-t] t^(a-1),{t,x,oo}]
double Loggamma(const double, const double);
        // returns Log[Gamma[a,x]] = Log[ Gamma[x] * P[a,x] ] 
        // where P[a,x] = Int[Exp[-t] t^(a-1),{t,0,x}]
#ifdef __COMPLEX__
complex<double> LogGamma(const complex<double>);
        // return Log[Gamma[x]]
#endif

//==============================================================================
// Exponential integrals
//==============================================================================

double En(const int, const double);
        // returns the exponential integral E_n(x)
double Ei(const double);
        // returns the exponential integral Ei(x)

/*
// Inverse error function
double InvErf(const double);
        // returns y such that erf(y)=x, not quite cheap
*/

//==============================================================================
// Bessel functions
//==============================================================================

double J0(const double);
        // return the Bessel function J_0(x)
double J1(const double);
        // return the Bessel function J_1(x)
double Jn(const int, const double);
        // return the Bessel function J_n(x)
double Y0(const double);
        // return the Bessel function Y_0(x) also known as N_0(x) [GR]
double Y1(const double);
        // return the Bessel function Y_1(x) also known as N_1(x) [GR]
double Yn(const int, const double);
        // return the Bessel function Y_n(x) also known as N_n(x) [GR]
double I0(const double);
        // return the modified Bessel function I_0(x)
double I1(const double);
        // return the modified Bessel function I_1(x)
double In(const int, const double);
        // return the modified Bessel function I_n(x)
double K0(const double);
        // return the modified Bessel function K_0(x)
double K1(const double);
        // return the modified Bessel function K_1(x)
double Kn(const int, const double);
        // return the modified Bessel function K_n(x)

//==============================================================================
// orthogonal polynomials
//==============================================================================

// Hermite polynomials

double HermiteH(const int, const double);
        // returns the nth Hermite polynomial
void HermiteH(const int, const double, double*);
        // evaluates the Hermite polynomials 0 to n
void NormSqHermite(const int, double*);
        // gives the inverse squared normalization constants for the H_n(x):
        // Int dx Exp[-x^2] H_n(x) H_m(x) = N_n delta_{nm}
double HermiteH_normalized(const int, const double);
        // returns the nth Hermite polynomial
        // normalized to be orthonormal w.r.t. the weight function exp(-x^2)
void HermiteH_normalized(const int, const double, double*);
        // evaluates the Hermite polynomials 0 to n
        // normalized to be orthonormal w.r.t. the weight function exp(-x^2)

} // namespace
#endif
