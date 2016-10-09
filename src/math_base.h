/** \file    math_base.h
    \brief   Base class for mathematical functions
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once

/// convenience function for squaring a number, used in many places
inline double pow_2(double x) { return x*x; }
inline float pow_2(float x) { return x*x; }
inline int pow_2(int x) { return x*x; }
inline unsigned int pow_2(unsigned int x) { return x*x; }
/// convenience function for raising a number to the 3rd power
inline double pow_3(double x) { return x*x*x; }

/// test if a number is neither infinity nor NaN
inline bool isFinite(double x) { return x==x && 1/x!=0; }

// some useful numbers (or even not-a-numbers)
#ifndef NULL
#define NULL 0
#endif

#ifndef INFINITY
#define INFINITY 1e10000
#endif

#ifndef NAN
#define NAN (INFINITY/INFINITY)
#endif

#ifndef M_PI
#define M_PI     3.14159265358979323846264338328
#endif

#ifndef M_SQRTPI
#define M_SQRTPI 1.77245385090551602729816748334
#endif

#ifndef M_SQRT2
#define M_SQRT2  1.41421356237309504880168872421
#endif

#define TWO_PI_CUBE 248.050213442398561403810520537

/** Functions and classes for basic and advanced math operations */
namespace math{

/** Prototype of a function of one variable that may provide up to two derivatives.
    This interface serves as the base for many mathematical routines throughout the code,
    some of them need only the value of the function, and some may need derivatives.
    However, not all classes implementing this interface are obliged to compute derivatives;
    the actual number of derivatives that can be produces is returned by a dedicated method 
    `numDerivs()`, which should be queried beforehand, and if necessary, the derivatives 
    must be estimated by the calling code itself from finite differences.
    Descendant classes should implement `evalDeriv()` and `numDerivs()` methods, 
    while `value()` and the `operator()` are simple shorthands provided by this base class.

    Note on the naming conventions here and throughout the code:
    `value(x)` stands for a member function returning a single number, where `x` may be 
    one or more arguments; whereas `eval***(x, *val, ...)` stands for a member function with 
    void return type, which computes something for the argument(s) `x` and stores the result(s)
    in output arguments(s) `*val`, ..., which optionally may be NULL to indicate that 
    the corresponding quantity needs not be computed.
*/
class IFunction {
public:
    virtual ~IFunction() {};

    /** Compute any combination of function, first and second derivative;
        each one is computed if the corresponding output parameter is not NULL.
        If the computation of a given derivative is not implemented, should return NaN. */
    virtual void evalDeriv(const double x,
        double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const=0;

    /** Query the number of derivatives implemented by this class */
    virtual unsigned int numDerivs() const=0;

    /** Convenience shorthand for computing only the value of the function */
    virtual double value(const double x) const {
        double val;
        evalDeriv(x, &val);
        return val;
    }

    /** Overloaded () operator enables to use derived classes as functors
        (in this case, as functions of one input argument returning one value) */
    double operator()(const double x) const { 
        return value(x); }
};

/** Prototype of a function that does not provide derivatives: 
    it 'swaps' the requirements for implementing the virtual methods - 
    the descendant classes should only implement the method 'value()', 
    while the method 'evalDeriv()' redirects itself to 'value()'.
*/
class IFunctionNoDeriv: public IFunction {
public:

    /** This needs to be implemented in the derived classes */
    virtual double value(const double x) const=0;

    /** Compute the value of the function only, by calling 'value()', derivatives are not available */
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        if(val)
            *val = value(x);
        if(der)
            *der = NAN;
        if(der2)
            *der2= NAN;
    }

    /** No derivatives, as one might guess */
    virtual unsigned int numDerivs() const { return 0; }

};

/** Prototype for a definite integral of a function */
class IFunctionIntegral {
public:
    virtual ~IFunctionIntegral() {};

    /** Compute the value of integral of the function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const=0;
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values. */
class IFunctionNdim {
public:
    virtual ~IFunctionNdim() {};

    /** evaluate the function.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
                    the vector of function values.
                    Should point to an existing array of length at least M.
    */
    virtual void eval(const double vars[], double values[]) const = 0;

    /// return the dimensionality of the input point (N)
    virtual unsigned int numVars() const = 0;

    /// return the number of elements in the output array of values (M)
    virtual unsigned int numValues() const = 0;
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values,
    and derivatives of these values w.r.t.the input variables (aka jacobian). */
class IFunctionNdimDeriv: public IFunctionNdim {
public:
    IFunctionNdimDeriv() {};
    virtual ~IFunctionNdimDeriv() {};

    /** evaluate the function and the derivatives.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
                    the vector of function values.
        \param[out] derivs is the M-by-N matrix (M rows, N columns) of partial derivatives 
                    of the vector-valued function by the input variables;
                    if a NULL pointer is passed, this does not need to be computed,
                    otherwise it must point to an existing array of sufficient length;
                    the indexing scheme is derivs[m*N+n] = df_m/dx_n.
    */
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const = 0;

    /** reimplement the evaluate function without derivatives */
    virtual void eval(const double vars[], double values[]) const {
        evalDeriv(vars, values);
    }
};

}  // namespace math
