/** \file    math_base.h
    \brief   Base class for mathematical functions
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once
#include <cstddef>   // defines NULL
#include <limits>    // defines infinity and NaN

// some useful numbers (or even not-a-numbers)

#ifndef INFINITY
#define INFINITY std::numeric_limits<double>::infinity()
#endif

#ifndef NAN
#define NAN      std::numeric_limits<double>::quiet_NaN()
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON        2.2204460492503131e-16
#endif
#define SQRT_DBL_EPSILON   1.4901161193847656e-08
#define ROOT3_DBL_EPSILON  6.0554544523933429e-06

#ifndef M_PI
#define M_PI     3.14159265358979323846264338328
#endif

#ifndef M_SQRTPI
#define M_SQRTPI 1.77245385090551602729816748334
#endif

#ifndef M_SQRT2
#define M_SQRT2  1.41421356237309504880168872421
#endif

#ifndef M_SQRT3
#define M_SQRT3  1.73205080756887729352744634151
#endif

#ifndef M_LN2
#define M_LN2    0.69314718055994530942
#endif

#define TWO_PI_CUBE 248.050213442398561403810520537

// a few very basic routines declared in the global namespace

/// convenience function for squaring a number, used in many places
template<typename T> inline T pow_2(T x) { return x*x; }

/// convenience function for raising a number to the 3rd power
template<typename T> inline T pow_3(T x) { return x*x*x; }

/// test if a number is neither infinity nor NaN
inline bool isFinite(double x) { return x>-INFINITY && x<INFINITY; }

/// sterilize NaN (replace with zero) and keep any other input unchanged
inline double nan2num(double x) { return isFinite(x) ? x : 0; }


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
        /*output*/ double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const = 0;

    /** Return the number of derivatives implemented by this class. */
    virtual unsigned int numDerivs() const = 0;

    /** Convenience shorthand for computing only the value of the function. */
    virtual double value(const double x) const
    {
        double val;
        evalDeriv(x, &val);
        return val;
    }

    /** Overloaded () operator enables to use derived classes as functors
        (in this case, as functions of one input argument returning one value) */
    double operator()(const double x) const
    {
        return value(x);
    }
};

/** Prototype of a function that provides three derivatives, extending the previous one. */
class IFunction3Deriv: public IFunction {
public:

    /** Evaluate the function and up to two derivatives at the given point. */
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const
    {
        evalDeriv(x, val, der, der2, NULL);
    }

    /** Actual computation of the function and three derivatives needs to be implemented
        in the derived classes; if any of the output pointers is NULL, the corresponding quantity
        does not need to be computed. */
    virtual void evalDeriv(const double x,
        /*output*/ double* val, double* der, double* der2, double* der3) const = 0;

    virtual unsigned int numDerivs() const { return 3; }
};

/** Prototype of a function that does not provide derivatives: 
    it 'swaps' the requirements for implementing the virtual methods -
    the descendant classes should only implement the method 'value()', 
    while the method 'evalDeriv()' redirects itself to 'value()'.
*/
class IFunctionNoDeriv: public IFunction {
public:

    /** This needs to be implemented in the derived classes. */
    virtual double value(const double x) const = 0;

    /** Compute the value of the function only, by calling 'value()', derivatives are not available. */
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const
    {
        if(val)
            *val = value(x);
        if(der)
            *der = NAN;
        if(der2)
            *der2= NAN;
    }

    /** No derivatives, as one might guess. */
    virtual unsigned int numDerivs() const { return 0; }
};

/** Prototype for a definite integral of a function. */
class IFunctionIntegral {
public:
    virtual ~IFunctionIntegral() {};

    /** Compute the value of integral of the function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const = 0;
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values. */
class IFunctionNdim {
public:
    virtual ~IFunctionNdim() {};

    /** Evaluate the function.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
        the vector of function values. Should point to an existing array of length at least M.
    */
    virtual void eval(const double vars[], double values[]) const = 0;

    /** Evaluate the function at several points in a single call.
        The default implementation simply loops over input points and calls eval() on each of them,
        but derived classes may provide an optimized version for P>1.
        This method is called, for instance, by integrateNdim() and sampleNdim() routines.
        \param[in]  npoints >= 1 is the number of input points (P).
        \param[in]  vars  is the array of P N-dimensional points, where N=numVars():
        d-th coordinate of p-th point is retrieved from vars[p * N + d], 0 <= d < N, 0 <= p < npoints.
        \param[out] values is the array of P M-dimensional function values, where M=numValues():
        v-th value at p-th point is stored in values[p * M + v], 0 <= v < M, 0 <= p < npoints.
        Should point to an existing array of length (at least) M*P.
    */
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        for(size_t p=0, N=numVars(), M=numValues(); p<npoints; p++)
            eval(vars + p*N, values + p*M);
    }

    /** Return the dimensionality of the input point (N). */
    virtual unsigned int numVars() const = 0;

    /** Return the number of elements in the output array of values (M). */
    virtual unsigned int numValues() const = 0;
};

/** Prototype of a function that has possibly M>>1 output values, of which only a small fraction
    could be nonzero at any point; it provides the interface for optimized accumulation of
    weighted points in the output array. */
class IFunctionNdimAdd: public IFunctionNdim {
public:
    /** Add a weighted point to the output array.
        \param[in]  vars  is the input point;
        \param[in]  mult  is the weight of input point;
        \param[in,out]  output  points to the external array that accumulates the data;
        all its elements that have a contribution from the input point are incremented by
        the appropriate amount, multiplied by the input factor 'mult'.
    */
    virtual void addPoint(const double vars[], const double mult, double output[]) const = 0;

    virtual void eval(const double vars[], double values[]) const
    {
        /// to compute all values at the given point, we fill the output array with zeros
        /// and invoke the `addPoint()` method (not very efficient, just fulfills the interface)
        for(unsigned int size=numValues(), i=0; i<size; i++)
            values[i] = 0.;
        addPoint(vars, 1., values);
    }
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values,
    and derivatives of these values w.r.t.the input variables (aka jacobian). */
class IFunctionNdimDeriv: public IFunctionNdim {
public:
    IFunctionNdimDeriv() {};
    virtual ~IFunctionNdimDeriv() {};

    /** Evaluate the function and the derivatives.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
        the vector of function values.
        \param[out] derivs is the M-by-N matrix (M rows, N columns) of partial derivatives 
        of the vector-valued function by the input variables; if a NULL pointer is passed,
        this does not need to be computed, otherwise it must point to an existing array
        of sufficient length; the indexing scheme is derivs[m*N+n] = df_m/dx_n.
    */
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const = 0;

    /** Reimplement the evaluate function without derivatives. */
    virtual void eval(const double vars[], double values[]) const
    {
        evalDeriv(vars, values);
    }
};

}  // namespace math
