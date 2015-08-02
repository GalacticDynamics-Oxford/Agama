/** \file    math_base.h
    \brief   Base class for mathematical functions of one variable
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

#ifndef INFINITY
#define INFINITY 1e10000
#endif
#ifndef NAN
#define NAN (INFINITY/INFINITY)
#endif

/** Functions and classes for basic and advanced math operations */
namespace math{

/** Prototype of a function of one variable that may provide up to two derivatives.
    This interface serves as the base for many mathematical routines throughout the code,
    some of them need only the value of the function, and some may need derivatives.
    However, not all classes implementing this interface are obliged to compute derivatives;
    the actual number of derivatives that can be produces is returned by a dedicated method 
    `numDerivs()`, which should be queried beforehand, and if necessary, the derivatives 
    must be estimated by the calling code itself from finite differences.
    Descendant classes should implement 'evalDeriv()' and 'numDerivs()' methods, 
    while 'value()' and the operator() are simple shorthands provided by this base class. */
class IFunction {
public:
    IFunction() {};
    virtual ~IFunction() {};

    /** Compute any combination of function, first and second derivative;
        each one is computed if the corresponding output parameter is not NULL.
        If the computation of a given derivative is not implemented, should return NaN. */
    virtual void evalDeriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const=0;

    /** Query the number of derivatives implemented by this class */
    virtual int numDerivs() const=0;

    /** Convenience shorthand for computing only the value of the function */
    virtual double value(const double x) const {
        double val;
        evalDeriv(x, &val);
        return val;
    }
    
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
    IFunctionNoDeriv() {};
    
    /** This needs to be implemented in the derived classes */
    virtual double value(const double x) const=0;

    /** Compute the value of the function only, by calling 'value()', derivatives are not available */
    virtual void evalDeriv(const double x, double* val=0, double* der=0, double* der2=0) const {
        if(val)
            *val = value(x);
        if(der)
            *der = NAN;
        if(der2)
            *der2= NAN;
    }

    /** No derivatives, as one might guess */
    virtual int numDerivs() const { return 0; }

};
    
}  // namespace math
