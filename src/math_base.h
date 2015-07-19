#pragma once

/// convenience function for squaring a number, used in many places
inline double pow_2(double x) { return x*x; }

#ifndef INFINITY
#define INFINITY 1e10000
#endif
#ifndef NAN
#define NAN (INFINITY/INFINITY)
#endif

namespace mathutils{

/** Prototype of a function of one variable that may provide up to two derivatives.
    This interface serves as the base for many mathematical routines throughout the code,
    some of them need only the value of the function, and some may need derivatives.
    However, not all classes implementing this interface are obliged to compute derivatives;
    the actual number of derivatives that can be produces is returned by a dedicated method 
    `numDerivs()`, which should be queried beforehand, and if necessary, the derivatives 
    must be estimated by the calling code itself from finite differences.  */
class IFunction {
public:
    IFunction() {};
    virtual ~IFunction() {};

    /** Compute any combination of function, first and second derivative;
        each one is computed if the corresponding output parameter is not NULL.
        If the computation of a given derivative is not implemented, should return NaN. */
    virtual void eval_deriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const=0;

    /** Query the number of derivatives implemented by this class */
    virtual int numDerivs() const=0;

    /** Convenience shorthand for computing only the value of the function */
    double value(const double x) const {
        double val;
        eval_deriv(x, &val);
        return val;
    }
};

}  // namespace mathutils
