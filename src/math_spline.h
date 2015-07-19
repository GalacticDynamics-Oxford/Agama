/** \file    math_spline.h
    \brief   spline interpolation class
    \author  Eugene Vasiliev
    \date    2011-2015

*/
#pragma once
#include "math_base.h"
#include <vector>

namespace mathutils{

/** Class that defines a cubic spline with natural or clamped boundary conditions */
class CubicSpline: public IFunction {
public:
    /** Initialize a cubic spline from the provided values of x and y
        (which should be arrays of equal length, and x values must be monotonically increasing).
        If deriv_left or deriv_right are provided, they set the slope at the lower or upper boundary
        (so-called clamped spline); if either of them is NaN, it means natural boundary condition.
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        double deriv_left=NAN, double deriv_right=NAN);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void eval_deriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const;

    virtual int numDerivs() const { return 2; }

    /** return the lower end of definition interval */
    double xlower() const { return xval.front(); }

    /** return the upper end of definition interval */
    double xupper() const { return xval.back(); }

    /** check if the spline is everywhere monotonic on the given interval */
    bool isMonotonic() const;

private:
    std::vector<double> xval, yval, cval;
};

}  // namespace
