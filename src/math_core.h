/** \file   math_core.h
    \brief  essential math routines (e.g., root-finding, integration)
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"

namespace math{

/** default relative accuracy of root-finder */
const double ACCURACY_ROOT=1e-6;

/** default relative accuracy of integration */
const double ACCURACY_INTEGR=1e-6;

/** limit on the maximum number of steps in ODE solver */
const int ODE_MAX_NUM_STEP=1e6;

/// \name  ---- Miscellaneous utility functions -----
///@{

/** test if a number is not infinity or NaN */
bool isFinite(double x);

/** compare two numbers with a relative accuracy eps: 
    \return -1 if x<y, +1 if x>y, or 0 if x and y are approximately equal */
int fcmp(double x, double y, double eps=1e-15);

/** return sign of a number */
inline double sign(double x) { return x>0?1.:x<0?-1.:0; }

/** return an integer power of a number */
double powInt(double x, int n);

/** ensure that the angle lies in [0,2pi) */
double wrapAngle(double x);

/** create a nearly monotonic sequence of angles by adding or subtracting 2Pi
    as many times as needed to bring the current angle to within Pi from the previous one.
    This may be used in a loop over elements of an array, by providing the previous 
    element, already processed by this function, as xprev. 
    Note that this usage scenario is not stable against error accumulation. */
double unwrapAngle(double x, double xprev);

///@}
/// \name  ----- root-finding and minimization routines -----
///@{

/** find a root of function on the interval [x1,x2].
    function must be finite at the ends of interval and have opposite signs (or be zero),
    otherwise NaN is returned.
    Interval can be (semi-)infinite, in which case an appropriate transformation is applied
    to the variable (but the function still should return finite value for an infinite argument).
    If the function interface provides derivatives, this may speed up the search.
    \param[in] rel_toler  determines the accuracy of root localization, relative to the range |x2-x1|.
*/
double findRoot(const IFunction& F, double x1, double x2, double rel_toler);

/** Find a local minimum on the interval [x1,x2].
    Interval can be (semi-)infinite, in which case an appropriate transformation is applied
    to the variable (but the function still should return finite value for an infinite argument).
    \param[in] xinit  is the optional initial guess point: 
    if provided (not NaN), this speeds up the determination of minimum, 
    but results in error if F(xinit) was not strictly lower than both F(x1) and F(x2). 
    Alternatively, if the location of the minimum is not known in advance, 
    xinit is set to NaN, and the function first performs a binary search to determine 
    the interval enclosing the minimum, and returns one of the endpoints if the function 
    turns out to be monotonic on the entire interval.
    \param[in] rel_toler  determines the accuracy of minimum localization, relative to the range |x2-x1|.
 */
double findMin(const IFunction& F, double x1, double x2, double xinit, double rel_toler);

/** Description of function behavior near a given point: the value and two derivatives,
    and the estimates of nearest points where the function takes on 
    strictly positive or negative values, or crosses zero.

    These estimates may be used to safely determine the interval of locating a root or minimum: 
    for instance, if one knows that f(x_1)=0 or very nearly zero (to within roundoff errors), 
    f(x_neg) is strictly negative,  and f(x) is positive at some interval between x_1 and x_neg,
    then one needs to find x_pos close to x_1 such that the root is strictly bracketed between 
    x_pos and x_neg (i.e. f(x_pos)>0). This is exactly the task for this little helper class.

    Note that the function is only computed at the given point (or, if its implementation 
    does not provide derivatives itself, they are computed numerically using one or two 
    nearby points);  thus the estimates provided by this utility class are only valid 
    as long as f(x0) is close to zero, or, more precisely, if the offsets ~|f/f'| are 
    much smaller than the scale of function variation ~|f'/f''|.
*/
class PointNeighborhood {
public:
    double f0, fder, fder2;  ///< the value, first and second derivative at the given point
    PointNeighborhood(const IFunction& fnc, double x0);

    /// return the estimated offset from x0 to the value of x where the function is positive
    double dxToPositive() const {
        return dxToPosneg(+1); }
    /// return the estimated offset from x0 to the value of x where the function is negative
    double dxToNegative() const {
        return dxToPosneg(-1); }
    /// return the estimated offset from x0 to the nearest root of f(x)=0
    double dxToNearestRoot() const;
private:
    double delta;   ///< a reasonably small value
    double dxToPosneg(double sgn) const;
};

///@}
/// \name ------ integration routines -------
///@{

/** integrate a function on a finite interval, using an adaptive 
    Gauss-Kronrod rule with maximum order up to 87. 
    If the function is well-behaved, this is the fastest method, 
    but if it cannot reach the required accuracy even using the highest-order rule,
    no further improvement can be made. */
double integrate(const IFunction& F, double x1, double x2, double rel_toler);

/** integrate a function on a finite interval, using a fixed-order Gauss-Legendre rule
    without error estimate. */
double integrateGL(const IFunction& F, double x1, double x2, unsigned int N);

/** integrate a function on a finite interval, using a fully adaptive integration routine 
    to reach the required tolerance; integrable singularities are handled properly. */
double integrateAdaptive(const IFunction& F, double x1, double x2, double rel_toler);

/** Helper class for integrand transformations.
    A function defined on a finite interval [x_low,x_upp], with possible integrable 
    singularities at endpoints, can be integrated on the interval [x1,x2] 
    (which may be narrower, i.e., x_low <= x1 <= x2 <= x_upp),
    by applying the following transformation.
    The integral  \f$  \int_{x1}^{x2} f(x) dx  \f$  is transformed into 
    \f$  \int_{y1}^{y2} f(x(y)) (dx/dy) dy  \f$,  where  0 <= y <= 1,
    \f$  x(y) = x_{low} + (x_{upp}-x_{low}) y^2 (3-2y)  \f$, and x1=x(y1), x2=x(y2). 

    This class may be used with any integration routine as follows:
    ~~~~
    ScaledIntegrandEndpointSing transf(fnc, x_low, x_upp);
    integrate*** (transf, transf.y_from_x(x1), transf.y_from_x(x2), ...)
    ~~~~
    Note that if x1==x_low then one may simply put 0 as the lower integration limit,
    and similarly 1 for the upper limit if x2==x_upp.
*/
class ScaledIntegrandEndpointSing: public IFunctionNoDeriv {
public:
    ScaledIntegrandEndpointSing(const IFunction& _F, double low, double upp) : 
        F(_F), x_low(low), x_upp(upp) {};

    /// return the value of input function at the unscaled coordinate x(y)
    /// times the conversion factor dx/dy
    virtual double value(const double y) const;

    // return the scaled variable y for the given original variable x
    double y_from_x(const double x) const;

    // return the original variable x for the given scaled variable y in [0,1]
    double x_from_y(const double y) const;

private:
    const IFunction& F;
    double x_low, x_upp;
};
    
///@}
/// \name ------ linear regression -----
///@{

/** perform a linear least-square fit (i.e., y=c*x+b);
    store the best-fit slope and intercept of the relation in the corresponding output arguments, 
    and store the rms scatter in the output argument 'rms' if it is not NULL. */
void linearFit(unsigned int N, const double x[], const double y[], 
    double& slope, double& intercept, double* rms=0);

/** perform a linear least-square fit without constant term (i.e., y=c*x);
    return the best-fit slope of the relation, and store the rms scatter 
    in the output argument 'rms' if it is not NULL. */
double linearFitZero(unsigned int N, const double x[], const double y[], double* rms=0);

///@}
/// \name ----- systems of ordinary differential equations ------
///@{

/** Prototype of a function that is used in integration of ordinary differential equation systems:
    dy/dt = f(t, y), where y is an N-dimensional vector. */
class IOdeSystem {
public:
    IOdeSystem() {};
    virtual ~IOdeSystem() {};
    
    /** compute the r.h.s. of the differential equation: 
        \param[in]  t    is the integration variable (time),
        \param[in]  y    is the vector of values of dependent variables, 
        \param[out] dydt should return the time derivatives of these variables */
    virtual void eval(const double t, const double y[], double* dydt) const=0;
    
    /** return the size of ODE system (number of variables) */
    virtual int size() const=0;
};

/** solve a system of differential equations */
class OdeSolver {
public:
    OdeSolver(const IOdeSystem& F, double abstoler, double reltoler);
    ~OdeSolver();
    int advance(double tstart, double tfinish, double *y);
private:
    void* impl;   ///< implementation details are hidden
};
///@}

}  // namespace
