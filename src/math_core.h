/** \file   math_core.h
    \brief  essential math routines (e.g., root-finding, integration)
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"
#include "math_ndim.h"

namespace math{

/** default relative accuracy of root-finder */
const double ACCURACY_ROOT=1e-6;

/** default relative accuracy of integration */
const double ACCURACY_INTEGR=1e-6;

/// \name  ---- Miscellaneous utility functions -----
///@{

/** test if a number is neither infinity nor NaN */
bool isFinite(double x);

/** compare two numbers with a relative accuracy eps: 
    \return -1 if x<y, +1 if x>y, or 0 if x and y are approximately equal */
int fcmp(double x, double y, double eps=1e-15);

/** return sign of a number */
inline double sign(double x) { return x>0?1.:x<0?-1.:0; }

/** return an integer power of a number */
double powInt(double x, int n);

/** return a pseudo-random number in the range [0,1) */
double random();

/** wraps the input argument into the range [0,2pi) */
double wrapAngle(double x);

/** create a nearly monotonic sequence of angles by adding or subtracting 2Pi as many times 
    as needed to bring the current angle `x` to within Pi from the previous one `xprev`.
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
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval (may be -INFINITY);
    \param[in] x2 is the upper end of the interval (may be +INFINITY);
    \param[in] relToler  determines the accuracy of root localization, relative to the range |x2-x1|.
*/
double findRoot(const IFunction& F, double x1, double x2, double relToler);

/** Find a local minimum on the interval [x1,x2].
    Interval can be (semi-)infinite, in which case an appropriate transformation is applied
    to the variable (but the function still should return finite value for an infinite argument).
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval (may be -INFINITY);
    \param[in] x2 is the upper end of the interval (may be +INFINITY);
    \param[in] xinit  is the optional initial guess point: 
    if provided (not NaN), this speeds up the determination of minimum, 
    but results in error if F(xinit) was not strictly lower than both F(x1) and F(x2). 
    Alternatively, if the location of the minimum is not known in advance, 
    xinit is set to NaN, and the function first performs a binary search to determine 
    the interval enclosing the minimum, and returns one of the endpoints if the function 
    turns out to be monotonic on the entire interval.
    \param[in] relToler  determines the accuracy of minimum localization, relative to the range |x2-x1|.
 */
double findMin(const IFunction& F, double x1, double x2, double xinit, double relToler);

/** Description of function behavior near a given point: the value and two derivatives,
    and the estimates of nearest points where the function takes on 
    strictly positive or negative values, or crosses zero.

    These estimates may be used to safely determine the interval of locating a root or minimum: 
    for instance, if one knows that f(x_1)=0 or very nearly zero (to within roundoff errors), 
    f(x_neg) is strictly negative,  and f(x) is positive at some interval between x_1 and x_neg,
    then one needs to find x_pos close to x_1 such that the root is strictly bracketed between 
    x_pos and x_neg (i.e. f(x_pos)>0). This is exactly the task for this little helper class.
    Another application is to post-process the result obtained by a root-finding routine,
    which ensures only that f(x_root) is approximately zero, but does not guarantee is sign,
    or even its magnitude (it only ensures that the root is located to a certain accuracy).
    If we need a point close to x_root with a known sign of f(x), this class can do the job.

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
    double dxToPosneg(double sgn) const;
};

///@}
/// \name ------ integration routines -------
///@{

/** integrate a function on a finite interval, using an adaptive 
    Gauss-Kronrod rule with maximum order up to 87. 
    If the function is well-behaved, this is the fastest method, 
    but if it cannot reach the required accuracy even using the highest-order rule,
    no further improvement can be made. 
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] relToler is the relative error tolerance;
    \param[out] error - if not NULL, output the error estimate in this variable;
    \param[out] numEval - if not NULL, output the number of function evaluations in this variable.
*/
double integrate(const IFunction& F, double x1, double x2, double relToler, 
    double* error=0, int* numEval=0);

/** integrate a function on a finite interval, using a fully adaptive integration routine 
    to reach the required tolerance; integrable singularities are handled properly. 
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] relToler is the relative error tolerance;
    \param[out] error - if not NULL, output the error estimate in this variable;
    \param[out] numEval - if not NULL, output the number of function evaluations in this variable.
*/
double integrateAdaptive(const IFunction& F, double x1, double x2, double relToler, 
    double* error=0, int* numEval=0);

/** integrate a function on a finite interval, using a fixed-order Gauss-Legendre rule
    without error estimate. 
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] N  is the number of points in Gauss-Legendre quadrature; 
    values up to 20 use pre-computed tables, and higher ones compute them on-the-fly (less efficient).
*/
double integrateGL(const IFunction& F, double x1, double x2, unsigned int N);

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

    /// return the scaled variable y for the given original variable x
    double y_from_x(const double x) const;

    /// return the original variable x for the given scaled variable y in [0,1]
    double x_from_y(const double y) const;

private:
    const IFunction& F;
    double x_low, x_upp;
};

/** N-dimensional integration (aka cubature).
    It computes the integral of a vector-valued function (each component is treated independently).
    The dimensions of integration volume and the length of result array are provided by 
    F.numVars() and F.numValues(), respectively. Integration boundaries should be finite.
    \param[in]  F  is the input function of N variables that produces a vector of M values,
    \param[in]  xlower  is the lower boundary of integration volume (vector of length N);
    \param[in]  xupper  is the upper boundary of integration volume (vector of length N);
    \param[in]  relToler  is the required relative error in each of the computed component of F;
    \param[in]  maxNumEval  is the upper limit on the number of function calls;
    \param[out] result  is the vector of length M, containing the values of integral for each component;
    \param[out] error  is the vector of length M, containing error estimates of the computed values,
                if this argument is set to NULL then no error information is stored;
    \param[out] numEval  is the actual number of function calls
                (if set to NULL, this information is not stored).
*/
void integrateNdim(const IFunctionNdim& F, const double xlower[], const double xupper[], 
    const double relToler, const unsigned int maxNumEval, 
    double result[], double error[]=0, int* numEval=0);

/** Sample points from an N-dimensional probability distribution function F.
    F should be non-negative in the given region, and the integral of F over this region should exist;
    still better is if F is bounded from above everywhere in the region. 
    The output consists of M sampling points from the given region, such that the density 
    of points in the neighborhood of any location X is proportional to the value of F(X).
    The samples are drawn from the probability distribution described by F using the standard 
    rejection sampling; the key algorithmic challenge is to make it efficient, i.e., minimize 
    the number of discarded trial points. This is achieved by adaptive importance sampling method.
    \param[in]  F  is the probability distribution, the dimensionality N of the problem 
                is given by F->numVars();
    \param[in]  xlower  is the lower boundary of sampling volume (vector of length N);
    \param[in]  xupper  is the upper boundary of sampling volume;
    \param[in]  numSamples  is the required number of sampling points (M);
    \param[out] samples  will be filled by samples, i.e. contain the matrix of M rows and N columns;
    \param[out] numTrialPoints (optional) if not NULL, will store the actual number of function
                evaluations (so that the efficiency of sampling is estimated as the ratio 
                numSamples/numTrialPoints);
    \param[out] integral (optional) if not NULL, will store the Monte Carlo estimate of the integral
                of F over the given region (this could be compared with the exact value, if known,
                to estimate the bias/error in sampling scheme). 
*/
void sampleNdim(const IFunctionNdim& F, const double xlower[], const double xupper[], 
    const unsigned int numSamples, Matrix<double>& samples, int* numTrialPoints=0, double* integral=0);

///@}

}  // namespace
