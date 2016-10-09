/** \file   math_core.h
    \brief  essential math routines (e.g., root-finding, integration)
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"

namespace math{
    
/// \name  ---- Miscellaneous utility functions -----
///@{

/** compare two numbers with a relative accuracy eps: 
    \return -1 if x<y, +1 if x>y, -2 if x==NAN, +2 if y==NAN, or 0 if x and y are approximately equal
*/
int fcmp(double x, double y, double eps=1e-15);

/** return sign of a number */
template<typename T>
inline T sign(T x) { return x>0 ? 1 : x<0 ? -1 : 0; }

/** return absolute value of a number */
template<typename T>
inline T abs(T x) { return x<0 ? -x : x; }

/** return an integer power of a number */
double powInt(double x, int n);


/** initialize the pseudo-random number generator with the given value,
    or a completely arbitrary value (depending on system time) if seed==0.
    In a multi-threaded case, each thread is initialized with a different seed.
    At startup (without any call to randomize()), the random seed always has the same value.
*/
void randomize(unsigned int seed=0);

/** return a pseudo-random number in the range [0,1).
    In a multi-threaded environment, each thread has access to its own instance of 
    pseudo-random number generator, which are initialized with different (but fixed) seeds.
    Therefore, if several threads process the data in an orderly way (using a static schedule),
    they will receive the same sequence of numbers on each run of the program,
    unless randomize(0) is called and as long as the number of threads is fixed.
*/
double random();

/** return two uncorrelated random numbers from the standard normal distribution */
void getNormalRandomNumbers(double& num1, double& num2);

/** return a quasirandom number from the Halton sequence.
    \param[in]  index  is the index of the number (should be >0, or better > ~10-20);
    \param[in]  base   is the base of the sequence, must be a prime number;
    if one needs an N-dimensional vector of ostensibly independent quasirandom numbers,
    one may call this function N times with different prime numbers as bases.
    \return  a number between 0 and 1.
    \note that that the numbers get increasingly more correlated as the base increases,
    thus it is not recommended to use more than ~6 dimensions unless index spans larger enough range.
*/
double quasiRandomHalton(unsigned int index, unsigned int base);

/** first ten prime numbers, may be used as bases in `quasiRandomHalton` */
static const unsigned int MAX_PRIMES = 10;  // not that there aren't more!
static const int PRIMES[MAX_PRIMES] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

/** wraps the input argument into the range [0,2pi) */
double wrapAngle(double x);

/** create a nearly monotonic sequence of angles by adding or subtracting 2Pi as many times 
    as needed to bring the current angle `x` to within Pi from the previous one `xprev`.
    This may be used in a loop over elements of an array, by providing the previous 
    element, already processed by this function, as xprev. 
    Note that this usage scenario is not stable against error accumulation. */
double unwrapAngle(double x, double xprev);


/** Perform a binary search in an array of sorted numbers x_0 < x_1 < ... < x_N
    to locate the index of bin that contains a given value x.
    \tparam  NumT is the numeric type of data in the array (double, float, int or unsigned int);
    \param[in]  x is the position, which must lie in the interval x_0 <= x <= x_N;
    \param[in]  arr is the array of bin boundaries sorted in ascending order (NOT CHECKED!)
    \param[in]  size is the number of elements in the array (i.e., number of bins plus 1);
    since this header does not include <vector>, we shall pass the array in the traditional
    way, as a pointer to the first element plus the number of elements.
    \returns the index k of the bin such that x_k <= x < x_{k+1}, where the last strict inequality
    is replaced by <= for the last bin (x=x_N still returns N-1).
    \throws std::invalid_argument exception if the point is outside the interval */
template<typename NumT>
unsigned int binSearch(const NumT x, const NumT arr[], const unsigned int size);

/** linearly interpolate the value y(x) between y1 and y2, for x between x1 and x2 */
inline double linearInterp(double x, double x1, double x2, double y1, double y2) {
    return ((x-x1)*y2 + (x2-x)*y1) / (x2-x1); }

/** cubic Hermite interpolation of y(x) for x1<=x<=x2 given the values (y1,y2) 
    and derivatives (dy1, dy2) of y at the boundaries of interval x1 and x2 */
inline double hermiteInterp(double x, double x1, double x2,
    double y1, double y2, double dy1, double dy2) {
    const double t = (x-x1) / (x2-x1);
    return pow_2(1-t) * ( (1+2*t) * y1 +   t   * dy1 * (x2-x1) )
         + pow_2(t)   * ( (3-2*t) * y2 + (t-1) * dy2 * (x2-x1) );
}


/** Class for computing running average and dispersion for a sequence of numbers */
class Averager {
public:
    Averager() : avg(0.), ssq(0.), num(0) {}
    /** Add a number to the list (the number itself is not stored) */
    void add(const double value) {
        double diff =  value-avg;
        avg += diff / (num+1);     // use Welford's stable recurrence relation
        ssq += diff * (value-avg); // for computing mean and variance of weighted function values
        num++;
    }
    /** Return the mean value of all elements added so far: 
        \f$  < x > = (1/N) \sum_{i=1}^N  x_i  \f$.  */
    double mean() const { return avg; }
    /** Return the dispersion of all elements added so far:
        \f$  D = (1/N) \sum_{i=1}^N  ( x_i - < x > )^2  \f$.  */
    double disp() const { return num>1 ? ssq / (num-1) : 0; }
    /** Return the number of elements */
    unsigned int count() const { return num; }
private:
    double avg, ssq;
    unsigned int num;
};

/** A product of two functions, to be used as a temporary object passed to
    integration or root-finding routines */
class FncProduct: public IFunction {
    const IFunction &f1, &f2; ///< references to two functions
public:
    FncProduct(const IFunction& fnc1, const IFunction& fnc2) :
        f1(fnc1), f2(fnc2) {}

    virtual unsigned int numDerivs() const {
        return f1.numDerivs() < f2.numDerivs() ? f1.numDerivs() : f2.numDerivs();
    }

    virtual void evalDeriv(const double x, double *val, double *der, double *der2) const {
        double v1, v2, d1, d2, dd1, dd2;
        bool needDer = der!=NULL || der2!=NULL, needDer2 = der2!=NULL;
        f1.evalDeriv(x, &v1, needDer ? &d1 : 0, needDer2 ? &dd1 : 0);
        f2.evalDeriv(x, &v2, needDer ? &d2 : 0, needDer2 ? &dd2 : 0);
        if(val)
            *val = v1 * v2;
        if(der)
            *der = v1 * d2 + v2 * d1;
        if(der2)
            *der2 = v1 * dd2 + 2 * d1 * d2 + v2 * dd1;
    }
};

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

///@}
/// \name ------ numerical derivatives -------
///@{

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
    double absx0;            ///< the abs.value of coordinate of the given point
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
    /// return the estimated distance between two roots of f(x)=0 if they exist
    double dxBetweenRoots() const;
private:
    double dxToPosneg(double sgn) const;
};

/** Evaluate the second derivative of a function from its values and first derivatives
    at three consecutive points, using high-accuracy Hermite interpolation.
    \param[in] x0, x1, x2 are the abscissa points (assuming that x1 is between x0 and x2);
    \param[in] f0, f1, f2 are the function values at these points;
    \param[in] df0, df1, df2 are the function derivatives at these points;
    \return the second derivative of function at x1
*/
double deriv2(double x0, double x1, double x2, double f0, double f1, double f2,
    double df0, double df1, double df2);

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
    double* error=NULL, int* numEval=NULL);

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
    double* error=NULL, int* numEval=NULL);

/** integrate a function on a finite interval, using a fixed-order Gauss-Legendre rule
    without error estimate. 
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] N  is the number of points in Gauss-Legendre quadrature; 
    values up to 20 use pre-computed tables, and higher ones compute them on-the-fly (less efficient).
*/
double integrateGL(const IFunction& F, double x1, double x2, unsigned int N);

/** prepare a table for Gauss-Legendre integration of one or many functions on the same interval.
    The integral is approximated by a weighted sum of function values over the array of points:
    \f$  \int_{x1}^{x2} f(x) dx = \sum_{i=0}^{N-1}  w_i f(x_i)  \f$.
    This function computes the coordinates x_i and weights w_i of these points 
    for the given order of quadrature; the user then may perform the above summation
    for as many functions as necessary, using the same table.
    \param[in]  x1  is the lower end of interval;
    \param[in]  x2  is the upper end of interval;
    \param[in]  N   is the number of points in the tables;
    \param[out] coords   points to the array that will be filled with coordinates of the points
    (it should be allocated before the call to this routine);
    \param[out] weights  will be filled with weights (array should be allocated beforehand).
*/
void prepareIntegrationTableGL(double x1, double x2, int N, double* coords, double* weights);

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
    double result[], double error[]=NULL, int* numEval=NULL);

///@}

}  // namespace
