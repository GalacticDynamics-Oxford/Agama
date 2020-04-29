/** \file   math_core.h
    \brief  essential math routines (e.g., root-finding, integration)
    \date   2015-2017
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

/** restrict the value x to the range [low..upp]; return low if x is NAN */
template<typename T>
inline T clip(const T& x, const T& low, const T& upp) { return x>upp ? upp : x>low ? x : low; }

/** return an integer power of a number */
double pow(double x, int n);

/** return a number raised to the given power,
    taking shortcuts for a few common values of n such as 0.5 or 2 */
double pow(double x, double n);


/** wraps the input argument into the range [0,2pi),
    taking the remainder of the division of x by the floating-point constant 2*M_PI
    (not the mathematical constant 2pi)
*/
double wrapAngle(double x);

/** create a nearly monotonic sequence of angles by adding or subtracting 2Pi as many times 
    as needed to bring the current angle `x` to within Pi from the previous one `xprev`.
    This may be used in a loop over elements of an array, by providing the previous 
    element, already processed by this function, as xprev. 
    Note that this usage scenario is not stable against error accumulation. */
double unwrapAngle(double x, double xprev);

/** optimized function for computing both sine and cosine at once;
    it performs a modified argument range reduction so that the floating-point values that are
    integer multiples of M_PI/2 correspond to exactly zero values of sine or cosine */
void sincos(double x, double& s, double& c);


/** Perform a binary search in an array of sorted numbers x_0 < x_1 < ... < x_N
    to locate the index of bin that contains a given value x.
    \tparam  NumT is the numeric type of data in the array (double, float, signed/unsigned int or long);
    \param[in]  x is the position;
    \param[in]  arr is the array of bin boundaries sorted in ascending order (assumed but NOT CHECKED!);
    \param[in]  size is the number of elements in the array (i.e., number of bins N plus 1);
    since this header file does not #include <vector>, we shall pass the array in the traditional
    way, as a pointer to the first element plus the number of elements.
    \returns the index k of the bin such that x_k <= x < x_{k+1}, where the last strict inequality
    is replaced by <= for the last bin (x=x_N still returns N-1);
    if x is strictly outside the grid, return -1 if x<x_0, or N if x>x_N.
*/
template<typename NumT>
ptrdiff_t binSearch(const NumT x, const NumT arr[], const size_t size);

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

///@}
/// \name  ----- algebraic transformations of functions -----
///@{

/// transform the interval  -inf <= u <= +inf  to  0 <= s <= 1:  u = 1 / (1-s) - 1 / s;
/// this is useful for finding roots on the entire real axis, if the magnitude of the unscaled
/// variable u is known to be moderate (e.g., it is a logarithm of some other quantity)
struct ScalingInf {};

/// transform a semi-infinite interval 
/// 0 <= u <= +inf,       or  -inf <= u <= -0,  or  
/// 0 < u0 <= u <= +inf,  or  -inf <= u <= u0 < 0,  to  0 <= s <= 1:
/// in the first  case (u0=+0.0),  u = exp( 1 / (1-s) - 1 / s ),
/// in the second case (u0=-0.0),  u =-exp( 1 / s - 1 / (1-s) ),
/// otherwise (u0!=0)              u = u0 * exp( s / (1-s) );
/// this is useful for finding roots on a semi-infinite interval which does not strictly enclose zero,
/// and the characteristic scale of the function is unknown (log-scaling solves this problem)
struct ScalingSemiInf {
    double u0;  ///< may be +-0.0 or any non-zero number
    ScalingSemiInf(double _u0=+0.0) : u0(_u0) {}
};

/// trivial linear rescaling of the interval  uleft <= u <= uright  to  0 <= s <= 1
struct ScalingLin {
    double uleft, uright;
    ScalingLin(double _uleft, double _uright) : uleft(_uleft), uright(_uright) {}
};

/// cubic rescaling of the interval  uleft <= u <= uright  to  0 <= s <= 1:
/// u = s^2 (3 - 2 s) (uright - uleft) + uleft;
/// this transformation stretches the region around the boundaries (du/ds ~ 0 at both ends)
/// and hence is suitable for removing integrable endpoint singularities:
/// \f$  \int_{uleft}^{uright} f(u) du = \int_0^1 f(u[s]) [du/ds] ds  \f$
struct ScalingCub {
    double uleft, uright;
    ScalingCub(double _uleft, double _uright) : uleft(_uleft), uright(_uright) {}
};

/// quintic rescaling of the interval  uleft <= u <= uright  to  0 <= s <= 1:
/// u = s^3 (10 - 15 s + 6 s^2) (uright - uleft) + uleft;
/// this transformation strongly stretches the region around the boundaries
/// (both du/ds and d2u/ds2 ~ 0 at both ends)
struct ScalingQui {
    double uleft, uright;
    ScalingQui(double _uleft, double _uright) : uleft(_uleft), uright(_uright) {}
};

/// return the scaled variable s for the given original variable u
template<typename Scaling> double scale(const Scaling& scaling, double u);

/// return the original variable u for the given scaled variable s which should be in the range [0:1],
/// and optionally the derivative du/ds
template<typename Scaling> double unscale(const Scaling& scaling, double s, double* duds=NULL);

/** Helper class for scaling transformations.
    A function f(u) is transformed into f(u[s]), where the scaled variable s in the interval [0:1]
    is related to u by one of the scaling transformations defined above.
    The function of the scaled argument optionally provides the first derivative df/ds = df/du du/ds.

    This class may be used with root-finding and minimization routines on (semi-)infinte intervals,
    especially when even the order of magnitude of the root is not known in advance:
    ~~~~
    MyFnc fnc;               // original function f(u)
    ScalingSemiInf scaling;  // transform [0..inf] to [0..1], or any other scaling transformation
    double root = findRoot(fnc, scaling, toler);
    ~~~~
    The templated overloaded routines findRoot and findMin internally create an instance of ScaledFnc for
    the user-provided scaling transformation, find the root in a scaled variable, and unscale it back.

    \tparam  Scaling  is one of the available scaling transformations (e.g., ScalingCub, ScalingInf).
*/
template<typename Scaling>
class ScaledFnc: public IFunction {
public:
    const Scaling scaling;  ///< the instance of the scaling transformations specifying its parameters
    const IFunction& fnc;   ///< the original function of the unscaled argument u
    ScaledFnc(const Scaling& _scaling, const IFunction& _fnc) : scaling(_scaling), fnc(_fnc) {}

    /// compute the original function f(u[s]) for the given value of scaled argument s
    virtual void evalDeriv(const double s, double* val=0, double* der=0, double* der2=0) const {
        double dfdu, duds;
        double u = unscale(scaling, s, der? &duds : NULL);
        fnc.evalDeriv(u, val, der? &dfdu : NULL);
        if(der)
            *der = dfdu * duds;
        if(der2)
            *der2= NAN;  // not available
    }

    virtual unsigned int numDerivs() const { return fnc.numDerivs()<1 ? 0 : 1; }
};

/** Helper class for integrand transformations.
    A function defined on a finite interval [x_lower,x_upper], with possible integrable 
    singularities at endpoints, can be integrated on the interval [x1,x2] 
    (which may be narrower, i.e., x_low <= x1 <= x2 <= x_upp), by transforming the integration variable.
    \f$  \int_{x1}^{x2} f(x) dx  =  \int_{s1}^{s2} f(x(s)) (dx/ds) ds  \f$,
    where  0 <= s <= 1  is the scaled integration variable, endpoints are defined as x1=x(s1), x2=x(s2),
    and x(s) is a `ScalingCub` transformation that stretches the regions around s=0, s=1.
    Similarly, if the interval is (semi-)infinite, then one may use `ScalingInf` or `ScalingSemiInf`.

    This class may be used with any integration routine as follows:
    ~~~~
    MyFnc fnc;                              // original function f(x)
    ScalingCub scaling(x_lower, x_upper);   // or any other scaling transformation
    integrate*** (ScaledIntegrand<ScalingCub>(scaling, fnc), scale(scaling, x1), scale(scaling, x2), ...)
    ~~~~
    Note that if x1==x_lower then one may simply put 0 as the lower integration limit,
    and similarly 1 for the upper limit if x2==x_upper.

    \tparam  Scaling  is one of the available scaling transformations (e.g., ScalingCub, ScalingInf).
*/
template<typename Scaling>
class ScaledIntegrand: public IFunctionNoDeriv {
public:
    const Scaling scaling;  ///< the instance of the scaling transformations specifying its parameters
    const IFunction& fnc;   ///< the original function of the unscaled argument u
    ScaledIntegrand(const Scaling& _scaling, const IFunction& _fnc) : scaling(_scaling), fnc(_fnc) {}

    /// compute the integrand: the product of the original function of unscaled argument
    /// multiplied by the derivative of scaling transformation
    virtual double value(const double s) const {
        double duds, u = unscale(scaling, s, &duds);
        return fnc.value(u) * duds;
    }
};

/** A wrapper providing the IFunction interface for an ordinary function with one parameter,
    e.g., exp(x) */
class FncWrapper: public IFunctionNoDeriv {
    double (*fnc)(double);  ///< function pointer to the actual function
public:
    FncWrapper(double (*f)(double)) : fnc(f) {}
    virtual double value(const double x) const { return fnc(x); }
};

/** A product of two functions, to be used as a temporary object passed to
    integration or root-finding routines */
class FncProduct: public IFunction {
    const IFunction &f1, &f2;  ///< references to two functions
public:
    FncProduct(const IFunction& fnc1, const IFunction& fnc2) :
        f1(fnc1), f2(fnc2) {}
    virtual unsigned int numDerivs() const {
        return f1.numDerivs() < f2.numDerivs() ? f1.numDerivs() : f2.numDerivs();
    }
    virtual void evalDeriv(const double x, /*output*/ double *val, double *der, double *der2) const;
};

/** Doubly-log-scaled function: return log(f(exp(logx))) and up to two derivatives w.r.t log(x) */
class LogLogScaledFnc: public IFunction {
    const IFunction& fnc;  ///< reference to the original function
public:
    LogLogScaledFnc(const IFunction& _fnc) : fnc(_fnc) {}
    virtual void evalDeriv(const double logx, /*output*/ double* logf, double* der, double* der2) const;
    virtual unsigned int numDerivs() const { return fnc.numDerivs(); }
};


// a couple of analytic functions providing the IFunction and IFunctionIntegral interfaces

/** A simple monomial function  f(x) = x^m */
class Monomial: public IFunction, public IFunctionIntegral {
    const int m;
public:
    Monomial(int _m) : m(_m) {};

    virtual void evalDeriv(const double x, /*output*/ double *val, double *der, double *der2) const;
    virtual unsigned int numDerivs() const { return 2; }
    virtual double integrate(double x1, double x2, int n=0) const;
};

/** A normalized Gaussian function centered at origin with a width sigma:
    \f$   f(x) = \exp( - [x / \sigma]^2 / 2 ) / (\sqrt{2\pi} \sigma)  \f$
*/
class Gaussian: public IFunction, public IFunctionIntegral {
    const double sigma;
public:
    Gaussian(double _sigma) : sigma(_sigma) {}

    /* Compute the value and up to two derivatives of the function */
    virtual void evalDeriv(const double x, /*output*/ double *val, double *der, double *der2) const;
    virtual unsigned int numDerivs() const { return 2; }

    /** Compute the value of integral of the function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const;
};

///@}
/// \name  ----- root-finding and minimization routines -----
///@{

/** find a root of a function F(x) on the finite interval [x1,x2].
    Function values at the endpoints of the interval must be finite and have opposite signs
    (or be zero), otherwise NaN is returned.
    If the function interface provides derivatives, this may speed up the search.
    \param[in] F  is the input function;
    \param[in] x1,x2 are two endpoints of the interval;
    \param[in] relToler  determines the accuracy of root localization, relative to the range |x2-x1|.
    \return  the root of equation F(x)=0, or NAN if the endpoints do not bracket the root.
    \throw   std::invalid_argument if the endpoints are not finite or relToler is not positive.
*/
double findRoot(const IFunction& F, double x1, double x2, double relToler);

/** find a root of a function F(x) with a given scaling transformation of the argument: x=x(s).
    This convenience routine allows to use (semi-)infinite intervals of x, transformed to 0<=s<=1
    (the calling code must choose a suitable transformation).
    The original interval for x is defined by the endpoints of the scaled variable: x(s=0), x(s=1).
    \tparam  Scaling  defines the type of the scaling transformation;
    \param[in]  F  is the input function F(x);
    \param[in]  scaling  is the instance of scaling transformation and defines its parameters,
    and implicitly the interval of the un-scaled variable x.
    \param[in]  relToler  determines the accuracy of root localization in the scaled variable s.
    \return  the root x of the input function (un-scaled), or NAN if the endpoints do not bracket it.
    \throw   std::invalid_argument if the endpoints are not finite or relToler is not positive.
*/
template<typename Scaling>
inline double findRoot(const IFunction& F, const Scaling& scaling, double relToler)
{
    return unscale(scaling, findRoot(ScaledFnc<Scaling>(scaling, F), 0, 1, relToler));
}

/** Find a local minimum of a function F(x) on the finite interval [x1,x2].
    \param[in] F  is the input function;
    \param[in] x1,x2  are the endpoints of the interval;
    \param[in] xinit  is the optional initial guess point: 
    if provided (not NaN), this may speed up the localization of the minimum,
    otherwise it will be determined automatically.
    \param[in] relToler  determines the accuracy of localization of the minimum,
    relative to the range |x2-x1|.
    \return  the point where F(x) attains a (local) minimum, or one of the endpoints of the interval;
    if the function produces an infinite value, return NAN.
    \throw  std::invalid_argument if the endpoints are not finite or relToler is not positive.
*/
double findMin(const IFunction& F, double x1, double x2, double xinit, double relToler);

/** find a local minimum of a function F(x) with a given scaling transformation of the argument: x=x(s).
    This routine allows to use (semi-)infinite intervals of x, similarly to findRoot.
    \tparam  Scaling  defines the type of the scaling transformation;
    \param[in]  F  is the input function F(x);
    \param[in]  scaling  is the instance of scaling transformation and defines its parameters,
    and implicitly the interval of the un-scaled variable x.
    \param[in]  xinit  is the optional initial guess point in unscaled coordinates (use NaN if not known);
    \param[in]  relToler  determines the accuracy of localization in the scaled variable s.
    \return  the point x (un-scaled) where the input function reaches a (local) minimum.
    \throw   std::invalid_argument if the endpoints are not finite or relToler is not positive.
*/
template<typename Scaling>
inline double findMin(const IFunction& F, const Scaling& scaling, double xinit, double relToler)
{
    return unscale(scaling,
        findMin(ScaledFnc<Scaling>(scaling, F), 0, 1, scale(scaling, xinit), relToler));
}

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

/** Evaluate the higher derivatives of a function from its values and first derivatives
    at three consecutive points, using high-accuracy Hermite interpolation.
    \param[in] x0, x1, x2 are the abscissa points (assuming that x1 is between x0 and x2);
    \param[in] f0, f1, f2 are the function values at these points;
    \param[in] df0, df1, df2 are the function derivatives at these points;
    \param[out] der2, der3, der4, der5  are the estimated 2nd to 5th derivatives at x1.
*/
void hermiteDerivs(double x0, double x1, double x2,
    double f0, double f1, double f2, double df0, double df1, double df2,
    /*output*/ double& der2, double& der3, double& der4, double& der5);

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
    If the integration interval is (semi-)infinite, one may apply an appropriate transformation of
    the integration variable, provided by the templated class ScaledIntegrand<Scaling>.
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] relToler is the relative error tolerance;
    \param[out] error - if not NULL, output the error estimate in this variable;
    \param[out] numEval - if not NULL, output the number of function evaluations in this variable.
*/
double integrateAdaptive(const IFunction& F, double x1, double x2, double relToler, 
    double* error=NULL, int* numEval=NULL);

/** integrate a function on a finite interval, using a built-in fixed-order Gauss-Legendre rule
    without error estimate.
    \param[in] F  is the input function;
    \param[in] x1 is the lower end of the interval;
    \param[in] x2 is the upper end of the interval;
    \param[in] N  is the number of points in Gauss-Legendre quadrature, must be <= MAX_GL_ORDER
*/
double integrateGL(const IFunction& F, double x1, double x2, int N);

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

/// built-in GL integration tables are available for every N up to MAX_GL_TABLE
const int MAX_GL_TABLE = 20;
/// built-in GL integration tables are available for some (but not every) N up to MAX_GL_ORDER
const int MAX_GL_ORDER = 33;

/// list of all built-in integration rules:  points and weights
extern const double * const GLPOINTS [MAX_GL_ORDER+1];
extern const double * const GLWEIGHTS[MAX_GL_ORDER+1];


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
