/** \file    math_spline.h
    \brief   spline interpolation and penalized spline approximation
    \author  Eugene Vasiliev, Walter Dehnen
    \date    2011-2024

This module implements various interpolation and smoothing algorithms in 1,2,3 dimensions.

###  1-dimensional case.
Let x[i], y[i], i=0..M-1  be two one-dimensional arrays of 'coordinates' and 'values'.
An interpolating function f(x) passing through these points may be constructed by several
different methods:
- Linear interpolator is the most trivial one, uses only the values of function at grid nodes.
- Cubic spline with natural or clamped boundary conditions. In the most familiar case of
natural boundary condition the second derivatives of f(x) at the left- and rightmost grid point
are set to zero; in the case of clamped spline instead the value of first derivative at these
boundaries must be provided.
In both cases the function and its first two derivatives are continuous on the entire domain.
It is also possible to apply a regularizing filter after the spline initialization, which
may modify the computed derivatives in order to avoid non-monotonic behaviour of the interpolator.
This makes the spline only continuous up to its first derivative, but reduces wiggles.
- If in addition to the values of function at grid points, its first derivatives y'[i]
are also given at all points, then a quintic spline is the right choice for interpolation.
It provides piecewise 5-th degree polynomial interpolation with three continuous derivatives
on the entire domain.

###  2-dimensional case.
In this case, {x[i], y[j]}, i=0..Mx-1, j=0..My-1  are the pairs of coordinates of nodes
on a separable 2d grid covering a rectangular domain, and z[i,j] are the values of a 2d function.
The interpolant can be constructed using several different methods:
- (bi-)linear interpolation yields only a continuous function but not its derivatives,
however it is guaranteed to preserve minima/maxima of the original data.
- (bi-)cubic interpolation is constructed globally and provides piecewise 3rd degree polynomials
in each coordinate on each cell of the 2d grid, with first and second derivatives being continuous
across the entire domain. Again the natural or clamped boundary conditions may be specified.
- If in addition to the function values z[i,j]=f(x[i], y[j]), its derivatives along each
coordinate df/dx and df/dy are known at each node, then a two-dimensional quintic spline may be
constructed which provides globally three times continuously differentiable interpolant.

In both 1d and 2d cases, quintic splines are better approximating a smooth function,
but only if its derivatives at grid nodes are known with sufficiently high accuracy
(i.e. trying to obtain them by finite differences or from a cubic spline is useless).

###  3-dimensional case.
For separable 3d grids there are linear and natural cubic spline interpolators,
both constructed from the array of values at the nodes of 3d grid.

###  B-splines.
This is an alternative framework for interpolation.
The value of the function is represented as a sum of basis function with adjustable amplitudes.
These basis functions are piecewise polynomials of degree N with compact support,
spanning at most N+1 adjacent intervals between grid nodes.
In D dimensions, the basis is formed by tensor products of B-splines in each coordinate.
Thus the interpolation is local, i.e. is determined by the amplitudes of at most (N+1)^D basis
functions that are possibly non-zero at the given point;
however, to find the amplitudes that yield the given values of function at all nodes of
the D-dimensional grid, one needs to solve a global linear system for all nodes,
except the case of a linear (N=1) interpolator.
The B-spline framework is most useful when one needs to construct the interpolator from
a large array of values not on a regular grid (penalized spline smoothing and density estimation).
The amplitudes of B-splines of degree N=1 and 3 can be used to construct ordinary
linear and cubic interpolators, which are evaluated more efficiently than B-splines.

###  Penalized spline smoothing.
The approach based on B-spline basis functions can be used also for constructing
a smooth approximation to the set of 'measurements'.

For instance, in one-dimensional case  {x[p], y[p], p=0..P-1}  are the data points, and we seek
a smooth function that passes close to these points but does not necessarily through them,
and moreover has an adjustable tradeoff between smoothness and mean-square deviation from data.
This approximating function is given as a weighted sum of 1d B-splines of degree 3,
where the amplitudes of these basis functions are obtained from a linear system for
the given data points and given amount of smoothing.

The formulation in terms of 1d third-degree B-splines is equivalent to a clamped cubic spline,
which is more efficient to compute, so after obtaining the amplitudes they should be converted
to the values of interpolating function at its nodes, plus two endpoint derivatives, and used to
construct a cubic spline.

The same approach works in more than one dimension. The amplitudes of a 2d B-spline interpolator
may be converted into its values and derivatives, and used to construct a 2d quintic spline.
In the 3d case, the amplitudes are directly used with a cubic (N=3) 3d B-spline interpolator.

*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"

namespace math{

///@{
/// \name One-dimensional interpolation

/** Common interface for one-dimensional piecewise-polynomial interpolators */
class BaseInterpolator1d: public IFunction3Deriv, public IFunctionIntegral {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    BaseInterpolator1d() {};

    /** Partially initialize a 1d interpolator from the provided grid in x, which should be
        monotonically increasing (zero-length and single-element arrays are acceptable).
        All interpolators return NaN when empty (initialized by default or with zero-length arrays).
    */
    BaseInterpolator1d(const std::vector<double>& xvalues);

    /** check if the interpolator is initialized */
    bool empty() const { return xval.empty(); }

    /** return the array of interpolator nodes */
    const std::vector<double>& xvalues() const { return xval; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.empty() ? NAN : xval.front(); }

    /** return the upper end of definition interval */
    double xmax() const { return xval.empty() ? NAN : xval.back(); }

    /** return the integral of the interpolator function times x^n on the interval [x1..x2];
        the interpolator is set to zero outside the interval xmin..xmax.
        Implemented in the derived classes.
    */
    using IFunctionIntegral::integrate;

    /** return the integral of the interpolator function S times another function F(x)
        on the interval [x1..x2], i.e. \f$  \int_{x1}^{x2} S(y) F(y) dy  \f$;
        the function F is specified by the interface that provides the integral
        \f$  \int_{y1}^{y2} F(y) * y^n dy  \f$  on any interval y1..y2 for any integer n>=0.
        The interpolator S is set to zero outside the interval xmin..xmax, unlike the evalDeriv()
        method, which may (or may not) return a linearly extrapolated value.
    */
    virtual double integrate(double x1, double x2, const IFunctionIntegral& f) const = 0;

    /** compute the convolution of the interpolator S with the provided kernel K, i.e.
        \f$ \int_{xmin}^{xmax} S(y) K(x-y) dy \f$;  S is set to zero outside the range of xvalues.
    */
    virtual double convolve(double x, const IFunctionIntegral& kernel) const = 0;

    /** find all local minima and maxima on the given interval x1..x2 (if not specified, this
        means the entire interpolation grid). Endpoints of the interval are considered as extrema,
        and the resulting array is sorted in order of increase. 
        No values are reported outside the interpolation grid (i.e. if x1<xmin(),
        the first element of the returned array is xmin, not x1).
    */
    virtual std::vector<double> extrema(double x1=NAN, double x2=NAN) const = 0;

    /** find all locations within the interval x1..x2 (by default the entire interpolation grid)
        at which the interpolator attains the given value y (by default 0), 
        sorted in order of increase.
        No values are reported outside the interpolation grid, even if the interpolator itself
        may be linearly extrapolated beyond its grid boundaries and cross y somewhere outside.
    */
    virtual std::vector<double> roots(double y=0, double x1=NAN, double x2=NAN) const = 0;

protected:
    std::vector<double> xval;  ///< grid nodes
};


/** Class that provides a simple piecewise-linear interpolation for an array of x,y values */
class LinearInterpolator: public BaseInterpolator1d {
public:
    LinearInterpolator() : BaseInterpolator1d() {};

    LinearInterpolator(const std::vector<double>& xvalues, const std::vector<double>& fvalues);

    using BaseInterpolator1d::evalDeriv;

    /** compute the value of interpolator and optionally its derivatives at point x; if the input
        location is outside the definition interval, a linear extrapolation is performed.
    */
    virtual void evalDeriv(double x, double* value, double* der, double* der2, double* der3) const;

    virtual double integrate(double x1, double x2, int n=0) const;

    virtual double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    virtual double convolve(double x, const IFunctionIntegral& kernel) const;

    virtual std::vector<double> extrema(double x1=NAN, double x2=NAN) const;

    virtual std::vector<double> roots(double y=0, double x1=NAN, double x2=NAN) const;

private:
    std::vector<double> fval;  ///< values of the interpolator at grid nodes
};


/** Piecewise-cubic spline interpolator with one or two continuous derivatives.
    The spline is defined by the values and first derivatives at each grid point.
    There are several possible ways of constructing a cubic spline:
    - from the function values at the grid points: this results in a natural cubic spline,
    constructed from the requirement that the second derivative is continuous at all interior
    points, and zero at the endpoints.
    - the same plus one or two endpoint derivatives results in a clamped cubic spline,
    which has the same smoothness properties, but a prescribed first derivative at the endpoint
    (hence the second derivative is generally not zero).
    - from the amplitudes of a 3rd degree B-spline interpolator: there are Nnodes+2 independent
    basis functions, and the resulting clamped cubic spline is exactly equivalent to the B-spline.
    - from the user-provided values and first derivatives at each grid point:
    this resulting Hermite interpolator has, in general, only one continuous derivative.
    A quintic spline, constructed from the same amount of input data, typically results in
    a much better interpolation accuracy, and has three continuous derivatives.

    In the first case, there is a further possibility of applying a monotonicity-preserving filter:
    if the input function values are monotonic over a certain range of grid indices,
    the spline would preserve this property, with the exception of a grid segment adjacent to
    a local maximum or minimum, where the value of interpolant is also allowed to reach an extremum
    inside the segment. This is achieved by modifying the first derivatives where necessary;
    as a result, the spline loses the property of being twice continuously differentiable,
    but generally behaves in a more regular way, avoiding spurious spikes.
    This procedure also manifestly violates the linear nature of the interpolator,
    hence it is not used by default.
*/
class CubicSpline: public BaseInterpolator1d {
public:
    CubicSpline() : BaseInterpolator1d() {};

    /** Construct a natural cubic spline from the function values at grid points,
        or a clamped cubic spline from the amplitudes of a 2nd- or 3rd-degree B-spline interpolator.
        In both cases the interpolated curve is twice (or only once for a 2nd-degree B-spline)
        continuously differentiable, unless an optional monotonic regularization filter is applied.
        \param[in]  xvalues - the array of grid nodes, should be monotonically increasing.
        \param[in]  fvalues - depending on the length of this array, it may mean two things:
        in the first case, an array of function values at grid nodes (same length as xvalues);
        in the second case, an array of B-spline amplitudes, with length equal to 
        xvalues.size()+1 (for a 2nd-degree B-spline) or xvalues.size()+2 (for a 3rd-degree B-spline);
        in this case the remaining arguments are ignored.
        \param[in]  regularize (optional, default false) - whether to apply the regularization
        procedure to impose monotonicity constraints and reduce wiggliness.
        This may require a modification of the internally assigned first derivative, 
        so the resulting spline is no longer twice continuously differentiable (only once);
        however, in general this produces a better-behaving interpolator.
        \param[in]  derivLeft (optional) - first derivative of the spline at the leftmost grid node;
        a default value NaN means a natural boundary condition (zero second derivative).
        Note that if regularize=true, the provided boundary derivative may not be respected.
        \param[in]  derivRight - same for the rightmost grid node.     
        \throw  std::invalid_argument or std::length_error if grid is too small or not monotonic,
        or the array sizes are incorrect, or they contain invalid values (infinities, NaN).
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        bool regularize=false, double derivLeft=NAN, double derivRight=NAN);

    /** Construct a piecewise-cubic Hermite interpolator from the provided values of function
        values and first derivatives. The curve has only one continuous derivative.
        \param[in]  xvalues  - the array of grid nodes, should be monotonically increasing.
        \param[in]  fvalues  - the array of function values at grid nodes, same length as xvalues.
        \param[in]  fderivs  - the array of function derivatives at grid nodes, same length.
        \throw  std::invalid_argument or std::length_error if grid is too small or not monotonic,
        or the array sizes are incorrect, or they contain invalid values (infinities, NaN).
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& fderivs);

    using BaseInterpolator1d::evalDeriv;

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed.
    */
    virtual void evalDeriv(double x, double* val, double* der, double* der2, double* der3) const;

    virtual double integrate(double x1, double x2, int n=0) const;

    virtual double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    virtual double convolve(double x, const IFunctionIntegral& kernel) const;

    virtual std::vector<double> extrema(double x1=NAN, double x2=NAN) const;

    virtual std::vector<double> roots(double y=0, double x1=NAN, double x2=NAN) const;

private:
    std::vector<double> fval;  ///< values of the interpolator at grid nodes
    std::vector<double> fder;  ///< first derivatives of the interpolator at grid nodes
};


/** Piecewise-quintic spline interpolator with 2-4 continuous derivatives.
    On each grid segment, y(x) is a 5th order polynomial specified by 6 coefficients --
    values and the first two derivatives at two adjacent grid nodes.
    There are several possible ways of constructing a quintic spline:
    - from the function values at the grid points: this results in a natural quintic spline,
    which has four continuous derivatives at all interior nodes, and vanishing 2nd and 4th
    derivatives at endpoints. The accuracy of interpolation is comparable to that of the
    natural cubic spline, however, it provides a greater smoothness.
    - the same plus first derivatives at one or two endpoint results in a clamped quintic spline,
    which has the same smoothness properties, but a prescribed first derivative at the endpoint
    (hence the second derivative is generally not zero).
    - from the values and first derivatives at all grid points: this is the "standard" quintic
    spline, which has three continuous derivatives and is superior in interpolation quality
    to a Hermite cubic interpolator constructed from the same amount of input data.
    - from the values, first and second derivatives at all grid points:
    this results in a piecewise-quintic Hermite interpolator, in which the 3rd and 4th derivatives
    are not necessarily continuous, however, the overall interpolation accuracy is even better
    (provided, of course, that one can supply exactly computed first and second derivatives as input).
*/
class QuinticSpline: public BaseInterpolator1d {
public:
    QuinticSpline() : BaseInterpolator1d() {};

    /** Construct a "natural" or "clamped" quintic spline from the provided values of x and f(x),
        optionally with first derivatives at one or both endpoints.
        The 1st and 2nd derivatives are determined from the condition that the 3rd and the 4th
        derivatives are continuous at each interior node.
        At the endpoints, the 4th derivative is set to zero, and either the 1st derivative
        is set explicitly by derivLeft/derivRight (clamped), or else the 2nd derivative
        is set to zero (natural).
        \param[in]  xvalues  - the array of grid node, should be monotonically increasing.
        \param[in]  fvalues  - the array of function values at grid nodes, same length as xvalues.
        \param[in]  derivLeft (optional) - first derivative of the spline at the leftmost grid node;
        a default value NaN means a natural boundary condition (zero second derivative).
        \param[in]  derivRight - same for the rightmost grid node.
        \throw  std::invalid_argument or std::length_error if grid is too small or not monotonic,
        or the array sizes are incorrect, or they contain invalid values (infinities, NaN).
    */
    QuinticSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        double derivLeft=NAN, double derivRight=NAN);

    /** Construct a "standard" quintic spline from the provided values of x, f(x) and f'(x).
        The 2nd derivatives are initialized from the condition that the 3rd derivative is continuous
        at each interior node, and the 4th derivative is zero at the endpoints of the grid.
        \param[in]  xvalues  - the array of grid nodes, should be monotonically increasing.
        \param[in]  fvalues  - the array of function values at grid nodes, same length as xvalues.
        \param[in]  fderivs  - the array of function derivatives at grid nodes, same length.
        \throw  std::invalid_argument or std::length_error if grid is too small or not monotonic,
        or the array sizes are incorrect, or they contain invalid values (infinities, NaN).
    */
    QuinticSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& fderivs);

    /** Construct a piecewise-quintic Hermite interpolator from the provided values of x,
        f(x), f'(x) and f''(x) at all grid nodes. The higher derivatives may not be continuous
        between adjacent grid segments, however, the accuracy of interpolation is superior
        to all other variants (of course, if the exact second derivatives can be provided).
    */
    QuinticSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& fderivs, const std::vector<double>& fderivs2);

    using BaseInterpolator1d::evalDeriv;

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed.
    */
    virtual void evalDeriv(double x, double* val, double* der, double* der2, double* der3) const;

    virtual double integrate(double x1, double x2, int n=0) const;

    virtual double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    virtual double convolve(double x, const IFunctionIntegral& kernel) const;

    virtual std::vector<double> extrema(double x1=NAN, double x2=NAN) const;

    virtual std::vector<double> roots(double y=0, double x1=NAN, double x2=NAN) const;

private:
    std::vector<double> fval;  ///< values of the interpolator at grid nodes
    std::vector<double> fder;  ///< first  derivatives of the interpolator at grid nodes
    std::vector<double> fder2; ///< second derivatives of the interpolator at grid nodes
};


/** Cubic or quintic spline with logarithmic scalings of both the value and the argument:
    f(x) = exp( S( ln(x) ) ), where S is represented as a spline.
    Input x values must be strictly positive; however, input f values may contain zero or
    negative elements - these points will be excluded from log-spline construction and
    interpolated without log-scaling when evaluating the function.
    This class is not derived from BaseInterpolator1d, since it does not fully implement its
    interface (i.e., does not provide integration, convolution and root-finding methods).
*/
class LogLogSpline: public IFunction {
public:
    /// empty constructor
    LogLogSpline() : nonnegative(true) {}

    /** Construct a natural or clamped cubic spline from the function values
        and optionally two endpoint derivatives.
        \param[in] xvalues - the array of grid nodes, must be monotonic and strictly positive.
        \param[in] fvalues - function values at grid nodes, same length as xvalues.
        \param[in] derivLeft (optional) - function derivative (before log-scaling) at the leftmost
        node; default value NaN means natural boundary condition (zero second derivative).
        \param[in] derivRight - same for the rightmost node.
    */
    LogLogSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        double derivLeft=NAN, double derivRight=NAN);

    /** Construct a quintic spline from function values and derivatives.
        \param[in]  xvalues  - the array of grid nodes, must be monotonic and strictly positive.
        \param[in]  fvalues  - function values at grid nodes, same length as xvalues.
        \param[in]  fderivs  - function derivatives at grid nodes, same length.
    */
    LogLogSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& fderivs);

    virtual void evalDeriv(double x, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

    /** return the number of derivatives that the interpolator provides */
    virtual unsigned int numDerivs() const { return 2; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

private:
    std::vector<double> xval;     ///< grid nodes
    std::vector<double> fval;     ///< values of the original function at grid nodes
    std::vector<double> fder;     ///< first derivatives of the original function at grid nodes
    std::vector<double> logxval;  ///< log-scaled coordinate
    std::vector<double> logfval;  ///< log-scaled function values
    std::vector<double> logfder;  ///< first derivatives of log-log scaled function at grid nodes
    std::vector<double> logfder2; ///< second derivatives of log-log function at grid nodes
    bool nonnegative;             ///< if the input was non-negative, the interpolated values will be >=0
};


/** One-dimensional B-spline interpolator class, self-contained.
    The value of interpolant is given by a weighted sum of components:
    \f$  f(x) = \sum_n  A_n  B_n(x) ,  0 <= n < numComp  \f$,
    where A_n are the amplitudes and B_n are basis functions, which are piecewise polynomials
    (B-splines) of degree N>=1 that are nonzero on a finite interval between at most N+1 grid
    points. The sum of all basis functions is always unity, and each function is non-negative.
    The interpolation is local - at any point, at most (N+1) basis functions are non-zero.
    The total number of components is (N_x+N-1), where N_x is the size of the grid.

    B-spline machinery is provided by two formally unrelated classes:
    this one (`BsplineWrapper`) implements the BaseInterpolator1d interface and contains both
    the interpolation grid and the array of amplitudes, and the sister class `BsplineInterpolator1d`
    only manages the basis functions for interpolation, but does not contain the amplitudes.

    \tparam  N is the degree of 1d B-splines: possible values are
    N=0 (rectangular histogram), N=1 (linear interpolator), N=2, and N=3 (clamped cubic spline).
*/
template<int N>
class BsplineWrapper: public BaseInterpolator1d {
public:
    /** Initialize a 1d interpolator from the provided arrays of grid nodes in x.
        There is no work done in the constructor apart from checking the validity of parameters.
        \param[in] xval are the grid nodes sorted in increasing order, must have at least 2 elements.
        \throw std::invalid_argument if the grid is invalid.
    */
    BsplineWrapper(const std::vector<double>& xval, const std::vector<double>& ampl);

    using BaseInterpolator1d::evalDeriv;

    /** compute the value of B-spline and optionally its derivatives at point x;
        if the point is outside the definition interval, return NaN (no extrapolation).
    */
    virtual void evalDeriv(double x, double* value, double* der, double* der2, double* der3) const;

    virtual double integrate(double x1, double x2, int n=0) const;

    virtual double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    virtual double convolve(double x, const IFunctionIntegral& kernel) const;

    virtual std::vector<double> extrema(double x1=NAN, double x2=NAN) const;

    virtual std::vector<double> roots(double y=0, double x1=NAN, double x2=NAN) const;

private:
    std::vector<double> ampl;   ///< array of B-spline amplitudes
};


/** One-dimensional B-spline interpolator class, non-self-contained.
    Unlike the sister class `BsplineWrapper`, this one does not contain the amplitudes of components,
    it only manages the basis functions for interpolation - e.g., `nonzeroBsplines()` computes 
    the values of all possibly non-zero basis functions at the given point, the method `eval()`
    implementing IFunctionNdimAdd interface computes the values of all numComp basis functions
    at the given point, and `interpolate()` computes the value of interpolant at the given
    point from the provided array of amplitudes, summing only over the non-trivial B-splines.

    \tparam  N is the degree of 1d B-splines: possible values are
    N=0 (rectangular histogram), N=1 (linear interpolator), N=2, and N=3 (clamped cubic spline).
*/
template<int N>
class BsplineInterpolator1d: public IFunctionNdimAdd {
public:

    /** Initialize a 1d interpolator from the provided arrays of grid nodes in x.
        There is no work done in the constructor apart from checking the validity of parameters.
        \param[in] xval are the grid nodes sorted in increasing order, must have at least 2 elements.
        \throw std::invalid_argument if the grid is invalid.
    */
    explicit BsplineInterpolator1d(const std::vector<double>& xval);

    /** Compute the values of all potentially non-zero interpolating basis functions or their
        derivatives at the given point, needed to obtain the value of interpolant f(x) at this point.
        \param[in]  x  is the point (which may lie outside the grid);
        \param[in]  derivOrder  is the order of derivative D of basis functions (0 <= D <= N);
        \param[out] values  is the array of (N+1) values of interpolation B-splines or their derivs.
        The sum of values of all B-splines is always 1, and values are non-negative.
        The special case when one of these weigths is 1 and the rest are 0 occurs at the corners of
        the interval, or, for a linear intepolator (N=1) also at all grid nodes,
        and means that the value of interpolant `f` is equal to the single element of the amplitudes
        array, which in the case N=1 should contain the values of the original function at grid nodes.
        If any of the coordinates of input point falls outside grid boundaries in the respective
        dimension, all weights are zero.
        \return  the index of the first basis function for which the values are stored.
    */
    unsigned int nonzeroComponents(double x, unsigned int derivOrder, double values[]) const;

    /** Compute the values of all numComp basis functions at the given point.
        \param[in]  x is the point (which may lie outside the grid);
        \param[out] values will contain the values of all basis functions at the given point
        (many of them may be zero); must point to an existing array of length numComp
        (no range check performed!).
        If the input point is outside the grid, all values will contain zeros.
    */
    virtual void eval(const double* x, double values[]) const;

    /** Add the values of non-zero B-splines at the given point, multiplied by the provided factor
        `mult`, to the array of accumulated values (similar to `eval()` but without zeroing down
        the remaining entries, only adding to the relevant ones).
    */
    virtual void addPoint(const double* x, double mult, double values[]) const;

    /** Compute the value of the interpolant `f` or its derivative at the given point.
        \param[in] x is the point (which may lie outside the grid);
        \param[in] amplitudes is the array of numComp amplitudes of each basis function;
        \param[in] derivOrder is the order of derivative (default is 0, i.e. the function itself);
        \return    the weighted sum of all potentially non-zero basis functions or their derivatives
        at this point, multiplied by their respective amplitudes, or 0 if the input location is outside
        the grid definition region.
        \throw  std::length_error if the length of `amplitudes` does not correspond to numComp.
    */
    double interpolate(double x, const std::vector<double> &amplitudes,
        const unsigned int derivOrder = 0) const;

    /** Compute the integral of f(x) * x^n over the interval [x1..x2], where f(x) is given
        by the weighted sum of basis functions, with weights provided in the array of amplitudes.
        \param[in]  x1  is the lower limit of integration;
        \param[in]  x2  is the upper limit;
        \param[in]  amplitudes  is the array of numComp amplitudes of basis functions;
        \param[in]  n   is the power-law index for the optional multiplier x^n;
        \return     the value of integral (exact up to machine precision).
        \throw  std::length_error if the length of `amplitudes` does not correspond to numComp.
    */
    double integrate(double x1, double x2, const std::vector<double> &amplitudes, int n=0) const;

    /** Construct a B-spline representation of the derivative of a B-spline interpolator.
        \param[in]  amplitudes  is the array of amplitudes of a B-spline of degree N;
        \return  an array of amplutides for a different B-spline of degree N-1 constructed
        for the same x-nodes and representing the derivative of the original B-spline;
        the size of this array is one less than the input array.
        \throw  std::length_error if the length of `amplitudes` does not correspond to numComp.
    */
    std::vector<double> deriv(const std::vector<double> &amplitudes) const;

    /** Construct a B-spline representation of the antiderivative of a B-spline interpolator.
        \param[in]  amplitudes  is the array of amplitudes of a B-spline of degree N;
        \return  an array of amplutides for a different B-spline of degree N+1 constructed
        for the same x-nodes and representing the antiderivative of the original B-spline;
        the size of this array is one element larger than the input array, and the 0th element is zero.
        \throw  std::length_error if the length of `amplitudes` does not correspond to numComp.
    */
    std::vector<double> antideriv(const std::vector<double> &amplitudes) const;

    /** The dimensions of interpolator (1) */
    virtual unsigned int numVars() const { return 1; }

    /** The number of components (basis functions) */
    virtual unsigned int numValues() const { return numComp; }

    /** return the boundaries of grid definition region */
    double xmin() const { return xval.front(); }
    double xmax() const { return xval.back();  }

    /** return the nodes of the grid used for interpolation */
    const std::vector<double>& xvalues() const { return xval; }

private:
    const unsigned int numComp;      ///< number of basis functions
    const std::vector<double> xval;  ///< grid nodes
};


/** Finite-element analysis using one-dimensional B-splines of degree N.

    This class provides methods for constructing discretized representations of any function using
    the basis set of B-splines or their derivatives; this amounts to integrating the original
    function multiplied by basis functions, which is performed on a pre-determined auxiliary grid
    of points (a few per each segment of the B-spline grid). The input function is evaluated at
    these points, and the values or derivatives of basis functions are pre-computed on the same
    grid during the construction.
*/
template<int N>
class FiniteElement1d {
public:
    /** Construct the object from the B-spline interpolator
        and initialize the arrays containing the values and derivatives of basis functions
    */
    explicit FiniteElement1d(const BsplineInterpolator1d<N>& interp);

    /** Compute the projection of a function f(x) onto the basis -- the vector of integrals
        of input function weighted with each of the basis functions B_n or their derivatives:
        \f$ v_n = \int f(x) B_n^{(D)}(x) dx \f$.
        \param[in]  fncValues   is the array of pre-computed values of input function f(x)
        at the grid of points returned by `integrPoints()`.
        \param[in]  derivOrder  is the order `D` of derivatives of basis functions (0 <= D <= N).
        \return  the vector v_n  of length `interp.numValues()` (number of basis functions).
        \throw   std::length_error if the length of fncValues differs from integrNodes.
    */
    std::vector<double> computeProjVector(
        const std::vector<double>& fncValues,
        unsigned int derivOrder=0) const;

    /** Compute the matrix of products of basis functions or their derivatives 
        weighted with input function f(x):
        \f$  A_{mn} = \int f(x) B_m^{(p)}(x) B_n^{(q)}(x) dx  \f$.
        \param[in]  fncValues    is the array of pre-computed values of input function f(x)
        at the grid of points returned by `integrPoints()`;
        it may be an empty array, meaning that f(x)=1 identically.
        \param[in]  derivOrderP  is the order `p` of derivatives of the row-wise basis functions.
        \param[in]  derivOrderQ  is the order `q` of derivatives of the column-wise basis functions.
        \return  the band matrix A_{mn}: a square matrix with size `interp.numValues()` and
        at most 2*N+1 nonzero values around the main diagonal in each row.
        \throw  std::length_error if the length of fncValues differs from integrNodes.
    */
    BandMatrix<double> computeProjMatrix(
        const std::vector<double>& fncValues = std::vector<double>(),
        unsigned int derivOrderP=0,
        unsigned int derivOrderQ=0) const;

    /** Add the values or derivatives of all B-splines at the given point, convolved with the given
        kernel and multiplied by the provided factor `mult`, to the array of accumulated values
        (similar to the method BsplineInterpolator1d::addPoint(), but with convolution and
        generalized to use any derivative of the basis functions).
        \param[in]  x  is the input point.
        \param[in]  mult  is the overall multiplicative factor for the output values.
        \param[in]  kernel  is the convolution kernel (e.g., math::Gaussian).
        \param[in]  derivOrder  is the order of derivatives of the basis functions (0 means values).
        \param[in,out]  values  is the output (accumulation) array of length interp.numValues(),
        whose elements will be incremented by the computed convolution integrals of all basis functions.
        \tparam SIGN  specifies the choice of the argument for the convolution kernel:
        if SIGN==True, the output values are \int dy B_i(y) K(x-y), otherwise  \int dy B_i(y) K(y-x).
    */
    template<bool SIGN>
    void addPointConv(const double x, double mult, const IFunctionIntegral& kernel,
        unsigned int derivOrder, double values[]) const;

    /** Compute the projection of a convolution of a function f(x) with the given kernel onto the basis:
        \f$  c_n = \int dx [ \int dy f(x) K(x-y) B_n^{(D)}(y) dy ]  \f$.
        Note that this method uses the values of input function collected at the pre-defined integration
        points (their number is ~ numBasisFnc * (N+1) );
        if the function is already represented by a B-spline interpolator, then the amplitudes of the
        convolved function are obtained as a product of the convolution matrix returned by 
        `computeConvMatrix()` and the vector of amplitudes of the B-spline interpolator.
        \param[in]  fncValues   is the array of pre-computed values of input function f(x)
        at the grid of points returned by `integrPoints()`.
        \param[in]  kernel      is the convolution kernel K.
        \param[in]  derivOrder  is the order `D` of derivatives of basis functions (0 <= D <= N).
        \return  the vector c_n  of length `interp.numValues()` (number of basis functions).
        \throw   std::length_error if the length of fncValues differs from integrNodes.
    */
    std::vector<double> computeConvVector(
        const std::vector<double>& fncValues,
        const IFunctionIntegral& kernel,
        unsigned int derivOrder=0) const;

    /** Compute the convolution matrix for the given kernel K:
        \f$  K_{mn} = \int dx \int dy  B_m^{(p)}(x) B_n^{(q)}(y) K(x-y)  \f$.
        If an interpolated function f(x) is represented by the vector of its basis-set amplitudes f_i,
        then the convolution of f with the kernel K (the integral  \f$ g(y) = \int f(x) K(x-y) dx \f$)
        may be approximated by the vector of amplitudes g_j:  g = A^{-1} K f,
        where A is the band matrix returned by `computeProjMatrix()`,
        and K is the convolution matrix returned by this routine.
        If the convolution kernel is (close to) a delta function, the matrix K is identical to A.
        \param[in]  kernel  is the convolution kernel (e.g., math::Gaussian).
        \param[in]  derivOrderP  is the order `p` of derivatives of the row-wise basis functions.
        \param[in]  derivOrderQ  is the order `q` of derivatives of the column-wise basis functions.
        \return the square matrix K_{mn} with size `interp.numValues()`.
    */
    Matrix<double> computeConvMatrix(
        const IFunctionIntegral& kernel,
        unsigned int derivOrderP=0,
        unsigned int derivOrderQ=0) const;

    /** Compute the amplitudes of B-spline interpolator for the input function f(x).
        This is a convenience function that performs the following steps:
        1) collect the function values at the points of the integration grid;
        2) find \f$ v_i = \int f(x) B_i(x) dx \$ using `computeProjVector()`;
        3) find \f$ A_{ij} = \int B_i(x) B_j(x) dx \$ using `computeProjMatrix()`;
        4) solve the linear system `A c = v`  to find the vector of amplitudes `c`.
        One can then use `interp.interpolate(x, c)` to compute the approximation for f(x).
        If one needs to construct interpolators for several different functions,
        it would be more efficient to store and re-use the matrix A_ij, i.e.,
        perform these steps manually. Similarly, in the case of weighted inner product,
        when the integrals for v_i and A_ij have an extra weight factor w(x),
        this modified sequence of steps could also be easily carried out manually.
        \param[in]  F  is the input function.
        \return  the vector of amplitudes `c` that can be used as the argument to B-spline
        `interpolate()` method as described above.
    */
    std::vector<double> computeAmplitudes(const IFunction& F) const;

    /// the B-spline interpolator constructed for the input grid
    const BsplineInterpolator1d<N> interp;

    /// coordinates of points used to integrate any function on the grid
    const std::vector<double>& integrPoints() const { return integrNodes; }

private:
    /// we use the order of Gauss-Legendre quadrature equal to N+1, so that the integration of
    /// polynomials up to degree 2N+1 (e.g., products of two basis functions) is exact
    static const int GLORDER = N+1;
    std::vector<double> integrNodes;   ///< nodes of the integration grid
    std::vector<double> integrWeights; ///< weights associated with the nodes of the integration grid
    std::vector<double> bsplValues;    ///< pre-computed values and all derivatives of basis functions
};


///@}
/// \name Two-dimensional interpolation
///@{

/** Generic two-dimensional interpolator class */
class BaseInterpolator2d: public IFunctionNdim {
public:
    BaseInterpolator2d() {}

    /** Initialize a 2d interpolator from the provided values of x, y and f.
        \param[in] xvalues - the grid in the x dimension, must be monotonically increasing
        and have at least 2 elements.
        \param[in] yvalues - same for the y dimension.
        \param[in] fvalues - the 2d array of function values at grid nodes,
        with the following indexing convention:  f(i, j) = f(x[i], y[j]).
        \throw std::length_error or std::invalid_argument if the array sizes are incorrect,
        or they contain infinity or NaN elements.
    */
    BaseInterpolator2d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const IMatrixDense<double>& fvalues);

    /** compute the value of the interpolating function and optionally its derivatives at point x,y;
        if the input location is outside the definition region, the result is NaN.
        Any combination of value, first and second derivatives is possible:
        if any of them is not needed, the corresponding pointer should be set to NULL.
    */
    virtual void evalDeriv(const double x, const double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const = 0;

    /** shortcut for computing the value of spline */
    double value(const double x, const double y) const {
        double v;
        evalDeriv(x, y, &v);
        return v;
    }

    /** IFunctionNdim interface */
    virtual void eval(const double vars[], double values[]) const {
        evalDeriv(vars[0], vars[1], values);
    }
    virtual unsigned int numVars()   const { return 2; }
    virtual unsigned int numValues() const { return 1; }

    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }

    /** check if the interpolator is initialized */
    bool empty() const { return fval.empty(); }

    /** return the array of grid nodes in x-coordinate */
    const std::vector<double>& xvalues() const { return xval; }

    /** return the array of grid nodes in y-coordinate */
    const std::vector<double>& yvalues() const { return yval; }

protected:
    std::vector<double> xval, yval;  ///< grid nodes in x and y directions
    std::vector<double> fval;        ///< flattened row-major 2d array of f values
};


/** Two-dimensional bilinear interpolator */
class LinearInterpolator2d: public BaseInterpolator2d {
public:
    LinearInterpolator2d() : BaseInterpolator2d() {}

    /** Initialize a bilinear 2d interpolator from the provided values of x, y and f.
        \param[in] xvalues - the grid in the x dimension, must be monotonically increasing
        and have at least 2 elements.
        \param[in] yvalues - same for the y dimension.
        \param[in] fvalues - the 2d array of function values at grid nodes,
        with the following indexing convention:  f(i, j) = f(x[i], y[j]).
        \throw std::length_error or std::invalid_argument if the array sizes are incorrect,
        or they contain infinity or NaN elements.
    */
    LinearInterpolator2d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const IMatrixDense<double>& fvalues)
    :
        BaseInterpolator2d(xvalues, yvalues, fvalues) {}

    /** Compute the value and/or derivatives of the interpolator;
        note that for the linear interpolator the 2nd derivatives are always zero. */
    virtual void evalDeriv(double x, double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;
};


/** Two-dimensional cubic spline */
class CubicSpline2d: public BaseInterpolator2d {
public:
    CubicSpline2d() : BaseInterpolator2d() {}

    /** Initialize a 2d cubic spline from the provided values of x, y and f.
        \param[in] xvalues - the grid in x dimension, must be monotonically increasing
        and have at least 2 elements.
        \param[in] yvalues - same for y dimension.
        \param[in] fvalues - the 2d array of function values at grid nodes,
        with the following indexing convention:  f(i, j) = f(x[i], y[j]).
        \param[in] regularize (optional, default false) - whether to apply a monotonic regularization
        filter to reduce wiggliness (in this case the spline coefficients are no longer linear 
        functions of input values, and the provided boundary derivatives may not be respected).
        \param[in] deriv_xmin (optional) - function derivative at the leftmost x node
        (same value for the entire side of the rectangle);
        default value NaN means a natural boundary condition.
        \param[in] deriv_xmax - same for the rightmost x node.
        \param[in] deriv_ymin - same for the leftmost y node;
        \param[in] deriv_ymax - same for the rightmost y node.
        \throw std::length_error or std::invalid_argument if the array sizes are incorrect,
        or they contain infinity or NaN elements.
    */
    CubicSpline2d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const IMatrixDense<double>& fvalues,
        bool regularize=false,
        double deriv_xmin=NAN, double deriv_xmax=NAN,
        double deriv_ymin=NAN, double deriv_ymax=NAN);

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(double x, double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;

private:
    /// flattened 2d arrays of derivatives in x and y directions, and mixed 2nd derivatives
    std::vector<double> fx, fy, fxy;
};


/** Two-dimensional quintic spline */
class QuinticSpline2d: public BaseInterpolator2d {
public:
    QuinticSpline2d() : BaseInterpolator2d() {}

    /** Initialize a 2d quintic spline from the provided values of x, y, f(x,y), df/dx, df/dy,
        and (optionally) the mixed second derivative d2f/dxdy.
        \param[in] xvalues - the grid in x dimension, must be monotonically increasing
        and have at least 2 elements.
        \param[in] yvalues - same for y dimension.
        \param[in] fvalues - the 2d array of function values at grid nodes,
        with the following indexing convention:  f(i, j) = f(x[i], y[j]).
        \param[in] dfdx - the 2d array of partial derivatives df/dx at grid nodes.
        \param[in] dfdy - the 2d array of partial derivatives df/dy at grid nodes.
        \param[in] d2fdxdy - the 2d array of mixed second derivatives d2f/dxdy at grid nodes
        (optional; if provided, this generally improves the accuracy of interpolation).
    */
    QuinticSpline2d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const IMatrixDense<double>& fvalues,
        const IMatrixDense<double>& dfdx,
        const IMatrixDense<double>& dfdy,
        const IMatrixDense<double>& d2fdxdy = Matrix<double>());

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(double x, double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;

private:
    /// flattened 2d arrays of various derivatives
    std::vector<double> fx, fy, fxx, fxy, fyy, fxxy, fxyy, fxxyy;
    void setupWoutMixedDeriv(size_t xsize, size_t ysize);
    void setupWithMixedDeriv(size_t xsize, size_t ysize);
};


///@}
/// \name Three-dimensional interpolation
///@{

/** Trilinear interpolator */
class LinearInterpolator3d: public IFunctionNdim {
public:
    LinearInterpolator3d() {}

    /** Construct the interpolator from the values at the nodes of a 3d grid.
        \param[in]  xvalues is the grid in x dimension with size nx>=2;
        \param[in]  yvalues is the grid in y dimension with size ny>=2;
        \param[in]  zvalues is the grid in z dimension with size nz>=2;
        \param[in]  fvalues is the flattened array of function values:
        fvalues[(i*ny + j) * nz + k] = f(xvalues[i], yvalues[j], zvalues[k]).
        \throw std::length_error if the grid sizes are incorrect,
        or std::invalid_argument if the grids are not monotonic, or contain invalid values.
    */
    LinearInterpolator3d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const std::vector<double>& zvalues,
        const std::vector<double>& fvalues);

    /** Compute the value of the interpolator at the given point;
        if it is outside the grid boundaries, return NaN.
    */
    double value(double x, double y, double z) const;

    /** check if the interpolator is initialized */
    bool empty() const { return fval.empty(); }

    // IFunctionNdim interface
    virtual void eval(const double point[3], double *val) const {
        *val = value(point[0], point[1], point[2]); }

    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }

private:
    std::vector<double> xval, yval, zval;  ///< grid nodes in x, y and z directions
    std::vector<double> fval;              ///< flattened 3d array of function values at 3d grid nodes
};


/** Three-dimensional cubic spline with natural boundary conditions */
class CubicSpline3d: public IFunctionNdim {
public:
    CubicSpline3d() {}

    /** Construct the spline interpolator from the values at the nodes of a 3d grid,
        or from the amplitudes of a BsplineInterpolator3d of degree N=3.
        \param[in]  xvalues  is the grid in x dimension with size nx>=2;
        \param[in]  yvalues  is the grid in y dimension with size ny>=2;
        \param[in]  zvalues  is the grid in z dimension with size nz>=2;
        \param[in]  fvalues is a flattened array with two alternative meanings.
        a) the array of function values taken at the nodes of the 3d grid:
        fvalues[(i*ny + j) * nz + k] = f(xvalues[i], yvalues[j], zvalues[k]);
        b) the array of B-spline amplitudes - in this context, the dimensions of
        the grid of amplitudes should be (nx+2) * (ny+2) * (nz+2).
        \param[in]  regularize (optional, default false) - whether to apply a monotonic
        regularization filter (only for the case "a").
        \throw std::length_error if the grid sizes are incorrect,
        or std::invalid_argument if the coordinate grids are not monotonic, or any of
        the input arrays contains invalid values.
    */
    CubicSpline3d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const std::vector<double>& zvalues,
        const std::vector<double>& fvalues,
        bool regularize=false);

    /** Compute the value of the interpolator at the given point;
        if it is outside the grid boundaries, return NaN.
    */
    double value(double x, double y, double z) const;

    /** check if the interpolator is initialized */
    bool empty() const { return fval.empty(); }

    // IFunctionNdim interface
    virtual void eval(const double point[3], double *val) const {
        *val = value(point[0], point[1], point[2]); }

    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }

private:
    std::vector<double> xval, yval, zval;  ///< grid nodes in x, y and z directions
    /// values and various derivatives of the function at 3d grid nodes
    std::vector<double> fval, fx, fy, fz, fxy, fxz, fyz, fxyz;
};


/** Three-dimensional B-spline interpolator class.
    The value of interpolant is given by a weighted sum of components:
    \f$  f(x,y,z) = \sum_n  A_n  B_n(x,y,z) ,  0 <= n < numComp  \f$,
    where A_n are the amplitudes and B_n are 3d basis functions, obtained by a tensor product of 
    three 1d interpolating basis functions, which are piecewise polynomials (B-splines) of degree
    N>=1 that are nonzero on a finite interval between at most N+1 grid points in each dimension.
    The interpolation is local - at any point, at most (N+1)^3 basis functions are non-zero.
    The total number of components numComp = (N_x+N-1) * (N_y+N-1) * (N_z+N-1), where N_x,N_y,N_z
    are the grid sizes in each dimension; the correspondence between the triplet of indices {i,j,k}
    and the index of the component n is given by two functions `indComp` and `decomposeIndComp`.
    For a linear interpolator (N=1) numComp is equal to the total number of nodes in the 3d grid,
    and the value of interpolant at each node of 3d grid is equal to the amplitude of
    the correspoding component; in other words, if we denote the grid nodes as X[i],Y[j],Z[k],
    0<=i<N_x, etc., then  f(X[i], Y[j], Z[k]) = A[indComp(i,j,k)].
    However, for higher-order interpolators (N>1) there is no 1:1 correspondence between
    the amplitudes of components and the values of interpolant at grid points (like a Bezier
    curve does not pass through its control points), and the number of components is larger
    than the total number of nodes.
    This class does not itself hold the amplitudes of components, it only manages
    the basis functions for interpolation - e.g., `nonzeroBsplines()` computes the values
    of all possibly non-zero basis functions at the given point, the method `eval()`
    implementing IFunctionNdim interface computes the values of all numComp basis functions
    at the given point, and `interpolate()` computes the value of interpolant at the given
    point from the provided array of amplitudes, summing only over the non-trivial B-splines.
    The sum of all basis functions is always unity, and each function is non-negative.
    \tparam  N is the degree of 1d B-splines
    (N=1 - linear, N=3 - cubic, other cases are not implemented).
    Note that the classes LinearInterpolator3d and CubicSpline3d perform the same task
    more efficiently, and also store all required arrays to compute the interpolant.
*/
template<int N>
class BsplineInterpolator3d: public IFunctionNdim {
public:
    /** Initialize a 3d interpolator from the provided 1d arrays of grid nodes in x, y and z
        dimensions.
        \param[in] xvalues, yvalues, zvalues are the nodes of grid in each dimension,
        sorted in increasing order, must have at least 2 elements.
        There is no work done in the constructor apart from checking the validity of parameters.
        \throw std::invalid_argument if the 1d grids are invalid.
    */
    BsplineInterpolator3d(
        const std::vector<double>& xvalues,
        const std::vector<double>& yvalues,
        const std::vector<double>& zvalues);

    /** Compute the values of all potentially non-zero interpolating basis functions
        at the given point, needed to obtain the value of interpolant f(x,y,z) at this point.
        \param[in]  point is the array of three coordinates of the point;
        \param[out] leftIndices is the array of indices of leftmost elements used for B-spline
        interpolation in each of 3 dimensions: N+1 consecutive elements are used per dimension;
        \param[out] values  is the array of (N+1)^3 weights (values of 3d interpolation B-splines)
        that must be multiplied by the amplitudes to compute the interpolant, namely:
        \f$  f(x,y,z) = \sum_{i=0}^N \sum_{j=0}^N \sum_{k=0}^N
             A[indComp(i+l[0], j+l[1], k+l[2])  \times  values[(i * (N+1) + j) * (N+1) + k]  \f$,
        where `l` is the shortcut for `leftIndices`, and `A` is the flattened array of amplitudes.
        The sum of weights of all B-splines is always 1, and weights are non-negative.
        The special case when one of these weigths is 1 and the rest are 0 occurs at the corners of
        the cube (the definition region), or, for a linear intepolator (N=1) also at all grid
        nodes, and means that the value of interpolant `f` is equal to the single element of the
        amplitudes array, which in the case N=1 should contain the values of the original function
        at grid nodes.
        If any of the coordinates of input point falls outside grid boundaries in the respective
        dimension, all weights are zero (except if point is NaN, in which case the result is also NaN).
    */
    void nonzeroComponents(const double point[3], unsigned int leftIndices[3], double values[]) const;

    /** Return the value of a single component (interpolation basis function) at the given point.
        Note that it is much more efficient to compute all possibly nonzero components at once,
        by calling `nonzeroComponents()`, than calling this function separately for each indComp;
        alternatively, `eval()` returns the values of all numComp (empty and non-empty) components.
        \param[in]  point  is the array of three coordinates of the point;
        \param[in]  indComp  is the index of component (0 <= indComp < numComp);
        \return  the value of a single interpolation basis function at this point,
        or zero if the point is outside the grid definition region;
        \throw std::out_of_range if indComp is out of range.
    */
    double valueOfComponent(const double point[3], unsigned int indComp) const;

    /** Report the region of 3d space where the interpolation basis function
        of the given component is nonzero.
        \param[in]  indComp is the index of component;
        \param[out] xlower  are the coordinates of the lower corner of the region;
        \param[out] xupper  same for the upper corner;
        \throw std::out_of_range error if indComp >= numComp.
    */
    void nonzeroDomain(unsigned int indComp, double xlower[3], double xupper[3]) const;

    /** Compute the values of all numComp basis functions at the given point.
        \param[in]  point is the array of three coordinates of the point;
        \param[out] values will contain the values of all basis functions at the given point
        (many of them may be zero); must point to an existing array of length numComp
        (no range check performed!).
        If the input point is outside the grid, all values will contain zeros.
    */
    virtual void eval(const double point[3], double values[]) const;

    /** Compute the value of the interpolant `f` at the given point.
        \param[in] point is the array of three coordinates of the point;
        \param[in] amplitudes is the array of numComp amplitudes of each basis function,
        provided by the caller;
        \return    the weighted sum of all potentially non-zero basis functions at this point,
        multiplied by their respective amplitudes, or 0 if the input location is outside
        the grid definition region.
        \throw std::length_error if the length of `amplitudes` does not correspond to numComp.
    */
    double interpolate(const double point[3], const std::vector<double> &amplitudes) const;

    /** The dimensions of interpolator (3) */
    virtual unsigned int numVars()   const { return 3; }

    /** The number of components (3d interpolation basis functions) */
    virtual unsigned int numValues() const { return numComp; }

    /** Return the index of element in the flattened 3d array of function values
        associated with the given triplet of indices in each of the 1d coordinate grids.
        The indices must satisfy  0 <= ind_x < N_x, 0 <= ind_y < N_y, 0 <= ind_z < N_z,
        where N_x=xval.size()+N-1, N_y=yval.size()+N-1, N_z=zval.size()+N-1,
        are the number of basis elements in each dimension;
        however, no range check is performed on the input indices!
    */
    unsigned int indComp(unsigned int ind_x, unsigned int ind_y, unsigned int ind_z) const {
        return index3d(ind_x, ind_y, ind_z, yval.size()+N-1, zval.size()+N-1);
    }

    /** return the index of element in the flattened 3d array with the given dimensions */
    static unsigned int index3d(unsigned int ind_x, unsigned int ind_y, unsigned int ind_z,
        unsigned int N_y, unsigned int N_z)
    {
        return (ind_x * N_y + ind_y) * N_z + ind_z;
    }

    /** Decompose the index of element in the flattened 3d array of function values
        into the three indices in each of the 1d coordinate grids (no range check is performed!)
    */
    void decomposeIndComp(const unsigned int indComp, unsigned int indices[3]) const {
        const unsigned int N_y = yval.size()+N-1, N_z = zval.size()+N-1;
        indices[2] = indComp % N_z,
        indices[1] = indComp / N_z % N_y,
        indices[0] = indComp / N_z / N_y;
    }

    /** return the boundaries of grid definition region */
    double xmin() const { return xval.front(); }
    double xmax() const { return xval.back();  }
    double ymin() const { return yval.front(); }
    double ymax() const { return yval.back();  }
    double zmin() const { return zval.front(); }
    double zmax() const { return zval.back();  }

    /** return the (sparse) matrix of roughness penalties */
    SparseMatrix<double> computeRoughnessPenaltyMatrix() const;

private:
    std::vector<double> xval, yval, zval;  ///< grid nodes in x, y and z directions
    const unsigned int numComp;            ///< total number of components
};


/** Fill the array of amplitudes for a 3d interpolator by collecting the values of the source
    function F at the nodes of 3d grid.
    For the case N=1, the values of source function at grid nodes are identical to the amplitudes,
    but for higher-degree interpolation this is not the case, and the amplitudes are obtained by
    solving a linear system with the size numComp*numComp, where numComp ~ (grid_size_in_1d+N-1)^3.
    This requires O(numComp^2) operations and could be prohibitively expensive if numComp > ~10^4,
    hence this routine should be used only for small grid sizes.
    To construct a 3d cubic spline from the function values only, one should use the CubicSpline3d
    class, which performs the initialization in O(numComp) operations, and also evaluates
    the interpolant several times faster than the equivalent B-spline, so for all practical
    purposes it should be preferred. One may supply the array of function values at the 3d grid,
    collected using this routine with N=1, to either a LinearInterpolator3d or CubicSpline3d.
    \tparam     N  is the degree of interpolator (implemented for N=1 and N=3);
    \param[in]  F  is the source function of 3 variables, returning one value;
    \param[in]  xvalues, yvalues, zvalues are the grids in each of three coordinates;
    \return  the array of amplitudes suitable to use with `BsplineInterpolator::interpolate()` routine;
    by construction, the values of interpolant at grid nodes should be equal to the values of source
    function (but the array of amplitudes does not have a simple interpretation in the case N>1).
    \throw  std::invalid_argument if the source function has incorrect dimensions,
    or possibly other exceptions that might arise in the solution of linear system in the case N>1.
*/
template<int N>
std::vector<double> createBsplineInterpolator3dArray(
    const IFunctionNdim& F,
    const std::vector<double>& xvalues,
    const std::vector<double>& yvalues,
    const std::vector<double>& zvalues);


/** Construct the array of amplitudes for a 3d interpolator representing a probability distribution
    function (PDF) from the provided array of points with weights, sampled from this PDF.
    \tparam     N  is the degree of interpolator (1 or 3);
    \param[in]  points  is the matrix with N_p rows and 3 columns, representing the sampled points;
    \param[in]  weights  is the array of point weights;
    \param[in]  xvalues, yvalues, zvalues are the grids in each of three coordinates;
    \return  the array of amplitudes suitable to use with `BsplineInterpolator::interpolate()` routine;
    \throw  std::invalid_argument if the array sizes are incorrect, or std::runtime_error in case
    of other possible problems.

    NOT AVAILABLE YET.
*/
template<int N>
std::vector<double> createBsplineInterpolator3dArrayFromSamples(
    const IMatrixDense<double>& points,
    const std::vector<double>& weights,
    const std::vector<double>& xvalues,
    const std::vector<double>& yvalues,
    const std::vector<double>& zvalues);


///@}
/// \name Penalized spline approximation (1d)
///@{

/** Penalized linear least-square fitting problem.
    Approximate the data series  {x[i], y[i], optionally w[i], i=0..numDataPoints-1}
    with a spline defined by  {X[k], Y[k], k=0..numKnots-1} in the least-square sense,
    possibly with additional penalty term for 'roughness' (curvature).

    Initialized once for a given set of x, w, X, and may be used to fit multiple sets of y
    with arbitrary degree of smoothing \f$ \lambda \f$.

    Internally, the approximation is performed by multi-parameter weighted linear
    least-square fitting:  minimize
      \f[
        \sum_{i=0}^{numDataPoints-1}  w_i (y_i - \hat y(x_i))^2 + \lambda \int [\hat y''(x)]^2 dx,
      \f]
    where
      \f[
        \hat y(x) = \sum_{p=0}^{numBasisFnc-1} A_p B_p(x)
      \f]
    is the approximated regression for input data,
    \f$ B_p(x) \f$ are its basis functions and \f$ A_p \f$ are amplitudes to be found.

    Basis functions are modified b-splines of degree 3 with knots at X[k], k=0..numKnots-1
    and natural boundary conditions (i.e., 2nd derivative is zero at endpoints);
    the number of basis functions is numKnots. Equivalently, the regression
    can be represented by a natural cubic spline with the same set of knots;
    b-splines are only used internally.

    LLS fitting is done by solving the following linear system:
      \f$ (\mathsf{C} + \lambda \mathsf{R}) \mathbf{A} = \mathbf{z} \f$,
    where  C  and  R  are square matrices of size numBasisFnc,
    w and z are vectors of the same size, and \f$ \lambda \f$ is a scalar (smoothing parameter).

    C = B^T W B, where B_ip is a matrix of size numDataPoints*numBasisFnc,
    containing value of p-th basis function at x[i], p=0..numBasisFnc-1, i=0..numDataPoints-1;
    W = diag(w) is a diagonal matrix of input point weights;
    z = B^T W y, where y[i] is the vector of original data points;
    R is the roughness penalty matrix:  \f$ R_pq = \int B''_p(x) B''_q(x) dx \f$.

*/
class SplineApprox {
public:
    /** construct the object for grid=X, xvalues=x, weights=w in the above formulation.
        Grid nodes must be sorted in ascending order.
        Data points do not necessarily need to lie within the grid boundaries;
        the fitted function will be linearly extrapolated outside the grid.
        \note OpenMP-parallelized loop over xvalues.
    */
    SplineApprox(
        const std::vector<double>& grid,
        const std::vector<double>& xvalues,
        const std::vector<double>& weights = std::vector<double>());

    ~SplineApprox();

    /** perform actual fitting for the array of y values with the given smoothing parameter.
        \param[in]  yvalues is the array of data points corresponding to x values
        that were passed to the constructor;
        \param[in]  edf  is the number of equivalent degrees of freedom - the parameter that
        controls the amount of smoothing.
        It ranges from 2 for a linear regression (infinite smoothing) to numKnots (no smoothing).
        Default value 0 is synonymous to no smoothing; other values outside this range are not allowed.
        \param[out] rmserror if not NULL, will contain the root-mean-square deviation of data points
        from the smoothing curve;
        \returns  the array of interpolated function values at grid knots (length: numKnots),
        which may be used to initialize a natural CubicSpline.
    */
    std::vector<double> fit(
        const std::vector<double> &yvalues,
        const double edf=0,
        double *rmserror=NULL) const;

    /** perform fitting with adaptive choice of smoothing parameter that minimizes
        the generalized cross-validation score (GCV), defined as
          GCV = (rmserror^2 * numDataPoints) / (numDataPoints-EDF)^2 .
        The input and output arguments are similar to `fit()`, with the difference that
        the number of equivalent degrees of freedom is not provided as input,
        but may be reported as output argument `edf` if the latter is not NULL.
    */
    std::vector<double> fitOptimal(
        const std::vector<double> &yvalues,
        double *rmserror=NULL,
        double* edf=NULL) const
    {
        return fitOversmooth(yvalues, 0., rmserror, edf);
    }

    /** perform an 'oversmooth' fitting with adaptive choice of smoothing parameter.
        smoothing>=0 determines the difference in ln(GCV) between the solution with
        optimal smoothing (lowest GCV) and the returned solution which is smoothed more than
        the optimal amount defined above.
        The other arguments have the same meaning as in `fitOptimal()`.
    */
    std::vector<double> fitOversmooth(
        const std::vector<double> &yvalues,
        const double smoothing,
        double *rmserror=NULL,
        double* edf=NULL) const;

    class Impl;         ///< opaque internal data for SplineApprox
private:
    const Impl* impl;   ///< internal object hiding the implementation details
    SplineApprox& operator= (const SplineApprox&);  ///< assignment operator forbidden
    SplineApprox(const SplineApprox&);              ///< copy constructor forbidden
};


/** Parameters of penalized log-density fit */
enum FitOptions {
    FO_PENALTY_2ND_DERIV = 0,  ///< use integral of squared second derivative in the penalty
    FO_PENALTY_3RD_DERIV = 1,  ///< use integral of squared third  derivative in the penalty
    FO_INFINITE_LEFT     = 2,  ///< fitted function extends to -infinity to the left  of the grid
    FO_INFINITE_RIGHT    = 4   ///< fitted function extends to +infinity to the right of the grid
};

/** Penalized log-spline approximation to a probability density distribution.
    Let P(x)>0 be a probability distribution function defined on the entire real axis,
    a semi-infinite interval [xmin,+inf) or (-inf,xmax], or a finite interval [xmin,xmax].
    Let  {x[i], w[i], i=0..numDataPoints-1}  be an array of samples drawn from this distribution,
    where  x[i] are their coordinates, and w[i]>=0 are weights. We follow the convention that
    the integral of P(x) over its domain is equal to the sum of w[i] (not necessarily unity).

    The task of this routine is to estimate P(x) from the samples.
    It will represent ln(P(x)) as a sum of numBasisFnc basis functions (B-splines of degree N>=1)
    defined by the grid nodes, which are provided by the user.
    The amplitudes of these basis functions are computed using the penalized maximum-likelihood
    approach for the input samples.
    In case that the interval is finite or semi-infinite, the corresponding endpoint of input grid
    should enclose all sample points; otherwise some sample points may be left out of the grid,
    since the estimated ln(P(x)) is linearly extrapolated beyond the grid boundary(-ies).

    By default, the fitting procedure finds the optimal value of the regularization parameter that
    maximizes the cross-validation score (smoothing out fluctuations in the estimate), and one may
    increase the smoothing at the expense of somewhat worse fit to the data.

    \tparam  N is the degree of B-splines (implemented only for 1 or 3).
    In the case N=1, basis functions are triangular-shaped blocks spanning two grid segments,
    and in the case N=3, we use a modified set of B-splines with natural boundary conditions.

    \param[in]  grid     are the grid nodes defining the interpolated ln(P),
    should be in increasing order.
    \param[in]  xvalues  are the coordinates x[i] of input samples.
    \param[in]  weights  are the weights w[i] of samples; should be non-negative.
    If not provided (empty array), all weights are assumed to be equal to 1.0/xvalues.size().
    \param[in]  options  is a bit field specifying several parameters for the fitting procedure:
    the choice between second and third derivative in the penalty term
    (only for N=3; the N=1 case always uses first derivatives),
    and the extent of the domain for the estimated density beyond each of the two grid endpoints:
    if FO_INFINTE_LEFT bit is set, the function ln(P(x)) is assumed to be defined
    for all x<knots[0] and will be linearly extrapolated to the left of the first grid node
    (obviously it will be declining towards x=-infinity); in this case input samples may
    have x[i]<knots[0] and they will be accounted for in the fit.
    If this bit is not set, the function P(x) is identically zero for x<knots[0], and any samples
    that appear to the left of the boundary are ignored in the fit.
    The same applies to FO_INFINITE_RIGHT bit and the last grid node.
    \param[in]  smoothing  is the parameter defining the tradeoff between smoothness
    and accuracy of approximation.
    Best-fit parameters (amplitudes) of a model without smoothing correspond to the absolute
    maximum likelihood of samples, but typically the model exhibits unpleasant fluctuations
    trying to over-fit the existing samples. To cope with overfitting, there are two methods.
    If smoothing==0 (default), this actually corresponds to the ``optimal smoothing'' determined
    by maximizing the cross-validation likelihood.
    Otherwise we first determine the expected dispersion 'logLrms' of log-likelihood
    for the given number of samples, and then find the value of smoothing parameter
    such that the log-likelihood of the returned model is lower than that of the optimal model
    by an amount smoothing*logLrms.
    For instance, setting smoothing=1.0 will yield a model that is within 1 sigma from
    the best-fitting optimally smoothed model.
    \return  the array of log-density values ln(P(x)) at grid points (same length as grid).
    For N=1, ln(P(x)) is piecewise-linear, and for N=3 it is a natural cubic spline defined by
    the values at grid nodes.
    \throw  std::invalid_argument exception if samples have negative weights,
    or their total weight is not positive, or grid points are invalid.
    \note OpenMP-parallelized loop over xvalues & weights.
*/
template<int N>
std::vector<double> splineLogDensity(
    const std::vector<double> &grid,
    const std::vector<double> &xvalues,
    const std::vector<double> &weights=std::vector<double>(),
    FitOptions options=FitOptions(),
    double smoothing=0);

///@}
/// \name Auxiliary routines for grid generation
///@{

/** generate a grid with uniformly spaced nodes.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost node (>0);
    \param[in]  xmax     is the location of the outermost node (should be >xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createUniformGrid(unsigned int nnodes, double xmin, double xmax);

/** generate a grid with exponentially spaced nodes, i.e., uniform in log(x):
    log(x[k]) = log(xmin) + log(xmax/xmin) * k/(nnodes-1), k=0..nnodes-1.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost node (>0);
    \param[in]  xmax     is the location of the outermost node (should be >xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createExpGrid(unsigned int nnodes, double xmin, double xmax);

/** generate a grid with exponentially growing spacing:
    x[k] = (exp(Z k) - 1)/(exp(Z) - 1), i.e., coordinates of nodes increase nearly linearly
    at the beginning and then nearly exponentially towards the end;
    the value of Z is computed so the the 1st element is at xmin and last at xmax.
    \param[in]  nnodes   is the total number of grid points (>=2);
    \param[in]  xmin     is the location of the innermost nonzero node (>0);
    \param[in]  xmax     is the location of the last node (should be >=nnodes*xmin);
    \param[in]  zeroelem -- if true, 0th node in the output array is placed at zero (otherwise at xmin);
    \return     the array of grid nodes.
*/
std::vector<double> createNonuniformGrid(unsigned int nnodes, double xmin, double xmax, bool zeroelem);

/** extend the input grid to negative values, by reflecting it about origin.
    \param[in]  input is the vector of N values that should start at zero
                and be monotonically increasing;
    \return     a new vector that has 2*N-1 elements, so that
                input[i] = output[N-1+i] = -output[N-1-i] for 0<=i<N
    \throw      std::invalid_argument if the input does not start with zero or is not increasing.
*/
std::vector<double> mirrorGrid(const std::vector<double> &input);

/** create a possibly non-uniform grid, symmetric about origin.
    \param[in]  nnodes  is the total number of grid nodes (hence the number of segments is nnodes-1);
    \param[in]  xmin  is the width of the central grid segment;
    \param[in]  xmax  is the outer edge of the grid (endpoints are at +-xmax);
    if it is provided, the grid segments are gradually stretched as needed,
    otherwise this implies uniform segments and hence xmax = 0.5 * (nnodes-1) * xmin.
*/
std::vector<double> createSymmetricGrid(unsigned int nnodes, double xmin, double xmax=NAN);

/** Construct a grid for interpolating a function with a cubic spline.
    x is supposed to be a log-scaled coordinate, i.e., it does not attain very large values
    (only the range [-100:100] is examined).
    The function is assumed to have linear asymptotic behaviour at x -> +- infinity,
    and the goal is to place the grid nodes such that the typical error in the interpolating
    spline is less than the provided tolerance eps.
    The error in the cubic spline approximation of a sufficiently smooth function
    is <= 5/384 h^4 |f""(x)|, where h is the grid spacing and f"" is the fourth derivative
    (which we have to estimate by finite differences, using the second derivatives provided
    by the function). Note, however, that if the input function is a spline interpolator itself,
    its smoothness is not quite as high, and the accuracy of the secondary interpolation deteriorates
    somewhat (but is still at an acceptable level, taking into account that the original function
    itself is an approximation).
    We start by picking an initial point from several trial points inside the range [-100:100],
    based on the highest absolute value of the second derivative.
    Then we scan the range of x in both directions, adding grid nodes at intervals determined
    by the above relation, and stop when the second derivative is less than the threshold eps.
    Typically the nodes will be more sparsely spaced towards the end of the grid.
    The approach is intended for functions that take x=log(y), so that the range of x is rather small.
    \param[in] fnc  is the function f(x), only its second derivative is examined
    (if the function does not provide it, a finite-difference approximation is constructed).
    \param[in] eps  is the tolerance parameter.
    \return  the grid in x.
*/
std::vector<double> createInterpolationGrid(const IFunction& fnc, double eps);

///@}
}  // namespace
