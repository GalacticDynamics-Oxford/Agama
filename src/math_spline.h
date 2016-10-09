/** \file    math_spline.h
    \brief   spline interpolation and penalized spline approximation
    \author  Eugene Vasiliev
    \date    2011-2016

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
- If in addition to the values of function at grid points, its first derivatives y'[i]
are also given at all points, then a quintic spline is the right choice for interpolation.
It provides piecewise 5-th degree polynomial interpolation with three continuous derivatives
on the entire domain.
- Alternatively, a cubic Hermite spline may be constructed from the same arrays x, y and y',
which provides locally cubic interpolation on each segment, but the values of 2nd derivative
are not continuous across segments. On the other hand, since the interpolation is local,
it is easy to ensure monotonic behaviour of the function by adjusting the derivatives
at the boundaries of segments where the monotonicity is violated (presently not implemented).

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
(i.e. trying to obtain them by finite differences is useless).

###  3-dimensional case.
In this case, the strategy is somewhat different: instead of a single object encapsulating
all data needed for interpolation, we provide the interface based on tensor product of
B-spline basis functions in each dimension, and their amplitudes are supplied by the user.
In other words, the value of function is represented as a sum of interpolating basis functions
with adjustable amplitudes, and each basis function is a separable function of three coordinates,
i.e. a product of three one-dimensional functions. These 1d basis functions are piecewise
polynomials of degree N with compact support spanning at most N+1 adjacent intervals between
nodes on their respective axis. Thus the interpolation is local, i.e. is determined by
the amplitudes of at most (N+1)^3 basis functions that are possibly non-zero at the given point;
however, to find the amplitudes that yield the given values of function at all nodes of a 3d grid,
one needs to solve a global linear system for all nodes, except the case of a linear (N=1)
interpolator.

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

###  Code origin
1d cubic spline is based on the GSL implementation by G.Jungman;
2d cubic spline is based on the interp2d library by D.Zaslavsky;
1d and 2d quintic splines are based on the code by W.Dehnen.
*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"

namespace math{

///@{
/// \name One-dimensional interpolation

/** Generic one-dimensional interpolator class */
class BaseInterpolator1d: public IFunction {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    BaseInterpolator1d() {};

    /** Initialize a 1d interpolator from the provided values of x and f(x);
        x should be at least of length 2 and monotonically increasing.
    */
    BaseInterpolator1d(const std::vector<double>& xvalues, const std::vector<double>& fvalues);

    /** return the number of derivatives that the interpolator provides */
    virtual unsigned int numDerivs() const { return 2; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0; }

    /** return the array of spline nodes */
    const std::vector<double>& xvalues() const { return xval; }

protected:
    std::vector<double> xval;  ///< grid nodes
    std::vector<double> fval;  ///< values of function at grid nodes
};

/** Class that provides a simple piecewise-linear interpolation for an array of x,y values */
class LinearInterpolator: public BaseInterpolator1d {
public:
    LinearInterpolator() : BaseInterpolator1d() {};

    LinearInterpolator(const std::vector<double>& xvalues, const std::vector<double>& fvalues);

    /** compute the value of interpolator and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;
};


/** Class that defines a cubic spline with natural or clamped boundary conditions */
class CubicSpline: public BaseInterpolator1d, public IFunctionIntegral {
public:
    CubicSpline() : BaseInterpolator1d() {};

    /** There are two possible ways of initializing a cubic spline:
        (1) from the values of the function at grid nodes, and optionally its endpoint derivatives;
        (2) from the amplitudes of B-spline functions returned by fitting routines.
        \param[in]  xvalues  - the array of grid nodes, should be monotonically increasing.
        \param[in]  fvalues  - in the first case, an array of function values
        at grid nodes (same length as xvalues);
        in the second case, an array of B-spline amplitudes, with length equal to xvalues.size()+2.
        \param[in]  derivLeft  (optional)  - first derivative of the spline at the leftmost
        grid node (only in the first case), default value NaN means a natural boundary condition
        (zero second derivative).
        \param[in]  derivRight - same for the rightmost grid node.
        \throws  std::invalid_argument exception if grid is too small or not monotonic,
        or the array sizes are incorrect.
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        double derivLeft=NAN, double derivRight=NAN);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

    /** return the integral of spline function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const;

    /** return the integral of spline function times another function f(x) on the interval [x1..x2]; 
        the other function is specified by the interface that provides the integral
        of f(x) * x^n for 0<=n<=3 */
    double integrate(double x1, double x2, const IFunctionIntegral& f) const;

    /** check if the spline is everywhere monotonic on the given interval */
    bool isMonotonic() const;

private:
    std::vector<double> fder2;  ///< second derivatives of function at grid nodes
};


/** Class that defines a piecewise cubic Hermite spline.
    Input consists of values of y(x) and dy/dx on a grid of x-nodes;
    result is a cubic function in each segment, with continuous first derivative at nodes
    (however second or third derivative is not continuous, unlike the case of quintic spline).
*/
class HermiteSpline: public BaseInterpolator1d {
public:
    HermiteSpline() : BaseInterpolator1d() {};

    /** Initialize the spline from the provided values of x, f(x) and f'(x)
        (which should be arrays of equal length, and x values must be monotonically increasing).
    */
    HermiteSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& yderivs);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

private:
    std::vector<double> fder;  ///< first derivatives of function at grid nodes
};


/** Class that defines a quintic spline.
    Given y and dy/dx on a grid, d^3y/dx^3 is computed such that the (unique) 
    polynomials of 5th degree between two adjacent grid points that give y,dy/dx,
    and d^3y/dx^3 on the grid are continuous in d^2y/dx^2, i.e. give the same
    value at the grid points. At the grid boundaries  d^3y/dx^3=0  is adopted.
*/
class QuinticSpline: public BaseInterpolator1d {
public:
    QuinticSpline() : BaseInterpolator1d() {};

    /** Initialize a quintic spline from the provided values of x, f(x) and f'(x)
        (which should be arrays of equal length, and x values must be monotonically increasing).
    */
    QuinticSpline(const std::vector<double>& xvalues, const std::vector<double>& fvalues,
        const std::vector<double>& fderivs);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

    /** return the value of 3rd derivative at a given point */
    double deriv3(const double x) const;

    /** two derivatives are returned by evalDeriv() method, and third derivative - by deriv3() */
    virtual unsigned int numDerivs() const { return 3; }

private:
    std::vector<double> fder;  ///< first derivatives of function at grid nodes
    std::vector<double> fder3; ///< third derivatives of function at grid nodes
};


///@}
/// \name Two-dimensional interpolation
///@{

/** Generic two-dimensional interpolator class */
class BaseInterpolator2d: public IFunctionNdim {
public:
    BaseInterpolator2d() {};
    /** Initialize a 2d interpolator from the provided values of x, y and f.
        The latter is 2d array with the following indexing convention:  f[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
    */
    BaseInterpolator2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& fvalues);

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
    virtual unsigned int numVars() const { return 2; }    
    virtual unsigned int numValues() const { return 1; }

    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }

    /** check if the interpolator is initialized */
    bool isEmpty() const { return xval.size()==0 || yval.size()==0; }

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
    LinearInterpolator2d() : BaseInterpolator2d() {};

    /** Initialize a 2d interpolator from the provided values of x, y and f.
        The latter is 2d array with the following indexing convention:  f[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
    */
    LinearInterpolator2d(const std::vector<double>& xgrid, const std::vector<double>& ygrid,
        const Matrix<double>& fvalues) : 
        BaseInterpolator2d(xgrid, ygrid, fvalues) {};

    /** Compute the value and/or derivatives of the interpolator;
        note that for the linear interpolator the 2nd derivatives are always zero. */
    virtual void evalDeriv(const double x, const double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;
};


/** Two-dimensional cubic spline */
class CubicSpline2d: public BaseInterpolator2d {
public:
    CubicSpline2d() : BaseInterpolator2d() {};

    /** Initialize a 2d cubic spline from the provided values of x, y and f.
        The latter is 2d array (Matrix) with the following indexing convention:  f(i,j) = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
        Derivatives at the boundaries of definition region may be provided as optional arguments
        (currently a single value per entire side of the rectangle is supported);
        if any of them is NaN this means a natural boundary condition.
    */
    CubicSpline2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& fvalues,
        double deriv_xmin=NAN, double deriv_xmax=NAN, double deriv_ymin=NAN, double deriv_ymax=NAN);

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(const double x, const double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;

private:
    /// flattened 2d arrays of derivatives in x and y directions, and mixed 2nd derivatives
    std::vector<double> fx, fy, fxy;
};


/** Two-dimensional quintic spline */
class QuinticSpline2d: public BaseInterpolator2d {
public:
    QuinticSpline2d() : BaseInterpolator2d() {};

    /** Initialize a 2d quintic spline from the provided values of x, y, f(x,y), df/dx and df/dy.
        The latter three are 2d arrays (variables of Matrix type) with the following indexing
        convention:  f(i,j) = f(x[i],y[j]), etc.
        Values of x and y arrays should monotonically increase.
    */
    QuinticSpline2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const Matrix<double>& fvalues, const Matrix<double>& dfdx, const Matrix<double>& dfdy);

    /** compute the value of spline and optionally its derivatives at point x,y */
    virtual void evalDeriv(const double x, const double y,
        double* value=NULL, double* deriv_x=NULL, double* deriv_y=NULL,
        double* deriv_xx=NULL, double* deriv_xy=NULL, double* deriv_yy=NULL) const;

private:
    /// flattened 2d arrays of various derivatives
    std::vector<double> fx, fy, fxxx, fyyy, fxyy, fxxxyy;
};


///@}
/// \name Three-dimensional interpolation
///@{

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
*/
template<int N>
class BsplineInterpolator3d: public math::IFunctionNdim {
public:
    /** Initialize a 3d interpolator from the provided 1d arrays of grid nodes in x, y and z dimensions.
        \param[in] xnodes, ynodes, znodes are the nodes of grid in each dimension,
        sorted in increasing order, must have at least 2 elements.
        There is no work done in the constructor apart from checking the validity of parameters.
        \throw std::invalid_argument if the 1d grids are invalid.
    */
    BsplineInterpolator3d(const std::vector<double>& xnodes,
        const std::vector<double>& ynodes, const std::vector<double>& znodes);

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
        the cube (the definition region), or, for a linear intepolator (N=1) also at all grid nodes,
        and means that the value of interpolant `f` is equal to the single element of the amplitudes
        array, which in the case N=1 should contain the values of the original function at grid nodes.
        If any of the coordinates of input point falls outside grid boundaries in the respective
        dimension, all weights are zero.
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
        \throw std::range_error if indComp is out of range.
    */
    double valueOfComponent(const double point[3], unsigned int indComp) const;

    /** Report the region of 3d space where the interpolation basis function
        of the given component is nonzero.
        \param[in]  indComp is the index of component;
        \param[out] xlower  are the coordinates of the lower corner of the region;
        \param[out] xupper  same for the upper corner;
        \throw std::range error if indComp >= numComp.
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
        \throw std::range_error if the length of `amplitudes` does not correspond to numComp.
    */
    double interpolate(const double point[3], const std::vector<double> &amplitudes) const;

    /** The dimensions of interpolator (3) */
    virtual unsigned int numVars()   const { return 3; }

    /** The number of components (3d interpolation basis functions) */
    virtual unsigned int numValues() const { return numComp; }

    /** Return the index of element in the flattened 3d array of function values
        associated with the given triplet of indices in each of the 1d coordinate grids.
        The indices must satisfy  0 <= ind_x < N_x+N-1, 0 <= ind_y < N_y+N-1, 0 <= ind_z < N_z+N-1,
        where N_x, N_y, N_z are the sizes of grids in each dimension provided to the constructor;
        however, no range check is performed on the input indices!
    */
    unsigned int indComp(unsigned int ind_x, unsigned int ind_y, unsigned int ind_z) const {
        return (ind_x * (ynodes.size()+N-1) + ind_y) * (znodes.size()+N-1) + ind_z;
    }

    /** Decompose the index of element in the flattened 3d array of function values
        into the three indices in each of the 1d coordinate grids (no range check is performed!)
    */
    void decomposeIndComp(const unsigned int indComp, unsigned int indices[3]) const {
        const unsigned int NN_y = ynodes.size()+N-1, NN_z = znodes.size()+N-1;
        indices[2] = indComp % NN_z,
        indices[1] = indComp / NN_z % NN_y,
        indices[0] = indComp / NN_z / NN_y;
    }
    
    /** return the boundaries of grid definition region */
    double xmin() const { return xnodes.front(); }
    double xmax() const { return xnodes.back();  }
    double ymin() const { return ynodes.front(); }
    double ymax() const { return ynodes.back();  }
    double zmin() const { return znodes.front(); }
    double zmax() const { return znodes.back();  }

    /** return the (sparse) matrix of roughness penalties */
    SpMatrix<double> computeRoughnessPenaltyMatrix() const;

private:
    std::vector<double> xnodes, ynodes, znodes;  ///< grid nodes in x, y and z directions
    const unsigned int numComp;                  ///< total number of components
};

/// trilinear interpolator
typedef BsplineInterpolator3d<1> LinearInterpolator3d;
/// tricubic interpolator
typedef BsplineInterpolator3d<3> CubicInterpolator3d;

/** Fill the array of amplitudes for a 3d interpolator by collecting the values of the source
    function F at the nodes of 3d grid.
    For the case N=1, the values of source function at grid nodes are identical to the amplitudes,
    but for higher-degree interpolation this is not the case, and the amplitudes are obtained by
    solving a linear system with the size numComp*numComp, where numComp ~ (grid_size_in_1d+N-1)^3.
    This could be prohibitively expensive if numComp > ~10^3, and hence this routine should be used
    only for small grid sizes.
    Keep in mind also that the amplitudes thus obtained may be negative even if the source function
    is everywhere non-negative.
    \tparam     N  is the degree of interpolator (implemented for N=1 and N=3);
    \param[in]  F  is the source function of 3 variables, returning one value;
    \param[in]  xnodes, ynodes, znodes are the grids in each of three coordinates;
    \return  the array of amplitudes suitable to use with `BsplineInterpolator::interpolate()` routine;
    by construction, the values of interpolant at grid nodes should be equal to the values of source
    function (but the array of amplitudes does not have a simple interpretation in the case N>1).
    \throw  std::invalid_argument if the source function has incorrect dimensions,
    or possibly other exceptions that might arise in the solution of linear system in the case N>1.
*/
template<int N>
std::vector<double> createInterpolator3dArray(const IFunctionNdim& F,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);

/** Construct the array of amplitudes for a 3d interpolator representing a probability distribution
    function (PDF) from the provided array of points with weights, sampled from this PDF.
    \tparam     N  is the degree of interpolator (1 or 3);
    \param[in]  points  is the matrix with N_p rows and 3 columns, representing the sampled points;
    \param[in]  weights  is the array of point weights;
    \param[in]  xnodes, ynodes, znodes are the grids in each of three coordinates;
    \return  the array of amplitudes suitable to use with `BsplineInterpolator::interpolate()` routine;
    \throw  std::invalid_argument if the array sizes are incorrect, or std::runtime_error in case
    of other possible problems.
*/
template<int N>
std::vector<double> createInterpolator3dArrayFromSamples(
    const Matrix<double>& points, const std::vector<double>& weights,
    const std::vector<double>& xnodes,
    const std::vector<double>& ynodes,
    const std::vector<double>& znodes);


///@}
/// \name Penalized spline approximation (1d)
///@{

/// opaque internal data for SplineApprox
class SplineApproxImpl;

/** Penalized linear least-square fitting problem.
    Approximate the data series  {x[i], y[i], i=0..numDataPoints-1}
    with a spline defined by  {X[k], Y[k], k=0..numKnots-1} in the least-square sense,
    possibly with additional penalty term for 'roughness' (curvature).

    Initialized once for a given set of x, X, and may be used to fit multiple sets of y
    with arbitrary degree of smoothing \f$ \lambda \f$.

    Internally, the approximation is performed by multi-parameter linear least-square fitting:
    minimize
      \f[
        \sum_{i=0}^{numDataPoints-1}  (y_i - \hat y(x_i))^2 + \lambda \int [\hat y''(x)]^2 dx,
      \f]
    where
      \f[
        \hat y(x) = \sum_{p=0}^{numBasisFnc-1} w_p B_p(x)
      \f]
    is the approximated regression for input data,
    \f$ B_p(x) \f$ are its basis functions and \f$ w_p \f$ are weights to be found.

    Basis functions are b-splines of degree 3 with knots at X[k], k=0..numKnots-1;
    the number of basis functions is numKnots+2. Equivalently, the regression
    can be represented by a clamped cubic spline with numKnots control points;
    b-splines are only used internally.

    LLS fitting is done by solving the following linear system:
      \f$ (\mathsf{A} + \lambda \mathsf{R}) \mathbf{w} = \mathbf{z} \f$,
    where  A  and  R  are square matrices of size numBasisFnc,
    w and z are vectors of the same size, and \f$ \lambda \f$ is a scalar (smoothing parameter).

    A = C^T C, where C_ip is a matrix of size numDataPoints*numBasisFnc,
    containing value of p-th basis function at x[i], p=0..numBasisFnc-1, i=0..numDataPoints-1.
    z = C^T y, where y[i] is vector of original data points
    R is the roughness penalty matrix:  \f$ R_pq = \int B''_p(x) B''_q(x) dx \f$.

*/
class SplineApprox {
public: 
    /** construct the object for grid=X, xvalues=x in the above formulation.
        Grid nodes must be sorted in ascending order.
        Data points do not necessarily need to lie within the grid boundaries;
        the fitted function will be linearly extrapolated outside the grid. 
    */
    SplineApprox(const std::vector<double> &grid, const std::vector<double> &xvalues);

    ~SplineApprox();

    /** perform actual fitting for the array of y values with the given smoothing parameter.
        \param[in]  yvalues is the array of data points corresponding to x values
        that were passed to the constructor;
        \param[in]  edf  is the number of equivalent degrees of freedom - the parameter that
        controls the amount of smoothing.
        It ranges from 2 for a linear regression (infinite smoothing) to numKnots+2 (no smoothing).
        Default value 0 is synonymous to no smoothing; other values outside this range are not allowed.
        \param[out] rmserror if not NULL, will contain the root-mean-square deviation of data points
        from the smoothing curve;
        \returns the array of amplitudes of B-splines, which may be used as an argument for
        the constructor of CubicSpline class that will provide the interpolated function;
        the length of this array is numKnots+2.
    */
    std::vector<double> fit(const std::vector<double> &yvalues, const double edf=0, 
        double *rmserror=NULL) const;

    /** perform fitting with adaptive choice of smoothing parameter that minimizes
        the value of AIC (Akaike information criterion), defined as 
          log(rmserror^2 * numDataPoints) + 2 * EDF / (numDataPoints-EDF-1) .
        The input and output arguments are similar to `fit()`, with the difference that
        the smoothing parameter (number of equivalent degrees of freedom) is not provided as input,
        but may be reported as output argument `edf` if the latter is not NULL.
    */
    std::vector<double> fitOptimal(const std::vector<double> &yvalues, 
        double *rmserror=NULL, double* edf=NULL) const {
        return fitOversmooth(yvalues, 0., rmserror, edf);
    }

    /** perform an 'oversmooth' fitting with adaptive choice of smoothing parameter.
        deltaAIC>=0 determines the difference in AIC (Akaike information criterion) between
        the solution with no smoothing and the returned solution which is smoothed more than
        the optimal amount defined above.
        The other arguments have the same meaning as in `fitOptimal()`.
    */
    std::vector<double> fitOversmooth(const std::vector<double> &yvalues, const double deltaAIC, 
        double *rmserror=NULL, double* edf=NULL) const;

private:
    const SplineApproxImpl* impl;   ///< internal object hiding the implementation details
    SplineApprox& operator= (const SplineApprox&);  ///< assignment operator forbidden
    SplineApprox(const SplineApprox&);              ///< copy constructor forbidden
};


/** Parameters of penalized log-density fit */
enum FitOptions {
    FO_PENALTY_2ND_DERIV = 0,  ///< use integral of squared second derivative in the penalty
    FO_PENALTY_3RD_DERIV = 1,  ///< use integral of squared third derivative in the penalty
    FO_INFINITE_LEFT     = 2,  ///< fitted function extends to -infinity to the left of the grid
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
    defined by the grid nodes (numBasisFnc = numGridNodes+N-1), where the grid nodes are provided
    by the user. In the case N=3 this is equivalent to a clamped cubic spline.
    The amplitudes of these basis functions are computed using the penalized maximum-likelihood
    approach for the input samples.
    In case that the interval is finite or semi-infinite, the corresponding endpoint of input grid
    should enclose all sample points; otherwise some sample points may be left out of the grid,
    since the estimated ln(P(x)) is linearly extrapolated beyond the grid boundary(-ies).

    Optionally the fitting procedure may employ regularization to smooth out fluctuations in
    the estimate, at the expense of somewhat worse fit to the data (this is only possible for N>1).

    \tparam  N is the degree of B-splines
    (implemented for 1 or 3, smoothing is possible only for N>1).
 
    \param[in]  grid     are the grid nodes defining the interpolated ln(P),
    should be in increasing order.
    \param[in]  xvalues  are the coordinates x[i] of input samples.
    \param[in]  weights  are the weights w[i] of samples; should be non-negative.
    \param[in]  options  is a bit field specifying several parameters for the fitting procedure:
    the choice between second and third derivative in the penalty term, and
    the extent of the domain for the estimated density beyond each of the two grid endpoints:
    if FO_INFINTE_LEFT bit is set, the function ln(P(x)) is assumed to be defined
    for all x<knots[0] and will be linearly extrapolated to the left of the first grid node
    (obviously it will be declining towards x=-infinity); in this case input samples may
    have x[i]<knots[0] and they will be accounted for in the fit.
    If this bit is not set, the function P(x) is identically zero for x<knots[0], and any samples
    that appear to the left of the boundary are ignored in the fit.
    The same applies to FO_INFINITE_RIGHT bit and the last grid node.
    \param[in]  smoothing  is the parameter defining the tradeoff between smoothness
    and accuracy of approximation (makes sense only for N>1).
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
    \return  the array of basis function amplitudes defining the log-density ln(P(x)).
    For N=1, their number is equal to the number of grid points,
    and ln(P(x)) is piecewise-linear, with the values at grid nodes equal to the amplitudes.
    For N=3, the amplitudes may be used to construct a clamped cubic spline for ln(P(x))
    by providing this array to the constructor of CubicSpline class.
    \throws  std::invalid_argument exception if samples have negative weights or lie
    outside the allowed boundaries, or grid points are invalid.
*/
template<int N>
std::vector<double> splineLogDensity(const std::vector<double> &grid,
    const std::vector<double> &xvalues, const std::vector<double> &weights,
    FitOptions options=FitOptions(), double smoothing=0);

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

/** create an almost uniform grid enclosing all input points,
    so that each bin contains at least minbin points from input array.
    \param[in]  nnodes    is the number of grid nodes (>=2);
    \param[in]  srcpoints is the input array of points;
    \param[in]  minbin    is the minimum number of points per bin;
    \return     the array of grid nodes. 
    NB: in the present implementation, the algorithm is not very robust 
    and works well only for nnodes*minbin << srcpoints.size, assuming that 
    'problematic' bins only are found close to endpoints but not in the middle of the grid.
*/
std::vector<double> createAlmostUniformGrid(unsigned int nnodes, 
    const std::vector<double> &srcpoints, unsigned int minbin);

/** extend the input grid to negative values, by reflecting it about origin.
    \param[in]  input is the vector of N values that should start at zero
                and be monotonically increasing;
    \return     a new vector that has 2*N-1 elements, so that
                input[i] = output[N-1+i] = -output[N-1-i] for 0<=i<N
    \throw      std::invalid_argument if the input does not start with zero or is not increasing.
*/
std::vector<double> mirrorGrid(const std::vector<double> &input);

///@}
}  // namespace
