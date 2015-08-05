/** \file    math_spline.h
    \brief   spline interpolation and penalized spline approximation
    \author  Eugene Vasiliev
    \date    2011-2015

Spline interpolation class is based on the GSL implementation by G.Jungman;
2d spline is based on interp2d library by D.Zaslavsky.
*/
#pragma once
#include "math_base.h"
#include <vector>

namespace math{

/** Class that defines a cubic spline with natural or clamped boundary conditions */
class CubicSpline: public IFunction {
public:
    /** empty constructor is required for the class to be used in std::vector and alike places */
    CubicSpline() {};

    /** Initialize a cubic spline from the provided values of x and y
        (which should be arrays of equal length, and x values must be monotonically increasing).
        If deriv_left or deriv_right are provided, they set the slope at the lower or upper boundary
        (so-called clamped spline); if either of them is NaN, it means a natural boundary condition.
    */
    CubicSpline(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        double deriv_left=NAN, double deriv_right=NAN);

    /** compute the value of spline and optionally its derivatives at point x;
        if the input location is outside the definition interval, a linear extrapolation is performed. */
    virtual void evalDeriv(const double x, double* value=0, double* deriv=0, double* deriv2=0) const;

    virtual unsigned int numDerivs() const { return 2; }

    /** return the lower end of definition interval */
    double xmin() const { return xval.size()? xval.front() : NAN; }

    /** return the upper end of definition interval */
    double xmax() const { return xval.size()? xval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0; }

    /** check if the spline is everywhere monotonic on the given interval */
    bool isMonotonic() const;

private:
    std::vector<double> xval;  ///< grid nodes
    std::vector<double> yval;  ///< values of function at grid nodes
    std::vector<double> cval;  ///< second derivatives of function at grid nodes
};


/** Two-dimensional cubic spline */
class CubicSpline2d {
public:
    CubicSpline2d() {};
    /** Initialize a 2d cubic spline from the provided values of x, y and z.
        The latter is 2d array with the following indexing convention:  z[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
        Derivatives at the boundaries of definition region may be provided as optional arguments
        (currently a single value per entire side of the rectangle is supported);
        if any of them is NaN this means a natural boundary condition.
    */
    CubicSpline2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const std::vector< std::vector<double> >& zvalues,
        double deriv_xmin=NAN, double deriv_xmax=NAN, double deriv_ymin=NAN, double deriv_ymax=NAN);

    /** compute the value of spline and optionally its derivatives at point x,y;
        if the input location is outside the definition region, the result is NaN. 
        Any combination of value, first and second derivatives is possible: 
        if any of them is not needed, the corresponding pointer should be set to NULL. 
    */
    void evalDeriv(const double x, const double y, 
        double* value=0, double* deriv_x=0, double* deriv_y=0,
        double* deriv_xx=0, double* deriv_xy=0, double* deriv_yy=0) const;

    /** shortcut for computing the value of spline */
    double value(const double x, const double y) const {
        double v;
        evalDeriv(x, y, &v);
        return v;
    }

    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }

    /** check if the spline is initialized */
    bool isEmpty() const { return xval.size()==0 || yval.size()==0; }

private:
    std::vector<double> xval, yval;  ///< grid nodes in x and y directions
    std::vector<double> zval;        ///< flattened 2d array of z values
    std::vector<double> zx, zy, zxy; ///< flattened 2d arrays of derivatives in x and y directions, and mixed 2nd derivatives
};


/** Two-dimensional bilinear interpolator */
class LinearInterpolator2d {
public:
    LinearInterpolator2d() {};
    /** Initialize a 2d linear interpolator from the provided values of x, y and z.
        The latter is 2d array with the following indexing convention:  z[i][j] = f(x[i],y[j]).
        Values of x and y arrays should monotonically increase.
    */
    LinearInterpolator2d(const std::vector<double>& xvalues, const std::vector<double>& yvalues,
        const std::vector< std::vector<double> >& zvalues);
    
    /** compute the value of interpolator and optionally its derivatives at point x,y;
        if the input location is outside the definition region, the result is NaN. 
        Any combination of value and first derivatives is possible: 
        if any of them is not needed, the corresponding pointer should be set to NULL. 
     */
    void evalDeriv(const double x, const double y, 
                   double* value=0, double* deriv_x=0, double* deriv_y=0) const;
    
    /** shortcut for computing the value of interpolator */
    double value(const double x, const double y) const {
        double v;
        evalDeriv(x, y, &v);
        return v;
    }
    
    /** return the boundaries of definition region */
    double xmin() const { return xval.size()? xval.front(): NAN; }
    double xmax() const { return xval.size()? xval.back() : NAN; }
    double ymin() const { return yval.size()? yval.front(): NAN; }
    double ymax() const { return yval.size()? yval.back() : NAN; }
    
    /** check if the interpolator is initialized */
    bool isEmpty() const { return xval.size()==0 || yval.size()==0; }
    
private:
    std::vector<double> xval, yval;  ///< grid nodes in x and y directions
    std::vector<double> zval;        ///< flattened 2d array of z values
};
    

/// opaque internal data for SplineApprox
class SplineApproxImpl;

/** Penalized linear least-square fitting problem.
    Approximate the data series  {x[i], y[i], i=0..numDataPoints-1}
    with spline defined at  {X[k], Y[k], k=0..numKnots-1} in the least-square sense,
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

    Basis functions are b-splines with knots at X[k], k=0..numKnots-1; the number of basis functions is numKnots+2.
    Equivalently, the regression can be represented by clamped cubic spline with numKnots control points;
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
    /** initialize workspace for xvalues=x, knots=X in the above formulation.
        knots must be sorted in ascending order, and all xvalues must lie 
        between knots.front() and knots.back()   */
    SplineApprox(const std::vector<double> &xvalues, const std::vector<double> &knots);
    ~SplineApprox();

    /** check if the basis-function matrix L is singular: if this is the case, 
        fitting procedure is much slower and cannot accomodate any smoothing */
    bool isSingular() const;

    /** perform actual fitting for the array of y values corresponding to the array of x values 
        passed to the constructor, with given smoothing parameter lambda.
        Return spline values Y at knots X, and if necessary, RMS error and EDF 
        (equivalent degrees of freedom) in corresponding output parameters (if they are not NULL).
        The spline derivatives at endpoints are returned in separate output arguments
        (to pass to initialization of clamped cubic spline):  the internal fitting process uses 
        b-splines, not natural cubic splines, therefore the endpoint derivatives are non-zero.
        EDF is equivalent number of free parameters in fit, increasing smoothing decreases EDF: 
        2<=EDF<=numKnots+2.  */
    void fitData(const std::vector<double> &yvalues, const double lambda, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0);

    /** perform fitting with adaptive choice of smoothing parameter lambda, to minimize AIC.
        AIC (Akaike information criterion) is defined as 
          log(rmserror*numDataPoints) + 2*EDF/(numDataPoints-EDF-1) .
        return spline values Y, rms error, equivalent degrees of freedom (EDF),
        and best-choice value of lambda. */
    void fitDataOptimal(const std::vector<double> &yvalues, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0, double *lambda=0);

    /** perform an 'oversmooth' fitting with adaptive choice of smoothing parameter lambda.
        The difference in AIC (Akaike information criterion) between the solution with no smoothing 
        and the returned solution is equal to deltaAIC (i.e. smooth more than optimal amount defined above).
        return spline values Y, rms error, equivalent degrees of freedom and best-choice value of lambda. */
    void fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC, 
        std::vector<double>& splineValues, double& derivLeft, double& derivRight,
        double *rmserror=0, double* edf=0, double *lambda=0);

private:
    SplineApproxImpl* impl;       ///< internal data hiding the implementation details
    SplineApprox& operator= (const SplineApprox&);  ///< assignment operator forbidden
    SplineApprox(const SplineApprox&);              ///< copy constructor forbidden
};


/** generates a grid with exponentially growing spacing.
    x[k] = (exp(Z k) - 1)/(exp(Z) - 1),
    and the value of Z is computed so the the 1st element is at xmin and last at xmax.
    \param[in]  nnodes   is the total number of grid points (>=2)
    \param[in]  xmin     is the location of the innermost nonzero node (>0)     
    \param[in]  xmax     is location of the last node (should be >=nnodes*xmin)
    \param[in]  zeroelem -- if true, 0th node in the output array is placed at zero (otherwise at xmin)
    \param[out] grid     is the array of grid nodes created by this routine
*/
void createNonuniformGrid(unsigned int nnodes, double xmin, double xmax, bool zeroelem, std::vector<double>& grid);

/** creates an almost uniform grid so that each bin contains at least minbin points from input array.
    input points are in srcpoints array and MUST BE SORTED in ascending order (assumed but not cheched).
    \param[in]  srcpoints is the input array of points,
    \param[in]  minbin    is the minimum number of points per bin,
    \param[in]  gridsize  is the required length of the output array,
    \param[out] grid      will contain the array of grid nodes. 
    NB: in the present implementation, the algorithm is not very robust and works well only for gridsize*minbin << srcpoints.size,
    assuming that 'problematic' bins only are found close to endpoints but not in the middle of the grid.
*/
void createAlmostUniformGrid(const std::vector<double> &srcpoints, unsigned int minbin, unsigned int& gridsize, std::vector<double>& grid);

}  // namespace
