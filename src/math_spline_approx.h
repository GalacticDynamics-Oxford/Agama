/** \file    math_spline_appr.h
    \brief   Penalized spline approximation
    \author  Eugene Vasiliev
    \date    2011-2015

*/
#pragma once
#include <vector>

namespace mathutils {

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
    /// initialize workspace for xvalues=x, knots=X in the above formulation.
    /// knots must be sorted in ascending order, and all xvalues must lie between knots.front() and knots.back()
    SplineApprox(const std::vector<double> &xvalues, const std::vector<double> &knots);
    ~SplineApprox();

    /// return status of the object
    bool isSingular() const;

    /** perform actual fitting for the array of y values corresponding to the array of x values 
        passed to the constructor, with given smoothing parameter lambda.
        Return spline values Y at knots X, and if necessary, rms error and EDF 
        (equivalent degrees of freedom) in corresponding output parameters (if they are not NULL).
        The spline derivatives at endpoints are returned in separate output arguments
        (to pass to initialization of clamped spline):  the internal fitting process uses b-splines, 
        not natural cubic splines, therefore the endpoint derivatives are non-zero.
        EDF is equivalent number of free parameters in fit, increasing smoothing decreases EDF: 
        2<=EDF<=numKnots.  */
    void fitData(const std::vector<double> &yvalues, const double lambda, 
        std::vector<double>& splineValues, double& der_left, double& der_right,
        double *rmserror=NULL, double* edf=NULL);

    /** perform fitting with adaptive choice of smoothing parameter lambda, to minimize AIC.
        AIC (Akaike information criterion) is defined as 
          log(rmserror*numDataPoints) + 2*EDF/(numDataPoints-EDF-1) .
        return spline values Y, rms error, equivalent degrees of freedom (EDF) and best-choice value of lambda. */
    void fitDataOptimal(const std::vector<double> &yvalues, 
        std::vector<double>& splineValues, double& der_left, double& der_right,
        double *rmserror=NULL, double* edf=NULL, double *lambda=NULL);

    /** perform an 'oversmooth' fitting with adaptive choice of smoothing parameter lambda.
        The difference in AIC (Akaike information criterion) between the solution with no smoothing 
        and the returned solution is equal to deltaAIC (i.e. smooth more than optimal amount defined above).
        return spline values Y, rms error, equivalent degrees of freedom and best-choice value of lambda. */
    void fitDataOversmooth(const std::vector<double> &yvalues, const double deltaAIC, 
        std::vector<double>& splineValues, double& der_left, double& der_right,
        double *rmserror=NULL, double* edf=NULL, double *lambda=NULL);

private:
    SplineApproxImpl* impl;       ///< internal data hiding the implementation details
    SplineApprox& operator= (const SplineApprox&);  ///< assignment operator forbidden
    SplineApprox(const SplineApprox&);              ///< copy constructor forbidden
};

}  // namespace
