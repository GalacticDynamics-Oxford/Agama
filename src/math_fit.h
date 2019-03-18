/** \file   math_fit.h
    \brief  routines for fitting in one or many dimensions
    \date   2015
    \author Eugene Vasiliev
*/
#pragma once
#include "math_base.h"
#include "math_linalg.h"

namespace math{

/// \name ------ linear regression ------
///@{

/** perform a linear least-square fit (i.e., c * x + b = y).
    \param[in]  x  is the array of independent variables;
    \param[in]  y  is the array of dependent variables;
    \param[in]  w  is the optional array of weight coefficients (= inverse square error in y values),
                if set to NULL this means equal weights;
    \param[out] slope  stores the best-fit slope of linear regression;
    \param[out] intercept  stores the best-fit intercept (value at x=0);
    \param[out] rms  optionally stores the rms scatter (if not NULL).
*/
void linearFit(const std::vector<double>& x, const std::vector<double>& y, 
    const std::vector<double>* w, double& slope, double& intercept, double* rms=NULL);

/** perform a linear least-square fit without constant term (i.e., c * x = y).
    \param[in]  x  is the array of independent variables;
    \param[in]  y  is the array of dependent variables;
    \param[in]  w  is the optional array of weight coefficients (= inverse square error in y values),
                if set to NULL this means equal weights;
    \param[out] rms  optionally stores the rms scatter (if not NULL).
    \return  the best-fit slope of linear regression. 
*/
double linearFitZero(const std::vector<double>& x, const std::vector<double>& y, 
    const std::vector<double>* w, double* rms=NULL);

/** perform a multi-parameter linear least-square fit, i.e., solve the system of equations
    `X c = y`  in the least-square sense (using singular-value decomposition).
    \param[in]  coefs  is the matrix of coefficients (X) with M rows and N columns;
    \param[in]  rhs  is the the array of M values (y);
    \param[in]  w  is the optional array of weights (= inverse square error in y values), 
                if set to NULL this means equal weights;
    \param[out] result  stores the solution (array of N coefficients in the regression);
    \param[out] rms  optionally stores the rms scatter (if not NULL).
*/
void linearMultiFit(const IMatrixDense<double>& coefs, const std::vector<double>& rhs, 
    const std::vector<double>* w, std::vector<double>& result, double* rms=NULL);

///@}
/// \name ------ nonlinear regression ------
///@{

/** perform a multi-parameter nonlinear least-square fit by the Levenberg--Marquardt method.
    Let `f(d; x)` be a single-valued function of data point `d` that depends on
    a vector of parameters `x` (of length N).
    We want to fit M data points with N parameters, minimizing the difference between `f(d_k; x)`
    and its target value `y_k` at this data point, for each k=1..M, by varying the parameters `x`.
    Define a multi-valued function F of N arguments (the parameters `x`) that provides M values,
    so that \f$  F_k(x) = f(d_k; x) - y_k  \f$, and also provides the Jacobian matrix of
    derivatives w.r.t. each parameter `x_i`, i=1..N, at each data point `d_k`, k=1..M.
    The fitting procedure varies the parameters `x` until the L2-norm of F, i.e.
    \f$ \sum_{i=k}^M [F_k(\vec x)]^2  \f$, reaches minimum.
    \param[in]  F  is the function whose L2-norm is minimized: its arguments are the values
    of parameters `x`, and its output values are `y_k - f(d_k; x)`, where the original nonlinear
    function `f`, data points `d_k` and the target values `y_k` must all be handled within F --
    the fitting algorithm only needs to know the difference at each point and the gradient
    w.r.t. each parameter at each point.
    It should throw an exception if the parameter values `x` are outside an acceptable range.
    \param[in]  xinit  is the array of starting values of parameters `x` (length N).
    \param[in]  relToler  is the stopping criterion: the change in parameter values during
    the step must satisfy |dx| < relToler * |x| to end the iterative procedure.
    \param[in]  maxNumIter  is the upper limit on the number of iterations.
    \param[out] result  is the array of best-fit parameters (length N).
    \return     the number of iterations taken.
*/
int nonlinearMultiFit(const IFunctionNdimDeriv& F, const double xinit[],
    const double relToler, const int maxNumIter, double result[]);

///@}
/// \name ------ multidimensional root-finding -------
///@{

/** solve a multidimensional system of equation.
    \param[in]  F  is the multivalued function of many variables that defines the NxN
    equation system (that is, `F.numVars() == F.numValues()`, where each equation involving
    N variables is represented by one element of the array of output values).
    It must provide the Jacobian matrix of derivatives of each function by all input vars.
    The equation system is \f$  F_i( \{x_k\} ) = 0, i=0..N-1, k=0..N-1  \f$.
    \param[in]  xinit  is the starting N-dimensional point for root finding;
    \param[in]  absToler  is the required tolerance on the value of each function at root;
    \param[in]  maxNumIter  is the upper limit on the number of iterations;
    \param[out] result will contain the array of variables that solve F(x)=0.
    \returns  the number of iterations taken.
*/
int findRootNdimDeriv(const IFunctionNdimDeriv& F, const double xinit[],
    const double absToler, const int maxNumIter, double result[]);

///@}
/// \name ------ multidimensional minimization -------
///@{

/** perform a multidimensional minimization of a function of N variables,
    using the Simplex algorithm of Nelder and Mead.
    \param[in]  F  is the function to be minimized (it may take N>=1 arguments 
    but must provide a single value);
    \param[in]  xinit  is the starting N-dimensional point for minimization;
    \param[in]  xstep  is the array of initial stepsizes in each of N dimensions;
    \param[in]  absToler  is the required tolerance on the location of minimum
    (average distance from the center of simplex to its corners):
    using a single control parameter demands that the input variables should be 
    scaled to comparable magnitudes;
    \param[in]  maxNumIter  is the upper limit on the number of iterations,
    the search is also terminated if (almost) no progress has been made for 
    10*N iterations;
    \param[out] result will contain the array of variables that minimize F.
    \returns  the number of iterations taken.
*/
int findMinNdim(const IFunctionNdim& F, const double xinit[], const double xstep[],
    const double absToler, const int maxNumIter, double result[]);

/** perform a gradient-based multidimensional minimization of a function of N variables.
    \param[in]  F  is the function to be minimized (it may take N>=1 arguments 
    but must provide a single value and a vector of derivatives w.r.t.each input variable);
    \param[in]  xinit  is the starting N-dimensional point for minimization;
    \param[in]  xstep  is the initial stepsize (same for each dimension);
    \param[in]  absToler  is the required tolerance on the location of minimum,
    defined as the norm of gradient vector;
    \param[in]  maxNumIter  is the upper limit on the number of iterations;
    \param[out] result will contain the array of variables that minimize F.
    \returns  the number of iterations taken.
*/
int findMinNdimDeriv(const IFunctionNdimDeriv& F, const double xinit[], const double xstep,
    const double absToler, const int maxNumIter, double result[]);

///@}
}  // namespace
