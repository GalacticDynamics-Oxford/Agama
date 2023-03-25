/** \file   math_optimization.h
    \brief  linear and quadratic optimization solvers
    \date   2009-2019
    \author Eugene Vasiliev
*/
#pragma once
#include "math_linalg.h"

namespace math{

/** Solve a linear optimization problem, i.e., a system of linear equations with
    constraints on the solution vector `x` and a linear objective function `F(x)`.
    The task is to find the vector `x` that solves the system of linear equations
    A x = rhs,  satisfies constraints  xmin <= x <= xmax, and minimizes the cost function
    F(x) = L x, where L is the vector of 'penalties'.
    \param[in]  A  is the matrix of the linear system (N_c rows, N_v columns, N_v>=N_c),
    accessed through the 'IMatrix' interface (e.g., a dense or a sparse matrix`);
    \param[in]  rhs  is the RHS of the linear system (N_c elements, or the equality constraints);
    \param[in]  L  is the vector of penalties (N_v elements, or empty vector meaning that
    all of them are zero);
    \param[in]  xmin  is the vector of lower limits imposed on the solution
    (N_v elements, or empty vector in the default case that all lower limits are zeros);
    \param[in]  xmax  is the vector of upper limits on the solution
    (N_v elements, or empty vector in the default case of no limits);
    \return  the solution vector `x` with N_v elements;
    \throw   std::invalid_argument if the sizes of input vectors/matrices are inconsistent,
    or std::runtime_error if the problem has no solution or the solver reports any other error.
    \tparam  NumT  is the numerical type of the input arrays (float or double).
    \note    This routine cannot be called from multiple threads simultaneously.
*/
template<typename NumT>
std::vector<double> linearOptimizationSolve(
    const     IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,
    const std::vector<NumT>& L = std::vector<NumT>(),
    const std::vector<NumT>& xmin = std::vector<NumT>(),
    const std::vector<NumT>& xmax = std::vector<NumT>());

/** Solve a quadratic optimization problem, i.e., a system of linear equations with
    constraints on the solution vector `x` and a quadratic objective function `F(x)`.
    The task is to find the vector `x` that solves the system of linear equations
    A x = rhs,  satisfies constraints  xmin <= x <= xmax, and minimizes the cost function
    F(x) = L x + (1/2) x^T Q x,  where L is the vector of 'linear penalties' and Q is the matrix
    of 'quadratic penalties'.
    \param[in]  A  is the matrix of the linear system (N_c rows, N_v columns, N_v>=N_c);
    \param[in]  rhs  is the RHS of the linear system (N_c elements, or the equality constraints);
    \param[in]  L  is the vector of linear penalties (N_v elements);
    \param[in]  Q  is the matrix of quadratic penalties (square matrix N_v x N_v),
    most commonly it would be a diagonal matrix specified by an array of N_v elements,
    as `BandMatrix<double>(vectorDiag)`. May also be an empty matrix, which means
    no quadratic term, so that the optimization problem is linear; however, the implementation
    of the solver may be different from that in `linearOptimizationSolve()`.
    \param[in]  xmin  is the vector of lower limits imposed on the solution
    (N_v elements, or empty vector in the default case that all lower limits are zeros);
    \param[in]  xmax  is the vector of upper limits on the solution
    (N_v elements, or empty vector in the default case of no limits; a values of INFINITY
    means no upper limit for any of its elements);
    \return  the solution vector `x` with N_v elements;
    \throw   std::invalid_argument if the sizes of input vectors/matrices are inconsistent,
    or std::runtime_error if the problem has no solution, the solver is not available or
    in case of any other error reported by the solver.
    \tparam  NumT  is the numerical type of the input arrays (float or double).
    \note    This routine cannot be called from multiple threads simultaneously.
*/
template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const     IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,
    const std::vector<NumT>& L = std::vector<NumT>(),
    const     IMatrix<NumT>& Q = BandMatrix<NumT>(),
    const std::vector<NumT>& xmin = std::vector<NumT>(),
    const std::vector<NumT>& xmax = std::vector<NumT>());

/** A modification of linear optimization problem that always has a solution, even if
    it does not satisfy all constraints exactly.
    The original linear system  A x = rhs  that has N_v variables and N_c constraints in the rhs
    is replaced with an augmented system, which has 2 N_c extra variables, so that each equation
    has two extra terms:  \f$ \sum_{v=0}^{N_v-1} A[c,v] x[v] + p[c] - q[c] = rhs[c]  \f$. 
    The vector L of linear penalties for original variables is also augmented with penalties
    for these additional variables, and the lower bounds for them are set to zero.
    Thus if the original system had a solution, the additional variables are all zero,
    but if it was infeasible, then either p[c] or q[c] will be larger than zero for some rows.
    The penalties for constraint violation are specified by an additional argument `consPenaltyLin`,
    while other arguments have the same meaning and default values as for `linearOptimizationSolve()`.
    Some constraint penalties may be infinity, meaning that these constraints must be satisfied exactly
    (no extra variables are added for these constraints).
    \param[in]  A  is the matrix of the linear system (N_c x N_v);
    \param[in]  rhs  is its right-hand side (N_c);
    \param[in]  L  is the vector of linear penalties for N_v variables;
    \param[in]  consPenaltyLin  is the vector of linear penalties for violating each of N_c constraints
    (in either positive or negative direction), must contain non-negative elements;
    \param[in]  xmin  is the vector of lower bounds on the solution (N_v or empty);
    \param[in]  xmax  is the vector of upper bounds (N_v);
    \tparam  NumT  is the numerical type of the input arrays (float or double).
    \return  the solution vector (N_v);
    \throw  std::runtime_error in case of a problem reported by the solver.
*/
template<typename NumT>
std::vector<double> linearOptimizationSolveApprox(
    const     IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,
    const std::vector<NumT>& L = std::vector<NumT>(),
    const std::vector<NumT>& consPenaltyLin = std::vector<NumT>(),
    const std::vector<NumT>& xmin = std::vector<NumT>(),
    const std::vector<NumT>& xmax = std::vector<NumT>());

/** A modification of quadratic optimization problem that always has a solution, even if
    it does not satisfy all constraints exactly. 
    The original linear system  A x = rhs  that has N_v variables and N_c constraints in the rhs is
    augmented with extra 2 N_c slack variables in the same way as in `linearOptimizationSolveApprox()`.
    The penalties for these extra variables being different from zero can be both linear and quadratic,
    governed by two additional vectors `consPenaltyLin` and `consPenaltyQuad` (either of them, but
    not both, may be an empty vector). In particular, when only quadratic penalties for constraint
    violation are used, this is equivalent to the constrained least-square fitting problem.
    Some constraint penalties may be infinity, meaning that these constraints must be satisfied exactly
    (no extra variables are added for these constraints).
    Other arguments have the same meaning and default values as for `quadraticOptimizationSolve()`.
    \param[in]  A  is the matrix of the linear system (N_c x N_v);
    \param[in]  rhs  is its right-hand side (N_c);
    \param[in]  L  is the vector of linear penalties for N_v variables;
    \param[in]  Q  is the matrix (probably a diagonal one, or even an empty one) of quadratic penalties
    for N_v variables;
    \param[in]  consPenaltyLin  is the vector of (non-negative) linear penalties for violating each of
    N_c constraints (may be empty, meaning no linear penalties);
    \param[in]  consPenaltyQuad is the vector of (non-negative) quadratic penalties for constraint
    violation (N_c elements, may be empty if no quadratic penalties - independently from the existence
    of quadratic penalties for variables in the `Q` matrix);
    \param[in]  xmin  is the vector of lower bounds on the solution (N_v or empty);
    \param[in]  xmax  is the vector of upper bounds (N_v);
    \tparam  NumT  is the numerical type of the input arrays (float or double).
    \return  the solution vector (N_v);
    \throw  std::runtime_error in case of a problem reported by the solver.
*/
template<typename NumT>
std::vector<double> quadraticOptimizationSolveApprox(
    const     IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,
    const std::vector<NumT>& L = std::vector<NumT>(),
    const     IMatrix<NumT>& Q = BandMatrix<NumT>(),
    const std::vector<NumT>& consPenaltyLin  = std::vector<NumT>(),
    const std::vector<NumT>& consPenaltyQuad = std::vector<NumT>(),
    const std::vector<NumT>& xmin = std::vector<NumT>(),
    const std::vector<NumT>& xmax = std::vector<NumT>());

}  // namespace
