#ifndef HAVE_PYTHON
#undef HAVE_CVXOPT
#endif

#ifdef HAVE_CVXOPT
#include <cvxopt.h>  // C interface to LP/QP solver written in Python
#if PY_MAJOR_VERSION >= 3
#define PyString_AsString PyUnicode_AsUTF8
#endif
#endif

#ifdef HAVE_GLPK
#include <glpk.h>    // GNU linear programming kit - optimization problem solver
#include <csetjmp>
#endif

#include "math_optimization.h"
#include "math_core.h"
#include "utils.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace math{

namespace{
// Auxiliary matrix interfaces representing various special-structure matrices constructed on the fly

/** Matrix augmented with an extra block at the right end, containing one value per column:

    +-------Nvar columns---------Naug--+     |v|
    | exactly constrained part |       |     |a|   | |
    |--------------------------|       |     |r|   |R|
    | approximately constr.pt. |1 -1   |  *  |s| = |H|
    |                          |     1 |     |_|   |S|
    +----------------------------------+     |#|   | |
          original matrix      extra cols    |#|

    The extra Naug columns added to the matrix correspond to "slack" variables:
    the linear combination of original variables in the corresponding row is augmented with
    one or two extra slack variables per constraint, whose linear coefficients are +1 or -1.
    These slack variables are penalized in the objective function, and if the original matrix
    equation had a feasible solution, these variables (#) would be zero.
    But if the RHS (constraints) cannot be satisfied exactly, then the slack variable,
    or one of the two variables in the row corresponding to the given constraint,
    will be nonzero and will contribute to the cost function (linear or quadratic, or both).
    In case of one extra variable per row, its lower and upper limits are not restricted,
    and it contributes to the quadratic penalty only (regardless of its sign, the penalty is >0).
    In case of two extra variables, they are both limited to non-negative values, but have opposite
    signs in the linear combination, so only one of them will be >0 and contribute to the penalty.
*/
template<typename NumT>
class AugmentedMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& M;             ///< the original matrix
    const std::vector<size_t>& rowAug;  ///< row index of the corresponding column in the augmented part
    const std::vector<NumT>&   valAug;  ///< values in the augmented part (one value in each column)
public:
    AugmentedMatrix(const IMatrix<NumT>& src,
        const std::vector<size_t>& _rowAug, const std::vector<NumT>& _valAug)
    :
        IMatrix<NumT>(src.rows(), src.cols() + _rowAug.size()), M(src), rowAug(_rowAug), valAug(_valAug)
    {
        assert(rowAug.size() == valAug.size());
    }
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;
    virtual size_t size() const { return rows() * cols(); }
    virtual NumT at(size_t row, size_t col) const {
        if(col < M.cols())
            return M.at(row, col);
        col -= M.cols();
        assert(col < rowAug.size());
        if(row == rowAug[col])
            return valAug[col];
        return 0;
    }
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {  // dense matrix indexing
        row = index / cols();
        col = index % cols();
        return at(row, col);
    }
};

/// Matrix of quadratic penalties with extra diagonal terms for slack variables
template<typename NumT>
class AugmentedQuadMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& Q;        ///< the original matrix
    const std::vector<NumT>& pen;  ///< additional elements along the main diagonal
    const size_t Qsize, Qrows, Nadd;
public:
    AugmentedQuadMatrix(const IMatrix<NumT>& src, const std::vector<NumT>& _pen):
        IMatrix<NumT>(src.rows()+_pen.size(), src.cols()+_pen.size()),
        Q(src), pen(_pen), Qsize(Q.size()), Qrows(Q.rows()), Nadd(pen.size()) {};
    virtual size_t size()    const { return Qsize + Nadd; }
    virtual NumT at(const size_t row, const size_t col) const {
        if(col < Qrows)
            return Q.at(row, col);
        if(col == row)
            return pen.at(col-Qrows);
        return 0;
    }
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {
        if(index<Qsize)
            return Q.elem(index, row, col);
        row = col = index - Qsize + Qrows;
        return pen.at(index-Qsize);
    }
};

/// Matrix interface for accessing a subset of rows of the original matrix
template<typename NumT>
class RowSubsetMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& M;
    const std::vector<size_t> rowSubset;
public:
    RowSubsetMatrix(const IMatrix<NumT>& src, const std::vector<size_t>& _rowSubset) :
        IMatrix<NumT>(_rowSubset.size(), src.cols()), M(src), rowSubset(_rowSubset) {}
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;
    virtual size_t size() const { return rows() * cols(); }
    virtual NumT at(size_t row, size_t col) const {
        assert(row < rows());
        return M.at(rowSubset[row], col);
    }
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {  // dense matrix indexing
        row = index / cols();
        col = index % cols();
        return at(row, col);
    }
};

/// Matrix interface for computing the product  M^T diag(p) M + Q  on the fly,  where the matrix M
/// has r rows and c columns, p is a diagonal matrix of length r, and matrix Q is c by c
template<typename NumT>
class SelfProductMatrix: public IMatrix<NumT> {
public:
    const IMatrix<NumT>& M;      ///< original matrix M
    const std::vector<NumT> p;   ///< scaling coefficients p for each row of the matrix M
    const IMatrix<NumT>& Q;      ///< additional matrix Q (may be empty)
    SelfProductMatrix(const IMatrix<NumT>& src,
        const std::vector<NumT>& _p, const IMatrix<NumT>& _Q)
    :
        IMatrix<NumT>(src.cols(), src.cols()), M(src), p(_p), Q(_Q)
    {
        assert(M.rows() == p.size() &&
            (Q.size()==0 || (M.cols() == Q.cols() && Q.rows() == Q.cols()) ));
    }
    using IMatrix<NumT>::cols;
    virtual size_t size() const { return cols() * cols(); }
    // note: this method is actually not used, as it is very slow;
    // instead, the matrix is initialized by a dedicated routine that uses fast blas operations
    virtual NumT at(size_t row, size_t col) const {
        NumT result = 0;
        for(size_t i=0; i<M.rows(); i++)
            if(p[i] < INFINITY)  // infinity means zero in this context
                result += p[i] * M.at(i, row) * M.at(i, col);
        if(Q.size())
            result += Q.at(row, col);
        return result;
    }
    // likewise, this is not used
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {
        row = index / cols();
        col = index % cols();
        return at(row, col);
    }
};

}  // internal namespace

#ifdef HAVE_GLPK
//------- LP solver from the GNU linear programming kit -------//
namespace{
static std::jmp_buf err_buf;
/// change the default behaviour on GLPK error to something that doesn't crash program right away
void glpk_error_harmless(void*) {
    std::longjmp(err_buf, 1);
}
}

template<typename NumT>
std::vector<double> linearOptimizationSolve(const IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,  const std::vector<NumT>& L,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    size_t numVariables   = A.cols();
    size_t numConstraints = A.rows();
    if( rhs.size()!=numConstraints ||
        (!L.empty()    && L.size()!=numVariables) ||
        (!xmin.empty() && xmin.size()!=numVariables) ||
        (!xmax.empty() && xmax.size()!=numVariables) )
        throw std::invalid_argument("linearOptimizationSolve: invalid size of input arrays");

    // error handling
    glp_error_hook(glpk_error_harmless, NULL);
    if(setjmp(err_buf)) { 
        // this trick saves the current execution state on the first call,
        // and if an error occurs inside GLPK then the modified error handler is invoked,
        // and the control is returned here to the following statement
        glp_free_env();
        throw std::runtime_error("Error in linearOptimizationSolve");
    }

    // set up the problem
    glp_prob* problem = glp_create_prob();
    glp_set_obj_dir(problem, GLP_MIN);
    glp_add_rows(problem, numConstraints);
    glp_add_cols(problem, numVariables);
    for(size_t c=0; c<numConstraints; c++)
        glp_set_row_bnds(problem, c+1, GLP_FX, rhs[c], rhs[c]);
    for(size_t v=0; v<numVariables; v++) {
        double xminval = xmin.empty()? 0 : xmin[v];
        if(!xmax.empty() && isFinite(xmax[v]))
            glp_set_col_bnds(problem, v+1, GLP_DB, xminval, xmax[v]);
        else
            glp_set_col_bnds(problem, v+1, GLP_LO, xminval, 0 /*ignored*/);
        if(!L.empty() && L[v]!=0)
            glp_set_obj_coef(problem, v+1, L[v]);
    }
    // fill the linear matrix using triplets (row, column, value)
    {   // open a block so that temp variables are destroyed after it is closed.
        std::vector<int> ic;
        std::vector<int> iv;
        std::vector<double> mat;
        size_t numTotal=A.size();
        ic.reserve (numTotal+1); ic.push_back(0);  // 0th element is always ignored
        iv.reserve (numTotal+1); iv.push_back(0);  // (GLPK uses indices starting from 1)
        mat.reserve(numTotal+1); mat.push_back(0);
        for(size_t i=0; i<numTotal; i++) {
            size_t c, v;
            double val = A.elem(i, c, v);
            if(val!=0) {
                ic.push_back(c+1);
                iv.push_back(v+1);
                mat.push_back(val);
            }
        }
        glp_load_matrix(problem, ic.size()-1, &ic.front(), &iv.front(), &mat.front());
    }

    // solve the problem using the interior-point method
    int status = glp_interior(problem, NULL);
    if(glp_ipt_status(problem) != GLP_OPT) status=1;  // infeasible

    // retrieve the solution
    std::vector<double> result(numVariables);
    for(size_t v=0; v<numVariables; v++) {
        double vmin = xmin.empty() ? 0 : xmin[v];
        double vmax = xmax.empty() ? INFINITY : xmax[v];
        // correct possible roundoff errors that may lead to the value being outside the limits
        result[v] = clip(glp_ipt_col_prim(problem, v+1), vmin, vmax);
    }
    glp_delete_prob(problem);

    if(status==0)
        return result;  // success
    throw std::runtime_error(
        status==1?           "linearOptimizationSolve: problem is infeasible" :
        status==GLP_EFAIL?   "linearOptimizationSolve: empty problem":
        status==GLP_ENOCVG?  "linearOptimizationSolve: bad convergence":
        status==GLP_EITLIM?  "linearOptimizationSolve: number of iterations exceeded limit":
        status==GLP_EINSTAB? "linearOptimizationSolve: numerical instability":
        "linearOptimizationSolve: unknown error");
}
#else
// GLPK is not available - use the quadratic solver instead (if possible)
template<typename NumT>
std::vector<double> linearOptimizationSolve(const IMatrix<NumT>& A,
    const std::vector<NumT>& rhs,  const std::vector<NumT>& L,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
#ifdef HAVE_CVXOPT
    return quadraticOptimizationSolve(A, rhs, L, BandMatrix<NumT>(std::vector<NumT>()), xmin, xmax);
#endif
    /*else*/ throw std::runtime_error("linearOptimizationSolve not implemented");
    (void)A; (void)rhs; (void)L; (void)xmin; (void)xmax;  // silence warnings about unused params
}
#endif

#ifdef HAVE_CVXOPT

namespace {  // internal

static bool solver_called = false;  ///< prevent calling the solver more than once
static PyObject
    /// functions from various modules (initialized once, when CVXOPT is loaded)
    *fnc_solvers_lp   = NULL,
    *fnc_solvers_qp   = NULL,
    *fnc_blas_syrk    = NULL,
    *fnc_blas_gemv    = NULL,
    *fnc_lapack_potrf = NULL,
    *fnc_lapack_potrs = NULL,
    *fnc_kkt_getsolve = NULL,
    *fnc_kkt_runsolve = NULL,
    /// pointers to various matrices, re-allocated every time the optimization solver is called
    *objectiveQuad    = NULL,
    *coefMatrix       = NULL,
    *objectiveLin     = NULL,
    *consIneq         = NULL,
    *consIneqRhs      = NULL,
    *coefRhs          = NULL,
    *matAscaled       = NULL,
    *matK             = NULL,
    *vecWdi           = NULL;

/// helper routine for calling a python function and cleaning up memory
static bool callPythonFunction(PyObject* fnc, PyObject* args)
{
    PyObject* result = PyObject_CallObject(fnc, args);
    bool success = result != NULL;
    Py_XDECREF(result);
    Py_DECREF(args);
    return success;
}

/// helper routine to create a CVXOPT-compatible dense matrix of a special kind:
/// M^T diag(p) M + Q, where the elements of the matrices M, Q and the diagonal matrix p
/// are accessed through a SelfProductMatrix proxy object.
/// To improve efficiency, we construct a temporary scaled matrix M, then multiply it
/// by its own transpose with a fast BLAS routine, and then (optionally) add the matrix Q
template<typename NumT>
PyObject* initPyMatrixSelfProduct(const SelfProductMatrix<NumT>& S)
{
    PySys_WriteStdout("Using a dense quadratic penalty matrix\n");
    // 1. create a temporary matrix, in which the rows of the input matrix M are multiplied
    // by square roots of the elements of the vector p, and the whole thing is transposed
    size_t nrows = S.M.rows(), ncols = S.M.cols();
    assert(S.Q.size()==0 || (S.Q.cols() == S.Q.rows() && S.Q.cols() == ncols));
    PyObject *MTscaled = (PyObject*)Matrix_New(ncols, nrows, DOUBLE);  // temporary matrix M^T sqrt(p)
    PyObject *pyMatrix = (PyObject*)Matrix_New(ncols, ncols, DOUBLE);  // result matrix
    if(!MTscaled || !pyMatrix) {
        Py_XDECREF(MTscaled);
        Py_XDECREF(pyMatrix);
        return NULL;
    }
    for(size_t ir=0; ir<nrows; ir++) {
        double p = sqrt(S.p[ir]);
        if(p==INFINITY)  p=0;   // infinity in this context means that the row is ignored
        for(size_t ic=0; ic<ncols; ic++)
            MAT_BUFD(MTscaled)[ir*ncols+ic] = S.M.at(ir, ic) * p;
    }

    // 2. call the external function syrk from blas, computing the matrix K = Mscaled^T Mscaled
    if(!callPythonFunction(fnc_blas_syrk, Py_BuildValue("(OO)", MTscaled, pyMatrix))) {
        PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, "Error in syrk");
        Py_DECREF(MTscaled);
        Py_DECREF(pyMatrix);
        return NULL;
    }

    // 3. add the elements of the other input matrix Q (if provided - a square ncol*ncol matrix)
    if(S.Q.size()) {
        if(S.Q.size() == S.Q.rows() * S.Q.cols()) {  // dense matrix - loop over rows and columns
            for(size_t i=0; i<ncols; i++)
                for(size_t j=0; j<ncols; j++)
                    MAT_BUFD(pyMatrix)[i*ncols+j] += S.Q.at(i, j);
        } else {  // loop over nonzero elements
            for(size_t i=0, size=S.Q.size(); i<size; i++) {
                size_t ir, ic;
                double val = S.Q.elem(i, ir, ic);
                MAT_BUFD(pyMatrix)[ic*ncols+ir] += val;
            }
        }
    }

    Py_DECREF(MTscaled);  // delete the temporary matrix
    return pyMatrix;
}

/// helper routine to create a CVXOPT-compatible dense or sparse matrix
template<typename NumT>
PyObject* initPyMatrix(const IMatrix<NumT>& M)
{
    // if we have a special kind of self-product matrix, use a more efficient initialization approach
    const SelfProductMatrix<NumT>* S = dynamic_cast<const SelfProductMatrix<NumT>*>(&M);
    if(S)
        return initPyMatrixSelfProduct<NumT>(*S);

    size_t nrows = M.rows(), ncols = M.cols(), ntotal = M.size();
    if(ntotal == nrows*ncols)
    {   // dense matrices are initialized item-by-item
        PyObject *pyMatrix = (PyObject*)Matrix_New(nrows, ncols, DOUBLE);
        if(!pyMatrix) return NULL;
        for(size_t ir=0; ir<nrows; ir++)
            for(size_t ic=0; ic<ncols; ic++)
                MAT_BUFD(pyMatrix)[ic*nrows+ir] = M.at(ir, ic);  // column-major order used in CVXOPT
        return pyMatrix;
    } else {
        // To initialize sparse matrices, additional 3 vectors are allocated,
        // which hold triplets {col, row, value}
        PyObject *rowind = (PyObject*)Matrix_New(ntotal, 1, INT);
        PyObject *colind = (PyObject*)Matrix_New(ntotal, 1, INT);
        PyObject *values = (PyObject*)Matrix_New(ntotal, 1, DOUBLE);
        if(!colind || !rowind || !values) {
            Py_XDECREF(colind);
            Py_XDECREF(rowind);
            Py_XDECREF(values);
            return NULL;
        }
        for(size_t i=0; i<ntotal; i++) {
            size_t ir, ic;
            double val = M.elem(i, ir, ic);
            MAT_BUFI(rowind)[i]=ir;
            MAT_BUFI(colind)[i]=ic;
            MAT_BUFD(values)[i]=val;
        }
        PyObject* pyMatrix = (PyObject*)SpMatrix_NewFromIJV(
            (matrix*)rowind, (matrix*)colind, (matrix*)values, nrows, ncols, DOUBLE);
        Py_DECREF(colind);
        Py_DECREF(rowind);
        Py_DECREF(values);
        return pyMatrix;
    }
}

/// Custom KKT solver for a diagonal Hessian matrix and diagonal inequality constraints matrix, Part I.
/// This function is called on every iteration, assembles and factorizes the scaled matrix of
/// normal equations (or, more correctly, Karush-Kuhn-Tucker eqns), and returns the solve function.
/// The intermediate results are stored in global variables defined above.
/// The most expensive operation is the computation of matrix K = A S^{-1} A^T,
/// where A is the (fixed) coefficients matrix (numConstraints rows, numVariables columns),
/// and S is a diagonal matrix of numVariables scaling coefficients (different on each iteration).
/// Unfortunately, there is no BLAS routine for this operation, but only for a simpler one (syrk):
/// the matrix product K = B B^T, which requires O(numVariables * numConstraints^2).
/// Hence we first compute the diagonal scaling matrix S, then assemble an auxiliary scaled matrix
/// Ascaled = A S^{-1/2}, and then pass it to syrk. The resulting matrix K is then Cholesky factorized
/// by the LAPACK routine potrf, and subsequently used in the KKT solver, Part II.
/// This custom solver is applicable in the most common case, when the quadratic penalty matrix
/// is diagonal (or absent), and there is exactly one inequality constraint per variable,
/// and it greatly speeds up the computation, compared to the general case (implemented natively
/// in CVXOPT) when the matrices H (objectiveQuad), G (consIneq), W (scaling) and S are non-diagonal.
static PyObject* kkt_getsolve(PyObject* /*self*/, PyObject* args)
{
    PyObject* W = NULL;
    if(!PyArg_ParseTuple(args, "O!", &PyDict_Type, &W))
        return NULL;
    int numConstraints = MAT_NROWS(coefMatrix),
        numVariables   = MAT_NCOLS(coefMatrix),
        numConsIneq    =  SP_NROWS(consIneq);
    vecWdi       = PyDict_GetItemString(W, "di");   // inverse diagonal elements of the scaling matrix W
    PyObject* u1 = PyDict_GetItemString(W, "r");    // other elements of scaling matrix:
    PyObject* u2 = PyDict_GetItemString(W, "v");    // these should not be provided
    PyObject* u3 = PyDict_GetItemString(W, "beta");
    PyObject* u4 = PyDict_GetItemString(W, "dnl");
    if( !vecWdi || MAT_NROWS(vecWdi) != numConsIneq || MAT_NCOLS(vecWdi) != 1 ||
        (u1 && MAT_NROWS(u1)>0) ||   // these arguments should not be provided
        (u2 && MAT_NROWS(u2)>0) ||   // (or, rather, should have zero length)
        (u3 && MAT_NROWS(u3)>0) ||
        (u4 && MAT_NROWS(u4)>0) )
    {
        PyErr_SetString(PyExc_TypeError,
            "Custom KKT solver can't deal with non-diagonal scaling transformations");
        return NULL;
    }

    // in the following fragment of code we don't call any Python C API functions,
    // so may temporarily release GIL and let Python do its own business for a while.
    // \note OpenMP-parallelized loop over internal arrays.
    Py_BEGIN_ALLOW_THREADS

    // multiply the matrix of coefficients A by a diagonal matrix S^{-1/2}, storing the result in Ascaled
    const double *A = MAT_BUFD(coefMatrix);
    double *Ascaled = MAT_BUFD(matAscaled);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int v=0; v<numVariables; v++) {
        double Gs = v < numConsIneq ? SP_VALD(consIneq)[v] * MAT_BUFD(vecWdi)[v] : 0;
        double S  = pow_2(Gs) + (objectiveQuad ? SP_VALD(objectiveQuad)[v] : 0);
        double Si = 1./sqrt(S);
        for(int i=v*numConstraints, up=i+numConstraints; i<up; i++)
            Ascaled[i] = A[i] * Si;
    }

    // now re-acquire GIL
    Py_END_ALLOW_THREADS

    // call the external function syrk from blas, computing the matrix K = Ascaled Ascaled^T
    if(!callPythonFunction(fnc_blas_syrk, Py_BuildValue("(OO)", matAscaled, matK))) {
        PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, "Error in syrk");
        return NULL;
    }

    // call the external function potrf from lapack, computing the Cholesky factorization of the matrix K
    if(!callPythonFunction(fnc_lapack_potrf, Py_BuildValue("(O)", matK))) {
        PyErr_Print();
        PyErr_SetString(PyExc_RuntimeError, "Error in potrf");
        return NULL;
    }

    // return the solve function
    Py_INCREF(fnc_kkt_runsolve);
    return fnc_kkt_runsolve;
}

/// Custom KKT solver for a diagonal Hessian matrix and diagonal inequality constraints matrix, Part II.
/// This function is called twice per iteration, and uses the pre-computed matrices Ascaled, K,
/// stored in the global variables, to solve the KKT equations, replacing the input vectors x,y,z
/// (rhs of these eqns) by the solution vectors.
static PyObject* kkt_runsolve(PyObject* /*self*/, PyObject* args)
{
    int numConstraints = MAT_NROWS(coefMatrix),
        numVariables   = MAT_NCOLS(coefMatrix),
        numConsIneq    =  SP_NROWS(consIneq);
    PyObject *vecX=NULL, *vecY=NULL, *vecZ=NULL;
    if(!PyArg_ParseTuple(args, "OOO", &vecX, &vecY, &vecZ))
        return NULL;
    if( MAT_NROWS(vecX) != numVariables   || MAT_NCOLS(vecX) != 1 ||
        MAT_NROWS(vecY) != numConstraints || MAT_NCOLS(vecY) != 1 ||
        MAT_NROWS(vecZ) != numConsIneq    || MAT_NCOLS(vecZ) != 1 )
    {
        PyErr_SetString(PyExc_ValueError, "Custom KKT solver: invalid size of input arrays");
        return NULL;
    }
    double *x = MAT_BUFD(vecX), *z = MAT_BUFD(vecZ);
    std::vector<double> S(numVariables), T(numVariables);

    for(int v=0; v<numVariables; v++) {
        S[v] = objectiveQuad ? SP_VALD(objectiveQuad)[v] : 0;
        double zGs = 0;
        if(v < numConsIneq) {
            double di = MAT_BUFD(vecWdi)[v], Gs = SP_VALD(consIneq)[v] * di;
            S[v] += pow_2(Gs);
            z[v] *= di;
            zGs = Gs * z[v];
        }
        x[v]  = (zGs + x[v]) / S[v];
        T[v]  = x[v];
    };

    // call the external function gemv from blas
    if(!callPythonFunction(fnc_blas_gemv,
        Py_BuildValue("(OOOsdd)", coefMatrix, vecX, vecY, /*trans*/ "N", /*alpha*/ 1.0, /*beta*/ -1.0)))
        return NULL;

    // call the external function potrs from lapack
    if(!callPythonFunction(fnc_lapack_potrs, Py_BuildValue("(OO)", matK, vecY)))
        return NULL;

    // again call the external function gemv from blas
    if(!callPythonFunction(fnc_blas_gemv,
        Py_BuildValue("(OOOsdd)", coefMatrix, vecY, vecX, /*trans*/ "T", /*alpha*/ 1.0, /*beta*/ 0.0)))
        return NULL;

    for(int v=0; v<numVariables; v++) {
        x[v] = T[v] - x[v] / S[v];
        if(v < numConsIneq) {
            double Gs =  SP_VALD(consIneq)[v] * MAT_BUFD(vecWdi)[v];
            z[v] = Gs * x[v] - z[v];
        }
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/// descriptors of these functions for Python
static PyMethodDef descr_kkt_getsolve = {"getsolve", kkt_getsolve, METH_VARARGS, ""};
static PyMethodDef descr_kkt_runsolve = {"runsolve", kkt_runsolve, METH_VARARGS, ""};

/// one-time initialization of CVXOPT python module
static void initCVXOPT()
{
    if(fnc_solvers_lp)  // if this routine has been called previously, no need to re-initialize
        return;
    Py_Initialize();
    if(import_cvxopt() < 0) {
        throw std::runtime_error("quadraticOptimizationSolve: error importing CVXOPT python module");
    }
    PyObject *module_solvers = PyImport_ImportModule("cvxopt.solvers");
    PyObject *module_blas    = PyImport_ImportModule("cvxopt.blas");
    PyObject *module_lapack  = PyImport_ImportModule("cvxopt.lapack");
    if(module_solvers && module_blas && module_lapack) {
        fnc_solvers_lp   = PyObject_GetAttrString(module_solvers, "conelp");
        fnc_solvers_qp   = PyObject_GetAttrString(module_solvers, "coneqp");
        fnc_blas_syrk    = PyObject_GetAttrString(module_blas,    "syrk");
        fnc_blas_gemv    = PyObject_GetAttrString(module_blas,    "gemv");
        fnc_lapack_potrf = PyObject_GetAttrString(module_lapack,  "potrf");
        fnc_lapack_potrs = PyObject_GetAttrString(module_lapack,  "potrs");
        fnc_kkt_getsolve = PyCFunction_New(&descr_kkt_getsolve, NULL);  // register our custom KKT solver
        fnc_kkt_runsolve = PyCFunction_New(&descr_kkt_runsolve, NULL);
    }
    if(!fnc_solvers_lp || !fnc_blas_syrk || !fnc_lapack_potrs || !fnc_kkt_runsolve) {
        throw std::runtime_error("quadraticOptimizationSolve: error initializing CVXOPT python module");
    }
    // never release the pointers to Python objects allocated above... not a big deal though
}

}  // internal namespace

template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& _xmin, const std::vector<NumT>& _xmax)
{
    size_t numVariables   = A.cols();
    size_t numConstraints = A.rows();
    bool doQP = Q.size()!=0;
    if( rhs.size()!=numConstraints ||
        (!L.empty()     && L.size()!=numVariables) ||
        (!_xmin.empty() && _xmin.size()!=numVariables) ||
        (!_xmax.empty() && _xmax.size()!=numVariables) ||
        (doQP && (Q.rows()!=numVariables || Q.cols()!=numVariables)) )
        throw std::invalid_argument("quadraticOptimizationSolve: invalid size of input arrays");

    // import and set up python extension module
    initCVXOPT();

    // make sure this function is not called more than once at any given time
    if(solver_called)
        throw std::runtime_error("quadraticOptimizationSolve is non-reentrant");
    solver_called = true;

    // count inequality constraints: up to two (lower and/or upper) constraints per variable
    std::vector<NumT> xmin = _xmin.empty() ? std::vector<NumT>(numVariables, 0) : _xmin;
    std::vector<NumT> xmax = _xmax.empty() ? std::vector<NumT>(numVariables, INFINITY) : _xmax;
    size_t numConsIneq = 0;
    for(size_t v=0; v<numVariables; v++) {
        if(xmin[v] > -INFINITY) numConsIneq++;
        if(xmax[v] < +INFINITY) numConsIneq++;
    }

    // check if we can use custom (optimized) KKT solver: a few conditions must be satisfied
    bool useCustomKKT =
        _xmax.empty() &&                           // only one (lower) inequality constraint per variable
        (!doQP || Q.size() == numVariables) &&     // quadratic matrix is diagonal or absent altogether
        A.size() == numVariables * numConstraints; // coefficient matrix is dense

    // create matrices (stored as global variables!!)
    objectiveQuad = doQP ? initPyMatrix(Q) : NULL;
    coefMatrix    = initPyMatrix(A);
    objectiveLin  = (PyObject*)(Matrix_New(numVariables, 1, DOUBLE));
    consIneq      = (PyObject*)(SpMatrix_New(numConsIneq, numVariables, numConsIneq, DOUBLE));
    consIneqRhs   = (PyObject*)(Matrix_New(numConsIneq, 1, DOUBLE));
    coefRhs       = (PyObject*)(Matrix_New(numConstraints, 1, DOUBLE));
    matAscaled    = useCustomKKT ? (PyObject*)(Matrix_New(numConstraints, numVariables, DOUBLE)) : NULL;
    matK          = useCustomKKT ? (PyObject*)(Matrix_New(numConstraints, numConstraints, DOUBLE)) : NULL;
    if( !objectiveLin  || (doQP && !objectiveQuad) || !consIneq || !consIneqRhs ||
        !coefMatrix || !coefRhs || (useCustomKKT && (!matAscaled || !matK ) ) )
    {
        PyErr_Print();
        Py_XDECREF(objectiveLin);
        Py_XDECREF(objectiveQuad);
        Py_XDECREF(coefMatrix);
        Py_XDECREF(consIneq);
        Py_XDECREF(consIneqRhs);
        Py_XDECREF(coefRhs);
        Py_XDECREF(matAscaled);
        Py_XDECREF(matK);
        solver_called = false;
        throw std::runtime_error("quadraticOptimizationSolve: error allocating matrices");
    }

    // init objective func, linear part
    for(size_t v=0; v<numVariables; v++)
        MAT_BUFD(objectiveLin)[v] = L.empty()? 0 : L[v];

    // init inequality constraints
    for(size_t v=0, c=0; v<numVariables; v++) {
        SP_COL(consIneq)[v] = c;
        if(xmin[v] > -INFINITY) {  // add lower limit
            SP_ROW (consIneq)[c] = v;
            SP_VALD(consIneq)[c] = -1.;
            MAT_BUFD(consIneqRhs)[c] = -xmin[v];
            c++;
        }
        if(xmax[v] < +INFINITY) {  // add upper limit
            SP_ROW (consIneq)[c]= v;
            SP_VALD(consIneq)[c]= 1.;
            MAT_BUFD(consIneqRhs)[c] = xmax[v];
            c++;
        }
    }
    SP_COL(consIneq)[numVariables] = numConsIneq;

    // init rhs
    for(size_t c=0; c<numConstraints; c++)
        MAT_BUFD(coefRhs)[c] = rhs[c];

    // construct the dictionary of named arguments
    PyObject *posargs = PyTuple_New(0);  // positional arguments - not used but must be present
    PyObject *args    = PyDict_New();    // named arguments
    if(doQP)
        PyDict_SetItemString(args, "P", objectiveQuad);
    PyDict_SetItemString(args, doQP ? "q" : "c", objectiveLin);
    PyDict_SetItemString(args, "G", consIneq);
    PyDict_SetItemString(args, "h", consIneqRhs);
    PyDict_SetItemString(args, "A", coefMatrix);
    PyDict_SetItemString(args, "b", coefRhs);
    if(useCustomKKT) {
        PyDict_SetItemString(args, "kktsolver", fnc_kkt_getsolve);
        PySys_WriteStdout("Using a custom optimized KKT solver\n");
    }
    PySys_WriteStdout("numVariables=%zu, numConstraints=%zu, numConsIneq=%zu\n",
        numVariables, numConstraints, numConsIneq);

    // call the solver
    PyObject *result_dict = PyObject_Call(doQP ? fnc_solvers_qp : fnc_solvers_lp, posargs, args);

    // analyze the result
    PyObject *status = result_dict ? PyDict_GetItemString(result_dict, "status") : NULL;
    bool feasible = status != NULL && std::string(PyString_AsString(status)) == "optimal";
    PyObject *sol = result_dict ? PyDict_GetItemString(result_dict, "x") : NULL;

    // retrieve and store the solution
    std::vector<double> result(numVariables);
    // correct possible roundoff errors that may lead to the value being outside the limits
    for(size_t v=0; feasible && v<numVariables; v++) {
        result[v] = clip(MAT_BUFD(sol)[v], (double)xmin[v], (double)xmax[v]);
    }

    // cleanup
    Py_XDECREF(result_dict);
    Py_DECREF(posargs);
    Py_DECREF(args);
    Py_DECREF(objectiveLin);
    Py_XDECREF(objectiveQuad);
    Py_DECREF(consIneq);
    Py_DECREF(consIneqRhs);
    Py_DECREF(coefMatrix);
    Py_DECREF(coefRhs);
    Py_XDECREF(matAscaled);
    Py_XDECREF(matK);
    solver_called = false;

    if(feasible)
        return result;
    if(!result_dict) {
        if(PyErr_ExceptionMatches(PyExc_KeyboardInterrupt))
            throw std::runtime_error(utils::CtrlBreakHandler::message());
        PyErr_Print();  // also clears the exception state
        throw std::runtime_error("quadraticOptimizationSolve: solver returned error");
    }
    throw std::runtime_error("quadraticOptimizationSolve: problem is infeasible");
}
#else
template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
#ifdef HAVE_GLPK
    if(Q.size()==0)  // linear problems will be redirected to the appropriate solver
        return linearOptimizationSolve(A, rhs, L, xmin, xmax);
#endif
    /*else*/ throw std::runtime_error("quadraticOptimizationSolve not implemented");
    // silence warnings about unused params
    (void)A; (void)rhs; (void)L; (void)Q; (void)xmin; (void)xmax;
}
#endif

template<typename NumT>
std::vector<double> linearOptimizationSolveApprox(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const std::vector<NumT>& consPenaltyLin,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    if(consPenaltyLin.empty())
        throw std::invalid_argument(
            "linearOptimizationSolveApprox: constraint penalties must be provided");
    size_t numVariables   = A.cols();
    size_t numConstraints = A.rows();
    if( rhs.size()!=numConstraints || consPenaltyLin.size()!=numConstraints ||
        (!L.empty() && L.size()!=numVariables) )
        throw std::invalid_argument("linearOptimizationSolveApprox: invalid size of input arrays");
    /// augment the original vectors with extra elements for slack variables
    std::vector<NumT> Laug(numVariables);
    if(!L.empty())   // copy the array of penalty coefs for variables
        std::copy(L.begin(), L.end(), Laug.begin());
    std::vector<size_t> rowAug;  // row indices in the augmented part of the matrix
    std::vector<NumT>   valAug;  // coefficients (+-1) in the augmented part of the matrix
    for(size_t c=0; c<numConstraints; c++) {
        if(!(consPenaltyLin[c]>=0))
            throw std::invalid_argument(
                "linearOptimizationSolveApprox: constraint penalties must be non-negative");
        if(consPenaltyLin[c]>0 && consPenaltyLin[c]<INFINITY) {
            rowAug.push_back(c);
            rowAug.push_back(c);
            valAug.push_back(1);
            valAug.push_back(-1);
            Laug.push_back(consPenaltyLin[c]);
            Laug.push_back(consPenaltyLin[c]);
        }
    }
    std::vector<NumT> xminAug(xmin);  // lower bounds on all variables (original + augmented)
    if(!xmin.empty())
        xminAug.insert(xminAug.end(), rowAug.size(), 0);
    std::vector<NumT> xmaxAug(xmax);
    if(!xmax.empty())
        xmaxAug.insert(xmaxAug.end(), rowAug.size(), INFINITY);
    std::vector<double> result = linearOptimizationSolve(
        AugmentedMatrix<NumT>(A, rowAug, valAug), rhs, Laug, xminAug, xmaxAug);
    result.resize(numVariables);  // chop off extra slack variables
    return result;
}

template<typename NumT>
std::vector<double> quadraticOptimizationSolveApprox(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& consPenaltyLin, const std::vector<NumT>& consPenaltyQuad,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    if(consPenaltyLin.empty() && consPenaltyQuad.empty())
        throw std::invalid_argument(
            "quadraticOptimizationSolveApprox: constraint penalties must be provided");
    size_t numVariables   = A.cols();
    size_t numConstraints = A.rows();
    if( rhs.size()!=numConstraints ||
        (!consPenaltyLin.empty()  && consPenaltyLin. size()!=numConstraints) ||
        (!consPenaltyQuad.empty() && consPenaltyQuad.size()!=numConstraints) ||
        (!L.empty() && L.size()!=numVariables) ||
        (Q.rows()>0 && Q.rows()!=numVariables) || (Q.rows()!=Q.cols()) )
        throw std::invalid_argument("quadraticOptimizationSolveApprox: invalid size of input arrays");

    // check which constraints are 'loose', i.e. penalty is not infinite,
    // and add one or two penalized variables for each loose constraint
    std::vector<NumT> xminAug(xmin);  // lower bounds on all variables (original + augmented)
    if(xmin.empty())                  // if original lower bounds are not provided, assume zeroes
        xminAug.assign(numVariables, 0);
    std::vector<size_t> rowExact;// indices of exact constraints
    std::vector<NumT>   rhsExact;// corresponding constraint values
    std::vector<size_t> rowAug;  // indices of loose constraints (rows in the augmented matrix)
    std::vector<NumT>   valAug;  // coefficients (+-1) in the augmented part of the matrix
    std::vector<NumT> Laug(numVariables), Qaug;
    if(!L.empty())               // copy the array of penalty coefs for variables
        std::copy(L.begin(), L.end(), Laug.begin());
    bool havePenLin = false;
    for(size_t c=0; c<numConstraints; c++) {
        NumT penLin  = consPenaltyLin .empty() ? 0 : consPenaltyLin [c];
        NumT penQuad = consPenaltyQuad.empty() ? 0 : consPenaltyQuad[c];
        if(penLin==INFINITY || penQuad==INFINITY) {
            // this constraint must be satisfied exactly
            rowExact.push_back(c);
            rhsExact.push_back(rhs[c]);
        } else if(penLin>0) {
            // if linear penalty is provided, add two slack variables for each constraint,
            // which are both required to be nonnegative
            rowAug.push_back(c);
            rowAug.push_back(c);
            valAug.push_back(1);
            valAug.push_back(-1);
            Laug.push_back(penLin);
            Laug.push_back(penLin);
            Qaug.push_back(penQuad);
            Qaug.push_back(penQuad);
            xminAug.push_back(0);
            xminAug.push_back(0);
            havePenLin = true;
        } else if(penQuad>0) {
            // if only quadratic penalty is provided, add one slack variable
            // without any inequality constraints (i.e. -infinity)
            rowAug.push_back(c);
            valAug.push_back(1);
            Laug.push_back(0);
            Qaug.push_back(penQuad);
            xminAug.push_back(-INFINITY);
        } else if(!(penLin>0) || !(penQuad>0))
            throw std::invalid_argument("quadraticOptimizationSolveApprox: "
                "constraint penalties must be nonnegative (possibly infinite)");
    }
    std::vector<NumT> xmaxAug(xmax);
    if(!xmax.empty()) {
        // if upper limits are provided, add infinite upper limits for the augmented variables
        xmaxAug.insert(xmaxAug.end(), rowAug.size(), INFINITY);
        assert(xminAug.size() == xmaxAug.size());
    }

    // if the number of constraints with quadratic penalty is large enough, or if the matrix Q is dense,
    // choose a different strategy: do not use slack variables, which would have increased
    // the size of the constraints matrix to numCons*(numVar+numCons),
    // but instead keep a subset of the original constraint matrix with size numConsExact*numVar
    // (where numConsExact is the number of constraints that have to be satisfied exactly),
    // assemble (or add to) a dense numVar*numVar matrix of quadratic penalties,
    // and add extra terms to the vector of linear penalties
    if(!consPenaltyQuad.empty() && !havePenLin &&
        ((Q.rows() > 0 && Q.size() == Q.cols() * Q.rows()) || numConstraints*2 >= numVariables))
    {
        std::vector<NumT> Ladd(numVariables);
        if(!L.empty())
            Ladd = L;
        for(size_t c=0; c<numConstraints; c++) {
            if(consPenaltyQuad[c] == INFINITY)
                continue;
            for(size_t v=0; v<numVariables; v++)
                Ladd[v] += -rhs[c] * consPenaltyQuad[c] * A.at(c, v);
        }
        return quadraticOptimizationSolve(
            RowSubsetMatrix<NumT>(A, rowExact), rhsExact, Ladd,
            SelfProductMatrix<NumT>(A, consPenaltyQuad, Q), xmin, xmax);
    }

    std::vector<double> result = quadraticOptimizationSolve(
        AugmentedMatrix<NumT>(A, rowAug, valAug), rhs, allZeros(Laug) ? std::vector<NumT>() : Laug,
        Q.size()==0 && allZeros(Qaug) ?   // in this case don't create any quadratic matrix
            static_cast<const IMatrix<NumT>&>(BandMatrix<NumT>()) :
            // if either the original quadratic matrix for numVariables was non-empty,
            // or the additional diagonal elements for numConstraints penalties were specified,
            // will create an augmented quadratic matrix, substituting the unspecified elements with zeros
            static_cast<const IMatrix<NumT>&>(AugmentedQuadMatrix<NumT>(Q.size()==0 ?
                static_cast<const IMatrix<NumT>&>(BandMatrix<NumT>(std::vector<NumT>(numVariables, 0))) :
                Q, Qaug)),
        xminAug, xmaxAug);
    result.resize(numVariables);  // chop off extra slack variables
    return result;
}

// explicit instantiations for NumT = float and double
template std::vector<double> linearOptimizationSolve(const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&);

template std::vector<double> linearOptimizationSolve(const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);

template std::vector<double> quadraticOptimizationSolve(const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&, const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&);

template std::vector<double> quadraticOptimizationSolve(const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&, const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&);

template std::vector<double> linearOptimizationSolveApprox(const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&);

template std::vector<double> linearOptimizationSolveApprox(const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);

template std::vector<double> quadraticOptimizationSolveApprox(const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&, const IMatrix<float>&,
    const std::vector<float>&, const std::vector<float>&,
    const std::vector<float>&, const std::vector<float>&);

template std::vector<double> quadraticOptimizationSolveApprox(const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&, const IMatrix<double>&,
    const std::vector<double>&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);
}
