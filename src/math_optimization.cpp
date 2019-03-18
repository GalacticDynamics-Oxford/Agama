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
#include <stdexcept>
#include <cmath>

namespace math{

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
        result[v] = clamp(glp_ipt_col_prim(problem, v+1), vmin, vmax);
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

/// helper routine to create CVXOPT-compatible dense or sparse matrix
template<typename NumT>
PyObject* initPyMatrix(const IMatrix<NumT>& M)
{
    size_t nrows = M.rows(), ncols = M.cols(), ntotal = M.size();
    if(ntotal == nrows*ncols)
    {   // dense matrices are initialized item-by-item
        PyObject *pyMatrix = (PyObject*)Matrix_New(nrows, ncols, DOUBLE);
        if(!pyMatrix) return NULL;
        for(size_t ir=0; ir<nrows; ir++)
            for(size_t ic=0; ic<ncols; ic++)
                MAT_BUFD(pyMatrix)[ic*nrows+ir] = M.at(ir, ic);  // column-major order used in CVXOPT
        return pyMatrix;
    } else {  // To initialize sparse matrices, additional 3 vectors are allocated
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

template<typename NumT>
std::vector<double> quadraticOptimizationSolve(
    const IMatrix<NumT>& A, const std::vector<NumT>& rhs,
    const std::vector<NumT>& L, const IMatrix<NumT>& Q,
    const std::vector<NumT>& xmin, const std::vector<NumT>& xmax)
{
    size_t numVariables   = A.cols();
    size_t numConstraints = A.rows();
    bool doQP = Q.size()!=0;
    if( rhs.size()!=numConstraints ||
        (!L.empty()    && L.size()!=numVariables) ||
        (!xmin.empty() && xmin.size()!=numVariables) ||
        (!xmax.empty() && xmax.size()!=numVariables) ||
        (doQP && (Q.rows()!=numVariables || Q.cols()!=numVariables)) )
        throw std::invalid_argument("quadraticOptimizationSolve: invalid size of input arrays");

    Py_Initialize();
    PyObject *solvers = import_cvxopt() < 0 ? NULL : PyImport_ImportModule("cvxopt.solvers");
    if(!solvers) {
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error importing CVXOPT python module");
    }
    PyObject *problem = PyObject_GetAttrString(solvers, doQP ? "qp" : "lp");
    if(!problem) {
        Py_DECREF(solvers);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error initializing CVXOPT python module");
    }

    // count inequality constraints: one (lower) or two (lower+upper) constraints per variable
    size_t numConsIneq = numVariables;
    for(size_t v=0; v<xmax.size(); v++)
        if(isFinite(xmax[v]))
           numConsIneq++;

    // create matrices
    PyObject *objectiveQuad = doQP ? initPyMatrix(Q) : NULL;
    PyObject *coefMatrix    = initPyMatrix(A);
    PyObject *objectiveLin  = (PyObject*)(Matrix_New(numVariables, 1, DOUBLE));
    PyObject *consIneq      = (PyObject*)(SpMatrix_New(numConsIneq, numVariables, numConsIneq, DOUBLE));
    PyObject *consIneqRhs   = (PyObject*)(Matrix_New(numConsIneq, 1, DOUBLE));
    PyObject *coefRhs       = (PyObject*)(Matrix_New(numConstraints, 1, DOUBLE));
    PyObject *args = PyTuple_New(doQP ? 6 : 5);
    if( !objectiveLin  || (doQP && !objectiveQuad) || !consIneq || !consIneqRhs ||
        !coefMatrix || !coefRhs || !args)
    {
        PyErr_Print();
        Py_DECREF(problem);
        Py_DECREF(solvers);
        Py_XDECREF(objectiveLin); 
        Py_XDECREF(objectiveQuad); 
        Py_XDECREF(consIneq); 
        Py_XDECREF(consIneqRhs); 
        Py_XDECREF(coefMatrix); 
        Py_XDECREF(coefRhs); 
        Py_XDECREF(args);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: error allocating matrices");
    }

    // init objective func, linear part
    for(size_t v=0; v<numVariables; v++)
        MAT_BUFD(objectiveLin)[v] = L.empty()? 0 : L[v];

    // init inequality constraints
    for(size_t v=0, c=0; v<numVariables; v++, c++) {
        SP_ROW(consIneq) [c] = v;
        SP_VALD(consIneq)[c] = -1.;
        SP_COL(consIneq) [v] = c;
        MAT_BUFD(consIneqRhs)[v]= xmin.empty()? 0 : xmin[v];
        if(!xmax.empty() && isFinite(xmax[v])) { // add upper constraint
            SP_ROW(consIneq) [c+1]= c-v+numVariables;
            SP_VALD(consIneq)[c+1]= 1.;
            MAT_BUFD(consIneqRhs)[c-v+numVariables] = xmax[v];
            c++;
        }
    }
    SP_COL(consIneq)[numVariables] = numConsIneq;

    // init rhs
    for(size_t c=0; c<numConstraints; c++)
        MAT_BUFD(coefRhs)[c] = rhs[c];

    // pack matrices into an argument tuple
    if(doQP)
        PyTuple_SetItem(args, 0, objectiveQuad);
    PyTuple_SetItem(args, doQP?1:0, objectiveLin);
    PyTuple_SetItem(args, doQP?2:1, consIneq);
    PyTuple_SetItem(args, doQP?3:2, consIneqRhs);
    PyTuple_SetItem(args, doQP?4:3, coefMatrix);
    PyTuple_SetItem(args, doQP?5:4, coefRhs);
    PyObject *solver = PyObject_CallObject(problem, args);
    if(!solver) {
        PyErr_Print();
        Py_DECREF(args);
        Py_DECREF(problem);
        Py_DECREF(solvers);
        //Py_Finalize();
        throw std::runtime_error("quadraticOptimizationSolve: solver returned error");
    }

    PyObject *status = PyDict_GetItemString(solver, "status");
    bool feasible = status != NULL && std::string(PyString_AsString(status)) == "optimal";
    PyObject *sol = PyDict_GetItemString(solver, "x");

    std::vector<double> result(numVariables);
    for(size_t v=0; feasible && v<numVariables; v++) {
        double vmin = xmin.empty() ? 0 : xmin[v];
        double vmax = xmax.empty() ? INFINITY : xmax[v];
        // correct possible roundoff errors that may lead to the value being outside the limits
        result[v] = clamp(MAT_BUFD(sol)[v], vmin, vmax);
    }

    Py_DECREF(solver);
    Py_DECREF(args);
    Py_DECREF(problem);
    Py_DECREF(solvers);
    //Py_Finalize();  // <-- uncommenting causes crash on repeated calls..
    if(feasible)
        return result;
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
    (void)A; (void)rhs; (void)L; (void)Q; (void)xmin; (void)xmax;  // silence warnings about unused params
}
#endif

namespace{
/** Matrix augmented with an extra block at the right end, containing pairs of (+1,-1) values

    +--------Nvar columns---------2*Naug--+     |v|
    |  exactly constrained part |         |     |a|   | |
    |---------------------------|         |     |r|   |R|
    |  approximately constr.pt. |1 -1     |  *  |s| = |H|
    |                           |     1 -1|     |_|   |S|
    +-------------------------------------+     |#|   | |
           original matrix       extra cols     |#|

    The extra 2*Naug columns added to the matrix correspond to "slack" variables:
    they are penalized in the objective function, and if the original matrix equation had
    a feasible solution, these variables (#) would be zero. But if the RHS (constraints)
    cannot be satisfied exactly, then either the odd or even-indexed slack variable
    corresponding to the given constrain will be >0, and will contribute to the cost function
    (linear or quadratic, or both).
*/
template<typename NumT>
class AugmentedMatrix: public IMatrix<NumT> {
    const IMatrix<NumT>& M;       ///< the original matrix
    const std::vector<size_t>& indAug;
public:
    AugmentedMatrix(const IMatrix<NumT>& src, const std::vector<size_t>& _indAug):
        IMatrix<NumT>(src.rows(), src.cols() + 2*_indAug.size()), M(src), indAug(_indAug) {};
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;
    virtual size_t size() const { return rows() * cols(); }
    virtual NumT at(size_t row, size_t col) const {
        if(col < M.cols())
            return M.at(row, col);
        col -= M.cols();
        if(row == indAug.at(col / 2))
            return col%2 ? +1 : -1;
        return 0;
    }
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {
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
}  // internal namespace

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
    std::vector<size_t> indAug;
    for(size_t c=0; c<numConstraints; c++) {
        if(isFinite(consPenaltyLin[c])) {
            if(consPenaltyLin[c]<0)
                throw std::invalid_argument(
                    "linearOptimizationSolveApprox: constraint penalties must be non-negative");
            indAug.push_back(c);
            Laug.push_back(consPenaltyLin[c]);
            Laug.push_back(consPenaltyLin[c]);
        }
    }
    std::vector<NumT> xminaug(xmin);
    if(!xmin.empty())
        xminaug.insert(xminaug.end(), 2*indAug.size(), 0);
    std::vector<NumT> xmaxaug(xmax);
    if(!xmax.empty())
        xmaxaug.insert(xmaxaug.end(), 2*indAug.size(), INFINITY);
    std::vector<double> result = linearOptimizationSolve(
        AugmentedMatrix<NumT>(A, indAug), rhs, Laug, xminaug, xmaxaug);
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
        (!L.empty() && L.size()!=numVariables) )
        throw std::invalid_argument("quadraticOptimizationSolveApprox: invalid size of input arrays");

    // check which constraints are 'loose', i.e. penalty is not infinite,
    // and add an extra pair of penalized variables for each loose constraint
    std::vector<NumT> Laug(numVariables), Qaug;
    std::vector<size_t> indAug;
    if(!L.empty())   // copy the array of penalty coefs for variables
        std::copy(L.begin(), L.end(), Laug.begin());
    for(size_t c=0; c<numConstraints; c++) {
        NumT penLin  = consPenaltyLin.empty()  ? 0 : consPenaltyLin [c];
        NumT penQuad = consPenaltyQuad.empty() ? 0 : consPenaltyQuad[c];
        if(penLin<0 || penQuad<0)
            throw std::invalid_argument(
                "quadraticOptimizationSolveApprox: constraint penalties must be provided");
        if(isFinite(penLin + penQuad)) {
            indAug.push_back(c);
            Laug.push_back(penLin);
            Laug.push_back(penLin);
            Qaug.push_back(penQuad);
            Qaug.push_back(penQuad);
        }
    }
    std::vector<NumT> xminaug(xmin);
    if(!xmin.empty())
        xminaug.insert(xminaug.end(), 2*indAug.size(), 0);
    std::vector<NumT> xmaxaug(xmax);
    if(!xmax.empty())
        xmaxaug.insert(xmaxaug.end(), 2*indAug.size(), INFINITY);
    std::vector<double> result = quadraticOptimizationSolve(
        AugmentedMatrix<NumT>(A, indAug), rhs, allZeros(Laug) ? std::vector<NumT>() : Laug,
        Q.size()==0 && allZeros(Qaug) ?   // in this case don't create any quadratic matrix
            static_cast<const IMatrix<NumT>&>(BandMatrix<NumT>()) :
            // if either the original quadratic matrix for numVariables was non-empty,
            // or the additional diagonal elements for numConstraints penalties were specified,
            // will create an augmented quadratic matrix, substituting the unspecified elements with zeros
            static_cast<const IMatrix<NumT>&>(AugmentedQuadMatrix<NumT>(Q.size()==0 ?
                static_cast<const IMatrix<NumT>&>(BandMatrix<NumT>(std::vector<NumT>(numVariables, 0))) :
                Q, Qaug)),
        xminaug, xmaxaug);
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
