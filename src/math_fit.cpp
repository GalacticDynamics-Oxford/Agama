#include "math_fit.h"
#include "math_core.h"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <stdexcept>
#include <string>

#ifdef HAVE_EIGEN
// necessary to change the global setting for storage order, because the solver interface
// does not allow for custom matrix types (i.e. with non-default order)
#define EIGEN_DEFAULT_TO_ROW_MAJOR
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#else
#include <gsl/gsl_multifit_nlin.h>
#include <gsl/gsl_multiroots.h>
#endif

namespace math{

namespace {
// ---- wrappers for GSL vector and matrix views (access the data arrays without copying) ----- //
struct Vec {
    explicit Vec(std::vector<double>& vec) :
        v(gsl_vector_view_array(&vec.front(), vec.size())) {}
    operator gsl_vector* () { return &v.vector; }
private:
    gsl_vector_view v;
};

struct VecC {
    explicit VecC(const std::vector<double>& vec) :
        v(gsl_vector_const_view_array(&vec.front(), vec.size())) {}
    operator const gsl_vector* () const { return &v.vector; }
private:
    gsl_vector_const_view v;
};

struct MatC {
    explicit MatC(const IMatrixDense<double>& mat) :
        m(gsl_matrix_const_view_array(mat.data(), mat.rows(), mat.cols())) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};

// ----- wrappers for multidimensional minimization routines ----- //
template <class T>
struct GslFncWrapper {
    const T& F;
    std::string error;
    int numCalls;
    explicit GslFncWrapper(const T& _F) : F(_F), numCalls(0) {}
};

double functionWrapperNdim(const gsl_vector* x, void* param) {
    double val;
    GslFncWrapper<IFunctionNdimDeriv>* p = static_cast<GslFncWrapper<IFunctionNdimDeriv>*>(param);
    p->numCalls++;
    try{
        p->F.eval(x->data, &val);
        return val;
    }
    catch(std::exception& e) {
        p->error = e.what();
        return NAN;
    }
}

void functionWrapperNdimDer(const gsl_vector* x, void* param, gsl_vector* df) {
    GslFncWrapper<IFunctionNdimDeriv>* p = static_cast<GslFncWrapper<IFunctionNdimDeriv>*>(param);
    p->numCalls++;
    try{
        p->F.evalDeriv(x->data, NULL, df->data);
    }
    catch(std::exception& e) {
        p->error = e.what();
    }
}

void functionWrapperNdimFncDer(const gsl_vector* x, void* param, double* f, gsl_vector* df) {
    GslFncWrapper<IFunctionNdimDeriv>* p = static_cast<GslFncWrapper<IFunctionNdimDeriv>*>(param);
    p->numCalls++;
    try{
        p->F.evalDeriv(x->data, f, df->data);
    }
    catch(std::exception& e) {
        p->error = e.what();
    }
}

// ----- wrappers for multidimensional nonlinear fitting ----- //
#ifndef HAVE_EIGEN
inline int functionWrapperNdimMvalFncDer(
    const gsl_vector* x, void* param, gsl_vector* f, gsl_matrix* df)
{
    GslFncWrapper<IFunctionNdimDeriv>* p = static_cast<GslFncWrapper<IFunctionNdimDeriv>*>(param);
    try{
        p->numCalls++;
        p->F.evalDeriv(x->data, f? f->data : NULL, df? df->data : NULL);
        // check that values and/or derivatives are ok
        bool ok=true;
        for(unsigned int i=0; f && i<p->F.numValues(); i++)
            ok &= isFinite(f->data[i]);
        for(unsigned int i=0; df && i<p->F.numVars()*p->F.numValues(); i++)
            ok &= isFinite(df->data[i]);
        if(!ok) {
            /*p->error = "Function is not finite";
            return GSL_FAILURE;*/
            for(unsigned int i=0; f && i<p->F.numValues(); i++)
                f->data[i] = 1e10;
        }
        return GSL_SUCCESS;
    }
    catch(std::exception& e){
        p->error = e.what();
        return GSL_FAILURE;
    }
}

int functionWrapperNdimMval(const gsl_vector* x, void* param, gsl_vector* f) {
    return functionWrapperNdimMvalFncDer(x, param, f, NULL);
}

int functionWrapperNdimMvalDer(const gsl_vector* x, void* param, gsl_matrix* df) {
    return functionWrapperNdimMvalFncDer(x, param, NULL, df);
}

#else
template <class T>
struct EigenFncWrapper {
    // definitions for automatic numerical differentiation framework
    typedef double Scalar;
    typedef Eigen::VectorXd InputType; //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> InputType;
    typedef Eigen::VectorXd ValueType; //Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ValueType;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacobianType;

    const T& F;
    mutable std::string error;
    mutable int numCalls;
    explicit EigenFncWrapper(const T& _F) : F(_F), numCalls(0) {}

    int operator()(const InputType &x, ValueType &f) const {
        try{
            numCalls++;
            F.eval(x.data(), f.data());
            for(unsigned int i=0; i<F.numValues(); i++)
                if(!isFinite(f[i])) {
                    for(unsigned int j=0; j<F.numValues(); j++)
                        f[j] = 1e10;
                    return 0;
                    /*error = "Function is not finite";
                    return -1;*/
                }
            return 0;
        }
        catch(std::exception& e){
            error = e.what();
            return -1;
        }
        return 0;
    }

    int df(const InputType &x, JacobianType &df) const {
        try{
            numCalls++;
            F.evalDeriv(x.data(), NULL, df.data());
            /*for(unsigned int i=0; i<F.numVars()*F.numValues(); i++)
                if(!isFinite(df.data()[i])) {
                    error = "Derivative is not finite";
                    return -1;
                }*/
            return 0;
        }
        catch(std::exception& e){
            error = e.what();
            return -1;
        }
        return 0;
    }

    int inputs() const { return F.numVars(); }
    int values() const { return F.numValues(); }
    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };
};
#endif

}  // internal namespace

// ----- linear least-square fit ------- //
double linearFitZero(const std::vector<double>& x, const std::vector<double>& y,
    const std::vector<double>* w, double* rms)
{
    if(x.size() != y.size() || (w!=NULL && w->size() != y.size()))
        throw std::invalid_argument("LinearFit: input arrays are not of equal length");
    double c, cov, sumsq;
    if(w==NULL)
        gsl_fit_mul(&x.front(), 1, &y.front(), 1, y.size(), &c, &cov, &sumsq);
    else
        gsl_fit_wmul(&x.front(), 1, &w->front(), 1, &y.front(), 1, y.size(), &c, &cov, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/y.size());
    return c;
}

void linearFit(const std::vector<double>& x, const std::vector<double>& y, 
    const std::vector<double>* w, double& slope, double& intercept, double* rms)
{
    if(x.size() != y.size() || (w!=NULL && w->size() != y.size()))
        throw std::invalid_argument("LinearFit: input arrays are not of equal length");
    double cov00, cov11, cov01, sumsq;
    if(w==NULL)
        gsl_fit_linear(&x.front(), 1, &y.front(), 1, y.size(),
            &intercept, &slope, &cov00, &cov01, &cov11, &sumsq);
    else
        gsl_fit_wlinear(&x.front(), 1, &w->front(), 1, &y.front(), 1, y.size(),
            &intercept, &slope, &cov00, &cov01, &cov11, &sumsq);
    if(rms!=NULL)
        *rms = sqrt(sumsq/y.size());
}

// ----- multi-parameter linear least-square fit ----- //
void linearMultiFit(const IMatrixDense<double>& coefs, const std::vector<double>& rhs, 
    const std::vector<double>* w, std::vector<double>& result, double* rms)
{
    if(static_cast<unsigned int>(coefs.rows()) != rhs.size())
        throw std::invalid_argument(
            "LinearMultiFit: number of rows in matrix is different from the length of RHS vector");
    result.assign(coefs.cols(), 0);
    gsl_matrix* covarMatrix =
        gsl_matrix_alloc(coefs.cols(), coefs.cols());
    gsl_multifit_linear_workspace* fitWorkspace =
        gsl_multifit_linear_alloc(coefs.rows(),coefs.cols());
    if(covarMatrix==NULL || fitWorkspace==NULL) {
        if(fitWorkspace)
            gsl_multifit_linear_free(fitWorkspace);
        if(covarMatrix)
            gsl_matrix_free(covarMatrix);
        throw std::bad_alloc();
    }
    double sumsq;
    if(w==NULL)
        gsl_multifit_linear(MatC(coefs), VecC(rhs), Vec(result), covarMatrix, &sumsq, fitWorkspace);
    else
        gsl_multifit_wlinear(MatC(coefs), VecC(*w), VecC(rhs), Vec(result),
            covarMatrix, &sumsq, fitWorkspace);
    gsl_multifit_linear_free(fitWorkspace);
    gsl_matrix_free(covarMatrix);
    if(rms!=NULL)
        *rms = sqrt(sumsq/rhs.size());
}

// ----- nonlinear least-square fit ----- //
int nonlinearMultiFit(const IFunctionNdimDeriv& F, const double xinit[],
    const double relToler, const int maxNumIter, double result[])
{
    const unsigned int Nparam = F.numVars();   // number of parameters to vary
    const unsigned int Ndata  = F.numValues(); // number of data points to fit
    if(Ndata < Nparam)
        throw std::invalid_argument(
            "nonlinearMultiFit: number of data points is less than the number of parameters to fit");
    for(unsigned int i=0; i<Nparam; i++)
        result[i] = xinit[i];
#ifdef HAVE_EIGEN
    EigenFncWrapper<IFunctionNdimDeriv> params(F);
    Eigen::VectorXd data = Eigen::Map<const Eigen::VectorXd>(xinit, Nparam);
    //Eigen::NumericalDiff< EigenFncWrapper<IFunctionNdimDeriv> > fw(params);
    //Eigen::LevenbergMarquardt< Eigen::NumericalDiff< EigenFncWrapper<IFunctionNdimDeriv> > > solver(fw);
    Eigen::LevenbergMarquardt< EigenFncWrapper<IFunctionNdimDeriv> > solver(params);
    if(solver.minimizeInit(data) == Eigen::LevenbergMarquardtSpace::ImproperInputParameters)
        params.error = "invalid input parameters";
#else
    GslFncWrapper<IFunctionNdimDeriv> params(F);
    gsl_multifit_function_fdf fnc;
    fnc.params = &params;
    fnc.p = Nparam;
    fnc.n = Ndata;
    fnc.f = functionWrapperNdimMval;
    fnc.df = functionWrapperNdimMvalDer;
    fnc.fdf = functionWrapperNdimMvalFncDer;
    gsl_multifit_fdfsolver* solver = gsl_multifit_fdfsolver_alloc(
        gsl_multifit_fdfsolver_lmsder, Ndata, Nparam);
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Nparam);
    if(gsl_multifit_fdfsolver_set(solver, &fnc, &v_xinit.vector) != GSL_SUCCESS)
        params.error = "invalid input parameters";
    const double* data = solver->x->data;
#endif
    bool carryon = true, converged = false;
    while(params.error.empty() && carryon && !converged) {
#ifdef HAVE_EIGEN
        carryon = solver.minimizeOneStep(data) == Eigen::LevenbergMarquardtSpace::Running;
#else
        carryon = gsl_multifit_fdfsolver_iterate(solver) == GSL_SUCCESS;
#endif
        carryon &= params.numCalls < maxNumIter;
        // store the current result and test for convergence
        converged = true;
        for(unsigned int i=0; i<Nparam; i++) {
            converged &= fabs(data[i] - result[i]) <= relToler * fabs(data[i]);
            result[i] = data[i];
        }
    }
    if(!converged)
        params.numCalls *= -1;  // signal of non-convergence
#ifndef HAVE_EIGEN
    gsl_multifit_fdfsolver_free(solver);
#endif
    if(!params.error.empty())
        throw std::runtime_error("Error in nonlinearMultiFit: "+params.error);
    return params.numCalls;
}

// ----- multidimensional root-finding ----- //

int findRootNdimDeriv(const IFunctionNdimDeriv& F, const double xinit[],
    const double absToler, const int maxNumIter, double result[])
{
    const unsigned int Ndim = F.numVars();
    if(F.numValues() != F.numVars())
        throw std::invalid_argument(
            "findRootNdimDeriv: number of equations must be equal to the number of variables");
#ifdef HAVE_EIGEN
    EigenFncWrapper<IFunctionNdimDeriv> fnc(F);
    Eigen::VectorXd vars = Eigen::Map<const Eigen::VectorXd>(xinit, Ndim);
    Eigen::HybridNonLinearSolver< EigenFncWrapper<IFunctionNdimDeriv> , double > solver(fnc);
    if(solver.solveInit(vars) == Eigen::HybridNonLinearSolverSpace::ImproperInputParameters)
        fnc.error = "invalid input parameters";
    solver.parameters.maxfev = maxNumIter;
    solver.useExternalScaling= true;
    solver.diag.setConstant(Ndim, 1.);
    const double* values = solver.fvec.data();
#else
    GslFncWrapper<IFunctionNdimDeriv> fnc(F);
    gsl_multiroot_function_fdf gfnc;
    gfnc.params = &fnc;
    gfnc.n = Ndim;
    gfnc.f = functionWrapperNdimMval;
    gfnc.df = functionWrapperNdimMvalDer;
    gfnc.fdf = functionWrapperNdimMvalFncDer;
    gsl_multiroot_fdfsolver* solver = gsl_multiroot_fdfsolver_alloc(
        gsl_multiroot_fdfsolver_hybridsj, Ndim);
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Ndim);
    if(gsl_multiroot_fdfsolver_set(solver, &gfnc, &v_xinit.vector) != GSL_SUCCESS)
        fnc.error = "invalid input parameters";
    const double* vars = solver->x->data;
    const double* values = solver->f->data;
#endif
    bool carryon = true, converged = false;
    while(fnc.error.empty() && carryon && !converged) {   // iterate
#ifdef HAVE_EIGEN
        Eigen::HybridNonLinearSolverSpace::Status sta = solver.solveOneStep(vars);
        carryon  = sta == Eigen::HybridNonLinearSolverSpace::Running ||
                   sta == Eigen::HybridNonLinearSolverSpace::NotMakingProgressJacobian;  // try harder!
#else
        int sta  = gsl_multiroot_fdfsolver_iterate(solver);
        carryon  = sta == GSL_SUCCESS || sta == GSL_ENOPROGJ;
#endif
        carryon &= fnc.numCalls < maxNumIter;
        // test for convergence
        converged = true;
        for(unsigned int i=0; i<Ndim; i++)
            converged &= fabs(values[i]) <= absToler;
    }
    if(!converged)
        fnc.numCalls *= -1;  // signal of error
    // store the found location of minimum
    for(unsigned int i=0; i<Ndim; i++)
        result[i] = vars[i];
#ifndef HAVE_EIGEN
    gsl_multiroot_fdfsolver_free(solver);
#endif
    if(!fnc.error.empty())
        throw std::runtime_error("Error in findRootNdimDeriv: "+fnc.error);
    return fnc.numCalls;
}

// ----- multidimensional minimization ----- //

int findMinNdim(const IFunctionNdim& F, const double xinit[], const double xstep[],
    const double absToler, const int maxNumIter, double result[])
{
    if(F.numValues() != 1)
        throw std::invalid_argument("findMinNdim: function must provide a single output value");
    const unsigned int Ndim = F.numVars();
    // instance of minimizer algorithm
    gsl_multimin_fminimizer* mizer = gsl_multimin_fminimizer_alloc(
        gsl_multimin_fminimizer_nmsimplex2, Ndim);
    gsl_multimin_function fnc;
    GslFncWrapper<IFunctionNdim> params(F);
    fnc.params = &params;
    fnc.n = Ndim;
    fnc.f = functionWrapperNdim;
    int numIter = 0;
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Ndim);
    gsl_vector_const_view v_xstep = gsl_vector_const_view_array(xstep, Ndim);
    if(gsl_multimin_fminimizer_set(mizer, &fnc, &v_xinit.vector, &v_xstep.vector ) == GSL_SUCCESS)
    {   // iterate
        double sizePrev = gsl_multimin_fminimizer_size(mizer);
        int numIterStall = 0;
        do {
            if(gsl_multimin_fminimizer_iterate(mizer) != GSL_SUCCESS)
                break;
            double sizeCurr = gsl_multimin_fminimizer_size(mizer);
            if(sizeCurr <= absToler)
                break;
            // check if the simplex is stuck
            if(fabs(sizeCurr-sizePrev)/sizePrev <= 1e-4)
                numIterStall++;
            else
                numIterStall = 0;  // reset counter
            if(numIterStall >= 10*(int)Ndim)  // no progress
                break;  // may need to restart it instead?
            sizePrev = sizeCurr;
            numIter++;
        } while(numIter<maxNumIter);
    }
    // store the found location of minimum
    for(unsigned int i=0; i<Ndim; i++)
        result[i] = mizer->x->data[i];
    gsl_multimin_fminimizer_free(mizer);
    if(!params.error.empty())
        throw std::runtime_error("Error in findMinNdim: "+params.error);
    return numIter;
}

int findMinNdimDeriv(const IFunctionNdimDeriv& F, const double xinit[], const double xstep,
    const double absToler, const int maxNumIter, double result[])
{
    if(F.numValues() != 1)
        throw std::invalid_argument("findMinNdimDeriv: function must provide a single output value");
    const unsigned int Ndim = F.numVars();
    // instance of minimizer algorithm
    gsl_multimin_fdfminimizer* mizer = gsl_multimin_fdfminimizer_alloc(
        gsl_multimin_fdfminimizer_vector_bfgs2, Ndim);
    gsl_multimin_function_fdf fnc;
    GslFncWrapper<IFunctionNdimDeriv> params(F);
    fnc.params = &params;
    fnc.n = Ndim;
    fnc.f = functionWrapperNdim;
    fnc.df = functionWrapperNdimDer;
    fnc.fdf = functionWrapperNdimFncDer;
    int numIter = 0;
    gsl_vector_const_view v_xinit = gsl_vector_const_view_array(xinit, Ndim);
    if(gsl_multimin_fdfminimizer_set(mizer, &fnc, &v_xinit.vector, xstep, 0.1) == GSL_SUCCESS)
    {   // iterate
        do {
            numIter++;
            if(gsl_multimin_fdfminimizer_iterate(mizer) != GSL_SUCCESS)
                break;
        } while(numIter<maxNumIter && 
            gsl_multimin_test_gradient(mizer->gradient, absToler) == GSL_CONTINUE);
    }
    // store the found location of minimum
    for(unsigned int i=0; i<Ndim; i++)
        result[i] = mizer->x->data[i];
    gsl_multimin_fdfminimizer_free(mizer);
    if(!params.error.empty())
        throw std::runtime_error("Error in findMinNdimDeriv: "+params.error);
    return numIter;
}

}  // namespace
