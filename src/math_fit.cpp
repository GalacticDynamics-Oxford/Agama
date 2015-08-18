#include "math_fit.h"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multimin.h>
#include <stdexcept>

namespace math{

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

struct Mat {
    explicit Mat(Matrix<double>& mat) :
        m(gsl_matrix_view_array(mat.getData(), mat.numRows(), mat.numCols())) {}
    operator gsl_matrix* () { return &m.matrix; }
private:
    gsl_matrix_view m;
};

struct MatC {
    explicit MatC(const Matrix<double>& mat) :
        m(gsl_matrix_const_view_array(mat.getData(), mat.numRows(), mat.numCols())) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};

// ----- linear regression ------- //

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

void linearMultiFit(const Matrix<double>& coefs, const std::vector<double>& rhs, 
    const std::vector<double>* w, std::vector<double>& result, double* rms)
{
    if(coefs.numRows() != rhs.size())
        throw std::invalid_argument(
            "LinearMultiFit: number of rows in matrix is different from the length of RHS vector");
    result.assign(coefs.numCols(), 0);
    gsl_matrix* covarMatrix =
        gsl_matrix_alloc(coefs.numCols(), coefs.numCols());
    gsl_multifit_linear_workspace* fitWorkspace =
        gsl_multifit_linear_alloc(coefs.numRows(),coefs.numCols());
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
        gsl_multifit_wlinear(MatC(coefs), VecC(*w), VecC(rhs), Vec(result), covarMatrix, &sumsq, fitWorkspace);
    gsl_multifit_linear_free(fitWorkspace);
    gsl_matrix_free(covarMatrix);
    if(rms!=NULL)
        *rms = sqrt(sumsq/rhs.size());
}

// ----- multidimensional minimization ------ //

static double functionWrapperNdim(const gsl_vector* x, void* param)
{
    double val;
    static_cast<IFunctionNdim*>(param)->eval(x->data, &val);
    return val;
}

int findMinNdim(const IFunctionNdim& F, const double xinit[], const double xstep[],
    const double absToler, const int maxNumIter, double result[])
{
    if(F.numValues() != 1)
        throw std::invalid_argument("N-dimensional minimization: function must provide a single output value");
    const unsigned int Ndim = F.numVars();
    // instance of minimizer algorithm
    gsl_multimin_fminimizer* mizer = gsl_multimin_fminimizer_alloc(
        gsl_multimin_fminimizer_nmsimplex2, Ndim);
    gsl_multimin_function fnc;
    fnc.n = Ndim;
    fnc.f = functionWrapperNdim;
    fnc.params = const_cast<IFunctionNdim*>(&F);
    int numIter = 0;
    if(gsl_multimin_fminimizer_set(mizer, &fnc,
        &gsl_vector_const_view_array(xinit, Ndim).vector,
        &gsl_vector_const_view_array(xstep, Ndim).vector ) == GSL_SUCCESS)
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
    return numIter;
}

}  // namespace
