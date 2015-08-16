#include "math_linalg.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <stdexcept>
#include <cassert>

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

// ------ wrappers for BLAS routines ------ //

void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y) {
    gsl_blas_daxpy(alpha, VecC(X), Vec(Y));
}

double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y) {
    double result;
    gsl_blas_ddot(VecC(X), VecC(Y), &result);
    return result;
}

double blas_dnrm2(const std::vector<double>& X) {
    return gsl_blas_dnrm2(VecC(X));
}

void blas_dgemv(CBLAS_TRANSPOSE TransA,
    double alpha, const Matrix<double>& A, const std::vector<double>& X, double beta, std::vector<double>& Y) {
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, MatC(A), VecC(X), beta, Vec(Y));
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X) {
    gsl_blas_dtrmv((CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, MatC(A), Vec(X));
}

void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C) {
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, 
        alpha, MatC(A), MatC(B), beta, Mat(C));
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B) {
    gsl_blas_dtrsm((CBLAS_SIDE_t)Side, (CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, 
        alpha, MatC(A), Mat(B));
}

// ------ linear algebra routines ------ //

void choleskyDecomp(Matrix<double>& A)
{
    gsl_linalg_cholesky_decomp(Mat(A));
}

void linearSystemSolveCholesky(const Matrix<double>& cholA, const std::vector<double>& y, 
    std::vector<double>& x)
{
    x.resize(y.size());
    gsl_linalg_cholesky_solve(MatC(cholA), VecC(y), Vec(x));
}

void singularValueDecomp(Matrix<double>& A, Matrix<double>& V, std::vector<double>& SV)
{
    V.resize(A.numCols(), A.numCols());
    SV.resize(A.numCols());
    std::vector<double> temp(A.numCols());
    gsl_linalg_SV_decomp(Mat(A), Mat(V), Vec(SV), Vec(temp));
}

void linearSystemSolveSVD(const Matrix<double>& U, const Matrix<double>& V, const std::vector<double>& SV,
    const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(U.numCols());
    gsl_linalg_SV_solve(MatC(U), MatC(V), VecC(SV), VecC(y), Vec(x));
}

void linearSystemSolveTridiag(const std::vector<double>& diag, const std::vector<double>& aboveDiag,
    const std::vector<double>& belowDiag, const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(diag.size());
    gsl_linalg_solve_tridiag(VecC(diag), VecC(aboveDiag), VecC(belowDiag), VecC(y), Vec(x));
}

void linearSystemSolveTridiagSymm(const std::vector<double>& diag, const std::vector<double>& offDiag,
    const std::vector<double>& y, std::vector<double>& x)
{
    x.resize(diag.size());
    gsl_linalg_solve_symm_tridiag(VecC(diag), VecC(offDiag), VecC(y), Vec(x));
}

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

}  // namespace
