#include "math_linalg.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

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
    if(A.numRows() >= A.numCols()*5) {   // use a modified algorithm for very 'elongated' matrices
        Matrix<double> tempmat(A.numCols(), A.numCols());
        gsl_linalg_SV_decomp_mod(Mat(A), Mat(tempmat), Mat(V), Vec(SV), Vec(temp));
    } else
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

}  // namespace
