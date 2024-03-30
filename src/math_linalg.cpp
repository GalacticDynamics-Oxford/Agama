#include "math_linalg.h"
#include "math_core.h"  // for binSearch
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef HAVE_EIGEN

// calm down excessively optimizing Intel compiler, which otherwise screws the SVD module
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1800)
#pragma float_control (strict,on)
#endif

// don't use internal OpenMP parallelization at the level of internal Eigen routines (why?)
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <Eigen/Cholesky>
#include <Eigen/SVD>

#else
// no Eigen

#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_version.h>
#if GSL_MAJOR_VERSION >= 2
#define HAVE_GSL_SPARSE
#include <gsl/gsl_spblas.h>
#else
#warning "Sparse matrix support is not available, replaced by dense matrices"
#endif

#endif

// uncomment the following line to activate range checks on matrix element access (slows down the code)
//#define DEBUG_RANGE_CHECK

namespace math{

// ------ utility routines with common implementations for both EIGEN and GSL ------ //
void matrixRangeCheck(bool condition)
{
    if(!condition)
        throw std::out_of_range("matrix index out of range");
}

void eliminateNearZeros(std::vector<double>& vec, double threshold)
{
    double mag=0;
    for(size_t t=0; t<vec.size(); t++)
        mag = fmax(mag, fabs(vec[t]));
    mag *= threshold;
    for(size_t t=0; t<vec.size(); t++)
        if(fabs(vec[t]) <= mag)
            vec[t]=0;
}

void eliminateNearZeros(Matrix<double>& mat, double threshold)
{
    double mag=0;
    for(size_t i=0; i<mat.rows(); i++)
        for(size_t j=0; j<mat.cols(); j++)
            mag = fmax(mag, fabs(mat(i,j)));
    mag *= threshold;
    for(size_t i=0; i<mat.rows(); i++)
        for(size_t j=0; j<mat.cols(); j++)
            if(fabs(mat(i,j)) <= mag)
                mat(i,j)=0;
}

template<> void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y)
{
    const size_t size = X.size();
    if(size!=Y.size())
        throw std::length_error("blas_daxpy: invalid size of input arrays");
    if(alpha==0) return;
    for(size_t i=0; i<size; i++)
        Y[i] += alpha*X[i];
}

template<> void blas_dmul(double alpha, std::vector<float>& Y)
{
    const size_t size = Y.size();
    for(size_t i=0; i<size; i++)
        Y[i] *= alpha;
}

template<> void blas_dmul(double alpha, std::vector<double>& Y)
{
    const size_t size = Y.size();
    for(size_t i=0; i<size; i++)
        Y[i] *= alpha;
}

double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y)
{
    const size_t size = X.size();
    if(size!=Y.size())
        throw std::length_error("blas_ddot: invalid size of input arrays");
    double result = 0;
    for(size_t i=0; i<size; i++)
        result += X[i]*Y[i];
    return result;
}

template<>
double blas_dnrm2(const std::vector<double>& X)
{
    double result = 0;
    for(size_t i=0; i<X.size(); i++)
        result += pow_2(X[i]);
    return result;
}

template<>
double blas_dnrm2(const Matrix<double>& X)
{
    double result = 0;
    for(size_t i=0; i<X.rows(); i++)
        for(size_t j=0; j<X.cols(); j++)
            result += pow_2(X(i,j));
    return result;
}

// --------- BAND MATRIX --------- //
/*  we use the following storage scheme for band matrices: a flattened array of size N * (2*B+1),
    where N is the number of rows and B is the bandwidth (B < N).
    An example of a matrix with N=8 and B=2, and its compact representation:
    |10  5  2  0  0  0  0  0 |     | -  - 10  5  2 |
    | 4 10  5  2  0  0  0  0 |     | -  4 10  5  2 |
    | 1  4 10  5  2  0  0  0 |     | 1  4 10  5  2 |
    | 0  1  4 10  5  2  0  0 |     | 1  4 10  5  2 |
    | 0  0  1  4 10  5  2  0 |     | 1  4 10  5  2 |
    | 0  0  0  1  4 10  5  2 |     | 1  4 10  5  2 |
    | 0  0  0  0  1  4 10  5 |     | 1  4 10  5  - |
    | 0  0  0  0  0  1  4 10 |     | 1  4 10  -  - |
    Here entries marked '-' are unused (refer to elements that would lie outside the original matrix).
*/
template<typename NumT>
BandMatrix<NumT>::BandMatrix(size_t size, size_t bandWidth, const NumT value) :
    IMatrix<NumT>(size, size), band(bandWidth), data(size * (2*bandWidth+1), value)
{
    if(bandWidth >= size)
        throw std::length_error("BandMatrix: bandwidth must be less than the matrix size");
}

template<typename NumT>
NumT BandMatrix<NumT>::at(size_t row, size_t col) const
{
    matrixRangeCheck(row < rows() && col < cols());
    if(col > row+band || row > col+band)
        return 0;  // no error if the element is out of bandwidth, just return zero
    return data[row * (2*band+1) + col+band-row];
}

template<typename NumT>
const NumT& BandMatrix<NumT>::operator() (size_t row, size_t col) const
{
    // if the element is outside bandwidth, cannot return zero because this is a reference, not a value
    matrixRangeCheck(row < rows() && col < cols() && col <= row+band && row <= col+band);
    return data[row * (2*band+1) + col+band-row];
}

template<typename NumT>
NumT& BandMatrix<NumT>::operator() (size_t row, size_t col)
{
    matrixRangeCheck(row < rows() && col < cols() && col <= row+band && row <= col+band);
    return data[row * (2*band+1) + col+band-row];
}

template<typename NumT>
size_t BandMatrix<NumT>::size() const
{
    return rows() * (2*band+1) - band*(band+1);
}

template<typename NumT>
NumT BandMatrix<NumT>::elem(size_t index, size_t &row, size_t &col) const
{
    const size_t
        nRows = rows(),
        width = 2*band+1,                           // width of one normal row
        total = nRows * width - band * (band+1),    // total number of nonzero elements
        nclip = pow_2(band) + band * (band+1) / 2;  // number of elements in clipped areas
    if(index < nclip) {
        // first few rows have clipped left edge
        row = 0;
        col = index;
        while(col >= row+band+1) {
            col -= row+band+1;
            row++;
        }
    } else if(index+nclip < total) {
        // the bulk of the matrix
        row = (index-nclip) / width + band;
        col = (index-nclip) % width + row-band;
    } else if(index < total) {
        // last few rows have clipped right edge
        row = 0;
        col = total-1-index;
        while(col >= row+band+1) {
            col -= row+band+1;
            row++;
        }
        row = nRows-1-row;
        col = nRows-1-col;
    } else
        matrixRangeCheck(false /*index out of range*/);
    return data[row * width + col+band-row];
}

template<typename NumT>
std::vector<Triplet> BandMatrix<NumT>::values() const
{
    const ptrdiff_t width = band * 2 + 1, nRows = rows();
    std::vector<Triplet> result;
    result.reserve(nRows * width);
    for(ptrdiff_t r=0; r<nRows; r++) {
        ptrdiff_t cl = std::max<ptrdiff_t>(0, r-band);
        ptrdiff_t cr = std::min<ptrdiff_t>(nRows-1, r+band);
        for(ptrdiff_t c = cl, m = r*width + cl-r+band; c <= cr; c++, m++)
            result.push_back(Triplet(r, c, data[m]));
    }
    return result;
}

template<>
void blas_daxpy(double alpha, const BandMatrix<double>& X, BandMatrix<double>& Y)
{
    const size_t size = X.rows(), band = X.bandwidth(), numElem = size * (band*2+1);
    if(Y.rows() != size || Y.bandwidth() != band)
        throw std::length_error("blas_daxpy: invalid size of input arrays");
    if(alpha==0) return;
    const double* x = &(X(0,0))-band;  // address of the first element in the flattened band matrix
    double* y = &(Y(0,0))-band;
    for(size_t i=0; i<numElem; i++)
        y[i] += alpha * x[i];
}

template<> void blas_dmul(double alpha, BandMatrix<double>& Y)
{
    const size_t size = Y.rows(), band = Y.bandwidth(), numElem = size * (band*2+1);
    double* y = &(Y(0,0))-band;
    for(size_t i=0; i<numElem; i++)
        y[i] *= alpha;
}

template<>
void blas_dgemv(CBLAS_TRANSPOSE Trans, double alpha, const BandMatrix<double>& mat,
    const std::vector<double>& vec, double beta, std::vector<double>& result)
{
    const ptrdiff_t size = mat.rows(), band = mat.bandwidth(), width = band * 2 + 1;
    if((ptrdiff_t)vec.size() != size || (ptrdiff_t)result.size() != size)
        throw std::length_error("blas_dgemv: invalid size of input arrays");
    if(beta==0)
        result.assign(size, 0.);
    else
        for(ptrdiff_t i=0; i<size; i++)
            result[i] *= beta;
    const double* data = &(mat(0,0))-band;  // address of the first element in the flattened band matrix
    if(Trans == CblasTrans) {
        for(ptrdiff_t r=0; r<size; r++) {
            double   val = alpha * vec[r];
            ptrdiff_t cl = std::max<ptrdiff_t>(0, r-band);
            ptrdiff_t cr = std::min<ptrdiff_t>(size-1, r+band);
            for(ptrdiff_t c = cl, m = r*width + cl-r+band; c <= cr; c++, m++)
                result[c] += val * data[m];
        }
    } else {
        for(ptrdiff_t r=0; r<size; r++) {
            ptrdiff_t cl = std::max<ptrdiff_t>(0, r-band);
            ptrdiff_t cr = std::min<ptrdiff_t>(size-1, r+band);
            for(ptrdiff_t c = cl, m = r*width + cl-r+band; c <= cr; c++, m++)
                result[r] += alpha * vec[c] * data[m];
        }
    }
}

std::vector<double> solveBand(const BandMatrix<double>& mat, const std::vector<double>& rhs)
{
    const ptrdiff_t size = mat.rows(), band = mat.bandwidth(), width = band * 2 + 1;
    if((ptrdiff_t)rhs.size() != size)
        throw std::length_error("solveBand: invalid size of input arrays");
    std::vector<double> x(rhs);             // result vector, which originally contains the r.h.s.
    std::vector<double> U(size*band);       // upper triangular part of the input matrix
    std::vector<double> L(width);           // temporary storage (one row of the input matrix)
    const double* data = &(mat(0,0))-band;  // hack into the internal representation of band matrix
    if(band==1) {  // fast track for a tridiagonal matrix
        for(ptrdiff_t i=0; i<size; i++) {   // forward pass
            double ipiv = 1. / (data[i*3+1] - (i>0 ? data[i*3] * U[i-1] : 0));
            U[i] = data[i*3+2] * ipiv;
            x[i] = (rhs[i] - (i>0 ? data[i*3] * x[i-1] : 0)) * ipiv;
        }
        // back-substitution
        for(ptrdiff_t i=size-2; i>=0; i--)
            x[i] -= U[i] * x[i+1];
        return x;
    }
    for(ptrdiff_t i=0; i<size; i++) {
        // copy the [non-zero elements of] i-th row of input matrix into the temporary array L
        std::copy(data + i*width, data + (i+1)*width, L.begin());
        // Gaussian elimination (only the current, i-th row, is kept at a time)
        for(ptrdiff_t c = 1; c < 2*band; c++) {
            ptrdiff_t rmin = std::max<ptrdiff_t>(0, std::max(band-i, c-band));
            ptrdiff_t rmax = std::min<ptrdiff_t>(band, c);
            for(ptrdiff_t r=rmin; r<rmax; r++)
                L[c] -= L[r] * U[(i-band+r) * band + c-r-1];
        }
        double ipiv = 1. / L[band];  // inverse of the diagonal element
        // store the elements of U matrix for the current row
        for(ptrdiff_t c = band+1; c <= 2*band; c++)
            U[i * band + c-band-1] = L[c] * ipiv;
        // forward pass for the solution vector (multiply x by L^-1)
        for(ptrdiff_t c = std::max<ptrdiff_t>(0, band-i); c < band; c++)
            x[i] -= L[c] * x[i-band+c];
        x[i] *= ipiv;
    }
    // back-substitution (multiply x by U^-1)
    for(ptrdiff_t i = size-2; i >= 0; i--) {
        for(ptrdiff_t j = std::min<ptrdiff_t>(band-1, size-2-i); j >= 0; j--)
            x[i] -= x[i+j+1] * U[i * band + j];
    }
    return x;
}


#ifdef HAVE_EIGEN
// --------- EIGEN-BASED IMPLEMENTATIONS --------- //

namespace{
// workaround for the lack of template typedef
template<typename MatrixType> struct Type;

template<typename NumT>
struct Type< Matrix<NumT> > {
    typedef Eigen::Matrix<NumT, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> T;
};
template<typename NumT>
struct Type< SparseMatrix<NumT> > {
    typedef Eigen::SparseMatrix<NumT, Eigen::ColMajor> T;
};

// shotrcut for type casting
template<typename M>
inline typename Type<M>::T& mat(M& m) {
    return *static_cast<typename Type<M>::T*>(m.impl);
}
template<typename M>
inline const typename Type<M>::T& mat(const M& m) {
    return *static_cast<const typename Type<M>::T*>(m.impl);
}

}

//------ dense matrix -------//

// default constructor
template<typename NumT>
Matrix<NumT>::Matrix() :
    IMatrixDense<NumT>(),
    impl(new typename Type<Matrix<NumT> >::T()) {}

// constructor without data initialization
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols) :
    IMatrixDense<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols)) {}

// constructor with an initial value
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols, const NumT value) :
    IMatrixDense<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols))
{
    mat(*this).fill(value);
}

// copy constructor from a dense matrix
template<typename NumT>
Matrix<NumT>::Matrix(const Matrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(mat(src))) {}

// copy constructor from a sparse matrix
template<typename NumT>
Matrix<NumT>::Matrix(const SparseMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(mat(src))) {}

// copy constructor from a band matrix
template<typename NumT>
Matrix<NumT>::Matrix(const BandMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(src.rows(), src.cols()))
{
    mat(*this).fill(0);
    const ptrdiff_t band = src.bandwidth(), width = band * 2 + 1, nRows = rows();
    const NumT* data = &(src(0,0))-band;
    for(ptrdiff_t r=0; r<nRows; r++) {
        ptrdiff_t cl = std::max<ptrdiff_t>(0, r-band);
        ptrdiff_t cr = std::min<ptrdiff_t>(nRows-1, r+band);
        for(ptrdiff_t c = cl, m = r*width + cl-r+band; c <= cr; c++, m++)
            mat(*this)(r, c) = data[m];
    }
}

// copy constructor from anything matrix-like (same as from sparse matrix)
template<typename NumT>
Matrix<NumT>::Matrix(const IMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(src.rows(), src.cols()))
{
    mat(*this).fill(0);
    size_t numelem = src.size();
    for(size_t k=0; k<numelem; k++) {
        size_t i, j;
        NumT v = src.elem(k, i, j);
        mat(*this)(i, j) = v;
    }
}

// constructor from triplets
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values) :
    IMatrixDense<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols))
{
    mat(*this).fill(0);
    for(size_t k=0; k<values.size(); k++)
        mat(*this)(values[k].i, values[k].j) = static_cast<NumT>(values[k].v);
}

template<typename NumT>
Matrix<NumT>::~Matrix()
{
    delete &(mat(*this));
}

/// access a matrix element
template<typename NumT>
const NumT& Matrix<NumT>::operator() (size_t row, size_t col) const
{
#ifdef DEBUG_RANGE_CHECK
    matrixRangeCheck(row < rows() && col < cols());
#endif
    return mat(*this).data()[row * cols() + col];
}

template<typename NumT>
NumT& Matrix<NumT>::operator() (size_t row, size_t col)
{
#ifdef DEBUG_RANGE_CHECK
    matrixRangeCheck(row < rows() && col < cols());
#endif
    return mat(*this).data()[row * cols() + col];
}

/// access the raw data storage
template<typename NumT>
NumT* Matrix<NumT>::data() {
    return mat(*this).data();
}

template<typename NumT>
const NumT* Matrix<NumT>::data() const {
    return mat(*this).data();
}

//------ sparse matrix -------//

template<typename NumT>
SparseMatrix<NumT>::SparseMatrix() :
    IMatrix<NumT>(), impl(new typename Type<SparseMatrix<NumT> >::T()) {}

// constructor from triplets
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols),
    impl(new typename Type<SparseMatrix<NumT> >::T(nRows, nCols))
{
    mat(*this).setFromTriplets(values.begin(), values.end());
}

// copy constructor from band matrix
template<typename NumT>
    SparseMatrix<NumT>::SparseMatrix(const BandMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(new typename Type<SparseMatrix<NumT> >::T(src.rows(), src.cols()))
{
    std::vector<Triplet> values = src.values();
    mat(*this).setFromTriplets(values.begin(), values.end());
}

// copy constructor from sparse matrix
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(const SparseMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(new typename Type<SparseMatrix<NumT> >::T(mat(src))) {}

template<typename NumT>
SparseMatrix<NumT>::~SparseMatrix()
{
    delete &(mat(*this));
}

// element access
template<typename NumT>
NumT SparseMatrix<NumT>::at(size_t row, size_t col) const
{
    return mat(*this).coeff(row, col);
}

template<typename NumT>
NumT SparseMatrix<NumT>::elem(const size_t index, size_t &row, size_t &col) const
{
    matrixRangeCheck(index < static_cast<size_t>(mat(*this).nonZeros()));
    row = mat(*this).innerIndexPtr()[index];
    col = binSearch(static_cast<int>(index), mat(*this).outerIndexPtr(), cols()+1);
    return mat(*this).valuePtr()[index];
}

template<typename NumT>
size_t SparseMatrix<NumT>::size() const
{
    return mat(*this).nonZeros();
}

template<typename NumT>
std::vector<Triplet> SparseMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    result.reserve(mat(*this).nonZeros());
    for(size_t j=0; j<cols(); ++j)
        for(typename Type<SparseMatrix<NumT> >::T::InnerIterator i(mat(*this), j); i; ++i)
            result.push_back(Triplet(i.row(), i.col(), i.value()));
    return result;
}

// ------ wrappers for BLAS routines ------ //

/// convert the result of Eigen operation into a std::vector
template<typename T>
inline std::vector<double> toStdVector(const T& src) {
    Eigen::VectorXd vec(src);
    return std::vector<double>(vec.data(), vec.data()+vec.size());
}

/// wrap std::vector into an Eigen-compatible interface
inline Eigen::Map<const Eigen::VectorXd> toEigenVector(const std::vector<double>& v) {
    return Eigen::Map<const Eigen::VectorXd>(&v.front(), v.size());
}

template<typename MatrixType>
void blas_daxpy(double alpha, const MatrixType& X, MatrixType& Y)
{
    if(alpha==1)
        mat(Y) += mat(X);
    else if(alpha!=0)
        mat(Y) += alpha * mat(X);
}

template<typename MatrixType>
void blas_dmul(double alpha, MatrixType& Y)
{
    mat(Y) *= alpha;
}

template<typename MatrixType>
void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const MatrixType& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    Eigen::VectorXd v;
    if(TransA==CblasNoTrans)
        v = mat(A) * toEigenVector(X);
    else
        v = mat(A).transpose() * toEigenVector(X);
    if(alpha!=1)
        v *= alpha;
    if(beta!=0)
        v += beta * toEigenVector(Y);
    Y.assign(v.data(), v.data()+v.size());
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X)
{
    Eigen::VectorXd v;
    if(Uplo==CblasUpper && Diag==CblasNonUnit) {
        if(TransA == CblasNoTrans)
            v = mat(A).triangularView<Eigen::Upper>() * toEigenVector(X);
        else
            v = mat(A).triangularView<Eigen::Upper>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasUpper && Diag==CblasUnit) {
        if(TransA == CblasNoTrans)
            v = mat(A).triangularView<Eigen::UnitUpper>() * toEigenVector(X);
        else
            v = mat(A).triangularView<Eigen::UnitUpper>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasLower && Diag==CblasNonUnit) {
        if(TransA == CblasNoTrans)
            v = mat(A).triangularView<Eigen::Lower>() * toEigenVector(X);
        else
            v = mat(A).triangularView<Eigen::Lower>().transpose() * toEigenVector(X);
    } else if(Uplo==CblasLower && Diag==CblasUnit) {
        if(TransA == CblasNoTrans)
            v = mat(A).triangularView<Eigen::UnitLower>() * toEigenVector(X);
        else
            v = mat(A).triangularView<Eigen::UnitLower>().transpose() * toEigenVector(X);
    } else
        throw std::invalid_argument("blas_dtrmv: invalid operation mode");
    X.assign(v.data(), v.data()+v.size());
}

template<typename MatrixType>
void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const MatrixType& A, const MatrixType& B, double beta, MatrixType& C)
{
    size_t NR1 = TransA==CblasNoTrans ? A.rows() : A.cols();
    size_t NC1 = TransA==CblasNoTrans ? A.cols() : A.rows();
    size_t NR2 = TransB==CblasNoTrans ? B.rows() : B.cols();
    size_t NC2 = TransB==CblasNoTrans ? B.cols() : B.rows();
    if(NC1 != NR2 || NR1 != C.rows() || NC2 != C.cols())
        throw std::length_error("blas_dgemm: incompatible matrix dimensions");
    if(TransA == CblasNoTrans) {
        if(TransB == CblasNoTrans) {
            if(beta==0)
                mat(C) = alpha * mat(A) * mat(B);
            else
                mat(C) = alpha * mat(A) * mat(B) + beta * mat(C);
        } else {
            if(beta==0)
                mat(C) = alpha * mat(A) * mat(B).transpose();
            else
                mat(C) = alpha * mat(A) * mat(B).transpose() + beta * mat(C);
        }
    } else {
        if(TransB == CblasNoTrans) {
            if(beta==0)
                mat(C) = alpha * mat(A).transpose() * mat(B);
            else
                mat(C) = alpha * mat(A).transpose() * mat(B) + beta * mat(C);
        } else {
            if(beta==0)
                mat(C) = alpha * mat(A).transpose() * mat(B).transpose();
            else  // in this rare case, and if MatrixType==SparseMatrix, we need to convert the storage order
                mat(C) = typename Type<MatrixType>::T(alpha * mat(A).transpose() * mat(B).transpose())
                    + beta * mat(C);
        }
    }
    // matrix shape should not have changed
    assert((size_t)mat(C).rows() == C.rows() && (size_t)mat(C).cols() == C.cols());
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B)
{
    if(alpha!=1) {
        mat(B) *= alpha;
        blas_dtrsm(Side, Uplo, TransA, Diag, 1, A, B);
        return;
    }
    if(Uplo==CblasUpper && Diag==CblasNonUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheLeft>(mat(B));
            else
                mat(A).triangularView<Eigen::Upper>().transpose().solveInPlace<Eigen::OnTheLeft>(mat(B));
        } else {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::Upper>().solveInPlace<Eigen::OnTheRight>(mat(B));
            else
                mat(A).triangularView<Eigen::Upper>().transpose().solveInPlace<Eigen::OnTheRight>(mat(B));
        }
    } else if(Uplo==CblasUpper && Diag==CblasUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::UnitUpper>().solveInPlace<Eigen::OnTheLeft>(mat(B));
            else
                mat(A).triangularView<Eigen::UnitUpper>().transpose().solveInPlace<Eigen::OnTheLeft>(mat(B));
        } else {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::UnitUpper>().solveInPlace<Eigen::OnTheRight>(mat(B));
            else
                mat(A).triangularView<Eigen::UnitUpper>().transpose().solveInPlace<Eigen::OnTheRight>(mat(B));
        }
    } else if(Uplo==CblasLower && Diag==CblasNonUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheLeft>(mat(B));
            else
                mat(A).triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheLeft>(mat(B));
        } else {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::Lower>().solveInPlace<Eigen::OnTheRight>(mat(B));
            else
                mat(A).triangularView<Eigen::Lower>().transpose().solveInPlace<Eigen::OnTheRight>(mat(B));
        }
    } else if(Uplo==CblasLower && Diag==CblasUnit) {
        if(Side==CblasLeft) {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::UnitLower>().solveInPlace<Eigen::OnTheLeft>(mat(B));
            else
                mat(A).triangularView<Eigen::UnitLower>().transpose().solveInPlace<Eigen::OnTheLeft>(mat(B));
        } else {
            if(TransA == CblasNoTrans)
                mat(A).triangularView<Eigen::UnitLower>().solveInPlace<Eigen::OnTheRight>(mat(B));
            else
                mat(A).triangularView<Eigen::UnitLower>().transpose().solveInPlace<Eigen::OnTheRight>(mat(B));
        }
    } else
        throw std::invalid_argument("blas_dtrsm: invalid operation mode");
    // matrix shape should not have changed
    assert((size_t)mat(B).rows() == B.rows() && (size_t)mat(B).cols() == B.cols());
}

// explicit function instantiations for two matrix types
template void blas_daxpy(double, const Matrix<double>&, Matrix<double>&);
template void blas_daxpy(double, const SparseMatrix<double>&, SparseMatrix<double>&);
template void blas_dmul(double, Matrix<double>&);
template void blas_dmul(double, SparseMatrix<double>&);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const Matrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const SparseMatrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const Matrix<double>&, const Matrix<double>&, double, Matrix<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const SparseMatrix<double>&, const SparseMatrix<double>&, double, SparseMatrix<double>&);


// ------ linear algebra routines ------ //

/// LU decomposition for dense matrices
typedef Eigen::PartialPivLU< Type< Matrix<double> >::T> LUDecompImpl;

/// LU decomposition for sparse matrices
typedef Eigen::SparseLU< Type< SparseMatrix<double> >::T> SparseLUDecompImpl;

LUDecomp::LUDecomp(const Matrix<double>& M) :
    sparse(false), impl(NULL)
{
    if(M.rows() != M.cols())
        throw std::runtime_error("LUDecomp needs a square matrix");
    impl = new LUDecompImpl(mat(M));
}

LUDecomp::LUDecomp(const SparseMatrix<double>& M) :
    sparse(true), impl(NULL)
{
    if(M.rows() != M.cols())
        throw std::runtime_error("LUDecomp needs a square matrix");
    SparseLUDecompImpl* LU = new SparseLUDecompImpl();
    LU->compute(mat(M));
    if(LU->info() != Eigen::Success) {
        delete LU;
        throw std::domain_error("Sparse LUDecomp failed");
    }
    impl = LU;
}

LUDecomp::LUDecomp(const LUDecomp& src) :
    sparse(src.sparse), impl(NULL)
{
    if(sparse)  // copy constructor not supported by Eigen
        throw std::runtime_error("Cannot copy Sparse LUDecomp");
    else
        impl = new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl));
}

LUDecomp::~LUDecomp()
{
    if(sparse)
        delete static_cast<const SparseLUDecompImpl*>(impl);
    else
        delete static_cast<const LUDecompImpl*>(impl);
}

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("LUDecomp not initialized");
    if(sparse)
        return toStdVector(static_cast<const SparseLUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
    else
        return toStdVector(static_cast<const LUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}


/// Cholesky decomposition for dense matrices
typedef Eigen::LLT< Type< Matrix<double> >::T, Eigen::Lower> CholeskyDecompImpl;

CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(NULL)
{
    if(M.rows() != M.cols())
        throw std::runtime_error("CholeskyDecomp needs a square matrix");
    impl = new CholeskyDecompImpl(mat(M));
    if(static_cast<const CholeskyDecompImpl*>(impl)->info() != Eigen::Success)
        throw std::domain_error("CholeskyDecomp failed");
}

CholeskyDecomp::~CholeskyDecomp() { delete static_cast<const CholeskyDecompImpl*>(impl); }

CholeskyDecomp::CholeskyDecomp(const CholeskyDecomp& src) :
    impl(new CholeskyDecompImpl(*static_cast<const CholeskyDecompImpl*>(src.impl))) {}

Matrix<double> CholeskyDecomp::L() const { 
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    size_t size = static_cast<const CholeskyDecompImpl*>(impl)->matrixLLT().rows();
    // inefficient: we first allocate an uninitialized matrix and then replace its content with L
    Matrix<double> L(size, size);
    mat(L) = static_cast<const CholeskyDecompImpl*>(impl)->matrixL();
    return L;
}

std::vector<double> CholeskyDecomp::solve(const std::vector<double>& rhs) const {
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    return toStdVector(static_cast<const CholeskyDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}


/// QR decomposition for dense matrices
typedef Eigen::HouseholderQR< Type< Matrix<double> >::T> QRDecompImpl;

QRDecomp::QRDecomp(const Matrix<double>& M) :
    impl(new QRDecompImpl(mat(M))) {}

QRDecomp::~QRDecomp() { delete static_cast<QRDecompImpl*>(impl); }

QRDecomp::QRDecomp(const QRDecomp& src) :
    impl(new QRDecompImpl(*static_cast<const QRDecompImpl*>(src.impl))) {}

void QRDecomp::QR(Matrix<double>& Q, Matrix<double>& R) const
{
    if(!impl)
        throw std::runtime_error("QRDecomp not initialized");
    const Type< Matrix<double> >::T& mQR = static_cast<const QRDecompImpl*>(impl)->matrixQR();
    Q = Matrix<double>(mQR.rows(), mQR.rows());
    R = Matrix<double>(mQR.rows(), mQR.cols());
    mat(Q) = static_cast<const QRDecompImpl*>(impl)->householderQ();
    mat(R) = static_cast<const QRDecompImpl*>(impl)->matrixQR().triangularView<Eigen::Upper>();
}

std::vector<double> QRDecomp::solve(const std::vector<double>& rhs) const {
    if(!impl)
        throw std::runtime_error("QRDecomp not initialized");
    return toStdVector(static_cast<const QRDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}


/// Singular-value decomposition for dense matrices
#if EIGEN_VERSION_AT_LEAST(3,3,0)
typedef Eigen::BDCSVD< Type< Matrix<double> >::T> SVDecompImpl;     // more efficient
#else
typedef Eigen::JacobiSVD< Type< Matrix<double> >::T> SVDecompImpl;  // slower
#endif

SVDecomp::SVDecomp(const Matrix<double>& M) :
    impl(new SVDecompImpl(mat(M), Eigen::ComputeThinU | Eigen::ComputeThinV)) {}

SVDecomp::~SVDecomp() { delete static_cast<SVDecompImpl*>(impl); }

SVDecomp::SVDecomp(const SVDecomp& src) :
    impl(new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl))) {}

Matrix<double> SVDecomp::U() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    const Type< Matrix<double> >::T& mU = static_cast<const SVDecompImpl*>(impl)->matrixU();
    Matrix<double> U(mU.rows(), mU.cols());
    mat(U) = mU;
    return U;
}

Matrix<double> SVDecomp::V() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    const Type< Matrix<double> >::T& mV = static_cast<const SVDecompImpl*>(impl)->matrixV();
    Matrix<double> V(mV.rows(), mV.cols());
    mat(V) = mV;
    return V;
}

std::vector<double> SVDecomp::S() const { 
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    return toStdVector(static_cast<const SVDecompImpl*>(impl)->singularValues());
}

std::vector<double> SVDecomp::solve(const std::vector<double>& rhs) const {
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    return toStdVector(static_cast<const SVDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}

// --------- END OF EIGEN-BASED IMPLEMENTATIONS --------- //
#else
// --------- GSL-BASED IMPLEMENTATIONS --------- //
namespace{
// allocate memory if the size is positive, otherwise return NULL
template<typename NumT>
inline NumT* xnew(size_t size) {
    return size==0 ? NULL : new NumT[size]; }
}

// if any GSL function triggers an error, it will be stored in these variables (defined in math_core.cpp)
extern bool exceptionFlag;
extern std::string exceptionText;

#define CALL_FUNCTION_OR_THROW(x) { \
    exceptionFlag = false; \
    x; \
    if(exceptionFlag) throw std::runtime_error(exceptionText); \
}

//------ dense matrix ------//

// default constructor
template<typename NumT>
Matrix<NumT>::Matrix() :
    IMatrixDense<NumT>(), impl(NULL) {}

// constructor without data initialization
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols) :
    IMatrixDense<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols)) {}

// constructor with a given initial value
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols, const NumT value) :
    IMatrixDense<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nCols*nRows, value);
}

// copy constructor from a dense matrix
template<typename NumT>
Matrix<NumT>::Matrix(const Matrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(rows()*cols()))
{
    std::copy(
        /*src  begin*/ static_cast<const NumT*>(src.impl),
        /*src  end  */ static_cast<const NumT*>(src.impl) + rows()*cols(),
        /*dest begin*/ static_cast<NumT*>(impl));
}

// copy constructor from a sparse matrix
template<typename NumT>
Matrix<NumT>::Matrix(const SparseMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(rows()*cols()))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + rows()*cols(), static_cast<NumT>(0));
    size_t numelem = src.size();
    for(size_t k=0; k<numelem; k++) {
        size_t i, j;
        NumT v = src.elem(k, i, j);
        static_cast<NumT*>(impl)[i*cols()+j] = v;
    }
}

// copy constructor from a band matrix
template<typename NumT>
Matrix<NumT>::Matrix(const BandMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()),
    impl(xnew<NumT>(rows()*cols()))
{
    const ptrdiff_t band = src.bandwidth(), width = band * 2 + 1, nRows = rows();
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + pow_2(nRows), static_cast<NumT>(0));
    const NumT* data = &(src(0,0))-band;
    for(ptrdiff_t r=0; r<nRows; r++) {
        ptrdiff_t cl = std::max<ptrdiff_t>(0, r-band);
        ptrdiff_t cr = std::min<ptrdiff_t>(nRows-1, r+band);
        for(ptrdiff_t c = cl, m = r*width + cl-r+band; c <= cr; c++, m++)
            static_cast<NumT*>(impl)[r*nRows+c] = data[m];
    }
}

// copy constructor from anything matrix-like (same as from sparse matrix)
template<typename NumT>
Matrix<NumT>::Matrix(const IMatrix<NumT>& src) :
    IMatrixDense<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(rows()*cols()))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + rows()*cols(), static_cast<NumT>(0));
    size_t numelem = src.size();
    for(size_t k=0; k<numelem; k++) {
        size_t i, j;
        NumT v = src.elem(k, i, j);
        static_cast<NumT*>(impl)[i*cols()+j] = v;
    }
}

// constructor from triplets
template<typename NumT>
Matrix<NumT>::Matrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values) :
    IMatrixDense<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nRows*nCols, static_cast<NumT>(0));
    size_t numelem = values.size();
    for(size_t k=0; k<numelem; k++) {
        size_t i = values[k].i, j = values[k].j;
        if(i<nRows && j<nCols)
            static_cast<NumT*>(impl)[i*nCols+j] += static_cast<NumT>(values[k].v);
        else {
            delete[] static_cast<NumT*>(impl);
            throw std::out_of_range("Matrix: triplet index out of range");
        }
    }
}

template<typename NumT>
Matrix<NumT>::~Matrix()
{
    delete[] static_cast<NumT*>(impl);
}

/// access the matrix element for reading
template<typename NumT>
const NumT& Matrix<NumT>::operator() (size_t row, size_t col) const
{
#ifdef DEBUG_RANGE_CHECK
    matrixRangeCheck(row < rows() && col < cols());
#endif
    return static_cast<const NumT*>(impl)[row * cols() + col];
}

/// access the matrix element for writing
template<typename NumT>
NumT& Matrix<NumT>::operator() (size_t row, size_t col)
{
#ifdef DEBUG_RANGE_CHECK
    matrixRangeCheck(row < rows() && col < cols());
#endif
    return static_cast<NumT*>(impl)[row * cols() + col];
}

/// access the raw data storage
template<typename NumT>
NumT* Matrix<NumT>::data() { return static_cast<NumT*>(impl); }

template<typename NumT>
const NumT* Matrix<NumT>::data() const { return static_cast<const NumT*>(impl); }

//------- sparse matrix --------//

// GSL sparse matrices are implemented only in version >= 2, and only with numerical format = double
#ifdef HAVE_GSL_SPARSE

// empty constructor
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix() :
    IMatrix<NumT>(), impl(NULL) {}

// constructor from triplets
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols), impl(NULL)
{
    if(nRows*nCols==0)
        return;  // no allocation in case of zero matrix
    size_t size = values.size();
    exceptionFlag = false;
    gsl_spmatrix* sp  = gsl_spmatrix_alloc_nzmax(nRows, nCols, size, GSL_SPMATRIX_TRIPLET);
    for(size_t k=0; k<size; k++)
        gsl_spmatrix_set(sp, values[k].i, values[k].j, values[k].v);
    impl = gsl_spmatrix_compcol(sp);
    gsl_spmatrix_free(sp);
    if(exceptionFlag) throw std::runtime_error(exceptionText);
}

// copy constructor from sparse matrix
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(const SparseMatrix<NumT>& srcObj) :
    IMatrix<NumT>(srcObj.rows(), srcObj.cols()), impl(NULL)
{
    if(rows()*cols() == 0)
        return;
    const gsl_spmatrix* src = static_cast<const gsl_spmatrix*>(srcObj.impl);
    exceptionFlag = false;
    gsl_spmatrix* dest = gsl_spmatrix_alloc_nzmax(src->size1, src->size2, src->nz, GSL_SPMATRIX_CCS);
    gsl_spmatrix_memcpy(dest, src);
    if(exceptionFlag) throw std::runtime_error(exceptionText);
    impl = dest;
}

// copy constructor from band matrix
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(const BandMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()), impl(NULL)
{
    size_t nRows = src.rows();
    if(nRows == 0)
        return;
    size_t size = src.size();
    exceptionFlag = false;
    gsl_spmatrix* sp  = gsl_spmatrix_alloc_nzmax(nRows, nRows, size, GSL_SPMATRIX_TRIPLET);
    for(size_t k=0; k<size; k++) {
        size_t row, col;
        double val = src.elem(k, row, col);
        gsl_spmatrix_set(sp, row, col, val);
    }
    impl = gsl_spmatrix_compcol(sp);
    gsl_spmatrix_free(sp);
    if(exceptionFlag) throw std::runtime_error(exceptionText);
}

template<typename NumT>
SparseMatrix<NumT>::~SparseMatrix()
{
    if(impl)
        gsl_spmatrix_free(static_cast<gsl_spmatrix*>(impl));
}

template<typename NumT>
NumT SparseMatrix<NumT>::at(size_t row, size_t col) const
{
    if(impl)
        return static_cast<NumT>(gsl_spmatrix_get(static_cast<const gsl_spmatrix*>(impl), row, col));
    else
        throw std::length_error("SparseMatrix: empty matrix");
}

template<typename NumT>
NumT SparseMatrix<NumT>::elem(const size_t index, size_t &row, size_t &col) const
{
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    matrixRangeCheck(impl != NULL && index < sp->nz);
    row = sp->i[index];
#if (GSL_MAJOR_VERSION == 2) && (GSL_MINOR_VERSION < 6)
    col = binSearch(index, sp->p, sp->size2+1);
#else
    col = binSearch(static_cast<int>(index), sp->p, sp->size2+1);  // type of sp->p changed to int*
#endif
    return static_cast<NumT>(sp->data[index]);
}

template<typename NumT>
size_t SparseMatrix<NumT>::size() const
{
    return impl==NULL ? 0 : static_cast<const gsl_spmatrix*>(impl)->nz; 
}

template<typename NumT>
std::vector<Triplet> SparseMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    if(impl == NULL)
        return result;
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    result.reserve(sp->nz);
    size_t col = 0;
    for(size_t k=0; k<sp->nz; k++) {
        while(col < static_cast<size_t>(sp->size2) && static_cast<size_t>(sp->p[col+1]) <= k)
            col++;
        result.push_back(Triplet(sp->i[k], col, sp->data[k]));
    }
    return result;
}

#else
// no GSL support for sparse matrices - implement them as dense matrices

template<typename NumT>
SparseMatrix<NumT>::SparseMatrix() :
    IMatrix<NumT>(), impl(NULL) {}

// constructor from triplets
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nRows*nCols, 0);
    size_t numelem = values.size();
    for(size_t k=0; k<numelem; k++) {
        size_t i = values[k].i, j = values[k].j;
        if(i<nRows && j<nCols)
            static_cast<NumT*>(impl)[i*nCols+j] += static_cast<NumT>(values[k].v);
        else {
            delete[] static_cast<NumT*>(impl);
            throw std::out_of_range("SparseMatrix: triplet index out of range");
        }
    }
}

// copy constructor from sparse matrix
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(const SparseMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(xnew<NumT>(rows()*cols()))
{
    std::copy(
        /*src  begin*/ static_cast<const NumT*>(src.impl),
        /*src  end  */ static_cast<const NumT*>(src.impl) + rows()*cols(),
        /*dest begin*/ static_cast<NumT*>(impl));
}

// copy constructor from band matrix
template<typename NumT>
SparseMatrix<NumT>::SparseMatrix(const BandMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(src.rows()*src.cols()))
{
    size_t nRows = src.rows();
    size_t size  = src.size();
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nRows*nRows, 0);
    for(size_t k=0; k<size; k++) {
        size_t row, col;
        NumT val = src.elem(k, row, col);
        static_cast<NumT*>(impl)[row*nRows+col] += static_cast<NumT>(val);
    }
}

template<typename NumT>
SparseMatrix<NumT>::~SparseMatrix()
{
    delete static_cast<NumT*>(impl);
}

template<typename NumT>
NumT SparseMatrix<NumT>::at(size_t row, size_t col) const
{
    matrixRangeCheck(row < rows() && col < cols());
    return static_cast<const NumT*>(impl)[row * cols() + col];
}

template<typename NumT>
NumT SparseMatrix<NumT>::elem(const size_t index, size_t &row, size_t &col) const
{
    row = index / cols();
    col = index % cols();
    matrixRangeCheck(row < rows());
    return static_cast<const NumT*>(impl)[index];
}

template<typename NumT>
size_t SparseMatrix<NumT>::size() const { return rows()*cols(); }

template<typename NumT>
std::vector<Triplet> SparseMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    for(size_t k=0; k<cols() * rows(); k++)
        if(static_cast<const NumT*>(impl)[k] != 0)
            result.push_back(Triplet(k / cols(), k % cols(), static_cast<const NumT*>(impl)[k]));
    return result;
}

#endif

// ------ wrappers for BLAS routines ------ //

namespace {
// wrappers for GSL vector and matrix views (access the data arrays without copying)
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
        m(gsl_matrix_view_array(mat.data(), mat.rows(), mat.cols())) {}
    Mat(double* data, size_t rows, size_t cols) :
        m(gsl_matrix_view_array(data, rows, cols)) {}
    operator gsl_matrix* () { return &m.matrix; }
private:
    gsl_matrix_view m;
};

struct MatC {
    explicit MatC(const Matrix<double>& mat) :
        m(gsl_matrix_const_view_array(mat.data(), mat.rows(), mat.cols())) {}
    MatC(const double* data, size_t rows, size_t cols) :
        m(gsl_matrix_const_view_array(data, rows, cols)) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};
} // internal namespace

template<> void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y)
{
    if(X.rows() != Y.rows() || X.cols() != Y.cols())
        throw std::length_error("blas_daxpy: incompatible sizes of input arrays");
    if(alpha==0) return;
    const size_t  size = X.size();
    const double* arrX = X.data();
    double* arrY = Y.data();
    for(size_t k=0; k<size; k++)
        arrY[k] += alpha*arrX[k];
}

template<> void blas_dmul(double alpha, Matrix<double>& Y)
{
    const size_t size = Y.size();
    double* y = Y.data();
    for(size_t k=0; k<size; k++)
        y[k] *= alpha;
}

template<> void blas_daxpy(double alpha, const SparseMatrix<double>& X, SparseMatrix<double>& Y)
{
    if(X.rows() != Y.rows() || X.cols() != Y.cols())
        throw std::length_error("blas_daxpy: incompatible sizes of input arrays");
    if(alpha==0) return;
#ifdef HAVE_GSL_SPARSE
    const gsl_spmatrix* spX = static_cast<const gsl_spmatrix*>(X.impl);
    gsl_spmatrix* spY = static_cast<gsl_spmatrix*>(Y.impl), *tmpX = NULL;
    exceptionFlag = false;
    if(alpha!=1) {  // allocate a temporary sparse matrix for X*alpha
        tmpX = gsl_spmatrix_alloc_nzmax(spX->size1, spX->size2, spX->nz, GSL_SPMATRIX_CCS);
        gsl_spmatrix_memcpy(tmpX, spX);
        gsl_spmatrix_scale(tmpX, alpha);
        spX = tmpX;
    }
    gsl_spmatrix* result = gsl_spmatrix_alloc_nzmax(spY->size1, spY->size2,
        std::max(spX->nz, spY->nz), GSL_SPMATRIX_CCS);
    gsl_spmatrix_add(result, spX, spY);
    gsl_spmatrix_free(spY);
    Y.impl = result;
    if(alpha!=1)
        gsl_spmatrix_free(tmpX);
    if(exceptionFlag) throw std::runtime_error(exceptionText);
#else
    const size_t size = X.rows() * X.cols();
    for(size_t k=0; k<size; k++)
        static_cast<double*>(Y.impl)[k] += alpha * static_cast<const double*>(X.impl)[k];
#endif
}

template<> void blas_dmul(double alpha, SparseMatrix<double>& Y)
{
#ifdef HAVE_GSL_SPARSE
    gsl_spmatrix_scale(static_cast<gsl_spmatrix*>(Y.impl), alpha);
#else
    const size_t size = Y.rows() * Y.cols();
    for(size_t k=0; k<size; k++)
        static_cast<double*>(Y.impl)[k] *= alpha;
#endif
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const Matrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, MatC(A), VecC(X), beta, Vec(Y)) )
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const SparseMatrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    if(A.impl == NULL)
        throw std::length_error("blas_dgemv: empty matrix");
#ifdef HAVE_GSL_SPARSE
    CALL_FUNCTION_OR_THROW(
    gsl_spblas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, static_cast<const gsl_spmatrix*>(A.impl),
        VecC(X), beta, Vec(Y)) )
#else
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha,
        MatC(static_cast<const double*>(A.impl), A.rows(), A.cols()), VecC(X), beta, Vec(Y)) )
#endif
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X)
{
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dtrmv((CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, MatC(A), Vec(X)) )
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C)
{
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB,
        alpha, MatC(A), MatC(B), beta, Mat(C)) )
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const SparseMatrix<double>& A, const SparseMatrix<double>& B,
    double beta, SparseMatrix<double>& C)
{
    if(A.impl == NULL || B.impl == NULL || C.impl == NULL)
        throw std::length_error("blas_dgemv: empty matrix");
#ifdef HAVE_GSL_SPARSE
    if(beta!=0)
        throw std::runtime_error("blas_dgemm: beta!=0 not implemented");
    const gsl_spmatrix* spA = static_cast<const gsl_spmatrix*>(A.impl);
    const gsl_spmatrix* spB = static_cast<const gsl_spmatrix*>(B.impl);
    gsl_spmatrix *trA = NULL, *trB = NULL;
    if(TransA != CblasNoTrans) {
        trA = gsl_spmatrix_alloc_nzmax(spA->size2, spA->size1, spA->nz, GSL_SPMATRIX_CCS);
        gsl_spmatrix_transpose_memcpy(trA, spA);
        spA = trA;
    }
    if(TransB != CblasNoTrans) {
        trB = gsl_spmatrix_alloc_nzmax(spB->size2, spB->size1, spB->nz, GSL_SPMATRIX_CCS);
        gsl_spmatrix_transpose_memcpy(trB, spB);
        spB = trB;
    }
    CALL_FUNCTION_OR_THROW(
    gsl_spblas_dgemm(alpha, spA, spB, static_cast<gsl_spmatrix*>(C.impl)) )
    if(TransA != CblasNoTrans)
        gsl_spmatrix_free(trA);
    if(TransB != CblasNoTrans)
        gsl_spmatrix_free(trB);
#else
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, alpha,
        MatC(static_cast<const double*>(A.impl), A.rows(), A.cols()),
        MatC(static_cast<const double*>(B.impl), B.rows(), B.cols()),
        beta, Mat(static_cast <double*>(C.impl), C.rows(), C.cols())) )
#endif
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B)
{
    CALL_FUNCTION_OR_THROW(
    gsl_blas_dtrsm((CBLAS_SIDE_t)Side, (CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag,
        alpha, MatC(A), Mat(B)) )
}

// ----- Linear algebra routines ----- //

/// LU decomposition implementation for GSL
struct LUDecompImpl {
    gsl_matrix* LU;
    gsl_permutation* perm;
    LUDecompImpl(const double* data, size_t rows, size_t cols) {
        if(rows!=cols)
            throw std::length_error("LUDecomp needs a square matrix");
        LU = gsl_matrix_alloc(rows, cols);
        perm = gsl_permutation_alloc(rows);
        if(!LU || !perm) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::bad_alloc();
        }
        int dummy;
        gsl_matrix_memcpy(LU, MatC(data, rows, cols));
        exceptionFlag = false;
        gsl_linalg_LU_decomp(LU, perm, &dummy);
        if(exceptionFlag) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::runtime_error(exceptionText);
        }
    }
    LUDecompImpl(const LUDecompImpl& src) {
        LU = gsl_matrix_alloc(src.LU->size1, src.LU->size2);
        perm = gsl_permutation_alloc(src.LU->size1);
        if(!LU || !perm) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::bad_alloc();
        }
        gsl_matrix_memcpy(LU, src.LU);
        gsl_permutation_memcpy(perm, src.perm);
    }
    ~LUDecompImpl() {
        gsl_permutation_free(perm);
        gsl_matrix_free(LU);
    }
private:
    LUDecompImpl& operator=(const LUDecompImpl&);
};

LUDecomp::LUDecomp(const Matrix<double>& M) :
    sparse(false), impl(new LUDecompImpl(M.data(), M.rows(), M.cols())) {}

// GSL does not offer LU decomposition of sparse matrices, so they are converted to dense ones
LUDecomp::LUDecomp(const SparseMatrix<double>& M) :
    sparse(false),
#ifdef HAVE_GSL_SPARSE
    impl(new LUDecompImpl(Matrix<double>(M).data(), M.rows(), M.cols()))
#else
    impl(new LUDecompImpl(static_cast<const double*>(M.impl), M.rows(), M.cols()))
#endif
{}

LUDecomp::~LUDecomp() { delete static_cast<LUDecompImpl*>(impl); }

LUDecomp::LUDecomp(const LUDecomp& src) :
    sparse(false), impl(new LUDecompImpl(*static_cast<const LUDecompImpl*>(src.impl))) {}

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("LUDecomp not initialized");
    if(rhs.size() != static_cast<const LUDecompImpl*>(impl)->LU->size1)
        throw std::length_error("LUDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    CALL_FUNCTION_OR_THROW(
    gsl_linalg_LU_solve(static_cast<const LUDecompImpl*>(impl)->LU,
        static_cast<const LUDecompImpl*>(impl)->perm, VecC(rhs), Vec(result)) )
    return result;
}

/// Cholesky decomposition implementation for GSL
CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(NULL)
{
    if(M.rows()!=M.cols())
        throw std::length_error("CholeskyDecomp needs a square matrix");
    gsl_matrix* L = gsl_matrix_alloc(M.rows(), M.cols());
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_memcpy(L, MatC(M));
    exceptionFlag = false;
    gsl_linalg_cholesky_decomp(L);
    if(exceptionFlag) {
        gsl_matrix_free(L);
        throw std::domain_error(exceptionText);
    }
    impl=L;
}

CholeskyDecomp::~CholeskyDecomp()
{
    gsl_matrix_free(static_cast<gsl_matrix*>(impl));
}

CholeskyDecomp::CholeskyDecomp(const CholeskyDecomp& srcObj) :
    impl(NULL)
{
    const gsl_matrix* src = static_cast<const gsl_matrix*>(srcObj.impl);
    gsl_matrix* L = gsl_matrix_alloc(src->size1, src->size2);
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_memcpy(L, src);
    impl = L;
}

Matrix<double> CholeskyDecomp::L() const
{
    const gsl_matrix* M = static_cast<const gsl_matrix*>(impl);
    if(!M || M->size1!=M->size2)
        throw std::runtime_error("CholeskyDecomp not initialized");
    Matrix<double> L(M->size1, M->size2, 0);
    // copy the lower triangular part of the matrix; the upper triangle contains garbage
    for(size_t i=0; i<M->size1; i++)
        for(size_t j=0; j<=i; j++)
            L(i,j) = M->data[i*M->size2+j];
    return L;
}

std::vector<double> CholeskyDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    if(rhs.size() != static_cast<const gsl_matrix*>(impl)->size1)
        throw std::length_error("CholeskyDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    CALL_FUNCTION_OR_THROW(
    gsl_linalg_cholesky_solve(static_cast<const gsl_matrix*>(impl), VecC(rhs), Vec(result)) )
    return result;
}

/// QR decomposition implementation for GSL
struct QRDecompImpl {
    Matrix<double> mat;
    std::vector<double> vec;
};

QRDecomp::QRDecomp(const Matrix<double>& M) :
    impl(new QRDecompImpl())
{
    QRDecompImpl* qr = static_cast<QRDecompImpl*>(impl);
    qr->mat = M;
    qr->vec.resize(std::min(M.cols(), M.rows()));
    CALL_FUNCTION_OR_THROW(gsl_linalg_QR_decomp(Mat(qr->mat), Vec(qr->vec)) )
}

QRDecomp::~QRDecomp() { delete static_cast<QRDecompImpl*>(impl); }

QRDecomp::QRDecomp(const QRDecomp& src) :
    impl(new QRDecompImpl(*static_cast<const QRDecompImpl*>(src.impl))) {}

void QRDecomp::QR(Matrix<double>& Q, Matrix<double>& R) const
{
    if(!impl)
        throw std::runtime_error("QRDecomp not initialized");
    QRDecompImpl* qr = static_cast<QRDecompImpl*>(impl);
    Q = Matrix<double>(qr->mat.rows(), qr->mat.rows());
    R = Matrix<double>(qr->mat.rows(), qr->mat.cols());
    gsl_linalg_QR_unpack(MatC(qr->mat), VecC(qr->vec), Mat(Q), Mat(R));
}

std::vector<double> QRDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("QRDecomp not initialized");
    const QRDecompImpl* qr = static_cast<const QRDecompImpl*>(impl);
    if(rhs.size() != qr->mat.rows())
        throw std::length_error("QRDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs);
    // compute result = Q^T rhs
    gsl_linalg_QR_QTvec(MatC(qr->mat), VecC(qr->vec), Vec(result));
    // if the matrix is non-square, select the upper left corner as R
    size_t size = std::min(qr->mat.cols(), qr->mat.rows());
    gsl_matrix_const_view R = gsl_matrix_const_submatrix (MatC(qr->mat), 0, 0, size, size);
    // solve R x = result, storing x in-place
    gsl_blas_dtrsv((CBLAS_UPLO_t)CblasUpper, (CBLAS_TRANSPOSE_t)CblasNoTrans,
        (CBLAS_DIAG_t)CblasNonUnit, &(R.matrix), Vec(result));
    result.resize(qr->mat.cols());
    return result;
}

/// singular-value decomposition implementation for GSL
struct SVDecompImpl {
    Matrix<double> U, V;
    std::vector<double> S;
};

SVDecomp::SVDecomp(const Matrix<double>& M) :
    impl(new SVDecompImpl())
{
    SVDecompImpl* sv = static_cast<SVDecompImpl*>(impl);
    sv->U = M;
    sv->V = Matrix<double>(M.cols(), M.cols());
    sv->S.resize(M.cols());
    std::vector<double> temp(M.cols());
    if(M.rows() >= M.cols()*5) {   // use a modified algorithm for very 'elongated' matrices
        Matrix<double> tempmat(M.cols(), M.cols());
        CALL_FUNCTION_OR_THROW(
        gsl_linalg_SV_decomp_mod(Mat(sv->U), Mat(tempmat), Mat(sv->V), Vec(sv->S), Vec(temp)) )
    } else {
        CALL_FUNCTION_OR_THROW(
        gsl_linalg_SV_decomp(Mat(sv->U), Mat(sv->V), Vec(sv->S), Vec(temp)) )
    }
    // chop off excessively small singular values which may destabilize solution of linear system
    double minSV = sv->S[0] * 1e-15 * std::max<size_t>(sv->S.size(), 10);
    for(size_t k=0; k<sv->S.size(); k++)
        if(sv->S[k] < minSV)
            sv->S[k] = 0;
}

SVDecomp::~SVDecomp() { delete static_cast<SVDecompImpl*>(impl); }

SVDecomp::SVDecomp(const SVDecomp& src) :
    impl(new SVDecompImpl(*static_cast<const SVDecompImpl*>(src.impl))) {}

Matrix<double> SVDecomp::U() const { return static_cast<const SVDecompImpl*>(impl)->U; }

Matrix<double> SVDecomp::V() const { return static_cast<const SVDecompImpl*>(impl)->V; }

std::vector<double> SVDecomp::S() const { return static_cast<const SVDecompImpl*>(impl)->S; }

std::vector<double> SVDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("SVDecomp not initialized");
    const SVDecompImpl* sv = static_cast<const SVDecompImpl*>(impl);
    if(rhs.size() != sv->U.rows())
        throw std::length_error("SVDecomp: incorrect size of RHS vector");
    std::vector<double> result(sv->U.cols());
    CALL_FUNCTION_OR_THROW(
    gsl_linalg_SV_solve(MatC(sv->U), MatC(sv->V), VecC(sv->S), VecC(rhs), Vec(result)) )
    return result;
}

#endif

// template instantiations to be compiled (both for Eigen and GSL)
template struct BandMatrix<float>;
template struct BandMatrix<double>;
template struct SparseMatrix<float>;
template struct SparseMatrix<double>;
template struct Matrix<float>;
template struct Matrix<double>;

}  // namespace
