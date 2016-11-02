#include "math_linalg.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

#ifdef HAVE_EIGEN

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

namespace math{

// ------ utility routines with common implementations for both EIGEN and GSL ------ //

void eliminateNearZeros(std::vector<double>& vec, double threshold)
{
    double mag=0;
    for(unsigned int t=0; t<vec.size(); t++)
        mag = fmax(mag, fabs(vec[t]));
    mag *= threshold;
    for(unsigned int t=0; t<vec.size(); t++)
        if(fabs(vec[t]) <= mag)
            vec[t]=0;
}

void eliminateNearZeros(Matrix<double>& mat, double threshold)
{
    double mag=0;
    for(unsigned int i=0; i<mat.rows(); i++)
        for(unsigned int j=0; j<mat.cols(); j++)
            mag = fmax(mag, fabs(mat(i,j)));
    mag *= threshold;
    for(unsigned int i=0; i<mat.rows(); i++)
        for(unsigned int j=0; j<mat.cols(); j++)
            if(fabs(mat(i,j)) <= mag)
                mat(i,j)=0;
}

template<> void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y)
{
    unsigned int size = X.size();
    if(size!=Y.size())
        throw std::invalid_argument("blas_daxpy: invalid size of input arrays");
    if(alpha==0) return;
    for(unsigned int i=0; i<size; i++)
        Y[i] += alpha*X[i];
}

double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y)
{
    unsigned int size = X.size();
    if(size!=Y.size())
        throw std::invalid_argument("blas_ddot: invalid size of input arrays");
    double result = 0;
    for(unsigned int i=0; i<size; i++)
        result += X[i]*Y[i];
    return result;
}

std::vector<double> solveTridiag(
    const std::vector<double>& diag,
    const std::vector<double>& aboveDiag,
    const std::vector<double>& belowDiag,
    const std::vector<double>& rhs)
{
    unsigned int size = diag.size();
    if(size<1 || rhs.size() != size || aboveDiag.size()+1 != size || belowDiag.size()+1 != size)
        throw std::invalid_argument("solveTridiag: invalid size of input arrays");
    std::vector<double> x(size);  // solution
    std::vector<double> t(size);  // temporary array
    // forward pass
    for(unsigned int i=0; i<size; i++) {
        double piv = diag[i] - (i>0 ? belowDiag[i-1] * t[i-1] : 0);
        if(piv == 0)
            throw std::domain_error("solveTridiag: zero pivot element on diagonal");
        t[i] = i<size-1 ? aboveDiag[i] / piv : 0;
        x[i] = (rhs[i] - (i>0 ? belowDiag[i-1] * x[i-1] : 0)) / piv;
    }
    // back-substitution
    for(unsigned int i=size-1; i>0; i--)
        x[i-1] -= t[i-1] * x[i];
    x.resize(diag.size());
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
struct Type< SpMatrix<NumT> > {
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
    IMatrix<NumT>(),
    impl(new typename Type<Matrix<NumT> >::T()) {}

// constructor without data initialization
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols) :
    IMatrix<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols)) {}

// constructor with an initial value
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols, const NumT value) :
    IMatrix<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols))
{
    mat(*this).fill(value);
}

// copy constructor from a dense matrix
template<typename NumT>
Matrix<NumT>::Matrix(const Matrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(mat(src))) {}

// copy constructor from a sparse matrix
template<typename NumT>
Matrix<NumT>::Matrix(const SpMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(new typename Type<Matrix<NumT> >::T(mat(src))) {}

// constructor from triplets
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols),
    impl(new typename Type<Matrix<NumT> >::T(nRows, nCols))
{
    mat(*this).fill(0);
    for(unsigned int k=0; k<values.size(); k++)
        mat(*this)(values[k].i, values[k].j) = static_cast<NumT>(values[k].v);
}

template<typename NumT>
Matrix<NumT>::~Matrix()
{
    delete &(mat(*this));
}

/// access a matrix element
template<typename NumT>
const NumT& Matrix<NumT>::operator() (unsigned int row, unsigned int col) const
{
#ifdef DEBUG_RANGE_CHECK
    return mat(*this)(row, col);
#else
    return mat(*this).data()[row * cols() + col];
#endif
}

template<typename NumT>
NumT& Matrix<NumT>::operator() (unsigned int row, unsigned int col)
{
#ifdef DEBUG_RANGE_CHECK
    return mat(*this)(row, col);
#else
    return mat(*this).data()[row * cols() + col];
#endif
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
SpMatrix<NumT>::SpMatrix() :
    IMatrix<NumT>(), impl(new typename Type<SpMatrix<NumT> >::T()) {}

// constructor from triplets
template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols),
    impl(new typename Type<SpMatrix<NumT> >::T(nRows, nCols))
{
    mat(*this).setFromTriplets(values.begin(), values.end());
}

// copy constructor from sparse matrix
template<typename NumT>
SpMatrix<NumT>::SpMatrix(const SpMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(new typename Type<SpMatrix<NumT> >::T(mat(src))) {}

template<typename NumT>
SpMatrix<NumT>::~SpMatrix()
{
    delete &(mat(*this));
}

// element access
template<typename NumT>
NumT SpMatrix<NumT>::operator() (unsigned int row, unsigned int col) const
{
    return mat(*this).coeff(row, col);
}

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    if(static_cast<int>(index) >= mat(*this).nonZeros())
        throw std::range_error("SpMatrix: element index out of range");
    row = mat(*this).innerIndexPtr()[index];
    col = binSearch(static_cast<int>(index), mat(*this).outerIndexPtr(), cols()+1);
    return mat(*this).valuePtr()[index];
}

template<typename NumT>
unsigned int SpMatrix<NumT>::size() const
{
    return mat(*this).nonZeros();
}

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    result.reserve(mat(*this).nonZeros());
    for(unsigned int j=0; j<cols(); ++j)
        for(typename Type<SpMatrix<NumT> >::T::InnerIterator i(mat(*this), j); i; ++i)
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

template<> void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y)
{
    if(alpha!=0)
        mat(Y) += alpha * mat(X);
}

template<typename MatrixType>
void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const MatrixType& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    Eigen::VectorXd v;
    if(TransA==CblasNoTrans)
        v = alpha * mat(A) * toEigenVector(X);
    else
        v = alpha * mat(A).transpose() * toEigenVector(X);
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
    unsigned int NR1 = TransA==CblasNoTrans ? A.rows() : A.cols();
    unsigned int NC1 = TransA==CblasNoTrans ? A.cols() : A.rows();
    unsigned int NR2 = TransB==CblasNoTrans ? B.rows() : B.cols();
    unsigned int NC2 = TransB==CblasNoTrans ? B.cols() : B.rows();
    if(NC1 != NR2 || NR1 != C.rows() || NC2 != C.cols())
        throw std::invalid_argument("blas_dgemm: incompatible matrix dimensions");
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
            else  // in this rare case, and if MatrixType==SpMatrix, we need to convert the storage order
                mat(C) = typename Type<MatrixType>::T(alpha * mat(A).transpose() * mat(B).transpose())
                    + beta * mat(C);
        }
    }
    // matrix shape should not have changed
    assert((unsigned int)mat(C).rows() == C.rows() && (unsigned int)mat(C).cols() == C.cols());
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
    assert((unsigned int)mat(B).rows() == B.rows() && (unsigned int)mat(B).cols() == B.cols());
}


// ------ linear algebra routines ------ //    

/// LU decomposition for dense matrices
typedef Eigen::PartialPivLU< Type< Matrix<double> >::T> LUDecompImpl;

/// LU decomposition for sparse matrices
typedef Eigen::SparseLU< Type< SpMatrix<double> >::T> SpLUDecompImpl;

LUDecomp::LUDecomp(const Matrix<double>& M) :
    sparse(false), impl(new LUDecompImpl(mat(M))) {}

LUDecomp::LUDecomp(const SpMatrix<double>& M) :
    sparse(true), impl(NULL)
{
    SpLUDecompImpl* LU = new SpLUDecompImpl();
    LU->compute(mat(M));
    if(LU->info() != Eigen::Success) {
        delete LU;
        throw std::runtime_error("Sparse LUDecomp failed");
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
        delete static_cast<const SpLUDecompImpl*>(impl);
    else
        delete static_cast<const LUDecompImpl*>(impl);
}

std::vector<double> LUDecomp::solve(const std::vector<double>& rhs) const
{
    if(!impl)
        throw std::runtime_error("LUDecomp not initialized");
    if(sparse)
        return toStdVector(static_cast<const SpLUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
    else
        return toStdVector(static_cast<const LUDecompImpl*>(impl)->solve(toEigenVector(rhs)));
}


/// Cholesky decomposition for dense matrices

typedef Eigen::LLT< Type< Matrix<double> >::T, Eigen::Lower> CholeskyDecompImpl;

CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(new CholeskyDecompImpl(mat(M))) 
{
    if(static_cast<const CholeskyDecompImpl*>(impl)->info() != Eigen::Success)
        throw std::domain_error("CholeskyDecomp failed");
}

CholeskyDecomp::~CholeskyDecomp() { delete static_cast<const CholeskyDecompImpl*>(impl); }

CholeskyDecomp::CholeskyDecomp(const CholeskyDecomp& src) :
    impl(new CholeskyDecompImpl(*static_cast<const CholeskyDecompImpl*>(src.impl))) {}

Matrix<double> CholeskyDecomp::L() const { 
    if(!impl)
        throw std::runtime_error("CholeskyDecomp not initialized");
    unsigned int size = static_cast<const CholeskyDecompImpl*>(impl)->matrixLLT().rows();
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

/// Singular-value decomposition for dense matrices
#if EIGEN_VERSION_AT_LEAST(3,3,0)
typedef Eigen::BDCSVD< Type< Matrix<double> >::T> SVDecompImpl;
#else
typedef Eigen::JacobiSVD< Type< Matrix<double> >::T> SVDecompImpl;
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

// allocate memory if the size is positive, otherwise return NULL
template<typename NumT>
inline NumT* xnew(unsigned int size) {
    return size==0 ? NULL : new NumT[size]; }
    
//------ dense matrix ------//

// default constructor
template<typename NumT>
Matrix<NumT>::Matrix() :
    IMatrix<NumT>(), impl(NULL) {}

// constructor without data initialization
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols) :
    IMatrix<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols)) {}

// constructor with a given initial value
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols, const NumT value) :
    IMatrix<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nCols*nRows, value);
}

// copy constructor from a dense matrix
template<typename NumT>
Matrix<NumT>::Matrix(const Matrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(rows()*cols()))
{
    std::copy(
        /*src  begin*/ static_cast<const NumT*>(src.impl),
        /*src  end  */ static_cast<const NumT*>(src.impl) + rows()*cols(),
        /*dest begin*/ static_cast<NumT*>(impl));
}

// copy constructor from a sparse matrix
template<typename NumT>
Matrix<NumT>::Matrix(const SpMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()), impl(xnew<NumT>(rows()*cols()))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + rows()*cols(), 0);
    unsigned int numelem = src.size();
    for(unsigned int k=0; k<numelem; k++) {
        unsigned int i, j;
        NumT v = src.elem(k, i, j);
        static_cast<NumT*>(impl)[i*cols()+j] = v;
    }
}

// constructor from triplets
template<typename NumT>
Matrix<NumT>::Matrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nRows*nCols, 0);
    unsigned int numelem = values.size();
    for(unsigned int k=0; k<numelem; k++) {
        unsigned int i = values[k].i, j = values[k].j;
        if(i<nRows && j<nCols)
            static_cast<NumT*>(impl)[i*nCols+j] += static_cast<NumT>(values[k].v);
        else {
            delete[] static_cast<NumT*>(impl);
            throw std::range_error("Matrix: triplet index out of range");
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
const NumT& Matrix<NumT>::operator() (unsigned int row, unsigned int col) const
{
#ifdef DEBUG_RANGE_CHECK
    if(row >= rows() || col >= cols())
        throw std::range_error("Matrix: index out of range");
#endif
    return static_cast<const NumT*>(impl)[row * cols() + col];
}

/// access the matrix element for writing
template<typename NumT>
NumT& Matrix<NumT>::operator() (unsigned int row, unsigned int col)
{
#ifdef DEBUG_RANGE_CHECK
    if(row >= rows() || col >= cols())
        throw std::range_error("Matrix: index out of range");
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
SpMatrix<NumT>::SpMatrix() :
    IMatrix<NumT>(), impl(NULL) {}

// constructor from triplets
template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols), impl(NULL)
{
    if(nRows*nCols==0)
        return;  // no allocation in case of zero matrix
    unsigned int size = values.size();
    gsl_spmatrix* sp  = gsl_spmatrix_alloc_nzmax(nRows, nCols, size, GSL_SPMATRIX_TRIPLET);
    for(unsigned int k=0; k<size; k++)
        gsl_spmatrix_set(sp, values[k].i, values[k].j, values[k].v);
    impl = gsl_spmatrix_compcol(sp);
    gsl_spmatrix_free(sp);
}

// copy constructor from sparse matrix
template<typename NumT>
SpMatrix<NumT>::SpMatrix(const SpMatrix<NumT>& srcObj) :
    IMatrix<NumT>(srcObj.rows(), srcObj.cols()), impl(NULL)
{
    if(rows()*cols() == 0)
        return;
    const gsl_spmatrix* src = static_cast<const gsl_spmatrix*>(srcObj.impl);
    gsl_spmatrix* dest = gsl_spmatrix_alloc_nzmax(src->size1, src->size2, src->nz, GSL_SPMATRIX_CCS);
    gsl_spmatrix_memcpy(dest, src);
    impl = dest;
}

template<typename NumT>
SpMatrix<NumT>::~SpMatrix()
{
    if(impl)
        gsl_spmatrix_free(static_cast<gsl_spmatrix*>(impl));
}

template<typename NumT>
NumT SpMatrix<NumT>::operator() (unsigned int row, unsigned int col) const
{
    if(impl)
        return static_cast<NumT>(gsl_spmatrix_get(static_cast<const gsl_spmatrix*>(impl), row, col));
    else
        throw std::range_error("SpMatrix: empty matrix");
}

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    if(impl == NULL || index >= sp->nz)
        throw std::range_error("SpMatrix: index out of range");
    row = sp->i[index];
    col = binSearch(static_cast<size_t>(index), sp->p, sp->size2+1);
    return static_cast<NumT>(sp->data[index]);
}

template<typename NumT>
unsigned int SpMatrix<NumT>::size() const
{
    return impl==NULL ? 0 : static_cast<const gsl_spmatrix*>(impl)->nz; 
}

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    if(impl == NULL)
        return result;
    const gsl_spmatrix* sp = static_cast<const gsl_spmatrix*>(impl);
    result.reserve(sp->nz);
    unsigned int col = 0;
    for(unsigned int k=0; k<sp->nz; k++) {
        while(col<sp->size2 && sp->p[col+1]<=k)
            col++;
        result.push_back(Triplet(sp->i[k], col, sp->data[k]));
    }
    return result;
}

#else
// no GSL support for sparse matrices - implement them as dense matrices

template<typename NumT>
SpMatrix<NumT>::SpMatrix() :
    IMatrix<NumT>(), impl(NULL) {}
    
// constructor from triplets
template<typename NumT>
SpMatrix<NumT>::SpMatrix(unsigned int nRows, unsigned int nCols, const std::vector<Triplet>& values) :
    IMatrix<NumT>(nRows, nCols), impl(xnew<NumT>(nRows*nCols))
{
    std::fill(static_cast<NumT*>(impl), static_cast<NumT*>(impl) + nRows*nCols, 0);
    unsigned int numelem = values.size();
    for(unsigned int k=0; k<numelem; k++) {
        unsigned int i = values[k].i, j = values[k].j;
        if(i<nRows && j<nCols)
            static_cast<NumT*>(impl)[i*nCols+j] += static_cast<NumT>(values[k].v);
        else {
            delete[] static_cast<NumT*>(impl);
            throw std::range_error("SpMatrix: triplet index out of range");
        }
    }
}

// copy constructor from sparse matrix
template<typename NumT>
SpMatrix<NumT>::SpMatrix(const SpMatrix<NumT>& src) :
    IMatrix<NumT>(src.rows(), src.cols()),
    impl(xnew<NumT>(rows()*cols()))
{
    std::copy(
        /*src  begin*/ static_cast<const NumT*>(src.impl),
        /*src  end  */ static_cast<const NumT*>(src.impl) + rows()*cols(),
        /*dest begin*/ static_cast<NumT*>(impl));
}

template<typename NumT>
SpMatrix<NumT>::~SpMatrix()
{
    delete static_cast<NumT*>(impl);
}

template<typename NumT>
NumT SpMatrix<NumT>::operator() (unsigned int row, unsigned int col) const
{
    if(row >= rows() || col >= cols())
        throw std::range_error("SpMatrix: index out of range");
    return static_cast<const NumT*>(impl)[row * cols() + col];
}

template<typename NumT>
NumT SpMatrix<NumT>::elem(const unsigned int index, unsigned int &row, unsigned int &col) const
{
    row = index / cols();
    col = index % cols();
    return operator()(row, col);
}

template<typename NumT>
unsigned int SpMatrix<NumT>::size() const { return rows()*cols(); }

template<typename NumT>
std::vector<Triplet> SpMatrix<NumT>::values() const
{
    std::vector<Triplet> result;
    for(unsigned int k=0; k<cols() * rows(); k++)
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
    Mat(double* data, unsigned int rows, unsigned int cols) :
        m(gsl_matrix_view_array(data, rows, cols)) {}
    operator gsl_matrix* () { return &m.matrix; }
private:
    gsl_matrix_view m;
};

struct MatC {
    explicit MatC(const Matrix<double>& mat) :
        m(gsl_matrix_const_view_array(mat.data(), mat.rows(), mat.cols())) {}
    MatC(const double* data, unsigned int rows, unsigned int cols) :
        m(gsl_matrix_const_view_array(data, rows, cols)) {}
    operator const gsl_matrix* () const { return &m.matrix; }
private:
    gsl_matrix_const_view m;
};
} // internal namespace

template<> void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y)
{
    if(X.rows() != Y.rows() || X.cols() != Y.cols())
        throw std::invalid_argument("blas_daxpy: incompatible sizes of input arrays");
    if(alpha==0) return;
    unsigned int size = X.rows() * X.cols();
    const double* arrX = X.data();
    double* arrY = Y.data();
    for(unsigned int k=0; k<size; k++)
        arrY[k] += alpha*arrX[k];
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const Matrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, MatC(A), VecC(X), beta, Vec(Y));
}

template<> void blas_dgemv(CBLAS_TRANSPOSE TransA, double alpha, const SpMatrix<double>& A,
    const std::vector<double>& X, double beta, std::vector<double>& Y)
{
    if(A.impl == NULL)
        throw std::invalid_argument("blas_dgemv: empty matrix");
#ifdef HAVE_GSL_SPARSE
    gsl_spblas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha, static_cast<const gsl_spmatrix*>(A.impl),
        VecC(X), beta, Vec(Y));
#else
    gsl_blas_dgemv((CBLAS_TRANSPOSE_t)TransA, alpha,
        MatC(static_cast<const double*>(A.impl), A.rows(), A.cols()), VecC(X), beta, Vec(Y));
#endif
}

void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X)
{
    gsl_blas_dtrmv((CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, MatC(A), Vec(X));
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const Matrix<double>& A, const Matrix<double>& B, double beta, Matrix<double>& C)
{
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, 
        alpha, MatC(A), MatC(B), beta, Mat(C));
}

template<> void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const SpMatrix<double>& A, const SpMatrix<double>& B, double beta, SpMatrix<double>& C)
{
    if(A.impl == NULL || B.impl == NULL || C.impl == NULL)
        throw std::invalid_argument("blas_dgemv: empty matrix");
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
    gsl_spblas_dgemm(alpha, spA, spB, static_cast<gsl_spmatrix*>(C.impl));
    if(TransA != CblasNoTrans)
        gsl_spmatrix_free(trA);
    if(TransB != CblasNoTrans)
        gsl_spmatrix_free(trB);
#else
    gsl_blas_dgemm((CBLAS_TRANSPOSE_t)TransA, (CBLAS_TRANSPOSE_t)TransB, alpha,
        MatC(static_cast<const double*>(A.impl), A.rows(), A.cols()),
        MatC(static_cast<const double*>(B.impl), B.rows(), B.cols()),
        beta, Mat(static_cast <double*>(C.impl), C.rows(), C.cols()));
#endif
}

void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B)
{
    gsl_blas_dtrsm((CBLAS_SIDE_t)Side, (CBLAS_UPLO_t)Uplo, (CBLAS_TRANSPOSE_t)TransA, (CBLAS_DIAG_t)Diag, 
        alpha, MatC(A), Mat(B));
}

// ----- Linear algebra routines ----- //

/// LU decomposition implementation for GSL
struct LUDecompImpl {
    gsl_matrix* LU;
    gsl_permutation* perm;
    LUDecompImpl(const double* data, unsigned int rows, unsigned int cols) {
        LU = gsl_matrix_alloc(rows, cols);
        perm = gsl_permutation_alloc(rows);
        if(!LU || !perm) {
            gsl_permutation_free(perm);
            gsl_matrix_free(LU);
            throw std::bad_alloc();
        }
        int dummy;
        gsl_matrix_memcpy(LU, MatC(data, rows, cols));
        gsl_linalg_LU_decomp(LU, perm, &dummy);
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
LUDecomp::LUDecomp(const SpMatrix<double>& M) :
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
        throw std::invalid_argument("LUDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    gsl_linalg_LU_solve(static_cast<const LUDecompImpl*>(impl)->LU,
        static_cast<const LUDecompImpl*>(impl)->perm, VecC(rhs), Vec(result));
    return result;
}
    
/// Cholesky decomposition implementation for GSL
CholeskyDecomp::CholeskyDecomp(const Matrix<double>& M) :
    impl(NULL)
{
    gsl_matrix* L = gsl_matrix_alloc(M.rows(), M.cols());
    if(!L)
        throw std::bad_alloc();
    gsl_matrix_memcpy(L, MatC(M));
    try{
        gsl_linalg_cholesky_decomp(L);
    }
    catch(std::domain_error&) {
        gsl_matrix_free(L);
        throw std::domain_error("CholeskyDecomp failed");
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
        throw std::invalid_argument("CholeskyDecomp: incorrect size of RHS vector");
    std::vector<double> result(rhs.size());
    gsl_linalg_cholesky_solve(static_cast<const gsl_matrix*>(impl), VecC(rhs), Vec(result));
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
    sv->V=Matrix<double>(M.cols(), M.cols());
    sv->S.resize(M.cols());
    std::vector<double> temp(M.cols());
    if(M.rows() >= M.cols()*5) {   // use a modified algorithm for very 'elongated' matrices
        Matrix<double> tempmat(M.cols(), M.cols());
        gsl_linalg_SV_decomp_mod(Mat(sv->U), Mat(tempmat), Mat(sv->V), Vec(sv->S), Vec(temp));
    } else
        gsl_linalg_SV_decomp(Mat(sv->U), Mat(sv->V), Vec(sv->S), Vec(temp));
    // chop off excessively small singular values which may destabilize solution of linear system
    double minSV = sv->S[0] * 1e-15 * std::max<unsigned int>(sv->S.size(), 10);
    for(unsigned int k=0; k<sv->S.size(); k++)
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
        throw std::invalid_argument("SVDecomp: incorrect size of RHS vector");
    std::vector<double> result(sv->U.cols());
    gsl_linalg_SV_solve(MatC(sv->U), MatC(sv->V), VecC(sv->S), VecC(rhs), Vec(result));
    return result;
}

#endif

// template instantiations to be compiled (both for Eigen and GSL)
template struct SpMatrix<float>;
template struct SpMatrix<double>;
template struct Matrix<float>;
template struct Matrix<double>;
//template struct Matrix<int>;
//template struct Matrix<unsigned int>;
template void blas_daxpy(double alpha, const std::vector<double>& X, std::vector<double>& Y);
template void blas_daxpy(double alpha, const Matrix<double>& X, Matrix<double>& Y);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const Matrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemv(CBLAS_TRANSPOSE, double, const SpMatrix<double>&,
    const std::vector<double>&, double, std::vector<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const Matrix<double>&, const Matrix<double>&, double, Matrix<double>&);
template void blas_dgemm(CBLAS_TRANSPOSE, CBLAS_TRANSPOSE,
    double, const SpMatrix<double>&, const SpMatrix<double>&, double, SpMatrix<double>&);

}  // namespace
