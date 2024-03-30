/** \file   math_linalg.h
    \brief  linear algebra routines
    \date   2015-2024
    \author Eugene Vasiliev

    This module defines the matrix class and the interface for linear algebra routines.
    The actual implementation is provided either by GSL or by Eigen.
    Eigen is a highly-optimized linear algebra library that offers excellent performance
    in run-time at the expense of a significantly longer compilation time
    (since it a header-only library entirely based on esoteric template constructions).
    GSL implementations of the same routines are typically several times slower.
    The switch of back-end is transparent for the rest of the code outside this module.

    Matrices may be defined over any numerical type, although all linear algebra
    routines operate only on matrices of double, and can be either dense or sparse.
    In dense matrices, data is stored in a flattened 1d array in row-major order,
    i.e., a row is a contiguous block in memory. Strides are not supported.
    Sparse matrices provide only a limited functionality -- construction, read-only element
    access, and a couple of most important operations (matrix-vector and matrix-matrix
    multiplications, and solution of square linear systems by LU decomposition).
    Moreover, GSL prior to version 2 does not have sparse matrix support, thus they are
    implemented as dense matrices, which of course may render some operations entirely
    impractical (the use of GSL is not recommended anyway).
    Again the user interface is agnostic to the actual back-end.
    A special subtype of a sparse matrix is a band matrix, which is represented by
    a separate class and allows read/write access and fast solution of linear systems.
*/
#pragma once
#include <vector>
#include <cstddef>  // defines NULL, size_t and ptrdiff_t

namespace math{

/// \name Matrix-related classes
///@{

/// \throw std::out_of_range exception if the condition is not satisfied
void matrixRangeCheck(bool condition);

/// a triplet of numbers specifying an element in a sparse matrix
struct Triplet {
    size_t i;  ///< row
    size_t j;  ///< column
    double v;  ///< value
    Triplet() : i(0), j(0), v(0) {}
    Triplet(const int _i, const int _j, const double _v) : i(_i), j(_j), v(_v) {}
    int row() const { return i; }
    int col() const { return j; }
    double value() const { return v; }
};


/** An abstract read-only interface for a matrix.
    Dense, sparse and band matrices implement this interface and add more specific methods.
    It may also be implemented by a user-defined proxy class in the following context:
    suppose we need to provide the element-wise access to the matrix for some routine,
    but do not want to store the entire matrix in memory, or this matrix is composed of
    several concatenated matrices and we don't want to create a temporary copy of them.
    It can be used to loop over non-zero elements of the original matrix without
    the need for a full two-dimensional loop.
*/
template<typename NumT>
struct IMatrix {
    IMatrix(size_t _nRows=0, size_t _nCols=0) :
        nRows(_nRows), nCols(_nCols) {}

    virtual ~IMatrix() {}

    /// overall size of the matrix (number of possibly nonzero elements)
    virtual size_t size() const = 0;

    /// return an element from the matrix at the specified position
    virtual NumT at(const size_t row, const size_t col) const = 0;

    /// return an element at the overall `index` from the matrix (0 <= index < size),
    /// together with its separate row and column indices; used to loop over all nonzero elements
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const = 0;

    /// get the number of matrix rows
    inline size_t rows() const { return nRows; }

    /// get the number of matrix columns
    inline size_t cols() const { return nCols; }

    friend void swap(IMatrix<NumT>& first, IMatrix<NumT>& second)
    {
        using std::swap;
        swap(first.nRows, second.nRows);
        swap(first.nCols, second.nCols);
    }
private:
    size_t nRows;     ///< number of rows (first index)
    size_t nCols;     ///< number of columns (second index)
};


/** An abstract interface for a row-major dense-storage matrix.
    Indexing scheme is  `M.at(row, column) = M.data()[ row * M.cols() + column ]`,
    and the actual storage is provided by derived classes.
*/
template<typename NumT>
struct IMatrixDense: public IMatrix<NumT> {
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;

    IMatrixDense(size_t nRows=0, size_t nCols=0) :
        IMatrix<NumT>(nRows, nCols) {}

    /// overall size of the matrix (= total number of elements, all of them are possibly nonzero)
    virtual size_t size() const { return rows() * cols(); }

    /// return an element from the matrix at the specified position (row and column index)
    /// \throw std::out_of_range if row or col are larger than the respective matrix dimension
    virtual NumT at(const size_t row, const size_t col) const {
        matrixRangeCheck(row < rows() && col < cols());
        return data()[row * cols() + col];
    }

    /// return the element with the given overall index and store its row and column indices
    /// \throw std::out_of_range if the element index is larger than the total number of elements
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {
        row = index / cols();
        col = index % cols();
        matrixRangeCheck(row < rows());
        return data()[index];
    }

    /// access the raw data storage
    /// (data may be modified, but bound checks are responsibility of the calling code)
    virtual NumT* data() = 0;

    /// access the raw data storage (const overload)
    virtual const NumT* data() const = 0;
};


/** Band-diagonal square matrices.
    A band matrix with dimension NxN and bandwidth B has nonzero elements A(i,j) only when |i-j| <= B.
    A diagonal matrix has bandwidth zero. An example of a penta-diagonal matrix (B=2) of size 8x8:
    \code
    |10  5  2  0  0  0  0  0 |
    | 4 10  5  2  0  0  0  0 |
    | 1  4 10  5  2  0  0  0 |
    | 0  1  4 10  5  2  0  0 |
    | 0  0  1  4 10  5  2  0 |
    | 0  0  0  1  4 10  5  2 |
    | 0  0  0  0  1  4 10  5 |
    | 0  0  0  0  0  1  4 10 |
    \endcode
    These matrices are stored in a compact way and can be used to efficiently solve certain types
    of linear systems (such as those arising in spline construction).
*/
template<typename NumT>
struct BandMatrix: public IMatrix<NumT> {
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;

    /// default empty constructor
    BandMatrix() : band(0) {}

    /// create a square matrix of given size and bandwidth, initialized to the given value.
    /// \param[in]  size  is the size of matrix (number of rows and columns);
    /// \param[in]  bandwidth  is the number of elements on each side of the diagonal;
    /// \param[in]  value (optional)  is the initial value of all elements;
    /// \throw  std::invalid_argument if bandwidth>=size.
    BandMatrix(size_t size, size_t bandwidth, const NumT value=0);

    /// create a diagonal matrix from the vector of diagonal values
    explicit BandMatrix(const std::vector<NumT>& src) :
        IMatrix<NumT>(src.size(), src.size()), band(0), data(src) {}

    /// access the matrix element for reading
    /// \throw std::out_of_range if the element is outside the band
    const NumT& operator() (size_t row, size_t col) const;

    /// access the matrix element for writing
    /// \throw std::out_of_range if the element is outside the band
    NumT& operator() (size_t row, size_t col);

    /// return the number of nonzero elements
    virtual size_t size() const;

    /// return an element from the matrix at the specified position (or zero if it is outside the band)
    /// \throw std::out_of_range if the row or col index is larger than the matrix dimension
    virtual NumT at(size_t row, size_t col) const;

    /// return the given nonzero element and store its row and column indices
    /// \throw std::out_of_range if the index is larger than the total number of elements in the matrix
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const;

    /// return the bandwidth of the matrix (number of nonzero elements on each side
    /// of the diagonal at each row; a diagonal matrix has bandwidth 0)
    inline size_t bandwidth() const { return band; }

    /// return all non-zero elements in a single array of triplets (row, column, value)
    std::vector<Triplet> values() const;

private:
    size_t band;             ///< bandwidth
    std::vector<NumT> data;  ///< actual storage of matrix elements
};


/// Read-only sparse matrices
template<typename NumT>
struct SparseMatrix: public IMatrix<NumT> {
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;

    /// default (empty) constructor
    SparseMatrix();

    /// create a matrix of given size from a list of triplets (row,column,value);
    SparseMatrix(size_t nRows, size_t nCols,
        const std::vector<Triplet>& values = std::vector<Triplet>());

    /// copy constructor from a sparse matrix
    SparseMatrix(const SparseMatrix<NumT>& src);

    /// copy constructor from a band matrix
    explicit SparseMatrix(const BandMatrix<NumT>& src);

#if __cplusplus >= 201103L
    // move constructor in C++11
    SparseMatrix(SparseMatrix&& src) : SparseMatrix() {
        swap(*this, src);
    }
#endif

    /// assignment from a sparse matrix
    SparseMatrix& operator=(SparseMatrix<NumT> src) {
        swap(*this, src);
        return *this;
    }

    /// needed for standard containers and assignment operator
    friend void swap(SparseMatrix<NumT>& first, SparseMatrix<NumT>& second) {
        using std::swap;
        swap(static_cast<IMatrix<NumT>&>(first), static_cast<IMatrix<NumT>&>(second));
        swap(first.impl, second.impl);
    }

    virtual ~SparseMatrix();

    /// return all non-zero elements in a single array of triplets (row, column, value)
    std::vector<Triplet> values() const;

    /// return the number of nonzero elements
    virtual size_t size() const;

    /// return the given nonzero element and store its row and column indices
    /// \throw std::out_of_range if index exceeds the total number of nonzero elements
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const;

    /// return an element from the matrix at the specified position, or zero if it is not stored
    /// \throw std::out_of_range if row or col exceed the respective matrix dimension
    virtual NumT at(size_t row, size_t col) const;

    void* impl;  ///< opaque implementation details
};


/** Ordinary matrices with dense storage */
template<typename NumT>
struct Matrix: public IMatrixDense<NumT> {
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;

    /// default (empty) constructor
    Matrix();

    /// create a matrix of given size (values are not initialized!)
    Matrix(size_t nRows, size_t nCols);

    /// create a matrix of given size, initialized to the given value
    Matrix(size_t nRows, size_t nCols, const NumT value);

    /// copy constructor from a matrix of the same type
    Matrix(const Matrix<NumT>& src);

    /// copy constructor from a sparse matrix
    explicit Matrix(const SparseMatrix<NumT>& src);

    /// copy constructor from a band matrix
    explicit Matrix(const BandMatrix<NumT>& src);

    /// copy constructor from anything matrix-like (least specialized)
    explicit Matrix(const IMatrix<NumT>& src);

    /// create a matrix from a list of triplets (row,column,value)
    Matrix(size_t nRows, size_t nCols, const std::vector<Triplet>& values);

    /// assignment from the same type of matrix
    Matrix& operator=(Matrix<NumT> src) {
        swap(*this, src);
        return *this;
    }

#if __cplusplus >= 201103L
    // move constructor in C++11
    Matrix(Matrix<NumT>&& src) : Matrix() {
        swap(*this, src);
    }
#endif

    /// needed for standard containers and assignment operator
    friend void swap(Matrix<NumT>& first, Matrix<NumT>& second) {
        using std::swap;
        swap(static_cast<IMatrix<NumT>&>(first), static_cast<IMatrix<NumT>&>(second));
        swap(first.impl, second.impl);
    }

    virtual ~Matrix();

    /// access the matrix element for reading
    /// (range check is only performed if DEBUG_RANGE_CHECK macro is set)
    const NumT& operator() (size_t row, size_t col) const;

    /// access the matrix element for writing (same remark about range check)
    NumT& operator() (size_t row, size_t col);

    /// access the raw data storage (flattened 2d array in row-major order):
    /// indexing scheme is  `M(row, column) = M.data[ row * M.cols() + column ]`;
    /// bound checks are the responsibility of the calling code
    virtual NumT* data();

    /// access the raw data storage (const overload)
    virtual const NumT* data() const;

    void* impl;     ///< opaque implementation details
};


/** An adapter for representing an externally managed memory area as a dense row-major matrix */
template<typename NumT>
struct MatrixView: public IMatrixDense<NumT> {
    using IMatrix<NumT>::rows;
    using IMatrix<NumT>::cols;

    /// construct the matrix view on the externally provided data storage with the given dimensions:
    /// the pointer must remain valid for the lifetime of this object
    MatrixView(size_t nRows, size_t nCols, NumT* _storage) :
        IMatrixDense<NumT>(nRows, nCols), storage(_storage) {}

    /// return the pointer to the external data storage
    virtual NumT* data() { return storage; }

    /// same but const-overload
    virtual const NumT* data() const { return storage; }
private:
    NumT* storage;  ///< external data storage (neither created nor deallocated by this object)
};

/** Interface for a matrix transposition without creating a new matrix */
template<typename NumT>
class TransposedMatrix: public math::IMatrix<NumT> {
    const math::IMatrix<NumT>& mat;       ///< the original matrix
public:
    TransposedMatrix(const math::IMatrix<NumT>& src):
    math::IMatrix<NumT>(src.cols(), src.rows()), mat(src) {};
    virtual size_t size() const { return mat.size(); }
    virtual NumT at(const size_t row, const size_t col) const {
        return mat.at(col, row);
    }
    virtual NumT elem(const size_t index, size_t &row, size_t &col) const {
        return mat.elem(index, col, row);
    }
};

///@}
/// \name Utility routines
///@{

/** check whether all elements of an array are zeros (return true for an empty array as well) */
template<typename NumT>
bool allZeros(const std::vector<NumT>& vec)
{
    for(size_t i=0; i<vec.size(); i++)
        if(vec[i]!=0)
            return false;
    return true;
}

/** check if all elements of a matrix are zeros */
template<typename NumT>
bool allZeros(const Matrix<NumT>& mat)
{
    for(size_t i=0; i<mat.rows(); i++)
        for(size_t j=0; j<mat.cols(); j++)
            if(mat(i,j) != 0)
                return false;
    return true;
}

/** check if all elements of a matrix accessed through IMatrix interface are zeros */
template<typename NumT>
bool allZeros(const IMatrix<NumT>& mat)
{
    for(size_t k=0; k<mat.size(); k++) {
        size_t i,j;
        if(mat.elem(k, i, j) != 0)
            return false;
    }
    return true;
}

/** zero out array elements with magnitude smaller than the threshold
    times the maximum element of the array;
*/
void eliminateNearZeros(std::vector<double>& vec, double threshold=2e-15);
void eliminateNearZeros(     Matrix<double>& mat, double threshold=2e-15);

///@}
/// \name  a subset of BLAS routines
///@{

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

/// dot product of two vectors
double blas_ddot(const std::vector<double>& X, const std::vector<double>& Y);

/// L2-norm of a vector or a matrix (sum of squares of all elements)
/// \tparam Type may be std::vector<double> or Matrix<double>
template<typename Type>
double blas_dnrm2(const Type& X);

/// sum of two vectors or two matrices:  Y := alpha * X + Y
/// \tparam Type may be std::vector<double>, Matrix<double>, SparseMatrix<double> or BandMatrix<double>
template<typename Type>
void blas_daxpy(double alpha, const Type& X, Type& Y);

/// multiply vector or matrix by a number:  Y := alpha * Y
/// \tparam Type may be std::vector<double>, Matrix<double>, SparseMatrix<double> or BandMatrix<double>
template<typename Type>
void blas_dmul(double alpha, Type& Y);

/// matrix-vector multiplication:  Y := alpha * A * X + beta * Y
/// \tparam MatrixType is Matrix<double>, SparseMatrix<double> or BandMatrix<double>
template<typename MatrixType>
void blas_dgemv(CBLAS_TRANSPOSE TransA,
    double alpha, const MatrixType& A, const std::vector<double>& X, double beta,
    std::vector<double>& Y);

/// matrix-vector multiplication for a triangular matrix A:  X := A * X
void blas_dtrmv(CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    const Matrix<double>& A, std::vector<double>& X);

/// matrix product:  C := alpha * A * B + beta * C
/// \tparam MatrixType is either Matrix<double> or SparseMatrix<double>
template<typename MatrixType>
void blas_dgemm(CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
    double alpha, const MatrixType& A, const MatrixType& B, double beta, MatrixType& C);

/// matrix product for a triangular matrix A:
/// B := alpha * A^{-1} * B  (if Side=Left)  or  alpha * B * A^{-1}  (if Side=Right)
void blas_dtrsm(CBLAS_SIDE Side, CBLAS_UPLO Uplo, CBLAS_TRANSPOSE TransA, CBLAS_DIAG Diag,
    double alpha, const Matrix<double>& A, Matrix<double>& B);

///@}
/// \name  Linear algebra routines
///@{

/** LU decomposition of a generic square positive-definite matrix M into lower and upper
    triangular matrices:  once created, it may be used to solve a linear system `M x = rhs`
    multiple times with different rhs.
*/
class LUDecomp {
    bool sparse; ///< flag specifying whether this is a dense or sparse matrix
    void* impl;  ///< opaque implementation details
public:
    // default constructor and other boilerplate definitions
    LUDecomp() : sparse(false), impl(NULL) {}
    LUDecomp(const LUDecomp& src);
    LUDecomp& operator=(LUDecomp src) {
        swap(*this, src);
        return *this;
    }
    friend void swap(LUDecomp& first, LUDecomp& second) {
        using std::swap;
        swap(first.impl,   second.impl);
        swap(first.sparse, second.sparse);
    }
    ~LUDecomp();

    /// Construct a decomposition for the given dense matrix M
    explicit LUDecomp(const Matrix<double>& M);

    /// Construct a decomposition for the given sparse matrix M
    explicit LUDecomp(const SparseMatrix<double>& M);

    /// Solve the matrix equation `M x = rhs` for x, using the LU decomposition of matrix M
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


/** Cholesky decomposition of a symmetric positive-definite square matrix M
    into a product of L L^T, where L is a lower triangular matrix.
    Once constructed, it be used for solving a linear system `M x = rhs` multiple times with
    different rhs; the cost of construction is roughly twice lower than LUDecomp.
*/
class CholeskyDecomp {
    void* impl;  ///< opaque implementation details
public:
    CholeskyDecomp() : impl(NULL) {}
    CholeskyDecomp(const CholeskyDecomp& src);
    CholeskyDecomp& operator=(CholeskyDecomp src) {
        swap(*this, src);
        return *this;
    }
    friend void swap(CholeskyDecomp& first, CholeskyDecomp& second) {
        using std::swap;
        swap(first.impl, second.impl);
    }
    ~CholeskyDecomp();

    /// Construct a decomposition for the given dense matrix M
    /// (only the lower triangular part of M is used).
    /// \throw std::domain_error if M is not positive-definite
    explicit CholeskyDecomp(const Matrix<double>& M);

    /// return the lower triangular matrix L of the decomposition
    Matrix<double> L() const;

    /// Solve the matrix equation `M x = rhs` for x, using the Cholesky decomposition of matrix M
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


class QRDecomp {
    void* impl;  ///< opaque implementation details
public:
    QRDecomp() : impl(NULL) {}
    QRDecomp(const QRDecomp& src);
    QRDecomp& operator=(QRDecomp src) {
        swap(*this, src);
        return *this;
    }
    friend void swap(QRDecomp& first, QRDecomp& second) {
        using std::swap;
        swap(first.impl, second.impl);
    }
    ~QRDecomp();

    /// Construct a decomposition for the given matrix M
    explicit QRDecomp(const Matrix<double>& M);

    /// retrieve the orthogonal matrix Q and the upper triangular matrix R from the decomposition
    void QR(Matrix<double>& Q, Matrix<double>& R) const;

    /// Solve the matrix equation `M x = rhs` for x, using the QR of matrix M:
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


/** Singular value decomposition of an arbitrary M-by-N matrix A (M>=N) into a product  U S V^T,
    where U is an orthogonal M-by-N matrix, S is a diagonal N-by-N matrix of singular values,
    and V is an orthogonal N-by-N matrix.
    It can be used to solve linear systems in the same way as LUDecomp or CholeskyDecomp,
    but is applicable also in the case of non-full-rank systems, both when the number of variables
    is larger than the number of equations (in this case it produces a least-square solution),
    and when the nullspace of the system is nontrivial (in this case the solution with the lowest
    norm out of all possible ones is returned).
    It provides both left and right singular vectors and corresponding singular values;
    for a symmetric positive-definite matrix, SVD is equivalent to eigendecomposition (i.e. U=V).
    SVD is also considerably slower than other decompositions.
*/
class SVDecomp {
    void* impl;  ///< opaque implementation details
public:
    SVDecomp() : impl(NULL) {}
    SVDecomp(const SVDecomp& src);
    SVDecomp& operator=(SVDecomp src) {
        swap(*this, src);
        return *this;
    }
    friend void swap(SVDecomp& first, SVDecomp& second) {
        using std::swap;
        swap(first.impl, second.impl);
    }
    ~SVDecomp();

    /// Construct a decomposition for the given matrix M
    explicit SVDecomp(const Matrix<double>& M);

    /// return the orthogonal matrix U that forms part of the decomposition
    Matrix<double> U() const;

    /// return the orthogonal matrix V that forms part of the decomposition
    Matrix<double> V() const;

    /// return the vector of singular values sorted in decreasing order
    std::vector<double> S() const;

    /// Solve the matrix equation `M x = rhs` for x, using the SVD of matrix M:
    /// if the system is overdetermined, the solution is obtained in the least-square sense;
    /// if the system is underdetermined, the solution with the smallest norm is returned.
    std::vector<double> solve(const std::vector<double>& rhs) const;
};


/** Solve a sparse linear system  A x = rhs, where A is a square band-diagonal matrix.
    The matrix A must be strongly non-degenerate and diagonally dominant (which is usually the case,
    e.g. when such matrix is constructed in the finite-difference context such as spline approximation);
    in other words, we do not use pivoting in solving the linear system and always take the diagonal
    element without checking (no error is reported if it turns out to be near-zero).
    \param[in] bandMatrix  is the square NxN matrix of the linear system.
    \param[in] rhs  is the right-hand side of the linear system, must have length N.
    \return  the solution vector `x` of length N.
    \throw  std::invalid_argument  if the matrix size is incorrect.
*/
std::vector<double> solveBand(const BandMatrix<double>& bandMatrix, const std::vector<double>& rhs);


///@}

}  // namespace
