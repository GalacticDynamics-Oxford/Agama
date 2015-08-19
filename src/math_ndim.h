/** \file    math_ndim.h
    \brief   Prototype of an N-dimensional function, and the matrix class
    \date    2015
    \author  Eugene Vasiliev
*/
#pragma once
#include <vector>

namespace math{

/** a simple class for two-dimensional matrices */
template<typename NumT>
class Matrix {
public:
    /// create an empty matrix
    Matrix() : nRows(0), nCols(0) {};

    /// create a matrix of given size
    Matrix(unsigned int _nRows, unsigned int _nCols) :
        nRows(_nRows), nCols(_nCols), data(nRows*nCols) {};

    /// create a matrix of given size and fill it with a value
    Matrix(unsigned int _nRows, unsigned int _nCols, double val) :
        nRows(_nRows), nCols(_nCols), data(nRows*nCols, val) {};

    /// resize an existing matrix
    void resize(unsigned int newRows, unsigned int newCols) {
        nRows = newRows;
        nCols = newCols;
        data.resize(nRows*nCols);
    }

    /// access the matrix element for reading (bound checks are performed by the underlying vector)
    const NumT& operator() (unsigned int row, unsigned int column) const {
        return data[row*nCols+column]; }

    /// access the matrix element for writing
    NumT& operator() (unsigned int row, unsigned int column) {
        return data[row*nCols+column]; }

    /// number of matrix rows
    unsigned int numRows() const { return nRows; }

    /// number of matrix columns
    unsigned int numCols() const { return nCols; }

    /// access raw data for reading (2d array in row-major order)
    const NumT* getData() const { return &data.front(); }

    /// access raw data for writing (2d array in row-major order)
    NumT* getData() { return &data.front(); }
private:
    unsigned int nRows;      ///< number of rows (first index)
    unsigned int nCols;      ///< number of columns (second index)
    std::vector<NumT> data;  ///< flattened data storage
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values. */
class IFunctionNdim {
public:
    IFunctionNdim() {};
    virtual ~IFunctionNdim() {};

    /** evaluate the function.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
                    the vector of function values.
                    Should point to an existing array of length at least M.
    */
    virtual void eval(const double vars[], double values[]) const = 0;

    /// return the dimensionality of the input point (N)
    virtual unsigned int numVars() const = 0;

    /// return the number of elements in the output array of values (M)
    virtual unsigned int numValues() const = 0;
};

/** Prototype of a function of N>=1 variables that computes a vector of M>=1 values,
    and derivatives of these values w.r.t.the input variables (aka jacobian). */
class IFunctionNdimDeriv: public IFunctionNdim {
public:
    IFunctionNdimDeriv() {};
    virtual ~IFunctionNdimDeriv() {};

    /** evaluate the function and the derivatives.
        \param[in]  vars   is the N-dimensional point at which the function should be computed.
        \param[out] values is the M-dimensional array (possibly M=1) that will contain
                    the vector of function values.
        \param[out] derivs is the M-by-N matrix (M rows, N columns) of partial derivatives 
                    of the vector-valued function by the input variables;
                    if a NULL pointer is passed, this does not need to be computed,
                    otherwise the shape of matrix will be resized as needed
                    (i.e. one may pass a pointer to an empty matrix and it will be resized).
    */
    virtual void evalDeriv(const double vars[], double values[], Matrix<double>* derivs=0) const = 0;

    /** reimplement the evaluate function without derivatives */
    virtual void eval(const double vars[], double values[]) const {
        evalDeriv(vars, values, NULL);
    }
};

}  // namespace