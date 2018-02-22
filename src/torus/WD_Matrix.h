/**
\file WD_Matrix.h
\brief Contains templated class Matrix.
Useful for working with matrices if size is known in advance. 
Various operations (e.g. Gaussian inversion) made convenient.

*                                                                              *
* Matrix.h                                                                     *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics                *
* address: 1 Keble Road, Oxford OX1 3NP, United Kingdom                        *
* e-mail : dehnen@thphys.ox.ac.uk                                              *
*                                                                              *
*******************************************************************************/

#ifndef Matrix_h
#define Matrix_h 1

#include "WD_Vector.h"

using std::cerr;
namespace WD{

template<class T, int N1, int N2>
class Matrix {

protected:
    T a[N1*N2];
    static void range_error();
    static void division_by_zero_error();

public:
    static void error(const char*);

    Matrix() {}
    Matrix(const T);
    Matrix(const Matrix& M) { for(int i=0;i<N1*N2;i++) a[i]=M.a[i];}
   ~Matrix() {}

    Matrix&      operator=            (const Matrix&);
    Matrix&      operator+=           (const Matrix&);
    Matrix&      operator-=           (const Matrix&);
    Matrix&      operator=            (const T);
    Matrix&      operator=            (const T**);
    Matrix&      operator+=           (const T);
    Matrix&      operator-=           (const T);
    Matrix&      operator*=           (const T);
    Matrix&      operator/=           (const T);

    Matrix       operator-            () const;
    Matrix       operator-            (const Matrix&) const;
    Matrix       operator+            (const Matrix&) const;
    int          operator==           (const Matrix&) const;
    int          operator!=           (const Matrix&) const;
    Matrix       operator+            (const T) const;
    Matrix       operator-            (const T) const;
    Matrix       operator*            (const T) const;
    Matrix       operator/            (const T) const;

    T*           operator[]           (const int n1) { return a+n1*N2; }
    T const*     operator[]           (const int n1) const { return a+n1*N2; }
    T            operator()           (const int n1, const int n2) const
                                          { return a[n1*N2+n2];}

    Vector<T,N1> column               (const int) const;
    void         fill_column          (const T, const int);
    void         multiply_column      (const T, const int);
    void         set_column           (const Vector<T,N1>&, const int);
    void         set_column           (const T*, const int);
    void         add_to_column        (const Vector<T,N1>&, const int);
    void         add_to_column        (const T*, const int);
    void         subtract_from_column (const Vector<T,N1>&, const int);
    void         subtract_from_column (const T*, const int);

    Vector<T,N2> row                  (const int) const;
    void         fill_row             (const T, const int);
    void         multiply_row         (const T, const int);
    void         set_row              (const Vector<T,N2>&, const int);
    void         set_row              (const T*, const int);
    void         add_to_row           (const Vector<T,N2>&, const int);
    void         add_to_row           (const T*, const int);
    void         subtract_from_row    (const Vector<T,N2>&, const int);
    void         subtract_from_row    (const T*, const int);
};


template<class T, int N1, int N2>
inline Matrix<T,N1,N2> operator+ (const T x, const Matrix<T,N1,N2>& M)
    { return M+x; }
template<class T, int N1, int N2>
inline Matrix<T,N1,N2> operator- (const T x, const Matrix<T,N1,N2>& M)
    { Matrix<T,N1,N2> N(x); return N-=M; }
template<class T, int N1, int N2>
inline Matrix<T,N1,N2> operator* (const T x, const Matrix<T,N1,N2>& M)
    { return M*x; }

////////////////////////////////////////////////////////////////////////////////
// class Matrix
////////////////////////////////////////////////////////////////////////////////
// protected member functions:

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::error(const char* msgs)
{ 
    cerr << " Matrix ERROR: "<<msgs<<'\n';
#ifndef ebug
    exit(1);
#endif
}

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::range_error()
{   error("out of range"); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::division_by_zero_error()
{   error("division by zero"); }

////////////////////////////////////////////////////////////////////////////////
// public member functions:
// (A) Constructors

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>::Matrix(const T fill_value)
    { for(int i=0; i<N1*N2; i++) a[i] = fill_value; }

// (B) arithmetic operators with assign: =, +=, -=, *=, /=

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator= (const Matrix<T,N1,N2>& M)
    { for(int i=0; i<N1*N2; i++) a[i] = M.a[i];
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator+= (const Matrix<T,N1,N2>& M)
    { for(int i=0; i<N1*N2; i++) a[i] += M.a[i];
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator-= (const Matrix<T,N1,N2>& M)
    { for(int i=0; i<N1*N2; i++) a[i] -= M.a[i];
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator= (const T fill_value)
    { for(int i=0; i<N1*N2; i++) a[i] = fill_value;
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator= (const T** array)
    { int i,k,l;
      for(k=i=0; k<N1; k++)
      for(l=0  ; l<N2; l++,i++) a[i] = array[k][l];
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator+= (const T m)
    { for(int i=0; i<N1*N2; i++) a[i] += m;
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator-= (const T m)
    { for(int i=0; i<N1*N2; i++) a[i] -= m;
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator*= (const T m)
    { for(int i=0; i<N1*N2; i++) a[i] *= m;
      return *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2>& Matrix<T,N1,N2>::operator/= (const T m)
    { if(m==0) division_by_zero_error();
      for(int i=0; i<N1*N2; i++) a[i] /= m;
      return *this; }

// (C) further arithmetic operators: all constant member functions

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator- () const
    { Matrix<T,N1,N2> P(0); return P-= *this; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator- (const Matrix<T,N1,N2>& M) const
    { Matrix<T,N1,N2> P(*this); return P-=M; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator+ (const Matrix<T,N1,N2>& M) const
    { Matrix<T,N1,N2> P(*this); return P+=M; }

template<class T, int N1, int N2>
inline int Matrix<T,N1,N2>::operator== (const Matrix<T,N1,N2>& M) const
    { for(int i=0; i<N1*N2; i++) if(a[i] != M.a[i]) return 0;
      return 1; }

template<class T, int N1, int N2>
inline int Matrix<T,N1,N2>::operator!= (const Matrix<T,N1,N2>& M) const
    { for(int i=0; i<N1*N2; i++) if(a[i] != M.a[i]) return 1;
      return 0; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator+ (const T x) const
    { Matrix<T,N1,N2> P(*this); return P+=x; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator- (const T x) const
    { Matrix<T,N1,N2> P(*this); return P-=x; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator* (const T x) const
    { Matrix<T,N1,N2> P(*this); return P*=x; }

template<class T, int N1, int N2>
inline Matrix<T,N1,N2> Matrix<T,N1,N2>::operator/ (const T x) const
    { Matrix<T,N1,N2> P(*this); return P/=x; }

// (D) column and row manipulations

template<class T, int N1, int N2>
inline Vector<T,N1> Matrix<T,N1,N2>::column(const int n2) const
    { Vector<T,N1> P; int i,j;
      for(j=n2,i=0; i<N1; i++,j+=N2) P[i] = *(a+j);
      return P; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::fill_column(const T fill_value, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      for(T*b=a+n2; b<a+N1*N2; b+=N2) *b = fill_value; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::multiply_column(const T multiplier, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      for(T*b=a+n2; b<a+N1*N2; b+=N2) *b *= multiplier; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::set_column(const Vector<T,N1>& V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b = V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::set_column(const T* V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b = V[i]; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::add_to_column(const Vector<T,N1>& V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b += V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::add_to_column(const T* V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b += V[i]; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::subtract_from_column(const Vector<T,N1>& V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b -= V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::subtract_from_column(const T* V, const int n2)
    { if(n2>=N2 || n2<0) range_error();
      T*b; int i;
      for((b=a+n2,i=0); i<N1; (i++,b+=N2)) *b -= V[i]; }

template<class T, int N1, int N2>
inline Vector<T,N2> Matrix<T,N1,N2>::row(const int n1) const
    { Vector<T,N2> P; int i, j;
      for(j=n1*N2,i=0; i<N2; i++,j++) P[i] = *(a+j);
      return P; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::fill_row(const T fill_value, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      for(T*b=a+n1*N2; b<a+(n1+1)*N2; b++) *b = fill_value; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::multiply_row(const T multiplier, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      for(T*b=a+n1*N2; b<a+(n1+1)*N2; b++) *b *= multiplier; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::set_row(const Vector<T,N2>& V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b = V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::set_row(const T* V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b = V[i]; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::add_to_row(const Vector<T,N2>& V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b += V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::add_to_row(const T* V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b += V[i]; }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::subtract_from_row(const Vector<T,N2>& V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b -= V(i); }

template<class T, int N1, int N2>
inline void Matrix<T,N1,N2>::subtract_from_row(const T* V, const int n1)
    { if(n1>=N1 || n1<0) range_error();
      T*b; int i;
      for((b=a+n1*N2,i=0); i<N2; (i++,b++)) *b -= V[i]; }


////////////////////////////////////////////////////////////////////////////////
// related functions/routines

template<class T, int N>
inline void set_to_unity(Matrix<T,N,N>& M)
{
    const T zero=0, one=1;
    M = zero;
    for(int i=0; i<N; i++) M[i][i] = one;
}

template<class T, int N>
inline Matrix<T,N,N> unity()
{
    const T zero=0, one=1;
    Matrix<T,N,N> M = zero;
    for(int i=0; i<N; i++) M[i][i] = one;
    return M;
}

template<class T, int N>
inline T trace(const Matrix<T,N,N>& M)
{ 
    T t = T(0);
    for(int i=0; i<N; i++) t += M(i,i); 
    return t;
}

template<class T, int L, int M, int N>
inline Matrix<T,L,N> operator* (const Matrix<T,L,M>& A, const Matrix<T,M,N>& B)
{
    int l,m,n;
    Matrix<T,L,N> C=0.;
    for(l=0; l<L; l++)
        for(n=0; n<N; n++)
            for(m=0; m<M; m++)
                C[l][n] += A(l,m) * B(m,n);
    return C;
}

template<class T, int L, int M, int N>
inline void multiply(Matrix<T,L,M>& A, Matrix<T,M,N>& B, Matrix<T,L,N>& C)
{
    T y,*pa=A[0],*pai,*pb,*pbi, *pc=C[0];
    int l,m,n;
    for(l=0; l<L; l++,pa+=M)
        for(pb=B[0],n=0; n<N; n++,pb++,pc++) {
            for(y=T(0),pai=pa,pbi=pb,m=0; m<M; m++,pai++,pbi+=N)
                y += *pai * *pbi;
            *pc = y;
        }
}

template<class T, int L, int M, int N>
inline void multiplyZ(Matrix<T,L,M>& A, Matrix<T,M,N>& B, Matrix<T,L,N>& C)
// on return C = A*B
{
    T y,*pa=A[0],*pai,*pb,*pbi, *pc=C[0];
    int l,m,n;
    for(l=0; l<L; l++,pa+=M)
        for(pb=B[0],n=0; n<N; n++,pb++,pc++) {
            for(y=T(0),pai=pa,pbi=pb,m=0; m<M; m++,pai++,pbi+=N)
                if((*pai) && (*pbi)) y += *pai * *pbi;
            *pc = y;
        }
}

template<class T, int N1, int N2>
inline void multiplyZ(Matrix<T,N1,N2>& A, const T x, Matrix<T,N1,N2>& C)
// on return C = A*x
{
    T *pa=A[0],*pc=C[0],*aup=pa+N1*N2;
    for(; pa<aup; pa++,pc++)
        if(*pa) *pc = *pa * x;
}

template<class T, int N1, int N2>
inline void apl_ml(Matrix<T,N1,N2>& A, const T x, Matrix<T,N1,N2>& C)
// on return C += A*x
{
    T *pa=A[0],*pc=C[0],*aup=pa+N1*N2;
    for(; pa<aup; pa++,pc++)
        if(*pa) *pc += *pa * x;
}

template<class T, int L, int M, int N>
inline void as_ml_ml(Matrix<T,L,M>& A, Matrix<T,M,N>& B, const T x, Matrix<T,L,N>& C)
// on return C = A*B*x
{
    T y,*pa=A[0],*pai,*pb,*pbi, *pc=C[0];
    int l,m,n;
    for(l=0; l<L; l++,pa+=M)
        for(pb=B[0],n=0; n<N; n++,pb++,pc++) {
            for(y=T(0),pai=pa,pbi=pb,m=0; m<M; m++,pai++,pbi+=N)
                if((*pai) && (*pbi)) y += *pai * *pbi;
            *pc = y * x;
        }
}

template<class T, int L, int M>
inline Vector<T,L> operator* (const Matrix<T,L,M>& A, const Vector<T,M>& B)
{
    int l,m;
             Vector<T,L> C=0.;
    for(l=0; l<L; l++)
        for(m=0; m<M; m++)
            C[l] += A(l,m) * B(m);
    return C;
}

template<class T, int L, int M>
inline Vector<T,M> operator* (const Vector<T,L>& B, const Matrix<T,L,M>& A) 
{
    int l,m;
             Vector<T,M> C=0.;
    for(m=0; m<M; m++)
        for(l=0; l<L; l++)
            C[m] += B(l) * A(l,m);
    return C;
}

template<class T, int N>
inline Matrix<T,N,N> operator% (const Vector<T,N>& a, const Vector<T,N>& b)
// outer product of two vectors
{
    int i;
    Matrix<T,N,N>  M;
    for(i=0; i<N; i++) M.set_row(a(i)*b,i);
    return M;
}

template<class T, int N>
inline void GaussInvert(Matrix<T,N,N>& A)
{
    int    i, icol=0, irow=0, j, k, l, ll;
    T      big, dum, pivinv, One=T(1), Zero=T(0);
    Vector<int,N>   ipiv=0,indxr,indxc;
    for(i=0; i<N; i++) {
        big = Zero;
        for(j=0; j<N; j++) {
            if(ipiv(j)!=1) for(k=0; k<N; k++) {
                if(ipiv(k)==0) {
                    if(WDabs(A(j,k)) >= big) {
                        big  = WDabs(A(j,k));
                        irow = j;
                        icol = k;
                    }
                } else if(ipiv(k)>1) A.error(" Matrix to invert is singular");
            }
            ++(ipiv[icol]);
            if(irow != icol)
                for(l=0; l<N; l++) WDswap(A[irow][l], A[icol][l]);
        }
        indxr[i] = irow;
        indxc[i] = icol;
        if(A(icol,icol)==Zero) A.error(" Matrix to invert is singular");
        pivinv = One/A(icol,icol);
        A[icol][icol] = One;
        for(l=0; l<N; l++) A[icol][l] *= pivinv;
        for(ll=0; ll<N; ll++)
            if(ll != icol) {
                dum         = A(ll,icol);
                A[ll][icol] = Zero;
                for(l=0; l<N; l++) A[ll][l] -= A(icol,l) * dum;
            }
    }
    for(l=N-1; l>=0; l--)
        if (indxr(l) != indxc(l) )
            for(k=0; k<N; k++) WDswap(A[k][indxr(l)], A[k][indxc(l)]);
}

template<class T, int N>
inline void invert(const Matrix<T,N,N>& A, Matrix<T,N,N>& A1)
{
    A1 = A;
    GaussInvert(A1);
}

template<class T, int N>
inline Matrix<T,N,N> inverse(const Matrix<T,N,N>& A)
{
    Matrix<T,N,N> A1 = A;
    GaussInvert(A1);
    return A1;
}


template<class T, int N, int M>
inline Matrix<T,N,M> operator! (const Matrix<T,M,N>& A)
{
    Matrix<T,N,M> At;
    int i,j;
    for(i=0; i<N; i++)
        for(j=0; j<M; j++)
            At[i][j] = A(j,i);
    return At;
}
} // namespace
#endif
