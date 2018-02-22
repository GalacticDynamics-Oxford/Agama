/**
\file WD_Numerics.h
\brief Various useful mathsy functions

*                                                                              *
* Numerics.h                                                                   *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1Keble Road, Oxford, OX1 3NP, United Kingdom.                       *
* e-mail:  dehnen@thphys.ox.ac.uk                                              *
*                                                                              *
*******************************************************************************/

#ifndef _Numerics_def_
#define _Numerics_def_ 1

#include "WD_Vector.h"
#include "WD_Matrix.h"
#include "WD_Numerics.templates"

namespace WD{
////////////////////////////////////////////////////////////////////////////////
// here only the non-inline non-template functions are listed: /////////////////
////////////////////////////////////////////////////////////////////////////////

int GaussJordan(double**,const int,double**,const int);
int GaussJordan(float**, const int,float**, const int);
int GaussJordan(double**,const int,double*);
int GaussJordan(float**, const int,float*);

int GaussBack(double**,const int,double*);
int GaussBack(float**, const int,float*);

int  CholeskyDecomposition(float**, const int);
void CholeskySolution(const float**, const int, float*);
void CholeskyInvertL(float**, const int);
void CholeskyInvertF(float**, const int);
inline void CholeskyInvert(float** a, const int n)
        { CholeskyInvertL(a,n); CholeskyInvertF(a,n); }

int  CholeskyDecomposition(double**, const int);
void CholeskySolution(const double**, const int, double*);
void CholeskyInvertL(double**, const int);
void CholeskyInvertF(double**, const int);
inline void CholeskyInvert(double** a, const int n)
        { CholeskyInvertL(a,n); CholeskyInvertF(a,n); }

int  LUDecomposition(float**, const int, int*, int&);
void LUSolution(const float**, const int, const int*, float*);
void LUInvert(const float**, float**, const int, const int*);

int  LUDecomposition(double**, const int, int*, int&);
void LUSolution(double**, const int, const int*, double*);
void LUInvert(double**, double**, const int, const int*);

void tred2(double**, const int, double*, double*, const char=1);
void tqli(double*, double*, const int, double**, const char=1);
inline void EigenSym(double** A, const int n, double* lam, const char EV=1)
{
    double* e = new double[n];
    tred2(A, n, lam, e, EV);
    tqli (lam, e, n, A, EV);
    delete[] e;
}
template<class FLOAT, int N>
inline void EigenSym(Matrix<FLOAT,N,N>& A, Vector<FLOAT,N>& lam,
                     const char EV=1)
{
    Vector<FLOAT,N> e;
    Tred2(A,lam,e,EV);
    Tqli (lam,e,A,EV);
}

void balanc(double**, const int);
void elmhes(double**, const int);
void hqr(double**, const int, double*, double*);
inline void EigenReal(double** A, const int n, double* lr, double* li)
{
    balanc(A,n);
    elmhes(A,n);
    hqr(A,n,lr,li);
}
inline void EigenNon(double** A, const int n, double* lr, double* li)
{ EigenReal(A,n,lr,li); }

#ifdef __COMPLEX__
template<class FLOAT, int N>
inline void EigenReal(const Matrix<FLOAT,N,N>& A, Vector<complex<FLOAT>,N>& W)
{
    Matrix<FLOAT,N,N> B(A);
    Balance     (B);
    Hessenberg  (B);
    QRHessenberg(B,W);
}
#endif

void GaussLegendre(double*,double*,const int);
void LegendrePeven(double*,const double,const int);
void dLegendrePeven(double*,double*,const double,const int);

double qbulir(double(*)(double),const double,const double,const double,double&);
inline double qbulir(double(*func)(double),const double a,const double b,
                     const double eps)
{ 
    double err; 
    return qbulir(func,a,b,eps,err);
}

double qsplin(double*, double*, double*, const int, const double, const double,
              const double);
float qsplin(float*, float*, float*, const int, const float, const float,
             const float);

double zbrent(double(*func)(double),const double,const double,const double);
float  zbrent(float(*func)(float),  const float, const float, const float);

double LevMar(const double*, const double*, const double*, const int, double*,
              const int*, const int, 
              double (*)(const double,const double*,double*,const int),
              const double, const int);

double FitGauss(               // return:    chi^2
                const int    , // input:     no. of data points
                const double*, // input:     x data
                const double*, // input:     y data
                const double*, // input:     sigma_y data
                double[3]    , // in/output: initial guess/best-fit para
                const int[3]); // input:     fit para or not

} // namespace
#endif
