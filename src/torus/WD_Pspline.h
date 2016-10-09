/**
\file WD_Pspline.h
\brief Contains templated spline classes needed by GalaxyPotential

 Pspline.h 

 C++ code written by Walter Dehnen, 1994, 
 Oxford University, Department of Physics, Theoretical Physics 
 address: 1 Keble Road, Oxford OX1 3NP, United Kingdom 
 e-mail : dehnen@thphys.ox.ac.uk 

------------------------------------------------------------------------------
 In the following 

S  denotes  a scalar: a type for which the following operators are defined 
            =, +=, -=, *=, /=, +, -, *, /, <, >, <=, >=, !=, and == 
            between S and which allows for explicit conversion from a float 
T  denotes  a type for which the following operators are defined 
            =, +=, -=, +, -, !=, and ==  between T 
            =, *=, /=, *, and /  between a T and a S 
            (e.g. a Vector or a Matrix) 

******************************************************************************

Following the notation in Press et al. (1992, hereafter NR) for cubic spli- 
nes, we first notice that for given y_i, y_i1 (where i1==i+1) there is a 
unique linear function (first order polynomial) 
    P_1(x) = A y_i + B y_i1                                                (1) 
with 
    A(x) = (x_i1-x) / dx_i ;   dx_i == x_i1 - x_i 
    B(x) = (x-x_i)  / dx_i = 1-A                                           (2) 
such that P_1(x_i)=y_i and P_1(x_i1)=y_i1 (NR, eq. 3.3.1). Here, we are 
interested in the case where the first derivative, dy(x), is also tabulated. 
Then one might use the unique cubic polynomial 
    P_3(x) = P_1(x) + C (dy_i  - [y_i1-y_i]/dx_i) 
                    + D (dy_i1 - [y_i1-y_i]/dx_i)                          (3) 
with 
    C(x) = dx_i (A^2 - A^3), 
    D(x) = dx_i (B^3 - B^2)                                                (4) 
such that P_3(x_i)=y_i, P_3(x_i1)=y_i1, dP_3(x_i)=dy_i, and dP_3(x_i1)=dy_i1 
in analogy to (1). Suppose, contrary to fact, that we additionally knew the 
tabulated values of the third derivatives, d3y(x). There exists a unique 
fifth order polynomial 
    P_5(x) = P_3(x) + E (d3y_i - 6[dy_i+dy_i1]/dx_i^2 + 12[y_i1-y_1]/dx_i^3) 
                    + F (d3y_i1- 6[dy_i+dy_i1]/dx_i^2 + 12[y_i1-y_1]/dx_i^3) 
with                                                                       (5) 
    E(x) = dx_i^2 (2A^2 - A - 1) C / 48, 
    F(x) = dx_i^2 (2B^2 - B - 1) D / 48                                    (6) 
such that P_5(x_i)=y_i, P_5(x_i1)=y_i1, dP_5(x_i)=dy_i, dP_5(x_i1)=dy_i1, 
d3P_5(x_i)=d3y_i, and d3P_5(x_i1)=d3y_i1. In close analogy to the cubic 
spline, the d3y_i can now be determined by the condition that the second 
derivatives are continuous at the grid points. This leads to the following 
N-2 equations for the d3y_i (with i0==i-1) 
    (dx_i0 d3y_i0 + dx_i d3y_i1)/(x_i1-x_i0) - 3 d3y_i    = 
    12/(x_i1-x_i0) ( 7[dy_i/dx_i+dy_i/dx_i0] + 3[dy_i0/dx_i0+dy_i1/dx_i] 
                    -10[{y_i-y_i0}/dx_i0 + {y_i1-y_i}/dx_i] ).             (7) 
Together with the boundary conditions 
    d3y_0 = d3y_{N-1} = 0.                                                 (8) 
this constrains the d3y_i completely. Moreover, the equations (7) form a 
tridiagonal set of equations and can be solved in O(N) operations in close 
analogy to cubic splines. 

*******************************************************************************/

#ifndef _Pspline_def_
#define _Pspline_def_ 1

#include "WD_FreeMemory.h"
#include "WD_Numerics.h"

namespace WD{
////////////////////////////////////////////////////////////////////////////////
template<class S>
inline void find_for_Pspline(int& klo, const int n, S const *x, const S xi)
// the ordinary find() returns n-1 (in klo) if and only if xi=x[n-1]. This
// is inconvenient for we want x[klo+1] to be defined. Therefore, this routine
// always returns klo such that xi is between x[klo] and x[klo+1] inclusive.
{
    find(klo,n,x,xi);
    if(klo==n-1) klo--;
}
////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
void Pspline(
        S*  const x,            // input:   table of points x
        T*  const Y,            // input:   table of y(x)
        T*  const Y1,           // input:   table of dy/dx
        int const n,            // input:   size of these tables
        T *Y3)                  // output:  table of d^3y/dx^3
// Given dy/dx on a grid, d^3y/dx^3 is computed such that the (unique) 
// polynomials of 5th order between two adjacent grid points that give y,dy/dx,
// and d^3y/dx^3 on the grid are continuous in d^2y/dx^2, i.e. give the same
// value at the grid points. At the grid boundaries  d^3y/dx^3=0  is adopted.
{
    const S      zero=0.,one=1.,three=3.,seven=7.,ten=10.,twelve=12.; 
    register int i;
    register S   p,sig,dx,dx1,dx2;
    register T   dy=Y[1]-Y[0], dy1=dy;
    S *v = new S[n-1];
    dx   = x[1]-x[0];
    Y3[0]= v[0] = zero;
    for(i=1; i<n-1; i++) {
        dx1  = x[i+1]-x[i];
        dx2  = x[i+1]-x[i-1];
        dy1  = Y[i+1]-Y[i];
        sig  = dx/dx2;
        p    = sig*v[i-1]-three;
        v[i] = (sig-one)/p;
        Y3[i]= twelve*(   seven*Y1[i]*dx2/(dx*dx1) 
                        + three*(Y1[i-1]/dx+Y1[i+1]/dx1)
                        - ten*(dy/(dx*dx) + dy1/(dx1*dx1))  ) / dx2;
        Y3[i]= (Y3[i] - sig*Y3[i-1] ) / p;
        dx   = dx1;
        dy   = dy1;
    }
    Y3[n-1] = zero;
    for(i=n-2; i>=0; i--)
        Y3[i] += v[i]*Y3[i+1];
    delete[] v;
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T Psplint(                      // return:  y(xi)
        S* const x,             // input:   table of points at x[lo]
        T* const Y,             // input:   table of y    at y[lo]
        T* const Y1,            // input:   table of y'   at y1[lo]
        T* const Y3,            // input:   table of y''' at y3[lo]
        S  const xi,            // input:   x-value where y is wanted
        T* dYi=0,               // output:  dy/dx(xi)     if dy  != 0
        T* d2Yi=0)              // output:  d^2y/d^2x(xi) if d2y != 0
{
    const    S zero=0.,one=1.,two=2.,five=5.,six=6.,nine=9.,fe=48.;
    register S h,hi,hf, A,B,C,D,Aq,Bq;
    if((h=x[1]-x[0])==zero) Numerics_error("bad X input in Psplint()");
    hi = one/h;
    hf = h*h;
    A  = hi*(x[1]-xi); Aq = A*A;
    B  = one-A;        Bq = B*B;
//  if(A*B<zero) Numerics_error("X not bracketed in Psplint()");
    C  = h*Aq*B;
    D  =-h*Bq*A;
    register T  t1 = hi*(Y[1]-Y[0]),
                C2 = Y1[0]-t1,
                C3 = Y1[1]-t1,
                t2 = six*(Y1[0]+Y1[1]-t1-t1)/hf,
                C4 = Y3[0]-t2,
                C5 = Y3[1]-t2;
    hf/= fe;
    register T  Yi = A*Y[0] + B*Y[1] + C*C2 + D*C3
                     + hf * ( C*(Aq+Aq-A-one)*C4 + D*(Bq+Bq-B-one)*C5 );
    if(dYi) {
        register S BAA=B-A-A, ABB=A-B-B;
        hf  += hf;
        *dYi = t1 + (A*ABB)*C2 + (B*BAA)*C3
             + hf*A*B*((one+A-five*Aq)*C4+ (one+B-five*Bq)*C5);
        if(d2Yi) {
            *d2Yi = BAA*C2 - ABB*C3;
            *d2Yi+= *d2Yi  + hf * ( (two*Aq*(nine*B-A)-one) * C4
                                   +(two*Bq*(B-nine*A)+one) * C5 );
            *d2Yi*= hi;
        }
    }
    return Yi;
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T PsplinT(                      // return:  y(xi)
        const S& x0,            // input:   x0 <= x
        const S& x1,            // input:   x  <= x1 
        const T& Y0,            // input:   Y(x0)
        const T& Y1,            // input:   Y(x1)
        const T& Y10,           // input:   dY(x0)
        const T& Y11,           // input:   dY(x1)
        const T& Y30,           // input:   d3Y(x0)
        const T& Y31,           // input:   d3Y(x0)
        const S& xi,            // input:   x-value where y is wanted
        T* dYi=0,               // output:  dy/dx(xi)     if dy  != 0
        T* d2Yi=0)              // output:  d^2y/d^2x(xi) if d2y != 0
{
    const    S zero=0.,one=1.,two=2.,five=5.,six=6.,nine=9.,fe=48.;
    register S h,hi,hf, A,B,C,D,Aq,Bq;
    if((h=x1-x0)==zero) Numerics_error("PsplinT bad X input");
    hi = one/h;
    hf = h*h;
    A  = hi*(x1-xi); Aq = A*A;
    B  = one-A;      Bq = B*B;
//  if(A*B<zero) Numerics_error("X not bracketed in Psplint()");
    C  = h*Aq*B;
    D  =-h*Bq*A;
    register T  t1 = hi*(Y1-Y0),
                C2 = Y10-t1,
                C3 = Y11-t1,
                t2 = six*(Y10+Y11-t1-t1)/hf,
                C4 = Y30-t2,
                C5 = Y31-t2;
    hf/= fe;
    register T  Yi = A*Y0+ B*Y1+ C*C2+ D*C3+ 
                     hf*(C*(Aq+Aq-A-one)*C4+ D*(Bq+Bq-B-one)*C5);
    if(dYi) {
        register S BAA=B-A-A, ABB=A-B-B;
        hf  += hf;
        *dYi = t1 + (A*ABB)*C2 + (B*BAA)*C3
             + hf*A*B*((one+A-five*Aq)*C4+ (one+B-five*Bq)*C5);
        if(d2Yi) {
            *d2Yi = BAA*C2 - ABB*C3;
            *d2Yi+= *d2Yi  + hf * ( (two*Aq*(nine*B-A)-one) * C4
                                   +(two*Bq*(B-nine*A)+one) * C5 );
            *d2Yi*= hi;
        }
    }
    return Yi;
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
void PsplinTarr(
        const S& xl,            // input:   xl <= x
        const S& xh,            // input:   x  <= xh 
        const S& xi,            // input:   x-value where y is wanted
        T* const yl,            // input:   Y_k(xl)
        T* const yh,            // input:   Y_k(xh)
        T* const y1l,           // input:   dY_k(xl)
        T* const y1h,           // input:   dY_k(xh)
        T* const y3l,           // input:   d3Y_k(xl)
        T* const y3h,           // input:   d3Y_k(xh)
        const int K,            // input:   k=0,...,K-1
        T* y,                   // output:  y_k(xi)
        T* dy=0,                // output:  dy_k/dx(xi)     if dy  != 0
        T* d2y=0)               // output:  d^2y_k/d^2x(xi) if d2y != 0
{
    const    S zero=0.,one=1.,two=2.,five=5.,six=6.,nine=9.,fe=48.;
    register S h,hi,hq,hf, A,B,C,D,E,F,Aq,Bq;
    register T C2=*yl,C3=C2,C4=C2,C5=C2, t1=C2,t2=C2;
    register T *Y=y,*Yl=yl,*Yh=yh,*Y1l=y1l,*Y1h=y1h,*Y3l=y3l,*Y3h=y3h,*YK=y+K;
    if((h=xh-xl)==zero) Numerics_error("PsplinTarr bad X input");
    hi = one/h;
    hq = h*h;
    hf = hq/fe;
    A  = hi*(xh-xi); Aq = A*A;
    B  = one-A;      Bq = B*B;
//  if(A*B<zero) Numerics_error("X not bracketed in Psplintarr()");
    C  = h*Aq*B;
    D  =-h*Bq*A;
    E  = hf*C*(Aq+Aq-A-one);
    F  = hf*D*(Bq+Bq-B-one);
    if(d2y) {
        register S hf2= hf+hf, BAA=B-A-A, ABB=A-B-B,
                   AB=A*B, ABh=hf2*AB, Cp=Aq-AB-AB, Dp=Bq-AB-AB,
                   Ep=ABh*(one+A-five*Aq), Fp=ABh*(one+B-five*Bq),
                   Epp=hf2*(two*Aq*(nine*B-A)-one), Fpp=hf2*(two*Bq*(B-nine*A)+one);
        register T *dY=dy, *d2Y=d2y;
        for(; Y<YK; Y++,Yl++,Yh++,Y1l++,Y1h++,Y3l++,Y3h++,dY++,d2Y++) {
            t1   = hi*(*Yh-*Yl);
            C2   = *Y1l-t1;
            C3   = *Y1h-t1;
            t2   = six*(*Y1l+*Y1h-t1-t1)/hq;
            C4   = *Y3l-t2;
            C5   = *Y3h-t2;
            *Y   = A**Yl + B**Yh + C*C2 + D*C3+ E*C4 + F*C5;
            *dY  = t1+ Cp*C2 + Dp*C3 + Ep*C4 + Fp*C5;
            *d2Y = BAA*C2 - ABB*C3;
            *d2Y+= *d2Y + Epp*C4 + Fpp*C5;
            *d2Y*= hi;
        }
    } else if(dy) {
        register S AB=A*B, ABh=(hf+hf)*AB, Cp=Aq-AB-AB, Dp=Bq-AB-AB,
                   Ep=ABh*(one+A-five*Aq), Fp=ABh*(one+B-five*Bq);
        register T *dY=dy;
        for(; Y<YK; Y++,Yl++,Yh++,Y1l++,Y1h++,Y3l++,Y3h++,dY++) {
            t1  = hi*(*Yh-*Yl);
            C2  = *Y1l-t1;
            C3  = *Y1h-t1;
            t2  = six*(*Y1l+*Y1h-t1-t1)/hq;
            C4  = *Y3l-t2;
            C5  = *Y3h-t2;
            *Y  = A**Yl + B**Yh + C*C2 + D*C3+ E*C4 + F*C5;
            *dY = t1+ Cp*C2 + Dp*C3 + Ep*C4 + Fp*C5;
        }
    } else {
        for(; Y<YK; Y++,Yl++,Yh++,Y1l++,Y1h++,Y3l++,Y3h++) {
            t1  = hi*(*Yh-*Yl);
            C2  = *Y1l-t1;
            C3  = *Y1h-t1;
            t2  = six*(*Y1l+*Y1h-t1-t1)/hq;
            C4  = *Y3l-t2;
            C5  = *Y3h-t2;
            *Y  = A**Yl+ B**Yh+ C*C2+ D*C3+ E*C4+ F*C5;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T Psplev(                       // return:  y(xi)
         S*  const x,           // input:   table of points
         T*  const Y,           // input:   table of y
         T*  const Y1,          // input:   table of y'
         T*  const Y3,          // input:   table of y'''
         int const n,           // input:   size of above tables
         S   const xi,          // input:   x-value where y is wanted
         T*  dyi=0,             // output:  y'(xi)   if dyi  != 0
         T*  d2yi=0)            // output:  y''(xi)  if d2yi != 0
// - x,y,y2 run from 0 to n-1 rather than from 1 to n;
// - y can be of more general type than scalar, e.g., Vector, Matrix.
// - takes old values of KLO and KHI to look whether they work again, else the
//   search for KLO, KHI is done using 'hunt' (NR) starting from linear 
//   interpolation.
{
  // N.B. this int used to be static, but that's bad with multithreading
    int lo=0;
    find_for_Pspline(lo,n,x,xi);
    return Psplint(x+lo,Y+lo,Y1+lo,Y3+lo,xi,dyi,d2yi);
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
void Pspline2D(
        S*  const x[2],  // input:   tables of points x0, x1
        T** const y[3],  // input:   tables of y, dy/dx0, dy/dx1
        int const n[2],  // input:   sizes of above tables: n[0],n[1]
        T** a[4])        // output:  tables of coeffs a[0],a[1],a[2],a[3]
{
// 2D Pspline with natural boundary conditions
    register   T   z=y[0][0][0];
    z = 0.;
    register int i,j;
    T *t = new T[n[0]];
    T *t1= new T[n[0]];
    T *t3= new T[n[0]];
// 1. for each x1 do 1D Pspline for y in x0
    for(j=0; j<n[1]; j++) {
        for(i=0; i<n[0]; i++) {
            t[i]  = y[0][i][j];         // y
            t1[i] = y[1][i][j];         // dy/dx0
        }
        Pspline(x[0],t,t1,n[0],t3);
        for(i=0; i<n[0]; i++)
            a[0][i][j] = t3[i];         // d^3y/dx0^3
    }
// 2. for each x0 do 1D Pspline for y and splines for dy/dx0, d^3y/dx0^3 in x1
    for(i=0; i<n[0]; i++) {
        Pspline(x[1],y[0][i],y[2][i],n[1],a[1][i]);
        spline (x[1],y[1][i],n[1],z,z,a[2][i],1,1);
        spline (x[1],a[0][i],n[1],z,z,a[3][i],1,1);
    }
    delete[] t;
    delete[] t1;
    delete[] t3;
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T Pspl2D(               // return:  f(xi,yi)
        S* const x,     // input:   array x at x[low]
        S* const y,     // input:   array x at y[low]
        T* const f0,    // input:   f(x,y)
        T* const f1,    // input:   df/dx(x,y)
        T* const f2,    // input:   df/dy(x,y)
        T* const a0,    // input:   a0(x,y)
        T* const a1,    // input:   a1(x,y)
        T* const a2,    // input:   a2(x,y)
        T* const a3,    // input:   a3(x,y)
        const S& xi,    // input:   x where f desired
        const S& yi,    // input:   y where f desired
        T*  d1=0,       // output:  gradient of y   if d1 != 0
        T** d2=0)       // output:  d^2y/dxi/dxj    if d2 != 0
{
    T fl[4]  = {f1[0],f1[1],a0[0],a0[1]}, fh[4]  = {f1[2],f1[3],a0[2],a0[3]},
      f2l[4] = {a2[0],a2[1],a3[0],a3[1]}, f2h[4] = {a2[2],a2[3],a3[2],a3[3]};
    if(d2) {
        T F[2], G[4], dF[2], dG[4], d2F[4], d2G[4];
        PsplinTarr      (y[0],y[1],yi,f0,f0+2,f2,f2+2,a1,a1+2,2,F,dF,d2F);
        splinTarr       (y[0],y[1],yi,fl,fh,f2l,f2h,4,G,dG,d2G);
        d2[1][1]=PsplinT(x[0],x[1],d2F[0],d2F[1],d2G[0],d2G[1],d2G[2],d2G[3], xi);
        d1[1]   =PsplinT(x[0],x[1], dF[0], dF[1], dG[0], dG[1], dG[2], dG[3], xi, d2[1]);
        d2[0][1]=d2[1][0];
        return   PsplinT(x[0],x[1],  F[0],  F[1],  G[0],  G[1],  G[2],  G[3], xi, d1, d2[0]);
    } else if(d1) {
        T F[2], G[4], dF[2], dG[4];
        PsplinTarr    (y[0],y[1],yi,f0,f0+2,f2,f2+2,a1,a1+2,2,F,dF);
        splinTarr     (y[0],y[1],yi,fl,fh,f2l,f2h,4,G,dG);
        d1[1] =PsplinT(x[0],x[1],dF[0],dF[1],dG[0],dG[1],dG[2],dG[3], xi);
        return PsplinT(x[0],x[1], F[0], F[1], G[0], G[1], G[2], G[3], xi, d1);
    }
    T F[2], G[4];
    PsplinTarr    (y[0],y[1],yi,f0,f0+2,f2,f2+2,a1,a1+2,2,F);
    splinTarr     (y[0],y[1],yi,fl,fh,f2l,f2h,4,G);
    return PsplinT(x[0],x[1],F[0],F[1],G[0],G[1],G[2],G[3],xi);
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T Psplev2D(              // return:  y(x0i,x1i)
        S*  const x[2],  // input:   tables of points x0, x1
        T** const y[3],  // input:   tables of y, dy/dx0, dy/dx1
        T** const a[4],  // input:   tables of coeffs a[0],a[1],a[2],a[3]
        int const n[2],  // input:   sizes of above tables: n[0],n[1]
        S   const xi[2], // input:   (x0,x1)-value where y is wanted
        T*  d1=0,        // output:  gradient of y   if d1 != 0
        T** d2=0)        // output:  d^2y/dxi/dxj    if d2 != 0
{
  // N.B. these 2 ints used to be static, but that's bad with multithreading
    int l0=0, l1=0;
    find_for_Pspline(l0,n[0],x[0],xi[0]);
    find_for_Pspline(l1,n[1],x[1],xi[1]);
    register int k0=l0+1, k1=l1+1;

/*
// for testing Pspl2D() above
    T f0[4] = {y[0][l0][l1],y[0][k0][l1],y[0][l0][k1],y[0][k0][k1]},
      f1[4] = {y[1][l0][l1],y[1][k0][l1],y[1][l0][k1],y[1][k0][k1]},
      f2[4] = {y[2][l0][l1],y[2][k0][l1],y[2][l0][k1],y[2][k0][k1]},
      a0[4] = {a[0][l0][l1],a[0][k0][l1],a[0][l0][k1],a[0][k0][k1]},
      a1[4] = {a[1][l0][l1],a[1][k0][l1],a[1][l0][k1],a[1][k0][k1]},
      a2[4] = {a[2][l0][l1],a[2][k0][l1],a[2][l0][k1],a[2][k0][k1]},
      a3[4] = {a[3][l0][l1],a[3][k0][l1],a[3][l0][k1],a[3][k0][k1]};
    return Pspl2D(x[0]+l0,x[1]+l1,f0,f1,f2,a0,a1,a2,a3,xi[0],xi[1],d1,d2);
*/
    T fl[2]  = {y[0][l0][l1],y[0][k0][l1]}, fh[2]  = {y[0][l0][k1],y[0][k0][k1]},
      f1l[2] = {y[2][l0][l1],y[2][k0][l1]}, f1h[2] = {y[2][l0][k1],y[2][k0][k1]},
      f3l[2] = {a[1][l0][l1],a[1][k0][l1]},f3h[2]={a[1][l0][k1],a[1][k0][k1]},
      flo[4] = {y[1][l0][l1],y[1][k0][l1],a[0][l0][l1],a[0][k0][l1]},
      fhi[4] = {y[1][l0][k1],y[1][k0][k1],a[0][l0][k1],a[0][k0][k1]},
      f2l[4] = {a[2][l0][l1],a[2][k0][l1],a[3][l0][l1],a[3][k0][l1]},
      f2h[4] = {a[2][l0][k1],a[2][k0][k1],a[3][l0][k1],a[3][k0][k1]};
    if(d2) {
        T F[2], G[4], dF[2], dG[4], d2F[2], d2G[4];
        PsplinTarr      (x[1][l1],x[1][k1],xi[1],fl,fh,f1l,f1h,f3l,f3h,2,F,dF,d2F);
        splinTarr       (x[1][l1],x[1][k1],xi[1],flo,fhi,f2l,f2h,4,G,dG,d2G);
        d2[1][1]=PsplinT(x[0][l0],x[0][k0],d2F[0],d2F[1],d2G[0],d2G[1],d2G[2],d2G[3],xi[0]);
        d1[1]   =PsplinT(x[0][l0],x[0][k0],dF[0],dF[1],dG[0],dG[1],dG[2],dG[3],xi[0],d2[1]);
        d2[0][1]=d2[1][0];
        return   PsplinT(x[0][l0],x[0][k0],F[0],F[1],G[0],G[1],G[2],G[3],xi[0],d1,d2[0]);
    } else if(d1) {
        T F[2], G[4], dF[2], dG[4];
        PsplinTarr(x[1][l1],x[1][k1],xi[1],fl,fh,f1l,f1h,f3l,f3h,2,F,dF);
        splinTarr (x[1][l1],x[1][k1],xi[1],flo,fhi,f2l,f2h,4,G,dG);
        d1[1] =PsplinT(x[0][l0],x[0][k0],dF[0],dF[1],dG[0],dG[1],dG[2],dG[3],xi[0]);
        return PsplinT(x[0][l0],x[0][k0],F[0],F[1],G[0],G[1],G[2],G[3],xi[0],d1);
    }
        T F[2], G[4];
        PsplinTarr(x[1][l1],x[1][k1],xi[1],fl,fh,f1l,f1h,f3l,f3h,2,F);
        splinTarr (x[1][l1],x[1][k1],xi[1],flo,fhi,f2l,f2h,4,G);
        return PsplinT(x[0][l0],x[0][k0],F[0],F[1],G[0],G[1],G[2],G[3],xi[0]);
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
void Pspline3D(
        S*   const x[3],     // input:   tables of points x0, x1, x2
        T*** const y[4],     // input:   tables of y, dy/dx0, dy/dx1, dy/dx2
        int  const n[3],     // input:   sizes of above tables: n[0],n[1],n[2]
        T*** a[11])          // output:  tables of coeffs a[i], i=0,...,10
{
// 3D Pspline with natural boundary conditions
    register   T   z=y[0][0][0][0];
    z = 0.;
    register int i,j,k,nn=fmax(n[0],n[1]);
    T *t = new T[nn];
    T *t1= new T[nn];
    T *t2= new T[nn];
    T *t3= new T[nn];
// 1   for each x2 do 2D Dslpine in x0,x1
    for(k=0; k<n[2]; k++) {
// 1.1 for each x1 do 1D Pspline for y in x0
        for(j=0; j<n[1]; j++) {
            for(i=0; i<n[0]; i++) {
                t[i]  = y[0][i][j][k];          // y
                t1[i] = y[1][i][j][k];          // dy/dx0
            }
            Pspline(x[0],t,t1,n[0],t3);
            for(i=0; i<n[0]; i++)
                a[0][i][j][k] = t3[i];          // d^3y/dx0^3
        }
// 1.2 for each x0 do 1D Pspline for y and splines for dy/dx0, d^3y/dx0^3 in x1
        for(i=0; i<n[0]; i++) {
            for(j=0; j<n[1]; j++) {
                t[j]  = y[0][i][j][k];          // y
                t1[j] = y[2][i][j][k];          // dy/dx0
            }
            Pspline(x[1],t,t1,n[1],t3);
            for(j=0; j<n[1]; j++) {
                a[1][i][j][k] = t3[j];
                t[j]  = y[1][i][j][k];
                t1[j] = a[0][i][j][k];
            }
            spline (x[1],t,n[1],z,z,t2,0,1);
            spline (x[1],t1,n[1],z,z,t3,0,1);
            for(j=0; j<n[1]; j++) {
                a[2][i][j][k] = t2[j];
                a[3][i][j][k] = t3[j];
            }
        }
    }
// 2   for each x0,x1 do 1D Pspline for y and splines for dy/dx0,dy/dx1 and a[i]
    for(i=0; i<n[0]; i++)
        for(j=0; j<n[1]; j++) {
            Pspline(x[2],y[0][i][j],y[3][i][j],n[2],a[4][i][j]);
            spline (x[2],y[1][i][j],n[2],z,z,a[5][i][j],0,1);
            spline (x[2],y[2][i][j],n[2],z,z,a[6][i][j],0,1);
            spline (x[2],a[0][i][j],n[2],z,z,a[7][i][j],0,1);
            spline (x[2],a[1][i][j],n[2],z,z,a[8][i][j],0,1);
            spline (x[2],a[2][i][j],n[2],z,z,a[9][i][j],0,1);
            spline (x[2],a[3][i][j],n[2],z,z,a[10][i][j],0,1);
        }
    delete[] t;
    delete[] t1;
    delete[] t2;
    delete[] t3;
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
T Psplev3D(                // return:  y(x0i,x1i)
        S*   const x[3],   // input:   tables of points x0, x1, x2
        T*** const y[4],   // input:   tables of y, dy/dx0, dy/dx1, dy/dx2
        T*** const a[11],  // input:   tables of coeffs a[i], i=0,...,10
        int  const n[3],   // input:   sizes of above tables: n[0],n[1],n[2]
        S    const xi[3],  // input:   (x0,x1,x2)-value where y is wanted
        T*   d1=0,         // output:  gradient of y   if d1 != 0
        T**  d2=0)         // output:  d^2y/dxi/dxj    if d2 != 0
{
  // N.B. these 3 ints used to be static, but that's bad with multithreading
    int l0=0, l1=0, l2=0;
    find_for_Pspline(l0,n[0],x[0],xi[0]);
    find_for_Pspline(l1,n[1],x[1],xi[1]);
    find_for_Pspline(l2,n[2],x[2],xi[2]);
    register int k0=l0+1,k1=l1+1,k2=l2+1;
    T Y0l[4]={y[0][l0][l1][l2],y[0][k0][l1][l2],y[0][l0][k1][l2],y[0][k0][k1][l2]},
      Y0h[4]={y[0][l0][l1][k2],y[0][k0][l1][k2],y[0][l0][k1][k2],y[0][k0][k1][k2]},
      Y3l[4]={y[3][l0][l1][l2],y[3][k0][l1][l2],y[3][l0][k1][l2],y[3][k0][k1][l2]},
      Y3h[4]={y[3][l0][l1][k2],y[3][k0][l1][k2],y[3][l0][k1][k2],y[3][k0][k1][k2]},
      A4l[4]={a[4][l0][l1][l2],a[4][k0][l1][l2],a[4][l0][k1][l2],a[4][k0][k1][l2]},
      A4h[4]={a[4][l0][l1][k2],a[4][k0][l1][k2],a[4][l0][k1][k2],a[4][k0][k1][k2]};
    T Bl[24] ={y[1][l0][l1][l2],y[1][k0][l1][l2],y[1][l0][k1][l2],y[1][k0][k1][l2],
               y[2][l0][l1][l2],y[2][k0][l1][l2],y[2][l0][k1][l2],y[2][k0][k1][l2],
               a[0][l0][l1][l2],a[0][k0][l1][l2],a[0][l0][k1][l2],a[0][k0][k1][l2],
               a[1][l0][l1][l2],a[1][k0][l1][l2],a[1][l0][k1][l2],a[1][k0][k1][l2],
               a[2][l0][l1][l2],a[2][k0][l1][l2],a[2][l0][k1][l2],a[2][k0][k1][l2],
               a[3][l0][l1][l2],a[3][k0][l1][l2],a[3][l0][k1][l2],a[3][k0][k1][l2]},
      Bh[24] ={y[1][l0][l1][k2],y[1][k0][l1][k2],y[1][l0][k1][k2],y[1][k0][k1][k2],
               y[2][l0][l1][k2],y[2][k0][l1][k2],y[2][l0][k1][k2],y[2][k0][k1][k2],
               a[0][l0][l1][k2],a[0][k0][l1][k2],a[0][l0][k1][k2],a[0][k0][k1][k2],
               a[1][l0][l1][k2],a[1][k0][l1][k2],a[1][l0][k1][k2],a[1][k0][k1][k2],
               a[2][l0][l1][k2],a[2][k0][l1][k2],a[2][l0][k1][k2],a[2][k0][k1][k2],
               a[3][l0][l1][k2],a[3][k0][l1][k2],a[3][l0][k1][k2],a[3][k0][k1][k2]},
      B2l[24]={a[5][l0][l1][l2],a[5][k0][l1][l2],a[5][l0][k1][l2],a[5][k0][k1][l2],
               a[6][l0][l1][l2],a[6][k0][l1][l2],a[6][l0][k1][l2],a[6][k0][k1][l2],
               a[7][l0][l1][l2],a[7][k0][l1][l2],a[7][l0][k1][l2],a[7][k0][k1][l2],
               a[8][l0][l1][l2],a[8][k0][l1][l2],a[8][l0][k1][l2],a[8][k0][k1][l2],
               a[9][l0][l1][l2],a[9][k0][l1][l2],a[9][l0][k1][l2],a[9][k0][k1][l2],
               a[10][l0][l1][l2],a[10][k0][l1][l2],a[10][l0][k1][l2],a[10][k0][k1][l2]},
      B2h[24]={a[5][l0][l1][k2],a[5][k0][l1][k2],a[5][l0][k1][k2],a[5][k0][k1][k2],
               a[6][l0][l1][k2],a[6][k0][l1][k2],a[6][l0][k1][k2],a[6][k0][k1][k2],
               a[7][l0][l1][k2],a[7][k0][l1][k2],a[7][l0][k1][k2],a[7][k0][k1][k2],
               a[8][l0][l1][k2],a[8][k0][l1][k2],a[8][l0][k1][k2],a[8][k0][k1][k2],
               a[9][l0][l1][k2],a[9][k0][l1][k2],a[9][l0][k1][k2],a[9][k0][k1][k2],
               a[10][l0][l1][k2],a[10][k0][l1][k2],a[10][l0][k1][k2],a[10][k0][k1][k2]};
    if(d2) {
        T F[4],dF[4],d2F[4],B[24],dB[24],d2B[24];
        PsplinTarr     (x[2][l2],x[2][k2],xi[2],Y0l,Y0h,Y3l,Y3h,A4l,A4h,4,F,dF,d2F);
        splinTarr      (x[2][l2],x[2][k2],xi[2],Bl,Bh,B2l,B2h,24,B,dB,d2B);
        d2[2][2]=Pspl2D(x[0]+l0,x[1]+l1,d2F,d2B,d2B+4,d2B+8,d2B+12,d2B+16,d2B+20,xi[0],xi[1]);
        d1[2]   =Pspl2D(x[0]+l0,x[1]+l1, dF, dB, dB+4, dB+8, dB+12, dB+16, dB+20,xi[0],xi[1],d2[2]);
        d2[0][2]=d2[2][0]; d2[1][2]=d2[2][1];
        return   Pspl2D(x[0]+l0,x[1]+l1,  F,  B,  B+4,  B+8,  B+12,  B+16,  B+20,xi[0],xi[1],d1,d2);
    } else if(d1) {
        T F[4],dF[4],B[24],dB[24];
        PsplinTarr   (x[2][l2],x[2][k2],xi[2],Y0l,Y0h,Y3l,Y3h,A4l,A4h,4,F,dF);
        splinTarr    (x[2][l2],x[2][k2],xi[2],Bl,Bh,B2l,B2h,24,B,dB);
        d1[2] =Pspl2D(x[0]+l0,x[1]+l1,dF,dB,dB+4,dB+8,dB+12,dB+16,dB+20,xi[0],xi[1]);
        return Pspl2D(x[0]+l0,x[1]+l1, F, B, B+4, B+8, B+12, B+16, B+20,xi[0],xi[1],d1);
    }
    T F[4],B[24];
    PsplinTarr   (x[2][l2],x[2][k2],xi[2],Y0l,Y0h,Y3l,Y3h,A4l,A4h,4,F);
    splinTarr    (x[2][l2],x[2][k2],xi[2],Bl,Bh,B2l,B2h,24,B);
    return Pspl2D(x[0]+l0,x[1]+l1,F,B,B+4,B+8,B+12,B+16,B+20,xi[0],xi[1]);
}

////////////////////////////////////////////////////////////////////////////////
template<class S, class T>
void Derivs(S*   const x[3],
            T*** const y[4],
            int  const n[3])
// given y(x,y,z) in y[0], y[i] are filled with dy/dx_i of a cubic spline
// with natural boundary at x_i,max and dy/dx_i=0 at x_i,min
{
    register int i,j,k, nn=fmax(n[0],n[1]);
    T zero=0., *z=new T[nn], *z1=new T[nn];

// dy/dx0
    for(j=0; j<n[1]; j++)
        for(k=0; k<n[2]; k++) {
            for(i=0; i<n[0]; i++)
                z[i]=y[0][i][j][k];
            SplinedY(x[0],z,n[0],z1,&zero);
            for(i=0; i<n[0]; i++)
                y[1][i][j][k]=z1[i];
        }

// dy/dx1
    for(i=0; i<n[0]; i++)
        for(k=0; k<n[2]; k++) {
            for(j=0; j<n[1]; j++)
                z[j]=y[0][i][j][k];
            SplinedY(x[1],z,n[1],z1,&zero);
            for(j=0; j<n[1]; j++)
                y[2][i][j][k]=z1[j];
        }

// dy/dx2
    for(i=0; i<n[0]; i++)
        for(j=0; j<n[1]; j++)
            SplinedY(x[2],y[0][i][j],n[2],y[3][i][j],&zero);

    delete z;
    delete z1;
}

} // namespace
#endif
