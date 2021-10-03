/*******************************************************************************
*                                                                              *
* Numerics.cc                                                                  *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford, OX1 3NP, United Kingdom.                      *
* e-mail:  w.dehnen1@physics.ox.ac.uk                                          *
*                                                                              *
*******************************************************************************/

#include <cmath>
#include <algorithm>
#include "WD_Numerics.h"
#include "WD_FreeMemory.h"

namespace WD{

////////////////////////////////////////////////////////////////////////////////
typedef int*     Pint;
typedef float*   Pflt;
typedef float**  PPflt;
typedef double*  Pdbl;
typedef double** PPdbl;

////////////////////////////////////////////////////////////////////////////////
int GaussJordan(PPdbl a, const int n, PPdbl b, const int m)
{
    int    i, icol=0, irow=0, j, k, l, ll;
    double big, dum, pivinv;
    Pint ipiv  = new int[n];
    Pint indxr = new int[n];
    Pint indxc = new int[n];
/* these integer arrays are used for bookkeeping on the pivoting */
    for(j=0; j<n; j++)
        ipiv[j]=0;
    for(i=0; i<n; i++) {
/* this is the main loop over the columns to be reduced */
        big = 0.;
        for(j=0; j<n; j++)
/* this is the outer loop to search for the pivot element */
            if(ipiv[j]!=1)
                for(k=0; k<n; k++) {
                    if(ipiv[k]==0) {
                        if(fabs(a[j][k]) >= big) {
                            big=fabs(a[j][k]);
                            irow=j;
                            icol=k;
                        }
                    } else if(ipiv[k]>1)
                        return Numerics_message("GaussJordan: Singular Matrix 1");
                }
        ++(ipiv[icol]);
/* We now have the pivot element, so we interchange rows, if needed, to put the
pivot element on the diagonal. The columns are not physically interchanged, only
relabeled: indx[i], the column of the ith pivot element, is the ith column that
is reduced, while indxr[i] is the row in which that pivot element was originally
located. If indxr[i]!=indxc[i] there is an implied column interchange. With this
form of bookkeeping, the solution b's will end up in the correct order, and the
inverse matrix will be scrambled by columns. */
        if(irow != icol) {
            for(l=0; l<n; l++)
                std::swap(a[irow][l], a[icol][l]);
            for(l=0; l<m; l++)
                std::swap(b[irow][l], b[icol][l]);
        }
        indxr[i]=irow;
        indxc[i]=icol;
/* We are now ready to divide the pivot row by the pivot element located at irow
and icol. */
        if(a[icol][icol]==0.)
            return Numerics_message("GaussJordan: Singular Matrix 2");
        pivinv = 1./a[icol][icol];
        a[icol][icol] = 1.;
        for(l=0; l<n; l++) 
            a[icol][l] *= pivinv;
        for(l=0; l<m; l++) 
            b[icol][l] *= pivinv;
        for(ll=0; ll<n; ll++)
/* Next we reduce the rows except for the pivot one */
            if(ll != icol) {
                dum = a[ll][icol];
                a[ll][icol] = 0.;
                for(l=0; l<n; l++)
                    a[ll][l] -= a[icol][l]*dum;
                for(l=0; l<m; l++)
                    b[ll][l] -= b[icol][l]*dum;
            }
    }
/* This is the end of the main loop over the columns of the reduction. It only
remains to unscramble the solution in view of the column interchanges. We do
this by interchanging pairs of columns in the reverse order that the permutation
was built up. */
    for(l=n-1; l>=0; l--) {
        if (indxr[l] != indxc[l] )
            for(k=0; k<n; k++)
                std::swap(a[k][indxr[l]], a[k][indxc[l]]);
    }
    delete[] ipiv;
    delete[] indxr;
    delete[] indxc;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
int GaussJordan(PPdbl a, const int n, Pdbl b)
{
    int    i, icol=0, irow=0, j, k, l, ll;
    double big, dum, pivinv;
    Pint ipiv  = new int[n];
    Pint indxr = new int[n];
    Pint indxc = new int[n];
/* these integer arrays are used for bookkeeping on the pivoting */
    for(j=0; j<n; j++)
        ipiv[j]=0;
    for(i=0; i<n; i++) {
/* this is the main loop over the columns to be reduced */
        big = 0.;
        for(j=0; j<n; j++)
/* this is the outer loop to search for the pivot element */
            if(ipiv[j]!=1)
                for(k=0; k<n; k++) {
                    if(ipiv[k]==0) {
                        if(fabs(a[j][k]) >= big) {
                            big=fabs(a[j][k]);
                            irow=j;
                            icol=k;
                        }
                    } else if(ipiv[k]>1)
                        return Numerics_message("GaussJordan: Singular Matrix 1");
                }
        ++(ipiv[icol]);
/* We now have the pivot element, so we interchange rows, if needed, to put the
pivot element on the diagonal. The columns are not physically interchanged, only
relabeled: indx[i], the column of the ith pivot element, is the ith column that
is reduced, while indxr[i] is the row in which that pivot element was originally
located. If indxr[i]!=indxc[i] there is an implied column interchange. With this
form of bookkeeping, the solution b's will end up in the correct order, and the
inverse matrix will be scrambled by columns. */
        if(irow != icol) {
            std::swap(b[irow], b[icol]);
            for(l=0; l<n; l++)
                std::swap(a[irow][l], a[icol][l]);
        }
        indxr[i]=irow;
        indxc[i]=icol;
/* We are now ready to divide the pivot row by the pivot element located at irow
and icol. */
        if(a[icol][icol]==0.)
            return Numerics_message("GaussJordan: Singular Matrix 2");
        pivinv = 1./a[icol][icol];
        a[icol][icol] = 1.;
        b[icol] *= pivinv;
        for(l=0; l<n; l++) 
            a[icol][l] *= pivinv;
        for(ll=0; ll<n; ll++)
/* Next we reduce the rows except for the pivot one */
            if(ll != icol) {
                dum = a[ll][icol];
                a[ll][icol] = 0.;
                b[ll] -= b[icol]*dum;
                for(l=0; l<n; l++)
                    a[ll][l] -= a[icol][l]*dum;
            }
    }
/* This is the end of the main loop over the columns of the reduction. It only
remains to unscramble the solution in view of the column interchanges. We do
this by interchanging pairs of columns in the reverse order that the permutation
was built up. */
    for(l=n-1; l>=0; l--) {
        if (indxr[l] != indxc[l] )
            for(k=0; k<n; k++)
                std::swap(a[k][indxr[l]], a[k][indxc[l]]);
    }
    delete[] ipiv;
    delete[] indxr;
    delete[] indxc;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
int GaussBack(PPdbl a, const int n, Pdbl b)
/* Gaussian elemination with backsubstitution (cf. Numerical Recipes sect. 2.2).
   The inverse of matrix a is NOT computed. */
{
    int     i,irow,j,l;
    double  pivinv, temp;
/* This is the main loop over the columns to be reduced. */
    for(i=0; i<n; i++) {
/* Partial pivoting: the biggest element in the column i, in and below row i */
        irow = i;
        for(j=i+1; j<n; j++)
            if(fabs(a[j][i]) > fabs(a[irow][i])) irow = j;
        if(a[irow][i] == 0.)
            return Numerics_message("GaussBack: Singular Matrix");
/* Interchange rows if necessary: */
        if(irow != i) {
            temp=b[i]; b[i]=b[irow]; b[irow]=temp;
            for(l=i; l<n; l++)
                { temp=a[i][l]; a[i][l]=a[irow][l]; a[irow][l]=temp; }
        }
/* Divide the pivot row by the pivot element: */
        pivinv  = 1./a[i][i];
        a[i][i] = 1.;
        b[i]   *= pivinv;
        for(l=i+1; l<n; l++)
            a[i][l] *= pivinv;
/* Reduce the rows i+1 to n-1: */
        for(j=i+1; j<n; j++) {
            if(a[j][i] != 0.) {
                temp    = a[j][i];
                a[j][i] = 0.;
                b[j]   -= temp * b[i];
                for(l=i+1; l<n; l++)
                    a[j][l] -= temp * a[i][l];
            }
        }
    }
/* Now the matrix is half of triangular shape, and the solution can be computed
   by backsubstitution: */
    for(i=n-2; i>=0; i--)
        for(j=i+1; j<n; j++)
            b[i] -= a[i][j] * b[j];
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
// now follow the routines for floats instead of doubles ///////////////////////
////////////////////////////////////////////////////////////////////////////////
int GaussJordan(PPflt a, const int n, PPflt b, const int m)
{
    int   i, icol=0, irow=0, j, k, l, ll;
    float big, dum, pivinv;
    Pint ipiv  = new int[n];
    Pint indxr = new int[n];
    Pint indxc = new int[n];
/* these integer arrays are used for bookkeeping on the pivoting */
    for(j=0; j<n; j++)
        ipiv[j]=0;
    for(i=0; i<n; i++) {
/* this is the main loop over the columns to be reduced */
        big = 0.;
        for(j=0; j<n; j++)
/* this is the outer loop to search for the pivot element */
            if(ipiv[j]!=1)
                for(k=0; k<n; k++) {
                    if(ipiv[k]==0) {
                        if(fabs(a[j][k]) >= big) {
                            big=fabs(a[j][k]);
                            irow=j;
                            icol=k;
                        }
                    } else if(ipiv[k]>1)
                        return Numerics_message("GaussJordan: Singular Matrix 1");
                }
        ++(ipiv[icol]);
/* We now have the pivot element, so we interchange rows, if needed, to put the
pivot element on the diagonal. The columns are not physically interchanged, only
relabeled: indx[i], the column of the ith pivot element, is the ith column that
is reduced, while indxr[i] is the row in which that pivot element was originally
located. If indxr[i]!=indxc[i] there is an implied column interchange. With this
form of bookkeeping, the solution b's will end up in the correct order, and the
inverse matrix will be scrambled by columns. */
        if(irow != icol) {
            for(l=0; l<n; l++)
                std::swap(a[irow][l], a[icol][l]);
            for(l=0; l<m; l++)
                std::swap(b[irow][l], b[icol][l]);
        }
        indxr[i]=irow;
        indxc[i]=icol;
/* We are now ready to divide the pivot row by the pivot element located at irow
and icol. */
        if(a[icol][icol] == 0.)
            return Numerics_message("GaussJordan: Singular Matrix 2");
        pivinv = 1./a[icol][icol];
        a[icol][icol] = 1.;
        for(l=0; l<n; l++) 
            a[icol][l] *= pivinv;
        for(l=0; l<m; l++) 
            b[icol][l] *= pivinv;
        for(ll=0; ll<n; ll++)
/* Next we reduce the rows except for the pivot one */
            if(ll != icol) {
                dum = a[ll][icol];
                a[ll][icol] = 0.;
                for(l=0; l<n; l++)
                    a[ll][l] -= a[icol][l]*dum;
                for(l=0; l<m; l++)
                    b[ll][l] -= b[icol][l]*dum;
            }
    }
/* This is the end of the main loop over the columns of the reduction. It only
remains to unscramble the solution in view of the column interchanges. We do
this by interchanging pairs of columns in the reverse order that the permutation
was built up. */
    for(l=n-1; l>=0; l--) {
        if (indxr[l] != indxc[l] )
            for(k=0; k<n; k++)
                std::swap(a[k][indxr[l]], a[k][indxc[l]]);
    }
    delete[] ipiv;
    delete[] indxr;
    delete[] indxc;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
int GaussJordan(PPflt a, const int n, Pflt b)
{
    int   i, icol=0, irow=0, j, k, l, ll;
    float big, dum, pivinv;
    Pint ipiv  = new int[n];
    Pint indxr = new int[n];
    Pint indxc = new int[n];
/* these integer arrays are used for bookkeeping on the pivoting */
    for(j=0; j<n; j++)
        ipiv[j]=0;
    for(i=0; i<n; i++) {
/* this is the main loop over the columns to be reduced */
        big = 0.;
        for(j=0; j<n; j++)
/* this is the outer loop to search for the pivot element */
            if(ipiv[j]!=1)
                for(k=0; k<n; k++) {
                    if(ipiv[k]==0) {
                        if(fabs(a[j][k]) >= big) {
                            big=fabs(a[j][k]);
                            irow=j;
                            icol=k;
                        }
                    } else if(ipiv[k]>1)
                        return Numerics_message("GaussJordan: Singular Matrix 1");
                }
        ++(ipiv[icol]);
/* We now have the pivot element, so we interchange rows, if needed, to put the
pivot element on the diagonal. The columns are not physically interchanged, only
relabeled: indx[i], the column of the ith pivot element, is the ith column that
is reduced, while indxr[i] is the row in which that pivot element was originally
located. If indxr[i]!=indxc[i] there is an implied column interchange. With this
form of bookkeeping, the solution b's will end up in the correct order, and the
inverse matrix will be scrambled by columns. */
        if(irow != icol) {
            std::swap(b[irow], b[icol]);
            for(l=0; l<n; l++)
                std::swap(a[irow][l], a[icol][l]);
        }
        indxr[i]=irow;
        indxc[i]=icol;
/* We are now ready to divide the pivot row by the pivot element located at irow
and icol. */
        if(a[icol][icol]==0.)
            return Numerics_message("GaussJordan: Singular Matrix 2");
        pivinv = 1./a[icol][icol];
        a[icol][icol] = 1.;
        b[icol] *= pivinv;
        for(l=0; l<n; l++) 
            a[icol][l] *= pivinv;
        for(ll=0; ll<n; ll++)
/* Next we reduce the rows except for the pivot one */
            if(ll != icol) {
                dum = a[ll][icol];
                a[ll][icol] = 0.;
                b[ll] -= b[icol]*dum;
                for(l=0; l<n; l++)
                    a[ll][l] -= a[icol][l]*dum;
            }
    }
/* This is the end of the main loop over the columns of the reduction. It only
remains to unscramble the solution in view of the column interchanges. We do
this by interchanging pairs of columns in the reverse order that the permutation
was built up. */
    for(l=n-1; l>=0; l--) {
        if (indxr[l] != indxc[l] )
            for(k=0; k<n; k++)
                std::swap(a[k][indxr[l]], a[k][indxc[l]]);
    }
    delete[] ipiv;
    delete[] indxr;
    delete[] indxc;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
int GaussBack(PPflt a, const int n, Pflt b)
/* Gaussian elemination with backsubstitution (cf. Numerical Recipes sect. 2.2).
   The inverse of matrix a is NOT computed. */
{
    int    i,irow,j,l;
    float  pivinv, temp;
/* This is the main loop over the columns to be reduced. */
    for(i=0; i<n; i++) {
/* Partial pivoting: the biggest element in the column i, in and below row i */
        irow = i;
        for(j=i+1; j<n; j++)
            if(fabs(a[j][i]) > fabs(a[irow][i])) irow = j;
        if(a[irow][i] == 0.)
            return Numerics_message("GaussBack: Singular Matrix");
/* Interchange rows if necessary: */
        if(irow != i) {
            temp=b[i]; b[i]=b[irow]; b[irow]=temp;
            for(l=i; l<n; l++)
                { temp=a[i][l]; a[i][l]=a[irow][l]; a[irow][l]=temp; }
        }
/* Divide the pivot row by the pivot element: */
        pivinv  = 1./a[i][i];
        a[i][i] = 1.;
        b[i]   *= pivinv;
        for(l=i+1; l<n; l++)
            a[i][l] *= pivinv;
/* Reduce the columns i+1 to n-1 of the rows i+1 to n-1: */
        for(j=i+1; j<n; j++) {
            if(a[j][i] != 0.) {
                temp    = a[j][i];
                a[j][i] = 0.;
                b[j]   -= temp * b[i];
                for(l=i+1; l<n; l++)
                    a[j][l] -= temp * a[i][l];
            }
        }
    }
/* Now the solution can be computed by backsubstitution: */
    for(i=n-2; i>=0; i--)
        for(j=i+1; j<n; j++)
            b[i] -= a[i][j] * b[j];
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
double qbulir(double(*func)(double), const double a, const double b, 
              const double eps_, double& err)
/*------------------------------------------------------------------------------
Quadrature program using the Bulirsch sequence and rational extrapolation. The
algorithm is published in Bulirsch & Stoer, Num. Math. 9, 271-278 (1967), where
a routine in ALGOL is given. This routine is a straightforward translation into
C++.
CAUTION: 
Do not use this routine for integrating low order polynomials (up to fourth
order) or periodic functions with period equal to the interval of integration
or linear combinations of both.
INPUT:  func   pointer to function to be integrated.
        a,b    lower and upper boundaries of the integration interval;
        eps    desired relativ accuracy;
OUTPUT: return approximated value for the integral;
        err    actual relative error of the return value.
------------------------------------------------------------------------------*/
{
    double ba=b-a;
    if(ba==0.) return 0.;

    int    i,n=2,nn=3,mx=25,m,mr, bo,bu=0,odd=1;
    double c,d1,ddt,den,e,eps,eta=1.e-7,gr,hm,nt,
                    sm,t,t1,t2,t2a,ta,tab=0.,tb,v=0.,w;
    double d[7]={0},dt[7]={0};

    while(eta+1. != 1.) eta *=0.5;
    eta  *=2.;                       // eta = actual computing accuracy

    eps   = std::max(eps_,eta);
    sm    = 0.;
    gr    = 0.;
    t1    = 0.;
    t2    = 0.5*((*func)(a)+(*func)(b));
    t2a   = t2;
    tb    = fabs(t2a);
    c     = t2*ba;
    dt[0] = c;

    for(m=1;m<=mx;m++) {           // iterate over the refinements
        bo = (m>=7);
        hm = ba/n;
        if(odd) {
            for(i=1;i<=n;i+=2) {
                w  = (*func)(a+i*hm);
                t2+= w;
                tb+= fabs(w);
            }
            nt  = t2;
            tab = tb * fabs(hm);
            d[1]=16./9.;
            d[3]=64./9.;
            d[5]=256./9.;
        } else {
            for(i=1;i<=n;i+=6) {
                w  = i*hm;
                t1+= (*func)(a+w) + (*func)(b-w);
            }
            nt  = t1+t2a;
            t2a =t2;
            d[1]=9./4.;
            d[3]=9.;
            d[5]=36.;
        }
        ddt  =dt[0];
        t    =nt*hm;
        dt[0]=t;
        nt   =dt[0];
        if(bo) {
            mr  =6;
            d[6]=64.;
            w   =144.;
        } else {
            mr  =m;
            d[m]=n*n;
            w   =d[m];
        }
        for(i=1;i<=mr;i++) {
            d1 =d[i]*ddt;
            den=d1-nt;
            e  =nt-ddt;
            if(den != 0.) {
                e /= den;
                v  = nt*e;
                nt = d1*e;
                t += v;
            } else {
                nt = 0.;
                v  = 0.;
            }
            ddt  = dt[i];
            dt[i]= v;
        }
        ta = c;
        c  = t;
        if(!bo) t -= v;
        v  = t-ta;
        t += v;
        err= fabs(v);
        if(ta<t) {
            d1 = ta;
            ta = t;
            t  = d1;
        }
        bo = bo || (ta<gr && t>sm);
        if(bu && bo && err < eps*tab*w) break;
        gr = ta;
        sm = t;
        odd= !odd;
        i  = n;
        n  = nn;
        nn = i+i;
        bu = bo;
        d[2]=4.;
        d[4]=16.;
    }
    v = tab*eta;
    if(err<v) err = v;
    if(m==mx) Numerics_error("qbulir exceeding maximum of iterations");
    return c;
}
////////////////////////////////////////////////////////////////////////////////
void GaussLegendre(Pdbl x, Pdbl w, const int n)
{
    double eps=1e-10;
    // eps != actual computing accuracy
    // because that was crashing the f'ing thing
    int j,i,m=(n+1)/2;
    double z1,z,pp,p3,p2,p1;
    for (i=0;i<m;i++) {
        z=cos(3.141592653589793*(i+0.75)/(n+0.5));
        do {
            p1 = 1.0;
            p2 = 0.0;
            for(j=0;j<n;j++) {
                p3 = p2;
                p2 = p1;
                p1 = ( (2*j+1)*z*p2 - j*p3 ) / double(j+1);
            }
            pp = n * (z*p1-p2) / (z*z-1.0);
            z1 = z;
            z  = z1 - p1 / pp;
        } while (fabs(z-z1)>eps);
        x[i]     =-z;
        x[n-1-i] = z;
        w[i]     = 2. / ((1.0-z*z)*pp*pp);
        w[n-1-i] = w[i];
    }
}
////////////////////////////////////////////////////////////////////////////////
void LegendrePeven(double* p, const double x, const int np)
// based on a routine from J.J. Binney
// evaluates even Legendre Polys up to l=2*(np-1) at x
{
    int    n,l,l2;
    double x2=x*x;
    p[0] = 1.;
    p[1] = 1.5*x2-0.5;
    for(n=2; n<np; n++) {
        l = 2*(n-1);
        l2= 2*l;
        p[n] = - p[n-2] * l*(l-1)         / double((l2+1)*(l2-1))
               + p[n-1] * (x2-(l2*l+l2-1) / double((l2-1)*(l2+3)));
        p[n]*= (l2+1)*(l2+3) / double((l+1)*(l+2));
    }
}
////////////////////////////////////////////////////////////////////////////////
void dLegendrePeven(double* p, double* d, const double x, const int np)
// based on a routine from J.J. Binney
// evaluates even Legendre Polys and its derivs up to l=2*(np-1) at x
{
    int    n,l,l2;
    double x2=x*x;
    p[0] = 1.;
    d[0] = 0.;
    p[1] = 1.5*x2-0.5;
    d[1] = 1.5;
    for(n=2; n<np; n++) {
        l = 2*(n-1);
        l2= 2*l;
        p[n] = - p[n-2] * l*(l-1)         / double((l2+1)*(l2-1))
               + p[n-1] * (x2-(l2*l+l2-1) / double((l2-1)*(l2+3)));
        p[n]*= (l2+1)*(l2+3) / double((l+1)*(l+2));
        d[n] = - d[n-2] * l*(l-1)         / double((l2+1)*(l2-1))
               + d[n-1] * (x2-(l2*l+l2-1) / double((l2-1)*(l2+3)))
               + p[n-1];
        d[n]*= (l2+1)*(l2+3) / double((l+1)*(l+2));
    }
    x2 = 2*x;
    for(n=0; n<np; n++)
        d[n] *= x2;
}
////////////////////////////////////////////////////////////////////////////////
double zbrent(double(*func)(double), const double x1, const double x2,
              const double tol)
{
    int    iter=0,itmax=100;
    double a,b,c=0.,d=0.,e=0.,fa,fb,fc,tol1,tolh=0.5*tol,eps=6.e-8,s,p,q,r,xm;

    fa=func(a=x1);
    fb=func(b=x2);
    if((fa>0. && fb>0.) || (fa<0. && fb<0.)) 
        Numerics_error("zbrent: root must be bracketed");
    fc=fb;
    while(iter++<itmax) {
        if((fc>0. && fb>0.) || (fc<0. && fb<0.)) {
            c = a;
            fc= fa;
            e = (d=b-a);
        } 
        if(fabs(fc)<fabs(fb)) {
            a=b;   b=c;   c=a;
            fa=fb; fb=fc; fc=fa;
        }
        tol1= eps*fabs(b)+tolh;
        xm  = 0.5*(c-b);
        if(fabs(xm)<tol1 || fb==0.) return b;
        if(fabs(e)>=tol1 && fabs(fa)>fabs(fb)) {
            s=fb/fa;
            if(a==c) {
                p = 2.*xm*s;
                q = 1.-s;
            } else {
                q = fa/fc;
                r = fb/fc;
                p = s*(2.*xm*q*(q-r)-(b-a)*(r-1.));
                q = (q-1.)*(r-1.)*(s-1.);
            }
            if(p>0.) q =-q;
                else p =-p;
            if(2.*p < std::min(3.*xm*q-fabs(tol1*q), fabs(e*q))) {
                e = d;
                d = p/q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a  = b;
        fa = fb;
        if(fabs(d)>tol1) b += d;
            else        b += (xm<0.) ? -tol1 : tol1;
        fb = func(b);
    }
    Numerics_error("zbrent exceeding iterations");
    return b;
}
////////////////////////////////////////////////////////////////////////////////
float zbrent(float(*func)(float), const float x1, const float x2,
             const float tol)
{
    int   iter=0,itmax=100;
    float a,b,c=0.,d=0.,e=0.,fa,fb,fc,tol1,tolh=0.5f*tol,eps=6.e-8f,s,p,q,r,xm;

    fa=func(a=x1);
    fb=func(b=x2);
    if((fa>0.f && fb>0.f) || (fa<0.f && fb<0.f)) 
        Numerics_error("zbrent: root must be bracketed");
    fc=fb;
    while(iter++<itmax) {
        if((fc>0.f && fb>0.f) || (fc<0.f && fb<0.f)) {
            c = a;
            fc= fa;
            e = (d=b-a);
        } 
        if(fabs(fc)<fabs(fb)) {
            a=b;   b=c;   c=a;
            fa=fb; fb=fc; fc=fa;
        }
        tol1= eps*fabs(b)+tolh;
        xm  = 0.5f*(c-b);
        if(fabs(xm)<tol1 || fb==0.) return b;
        if(fabs(e)>=tol1 && fabs(fa)>fabs(fb)) {
            s=fb/fa;
            if(a==c) {
                p = 2.f*xm*s;
                q = 1.f-s;
            } else {
                q = fa/fc;
                r = fb/fc;
                p = s*(2.f*xm*q*(q-r)-(b-a)*(r-1.f));
                q = (q-1.f)*(r-1.f)*(s-1.f);
            }
            if(p>0.f) q =-q;
                else p =-p;
            if(2.f*p < std::min(3.f*xm*q-fabs(tol1*q), fabs(e*q))) {
                e = d;
                d = p/q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a  = b;
        fa = fb;
        if(fabs(d)>tol1) b += d;
            else        b += (xm<0.f) ? -tol1 : tol1;
        fb = func(b);
    }
    Numerics_error("zbrent exceeding iterations");
    return b;
}

////////////////////////////////////////////////////////////////////////////////
void heap_index(Pint A, const int n, Pint indx)
// the numbers 0 to n-1 are ordered in ascending order of A[i]
{
    int l,j,ir,indxt,i;
    int q;
    for(j=0; j<n; j++) indx[j] = j;
    l = n>>1;
    ir= n-1;
    for(;;) {
        if(l>0)
            q = A[indxt=indx[--l]];
        else {
            q = A[indxt=indx[ir]];
            indx[ir] = indx[0];
            if(--ir == 0) {
                indx[0] = indxt;
                return;
            }
        }
        i = l;
        j = (l<<1) + 1;
        while(j<=ir) {
            if(j < ir && A[indx[j]] < A[indx[j+1]] ) j++;
            if(q < A[indx[j]] ) {
                indx[i] = indx[j];
                j+= 1+(i=j);
            } else
                j = ir+1;
        }
        indx[i] = indxt;
    }
}

////////////////////////////////////////////////////////////////////////////////
void heap_index(int(*func)(const int), const int n, Pint indx)
// the numbers 0 to n-1 are ordered in ascending order of func(int)
{
    int l,j,ir,indxt,i;
    int q;
    for(j=0; j<n; j++) indx[j] = j;
    l = n>>1;
    ir= n-1;
    for(;;) {
        if(l>0)
            q = (*func)(indxt=indx[--l]);
        else {
            q = (*func)(indxt=indx[ir]);
            indx[ir] = indx[0];
            if(--ir == 0) {
                indx[0] = indxt;
                return;
            }
        }
        i = l;
        j = (l<<1) + 1;
        while(j<=ir) {
            if(j < ir && (*func)(indx[j]) < (*func)(indx[j+1]) ) j++;
            if(q < (*func)(indx[j]) ) {
                indx[i] = indx[j];
                j+= 1+(i=j);
            } else
                j = ir+1;
        }
        indx[i] = indxt;
    }
}

////////////////////////////////////////////////////////////////////////////////
void heap_index(float(*func)(const int), const int n, Pint indx)
// the numbers 0 to n-1 are ordered in ascending order of func(int)
{
    int l,j,ir,indxt,i;
    float q;
    for(j=0; j<n; j++) indx[j] = j;
    l = n>>1;
    ir= n-1;
    for(;;) {
        if(l>0)
            q = (*func)(indxt=indx[--l]);
        else {
            q = (*func)(indxt=indx[ir]);
            indx[ir] = indx[0];
            if(--ir == 0) {
                indx[0] = indxt;
                return;
            }
        }
        i = l;
        j = (l<<1) + 1;
        while(j<=ir) {
            if(j < ir && (*func)(indx[j]) < (*func)(indx[j+1]) ) j++;
            if(q < (*func)(indx[j]) ) {
                indx[i] = indx[j];
                j+= 1+(i=j);
            } else
                j = ir+1;
        }
        indx[i] = indxt;
    }
}

////////////////////////////////////////////////////////////////////////////////
void heap_index(double(*func)(const int), const int n, Pint indx)
// the numbers 0 to n-1 are ordered in ascending order of func(int)
{
    int l,j,ir,indxt,i;
    double q;
    for(j=0; j<n; j++) indx[j] = j;
    l = n>>1;
    ir= n-1;
    for(;;) {
        if(l>0)
            q = (*func)(indxt=indx[--l]);
        else {
            q = (*func)(indxt=indx[ir]);
            indx[ir] = indx[0];
            if(--ir == 0) {
                indx[0] = indxt;
                return;
            }
        }
        i = l;
        j = (l<<1) + 1;
        while(j<=ir) {
            if(j < ir && (*func)(indx[j]) < (*func)(indx[j+1]) ) j++;
            if(q < (*func)(indx[j]) ) {
                indx[i] = indx[j];
                j+= 1+(i=j);
            } else
                j = ir+1;
        }
        indx[i] = indxt;
    }
}

////////////////////////////////////////////////////////////////////////////////
float qsplin(Pflt x, Pflt y, Pflt y2, const int n, const float al,
             const float x1, const float x2)
// integrates the spline given by the arrays times x^al from x1 to x2
{
    if(x1 == x2) return 0.f;
    if((x[n-1]-x1)*(x1-x[0]) < 0.f) Numerics_error("qsplin: x1 not in range");
    if((x[n-1]-x2)*(x2-x[0]) < 0.f) Numerics_error("qsplin: x2 not in range");
    if(x1 > x2) Numerics_error("qsplin: x1 > x2");
    if(x1 < 0.f && al < 0.f) Numerics_error("qsplin: integral complex");
    if(x1 == 0.f && al <=-1.f) Numerics_error("qsplin: integral diverging");
    int   i,k;
    float q=0.f, ali, ali1, h, h2, t, xl, xh;
    float          a[4];
// find k with x[k] <= x1 <= x[k+1]
    k = hunt(x,n,x1,int((x1-x[0])/(x[n-1]-x[0])*(n-1)));
// go up until x[k] >= x2 and add to the integral
    while(x[k] < x2) {
      xl   = (x1 > x[k])? x1 : x[k];
      xh   = (x2 < x[k+1])? x2 : x[k+1];
        h    = x[k+1]-x[k];
        if(h==0.) Numerics_error("qsplin: bad x input");
        h2   = h*h;
        a[0] = x[k+1]*y[k] - x[k]*y[k+1] + x[k]*x[k+1]/6.f *
               ( (x[k+1]+h)*y2[k] - (x[k]-h)*y2[k+1] );
        a[1] = y[k+1]-y[k] + (  (h2-3.f*x[k+1]*x[k+1]) * y2[k]
                             -(h2-3.f*x[k]*x[k]) * y2[k+1] ) / 6.f;
        a[2] = 0.5f * (x[k+1]*y2[k] - x[k]*y2[k+1]);
        a[3] = (y2[k+1] - y2[k]) / 6.f;
        for(i=0,ali=al,t=0.f; i<4; i++,ali+=1.f)
            if(ali == -1.f)
                t += a[i] * log(xh/xl);
            else {
                ali1 = ali+1.f;
                t   += a[i] * (powf(xh,ali1) - powf(xl,ali1)) / ali1;
            }
        q += t / h;
        k++;
    }
    return q;
}

////////////////////////////////////////////////////////////////////////////////
double qsplin(Pdbl x, Pdbl y, Pdbl y2, const int n, const double al,
              const double x1, const double x2)
// integrates the spline given by the arrays times x^al from x1 to x2
{
    if(x1 == x2) return 0.;
    if((x[n-1]-x1)*(x1-x[0]) < 0.) Numerics_error("qsplin: x1 not in range");
    if((x[n-1]-x2)*(x2-x[0]) < 0.) Numerics_error("qsplin: x2 not in range");
    if(x1 > x2) Numerics_error("qsplin: x1 > x2");
    if(x1 < 0. && al < 0.) Numerics_error("qsplin: integral complex");
    if(x1 == 0. && al <=-1.) Numerics_error("qsplin: integral diverging");
    int    i,k;
    double q=0., ali, ali1, h, h2, t, xl, xh;
    double          a[4];
// find k with x[k] <= x1 <= x[k+1]
    k = hunt(x,n,x1,int((x1-x[0])/(x[n-1]-x[0])*(n-1)));
// go up until x[k] >= x2 and add to the integral
    while(x[k] < x2) {
        xl   = std::max(x[k],x1);
        xh   = std::min(x[k+1],x2);
        h    = x[k+1]-x[k];
        if(h==0.) Numerics_error("qsplin: bad x input");
        h2   = h*h;
        a[0] = x[k+1]*y[k] - x[k]*y[k+1] + x[k]*x[k+1]/6. *
               ( (x[k+1]+h)*y2[k] - (x[k]-h)*y2[k+1] );
        a[1] = y[k+1]-y[k] + (  (h2-3.*x[k+1]*x[k+1]) * y2[k]
                             -(h2-3.*x[k]*x[k]) * y2[k+1] ) / 6.;
        a[2] = 0.5 * (x[k+1]*y2[k] - x[k]*y2[k+1]);
        a[3] = (y2[k+1] - y2[k]) / 6.;
        for(i=0,ali=al,t=0.; i<4; i++,ali+=1.)
            if(ali == -1.)
                t += a[i] * log(xh/xl);
            else {
                ali1 = ali+1.;
                t   += a[i] * (pow(xh,ali1) - pow(xl,ali1)) / ali1;
            }
        q += t / h;
        k++;
    }
    return q;
}

////////////////////////////////////////////////////////////////////////////////
int CholeskyDecomposition(PPdbl a, const int n)
// given a positive definite symmetric matrix a[0...n-1][0...n-1] this routine
// constructs its Cholesky decomposition A = L L^t. On output L is returned in
// the lower triangle. (only the upper one has been used, however, the diagonal
// elements will be overwritten).
{
    int    i,j,k;
    double sum;
    for(i=0; i<n; i++)
        for(j=i; j<n; j++) {
            for(sum=a[i][j],k=i-1; k>=0; k--)
                sum -= a[i][k]*a[j][k];
            if(i==j) {
                if(sum <= 0.)
                    return Numerics_message("CholeskyDecomposition: Matrix not pos def");
                a[i][i] = sqrt(sum);
            } else
                a[j][i] = sum / a[i][i];
        }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
void CholeskySolution(const double** a, const int n, Pdbl b)
{
    int    i,k;
    double sum;
    for(i=0; i<n; i++) {
        for(sum=b[i],k=i-1; k>=0; k--) sum -= a[i][k]*b[k];
        b[i] = sum / a[i][i];
    }
    for(i=n-1; i>=0; i--) {
        for(sum=b[i],k=i+1; k<n; k++) sum -= a[k][i]*b[k];
        b[i] = sum / a[i][i];
    }
}

////////////////////////////////////////////////////////////////////////////////
void CholeskyInvertL(PPdbl a, const int n)
// given a, the output of CholeskyDecomposition(), we compute in place the
// inverse of the lower triangular matrix L
{
    int i,j,k;
    double sum;
    for(i=0; i<n; i++) {
        a[i][i] = 1./a[i][i];
        for(j=i+1; j<n; j++) {
            for(k=i,sum=0.; k<j; k++) sum -= a[j][k]*a[k][i];
            a[j][i] = sum/a[j][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void CholeskyInvertF(PPdbl a, const int n)
// given a, the output of CholeskyInvertL(), we compute in place the
// inverse fo the original input matrix to CholeskyDecomposition().
{
    int i,j,k;
    double sum;
// 1st compute L^-1^T * L^-1 in upper right triangle
    for(i=0; i<n; i++)
        for(j=n-1; j>=i; j--) {
            for(k=j,sum=0.; k<n; k++) sum += a[k][i]*a[k][j];
            a[i][j] = sum;
        }
// 2nd use symmetry to fill in lower left triangle
    for(i=0; i<n; i++)
    for(j=i+1; j<n; j++) a[j][i] = a[i][j]; 
}

////////////////////////////////////////////////////////////////////////////////
int LUDecomposition(PPdbl a, const int n, Pint indx, int& d)
{
    const double    tiny = 1.e-20;
    int    i,imax,j,k;
    double big, dum, sum;
    Pdbl            vv = new double[n];

    d = 1;
    for(i=0; i<n; i++) {
        big = 0.;
        for(j=0; j<n; j++) big = fmax(big, fabs(a[i][j]));
        if(big==0.) return Numerics_message("LUDecomposition: singular matrix");
        vv[i] = 1./big;
    }
    for(j=0; j<n; j++) {
        imax = j;
        for(i=0; i<j; i++) {
            sum = a[i][j];
            for(k=0; k<i; k++) sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
        }
        big = 0.;
        for(i=j; i<n ; i++) {
            sum = a[i][j];
            for(k=0; k<j; k++) sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
            if( (dum = vv[i]*fabs(sum)) >= big) {
                big = dum;
                imax= i;
            }
        }
        if(j!=imax) {
            for(k=0; k<n; k++) {
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            d =-d;
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if(a[j][j] == 0.) a[j][j] = tiny;
        if(j<n-1) {
            dum = 1./a[j][j];
            for(i=j+1; i<n; i++) a[i][j] *= dum;
        }
    }
    delete[] vv;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
void LUSolution(double **a, const int n, const int *indx, Pdbl b)
{
    int    i, ii=-1, ip, j;
    double sum;
    for(i=0; i<n; i++) {
        ip    = indx[i];
        sum   = b[ip];
        b[ip] = b[i];
        if(ii >=0 ) for(j=ii; j<i; j++) sum -= a[i][j]*b[j];
        else if (sum) ii=i;
        b[i] = sum;
    }
    for(i=n-1; i>=0; i--) {
        sum = b[i];
        for(j=i+1; j<n; j++) sum -= a[i][j]*b[j];
        b[i] = sum / a[i][i];
    }
}

////////////////////////////////////////////////////////////////////////////////
void LUInvert(double **a, double **y, const int n, const int *indx)
{
    int   i,j;
    double *col = new double[n];
    for(j=0; j<n; j++) {
        for(i=0; i<n; i++) col[i] = 0.;
        col[j] = 1.;
        LUSolution(a,n,indx,col);
        for(i=0; i<n; i++) y[i][j] = col[i];
    }
    delete[] col;
}

////////////////////////////////////////////////////////////////////////////////
int CholeskyDecomposition(PPflt a, const int n)
// given a positive definite symmetric matrix a[0...n-1][0...n-1] this routine
// constructs its Cholesky decomposition A = L L^t. On output L is returned in
// the lower triangle. (only the upper one has been used, however, the diagonal
// elements will be overwritten).
{
    int   i,j,k;
    float sum;
    for(i=0; i<n; i++)
        for(j=i; j<n; j++) {
            for(sum=a[i][j],k=i-1; k>=0; k--)
                sum -= a[i][k]*a[j][k];
            if(i==j) {
                if(sum <= 0.)
                    return Numerics_message("CholeskyDecomposition: Matrix not pos def");
                a[i][i] = sqrt(sum);
            } else
                a[j][i] = sum / a[i][i];
        }
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
void CholeskySolution(const float** a, const int n, Pflt b)
{
    int   i,k;
    float sum;
    for(i=0; i<n; i++) {
        for(sum=b[i],k=i-1; k>=0; k--) sum -= a[i][k]*b[k];
        b[i] = sum / a[i][i];
    }
    for(i=n-1; i>=0; i--) {
        for(sum=b[i],k=i+1; k<n; k++) sum -= a[k][i]*b[k];
        b[i] = sum / a[i][i];
    }
}

////////////////////////////////////////////////////////////////////////////////
void CholeskyInvertL(PPflt a, const int n)
// given a, the output of CholeskyDecomposition(), we compute in place the
// inverse of the lower triangular matrix L
{
    int i,j,k;
    double sum;
    for(i=0; i<n; i++) {
        a[i][i] = 1./a[i][i];
        for(j=i+1; j<n; j++) {
            for(k=i,sum=0.; k<j; k++) sum -= a[j][k]*a[k][i];
            a[j][i] = sum/a[j][j];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void CholeskyInvertF(PPflt a, const int n)
// given a, the output of CholeskyInvertL(), we compute in place the
// inverse fo the original input matrix to CholeskyDecomposition().
{
    int i,j,k;
    double sum;
// 1st compute L^-1^T * L^-1 in upper right triangle
    for(i=0; i<n; i++)
        for(j=n-1; j>=i; j--) {
            for(k=j,sum=0.; k<n; k++) sum += a[k][i]*a[k][j];
            a[i][j] = sum;
        }
// 2nd use symmetry to fill in lower left triangle
    for(i=0; i<n; i++)
    for(j=i+1; j<n; j++) a[j][i] = a[i][j]; 
}

////////////////////////////////////////////////////////////////////////////////
int LUDecomposition(PPflt a, const int n, Pint indx, int& d)
{
    const float    tiny = 1.e-20f;
    int   i,imax,j,k;
    float big, dum, sum;
    Pflt           vv = new float[n];

    d = 1;
    for(i=0; i<n; i++) {
        big = 0.;
        for(j=0; j<n; j++) big = fmax(double(big), fabs(a[i][j]));
        if(big==0.) return Numerics_message("LUDecomposition: singular matrix");
        vv[i] = 1./big;
    }
    for(j=0; j<n; j++) {
        imax = j;
        for(i=0; i<j; i++) {
            sum = a[i][j];
            for(k=0; k<i; k++) sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
        }
        big = 0.;
        for(i=j; i<n ; i++) {
            sum = a[i][j];
            for(k=0; k<j; k++) sum -= a[i][k]*a[k][j];
            a[i][j] = sum;
            if( (dum = vv[i]*fabs(sum)) >= big) {
                big = dum;
                imax= i;
            }
        }
        if(j!=imax) {
            for(k=0; k<n; k++) {
                dum = a[imax][k];
                a[imax][k] = a[j][k];
                a[j][k] = dum;
            }
            d =-d;
            vv[imax] = vv[j];
        }
        indx[j] = imax;
        if(a[j][j] == 0.) a[j][j] = tiny;
        if(j<n-1) {
            dum = 1./a[j][j];
            for(i=j+1; i<n; i++) a[i][j] *= dum;
        }
    }
    delete[] vv;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
void LUSolution(const float **a, const int n, const int *indx, Pflt b)
{
    int   i, ii=-1, ip, j;
    float sum;
    for(i=0; i<n; i++) {
        ip    = indx[i];
        sum   = b[ip];
        b[ip] = b[i];
        if(ii >=0 ) for(j=ii; j<i; j++) sum -= a[i][j]*b[j];
        else if (sum) ii=i;
        b[i] = sum;
    }
    for(i=n-1; i>=0; i--) {
        sum = b[i];
        for(j=i+1; j<n; j++) sum -= a[i][j]*b[j];
        b[i] = sum / a[i][i];
    }
}

////////////////////////////////////////////////////////////////////////////////
void LUInvert(const float **a, float **y, const int n, const int *indx)
{
    int   i,j;
    float *col = new float[n];
    for(j=0; j<n; j++) {
        for(i=0; i<n; i++) col[i] = 0.f;
        col[j] = 1.f;
        LUSolution(a,n,indx,col);
        for(i=0; i<n; i++) y[i][j] = col[i];
    }
    delete[] col;
}

////////////////////////////////////////////////////////////////////////////////
void tred2(PPdbl a, const int n, Pdbl d, Pdbl e, const char EV)
{
    int     l,k,j,i;
    double  scale,hh,h,g,f;
    for(i=n-1; i>0; i--) {
        l = i - 1;
        h = scale = 0.;
        if(l>0) {
            for(k=0; k<=l; k++)
                scale += fabs(a[i][k]);
            if(scale==0.)
                e[i] = a[i][l];
            else {
                for(k=0; k<=l; k++) {
                    a[i][k] /= scale;
                    h       += a[i][k] * a[i][k];
                }
                f       = a[i][l];
                g       = (f>=0.)? -sqrt(h) : sqrt(h);
                e[i]    = scale * g;
                h      -= f * g;
                a[i][l] = f - g;
                f       = 0.;
                for(j=0; j<=l; j++) {
                    if(EV) a[j][i] = a[i][j] / h;
                    for(k=0,g=0.; k<=j; k++)
                        g += a[j][k]*a[i][k];
                    for(k=j+1; k<=l; k++)
                        g += a[k][j]*a[i][k];
                    e[j] = g/h;
                    f += e[j]*a[i][j];
                }
                hh=f/(h+h);
                for(j=0; j<=l; j++) {
                    f=a[i][j];
                    e[j]=g=e[j]-hh*f;
                    for(k=0; k<=j; k++)
                        a[j][k] -= f*e[k]+g*a[i][k];
                }
            }
        } else
            e[i]=a[i][l];
        d[i]=h;
    }
    d[0] = e[0] = 0.;
    if(EV) {
        for(i=0; i<n; i++) {
            l=i-1;
            if(d[i]) {
                for(j=0; j<=l; j++) {
                    for(k=0,g=0.; k<=l; k++)
                        g += a[i][k]*a[k][j];
                    for(k=0; k<=l; k++)
                        a[k][j] -= g*a[k][i];
                }
            }
            d[i]=a[i][i];
            a[i][i]=1.0;
            for(j=0; j<=l; j++)
                a[j][i] = a[i][j] = 0.;
        }
    } else
        for(i=0; i<n; i++)
            d[i] = a[i][i];

}

////////////////////////////////////////////////////////////////////////////////
void tqli(Pdbl d, Pdbl e, const int n, PPdbl z, const char EV)
{
    int    m,l,iter,i,k;
    double s,r,p,g,f,dd,c,b;
    for(i=1; i<n; i++)
        e[i-1] = e[i];
    e[n-1] = 0.;
    for(l=0; l<n; l++) {
        iter = 0;
        do {
            for(m=l; m<n-1; m++) {
                dd = fabs(d[m]) + fabs(d[m+1]);
                if( fabs(e[m])+dd == dd ) break;
            }
            if(m != l) {
                if(iter++ == 30) Numerics_error("tqli: too many iterations");
                g = (d[l+1]-d[l])/(2.*e[l]);
                r = (g==0.)? 1. : hypot(g,1.);
                g = d[m]-d[l]+e[l]/(g+sign(r,g));
                s = c = 1.;
                p = 0.;
                for(i=m-1; i>=l; i--) {
                    f = s*e[i];
                    b = c*e[i];
                    e[i+1] = r = hypot(f,g);
                    if(r==0.) {
                        d[i+1] -= p;
                        e[m]    = 0.;
                        break;
                    }
                    s = f/r;
                    c = g/r;
                    g = d[i+1]-p;
                    r = (d[i]-g)*s+2.*c*b;
                    p = s*r;
                    d[i+1] = g+p;
                    g = c*r-b;
                    if(EV)
                        for(k=0; k<n; k++) {
                            f         = z[k][i+1];
                            z[k][i+1] = s*z[k][i] + c*f;
                            z[k][i]   = c*z[k][i] - s*f;
                        }
                }
                if(r==0. && i>=l) continue;
                d[l]-= p;
                e[l] = g;
                e[m] = 0.;
            }
        } while(m != l);
    }
}

////////////////////////////////////////////////////////////////////////////////
void balanc(double **a, const int n)
{
    const    double radix=2., sqrdx=radix*radix;
    int    last=0,j,i;
    double s,r,g,f,c;

    while(last==0) {
        last=1;
        for(i=0; i<n; i++) {
            r = c = 0.0;
            for(j=0; j<n; j++)
                if(j!=i) {
                    c += fabs(a[j][i]);
                    r += fabs(a[i][j]);
                }
            if(c&&r) {
                g = r/radix;
                f = 1.;
                s = c+r;
                while(c<g) {
                    f *= radix;
                    c *= sqrdx;
                }
                g = r * radix;
                while(c>g) {
                    f /= radix;
                    c /= sqrdx;
                }
                if((c+r)/f < 0.95*s) {
                    last = 0;
                    g    = 1. / f;
                    for(j=0; j<n; j++) a[i][j] *= g;
                    for(j=0; j<n; j++) a[j][i] *= f;
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void elmhes(double** a, const int n)
{
    int    m,j,i;
    double y,x;
    for(m=1; m<n-1; m++) {
        x = 0.0;
        i = m;
        for(j=m; j<n; j++)
            if(fabs(a[j][m-1]) > fabs(x)) {
                x = a[j][m-1];
                i = j;
            }
        if(i!=m) {
            for(j=m-1; j<n; j++) std::swap(a[i][j],a[m][j]);
            for(j=0;   j<n; j++) std::swap(a[j][i],a[j][m]);
        }
        if(x) {
            for(i=m+1; i<n ;i++)
                if((y=a[i][m-1]) != 0.) {
                    y        /= x;
                    a[i][m-1] = y;
                    for(j=m; j<n; j++) a[i][j]-= y * a[m][j];
                    for(j=0; j<n; j++) a[j][m]+= y * a[j][i];
                }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void hqr(double** a, const int n, double* wr, double* wi)
{
    int    nn,m,l,k,j,its,i,mmin;
    double z=0.,y=0.,x=0.,w=0.,v=0.,u=0.,t=0.,s=0.,r=0.,q=0.,p=0.,anrm;

    anrm = fabs(a[0][0]);
    for(i=1; i<n; i++)
        for(j=(i-1); j<n; j++)
            anrm += fabs(a[i][j]);
    nn = n-1;
    t  = 0.;
    while(nn >= 0) {
        its = 0;
        do {
            for(l=nn; l>=1; l--) {
                s = fabs(a[l-1][l-1]) + fabs(a[l][l]);
                if(s==0.0) s = anrm;
                if(fabs(a[l][l-1])+s == s) break;
            }
            x = a[nn][nn];
            if(l==nn) {
                wr[nn]   = x + t;
                wi[nn--] = 0.;
            } else {
                y = a[nn-1][nn-1];
                w = a[nn][nn-1] * a[nn-1][nn];
                if(l==(nn-1)) {
                    p = 0.5*(y-x);
                    q = p*p + w;
                    z = sqrt(fabs(q));
                    x+= t;
                    if(q>=0.) {
                        z        = p+sign(z,p);
                        wr[nn-1] = wr[nn]=x+z;
                        if(z) wr[nn] = x-w/z;
                        wi[nn-1] = wi[nn] = 0.;
                    } else {
                        wr[nn-1] = wr[nn] = x + p;
                        wi[nn-1] =-(wi[nn]=z);
                    }
                    nn-= 2;
                } else {
                    if(its==30) Numerics_error("hqr: exceeding iterations");
                    if(its==10 || its==20) {
                        t += x;
                        for(i=0; i<nn; i++) a[i][i] -= x;
                        s = fabs(a[nn][nn-1]) + fabs(a[nn-1][nn-2]);
                        y = x = 0.75*s;
                        w =-0.4375 * s*s;
                    }
                    ++its;
                    for(m=(nn-2); m>=l; m--) {
                        z = a[m][m];
                        r = x-z;
                        s = y-z;
                        p = (r*s-w) / a[m+1][m] + a[m][m+1];
                        q = a[m+1][m+1] - z - r - s;
                        r = a[m+2][m+1];
                        s = fabs(p)+fabs(q)+fabs(r);
                        p/= s;
                        q/= s;
                        r/= s;
                        if(m==l) break;
                        u = fabs(a[m][m-1]) * (fabs(q)+fabs(r));
                        v = fabs(p) * (fabs(a[m-1][m-1])+fabs(z)+fabs(a[m+1][m+1]));
                        if(u+v == v) break;
                    }
                    for(i=m+2; i<=nn; i++) {
                        a[i][i-2] = 0.;
                        if(i != (m+2)) a[i][i-3] = 0.;
                    }
                    for(k=m; k<=nn-1; k++) {
                        if(k != m) {
                            p = a[k][k-1];
                            q = a[k+1][k-1];
                            r = 0.;
                            if(k!=(nn-1)) r = a[k+2][k-1];
                            if((x=fabs(p)+fabs(q)+fabs(r))!=0.) {
                                p /= x;
                                q /= x;
                                r /= x;
                            }
                        }
                        if((s=sign(sqrt(p*p+q*q+r*r),p))!=0.) {
                            if(k==m) {
                                if(l!=m) a[k][k-1] = -a[k][k-1];
                            } else
                                a[k][k-1] = -s*x;
                            p+= s;
                            x = p/s;
                            y = q/s;
                            z = r/s;
                            q/= p;
                            r/= p;
                            for(j=k; j<=nn; j++) {
                                p = a[k][j] + q * a[k+1][j];
                                if(k!=(nn-1)) {
                                    p += r*a[k+2][j];
                                    a[k+2][j] -= p*z;
                                }
                                a[k+1][j]-= p*y;
                                a[k][j]  -= p*x;
                            }
                            mmin = nn<k+3 ? nn : k+3;
                            for(i=l; i<=mmin; i++) {
                                p = x * a[i][k] + y * a[i][k+1];
                                if(k!=(nn-1)) {
                                    p += z*a[i][k+2];
                                    a[i][k+2] -= p*r;
                                }
                                a[i][k+1]-= p*q;
                                a[i][k]  -= p;
                            }
                        }
                    }
                }
            }
        } while (l<nn-1);
    }
}
////////////////////////////////////////////////////////////////////////////////
static double LevCof(const double *x,
            const double *y,
            const double *sig,
            const int    N,
                  double *a,
            const int    *fit,
            const int    M,
                  double **A,
                  double *B,
            double (*f)(const double,const double*,double*,const int))
{
    int    i,j,k,l,mf,n;
    double si,dy,wt,cq=0.;
    double *dyda=new double[M];

    for(i=mf=0; i<M; i++) if(fit[i]) mf++;
    for(j=0; j<mf; j++) {
        B[j] = 0.;
        for(l=0; l<mf; l++)
            A[j][l] = 0.;
    }
    for(n=0; n<N; n++) {
        dy = y[n]- (*f)(x[n],a,dyda,M);
        si = 1./(sig[n]*sig[n]);
        cq+= pow(dy/sig[n],2);
        for(i=j=0; i<M; i++)
            if(fit[i]) {
                wt = dyda[i]*si;
                for(k=l=0; k<=M; k++)
                    if(fit[k]) A[j][l++] += wt*dyda[k];
                B[j++] += dy*wt;
            }
    }
    delete[] dyda;
    return cq;
}

double LevMar(
              const double *x,
              const double *y,
              const double *sig,
              const int    N,
                    double *a,
              const int    *fit,
              const int    M,
              double (*f)(const double,const double*,double*,const int),
              const double dcmax,
              const int    itmax)
{
    int    it=0,i,j,mf;
    int    mm[2];
    double dc,lam=0.125,tm, cq, cqo;
    double **A, *B, **Ay, *By, *ay;

    Alloc1D(ay,M);
    for(i=mf=0; i<M; i++) {
        ay[i] = a[i];
        if(fit[i]) mf++;
    }
    mm[0]=mm[1]=mf;
    Alloc2D(A,mm); Alloc2D(Ay,mm);
    Alloc1D(B,mf); Alloc1D(By,mf);

    cqo = LevCof(x,y,sig,N,a,fit,M,A,B,f);
    for(dc=0.,i=0; i<mf; i++) dc += B[i]*B[i];
    dc = sqrt(dc)/double(N);

    while(dc > dcmax && it++ < itmax) {
        tm = 1.+lam;
        for(i=0; i<mf; i++) {
            for(j=0; j<mf; j++) Ay[i][j] = A[i][j];
            Ay[i][i] *= tm;
            By[i]     = B[i];
        }
        GaussBack(Ay,mf,By);
        for(i=j=0; i<M; i++) if(fit[i]) ay[i] = a[i] + By[j++];
        if(cqo>(cq=LevCof(x,y,sig,N,ay,fit,M,Ay,By,f))) {
            lam *= 0.125;
            cqo = cq;
            for(dc=0.,i=0; i<mf; i++) {
                B[i] = By[i];
                dc  += B[i]*B[i];
                for(j=0; j<mf; j++) A[i][j] = Ay[i][j];
            }
            dc = sqrt(dc)/double(N);
        } else 
            lam *= 8;
    }
    Free1D(ay); Free1D(By); Free1D(B); Free2D(A); Free2D(Ay);
    return cqo;
}
////////////////////////////////////////////////////////////////////////////////
inline double gauss_fit(const double x,const double *p, double *df, const int D)
{
    if(D != 3)Numerics_error("FitGauss: D != 3");
    double fg,tm;
    tm   =(x-p[1])/p[2];
    df[0]=exp(-0.5*pow(tm,2));
    fg   =p[0]*df[0];
    df[1]=tm/p[2]*fg;
    df[2]=tm*df[1];
    return fg;
}

double FitGauss(                  // return:    chi^2
                const double *x,  // input:     x data
                const double *y,  // input:     y data
                const double *dy, // input:     sigma_y data
                const int    N,   // input:     no. of data points
                      double *p,  // in/output: initial guess/best-fit para
                const int    *f)  // input:     fit para or not
{
    return LevMar(x,y,dy,N,p,f,3,&gauss_fit,1.e-6,100);
}
} // namespace
////end of Numerics.cc//////////////////////////////////////////////////////////
