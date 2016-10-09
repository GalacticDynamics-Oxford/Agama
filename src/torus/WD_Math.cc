/*******************************************************************************
*                                                                              *
* WD_Math.cc                                                                    *
*                                                                              *
* C++ code written by Walter Dehnen, 1994/95,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1Keble Road, Oxford, OX1 3NP, United Kingdom.                       *
* e-mail:  dehnen@thphys.ox.ac.uk                                              *
*                                                                              *
********************************************************************************
*                                                                              *
* some special mathematical functions, mostly from Press et al. (1992).        *
*                                                                              *
*******************************************************************************/

#include <iostream>
#include <complex>
using std::complex;
using std::cerr;
#include "WD_Math.h"
#include <cstdlib>

namespace WD{

const complex<double> IMAG = complex<double>(0,1);      // i
const complex<double> ITPi = IMAG * (2*M_PI);           // 2 i Pi
const double SPi   = 1.772453850905516027298167483341;  // Sqrt[Pi]
const double STPi  = 2.506628274631000502415765284811;  // Sqrt[2 Pi]
const double iSTPi = 1./STPi;                           // 1./Sqrt[2 Pi]

template<class S>  S    WDabs  (const S x) 
               { return (x<0)? -x : x; }
template<class S>  int  sign(const S x) 
               { return (x<0)? -1:((x>0)? 1:0 ); }
template<class S>  S    sign(const S x, const S s) 
               { return (s>0)? WDabs(x) : -WDabs(x);}


////////////////////////////////////////////////////////////////////////////////
// 0. auxiliary constants etc.
const int    maxit  = 100;
const double fpmin  = 1.e-40,
             eps    = 1.e-9,
             logeps =-20.72326583694641115616192309216;

////////////////////////////////////////////////////////////////////////////////
// 1. auxiliary functions *************************************************** //
////////////////////////////////////////////////////////////////////////////////
static void gser(double& gamser, const double a, const double x, double& lng)
{
    lng=LogGamma(a);
    if(x<=0.) {
        if(x<0.) MathError("x<0 in gser()");
        gamser=0.;
        return;
    }
    int    n;
    double sum,del,ap;
    ap  = a;
    del = sum = 1.0/a;
    for(n=1; n<=maxit; n++) {
        ++ap;
        del *= x/ap;
        sum += del;
        if(WDabs(del) < WDabs(sum)*eps) {
            gamser = sum*exp(-x+a*log(x)-lng);
            return;
        }
    }
    MathError("a too large, maxit too small in gser()");
}

static void gcf(double& gammcf, double a, double x, double& lng)
{
    int i;
    double an,b,c,d,del,h;

    lng = LogGamma(a);
    b   = x+1.-a;
    c   = 1./fpmin;
    d   = 1./b;
    h   = d;
    for(i=1; i<=maxit; i++) {
        an =-i*(i-a);
        b += 2.;
        d  = an*d+b; if(WDabs(d)<fpmin) d=fpmin;
        c  = b+an/c; if(WDabs(c)<fpmin) c=fpmin;
        d  = 1./d;
        del= d*c;
        h *= del;    if(WDabs(del-1.)<eps) break;
    }
    if(i>maxit) MathError("a too large, maxit too small in gcf()");
    gammcf = exp(-x+a*log(x)-lng) * h;
}

////////////////////////////////////////////////////////////////////////////////
// 2. functions defined in Math.h ******************************************* //
////////////////////////////////////////////////////////////////////////////////

void MathError(const char* msgs)
{
    cerr<<" error in Mathematics: "<<msgs<<'\n';
#ifndef ebug
    exit(1);
#endif
}

void MathWarning(const char* msgs)
{
    cerr<<" Warning in Mathematics: "<<msgs<<'\n';
}

//==============================================================================
// volume of the unit sphere in n dimensions
//==============================================================================
double SphVol(const int d)
{
    if(d==1) return 2;
    if(d==2) return M_PI;
//  if(d==3) return FPi/3.;
    int k,n;
    double 
    cn = M_PI,
    ae = 2.,
    ao = M_PI/2;
    for(k=1,n=2;;k++) {
        ae *= n/double(n+1);
        cn *= ae;
        if(++n==d) break;
        ao *= n/double(n+1);
        cn *= ao;
        if(++n==d) break;
    }
    return cn;
}

//==============================================================================
// logarithms of complex trigonometric and hyperbolic functions
//==============================================================================
complex<double> lnsin(const complex<double>& x)         // log(sin(x))
{
    double 
    ep = exp(-2*WDabs(imag(x))),
    em = sin(real(x))*(1.+ep);
    ep = cos(real(x))*(1.-ep);
    return complex<double>
        (WDabs(imag(x))+0.5*log(0.25*(em*em+ep*ep)),atan2(sign(imag(x))*ep,em));
}

complex<double> lncos(const complex<double>& x)         // log(cos(x))
{
    double 
    ep = exp(-2*WDabs(imag(x))),
    em = cos(real(x))*(1.+ep);
    ep = sin(real(x))*(1.-ep);
    return complex<double>
        (WDabs(imag(x))+0.5*log(0.25*(em*em+ep*ep)),atan2(-sign(imag(x))*ep,em));
}

complex<double> lnsinh(const complex<double>& x)        // log(sinh(x))
{
    double 
    ep = exp(-2*WDabs(real(x))),
    em = sin(imag(x))*(1.+ep);
    ep = cos(imag(x))*(1.-ep);
    return complex<double>
        (WDabs(real(x))+0.5*log(0.25*(em*em+ep*ep)),atan2(em,sign(real(x))*ep));
}

complex<double> lncosh(const complex<double>& x)        // log(cosh(x))
{
    double 
    ep = exp(-2*WDabs(real(x))),
    em = cos(imag(x))*(1.+ep);
    ep = sin(imag(x))*(1.-ep);
    return complex<double>
        (WDabs(real(x))+0.5*log(0.25*(em*em+ep*ep)),atan2(sign(real(x))*ep,em));
}

//==============================================================================
// Gamma functions
//==============================================================================

double LogGamma(const double x)
{
    const double cof[6]={ 76.18009172947146, -86.50532032941677,
                          24.01409824083091, -1.231739572450155,
                          1.208650973866179e-3, -5.395239384953e-6 };
    if(x<= 0.) {
      if( -x == int(-x) ) MathError("LogGamma called at z = -n");
        return log(M_PI/sin(M_PI*x)) - LogGamma(1.-x);
    }

    double ser=1.000000000190015,
    y   = x,
    tmp = y+5.5;
    tmp-= (y+0.5) * log(tmp);
    for(int j=0; j<6; j++) ser+= cof[j]/++y;
    return -tmp + log(STPi*ser/x);
}

complex<double> LogGamma(const complex<double> z)
{
    const double c[6]={ 76.18009172947146, -86.50532032941677,
                        24.01409824083091, -1.231739572450155,
                        1.208650973866179e-3, -5.395239384953e-6 };
    if( imag(z)==0. && real(z) <= 0. && -real(z) == -int(real(z)) ) 
        MathError("LogGamma called at z = -n");

    char turn = (real(z)<1)? 1 : 0;
    complex<double> 
    ser = 1.000000000190015,
    y   = turn? 2.-z : z,
    tmp = y+4.5;
    tmp-= (y-0.5)*log(tmp);
    for(int j=0; j<6; j++) {
        ser+= c[j]/y;
        y  += 1.;
    }
    if(turn) {
        y    = M_PI*z-M_PI;
        tmp -= log(STPi*ser/y) + lnsin(y);
//      tmp -= log(STPi*ser*sin(y)/y);
    } else 
        tmp = log(STPi*ser) - tmp;
    while(imag(tmp)> M_PI) tmp-= ITPi;
    while(imag(tmp)<-M_PI) tmp+= ITPi;
    return tmp;
}

double GammaP(const double a, const double x)
{
    if(x<0. || a<=0.) MathError("invalid arguments in GammaP()");
    if(x<(a+1.)) {
        double gamser,lng;
        gser(gamser,a,x,lng);
        return gamser;
    }
    double gamcf,lng;
    gcf(gamcf,a,x,lng);
    return 1.-gamcf;
}

double Loggamma(const double a, const double x)
{
    if(x<0. || a<=0.) MathError("invalid arguments in Loggamma()");
    if(x<(a+1.)) {
        double gamser,lng;
        gser(gamser,a,x,lng);
        return log(gamser)+lng;
    }
    double gamcf,lng;
    gcf(gamcf,a,x,lng);
    return log(1.-gamcf)+lng;
}

double LogGamma(const double a, const double x)
{
    if(x<0. || a<=0.) MathError("invalid arguments in Loggamma()");
    if(x<(a+1.)) {
        double gamser,lng;
        gser(gamser,a,x,lng);
        return log(1.-gamser)+lng;
    }
    double gamcf,lng;
    gcf(gamcf,a,x,lng);
    return log(gamcf)+lng;
}

//==============================================================================
// Exponential integrals
//==============================================================================

double En(const int n, const double x)
{
    if(n<0 || x<0. || (x==0. && n<=1)) MathError("bad argumends in En()");
    if(n==0)  return exp(-x)/x;
    if(x==0.) return 1./double(n-1);
    double ans;
    if(x>1.) {
        int    i,nm1=n-1;
        double a,b,c,d,del,h;
        b = x+n;
        c = 1./fpmin;
        d = 1./b;
        h = d;
        for(i=1; i<=maxit; i++) {
            a   =-i*(nm1+i);
            b  += 2.;
            d   = 1./(a*d+b);
            c   = b+a/c;
            del = c*d;
            h  *= del;
            if(WDabs(del-1.) < eps) return h*exp(-x);
        }
        ans = h*exp(-x);
        MathWarning("continued fraction failed in En()");
    } else {
        int    i,ii,nm1=n-1;
        double del,fac,psi;
        ans = nm1? 1./double(nm1) : -log(x)-EulerGamma;
        fac = 1.;
        for(i=1; i<=maxit; i++) {
            fac *=-x/double(i);
            if(i!=nm1)
                del =-fac/double(i-nm1);
            else {
                psi =-EulerGamma;
                for(ii=1; ii<=nm1; ii++)
                    psi+= 1./double(ii);
                del = fac*(psi-log(x));
            }
            ans += del;
            if(WDabs(del) < WDabs(ans)*eps) return ans;
        }
        MathWarning("series failed in En()");
    }
    return ans;
}

double Ei(const double x)
{
    if(x<=0.)   return -En(1,-x);
    if(x<fpmin) return log(x)+EulerGamma;
    int    k;
    double fact=1.,sum=0.,term=1.;
    if(x<=-logeps) {
        for(k=1; k<=maxit; k++) {
            fact*= x/k;
            term = fact/k;
            sum += term;
            if(term<eps*sum) break;
        }
        if(k>maxit) MathError("Series failed in Ei()");
        return sum+log(x)+EulerGamma;
    }
    for(k=1; k<=maxit; k++) {
        fact = term;
        term*= k/x;
        if(term<eps) break;
        if(term<fact) sum+= term;
        else {
            sum -= fact;
            break;
        }
    }
    if(k>maxit) MathError("Series failed in Ei()");
    return exp(x)*(1.0+sum)/x;
}

//==============================================================================
// Bessel functions
//==============================================================================

double J0(const double x)
{
    double ax=WDabs(x),y,ans1,ans2;
    if(ax < 8.) {
        y    = x*x;
        ans1 = 57568490574.0+y*(-13362590354.0+y*(651619640.7
                +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
        ans2 = 57568490411.0+y*(1029532985.0+y*(9494680.718
                +y*(59272.64853+y*(267.8532712+y*1.0))));
        return ans1/ans2;
    }
    double z=8./ax, xx=ax-0.785398164;
    y   =z*z;
    ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
            +y*(-0.2073370639e-5+y*0.2093887211e-6)));
    ans2=       -0.1562499995e-1+y*(0.1430488765e-3
            +y*(-0.6911147651e-5+y*(0.7621095161e-6-y*0.934935152e-7)));
    return sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
}

double J1(const double x)
{
    double ax=WDabs(x),y,ans1,ans2;
    if(ax < 8.) {
        y    = x*x;
        ans1 = x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
                +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
        ans2 = 144725228442.0+y*(2300535178.0+y*(18583304.74
                +y*(99447.43394+y*(376.9991397+y*1.0))));
        return ans1/ans2;
    }
    double z=8./ax, xx=ax-2.356194491;
    y    = z*z;
    ans1 = 1.0+y*(0.183105e-2+y*(-0.3516396496e-4
              +y*(0.2457520174e-5+y*(-0.240337019e-6))));
    ans2 =        0.04687499995+y*(-0.2002690873e-3
              +y*(0.8449199096e-5+y*(-0.88228987e-6 +y*0.105787412e-6)));
    return sign(x) * sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
}

double Jn(const int n, const double x)
{
    if(n<0) MathError(" negative n in Jn(x)");
    if(n==0)  return J0(x);
    if(n==1)  return J1(x);
    if(n==-1) return -J1(x);

    const double acc=60., bigno=1.e10, bigni=1.e-10;
    int    j,jsum,m;
    double ax,bj,bjm,bjp,sum,tox,ans;

    ax=WDabs(x);
    if(ax==0.0) return 0.0;
    if(ax>double(n)) {
        tox = 2./ax;
        bjm = J0(ax);
        bj  = J1(ax);
        for(j=1; j<n; j++) {
            bjp=j*tox*bj-bjm;
            bjm=bj;
            bj=bjp;
        }
        ans=bj;
    } else {
        tox  = 2./ax;
        m    = 2*((n+(int) sqrt(acc*n))/2);
        jsum = 0;
        bjp  = ans = sum = 0.;
        bj   = 1.;
        for (j=m;j>0;j--) {
            bjm = j*tox*bj-bjp;
            bjp = bj;
            bj  = bjm;
            if(WDabs(bj) > bigno) {
                bj  *= bigni;
                bjp *= bigni;
                ans *= bigni;
                sum *= bigni;
            } 
            if(jsum) sum += bj;
            jsum=!jsum;
            if(j==n) ans=bjp;
        }
        sum=2.0*sum-bj;
        ans /= sum;
    }
    return x < 0.0 && (n & 1) ? -ans : ans;
}

double Y0(const double x)
{
    if(x<0.) MathError(" negative argument in Y0(x)");
    double y,ans1,ans2;
    if(x < 8.0) {
        y    = x*x;
        ans1 =-2957821389.0+y*(7062834065.0+y*(-512359803.6
                +y*(10879881.29+y*(-86327.92757+y*228.4622733))));
        ans2 = 40076544269.0+y*(745249964.8+y*(7189466.438
                +y*(47447.26470+y*(226.1030244+y*1.0))));
        return (ans1/ans2)+0.636619772*J0(x)*log(x);
    }
    double z=8./x, xx=x-0.785398164;
        y    = z*z;
        ans1 = 1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
                  +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 =        -0.1562499995e-1+y*(0.1430488765e-3
                  +y*(-0.6911147651e-5+y*(0.7621095161e-6
                  +y*(-0.934945152e-7))));
    return sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
}

double Y1(const double x)
{
    if(x<0.) MathError(" negative argument in Y1(x)");
    double y,ans1,ans2;
    if(x < 8.) {
        y    = x*x;
        ans1 = x*(-0.4900604943e13+y*(0.1275274390e13
                +y*(-0.5153438139e11+y*(0.7349264551e9
                +y*(-0.4237922726e7+y*0.8511937935e4)))));
        ans2 = 0.2499580570e14+y*(0.4244419664e12
                +y*(0.3733650367e10+y*(0.2245904002e8
                +y*(0.1020426050e6+y*(0.3549632885e3+y)))));
        return (ans1/ans2)+0.636619772*(J1(x)*log(x)-1.0/x);
    }
    double z=8./x, xx=x-2.356194491;
    y    = z*z;
    ans1 = 1.0+y*(0.183105e-2+y*(-0.3516396496e-4
              +y*(0.2457520174e-5+y*(-0.240337019e-6))));
    ans2 =    0.04687499995+y*(-0.2002690873e-3
          +y*(0.8449199096e-5+y*(-0.88228987e-6+y*0.105787412e-6)));
    return sqrt(0.636619772/x)*(sin(xx)*ans1+z*cos(xx)*ans2);
}

double Yn(const int n, const double x)
{
    if(n<0)  MathError(" negative n in Yn(x)");
    if(x<0.) MathError(" negative argument in Yn(x)");
    if(n==0) return Y0(x);
    if(n==1) return Y1(x);
    int j;
    double by=Y1(x),bym=Y0(x),byp,tox=2./x;
    for(j=1; j<n; j++) {
        byp = j*tox*by-bym;
        bym = by;
        by  = byp;
    }
    return by;
}

double I0(const double x)
{
    double ax=WDabs(x),y;
    if(ax < 3.75) {
        y = x/3.75;
        y*= y;
        return 1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                  +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    }
    y = 3.75/ax;
    return (exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
                +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                +y*0.392377e-2))))))));
}

double I1(const double x)
{
    double ans,ax=WDabs(x),y;
    if(ax < 3.75) {
        y = x/3.75;
        y*= y;
        ans=ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
                +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))));
    } else {
        y=3.75/ax;
        ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
                -y*0.420059e-2));
        ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
                +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))));
        ans *= (exp(ax)/sqrt(ax));
    }
    return x < 0.0 ? -ans : ans;
}

double In(const int n, const double x)
{
    const double acc=60., bigno=1.e10, bigni=1.e-10;
    if(n<0) MathError(" negative n in In(x)");
    if(n==0)  return I0(x);
    if(n==1)  return I1(x);
    if(x==0.) return 0.;

    int    j;
    double bi,bim,bip,tox,ans;

    tox=2.0/WDabs(x);
    bip=ans=0.0;
    bi=1.0;
    for(j=2*(n+(int)sqrt(acc*n)); j>0; j--) {
        bim = bip+j*tox*bi;
        bip = bi;
        bi  = bim;
        if(WDabs(bi) > bigno) {
            ans *= bigni;
            bi  *= bigni;
            bip *= bigni;
        }
        if(j==n) ans=bip;
    }
    ans *= I0(x)/bi;
    return x < 0.0 && (n & 1) ? -ans : ans;
}

double K0(const double x)
{
    if(x<0.) MathError(" negative argument in K0(x)");
    double y;
    if(x <= 2.) {
        y = x*x/4.;
        return (-log(x/2.0)*I0(x))+(-0.57721566+y*(0.42278420
                +y*(0.23069756+y*(0.3488590e-1+y*(0.262698e-2
                +y*(0.10750e-3+y*0.74e-5))))));
    }
    y = 2./x;
    return (exp(-x)/sqrt(x))*(1.25331414+y*(-0.7832358e-1
                +y*(0.2189568e-1+y*(-0.1062446e-1+y*(0.587872e-2
                +y*(-0.251540e-2+y*0.53208e-3))))));
}

double K1(const double x)
{
    if(x<0.) MathError(" negative argument in K1(x)");
    double y;
    if(x <= 2.) {
        y=x*x/4.0;
        return (log(x/2.0)*I1(x))+(1.0/x)*(1.0+y*(0.15443144
                +y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1
                +y*(-0.110404e-2+y*(-0.4686e-4)))))));
    }
    y = 2./x;
    return (exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619
                +y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2
                +y*(0.325614e-2+y*(-0.68245e-3)))))));
}

double Kn(const int n, const double x)
{
    if(n<0)  MathError(" negative n in Kn(x)");
    if(x<0.) MathError(" negative argument in Kn(x)");
    if(n==0) return K0(x);
    if(n==1) return K1(x);
    int j;
    double bk,bkm,bkp,tox;

    tox = 2./x;
    bkm = K0(x);
    bk  = K1(x);
    for(j=1; j<n; j++) {
        bkp = bkm+j*tox*bk;
        bkm = bk;
        bk  = bkp;
    }
    return bk;
}

//==============================================================================
// orthogonal polynomials
//==============================================================================

// Hermite Polynomials

double HermiteH(const int n, const double x)
{
    if(n==0) return 1;
    if(n==1) return 2.*x;
    int    i=1;
    double h0=1., h1=2*x, hi=h1;
    while(n>i) {
        hi = 2. * (x*h1 - i*h0);
        h0 = h1;
        h1 = hi;
        i++;
    }
    return hi;
}

void HermiteH(const int n, const double x, double *H)
{
    H[0] = 1.;  if(n==0) return;
    H[1] = 2*x; if(n==1) return;
    int i=1;
    while(n>i) {
        H[i+1] = 2 * (x*H[i] - 2*H[i-1]);
        i++;
    }
}

void NormSqHermite(const int n, double *N)
{
    N[0] = SPi;         if(n==0) return;        // Sqrt[Pi]
    N[1] = 2*SPi;       if(n==1) return;        // 2*Sqrt[Pi]
    int i=1;
    while(n>=i++) N[i] = 2*i*N[i-1];
}

double HermiteH_normalized(const int n, const double x)
{
    if(n==0) return 1.   / sqrt(M_PI);
    if(n==1) return 2.*x / sqrt(M_PI*2);;
    int    i=1, N=2;
    double h0=1., h1=2*x, hi=h1;
    while(n>i) {
        hi = 2. * (x*h1 - i*h0);
        h0 = h1;
        h1 = hi;
        i++;
        N *= 2 * i;
    }
    return hi / sqrt(N*M_PI);
}

void HermiteH_normalized(const int n, const double x, double *H)
{
    if(n>=0) H[0] = 1.;
    if(n>0)  H[1] = 2*x;
    int i=1;
    while(n>i) {
        H[i+1] = 2 * (x*H[i] - 2*H[i-1]);
        i++;
    }
    int N=1;
    i=0;
    while(n>=i) {
        H[i] /= sqrt(N*M_PI);
        i++;
        N *= 2 * i; 
    }
}

} // namespace
// end of Math.cc //////////////////////////////////////////////////////////////
