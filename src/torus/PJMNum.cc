


#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
namespace torus {
using std::cerr;

// A lot like Press et al's version.
////////////////////////////////////////////////////////////////////////////////
double trapzd(double(*func)(double), const double a, const double b,const int n)
{
  static double s;

  if(n==1)
    return (s=0.5*(b-a)*func(a)+func(b));
  else {
    double x,tmp,sum,del; 
    int it,j;
    for(it=1,j=1;j<n-1;j++) it <<= 1;
    tmp = 1./double(it);
    del = (b-a)*tmp;
    x = a+0.5*del;
    for(sum=0.,j=0;j<it;j++,x+=del) sum += func(x);  
    s = 0.5*(s+(b-a)*sum*tmp);
    return s;
  }
}

////////////////////////////////////////////////////////////////////////////////
void polint(double *xa, double *ya, const int n, 
	      const double x, double &y, double &dy) 
{
  int ns=0;
  double *c = new double[n], *d = new double[n], dift, dif = fabs(x-xa[0]),ho,hp,w,den;
  for(int i=0;i!=n;i++) {
    if((dift=fabs(x-xa[i]))<dif) {
      ns=i; dif=dift;
    }
    c[i] = ya[i];
    d[i] = ya[i];
  }
  y=ya[ns--];
  for(int m=1;m!=n;m++) {
    for(int i=0;i<n-m;i++) {
      ho=xa[i]-x;
      hp=xa[i+m]-x;
      w=c[i+1]-d[i];
      if((den=ho-hp) == 0.) {cerr << "error in polint\n";}// more needed?
      den=w/den;
      d[i] = hp*den;
      c[i] = ho*den;
    }
    dy=(2*(ns+1)<(n-m)) ? c[ns+1] : d[ns--];
    y += dy;
  }
  delete[] c;
  delete[] d;
}

////////////////////////////////////////////////////////////////////////////////
double qromb(double(*func)(double), const double a, 
	     const double b, const double EPS) {

  const int JMAX=20, JMAXP = JMAX+1, K=5;
  double ss=0,dss, s[JMAX], h[JMAXP], s_t[K], h_t[K];
  
  h[0]=1.;
  for(int j=1; j<=JMAX;j++) {
    s[j-1] = trapzd(func,a,b,j);
    if(j>=K) {
      for(int i=0;i<K;i++) {
	h_t[i] = h[j-K+i];
	s_t[i] = s[j-K+i];
      }
      polint(h_t,s_t,K,0.,ss,dss);
      if(fabs(dss) <= EPS*fabs(ss)) return ss;
    }
    h[j] = 0.25*h[j-1];
  }
  //cerr << "too many steps in qromb\n";
  return ss;
}

////////////////////////////////////////////////////////////////////////////////
double probks(const double alam) {
  const double EPS1=1.e-6, EPS2=1.e-16;
  int j;
  double a2,fac=2.,sum=0.,term,termbf=0.;
  
  a2 = -2.*alam*alam;
  for(j=1;j!=100;j++) {
    term=fac*exp(a2*j*j);
    sum += term;
    if(fabs(term) <=EPS1*termbf || fabs(term) <= EPS2*sum) return sum;
    fac = -fac;
    termbf=fabs(term);
  }
  return 1.; // if not converged
}


void kstwo(double* data1, int n1, double* data2, int n2, 
	   double &d, double &prob) {
  int j1=0,j2=0;
  double d1,d2,dt,en1,en2,en,fn1=0.,fn2=0.;
  std::sort(data1,data1+n1);
  std::sort(data2,data2+n2);
  en1 = n1; en2 = n2; d=0.;
  while(j1<n1 && j2<n2) {
    if((d1=data1[j1]) <= (d2=data2[j2])) fn1=j1++/en1;
    if(d2<=d1) fn2=j2++/en2;
    if((dt=fabs(fn2-fn1))>d) d=dt;
  }
  en = sqrt(en1*en2/(en1+en2));
  prob = probks((en+0.12+0.11/en)*d);
}

} // namespace