/**
\file   PJMNum.h
\brief  More useful mathsy functions.

 */

// Creating my own set of Numerical recipe style things. Something I probably
// should have done a long time ago
//
//     PJMNum.h started 23/10/08
//


#ifndef _PJMNum_
#define _PJMNum_ 1 

namespace torus {

double trapzd(double(*func)(double), const double, const double, const int);
void   polint(double*, double*, const int, const double, double&, double&);
double qromb(double(*func)(double), const double, const double, const double = 1.e-6);
double probks(const double);
void   kstwo(double*, int, double*, int, double&, double&);





template <class C>
double trapzd(const C* const o, double(C::*func)(double) const,
	      const double a, const double b,const int n) {
  static double s;

  if(n==1)
    return (s=0.5*(b-a)*(o->*func)(a)+(o->*func)(b));
  else {
    double x,tmp,sum,del; 
    int it,j;
    for(it=1,j=1;j<n-1;j++) it <<= 1;
    tmp = 1./double(it);
    del = (b-a)*tmp;
    x = a+0.5*del;
    for(sum=0.,j=0;j<it;j++,x+=del) sum += (o->*func)(x);  
    s = 0.5*(s+(b-a)*sum*tmp);
    return s;
  }

}

template <class C>
double qromb(const C* const o, double(C::*func)(double) const, const double a,
	      const double b)
{
  const double EPS = 1.e-6;
  const int JMAX=20, JMAXP = JMAX+1, K=5;
  double ss,dss, s[JMAX], h[JMAXP], s_t[K], h_t[K];
  
  h[0]=1.;
  for(int j=1; j<=JMAX;j++) {
    s[j-1] = trapzd(o,func,a,b,j);
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

template <class C>
class stored_qromb {
 public:
  C* o;
  double(C::*func)(double) const;
  double a,b,*table;
  int ntab, ntabMAX, tabMAX, nmax;
  double sout;
  stored_qromb(C*,double(C::*)(double) const);
  void   sort_table();
  double trapzd_store(const int);
  double qromb_store(const double, const double);
  ~stored_qromb() {delete[] table;}
};




template <class C>
stored_qromb<C>::stored_qromb(C *obj,double(C::*f)(double) const) : o(obj), 
  func(f), ntab(0), tabMAX(11), sout(0)
{
  ntabMAX = pow(2.,tabMAX)+1;
  table = new double[ntabMAX];
}

template <class C>
double stored_qromb<C>::trapzd_store(const int n) {
  // static double s;
  if(n<=tabMAX) nmax = n;
  if(n==1) {
    table[ntab++] = (b-a)*(o->*func)(a);
    table[ntab++] = (b-a)*(o->*func)(b);
    sout=0.5*(table[0]+table[1]);
    //table[0] = a;
    //table[1] = b;
  }
  else {
    double x,tmp,sum,del; 
    int it,j;
    for(it=1,j=1;j<n-1;j++) it <<= 1;
    tmp = 1./double(it);
    del = (b-a)*tmp;
    x = a+0.5*del;
    for(sum=0.,j=0;j<it;j++,x+=del) {
      double tmp_tab = (o->*func)(x);
      //cerr << tmp_tab << '\n';
      if(n<=tabMAX) table[ntab++] = (b-a)*tmp_tab;
      //table[ntab++] = x;
      sum += tmp_tab;  
    }
    sout = 0.5*(sout+(b-a)*sum*tmp);
    //cerr << sout << '\n';
  }
  return sout;
}

template <class C>
double  stored_qromb<C>::qromb_store(const double a_in, const double b_in)
{
  const double EPS = 1.e-6;
  const int JMAX=20, JMAXP = JMAX+1, K=5;
  double ss,dss, s[JMAX], h[JMAXP], s_t[K], h_t[K];
  a = a_in; b = b_in;
  ntab = 0;
  sout=0.;
  h[0]=1.;
  for(int j=1; j<=JMAX;j++) {
    s[j-1] = trapzd_store(j);
    if(j>=K) {
      for(int i=0;i<K;i++) {
	h_t[i] = h[j-K+i];
	s_t[i] = s[j-K+i];
      }
      polint(h_t,s_t,K,0.,ss,dss);
      if(fabs(dss) <= EPS*fabs(ss)) {
	sort_table();
	return ss;
      }
    }
    h[j] = 0.25*h[j-1];
  }
  //cerr << "too many steps in qromb\n";
  sort_table();
  return ss;
}

template <class C>
void stored_qromb<C>::sort_table() {
  //cerr << ntab <<  ' ' << ntabMAX << '\n';

  double *integrands, *cumulative;
  integrands = new double[ntab];
  cumulative = new double[ntabMAX];
  integrands[0] = table[0];
  integrands[ntab-1] = table[ntab-1];
  int oldend = ntab;
  //for(int i=0;i!=ntab;i++) cerr << table[i] << '\n';
  for(int i=0;i!=nmax-1;i++) { // ??
    int /*n = nmax-i,*/ step = pow(2.,i+1), point=pow(2.,i); // for output table
    //cerr <<"point step "<< point << ' ' << step << '\n';
    int it=1;
    for(int j=1;j<nmax-i-1;j++) it <<= 1;
    //cerr << oldend << ' ' << it << '\n';
    for(int i=0;i!=it;i++) {
      //cerr << table[oldend-it+i] << '\n';
      integrands[point] = table[oldend-it+i];
      point += step;
    }
    oldend -= it;
  }
  //for(int i=0;i!=ntab;i++) cerr << integrands[i] << '\n';
  cumulative[0] = 0.;
  for(int i=1;i!=ntab;i++) {
    cumulative[i] = cumulative[i-1]+(0.5*(integrands[i-1]+integrands[i]))/double(ntab-1);
    //cerr << cumulative[i] << '\n';
  }
  for(int i=ntab;i!=ntabMAX;i++)  cumulative[i] = 0.;
  delete[] integrands;
  delete[] table;
  table = cumulative;
}

}  // namespace
#endif
