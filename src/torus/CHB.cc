/*******************************************************************************
*                                                                              *
*  CHB.cc                                                                      *
*                                                                              *
* C++ code written by Paul McMillan, 2008                                      *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford OX1 3NP, United Kingdom                        *
* e-mail:  p.mcmillan1@physics.ox.ac.uk                                        *
*                                                                              *
*******************************************************************************/
#include <cmath>
#include <iostream>
#include "Pi.h"
#include "CHB.h"
#include <cstdlib>

namespace torus{
using std::cerr;

void Cheby::getchebc() { // get the coefficients
  delete[] s;
  tabsize = NChb*(NChb+1)/2;
  s = new double[tabsize]; 
  int mp;
  double twn=0.25;
  for(int m=0; m!=tabsize; m++) s[m]=0.;

  for(int m=0;m!=NChb;m++) {
    twn *= 2.;
    mp  = (m*(m+1))/2; 
    s[mp+m]=twn;
    if(m>=2) {
      int mm = m, n=mm/2;
      for(int k=1;k!=n+1;k++) 
	s[mp+m-(2*k)]=-0.25*s[mp+m-(2*k)+2]*double((mm-2*k+2)*(mm-2*k+1))/
	  double((mm-k)*k);
    }
  } 
  s[0] = 1.;
}
////////////////////////////////////////////////////////////////////////////////

Cheby::Cheby() {
  NChb = 0;
  s =  0;
  e1 = 0;
  tabsize = 0;
}


////////////////////////////////////////////////////////////////////////////////

Cheby::Cheby(const int N) {
  NChb = N;
  s = 0;
  getchebc();
  e1 = new double[NChb];
}

////////////////////////////////////////////////////////////////////////////////

Cheby::Cheby(double * inp, const int N) {
  setcoeffs(inp,N);
}

////////////////////////////////////////////////////////////////////////////////

void Cheby::setcoeffs(double * inp, const int N) {
  NChb = N;
  s = 0;
  getchebc();
  e1 = new double[NChb];
  for(int i=0;i!=NChb;i++)
    e1[i] = inp[i];
}

////////////////////////////////////////////////////////////////////////////////

Cheby::Cheby(const Cheby& Ch) {
  if(Ch.NChb) {
    NChb    = Ch.NChb;
    tabsize = Ch.tabsize;
    s = new double[tabsize];
    for(int i=0;i!=tabsize;i++)
      s[i] = Ch.s[i];
    e1 = new double[NChb];
    for(int i=0;i!=NChb;i++)
      e1[i] = Ch.e1[i];
  } else {
    NChb=0; tabsize =0;
    s = 0; e1=0;
  }
}

////////////////////////////////////////////////////////////////////////////////
Cheby::Cheby(double * x, double * y, const int np, const int N) {
  NChb = N;
  if(NChb>50) { cerr << "too many coeffs in Cheby\n"; std::exit(0); }
  s = 0;
  getchebc();
  e1 = new double[NChb];
  chebyfit(x,y,np,N);
}
////////////////////////////////////////////////////////////////////////////////
Cheby::~Cheby() {
  if(NChb>0) {
    delete[] s;
    delete[] e1;
  }
}

////////////////////////////////////////////////////////////////////////////////
Cheby&  Cheby::operator= (const Cheby& Ch) {
  if(NChb == Ch.NChb) 
    for(int i=0;i!=NChb;i++)
      e1[i] = Ch.e1[i];
  else {
    delete[] s;
    delete[] e1;
    NChb    = Ch.NChb;
    tabsize = Ch.tabsize;
    s = new double[tabsize];
    for(int i=0;i!=tabsize;i++)
      s[i] = Ch.s[i];
    e1 = new double[NChb];
    for(int i=0;i!=NChb;i++)
      e1[i] = Ch.e1[i];
  }
  return *this;
}

////////////////////////////////////////////////////////////////////////////////
void Cheby::writecoeffs(std::ostream& out) const
{
  for(int i=0;i!=NChb;i++) out << e1[i] << " ";
  out << "\n";
}


////////////////////////////////////////////////////////////////////////////////
// fit Chebyshev polynomial with NC coefficients to y(x), given at np points 
void Cheby::chebyfit(double * x, double * y, const int np, const int NC)  
{
  // need to determine s here.
  if(NC > NChb) { 
    if(NChb) delete[] e1;
    NChb = NC;      
    e1 = new double[NChb];
    getchebc(); 
  } else if(NC && NC != NChb) {
    if(NChb) delete[] e1;
    NChb = NC;      
    e1 = new double[NChb];
  }
  double alpha = 2./(x[np-1]-x[0]),
    beta = -(x[np-1]+x[0])/(x[np-1]-x[0]),  // can be done quicker
    Q = alpha/beta,
    *xtmp=new double[np], *coeff[3]={new double[np-1], new double[np-2], new double[np-2]},
    bin[50],s11[50+1],dydx,
    th1,c1,c11,s1, fac, sum,a,b,c,ain0, sum1,sum2,sum3,cmk,betam,//q,
    work[50+2][2], ain[50+1];

  for(int i=0;i!=np;i++)   xtmp[i]=x[i]*alpha + beta;

  for(int i=0;i!=np-2;i++) { // work out quadratic coefs over intervals
    int i1=i+1,i2=i+2;
    dydx=(y[i2]-y[i])/(xtmp[i2]-xtmp[i]);
    coeff[2][i] = ((y[i1]-y[i])-(xtmp[i1]-xtmp[i])*dydx)/
                   ((xtmp[i1]-xtmp[i])*(xtmp[i1]-xtmp[i2]));
    coeff[1][i] = dydx-coeff[2][i]*(xtmp[i]+xtmp[i2]);
    coeff[0][i] = y[i]+xtmp[i]*(coeff[2][i]*xtmp[i2]-dydx);
  }
  for(int i=0;i!=NChb;i++) bin[i] = 0.;

  for(int i=0;i!=np-1;i++) { 
    for(int j=0;j!=2;j++) {
      if(!(i!=0 && j==0) ) { // otherwise this doesn't need to be done (i+j)
	c1 = (i+j==np-1)? 1. : (!(i+j))? -1. : xtmp[i+j];
	c11= c1;
	th1= acos(c1);
	s1 = sin(th1);
	s11[0] = s1;
	for(int k=1;k!=NChb+1;k++) {
	  s11[k]=s11[k-1]*c1+c11*s1;
	  c11   =c11*c1-s11[k-1]*s1;  // find the NChb+1 values of sin(k*th1) 
	}
	for(int m=1;m!=NChb+2;m++) { // exact copy, to be cleaned later
	  if(!(m%2)) {
	    int n=m/2;
	    fac=1.; sum=0.;
	    for(int k=0;k!=n;k++) { 
	      sum += fac*s11[m-2*k-1]/double(m-2*k); 
	      fac *= double(m-k)/double(k+1); 
	    }
	    sum = (.5*fac*th1+sum)/double(pow(2,m-1)); 
	    //cerr << th1 << 'n';
	  } else {
	    int n=(m+1)/2;
	    fac=1.; sum=0.;
	    for(int k=0;k!=n;k++) { 
	      sum += fac*s11[m-2*k-1]/double(m-2*k); 
	      fac *= double(m-k)/double(k+1); 
	    }
	    sum = sum/double(pow(2,m-1));
	  }
	  work[m][j] = sum;
	}
	work[0][j]=th1;
      } // endif
    } 
    for(int k=0; k!=NChb+1; k++)   ain[k]=work[k+1][0]-work[k+1][1];
    ain0 = work[0][0]-work[0][1];
    for(int k=0; k!=NChb+2; k++) work[k][0] = work[k][1];


    if(!(i))         { a = coeff[0][i];   b = coeff[1][i];   c = coeff[2][i]; }
    else if(i==np-2) { a = coeff[0][i-1]; b = coeff[1][i-1]; c = coeff[2][i-1];}
    else { a=0.5*(coeff[0][i]+coeff[0][i-1]); b=0.5*(coeff[1][i]+coeff[1][i-1]);
           c=0.5*(coeff[2][i]+coeff[2][i-1]);}

    for(int m=0;m!=NChb;m++) { 
      int mp = m*(m+1)/2; 
      sum1 = s[mp]*ain0;   // s is determined by getchebc
      sum2 = s[mp]*ain[0];
      sum3 = s[mp]*ain[1];
      if(m) {
	for(int k=1;k<=m;k++) { 
	  sum1 += s[mp+k]*ain[k-1];
	  sum2 += s[mp+k]*ain[k];
	  sum3 += s[mp+k]*ain[k+1];
	}
      }
      bin[m] += a*sum1 + b*sum2 + c*sum3;
    } 
  }
  for(int m=0;m!=NChb;m++) bin[m] *= 2./Pi;
  bin[0] *= 0.5;
  for(int k=0;k!=NChb;k++) { 
    e1[k] = 0.;
    for(int m=k;m!=NChb;m++) {
      int mp = m*(m+1)/2;
      e1[k] += bin[m]*s[mp+k]; 
    }
  } 
  double qk =1.;
  for(int k=0;k!=NChb;k++) {
    cmk   = 1.;
    betam = pow(beta,k);
    e1[k] *= betam;
    if(k<=NChb-2) { 
      for(int m=k+1;m!=NChb;m++) {
	betam *= beta;
	cmk   *= double(m)/double(m-k);
	e1[k] += e1[m]*cmk*betam;
      } 
    } 
    e1[k]*=qk;
    qk *=Q;
  }
  delete[] xtmp;
  delete[] coeff[0];
  delete[] coeff[1];
  delete[] coeff[2];
}

////////////////////////////////////////////////////////////////////////////////
// For Chebshev polynomial C(x) w. input x, these are: x, C(x), C'(x), C''(x)
void Cheby::unfitderiv(const double x,double &y, double &dy, double &d2y) const
{
  int nCm1=NChb-1, nCm2= nCm1-1;
  y   = e1[nCm1];
  dy  = e1[nCm1] * nCm1;
  d2y = e1[nCm1] * nCm1 * nCm2;
  for(int i=nCm2;i>=0;i--) {
    y = y*x + e1[i];
    if(i>0) dy  = dy*x  + e1[i]*i;
    if(i>1) d2y = d2y*x + e1[i]*i*(i-1);
  }
}
////////////////////////////////////////////////////////////////////////////////
// For Chebshev polynomial C(x) w. input x, these are: x, C(x), C'(x)
void Cheby::unfitderiv(const double x,double &y, double &dy) const
{
  int nCm1=NChb-1, nCm2= nCm1-1;
  y   = e1[nCm1];
  dy  = e1[nCm1] * nCm1;
  for(int i=nCm2;i>=0;i--) {
    y = y*x + e1[i];
    if(i>0) dy  = dy*x  + e1[i]*i;
  }
}
////////////////////////////////////////////////////////////////////////////////
// given x, returns C(x)
double Cheby::unfit1(const double x) const
{
  double y=e1[NChb-1];
  for(int j=NChb-2;j>=0;j--)       
    y = y*x + e1[j];
  return y;
}

////////////////////////////////////////////////////////////////////////////////
// Input table x[n], output table C(x)[n], number of terms n.
void Cheby::unfitn(double * x, double * y, const int n) const {
  for(int i=0; i!=n; i++) {
    y[i]=e1[NChb-1];
    for(int j=NChb-2;j>=0;j--)       
      y[i] = y[i]*x[i] + e1[j];
  }
}
} // namespace