/*

galpot.cc 

C++ code 

Copyright Walter Dehnen, 1996-2004 
e-mail:   walter.dehnen@astro.le.ac.uk 
address:  Department of Physics and Astronomy, University of Leicester 
          University Road, Leicester LE1 7RH, United Kingdom 

Modifications by Eugene Vasiliev, June 2015 

------------------------------------------------------------------------

source code for class GalaxyPotential and dependencies 

Version 0.0    15. July      1997 
Version 0.1    24. March     1998 
Version 0.2    22. September 1998 
Version 0.3    07. June      2001 
Version 0.4    22. April     2002 
Version 0.5    05. December  2002 
Version 0.6    05. February  2003 
Version 0.7    23. September 2004  fixed "find(): x out of range" error 
Version 0.8    24. June      2005  explicit construction of tupel 

----------------------------------------------------------------------*/
#include "potential_galpot.h" 
#include "potential_composite.h"
#include "WD_FreeMemory.h"
#include "WD_Numerics.h"
#include "WD_Pspline.h"
#include "WD_Vector.h"
#include <cmath>
#include <stdexcept>

namespace potential{

inline double sign(double x) { return x>0?1.:x<0?-1.:0; }

const int    GALPOT_LMAX=80;     ///< maximum l for the Multipole expansion 
const int    GALPOT_NRAD=201;    ///< DEFAULT number of radial points in Multipole 
const double GALPOT_RMIN=1.e-4,  ///< DEFAULT min radius of logarithmic radial grid in Multipole
             GALPOT_RMAX=1.e3;   ///< DEFAULT max radius of logarithmic radial grid

//----- disk density and potential -----//

/** simple exponential radial density profile without inner hole or wiggles */
class DiskDensityRadialExp: public math::IFunction {
public:
    DiskDensityRadialExp(double _surfaceDensity, double _scaleLength): 
        surfaceDensity(_surfaceDensity), scaleLength(_scaleLength) {};
private:
    const double surfaceDensity, scaleLength;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        double val = surfaceDensity * exp(-R/scaleLength);
        if(f)
            *f = val;
        if(fprime)
            *fprime = -val/scaleLength;
        if(fpprime)
            *fpprime = val/pow_2(scaleLength);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** more convoluted radial density profile - exponential with possible inner hole and modulation */
class DiskDensityRadialRichExp: public math::IFunction {
public:
    DiskDensityRadialRichExp(const DiskParam& _params): params(_params) {};
private:
    const DiskParam params;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        if(params.innerCutoffRadius && R==0.) {
            if(f) *f=0;
            if(fprime)  *fprime=0;
            if(fpprime) *fpprime=0;
            return;
        }
        const double Rrel = R/params.scaleLength;
        const double cr = params.modulationAmplitude ? params.modulationAmplitude*cos(Rrel) : 0;
        const double sr = params.modulationAmplitude ? params.modulationAmplitude*sin(Rrel) : 0;
        double val = params.surfaceDensity * exp(-params.innerCutoffRadius/R-Rrel+cr);
        double fp  = params.innerCutoffRadius/(R*R)-(1+sr)/params.scaleLength;
        if(fpprime)
            *fpprime = (fp*fp-2*params.innerCutoffRadius/(R*R*R)-cr/pow_2(params.scaleLength))*val;
        if(fprime)
            *fprime  = fp*val;
        if(f) 
            *f = val;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** exponential vertical disk density profile */
class DiskDensityVerticalExp: public math::IFunction {
public:
    DiskDensityVerticalExp(double _scaleHeight): scaleHeight(_scaleHeight) {};
private:
    const double scaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z/scaleHeight);
        double      h        = exp(-x);
        if(H)       *H       = scaleHeight/2*(h-1+x);
        if(Hprime)  *Hprime  = sign(z)*(1.-h)/2;
        if(Hpprime) *Hpprime = h/(2*scaleHeight);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** isothermal (sech^2) vertical disk density profile */
class DiskDensityVerticalIsothermal: public math::IFunction {
public:
    DiskDensityVerticalIsothermal(double _scaleHeight): scaleHeight(_scaleHeight) {};
private:
    const double scaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z/scaleHeight);
        double      h        = exp(-x);
        double      sh1      = 1.+h;
        if(H)       *H       = scaleHeight*(0.5*x+log(0.5*sh1));
        if(Hprime)  *Hprime  = 0.5*sign(z)*(1.-h)/sh1;
        if(Hpprime) *Hpprime = h/(sh1*sh1*scaleHeight);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** vertically thin disk profile */
class DiskDensityVerticalThin: public math::IFunction {
public:
    DiskDensityVerticalThin() {};
private:
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        if(H)       *H       = fabs(z)/2;
        if(Hprime)  *Hprime  = sign(z)/2;
        if(Hpprime) *Hpprime = 0;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** helper routine to create an instance of radial density function */
const math::IFunction* createRadialDiskFnc(const DiskParam& params) {
    if(params.scaleLength<=0)
        throw std::invalid_argument("Disk scale length cannot be <=0");
    if(params.innerCutoffRadius<0)
        throw std::invalid_argument("Disk inner cutoff radius cannot be <0");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0)
        return new DiskDensityRadialExp(params.surfaceDensity, params.scaleLength);
    else
        return new DiskDensityRadialRichExp(params);
}

/** helper routine to create an instance of vertical density function */
const math::IFunction* createVerticalDiskFnc(const DiskParam& params) {
    if(params.scaleHeight>0)
        return new DiskDensityVerticalExp(params.scaleHeight);
    if(params.scaleHeight<0)
        return new DiskDensityVerticalIsothermal(-params.scaleHeight);
    else
        return new DiskDensityVerticalThin();
}

double DiskResidual::densityCyl(const coord::PosCyl &pos) const
{
    if(pos.z==0) return 0;
    double h, H, Hp, F, f, fp, fpp, r=hypot(pos.R, pos.z);
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    radialFnc  ->evalDeriv(pos.R, &F);
    return (F-f)*h - 2*fp*(H+pos.z*Hp)/r - fpp*H;
}

double DiskAnsatz::densityCyl(const coord::PosCyl &pos) const
{
    double h, H, Hp, f, fp, fpp, r=hypot(pos.R, pos.z);
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    return f*h + (pos.z!=0 ? 2*fp*(H+pos.z*Hp)/r : 0) + fpp*H;
}

void DiskAnsatz::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r=hypot(pos.R, pos.z);
    double h, H, Hp, f, fp, fpp;
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    f*=4*M_PI; fp*=4*M_PI; fpp*=4*M_PI;
    double Rr=pos.R/r, zr=pos.z/r;
    if(r==0) { Rr=0; zr=0; r=1e-100; }
    if(potential) {
        *potential = f*H;
    }
    if(deriv) {
        deriv->dR = H*fp*Rr;
        deriv->dz = H*fp*zr + Hp*f;
        deriv->dphi=0;
    }
    if(deriv2) {
        deriv2->dR2 = H*(fpp*Rr*Rr + fp*zr*zr/r);
        deriv2->dz2 = H*(fpp*zr*zr + fp*Rr*Rr/r) + 2*fp*Hp*zr + f*h;
        deriv2->dRdz= H*Rr*zr*(fpp - fp/r) + fp*Hp*Rr;
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

//----- spheroid density -----//
SpheroidDensity::SpheroidDensity (const SphrParam &_params) :
    BaseDensity(), params(_params)
{
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Spheroid scale radius cannot be <=0");
    if(params.axisRatio<=0)
        throw std::invalid_argument("Spheroid axis ratio cannot be <=0");
    if(params.outerCutoffRadius<0)
        throw std::invalid_argument("Spheroid outer cutoff radius cannot be <0");
};

double SpheroidDensity::densityCyl(const coord::PosCyl &pos) const
{
    double m   = hypot(pos.R, pos.z/params.axisRatio);
    double m0  = m/params.scaleRadius;
    double rho = params.densityNorm;
    if(params.gamma==0.5)rho /= sqrt(m0); else
    if(params.gamma==1.) rho /= m0;       else 
    if(params.gamma==2.) rho /= m0*m0;    else
    if(params.gamma!=0.) rho /= pow(m0, params.gamma);
    m0 += 1;
    const double beg = params.beta-params.gamma;
    if(beg==1.) rho /= m0;       else
    if(beg==2.) rho /= m0*m0;    else
    if(beg==3.) rho /= m0*m0*m0; else
    rho /= pow(m0,beg);
    if(params.outerCutoffRadius)
        rho *= exp(-pow_2(m/params.outerCutoffRadius));
    return rho;
}

//----- multipole potential -----//
const int N =GALPOT_LMAX/2+1; // number of multipoles
const int N2=3*N/2;           // number of grid point for cos[theta] in [0,1]
const int N4=5*N/2;           // number of points used to integrate over cos[theta]

typedef WD::Vector<double,N> DBN;

void Multipole::AllocArrays()
{
    X[0] = new double[K[0]];
    X[1] = new double[K[1]];
    WD::Alloc2D(Y[0],K);
    WD::Alloc2D(Y[1],K);
    WD::Alloc2D(Y[2],K);
    WD::Alloc2D(Z[0],K);
    WD::Alloc2D(Z[1],K);
    WD::Alloc2D(Z[2],K);
    WD::Alloc2D(Z[3],K);
}

Multipole::Multipole(const BaseDensity& source_density,
                     const double rmin, const double rmax,
                     const int num_grid_points,
                     const double _gamma, const double _beta)
{
    if((source_density.symmetry() & ST_AXISYMMETRIC) != ST_AXISYMMETRIC)
        throw std::invalid_argument("Error in Multipole expansion: source density must be axially symmetric");
    K[0] = num_grid_points;
    K[1] = N2;
    AllocArrays();
    setup(source_density, rmin, rmax, _gamma, _beta);
}

Multipole::~Multipole()
{
    delete[] X[0]; 
    delete[] X[1];
    WD::Free2D(Y[0]);
    WD::Free2D(Y[1]);
    WD::Free2D(Y[2]);
    WD::Free2D(Z[0]);
    WD::Free2D(Z[1]);
    WD::Free2D(Z[2]);
    WD::Free2D(Z[3]);
}

void Multipole::setup(const BaseDensity& source_density,
                      const double ri, const double ra,
                      const double g, const double b)
{
  Rmin = ri;
  Rmax = ra;
  gamma= g; 
  beta = b;
  lRmin= log(Rmin);
  lRmax= log(Rmax); 
  g2   = 2.-gamma;

  const    DBN    Zero=DBN(0.);
  const    double half=0.5, three=3., sixth=1./6.,
    dlr =(lRmax-lRmin)/double(K[0]-1);
  int    i,l,k,ll,lli1;
  double dx,dx2,xl_ll,xh_ll,risq,ril2,dP;
  DBN    A[4],P2l,dP2l;

  DBN    EX;
  //
  // 0  check for inconsistencies in input
  //
  if(beta>0. && beta<3.) {
    //std::cerr<<" Warning: beta= "<<beta
    //     <<" unsuitable for Multipole expansion;"
    //     <<" we'll take beta=3.2\n";
    beta=3.2;
  }
  //
  // 1  compute expansion of the density
  //
  double
    *ct   = new double[N4],
    *st   = new double[N4],
    *wi   = new double[N4],
    *r    = new double[K[0]];
  DBN
    *W    = new DBN   [N4],
    *rhol = new DBN   [K[0]],
    *rhl2 = new DBN   [K[0]];
  //
  // 1.1 set points and weights for integration over cos(theta)
  //
  WD::GaussLegendre(ct,wi,N4);
  for(i=0; i<N4; i++) {
    ct[i] = 0.5 * (ct[i]+1.);
    st[i] = sqrt(1.-ct[i]*ct[i]);
    wi[i] = 0.5 * wi[i];
    WD::LegendrePeven(W[i],ct[i]);
    W[i] *= wi[i] * 4*M_PI;
  }
  //
  // 1.2 integrate over cos(theta)
  //
  for(k=0; k<K[0]; k++) {
    X[0][k] = k<K[0]-1? lRmin+dlr*k : lRmax;  // v0.7
    r[k]    = exp(X[0][k]);
    rhol[k] = 0.;
    for(i=0; i<N4; i++)
        rhol[k] += W[i] * source_density.density(coord::PosCyl(r[k]*st[i], r[k]*ct[i], 0));
  }
  delete[] ct;
  delete[] st;
  delete[] wi;
  delete[] W;
  //
  // 1.3 establish spline in r needed for integration
  //
  WD::spline(r,rhol,K[0],(-gamma/r[0])*rhol[0],Zero,rhl2,0,1);
  //
  // 2. compute potential's expansion
  //
  DBN
    *P1   = new DBN[K[0]],
    *P2   = new DBN[K[0]],
    *Phil = new DBN[K[0]],
    *dPhl = new DBN[K[0]];
  //
  // 2.1 set P1[k][l] r[k]^(-1-2l) = Int[rho_2l(x,l) x^(2l+2), {x,0,r[k]}]
  //
  //     for r < Rmin we take  rho_2l proportional r^-gamma
  //
  risq  = Rmin*Rmin;
  for(l=0; l<N; l++) {
    P1[0][l] = rhol[0][l] * risq / double(2*l+3-gamma);
    EX[l]    = exp(-(1+2*l)*dlr);
  }
  for(k=0; k<K[0]-1; k++) {
    dx   = r[k+1]-r[k];
    dx2  = dx*dx;
    A[0] = r[k+1]*rhol[k] - r[k]*rhol[k+1] + sixth*r[k]*r[k+1] *
      ( (r[k+1]+dx)*rhl2[k] - (r[k]-dx)*rhl2[k+1] );
    A[1] = rhol[k+1]-rhol[k]
      + sixth * ( (dx2-three*r[k+1]*r[k+1]) * rhl2[k]
          -(dx2-three*r[k]*r[k])     * rhl2[k+1] );
    A[2] = half  * (r[k+1]*rhl2[k] - r[k]*rhl2[k+1]);
    A[3] = sixth * (rhl2[k+1]-rhl2[k]);
    for(l=0,ll=2; l<N; l++,ll+=2) {
      xl_ll = r[k]*EX(l);
      xh_ll = r[k+1];
      for(i=0,lli1=ll+1,dP=0.; i<4; i++,lli1++) {
        xl_ll*= r[k];
        xh_ll*= r[k+1];
        dP   += A[i](l) * (xh_ll - xl_ll) / lli1;
      }
      P1[k+1][l] = EX(l) * P1[k](l) + dP / dx;
    }
  }
  //
  // 2.2 set P2[k][l] = r[k]^(2l) Int[rho_2l(x,l) x^(1-2l), {x,r[k],Infinity}]
  //
  //     for r > Rmax we take  rho_2l proportional r^-beta if beta>0
  //                                  = 0                  if beta<=0
  //
  if(beta>0.) {
    risq  = Rmax*Rmax;
    for(l=0; l<N; l++) {
      P2[K[0]-1][l] = rhol[K[0]-1][l] * risq / double(beta+2*l-2);
      EX[l] = exp(-2*l*dlr);
    }
  } else {
    P2[K[0]-1] = 0.;
    for(l=0; l<N; l++)
      EX[l] = exp(-2*l*dlr);
  }
  for(k=K[0]-2; k>=0; k--) {
    risq = r[k]*r[k];
    dx   = r[k+1]-r[k];
    dx2  = dx*dx;
    A[0] = r[k+1]*rhol[k] - r[k]*rhol[k+1] + sixth*r[k]*r[k+1] *
      ( (r[k+1]+dx)*rhl2[k] - (r[k]-dx)*rhl2[k+1] );
    A[1] = rhol[k+1]-rhol[k]
      + sixth * ( (dx2-three*r[k+1]*r[k+1]) * rhl2[k]
          -(dx2-three*r[k]*r[k])     * rhl2[k+1] );
    A[2] = half  * (r[k+1]*rhl2[k] - r[k]*rhl2[k+1]);
    A[3] = sixth * (rhl2[k+1]-rhl2[k]);
    for(l=0,ll=1,ril2=1.; l<N; l++,ll-=2,ril2*=risq) {
      xl_ll = r[k];
      xh_ll = r[k+1]*EX(l);
      for(i=0,lli1=ll+1,dP=0.; i<4; i++,lli1++) {
        xl_ll *= r[k];
        xh_ll *= r[k+1];
        if(lli1) dP += A[i](l) * (xh_ll - xl_ll) / lli1;
        else     dP += A[i](l) * ril2 * dlr;
      }
      P2[k][l] = EX(l) * P2[k+1](l) + dP / dx;
    }
  }
  //
  // 2.3 put together the Phi_2l(r) and dPhi_2l(r)/dlog[r]
  //
  for(k=0; k<K[0]; k++)
    for(l=ll=0; l<N; l++,ll+=2) {
      Phil[k][l] =-P1[k](l) - P2[k](l);                   // Phi_2l
      dPhl[k][l] = (ll+1)*P1[k](l) - ll*P2[k](l);         // dPhi_2l/dlogr
    }
  if(gamma<2)
    Phi0 = Phil[0](0) - dPhl[0](0) / g2;
  delete[] r;
  delete[] rhol;
  delete[] rhl2;
  delete[] P1;
  delete[] P2;
  //
  // 3. establish L_circ(R) on the logarithmic grid (skipped)
  //
  // 4.  Put potential and its derivatives on a 2D grid in log[r] & cos[theta]
  //
  // 4.1 set linear grid in theta
  //
  for(i=0; i<N2; i++) 
    X[1][i] = double(i) / double(N2-1);
  //
  // 4.2 set dPhi/dlogr & dPhi/dcos[theta] 
  //
  for(i=0; i<N2; i++) {
    WD::dLegendrePeven(P2l,dP2l,X[1][i]);
    for(k=0; k<K[0]; k++) {
      Y[0][k][i] = Phil[k] * P2l;         // Phi
      Y[1][k][i] = dPhl[k] * P2l;         // d Phi / d logR
      Y[2][k][i] = Phil[k] * dP2l;        // d Phi / d cos(theta)
    }
  }
  delete[] Phil;
  delete[] dPhl;
  //
  // 4.3 establish 2D Pspline of Phi in log[r] & cos[theta]
  //
  WD::Pspline2D(X,Y,K,Z);
}

void Multipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r=hypot(pos.R, pos.z);
    double ct=pos.z/r, st=pos.R/r;  // cos(theta), sin(theta)
    if(r==0) {ct=st=0;}
    double Xi[2];
    double lr=log(r), Phi;
    Xi[0] = fmin(lRmax,fmax(lRmin,lr));
    Xi[1] = fabs(ct);
    double der[2];      // first derivatives from spline by its arguments (log r, z/r)
    double der2a[2],der2b[2]; 
    double* der2[]={der2a,der2b};  // second derivatives from spline, if required
    if(deriv2)
        Phi = WD::Psplev2D(X,Y,Z,K,Xi,der,der2);
    else
        Phi = WD::Psplev2D(X,Y,Z,K,Xi,der);
    if(deriv) der[1]*= sign(ct);
    if(deriv2) der2[0][1]*= sign(ct);
    if(lr < lRmin) {
        if(g2>0.) {
            Phi = (Phi-Phi0)*exp(g2*(lr-Xi[0]));
            if(deriv) der[0] = g2*Phi;
            if(deriv2) der2[0][0] = g2*g2*Phi;
            Phi+= Phi0;
        } else if(g2==0.) {
            if(deriv) der[0] = Phi/lRmin;
            if(deriv2) der2[0][0] = 0.;
            Phi*= lr/lRmin;
        } else {
            Phi*= exp(g2*(lr-Xi[0]));
            if(deriv) der[0] = g2*Phi;
            if(deriv2) der2[0][0] = g2*g2*Phi;
        }
    } else if(lr > lRmax) {
        Phi *= Rmax/r;
        if(deriv) der[0] =-Phi;
        if(deriv2) der2[0][0] = Phi;
    }
    if(potential)
        *potential=Phi;
    if(r==0) r=1e-100;  // safety measure
    if(deriv) {
        deriv->dR=(der[0]-der[1]*ct)*st/r;
        deriv->dz=(der[0]*ct+der[1]*st*st)/r;
        deriv->dphi=0;
    }
    if(deriv2) {
        double z2=ct*ct, R2=st*st;
        deriv2->dR2 = (der2[0][0]*R2 + der2[1][1]*R2*z2 - der2[0][1]*2*R2*ct
                      + der[0]*(z2-R2) + der[1]*(2*R2-z2)*ct) / (r*r);  // d2/dR2
        deriv2->dz2 = (der2[0][0]*z2 + der2[1][1]*R2*R2 + der2[0][1]*2*R2*ct
                      + der[0]*(R2-z2) - der[1]*3*R2*ct) / (r*r);       // d2/dz2
        deriv2->dRdz= (der2[0][0]*ct*st - der2[1][1]*ct*st*R2 + der2[0][1]*st*(R2-z2)
                      - der[0]*2*ct*st + der[1]*st*(2*z2-R2)) / (r*r);  // d2/dRdz
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

//----- GalaxyPotential refactored into a Composite potential -----//
const potential::BasePotential* createGalaxyPotential(
    const std::vector<DiskParam>& DiskParams, 
    const std::vector<SphrParam>& SphrParams)
{
    // keep track of inner/outer slopes of spheroid density profiles..
    double gamma=0, beta=1.e3;
    // ..and the length scales of all components
    double lengthMin=1e100, lengthMax=0;
    
    // first create a set of density components for the multipole
    // (all spheroids and residual part of disks)
    std::vector<const BaseDensity*> componentsDens;
    for(size_t i=0; i<DiskParams.size(); i++) {
        componentsDens.push_back(new DiskResidual(DiskParams[i]));
        lengthMin = fmin(lengthMin, DiskParams[i].scaleLength);
        lengthMax = fmax(lengthMax, DiskParams[i].scaleLength);
        if(DiskParams[i].innerCutoffRadius>0)
            lengthMin = fmin(lengthMin, DiskParams[i].innerCutoffRadius);
    }
    for(size_t i=0; i<SphrParams.size(); i++) {
        componentsDens.push_back(new SpheroidDensity(SphrParams[i]));
        gamma = fmax(gamma, SphrParams[i].gamma);
        lengthMin = fmin(lengthMin, SphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, SphrParams[i].scaleRadius);
        if(SphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, SphrParams[i].outerCutoffRadius);
        else
            beta = fmin(beta, SphrParams[i].beta);
    }
    // if cutoff radius is provided or there are no spheroidal components, outer slope is undetermined
    if(beta==1.e3) beta=-1;
    if(componentsDens.size()==0)
        throw std::invalid_argument("Empty parameters in GalPot");
    const CompositeDensity dens(componentsDens);

    // create multipole potential from this combined density
    double rmin = GALPOT_RMIN * lengthMin;
    double rmax = GALPOT_RMAX * lengthMax;
    const BasePotential* mult=new Multipole(dens, rmin, rmax, GALPOT_NRAD, gamma, beta);

    // now create a composite potential from the multipole and non-residual part of disk potential
    std::vector<const BasePotential*> componentsPot;
    componentsPot.push_back(mult);
    for(size_t i=0; i<DiskParams.size(); i++)  // note that we create another class of objects than above
        componentsPot.push_back(new DiskAnsatz(DiskParams[i]));
    return new CompositeCyl(componentsPot);
}

} // namespace
