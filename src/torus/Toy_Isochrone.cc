/*******************************************************************************
*                                                                              *
* Toy_Isochrone.cc                                                             *
*                                                                              *
* C++ code written by Walter Dehnen, 1994-96,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail:  paul@astro.lu.se                                                    *
* github:  https://github.com/PaulMcMillan-Astro/Torus                         *
*                                                                              *
*******************************************************************************/

#include "Toy_Isochrone.h"
#include <cmath>

namespace torus{

inline double pow_2(double x) {return x*x;}
inline double pow_3(double x) {return x*x*x;}
    
////////////////////////////////////////////////////////////////////////////////
// class ToyIsochrone


void ToyIsochrone::psifunc(const double p, double& f, double& df) const
{
    psi  = p;
    spsi = sin(psi);
    cpsi = cos(psi);
    f    = psi-eps*spsi-tr;
    df   = 1.-eps*cpsi;
}

void ToyIsochrone::psisolve() const
// solves  tr = psi - eps * sin(psi);    for psi.
{
    const double tol=1.e-8;
    psi = tr;
    if(fabs(eps)<tol || fabs(tr)<tol || fabs(tr-Pi)<tol || fabs(tr-TPi)<tol) {
        spsi = sin(psi);
        cpsi = cos(psi);
        return;
    }
#if 0
    double ps=psi;
    if(ps<Pi) ps = rtsafe(this,&ToyIsochrone::psifunc,ps,Pi,tol);
	else  ps = rtsafe(this,&ToyIsochrone::psifunc,Pi,ps,tol);
    if(ps!=psi) {
        psi  = ps;
        spsi = sin(psi);
        cpsi = cos(psi);
    }
#else
    // modification of the original code from Eugene Vasiliev
    bool signpsi=tr>Pi;
    psi=Pi/2+(Pi/8)/eps*(sqrt(Pi*Pi-8*Pi*eps+16*eps*(eps+(signpsi?2*Pi-tr:tr)))-Pi);
    if(signpsi) psi=2*Pi-psi;
    double deltapsi=0;
    int niter=0;
    do {  // Newton's method for solving the Kepler equation (f=0)
        spsi=sin(psi);
        cpsi=cos(psi);
        double f=psi-eps*spsi-tr;
        double df=1.-eps*cpsi;
        deltapsi=-f/df;
        deltapsi=-f/(df+0.5*deltapsi*eps*spsi);  // refinement using second derivative (thanks to A.Gurkan)
        psi+=deltapsi;
        niter++;
    } while(fabs(deltapsi)>tol && niter<42);
#endif
    
}

double ToyIsochrone::catan(const double& aa) const
// Evaluates the function atan(aa tan(psi/2)).
// using tan(x/2) = sin(x)/[1+cos(x)]
{

// Three ranges.
    if(cpsi>=0. && spsi>=0.)                //      0 > psi > Pi/2
        return atan2(aa*spsi,1.+cpsi); 
    else if(cpsi<0. && spsi>=0)             //   Pi/2 > psi > Pi
	return Pih-atan2(1.+cpsi,spsi*aa);
    else if(cpsi<0.) {                      //     Pi > psi > 3Pi/2
	double temp=atan2(1.+cpsi,spsi*aa);
	if(temp>0) return Pi3h-temp;
	return Pih-temp;
    } else                                  //  3Pi/2 > psi > 2Pi
        return Pi+atan2(aa*spsi,1.+cpsi); 
}

double ToyIsochrone::wfun(const double& fac) const
{
    if(fac==0.) return Pih; // radial orbit <=> e==1.
    else {
      if((1.-e)==0.) return catan(1.e99) + 
		       catan(sqrt((a*(1.+e)+2.)/(a*(1.-e)+2.)))*fac;
      else return   catan(sqrt((1.+e)/(1.-e)))
	       + catan(sqrt((a*(1.+e)+2.)/(a*(1.-e)+2.)))*fac;
    }
}

double ToyIsochrone::wfundw(const double& fac, double& dwdpsi) const
{
    if(fac==0.) { // radial orbit <=> e==1.
	dwdpsi = 0.;
	return Pih;
    } else {
        double a1   = sqrt( (1.+e) / (1.-e) );
        double a2   = sqrt( (a*(1.+e)+2.) / (a*(1.-e)+2.) );
        double cps1 = 1.+cpsi;
        double sps1 = 1.-cpsi;
        dwdpsi      = 1./(cps1/a1+sps1*a1) + fac/(cps1/a2+sps1*a2);
        return        catan(a1) + catan(a2)*fac;
    }
}

double ToyIsochrone::Fint(double p, double q, double Pc, double Qc) const
// Evaluates int_0^psi (Pc+Qc cos[t])/ (p+q cos[t])^2  dt.
{
    double subint = 2./sqrt(p*p-q*q)*catan(sqrt((p-q)/(p+q)));
    return ((p*Qc-q*Pc)*spsi/(p+q*cpsi) + (p*Pc-q*Qc)*subint)/(p*p-q*q);
}

////////////////////////////////////////////////////////////////////////////////

vec4 ToyIsochrone::lower_bounds(const double x, const double v) const
{
    return -upper_bounds(x,v);
}

vec4 ToyIsochrone::upper_bounds(const double x, const double v) const
{
    double a[4] = {sqrt(1.e5*x)*v, sqrt(1.e3*x), 1.e2*x*v, 1.*x};
    return vec4((const double*)a); 
}

void ToyIsochrone::Set()
{
    M = gamma*gamma;
    b = beta*beta;
    if(b==0.) throw std::runtime_error("Torus Error -2: ToyIsochrone: scale radius = 0 not allowed");
    if(M==0.) throw std::runtime_error("Torus Error -2: ToyIsochrone: mass = 0 not allowed");
    if(b>0. && M>0.) {
	sMb  = sqrt(M*b);
	sMob = sMb/b;
	jp   = fabs(Lz)/sMb;
    }
    derivs_ok = false;
}

void ToyIsochrone::dump() const
{
    cerr<< "\ntest output from Iso:"
        << "\njr,jt,tr,tt     = " << jr <<','<<   jt<<','<<   tr<<','<< tt
        << "\nr,th,p_r,p_th   = " <<  r <<','<<   th<<','<<   pr<<','<< pt
        << "\na,e,eps,u       = " <<  a <<','<<    e<<','<<  eps<<','<< u
	<< "\nH,wr,wt,at      = " <<  H <<','<<   wr<<','<<   wt<<','<< at
	<< "\nwh,sq,sGam,chi  = " << wh <<','<<   sq<<','<< sGam<<','<< chi
	<< "\npsi,spsi,cpsi   = " << psi<<','<< spsi<<','<< cpsi<<'\n'; 
}

ToyIsochrone::ToyIsochrone()
{  
    IsoPar p=1.f;
    set_parameters(p);
    Set();
}

ToyIsochrone::ToyIsochrone(const IsoPar& p)
{ 
    set_parameters(p);
    Set();
}

ToyIsochrone::ToyIsochrone(const ToyIsochrone& I): gamma(I.gamma),beta(I.beta),Lz(I.Lz),r0(I.r0)
{ 
    Set();
}

PSPD ToyIsochrone::ForwardWithDerivs(const PSPD& JT, double dQdT[2][2]) const
// Transforms action-angle variables (JT) to phasespace co-ordinates (r,theta)
// and their conjugated momenta, the derivs of (r,theta) w.r.t. the angle vars
// are also returned; theta is latitude rather than polar angle.
{
  derivs_ok = true;
    double e2,schi,cchi,csth;
    static   double fac, dw;

// Extract and scale the actions and angles.
    jr = double(JT(0)) / sMb;
    jt = double(JT(1)) / sMb;
    tr = double(JT(2));
    tt = double(JT(3));
    if(std::isnan(tr) || std::isinf(tr) || fabs(tr)>INT_MAX) 
      tr = 0.; // just in case  
    tr = math::wrapAngle(tr);
    if(jr<0. || jt<0.|| this->jp<0.) {
        throw std::runtime_error("Torus Error -2: ToyIsochrone: negative action(s)"); 
	return PSPD(0.);
    }
// Set up auxilliary variables independent of the angles tr and tt.
    at  = this->jp+jt;
    sGam= (this->jp==0.) ? 1. : sqrt(1.-pow_2(this->jp/at));
    sq  = hypot(2.,at);
    fac = at/sq;
    HH  = 2./(2.*jr+at+sq);
    H2  = HH*HH;
    H   =-0.5*H2;
    a   = 1./H2-1.;
    if(at==0.) e = 1.;
    else {
        e2    = 1. - pow_2(at/a)/H2;
        e     = (e2>0) ? sqrt(e2) : 0;
        if(e2<-1.e-3) throw std::runtime_error("Torus Error -2: ToyIsochrone: bad e in ForwardWithDerivs");
    }
    ae  = a*e;
    eps = ae*H2;
    wr  = H2*HH;
    wt0r= 0.5*(1.+fac);
    wt  = wt0r*wr;
// Solve for the relevant other variables.
    psisolve();           // gives psi, cpsi, spsi
    wh  = wfundw(fac,dw); // fac is input, dw is output
    chi = tt-wt0r*tr+wh;
    u   = a*(1.-e*cpsi);
// Calculate co-ordinates and the conjugate momenta.
    schi= sin(chi);
    cchi= cos(chi);
    r   = sqrt(u*(u+2.));
    csth= cos(th = asin(sGam*schi));
    pr  = HH*ae*spsi/r;
    pt  = (at==0) ? 0. : at*sGam*cchi/csth;
// Calculate derivatives of the coordinates
    dQdT[0][0] = ae*spsi/r/H2*b;
    dQdT[0][1] = 0.;
    dQdT[1][1] = sGam*cchi/csth;
    dQdT[1][0] = dQdT[1][1] * (dw/(1.-eps*cpsi)-wt0r);
// re-scale and return
    return PSPD(b*r+r0, th, pr*sMob, pt*sMb);
}

PSPD ToyIsochrone::ForwardWithDerivs(const PSPD& JT, double dQdT[2][2], 
			       double dPdT[2][2] ) const
// Transforms action-angle variables (JT) to phasespace co-ordinates (r,theta)
// and their conjugated momenta, the derivs of (r,theta,pr,ptheta) w.r.t. the 
//angle vars are also returned; theta is latitude rather than polar angle.
{
    derivs_ok = true;
    double e2,schi,cchi,csth,dchidtr,ir,icsth;
    static   double fac, dw;

// Extract and scale the actions and angles.
    jr = double(JT(0)) / sMb;
    jt = double(JT(1)) / sMb;
    tr = double(JT(2));
    tt = double(JT(3));
    if(std::isnan(tr) || std::isinf(tr) || fabs(tr)>INT_MAX) 
      tr = 0.; // just in case  
    tr = math::wrapAngle(tr);
    if(jr<0. || jt<0.|| this->jp<0.) {
       throw std::runtime_error("Torus Error -2: ToyIsochrone: negative action(s)"); 
	return PSPD(0.);
    }
// Set up auxilliary variables independent of the angles tr and tt.
    at  = this->jp+jt;
    sGam= (this->jp==0.) ? 1. : sqrt(1.-pow_2(this->jp/at));
    sq  = hypot(2.,at);
    fac = at/sq;
    HH  = 2./(2.*jr+at+sq);
    H2  = HH*HH;
    H   =-0.5*H2;
    a   = 1./H2-1.;
    if(at==0.) e = 1.;
    else {
        e2    = 1. - pow_2(at/a)/H2;
        e     = (e2>0) ? sqrt(e2) : 0;
        if(e2<-1.e-3) throw std::runtime_error("Torus Error -2: ToyIsochrone: bad e in ForwardWithDerivs");
    }
    ae  = a*e;
    eps = ae*H2;
    wr  = H2*HH;
    wt0r= 0.5*(1.+fac);
    wt  = wt0r*wr;
// Solve for the relevant other variables.
    psisolve();           // gives psi, cpsi, spsi
    wh  = wfundw(fac,dw); // fac is input, dw is output (dwh/dpsi)
    chi = tt-wt0r*tr+wh;
    u   = a*(1.-e*cpsi);
// Calculate co-ordinates and the conjugate momenta.
    schi = sin(chi);
    cchi = cos(chi);
    r    = sqrt(u*(u+2.));
    ir   = 1./r;
    csth = cos(th = asin(sGam*schi));
    icsth= 1./csth; 
    pr   = HH*ae*spsi*ir;
    pt   = (at==0) ? 0. : at*sGam*cchi/csth;

// Calculate derivatives of the coordinates
    dQdT[0][0] = ae*spsi*ir/H2*b;
    dQdT[0][1] = 0.;
    dQdT[1][1] = sGam*cchi*icsth;
    dQdT[1][0] = dQdT[1][1] * (dchidtr = dw/(1.-eps*cpsi)-wt0r);

    dPdT[0][0] = HH*ae*ir*((-1.+1./(1.-eps*cpsi))/eps-spsi*ir*dQdT[0][0]/b)*sMob;
    dPdT[0][1] = 0.;
    dPdT[1][1] = (at==0) ? 0. : at*sGam*icsth*
                                (sGam*schi*icsth*cchi*dQdT[1][1]-schi)*sMb;
    dPdT[1][0] = (at==0) ? 0. : at*sGam*icsth*(sGam*schi*icsth*cchi*dQdT[1][0] 
					       - schi*dchidtr)*sMb;
// re-scale and return
    return PSPD(b*r+r0, th, pr*sMob, pt*sMb);
}


void ToyIsochrone::Derivatives(double dQPdJ[4][2]) const
// Calculates the derivatives of the spherical phasespace co-ordinates w.r.t.
// Jr,Jt. A call of one of Forward() or ForwardWithDerivs() has to be preceeded.
{
  if(!derivs_ok)
      throw std::runtime_error("Torus Error -2:  `ToyIsochrone::Derivatives()' called without For/Backward");

    double Hx,wrx,wtx,ax,axa,ex,exe,axe,exa,psix,ux,xGam,wx,chix,HxH,
		    schi,cchi,sith,csth,fac,cGam;
    schi=sin(chi);
    cchi=cos(chi);
    sith=sin(th);
    csth=cos(th);
    fac = at/HH;
    cGam= sqrt(1.-sGam*sGam);
// w. r. t. Jr:
    Hx  = wr;
    wrx = 1.5*wr/H;
    wtx = wt*wrx;
    wrx*= wr;
    HxH = 0.5*Hx/H;
    ax  = HxH/H;
    axa = ax/a;
    ex  = pow_2(at/a) / (e*H2) * (HxH+axa);
    exe = ex/e;
    axe = ax*e;
    exa = ex*a;
    psix= (exa+H2*axe)*spsi/(u+1.);
    ux  = u*axa-cpsi*exa+ae*spsi*psix;
    wx  =-wh*HxH + fac*(u+1.)/(r*r)*psix
         -0.5*fac*(  Fint(a,-ae,ax,-axe-exa) + Fint(a+2.,-ae,ax,-axe-exa));
    chix= tr/wr * (wrx*wt/wr - wtx) + wx;
    dQPdJ[0][0] = (u+1.)/r*ux;
    dQPdJ[1][0] = sGam*cchi*chix/csth;
    dQPdJ[2][0] = pr*(HxH -dQPdJ[0][0]/r +axa +exe) + HH*ae*cpsi/r*psix;
    dQPdJ[3][0] = at*sGam*(-schi/csth*chix +cchi*sith/pow_2(csth)*dQPdJ[1][0]);
    dQPdJ[0][0]/= sMob;
    dQPdJ[1][0]/= sMb;
    dQPdJ[2][0]/= b;
// w.r.t. Jt:
    Hx  = wt;
    wrx = wtx;
    wtx = wt0r*wrx + 2.*wr/pow_3(sq);
    HxH = 0.5*Hx/H;
    ax  = HxH/H;
    axa = ax/a;
    ex  = pow_2(at/a) / (e*H2) * (HxH+axa-1./at);
    exe = ex/e;
    axe = ax*e;
    exa = ex*a;
    psix= (exa+H2*axe)*spsi/(u+1.);
    ux  = u*axa-cpsi*exa+ae*spsi*psix;
    axe = ax*e;
    exa = ex*a;
    wx  = wh*(1./at-HxH) + fac*(u+1.)/(r*r)*psix
         -0.5*fac*(  Fint(a,-ae,ax,-axe-exa) + Fint(a+2.,-ae,ax,-axe-exa));
    chix= tr/wr * (wrx*wt/wr - wtx) + wx;
    if(jt==0.) {
	dQPdJ[1][1] = 0.;
	dQPdJ[3][1] = 100.;		// actually, it's infinite
    } else {
        xGam= cGam/(sGam*at);
        dQPdJ[1][1] = (sGam*cchi*chix+schi*cGam*xGam)/csth;
        dQPdJ[3][1] = pt/at + at *( cchi*cGam/csth*xGam + sGam 
		      * (-schi/csth*chix +cchi*sith/pow_2(csth)*dQPdJ[1][1]));
    }
    dQPdJ[0][1] = (u+1.)/r*ux;
    dQPdJ[2][1] = pr*(HxH -dQPdJ[0][1]/r +axa +exe) + HH*ae*cpsi/r*psix;
    dQPdJ[0][1]/= sMob;
    dQPdJ[1][1]/= sMb;
    dQPdJ[2][1]/= b;
}

void ToyIsochrone::Derivatives(double dQPdJ[4][2], Pdble dQPdA[4]) const
// Calculates the derivatives of the spherical phasespace co-ordinates w.r.t.
// Jr,Jt and gamma,beta,Lz,r0. A call of one of Forward() or ForwardWithDerivs()
// has to be preceeded.
{
    if(!derivs_ok)
      throw std::runtime_error("Torus Error -2:  `ToyIsochrone::Derivatives()' called without For/Backward");

    double Hx,wrx,wtx,ax,axa,ex,exe,axe,exa,psix,ux,xGam,wx,chix,HxH,
		    schi,cchi,sith,csth,fac,cGam,temp;
    schi=sin(chi);
    cchi=cos(chi);
    sith=sin(th);
    csth=cos(th);
    fac = at/HH;
    cGam= sqrt(1.-sGam*sGam);
// w. r. t. Jr:
    Hx  = wr;
    wrx = 1.5*wr/H;
    wtx = wt*wrx;
    wrx*= wr;
    HxH = 0.5*Hx/H;
    ax  = HxH/H;
    axa = ax/a;
    ex  = pow_2(at/a) / (e*H2) * (HxH+axa);
    exe = ex/e;
    axe = ax*e;
    exa = ex*a;
    psix= (exa+H2*axe)*spsi/(u+1.);
    ux  = u*axa-cpsi*exa+ae*spsi*psix;
    wx  =-wh*HxH + fac*(u+1.)/(r*r)*psix
         -0.5*fac*(  Fint(a,-ae,ax,-axe-exa) + Fint(a+2.,-ae,ax,-axe-exa));
    chix= tr/wr * (wrx*wt/wr - wtx) + wx;
    dQPdJ[0][0] = (u+1.)/r*ux;
    dQPdJ[1][0] = sGam*cchi*chix/csth;
    dQPdJ[2][0] = pr*(HxH -dQPdJ[0][0]/r +axa +exe) + HH*ae*cpsi/r*psix;
    dQPdJ[3][0] = at*sGam*(-schi/csth*chix +cchi*sith/pow_2(csth)*dQPdJ[1][0]);
    dQPdJ[0][0]/= sMob;
    dQPdJ[1][0]/= sMb;
    dQPdJ[2][0]/= b;
// w.r.t. Jt:
    Hx  = wt;
    wrx = wtx;
    wtx = wt0r*wrx + 2.*wr/pow_3(sq);
    HxH = 0.5*Hx/H;
    ax  = HxH/H;
    axa = ax/a;
    ex  = pow_2(at/a) / (e*H2) * (HxH+axa-1./at);
    exe = ex/e;
    axe = ax*e;
    exa = ex*a;
    psix= (exa+H2*axe)*spsi/(u+1.);
    ux  = u*axa-cpsi*exa+ae*spsi*psix;
    axe = ax*e;
    exa = ex*a;
    wx  = wh*(1./at-HxH) + fac*(u+1.)/(r*r)*psix
          -0.5*fac*(  Fint(a,-ae,ax,-axe-exa) + Fint(a+2.,-ae,ax,-axe-exa));
    chix= tr/wr * (wrx*wt/wr - wtx) + wx;
    if(jt==0.) {
	dQPdJ[1][1] = 0.;
	dQPdJ[3][1] = 100.;		// actually, it's infinite
    } else {
        xGam= cGam/(sGam*at);
        dQPdJ[1][1] = (sGam*cchi*chix+schi*cGam*xGam)/csth;
        dQPdJ[3][1] = pt/at + at *( cchi*cGam/csth*xGam + sGam 
		      * (-schi/csth*chix +cchi*sith/pow_2(csth)*dQPdJ[1][1]));
    }
    dQPdJ[0][1] = (u+1.)/r*ux;
    dQPdJ[2][1] = pr*(HxH -dQPdJ[0][1]/r +axa +exe) + HH*ae*cpsi/r*psix;
    dQPdJ[0][1]/= sMob;
    dQPdJ[1][1]/= sMb;
    dQPdJ[2][1]/= b;
// w.r.t. Jp = fabs(Lz):
    if(jt==0.) {
	dQPdA[1][2] = 0.;
	dQPdA[3][2] = 100.;		// actually, it's infinite
    } else {
        xGam= (cGam-1.)/(sGam*at);
        dQPdA[1][2] = (sGam*cchi*chix+schi*cGam*xGam)/csth;
        dQPdA[3][2] = pt/at + at *( cchi*cGam/csth*xGam + sGam *
		      (-schi/csth*chix +cchi*sith/pow_2(csth)*dQPdA[1][2]));
    }
    dQPdA[0][2] = dQPdJ[0][1];
    dQPdA[2][2] = dQPdJ[2][1];
    dQPdA[1][2]/= sMb;
// w.r.t. M and b:
    temp = 0.5*(dQPdJ[0][0]*jr+dQPdJ[0][1]*jt+dQPdA[0][2]*jp);
    dQPdA[0][0] =-temp / sMob;
    dQPdA[0][1] = r - temp * sMob;
    temp = 0.5*(dQPdJ[1][0]*jr+dQPdJ[1][1]*jt+dQPdA[1][2]*jp);
    dQPdA[1][0] =-temp / sMob;
    dQPdA[1][1] =-temp * sMob;
    temp = 0.5*(dQPdJ[2][0]*jr+dQPdJ[2][1]*jt+dQPdA[2][2]*jp);
    dQPdA[2][0] = (0.5*pr/b - temp) / sMob;
    dQPdA[2][1] =-(0.5*pr/b + temp) * sMob;
    temp = 0.5*(dQPdJ[3][0]*jr+dQPdJ[3][1]*jt+dQPdA[3][2]*jp);
    dQPdA[3][0] = (0.5*pt - temp) / sMob;
    dQPdA[3][1] = (0.5*pt - temp) * sMob;
// w.r.t. r0:
    dQPdA[0][3] = 1.;
    dQPdA[1][3] = 0.;
    dQPdA[2][3] = 0.;
    dQPdA[3][3] = 0.;
// transfer to gamma, beta, and Lz
    double tgamma=2*gamma, tbeta=2*beta;
    for(short i=0; i<4; i++) {
	dQPdA[i][0] *= tgamma;
	dQPdA[i][1] *= tbeta;
	if(Lz < 0.) dQPdA[i][2] = -dQPdA[i][2];
    }
}

PSPD ToyIsochrone::Forward(const PSPD& JT) const
// Transforms action-angle variables (JT) to phasespace co-ordinates (r,theta)
// and their conjugated momenta; theta is latitude rather than polar angle.
{
    derivs_ok = true;
    double e2;
    static   double fac;
// Extract and scale the actions and angles.
    jr = fmax(0, double(JT(0)) / sMb);
    jt = fmax(0, double(JT(1)) / sMb);
    tr = double(JT(2));
    tt = double(JT(3));
// Make sure that tr is in the correct range
    if(std::isnan(tr) || std::isinf(tr) || fabs(tr)>INT_MAX) 
      tr = 0.; // just in case  
    tr = math::wrapAngle(tr);
    if(jr<0. || jt<0.|| jp<0.) {
        throw std::runtime_error("Torus Error -2: ToyIsochrone: negative action(s)"); 
	return PSPD(0.);
    }
// Set up auxilliary variables independent of the angles tr and tt.
    at  = jp+jt;
    sGam= (jp==0.) ? 1. : sqrt(1.-pow_2(jp/at));
    sq  = hypot(2.,at);
    fac = at/sq;
    HH  = 2./(2.*jr+at+sq);
    H2  = HH*HH;
    H   =-0.5*H2;
    a   = 1./H2-1.;
    if(at==0.) e = 1.;
    else {
        e2    = 1. - pow_2(at/a)/H2;
        e     = (e2>0) ? sqrt(e2) : 0;
        if(e2<-1.e-3) throw std::runtime_error("Torus Error -2: ToyIsochrone: bad e in Forward");
    }
    ae  = a*e;
    eps = ae*H2;
    wr  = H2*HH;
    wt0r= 0.5*(1.+fac);
    wt  = wt0r*wr;
    //cerr << jt << " " << wt << " " << jt*wt<< "\n";
// Solve for the relevant other variables.
    psisolve(); 
    wh  = wfun(fac);
    chi = tt-wt0r*tr+wh;
    u   = a*(1.-e*cpsi);
// Calculate co-ordinates and the conjugate momenta.
    r   = (u==0.)?  0. : sqrt(u*(u+2.));
    th  = asin(sGam*sin(chi));
    pt  = (at==0.)? 0. : at*sGam*cos(chi)/cos(th);
    if(r==0.) pr = sqrt(1.-H2);
        else  pr = HH*ae*spsi/r;
// re-scale and return
    return PSPD(fmax(b*r+r0,b*1.e-6), th, pr*sMob, pt*sMb); 
}

PSPD ToyIsochrone::Backward(const PSPD& QP) const
// Transforms phasespace co-ordinates (r,theta) and their conjugated momenta
// (QP) to action-angle variables (JT);
// theta is latitude rather than polar angle.
{
    derivs_ok = true;
    double e2,csth;
    static   double fac;
// extract and scale co-ordinates
    r   = (QP(0)-r0) / b;
    th  = QP(1);
    pr  = QP(2) / sMob;      // pr  = dr/dt        => [pr]  = Sqrt[GM/b]
    pt  = QP(3) / sMb;       // pth = r^2 * dth/dt => [pth] = Sqrt[GMb]
    csth= cos(th);
// Perform some consistency checks
    if(csth==0.&&jp!=0.) {
	throw std::runtime_error("Torus Error -2: ToyIsochrone: Jp!=0 && R=0 in Backward");
	return PSPD(0.);
    }
    if(r==0. && pt!=0.) {
	throw std::runtime_error("Torus Error -2: ToyIsochrone: pt!=0 && r=0 in Backward");
	return PSPD(0.);
    }
// Set up auxialiary variables
    at  = (jp==0.)?  fabs(pt) : hypot(pt,jp/csth);
    sq  = hypot(2.,at);
    wt0r= 0.5*(1.+at/sq);
    sGam= (jp==0.)? 1. : sqrt(1.-pow_2(jp/at));
    u   = hypot(r,1.)-1.;
    H2  = (pt==0.)? -pr*pr : -pr*pr-pow_2(pt/r);
    if(jp!=0.) H2 -= pow_2(jp/(r*csth));
    H2 += 2./(u+2.);
    if(H2<=0.) {
	throw std::runtime_error("Torus Error -2: ToyIsochrone: H>=0 in Backward");
	return PSPD(0.);
    }
    H   =-0.5*H2;
    HH  = sqrt(H2);
    a   = 1./H2-1.;
    if(at==0.) e = 1.;
    else {
        e2 = 1. - pow_2(at/a)/H2;
        e  = (e2>0) ? sqrt(e2) : 0;
    }
    ae  = a*e;
    eps = ae*H2;
    fac = at/sq;
    wr  = H2*HH;
    wt  = wt0r*wr;
    //psisolve();
    //wh  = wfun(fac);
    jt  = at-jp;
    if(e==0.) { // circular orbit
      wh  = wfun(fac);
      psi = wh/wt0r;
	cpsi= cos(psi);
        spsi= sin(psi);
        jr  = 0.;
        tr  = psi;
        if(sGam==0.)      chi = 0.;
        else if(sGam==1.) chi = (pt<0) ? Pi-fabs(th) : fabs(th);
        else              chi = acos(fmax(-1.,fmin(1.,pt*csth/at/sGam)));
        if(th<0.) chi=TPi-chi;
        tt  = chi;
    } else {
	cpsi = fmax(-1.,fmin(1.,(1.-u/a)/e));
        if(cpsi==1.)       psi = 0.;
        else if(cpsi==-1.) psi = Pi;
        else               psi = (pr<0.)? TPi-acos(cpsi) : acos(cpsi);
        spsi= sin(psi);
        jr  = fmax(0., 1./HH-0.5*(at+sq));
        tr  = psi-eps*spsi;
        if(sGam==0.)      chi = 0.;
        else if(sGam==1.) chi = (pt<0) ? Pi-fabs(th) : fabs(th);
        else              chi = acos(fmax(-1.,fmin(1.,pt*csth/at/sGam)));
        if(th<0.) chi=TPi-chi;
	wh  = wfun(fac);
        tt  = wt0r*tr+chi-wh;
    }
    if(std::isnan(tt) || std::isinf(tt) || fabs(tt)>INT_MAX) 
      tt = 0.; // just in case  
    tt = math::wrapAngle(tt);
// re-scale and return
    return PSPD(jr*sMb, jt*sMb, tr, tt);
}

////////////////////////////////////////////////////////////////////////////////
PSPT ToyIsochrone::Backward3D(const PSPT& QP3) const
{ // For the 3D case
  PSPD Jt2, QP2 = QP3.Give_PSPD();
  PSPT Jt3;

  Jt2 = Backward(QP2);
  Jt3.Take_PSPD(Jt2); // do the 2D part

  Jt3[2] = QP3(5); // old: QP3(0)*cos(QP3(1))*QP3(5); // Angular momentum. 
  if(QP3(1) == 0. && QP3(4) == 0.) { 
    // This should be 
    // Jt3(4) = tt  = wt0r*tr+chi-wh; so wt0r = omega_theta/omega_r; 
    // chi= angle in some plane;  wh = mess of arctans.
    Jt3[5] = (QP3(5)>0.)? QP3(2)-wh+wt0r*Jt3(3) : QP3(2)+wh-wt0r*Jt3(3);
    AlignAngles3D(Jt3);
    return Jt3;
  }
  double sinu = (Jt3(2)>0.)? tan(QP3(1))*Lz/sqrt(Jt3(1)*(Jt3(1)+2.*fabs(Lz))) :
    -tan(QP3(1))*Lz/sqrt(Jt3(1)*(Jt3(1)+2.*fabs(Lz))),
    u = (sinu>1.)? Pih : (sinu<-1.)? -Pih : asin(sinu);
  if(QP3(4)<0.) u= Pi-u; 
  Jt3[5] = QP3(2)-u;
  Jt3[5] += (Jt3(2)>0.)? Jt3(4) : -Jt3(4);
  AlignAngles3D(Jt3);
  return Jt3;
}

////////////////////////////////////////////////////////////////////////////////
PSPT ToyIsochrone::Forward3D(const PSPT& JT3) const
{ // For the 3D case
  PSPD QP2, JT2 = JT3.Give_PSPD();
  PSPT QP3;
  
  QP2 = Forward(JT2); 
  QP3.Take_PSPD(QP2);  // do the 2D part

  QP3[5] = JT3(2); // old:JT3(2)/(QP3(0)*cos(QP3(1))); // Angular velocity/mom
  if(JT3(1) == 0.) {
    QP3[2] = (QP3(5)>0.)? JT3(5)+wh-wt0r*JT3(3) : JT3(5)-wh+wt0r*JT3(3);
    if(std::isnan(QP3[2]) || std::isinf(QP3[2]) || fabs(QP3[2])>INT_MAX) 
      QP3[2] = 0.; // just in case  
    QP3[2] = math::wrapAngle(QP3(2));
    return QP3;
  }

  double sinu = (JT3(2)>0.)? tan(QP3(1))*Lz/sqrt(JT3(1)*(JT3(1)+2.*fabs(Lz))) :
    -tan(QP3(1))*Lz/sqrt(JT3(1)*(JT3(1)+2.*fabs(Lz))),
    u = (sinu>1.)? Pih : (sinu<-1.)? -Pih : asin(sinu);
  if(QP3(4)<0.) u= Pi-u; 
  QP3[2] = JT3(5)+u;

  QP3[2] -= (JT3(2)>0.)? JT3(4) : -JT3(4); // Note opposite sign to Backward
  if(std::isnan(QP3[2]) || std::isinf(QP3[2]) || fabs(QP3[2])>INT_MAX) 
    QP3[2] = 0.; // just in case  
  QP3[2] = math::wrapAngle(QP3(2));

  return QP3;
}


} // namespace
///end of Isochrone.cc//////////////////////////////////////////////////////////
