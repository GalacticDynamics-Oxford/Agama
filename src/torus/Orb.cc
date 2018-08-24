/*******************************************************************************
*                                                                              *
*  Orb.cc                                                                      *
*                                                                              *
* C++ code written by Walter Dehnen, 1994-96,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/

#include <cmath>
#include <iostream>
#include <fstream>
#include "Pi.h"
#include "Orb.h"
#include "WD_Matrix.h"

namespace torus{
using std::ofstream;


typedef Vector<double,4>	DB4;
typedef Matrix<double,4,4>	DB44;

Record::Record(const PSPD& W, Potential* Phi, const double dE) : 
	Pot(Phi), R(W(0)), z(W(1)), pR(W(2)), pz(W(3))
{
  E   = Pot->eff(R,z,aR,az) + 0.5 * (pR*pR + pz*pz);
  set_tolerance(dE);
  set_maxstep();
}

Record::Record(const Record& X) :
	Pot(X.Pot), R(X.R), z(X.z), pR(X.pR), pz(X.pz), aR(X.aR), az(X.az),
 	E(X.E), tol(X.tol) {} 

Record& Record::operator=(const Record& X)
{
    Pot = X.Pot;
    R   = X.R;
    z   = X.z;
    pR  = X.pR;
    pz  = X.pz;
    aR  = X.aR;
    az  = X.az;
    E   = X.E;
    tol = X.tol;
    return *this;
}

void Record::LeapFrog(const double dt)
{
    double dth = 0.5 * dt;
    pR-= aR * dth;
    pz-= az * dth;
    R += pR * dt;
    z += pz * dt;
    E  = Pot->eff(R,z,aR,az);
    pR-= aR * dth;
    pz-= az * dth;
    E += 0.5 * (pR*pR + pz*pz);
}

inline PSPD RKint(const PSPD Y, double& P, Potential* Ph)
{
    double aR,az;
    P = Ph->eff(Y(0),Y(1),aR,az);
    return PSPD(Y(2),Y(3),-aR,-az);
} 

void Record::RungeKutta(const double dt)
{
    PSPD   dY,Y0=PSPD(R,z,pR,pz),Y1=Y0;
    Y1+=(dY=dt*PSPD(pR,pz,-aR,-az))/6.;
    Y1+=(dY=dt*RKint(Y0+.5*dY,E,Pot))/3.;
    Y1+=(dY=dt*RKint(Y0+.5*dY,E,Pot))/3.;
    Y1+=(dY=dt*RKint(Y0+dY,E,Pot))/6.;
    R=Y1(0); z=Y1(1); pR=Y1(2); pz=Y1(3);
    E=Pot->eff(R,z,aR,az)+0.5*(pR*pR+pz*pz);
}

void Record::step2_by(double& dt, const double f)
{
    Record next = *this;
    double fac=(f<=1.)? 2.:f, FAC=pow(fac,3), dE;
    next.LeapFrog(dt);
    dE = fabs(next.E - this->E);
    while(dt<dtm && FAC*dE < tol) {
	dt  *= fac;
	next = *this;
        next.LeapFrog(dt);
        dE = fabs(next.E - this->E);
    }
    while(dE>tol && dt>1.e-10)  {
	dt  /= fac;
	next = *this;
        next.LeapFrog(dt);
        dE = fabs(next.E - this->E);
    }
    *this = next;
}

void Record::step4_by(double& dt, const double f)
{
    Record next = *this;
    double fac=(f<=1.)? 1.4:f, FAC=pow(fac,5), dE;
    next.FourSymp(dt);
    dE = fabs(next.E - this->E);
    while(dt<dtm && FAC*dE < tol) {
	dt  *= fac;
	next = *this;
        next.FourSymp(dt);
        dE = fabs(next.E - this->E);
    }
    while(dE>tol && dt>1.e-10) {
	dt  /= fac;
	next = *this;
        next.FourSymp(dt);
        dE = fabs(next.E - this->E);
    }
    *this = next;
}

void Record::stepRK_by(double& dt, const double f)
{
    Record next = *this;
    double fac=(f<=1.)? 1.4:f, FAC=pow(fac,5), dE;
    next.RungeKutta(dt);
    dE = fabs(next.E - this->E);
    while(dt<dtm && FAC*dE < tol) {
	dt  *= fac;
	next = *this;
        next.RungeKutta(dt);
        dE = fabs(next.E - this->E);
    }
    while(dE>tol && dt>1.e-10)  {
	dt  /= fac;
	next = *this;
        next.RungeKutta(dt);
        dE = fabs(next.E - this->E);
    }
    *this = next;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
Record3D::Record3D(const PSPT& W, Potential* Phi, const double dE) : 
  Pot(Phi), R(W(0)), z(W(1)), phi(W(2)), pR(W(3)), pz(W(4)), 
  phid(W(5)/R), phidd(-2*phid*pR/R), Jphi(R*W(5))
{
  E   = Pot->eff(R,z,aR,az) + 0.5 * (pR*pR + pz*pz);
  set_tolerance(dE);
  set_maxstep();
}

Record3D::Record3D(const Record3D& X) :
Pot(X.Pot), R(X.R), z(X.z), phi(X.phi), pR(X.pR), pz(X.pz), phid(X.phid), 
aR(X.aR), az(X.az), phidd(X.phidd), Jphi(X.Jphi), E(X.E), tol(X.tol) {} 

Record3D& Record3D::operator=(const Record3D& X)
{
    Pot   = X.Pot;
    R     = X.R;
    z     = X.z;
    phi   = X.phi;
    pR    = X.pR;
    pz    = X.pz;
    phid  = X.phid;
    aR    = X.aR;
    az    = X.az;
    phidd = X.phidd;
    Jphi  = X.Jphi;
    E     = X.E;
    tol   = X.tol;
    return *this;
}

inline PSPT RK3Dint(const PSPT Y, double& P, Potential* Ph)
{
  double aR,az,phidd;
  P     = Ph->eff(Y(0),Y(1),aR,az);
  phidd = -2.*Y(3)*Y(5)/Y(0);
  return PSPT(Y(3),Y(4),Y(5),-aR,-az,phidd); // N.B. 3rd/6th component not 
                                             // velocity/acceleration (angular) 
}
 
void Record3D::RungeKutta(const double dt)
{
  phid = Jphi/(R*R);
  PSPT dY,Y0(R,z,phi,pR,pz,phid),Y1=Y0;
  Y1+=(dY=dt*PSPT(pR,pz,phid,-aR,-az,phidd))/6.;
  Y1+=(dY=dt*RK3Dint(Y0+.5*dY,E,Pot))/3.;
  Y1+=(dY=dt*RK3Dint(Y0+.5*dY,E,Pot))/3.;
  Y1+=(dY=dt*RK3Dint(Y0+dY,E,Pot))/6.;
  R=Y1(0); z=Y1(1); phi=Y1(2); pR=Y1(3); pz=Y1(4); phid=Y1(5);
  E=Pot->eff(R,z,aR,az)+0.5*(pR*pR+pz*pz);
  phidd = -2.*Y1(3)*Y1(5)/Y1(0);
}

void Record3D::stepRK_by(double& dt, const double f)
{
    Record3D next = *this;
    double fac=(f<=1.)? 1.4:f, FAC=pow(fac,5), dE;
    next.RungeKutta(dt);
    dE = fabs(next.E - this->E);
    while(dt<dtm && FAC*dE < tol) {
	dt  *= fac;
	next = *this;
        next.RungeKutta(dt);
        dE = fabs(next.E - this->E);
    }
    while(dE>tol && dt>1.e-10)  {
	dt  /= fac;
	next = *this;
        next.RungeKutta(dt);
        dE = fabs(next.E - this->E);
    }
    //cerr << next.QP3D() << '\n';
    *this = next;
    phid = Jphi/(R*R);
    while (phi< 0.)  phi += 2*Pi;
    while (phi>2*Pi) phi -= 2*Pi;
}
///////////////////////////////////////////////////////////////////////////////

inline PSPD fintz(const PSPD& W, const double z, Potential* Phi)
{
    double   FR, Fz;
    Phi->eff(W(0), z, FR, Fz);
    return PSPD(W(2)/W(3), 1./W(3), -FR/W(3), -Fz/W(3));
}

static double find_sos(const PSPD W, Potential* Phi, double& R, double& pR)
// returns dt = (time at which z=0) - (time at which W)
{
    PSPD   y=W, y1;
    double h=-W(1);
    y[1] = 0.;
    if(h!=0.) {
	y1 = h * fintz(W, -h, Phi);
        y += y1/6.;
        y1 = h * fintz(W+y1*0.5, -0.5*h, Phi);
        y += y1/3.;
        y1 = h * fintz(W+y1*0.5, -0.5*h, Phi);
        y += y1/3.;
        y1 = h * fintz(W+y1, 0., Phi);
        y += y1/6.;
    }
    R = y(0);
    pR= y(2);
    return y(1);
}

int StartfromERpR(                 // return      error flag
    Potential*   Phi,              // input:      pointer to a Potential
    const double E,                // input:      orbit's energy ...
    const double R,                // input:      starting value for R
    const double pR,               // input:      starting value for pR
    PSPD&        W)                // output:     starting point (R,0,pR,pz)
{
    double dPdR, dPdz, pzsq;
    pzsq = 2.*(E-Phi->eff(R,0.,dPdR,dPdz))-pR*pR;
    if(pzsq<0.) return 1;
    W = PSPD(R, 0., pR, sqrt(pzsq));
    return 0;
}

int orbit(                         // return:     error flag
    Potential*   Phi,              // input:      pointer to a Potential
    const PSPD&  W0,               // input:      start point (R,z,pR,pz)
    const double tol,              // input:      and tolerance for dE
    const int    NT,               // input:      dyn. times to be integrated
    const double path_length,      // input:      path length between outputs
    const char*  orbfile,          // input:      file name for output
    const char*  sosfile,          // input:      file name for output of SoS
    double&      Oz)               // output:     Omega_z 
{
    const double Pith = Pi/3.;
    ofstream orbout, sosout;
    if(orbfile[0]) {
        orbout.open(orbfile);
	if(!orbout) { cerr<<" cannot open file " << orbfile << '\n'; exit(1); }
    }
    if(sosfile[0]) {
        sosout.open(sosfile);
	if(!sosout) { cerr<<" cannot open file " << sosfile << '\n'; exit(1); }
    }
    int    nz=0;
    PSPD   W=W0;
    double path, t=0, tdt, Sz=0., Siz=0.;
    double          dt=1.e-4, Rsos, pRsos;
    Record X(W,Phi);
    X.set_tolerance(tol);
    path = 0;
    while(nz<NT) {
  	X.stepRK_by(dt);
	if(W(3)>0. && (W(1)*X.QP()(1)) <= 0.) {  // near z=0, pz>0
	    nz++;
	    tdt = t + find_sos(W, Phi, Rsos, pRsos);
	    Sz += tdt;
	    Siz+= nz*tdt;
	    if(sosfile[0])
	        sosout<<' '<<Rsos<<' '<<pRsos<<'\n';
	}
	if(orbfile[0] &&
	   (path+=hypot(X.QP()(0)-W(0),X.QP()(1)-W(1)))>path_length) {
	    path = 0.;
	    orbout.precision(9);
	    orbout //<<t<<' '
		  <<W(0)<<' '<<W(1)<<' '<< W(2)<<' '<<W(3)<<' '
		  <<X.energy()<< '\n';
	    orbout.precision(6);
	}
	t += dt;
	W  = X.QP();
    }
    Oz = Pith*nz*(nz*nz-1) / (2*Siz-(nz+1)*Sz);
    return 0;
}

} // namespace
//end of Orb.cc/////////////////////////////////////////////////////////////////
