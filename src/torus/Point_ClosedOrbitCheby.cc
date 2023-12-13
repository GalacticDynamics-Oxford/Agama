
/*
*
* C++ code written by Paul McMillan, 2008                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*/

#include "Point_ClosedOrbitCheby.h"
#include "PJMNum.h"
#include "Orb.h"

namespace torus{

// Routine needed for external integration routines ----------------------------

double PoiClosedOrbit::actint(double theta) const {
  double psi = asin(theta/thmaxforactint); 
  double tmp = vr2.unfit1(psi) * drdth2.unfit1(psi) + pth2.unfit1(psi*psi);
  return tmp;
}
//------------------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////////
// function: do_orbit - Integrates an orbit from z=0 upwards and back to z=0
//                      Tabulates coordinates in between.
void PoiClosedOrbit::do_orbit(PSPD clo,   double delt, Potential* Phi,
	      double* time, double* tbR,  double* tbz,  double* tbr,
	      double* tbvr, double* tbpth,double* tbdrdth, int &np, int Nt) 
{
  double dt, ot,t=0.;
  Record X(clo,Phi,1.e-12);
  X.set_maxstep(delt);
  PSPD W = clo;
  dt = delt;
  for(np=0; np!=Nt && W(1)*X.QP(1) > 0;np++) {
    ot = t;
    W = X.QP();
    do {
      X.stepRK_by(dt);
      t += dt;
    } while(t-ot < delt);
    time[np]   = t;
    tbR[np]    = X.QP(0);
    tbz[np]    = X.QP(1);
    tbr[np]    = hypot(X.QP(0),X.QP(1));
    tbvr[np]   = (X.QP(0)*X.QP(2) + X.QP(1)*X.QP(3))/tbr[np];
    tbpth[np]  = X.QP(0)*X.QP(3) - X.QP(1)*X.QP(2);
    tbdrdth[np]= tbr[np]*tbr[np]*tbvr[np]/tbpth[np];
  }
}

////////////////////////////////////////////////////////////////////////////////
// function: set_Rstart - iterates towards correct Rstart for given energy (E)
void PoiClosedOrbit::set_Rstart(double& Rstart,double Rstop, double& odiff,double& dr,
			  bool& done, bool& either_side, bool& first) 
{
  double diff, small =1.e-4;
  if(fabs(diff=Rstart-Rstop) < small*Rstart) done = true;
  else done = false;
      
  if(either_side && !done) {
    if(diff*odiff>0.) dr *= 0.5;
    else              dr *=-0.5;
    odiff = diff;
  } else if(first && !done) {
    odiff = diff;
    first=false;
  } else if(diff*odiff < 0. && !done) {
    either_side = true;
    odiff = diff;
    dr *= -0.5;
  } else if(fabs(diff) > fabs(odiff) && !done) {
    dr *= -1.;
    Rstart += dr;
  }
  if(!done) Rstart += dr;
}
////////////////////////////////////////////////////////////////////////////////
// function: set_E - iterates towards correct energy (E) for given J_l
void PoiClosedOrbit::set_E(const double tJl,double& odiffJ,double& E, 
		     double& dE, bool& done,bool& es_Jl,bool& firstE)
{
  double diffJ;
  if(fabs(diffJ=tJl-Jl)>0.0005*Jl) done = false;
  if(es_Jl && !done) {
    if(diffJ*odiffJ>0.) dE *= 0.5;
    else                dE *=-0.5;
    odiffJ = diffJ;
  } else if(firstE && !done) {
    odiffJ = diffJ;
    firstE=false;
  } else if(diffJ*odiffJ < 0. && !done) {
    es_Jl = true;
    odiffJ = diffJ;
    dE *= -0.5;
  } else if(fabs(diffJ) > fabs(odiffJ) && !done) {
    dE *= -1.;
    E += dE;
  }
  if(!done) E = (E+dE<0.)? E+dE : 0.95*E;
}

////////////////////////////////////////////////////////////////////////////////
// RewriteTables - organise so we have only the upwards movement, to thmax
void PoiClosedOrbit::RewriteTables(const int n, double *time, double *tbR,double *tbz,
			    double *tbr, double *tbvr, double *tbpth, 
			    double *tbir, double *tbth, int &imax)
{
  int klo,khi;
  double tmax,rmax;
  thmax=0.;
  for(int i=0;i!=n;i++) {
    tbth[i] = atan2(tbz[i],tbR[i]);
    if(tbth[i] > thmax) {
      imax=i; thmax=tbth[i];
    }
  }

  klo = (tbpth[imax] > 0.)? imax : imax-1;
  khi = klo + 1;
  // Estimate maximum th, and r at that point, assuming ~const acceleration 
  tmax=(time[klo]*tbpth[khi]-time[khi]*tbpth[klo])/(tbpth[khi]-tbpth[klo]);
  thmax=tbth[klo]+(tbpth[klo] + .5*(tbpth[khi]-tbpth[klo])*(tmax-time[klo])
	/(time[khi]-time[klo]))*(tmax-time[klo])/pow(.5*(tbr[khi]+tbr[klo]),2);
  rmax=tbr[klo]+(tbvr[klo]+.5*(tbvr[khi]-tbvr[klo])*(tmax-time[klo])
	       /(time[khi]-time[klo]))*(tmax-time[klo]);
  imax = khi;
  omz  = Pih/tmax;        // Frequency of vertical oscillation
  tbth[imax]  = thmax;
  tbpth[imax] = 0.;
  tbvr[imax]  = 0.;
  tbr[imax]   = rmax;
  imax++;
  for(int i=0;i!=imax;i++) tbir[i] = 1./tbr[i];
}

////////////////////////////////////////////////////////////////////////////////
// chebderivs - returns dy/dth and dz/dth using Chebyshev fits to the orbit
vec2 PoiClosedOrbit::chebderivs(const double psi, const vec2 yz) {
  double /*t = thmax * sin(psi),*/ t0 = yz[0]*yz[1], ptho,
    tvr,tdrdth,tpth,tiny=1.e-20,tmp,tmp2;
  vec2 dyzdt;
  tvr    = vr2.unfit1(psi);
  tdrdth = drdth2.unfit1(psi);
  tpth   = pth2.unfit1(psi*psi);
  if(psi == 0.) tdrdth = 0.;  // symmetry, should be firmly stated
  tmp2 = cos(t0);
  tmp = alt2-Lz*Lz/(tmp2*tmp2);
  ptho = (tmp>tiny)? sqrt(tmp) : sqrt(tiny);
  tmp = (yz[1]>tiny)? yz[1] : tiny;
  dyzdt[0] = tvr*tdrdth/(tmp*ptho) * thmax*cos(psi);
  dyzdt[1] = tpth/(yz[0]*ptho) * thmax*cos(psi);
  return dyzdt;
}

////////////////////////////////////////////////////////////////////////////////
// stepper - take a Runge-Kutta step in y and z, return uncertainty
double PoiClosedOrbit::stepper(vec2 &tyz,vec2 &dyzdt,const double psi,const double h) 
{
  double hh=h*0.5,  psihh=psi+hh;
  vec2 tmpyz, k2, tmpyz2, k2b, tmpyz3;
  tmpyz = tyz + hh*dyzdt; 
  k2 = chebderivs(psihh, tmpyz);
  tmpyz = tyz + h*k2;
 // Try doing that twice:
  tmpyz2 = tyz + 0.25*h*dyzdt; 
  k2b  = chebderivs(psi+0.25*h, tmpyz2);
  tmpyz2 = tyz+ 0.5*h*k2b;
  k2b = chebderivs(psi+0.5*h, tmpyz2);
  tmpyz3 = tmpyz2 + 0.25*h*k2b;
  k2b = chebderivs(psi+0.75*h, tmpyz3);
  tyz = tmpyz2+ 0.5*h*k2b;
  
  double err0 = fabs(tyz[0]-tmpyz[0])*0.15,    
    err1 = fabs(tyz[1]-tmpyz[1])*0.15; // error estimates (as O(h^3))
  tyz = tmpyz;
  return (err0>err1)? err0 : err1;
}

////////////////////////////////////////////////////////////////////////////////
// yzrkstep - take a RK step in y & z, step size determined by uncertainty.
void PoiClosedOrbit::yzrkstep(vec2 &yz, vec2 &dyzdt, const double tol, double& psi,
	       const double h0, double& hnext, const double hmin, 
	       const double hmax) {
  bool done=false;
  double err,fac =1.4, fac3=pow(fac,3), h=h0;
  vec2 tmpyz;
  do {  
    tmpyz = yz;    
    err = stepper(tmpyz,dyzdt,psi,h);
    if(err*fac3<tol && h*fac<hmax) h*=fac;
    else done = true;
  } while(!done);

  while(err>tol && h>hmin) {
    h /=fac;
    if(h<hmin) h=hmin;
    tmpyz = yz;  
    err = stepper(tmpyz,dyzdt,psi,h);
  }
  psi += h;
  yz = tmpyz;
  hnext=h;
}
////////////////////////////////////////////////////////////////////////////////
// yfn - Interpolate to return y(th) or z(th) for given th=t
double PoiClosedOrbit::yfn(double t, vec2 * ys,const int which,double * thet, int n) 
{
  int klo=0,khi=n-1,k;
  while(khi-klo > 1) {
    k=(khi+klo)/2;
    if(thet[k]>t) khi = k;
    else          klo = k;
  }
  double h = thet[khi] - thet[klo];
  if(h==0.) cerr << "bad theta input to yfn: klo= "<<klo<<"  khi= "<< khi<<"\n";
  double a = (thet[khi] - t), b = (t - thet[klo]);
  return (a*ys[klo][which] + b*ys[khi][which])/h; 
}


////////////////////////////////////////////////////////////////////////////////
// Find point transform suitable for given potential and Actions
////////////////////////////////////////////////////////////////////////////////

void PoiClosedOrbit::set_parameters(Potential *Phi, const Actions J) {
  Jl = J(1);
  Lz = fabs(J(2));
  alt2=(fabs(Lz)+Jl)*(fabs(Lz)+Jl);
  Phi->set_Lz(Lz);
  if(Jl<=0.) { 
    cerr << "PoiClosedOrbit called for Jl <= 0. Not possible.\n";
    return;
  }
  bool first=true,firstE=true, either_side=false, es_Jl=false, done=false;
  const int Nt=1024;
  int np=0,norb=0,nE=0,nEmax = 50000,imax, NCheb=10;
  double time[Nt], tbR[Nt], tbz[Nt], tbr[Nt], tbvr[Nt], tbpth[Nt], tbdrdth[Nt],
    tbir[Nt],tbth[Nt]; // tables from orbit integration

  // Could improve - starting radius should be guessed from Jz+Jphi

  double Rstart0 = Phi->RfromLc(Lz), Rstart = Rstart0, dr=0.1*Rstart, Rstop,
    E = Phi->eff(Rstart,0.), dE, tiny=1.e-9, odiff=0,odiffJ=0,
    delt=0.002*Rstart*Rstart/Lz, pot, tJl, *psi=0, *psisq=0, *tbth2=0, 
    Escale = Phi->eff(2.*Rstart,0.)-E; // positive number
  PSPD  clo(0);
  Cheby rr2;
  E += tiny*Escale;
  dE = 0.08*Escale;

  // For any given Jl, we do not know the corresponding closed orbit, and 
  // don't even know any point on it. Therefore we have to start by guessing
  // an energy (E) and iterate to the correct value. Furthermore, for any 
  // given E, we don't actually know the closed orbit, so we have to guess a 
  // starting point, then integrate the orbit and use that to improve our 
  // guess
  for(nE=0;!done && nE!=nEmax;nE++) {           // iterate energy 
    for(norb=0;norb!=200 && !done;norb++) {  // for each energy, iterate start R
      //cerr << " Rstart ini " << Rstart << " E = "<<E<<"\n";
      while((pot=Phi->eff(Rstart,tiny))>=E) {// avoid unphysical starting points
	if(first) Rstart += 0.5*(Rstart0-Rstart);  
	else Rstart = 0.01*clo[0] + 0.99*Rstart; // if use mean -> closed loop
	//cerr << Rstart << ' ' << pot << ' ' << E<< "\n";
      }
      clo[0] = Rstart; clo[1] = tiny;        // clo = starting point
      clo[2] = 0.;  clo[3] = sqrt(2.*(E-pot)); 
      //cerr << " Rstart " << Rstart << "\n";
      do {                          // integrate orbit (with enough datapoints) 
	delt= (np==Nt)? delt*2 : (np<0.25*Nt &&np)? delt*.9*np/double(Nt):delt;
	do_orbit(clo,delt,Phi,time,tbR,tbz,tbr,tbvr,tbpth,tbdrdth,np,Nt); 
      } while(np==Nt || np < Nt/4);
      Rstop = tbR[np-2]-tbz[np-2]/(tbz[np-1]-tbz[np-2])*(tbR[np-1]-tbR[np-2]); 
      set_Rstart(Rstart,Rstop,odiff,dr,done,either_side,first);//pick new Rstart
    } // end iteration in Rstart
    // clean up tables of values
    RewriteTables(np, time,tbR,tbz,tbr,tbvr,tbpth,tbir, tbth, imax);
    // find Jl, having set up chebyshev functions to do so.
    psi =   new double[imax];
    psisq = new double[imax];
    tbth2 = new double[imax];
    for(int i=0; i!=imax; i++){
      tbth2[i] = tbth[i]*tbth[i];
      psi[i]   = (tbth[i] >= thmax)? Pih : asin(tbth[i]/thmax);
      psisq[i] = psi[i] * psi[i];
    }
    drdth2.chebyfit(psi,tbdrdth,imax-2,NCheb);
    vr2.chebyfit   (psi,  tbvr, imax,NCheb);
    pth2.chebyfit  (psisq,tbpth,imax,NCheb);
    thmaxforactint = thmax;
    //tJl = 2.*qromb(&actint,0,thmax)/Pi;       // Find Jl
    tJl = 2.*qromb(this,&PoiClosedOrbit::actint,0,thmax)/Pi;
    set_E(tJl,odiffJ,E,dE,done,es_Jl,firstE); // pick new E

    if(!done && nE!=nEmax-1) { delete[] psi; delete[] psisq; delete[] tbth2; }
    dr=0.1*Rstart;
    first = true;
    either_side = false;
  } // end iterative loop

  for(int i=0; i!=imax; i++) tbth2[i]   = tbth[i]*tbth[i];
  xa.chebyfit  (tbth2, tbir, imax, NCheb);  // x
  rr2.chebyfit (tbth2,  tbr, imax, NCheb);


  double thmax2 = acos(sqrt(Lz*Lz/alt2));
  int many=100000;
  double tpsi,*thet;
  vec2   *yzfull,yz,dyzdt;

  thet  = new double[many];
  yzfull= new vec2[many];

  yz[0] = 1.; yz[1] = 0.; tpsi = 0;
  dyzdt = chebderivs(tpsi,yz);
  int np2, nr=15;
  double tol=2.e-10, h0=5.e-4,hnext=h0,tmp=0.;

  for(np2=0;tmp<0.99*thmax && yz[0]*yz[1]<0.99*thmax2&& np2!=many;np2++) {
    h0 = (hnext<0.002)? hnext : 0.002; 
    yzrkstep(yz,dyzdt,tol,tpsi,h0,hnext,1.e-8,2.e-3);
    tmp =  thmax*sin(tpsi);
    thet[np2] = tmp;
    yzfull[np2][0] = yz[0];
    yzfull[np2][1] = yz[1];
    dyzdt = chebderivs(tpsi,yz);
  }


  double *rr = new double[nr], *yy = new double[nr];
  rr[0] = 0.;
  yy[0] = 1.;
  for(int i=1;i!=nr-1;i++) { // this nr is new and ~15
    double tmpth = sqrt((i-1)/double(nr-2))*thmax;
    rr[i] = rr2.unfit1(tmpth*tmpth); // possibly unfitn
    yy[i] = yfn(tmpth,yzfull,0,thet,np2);
  }
  rr[nr-1] = 2*rr[nr-2];
  yy[nr-1] = yy[nr-2];

  int NCheb2 = 2*NCheb;
  ya.chebyfit(rr,yy,nr,NCheb2); 

  //double outyz[nr];
  //ya.unfitn(rr,outyz,nr);
  //PJMplot2 graph(rr,yy,nr,rr,outyz,nr);
  //graph.plot();

  // extend zz to all theta<pi/2 and fit to theta*(apoly in theta**2) 
  double *tmpth2 = new double[nr], *zz = new double[nr];
  for(int i=0;i!=nr;i++) {
    tmpth2[i] = (i+1.)/double(nr)*thmax*thmax;
    tmp = sqrt(tmpth2[i]);
    zz[i] = yfn(tmp,yzfull,1,thet,np2)/tmp; 
  }

  za.chebyfit(tmpth2,zz,nr,NCheb); // note that this is in fact z/theta.

  //za.unfitn(tmpth2,outyz,nr);
  //graph.putvalues(tmpth2,zz,nr,tmpth2,outyz,nr);
  //graph.findlimits();
  //graph.plot();

  double x1 = thmax*thmax, x2=Pih*Pih, y1x, dy1x, y1z, dy1z, delx = x2-x1;

  xa.unfitderiv(x1,y1x,dy1x);
  za.unfitderiv(x1,y1z,dy1z);
  // define coefficients such that quadratic goes through final point and
  // has correct gradient at that point. Then take same values of xa and za 
  // at th = pi/2 
  ax = -dy1x/delx;
  bx = dy1x-2*ax*x1;
  cx = y1x - x1*(ax*x1 + bx);

  az = -dy1z/delx;
  bz = dy1z-2*az*x1;
  cz = y1z - x1*(az*x1 + bz);

  delete[] psi;   
  delete[] psisq; 
  delete[] tbth2; 
  delete[] thet;
  delete[] yzfull;
  delete[] rr;
  delete[] yy;
  delete[] tmpth2;
  delete[] zz;
}

// various numbers needed for Derivatives() 
double R,z,r,th,th2,ir,costh,sinth,pr,pth,xpp,ypp,zpp,dx,dy,dz,d2x,d2y,d2z,
  rt,tht,prt,ptht;
double drtdr, drtdth, dthtdr, dthtdth;
double dthdtht, dthdrt, drdtht, drdrt;

PoiClosedOrbit::PoiClosedOrbit(const double* param) {
  set_parameters(param);
}

void PoiClosedOrbit::set_parameters(const double* param) {
  int ncx,ncy,ncz;
  Jl = param[0]; Lz = param[1]; thmax = param[2]; omz = param[3];
  ncx = int(param[4]);
  double *chx = new double[ncx];
  for(int i=0;i!=ncx;i++) chx[i] = param[5+i];
  ncy = int(param[5+ncx]);
  double *chy = new double[ncy];
  for(int i=0;i!=ncy;i++) chy[i] = param[6+ncx+i];
  ncz = int(param[6+ncx+ncy]);
  double *chz = new double[ncz];
  for(int i=0;i!=ncz;i++) chz[i] = param[7+ncx+ncy+i];
  xa.setcoeffs(chx,ncx);
  ya.setcoeffs(chy,ncy);
  za.setcoeffs(chz,ncz);

  double x1 = thmax*thmax, x2=Pih*Pih, y1x, dy1x, y1z, dy1z, delx = x2-x1;

  xa.unfitderiv(x1,y1x,dy1x);
  za.unfitderiv(x1,y1z,dy1z);
  // define coeeficients such that quadratic goes through final point and
  // has correct gradient at that point. Then take same values of xa and za 
  // at th = pi/2 
  ax = -dy1x/delx;
  bx = dy1x-2*ax*x1;
  cx = y1x - x1*(ax*x1 + bx);

  az = -dy1z/delx;
  bz = dy1z-2*az*x1;
  cz = y1z - x1*(az*x1 + bz);

  delete[] chx;
  delete[] chy;
  delete[] chz;
}

PoiClosedOrbit::PoiClosedOrbit(Actions J, Cheby ch1, Cheby ch2, Cheby ch3, 
		   double tmx, double om) {
  set_parameters(J,ch1,ch2,ch3,tmx,om);
}

PoiClosedOrbit::PoiClosedOrbit() {}

///////////////////////////////////////////////////////////////////////////////
PoiClosedOrbit::PoiClosedOrbit(Potential *Phi, const Actions J) {
  set_parameters(Phi,J);
}


///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// The actual transforms                                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

PSPD    PoiClosedOrbit::Forward           (const PSPD &qp)                const 
{
  //first convert from toy coords to real  
  // first guess, th = th^T
  th = qp(1);
  th2 = th*th;
  // then use r^T = x(th)*r
  if(fabs(th)<=thmax) xa.unfitderiv(th2,xpp,dx,d2x);
  else {
    xpp = (ax*th2+bx)*th2+cx;
    dx = 2*ax*th2+bx;
    d2x= 2*ax;
  }
  double ixpp = 1./xpp;
  dx  = 2*th*dx;
  r = qp(0)*ixpp;
  ya.unfitderiv(r,ypp,dy,d2y);
  if(fabs(th)<=thmax) za.unfitderiv(th2,zpp,dz,d2z);
  else {
    zpp = (az*th2+bz)*th2+cz;
    dz = 2*az*th2+bz;
    d2z= 2*az;
  }
  dz  = zpp + 2.*th2*dz; 
  zpp  = zpp*th;
  double tmptht = ypp*zpp;
  int tmpint=0;
  do {
    tmpint++;
    dthtdth = ypp*dz - zpp*dy*r*ixpp*dx; // note not holding usual constants
    th += (qp(1)-tmptht)/dthtdth;
    th2 = th*th;
    // then use r^T = x(th)*r
    if(fabs(th)<=thmax) xa.unfitderiv(th2,xpp,dx,d2x);
    else {
      xpp = (ax*th2+bx)*th2+cx;
      dx = 2*ax*th2+bx;
      d2x= 2*ax;
    }
    double ixpp = 1./xpp;
    d2x = 2*dx+4*th2*d2x;  // because given is d/dth2
    dx  = 2*th*dx;
    r = qp(0)*ixpp;
    ya.unfitderiv(r,ypp,dy,d2y);
    if(fabs(th)<=thmax) za.unfitderiv(th2,zpp,dz,d2z);
    else {
      zpp = (az*th2+bz)*th2+cz;
      dz = 2*az*th2+bz;
      d2z= 2*az;
    }
    d2z = th*(6.*dz + 4*th2*d2z); // because given is d(z/th)/dth2
    dz  = zpp + 2.*th2*dz; 
    zpp  = zpp*th;
    tmptht = ypp*zpp;
  }while(fabs(tmptht-qp(1))>0.00000001 && tmpint<100);

  costh = cos(th); sinth = sin(th); ir = 1./r;
  drtdr = xpp; drtdth = r*dx; dthtdr = dy*zpp; dthtdth = ypp*dz;
  rt = qp(0); tht = qp(1); prt = qp(2); ptht = qp(3);
  pr = drtdr*prt + dthtdr*ptht; pth = drtdth*prt + dthtdth*ptht;

  double idet = 1./(dthtdth*drtdr-drtdth*dthtdr);
  dthdtht = drtdr*idet; dthdrt = -dthtdr*idet; drdtht=-drtdth*idet;
  drdrt = dthtdth*idet; // needed by Derivatives()

// last convert from rth to Rz
  R = r*costh;
  z = r*sinth;
  double pR = costh*pr - sinth*ir*pth;
  double pz = sinth*pr + costh*ir*pth;
  PSPD QP(R,z,pR,pz);
  return QP;
}

PSPD   PoiClosedOrbit::Backward           (const PSPD &QP)                const 
{
  PSPD qp; 
  R=QP(0); z=QP(1); r=hypot(QP(0),QP(1)); th=atan2(QP(1),QP(0)); 
  double th2=th*th; 
  ir=1./r; costh=QP(0)*ir; sinth=QP(1)*ir; 
  pr=QP(2)*costh+QP(3)*sinth; pth=-QP(2)*QP(1)+QP(3)*QP(0);

  if(fabs(th)<=thmax) xa.unfitderiv(th2,xpp,dx,d2x);
  else {
    xpp = (ax*th2+bx)*th2+cx;
    dx = 2*ax*th2+bx;
    d2x= 2*ax;
  }
  ya.unfitderiv(r,ypp,dy,d2y);
  if(fabs(th)<=thmax) za.unfitderiv(th2,zpp,dz,d2z);
  else {
    zpp = (az*th2+bz)*th2+cz;
    dz = 2*az*th2+bz;
    d2z= 2*az;
  }

  d2x = 2*dx+4*th2*d2x;  // because given is d/dth2
  dx  = 2*th*dx;

  d2z = th*(6.*dz + 4*th2*d2z);   // given is z/th and d/dth2
  dz  = zpp + 2.*th2*dz; 
  zpp  = zpp*th;
  
  rt  = r  * xpp;
  tht = ypp * zpp;
  qp[0] = rt;
  qp[1] = tht;
  // unfortunately getting p(r,th)^T is harder. Need to know 
  // d(r,th)/d(r,th)^t, but can only find inverses directly.
  
  drtdr = xpp; drtdth = r*dx; dthtdr = dy*zpp; dthtdth = ypp*dz;

  double idet = 1./(dthtdth*drtdr-drtdth*dthtdr);
  dthdtht = drtdr*idet; dthdrt = -dthtdr*idet; drdtht=-drtdth*idet;
  drdrt = dthtdth*idet;
  prt  = pr*drdrt+pth*dthdrt; 
  ptht = pr*drdtht+pth*dthdtht; 
  qp[2] = prt;
  qp[3] = ptht;
  return qp;

}

////////////////////////////////////////////////////////////////////////////////
PSPT PoiClosedOrbit::Forward3D(const PSPT& w3) const
{
  PSPT W3 = w3;
  PSPD w2 = w3.Give_PSPD();
  W3.Take_PSPD(Forward(w2));
  W3[5] /= W3(0);  // p_phi^T -> v_phi 
  return W3;
}
////////////////////////////////////////////////////////////////////////////////
PSPT PoiClosedOrbit::Backward3D(const PSPT& W3) const
{
  PSPT w3 = W3;
  PSPD W2 = W3.Give_PSPD();
  w3.Take_PSPD(Backward(W2));
  w3[5] *= W3(0); // correct because this is v_phi, not p_phi 
  return w3;
}
////////////////////////////////////////////////////////////////////////////////

void    PoiClosedOrbit::Derivatives(double dQPdqp[4][4]) const {

  double dpRdr = pth*sinth*ir*ir, dpRdth = -pr*sinth-pth*costh*ir, 
    dpRdpr = costh, dpRdpth = -sinth*ir,
    dpzdr = -costh*ir*ir*pth, dpzdth = costh*pr+-sinth*ir*pth, 
    dpzdpr = sinth, dpzdpth = costh*ir;//, dQPdqp[4][4];
  
  double d2rtdr2 = 0., d2rtdrdth = dx, d2rtdth2 = r*d2x,
    d2thtdr2 = d2y*zpp, d2thtdrdth = dy*dz, d2thtdth2 = ypp*d2z;

  double dprdrt = (d2rtdr2*drdrt + d2rtdrdth*dthdrt)*prt +
    (d2thtdr2*drdrt + d2thtdrdth*dthdrt)*ptht,
    dprdtht = (d2rtdr2*drdtht + d2rtdrdth*dthdtht)*prt +
    (d2thtdr2*drdtht + d2thtdrdth*dthdtht)*ptht, 
    dprdprt = drtdr, dprdptht = dthtdr;
  
  double dpthdrt = (d2rtdrdth*drdrt + d2rtdth2*dthdrt)*prt +
    (d2thtdrdth*drdrt + d2thtdth2*dthdrt)*ptht,
    dpthdtht = (d2rtdrdth*drdtht + d2rtdth2*dthdtht)*prt +
    (d2thtdrdth*drdtht + d2thtdth2*dthdtht)*ptht, 
    dpthdprt = drtdth, dpthdptht = dthtdth;

  dQPdqp[0][0] = costh*drdrt - z*dthdrt;    //dR/dr*dr/drt + dR/dth*dth/drt
  dQPdqp[0][1] = costh*drdtht - z*dthdtht;  //dR/dr*dr/dtht + dR/dth*dth/dtht
  dQPdqp[0][2] = 0.;
  dQPdqp[0][3] = 0.;
  
  dQPdqp[1][0] = sinth*drdrt  + R*dthdrt;  //dz/dr*dr/drt + dz/dth*dth/drt
  dQPdqp[1][1] = sinth*drdtht + R*dthdtht; //dz/dr*dr/dtht + dz/dth*dth/dtht
  dQPdqp[1][2] = 0.;
  dQPdqp[1][3] = 0.;
  
  dQPdqp[2][0] = dpRdr*drdrt + dpRdth*dthdrt + dpRdpr*dprdrt + dpRdpth*dpthdrt;
  dQPdqp[2][1] = dpRdr*drdtht+ dpRdth*dthdtht+ dpRdpr*dprdtht+ dpRdpth*dpthdtht;
  dQPdqp[2][2] = dpRdpr*dprdprt+ dpRdpth*dpthdprt;
  dQPdqp[2][3] = dpRdpr*dprdptht+dpRdpth*dpthdptht;
 
  dQPdqp[3][0] = dpzdr*drdrt + dpzdth*dthdrt + dpzdpr*dprdrt + dpzdpth*dpthdrt;
  dQPdqp[3][1] = dpzdr*drdtht+ dpzdth*dthdtht+ dpzdpr*dprdtht+ dpzdpth*dpthdtht;
  dQPdqp[3][2] = dpzdpr*dprdprt+ dpzdpth*dpthdprt;
  dQPdqp[3][3] = dpzdpr*dprdptht+dpzdpth*dpthdptht;
 
}

} // namespace