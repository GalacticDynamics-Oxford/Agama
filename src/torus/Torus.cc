/*******************************************************************************
*                                                                              *
* Torus.cc                                                                     *
*                                                                              *
* C++ code written by Walter Dehnen, 1995,                                     *
*                     Paul McMillan, 2007-                                     *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/

#include <iomanip>
#include "Torus.h"
#include "Point_None.h"
#include "Point_ClosedOrbitCheby.h"
#include "Toy_Isochrone.h"
#include "WD_Numerics.h"
#include <cmath>

namespace torus{

typedef Vector<double,2>   DB2;
typedef Matrix<double,2,2> DB22;
typedef Vector<double,4>   DB4;
typedef Matrix<double,4,4> DB44;

////////////////////////////////////////////////////////////////////////////////
// class Torus ************************************************************** //
////////////////////////////////////////////////////////////////////////////////

static double RforSOS;
static PSPD   Jtroot;

void Torus::SetMaps(const double* pp,
	            const vec4 &tp,
	            const GenPar &sn,
	            const AngPar &ap)
{
  delete PT;
  PT = new PoiClosedOrbit(pp);
  if(!TM) TM = new ToyIsochrone;
  TM->set_parameters(tp);
  GF.set_parameters(sn);
  AM.set_parameters(ap);
}
void Torus::SetMaps(const vec4 &tp,
	            const GenPar &sn,
	            const AngPar &ap)
{
  delete PT;
  PT = new PoiNone; 
  if(!TM) TM = new ToyIsochrone;
  TM->set_parameters(tp);
  GF.set_parameters(sn);
  AM.set_parameters(ap);
}


void Torus::SetPP(Potential * Phi, const Actions JPT)
{ 
  delete PT;
  PT = new PoiClosedOrbit(Phi,JPT);
}

void Torus::SetPP(double * param) 
{
  delete PT;
  PT = new PoiClosedOrbit(param);
}

void Torus::SetPP(Actions J, Cheby cb1, Cheby cb2, Cheby cb3, 
		  double thmax, double freq) 
{
  delete PT;
  PT = new PoiClosedOrbit(J,cb1,cb2,cb3,thmax,freq);
}

void Torus::SetPP()
{ 
  delete PT;  
  PT = new PoiNone;
}

void Torus::SetTP(const vec4& tp)
{ 
    if(!TM) TM = new ToyIsochrone;
    TM->set_parameters(tp);
}


void Torus::DelMaps()
{
    delete PT;
    delete TM;
}

void Torus::show(ostream& out) const
{
    out //<<" Actions                = "<<J(0)<<','<<J(1)<<','<<J(2)<<'\n'
	<<" E, dE                  = "<<E<<' ';
    if(J(0) && J(1))
      out << (  hypot(Om(0),Om(1)) * sqrt(J(0)*J(1)) * dc(0) ) <<'\n';
    else
      out << (  hypot(Om(0),Om(1)) * (J(0) + J(1)) * dc(0) ) <<'\n';
    out //<<" Frequencies            = "<<Om(0)<<','<<Om(1)<<','<<Om(2)<<'\n'
	<<" dJ, chi_rms            = "<<dc(0)<<','<<dc(1)<<','<<dc(2)<<','<<dc(3)<<'\n'
      //<<" parameters of PoiTra   = "<<PP()<<'\n'
	<<" parameters of ToyMap   : M="<<pow_2(TP()(0))<<", b="<<pow_2(TP()(1))<<", Lz="<<TP()(2)<<", r0="<<TP()(3)
    <<";  # of terms in GF = " << SN().NumberofTerms()<<'\n';
	/*<<" number of Sn, log|Sn|  : ";
	SN().write_log(out);
    out <<"\n log|dSn/dJr|           : ";
	AM.dSdJ1().write_log(out);
    out <<"\n log|dSn/dJl|           : ";
	AM.dSdJ2().write_log(out);
    out <<"\n log|dSn/dJp|           : ";
	AM.dSdJ3().write_log(out);	*/
}


////////////////////////////////////////////////////////////////////////////////
// Deletes all |Sn| < a*|max_Sn|, creates new terms around all |Sn| > b*|max_Sn|
// up to maximum Nmax, sets all |Sn| < off*|max_Sn| to zero
void Torus::TailorAndCutSN(const double ta, const double tb, const double off,
		           const int Nmax)
{
    GenPar SN=GF.parameters();
    SN.tailor(ta,tb,Nmax);
    SN.cut(off);
    GF.set_parameters(SN);
    SN=0.;
    AM.set_parameters(AngPar(SN,SN,SN));
}

////////////////////////////////////////////////////////////////////////////////
void Torus::LevCof(const PSPD     &Jt,
    	           const Position &Q,
		   const double   &a,
		   const double   &b, 
		   PSPD	          &QP,
		   double         &ch,
 		   DB2            &B,
 		   DB22           &A,
 		   DB22           &dQdt) const
{
// May need a rewrite, to include possibility of effectively a>>1
// computes chi, where
//		chi^2 = a^2 * (R0 - R[th])^2 + b^2 * (z0 - z[th])^2
// and its derivatives w.r.t. th
    int    i,j,k;
             double dQPdqp[4][4], dqdt[2][2], dqpdj[4][2], djdt[2][2];
	     DB22   dq;
	     double dR, dz, aq=a*a,bq=b*b;
    QP = TM->ForwardWithDerivs(GF.ForwardWithDerivs(Jt,djdt),dqdt) >> (*PT);
    PT->Derivatives(dQPdqp);
    TM->Derivatives(dqpdj);
    for(i=0; i<2; i++)
      for(j=0; j<2; j++) {
	dq[i][j] = dqdt[i][j] + dqpdj[i][0]*djdt[0][j] + dqpdj[i][1]*djdt[1][j];
      }
    for(i=0; i<2; i++)
      for(j=0; j<2; j++)
	for(k=0,dQdt[i][j]=0.; k<2; k++)
	  dQdt[i][j]+= dQPdqp[i][k] * dq[k][j];
	  
    dR      = (QP(0)==0.)? 1e99 : a*(Q(0)-QP(0));
    dz      = b*(Q(1)-QP(1));
    ch      = hypot(dR,dz);
    B[0]    = a*dR*dQdt[0][0] + b*dz*dQdt[1][0];
    
    if(!QP(1) && !QP(3)) ch = 1e99;
    B[1]    = a*dR*dQdt[0][1] + b*dz*dQdt[1][1];
    A[0][0] = aq*pow(dQdt[0][0],2) + bq*pow(dQdt[1][0],2);
    A[0][1] = aq*dQdt[0][0]*dQdt[0][1] + bq*dQdt[1][0]*dQdt[1][1];
    A[1][0] = A[0][1];
    A[1][1] = aq*pow(dQdt[0][1],2) + bq*pow(dQdt[1][1],2);
}


////////////////////////////////////////////////////////////////////////////////
void Torus::LevCof(const PSPD         &Jt,
    	           const PSPD         &QP_aim,
		   const double       &a,
		   const double       &b,
		   const double       &vscale, 
		   PSPD	              &QP,
		   double             &ch,
 		   DB2                &B,
 		   DB22               &A,
 		   Matrix<double,4,2> &dQPdt) const
{
// May need a rewrite, to include possibility of effectively a>>1
// computes chi, where
//		chi^2 = a^2 * (R0 - R[th])^2 + b^2 * (z0 - z[th])^2
//                      + vscale^2 * ( (vR0 - vR[th])^2 + (vz0 - vz[th])^2 )
// and its derivatives w.r.t. th
    int    i,j,k;
    double dQPdqp[4][4], dqdt[2][2], dpdt[2][2], dqpdj[4][2], djdt[2][2];
    Matrix<double,4,2>   dqp;
    double dR, dz, dvR, dvz, aq=a*a, bq=b*b, vscale2=vscale*vscale;


    QP = TM->ForwardWithDerivs(GF.ForwardWithDerivs(Jt,djdt),dqdt,dpdt)>> (*PT);
    PT->Derivatives(dQPdqp);
    TM->Derivatives(dqpdj);
    for(i=0; i<2; i++)
      for(j=0; j<2; j++) {
	dqp[i][j]= dqdt[i][j] + dqpdj[i][0]*djdt[0][j] + dqpdj[i][1]*djdt[1][j];
      }
    for(i=2; i<4; i++)
      for(j=0; j<2; j++) {
	dqp[i][j]= dpdt[i-2][j]+dqpdj[i][0]*djdt[0][j] + dqpdj[i][1]*djdt[1][j];
      }

    for(i=0; i<4; i++)
      for(j=0; j<2; j++)
	for(k=0,dQPdt[i][j]=0.; k<4; k++)
	  dQPdt[i][j]+= dQPdqp[i][k] * dqp[k][j];
	  
    dR      = (QP(0)==0.)? 1e99 : a*(QP_aim(0)-QP(0));
    dz      = b*(QP_aim(1)-QP(1));
    dvR     = vscale*(QP_aim(2)-QP(2));
    dvz     = vscale*(QP_aim(3)-QP(3));
    ch      = sqrt(dR*dR+dz*dz+dvR*dvR+dvz*dvz);
    if(!QP(1) && !QP(3)) ch = 1e99;

    B[0]    = a*dR*dQPdt[0][0] + b*dz*dQPdt[1][0] 
      + vscale*dvR*dQPdt[2][0] + vscale*dvz*dQPdt[3][0];
    B[1]    = a*dR*dQPdt[0][1] + b*dz*dQPdt[1][1]
      + vscale*dvR*dQPdt[2][1] + vscale*dvz*dQPdt[3][1];

    A[0][0] = aq*pow(dQPdt[0][0],2) + bq*pow(dQPdt[1][0],2) 
      + vscale2*(pow(dQPdt[2][0],2) + pow(dQPdt[3][0],2));
    A[0][1] = aq*dQPdt[0][0]*dQPdt[0][1] + bq*dQPdt[1][0]*dQPdt[1][1] 
      + vscale2*(dQPdt[2][0]*dQPdt[2][1] + dQPdt[3][0]*dQPdt[3][1]);
    A[1][0] = A[0][1];
    A[1][1] = aq*pow(dQPdt[0][1],2) + bq*pow(dQPdt[1][1],2)
      + vscale2*(pow(dQPdt[2][1],2) + pow(dQPdt[3][1],2));
}

void Torus::LevCof(const PSPD         &Jt,
    	           const PSPD         &QP_aim,
		   const Vector<double,4> &sc, 
		   PSPD	              &QP,
		   double             &ch,
 		   DB2                &B,
 		   DB22               &A,
 		   Matrix<double,4,2> &dQPdt) const
{
// May need a rewrite, to include possibility of effectively a>>1
// computes chi, where
//		chi^2 = a^2 * (R0 - R[th])^2 + b^2 * (z0 - z[th])^2
//                      + vscale^2 * ( (vR0 - vR[th])^2 + (vz0 - vz[th])^2 )
// and its derivatives w.r.t. th
    int    i,j,k;
    double dQPdqp[4][4], dqdt[2][2], dpdt[2][2], dqpdj[4][2], djdt[2][2];
    Matrix<double,4,2>   dqp;
    double dR, dz, dvR, dvz;//, aq=a*a, bq=b*b, vscale2=vscale*vscale;
    Vector<double,4> scq;
    for(int i=0;i!=4;i++) scq[i] = sc[i]*sc[i];

    QP = TM->ForwardWithDerivs(GF.ForwardWithDerivs(Jt,djdt),dqdt,dpdt)>> (*PT);
    PT->Derivatives(dQPdqp);
    TM->Derivatives(dqpdj);
    for(i=0; i<2; i++)
      for(j=0; j<2; j++) {
	dqp[i][j]= dqdt[i][j] + dqpdj[i][0]*djdt[0][j] + dqpdj[i][1]*djdt[1][j];
      }
    for(i=2; i<4; i++)
      for(j=0; j<2; j++) {
	dqp[i][j]= dpdt[i-2][j]+dqpdj[i][0]*djdt[0][j] + dqpdj[i][1]*djdt[1][j];
      }

    for(i=0; i<4; i++)
      for(j=0; j<2; j++)
	for(k=0,dQPdt[i][j]=0.; k<4; k++)
	  dQPdt[i][j]+= dQPdqp[i][k] * dqp[k][j];
	  
    dR      = (QP(0)==0.)? 1e99 : sc[0]*(QP_aim(0)-QP(0));
    dz      = sc[1]*(QP_aim(1)-QP(1));
    dvR     = sc[2]*(QP_aim(2)-QP(2));
    dvz     = sc[3]*(QP_aim(3)-QP(3));
    ch      = sqrt(dR*dR+dz*dz+dvR*dvR+dvz*dvz);
    if(!QP(1) && !QP(3)) ch = 1e99;

    B[0]    = sc[0]*dR*dQPdt[0][0] + sc[1]*dz*dQPdt[1][0] 
      + sc[2]*dvR*dQPdt[2][0] + sc[3]*dvz*dQPdt[3][0];
    B[1]    = sc[0]*dR*dQPdt[0][1] + sc[1]*dz*dQPdt[1][1]
      + sc[2]*dvR*dQPdt[2][1] + sc[3]*dvz*dQPdt[3][1];

    A[0][0] = scq[0]*powf(dQPdt[0][0],2) + scq[1]*powf(dQPdt[1][0],2) 
      + scq[2]*powf(dQPdt[2][0],2) + scq[3]*powf(dQPdt[3][0],2);
    A[0][1] = scq[0]*dQPdt[0][0]*dQPdt[0][1] + scq[1]*dQPdt[1][0]*dQPdt[1][1] 
      + scq[2]*dQPdt[2][0]*dQPdt[2][1] + scq[3]*dQPdt[3][0]*dQPdt[3][1];
    A[1][0] = A[0][1];
    A[1][1] = scq[0]*powf(dQPdt[0][1],2) + scq[1]*powf(dQPdt[1][1],2)
      + scq[2]*powf(dQPdt[2][1],2) + scq[3]*powf(dQPdt[3][1],2);
}





////////////////////////////////////////////////////////////////////////////////
inline bool velocities_are_dependent(		// return: v1, v2 dependent?
	const double    norm_det,		// input:  z/R*det(D1)/r^2
	const Velocity& v1,			// input:  v1
	const Velocity& v2,			// input:  v2
	const double    tolerance,              // input: tolerance
	double&         stat)                   // output: discriminant 
{
    if( sign(v1(0)) * sign(v1(1)) * sign(v2(0)) * sign(v2(1)) < 0) return false;
    double x   = hypot(fabs(v1(0))-fabs(v2(0)), 
				fabs(v1(1))-fabs(v2(1))),
		    y   = hypot(v1(0),v1(1)),
		    eps = tolerance * sqrt(norm_det);
    //if(tolerance<0.05) cerr << x << " " << y << " " <<sqrt(norm_det) << "\n";
    if ( x > eps * y ) return false;
    stat = x/(eps*y);
    return true;
}

int Torus::containsPoint(    // return:	    error flag (see below)
    const Position &Q,       // input:      (R,z,phi)
          Velocity &v1,      // output:	    (vR,vz,vphi)_1
	  DB22     &D1,	     // output:     {d(R,z)/d(T1,T2)}_1    
          Angles   &A1,      // output:     T    
          Velocity &v2,      // output:	    (vR,vz,vphi)_2
          DB22     &D2,	     // output:     {d(R,z)/d(T1,T2)}_2       
          Angles   &A2,      // output:     T    
          bool     needA,    // input:      angles out?    
          bool     toy,      // input:      toy angles?
          bool     useA,     // input:      use input angles?
          double   delr)     // input:      tolerance in position 
    const
// Returns 1 if (R,z,phi) is ever hit by the orbit, and 0 otherwise. If the 
// torus passes through the point given, this happens four times, in each case
// with a different velocity. However, only two of these are independent, since 
// changing the sign of both vR and vz simultaneously gives the same orbit. For
// each of these two both possible velocities and the determinant
// | d(x,y,z)/d(Tr,Tl,phi) | is returned. The latter vanishes on the edge of the
// orbit, such that its inverse, the density of the orbit, diverges there
// (that's the reason why not the density itself is returned).
//
// We'll use Levenberg-Marquardt to minimize [R-R(th)]^2 + [z-z(th)]^2
// If the minimum is zero the angles found yield our points (R,z),
// if otherwise the minimum is non-zero (R,z) is never reached by the orbit
{
  // Zeroth: Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in containsPoint\n";
    return 0;
  }
  // First: Special case of Jl=0. Doing it normally creates infinite loop. 
  if(J(1)==0.){ 
    if(Q(1)!=0.) return 0;   
    Angles Ang = 0.;
    PSPD tmp = Map(Ang), QP =0., JT, Jt, Jtry;
    PSPT   QP3D, Jt3D, JT3D;
    double chio,rmin,rmax, dTdt[2][2];;
    DB22   A, Atry, dQdt, dQdtry;
    DB2    B, Btry, dt;
    rmin=tmp(0);
    Ang[0]=Pi;   tmp = Map(Ang);
    rmax=tmp(0);
    if(Q(0)<rmin || Q(0)>rmax) return 0;  
    // Check that it's in range. If not...   
    Jt = PSPD(J(0),J(1),1.,0.);
    const int maxit1=100;
    int it=0;
    const double rtiny=Q(0)*1.e-4;
    while(fabs(QP(0)-Q(0))>rtiny){
      LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);
      Jt[2] -= (QP(0)-Q(0))/sqrt(A(0,0));
        Jt[2] = math::wrapAngle(Jt(2));
      if(Jt(2)>Pi) Jt[2] = TPi - Jt(2);
      it++;
      if((rmin-QP(0))*(QP(0)-rmax) < 0.) {
	cerr << "out of range in Newton-Raphson within containsPoint\n";
	return 0;
      }
      if(it == maxit1) {
	cerr << "too many iterations in Newton-Raphson within containsPoint\n";
	return 0;
      }
    }
    v1[0]    = QP(2);
    v1[1]    = QP(3);
    v1[2]    = J(2)/QP(0); 
    if(needA) {   
      QP3D.Take_PSPD(QP);
      QP3D[2] = Q(2); QP3D[5] = v1(2);
      Jt3D = QP3D << (*PT) << (*TM); // find theta_phi
      Jt3D.Take_PSPD(Jt);
      Jt3D[2] = J(2); // just in case.
    
      JT3D       = AM.Backward3DWithDerivs(Jt3D,dTdt);
      JT = JT3D.Give_PSPD();
      if(toy) {A1[0] = Jt3D(3); A1[1] = Jt3D(4); A1[2] = Jt3D(5);}
      else    {A1[0] = JT3D(3); A1[1] = JT3D(4); A1[2] = JT3D(5);}
      A2 = A1; 
    } else 
      JT = AM.BackwardWithDerivs(Jt,dTdt);

    D1       = 0.;
    D1[0][0] = dQdt(0,0)/dTdt[0][0];
    v2 = v1; 
    D2 = 0.;
    D2[0][0] = D1[0][0];
    return 1;
  }  
//------------------------------------------------------------------------------
// If it isn't  the special case. Do this the hard way.
    const    int    maxit1=100,maxit2=32;
    const    double tiny=1.e-8,
		    hit[64]={1/4.,1.,1/8.,15/8.,3/4.,5/4.,0.,1/2.,3/2.,7/4.,
		    	     3/8.,5/8.,7/8.,9/8.,11/8.,13/8.,
			     1/16.,3/16.,5/16.,7/16.,9/16.,11/16.,13/16.,15/16.,
			     17/16.,19/16.,21/16.,23/16.,25/16.,27/16.,29/16.,
			     31/16.,
			     1/32.,3/32.,5/32.,7/32.,9/32.,11/32.,13/32.,15/32.,
			     17/32.,19/32.,21/32.,23/32.,25/32.,27/32.,29/32.,
			     31/32.,33/32.,35/32.,37/32.,39/32.,41/32.,43/32.,
			     45/32.,47/32.,49/32.,51/32.,53/32.,55/32.,57/32.,
			     59/32.,61/32.,63/32.}; // for finding 2nd theta
    int    it=0, tried=0;
    DB22   A, Atry, dQdt, dQdtry;                   // for Lev-Mar
    DB2    B, Btry, dt;                             // for Lev-Mar
    double chi, chio, dTdt[2][2];                   // ditto
    PSPD   QP;
    PSPD   JT, Jt, Jtry;
    PSPT   QP3D, Jt3D, JT3D;
    double lam=0.5, lam1, det, /*JT3_0,*/ det1,
      rq   = Q(0)*Q(0) + Q(1)*Q(1),            
      rtin = (delr)? delr : sqrt(rq)*tiny;          // tolerance in position 

    

    Jt = PSPD(J(0),J(1),Pih,0.);                   // guess
    if(useA) { //cerr << A1 << "\n"; 
      Jt[2] = A1[0]; Jt[3] = A1[1]; }     // if guess is given
    LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);           // find detc/ detc
    if(std::isnan(B(0))) return 0;
    while(chio>rtin && maxit1>it++ && lam < 1.e20 ) {  // Lev Mar iteration
      //cerr << it << " ";
      lam1  = 1.+lam;
	det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
	dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
	dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
	//cerr << A << B << "\n" << dt << " ";
	Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
	if(std::isnan(Jtry(2)) || std::isinf(Jtry(2)) || fabs(Jtry(2))>INT_MAX)
	  Jtry[2] = 0.; // give up
	if(std::isnan(Jtry(3)) || std::isinf(Jtry(3)) || fabs(Jtry(3))>INT_MAX)
	  Jtry[3] = 0.; // give up
	if(fabs(Jtry(2))>100.) Jtry[2] -= TPi*int(Jtry(2)*iTPi); 
	if(fabs(Jtry(3))>100.) Jtry[3] -= TPi*int(Jtry(3)*iTPi); 
	//cerr << Jtry << "\n";
	while (Jtry(2)< 0. ) Jtry[2] += TPi;
	while (Jtry(3)< 0. ) Jtry[3] += TPi;
	while (Jtry(2)> TPi) Jtry[2] -= TPi;
	while (Jtry(3)> TPi) Jtry[3] -= TPi;
	//cerr << "here\n";
	LevCof(Jtry,Q,1.,1.,QP,chi,Btry,Atry,dQdtry);
	if(chi<chio  && !std::isnan(Btry(0))) {
	    lam *= 0.125;
	    chio = chi;
	    Jt   = Jtry;
            A    = Atry;
	    B    = Btry;
	    dQdt = dQdtry;
	} else
	    lam *= 8.;
    }
    //cerr << "\n";
    if(chio > rtin) return 0;

    v1[0]    = QP(2);
    v1[1]    = QP(3);
    v1[2]    = J(2)/QP(0); 
    if(needA) {
      QP3D.Take_PSPD(QP);
      QP3D[2] = Q(2); QP3D[5] = v1(2);
      Jt3D = QP3D << (*PT) << (*TM);                     // find theta_phi
      Jt3D.Take_PSPD(Jt); Jt3D[2] = J(2); // just in case
      JT3D       = AM.Backward3DWithDerivs(Jt3D,dTdt);   // always needed
      JT = JT3D.Give_PSPD();
      if(toy) {A1[0] = Jt3D(3); A1[1] = Jt3D(4); A1[2] = Jt3D(5);}
      else    {A1[0] = JT3D(3); A1[1] = JT3D(4); A1[2] = JT3D(5);}
    } else 
      JT = AM.BackwardWithDerivs(Jt,dTdt);

    D1[0][0] = dQdt(0,0)*dTdt[1][1] - dQdt(0,1)*dTdt[1][0];
    D1[0][1] =-dQdt(0,0)*dTdt[0][1] + dQdt(0,1)*dTdt[0][0];
    D1[1][0] = dQdt(1,0)*dTdt[1][1] - dQdt(1,1)*dTdt[1][0];
    D1[1][1] =-dQdt(1,0)*dTdt[0][1] + dQdt(1,1)*dTdt[0][0];
    D1      /= dTdt[0][0]*dTdt[1][1]-dTdt[0][1]*dTdt[1][0];
    det1     = fabs(Q(1)/Q(0)) * fabs(D1(0,0)*D1(1,1)-D1(0,1)*D1(1,0)) / rq;
   
    JT[2] = TPi-JT(2); 
    double Jt2_0 = Jt[2],
    Jt3_0 = Jt[3];
// Try to find other independent velocity. 
// It must not fulfill the criterion in the do-while loop

    if(Q(1) == 0.) {	// in symmetry plane, second pair of Vs is dependent.
	v2 = v1; v2[0] *=-1.;
	D2 = D1; D2[0][0] *=-1.; D2[0][1] *=-1.;
	A2 = A1; A2[1] = (A2(1)>Pi)? -Pi+A2(1) : Pi+A2(1); 
	return 1;
    }
    bool usedA=false;                  // used supplied guess
    bool notdone=true;
    double depend_tol = 0.1;           // tolerance for whether v are dependent
    double stat, beststat=0;
    PSPD bestJt=0.;
    do {
	it    = 0;
	lam   = 0.5;
	do {
	  if(useA && !usedA) {        // use supplied guess
	    //cerr << A2 << "\n"; 
	    Jt[2] = A2[0]; Jt[3] = A2[1]; usedA = true;
	  } else { 
	    Jt[2] = Jt2_0;           // Or take a shot based on other theta
	    Jt[3] = Jt3_0 + Pi*hit[tried++];
	  }
	  LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);
	} while (QP(0) == 0. && tried<64); // in case of negative actions
	if(QP(0) == 0.) it = maxit2;       // abort

        while(chio>rtin && maxit2>it++ && lam < 1.e20 ) {  // Lev-Mar iteration
	    lam1  = 1.+lam;
	    det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
	    dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
	    dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
	    Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
	    while (Jtry(2)< 0. ) Jtry[2] += TPi;
	    while (Jtry(3)< 0. ) Jtry[3] += TPi;
	    while (Jtry(2)> TPi) Jtry[2] -= TPi;
	    while (Jtry(3)> TPi) Jtry[3] -= TPi;
            LevCof(Jtry,Q,1.,1.,QP,chi,Btry,Atry,dQdtry);
	    
	    if(chi<chio  && !std::isnan(Btry(0))) { // better
	        lam *= 0.125;
	        chio = chi;
	        Jt   = Jtry;
                A    = Atry;
	        B    = Btry;
	        dQdt = dQdtry;
	    } else                                  // worse
	        lam *= 8.;
        }
        v2[0] = QP(2);
        v2[1] = QP(3);
        v2[2] = J(2)/Q(0); 
	while(JT(3)>TPi) JT[3]-=TPi;
	depend_tol = (tried<5)? 0.1 : (tried<10)? 0.05 : 0.01; // tolerance
	if(chio<=rtin) {
	  notdone = velocities_are_dependent(det1,v1,v2,depend_tol,stat);
	  if(notdone && stat>beststat) {
	    beststat = stat; bestJt = Jt;
	  }
	}
    } while ( (chio>rtin || notdone ) && 64>tried);

    if(tried>=64 && notdone) {
    //if( chio>rtin || velocities_are_dependent(det1,v1,v2,depend_tol))
    	cerr<<" containsPoint() failed at (R,z)=("<<Q(0)<<","<<Q(1)<<")\n";
	Jt = bestJt;
	LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);
    }

    if(needA) {
      QP3D.Take_PSPD(QP);
      QP3D[2] = Q(2); QP3D[5] = v2(2);
      Jt3D = QP3D << (*PT) << (*TM);       // find theta_phi
      Jt3D.Take_PSPD(Jt);  Jt3D[2] = J(2); // just in case.
    
      JT3D  = AM.Backward3DWithDerivs(Jt3D,dTdt);
      JT    = JT3D.Give_PSPD();
      if(toy) {A2[0] = Jt3D(3); A2[1] = Jt3D(4); A2[2] = Jt3D(5);}
      else    {A2[0] = JT3D(3); A2[1] = JT3D(4); A2[2] = JT3D(5);}
    } else
      JT = AM.BackwardWithDerivs(Jt,dTdt);

    D2[0][0] = dQdt(0,0)*dTdt[1][1] - dQdt(0,1)*dTdt[1][0];
    D2[0][1] =-dQdt(0,0)*dTdt[0][1] + dQdt(0,1)*dTdt[0][0];
    D2[1][0] = dQdt(1,0)*dTdt[1][1] - dQdt(1,1)*dTdt[1][0];
    D2[1][1] =-dQdt(1,0)*dTdt[0][1] + dQdt(1,1)*dTdt[0][0];
    D2      /= (dTdt[0][0]*dTdt[1][1]-dTdt[0][1]*dTdt[1][0]);
    
    //if(tried>=64) cerr << D2;

    return 1;
}

void Torus::CheckLevCof(PSPD QP_aim, Angles A_in) {
  PSPD Jt = PSPD(J(0),J(1),A_in[0],A_in[1]), Jtry,QP,oQP;
  double small = 1.e-4;
  double chi,chio,sc;
  DB22   A;
  Matrix<double,4,2> dQPdt;
  DB2    B;
  double tmpx2 = powf(Rmax-Rmin,2) + powf(zmax,2), tmpv2;
  Angles tmpA = 0.; tmpA[0] = Pih;
  //Position tmpq;
  //PSPT   tmp3Dqp = MapfromToy3D(A_in);
  PSPD   tmpqp = MapfromToy(tmpA);
  tmpv2 = tmpqp(2)*tmpqp(2) + tmpqp(3)*tmpqp(3);
  sc = sqrt(tmpx2/tmpv2);

  LevCof(Jt,QP_aim,1.,1.,sc,oQP,chio,B,A,dQPdt);
  //cerr << dQPdt;
  Jtry = Jt; Jtry[2] = A_in[0] + small;
  
  LevCof(Jtry,QP_aim,1.,1.,sc,QP,chi,B,A,dQPdt);
  //for(int i=0;i!=4;i++) cerr << (QP[i]-oQP[i])/small << ' ';
  //cerr << '\n';
  cerr << -2*B(0) << ' ' << (chi*chi-chio*chio)/small << '\n';

  Jtry = Jt; Jtry[3] = A_in[1] + small;
  LevCof(Jtry,QP_aim,1.,1.,sc,QP,chi,B,A,dQPdt);
  //for(int i=0;i!=4;i++) cerr << (QP[i]-oQP[i])/small << ' ';
  //cerr << '\n';

  cerr << -2*B(1) << ' ' << (chi*chi-chio*chio)/small << '\n';

}

void Torus::CheckLevCof(Position Q_aim, Angles A_in) {
  PSPD Jt = PSPD(J(0),J(1),A_in[0],A_in[1]), Jtry,QP,oQP;
  double small = 1.e-5;
  double chi,chio;
  DB22   A ;
  Matrix<double,2,2> dQdt;
  DB2    B;

  LevCof(Jt,Q_aim,1.,1.,oQP,chio,B,A,dQdt);
  //cerr << dQdt(0,0) << ' ' << dQdt(1,0) << ' ';
  Jtry = Jt; Jtry[2] = A_in[0] + small;
  LevCof(Jtry,Q_aim,1.,1.,QP,chi,B,A,dQdt);
  cerr << -2*B(0) << ' ' 
       << (chi*chi-chio*chio)/small << '\n';
  //<< (QP(0)-oQP(0))/small << ' ' <<  (QP(1)-oQP(1))/small << '\n';
  Jtry = Jt; Jtry[3] = A_in[1] + small;
  LevCof(Jtry,Q_aim,1.,1.,QP,chi,B,A,dQdt);
  cerr << -2*B(1) << ' ' << (chi*chi-chio*chio)/small << '\n';

}

////////////////////////////////////////////////////////////////////////////////
DB2 Torus::DistancetoPSP(const PSPD &QP_aim, double &scale, Angles &Aclosest) const
// We'll use Levenberg-Marquardt to minimize 
// [R-R(th)]^2 + [z-z(th)]^2 + scale^2 * ( [vR-vR(th)]^2 + [vz-vz(th)]^2 )
{
  const    int    maxit=100;
  const    double tiny=1.e-8;
  int    it=0;
           DB22   A, Atry;
	   Matrix<double,4,2> dQPdt, dQPdtry;
           DB2    B, Btry, dt;
           double chi, chio=1.e99;
  double lam=0.5, lam1, det, r=hypot(QP_aim(0),QP_aim(1)), rtin=r*tiny;
  PSPD   QP, QP_best;
  PSPD   Jt, Jtry;
  double sc;
  DB2 out=0.;
  Angles Astart(0.);
  if(scale==0.) {
    //cerr << Rmin << ' ' << Rmax << ' ' << zmax << '\n';
    double tmpx2 = powf(0.5*(Rmax-Rmin),2) + powf(zmax,2), tmpv2;
    Angles tmpA = 0.; tmpA[0] = Pih;
    PSPD   tmpqp = MapfromToy(tmpA);
    //cerr << tmpqp << ' ' <<  Rmin << ' ' << Rmax << ' ' << zmax << '\n';
    tmpv2 = tmpqp(2)*tmpqp(2) + tmpqp(3)*tmpqp(3);
    sc = sqrt(tmpx2/tmpv2);
    scale = sc;
    //cerr << sc << '\n';
  } else sc = scale;


  // Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in DistancetoPoint\n";
    out = 0.;
    return out;
  }
  
  // find a starting point using a course grid 
  const int ngridr = 30, ngridz=30;
  Angles Atest;
  for(int i=0;i!=ngridr;i++) {
    Atest[0] = TPi*i/double(ngridr);
    for(int j=0;j!=ngridz;j++) {
      Atest[1] = TPi*j/double(ngridz);
      QP = MapfromToy(Atest);
      chi = pow(QP(0)-QP_aim(0),2) +pow(QP(1)-QP_aim(1),2) +
	sc*sc*(pow(QP(2)-QP_aim(2),2) +pow(QP(3)-QP_aim(3),2));
      if(chi<chio) {
	Astart = Atest;
	chio = chi;
      }
    }
    //std::cout << '\n';
  }

  //Astart = 1.;

  Jt = PSPD(J(0),J(1),Astart[0],Astart[1]);
  LevCof(Jt,QP_aim,1.,1.,sc,QP,chio,B,A,dQPdt);
  if(std::isnan(B(0))) {
    out = 0.; return out;
  }
  QP_best = QP;
  //cerr << Jt << '\n';
  //cerr << QP_aim << ' ' << QP_best << '\n';
  while(chio>rtin && maxit>it++ && lam < 1.e20 ) {
    lam1  = 1.+lam;
    det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
    dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
    dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
    Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
    //cerr << Jtry << ' ' << chio << '\n';
    while (Jtry(2)< 0. ) Jtry[2] += TPi;
    while (Jtry(3)< 0. ) Jtry[3] += TPi;
    while (Jtry(2)> TPi) Jtry[2] -= TPi;
    while (Jtry(3)> TPi) Jtry[3] -= TPi;
    LevCof(Jtry,QP_aim,1.,1.,sc,QP,chi,Btry,Atry,dQPdtry);
    //cerr << chi << ' ' << Jtry[2] << ' ' << Jtry[3] << '\n';
    if(chi<chio  && !std::isnan(Btry(0))) {
      lam *= 0.125;
      chio = chi;
      Jt   = Jtry;
      A    = Atry;
      B    = Btry;
      QP_best = QP;
      dQPdt = dQPdtry;
    } else
      lam *= 8.;
  }
  Aclosest[0] = Jt[2]; Aclosest[1] = Jt[3];
  //cerr << QP_aim << ' ' << QP_best << '\n';
  if(chio < rtin) { out = 0.; return out; }
  else { 
    out[0] = hypot(QP_best(0)-QP_aim(0),QP_best(1)-QP_aim(1));
    out[1] = hypot(QP_best(2)-QP_aim(2),QP_best(3)-QP_aim(3));
    return out;
  }
}


////////////////////////////////////////////////////////////////////////////////
Vector<double,4> Torus::DistancetoPSP(const PSPD &QP_aim, 
				      Vector<double,4> &scales, 
				      Angles &Aclosest) const
// We'll use Levenberg-Marquardt to minimize 
// (scale[0]*[R-R(th)])^2   + (scale[1]*[z-z(th)])^2 + 
// (scale[2]*[vR-vR(th)])^2 + (scale[3]*[vz-vz(th)])^2 
{
  const    int    maxit=100;
  const    double tiny=1.e-8;
  int    it=0;
           DB22   A, Atry;
	   Matrix<double,4,2> dQPdt, dQPdtry;
           DB2    B, Btry, dt;
           double chi, chio=1.e99;
  double lam=0.5, lam1, det, r=hypot(QP_aim(0),QP_aim(1)), rtin=r*tiny;
  PSPD   QP, QP_best;
  PSPD   Jt, Jtry;
  Vector<double,4> sc;
  Vector<double,4>  out=0.;
  Angles Astart(0.);
  if(scales==0.) {
    //FindLimits();
    //cerr << Rmin << ' ' << Rmax << ' ' << zmax << '\n';
    Angles tmpA = 0.; tmpA[0] = Pih;
    PSPD   tmpqp = MapfromToy(tmpA);
    double tmpR2 = powf(0.5*(Rmax-Rmin),2), tmpz2 = zmax*zmax, 
      tmpvR2 = tmpqp(2)*tmpqp(2), tmpvz2 =tmpqp(3)*tmpqp(3);
    //cerr << tmpqp << ' ' <<  Rmin << ' ' << Rmax << ' ' << zmax << '\n';
    sc[0] = 1./sqrt(tmpR2);
    sc[1] = 1./sqrt(tmpz2);
    sc[2] = 1./sqrt(tmpvR2);
    sc[3] = 1./sqrt(tmpvz2);
      //sc = sqrt(tmpR2/tmpv2);
    scales = sc;
    //cerr << sc << '\n';
  } else sc = scales;


  // Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in DistancetoPoint\n";
    out = 0.;
    return out;
  }
  
  // find a starting point using a course grid 
  const int ngridr = 30, ngridz=30;
  Angles Atest;
  for(int i=0;i!=ngridr;i++) {
    Atest[0] = TPi*i/double(ngridr);
    for(int j=0;j!=ngridz;j++) {
      Atest[1] = TPi*j/double(ngridz);
      QP = MapfromToy(Atest);
      chi = sc[0]*sc[0]*powf(QP(0)-QP_aim(0),2) + 
	sc[1]*sc[1]*powf(QP(1)-QP_aim(1),2) +
	sc[2]*sc[2]*powf(QP(2)-QP_aim(2),2) +
	sc[3]*sc[3]*powf(QP(3)-QP_aim(3),2);
      if(chi<chio) {
	Astart = Atest;
	chio = chi;
      }
      //if(chi<0.1) 
      //std::cout << chi << ' ';
	//else cerr << "0.1 ";
    }
    //std::cout << '\n';
  }

  //Astart = 1.;

  Jt = PSPD(J(0),J(1),Astart[0],Astart[1]);
  LevCof(Jt,QP_aim,sc,QP,chio,B,A,dQPdt);
  if(std::isnan(B(0))) {
    out = 0.; return out;
  }
  QP_best = QP;
  //cerr << Jt << '\n';
  //cerr << QP_aim << ' ' << QP_best << '\n';
  while(chio>rtin && maxit>it++ && lam < 1.e20 ) {
    lam1  = 1.+lam;
    det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
    dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
    dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
    Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
    //cerr << Jtry << ' ' << chio << '\n';
    while (Jtry(2)< 0. ) Jtry[2] += TPi;
    while (Jtry(3)< 0. ) Jtry[3] += TPi;
    while (Jtry(2)> TPi) Jtry[2] -= TPi;
    while (Jtry(3)> TPi) Jtry[3] -= TPi;
    LevCof(Jtry,QP_aim,1.,1.,sc[0],QP,chi,Btry,Atry,dQPdtry);
    //cerr << chi << ' ' << Jtry[2] << ' ' << Jtry[3] << '\n';
    if(chi<chio  && !std::isnan(Btry(0))) {
      lam *= 0.125;
      chio = chi;
      Jt   = Jtry;
      A    = Atry;
      B    = Btry;
      QP_best = QP;
      dQPdt = dQPdtry;
    } else
      lam *= 8.;
  }
  Aclosest[0] = Jt[2]; Aclosest[1] = Jt[3];
  //cerr << QP_aim << ' ' << QP_best << '\n';
  if(chio < rtin) { out = 0.; return out; }
  else { 
    out[0] = hypot(QP_best(0)-QP_aim(0),QP_best(1)-QP_aim(1));
    out[1] = hypot(QP_best(2)-QP_aim(2),QP_best(3)-QP_aim(3));
    return out;
  }
}







////////////////////////////////////////////////////////////////////////////////
double Torus::DistancetoPoint(const Position &Q) const
// We'll use Levenberg-Marquardt to minimize [R-R(th)]^2 + [z-z(th)]^2
{
    const    int    maxit=100;
    const    double tiny=1.e-8;
    int    it=0;
	     DB22   A, Atry, dQdt, dQdtry;
	     DB2    B, Btry, dt;
	     double chi, chio;
    double lam=0.5, lam1, det, r=hypot(Q(0),Q(1)), rtin=r*tiny;
	     PSPD   QP;
    PSPD   Jt, Jtry;

  // Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in DistancetoPoint\n";
    return 0;
  }

    Jt = PSPD(J(0),J(1),Pih,0.);
    LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);
    if(std::isnan(B(0))) return 0;

    while(chio>rtin && maxit>it++ && lam < 1.e20 ) {
	lam1  = 1.+lam;
	det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
	dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
	dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
	Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
        LevCof(Jtry,Q,1.,1.,QP,chi,Btry,Atry,dQdtry);
	if(chi<chio  && !std::isnan(Btry(0))) {
	    lam *= 0.125;
	    chio = chi;
	    Jt   = Jtry;
            A    = Atry;
	    B    = Btry;
	    dQdt = dQdtry;
	} else
	    lam *= 8.;
    }

    return (chio > rtin)? chio : 0.;
}


////////////////////////////////////////////////////////////////////////////////
double Torus::DistancetoPoint(const Position &Q, double &thr, double &thz) const
// We'll use Levenberg-Marquardt to minimize [R-R(th)]^2 + [z-z(th)]^2
{
    const    int    maxit=100;
    const    double tiny=1.e-8;
    int    it=0;
	     DB22   A, Atry, dQdt, dQdtry;
	     DB2    B, Btry, dt;
	     double chi, chio;
    double lam=0.5, lam1, det, r=hypot(Q(0),Q(1)), rtin=r*tiny;
	     PSPD   QP;
    PSPD   Jt, Jtry;

  // Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in DistancetoPoint\n";
    return 0;
  }
  if(thr>TPi || thz>TPi || thr<0. || thz<0. ) { thr=Pih; thz=0.; }
  Jt = PSPD(J(0),J(1),Pih,0.);
  //Jt = PSPD(J(0),J(1),thr,thz);
    LevCof(Jt,Q,1.,1.,QP,chio,B,A,dQdt);
    if(std::isnan(B(0))) return 0;

    while(chio>rtin && maxit>it++ && lam < 1.e20 ) {
      //if(it==30) chih = chio;
	lam1  = 1.+lam;
	det   = A(0,0)*A(1,1)*pow(lam1,2) - pow(A(0,1),2);
	dt[0] = (B(0)*lam1*A(1,1)-B(1)*A(0,1)) / det;
	dt[1] = (B(1)*lam1*A(0,0)-B(0)*A(0,1)) / det;
	Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
        LevCof(Jtry,Q,1.,1.,QP,chi,Btry,Atry,dQdtry);
	if(chi<chio  && !std::isnan(Btry(0))) {
	    //itb  = it;
	    lam *= 0.125;
	    chio = chi;
	    Jt   = Jtry;
            A    = Atry;
	    B    = Btry;
	    dQdt = dQdtry;
	} else
	    lam *= 8.;
    }
    thr = Jt[2]; thz = Jt[3];
    return (chio > rtin)? chio : 0.;
}


////////////////////////////////////////////////////////////////////////////////
double Torus::DistancetoRadius(const double R) const
// We'll use Levenberg-Marquardt to minimize [R-R(th)]^2
{
    const    int      maxit=1000;
    const    double   tiny=1.e-4;
    int      it=0;
	     DB22     A, Atry, dQdt, dQdtry;
	     DB2      B, Btry, dt;
	     double   chi, chio;
    double   lam=0.5, lam1, rtin=R*tiny;
	     PSPD     QP;
	     Position Q=0.;
    PSPD     Jt, Jtry;
             Angles Ang = 0.;
	     double rmin,rmax;
	     
  // Avoid bugs causing crashes/loops
  if(J(0)<0. || J(1) < 0.) {
    cerr << "Warning: negative actions in DistancetoRadius\n";
    return 0;
  }
    Ang[1] = Pih;    
    rmin = (Map(Ang))(0);  // Questionable, but certainly 
    Ang[0] = Pi; Ang[1] = 0.;
    rmax = (Map(Ang))(0);
    if(R<=rmax && R>=rmin) return 0.; // Added for a quicker, easier thing

    Q[0] = R;
    Jt   = PSPD(J(0),J(1),Pih,0.);
    LevCof(Jt,Q,1.,0.,QP,chio,B,A,dQdt);
    while(chio>rtin && maxit>it++ && lam < 1.e20 ) {
	lam1  = 1.+lam;
	dt[0] = B(0)/A(0,0)/lam1;
	dt[1] = B(1)/A(1,1)/lam1;
	Jtry  = PSPD(J(0),J(1),Jt(2)+dt(0),Jt(3)+dt(1));
	LevCof(Jtry,Q,1.,0.,QP,chi,Btry,Atry,dQdtry);
	if(chi<chio && !(std::isnan(Btry[0])) ) {
	  lam *= 0.125;
	  chio = chi;
	  Jt   = Jtry;
	  A    = Atry;
	  B    = Btry;
	  dQdt = dQdtry;
	} else
	  lam *= 8.;
    }

    return (chio > rtin)? chio : 0.;
}





////////////////////////////////////////////////////////////////////////////////
void Torus::SOSroot(const double t2, double& z, double& dz) const
{
    double          dQPdqp[4][4], dqdt[2][2], dqpdj[4][2], djdt[2][2];
    PSPD   jt, QP;
    int    i,j;
    double dqdt2;
    Jtroot[3] = t2;
    jt        = GF.ForwardWithDerivs(Jtroot, djdt);
    QP        = TM->ForwardWithDerivs(jt, dqdt) >> (*PT);
    PT->Derivatives(dQPdqp);
    TM->Derivatives(dqpdj);
    z  = QP(1);
    dz = 0.;
    for(i=0; i<2; i++) {
	dqdt2 = dqdt[i][1];
	for(j=0; j<2; j++)
	    dqdt2 += dqpdj[i][j] * djdt[j][1];
        dz += dqdt2 * dQPdqp[1][i];
    }
}
////////////////////////////////////////////////////////////////////////////////
void Torus::SOS(ostream& to, const int Nthr) const
{
    Jtroot = PSPD(J(0),J(1),0.,0.);
    to << (Jtroot>>GF>>(*TM)>>(*PT)) <<"    "<<Jtroot(3)<<'\n';
    const double allowed = -1.e-8;
    double tempz1,tempz2,tempdz;
    for(int ithr=1; ithr<Nthr; ithr++) {
      Jtroot[2] = double(ithr) * Pi / double(Nthr);
      SOSroot(-Pih,tempz1,tempdz); SOSroot(Pih,tempz2,tempdz);
      if(tempz1*tempz2<0.){
	Jtroot[3] = rtsafe(this,&Torus::SOSroot, -Pih, -(-Pih), -allowed);
	to << (Jtroot>>GF>>(*TM)>>(*PT)) <<"    "<<Jtroot(3)<<'\n';
      }
    }
    Jtroot[2] = Pi;
    Jtroot[3] = 0.;
    to << (Jtroot>>GF>>(*TM)>>(*PT)) <<"    "<<Jtroot(3)<<'\n';
}

////////////////////////////////////////////////////////////////////////////////
void Torus::SOS_z_root(const double t2, double& RmRS, double& dR) const
{
    double          dQPdqp[4][4], dqdt[2][2], dqpdj[4][2], djdt[2][2];
    PSPD   jt, QP;
    int    i,j;
    double dqdt2;
    Jtroot[2] = t2;
    jt        = GF.ForwardWithDerivs(Jtroot, djdt);
    QP        = TM->ForwardWithDerivs(jt, dqdt) >> (*PT);
    PT->Derivatives(dQPdqp);
    TM->Derivatives(dqpdj);
    RmRS  = QP(0)-RforSOS;
    dR = 0.;
    for(i=0; i<2; i++) {
	dqdt2 = dqdt[i][0];
	for(j=0; j<2; j++)
	    dqdt2 += dqpdj[i][j] * djdt[j][0];
        dR += dqdt2 * dQPdqp[0][i];
    }
}
////////////////////////////////////////////////////////////////////////////////
void Torus::SOS_z(ostream& to, const double RSOS, const int Nthz) const
{
  Jtroot = PSPD(J(0),J(1),0.,0.);
  RforSOS = RSOS;
  int ntry=100;
  double tmpR1,tmpR2,thmin,thmax;
  const double allowed = -1.e-8;
  // do this slowly but safely-ish
  for(int ithz=0; ithz<Nthz; ithz++) {
    Jtroot[3] = double(ithz) * TPi / double(Nthz);
    // OK first, can I do this the easy way?
    Jtroot[2] = thmin = 0.; tmpR1 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    Jtroot[2] = thmax = Pi; tmpR2 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    for(int i=0;i!=ntry && (tmpR1-RSOS)*(tmpR2-RSOS)>0;i++) {
      if(i) { thmin = thmax; tmpR1 = tmpR2; }
      Jtroot[2] = thmax = double(i+1)*TPi/double(ntry);
      tmpR2 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    }
    if((tmpR1-RSOS)*(tmpR2-RSOS)<0) {
      Jtroot[2] = rtsafe(this,&Torus::SOS_z_root, thmin, thmax, -allowed);
      to <<  (Jtroot>>GF>>(*TM)>>(*PT)) <<"    "<<Jtroot(3)<<'\n';
    }
  }
}
////////////////////////////////////////////////////////////////////////////////
int Torus::SOS_z(double* outz, double* outvz, 
		 const double RSOS, const int Nthz) const
{
  Jtroot = PSPD(J(0),J(1),0.,0.);
  RforSOS = RSOS;
  int ntry=100,nout=0;
  double tmpR1,tmpR2,thmin,thmax;
  const double allowed = -1.e-8;
  // do this slowly but safely-ish
  for(int ithz=0; ithz<Nthz; ithz++) {
    Jtroot[3] = double(ithz) * TPi / double(Nthz);
    // OK first, can I do this the easy way?
    Jtroot[2] = thmin = 0.; tmpR1 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    Jtroot[2] = thmax = Pi; tmpR2 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    for(int i=0;i!=ntry && (tmpR1-RSOS)*(tmpR2-RSOS)>0;i++) {
      if(i) { thmin = thmax; tmpR1 = tmpR2; }
      Jtroot[2] = thmax = double(i+1)*TPi/double(ntry);
      tmpR2 = (Jtroot>>GF>>(*TM)>>(*PT))(0);
    }
    if((tmpR1-RSOS)*(tmpR2-RSOS)<0) {
      Jtroot[2] = rtsafe(this,&Torus::SOS_z_root, thmin, thmax, -allowed);
      PSPD outQP=(Jtroot>>GF>>(*TM)>>(*PT));
      outz[nout]=outQP[1]; outvz[nout]=outQP[3]; nout++;
    }
  }
  return nout;
}

////////////////////////////////////////////////////////////////////////////////

void Torus::AutoTorus(Potential * Phi, const Actions Jin, const double R0) {

  IsoPar IP = 0.;
  double Rc = Phi->RfromLc(Jin(1)+fabs(Jin(2)));
  IP[1] = (R0)? sqrt(R0) : sqrt(Rc);
  double dPdR,dPdz, atmp, btmp;
  (*Phi)(Rc,0.,dPdR,dPdz);
  btmp = (R0)? R0 : Rc;
  atmp = (R0)? sqrt(R0*R0+Rc*Rc) : Rc*sqrt(2.);
  IP[0] = sqrt(dPdR*atmp*powf(btmp+atmp,2.)/Rc);
  IP[2] = fabs(Jin(2));
  GenPar SN;
  SN.MakeGeneric();
  GenPar s1(SN);
  s1=0.;
  SetMaps(IP,SN,AngPar(s1,s1,s1));
  //SetActions(Jin);
  J = Jin;
  E = 0.;
  Om = 0.;
  dc = 0.;
}

void Torus::AutoPTTorus(Potential *Phi, const Actions Jin, const double R0) {
  // Using a point transform rather fixes the paramaters of the toymap
  IsoPar IP = 0.;
  IP[1] = (R0)? sqrt(R0) : sqrt(3.);
  IP[2] = fabs(Jin(2));
  double tmp1 = pow(IP(1),2),       // b
    tmp2 = sqrt(tmp1*tmp1+1.);      // sqrt(b^2+r^2), note r=1
  IP[0] = (IP(2)+Jin(1))*(tmp1+tmp2)*sqrt(tmp2); // sqrt(GM). Again, r=1 
  GenPar SN;
  SN.MakeGeneric();
  GenPar s1(SN);
  s1=0.;
  J = Jin;
  SetMaps(IP,SN,AngPar(s1,s1,s1));
  SetPP(Phi,Jin);
}



void Torus::FindLimits() { // Approximate.
  Angles Ang = 0.;
  Ang[1] = Pih;    
  Rmin = (MapfromToy(Ang))(0);
  Ang[0] = Pi;
  zmax = (MapfromToy(Ang))(1);
  Ang[0] = Pi; Ang[1] =0.;
  Rmax = (MapfromToy(Ang))(0);
}


} // namespace