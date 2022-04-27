/*******************************************************************************
*                                                                              *
* Fit.cc                                                                       *
*                                                                              *
* C++ code written by Walter Dehnen, 1995-96,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*------------------------------------------------------------------------------*
*                                                                              *
* 1 fit the Sn, and/or parameters for the ToyMap and/or PoiTra                 *
*                                                                              *
* 2 compute the dS/dJ and frequencies                                          *
*                                                                              *
*******************************************************************************/
#include <iostream>
#include <iomanip>
#include <fstream>
#include "Fit.h"
#include "Orb.h"
#include <cmath>
#include "WD_Numerics.h"

namespace torus{

////////////////////////////////////////////////////////////////////////////////
// typedefs and constants

typedef Matrix<double,2,2> DB22;
typedef Vector<double,2>   DB2;

const   double  dch_tol = 1.e3;	// upper limit for dchirms/dparameter

////////////////////////////////////////////////////////////////////////////////
// routine SbyLevMar() ****************************************************** //
////////////////////////////////////////////////////////////////////////////////

inline int minmax(double& x, const double min, const double max)
{
    if( x < min ) { x = min; return 1; }
    if( x > max ) { x = max; return 1; }
    return 0;
}

inline double Heff(const PSPD& QP, double dHdQP[4], Potential* Phi)
{
    dHdQP[2] = double(QP(2));
    dHdQP[3] = double(QP(3));
    if(std::isinf(QP(0))) return 0.5*(pow(dHdQP[2],2) +pow(dHdQP[3],2));
    if(std::isinf(QP(1))) 
      return 0.5*(pow(dHdQP[2],2) +pow(dHdQP[3],2)) 
	     + Phi->Lzsquare()/(pow(QP(0),2));
    return 0.5*(pow(dHdQP[2],2) +pow(dHdQP[3],2))
	   + Phi->eff(double(QP(0)),double(QP(1)),dHdQP[0],dHdQP[1]);
}

static void Deletion(const int M, double**A, double**B, double*a, double*b)
{
    int m;
    for(m=0; m<M; m++) {
	delete[] A[m];
	delete[] B[m];
    }
    delete[] A;
    delete[] B;
    delete[] a;
    delete[] b;
}
////////////////////////////////////////////////////////////////////////////////
static int LevMarCof(        // return:        error flag (see below)
    const char     fit[3],   // Input:         what should be fit?
    const Actions& J,        // Input:         Actions of orbit 
    Potential*     Phi,      // Input:         pointer to Potential
    const int      mfit,     // Input:         number of parameters to be fit
    GenFncFit&     GF,       // Input:         generating function
    ToyMap&        TM,       // Input:         toy-potential map
    PoiTra&        PT,       // Input:         canonical map
    double&        Hav,      // Output:        average H
    double&        chirms,   // Output:        rms deviation from average H
    double&        dchisq,   // Output:        | dchi^2 / da |
    double         *bk,      // Output:        array  with first deriv. of chisq
    double         **akl)    // Output:        matrix with sec. deriv. of chisq
/*==============================================================================
on the meaning of the return value:
    0  -> everything went ok
   -2  -> negative action(s) occured at least at one point in angle space
   -4  -> dchisq/da > dch_tol (static const)
--------------------------------------------------------------------------------
definition of chi to be minimized w.r.t. the parameters:

           2                2          2          2
        chi  = < [< H > - H]  >  =  < H  > - < H > 

             SUM_angles A(angles,..)
with < A > = -----------------------   and H = H(angles, parameters p).
             Number of angles

Then
                         2
                 1  d chi               dH               dH
        b_k = - --- ------ = < H > * < ---- >  -  < H * ---- >
                 2   dp_k              dp_k             dp_k

and (neglecting second derivatives of H, see Press et al.)

                      2   2
                1    d chi        dH     dH         dH         dH
        a_kl = --- --------- = < ---- * ---- > - < ---- > * < ---- >.
                2  dp_k dp_l     dp_k   dp_l       dp_k       dp_l

Also
        |      2 |
        | d chi  |  
        | -----  | = 2 Sqrt{ SUM_k (b_k)^2 }.
        | d p_k  |

The function H(angles, parameters) and its derivatives are obtained as follows
(upper-case letters denote target and lower-case toy coordinates, J are actions,
t angles, q usual phase space co-ordinates, and p their conjugate momenta).

1. (J,t) -> (j,t)  by generating function:
    j_i             = J_i + 2 * SUM_(n1,n2) S_(n1,n2) ni cos{n1*t1 + n2*t2}
    dj_i/dS_(n1,n2) = 2 * ni * cos{n1*t1 + n2*t2}
2. (j,t) -> (q,p)  by toy-potential mapping, parameters alpha
3. (q,p) -> (Q,P)  by canonical transformation, parameters beta

                                  2             2            2
H(J,t)        = H_eff(Q,P) = 1/2 P  + Pot(Q) + J_phi / (2 * R ) 
dH/dbeta      = dH_eff/d(Q,P) * d(Q,P)/dbeta
dH/dalpha     = dH_eff/d(Q,P) * d(Q,P)/d(q,p) * d(q,p)/dalpha
dH/dS_(n1,n2) = dH_eff/d(Q,P) * d(Q,P)/d(q,p) * d(q,p)/dj * dj/dS_(n1,n2)

==============================================================================*/
{
    int    i1, i2, j, k, l, m;
    PSPD   jt, QP;
    double          dqpdj[4][2], dQPdqp[4][4], dHdQP[4], dHdqp[4]={0};
    Pdble           dHda, dHavda, dqpdalfa[4], dQPdbeta[4];
    double H, Hsqav, temp,chims;

    dHda   = new double[mfit];
    dHavda = new double[mfit];
    Hav    = 0.;
    Hsqav  = 0.;
    //cerr << TM.parameters() << '\n' << PT.parameters() << '\n';
    for(k=0; k<4; k++) {
        dqpdalfa[k] = new double[TM.NumberofParameters()];
        dQPdbeta[k] = new double[PT.NumberofParameters()];
    }
    for(k=0; k<mfit; k++) {
        dHavda[k] = 0.;
	bk[k]     = 0.;
	for(l=k; l<mfit; l++)
	    akl[k][l] = 0.;
    }
//
// We make a difference between the cases with and without fitting of the Sn
// this is to avoid the construction of the GenPar dj1dS, dj2dS in the latter
//
// 1. case: fit of the Sn
//          bomb around the torus and sum up the derivatives
//
  
    if(fit[0]) {
        double  dHdj[2];
        GenPar  dj1dS(GF.parameters()), dj2dS(GF.parameters()); 
        for(i1=0; i1<GF.N_th1(); i1++)
        for(i2=0; i2<GF.N_th2(); i2++) {
	    jt = GF.MapWithDerivs(J(0),J(1),i1,i2,dj1dS,dj2dS);
            if (jt[0] < 0. || jt[1] < 0.) {  // negative actions
                for(k=0;k<4;k++) { delete[] dqpdalfa[k]; delete[] dQPdbeta[k]; }
                delete[] dHda; delete[] dHavda;
	        return -2;
	    }
            QP = jt >> TM;
	    //cerr <<"\n"<< QP << "\n"; 
	    QP = QP >> PT;
	    //cerr << QP << "\n"; 
 	    if(fit[1]) TM.Derivatives(dqpdj, dqpdalfa);
		else   TM.Derivatives(dqpdj);
            //if(fit[2]) PT.Derivatives(dQPdqp, dQPdbeta); // NB fix needed
	    PT.Derivatives(dQPdqp);
	    H = Heff(QP, dHdQP, Phi);
	    if(fit[0] || fit[1])
	      for(j=0; j<4; j++)
		for(k=0, dHdqp[j]=0.; k<4; k++)
		  dHdqp[j] += dHdQP[k] * dQPdqp[k][j];
	    m = 0;
	    if(fit[2])
		for(j=0; j<PT.NumberofParameters(); j++,m++)
		    for(k=0, dHda[m]=0.; k<4; k++)
  			dHda[m] += dHdQP[k] * dQPdbeta[k][j];
	    if(fit[1])
		for(j=0; j<TM.NumberofParameters(); j++,m++)
		    for(k=0, dHda[m]=0.; k<4; k++)
			dHda[m] += dHdqp[k] * dqpdalfa[k][j];
	    dHdj[0] = dHdj[1] = 0.;
            for(k=0; k<4; k++) {
	        dHdj[0] += dHdqp[k] * dqpdj[k][0];
	        dHdj[1] += dHdqp[k] * dqpdj[k][1];
	    }
	    for(j=0; j<dj1dS.NumberofTerms(); j++,m++)
		dHda[m] = dHdj[0] * dj1dS(j) + dHdj[1] * dj2dS(j);
            if(m!=mfit) cerr<<"wrong MFIT in LevMarCof()\n";
	    Hav   += H;
	    Hsqav += H*H;
	    for(k=0; k<mfit; k++) {
		dHavda[k] += dHda[k];
		bk[k]     += H * dHda[k];
		for(l=k; l<mfit; l++)
		    akl[k][l] += dHda[k] * dHda[l];
	    }
	} // done bombing around the torus
    } else { 
//
// 2. case: no fit of the Sn
//          bomb around the torus and sum up the derivatives
//
        for(i1=0; i1<GF.N_th1(); i1++)
        for(i2=0; i2<GF.N_th2(); i2++) {
	    jt = GF.Map(J(0),J(1),i1,i2);
            if (jt[0] < 0. || jt[1] < 0.) {  // negative actions
                for(k=0;k<4;k++) { delete[] dqpdalfa[k]; delete[] dQPdbeta[k]; }
                delete[] dHda; delete[] dHavda;
	        return -2;
	    }	    
            QP = jt >> TM;
	    //cerr << QP << " "; 
	    QP = QP >> PT;
	    //cerr << QP << "\n";
	    if(fit[1]) {         
	      TM.Derivatives(dqpdj, dqpdalfa);
	      PT.Derivatives(dQPdqp); 
	    }
	    
	    H = Heff(QP, dHdQP, Phi);
	    if(fit[1])
                for(j=0; j<4; j++)
                    for(k=0, dHdqp[j]=0.; k<4; k++)
	                dHdqp[j] += dHdQP[k] * dQPdqp[k][j];
	    m = 0;
	    if(fit[2])
		for(j=0; j<PT.NumberofParameters(); j++,m++)
		    for(k=0, dHda[m]=0.; k<4; k++)
  			dHda[m] += dHdQP[k] * dQPdbeta[k][j];
	    if(fit[1])
		for(j=0; j<TM.NumberofParameters(); j++,m++)
		    for(k=0, dHda[m]=0.; k<4; k++)
			dHda[m] += dHdqp[k] * dqpdalfa[k][j];
            if(m!=mfit) cerr<<"wrong MFIT in LevMarCof()\n";
	    Hav   += H;
	    Hsqav += H*H;
	    for(k=0; k<mfit; k++) {
		dHavda[k] += dHda[k];
		bk[k]     += H * dHda[k];
		for(l=k; l<mfit; l++)
		    akl[k][l] += dHda[k] * dHda[l];
	    }
	}
    }
//
// Done bombing around the torus.
// Now normalize and finally compute bk, akl, delta H, and dchisq
//
    temp    = 1./double(GF.N_th1()*GF.N_th2());
    Hav    *= temp;
    Hsqav  *= temp;
    chims  = Hsqav - Hav*Hav;
    chirms = (chims>0.)? sqrt(chims) : 0 ;
    dchisq  = 0.;
    for(k=0; k<mfit; k++) {
	dHavda[k] *= temp;
	bk[k]      = Hav * dHavda[k] - bk[k] * temp;
	dchisq    += bk[k] * bk[k];
    }
    for(k=0; k<mfit; k++)
	for(l=k; l<mfit; l++) 
	    akl[l][k] = akl[k][l] = temp * akl[k][l] - dHavda[k] * dHavda[l]; 
    dchisq = 2. * sqrt(dchisq);
    for(k=0; k<4; k++) {
	delete[] dqpdalfa[k];
	delete[] dQPdbeta[k];
    }
    
    delete[] dHda;
    delete[] dHavda;
    if(dchisq > dch_tol) { return -4;}
    return 0;
}
////////////////////////////////////////////////////////////////////////////////

static int ChirmsOnly(         // return:        error flag (see below)
    const Actions&   J,        // Input:         Actions of orbit 
    Potential*       Phi,      // Input:         pointer to Potential
    const GenFncFit& GF,       // Input:         Generating function
    const ToyMap&    TM,       // Input:         toy-potential map
    const PoiTra&    PT,       // Input:         canonical map
    double&          Hav,      // Output:        average H
    double&          chirms)   // Output:        rms deviation from average H
/*==============================================================================
on the meaning of the return value:
    0  -> everything went ok
   -2  -> negative action(s) occured at least at one point in angle space
==============================================================================*/
{
    int     i1, i2;
    PSPD    jt, QP;
    double  H, Hsqav, temp;
    Hav    = 0.;
    Hsqav  = 0.;
// bomb around the torus and sum up the H and H^2
    for(i1=0; i1<GF.N_th1(); i1++)
        for(i2=0; i2<GF.N_th2(); i2++) {
	    jt    = GF.Map(J(0),J(1),i1,i2);
            if(jt[0] < 0. || jt[1] < 0.) return -2;
            QP    = jt >> TM >> PT;
	    H     = 0.5 * (QP(2)*QP(2)+QP(3)*QP(3))
	            + Phi->eff(double(QP(0)),double(QP(1)));
            Hav  += H;
	    Hsqav+= H*H;
    }
    temp    = 1./double(GF.N_th1()*GF.N_th2());
    Hav    *= temp;
    Hsqav  *= temp;
    chirms  = sqrt(Hsqav - Hav*Hav);
    return 0;
}

////////////////////////////////////////////////////////////////////////////////

int SbyLevMar(               // return:        error flag (see below)
    const Actions&  J,     // Input:         Actions of Torus to be fit
    Potential*      Phi,     // Input:         Potential to be fit
    const int       option,  // Input:         option for fit (see below)
    const int       N_th1,   // Input:         # of theta_r for fit
    const int       N_th2,   // Input:         # of theta_th for fit
    const int       max_iter,// Input:         max. # of iterations
    const double    tol1,    // Input:         stop if   dH_rms      < tol1 ...
    const double    tol2,    // Input:         AND  if   |dchi^2/dp| < tol2
    GenPar&         Sn,      // Input/Output:  parameters of generating function
    PoiTra&         PT,      // Input/Output:  canonical map with parameters
    ToyMap&         TM,      // Input/Output:  toy-potential map with parameters
    double&         lambda,  // Input/Output:  lambda of the iterations
    double&         mean_H,  // Output:        mean H
    double&         delta_H, // Output:        rms deviation from mean
    int&            negact,  // Output:        # of occurence of neg. actions
    const double  /*exp_H*/, // Input:	       estimate of expected mean H
    const int       err)     // Input:	       error output
/*==============================================================================
   meaning of return: positive -> =number of iterations
                      -2       -> negativ actions already for input parameters
		      -3       -> singular matrix occuring (very strange indeed)
		      -4       -> dch > 10^3 or rel. change in <H>  >=  2

   meaning of option:  0       -> fit everthing
		       add 1   -> don't fit S_n of Generating function
		       add 2   -> don't fit Isochrone parameters
		       add 4   -> don't fit Point transformation parameters
==============================================================================*/
{
    GenFncFit GF(Sn, N_th1, N_th2);
    Phi->set_Lz(J(2));
    // special case: no fit, only <H> and <(H-<H>)^2> wanted
    if(option >= 7 || max_iter==0) {
	if (ChirmsOnly(J,Phi,GF,TM,PT,mean_H,delta_H)) return -2;
	return 0;
    }
// normal case: fit some parameter(s)
    int    j,k,m;
    char            fit[3];
    int             iterations=0, mfit=0, F;
    double          chirms, dchisq, dchtry, temp, damp=1.,
                    H_av, *dA, *B, **AA, **AAtry;
    double //H_av0, rHav, 
		    Rc     = Phi->RfromLc(J(2)), 	// Rc = length scale
                    vc     = WDabs(J(2)) / Rc;		// vc = velocity scale
		
    vec4 	    tp(TM.parameters()), tpy(tp),
                    tpmin(TM.lower_bounds(Rc,vc)),tpmax(TM.upper_bounds(Rc,vc));
    // if(PT.NumberofParameters()>0) {
//       tpmin[0] = 0.666*tp[0];
//       tpmax[0] = 1.5*tp[0];
//       //tpmin[3] =0.; tpmax[3] =0.;
//     } 

    // STABILITY
    //tpmin[3] =0.; tpmax[3] =0.;
    // STABILITY
// initialisation
    negact = 0;
    if(option >= 4)   fit[2] = 0;
        else        { fit[2] = 1; mfit += PT.NumberofParameters(); }
    if(option%4 >= 2) fit[1] = 0;
        else        { fit[1] = 1; mfit += TM.NumberofParameters(); }
    if(option%2 >= 1) fit[0] = 0;
        else        { fit[0] = 1; mfit += Sn.NumberofTerms(); }
    dA    = new double[mfit];        // 
    B     = new double[mfit];        // The vectors and matrices
    AA    = new  double* [mfit];     // of derivatives required for
    AAtry = new  double* [mfit];     // Levenberg-Marquad
    for(m=0; m<mfit; m++) {
	AA[m]    = new double[mfit];
	AAtry[m] = new double[mfit];
    }
//
// evaluate initial chi^2 and its derivatives
// if negative actions occur, try to salvage, then set Sn=0 and try again
    if((F=LevMarCof(fit,J,Phi,mfit,GF,TM,PT,mean_H,delta_H,dchisq,B,AA))==-2){
      if(err) cerr<<"SbyLevMar: init: NEGATIVE ACTIONS -> set S_n=0.99*S_n\n";
      GF.set_parameters((Sn *= 0.99));
      if((F=LevMarCof(fit,J,Phi,mfit,GF,TM,PT,mean_H,delta_H,dchisq,B,AA))
	 ==-2)	{
	if(err) cerr<<"SbyLevMar: init: NEGATIVE ACTIONS -> set S_n=0.5*S_n\n";
	GF.set_parameters((Sn *= 0.5));
	if((F=LevMarCof(fit,J,Phi,mfit,GF,TM,PT,mean_H,delta_H,dchisq,B,AA))
	   ==-2)	{
	  if(err) 
	    cerr<<"SbyLevMar: init: still NEGATIVE ACT. -> set S_n=0.2*S_n\n";
	  GF.set_parameters((Sn *= 0.2));
	  if((F=LevMarCof(fit,J,Phi,mfit,GF,TM,PT,mean_H,delta_H,dchisq,B,AA))
	     ==-2) {
	    if(err)  
	      cerr<<"SbyLevMar: init: still NEGATIVE ACT. -> set S_n=0.\n"; 
	    GF.set_parameters((Sn = 0.));
	    F=LevMarCof(fit,J,Phi,mfit,GF,TM,PT,mean_H,delta_H,dchisq,B,AA);
	  }
	}
      }
    }
    //H_av0 = (exp_H)? exp_H : mean_H;  // first Average value of H
    //rHav  = mean_H/H_av0;             // ratio of average to initial average
    if(err) cerr<<"SbyLevMar: init:";
    if(F) {
      if(err)
          cerr<<" LevMarCof() failed\n";
        Deletion(mfit,AA,AAtry,dA,B);
        return F; 
    } 
    if(err) 
      cerr<<" lam, <H>, dH, dch = "
          <<lambda<<' ' <<mean_H<<' '<<delta_H<<' '<< dchisq //<<'\n';
	//<<"  a=" << cpy 
	  <<"; b="<<tpy(0)<<','<<tpy(1)<<','<<tpy(2)<<','<<tpy(3)<<'\n';

//##############################################################################
// iteration (i.e. the main part of this routine)
    while((delta_H>tol1 || dchisq>tol2)&& max_iter>iterations && lambda<1.e40) {
	iterations++;
// 1. set up matrix A = ..(lambda), dA=B;
	temp = 1.+lambda;
	for(m=0; m<mfit; m++) {
	    dA[m]     = B[m];
	    for(k=0; k<mfit; k++)
	        AAtry[k][m] = AA[k][m];
            AAtry[m][m] *= temp;
	}
// 2. solve for change in parameters
        if(GaussBack(AAtry, mfit, dA)) {
            Deletion(mfit,AA,AAtry,dA,B);
	    return -3;
        }
// 3. change parameters => trial parameters, check for bounds
	m = 0;                     // Note m incremented for all 3 fit[ ]
        // if(fit[2]) {
//    	    for(j=0; j<PT.NumberofParameters(); j++,m++) {
//       		cpy[j] = cp(j)+dA[m];
// 		minmax(cpy[j],cpmin(j),cpmax(j));
// 	    }
//             PT.set_parameters(cpy);
// 	}
        if(fit[1]) {
	    for(j=0; j<TM.NumberofParameters(); j++,m++) {
		tpy[j] = tp(j)+dA[m];
		minmax(tpy[j],tpmin(j),tpmax(j));
	    }
            TM.set_parameters(tpy);
	}
        if(fit[0]) {
            GenPar Stry(Sn);
	    for(j=0; j<Sn.NumberofTerms(); (j++,m++))
		Stry[j] += damp * dA[m];
            GF.set_parameters(Stry);
	}
        if(m!=mfit) cerr<<"wrong MFIT in SbyLevMar()\n";
// 4. compute chi^2, A, b for trial parameters.
	F = LevMarCof(fit,J,Phi,mfit,GF,TM,PT,H_av,chirms,dchtry,dA,AAtry);
	//rHav = H_av/H_av0;
// 5. accept or reject trial parameters
	if(F) {
	  if(F==-2) {
	    damp = fmax(0.5*damp,0.0078125);
	    negact++;
	    if (err)
	      cerr<<"SbyLevMar: it "<<iterations<<": NEG. ACTIONS;   "
	        //<< cpy << 
		" b=" << tpy(0)<<','<<tpy(1)<<','<<tpy(2)<<','<<tpy(3) <<'\n';
	  } else 
	    if(err)
	    cerr<<"SbyLevMar: it "<<iterations<<": |dCHI^2/dA|>TOLERANCE\n";
	  lambda *= 4.;
	} else if(chirms < delta_H) {
	  damp    = fmin(2.*damp,1.);
	  mean_H  = H_av;
	  delta_H = chirms;
	  dchisq  = dchtry;
	  if(err)
	    cerr<<"SbyLevMar: it "<<iterations
	    <<": lam, <H>, dH, dch = "<<lambda<<' '<<mean_H
		<<' '<<delta_H<<' '<< dchisq//<<"  a="<< cpy 
		<<"; b="<<tpy(0)<<','<<tpy(1)<<','<<tpy(2)<<','<<tpy(3)<<'\n';
	  lambda *= 0.5;
	  //if(fit[2]) cp = PT.parameters();
	  if(fit[1]) tp = TM.parameters();
	  if(fit[0]) Sn = GF.parameters();
	  for(m=0; m<mfit; m++) {
	    B[m] = dA[m];
	    for(k=0; k<mfit; k++)
	      AA[k][m] = AAtry[k][m];
	  }
	} else {
	  if(err){
	    cerr<<"SbyLevMar: it "<<iterations<<": no improvement;"//a="<<cpy<<
		<< " dH = "<<delta_H<<"; b="<<tpy(0)<<','<<tpy(1)<<','<<tpy(2)<<','<<tpy(3)<<'\n';
	  }
	    lambda *= 4.;
        }
    }
    //if(fit[2]) PT.set_parameters(cp);
    if(fit[1]) TM.set_parameters(tp); 
    Deletion(mfit,AA,AAtry,dA,B);
    return iterations;
}
////////////////////////////////////////////////////////////////////////////////
// routine dSbyInteg() ****************************************************** //
////////////////////////////////////////////////////////////////////////////////
inline void z0NewTdim(double**& M, double*& T1, double*& T3, int& Tdim)
{
    int Nold=Tdim;
    Tdim*=2;
    double* T1new = new double[Tdim];
    double* T3new = new double[Tdim];
    double** Mnew = new double* [Tdim];
    for(int i=0; i<Nold; i++) {
	T1new[i] = T1[i];
	T3new[i] = T3[i];
	Mnew[i]  = M[i];
    }
    delete[] T1;
    delete[] T3;
    delete[] M;
    T1 = T1new;
    T3 = T3new;
    M  = Mnew;
}

inline void z0AddEquation(int*g, double** M, double* T1, double* T3,
			int& I, int& Tdim, const int Nr, const int Mdim,
			const GenPar& Sn, const double t1, const double t3,
			const double dT, const double time, int& grid)
{
  // Puts theta into the vectors T1,T3 and puts the #n -2sin(n.theta) values
  // into a column of the matrix M
    int i;
    M[I]  = new double[Mdim];
    T1[I] = t1;
    T3[I] = t3;
    for(i=0; i<Mdim-1; i++)
      M[I][i] =-2.*sin( Sn.n1(i)*t1);
    M[I][Mdim-1]= time;
    i = int(t1/dT);     // This grid enables us to check whether each cell 
                        // in theta space has been visited    
    if(i>=0 && i<Nr ) grid = (g[i]+=1);
    else                       grid = 0;
    if(Tdim==(I+=1)) z0NewTdim(M,T1,T3,Tdim);
}

inline void z0RemoveEquation(int*g, double** M, double* T1, double* /*T3*/,
			   int& I, const int Nr, const double dT)
{
    I--;
    int i;
    delete[] M[I];
    i = int(T1[I]/dT);
    if(i>=0 && i<Nr) g[i] -= 1;
}

inline void z0RemoveStrip(int*g, double**M, double*T1, double*T3, int*Kl,
		        int& I, const int K, const int Nr, const double dT)
{
    int k=(K==0)? 0 : Kl[K-1];
    while(I>k) z0RemoveEquation(g,M,T1,T3,I,Nr,dT);
}

static int z0AddStrip(const PSPT& Jt3, Potential* Phi, const GenPar& Sn,
		    const GenFnc& GF, const PoiTra& PT, const ToyMap& TM,
                    double& dtm, double**M, double*T1, double*T3, 
		    int*Kl, int*g,
		    int& I, int& K, int& Tdim, const int Nr, const int Mdim, 
		    const int NMIN, int& integ, const int INTOL, const int err)
{
    const double    Etol = 1.e-14,
		    tiny = 1.e-6;
    int toruserrno = 0;
    int             grid;
    double          dtime= pow(2.,-30);  // Tiny number, ensuring v. accurate RK
                                         // integration. Kinda arbitrary, no?
    int    n;
   // PSPD   jt=Jt>>GF, QP;
    PSPT   jt3=Jt3>>GF, QP3;
    double dt,dt1,dt3,time=0.,t1=Jt3(3),t3=Jt3(5),//t3new,Lperp,u,
		    dT=Pi/double(Nr);
    if(jt3(0)<0.) jt3[0]=tiny; 
    QP3 = jt3 >> TM >> PT;
    //cerr << jt3(5) << ' ' << QP3(2);
    jt3    = QP3 << PT << TM;
    //cerr << ' ' << jt3(5) << '\n';
    if(toruserrno) {
	if(err) cerr<<"error in map, toruserrno="<<toruserrno<<'\n';
        return 1;
    }
    z0AddEquation(g,M,T1,T3,I,Tdim,Nr,Mdim,Sn,t1,t3,dT,time,grid);
    Record3D R(QP3, Phi);
    R.set_tolerance(Etol);
    R.set_maxstep(dtm);
    for(n=0; (t1<Pi && !grid) || n<NMIN; ) {
      for(dt=0.; dt<dT; ) {
	    R.stepRK_by(dtime);
	    time += dtime;
	    jt3    = R.QP3D() << PT << TM;
	    //cerr << jt3(5) << ' ' << (R.QP3D())(2) << '\n';
	    if(INTOL<=integ++ || toruserrno) {
		if(err && toruserrno) 
		    cerr<<" error in map, toruserrno="<<toruserrno<<'\n';
		if(n<NMIN) {
		    z0RemoveStrip(g,M,T1,T3,Kl,I,K,Nr,dT);
		    return 1;
		} else {
                    Kl[K++]=I;
                    return 0;
		}
	    }
    	    if(     jt3(3)-t1 <-4.) t1 += (dt1=jt3(3)+TPi-t1);
    	    else if(jt3(3)-t1 > 4.) t1 += (dt1=jt3(3)-TPi-t1);
    	    else                    t1 += (dt1=jt3(3)-t1);
	    if(     jt3(5)-t3 <-4.) t3 += (dt3=jt3(5)+TPi-t3);
	    else if(jt3(5)-t3 > 4.) t3 += (dt3=jt3(5)-TPi-t3);
	    else                    t3 += (dt3=jt3(5)-t3);
	    //if(     (R.QP3D())(5)-t3 <-4.) t3 += (dt3=(R.QP3D())(5)+TPi-t3);
	    //else if((R.QP3D())(5)-t3 > 4.) t3 += (dt3=(R.QP3D())(5)-TPi-t3);
	    //else                    t3 += (dt3=(R.QP3D())(5)-t3);
	    if(dt1 > dT) {
		dtime*= 0.5*dT/dt1;
		dtm   = dtime;
		R.set_maxstep(dtm);
	    }
            dt += dt1;
	}
	n++;
	//cerr << t1 << ' ' << t3 << '\n';
        z0AddEquation(g,M,T1,T3,I,Tdim,Nr,Mdim,Sn,t1,t3,dT,time,grid);
    }
    Kl[K++]=I;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
int z0dSbyInteg(               // return:     error flag (see below)
    const Actions& J,        // Input:      Actions of Torus to be fit
    Potential* Phi,          // Input:      pointer to Potential
    const int Nr,            // Input:      # of grid cells in Pi
    const GenPar& Sn,        // Input:      parameters of generating function
    const PoiTra& PT,        // Input:      canonical map with parameters
    const ToyMap& TM,        // Input:      toy-potential map with parameters
    const double  /*dO*/,        // Input:      delta Omega
    Frequencies   &Om,	     // In/Output:  Omega_r, Omega_l
    Errors   &chi,	     // Output:     chi_rms for fit of dSn/dJi
    AngPar& Ap,              // Output:     dSn/dJr & dSn/dJl
    const int IperCell,	     // Input:	    max tol steps on average per cell
    const int err)	     // Input:	    error output?
//==============================================================================
// meaning of return:  0       -> everything went ok 
//                    -1       -> N too small, e.g., due to:
//                                - H>0 occured too often in backward ToyIsochrone
//                                - orbit integrations stopped (too long)
//                    -4       -> negative Omega => reduce Etol
//                    -5       -> Matrix M^tM not pos. def. => something wrong
//		      -6       -> max{(dS/dJ)_n} > 1
//==============================================================================
{
             //char   constraint=(dO<Om(0) && dO<Om(1) && dO>0.);
    const    int    Nm=12;
             int    *g, *Kl, I=0, K=0, integ=0, Tdim, xdim, Mdim,
		    MAXHPOS=(Nr*Nr)/5, INTOL=Nr*Nr*IperCell; 
             double  **M, *T1, *T3;
    int    hpos=0, F, i, j, k, n, i1=0;
             double dtm = 1.;
    double temp;
             PSPT   Jt3;
             GenFnc GF(Sn);

	     Ap = AngPar(Sn,Sn,Sn);

// 1. initialize and allocate memory
    for(i=0;i!=3;i++) Jt3[i] = J(i);
    Mdim  = Sn.NumberofTerms()+1;
    Tdim  = fmax(200, 8*Nr);
    Kl    = new int[Tdim];
    T1    = new double[Tdim];
    T3    = new double[Tdim];
    M     = new double* [Tdim];
    g     = new int [Nr];
    for(i=0; i<Nr; i++) g[i] = 0;
    if(err) cerr<<"dSbyInteg: ";
    bool fail=false;
// 2.  fill in the matrix M and vectors T1, T3 by orbit integrations
// 2.1 add strips to fill the theta plane densely (with resolution Pi/Nr)
    for(i1=0; i1<Nr && !fail; i1++) if(!g[i1]) {
	  // Make sure there's points everywhere on the grid (either start 
	  // point or gets orbit integrated to). 0 < theta < Pi.
        Jt3[3] = Pi*(i1+.6)/double(Nr);
        Jt3[4] = 0.;
	Jt3[5] = Pi*(i1+.6)/double(Nr);
	//cerr << "Jt3[5]= "<< Jt3(5)<< "\n";
        F=z0AddStrip(Jt3,Phi,Sn,GF,PT,TM,dtm,M,T1,T3,Kl,g,I,K,Tdim,Nr,Mdim,Nm,
		   integ,INTOL,err);
        if(integ > INTOL) {
	  if(err) cerr<<" too many time steps"; 
	  fail = true; }
	if(F) hpos++; 
        if(hpos++ > MAXHPOS) {
	  if(err)cerr<<" too many map errors"; 
	  fail = true; }
      }
// 2.2 if omega is constrained, add equation
//    if(I && constraint) {
//        A    = 1./dO;
//        M[I] = new double[Mdim];
//        for(i=0; i<Mdim-1; i++) M[I][i]=0.;
//        M[I][Mdim-1] = A;
//	T1[I] = A*Om(0);
//
//	I+=1;
//    }
 // 2.3 compute number of unknowns, abort if more than half number of equations
    xdim = K+Mdim;
    if(I<2*xdim) {
      for(i=0; i<I; i++)  delete[] M[i];
      delete[] g; delete[] M; delete[] T1; delete[] T3; delete[] Kl;
      if(err) cerr<<"  "<<xdim<<" unknowns in "<<I<<" equations\n";
      return -1;
    }
    if(err) cerr<<"linear eqs";
// 3. compute normal equations; and solve for omega and dSdJi
    double **N = new double* [xdim];
    for(i=0; i<Mdim; i++) {
        N[i] = new double[xdim];
// 3.1 Nij i,j = dS/dJ, Omega
	for(j=i; j<Mdim; j++) {
	    temp = M[0][i] * M[0][j];
	    for(n=1; n<I; n++)
	      temp += M[n][i] * M[n][j]; // Sum over all theta
            N[i][j] = temp;
	}
// 3.2 Nij i = dS/dJ, Omega; j = phases
	for(n=k=0,j=Mdim; k<K; k++,j++) {
	    for(temp=0.; n<Kl[k]; n++)
		temp += M[n][i];
	    N[i][j] = temp;
	}
    }
// 3.3 Nij i,j = phases
    for(n=k=0,i=Mdim; k<K; n=Kl[k],k++,i++) {
        N[i] = new double[xdim];
	N[i][i] = Kl[k]-n;
	for(j=i+1; j<xdim; j++)
	    N[i][j] = 0.;
    }
    if(err) cerr<<" and normal eqs computed";
// 3.4 Cholesky decompose the normal equations
// Not quite the usual (Press et al) method, which gives a matrix and a vector.
// In this case the vector is instead given as the diagonal of the matrix, which
// seems to make sense.
    if(CholeskyDecomposition(N,xdim)) {
        for(i=0; i<I; i++)    delete[] M[i];
        for(i=0; i<xdim; i++) delete[] N[i];
	delete[] g; delete[] M; delete[] T1; delete[] T3; 
	delete[] Kl; delete[] N; 
        return -5;
    }
    //cerr << "\n\n";
    //for(i=0; i!=I; i++) cerr << T3[i] << "\n";
    if(err) cerr<<", N decomposed";
    double *L1 = new double[xdim];
    double *L3 = new double[xdim];
// 3.5 Solve for Omega_r, dS/dJr, phases_r
    for(i=0; i<Mdim; i++)
	for(L1[i]=0.,n=0; n<I; n++)
	    L1[i] += M[n][i] * T1[n];
    for(n=k=0,i=Mdim; k<K; k++,i++)
	for(L1[i]=0.; n<Kl[k]; n++)
	    L1[i] += T1[n];
    // Find vector V such that N V = L1 (original N, not the output of 
    // CholeskyDecomposition, which is a halfway house to this) 
    CholeskySolution((const double**)N,xdim,L1); // Not standard CS either, this
    for(n=k=0,chi[1]=0.; k<K; k++)               // returns answer (V) in L1, 
      for(;n<Kl[k];n++) {                        // not in a separate vector.
	    temp = L1[Mdim+k];
            for(i=0; i<Mdim; i++)
	        temp += M[n][i] * L1[i];
	    chi[1] += pow(T1[n]-temp,2);
        }
    chi[1] = sqrt(chi(1)/double(I));
    Om[0]  = L1[Mdim-1];
    for(i=0; i<Mdim-1; i++)
      Ap.dSdJ1(i,L1[i]);

    chi[2] = 0.;
    //Om[1]  = 0.; //Leave unchanged
    for(i=0; i<Mdim-1; i++)
      Ap.dSdJ2(i,0.); 
    //      3.6 Solve for Omega_phi, dS/dJphi, phases_phi
    for(i=0; i<Mdim; i++)
	for(L3[i]=0.,n=0; n<I; n++)
	    L3[i] += M[n][i] * T3[n];
    for(n=k=0,i=Mdim; k<K; k++,i++)
	for(L3[i]=0.; n<Kl[k]; n++)
	    L3[i] += T3[n];
    CholeskySolution((const double**)N,xdim,L3);
    for(n=k=0,chi[3]=0.; k<K; k++)
	for(;n<Kl[k];n++) {
	    temp = L3[Mdim+k];
            for(i=0; i<Mdim; i++)
	        temp += M[n][i] * L3[i];
	    chi[3] += pow(T3[n]-temp,2);
        }
    chi[3] = sqrt(chi(3)/double(I));// Need to sort out something out
    Om[2]  = L3[Mdim-1];
    for(i=0; i<Mdim-1; i++) {
      //cerr << i << ' ' << L3[i] << "\t\t";
      Ap.dSdJ3(i,L3[i]); 
    }
    if(err) cerr<<", and eqs. solved\nchi = " << chi << "\n";

    for(i=0; i<I; i++)    delete[] M[i];
    for(i=0; i<xdim; i++) delete[] N[i];
    delete[] g; delete[] M; delete[] T1; delete[] T3; delete[] Kl;
    delete[] N; delete[] L1; delete[] L3; 
    if(Om(0)<0. || Om(1)<0.) return -4;
    if(Ap.dSdJ1().maxS()>1.f || Ap.dSdJ2().maxS()>1.f) return -5;
    return 0;
}	   

////////////////////////////////////////////////////////////////////////////////

inline void NewTdim(double**& M, double*& T1, double*& T2, double*& T3, int& Tdim)
{
    int Nold=Tdim;
    Tdim*=2;
    double* T1new = new double[Tdim];
    double* T2new = new double[Tdim];
    double* T3new = new double[Tdim];
    double** Mnew = new double* [Tdim];
    for(int i=0; i<Nold; i++) {
	T1new[i] = T1[i];
	T2new[i] = T2[i];
	T3new[i] = T3[i];
	Mnew[i]  = M[i];
    }
    delete[] T1;
    delete[] T2;
    delete[] T3;
    delete[] M;
    T1 = T1new;
    T2 = T2new;
    T3 = T3new;
    M  = Mnew;
}

inline void AddEquation(int**g, double** M, double* T1, double* T2, double* T3,
			int& I, int& Tdim, const int Nr, const int Mdim,
			const GenPar& Sn, const double t1, const double t2,
			const double t3, const double dT, const double time, int& grid)
{
  // Puts theta into the vectors T1,T2,T3 and puts the #n -2sin(n.theta) values
  // into a column of the matrix M
    int i,j;
    M[I]  = new double[Mdim];
    T1[I] = t1;
    T2[I] = t2;
    T3[I] = t3;
    for(i=0; i<Mdim-1; i++)
	M[I][i] =-2.*sin( Sn.n1(i)*t1+Sn.n2(i)*t2 );
    M[I][Mdim-1]= time;
    i = int(t1/dT);     // This grid enables us to check whether each cell 
    j = int(t2/dT);     // in thetaR/z space has been visited    
    if(i>=0 && i<Nr && j>=0 && j<Nr) grid = (g[i][j]+=1);
    else                             grid = 0;
    if(Tdim==(I+=1)) NewTdim(M,T1,T2,T3,Tdim);
}

inline void RemoveEquation(int**g, double** M, double* T1, double* T2, double* /*T3*/,
			   int& I, const int Nr, const double dT)
{
    I--;
    int i,j;
    delete[] M[I];
    i = int(T1[I]/dT);
    j = int(T2[I]/dT);
    if(i>=0 && i<Nr && j>=0 && j<Nr) g[i][j] -= 1;
}

inline void RemoveStrip(int**g, double**M, double*T1, double*T2, double*T3, int*Kl,
		        int& I, const int K, const int Nr, const double dT)
{
    int k=(K==0)? 0 : Kl[K-1];
    while(I>k) RemoveEquation(g,M,T1,T2,T3,I,Nr,dT);
}

static int AddStrip(const PSPT& Jt3, Potential* Phi, const GenPar& Sn,
		    const GenFnc& GF, const PoiTra& PT, const ToyMap& TM,
                    double& dtm, double**M, double*T1, double*T2, double*T3, 
		    int*Kl, int**g,
		    int& I, int& K, int& Tdim, const int Nr, const int Mdim, 
		    const int NMIN, int& integ, const int INTOL, const int err)
{
  //if(err) cerr << " integ in  " << integ<< "\n";
    const double    Etol = 1.e-14,
		    tiny = 1.e-6;
    int toruserrno = 0;
    int             grid;
    double          dtime= pow(2.,-30);  // Tiny number, ensuring v. accurate RK
                                         // integration. Kinda arbitrary, no?
    int    n;
    //PSPD   jt=Jt>>GF, QP;
    PSPT   jt3=Jt3>>GF, QP3;
    double dt,dt1,dt2,dt3,time=0.,t1=Jt3(3),t2=Jt3(4),t3=Jt3(5),//t3new,Lperp,u,
           dT=Pi/double(Nr);
    if(jt3(0)<0.) jt3[0]=tiny;
    if(jt3(1)<0.) jt3[1]=tiny;
    QP3 = jt3 >> TM >> PT;
    //cerr << QP3 << " " << jt3 << "\n";
    //cerr << jt3 << "\n";
    if(toruserrno) {
	if(err) cerr<<"error in map, toruserrno="<<toruserrno<<'\n';
            return 1;
    }
    
    AddEquation(g,M,T1,T2,T3,I,Tdim,Nr,Mdim,Sn,t1,t2,t3,dT,time,grid);
    Record3D R(QP3,Phi);
    R.set_tolerance(Etol);
    R.set_maxstep(dtm);
    for(n=0; (t1<Pi && t2<Pi && !grid) || n<NMIN; ) {
      //if(t1<Pi && t2<Pi && !grid) 
      //cerr << "This is a surprise" << t1 << ' ' << t2 << "\n" 
      //     << "See Fit.cc, AddStrip() and see if you can understand it\n";
      for(dt=0.; dt<dT; ) {
	R.stepRK_by(dtime);
	//cerr << R.QP3D(2)  << "    ";
	    time += dtime;
	    jt3    = R.QP3D() << PT << TM;
	    //cerr << jt3 << "\n";
	    //if(!(integ%100)) cerr << integ <<" "<<  R.QP3D() << " " << dtime<< "\n";
	    if(INTOL<=integ++ || toruserrno) {
		if(err && toruserrno) 
		    cerr<<" error in map, toruserrno="<<toruserrno<<'\n';
		if(n<NMIN) {
		  RemoveStrip(g,M,T1,T2,T3,Kl,I,K,Nr,dT);
		    return 1;
		} else {
                    Kl[K++]=I;
                    return 0;
		}
	    }
    	    if(     jt3(3)-t1 <-4.) t1 += (dt1=jt3(3)+TPi-t1);
    	    else if(jt3(3)-t1 > 4.) t1 += (dt1=jt3(3)-TPi-t1);
    	    else                    t1 += (dt1=jt3(3)-t1);
    	    if(     jt3(4)-t2 <-4.) t2 += (dt2=jt3(4)+TPi-t2);
    	    else if(jt3(4)-t2 > 4.) t2 += (dt2=jt3(4)-TPi-t2);
    	    else                    t2 += (dt2=jt3(4)-t2);
	    if(     jt3(5)-t3 <-4.) t3 += (dt3=jt3(5)+TPi-t3);
    	    else if(jt3(5)-t3 > 4.) t3 += (dt3=jt3(5)-TPi-t3);
    	    else                    t3 += (dt3=jt3(5)-t3);
	    //cerr << t1 << ' '<< t2 << ' ';
	    if((dt1=hypot(dt1,dt2)) > dT) {
		dtime*= 0.5*dT/dt1;
		dtm   = dtime;
		R.set_maxstep(dtm);
	    }
            dt += dt1;
	    //cerr << "  time  " << time << "\t";
	    //if(!(integ%3)) cerr << "\n";
	}
	n++;
        AddEquation(g,M,T1,T2,T3,I,Tdim,Nr,Mdim,Sn,t1,t2,t3,dT,time,grid);
    }
    //if(err) cerr << " integ out  " << integ<< "\n";
    Kl[K++]=I;
    return 0;
}
	   
////////////////////////////////////////////////////////////////////////////////
int dSbyInteg(               // return:     error flag (see below)
    const Actions& J,        // Input:      Actions of Torus to be fit
    Potential* Phi,          // Input:      pointer to Potential
    const int Nr,            // Input:      # of grid cells in Pi
    const GenPar& Sn,        // Input:      parameters of generating function
    const PoiTra& PT,        // Input:      canonical map with parameters
    const ToyMap& TM,        // Input:      toy-potential map with parameters
    const double  dO,        // Input:      delta Omega
    Frequencies   &Om,	     // In/Output:  Omega_r, Omega_l
    Errors   &chi,	     // Output:     chi_rms for fit of dSn/dJi
    AngPar& Ap,              // Output:     dSn/dJr & dSn/dJl
    const int IperCell,	     // Input:	    max tol steps on average per cell
    const int err)	     // Input:	    error output?
//==============================================================================
// meaning of return:  0       -> everything went ok 
//                    -1       -> N too small, e.g., due to:
//                                - H>0 occured too often in backward ToyIsochrone
//                                - orbit integrations stopped (too long)
//                    -4       -> negative Omega => reduce Etol
//                    -5       -> Matrix M^tM not pos. def. => something wrong
//		      -6       -> max{(dS/dJ)_n} > 1
//==============================================================================
{
  if(!J(1)) {
    int F = z0dSbyInteg(J,Phi,Nr,Sn,PT,TM,dO,Om,chi,Ap,IperCell,err);
    return F;
  }
             //bool   constraint=false; //History 
    const    int    Nm=12;
             int    **g, *Kl, I=0, K=0, integ=0, Tdim, xdim, Mdim,
		    MAXHPOS=(Nr*Nr)/5, INTOL=Nr*Nr*IperCell; 
             double  **M, *T1, *T2, *T3;
    int    hpos=0, F, i, j, k, n, i1=0, i2=0;
             double dtm = 1.;
    double temp;
             PSPT   Jt3;
             GenFnc GF(Sn);

    Ap = AngPar(Sn,Sn,Sn);

// 1. initialize and allocate memory
    for(i=0;i!=3;i++) Jt3[i] = J(i);
    Mdim  = Sn.NumberofTerms()+1;
    Tdim  = fmax(200, 8*Nr*Nr);
    Kl    = new int[Tdim];
    T1    = new double[Tdim];
    T2    = new double[Tdim];
    T3    = new double[Tdim];
    M     = new double* [Tdim];
    g     = new int* [Nr];
    for(i=0; i<Nr; i++) {
	g[i] = new int[Nr];
	for(j=0; j<Nr; j++)
	    g[i][j] = 0;
    }

    if(err) cerr<<"dSbyInteg: ";
    bool fail=false;
// 2.  fill in the matrix M and vectors T1, T2 by orbit integrations
// 2.1 add strips to fill the theta plane densely (with resolution Pi/Nr)
    for(i2=0; i2<Nr && !fail; i2++) 
      for(i1=0; i1<Nr  && !fail; i1++) if(!g[i1][i2]) {
	  // Make sure there's points everywhere on the grid (either start 
	  // point or gets orbit integrated to). 0 < theta < Pi.
        Jt3[3] = Pi*(i1+.0)/double(Nr);
        Jt3[4] = Pi*(i2+.0)/double(Nr);
	Jt3[5] = 0*Pi*((Nr-1-i2)+.6)/double(Nr); // no, it doesn't really matter
        F=AddStrip(Jt3,Phi,Sn,GF,PT,TM,dtm,M,T1,T2,T3,Kl,g,I,K,Tdim,Nr,Mdim,Nm,
		   integ,INTOL,err);
        if(integ > INTOL) {
	  if(err) cerr<<" too many time steps "<< integ << ' ' << i1 << i2 <<'\n'; 
	  fail = true;}
	if(F) hpos++; 
        if(hpos++ > MAXHPOS) {
	  if(err)cerr<<" too many map errors"; 
	  fail = true;}
	}
// 2.2 if omega is constrained, add equation
//    if(I && constraint) {
//        A    = 1./dO;
//        M[I] = new double[Mdim];
//        for(i=0; i<Mdim-1; i++) M[I][i]=0.;
//        M[I][Mdim-1] = A;
//	T1[I] = A*Om(0);
//	T2[I] = A*Om(1);
//	I+=1;
//    }
 // 2.3 compute number of unknowns, abort if more than half number of equations
    //for(i=0;i!=I;i++) cerr << i <<' ' << T3[i]<< '\n'; 
    xdim = K+Mdim;
    if(I<2*xdim) {
        for(i=0; i<Nr; i++) delete[] g[i];
	for(i=0; i<I; i++)  delete[] M[i];
	delete[] g; delete[] M; delete[] T1; delete[] T2; delete[] T3; 
	delete[] Kl;
	if(err) cerr<<"  "<<xdim<<" unknowns in "<<I<<" equations\n";
        return -1;
    }
    if(err) cerr<<"linear eqs";
// 3. compute normal equations; and solve for omega and dSdJi
    double **N = new double* [xdim];
    for(i=0; i<Mdim; i++) {
        N[i] = new double[xdim];
// 3.1 Nij i,j = dS/dJ, Omega
	for(j=i; j<Mdim; j++) {
	    temp = M[0][i] * M[0][j];
	    for(n=1; n<I; n++)
	      temp += M[n][i] * M[n][j]; // Sum over all theta
            N[i][j] = temp;
	}
// 3.2 Nij i = dS/dJ, Omega; j = phases
	for(n=k=0,j=Mdim; k<K; k++,j++) {
	    for(temp=0.; n<Kl[k]; n++)
		temp += M[n][i];
	    N[i][j] = temp;
	}
    }
// 3.3 Nij i,j = phases
    for(n=k=0,i=Mdim; k<K; n=Kl[k],k++,i++) {
        N[i] = new double[xdim];
	N[i][i] = Kl[k]-n;
	for(j=i+1; j<xdim; j++)
	    N[i][j] = 0.;
    }
    if(err) cerr<<" and normal eqs computed";
// 3.4 Cholesky decompose the normal equations
// Not quite the usual (Press et al) method, which returns a matrix and a vector.
// In this case the vector is instead given as the diagonal of the matrix, which
// seems to make sense.
    if(CholeskyDecomposition(N,xdim)) {
        for(i=0; i<Nr; i++)   delete[] g[i];
	for(i=0; i<I; i++)    delete[] M[i];
        for(i=0; i<xdim; i++) delete[] N[i];
	delete[] g; delete[] M; delete[] T1; 
	delete[] T2; delete[] T3; delete[] Kl; delete[] N; 
        return -5;
    }
    if(err) cerr<<", N decomposed";
    double *L1 = new double[xdim];
    double *L2 = new double[xdim];
    double *L3 = new double[xdim];
    // ofstream to;
//     to.open("thetas.tab");
//     for(n=k=0,i=Mdim; k<K; k++,i++) {
//       for(;n<Kl[k]; n++)
// 	to << T1[n] << " " << T2[n] << "\n";
//       to << "0 0\n";
//     }
//     to.close();
// 3.5 Solve for Omega_r, dS/dJr, phases_r
    for(i=0; i<Mdim; i++)
	for(L1[i]=0.,n=0; n<I; n++)
	    L1[i] += M[n][i] * T1[n];
    for(n=k=0,i=Mdim; k<K; k++,i++)
	for(L1[i]=0.; n<Kl[k]; n++)
	    L1[i] += T1[n];
    // Find vector V such that N V = L1 (original N, not the output of 
    // CholeskyDecomposition, which is a halfway house to this) 
    CholeskySolution((const double**)N,xdim,L1); // Not standard CS either, this
    for(n=k=0,chi[1]=0.; k<K; k++)               // returns answer (V) in L1, 
      for(;n<Kl[k];n++) {                        // not in a separate vector.
	    temp = L1[Mdim+k];
            for(i=0; i<Mdim; i++)
	        temp += M[n][i] * L1[i];
	    chi[1] += pow(T1[n]-temp,2);
        }
    chi[1] = sqrt(chi(1)/double(I));
    Om[0]  = L1[Mdim-1];
    for(i=0; i<Mdim-1; i++)
      Ap.dSdJ1(i,L1[i]);
// 3.6 Solve for Omega_l, dS/dJl, phases_l
    for(i=0; i<Mdim; i++)
	for(L2[i]=0.,n=0; n<I; n++)
	    L2[i] += M[n][i] * T2[n];
    for(n=k=0,i=Mdim; k<K; k++,i++)
	for(L2[i]=0.; n<Kl[k]; n++)
	    L2[i] += T2[n];
    CholeskySolution((const double**)N,xdim,L2);
    for(n=k=0,chi[2]=0.; k<K; k++)
	for(;n<Kl[k];n++) {
	    temp = L2[Mdim+k];
            for(i=0; i<Mdim; i++)
	        temp += M[n][i] * L2[i];
	    chi[2] += pow(T2[n]-temp,2);
        }
    chi[2] = sqrt(chi(2)/double(I));
    Om[1]  = L2[Mdim-1];
    for(i=0; i<Mdim-1; i++){
      //cerr << i << ' ' << L2[i] << "\t";
      Ap.dSdJ2(i,L2[i]); 
    }
    // 3.6 Solve for Omega_phi, dS/dJphi, phases_phi
    for(i=0; i<Mdim; i++)
	for(L3[i]=0.,n=0; n<I; n++)
	    L3[i] += M[n][i] * T3[n];
    for(n=k=0,i=Mdim; k<K; k++,i++)
	for(L3[i]=0.; n<Kl[k]; n++)
	    L3[i] += T3[n];
    CholeskySolution((const double**)N,xdim,L3);
    for(n=k=0,chi[3]=0.; k<K; k++)
	for(;n<Kl[k];n++) {
	    temp = L3[Mdim+k];
            for(i=0; i<Mdim; i++)
	        temp += M[n][i] * L3[i];
	    chi[3] += pow(T3[n]-temp,2);
        }
    chi[3] = sqrt(chi(3)/double(I));// Need to sort out something out
    Om[2]  = L3[Mdim-1];
    for(i=0; i<Mdim-1; i++) {
      //cerr << i << ' ' << L3[i] << "\t\t";
      Ap.dSdJ3(i,L3[i]); 
    }
    if(err) cerr<<", and eqs. solved\nchi = " << chi << "\n";
    
    for(i=0; i<Nr; i++)   delete[] g[i];
    for(i=0; i<I; i++)    delete[] M[i];
    for(i=0; i<xdim; i++) delete[] N[i];
    delete[] g; delete[] M; delete[] T1; delete[] T2; delete[] T3; delete[] Kl;
    delete[] N; delete[] L1; delete[] L2; delete[] L3;
    if(Om(0)<0. || Om(1)<0.) return -4;
    if(Ap.dSdJ1().maxS()>1.f || Ap.dSdJ2().maxS()>1.f) return -6;
    return 0;
}
////////////////////////////////////////////////////////////////////////////////
// routine Omega() ********************************************************** //
////////////////////////////////////////////////////////////////////////////////
int Omega(                   // return:    error flag, see below
    Potential     *Phi,      // Input:     pointer to Potential
    const Actions &J,        // Input:     actions
    const GenPar  &Sn,       // Input:     parameters of generating function
    const PoiTra  &PT,       // Input:     canonical map with parameters
    const ToyMap  &TM,       // Input:     toy-potential map with parameters
    const double  t1,        // Input:     start toy angle
    const double  t2,        // Input:     start toy angle
    const double  thmax,     // Input:     difference in angle to integrate over
    Frequencies   &Om,	     // Output:	   Omega_r, Omega_l, Omega_p
    double        &dOrl,     // Output:    delta(Omega_rl)
    double        &dOp)      // Output:    delta(Omega_phi)
// meaning of the return value:   0     everything ok
//			         -1     too many time steps (never gets there)
//				 -2     error in backward maps
{
    const    double Etol=1.e-14;
    const    int    Intol=100000;
    int    N=1,integ=0;
    double det,t=0,th1=t1,th2=t2,x,
		    St=0.,Stt=0.,S1=t1,S2=t2,S1t=0.,S2t=0.,
		    Sx=0.,Sxx=0;
    double          dt=1.e-4;
    GenFnc          GF(Sn);
    PSPD   jt,QP;
    int toruserrno = 0;
    QP = PSPD(J(0),J(1),t1,t2) >> GF >> TM;
    if(toruserrno) return -2;
    QP = QP >> PT;
    if(toruserrno) return -2;
    Phi->set_Lz(J(2));
    Record X(QP,Phi);
    X.set_tolerance(Etol);
    while( (th1-t1<thmax || th2-t2<thmax) && Intol > integ++ ) {
// integrate one time step
        X.stepRK_by(dt);
	t+=dt;
// compute toy angles, shift to get increasing order and add to sums
	jt = X.QP() << PT << TM;
	if(toruserrno) return -2;
	if(std::isnan(jt(2)) || std::isinf(jt(2)) || fabs(jt(2))>INT_MAX)
	  jt[2] = th1+0.0001;        // in case of major failure
	if(std::isnan(jt(3)) || std::isinf(jt(3)) || fabs(jt(3))>INT_MAX)
	  jt[3] = th2+0.0001;        // in case of major failure
	while(jt(2)<th1-Pi) jt[2]+=TPi;
	while(jt(3)<th2-Pi) jt[3]+=TPi;
	th1 = jt(2);
	th2 = jt(3);
	N++;
	St += t;
	Stt+= t*t;
	S1 += th1;
	S1t+= th1*t;
	S2 += th2;
	S2t+= th2*t;
	x   = pow(X.QP(0),-2);
	Sx += x;
	Sxx+= x*x;
    }
// compute Omega from a least square fit of a straight line to theta(t)
    det   = N*Stt-St*St;
    Om[0] =(N*S1t-St*S1) / det;
    Om[1] = (J(1)==0.)? 0. : (N*S2t-St*S2) / det;
    dOrl  = sqrt(double(N)/det);
    Om[2] = J(2)*Sx/double(N);
    dOp   = J(2)*sqrt(Sxx)/double(N);

    if(integ >= Intol) return -1;
    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// routine FullFit() ******************************************************** //
////////////////////////////////////////////////////////////////////////////////
inline double NearResonant(const Frequencies &Om)
{
    double res=1., O01=Om(0)/Om(1), O10=Om(1)/Om(0);
    res = fmin(res, WDabs(O01 - 1.) );
    res = fmin(res, WDabs(2. - O10) );
    res = fmin(res, WDabs(O01 - 2.) );
    res = fmin(res, WDabs(3.-2*O10) );
    res = fmin(res, WDabs(2*O01-3.) );
    res = fmin(res, WDabs(4.-3*O10) );
    res = fmin(res, WDabs(3*O01-4.) );
    return res;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

int AllFit(	            // return:	error flag (see below)
    const Actions &J,       // input:	Actions of Torus to be fit
    Potential     *Phi,	    // input:	pointer to Potential
    const double  tol,	    // input:	goal for |dJ|/|J|
    const int	  Max,	    // input:	max. number of S_n
    const int	  Ni,	    // input:	max. number of iterations
    const int     OD,	    // input:	overdetermination of eqs. for angle fit
    const int     Nrs,      // input:	min. number of cells for fit of dS/dJ
    PoiTra	  &PT,      // in/output:	canonical map with parameters
    ToyMap	  &TM,      // in/output:	toy-pot. map with parameters
    GenPar	  &SN,      // in/output:	parameters of gen. function
    AngPar	  &AP,      // output:	dSn/dJr & dSn/dJl
    Frequencies   &Om,	    // output:	O_r, O_l, O_p
    double	  &Hav,     // output:	mean H
    Errors        &d, 	    // output:	actual |dJ|/|J| & chirms of angle fits
    const int     type,     // input:   Type of Fit (Full (0)/Half(1))
    const bool    safe,     // input:   Safe (vary one thing at a time) fit? 
    const int     Nta,      // input:   Number of tailorings
    const int     ipc,	    // input:   max tol steps on average per cell
    const double  eH,       // input:	estimate of expected mean H
    const int     Nth,	    // input:	min. No of theta per dimension
    const int     err,	    // input:	error output?
    const bool useNewAngMap)// input:   whether to use new method for angle mapping
{

  const    int    Ni0 = 10;            // max. No of iteration in 0. fit
  const    double  ta0 = 5.e-5,        // threshold for deletion of Sn
    tb0 = 3.e-3;   	               // threshold for creation of Sn
  const    double BIG = 1.e10,         // tolerance on H if otherwise undefined
    off0 = 0.002,   	  	       // threshold for setting Sn=0
    l0  = 1./128., 	  	       // default start value of lambda
    //tl0 = 3.e-7,   	  	       // tolerance for dchisq of 0. fit
    tl1 = 3.e-7,   	  	       // tolerance for dchisq of 1. & 2. fit
    tl8 = 3.e-5,          	       // tolarance for dchisq/(J*O) of --
    tla = 1.e-2;   	  	       // tolerance for chi_rms of angle fit
  int    Ni1 = Ni/5,          // max. No of iteration in 1. fit
    Ni2 = (Nta)? (Ni-Ni1)/Nta : Ni-Ni1,// max. No of iteration in 2. fit
    n1 = fmax(Nth, 6*(SN.NumberofN1()/4+1)),
    n2 = fmax(Nth, 6*(SN.NumberofN2()/4+1)),
    nrs= Nrs, F, sf=(safe)? 2 : 0;
  double tl9=0.95*tol, fac=1., Jabs, tlH, tlC, rs;
  int    ngA;
  double  tailorparam = (J(0) > J(1) && J(1))? J(0)/J(1) : 
                       (J(1) > J(0) && J(0))? J(1)/J(0) : 1;
  double ta = ta0*tailorparam, tb = tb0*tailorparam, off = off0*tailorparam;

  double l,dH,dO,dOp,odJ;
  GenPar oSN;
  // Prepare for fit.
  if(Ni1<1 || J(0)<=0. || J(1)<0. ) return -1; // bad input 
  Jabs = (J(0) && J(1))? sqrt(J(0)*J(1)) : J(0) + J(1);  // Typical Action
  
  Phi->set_Lz(J(2));
  
  if(J(1) == 0.) SN.Jz0(); // remove n_l != 0 terms

//----------------------------------------------------------------------------
// pre-fit  (no fit yet of the Sn)
//
  if(err>=2) cerr<<" 0. Fit of ToyPar & CanPar\n"; // NOTE 0 below: temporary
  F = SbyLevMar(J,Phi,5,n1,n2,Ni0,BIG,0.,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  // Function (above) which does the Levenberg-Marquad iteration
  // return value is #iterations or negative number for error
  if(F<0) { 
    if(err) cerr<<" SbyLevMar() returned "<<F<<'\n';
    if(F== -3) return -4;
    return -1; 
  }
//----------------------------------------------------------------------------
// first fit
//
//    if estimates for the frequencies given, use them to constrain dH
  if(err>=2) cerr<<" 1. Fit: max. "<<Ni1<<" iterations.\n";
  if(Om(0) && (Om(1) || !(J(1)))) {
    tlH = hypot(Om(0),Om(1)) * Jabs * tl9;
    tlC = (Om(0)*J(0) + Om(1)*J(1)) * tl8;// switch out ,4,s with ,6,s
    F = SbyLevMar(J,Phi,4+sf,n1,n2,Ni1,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  } else  //    otherwise don't constrain dH
    F = SbyLevMar(J,Phi,4+sf,n1,n2,Ni1,BIG,tl1,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  if(F<0) { 
    if(err) {cerr<<" 1. Fit: max. "<<Ni1<<" iterations.\n";
      cerr<<" SbyLevMar() returned "<<F<<'\n';}
    if(F== -3) return -4;
    return -1; 
  }
  
//----------------------------------------------------------------------------  
//    Estimate Omega and hence |dJ|/|J| from Orbit Integration
//  We use a least squares fit for omega R and l; omega phi = mean (Lz/R^2)
  if(type==1) { // if no Angle Fit
    F = Omega(Phi,J,SN,PT,TM,Pi,0.,64.*Pi,Om,dO,dOp);
    if(F && F!=-1 && type==1) F = Omega(Phi,J,SN,PT,TM,Pih,0.,32.*Pi,Om,dO,dOp);
    if(F && F!=-1 && type==1) F = Omega(Phi,J,SN,PT,TM,Pi,Pih,32.*Pi,Om,dO,dOp);
    if(F || Om(0)<Om(2) || (Om(1)<Om(2) && J(1)) ) {
      Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));
      dO = Om(1);  
    } else {
      if(!J(1)) Om[1] = (Phi->KapNuOm(Phi->RfromLc(WDabs(J(2)))))(1);
      rs   = NearResonant(Om);
      if(rs>0.2) dO = fmax(dO,1.e-4/rs*fmin(Om(0),Om(1)));
      else 	 dO = fmax(dO,0.02*fmin(Om(0),Om(1)));
    }
  } else if(!Om(0) || !Om(1) || !Om(2))                 // if no estimates
    Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));         // and doing angle fit
  if(!isFinite(Om(0)+Om(1)+Om(2)))
    cerr << "**BAD** ";
  if(err>=2) cerr << "Omega estimate: "<< Om(0)<<','<<Om(1)<<','<<Om(2) << "\n";
  fac  = 1. / ( hypot(Om(0),Om(1))*Jabs );
  d[0] = dH*fac;

//---------------------------------------------------------------------------- 
//    If fit satisfactory, determine the dS/dJ for the angle map and Omega
//    or if we have no idea what omega is, try another way of finding it
  bool fail = false, tryang=false;
  if(type==0 && d(0) <= tol){// if doing angle fit, and worth try
    tryang = true;
    if(err>=2) cerr<<" Fit of dS/dJi.\n";
    while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
    if(!J(1)) nrs = OD*(SN.NumberofTerms());
      if(useNewAngMap)
          F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);      
      else
          F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);      
    if(F) {
      if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
      fail = true;
    }
    if(d(1)>tla||d(2)>tla||d(3)>tla) { // chi_rms of dSbyInteg fit
      if(err) cerr << "try more terms in AngMap";
      //if(dO!=0.) dO = fmax(dO, 0.5*min(Om(0),Om(1)));
      SN.tailor(0., -1, Max);
      if(J(1) == 0.) SN.Jz0();
      while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
      if(!J(1)) nrs = OD*(SN.NumberofTerms());
        if(useNewAngMap)
            F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
        else
            F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
      if(F) {
	if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	fail=true;
      } else fail=false;
    }
    fac    = 1. / ( hypot(Om(0),Om(1))*Jabs );
    d[0] = dH*fac;
    if(err) cerr<<"Angle fit done; dJ="<<d(0)<<", Omega="<<Om(0)<<','<<Om(1)<<','<<Om(2)<<'\n';
  }

  if(d(0) <= tol && !fail) return 0; // DONE!
//---------------------------------------------------------------------------|
// else we do second fit if |dJ|/|J| above aimed tolerance, tailoring and    |
// cutting the SN. And repeating if needed (indefinitely).                   |
//---------------------------------------------------------------------------/

  if(err) cerr<<" dJ="<<d(0)<<" --> 2. Fit: tailor set of Sn ("<<
      SN.NumberofTerms()<<" terms), max. "
	      <<Ni2<<" iterations.\n";
  double tmp = 1./double(SN.NumberofTerms());
  ta *= tmp;    tb *= tmp;    off *= tmp;
  oSN = SN;
  odJ = d(0);
  vec4 TP = TM.parameters();
  if(!tryang) {   // only do this if no angle fit done
    SN.tailor(ta,tb,Max);
    SN.cut(off);
  } else tryang = false;
  if(J(1) == 0.) SN.Jz0(); 
  n1=fmax(Nth, 6*(SN.NumberofN1()/4+1));
  n2=fmax(Nth, 6*(SN.NumberofN2()/4+1));
  if(l>l0) { l/=256; if(l<l0) l=l0; }
  tlH = dH*tl9/d(0);
  tlC = (Om(0)*J(0) + Om(1)*J(1)) * tl8;
  F=SbyLevMar(J,Phi,4+sf,n1,n2,Ni2,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  if(F<0) { 
    if(err) {cerr<<" dJ="<<d(0)<<" --> 2. Fit: tailor set of Sn, max. "
		 <<Ni2<<" iterations.\n";
      cerr<<" SbyLevMar() returned "<<F<<'\n';}
    if(F== -3) return -4;
    return -1; 
  }
  if((d[0]=dH*fac)>odJ) {
    if(err) cerr << "Tailored set not an improvement. Reverting\n";
    SN   = oSN;
    d[0] = odJ;
    dH   = d(0)/fac;
    TM.set_parameters(TP);
  }
//-------------- LOOP ----------------------------------------------------------
  bool done=false;
  for(int i=0;(i< (Nta-2) && !done);i++) {
    //    if still not converged: enlarge the set of SN and fit again
    if(err) cerr<<" dJ="<<d(0)<<" --> "<< i+1 <<"th tailor set of SN (of"
		<<Nta-2 << ", "<<SN.NumberofTerms()<<" GF terms), max. "
		<<Ni2<<" iterations: ";
    if((d[0]=dH*fac)>=tol) {
      oSN = SN;
      odJ = d[0];
      if(!tryang) {
	if(i%2) 
	  SN.edgetailor(0.25,Max);
	if(!i) SN.tailor(0.,tb*=0.1f,Max);
	else SN.tailor(0.,-1,Max); // always add new terms 
      } else tryang = false;
      if(J(1) == 0.) SN.Jz0();
      if(i==2 || i==6) Ni2 = (Ni2<=4)? Ni2 : (Ni2<8)? 4 : Ni2/2;
      if(i==11) Ni2 = 2;
      if(J(1) == 0.) SN.Jz0();
      n1=fmax(Nth, 6*(SN.NumberofN1()/4+1));
      n2=fmax(Nth, 6*(SN.NumberofN2()/4+1));
      tlH = dH*tl9/d(0);
      F=SbyLevMar(J,Phi,4+sf,n1,n2,Ni2,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
      if(err) {  cerr<<" => M="<<pow_2(TM.parameters()[0])<<", b="<<pow_2(TM.parameters()[1])
          <<", Lz="<<TM.parameters()[2]<<", r0="<<TM.parameters()[3]<<'\n';} 
      if(F<0) { 
	if(err) {cerr<<" dJ="<<d(0)<<" --> again tailor set of SN, max. "
		     <<Ni2<<" iterations.\n";
	  cerr<<" SbyLevMar() returned "<<F<<'\n';}
	if(F== -3) return -4;
	return -1; 
      }
    }
    if((d[0]=dH*fac)<tol || i == (Nta-2)-1) {
      tryang = true;
      if(type==1) { // if no Angle Fit
	F = Omega(Phi,J,SN,PT,TM,Pi,0.,64.*Pi,Om,dO,dOp);
	if(F && F!=-1) F = Omega(Phi,J,SN,PT,TM,Pih,0.,32.*Pi,Om,dO,dOp);
	if(F || Om(0)<Om(2) || (Om(1)<Om(2) && J(1)) ) {                       
	  Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));
	  dO = Om(1);  
	} else {
	  if(!J(1)) Om[1] = (Phi->KapNuOm(Phi->RfromLc(WDabs(J(2)))))(1);
	  rs   = NearResonant(Om);
	  if(rs>0.2) dO = fmax(dO,1.e-4/rs*fmin(Om(0),Om(1)));
	  else 	 dO = fmax(dO,0.02*fmin(Om(0),Om(1)));
	}
	fac  = 1. / ( hypot(Om(0),Om(1))*Jabs );
	d[0] = dH*fac;
	if(d(0) < tol) done=true;
      }

//    Determine the dS/dJ for the angle map
      if(type==0) {
	if(err>=2) cerr<<" Fit of dS/dJi.\n";
	nrs = Nrs;
	while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
	if(!J(1)) nrs = OD*(SN.NumberofTerms());
          if(useNewAngMap)
              F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err>=2);
          else
              F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err>=2);
	if(F) {
	  if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	}
	if(d(1)>tla || d(2)>tla || d(3)>tla || F) {
	  if(err) cerr << "try more terms in AngMap\n";
	  if(dO!=0.) dO = fmax(dO, 0.5*fmin(Om(0),Om(1)));
	  SN.tailor(0., -1, Max);
	  if(J(1) == 0.) SN.Jz0();
	  while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
	  if(!J(1)) nrs = OD*(SN.NumberofTerms());
	  if(useNewAngMap)
	    F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
	  else
	    F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
	  if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	  if(F) {
	    return -4;
	  }
	}
	fac = 1. / ( hypot(Om(0),Om(1))*Jabs );
	d[0]= dH*fac;
	if(d(0) < tol) done=true;
	if(err) cerr<<" dJ = "<<d(0)<<'\n';
      } 
    } else tryang = false;
  }
//---------- END LOOP ----------------------------------------------------------
 // Return
    if(d(0) > 2*tol) return -3;
    if(d(0) > tol)   return -2;
    return 0;
}

// end of Fit.cc ///////////////////////////////////////////////////////////////

int LowJzFit(	            // return:	error flag (see below)
    const Actions &J,       // input:	Actions of Torus to be fit
    Potential     *Phi,	    // input:	pointer to Potential
    const double  tol,	    // input:	goal for |dJ|/|J|
    const int	  Max,	    // input:	max. number of S_n
    const int	  Ni,	    // input:	max. number of iterations
    const int     OD,	    // input:	overdetermination of eqs. for angle fit
    const int     Nrs,      // input:	min. number of cells for fit of dS/dJ
    PoiTra	  &PT,      // in/output:	canonical map with parameters
    ToyMap	  &TM,      // in/output:	toy-pot. map with parameters
    GenPar	  &SN,      // in/output:	parameters of gen. function
    AngPar	  &AP,      // output:	dSn/dJr & dSn/dJl
    Frequencies   &Om,	    // output:	O_r, O_l, O_p
    double	  &Hav,     // output:	mean H
    Errors        &d, 	    // output:	actual |dJ|/|J| & chirms of angle fits
    const int     type,     // input:    Type of Fit (Full (0)/Half(1))
    const int     Nta,      // input:     Number of tailorings
    const int     ipc,	    // input:     max tol steps on average per cell
    double const  eH,       // input:	estimate of expected mean H
    const int     Nth,	    // input:	min. No of theta per dimension
    const int     err,	    // input:	error output?
    const bool useNewAngMap)// input:   whether to use new method for angle mapping
{
  if(J(1) == 0.) {
    if(Ni>800)
      return AllFit(J,Phi,tol,Max,Ni,OD,Nrs,PT,TM,SN,AP,
		    Om,Hav,d,type,false,Ni-3,ipc,eH,Nth,err,useNewAngMap); 
    else 
      return AllFit(J,Phi,tol,Max,800,OD,Nrs,PT,TM,SN,AP,
		    Om,Hav,d,type,false,100,ipc,eH,Nth,err,useNewAngMap); // Let it run on
  }
  //int F;
  double tolJz0 = tol*J(1)/J(0); // Alter tolerance for planar fit
  Actions Jz0=J; Jz0[1] = 0.;
  /*F =*/ AllFit(Jz0,Phi,tolJz0,Max,Ni,OD,Nrs,PT,TM,SN,AP,
  	     Om,Hav,d,type,false,25,ipc,eH,Nth,err,useNewAngMap);// Orbit in plane, precise
  int nadd = 4 + SN.NumberofTerms()/10;
  SN.addn1eq0(nadd);  // add terms with n1=0
  return AllFit(J,Phi,tol,Max,Ni,OD,Nrs,PT,TM,SN,AP,
		Om,Hav,d,type,false,Nta,ipc,eH,Nth,err,useNewAngMap); 

}



int PTFit(	            // return:	error flag (see below)
    const Actions &J,       // input:	Actions of Torus to be fit
    Potential     *Phi,	    // input:	pointer to Potential
    const double  tol,	    // input:	goal for |dJ|/|J|
    const int	  Max,	    // input:	max. number of S_n
    const int	  Ni,	    // input:	max. number of iterations
    const int     OD,	    // input:	overdetermination of eqs. for angle fit
    const int     Nrs,      // input:	min. number of cells for fit of dS/dJ
    PoiTra	  &PT,      // in/output:	canonical map with parameters
    ToyMap	  &TM,      // in/output:	toy-pot. map with parameters
    GenPar	  &SN,      // in/output:	parameters of gen. function
    AngPar	  &AP,      // output:	dSn/dJr & dSn/dJl
    Frequencies   &Om,	    // output:	O_r, O_l, O_p
    double	  &Hav,     // output:	mean H
    Errors        &d, 	    // output:	actual |dJ|/|J| & chirms of angle fits
    const int     type,     // input:   Type of Fit (Full (0)/Half(1))
    const int     Nta,      // input:   Number of tailorings
    const int     ipc,	    // input:   max tol steps on average per cell
    const double  eH,       // input:	estimate of expected mean H
    const int     Nth,	    // input:	min. No of theta per dimension
    const int     err,	    // input:	error output?
    const bool useNewAngMap)// input:   whether to use new method for angle mapping
{

  const    int    Ni0 = 10;            // max. No of iteration in 0. fit
  const    double  //ta0 = 5.e-5,        // threshold for deletion of Sn
    tb0 = 3.e-3;   	               // threshold for creation of Sn
  const    double BIG = 1.e10,         // tolerance on H if otherwise undefined
    //off0 = 0.002,   	  	       // threshold for setting Sn=0
    l0  = 1./128., 	  	       // default start value of lambda
    //tl0 = 3.e-7,   	  	       // tolerance for dchisq of 0. fit
    tl1 = 3.e-7,   	  	       // tolerance for dchisq of 1. & 2. fit
    tl8 = 3.e-5,          	       // tolarance for dchisq/(J*O) of --
    tla = 1.e-2;   	  	       // tolerance for chi_rms of angle fit
  int    Ni1 = Ni/5,          // max. No of iteration in 1. fit
    Ni2 = (Nta)? (Ni-Ni1)/Nta : Ni-Ni1,// max. No of iteration in 2. fit
    n1 = fmax(Nth, 6*(SN.NumberofN1()/4+1)),
    n2 = fmax(Nth, 6*(SN.NumberofN2()/4+1)),
    nrs= Nrs, F;
  double tl9=0.95*tol, fac=1., Jabs, tlH, tlC, rs;
  int    ngA;
  double  tailorparam = (J(0) > J(1) && J(1))? J(0)/J(1) : 
                       (J(1) > J(0) && J(0))? J(1)/J(0) : 1;
  double /*ta = ta0*tailorparam,*/ tb = tb0*tailorparam /*, off = off0*tailorparam*/;

  double l,dH,dO,dOp,odJ;
  
  GenPar oSN;
  // Prepare for fit.
  if(Ni1<1 || J(0)<0. || J(1)<0. ) return -1; // bad input 
  Jabs = (J(0) && J(1))? sqrt(J(0)*J(1)) : J(0) + J(1);  // Typical Action
  
  Phi->set_Lz(J(2));


  SN.JR0();

  // check that the Point transform is set up correctly
  if(!(PT.NumberofParameters())) {
    if(err) cerr << "Point transform not set up before PTFit\n";
    return -1;
  } else {
    double *tmptab = new double[PT.NumberofParameters()];
    PT.parameters(tmptab);
    if(tmptab[0] != J(1) || tmptab[1] != WDabs(J(2))) {
      if(err) cerr << "Point transform set up wrong before Fit\n";
      delete[] tmptab;
      return -1;
    }
    Om = Phi->KapNuOm(Phi->RfromLc(J(2))); // Wrong for phi, it'll have to do
    Om[1] = tmptab[3];
    if(J(0)==0) {
      SN.cut(0.); // point transform enough. Om_phi wrong, but it'll do
      AP = AngPar(SN,SN,SN);
      delete[] tmptab;
      return 0;
    }
    delete[] tmptab;
  }

  
  int sf = 2; // 0 (not safe) or 2 (safe)

//----------------------------------------------------------------------------
// pre-fit  (no fit yet of the Sn)
//
  //if(err) cerr<<" 0. Fit of ToyPar & CanPar\n"; // NOTE 0 below: temporary
  F = SbyLevMar(J,Phi,5+sf,n1,n2,Ni0,BIG,0.,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  // Function (above) which does the Levenberg-Marquad iteration
  // return value is #iterations or negative number for error
  if(F<0) { 
    if(err) cerr<<" SbyLevMar() returned "<<F<<'\n';
    if(F== -3) return -4;
    return -1; 
  }
//----------------------------------------------------------------------------
// first fit
//
//    if estimates for the frequencies given, use them to constrain dH
  if(err) cerr<<" 1. Fit: max. "<<Ni1<<" iterations.\n";
  if(Om(0) && (Om(1) || !(J(1)))) {
    tlH = hypot(Om(0),Om(1)) * Jabs * tl9;
    tlC = (Om(0)*J(0) + Om(1)*J(1)) * tl8;// switch out ,4,s with ,6,s
    if(sf==0)
      F = SbyLevMar(J,Phi,6,n1,n2,Ni1,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
    F = SbyLevMar(J,Phi,5+sf,n1,n2,Ni1,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  } else { //    otherwise don't constrain dH
    if(sf==0)
      F = SbyLevMar(J,Phi,6,n1,n2,Ni1,BIG,tl1,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
    F = SbyLevMar(J,Phi,5+sf,n1,n2,Ni1,BIG,tl1,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  }
  if(F<0) { 
    if(err) {cerr<<" 1. Fit: max. "<<Ni1<<" iterations.\n";
      cerr<<" SbyLevMar() returned "<<F<<'\n';}
    if(F== -3) return -4;
    return -1; 
  }


  
//----------------------------------------------------------------------------  
//    Estimate Omega and hence |dJ|/|J| from Orbit Integration
//  We use a least squares fit for omega R and l; omega phi = mean (Lz/R^2)
  if(type==1) { // if no Angle Fit
    F = Omega(Phi,J,SN,PT,TM,Pi,0.,64.*Pi,Om,dO,dOp);
    if(F && F!=-1 && type==1) F = Omega(Phi,J,SN,PT,TM,Pih,0.,32.*Pi,Om,dO,dOp);
    if(F && F!=-1 && type==1) F = Omega(Phi,J,SN,PT,TM,Pi,Pih,32.*Pi,Om,dO,dOp);
    if(F || Om(0)<Om(2) || (Om(1)<Om(2) && J(1)) ) {
      Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));
      dO = Om(1);  
    } else {
      if(!J(1)) Om[1] = (Phi->KapNuOm(Phi->RfromLc(WDabs(J(2)))))(1);
      rs   = NearResonant(Om);
      if(rs>0.2) dO = fmax(dO,1.e-4/rs*fmin(Om(0),Om(1)));
      else 	 dO = fmax(dO,0.02*fmin(Om(0),Om(1)));
    }
  } else if(!Om(0) || !Om(1) || !Om(2))                 // if no estimates
    Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));         // and doing angle fit
  if(err) cerr << "Omega estimate: "<< Om << "\n";
  fac  = 1. / ( hypot(Om(0),Om(1))*Jabs );
  d[0] = dH*fac;

//---------------------------------------------------------------------------- 
//    If fit satisfactory, determine the dS/dJ for the angle map and Omega
//    or if we have no idea what omega is, try another way of finding it
  bool fail = false;
  if(type==0 && d(0) <= tol){// if doing angle fit, and worth try
    if(err) cerr<<" Fit of dS/dJi.\n";
    while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
    if(!J(1)) nrs = OD*(SN.NumberofTerms());
      if(useNewAngMap)
          F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);      
      else
          F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);      
    if(F) {
      if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
      fail = true;
    }
    if(d(1)>tla||d(2)>tla||d(3)>tla) { // chi_rms of dSbyInteg fit
      if(err) cerr << "try more terms in AngMap";
      //if(dO!=0.) dO = fmax(dO, 0.5*fmin(Om(0),Om(1)));
      SN.tailor(0., -1, Max);
      if(J(1) == 0.) SN.Jz0();
      while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
      if(!J(1)) nrs = OD*(SN.NumberofTerms());
        if(useNewAngMap)
            F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
        else
      F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
      if(F) {
	if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	fail=true;
      } else fail=false;
    }
    fac    = 1. / ( hypot(Om(0),Om(1))*Jabs );
    d[0] = dH*fac;
    if(err) cerr<<" dJ = "<<d(0)<<'\n';
  }

  if(d(0) <= tol && !fail) return 0; // DONE!
//---------------------------------------------------------------------------|
// else we do second fit if |dJ|/|J| above aimed tolerance, tailoring and    |
// cutting the SN. And repeating if needed (indefinitely).                   |
//---------------------------------------------------------------------------/

  if(err) cerr<<" dJ="<<d(0)<<" --> 2. Fit: tailor set of Sn, max. "
	      <<Ni2<<" iterations.\n";
  double tmp = 1./double(SN.NumberofTerms());
  /*ta *= tmp;*/    tb *= tmp;    /*off *= tmp;*/
  oSN = SN;
  odJ = d(0);
  vec4 TP = TM.parameters();

  SN.tailor(0.,tb,Max);
  SN.JR0();

  n1=fmax(Nth, 6*(SN.NumberofN1()/4+1));
  n2=fmax(Nth, 6*(SN.NumberofN2()/4+1));
  if(l>l0) { l/=256; if(l<l0) l=l0; }
  tlH = dH*tl9/d(0);
  tlC = (Om(0)*J(0) + Om(1)*J(1)) * tl8;
  F=SbyLevMar(J,Phi,4+sf,n1,n2,Ni2,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
  if(F<0) { 
    if(err) {cerr<<" dJ="<<d(0)<<" --> 2. Fit: tailor set of Sn, max. "
		 <<Ni2<<" iterations.\n";
      cerr<<" SbyLevMar() returned "<<F<<'\n';}
    if(F== -3) return -4;
    return -1; 
  }
  if((d[0]=dH*fac)>odJ) {
    if(err) cerr << "Tailored set not an improvement. Reverting\n";
    SN   = oSN;
    d[0] = odJ;
    dH   = d(0)/fac;
    TM.set_parameters(TP);
  }
//-------------- LOOP ----------------------------------------------------------
  bool done=false;
  for(int i=0;(i< (2*Nta-2) && !done);i++) {
    //    if still not converged: enlarge the set of SN and fit again
    if(err) cerr<<" dJ="<<d(0)<<" --> "<< i+1 <<"th tailor set of SN, max. "
		<<Ni2<<" iterations.\n";
    if((d[0]=dH*fac)>=tol) {
      oSN = SN;
      odJ = d[0];
      if(i%2) 
	SN.edgetailor(0.25,Max);
      if(!i) SN.tailor(0.,tb*=0.1f,Max);
      else SN.tailor(0.,-1,Max); // always add new terms 
      if(i<Nta-3)  SN.JR0();
      else if(i<Nta) {
	if(i==Nta-3) SN.AddTerm(1,0);
	SN.NoMix(); }
      //SN.write_log(cerr);
      if(J(1) == 0.) SN.Jz0();
      if(i==2 || i==6) Ni2 = (Ni2<=4)? Ni2 : (Ni2<8)? 4 : Ni2/2;
      if(i==11) Ni2 = 2;
      n1=fmax(Nth, 6*(SN.NumberofN1()/4+1));
      n2=fmax(Nth, 6*(SN.NumberofN2()/4+1));
      tlH = dH*tl9/d(0);
      F=SbyLevMar(J,Phi,4+sf,n1,n2,Ni2,tlH,tlC,SN,PT,TM,l=l0,Hav,dH,ngA,eH,err>=2);
      if(F<0) { 
	if(err) {cerr<<" dJ="<<d(0)<<" --> again tailor set of SN, max. "
		     <<Ni2<<" iterations.\n";
	  cerr<<" SbyLevMar() returned "<<F<<'\n';}
	if(F== -3) return -4;
	return -1; 
      }
    }
    if((d[0]=dH*fac)<tol || i == (2*Nta-2)-1) {
      if(type==1) { // if no Angle Fit
	F = Omega(Phi,J,SN,PT,TM,Pi,0.,64.*Pi,Om,dO,dOp);
	if(F && F!=-1) F = Omega(Phi,J,SN,PT,TM,Pih,0.,32.*Pi,Om,dO,dOp);
	if(F || Om(0)<Om(2) || (Om(1)<Om(2) && J(1)) ) {                       
	  Om = Phi->KapNuOm(Phi->RfromLc(WDabs(J(2))));
	  dO = Om(1);  
	} else {
	  if(!J(1)) Om[1] = (Phi->KapNuOm(Phi->RfromLc(WDabs(J(2)))))(1);
	  rs   = NearResonant(Om);
	  if(rs>0.2) dO = fmax(dO,1.e-4/rs*fmin(Om(0),Om(1)));
	  else 	 dO = fmax(dO,0.02*fmin(Om(0),Om(1)));
	}
	fac  = 1. / ( hypot(Om(0),Om(1))*Jabs );
	d[0] = dH*fac;
	if(d(0) < tol) done=true;
      }

//    Determine the dS/dJ for the angle map
      if(type==0) {
	if(err) cerr<<" Fit of dS/dJi.\n";
	nrs = Nrs;
	while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
	if(!J(1)) nrs = OD*(SN.NumberofTerms());
          if(useNewAngMap)
              F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
          else
	F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
	if(F) {
	  if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	}
	if(d(1)>tla || d(2)>tla || d(3)>tla || F) {
	  if(err) cerr << "try more terms in AngMap\n";
	  if(dO!=0.) dO = fmax(dO, 0.5*fmin(Om(0),Om(1)));
	  SN.tailor(0., -1, Max);
	  if(J(1) == 0.) SN.Jz0();
	  while( ((2*nrs-1)*nrs) <= OD*(2*nrs+SN.NumberofTerms())) nrs++;
	  if(!J(1)) nrs = OD*(SN.NumberofTerms());
	  if(useNewAngMap)
	    F = dSbySampling(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
	  else
	    F = dSbyInteg(J,Phi,nrs,SN,PT,TM,dO,Om,d,AP,ipc,err);
	  if(F) {
	    if(err) cerr<<" dSbyInteg() returned "<<F<<'\n';
	    return -4;
	  }
	}
	fac = 1. / ( hypot(Om(0),Om(1))*Jabs );
	d[0]= dH*fac;
	if(d(0) < tol) done=true;
	if(err) cerr<<" dJ = "<<d(0)<<'\n';
      } 
    }
  }
//---------- END LOOP ----------------------------------------------------------
 // Return
    if(d(0) > 2*tol) return -3;
    if(d(0) > tol)   return -2;
    return 0;
}

} // namespace
