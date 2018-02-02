#include "PJMCoords.h"
#include <cmath>
#include "Constants.h"
#include "Pi.h"
namespace torus{
////////////////////////////////////////////////////////////////////////////////
// GCA <--> GCY

void OmniCoords::GCYfromGCA()
{
    double s, c;
    rv[5][0] = hypot(rv[4](0), rv[4](1));
    rv[5][1] = rv[4](2);
    if(rv[5](0)) {
        s = rv[4](1)/rv[5](0);
        c = rv[4](0)/rv[5](0);
        rv[5][2] = (s>0.)? acos(c) : TPi-acos(c);
    } else {
        s = 0.;
        c = 1.;
        rv[5][2] = 0.;
    }
    rv[5][3] = c*rv[4](3)+s*rv[4](4);
    rv[5][5] =-s*rv[4](3)+c*rv[4](4);
    rv[5][4] = rv[4](5);
    know[5] = true;
}

void OmniCoords::GCAfromGCY()
{
    double s=sin(rv[5](2)), c=cos(rv[5](2));
    rv[4][0] = c*rv[5](0);
    rv[4][1] = s*rv[5](0);
    rv[4][2] = rv[5](1);
    rv[4][3] = c*rv[5](3)-s*rv[5](5);
    rv[4][4] = s*rv[5](3)+c*rv[5](5);
    rv[4][5] = rv[5](4);
    know[4] = true;
}

////////////////////////////////////////////////////////////////////////////////
// GCA <--> LSR

void OmniCoords::LSRfromGCA()
{
    static double z=0.,s=0.,c=1.;
    rv[3][0] = Rsun-rv[4](0);   // note switch in sign
    rv[3][1] =-rv[4](1);        //       ditto
    rv[3][2] = rv[4](2)-zsun;
    rv[3][3] =-rv[4](3);
    rv[3][4] = vcsun-rv[4](4);
    rv[3][5] = rv[4](5);
    if(zsun) { // need to rotate a bit for GC to have rv[3][2]=0
	double t;
	if(z!=zsun) {
	    z = zsun;
	    t = hypot(zsun,Rsun);
	    s = zsun/t;
	    c = Rsun/t;
	}
	t      = rv[3](0);
	rv[3][0] = c*t - s*rv[3](2);
	rv[3][2] = s*t + c*rv[3](2);
	t      = rv[3](3);
	rv[3][3] = c*t - s*rv[3](5);
	rv[3][5] = s*t + c*rv[3](5);
    }
    know[3] = true;
}

void OmniCoords::GCAfromLSR()
{
    static double z=0.,s=0.,c=1.;
    if(zsun) { // need to rotate a bit for GC to have rv[3][2]=0
        vec6 in=rv[3];
	double t;
	if(z!=zsun) {
	    z = zsun;
	    t = hypot(zsun,Rsun);
	    s = zsun/t;
	    c = Rsun/t;
	}
 	t     = in(0);
	in[0] = c*t + s*in(2);
	in[2] =-s*t + c*in(2);
 	t     = in(3);
	in[3] = c*t + s*in(5);
	in[5] =-s*t + c*in(5);
	rv[4][0] = Rsun-in(0);
	rv[4][1] =-in(1);
	rv[4][2] = in(2)+zsun;
	rv[4][3] =-in(3);
	rv[4][4] = vcsun-in(4);
	rv[4][5] = in(5);
    } else {
	rv[4][0] = Rsun-rv[3](0);
	rv[4][1] =-rv[3](1);
	rv[4][2] = rv[3](2);
	rv[4][3] =-rv[3](3);
	rv[4][4] = vcsun-rv[3](4);
	rv[4][5] = rv[3](5);
    }
    know[4] = true;
}


////////////////////////////////////////////////////////////////////////////////
// LSR  <-->  HCA
void OmniCoords::LSRfromHCA() 
{
    rv[3][0] = rv[2](0);
    rv[3][1] = rv[2](1);
    rv[3][2] = rv[2](2);
    rv[3][3] = rv[2](3) + Usun;
    rv[3][4] = rv[2](4) + Vsun;
    rv[3][5] = rv[2](5) + Wsun;
    know[3] = true;
}
void OmniCoords::HCAfromLSR()
{
    rv[2][0] = rv[3](0);
    rv[2][1] = rv[3](1);
    rv[2][2] = rv[3](2);
    rv[2][3] = rv[3](3) - Usun;
    rv[2][4] = rv[3](4) - Vsun;
    rv[2][5] = rv[3](5) - Wsun;
    know[2] = true;
}

////////////////////////////////////////////////////////////////////////////////
// HCA <--> HGP

void OmniCoords::HGPfromHCA()
{
    double R=hypot(rv[2](0),rv[2](1));
    rv[1][0] = hypot(R,rv[2](2));
    double 
    cl	   = (R==0.)?       1. : rv[2](0)/R,
    sl	   = (R==0.)?       0. : rv[2](1)/R,
    cb	   = (rv[1](0)==0.)?  1. : R/rv[1](0),
    sb	   = (rv[1](0)==0.)?  0. : rv[2](2)/rv[1](0),
    temp   = cl*rv[2](3) + sl*rv[2](4);
    rv[1][1] = (sl<0.)? TPi-acos(cl) : acos(cl);
    rv[1][2] = asin(sb);
    rv[1][3] = cb*temp + sb*rv[2](5);
    rv[1][4] = (rv[1](0)==0.)? 0. : (cl*rv[2](4) - sl*rv[2](3)) / rv[1](0);
    rv[1][5] = (rv[1](0)==0.)? 0. : (cb*rv[2](5) - sb*temp  ) / rv[1](0);
    know[1] = true;
}

void OmniCoords::HCAfromHGP()
{
    double 
    cl     = cos(rv[1](1)),
    sl     = sin(rv[1](1)),
    cb     = cos(rv[1](2)),
    sb     = sin(rv[1](2)),
    R      = cb*rv[1](0),
    vl     = rv[1](0)*rv[1](4),
    vb     = rv[1](0)*rv[1](5),
    temp   = cb*rv[1](3) - sb*vb;
    rv[2][0] = cl*R;
    rv[2][1] = sl*R;
    rv[2][2] = sb*rv[1](0);
    rv[2][3] = cl*temp-sl*vl;
    rv[2][4] = sl*temp+cl*vl;
    rv[2][5] = sb*rv[1](3)+cb*vb;
    know[2] = true;
}

////////////////////////////////////////////////////////////////////////////////
// HCA <--> HEQ at julian epoch

void OmniCoords::SetTrans()
{
// see `explanatory supplement to the astronomical almanac' p.104,173-4
    const double A[6]={0.01118086087,  6.770713945e-6,-6.738910167e-10,
		       1.463555541e-6,-1.667759063e-9, 8.720828496e-8},
		 B[6]={0.01118086087,  6.770713945e-6,-6.738910167e-10,
		       5.307158404e-6, 3.199770295e-10,8.825063437e-8},	
		 C[6]={0.009717173455,-4.136915141e-6,-1.052045688e-9,
		       -2.06845757e-6,-1.052045688e-9,-2.028121072e-7};
    if(epoch == 1991.25) {
        int i,j;
	for(i=0; i<3; i++) for(j=0; j<3; j++) {
	    EtoP[i][j] = GalactoConstants::EtoPJ1991[i][j];
	    if(EtoP[i][j]==0.) {
	      cerr << "Something has gone wrong in OmniCoords.\n"
		   << "I've seen this before when OmniCoords declared as "
		   << "global variable\n";
	      exit(1); 
	    }
	  }
        return;
    }
    if(epoch == 2000.) {
        int i,j;
	for(i=0; i<3; i++) for(j=0; j<3; j++) {
	    EtoP[i][j] = GalactoConstants::EtoPJ2000[i][j];
	    if(EtoP[i][j]==0.) {
	      cerr << "Something has gone badly wrong in OmniCoords.\n"
		   << "I've seen this before when OmniCoords declared as "
		   << "global variable\n";
	      exit(1); 
	    }
	  }
        return;
    }
    int    i,j,k;
    double
    T  = 0.01*(epoch-2000.),
    t  =-T,
    zt = ( ( A[5]*t + (A[4]*T+A[3]) )*t + ((A[2]*T+A[1])*T+A[0]) )*t, // z_a
    th = ( ( B[5]*t + (B[4]*T+B[3]) )*t + ((B[2]*T+B[1])*T+B[0]) )*t, // theta_a
    Ze = ( ( C[5]*t + (C[4]*T+C[3]) )*t + ((C[2]*T+C[1])*T+C[0]) )*t, // zeta_a
    czt= cos(zt),
    szt= sin(zt),
    cth= cos(th),
    sth= sin(th),
    cZe= cos(Ze),
    sZe= sin(Ze);
    double P[3][3] = {{ czt*cth*cZe-szt*sZe, -czt*cth*sZe-szt*cZe, -czt*sth},
		      { szt*cth*cZe+czt*sZe, -szt*cth*sZe+czt*cZe, -szt*sth},
		      { szt*cZe            , -sth*sZe            ,  cth    }};
    for(i=0; i<3; i++) for(j=0; j<3; j++) {
	EtoP[i][j] = 0.;
	for(k=0; k<3; k++)
	    EtoP[i][j] += GalactoConstants::EtoPJ2000[i][k] * P[k][j];
    }
}

void OmniCoords::HEQfromHCA()
{
    int i,j;
    vec6  h=0.;
    for(i=0; i<3; i++) for(j=0; j<3; j++) {
	h[i]   += rv[2](j)   * EtoP(i,j);
	h[i+3] += rv[2](j+3) * EtoP(i,j);
    }
    double R=hypot(h(0),h(1));
    rv[0][0] = hypot(R,h(2));
    double
    ca	   = (R==0.)?       1. : h(0)/R,
    sa	   = (R==0.)?       0. : h(1)/R,
    cd	   = (rv[0](0)==0.)?  1. : R/rv[0](0),
    sd	   = (rv[0](0)==0.)?  0. : h(2)/rv[0](0),
    temp   = ca*h(3) + sa*h(4);
    rv[0][1] = (sa<0.)? TPi-acos(ca) : acos(ca);
    rv[0][2] = asin(sd);
    rv[0][3] = cd*temp + sd*h(5);
    rv[0][4] = (rv[0](0)==0.)?  0. : (ca*h(4)-sa*h(3)) / rv[0](0);
    rv[0][5] = (rv[0](0)==0.)?  0. : (cd*h(5)-sd*temp) / rv[0](0);
    know[0] = true;
}

void OmniCoords::HCAfromHEQ()
{
           int i,j;
    vec6     h;
    double
    ca   = cos(rv[0](1)),
    sa   = sin(rv[0](1)),
    cd   = cos(rv[0](2)),
    sd   = sin(rv[0](2)),
    R    = cd*rv[0](0),
    va   = rv[0](0)*rv[0](4),
    vd   = rv[0](0)*rv[0](5),
    temp = cd*rv[0](3)-sd*vd;
    rv[2]=0.;
    h[0] = ca*R;
    h[1] = sa*R;
    h[2] = sd*rv[0](0);
    h[3] = ca*temp  -sa*va;
    h[4] = sa*temp  +ca*va;
    h[5] = sd*rv[0](3)+cd*vd;
    for(i=0; i<3; i++) for(j=0; j<3; j++) {
	rv[2][i]   += h(j)   * EtoP(j,i);
	rv[2][i+3] += h(j+3) * EtoP(j,i);
    }
    know[2] = true;
}

} // namespace