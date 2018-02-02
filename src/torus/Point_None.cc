/*******************************************************************************
*                                                                              *
* Point_None.cc                                                                *
*                                                                              *
* C++ code written by Walter Dehnen, 1994-97,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/

#include "Point_None.h"

namespace torus {
////////////////////////////////////////////////////////////////////////////////
// class PoiNone


////////////////////////////////////////////////////////////////////////////////
PoiNone::PoiNone()
{ 
  derivs_ok=false; 
}
////////////////////////////////////////////////////////////////////////////////
PoiNone::PoiNone(const PoiNone& /*P*/)
{ 
 derivs_ok=false; 
}

////////////////////////////////////////////////////////////////////////////////
PSPD PoiNone::Forward(const PSPD& w) const
{
    derivs_ok=true; 
// extract w=(q,p)
    r  = w(0);
    th = w(1);
    pr = w(2);
    pt = w(3);
    ir = 1./r;
    ct = cos(th);
    st = sin(th);

    R = r*ct;
    z = r*st;
    pR = ct*pr - st*ir*pt;
    pz = st*pr + ct*ir*pt;

    return PSPD(R,z,pR,pz);		
}
////////////////////////////////////////////////////////////////////////////////
PSPT PoiNone::Forward3D(const PSPT& w3) const
{
  PSPT W3 = w3;
  PSPD w2 = w3.Give_PSPD();
  W3.Take_PSPD(Forward(w2));
  W3[5] /= W3(0);             // p_phi -> v_phi 
  return W3;
}
////////////////////////////////////////////////////////////////////////////////
PSPD PoiNone::Backward(const PSPD& W) const
{
    derivs_ok=true; 

    R  = W(0);
    z  = W(1);
    pR = W(2);
    pz = W(3);

    r=hypot(R,z); 
    th=atan2(z,R); 
    ir = 1./r;
    ct = R*ir;
    st = z*ir;
    pr = pR*ct + pz*st;
    pt = -pR*z + pz*R;

    return PSPD(r,th,pr,pt);
}
////////////////////////////////////////////////////////////////////////////////
PSPT PoiNone::Backward3D(const PSPT& W3) const
{
  PSPT w3 = W3;
  PSPD W2 = W3.Give_PSPD();
  w3.Take_PSPD(Backward(W2));
  w3[5] *= W3(0);               // v_phi -> p_phi 
  return w3;
}
////////////////////////////////////////////////////////////////////////////////
void PoiNone::Derivatives(double dWdw[4][4]) const
{
    if(!derivs_ok)
	throw std::runtime_error("Torus Error -3: PoiNone::Derivatives() called without (For/Back)ward");
 
    dWdw[0][0] = ct;    	       	// dR/dr
    dWdw[0][1] = -z;    		// dR/dth
    dWdw[0][2] = 0.;			// dR/dpr
    dWdw[0][3] = 0.;			// dR/dpth

    dWdw[1][0] = st;    		// dz/dr
    dWdw[1][1] = R;     		// dz/dth
    dWdw[1][2] = 0.;			// dz/dpr
    dWdw[1][3] = 0.;			// dz/dpth

    dWdw[2][0] = pt*st*ir*ir;          // dpR/dr
    dWdw[2][1] = -pr*st-pt*ct*ir;      // dpR/dth
    dWdw[2][2] = ct;	                // dpR/dpr
    dWdw[2][3] = -st*ir;	        // dpR/dpth

    dWdw[3][0] = -ct*ir*ir*pt;		// dpz/dr
    dWdw[3][1] = ct*pr-st*ir*pt;	// dpz/dth
    dWdw[3][2] = st;			// dpz/dpr
    dWdw[3][3] = ct*ir;			// dpz/dpth
}

} // namespace
///end of Point_None.cc //////////////////////////////////////////////////////
