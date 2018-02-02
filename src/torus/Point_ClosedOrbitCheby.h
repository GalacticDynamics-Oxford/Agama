/***************************************************************************//**
*                                                                              *
\file Point_ClosedOrbitCheby.h 
\brief Contains class PoiClosedOrbit.
Code for the point transform used when Tori have very low J_R -- i.e. are 
nearly closed orbits in R-z plane.

*                                                                              *
* C++ code written by Paul McMillan, 2008                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/
//
// Header file for a point transformation. 
// Input (r,TH,{phi},pr,pTH,{pphi}), Output (R,z,{phi},vR,vz,{v_phi})
//


#ifndef _PoiCh_tr_
#define _PoiCh_tr_ 1

#include "Potential.h"
#include "Maps.h"
#include "Types.h"
#include "CHB.h"

namespace torus{
////////////////////////////////////////////////////////////////////////////////

/**

\brief Point transform used when fitting orbits with J_r << J_l, which
cannot be fit otherwise. The transform is defined by:

 r^T  = x(th)*r
 th^T = y(r)*z(th)
 
 This is done so that r^T is constant for a J_R = 0 orbit (as required), and 
 then fixes th^T so that pth^T has the correct dependence on th^T

------------------------------------------------------------------------------

 The class PoiClosedOrbit contains routines which implement this point transform
 and ones which find it for a given J_l, J_phi and gravitational potential.

 1. Finding the transform:
    Done by the function set_parameters(Potential*, Actions). This finds the
 correct closed (J_R=0) orbit in the target potential, and determines the
 functions x, y, and z (above), storing them as Chebyshev polynomials.

 2. Performing the transform:
    Done in the usual way, with Forward and Backward. x,y and z (above) found
 from the Chebyshev polynomials (for th<thmax) or a quadratic for x and z if
 th>thmax - this ensures that x and z are accurately fit, and don't trend
 off to extreme values for th>thmax.  Note that we actually store z' = z/th


*///////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
class PoiClosedOrbit : public PoiTra {
 private:
  // properties of the point transform
  double Jl,Lz,alt2,thmax,omz, 
    ax,bx,cx, az,bz,cz;   // coefficients of quadratic for x and z for th>thmax
  Cheby xa,ya,za;         // Chebyshev polynomials which define transform
                          // N.B. za stores z' = z/th because that's ~constant
  
  double thmaxforactint;  // used to find the action
  Cheby vr2, drdth2, pth2;//         ''
  double actint(double) const;
// ROUTINES USED TO FIND THE TRANSFORM -----------------------------------------
  // Integrate orbit over Pi in th_z & store output
  void do_orbit       (PSPD, double, Potential*, double*, double*, double*, 
		       double*, double*, double*, double*, int&, int); 
  // Shift starting radius towards that of closed orbit (or note we're there)
  void set_Rstart     (double&,double,double&,double&,bool&,bool&,bool&);
  // Shift orbit energy so that closed orbit of that energy has J_l = target
  void set_E          (const double,double&,double&,double&,bool&,bool&,bool&);
  // Rewrite tables so I can fit to chebyshev (also find thmax)
  void RewriteTables  (const int, double *, double *, double *, double *, 
		       double *, double *, double *, double *, int &);
  // Routines needed to find y and z
  vec2 chebderivs     (const double, const vec2);
  double stepper      (vec2&, vec2&, const double, const double );
  void yzrkstep       (vec2&, vec2&, const double, double&, const double, 
		       double&, const double, const double) ;
  double yfn          (double, vec2 *, const int, double * , int); 
//------------------------------------------------------------------------------
 public:
  // various constructors
  PoiClosedOrbit();                                          
  PoiClosedOrbit(Actions, Cheby, Cheby, Cheby, double,double);
  PoiClosedOrbit(const double*);                              
  PoiClosedOrbit(Potential *, const Actions);                 
  ~PoiClosedOrbit() {}
  void    set_parameters    (Actions, Cheby, Cheby, Cheby, double,double);
  void    set_parameters    (Potential *, const Actions);
  void    set_parameters    (const double*);
  void    parameters	    (double *)	                 const;
  int     NumberofParameters()                           const;
  PSPD    Forward           (const PSPD&)                const;
  PSPD    Backward          (const PSPD&)                const;
  PSPT    Forward3D         (const PSPT&)                const;
  PSPT    Backward3D        (const PSPT&)                const;
  void    Derivatives       (double[4][4])               const;
};


inline void PoiClosedOrbit::set_parameters(Actions J, Cheby Ch1, Cheby Ch2, 
					   Cheby Ch3, double thmx, double om)
{
  Jl = J(1);
  Lz = fabs(J(2));
  alt2=(fabs(Lz)+Jl)*(fabs(Lz)+Jl);
  omz = om;

  xa = Ch1;
  ya = Ch2;
  za = Ch3;
  thmax = thmx;

  // define coefficients such that quadratic goes through final point and
  // has correct gradient at that point. Then take same values of xa and za 
  // at th = pi/2 
  double x1 = thmax*thmax, x2=Pih*Pih, y1x, dy1x, y1z, dy1z, delx = x2-x1;

  xa.unfitderiv(x1,y1x,dy1x);
  za.unfitderiv(x1,y1z,dy1z);

  ax = -dy1x/delx;
  bx = dy1x-2*ax*x1;
  cx = y1x - x1*(ax*x1 + bx);

  az = -dy1z/delx;
  bz = dy1z-2*az*x1;
  cz = y1z - x1*(az*x1 + bz);
}

inline int PoiClosedOrbit::NumberofParameters() const {
  return 7 + xa.NChb + ya.NChb + za.NChb; // 2 actions, thmax, 3*NCheb +.
}

inline void PoiClosedOrbit::parameters(double *tmp) const {
  int ncx = xa.NChb, ncy = ya.NChb, ncz = za.NChb;
  tmp[0] = Jl; tmp[1] = Lz; tmp[2] = thmax; tmp[3] = omz;
  tmp[4] = ncx;
  for(int i=0;i!=ncx;i++) tmp[5+i] = xa.e1[i];
  tmp[5+ncx] = ncy;
  for(int i=0;i!=ncy;i++) tmp[6+ncx+i] = ya.e1[i];
  tmp[6+ncx+ncy] = ncz;
  for(int i=0;i!=ncz;i++) tmp[7+ncx+ncy+i] = za.e1[i];
  
}

} // namespace
#endif
