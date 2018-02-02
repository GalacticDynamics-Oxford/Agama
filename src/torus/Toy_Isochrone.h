/***************************************************************************//**
\file Toy_Isochrone.h
\brief Contains class ToyIsochrone & IsoPar. 
Code for the mapping from angles and actions to position and momentum in the 
isochrone potential, used as the toy mapping.

*                                                                              *
* C++ code written by Walter Dehnen, 1994-96,                                  *
*                     Paul McMillan, 2007-                                     *
* e-mail:  paul@astro.lu.se                                                    *
* github:  https://github.com/PaulMcMillan-Astro/Torus                         *
*                                                                              *
********************************************************************************
*                                                                              *
* class ToyIsochrone    class for the mapping between action-angle variables   *
*                  and ordinary phase space coordinates for orbits in the      *
*                  effective Isochrone potential                               *
*                                                                              *
*                                    -   GM                        Lz^2        *
*                  Phi(r,th) = -------------------------- + ------------------ *
*                              b + Sqrt[ b^2 + (r-r0)^2 ]   2((r-r0)sin(th))^2 *
*                                                                              *
*                  The paramters are GM, b, Lz and r0. However, GM and b are   *
*                  not allowed to become negative, which may cause problems    *
*                  for fitting algorithms. Therefore, class ToyIsochrone uses  *
*                  the parameters gamma^2 = GM, and beta^2 = b, such that beta *
*                  and gamma can be `any' real number, except for zero.        *
*                                                                              *
* class IsoPar     is a VecPar with 4 elements for holding the parameters      *
*                  of a ToyIsochrone.                                          *
*                                                                              *
*******************************************************************************/

#ifndef _ToyIsochrone_def_
#define _ToyIsochrone_def_ 1

#include "Maps.h"

namespace torus{
typedef Vector<double,4> IsoPar; // Vector.h
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/** \brief Mapping between toy action-angle coordinates and
    position-momentum coordinates for the generalised effective
    isochrone potential.

                                                                              
    class for the mapping between action-angle variables and ordinary
    phase space coordinates for orbits in the effective Isochrone
    potential
                                                                              
                       -   GM                        Lz^2        
     Phi(r,th) = -------------------------- + ------------------ 
                 b + Sqrt[ b^2 + (r-r0)^2 ]   2((r-r0)sin(th))^2 
                                                                              
    The parameters are GM, b, Lz and r0. However, GM and b are not
    allowed to become negative, which may cause problems for fitting
    algorithms. Therefore, class ToyIsochrone uses the parameters
    gamma^2 = GM, and beta^2 = b, such that beta and gamma can be
    `any' real number, except for zero. 

 */
class ToyIsochrone : public ToyMap {
private:
    double gamma,beta,Lz,r0;	// parameters
    double M,b;			// mass & scale length
    double sMb,sMob,jp;		// scaling and J_phi
    mutable double a,e,ae,eps,u,H,chi,psi,cpsi,spsi,sGam, jr,jt,tr,tt, 
      r,th,pr,pt,wr,wt,wt0r,at,wh,sq, HH,H2;
    mutable bool derivs_ok;
    void   Set();
    void   psifunc(const double, double&, double&) const;
    void   psisolve() const;
    double catan(const double&) const;
    double wfun(const double&) const;
    double wfundw(const double&, double&) const;
    double Fint(double,double,double,double) const;
public:
    void dump() const;
    ToyIsochrone();
    ToyIsochrone(const IsoPar& );
    ToyIsochrone(const ToyIsochrone& );
   ~ToyIsochrone() {}
    void    set_parameters    (const IsoPar&);
    vec4  parameters        ()                           const;
    vec4  lower_bounds      (const double, const double) const;
    vec4  upper_bounds      (const double, const double) const;
    void    parameters        (vec4*)                    const;
    int     NumberofParameters()                           const { return 4; }
    void    Derivatives       (double[4][2])               const;
    void    Derivatives       (double[4][2], double*[4])   const;
    PSPD    ForwardWithDerivs (const PSPD&, double[2][2])  const;
    PSPD    ForwardWithDerivs (const PSPD&, double[2][2],
			       double[2][2])               const;
    PSPD    Forward           (const PSPD&)                const;
    PSPD    Backward          (const PSPD&)                const;
    PSPT    Forward3D         (const PSPT&)                const;
    PSPT    Backward3D        (const PSPT&)                const;
};

inline void ToyIsochrone::set_parameters(const IsoPar& p)
{ 
    gamma = p(0);
    beta  = p(1);
    Lz    = p(2);
    r0    = p(3);
    Set();
}

inline vec4 ToyIsochrone::parameters() const
{
    double a[4] = {gamma,beta,Lz,r0};
    return vec4((const double*)a); 
}

inline void ToyIsochrone::parameters(vec4* p) const
{
	(*p)[0] = gamma;
	(*p)[1] = beta;
	(*p)[2] = Lz;
	(*p)[3] = r0;
}

} // namespace
#endif
