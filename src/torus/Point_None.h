/***************************************************************************//**
\file Point_None.h 
\brief Contains class PoiNone. Code for the default point tranform where one isn't needed (i.e. most cases).

*                                                                              *
* C++ code written by Walter Dehnen, 1994-97,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/
//
// Header file for a no point transformation. 
// Input (r,TH,{phi},pr,pTH,{pphi}), Output (R,z,{phi},vR,vz,{v_phi})
//
// I.e. just converts from spherical polar coords to cylindrical.
//
//

#ifndef _NoTra_
#define _NoTra_ 1

#include "Maps.h"
namespace torus{
////////////////////////////////////////////////////////////////////////////////

/** \brief Null point transform.

    Input (r,TH,{phi},pr,pTH,{pphi}), Output (R,z,{phi},vR,vz,{v_phi})
 */
class PoiNone : public PoiTra {

public:
    PoiNone();
    PoiNone(const PoiNone& );
    ~PoiNone() {}
    mutable bool derivs_ok;
    mutable double r,th,pr,pt,ir, ct,st,R,pR,z,pz;
    void    parameters	      (double *)		   const;
    int     NumberofParameters()                           const { return 0; }
    PSPD    Forward           (const PSPD&)                const;
    PSPD    Backward          (const PSPD&)                const;
    PSPT    Forward3D         (const PSPT&)                const;
    PSPT    Backward3D        (const PSPT&)                const;
    void    Derivatives       (double[4][4])               const;
};

inline void PoiNone::parameters (double *) const
{
  // Not applicable for this map. It has no parameters.
}

} // namespace
#endif
