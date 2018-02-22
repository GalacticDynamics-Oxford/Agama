/***************************************************************************//**
\file Maps.h 
\brief Contains classes PhaseSpaceMap PhaseSpaceMapWithBackward ToyMap PoiTra.
Base classes for all the transformations needed to describe a Torus

*                                                                              *
* C++ code written by Walter Dehnen, 1995-96,                                  *
*                     Paul McMillan, 2007-                                     *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
********************************************************************************
*                                                                              *
* abstract base classes:                                                       *
* PhaseSpaceMap                = phase-space maps                              *
* PhaseSpaceMapWithBackward    = phase-space maps with backward mapping        *
* ToyMap                       = toy-potential maps (e.g. isochrone)           *
* PoiTra                       = canonical maps (e.g. point transformation)    *
*                                                                              *
* non-base classes:                                                            *
* GenPar                       = parameters of generating functions            *
* AngPar                       = parameters of angle map                       *
* GenFnc                       = generating function                           *
* GenFncFit                    = generating function when heavily used         *
* AngMap                       = angle maps                                    *
*                                                                              *
********************************************************************************
*                                                                              *
* Inheritance tree  (including classes defined elsewhere)                      *
*                                                                              *
*                             PhaseSpaceMap                                    *
*                             /      |     \                                   *
*                          Torus   GenFnc  PhaseSpaceMapWithBackward           *
*                                     |         |       |       |              *
*                                GenFncFit   AngMap  ToyMap  PoiTra            *
*                                                       |                      *
*                                                    ToyIsochrone              *
*                                                                              *
********************************************************************************
*                                                                              *
* The term `forward' for a `PhaseSpaceMap' always means the mapping needed to  *
* go from action-angle variables to ordinary phase-space coordinates, whereas  *
* `backward' denotes the opposite directions.                                  *
*                                                                              *
*******************************************************************************/

#ifndef _Maps_
#define _Maps_ 1

#include <iostream>
#include <cmath>
#include "Pi.h"
#include <stdexcept>
#include "Types.h"
#include "math_core.h"
#include <climits>

namespace torus{
////////////////////////////////////////////////////////////////////////////////
/**
\brief Base class for any transformation on the route from
action-angle coordinates to ordinary phase-space coordinates.
 */
class PhaseSpaceMap {
public:
            PhaseSpaceMap          () {}
    virtual~PhaseSpaceMap          () {}
    virtual int  NumberofParameters()            const=0;
    virtual PSPD Forward           (const PSPD&) const=0;
    virtual PSPT Forward3D         (const PSPT&) const=0;
};

inline PSPD operator>> (const PSPD& X, const PhaseSpaceMap& P)
		{ return P.Forward(X); }
inline PSPT operator>> (const PSPT& X, const PhaseSpaceMap& P)
		{ return P.Forward3D(X); }

////////////////////////////////////////////////////////////////////////////////
/**
\brief Base class for any transformation on the route from
action-angle coordinates to ordinary phase-space coordinates that can
work in either direction.
 */
class PhaseSpaceMapWithBackward : public PhaseSpaceMap {
public:
            PhaseSpaceMapWithBackward   () {}
    virtual~PhaseSpaceMapWithBackward   () {}
    virtual PSPD Backward               (const PSPD&) const=0;
    virtual PSPT Backward3D             (const PSPT&) const=0;
};

inline PSPD operator<< (const PSPD& X, const PhaseSpaceMapWithBackward& P)
		{ return P.Backward(X); }
inline PSPT operator<< (const PSPT& X, const PhaseSpaceMapWithBackward& P)
		{ return P.Backward3D(X); }

////////////////////////////////////////////////////////////////////////////////
/**
\brief Base class for any conversion between toy action-angle
coordinates and position-momentum coordinates.
 */

class ToyMap : public PhaseSpaceMapWithBackward {
public:
    virtual vec4    parameters       ()                  	 const=0;
    virtual void    parameters       (vec4 *)                  const=0;
    virtual void    set_parameters   (const vec4&)                  =0;
    virtual vec4    lower_bounds     (const double,const double) const=0;
    virtual vec4    upper_bounds     (const double,const double) const=0;
    virtual void    Derivatives      (double[4][2])              const=0;
    virtual void    Derivatives      (double[4][2], Pdble[4])    const=0;
    virtual PSPD    ForwardWithDerivs(const PSPD&, double[2][2]) const=0;
    virtual PSPD    ForwardWithDerivs(const PSPD&, double[2][2], 
				      double[2][2])              const=0;
};

////////////////////////////////////////////////////////////////////////////////
/**
\brief Base class for any point transform.
 */
class PoiTra : public PhaseSpaceMapWithBackward {
public:
  virtual void    parameters       (double *)                  const=0;
  virtual void    Derivatives      (double[4][4])	       const=0; 
};

inline void AlignAngles(PSPD& JT)
{
  if(std::isnan(JT(2)) || std::isinf(JT(2)) || fabs(JT(2))>INT_MAX)
    JT[2] = 0.;        // in case of major failure
  if(std::isnan(JT(3)) || std::isinf(JT(3)) || fabs(JT(3))>INT_MAX)
    JT[3] = 0.;        // in case of major failure
  JT[2] = math::wrapAngle(JT(2));
  JT[3] = math::wrapAngle(JT(3));
}
inline void AlignAngles3D(PSPT& JT)
{
  if(std::isnan(JT(3)) || std::isinf(JT(3)) || fabs(JT(3))>INT_MAX)
    JT[3] = 0.;        // in case of major failure
  if(std::isnan(JT(4)) || std::isinf(JT(4)) || fabs(JT(4))>INT_MAX)
    JT[4] = 0.;        // in case of major failure
  if(std::isnan(JT(5)) || std::isinf(JT(5)) || fabs(JT(5))>INT_MAX)
    JT[5] = 0.;        // in case of major failure
  JT[3] = math::wrapAngle(JT(3));
  JT[4] = math::wrapAngle(JT(4));
  JT[5] = math::wrapAngle(JT(5));
}

} // namespace
#endif
