/***************************************************************************//**
\file  Orb.h 
\brief Contains classes Record Record3D, function orbit.
Classes and functions used for integrating orbits in gravitational potentials

*                                                                              *
* C++ code written by Walter Dehnen, 1994-96,                                  *
*                     Paul McMillan, 2007                                      *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
*******************************************************************************/

#ifndef _TorusOrb_
#define _TorusOrb_ 1

#include "Types.h"
#include "Potential.h"

namespace torus{
/** \brief Integrates orbits in effective potentials. */

class Record {
private:
    Potential*  Pot;                       // pointer to the potential
    double      R,z, pR,pz, aR,az;         // coordinates and deriv. of Phi
    double      E, tol, dtm;               // energy, tolerance, max time step
    void        LeapFrog(const double);    // 2nd order symplectic integrator
    void        FourSymp(const double);    // 4th order symplectic integrator
    void        RungeKutta(const double);  // 4th order Runge-Kutta integrator
public:
    Record                  (const PSPD&, Potential*, const double=0.);
    Record                  (const Record&);
   ~Record                  () {}
    Record&    operator=    (const Record&);
    void       set_tolerance(const double dE=0.) {tol= (dE)? fabs(dE) : 1.e-9;}
    void       set_maxstep  (const double dt=0.) {dtm= (dt)? fabs(dt) : 1.;}
    void       step2_by     (double&, const double=2.);
    void       step4_by     (double&, const double=1.4);
    void       stepRK_by    (double&, const double=1.4);
    PSPD       QP           () const { return PSPD(R,z,pR,pz); }
    double     QP	    (const int) const;
    double     energy       () const { return E; }
    Potential* potential    () const { return Pot; }
};

inline double Record::QP(const int i) const
{
    switch (i) {
    case 0: return R;
    case 1: return z;
    case 2: return pR;
    case 3: return pz;
    default: return 0.;
    }
}

inline void Record::FourSymp(const double dt)
    { const double s1= 1.35120719195965763404768780897,
                   s0=-1.70241438391931526809537561794;
      double s1dt=s1*dt;
      LeapFrog(s1dt); LeapFrog(s0*dt); LeapFrog(s1dt); }


////////////////////////////////////////////////////////////////////////////////


/** \brief Integrates orbits in gravitational potentials. */

class Record3D {
private:
    Potential*  Pot;                       // pointer to the potential
    double      R,z, phi, pR,pz, phid, aR,az, phidd;    
    // coordinates and deriv. of Phi
    double      Jphi, E, tol, dtm;         // energy, tolerance, max time step
    void        RungeKutta(const double);  // 4th order Runge-Kutta integrator
public:
    Record3D                  (const PSPT&, Potential*, const double=0.);
    Record3D                  (const Record3D&);
   ~Record3D                  () {}
    Record3D&    operator=    (const Record3D&);
    void       set_tolerance(const double dE=0.) {tol= (dE)? fabs(dE) : 1.e-9;}
    void       set_maxstep  (const double dt=0.) {dtm= (dt)? fabs(dt) : 1.;}
    void       stepRK_by    (double&, const double=1.4);
    PSPD       QP           () const { return PSPD(R,z,pR,pz); }
    double     QP	    (const int) const;
    PSPT       QP3D         () const {return PSPT(R,z,phi,pR,pz,R*phid); }
    double     QP3D	    (const int) const;  
    double     azi          () const { return phi; }
    double     Lz           () const { return Jphi; }
    double     energy       () const { return E; }
    Potential* potential    () const { return Pot; }
};

inline double Record3D::QP(const int i) const
{
    switch (i) {
    case 0: return R;
    case 1: return z;
    case 2: return pR;
    case 3: return pz;
    default: return 0.;
    }
}

inline double Record3D::QP3D(const int i) const
{
  switch (i) {
  case 0: return R;
  case 1: return z;
  case 2: return phi;
  case 3: return pR;
  case 4: return pz;
  case 5: return R*phid;
  default: return 0.;
  }
}


////////////////////////////////////////////////////////////////////////////////

int StartfromERpR(                // return      error flag
    Potential*,                   // input:      pointer to a Potential
    const double,                 // input:      orbit's energy ...
    const double,                 // input:      starting value for R
    const double,                 // input:      starting value for pR
    PSPD&);    			  // output:     starting point (R,0,pR,pz)

////////////////////////////////////////////////////////////////////////////////

int orbit(                        // return:     error flag
    Potential*,                   // input:      pointer to a Potential
    const PSPD&,		  // input:      start (R,z,pR,pz)
    const double,                 // input:      and tolerance for dE
    const int,                    // input:      dyn. times to be integrated
    const double,                 // input:      path length between outputs
    const char*,                  // input:      file name for output
    const char*,                  // input:      file name for output of SoS
    double&);                     // output:     Omega_l

////////////////////////////////////////////////////////////////////////////////

inline int orbit(                 // return:     error flag
    Potential*   Phi,             // Input:      pointer to a Potential
    const double E,               // Input:      orbit's energy ...
    const double tol,             // Input:      and tolerance for dE
    const double R,               // Input:      starting value for R
    const double pR,              // Input:      starting value for pR
    const int    NT,              // Input:      dyn. times to be integrated
    const double pl,              // Input:      path length between outputs
    const char*  f1,              // Input:      file name for output
    const char*  f2,              // Input:      file name for output of SoS
    double&      Oz)              // Output:     Omega_l
{
    PSPD W0;
    if( StartfromERpR(Phi,E,R,pR,W0) ) return 1;
    return orbit(Phi,W0,tol,NT,pl,f1,f2,Oz);
}

} // namespace
#endif
