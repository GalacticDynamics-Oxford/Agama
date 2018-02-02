/***************************************************************************//**
\file Types.h
\brief Contains classes PSPD PSPT, and typedefs for various Vector.h style 
Vectors used within Torus code (e.g. Actions)

*                                                                              *
* Types.h                                                                      *
*                                                                              *
* C++ code written by Walter Dehnen, 1994-96,                                  *
* Oxford University, Department of Physics, Theoretical Physics.               *
* address: 1 Keble Road, Oxford OX1 34P, United Kingdom                        *
* e-mail:  p.mcmillan1@physics.ox.ac.uk                                        *
*                                                                              *
*******************************************************************************/

#ifndef _TorusTypes_
#define _TorusTypes_ 1

#include "WD_Vector.h"

namespace torus{
using namespace WD;
using std::ostream;
using std::istream;

typedef Vector<double,3> Frequencies;
typedef Vector<double,3> Actions;
typedef Vector<double,3> Angles;
typedef Vector<double,3> Position;
typedef Vector<double,3> Velocity;
typedef Vector<double,4> Errors;

typedef Vector<double,2> vec2;
typedef Vector<double,3> vec3;
typedef Vector<double,4> vec4;
typedef Vector<double,6> vec6;

/** \brief Phase Space Point Doublet -- phase space point in 2D space

PSPD is used for any phase-space point in 2D -
e.g. (J_r,J_z,theta_r,theta_z) or (R,z,v_R,v_z)
 */
class PSPD{
protected:
    double a[4];
    void range_error() const       {cerr<<" PSPD: range error\n";      exit(1);}
    void div_by_zero_error() const {cerr<<" PSPD: division by zero\n"; exit(1);}
public:
    PSPD               () {}
    PSPD               (const double);
    PSPD               (const PSPD&);
    PSPD               (const double, const double, const double, const double);
   ~PSPD               () {}

    PSPD&   operator=  (const PSPD&);
    PSPD&   operator+= (const PSPD&);
    PSPD&   operator-= (const PSPD&);
    PSPD&   operator=  (const double);
    PSPD&   operator+= (const double);
    PSPD&   operator-= (const double);
    PSPD&   operator*= (const double);
    PSPD&   operator/= (const double);

    PSPD    operator-  () const;
    PSPD    operator+  (const PSPD&) const;
    PSPD    operator-  (const PSPD&) const;
    double  operator*  (const PSPD&) const;
    int     operator== (const PSPD&) const;
    int     operator!= (const PSPD&) const;
    PSPD    operator+  (const double) const;
    PSPD    operator-  (const double) const;
    PSPD    operator*  (const double) const;
    PSPD    operator/  (const double) const;

    double  norm       () const;
    double& operator[] (const int n)       { return  a[n]; }
    double  operator() (const int n) const { return  a[n]; }
};

ostream& operator<< (ostream&, const PSPD&);
istream& operator>> (istream&, PSPD&);
double   norm       (const PSPD&);
PSPD     operator+  (const double, const PSPD&);
PSPD     operator-  (const double, const PSPD&);
PSPD     operator*  (const double, const PSPD&);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// PSPD: member functions

inline PSPD::PSPD(const double fill_value)
    { for(int i=0; i<4; i++) a[i] = fill_value; }

inline PSPD::PSPD(const PSPD& V)
    { for(int i=0; i<4; i++) a[i] = V.a[i]; }

inline PSPD::PSPD(const double x0, const double x1, const double x2,
	  const double x3)
    { a[0]=x0; a[1]=x1; a[2]=x2; a[3]=x3; }

inline PSPD& PSPD::operator= (const PSPD& V)
    { for(int i=0; i<4; i++) a[i] = V.a[i];
      return *this; }

inline PSPD& PSPD::operator= (const double fill_value)
    { for(int i=0; i<4; i++) a[i] = fill_value;
      return *this; }

inline PSPD& PSPD::operator+= (const PSPD& V)
    { for(int i=0; i<4; i++) a[i] += V.a[i];
      return *this; }

inline PSPD& PSPD::operator-= (const PSPD& V)
    { for(int i=0; i<4; i++) a[i] -= V.a[i];
      return *this; }

inline PSPD& PSPD::operator+= (const double m)
    { for(int i=0; i<4; i++) a[i] += m;
      return *this; }

inline PSPD& PSPD::operator-= (const double m)
    { for(int i=0; i<4; i++) a[i] -= m;
      return *this; }

inline PSPD& PSPD::operator*= (const double m)
    { for(int i=0; i<4; i++) a[i] *= m;
      return *this; }

inline PSPD& PSPD::operator/= (const double m)
    { if(m==double(0.)) div_by_zero_error();
      for(int i=0; i<4; i++) a[i] /= m;
      return *this; }

inline PSPD PSPD::operator- () const
    { PSPD P(0); return P-=*this; }

inline PSPD PSPD::operator+ (const PSPD& V) const
    { PSPD P(*this); return P+=V; }

inline PSPD PSPD::operator- (const PSPD& V) const
    { PSPD P(*this); return P-=V; }

inline double PSPD::operator* (const PSPD& V) const
    { return a[0]*V.a[0] + a[1]*V.a[1] + a[2]*V.a[2] + a[3]*V.a[3]; }

inline int PSPD::operator== (const PSPD& V) const
    { for(int i=0; i<4; i++) if(a[i] != V.a[i]) return 0;
      return 1; }

inline int PSPD::operator!= (const PSPD& V) const
    { for(int i=0; i<4; i++) if(a[i] != V.a[i]) return 1;
      return 0; }

inline PSPD PSPD::operator+ (const double x) const
    { PSPD P(*this); return P+=x; }

inline PSPD PSPD::operator- (const double x) const
    { PSPD P(*this); return P-=x; }

inline PSPD PSPD::operator* (const double x) const
    { PSPD P(*this); return P*=x; }

inline PSPD PSPD::operator/ (const double x) const
    { PSPD P(*this); return P/=x; }

inline double PSPD::norm() const 
    { return a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3]; }

// class PSPD: related functions

inline ostream& operator<< (ostream& s, const PSPD& V)
    { s << ' ';
      for(int i=0; i<4; i++) s << ' ' << V(i);
      s << ' ';
      return s; }

inline istream& operator>> (istream& s, PSPD& V)
    { for(int i=0; i<4; i++) s >> V[i];
      return s; }

inline double norm      (const PSPD& V)
    { return V.norm(); }

inline PSPD   operator+ (const double x, const PSPD& V)
    { return V+x; }

inline PSPD   operator- (const double x, const PSPD& V)
    { PSPD P(x);
      return P-=V; }

inline PSPD   operator* (const double x, const PSPD& V)
    { return V*x; }


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/** \brief Phase Space Point Triplet -- phase space point in 3D space

PSPD is used for any phase-space point in 3D -
e.g. (J_r,J_z,J_phi,theta_r,theta_z,theta_phi) or (R,z,phi,v_R,v_z,v_phi)
 */
class PSPT{
protected:
    double a[6];
    void range_error() const       {cerr<<" PSPT: range error\n";      exit(1);}
    void div_by_zero_error() const {cerr<<" PSPT: division by zero\n"; exit(1);}
public:
    PSPT               () {}
    PSPT               (const double);
    PSPT               (const PSPT&);
    PSPT               (const double, const double, const double, 
	    const double, const double, const double);
   ~PSPT               () {}

    PSPT&   operator=  (const PSPT&);
    PSPT&   operator+= (const PSPT&);
    PSPT&   operator-= (const PSPT&);
    PSPT&   operator=  (const double);
    PSPT&   operator+= (const double);
    PSPT&   operator-= (const double);
    PSPT&   operator*= (const double);
    PSPT&   operator/= (const double);

    PSPT    operator-  () const;
    PSPT    operator+  (const PSPT&) const;
    PSPT    operator-  (const PSPT&) const;
    double  operator*  (const PSPT&) const;
    int     operator== (const PSPT&) const;
    int     operator!= (const PSPT&) const;
    PSPT    operator+  (const double) const;
    PSPT    operator-  (const double) const;
    PSPT    operator*  (const double) const;
    PSPT    operator/  (const double) const;

    double  norm       () const;
    double& operator[] (const int n)       { return  a[n]; }
    double  operator() (const int n) const { return  a[n]; }

    PSPD    Give_PSPD  () const;
    void    Take_PSPD  (const PSPD&);
};

ostream& operator<< (ostream&, const PSPT&);
istream& operator>> (istream&, PSPT&);
double   norm       (const PSPT&);
PSPT     operator+  (const double, const PSPT&);
PSPT     operator-  (const double, const PSPT&);
PSPT     operator*  (const double, const PSPT&);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// PSPT: member functions

inline PSPT::PSPT(const double fill_value)
    { for(int i=0; i<6; i++) a[i] = fill_value; }

inline PSPT::PSPT(const PSPT& V)
    { for(int i=0; i<6; i++) a[i] = V.a[i]; }

inline PSPT::PSPT(const double x0, const double x1, const double x2,
	  const double x3, const double x4, const double x5)
    { a[0]=x0; a[1]=x1; a[2]=x2; a[3]=x3; a[4]=x4; a[5]=x5;}

inline PSPT& PSPT::operator= (const PSPT& V)
    { for(int i=0; i<6; i++) a[i] = V.a[i];
      return *this; }

inline PSPT& PSPT::operator= (const double fill_value)
    { for(int i=0; i<6; i++) a[i] = fill_value;
      return *this; }

inline PSPT& PSPT::operator+= (const PSPT& V)
    { for(int i=0; i<6; i++) a[i] += V.a[i];
      return *this; }

inline PSPT& PSPT::operator-= (const PSPT& V)
    { for(int i=0; i<6; i++) a[i] -= V.a[i];
      return *this; }

inline PSPT& PSPT::operator+= (const double m)
    { for(int i=0; i<6; i++) a[i] += m;
      return *this; }

inline PSPT& PSPT::operator-= (const double m)
    { for(int i=0; i<6; i++) a[i] -= m;
      return *this; }

inline PSPT& PSPT::operator*= (const double m)
    { for(int i=0; i<6; i++) a[i] *= m;
      return *this; }

inline PSPT& PSPT::operator/= (const double m)
    { if(m==double(0.)) div_by_zero_error();
      for(int i=0; i<6; i++) a[i] /= m;
      return *this; }

inline PSPT PSPT::operator- () const
    { PSPT P(0); return P-=*this; }

inline PSPT PSPT::operator+ (const PSPT& V) const
    { PSPT P(*this); return P+=V; }

inline PSPT PSPT::operator- (const PSPT& V) const
    { PSPT P(*this); return P-=V; }

inline double PSPT::operator* (const PSPT& V) const
    { return a[0]*V.a[0] + a[1]*V.a[1] + a[2]*V.a[2] + a[3]*V.a[3]; }

inline int PSPT::operator== (const PSPT& V) const
    { for(int i=0; i<6; i++) if(a[i] != V.a[i]) return 0;
      return 1; }

inline int PSPT::operator!= (const PSPT& V) const
    { for(int i=0; i<6; i++) if(a[i] != V.a[i]) return 1;
      return 0; }

inline PSPT PSPT::operator+ (const double x) const
    { PSPT P(*this); return P+=x; }

inline PSPT PSPT::operator- (const double x) const
    { PSPT P(*this); return P-=x; }

inline PSPT PSPT::operator* (const double x) const
    { PSPT P(*this); return P*=x; }

inline PSPT PSPT::operator/ (const double x) const
    { PSPT P(*this); return P/=x; }

inline double PSPT::norm() const 
    { return a[0]*a[0]+a[1]*a[1]+a[2]*a[2]+a[3]*a[3]+a[4]*a[4]+a[5]*a[5]; }

inline PSPD PSPT::Give_PSPD() const { 
  PSPD tmp;
  tmp[0] = a[0]; tmp[1] = a[1];
  tmp[2] = a[3]; tmp[3] = a[4];
  return tmp; }

inline void PSPT::Take_PSPD(const PSPD& tmp) {
  a[0] = tmp(0); a[1] = tmp(1);
  a[3] = tmp(2); a[4] = tmp(3);
}



// class PSPT: related functions

inline ostream& operator<< (ostream& s, const PSPT& V)
{ s << ' ';
  for(int i=0; i<6; i++) s << ' ' << V(i);
  s << ' ';
  return s; }

inline istream& operator>> (istream& s, PSPT& V)
{ for(int i=0; i<6; i++) s >> V[i];
  return s; }

inline double norm      (const PSPT& V)
{ return V.norm(); }

inline PSPT   operator+ (const double x, const PSPT& V)
{ return V+x; }

inline PSPT   operator- (const double x, const PSPT& V)
{ PSPT P(x);
  return P-=V; }

inline PSPT   operator* (const double x, const PSPT& V)
{ return V*x; }



typedef double*                 Pdble;
} // namespace
#endif
