/***************************************************************************//**
\file GeneratingFunction.h 
\brief File containing classes that handle generating function and derivatives for conversions between toy and true angle-action variables

*                                                                              *
* C++ code written by Walter Dehnen, 1995-96,                                  *
*                     Paul McMillan, 2007-                                     *
* e-mail: paul@astro.lu.se                                                     *
* github: https://github.com/PaulMcMillan-Astro/Torus                          *
*                                                                              *
********************************************************************************
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
*                                                       |       |              *
*                                                    ToyIsochrone  PoiTra      *
*                                                                              *
********************************************************************************
*                                                                              *
* The term `forward' for a `PhaseSpaceMap' always means the mapping needed to  *
* go from action-angle variables to ordinary phase-space coordinates, whereas  *
* `backward' denotes the opposite directions.                                  *
*                                                                              *
*******************************************************************************/


#ifndef _GenFnc_
#define _GenFnc_ 1

#include "Maps.h"

namespace torus{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/**
   \brief Parameters of GenFnc. The various values S_n
   and the corresponding vector of two integers n.
 */
class GenPar {
private:
  short   ntot, nn1, nn2, *N1, *N2;  // n for the S_n 
  double   *S;                       // S_n
  void    sortSn();                  // Put in correct order
  void    findNN();                  // Find range of n used
  void    SanityCheck();             // Check no repeats, other invalid terms.
public:
    GenPar(const int=0);
    GenPar(const GenPar&);
   ~GenPar();

   int     same_terms_as(const GenPar&) const;  // identical n, yes or no?
   void    write        (ostream&) const;       // write to file
   void    read         (istream&);             // read from file
   void    read         (int*,int*,double*,short);// read from file
   void    write_log    (ostream&) const;       // write (human readable)
   void    MakeGeneric  ();                     // Make this a typical GenPar
   void    tailor       (const double, const double, const int);
   void    edgetailor   (const double, const int);
   void    cut          (const double);
   int     Jz0          ();
   int     JR0          ();
   int     NoMix        ();
   int     AddTerm      (const int, const int);
   void    Build_JR0    (const int);
   void    addn1eq0     (const int);
   double  maxS	 () const;
    
   GenPar& operator=    (const GenPar&);
   GenPar& operator+=   (const GenPar&);
   GenPar& operator-=   (const GenPar&);
   GenPar& operator=    (const double);
   GenPar& operator*=   (const double);
   GenPar& operator/=   (const double);
   GenPar  operator-    () const;

   GenPar  operator+    (const GenPar&) const;
   GenPar  operator-    (const GenPar&) const;
   GenPar  operator*    (const double&) const;
   GenPar  operator/    (const double&) const;
   

   int     operator==   (const GenPar&) const;
   int     operator!=   (const GenPar&) const;
   double   operator()   (const int) const;
   double&  operator[]   (const int);
   
   void    put          (ostream&) const;
   void    get          (istream&);
   void    put_terms    (ostream&) const;
   void    get_terms    (istream&);
   int     skip         (istream&) const;
   void    skip_terms   (istream&, const int) const;

    int     NumberofTerms() const;
    int     NumberofTermsUsed() const;
    int     NumberofN1   () const;
    int     NumberofN2   () const;
    int     n1           (const int) const;
    int     n2           (const int) const;
};
inline double   GenPar::operator()   (const int i) const { return S[i]; }
inline double&  GenPar::operator[]   (const int i) { return S[i]; }
inline int     GenPar::n1           (const int i) const { return N1[i]; }
inline int     GenPar::n2           (const int i) const { return N2[i]; }
inline int     GenPar::NumberofTerms() const { return ntot; }
inline int     GenPar::NumberofN1() const { return nn1; }
inline int     GenPar::NumberofN2() const { return nn2; }

inline istream& operator>> (istream& from, GenPar& G)
	{ G.read(from); return from; }
inline ostream& operator<< (ostream& to, GenPar& G)
	{ G.write(to); return to; }
inline GenPar operator* (const double x, const GenPar& GP) 
{ return GP*x; }

////////////////////////////////////////////////////////////////////////////////

/**
   \brief Parameters of AngMap.  The various values d
   S_n/d J_i and corresponding vector of two integers n. (see
   eq 6 of McMillan & Binney 2008)
 */
class AngPar {  // class completely inline
    friend class AngMap;
private:
    GenPar dS1, dS2, dS3;
    void CheckTerms() const;
public:
    AngPar(const int=0);
    AngPar(const AngPar& ap);
    AngPar(const GenPar& s1, const GenPar& s2, const GenPar& s3);
   ~AngPar() {}
    GenPar  dSdJ1() const		{ return dS1; }
    GenPar  dSdJ2() const		{ return dS2; }
    GenPar  dSdJ3() const		{ return dS3; }
    double   dSdJ1(const int i) const 	{ return dS1(i); }
    double   dSdJ2(const int i) const 	{ return dS2(i); }
    double   dSdJ3(const int i) const 	{ return dS3(i); }
    void    dSdJ1(const int i, const double val)  { dS1[i] = val; }
    void    dSdJ2(const int i, const double val)  { dS2[i] = val; }
    void    dSdJ3(const int i, const double val)  { dS3[i] = val; }

    AngPar&  operator=    (const AngPar&);
    AngPar&  operator+=   (const AngPar&);
    AngPar&  operator-=   (const AngPar&);
    AngPar&  operator=    (const double);
    AngPar&  operator*=   (const double);
    AngPar&  operator/=   (const double);

    int         operator==   (const AngPar&) const;
    int         operator!=   (const AngPar&) const;
    double       operator()   (const int) const;
    double&      operator[]   (const int);
    int         NumberofTerms() const;

    void        put          (ostream&) const;
    void        get          (istream&);
};

inline void AngPar::CheckTerms() const
	{ if(!dS1.same_terms_as(dS2) || !dS1.same_terms_as(dS3))
	      throw std::runtime_error("Torus Error -4: mismatch between dS1/2/3 in AngPar"); }
inline AngPar::AngPar(const int n): dS1(n), dS2(n), dS3(n) {}
inline AngPar::AngPar(const AngPar& a): 
	      dS1(a.dS1),dS2(a.dS2),dS3(a.dS3)
{ CheckTerms(); }
inline AngPar::AngPar(const GenPar& s1, const GenPar& s2, const GenPar& s3): 
	      dS1(s1), dS2(s2), dS3(s3)
	{ CheckTerms(); }
inline AngPar& AngPar::operator=(const AngPar& a)
	{ dS1=a.dS1; dS2=a.dS2; dS3=a.dS3; return *this; }
inline AngPar& AngPar::operator+=(const AngPar& a)
	{ dS1+=a.dS1; dS2+=a.dS2;  dS3+=a.dS3; return *this; }
inline AngPar& AngPar::operator-=(const AngPar& a)
	{ dS1-=a.dS1; dS2-=a.dS2; dS3-=a.dS3; return *this; }
inline AngPar& AngPar::operator=(const double x)
	{ dS1=x; dS2=x; dS3=x; return *this; }
inline AngPar& AngPar::operator*=(const double x)
	{ dS1*=x; dS2*=x; dS3*=x; return *this; }
inline AngPar& AngPar::operator/=(const double x)
	{ dS1/=x; dS2/=x; dS3/=x; return *this; }
inline int AngPar::operator== (const AngPar& a) const
	{ return (dS1==a.dS1 && dS2==a.dS2 && dS3==a.dS3 ); }
inline int AngPar::operator!= (const AngPar& a) const
	{ return (dS1!=a.dS1 || dS2!=a.dS2 || dS3!=a.dS3); }
inline double AngPar::operator() (const int i) const
	{ throw std::runtime_error("Torus Error -4: AngPar::operator() called"); return dS1(i); }
inline double& AngPar::operator[] (const int i)
	{ throw std::runtime_error("Torus Error -4: AngPar::operator[] called"); return dS1[i]; }
inline int AngPar::NumberofTerms() const { return dS1.NumberofTerms(); }
inline void AngPar::put(ostream& to) const
	{ dS1.put(to); to<<'\n'; dS2.put(to); dS3.put(to); to<<'\n'; }
inline void AngPar::get(istream& from)
	{ dS1.get(from); dS2.get(from); dS3.get(from); }

////////////////////////////////////////////////////////////////////////////////
/**
   \brief Mapping between the true actions and the toy actions as a
   function of toy angle. Has parameters GenPar. See eq. 5 McMillan &
   Binney 2008.
 */
class GenFnc : public PhaseSpaceMap {
protected:
    GenPar Sn;
public:
    GenFnc() {}
    GenFnc(const GenPar& s): Sn(s) {}
    GenFnc(const GenFnc& g): PhaseSpaceMap(), Sn(g.Sn) 	{}
   ~GenFnc() {}
    GenFnc& operator=         (const GenFnc&);
    void    set_parameters    (const GenPar& s) { Sn=s; }
    GenPar  parameters        ()                    	  const;
    void    parameters        (GenPar*)			  const;
    double  coeff              (const int)		  const;
    int     NumberofParameters()			  const;
    int     n1		      (const int)		  const;
    int     n2		      (const int)		  const;
    PSPD    ForwardWithDerivs (const PSPD&, double[2][2]) const;
    PSPD    Forward           (const PSPD&)		  const;
    PSPT    Forward3D         (const PSPT&)		  const;
};

inline GenPar  GenFnc::parameters() const
    { return Sn; }
inline void    GenFnc::parameters(GenPar* s) const
    { *s = Sn; }
inline double   GenFnc::coeff(const int i) const
    { return Sn(i); }
inline int     GenFnc::n1(const int i) const
    { return Sn.n1(i); }
inline int     GenFnc::n2(const int i) const
    { return Sn.n2(i); }
inline int     GenFnc::NumberofParameters() const
    { return Sn.NumberofTerms(); }
inline GenFnc& GenFnc::operator=         (const GenFnc& gf)
    { Sn=gf.Sn; return *this;}

////////////////////////////////////////////////////////////////////////////////

/**
   \brief Mapping between the true actions and the toy actions as a
   function of toy angle for cases where a Torus is being fit, so the
   toy angles are evenly spaced over (half of) the torus.
 */
class GenFncFit : public GenFnc {
private:
    int    Nth1, Nth2;      // Number of points for theta1 fit, theta2 fit
    double Pin1, Pin2;      // Pi/Nth1, Pi/Nth2
    double **cc1, **ss1, **cc2, **ss2; // cos(Sn.n1 * angles evenly spaced
                                       // round half the torus) etc
    void   FreeTrig();
    void   SetTrig();
    void   AllocAndSetTrig();
public:
    GenFncFit(const GenFnc&, const int, const int);
    GenFncFit(const GenPar&, const int, const int);
   ~GenFncFit();
    void set_Nth       (const int, const int);
    void set_parameters(const GenPar&);
    int  N_th1         () const { return Nth1; }
    int  N_th2         () const { return Nth2; }
    PSPD Map           (const double, const double, const int, const int) const;
    PSPD MapWithDerivs (const double, const double, const int, const int,
                        GenPar&, GenPar&) const;
};

inline GenFncFit::GenFncFit(const GenFnc& G, const int nt1, const int nt2)
       : GenFnc(G), Nth1(nt1), Nth2(nt2) { AllocAndSetTrig(); }
inline GenFncFit::GenFncFit(const GenPar& S, const int nt1, const int nt2)
       : GenFnc(S), Nth1(nt1), Nth2(nt2) { AllocAndSetTrig(); }
inline GenFncFit::~GenFncFit() { FreeTrig(); }

////////////////////////////////////////////////////////////////////////////////


/**
   \brief Mapping between the true angles and the toy angles as a
   function of toy angle. Has parameters AngPar. See eq. 6 McMillan &
   Binney 2008.
 */
class AngMap : public PhaseSpaceMapWithBackward {
private:
    AngPar A;
    PSPD NewtonStep(double&, double&, double&, const PSPD, const PSPD&) const;
    PSPD Map       (const PSPD&) const;
public:
    AngMap() {}
    AngMap(const AngPar&);
    AngMap(const AngMap&);
   ~AngMap() {}
    AngMap& operator=         (const AngMap&);
    void    set_parameters    (const AngPar&);
    AngPar  parameters        ()            const;
    void    parameters        (AngPar* )    const;
    GenPar  dSdJ1             () 	    const;
    GenPar  dSdJ2             () 	    const;
    GenPar  dSdJ3             () 	    const;
    double   dSdJ1             (const int)   const;
    double   dSdJ2             (const int)   const;
    double   dSdJ3             (const int)   const;
    int     NumberofParameters()            const;
    PSPD    Forward           (const PSPD&) const; 
    PSPD    Backward          (const PSPD&) const;
    PSPD    BackwardWithDerivs(const PSPD&, double[2][2]) const; 
    PSPT    Forward3D         (const PSPT&) const;
    PSPT    Backward3D        (const PSPT&) const;
    PSPT    Backward3DWithDerivs(const PSPT&, double[2][2]) const;
};
inline         AngMap::AngMap (const AngPar& a) : A(a) {}
inline         AngMap::AngMap (const AngMap& am) : PhaseSpaceMapWithBackward(),
						   A(am.A) {}
inline void    AngMap::set_parameters (const AngPar& a) { A=a; }
inline AngPar  AngMap::parameters() const	 { return A; }
inline void    AngMap::parameters(AngPar *a) const	 { *a = A; }
inline GenPar  AngMap::dSdJ1 () const		 { return A.dSdJ1(); }
inline GenPar  AngMap::dSdJ2 () const		 { return A.dSdJ2(); }
inline GenPar  AngMap::dSdJ3 () const		 { return A.dSdJ3(); }
inline double   AngMap::dSdJ1 (const int i) const { return A.dSdJ1(i); }
inline double   AngMap::dSdJ2 (const int i) const { return A.dSdJ2(i); }
inline double   AngMap::dSdJ3 (const int i) const { return A.dSdJ3(i); }
inline int     AngMap::NumberofParameters() const { return A.NumberofTerms(); }
inline AngMap& AngMap::operator= (const AngMap& am) {A=am.A; return *this;}
inline PSPD    AngMap::Backward(const PSPD& Jt) const { return Map(Jt); }

////////////////////////////////////////////////////////////////////////////////

} // namespace

#endif
