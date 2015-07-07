#ifndef ANALYTICPOTENTIALS_H
#define ANALYTICPOTENTIALS_H

#include "BaseClasses.hpp"
#include "GSLInterface.hpp"
#include "falPot.h"
#include "coordsys.hpp"
#include "coordtransforms.hpp"

//============================================================================
///
///	## Spherical NFW potential
///		rs is the scale radius
///		GM is G times the mass M
///		qy and qz are the y and z flattenings respectively
///
///	 Phi = -GM/sqrt{x^2+y^2+z^2}
///
//============================================================================

class NFWsphere: public PotentialBase{
private:
  double rs, GM, rho0;
public:
  NFWsphere(double RS, double m): rs(RS), GM(GRAV*m) {
    rho0 = m / (4. * PI * pow(rs,3));
  }
  double Phi(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  double dPhidr(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double d2Phidr(VecDoub_I &x) const;
  double d2PhidR(VecDoub_I &x) const;
  double d2Phidz(VecDoub_I &x) const;
  void EpicycleFrequencies(const double, VecDoub_O&) const;
  VecDoub Forces(VecDoub_I &x) const;
  double density(VecDoub_I &x) const;
};

//============================================================================
/// Axisymmetric Stackel potential with perfect ellipsoidal density
///
/// Requires an oblate spheroidal coordinate system instance
///
/// (adapted from Jason Sanders)
//============================================================================
class StackelProlate_PerfectEllipsoid: public PotentialBase{
private:
  double Const;
  OblateSpheroidCoordSys *CS;
public:
  StackelProlate_PerfectEllipsoid(double Rho0, double alpha) {
    CS    = new OblateSpheroidCoordSys(alpha);
    Const = 2.*PI*GRAV*Rho0*(-CS->alpha());
  }
  virtual ~StackelProlate_PerfectEllipsoid(){delete CS;}
  inline double alpha()const {return CS->alpha();}
  inline double gamma()const {return CS->gamma();}
  double G (double tau) const;
  double GPrime (double tau) const;
  VecDoub Vderivs (VecDoub_I &tau) const;
  inline VecDoub x2tau(VecDoub_I &x) const{return CS->x2tau(x);}
  inline VecDoub xv2tau(VecDoub_I &x) const{return CS->xv2tau(x);}
  double Phi(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  VecDoub Forces(VecDoub_I &x) const;
  double Phi_tau(VecDoub_I &tau) const;
  VecDoub x2ints(VecDoub_I &x, VecDoub_O *tau = NULL) const;
};

//============================================================================
///
///	Axisymmetric Miyamoto-Nagai potential
///
///	Phi = -GM/sqrt(R^2+(A+sqrt(z^2+b^2))^2)
///
/// (adapted from Jason Sanders)
//============================================================================
class MiyamotoNagai: public PotentialBase{
private:
  double A, GM, M, Bq;
public:
  MiyamotoNagai(double a, double m, double b)
    : A(a), GM(GRAV*m), M(m), Bq(b*b) {}
  double Phi(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  double d2PhidR(VecDoub_I &x) const;
  double d2Phidz(VecDoub_I &x) const;
  void EpicycleFrequencies(const double, VecDoub_O&) const;
  double Vc(const double R) const;
  VecDoub Forces(VecDoub_I &x) const;
  double density(VecDoub_I &x) const;
};

//============================================================================
/// Wrapper for potentials produced by the GalPot code
///
/// (adapted from Jason Sanders)
//============================================================================
class GalPot: public PotentialBase{
private:
  GalaxyPotential *PhiWD;
public:
  GalPot(std::string TpotFile);
  GalPot(int, DiskPar*, int, SphrPar*);
  void reset(const int, DiskPar*, const int, SphrPar*);
  inline DiskPar DiskParameter(const int i) {return PhiWD->DiskParameter(i);}
  inline SphrPar SpheroidParameter(const int i) {return PhiWD->SpheroidParameter(i);}
  double Phi(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  void EpicycleFrequencies(const double, VecDoub_O&) const;
  VecDoub Forces (VecDoub_I &x) const;
  double density(VecDoub_I &x) const;
  inline double DisksSurfaceDensity(const double R) const
  { return PhiWD->DisksSurfaceDensity(R);}
  double DisksStellarDensity(VecDoub_I &x) const;
  inline double Rc(const double Lz) const {return PhiWD->RfromLc(fabs(Lz)/conv::kpcMyr2kms);}
  GalaxyPotential *PWD() const{return PhiWD;}
  void printParameterFile(char*);
};

//============================================================================
///
///	Triaxial Stackel potential with perfect ellipsoidal density
///
///	Requires a confocal ellipsoidal coordinate system instance
///
/// (adapted from Jason Sanders)
//============================================================================

class StackelTriaxial: public PotentialBase{
private:
  double Const, a, b, c, l, Flm, Elm, sinm;
  ConfocalEllipsoidalCoordSys *CS;
  VecDoub Vderivs(VecDoub_I &tau) const;
public:
  StackelTriaxial(double Rho0, double alpha, double beta){
    CS    = new ConfocalEllipsoidalCoordSys(alpha,beta);
    a     = sqrt(-CS->alpha());b = sqrt(-CS->beta());c = sqrt(-CS->gamma());
    l     = acos(c/a);double sinL = (1-(c/a)*(c/a));
    sinm  = sqrt((1.-(b/a)*(b/a))/sinL);
    Flm   = ellint_first(l,sinm);
    Elm   = ellint_second(l,sinm);
    Const = 2.*PI*GRAV*Rho0*b*c/sqrt(sinL);
    // std::cout<<(-Const/a/a/sinL/sinm/sinm/3.*((1-sinm*sinm)*Flm+(2.*sinm*sinm-1.)*Elm-b*sinm*sinm/a*sqrt(sinL)*cos(l)))
    // 	<<" "<<-Const/a/a/sinL/sinm/sinm/(1-sinm*sinm)*(Elm-(1-sinm*sinm)*Flm-c*sinm*sinm*sin(l)/b)
    // 	<<" "<<-Const*0.5/a/a/3./sinL/(1-sinm*sinm)*(-(1-sinm*sinm)*Flm+(2.*(1-sinm*sinm)-1)*Elm+b*sin(l)*(b*b/c/c-(1-sinm*sinm))/c)<<std::endl;
  }
  virtual ~StackelTriaxial(){delete CS;}
  inline double alpha() const{return CS->alpha();}
  inline double beta() const{return CS->beta();}
  inline double gamma() const{return CS->gamma();}
  double G(double tau) const;double GPrime(double tau) const;
  inline VecDoub x2tau(VecDoub x) const{return CS->x2tau(x);}
  inline VecDoub xv2tau(VecDoub x) const{return CS->xv2tau(x);}
  double Phi(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  VecDoub Forces(VecDoub_I &x) const;
  double Phi_tau(VecDoub_I &tau) const;
  VecDoub tau2ints(VecDoub_I &tau) const;
};

//============================================================================
///
///	## Triaxial NFW potential
///		rs is the scale radius
///		GM is G times the mass M
///		qy and qz are the y and z flattenings respectively
///
///	 \f[\Phi = \frac{-GM}{\sqrt{x^2+y^2/q_y^2+z^2/q_z^2}}
///		\log\Big(1+\frac{\sqrt{x^2+y^2/q_y^2+z^2/q_z^2}}{R_s}\Big)\f]
///
/// (adapted from Jason Sanders)
//============================================================================
class NFW: public PotentialBase{
private:
  double rs, GM, q1, q2;
public:
  NFW(double RS, double gm, double qy, double qz): rs(RS), GM(gm), q1(qy*qy), q2(qz*qz){}
  double Phi (VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  VecDoub Forces (VecDoub_I &x) const;
  double density (VecDoub_I &x) const;
};

//============================================================================
///
///	Triaxial logarithmic potential
///
///	Phi = Vc^2/2 log(x^2+y^2/q1^2+z^2/q2^2)
///
/// (adapted from Jason Sanders)
//============================================================================
class Logarithmic: public PotentialBase{
private:
  double Vc2, q1, q2;
public:
  Logarithmic(double VC, double P, double Q): Vc2(VC*VC), q1(P*P),q2(Q*Q){}
  double Phi (VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  VecDoub Forces (VecDoub_I &x) const;
};

class PlummerSphere: public PotentialBase{
private:
  double rs,rs2, GM, rho0;
public:
  PlummerSphere(double RS, double m): rs(RS),rs2(RS*RS), GM(GRAV*m) {
    rho0 = m * 3. / (4. * PI * pow(rs,3));
  }
  double Phi(VecDoub_I &x) const;
  double dPhidr(VecDoub_I &x) const;
  double d2Phidr(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double d2PhidR(VecDoub_I &x) const;
  double d2Phidz(VecDoub_I &x) const;
  void EpicycleFrequencies(const double, VecDoub_O&) const;
  VecDoub Forces(VecDoub_I &x) const;
  double density(VecDoub_I &x) const;
  double Mass(const double r) const;
};

#ifdef HAVE_SMILE
class SmilePotential: public PotentialBase{
private:
  const void* pot;
public:
  SmilePotential(const char* filename);
  SmilePotential(const int numParams, const char* const params[]);
  ~SmilePotential();
  double Phi (VecDoub_I &x) const;
  VecDoub Forces (VecDoub_I &x) const;
  double density(VecDoub_I &x) const;
  double dPhidr(VecDoub_I &x) const;
  double dPhidR(VecDoub_I &x) const;
  double dPhidx(VecDoub_I &x) const;
  double dPhidy(VecDoub_I &x) const;
  double dPhidz(VecDoub_I &x) const;
  double d2Phidr(VecDoub_I &x) const;
  double d2PhidR(VecDoub_I &x) const;
  double d2Phidz(VecDoub_I &x) const;
  void EpicycleFrequencies(const double, VecDoub_O&) const;
};
#endif

#endif
