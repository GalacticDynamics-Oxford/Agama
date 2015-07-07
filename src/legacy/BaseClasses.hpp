#ifndef BASECLASSES_H
#define BASECLASSES_H

#include <iostream>
#include "nr3.h"
#include "utils.hpp"
#include "potential_base.h"

/* ####################################################

   Basic functionality for a Galaxy potential

   #################################################### */
inline coord::PosCar toPosCar(VecDoub_I &x) { return coord::PosCar(x[0],x[1],x[2]); }
inline coord::PosVelCar toPosVelCar(VecDoub_I &x) { return coord::PosVelCar(x[0],x[1],x[2],x[3],x[4],x[5]); }
inline VecDoub toVec(const coord::PosVelCar p) { 
    VecDoub x(6); x[0]=p.x; x[1]=p.y; x[2]=p.z; x[3]=p.vx; x[4]=p.vy; x[5]=p.vz; return x; }

class PotentialBase{
public:
  const potential::BasePotential& pot;
  PotentialBase(const potential::BasePotential& p) : pot(p) {};
  inline double Phi(VecDoub_I &x) const {
    double phi;
    pot.eval(toPosCar(x), &phi);
    return phi;
  }
 /* virtual double dPhidx(VecDoub_I &x) const {
    cout << "dPhidx(): Not defined for this potential!\n"; exit(-1);
    return 0;}
  virtual double dPhidy(VecDoub_I &x) const {
    cout << "dPhidy(): Not defined for this potential!\n"; exit(-1);
    return 0;}
  virtual double dPhidz(VecDoub_I &x) const {
    cout << "dPhidz(): Not defined for this potential!\n"; exit(-1);
    return 0;}
  virtual double dPhidr(VecDoub_I &x) const {
    cout << "dPhidr(): Not defined for this potential!\n"; exit(-1);
    return 0;}
  virtual double dPhidR(VecDoub_I &x) const {
    cout << "dPhidR(): Not defined for this potential!\n"; exit(-1);
    return 0;}*/
  VecDoub Forces(VecDoub_I &x) const {
    coord::GradCar f;
    pot.eval(toPosCar(x), NULL, &f);
    VecDoub forces(3);
    forces[0] = -f.dx; forces[1] = -f.dy; forces[2] = -f.dz;
    return forces;
  }
/*  virtual void EpicycleFrequencies(const double R, VecDoub_O &freqs) const {

    VecDoub x(3); x[0] = R; x[1] = x[2] = 0.0;

    freqs[0] = 0.0;//sqrt( d2PhidR(x) + 3.0*dPhidR(x)/R ); // kappa
    freqs[1] = sqrt( dPhidR(x)/R );                    // Omega_phi
    freqs[2] = 0.0;//sqrt( d2Phidz(x) );                   // nu_z

    return;
  }
  virtual double density(VecDoub_I &x) const {
    cout << "density(): Not defined for this potential!\n"; exit(-1);
    return 0;
  }
  inline virtual double Vc(const double R) const {
    VecDoub x(3,0.0); x[0] = R;
    return sqrt(dPhidR(x)*R);
  }*/
  //virtual double Rc(const double Lz) const;
  /* Returns circular radius for given Lz */

  inline double H(VecDoub_I &xv) const {
    /* returns Hamiltonian at Cartesian (x,v) */
    double X[3] = {xv[0],xv[1],xv[2]};
    VecDoub x(3,X);
    return 0.5*(xv[3]*xv[3]+xv[4]*xv[4]+xv[5]*xv[5])+Phi(x);
  }
  inline double Lz(VecDoub_I &xv) const {
    /* returns z-component of angular momentum at Cartesian (x,v) */
    return xv[0]*xv[4]-xv[1]*xv[3];
  }
//  VecDoub dPhidRdz(VecDoub_I &Rz) const;
  // returns second derivative of potential wrt R and z
  // assumes axisymmetry -- takes VecDoub Rz = {R,z}
  // returns (dP/dRdR, dPdRdz, dPdzdz)

  //double DeltaGuess(VecDoub_I &x) const;

};

#if 0
/* ####################################################

   Basic functionality for a object converting phase
   space coordinates to actions (and angles)

   #################################################### */

class Action_Finder{
  /* General base class for PotentialBase */
public:
    inline virtual VecDoub actions(VecDoub_I &w) const{
    /* actions for Cartesian 6D phase space position w */
    /* -- to be overridden by derived classes -- */
    return VecDoub(3,0.0);
  }
#ifdef BOOST
  // ----------------------------------------------------------
  // Stuff for Python
  // ----------------------------------------------------------
  inline virtual boost::python::numeric::array actions_py(boost::python::numeric::array f){
    VecDoub x(6,0.0);
    for(int i=0;i<6;i++)
      x[i]=boost::python::extract<double>(f[i]);
    x = actions(x);
    return boost::python::numeric::array(boost::python::make_tuple(x[0],x[1],x[2]));
  }
#endif

};

/* ####################################################

   Basic functionality for an action-based distribution function (f(J))

   #################################################### */

class DistributionFunction {
public:
    double Norm; // Normalisation factor
    inline virtual double Prob(VecDoub_I &J,void *params = NULL) const {
      /* Evaluates the DF at the Action coordinates J */
      /* Optional argument 'Pot' in case additional information on the
	 Potential is required (e.g. epicycle frequencies) */
      /* -- to be overridden by derived classes -- */
      cout << "DistributionFunction::Prob() not defined properly!" << endl;
      exit(-1);
      return 0.5;
    }
};

/* ####################################################

   Basic functionality for a complete Galaxy model

   #################################################### */

class GalaxyModel
{
  /*
     The class "GalaxyModel" combines Potential, DF and the actionfinder.
     It can also be used as a PotentialBase and a Distribution function.
  */
protected:
  PotentialBase *Pot;
  Action_Finder *AF;
  DistributionFunction *DF;
  int vlimits(VecDoub_I &x, VecDoub_O &vmin, VecDoub_O &vmax) const;           // Returns escape velocity at each set of spatial coordinates
  double KzMoments(VecDoub_I&, double*,double*) const;
public:
  // Phase space position of the Sun in the model:
  double R0,z0;
  double Usun,Vsun,Wsun; // Solar velocity w.r.t. the local standard of rest
  double H0; // Hubble constant in unit km/s / kpc (e.g. 0.073)
  double rel_error_moments,abs_error_moments;
  bool use_octint;
  GalaxyModel(PotentialBase *inPot, Action_Finder *inAF, DistributionFunction *inDF, char *SolarParameters_Filename)
    : Pot(inPot), AF(inAF), DF(inDF),
      rel_error_moments(1e-2), abs_error_moments(0.0),use_octint(false)
  {
    std::ifstream from; from.open(SolarParameters_Filename);
    if (from.is_open()) {
      std::string tmp;
      from >> tmp >> R0;
      from >> tmp >> z0;
      from >> tmp >> Usun;
      from >> tmp >> Vsun;
      from >> tmp >> Wsun;
      from >> tmp >> H0;
      from.close();
    } else {
      std::cout << "Error in GalaxyModel:\n";
      std::cout << "Could not open file with solar phase space position:";
      std::cout << SolarParameters_Filename << endl;
      exit(-2);
    }
  }
  inline void resetDF(DistributionFunction *inDF) { DF = inDF; }

  inline double Phi(VecDoub_I &x) const { return Pot->Phi(x);}
  inline double dPhidx(VecDoub_I &x) const { return Pot->dPhidx(x);}
  inline double dPhidy(VecDoub_I &x) const { return Pot->dPhidy(x);}
  inline double dPhidz(VecDoub_I &x) const { return Pot->dPhidz(x);}
  inline double dPhidr(VecDoub_I &x) const { return Pot->dPhidr(x);}
  inline double dPhidR(VecDoub_I &x) const { return Pot->dPhidR(x);}
  inline VecDoub Forces(VecDoub_I &x) const { return Pot->Forces(x);}
  inline double density(VecDoub_I &x) const { return Pot->density(x);}
  inline double Vc(const double &R) const { return Pot->Vc(R);}
  inline double Prob(VecDoub_I &w, void *params = NULL) const{
    if (w.size()==3){
        return DF->Prob(w,params);
    }else{
        VecDoub J    = AF->actions(w);
        double prob = DF->Prob(J,params);
        if (std::isnan(prob)==true){  // Have this problem sometimes when actions are really large
            //print("Large actions...returning 0 for the DF...");
            return 0.;
        } else{
            return prob;
        }
    }
  }
  inline VecDoub actions(VecDoub_I &w) const{ return AF->actions(w); }

  inline VecDoub getSolarPosition() {
    VecDoub Pos(6);
    Pos[0] = R0; Pos[1] = z0;
    Pos[2] = Usun; Pos[3] = Vc(R0)+Vsun; Pos[4] = Wsun;
    Pos[5] = H0;
    return Pos;
  }

  double moments(VecDoub_I &x) const;                                          // Returns density (0th moment)
  double moments(VecDoub_I &x, double &Vbar) const;                           // Returns density and mean V velocity (=Vbar);
  double moments(VecDoub_I &x, VecDoub_O &SecondMoments) const;                // Returns density and second moments (velocity dispersions)
  double moments(VecDoub_I &x, double &Vbar, VecDoub_O &SecondMoments) const; // The two above combined...

  double Kz(VecDoub_I&) const;                                          // Return vertical force

};

/* ####################################################

   Basic functionality for a data set

   #################################################### */

class DataClass {
public:
  inline virtual double logLikelihood(GalaxyModel *Model) {
    /* returns log(Likelihood) of the data set given the Galaxy Model */
    /* -- to be overridden by derived classes -- */
    exit(-1);
  }
};


/* ####################################################

   Usefull auxiliary function

   #################################################### */

class CircularSpeed {
private:
  GalaxyModel *GM;
  PotentialBase *Pot;
  int flag;
public:
  CircularSpeed(GalaxyModel *inGM)
    : GM(inGM), flag(0) {}
  CircularSpeed(PotentialBase *inPot)
    : Pot(inPot),flag(1) {}
  double operator()(const double R) const {
    if (flag==0) return GM->Vc(R);
    else return Pot->Vc(R);
  }
};

VecDoub readSolarParameters(char fname[]);
// Reads and returns content of a file like "parfiles/SolarParameters".
#endif

#endif
