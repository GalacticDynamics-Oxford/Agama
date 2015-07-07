#ifndef STACKEL_JS_H
#define STACKEL_JS_H

//#include "AnalyticPotentials.hpp"
//#include "BaseClasses.hpp"
#include "coordsys.hpp"

// ============================================================================
// Axisymmetric Stackel Perfect Ellipsoid Potential
// ============================================================================
#if 0
struct root_struct_axi{
	StackelProlate_PerfectEllipsoid *P;
	VecDoub Ints;
	root_struct_axi(StackelProlate_PerfectEllipsoid *PP, VecDoub ints)
		:P(PP),Ints(ints){}
};

struct action_struct_axi{
	root_struct_axi RS;
	double taubargl, Deltagl;
	action_struct_axi(StackelProlate_PerfectEllipsoid *PP, VecDoub ints, double tb, double Dl)
		:RS(PP,ints),taubargl(tb),Deltagl(Dl){}
};

class Actions_AxisymmetricStackel : public Action_Finder{
	private:
		StackelProlate_PerfectEllipsoid *Pot;
		VecDoub find_limits(VecDoub_I x, VecDoub_I ints) const;
	public:
		Actions_AxisymmetricStackel(StackelProlate_PerfectEllipsoid *pot): Pot(pot){}
		VecDoub actions(VecDoub_I &x);
};

// ==========================================================================================
// Axisymmetric Stackel Fudge
// ==========================================================================================
#endif
class Actions_AxisymmetricStackel_Fudge{
private:
    const potential::BasePotential& poten;
    double E, I2;
    double Kt[2];
    VecDoub find_limits(const coord::PosVelProlSph& tau) const;
    void integrals(const coord::PosVelProlSph& tau);
public:
    const coord::ProlSph CS;
    Actions_AxisymmetricStackel_Fudge(const potential::BasePotential &pot,double a): 
            poten(pot), CS(a,-1) {}
		/*inline double Phi_tau(VecDoub_I tau) const{
			return Pot->Phi(CS->tau2x(tau));
		}
		inline double Phi_tau(double l, double n) const{
		    double tmp[3] = {l,0.,n};
		    VecDoub tau(3,tmp);
			return Pot->Phi(CS->tau2x(tau));
		}
		inline double chi_lam(VecDoub_I tau) const{return -(tau[0]-tau[2])*Phi_tau(tau);}
		inline double chi_nu(VecDoub_I tau) const{ return -(tau[2]-tau[0])*Phi_tau(tau);}*/
    double chi(double lambda, double nu) const {
        double phi;
        poten.eval(coord::toPosCyl(coord::PosProlSph(lambda, nu, 0, CS)), &phi);
        return phi;
    }

		VecDoub actions(const coord::PosVelCyl& point);
		//VecDoub angles(VecDoub_I x);
};

struct root_struct_axi_fudge{
	const Actions_AxisymmetricStackel_Fudge *ASF;
	VecDoub Ints;
	const coord::PosVelProlSph& tau_i;
	int swit;
	root_struct_axi_fudge(const Actions_AxisymmetricStackel_Fudge *ASF, VecDoub ints, 
                          const coord::PosVelProlSph& tau_i, int swit)
		:ASF(ASF),Ints(ints),tau_i(tau_i),swit(swit){}
};

struct action_struct_axi_fudge{
	const Actions_AxisymmetricStackel_Fudge *ASF;
	VecDoub Ints;
	const coord::PosVelProlSph& tau_i;
	double taubargl, Deltagl;
	int swit;
	action_struct_axi_fudge(const Actions_AxisymmetricStackel_Fudge *ASF, VecDoub ints, 
                            const coord::PosVelProlSph& tau_i, double tb, double Dl,int swit)
		:ASF(ASF),Ints(ints),tau_i(tau_i),taubargl(tb),Deltagl(Dl),swit(swit){}
};

// ==========================================================================================
// Triaxial Stackel Perfect Ellipsoid Potential
// ==========================================================================================
#if 0
struct root_struct_triax{
	StackelTriaxial *P;
	VecDoub Ints;
	root_struct_triax(StackelTriaxial *PP, VecDoub ints)
		:P(PP),Ints(ints){}
};

struct action_struct_triax{
	StackelTriaxial *P;
	VecDoub Ints;
	double taubargl, Deltagl;
	action_struct_triax(StackelTriaxial *PP, VecDoub ints, double tb, double Dl)
		:P(PP),Ints(ints),taubargl(tb),Deltagl(Dl){}
};

class Actions_TriaxialStackel : public Action_Finder{
	private:
		bool freq_yes;
		StackelTriaxial *Pot;
		VecDoub find_limits(VecDoub_I x,VecDoub_I ints) const;
	public:
		Actions_TriaxialStackel(StackelTriaxial *pot): Pot(pot){
			freq_yes = 0;
		};
		inline void set_freq_yes(bool s){freq_yes=s;}
		VecDoub actions(VecDoub_I &x0);
};

// ==========================================================================================
// Triaxial Stackel Fudge
// ==========================================================================================

class Actions_TriaxialStackel_Fudge : public Action_Finder{
	private:
		PotentialBase *Pot;
		double E;
		VecDoub Jt, Kt;
		VecDoub find_limits(VecDoub_I x) const;
		void integrals(VecDoub_I tau);
	public:
		ConfocalEllipsoidalCoordSys *CS;
		Actions_TriaxialStackel_Fudge(PotentialBase *pot,double a,double b): Pot(pot){
			CS = new ConfocalEllipsoidalCoordSys(a,b);
			Jt.resize(3);Kt.resize(3);
		};
		inline double Phi_tau(VecDoub_I tau) const{
			return Pot->Phi(CS->tau2x(tau));
		}
		inline double Phi_tau(double l, double m, double n) const{
		    double tmp[3] = {l,m,n};
			return Pot->Phi(CS->tau2x(VecDoub_I(3,tmp)));
		}
		inline double chi_lam(VecDoub_I tau) const{return (tau[0]-tau[1])*(tau[2]-tau[0])*Phi_tau(tau);}
		inline double chi_mu(VecDoub_I tau) const{ return (tau[1]-tau[2])*(tau[0]-tau[1])*Phi_tau(tau);}
		inline double chi_nu(VecDoub_I tau) const{ return (tau[2]-tau[0])*(tau[1]-tau[2])*Phi_tau(tau);}
		double ptau_tau(double tau, VecDoub_I ints, VecDoub_I tau_i, int swit) const;
		VecDoub actions(VecDoub_I &x);
		VecDoub angles(VecDoub_I x);
		double sos(VecDoub_I x, int comp,std::string outfile);
};


struct root_struct_triax_fudge{
	const Actions_TriaxialStackel_Fudge *ATSF;
	VecDoub Ints;
	VecDoub tau_i;
	int swit;
	root_struct_triax_fudge(const Actions_TriaxialStackel_Fudge *ATSF, VecDoub ints, VecDoub tau_i, int swit)
		:ATSF(ATSF),Ints(ints),tau_i(tau_i),swit(swit){}
};

struct action_struct_triax_fudge{
	const Actions_TriaxialStackel_Fudge *ATSF;
	VecDoub Ints;
	VecDoub tau_i;
	double taubargl, Deltagl;
	int swit;
	action_struct_triax_fudge(const Actions_TriaxialStackel_Fudge *ATSF, VecDoub ints, VecDoub tau_i,double tb, double Dl,int swit)
		:ATSF(ATSF),Ints(ints),tau_i(tau_i),taubargl(tb),Deltagl(Dl),swit(swit){}
};
#endif
#endif
// ==========================================================================================
