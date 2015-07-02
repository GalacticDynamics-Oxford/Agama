#include "actions_staeckel.h"

namespace actions{

    AxisymIntegrals findIntegralsOfMotionOblatePerfectEllipsoid
        (const potential::StaeckelOblatePerfectEllipsoid& pot, const coord::PosVelCyl& point)
    {
        AxisymIntegrals Ints;
        Ints.H = potential::totalEnergy(pot, point);
        Ints.Lz= coord::Lz(point);
        const coord::ProlSph& coordsys=pot.coordsys();
        coord::PosDerivT<coord::Cyl, coord::ProlSph> derivs;
        const coord::PosProlSph coords= coord::toPosDeriv<coord::Cyl, coord::ProlSph>
            (point, coordsys, &derivs);
        double lambdadot = derivs.dlambdadR*point.vR + derivs.dlambdadz*point.vz;
        double Glambda   = pot.eval_G(coords.lambda);
        Ints.I3 = (coords.lambda+coordsys.gamma) * 
            (Ints.H - pow_2(Ints.Lz)/2/(coords.lambda+coordsys.alpha) + Glambda) -
            pow_2(lambdadot*(coords.lambda-coords.nu)) / 
            (8*(coords.lambda+coordsys.alpha)*(coords.lambda+coordsys.gamma));
        return Ints;
    }

#if 0
// ============================================================================
// Oblate Stackel Angle-action calculator
// ============================================================================
struct root_struct_axi{
    potential::StackelProlate_PerfectEllipsoid *P;
    VecDoub Ints;
    root_struct_axi(potential::StackelProlate_PerfectEllipsoid *PP, VecDoub ints)
        :P(PP),Ints(ints){}
};

struct action_struct_axi{
    root_struct_axi RS;
    double taubargl, Deltagl;
    action_struct_axi(StackelProlate_PerfectEllipsoid *PP, VecDoub ints, double tb, double Dl)
        :RS(PP,ints),taubargl(tb),Deltagl(Dl){}
};

double ptau2ROOT_AxiSym(double tau, void *params) {
	/* for finding roots of p_tau^2*2.0*(tau+Alpha)  */
	root_struct_axi *RS = (root_struct_axi *) params;
	return (RS->Ints[0]-RS->Ints[1]/(tau+RS->P->alpha())-RS->Ints[2]/(tau+RS->P->gamma())+RS->P->G(tau));
	}

VecDoub Actions_AxisymmetricStackel::find_limits(VecDoub_I tau, VecDoub_I ints) const{

	double lambda = tau[0], nu = tau[2];
	// create a structure to store parameters for ptau2ROOT
	root_struct_axi RS(Pot,ints);
	// find roots of p^2(lam)
	double laminner=lambda;
	while(ptau2ROOT_AxiSym(laminner, &RS)>0.0)	laminner-=.1*(laminner+Pot->alpha());
	double lamouter=lambda;
	while(ptau2ROOT_AxiSym(lamouter, &RS)>0.)	lamouter*=1.1;

	root_find RF(SMALL,100);
	VecDoub limits(4);
	limits[0] = (RF.findroot(&ptau2ROOT_AxiSym,laminner,lambda,&RS));
	limits[1] = (RF.findroot(&ptau2ROOT_AxiSym,lambda,lamouter,&RS));
	limits[2] = (-Pot->gamma()+TINY);
	// find root of p^2(nu)
	double nuouter=nu;
	while(ptau2ROOT_AxiSym(nuouter, &RS)<0.)	nuouter+=0.1*(-Pot->alpha()-nuouter);
	limits[3] = (RF.findroot(&ptau2ROOT_AxiSym,nu,nuouter,&RS));
	return limits;
}

double ptau2_AxiSym(double tau, void *params) {
	//p^2(tau) using global integrals
	root_struct_axi *RS = (root_struct_axi *) params;
	return (RS->Ints[0]-RS->Ints[1]/(tau+RS->P->alpha())-RS->Ints[2]/(tau+RS->P->gamma())+RS->P->G(tau))
			/(2.0*(tau+RS->P->alpha()));
}

double J_integrand_AxiSym(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_axi * AS = (action_struct_axi *) params;
	double tau=AS->taubargl+AS->Deltagl*sin(theta);
	return sqrt(MAX(0.,ptau2_AxiSym(tau,&(AS->RS))))*cos(theta);
}

Actions Actions_AxisymmetricStackel::actions(const coord::PosVelCar& point) const {
	VecDoub   tau       = Pot->xv2tau(x);
	VecDoub integrals   = Pot->x2ints(x,&tau);
	VecDoub limits      = find_limits(tau,integrals);
	Actions actions;
	// JR
	double taubar      = 0.5*(limits[0]+limits[1]);
	double Delta       = 0.5*(limits[1]-limits[0]);
	action_struct_axi AS(Pot,integrals,taubar,Delta);
	actions.Jr          = (Delta*GaussLegendreQuad(&J_integrand_AxiSym,-.5*PI,.5*PI,&AS)/PI);
	// Lz
	actions.Jphi        = (sqrt(2.0*integrals[1]));
	// Jz
	taubar              = 0.5*(limits[2]+limits[3]);
	Delta               = 0.5*(limits[3]-limits[2]);
	AS                  = action_struct_axi (Pot,integrals,taubar,Delta);
	actions.Jz          = (2.*Delta*GaussLegendreQuad(&J_integrand_AxiSym,-.5*PI,.5*PI,&AS)/PI);
	return actions;
}
#endif

}  // namespace actions
