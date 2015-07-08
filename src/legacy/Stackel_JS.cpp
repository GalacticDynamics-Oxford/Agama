/* ####################################################

   Basic functionality for a object converting phase
   space coordinates to actions (and angles)

#################################################### */


#include "BaseClasses.hpp"
#include "Stackel_JS.hpp"
#include "GSLInterface.hpp"

#if 0
// ============================================================================
// Prolate Stackel Angle-action calculator
// ============================================================================

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

VecDoub Actions_AxisymmetricStackel::actions(VecDoub_I &x) {
	VecDoub   tau       = Pot->xv2tau(x);
	VecDoub integrals   = Pot->x2ints(x,&tau);
	VecDoub limits      = find_limits(tau,integrals);
	VecDoub actions(3);
	// JR
	double taubar      = 0.5*(limits[0]+limits[1]);
	double Delta       = 0.5*(limits[1]-limits[0]);
	action_struct_axi AS(Pot,integrals,taubar,Delta);
	actions[0]          = (Delta*GaussLegendreQuad(&J_integrand_AxiSym,-.5*PI,.5*PI,&AS)/PI);
	// Lz
	actions[1]          = (sqrt(2.0*integrals[1]));
	// Jz
	taubar              = 0.5*(limits[2]+limits[3]);
	Delta               = 0.5*(limits[3]-limits[2]);
	AS                  = action_struct_axi (Pot,integrals,taubar,Delta);
	actions[2]          = (2.*Delta*GaussLegendreQuad(&J_integrand_AxiSym,-.5*PI,.5*PI,&AS)/PI);
	return actions;
}
#endif
// ============================================================================
// Prolate Stackel Angle-action Fudge
// ============================================================================

double ptau2ROOT_AxiSym_Fudge(double tau, void *params) {
	/* for finding roots of p_tau^2*2.0*(tau+Alpha)*(tau+Gamma)  */
	root_struct_axi_fudge *RS = (root_struct_axi_fudge *) params;
	double phi = 0.;
    if(RS->swit==0)	{
        phi = -(tau-RS->tau_i.nu) * RS->ASF->chi(tau, RS->tau_i.nu);
	}else if (RS->swit==1) {
        phi = -(tau-RS->tau_i.lambda) * RS->ASF->chi(RS->tau_i.lambda, tau);
	}
	double Alpha = RS->ASF->CS.alpha, Gamma = RS->ASF->CS.gamma;
	return (RS->Ints[0]*(tau+Gamma)-RS->Ints[1]*(tau+Gamma)/(tau+Alpha)-RS->Ints[2]+phi);
}

void Actions_AxisymmetricStackel_Fudge::integrals(const coord::PosVelProlSph& tau) {

	double a   = CS.alpha, c = CS.gamma;

	double Pl2 = (tau.lambda-tau.nu)/((a+tau.lambda)*(c+tau.lambda))/4.;
	double Pn2 = (tau.nu-tau.lambda)/((a+tau.nu)*(c+tau.nu))/4.;

	double pl  = tau.lambdadot*Pl2; pl*=pl;
	double pn  = tau.nudot*Pn2; pn*=pn;
    double Phi = (tau.nu-tau.lambda)*chi(tau.lambda, tau.nu);
	Kt[0]       = -2.*(a+tau.lambda)*(c+tau.lambda)*pl+(tau.lambda+c)*E-(tau.lambda+c)/(tau.lambda+a)*I2 + Phi;
	Kt[1]       = -2.*(a+tau.nu)*(c+tau.nu)*pn+(tau.nu+c)*E-(tau.nu+c)/(tau.nu+a)*I2 - Phi;
}

VecDoub Actions_AxisymmetricStackel_Fudge::find_limits(const coord::PosVelProlSph& tau) const{

	double lambda = tau.lambda, nu = tau.nu;
	VecDoub limits(4);
	// create a structure to store parameters for ptau2ROOT
	double tmp[3] = {E,I2,Kt[0]};
	VecDoub ints(3,tmp);
	root_struct_axi_fudge RS_l(this,ints,tau,0);

	// find roots of p^2(lam)
    int niter=0;
	double laminner=lambda;
	while(ptau2ROOT_AxiSym_Fudge(laminner, &RS_l)>0.0 && niter<1000) { niter++;	laminner-=.1*(laminner+CS.alpha); }
	double lamouter=lambda;
    niter=0;
	while(ptau2ROOT_AxiSym_Fudge(lamouter, &RS_l)>0. && niter<1000) { niter++;	lamouter*=1.1; }

	root_find RF(SMALL,100);
	limits[0] = (RF.findroot(&ptau2ROOT_AxiSym_Fudge,laminner,lambda,&RS_l));
	limits[1] = (RF.findroot(&ptau2ROOT_AxiSym_Fudge,lambda,lamouter,&RS_l));

	limits[2] = (-CS.gamma+TINY);
	// find root of p^2(nu)
	double nuouter=nu;
	ints[2]   = Kt[1];
	root_struct_axi_fudge RS_n(this,ints,tau,1);
    niter=0;
	while(ptau2ROOT_AxiSym_Fudge(nuouter, &RS_n)<0.  && niter<100) { niter++;	nuouter+=0.1*(-CS.alpha-nuouter); }
	limits[3] = (RF.findroot(&ptau2ROOT_AxiSym_Fudge,nu,nuouter,&RS_n));
	return limits;
}

double J_integrand_AxiSym_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_axi_fudge * AS = (action_struct_axi_fudge *) params;
	double tau        = AS->taubargl+AS->Deltagl*sin(theta);
	double phi        = 0.;
	if(AS->swit==0)	{
        phi = -(tau-AS->tau_i.nu) * AS->ASF->chi(tau, AS->tau_i.nu);
	}else if (AS->swit==1) {
        phi = -(tau-AS->tau_i.lambda) * AS->ASF->chi(AS->tau_i.lambda, tau);
	}
	double Alpha      = AS->ASF->CS.alpha, Gamma = AS->ASF->CS.gamma;
	double ptau2      = AS->Ints[0]*(tau+Gamma)-AS->Ints[1]*(tau+Gamma)/(tau+Alpha)-AS->Ints[2]+phi;
	ptau2             /= (tau+Alpha)*(tau+Gamma)*2.;
	return sqrt(MAX(0.,ptau2))*cos(theta);
}
#if 0
double dJdE_integrand_AxiSym_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_axi_fudge * AS = (action_struct_axi_fudge *) params;
	double tau    = AS->taubargl+AS->Deltagl*sin(theta);
	double phi    = 0.;
	if(AS->swit==0){
        double tmp[3] = {tau,0.,AS->tau_i[2]};
        phi = AS->ASF->chi_lam(VecDoub(3,tmp));
    } else if (AS->swit==1){
        double tmp[3] = {AS->tau_i[0],0.,tau};
        phi = AS->ASF->chi_nu(VecDoub(3,tmp));
    }
	double Alpha  = AS->ASF->CS.alpha, Gamma = AS->ASF->CS.gamma;
	double ptau2  = AS->Ints[0]*(tau+Gamma)-AS->Ints[1]*(tau+Gamma)/(tau+Alpha)-AS->Ints[2]+phi;
	ptau2 /= (tau+Alpha)*(tau+Gamma)*2.;
	return 0.25*cos(theta)/(sqrt(MAX(SMALL,ptau2))*(tau+Alpha));
}

double dJdI2_integrand_AxiSym_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_axi_fudge * AS = (action_struct_axi_fudge *) params;
	double tau    = AS->taubargl+AS->Deltagl*sin(theta);
	double phi    = 0.;
	if(AS->swit==0)	{
        double tmp[3] = {tau,0.,AS->tau_i[2]};
	    phi = AS->ASF->chi_lam(VecDoub(3,tmp));
	} else if(AS->swit==1) {
        double tmp[3] = {AS->tau_i[0],0.,tau};
		phi = AS->ASF->chi_nu(VecDoub(3,tmp));
	}
	double Alpha  = AS->ASF->CS.alpha, Gamma = AS->ASF->CS.gamma;
	double ptau2  = AS->Ints[0]*(tau+Gamma)-AS->Ints[1]*(tau+Gamma)/(tau+Alpha)-AS->Ints[2]+phi;
	ptau2 /= (tau+Alpha)*(tau+Gamma)*2.;
	return -0.25*cos(theta)/(sqrt(MAX(SMALL,ptau2))*(tau+Alpha)*(tau+Alpha));
}

double dJdI3_integrand_AxiSym_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_axi_fudge * AS = (action_struct_axi_fudge *) params;
	double tau    = AS->taubargl+AS->Deltagl*sin(theta);
	double phi    = 0.;
	if(AS->swit==0) {
        double tmp[3] = {tau,0.,AS->tau_i[2]};
        phi            = AS->ASF->chi_lam(VecDoub(3,tmp));
	} else if(AS->swit==1){
        double tmp[3] = {AS->tau_i[0],0.,tau};
        phi            = AS->ASF->chi_nu(VecDoub(3,tmp));
	}
	double Alpha = AS->ASF->CS.alpha, Gamma = AS->ASF->CS.gamma;
	double ptau2 = AS->Ints[0]*(tau+Gamma)-AS->Ints[1]*(tau+Gamma)/(tau+Alpha)-AS->Ints[2]+phi;
	ptau2 /= (tau+Alpha)*(tau+Gamma)*2.;
	return -0.25*cos(theta)/(sqrt(MAX(SMALL,ptau2))*(tau+Alpha)*(tau+Gamma));
}
#endif
VecDoub Actions_AxisymmetricStackel_Fudge::actions(const coord::PosVelCyl& point) {
    const coord::PosVelProlSph tau = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, CS);
	E = potential::totalEnergy(poten, point); 
    I2 = 0.5*pow_2(coord::Lz(point));
	integrals(tau);
	VecDoub limits    = find_limits(tau);
    /*std::cout << "lam="<<tau.lambda<<", nu="<<tau.nu<<"; E="<<E<<", I2="<<I2<<
    ", Kt0="<<Kt[0]<<", Kt1="<<Kt[1]<<"; limits="
    <<limits[0]<<":"<<limits[1]<<", "<<limits[2]<<":"<<limits[3]<<"\n";*/
	VecDoub acts(3);
	// JR
	double taubar    = 0.5*(limits[0]+limits[1]);
	double Delta     = 0.5*(limits[1]-limits[0]);
	double tmp[3]    = {E,I2,Kt[0]};
	VecDoub ints(3,tmp);
	action_struct_axi_fudge AS1(this,ints,tau,taubar,Delta,0);
	acts[0]        = (Delta*GaussLegendreQuad(&J_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS1)/PI);
	// Lz
	acts[1]        = (sqrt(2.0*I2));
	// Jz
	taubar            = 0.5*(limits[2]+limits[3]);
	Delta             = 0.5*(limits[3]-limits[2]);
	ints[2]           = Kt[1];
    action_struct_axi_fudge AS2(this,ints,tau,taubar,Delta,1);
	acts[2]        = (2.*Delta*GaussLegendreQuad(&J_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS2)/PI);
	return acts;
}
#if 0
VecDoub Actions_AxisymmetricStackel_Fudge::angles(VecDoub_I x){
	VecDoub   tau    = CS->xv2tau(x);
	E                = Pot->H(x); I2 = 0.5*(x[0]*x[4]-x[1]*x[3])*(x[0]*x[4]-x[1]*x[3]);
	integrals(tau);
	VecDoub limits   = find_limits(tau);
	VecDoub angles(6);
	// thetaR
	double taubar   = 0.5*(limits[0]+limits[1]);
	double Delta    = 0.5*(limits[1]-limits[0]);
	double tmp1[3]  = {E,I2,Kt[0]};
	VecDoub ints(3,tmp1);
	action_struct_axi_fudge AS(this,ints,tau,taubar,Delta,0);
	VecDoub dJdIl(3,0.);
	dJdIl[0]         = Delta*GaussLegendreQuad(&dJdE_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIl[1]         = Delta*GaussLegendreQuad(&dJdI2_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIl[2]         = Delta*GaussLegendreQuad(&dJdI3_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	VecDoub dSdI(3,0.);
	double thetaLim = asin(MAX(-1.,MIN(1.,(tau.lambda-taubar)/Delta)));
	double lsign    = SIGN_JS(tau[3]);
	dSdI[0]         += lsign*Delta*GaussLegendreQuad(&dJdE_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[1]         += lsign*Delta*GaussLegendreQuad(&dJdI2_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[2]         += lsign*Delta*GaussLegendreQuad(&dJdI3_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	// Lz
	double tmp2[3]  = {0.,1./sqrt(2.*I2),0.};
	VecDoub dJdIp(3,tmp2);
	dSdI[1]         +=SIGN_JS(tau[4])*tau[1]/sqrt(2.*I2);
	// Jz
	taubar           = 0.5*(limits[2]+limits[3]);
	Delta            = 0.5*(limits[3]-limits[2]);
	ints[2]          = Kt[1];
	AS               = action_struct_axi_fudge(this,ints,tau,taubar,Delta,1);
	VecDoub dJdIn(3,0.);
	dJdIn[0]         = 2.*Delta*GaussLegendreQuad(&dJdE_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIn[1]         = 2.*Delta*GaussLegendreQuad(&dJdI2_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIn[2]         = 2.*Delta*GaussLegendreQuad(&dJdI3_integrand_AxiSym_Fudge,-.5*PI,.5*PI,&AS)/PI;
	thetaLim         = asin(MAX(-1.,MIN(1.,(tau.nu-taubar)/Delta)));
	lsign            = SIGN_JS(tau[5]);
	dSdI[0]         += lsign*Delta*GaussLegendreQuad(&dJdE_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[1]         += lsign*Delta*GaussLegendreQuad(&dJdI2_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[2]         += lsign*Delta*GaussLegendreQuad(&dJdI3_integrand_AxiSym_Fudge,-.5*PI,thetaLim,&AS);
	double Determinant = dJdIp[1]*(dJdIl[0]*dJdIn[2]-dJdIl[2]*dJdIn[0]);

	double dIdJ[3][3];
	dIdJ[0][0]       = det2(dJdIp[1],dJdIp[2],dJdIn[1],dJdIn[2])/Determinant;
	dIdJ[0][1]       = -det2(dJdIl[1],dJdIl[2],dJdIn[1],dJdIn[2])/Determinant;
	dIdJ[0][2]       = det2(dJdIl[1],dJdIl[2],dJdIp[1],dJdIp[2])/Determinant;
	dIdJ[1][0]       = -det2(dJdIp[0],dJdIp[2],dJdIn[0],dJdIn[2])/Determinant;
	dIdJ[1][1]       = det2(dJdIl[0],dJdIl[2],dJdIn[0],dJdIn[2])/Determinant;
	dIdJ[1][2]       = -det2(dJdIl[0],dJdIl[2],dJdIp[0],dJdIp[2])/Determinant;
	dIdJ[2][0]       = det2(dJdIp[0],dJdIp[1],dJdIn[0],dJdIn[1])/Determinant;
	dIdJ[2][1]       = -det2(dJdIl[0],dJdIl[1],dJdIn[0],dJdIn[1])/Determinant;
	dIdJ[2][2]       = det2(dJdIl[0],dJdIl[1],dJdIp[0],dJdIp[1])/Determinant;

	angles[0] = (dSdI[0]*dIdJ[0][0]+dSdI[1]*dIdJ[1][0]+dSdI[2]*dIdJ[2][0]);
	angles[1] = (dSdI[0]*dIdJ[0][1]+dSdI[1]*dIdJ[1][1]+dSdI[2]*dIdJ[2][1]);
	angles[2] = (dSdI[0]*dIdJ[0][2]+dSdI[1]*dIdJ[1][2]+dSdI[2]*dIdJ[2][2]);
	angles[3] = (det2(dJdIp[1],dJdIp[2],dJdIn[1],dJdIn[2])/Determinant);
	angles[4] = (det2(dJdIn[1],dJdIn[2],dJdIl[1],dJdIl[2])/Determinant);
	angles[5] = (det2(dJdIl[1],dJdIl[2],dJdIp[1],dJdIp[2])/Determinant);

	if(tau[5]<0.0){angles[2]+=PI;}
	if(x[2]<0.0){angles[2]+=PI;}
	if(angles[2]>2.*PI)	angles[2]-=2.0*PI;
	if(angles[0]<0.0)	angles[0]+=2.0*PI;
	if(angles[1]<0.0)	angles[1]+=2.0*PI;
	if(angles[2]<0.0)	angles[2]+=2.0*PI;

	return angles;
}
#endif
// ============================================================================
// Triaxial Stackel Angle-action calculator
// ============================================================================
#if 0
double ptau2ROOT_Triax(double tau, void *params) {
	/* for finding roots of p_tau^2*2.0*(tau+Alpha)  */
	root_struct_triax *RS = (root_struct_triax *) params;
	return (RS->Ints[0]-RS->Ints[1]/(tau+RS->P->alpha())-RS->Ints[2]/(tau+RS->P->gamma())+RS->P->G(tau))
			/(2.0*(tau+RS->P->beta()));
	}

VecDoub Actions_TriaxialStackel::find_limits(VecDoub_I tau,VecDoub_I ints)const{

	double lambda = tau[0], mu=tau[1], nu = tau[2];
	root_find RF(SMALL,100);
	VecDoub limits(6);
	// create a structure to store parameters for ptau2ROOT
	root_struct_triax RS(Pot,ints);

	// find roots of p^2(lam)
	double laminner=lambda, lamouter=lambda;
	while(ptau2ROOT_Triax(laminner, &RS)>0.0 and (laminner+Pot->alpha())>SMALL)	laminner-=.1*(laminner+Pot->alpha());
	if((laminner+Pot->alpha())>SMALL) limits[0] = (RF.findroot(&ptau2ROOT_Triax,laminner,lambda,&RS));
	else limits[0] = (-Pot->alpha());
	while(ptau2ROOT_Triax(lamouter, &RS)>0.)	lamouter*=1.1;
	limits[1] = (RF.findroot(&ptau2ROOT_Triax,lambda,lamouter,&RS));

	// find root of p^2(mu)
	double muinner=mu, muouter=mu;
	while(ptau2ROOT_Triax(muinner, &RS)>0. and (muinner+Pot->beta())>SMALL)	muinner-=.1*(muinner+Pot->beta());
	if((muinner+Pot->beta())>SMALL) limits[2] = (RF.findroot(&ptau2ROOT_Triax,muinner,mu,&RS));
	else limits[2] = (-Pot->beta());
	while(ptau2ROOT_Triax(muouter, &RS)>0. and (muouter+Pot->alpha())<-SMALL)	muouter+=0.1*(-Pot->alpha()-muouter);
	if((muouter+Pot->alpha())<-SMALL) limits[3] = (RF.findroot(&ptau2ROOT_Triax,mu,muouter,&RS));
	else limits[3] = (-Pot->alpha());

	// find root of p^2(nu)
	double nuinner=nu, nuouter=nu;
	while(ptau2ROOT_Triax(nuinner, &RS)>0. and (nuinner+Pot->gamma())>SMALL)	nuinner-=.1*(nuinner+Pot->gamma());
	if((nuinner+Pot->gamma())>SMALL) limits[4] = (RF.findroot(&ptau2ROOT_Triax,nuinner,nu,&RS));
	else limits[4] = (-Pot->gamma());
	while(ptau2ROOT_Triax(nuouter, &RS)>0. and (nuouter+Pot->beta())<-SMALL)	nuouter+=0.1*(-Pot->beta()-nuouter);
	if((nuouter+Pot->beta())<-SMALL) limits[5] = (RF.findroot(&ptau2ROOT_Triax,nu,nuouter,&RS));
	else limits[5] = (-Pot->beta());

	return limits;
}

double J_integrand_Triax(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax * AS = (action_struct_triax *) params;
	double tau  = AS->taubargl+AS->Deltagl*sin(theta);
	double ptau = ((tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->Ints[0]
			-AS->Ints[1]*(tau+AS->P->gamma())-AS->Ints[2]*(tau+AS->P->alpha())
			+(tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->P->G(tau))
			/(2.0*(tau+AS->P->alpha())*(tau+AS->P->beta())*(tau+AS->P->gamma()));
	return sqrt(MAX(0.,ptau))*cos(theta);
}

double dJdH_integrand_Triax(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax * AS = (action_struct_triax *) params;
	double tau  = AS->taubargl+AS->Deltagl*sin(theta);
	double ptau = ((tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->Ints[0]
			-AS->Ints[1]*(tau+AS->P->gamma())-AS->Ints[2]*(tau+AS->P->alpha())
			+(tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->P->G(tau))
			/(2.0*(tau+AS->P->alpha())*(tau+AS->P->beta())*(tau+AS->P->gamma()));
	return sqrt(MAX(0.,1./ptau))*cos(theta)/(tau+AS->P->beta());
}

double dJdI2_integrand_Triax(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax * AS = (action_struct_triax *) params;
	double tau  = AS->taubargl+AS->Deltagl*sin(theta);
	double ptau = ((tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->Ints[0]
			-AS->Ints[1]*(tau+AS->P->gamma())-AS->Ints[2]*(tau+AS->P->alpha())
			+(tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->P->G(tau))
			/(2.0*(tau+AS->P->alpha())*(tau+AS->P->beta())*(tau+AS->P->gamma()));
	return -sqrt(MAX(0.,1./ptau))*cos(theta)/(tau+AS->P->beta())/(tau+AS->P->alpha());
}

double dJdI3_integrand_Triax(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax * AS = (action_struct_triax *) params;
	double tau  = AS->taubargl+AS->Deltagl*sin(theta);
	double ptau = ((tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->Ints[0]
			-AS->Ints[1]*(tau+AS->P->gamma())-AS->Ints[2]*(tau+AS->P->alpha())
			+(tau+AS->P->alpha())*(tau+AS->P->gamma())*AS->P->G(tau))
			/(2.0*(tau+AS->P->alpha())*(tau+AS->P->beta())*(tau+AS->P->gamma()));
	return -sqrt(MAX(0.,1./ptau))*cos(theta)/(tau+AS->P->beta())/(tau+AS->P->gamma());
}

VecDoub Actions_TriaxialStackel::actions(VecDoub_I &x)  {
	VecDoub tau       = Pot->xv2tau(x);

	VecDoub integrals = Pot->tau2ints(tau);
	VecDoub limits    = find_limits(tau,integrals);
	VecDoub actions(6), freqs(9);

	// We need to check which coordinates are oscillating and which are circulating
	// and multiply by the appropriate factor
	double tmp[3] = {1.,1.,1.};
	VecDoub circ(3,tmp);
	if(limits[0]==-Pot->alpha()) circ[0] = 1.; else circ[0] = 0.5;
	// if(limits[2]==-Pot->beta() and limits[3]==-Pot->alpha()) circ[1] = 1.; else circ[1] = 0.5;
	// if(limits[4]==-Pot->gamma() and limits[5]==-Pot->beta()) circ[2] = 0.5; else circ[2] = 1.;

	// JR
	double taubar      = 0.5*(limits[0]+limits[1]);
	double Delta       = 0.5*(limits[1]-limits[0]);
	action_struct_triax AS(Pot,integrals,taubar,Delta);
	actions[0] = (2.*circ[0]*Delta*GaussLegendreQuad(&J_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);

	if(freq_yes){
		freqs[0] = (0.5*circ[0]*Delta*GaussLegendreQuad(&dJdH_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[1] = (0.5*circ[0]*Delta*GaussLegendreQuad(&dJdI2_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[2] = (0.5*circ[0]*Delta*GaussLegendreQuad(&dJdI3_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
	}
	// return actions;

	// Jp
	taubar       = 0.5*(limits[2]+limits[3]);
	Delta        = 0.5*(limits[3]-limits[2]);
	AS           = action_struct_triax(Pot,integrals,taubar,Delta);
	actions[1]   = (2.*circ[1]*Delta*GaussLegendreQuad(&J_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);

	if(freq_yes){
		freqs[3] = (0.5*circ[1]*Delta*GaussLegendreQuad(&dJdH_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[4] = (0.5*circ[1]*Delta*GaussLegendreQuad(&dJdI2_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[5] = (0.5*circ[1]*Delta*GaussLegendreQuad(&dJdI3_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
	}

	// Jz
	taubar       = 0.5*(limits[4]+limits[5]);
	Delta        = 0.5*(limits[5]-limits[4]);
	AS           = action_struct_triax(Pot,integrals,taubar,Delta);
	actions[2]   = (2.*circ[2]*Delta*GaussLegendreQuad(&J_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);

	if(freq_yes){
		freqs[6] = (0.5*circ[2]*Delta*GaussLegendreQuad(&dJdH_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[7] = (0.5*circ[2]*Delta*GaussLegendreQuad(&dJdI2_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		freqs[8] = (0.5*circ[2]*Delta*GaussLegendreQuad(&dJdI3_integrand_Triax,-.5*PI,.5*PI,&AS)/PI);
		double det = freqs[0]*(freqs[4]*freqs[8]-freqs[5]*freqs[7])
					-freqs[1]*(freqs[3]*freqs[8]-freqs[5]*freqs[6])
					+freqs[2]*(freqs[3]*freqs[7]-freqs[4]*freqs[6]);
        actions.resize(6);
		actions[3] = ((freqs[4]*freqs[8]-freqs[5]*freqs[7])/det);
		actions[4] = ((freqs[7]*freqs[2]-freqs[8]*freqs[1])/det);
		actions[5] = ((freqs[1]*freqs[5]-freqs[2]*freqs[4])/det);
	}

	return actions;
}

// ============================================================================
// Triaxial Stackel Fudge
// ============================================================================

double ptau2ROOT_Triax_Fudge(double tau, void *params) {
	/* for finding roots of p_tau^2*2.0*(tau+Alpha)*(tau+Beta)*(tau+Gamma)  */
	root_struct_triax_fudge *RS = (root_struct_triax_fudge *) params;
	double phi = 0.;
	if(RS->swit==0){
	    double tmp[3] = {tau,RS->tau_i[1],RS->tau_i[2]};
        phi = RS->ATSF->chi_lam(VecDoub(3,tmp));
	} else if (RS->swit==1){
	    double tmp[3] = {RS->tau_i[0],tau,RS->tau_i[2]};
	    phi = RS->ATSF->chi_mu(VecDoub(3,tmp));
	} else if(RS->swit==2){
	    double tmp[3] = {RS->tau_i[0],RS->tau_i[1],tau};
	    phi = RS->ATSF->chi_nu(VecDoub(3,tmp));
	}
	// std::cout<<RS->Ints[0]<<" "<<tau<<" "<<RS->Ints[1]<<" "<<RS->Ints[2]<<" "<<phi<<std::endl;
	double p = (RS->Ints[0]*tau*tau-RS->Ints[1]*tau+RS->Ints[2]+phi);
	return p;//(RS->Ints[0]*tau*tau-RS->Ints[1]*tau+RS->Ints[2]+phi);
	}

void Actions_TriaxialStackel_Fudge::integrals(VecDoub_I tau){

	double a   = CS->alpha(), b = CS->beta(), c = CS->gamma();

	double Pl2 = (tau[0]-tau[1])*(tau[0]-tau[2])/((a+tau[0])*(b+tau[0])*(c+tau[0]))/4.;
	double Pm2 = (tau[1]-tau[2])*(tau[1]-tau[0])/((a+tau[1])*(b+tau[1])*(c+tau[1]))/4.;
	double Pn2 = (tau[2]-tau[1])*(tau[2]-tau[0])/((a+tau[2])*(b+tau[2])*(c+tau[2]))/4.;

	double pl  = tau[3]*Pl2; pl*=pl;
	double pm  = tau[4]*Pm2; pm*=pm;
	double pn  = tau[5]*Pn2; pn*=pn;

	Jt[0]       = (tau[1]+tau[2])*E+0.5*pm/Pm2*(tau[0]-tau[1])+0.5*pn/Pn2*(tau[0]-tau[2]);
	Jt[1]       = (tau[0]+tau[2])*E+0.5*pl/Pl2*(tau[1]-tau[0])+0.5*pn/Pn2*(tau[1]-tau[2]);
	Jt[2]       = (tau[0]+tau[1])*E+0.5*pl/Pl2*(tau[2]-tau[0])+0.5*pm/Pm2*(tau[2]-tau[1]);
	Kt[0]       = 2.*(a+tau[0])*(b+tau[0])*(c+tau[0])*pl-tau[0]*tau[0]*E+tau[0]*Jt[0]-chi_lam(tau);
	Kt[1]       = 2.*(a+tau[1])*(b+tau[1])*(c+tau[1])*pm-tau[1]*tau[1]*E+tau[1]*Jt[1]-chi_mu(tau);
	Kt[2]       = 2.*(a+tau[2])*(b+tau[2])*(c+tau[2])*pn-tau[2]*tau[2]*E+tau[2]*Jt[2]-chi_nu(tau);
}

VecDoub Actions_TriaxialStackel_Fudge::find_limits(VecDoub_I tau) const{

	double lambda = tau[0], mu=tau[1], nu = tau[2];
	root_find RF(1e-3,100);
	VecDoub limits(6);
	double tmp[3] = {E,Jt[0],Kt[0]};
	VecDoub ints(3,tmp);
	root_struct_triax_fudge RS_l(this,ints,tau,0);

	// find roots of p^2(lam)
	double laminner=lambda, lamouter=lambda;
	if(ptau2ROOT_Triax_Fudge(lambda, &RS_l)>0.0){
	while(ptau2ROOT_Triax_Fudge(laminner, &RS_l)>0.0 and (laminner+CS->alpha())>SMALL)	laminner-=.1*(laminner+CS->alpha());
	if((laminner+CS->alpha())>SMALL) limits[0] = (RF.findroot(&ptau2ROOT_Triax_Fudge,laminner,lambda,&RS_l));
	else limits[0] = (-CS->alpha());
	while(ptau2ROOT_Triax_Fudge(lamouter, &RS_l)>0.)	lamouter*=1.1;
	limits[1] = (RF.findroot(&ptau2ROOT_Triax_Fudge,lambda,lamouter,&RS_l));
	}
	else{ limits[0] = (lambda-TINY);limits[1] = (lambda+TINY);}

	// find root of p^2(mu)
	ints[1]=Jt[1];ints[2]=Kt[1];
	root_struct_triax_fudge RS_m(this,ints,tau,1);
	double muinner=mu, muouter=mu;
	if(ptau2ROOT_Triax_Fudge(mu, &RS_m)<0.0){
	while(ptau2ROOT_Triax_Fudge(muinner, &RS_m)<0. and (muinner+CS->beta())>SMALL)	muinner-=.1*(muinner+CS->beta());
	if((muinner+CS->beta())>SMALL) limits[2] = (RF.findroot(&ptau2ROOT_Triax_Fudge,muinner,mu,&RS_m));
	else limits[2] = (-CS->beta());
	while(ptau2ROOT_Triax_Fudge(muouter, &RS_m)<0. and (muouter+CS->alpha())<-SMALL)	muouter+=0.1*(-CS->alpha()-muouter);
	if((muouter+CS->alpha())<-SMALL) limits[3] = (RF.findroot(&ptau2ROOT_Triax_Fudge,mu,muouter,&RS_m));
	else limits[3] = (-CS->alpha());
	}
	else{ limits[2] = (mu-TINY);limits[3] = (mu+TINY);}

	// find root of p^2(nu)
	ints[1]=Jt[2];ints[2]=Kt[2];
	root_struct_triax_fudge RS_n(this,ints,tau,2);
	double nuinner=nu, nuouter=nu;
	if(ptau2ROOT_Triax_Fudge(nu, &RS_n)>0.0){
	while(ptau2ROOT_Triax_Fudge(nuinner, &RS_n)>0. and (nuinner+CS->gamma())>SMALL)	nuinner-=.1*(nuinner+CS->gamma());
	if((nuinner+CS->gamma())>SMALL) limits[4] = (RF.findroot(&ptau2ROOT_Triax_Fudge,nuinner,nu,&RS_n));
	else limits[4] = (-CS->gamma());
	while(ptau2ROOT_Triax_Fudge(nuouter, &RS_n)>0. and (nuouter+CS->beta())<-SMALL)	nuouter+=0.1*(-CS->beta()-nuouter);
	if((nuouter+CS->beta())<-SMALL) limits[5] = (RF.findroot(&ptau2ROOT_Triax_Fudge,nu,nuouter,&RS_n));
	else limits[5] = (-CS->beta());
	}
	else{ limits[4] = (nu-TINY);limits[5] = (nu+TINY);}

	return limits;
}

double J_integrand_Triax_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax_fudge * AS = (action_struct_triax_fudge *) params;
	double tau = AS->taubargl+AS->Deltagl*sin(theta);
	double a   = AS->ATSF->CS->alpha(), b = AS->ATSF->CS->beta(), c = AS->ATSF->CS->gamma();
	double phi = 0.;
	if(AS->swit==0){
	    double tmp[3] = {tau,AS->tau_i[1],AS->tau_i[2]};
	    phi = AS->ATSF->chi_lam(VecDoub(3,tmp));
	} else if(AS->swit==1){
	    double tmp[3] = {AS->tau_i[0],tau,AS->tau_i[2]};
	    phi = AS->ATSF->chi_mu(VecDoub(3,tmp));
	} else if(AS->swit==2)	{
	    double tmp[3] = {AS->tau_i[0],AS->tau_i[1],tau};
	    phi = AS->ATSF->chi_nu(VecDoub(3,tmp));
	}
	double ptau = (AS->Ints[0]*tau*tau-AS->Ints[1]*tau+AS->Ints[2]+phi)/(2.0*(tau+a)*(tau+b)*(tau+c));
	return sqrt(MAX(0.,ptau))*cos(theta);
}

double dJdE_integrand_Triax_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax_fudge * AS = (action_struct_triax_fudge *) params;
	double tau = AS->taubargl+AS->Deltagl*sin(theta);
	double a   = AS->ATSF->CS->alpha(), b = AS->ATSF->CS->beta(), c = AS->ATSF->CS->gamma();
	double phi = 0.;
	if(AS->swit==0)	{
	    double tmp[3] = {tau,AS->tau_i[1],AS->tau_i[2]};
	    phi = AS->ATSF->chi_lam(VecDoub(3,tmp));
	}else if(AS->swit==1){
	    double tmp[3] = {AS->tau_i[0],tau,AS->tau_i[2]};
		phi = AS->ATSF->chi_mu(VecDoub(3,tmp));
	}else if(AS->swit==2){
        double tmp[3] = {AS->tau_i[0],AS->tau_i[1],tau};
	    phi = AS->ATSF->chi_nu(VecDoub(3,tmp));
	}
	double ptau = (AS->Ints[0]*tau*tau-AS->Ints[1]*tau+AS->Ints[2]+phi)/(2.0*(tau+a)*(tau+b)*(tau+c));
	return 0.25*cos(theta)*tau*tau/(sqrt(MAX(1e-6,ptau))*(tau+a)*(tau+b)*(tau+c));
}

double dJdJ_integrand_Triax_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax_fudge * AS = (action_struct_triax_fudge *) params;
	double tau = AS->taubargl+AS->Deltagl*sin(theta);
	double a   = AS->ATSF->CS->alpha(), b = AS->ATSF->CS->beta(), c = AS->ATSF->CS->gamma();
	double phi = 0.;
	if(AS->swit==0)	{
	    double tmp[3] = {tau,AS->tau_i[1],AS->tau_i[2]};
        phi = AS->ATSF->chi_lam(VecDoub(3,tmp));
	} else if(AS->swit==1)	{
	    double tmp[3] = {AS->tau_i[0],tau,AS->tau_i[2]};
	    phi = AS->ATSF->chi_mu(VecDoub(3,tmp));
	} else if(AS->swit==2)	{
	    double tmp[3] = {AS->tau_i[0],AS->tau_i[1],tau};
	    phi = AS->ATSF->chi_nu(VecDoub(3,tmp));
	}
	double ptau = (AS->Ints[0]*tau*tau-AS->Ints[1]*tau+AS->Ints[2]+phi)/(2.0*(tau+a)*(tau+b)*(tau+c));
	return -0.25*cos(theta)*tau/(sqrt(MAX(1e-6,ptau))*(tau+a)*(tau+b)*(tau+c));
}

double dJdK_integrand_Triax_Fudge(double theta, void *params) {
	// need to set taubargl and Deltagl
	action_struct_triax_fudge * AS = (action_struct_triax_fudge *) params;
	double tau = AS->taubargl+AS->Deltagl*sin(theta);
	double a   = AS->ATSF->CS->alpha(), b = AS->ATSF->CS->beta(), c = AS->ATSF->CS->gamma();
	double phi = 0.;
	if(AS->swit==0)	{
	    double tmp[3] = {tau,AS->tau_i[1],AS->tau_i[2]};
		phi = AS->ATSF->chi_lam(VecDoub(3,tmp));
	} else if(AS->swit==1)	{
	    double tmp[3] = {AS->tau_i[0],tau,AS->tau_i[2]};
	    phi = AS->ATSF->chi_mu(VecDoub(3,tmp));
	} else if(AS->swit==2){
        double tmp[3] = {AS->tau_i[0],AS->tau_i[1],tau};
        phi = AS->ATSF->chi_nu(VecDoub(3,tmp));
    }
	double ptau = (AS->Ints[0]*tau*tau-AS->Ints[1]*tau+AS->Ints[2]+phi)/(2.0*(tau+a)*(tau+b)*(tau+c));
	return 0.25*cos(theta)/(sqrt(MAX(1e-6,ptau))*(tau+a)*(tau+b)*(tau+c));
}

VecDoub Actions_TriaxialStackel_Fudge::actions(VecDoub_I &x) {
	VecDoub tau = CS->xv2tau(x);
	E = Pot->H(x);
	integrals(tau);
	// std::cerr<<Jt[0]<<" "<<Jt[1]<<" "<<Jt[2]<<" "<<Kt[0]<<" "<<Kt[1]<<" "<<Kt[2]<<std::endl;
	VecDoub limits = find_limits(tau);
	VecDoub actions(4);
	// JR
	double tmp1[3] = {E,Jt[0],Kt[0]};
	VecDoub ints(3,tmp1);
	double taubar = 0.5*(limits[0]+limits[1]);
	double Delta  = 0.5*(limits[1]-limits[0]);
	action_struct_triax_fudge AS(this,ints,tau,taubar,Delta,0);
	double tmp2[3] = {1.,1.,1.};
	VecDoub circ(3,tmp2);
	// if(limits[0]==-CS->alpha()) circ[0] = 1.; else circ[0] = 0.5;
	actions[0] = (2.*circ[0]*Delta*GaussLegendreQuad(&J_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI);
	// Jp
	ints[1]    = Jt[1];ints[2]=Kt[1];
	taubar     = 0.5*(limits[2]+limits[3]);
	Delta      = 0.5*(limits[3]-limits[2]);
	AS         = action_struct_triax_fudge(this,ints,tau,taubar,Delta,1);
	actions[1] = (2.*circ[1]*Delta*GaussLegendreQuad(&J_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI);

	// Jz
	ints[1]    = Jt[2];ints[2]=Kt[2];
	taubar     = 0.5*(limits[4]+limits[5]);
	Delta      = 0.5*(limits[5]-limits[4]);
	AS         = action_struct_triax_fudge(this,ints,tau,taubar,Delta,2);
	actions[2] = (2.*circ[2]*Delta*GaussLegendreQuad(&J_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI);
	// for(auto i:limits)actions.push_back(i);
	// printVector(limits);

	// 0 box, 1 short-axis loop, 2 inner long-axis loop, 3 outer long-axis loop
	if(limits[2]<-CS->beta()+SMALL and limits[3]>-CS->alpha()-SMALL) actions[3] = 1;
	else if(limits[5]>-CS->beta()-SMALL and limits[0]<-CS->alpha()+SMALL) actions[3] = 2;
	else if(limits[5]>-CS->beta()-SMALL and limits[3]>-CS->alpha()-SMALL) actions[3] = 3;
	else if(limits[0]<-CS->alpha()+SMALL and limits[2]<-CS->beta()+SMALL and limits[4]<-CS->gamma()+SMALL) actions[3] = 0;
	else actions[3] = -1;
	for (int i=0; i<4; ++i)
		if(std::isinf(actions[i]) or std::isnan(actions[i]) or actions[i]<0.)
			actions[i]=0.;
	return actions;
}

VecDoub Actions_TriaxialStackel_Fudge::angles(VecDoub_I x){
	VecDoub tau    = CS->xv2tau(x);
	E              = Pot->H(x);
	integrals(tau);
	VecDoub limits   = find_limits(tau);
	VecDoub angles(11);
	// lam
	double tmp[3]   = {E,Jt[0],Kt[0]};
	VecDoub ints(3,tmp);
	double taubar   = 0.5*(limits[0]+limits[1]);
	double Delta    = 0.5*(limits[1]-limits[0]);
	action_struct_triax_fudge AS(this,ints,tau,taubar,Delta,0);
	double circ     = 0.5;
	if(limits[0]==-CS->alpha()) circ = 1.;
	VecDoub dJdIl(3,0.);
	dJdIl[0]         = 2.*circ*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIl[1]         = 2.*circ*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIl[2]         = 2.*circ*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	VecDoub dSdI(3,0.);
	double thetaLim = asin(MAX(-1.,MIN(1.,(tau[0]-taubar)/Delta)));
	double lsign    = SIGN_JS(tau[3]);
	dSdI[0]         += lsign*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[1]         += lsign*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[2]         += lsign*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);

	if(x[0]<0 and circ==1.)  for(int i=0;i<3;i++) dSdI[i]+=dJdIl[i]*PI;
	if(lsign<0) for(int i=0;i<3;i++) dSdI[i]+=dJdIl[i]*PI/circ;

	// Lz
	ints[1]          = Jt[1];ints[2]=Kt[1];
	taubar           = 0.5*(limits[2]+limits[3]);
	Delta            = 0.5*(limits[3]-limits[2]);
	AS               = action_struct_triax_fudge(this,ints,tau,taubar,Delta,1);
	VecDoub dJdIm(3,0.);
	circ             = 1;
	if(limits[3]!=-CS->alpha() and limits[2]!=-CS->beta()) circ = 0.5;
	dJdIm[0]         = 2.*circ*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIm[1]         = 2.*circ*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIm[2]         = 2.*circ*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	thetaLim         = asin(MAX(-1.,MIN(1.,(tau[1]-taubar)/Delta)));
	lsign = SIGN_JS(tau[4]);
	dSdI[0]         += lsign*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[1]         += lsign*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[2]         += lsign*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);

	// Short and Box
	if(limits[2]==-CS->beta()){
		if(lsign<0)for(int i=0;i<3;i++) dSdI[i]+=dJdIm[i]*PI;
		if(x[1]<0) for(int i=0;i<3;i++) dSdI[i]+=dJdIm[i]*PI;
	}
	// Outer long-axis loop
	else if(limits[3]==-CS->alpha() and limits[2]!=-CS->beta()){
		for(int i=0;i<3;i++) dSdI[i]=dJdIm[i]*PI/2.+dSdI[i];
		if(x[0]<0) for(int i=0;i<3;i++) dSdI[i]+=dJdIm[i]*PI;
		// if(lsign>0)for(int i=0;i<3;i++) dSdI[i]+=dJdIm[i]*PI;
	}
	// Inner long-axis loop
	else if(limits[3]!=-CS->alpha() and limits[2]!=-CS->beta()){
		if(lsign<0) for(int i=0;i<3;i++) dSdI[i]+=dJdIl[i]*PI/circ;
	}

	// Jz
	ints[1]             = Jt[2];ints[2]=Kt[2];
	taubar              = 0.5*(limits[4]+limits[5]);
	Delta               = 0.5*(limits[5]-limits[4]);
	AS                  = action_struct_triax_fudge(this,ints,tau,taubar,Delta,2);
	VecDoub dJdIn(3,0.);
	dJdIn[0]            = 2.*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIn[1]            = 2.*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	dJdIn[2]            = 2.*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,.5*PI,&AS)/PI;
	thetaLim            = asin(MAX(-1.,MIN(1.,(tau[2]-taubar)/Delta)));
	lsign = SIGN_JS(tau[5]);
	dSdI[0]            += lsign*Delta*GaussLegendreQuad(&dJdE_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[1]            += lsign*Delta*GaussLegendreQuad(&dJdJ_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);
	dSdI[2]            += lsign*Delta*GaussLegendreQuad(&dJdK_integrand_Triax_Fudge,-.5*PI,thetaLim,&AS);

	if(x[2]<0)  for(int i=0;i<3;i++) dSdI[i]+=dJdIn[i]*PI;
	if(lsign<0) for(int i=0;i<3;i++) dSdI[i]+=dJdIn[i]*PI;

	double Determinant = dJdIl[0]*det2(dJdIm[1],dJdIm[2],dJdIn[1],dJdIn[2])
						- dJdIl[1]*det2(dJdIm[0],dJdIm[2],dJdIn[0],dJdIn[2])
						+ dJdIl[2]*det2(dJdIm[0],dJdIm[1],dJdIn[0],dJdIn[1]);

	double dIdJ[3][3];
	dIdJ[0][0]          = det2(dJdIm[1],dJdIm[2],dJdIn[1],dJdIn[2])/Determinant;
	dIdJ[0][1]          = -det2(dJdIl[1],dJdIl[2],dJdIn[1],dJdIn[2])/Determinant;
	dIdJ[0][2]          = det2(dJdIl[1],dJdIl[2],dJdIm[1],dJdIm[2])/Determinant;
	dIdJ[1][0]          = -det2(dJdIm[0],dJdIm[2],dJdIn[0],dJdIn[2])/Determinant;
	dIdJ[1][1]          = det2(dJdIl[0],dJdIl[2],dJdIn[0],dJdIn[2])/Determinant;
	dIdJ[1][2]          = -det2(dJdIl[0],dJdIl[2],dJdIm[0],dJdIm[2])/Determinant;
	dIdJ[2][0]          = det2(dJdIm[0],dJdIm[1],dJdIn[0],dJdIn[1])/Determinant;
	dIdJ[2][1]          = -det2(dJdIl[0],dJdIl[1],dJdIn[0],dJdIn[1])/Determinant;
	dIdJ[2][2]          = det2(dJdIl[0],dJdIl[1],dJdIm[0],dJdIm[1])/Determinant;

	angles[0] = dSdI[0]*dIdJ[0][0]+dSdI[1]*dIdJ[1][0]+dSdI[2]*dIdJ[2][0];
	angles[1] = dSdI[0]*dIdJ[0][1]+dSdI[1]*dIdJ[1][1]+dSdI[2]*dIdJ[2][1];
	angles[2] = dSdI[0]*dIdJ[0][2]+dSdI[1]*dIdJ[1][2]+dSdI[2]*dIdJ[2][2];
	angles[3] = det2(dJdIm[1],dJdIm[2],dJdIn[1],dJdIn[2])/Determinant;
	angles[4] = det2(dJdIn[1],dJdIn[2],dJdIl[1],dJdIl[2])/Determinant;
	angles[5] = det2(dJdIl[1],dJdIl[2],dJdIm[1],dJdIm[2])/Determinant;

	for(int i=0;i<3;i++){
		if(angles[i]<0.)angles[i]+=2.*PI;
		if(angles[i]>2.*PI)angles[i]-=2.*PI;
		angles[6+i] = (x[i]);
	}
	angles[9]  = (tau[1]);
	angles[10] = (SIGN_JS(tau[4]));
	return angles;
}

double Actions_TriaxialStackel_Fudge::sos(VecDoub_I x, int comp, std::string file){
	std::ofstream outfile; outfile.open(file.c_str(), ios::app);
	outfile<<x[comp]/*sqrt(x[0]*x[0]+x[1]*x[1]/0.95/0.95+x[2]*x[2]/0.85/0.85)*/<<" "<<x[comp+3]<<std::endl;

	VecDoub tau = CS->xv2tau(x);
	E = Pot->H(x);
	integrals(tau);
	VecDoub limits = find_limits(tau);
	double a = CS->alpha(), b = CS->beta(), c = CS->gamma();
	double up = 0., down = 0.; int coord = 0;
	if(comp==0){ down = limits[0]+SMALL; up = limits[1]-SMALL; coord = 0;}
	if(comp==1){ down = limits[0]+SMALL; up = limits[1]-SMALL; coord = 0;}
	if(comp==2){ down = limits[2]+SMALL; up = limits[3]-SMALL; coord = 1;}
	double tmp1[3] = {E,Jt[coord],Kt[coord]};
	VecDoub_I ints(3,tmp1);

	double l;
	for(int n=0;n<200;n++){
		l = sqrt(down+a)+n*(sqrt(up+a)-sqrt(down+a))/199.;
		l = l*l-a;
		if(comp==0){
			outfile<<sqrt(MAX(0.,l+a))<<" ";
			if(fabs(l+b)<SMALL or fabs(l+c)<SMALL) outfile<<0.;
			else {
                double tmp2[3] = {l,tau[1],tau[2]};
			    outfile<<sqrt(MAX(0.,2.*(ints[0]*l*l-ints[1]*l+ints[2]+chi_lam(VecDoub_I(3,tmp2))/((l+b)*(l+c)))));
            }
		}
		if(comp==1){
			outfile<<sqrt(MAX(0.,l+b))<<" ";
			if(fabs(l+a)<SMALL or fabs(l+c)<SMALL) outfile<<0.;
			else {
                double tmp2[3] = {l,tau[1],tau[2]};
			    outfile<<sqrt(MAX(0.,2.*(ints[0]*l*l-ints[1]*l+ints[2]+chi_lam(VecDoub_I(3,tmp2))/((l+a)*(l+c)))));
			}
		}
		if(comp==2){
			outfile<<sqrt(MAX(0.,l+c))<<" ";
			if(fabs(l+a)<SMALL or fabs(l+b)<SMALL) outfile<<0.;
			else {
                double tmp2[3] = {tau[0],l,tau[2]};
			    outfile<<sqrt(MAX(0.,2.*(ints[0]*l*l-ints[1]*l+ints[2]+chi_mu(VecDoub_I(3,tmp2))/((l+a)*(l+b)))));
			}
		}
		outfile<<std::endl;
	}
	outfile.close();
	return 0;
}
#endif
// ============================================================================
// Stackel_JS.cpp
