#include "potential_analytic.h"
#include "potential_factory.h"
#include "potential_dehnen.h"
#include "potential_ferrers.h"
#include "potential_utils.h"
#include "units.h"
#include "utils.h"
#include "math_core.h"
#include "debug_utils.h"
#include "actions_spherical.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <stdexcept>

const double eps=1e-6;  // accuracy of comparison
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

bool testPotential(const potential::BasePotential& potential)
{
    bool ok=true;
    std::cout << potential.name();
    double val0 = potential.value(coord::PosCar(0,0,0));
    std::cout << " at origin is "<<val0;
    ok &= isFinite(val0);
    double mtot = potential.totalMass();
    double minf = potential.enclosedMass(INFINITY);
    std::cout << "; total mass is "<<mtot<<
        ", or enclosed mass M(r<inf) is "<<minf<<
        ", M(r<=0) is "<<potential.enclosedMass(0)<<
        ", M(r<1) is "<<potential.enclosedMass(1)<<
        ", M(r<10) is "<<potential.enclosedMass(10)<<
        ", M(r<1e9) is "<<potential.enclosedMass(1e9)<<"\n";
    // test interpolated potential
    try{
        potential::Interpolator2d interp(potential);
        actions::ActionFinderSpherical af(potential);
        std::ofstream strm, strmr;
        if(output) {
            strm. open((std::string("test_pot_" )+potential.name()).c_str());
            strmr.open((std::string("testr_pot_")+potential.name()).c_str());
        }
        strm << std::setprecision(15) << "E R Rc_root(E) Rc_interp(E) Lc Lc_root(E) Lc_interp(E)\n";
        strmr<< std::setprecision(15);
        for(double lr=-16; lr<=24; lr+=.25) {
            double r = pow(2., lr);
            double Phi;
            coord::GradCyl grad;
            coord::HessCyl hess;
            potential.eval(coord::PosCyl(r,0,0), &Phi, &grad, &hess);
            double E   = Phi + 0.5*pow_2(v_circ(potential, r));
            double Lc  = v_circ(potential, r) * r;
            double RcR = R_circ(potential, E);
            double RcI = interp.pot.R_circ(E);
            double LcR = L_circ(potential, E);
            double LcI = interp.pot.L_circ(E);
            double Rm  = R_max(potential, E);
            double PhiI, dPhiI, d2PhiI, dRofPhi;
            interp.pot.evalDeriv(r, &PhiI, &dPhiI, &d2PhiI);
            double RofPhi = interp.pot.R_max(PhiI, &dRofPhi);
            strm << E << ' ' << r << ' ' << RcR << ' ' << RcI << ' ' <<
                Lc << ' ' << LcR << ' ' << LcI << '\t' <<
                Phi << ' ' << grad.dR << ' ' << hess.dR2 << ' ' <<
                PhiI << ' ' << dPhiI << ' ' << d2PhiI << ' ' << 
                Rm << ' ' << RofPhi << ' ' << dRofPhi << '\n';
            for(double ll=0; ll<1; ll+=1./128) {
                double L = Lc * sin(M_PI_2*ll);
                double Rmin, Rmax, RminI, RmaxI;
                try{
                    findPlanarOrbitExtent(potential, E, L, Rmin, Rmax);
                }
                catch (std::runtime_error&) {
                    Rmin=Rmax=NAN;
                }
                interp.findPlanarOrbitExtent(E, L, RminI, RmaxI);
                double Jr=af.Jr(E, L);
                actions::Actions act = actions::actionsSpherical(potential,
                    coord::PosVelCyl(Rmax, 0, 0, 0, 0, L/Rmax));
                strmr << E << ' ' << pow_2(L/Lc) << ' ' <<
                    Rmin/r << ' ' << RminI/r << ' ' << Rmax/r << ' ' << RmaxI/r << ' ' <<
                    act.Jr/Lc << ' ' << Jr/Lc << '\n'; 
            }
            strmr<<'\n';
        }
    }
    catch(std::exception& e) {
        std::cout << "Cannot create interpolator: "<<e.what()<<"\n";
    }
    return ok;
}

template<typename coordSysT>
bool testPotentialAtPoint(const potential::BasePotential& potential,
    const coord::PosVelT<coordSysT>& point)
{
    bool ok=true;
    double E = potential::totalEnergy(potential, point);
    if(isAxisymmetric(potential)) {
        try{
            double Rc  = R_circ(potential, E);
            double vc  = v_circ(potential, Rc);
            double E1  = potential.value(coord::PosCyl(Rc, 0, 0)) + 0.5*vc*vc;
            double Lc1 = L_circ(potential, E);
            double Rc1 = R_from_Lz(potential, Lc1);
            ok &= math::fcmp(Rc, Rc1, 2e-10)==0 && math::fcmp(E, E1, 1e-11)==0;
            if(!ok)
                std::cout << potential.name()<<"  "<<coordSysT::name()<<"  " << point << "\033[1;31m ** \033[0m"
                "E="<<E<<", Rc(E)="<<Rc<<", E(Rc)="<<E1<<", Lc(E)="<<Lc1<<", Rc(Lc)="<<Rc1 << "\n";
        }
        catch(std::exception &e) {
            std::cout << potential.name()<<"  "<<coordSysT::name()<<"  " << point << e.what() << "\n";
        }
    }
    return ok;
}

potential::PtrPotential make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    potential::PtrPotential gp = potential::readGalaxyPotential(params_file, units::galactic_Myr);
    std::remove(params_file);
    if(gp.get()==NULL)
        std::cout<<"Potential not created\n";
    return gp;
}

const char* test_galpot_params[] = {
"2\n"  // stanard galaxy model
"7.52975e+08 3 0.3 0 0\n"
"1.81982e+08 3.5 0.9 0 0\n"
"2\n"
"9.41496e+10 0.5 0 1.8 0.075 2.1\n"
"1.25339e+07 1 1 3 17 0\n", 
"3\n"  // test various types of discs
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n"  };// log density profile with cutoff

/*const int numtestpoints=3;
const double init_cond[numtestpoints][6] = {
  {1, 0.5, 0.2, 0.1, 0.2, 0.3},
  {2, 0, 0, 0, 0, 0},
  {0, 0, 1, 0, 0, 0} };
*/
/// define test suite in terms of points for various coord systems
const int numtestpoints=5;
const double posvel_car[numtestpoints][6] = {
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {0,-1, 2, 0,   0.2,-0.5},   // point in y-z plane 
    {2, 0,-1, 0,  -0.3, 0.4},   // point in x-z plane
    {0, 0, 1, 0,   0,   0.5},   // point along z axis
    {0, 0, 0,-0.4,-0.2,-0.1}};  // point at origin with nonzero velocity
const double posvel_cyl[numtestpoints][6] = {   // order: R, z, phi
    {1, 2, 3, 0.4, 0.2, 0.3},   // ordinary point
    {2,-1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {0, 2, 0, 0,  -0.5, 0  },   // point along z axis, vphi must be zero
    {0,-1, 2, 0.5, 0.3, 0  },   // point along z axis, vphi must be zero, but vR is non-zero
    {0, 0, 0, 0.3,-0.5, 0  }};  // point at origin with nonzero velocity in R and z
const double posvel_sph[numtestpoints][6] = {   // order: R, theta, phi
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {2, 1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {1, 0, 0,-0.5, 0,   0  },   // point along z axis, vphi must be zero
    {1,3.14159, 2, 0.5, 0.3, 1e-4},   // point almost along z axis, vphi must be small, but vtheta is non-zero
    {0, 2,-1, 0.5, 0,   0  }};  // point at origin with nonzero velocity in R

int main() {
    std::vector<potential::PtrPotential> pots;
    pots.push_back(potential::PtrPotential(new potential::Plummer(10.,5.)));
    pots.push_back(potential::PtrPotential(new potential::Isochrone(1.,1.)));
    pots.push_back(potential::PtrPotential(new potential::NFW(10.,10.)));
    pots.push_back(potential::PtrPotential(new potential::MiyamotoNagai(5.,2.,0.2)));
    pots.push_back(potential::PtrPotential(new potential::Logarithmic(1.,0.01,.8,.5)));
    pots.push_back(potential::PtrPotential(new potential::Logarithmic(1.,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Ferrers(1.,0.9,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Dehnen(2.,1.,1.5,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Dehnen(2.,1.,1.5,1.,.5)));
    pots.push_back(make_galpot(test_galpot_params[0]));
    pots.push_back(make_galpot(test_galpot_params[1]));
    bool allok=true;
    std::cout << std::setprecision(10);
    for(unsigned int ip=0; ip<pots.size(); ip++) {
        allok &= testPotential(*pots[ip]);
        for(int ic=0; ic<numtestpoints; ic++) {
            allok &= testPotentialAtPoint(*pots[ip], coord::PosVelCar(posvel_car[ic]));
            allok &= testPotentialAtPoint(*pots[ip], coord::PosVelCyl(posvel_cyl[ic]));
            allok &= testPotentialAtPoint(*pots[ip], coord::PosVelSph(posvel_sph[ic]));
        }
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}