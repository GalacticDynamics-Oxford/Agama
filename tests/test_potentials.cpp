#include "potential_analytic.h"
#include "potential_factory.h"
#include "potential_dehnen.h"
#include "potential_ferrers.h"
#include "units.h"
#include "math_core.h"
#include "debug_utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const double eps=1e-6;  // accuracy of comparison

bool testPotential(const potential::BasePotential* potential)
{
    if(potential==NULL) return true;
    bool ok=true;
    std::cout << potential->name();
    double val0 = potential->value(coord::PosCar(0,0,0));
    std::cout << " at origin is "<<val0;
    ok &= math::isFinite(val0);
    double mtot = potential->totalMass();
    double minf = potential->enclosedMass(INFINITY);
    std::cout << "; total mass is "<<mtot<<
        ", or enclosed mass M(r<inf) is "<<minf<<
        ", M(r<=0) is "<<potential->enclosedMass(0)<<
        ", M(r<1e9) is "<<potential->enclosedMass(1e9)<<"\n";
    return ok;
}

template<typename coordSysT>
bool testPotentialAtPoint(const potential::BasePotential* potential,
    const coord::PosVelT<coordSysT>& point)
{
    if(potential==NULL) return true;
    bool ok=true;
    std::cout << potential->name()<<"  "<<coordSysT::name()<<"  " << point;
    double E = potential::totalEnergy(*potential, point);
    if((potential->symmetry() & potential::ST_ZROTSYM) == potential::ST_ZROTSYM) {  // test only axisymmetric potentials
        try{
            double Rc  = R_circ(*potential, E);
            double vc  = v_circ(*potential, Rc);
            double E1  = potential->value(coord::PosCyl(Rc, 0, 0)) + 0.5*vc*vc;
            double Lc1 = L_circ(*potential, E);
            double Rc1 = R_from_Lz(*potential, Lc1);
            ok &= math::fcmp(Rc, Rc1, 1e-11)==0 && math::fcmp(E, E1, 1e-11)==0;
            if(!ok) std::cout << "\033[1;31m ** \033[0m"
                "E="<<E<<", Rc(E)="<<Rc<<", E(Rc)="<<E1<<", Lc(E)="<<Lc1<<", Rc(Lc)="<<Rc1;
        }
        catch(std::exception &e) {
            std::cout << e.what();
        }
    }
    std::cout << "\n";
    return ok;
}

const potential::BasePotential* make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    const potential::BasePotential* gp = potential::readGalaxyPotential(params_file, units::galactic_Myr);
    std::remove(params_file);
    if(gp==NULL)
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

const int numpotentials=10;
const potential::BasePotential* pots[numpotentials] = {NULL};

int main() {
    pots[0] = new potential::Plummer(10.,5.);
    pots[1] = new potential::NFW(10.,10.);
    pots[2] = new potential::MiyamotoNagai(5.,2.,0.2);
    pots[3] = new potential::Logarithmic(1.,0.01,.8,.5);
    pots[4] = new potential::Logarithmic(1.,.7,.5);
    pots[5] = new potential::Ferrers(1.,0.9,.7,.5);
    pots[6] = new potential::Dehnen(2.,1.,.7,.5,1.5);
    pots[7] = make_galpot(test_galpot_params[0]);
    pots[8] = make_galpot(test_galpot_params[1]);
    bool allok=true;
    std::cout << std::setprecision(10);
    for(int ip=0; ip<numpotentials; ip++) {
        allok &= testPotential(pots[ip]);
        for(int ic=0; ic<numtestpoints; ic++) {
            allok &= testPotentialAtPoint(pots[ip], coord::PosVelCar(posvel_car[ic]));
            allok &= testPotentialAtPoint(pots[ip], coord::PosVelCyl(posvel_cyl[ic]));
            allok &= testPotentialAtPoint(pots[ip], coord::PosVelSph(posvel_sph[ic]));
        }
    }
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    for(int ip=0; ip<numpotentials; ip++)
        delete pots[ip];
    return 0;
}