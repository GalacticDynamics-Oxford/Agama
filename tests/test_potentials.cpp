#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_factory.h"
#include "potential_ferrers.h"
#include "potential_galpot.h"
#include "potential_multipole.h"
#include "potential_utils.h"
#include "utils.h"
#include "debug_utils.h"
#include "math_core.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const double eps=1e-6;  // accuracy of comparison
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

std::string checkLess(double val, double max, bool &ok)
{
    if(!(val<max))
        ok = false;
    return utils::pp(val, 7) + (val<max ? "" : "\033[1;31m ** \033[0m");
}

bool testPotential(const potential::BasePotential& potential)
{
    bool ok=true;
    std::cout << "\033[1;33m " << potential.name() << " \033[0m";
    double val0 = potential.value(coord::PosCar(0,0,0));
    std::cout << " at origin is "<<val0;
    ok &= val0==val0;
    double mtot = potential.totalMass();
    double minf = potential.enclosedMass(INFINITY);
    std::cout << "; total mass is "<<mtot<<
        ", or enclosed mass M(r<inf) is "<<minf<<
        ", M(r<=0) is "<<potential.enclosedMass(0)<<
        ", M(r<1) is "<<potential.enclosedMass(1)<<
        ", M(r<10) is "<<potential.enclosedMass(10)<<
        ", M(r<1e9) is "<<potential.enclosedMass(1e9)<<"\n";
    if(!isZRotSymmetric(potential) ||
        potential.name() == potential::Logarithmic::myName() )
        // non-axisymmetric or infinite potentials are not amenable for further tests
        return ok;
    // test interpolated potential
    try{
        potential::Interpolator interp(potential);
        std::ofstream strm;
        if(output) {
            std::string filename = std::string("test_pot_" )+potential.name();
            // gnuplot script for plotting the results
            strm.open((filename+".plt").c_str());
            strm << "set term pdf enhanced size 15cm,10cm\nset output '"+filename+".pdf'\n"
            "set logscale\nset xrange [1e-7:1e9]\nset yrange [1e-16:1e-4]\n"
            "set format x '10^{%T}'\nset format y '10^{%T}'\nset multiplot layout 2,2\n"
            "plot '"+filename+".dat' u 2:(abs($3/$2-1)) w l title 'Rcirc(E),root', \\\n"
            "  '' u 2:(abs($4/$2-1)) w l title 'Rcirc(E),interp', \\\n"
            "  '' u 2:(abs($8/$2-1)) w l title 'Rcirc(Lz),root', \\\n"
            "  '' u 2:(abs($9/$2-1)) w l title 'Rcirc(Lz),interp'\n"
            "plot '' u 2:(abs($6/$5-1)) w l title 'Lcirc(E),root', \\\n"
            "  '' u 2:(abs($7/$5-1)) w l title 'Lcirc(E),interp'\n"
            "plot '' u 2:(abs($10/$2-1)) w l title 'Rmax(E),root', \\\n"
            "  '' u 2:(abs($11/$2-1)) w l title 'Rmax(E),interp'\n"
            "plot '' u 2:(abs($15/$12-1)) w l title 'Phi(r),interp', \\\n"
            "  '' u 2:(abs($16/$13-1)) w l title 'dPhi/dr,interp', \\\n"
            "  '' u 2:(abs($17/$14-1)) w l title 'rho,interp'\n";
            strm.close();
            strm.open((filename+".dat").c_str());
            strm << std::setprecision(16) << "E\t"
            "R Rc_root(E) Rc_interp(E)\t"
            "Lc Lc_root(E) Lc_interp(E)\t"
            "Rc_root(Lc) Rc_interp(Lc)\t"
            "Rmax_root(Phi) Rmax_interp(Phi)\t"
            "Phi dPhi/dR rho Phi_interp dPhi/dr_interp rho_interp\n";
        }
        double sumw = 0, errRcR = 0, errRcI = 0, errLcR = 0, errLcI = 0,
        errRLR = 0, errRLI = 0, errRmR = 0, errRmI = 0, errPhiI = 0, errdPhiI = 0;
        for(double lr=-7; lr<=9; lr+=.0625) {
            double r = pow(10., lr);
            double Phi;
            coord::GradCyl grad;
            coord::HessCyl hess;
            potential.eval(coord::PosCyl(r,0,0), &Phi, &grad, &hess);
            double E   = Phi + 0.5*pow_2(v_circ(potential, r));
            double Lc  = v_circ(potential, r) * r;
            double RcR = R_circ(potential, E);
            double RcI = interp.R_circ(E);
            double LcR = L_circ(potential, E);
            double LcI = interp.L_circ(E);
            double RLR = R_from_Lz(potential, Lc);
            double RLI = interp.R_from_Lz(Lc);
            double RmR = R_max(potential, Phi);
            double RmI = interp.R_max(Phi);
            double PhiI, dPhiI, d2PhiI;
            interp.evalDeriv(r, &PhiI, &dPhiI, &d2PhiI);
            double truedens = (hess.dR2 + 2*grad.dR/r) / (4*M_PI);
            double intdens  = (d2PhiI   + 2*dPhiI  /r) / (4*M_PI);

            // density-weighted error: integrate |x-x_true|^2 r^3 d log(r)
            double weight = pow_3(r) * fmax(truedens, 0);
            sumw    += weight;
            errRcR  += weight * pow_2((r   - RcR) / (r   + RcR) * 2);
            errRcI  += weight * pow_2((r   - RcI) / (r   + RcI) * 2);
            errRLR  += weight * pow_2((r   - RLR) / (r   + RLR) * 2);
            errRLI  += weight * pow_2((r   - RLI) / (r   + RLI) * 2);
            errRmR  += weight * pow_2((r   - RmR) / (r   + RmR) * 2);
            errRmI  += weight * pow_2((r   - RmI) / (r   + RmI) * 2);
            errLcR  += weight * pow_2((Lc  - LcR) / (Lc  + LcR) * 2);
            errLcI  += weight * pow_2((Lc  - LcI) / (Lc  + LcI) * 2);
            errPhiI += weight * pow_2((Phi - PhiI)/ (Phi + PhiI)* 2);
            errdPhiI+= weight * pow_2((grad.dR - dPhiI) / (grad.dR + dPhiI) * 2);

            strm << E << '\t' <<
                r   << ' ' << RcR << ' ' << RcI << '\t' <<
                Lc  << ' ' << LcR << ' ' << LcI << '\t' <<
                RLR << ' ' << RLI << '\t' <<
                RmR << ' ' << RmI << '\t' <<
                Phi << ' ' << grad.dR<< ' ' << truedens << ' ' <<
                PhiI<< ' ' << dPhiI  << ' ' << intdens  << '\n';
        }
        errLcR = sqrt(errLcR / sumw);
        errLcI = sqrt(errLcI / sumw);
        errRcR = sqrt(errRcR / sumw);
        errRcI = sqrt(errRcI / sumw);
        errRmR = sqrt(errRmR / sumw);
        errRmI = sqrt(errRmI / sumw);
        errRLR = sqrt(errRLR / sumw);
        errRLI = sqrt(errRLI / sumw);
        errPhiI= sqrt(errPhiI/ sumw);
        errdPhiI=sqrt(errdPhiI/sumw);
        // check if the errors are within design tolerance;
        // for the composite potential (aka GalPot = Multipole + DiskAnsatz) or Multipole
        // we loosen the tolerance limits, as the Multipole potential is intrinsically
        // only an approximation and not infinitely smooth, thus its interpolated version
        // is not required to be exceedingly accurate
        double tol =
            potential.name() == potential::Multipole::myName() ? 1e4 :
            potential.name() == potential::CylSpline::myName() ? 1e2 :
            potential.name() == potential::CompositeCyl::myName() ? 1e2 : 1.;
        std::cout << "Density-weighted RMS errors"
        ": Phi(r)="     + checkLess(errPhiI, 1e-10 * tol, ok) +
        ", dPhi/dr="    + checkLess(errdPhiI,1e-08 * tol, ok) +
        ", Lc,root(E)=" + checkLess(errLcR,  1e-12,       ok) +
        ", Lc,int(E)="  + checkLess(errLcI,  1e-08 * tol, ok) +
        ", Rc,root(E)=" + checkLess(errRcR,  1e-12,       ok) +
        ", Rc,int(E)="  + checkLess(errRcI,  1e-09 * tol, ok) +
        ", Rc,root(Lz)="+ checkLess(errRLR,  1e-12,       ok) +
        ", Rc,int(Lz)=" + checkLess(errRLI,  1e-09 * tol, ok) +
        ", Rm,root(E)=" + checkLess(errRmR,  1e-12,       ok) +
        ", Rm,int(E)="  + checkLess(errRmI,  1e-09 * tol, ok) + "\n";
    }
    catch(std::exception& e) {
        std::cout << "Cannot create interpolator: "<<e.what()<<"\n";
        ok = false;
    }
#if 1
    try{
        potential::Interpolator2d interp(potential);
        /*actions::ActionFinderSpherical af(potential);
        std::ofstream strm;
        if(output)
            strm.open((std::string("testr_pot_")+potential.name()).c_str());
        strm << std::setprecision(15);
        for(double lr=-16.125; lr<=24.125; lr+=.25) {
            double r  = pow(2., lr);
            double Lc = v_circ(potential, r) * r;
            double E  = potential.value(coord::PosCyl(r, 0, 0)) + 0.5 * pow_2(Lc/r);
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
                strm << E << ' ' << pow_2(L/Lc) << ' ' <<
                    Rmin/r << ' ' << RminI/r << ' ' << Rmax/r << ' ' << RmaxI/r << ' ' <<
                    act.Jr/Lc << ' ' << Jr/Lc << '\n'; 
            }
            strm<<'\n';
        }*/
    }
    catch(std::exception& e) {
        std::cout << "Cannot create 2d interpolator: "<<e.what()<<"\n";
    }
#endif
    return ok;
}

template<typename coordSysT>
bool testPotentialAtPoint(const potential::BasePotential& potential,
    const coord::PosVelT<coordSysT>& point)
{
    bool ok=true;
    double E = potential::totalEnergy(potential, point);
    if(isAxisymmetric(potential) && E!=-INFINITY && E<0) {
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
// stanard galaxy model
"2\n"
"7.52975e+08 3 0.3 0 0\n"
"1.81982e+08 3.5 0.9 0 0\n"
"2\n"
"9.41496e+10 0.5 0 1.8 0.075 2.1\n"
"1.25339e+07 1 1 3 17 0\n",
// various types of discs
"3\n"
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n", // log density profile with a cutoff
// an extreme case of a spheroid profile with a very steep cusp and slow fall-off
"0\n"
"1\n"
"1e10 0.5 2.5 2.5 1 0\n"
};

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
    pots.push_back(potential::PtrPotential(new potential::Ferrers(1.,0.9,.7,.5)));
    pots.push_back(potential::PtrPotential(new potential::Dehnen(2.,1.,1.5,1.,1.)));
    pots.push_back(potential::CylSpline::create(potential::DiskDensity(
        potential::DiskParam(1., 2., -0.2, 0, 0)), 0, 20, 0.1, 500, 20, 0.01, 50));
    pots.push_back(make_galpot(test_galpot_params[0]));
    pots.push_back(make_galpot(test_galpot_params[1]));
    pots.push_back(make_galpot(test_galpot_params[2]));
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