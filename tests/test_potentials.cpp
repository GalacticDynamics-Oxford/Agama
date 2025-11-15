/** \name   test_potentials.cpp
    \author Eugene Vasiliev
    \date   2015-2024

    Test various aspects of potential evaluation and associated utility functions
    (finding peri/apocenter radii, circular orbits, 1d interpolations of these quantities)
*/
#include "potential_composite.h"
#include "potential_cylspline.h"
#include "potential_multipole.h"
#include "potential_factory.h"
#include "potential_utils.h"
#include "utils.h"
#include "math_random.h"
#include "debug_utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;
const char* err = "\033[1;31m ** \033[0m";
std::string checkLess(double val, double max, bool &ok)
{
    if(!(val<max))
        ok = false;
    return utils::pp(val, 7) + (val<max ? "" : err);
}


bool testPotential(const potential::BasePotential& potential)
{
    bool ok=true;
    std::cout << "\033[1;33m " << potential.name() << " \033[0m";
    // check that the potential is defined (not NAN) at zero and infinity
    double
    val0car = potential.value(coord::PosCar(0,0,0)),
    val8car = potential.value(coord::PosCar(INFINITY,0,0)),
    val0cyl = potential.value(coord::PosCyl(0,0,0)),
    val8cyl = potential.value(coord::PosCyl(INFINITY,0,0)),
    val0sph = potential.value(coord::PosSph(0,0,0)),
    val8sph = potential.value(coord::PosSph(INFINITY,0,0));
    std::cout << " at origin is " << val0car << '/' << val0cyl << '/' << val0sph;
    if(!(val0car == val0cyl && val0car == val0sph)) {   // allowed to be -inf, but not NaN
        ok = false;
        std::cout << err;
    }
    std::cout << ", at r=inf is " << val8car << '/' << val8cyl << '/' << val8sph;
    if(!(val8car == val8cyl && val8car == val8sph)) {   // allowed to be infinite, but not NaN
        ok = false;
        std::cout << err;
    }

    double Rmax0  = R_max (potential, val0car), Rmax8  = R_max (potential, val8car),
           Rcirc0 = R_circ(potential, val0car), Rcirc8 = R_circ(potential, val8car),
           Rlz0   = R_from_Lz(potential, 0),    Rlz8   = R_from_Lz(potential, INFINITY);
    std::cout << "; Rmax(E=" << val0car << ")=" << Rmax0;
    if(Rmax0 != 0) {
        ok = false;
        std::cout << err;
    }
    std::cout << "; Rmax(E=" << val8car << ")=" << Rmax8;
    if(Rmax8 != INFINITY && val8car == 0) {
        ok = false;
        std::cout << err;
    }
    std::cout << "; Rcirc(E=" << val0car << ")=" << Rcirc0;
    if(Rcirc0 != 0) {
        ok = false;
        std::cout << err;
    }
    std::cout << "; Rcirc(E=" << val8car << ")=" << Rcirc8;
    if(Rcirc8 != INFINITY && val8car == 0) {
        ok = false;
        std::cout << err;
    }
    std::cout << "; Rcirc(L=0)=" << Rlz0;
    if(Rlz0 != 0) {
        ok = false;
        std::cout << err;
    }
    std::cout << "; Rcirc(L=inf)=" << Rlz8;
    if(Rlz8 != INFINITY) {
        ok = false;
        std::cout << err;
    }

    double mtot = potential.totalMass();
    double minf = potential.enclosedMass(INFINITY);
    std::cout << "; total mass is "<<mtot<<
        ", or enclosed mass M(r<inf) is "<<minf<<
        ", M(r<=0) is "<<potential.enclosedMass(0)<<
        ", M(r<1) is "<<potential.enclosedMass(1)<<
        ", M(r<10) is "<<potential.enclosedMass(10)<<
        ", M(r<1e9) is "<<potential.enclosedMass(1e9)<<"\n";
    if(!isZRotSymmetric(potential) || val8car!=0)
        // non-axisymmetric or infinite potentials are not amenable for further tests
        return ok;
    // test interpolated potential
    try{
        std::ofstream strm;
        if(output) {
            std::string filename = std::string("test_pot_" )+potential.name();
            writePotential(filename+".ini", potential);
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
            "plot '' u 2:(abs($16/$12-1)) w l title 'Phi(r),interp', \\\n"
            "  '' u 2:(abs($17/$13-1)) w l title 'dPhi/dr,interp', \\\n"
            "  '' u 2:(abs($19/$15-1)) w l title 'rho,interp'\n";
            strm.close();
            strm.open((filename+".dat").c_str());
            strm << std::setprecision(16) << "E\t"
            "R Rc_root(E) Rc_interp(E)\t"
            "Lc Lc_root(E) Lc_interp(E)\t"
            "Rc_root(Lc) Rc_interp(Lc)\t"
            "Rmax_root(Phi) Rmax_interp(Phi)\t"
            "Phi dPhi/dR d2Phi/dR2 rho Phi_interp dPhi/dR_interp d2Phi/dR2_interp rho_interp "
            "d3Phi/dR3_interp d3Phi/drR3_finite_dif\n";
        }
        potential::Interpolator interp(potential);
        double sumw = 0, errRcR = 0, errRcI = 0, errLcR = 0, errLcI = 0,
        errRLR = 0, errRLI = 0, errRmR = 0, errRmI = 0,
        errPhiI = 0, errdPhiI = 0, errd2PhiI = 0, errd3PhiI = 0;
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
            double PhiI, dPhiI, d2PhiI, d3PhiI, d2PhiI_minusdeltar, d2PhiI_plusdeltar;
            interp.evalDeriv(r, &PhiI, &dPhiI, &d2PhiI, &d3PhiI);
            const double deltar = r * ROOT3_DBL_EPSILON;
            interp.evalDeriv(r - deltar, NULL, NULL, &d2PhiI_minusdeltar);
            interp.evalDeriv(r + deltar, NULL, NULL, &d2PhiI_plusdeltar);
            double d3PhiIfd = 0.5 * (d2PhiI_plusdeltar - d2PhiI_minusdeltar) / deltar;
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
            errd2PhiI+=weight * pow_2((hess.dR2-d2PhiI) / (fabs(hess.dR2) + fabs(d2PhiI)) * 2);
            errd3PhiI+=weight * pow_2((d3PhiIfd-d3PhiI) / (fabs(d3PhiIfd) + fabs(d3PhiI)) * 2);

            strm << E << '\t' <<
                r   << ' ' << RcR << ' ' << RcI << '\t' <<
                Lc  << ' ' << LcR << ' ' << LcI << '\t' <<
                RLR << ' ' << RLI << '\t' <<
                RmR << ' ' << RmI << '\t' <<
                Phi << ' ' << grad.dR << ' ' << hess.dR2 << ' ' << truedens << '\t' <<
                PhiI<< ' ' << dPhiI   << ' ' << d2PhiI   << ' ' << intdens  << ' '  <<
                d3PhiI << ' ' << d3PhiIfd << '\n';
        }
        errLcR  = sqrt(errLcR / sumw);
        errLcI  = sqrt(errLcI / sumw);
        errRcR  = sqrt(errRcR / sumw);
        errRcI  = sqrt(errRcI / sumw);
        errRmR  = sqrt(errRmR / sumw);
        errRmI  = sqrt(errRmI / sumw);
        errRLR  = sqrt(errRLR / sumw);
        errRLI  = sqrt(errRLI / sumw);
        errPhiI = sqrt(errPhiI/ sumw);
        errdPhiI= sqrt(errdPhiI/sumw);
        errd2PhiI=sqrt(errd2PhiI/sumw);
        errd3PhiI=sqrt(errd3PhiI/sumw);
        // check if the errors are within design tolerance;
        // for the composite potential (aka GalPot = Multipole + DiskAnsatz) or Multipole
        // we loosen the tolerance limits, as the Multipole potential is intrinsically
        // only an approximation and not infinitely smooth, thus its interpolated version
        // is not required to be exceedingly accurate
        double tol =
            potential.name() == potential::CylSpline::myName() ? 5.0 :
            potential.name() == potential::Multipole::myName() ? 10. :
            potential.name().substr(0,9) == "Composite"        ? 200 : 1.;
        std::cout << "Density-weighted RMS errors"
        ": Phi(r)="     + checkLess(errPhiI,  1e-10 * tol, ok) +
        ", dPhi/dr="    + checkLess(errdPhiI, 1e-08 * tol, ok) +
        ", d2Phi/dr2="  + checkLess(errd2PhiI,1e-06 * tol, ok) +
        ", d3Phi/dr3="  + checkLess(errd3PhiI,1e-06 * tol, ok) +
        ", Lc,root(E)=" + checkLess(errLcR,   1e-12,       ok) +
        ", Lc,int(E)="  + checkLess(errLcI,   1e-08 * tol, ok) +
        ", Rc,root(E)=" + checkLess(errRcR,   1e-12,       ok) +
        ", Rc,int(E)="  + checkLess(errRcI,   1e-09 * tol, ok) +
        ", Rc,root(Lz)="+ checkLess(errRLR,   1e-12,       ok) +
        ", Rc,int(Lz)=" + checkLess(errRLI,   1e-09 * tol, ok) +
        ", Rm,root(E)=" + checkLess(errRmR,   2e-12,       ok) +
        ", Rm,int(E)="  + checkLess(errRmI,   1e-09 * tol, ok) + "\n";
    }
    catch(std::exception& e) {
        std::cout << "Cannot create interpolator: "<<e.what()<<"\n";
        ok = false;
    }
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

bool testPrincipalAxes(double radius, double p, double q, double alpha, double beta, double gamma)
{
    const coord::PosCar point(0.5, 0.6, 0.7);  // fiducial point for checking the rotated density
    const std::string params = "type=spheroid gamma=0 beta=4 alpha=1";
    potential::PtrDensity den0 = potential::createDensity(utils::KeyValueMap(params +
        " p=" + utils::toString(p, 16) +
        " q=" + utils::toString(q, 16) +
        (alpha+beta+gamma!=0 ? " orientation=" +
        utils::toString(alpha, 16) + "," +
        utils::toString(beta , 16) + "," +
        utils::toString(gamma, 16) : ""), " "));
    // normalize the range of input angles to alpha=-pi..pi, beta=0..pi/2, gamma=0..pi
    if(beta > M_PI/2) {
        beta = M_PI-beta;
        alpha= M_PI+alpha;
        gamma= M_PI-gamma;
    }
    alpha = fmod(alpha+M_PI, M_PI*2) - M_PI;
    gamma = fmod(gamma+M_PI, M_PI);
    potential::PtrDensity den1 = potential::createDensity(utils::KeyValueMap(params +
        " p=" + utils::toString(p, 16) +
        " q=" + utils::toString(q, 16) +
        " orientation=" +
        utils::toString(alpha, 16) + "," +
        utils::toString(beta , 16) + "," +
        utils::toString(gamma, 16), " "));
    // check that the above alterations of angles do not affect the value of the rotated density
    double ratio = den1->density(point) / den0->density(point);
    bool ok = fabs(ratio-1) < 1e-14;

    // determine the axis ratios and angles
    double axes[3], angles[3];
    principalAxes(*den0, radius, axes, angles);
    potential::PtrDensity den2 = potential::createDensity(utils::KeyValueMap(params +
        " p=" + utils::toString(axes[1]/axes[0], 16) +
        " q=" + utils::toString(axes[2]/axes[0], 16) +
        " orientation=" + utils::toString(angles[0], 16) + "," +
        utils::toString(angles[1], 16) + "," + utils::toString(angles[2], 16), " "));
    // check that the density constructed with the recovered angles is similar
    ratio = den2->density(point) / den0->density(point);
    double
    difden   = fabs(ratio-1),
    difp     = fabs(p - axes[1]/axes[0]),
    difq     = fabs(q - axes[2]/axes[0]),
    difbeta  = fabs(beta-angles[1]),
    difalpha = fmin(fabs(alpha-angles[0]), fabs(fabs(alpha-angles[0])-M_PI*2)),
    difgamma = fmin(fabs(gamma-angles[2]), fabs(fabs(gamma-angles[2])-M_PI)),
    epsden   = 1e-3,
    epsaxes  = 2e-4,
    epsangles= 1e-3;
    if(output) {
        std::cout << checkLess(difden, epsden, ok) + "\n"
            "p: " + utils::pp(p, 10) + " => " + utils::pp(axes[1]/axes[0], 10) +
            " (dif: " + checkLess(difp, epsaxes, ok) + ")\n"
            "q: " + utils::pp(q, 10) + " => " + utils::pp(axes[2]/axes[0], 10) +
            " (dif: " + checkLess(difq, epsaxes, ok) + ")\n"
            "a: " + utils::pp(alpha, 10) + " => " + utils::pp(angles[0], 10) +
            " (dif: " + checkLess(difalpha, epsangles, ok) + ")\n"
            "b: " + utils::pp(beta , 10) + " => " + utils::pp(angles[1], 10) +
            " (dif: " + checkLess(difbeta , epsangles, ok) + ")\n"
            "c: " + utils::pp(gamma, 10) + " => " + utils::pp(angles[2], 10) +
            " (dif: " + checkLess(difgamma, epsangles, ok) + ")\n\n";
    } else {  // no output, just comparison
        ok &= (difden < epsden) && (difp < epsaxes) && (difq < epsaxes) &&
            (difalpha < epsangles) && (difbeta < epsangles) && (difgamma < epsangles);
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

// save a few keystrokes
inline void addPot(std::vector<potential::PtrPotential>& pots, const char* params) {
    pots.push_back(potential::createPotential(utils::KeyValueMap(params))); }

int main() {
    bool allok=true;

    // test the recovery of shape and orientation
    allok &= testPrincipalAxes(INFINITY, 0.7, 0.4, 0, 0, 0);
    for(int i=0; i<100; i++) {
        allok &= testPrincipalAxes(0.5,   // radius
            (math::random()*0.4+0.6),     // intermediate axis
            (math::random()*0.3+0.3),     // minor axis
            (math::random()*2-1) * M_PI,  // orientation angles: alpha,
            (math::random()    ) * M_PI,  // beta,
            (math::random()*2-1) * M_PI); // gamma
    }
    /*for(int i=0; i<20; i++) {  // axisymmetric - TODO
        allok &= testPrincipalAxes(0.5, 1, math::random()*0.49+0.5, 0, 0, 0);
    }*/
    if(!allok)
        std::cout << "\033[1;31mDetermination of principal axes failed\033[0m\n";

    std::vector<potential::PtrPotential> pots;
    addPot(pots, "type=Plummer, mass=10, scaleRadius=5");
    addPot(pots, "type=Isochrone, mass=3, scaleRadius=2");
    addPot(pots, "type=NFW, mass=10, scaleRadius=10");
    addPot(pots, "type=MiyamotoNagai, mass=5, scaleRadius=2, scaleHeight=0.2");
    addPot(pots, "type=LongMurali, mass=4, scaleRadius=1, scaleHeight=0.2, barLength=3");
    addPot(pots, "type=Logarithmic, v0=2, scaleRadius=0.01, p=0.8, q=0.5");
    addPot(pots, "type=Ferrers, mass=1, scaleRadius=0.9, p=0.8, q=0.5");
    addPot(pots, "type=Dehnen, mass=2, scaleRadius=1, gamma=1.5");
    addPot(pots, "type=PerfectEllipsoid, q=0.6");
    addPot(pots, "type=Multipole, density=Spheroid, densityNorm=1e5, scaleRadius=1.234e-5, "
        "gamma=-2.0, beta=2.99, alpha=2.5, gridSizeR=64");
    addPot(pots, "density=Disk, surfaceDensity=1, scaleRadius=2, scaleHeight=-0.2, "
        "type=CylSpline, gridSizeR=20, rmin=0.1, rmax=500, gridSizeZ=20, zmin=0.01, zmax=50");
    pots.push_back(make_galpot(test_galpot_params[0]));
    pots.push_back(make_galpot(test_galpot_params[1]));
    //pots.push_back(make_galpot(test_galpot_params[2]));
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
