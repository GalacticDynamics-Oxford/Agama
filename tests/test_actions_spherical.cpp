/** \file    test_actions_spherical.cpp
    \author  Eugene Vasiliev
    \date    2018

    This program tests the accuracy of the interpolated peri/apocenter finder
    and the interpolated spherical action finder.
    For each trial potential, we create a grid in radius (energy) and L/Lcirc(E)
    and compare the values of non-interpolated vs. interpolated routines;
    the accuracy is expected to be somewhat worse towards the endpoints
    (purely radial or circular orbits), but the weighted average difference
    should be smaller than the predefined limit.
*/
#include "actions_spherical.h"
#include "potential_utils.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "math_core.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

const double epsR = 1e-9;  // required relative accuracy of Rperi,Rapo interpolation
const double epsJ = 1e-6;  // same for Jr(E,L)
const double epsE = 1e-6;  // same for E(Jr,L)
const char*  err  = "\033[1;31m ** \033[0m";

bool testPotential(const potential::BasePotential& pot)
{
    potential::Interpolator2d intp(pot);
    actions::ActionFinderSpherical af(pot);
    std::ofstream strm, strE;
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open(("test_actions_spherical_"+std::string(pot.name())+".dat").c_str());
        strm << "# Energy       \tL/Lcirc\tRperi/Rcir,root\tRperi/Rcirc,int\t"
            "Rapo/Rcirc,root\tRapo/Rcirc,int \tJr/Lcirc,true  \tJr/Lcirc,interp\n";
        strE.open(("test_actions_spherical_"+std::string(pot.name())+"_energy.dat").c_str());
        strE << "# Lcirc\tL/Lcirc\tE(L,Jr)/Ecirc,r\tE(L,Jr)/Ecirc/i\n";
    }
    double Phi0 = pot.value(coord::PosCyl(0,0,0));
    math::Averager errR, errJ, errE;
    std::vector<double> gridLogR = math::createSymmetricGrid(201, 0.1, 15.);
    double sumwr = 0;   // integral of density over radius, used to normalize density-weighted errors
    for(size_t n=0; n<gridLogR.size(); n++) {
        double r = pow(10., gridLogR[n]);
        double Phi;
        coord::GradCyl grad;
        pot.eval(coord::PosCyl(r,0,0), &Phi, &grad);
        if(Phi<=Phi0*(1-1e-11))
            continue;  // roundoff
        double E  = Phi + 0.5*r*grad.dR;  // energy of a circular orbit at this radius
        double Lc = r*sqrt(r*grad.dR);    // ang.mom. of a circular orbit at this radius/this energy
        double wr = pot.density(coord::PosCyl(r,0,0)) * pow_3(r);  // density-weighted radial integration
        sumwr += wr;
        for(int i=0; i<=100; i++) {
            double wl, L = math::unscale(math::ScalingQui(0,1), i*0.01, &wl) * Lc, Jr = Lc-L;
            // test the accuracy of interpolation of peri/apocenter radii
            double R1s, R2s, R1i, R2i;
            potential::findPlanarOrbitExtent(pot, E, L, R1s, R2s);
            intp.findPlanarOrbitExtent(E, L, R1i, R2i);
            // test the accuracy of action interpolation
            coord::PosVelCyl point(r, 0, 0, sqrt(fmax(0, r*grad.dR - pow_2(L/r))), 0, L/r);
            actions::Actions acs;
            actions::evalSpherical(pot, point, &acs);
            double as = acs.Jr;
            double ai = af.Jr(E, L);
            // test the accuracy of computing energy from actions
            double Es = actions::computeHamiltonianSpherical(pot, actions::Actions(Jr, L, 0));
            double Ei = af.E(actions::Actions(Jr, L, 0));
            errR.add((R1s-R1i)/r * wl * wr);
            errR.add((R2s-R2i)/r * wl * wr);
            errJ.add((as-ai)/Lc  * wl * wr);
            errE.add((Ei/Es-1)   * wl * wr);
            if(utils::verbosityLevel >= utils::VL_VERBOSE) {
                strm << utils::pp(E, 15) + '\t' + utils::pp( L/Lc,  7) + '\t' +
                    utils::pp(R1s/r, 15) + '\t' + utils::pp(R1i/r, 15) + '\t' +
                    utils::pp(R2s/r, 15) + '\t' + utils::pp(R2i/r, 15) + '\t' +
                    utils::pp(as/Lc, 15) + '\t' + utils::pp(ai/Lc, 15) + '\n';
                strE << utils::pp(Lc, 7) + '\t' + utils::pp(L/Lc,   7) + '\t' +
                    utils::pp(Es/E,  15) + '\t' + utils::pp(Ei/E,  15) + '\n';
            }
        }
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm << '\n';
            strE << '\n';
        }
    }
    double rmsR = sqrt(pow_2(errR.mean()) + errR.disp()) / sumwr;
    double rmsJ = sqrt(pow_2(errJ.mean()) + errJ.disp()) / sumwr;
    double rmsE = sqrt(pow_2(errE.mean()) + errE.disp()) / sumwr;
    bool okR = rmsR < epsR, okJ = rmsJ < epsJ, okE = rmsE < epsE;
    std::cout << pot.name() <<
        ": relative rms error in Rperi,Rapo = " << rmsR << (okR ? "" : err) <<
        ", in Jr(E,L) = " << rmsJ << (okJ ? "" : err) <<
        ", in E(Jr,L) = " << rmsE << (okE ? "" : err) << "\n";
    return okR && okJ && okE;
}

inline void addPot(std::vector<potential::PtrPotential>& pots, const char* params) {
    pots.push_back(potential::createPotential(utils::KeyValueMap(params))); }

int main()
{
    bool allok = true;
    std::vector<potential::PtrPotential> pots;
    // a tough collection of assorted troubles: central singularity and shallow outer density fall-off
    addPot(pots, "type=Plummer mass=1e15 scaleradius=1000");
    addPot(pots, "type=Plummer mass=1e6 scaleradius=0");  // a point mass
    addPot(pots, "type=Spheroid gamma=2.3 beta=2.7 alpha=2 scaleRadius=100000 densitynorm=1");
    allok &= testPotential(potential::Composite(pots));
    // a very shallow potential at small radii
    addPot(pots, "type=Spheroid gamma=-2.0 beta=4.0 alpha=2.0 densitynorm=0.3183098861837907");
    allok &= testPotential(*pots.back());
    // just a typical case: a marginal central singularity
    addPot(pots, "type=Dehnen gamma=2");
    allok &= testPotential(*pots.back());
    // another typical nasty case: logarithmically-diverging total mass
    addPot(pots, "type=NFW");
    allok &= testPotential(*pots.back());
    // a very mild case (cored density)
    addPot(pots, "type=Isochrone scaleradius=1e-3");
    allok &= testPotential(*pots.back());
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
