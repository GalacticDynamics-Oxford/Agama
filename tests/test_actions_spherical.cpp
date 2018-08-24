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

const double epsR = 1e-6;  // required relative accuracy of Rperi,Rapo interpolation
const double epsJ = 1e-3;  // same for Jr
const char*  err  = "\033[1;31m ** \033[0m";

bool testPotential(const potential::BasePotential& pot)
{
    potential::Interpolator2d intp(pot);
    actions::ActionFinderSpherical af(pot);
    std::ofstream strm;
    if(utils::verbosityLevel >= utils::VL_VERBOSE)
        strm.open(("test_actions_spherical_"+std::string(pot.name())+".dat").c_str());
    math::Averager errR, errJ;
    for(double r=1e-8; r<1e8; r*=1.25) {
        double Phi;
        coord::GradCyl grad;
        pot.eval(coord::PosCyl(r,0,0), &Phi, &grad);
        double E  = Phi + 0.5*r*grad.dR;  // energy of a circular orbit at this radius
        double Lc = r*sqrt(r*grad.dR);    // ang.mom. of a circular orbit at this radius/this energy
        for(int i=0; i<=100; i++) {
            double w, L = math::unscale(math::ScalingQui(0,1), i*0.01, &w) * Lc;
            // test the accuracy of interpolation of peri/apocenter radii
            double R1s, R2s, R1i, R2i;
            potential::findPlanarOrbitExtent(pot, E, L, R1s, R2s);
            intp.findPlanarOrbitExtent(E, L, R1i, R2i);
            // test the accuracy of action interpolation
            coord::PosVelCyl point(r, 0, 0, sqrt(fmax(0, r*grad.dR - pow_2(L/r))), 0, L/r);
            double as = actions::actionsSpherical(pot, point).Jr;
            double ai = af.Jr(E, L);
            errR.add((R1s-R1i)/r * w);
            errR.add((R2s-R2i)/r * w);
            errJ.add((as-ai)/Lc  * w);
            if(utils::verbosityLevel >= utils::VL_VERBOSE)
                strm << utils::pp(E, 15) + '\t' + utils::pp( L/Lc, 15) + '\t' +
                    utils::pp(R1s/r, 15) + '\t' + utils::pp(R1i/r, 15) + '\t' +
                    utils::pp(R2s/r, 15) + '\t' + utils::pp(R2i/r, 15) + '\t' +
                    utils::pp(as/Lc, 15) + '\t' + utils::pp(ai/Lc, 15) + '\n';
        }
        if(utils::verbosityLevel >= utils::VL_VERBOSE)
            strm << '\n';
    }
    double rmsR = sqrt(pow_2(errR.mean()) + errR.disp());
    double rmsJ = sqrt(pow_2(errJ.mean()) + errJ.disp());
    bool okR = rmsR < epsR, okJ = rmsJ < epsJ;
    std::cout << pot.name() <<
        ": relative rms error in Rperi,Rapo = " << rmsR << (okR ? "" : err) <<
        ", in Jr = " << rmsJ << (okJ ? "" : err) << "\n";
    return okR && okJ;
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
    potential::CompositeCyl potc(pots);
    allok &= testPotential(potc);
    // just a typical case: a marginal central singularity
    addPot(pots, "type=Dehnen gamma=2");
    allok &= testPotential(*pots.back());
    // a very mild case
    addPot(pots, "type=Isochrone scaleradius=1e-3");
    allok &= testPotential(*pots.back());
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
