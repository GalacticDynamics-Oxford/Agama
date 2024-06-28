/** \file    test_actions_torus.cpp
    \date    August 2015
    \author  Eugene Vasiliev

    This example demonstrates the conversion between action/angle and position/velocity
    phase spaces, in both directions, using the Staeckel fudge and the Torus machinery.

    We create an instance of potential and a Torus, which is a complete description
    of a particular orbit, specified by three actions.
    Then we generate the trajectory using the angle mapping from the torus,
    without actually doing orbit integration in the real potential
    (the accuracy of such representation is checked elsewhere).
    For each point on the trajectory, we compute the actions and angles using
    the Staeckel fudge, and compare them to the values used to create the torus.
    They should agree to within reasonable limits, demonstrating the accuracy
    of action recovery by the fudge approximation (we assume that the torus
    provides 'exact' values, and the fudge returns an approximation to these).
*/
#include "actions_staeckel.h"
#include "actions_torus.h"
#include "potential_factory.h"
#include "units.h"
#include "utils.h"
#include "debug_utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <vector>

const units::InternalUnits unit(units::galactic_Myr);//(1.*units::Kpc, 977.8*units::Myr);
const unsigned int NUM_ANGLE_SAMPLES = 64;
const double NUM_ANGLE_PERIODS = 4;
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

bool test_actions(const potential::BasePotential& poten,
    const actions::BaseActionFinder& finder, const actions::BaseActionMapper& mapper, const actions::Actions actions)
{
    actions::ActionStat acts;
    actions::AngleStat  angs;
    actions::Frequencies freq;
    actions::Actions newactions;
    actions::Angles angles, newangles;
    angles.thetar = angles.thetaz = angles.thetaphi = 0;
    coord::PosVelCyl xv = mapper.map(actions::ActionAngles(actions, angles), &freq);  // obtain the values of frequencies
    double fr0 = fmax(freq.Omegar, fmax(freq.Omegaz, freq.Omegaphi));
    for(unsigned int i=0; i<NUM_ANGLE_SAMPLES; i++) {
        angles.thetar   = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 2*M_PI * freq.Omegar/fr0 );
        angles.thetaz   = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 2*M_PI * freq.Omegaz/fr0 );
        angles.thetaphi = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 2*M_PI * freq.Omegaphi/fr0 );
        xv = mapper.map(actions::ActionAngles(actions, angles));
        finder.eval(xv, &newactions, &newangles);
        angs.add(i*1.0, newangles);
        acts.add(newactions);
        if(output)
            std::cout << "Point: " << xv << "Energy: "<<totalEnergy(poten, xv)<<
            "\nOrig:  " << actions << angles << "\nFudge: " << newactions << newangles << "\n";
    }
    acts.finish();
    angs.finish();
    double scatter = (acts.rms.Jr+acts.rms.Jz) / (acts.avg.Jr+acts.avg.Jz);
    double scatterNorm = 0.33 * sqrt( (acts.avg.Jr+acts.avg.Jz) / (acts.avg.Jr+acts.avg.Jz+fabs(acts.avg.Jphi)) );
    bool tolerable = scatter < scatterNorm && 
        angs.dispr < 0.1 && angs.dispz < 1.0 && angs.dispphi < 0.05;
    const double dim = unit.to_Kpc_kms;
    std::cout << 
        acts.avg.Jr*dim <<" "<< acts.rms.Jr*dim <<" "<< 
        acts.avg.Jz*dim <<" "<< acts.rms.Jz*dim <<" "<< 
        acts.avg.Jphi*dim <<" "<< acts.rms.Jphi*dim <<"  "<< 
        angs.freqr <<" "<< angs.freqz <<" "<< angs.freqphi <<"  "<<
        angs.dispr <<" "<< angs.dispz <<" "<< angs.dispphi <<"  "<<
        std::endl;
    return tolerable;
}

potential::PtrPotential make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    potential::PtrPotential gp = potential::readGalaxyPotential(params_file, unit);
    std::remove(params_file);
    if(gp.get()==NULL)
        std::cout<<"Potential not created\n";
    return gp;
}

const char* test_galpot_params =
// BestFitPotential.Tpot
"3\n"
"5.63482e+08 2.6771 0.1974 0 0\n"
"2.51529e+08 2.6771 0.7050 0 0\n"
"9.34513e+07 5.3542 0.04 4 0\n"
"2\n"
"9.49e+10    0.5  0  1.8  0.075   2.1\n"
"1.85884e+07 1.0  1  3    14.2825 250.\n";

int main(int argc, const char* argv[]) {
    bool allok = true;
    potential::PtrPotential pot;
    utils::KeyValueMap params(argc, argv);
    if(argc==2 && std::string(argv[1]).find('=')==std::string::npos)
    {   // probably passed the ini file name
        pot = potential::readPotential(argv[1], units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun));
    } else if(argc>1 && std::string(argv[1]).find('=')!=std::string::npos)
    {   // probably passed several key=value parameters
        pot = potential::createPotential(params, units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun));
    } else
        pot = make_galpot(test_galpot_params);
    double Jr   = params.getDouble("Jr",   100);
    double Jz   = params.getDouble("Jz",   100);
    double Jphi = params.getDouble("Jphi", 1000);
    actions::Actions acts;
    acts.Jr   = Jr   * unit.from_Kpc_kms;
    acts.Jz   = Jz   * unit.from_Kpc_kms;
    acts.Jphi = Jphi * unit.from_Kpc_kms;
    actions::ActionMapperTorus mapper(pot);
    actions::ActionFinderAxisymFudge finder(pot, false);
    allok &= test_actions(*pot, finder, mapper, acts);
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}