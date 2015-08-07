#include "actions_staeckel.h"
#include "actions_torus.h"
#include "potential_factory.h"
#include "units.h"
#include "debug_utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <ctime>
#include <vector>

const units::InternalUnits unit(units::galactic_Myr);//(1.*units::Kpc, 977.8*units::Myr);
const unsigned int NUM_ANGLE_SAMPLES = 64;
const double NUM_ANGLE_PERIODS = 4;
const bool output = true;

bool test_actions(const potential::BasePotential& poten,
    const actions::BaseActionFinder& finder, const actions::BaseActionMapper& mapper, const actions::Actions actions)
{
    actionstat acts;
    anglestat angs;
    for(unsigned int i=0; i<NUM_ANGLE_SAMPLES; i++) {
        actions::Angles angles;
        angles.thetar   = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 6.27 );
        angles.thetaz   = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 8.45 );
        angles.thetaphi = math::wrapAngle( i*NUM_ANGLE_PERIODS/NUM_ANGLE_SAMPLES * 4.96 );
        coord::PosVelCyl xv = mapper.map(actions::ActionAngles(actions, angles));
        actions::ActionAngles aa = finder.actionAngles(xv);
        angs.add(i*1.0, aa);
        acts.add(aa);
        if(output)
            std::cout << "Point: " << xv << "Energy: "<<totalEnergy(poten, xv)<<
            "\nOrig:  " << actions << angles << "\nFudge: " << aa << "\n";
    }
    acts.finish();
    angs.finish();
    double scatter = (acts.disp.Jr+acts.disp.Jz) / (acts.avg.Jr+acts.avg.Jz);
    double scatterNorm = 0.33 * sqrt( (acts.avg.Jr+acts.avg.Jz) / (acts.avg.Jr+acts.avg.Jz+fabs(acts.avg.Jphi)) );
    bool tolerable = scatter < scatterNorm && 
        angs.dispr < 0.1 && angs.dispz < 1.0 && angs.dispphi < 0.05;
    const double dim = unit.to_Kpc*unit.to_Kpc/unit.to_Myr;//unit.to_Kpc_kms;
    std::cout << 
        acts.avg.Jr*dim <<" "<< acts.disp.Jr*dim <<" "<< 
        acts.avg.Jz*dim <<" "<< acts.disp.Jz*dim <<" "<< 
        acts.avg.Jphi*dim <<" "<< acts.disp.Jphi*dim <<"  "<< 
        angs.freqr <<" "<< angs.freqz <<" "<< angs.freqphi <<"  "<<
        angs.dispr <<" "<< angs.dispz <<" "<< angs.dispphi <<"  "<<
        std::endl;
    return tolerable;
}

const potential::BasePotential* make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    const potential::BasePotential* gp = potential::readGalaxyPotential(params_file, unit);
    std::remove(params_file);
    if(gp==NULL)
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
    const potential::BasePotential* pot;
    utils::KeyValueMap params(argc, argv);
    if(argc>1) {
        potential::ConfigPotential config;
        potential::parseConfigPotential(params, config);
        pot = potential::createPotential(config);
    } else
        pot = make_galpot(test_galpot_params);
    double Jr   = params.getDouble("Jr", 100);
    double Jz   = params.getDouble("Jz", 100);
    double Jphi = params.getDouble("Jphi", 1000);
    actions::Actions acts;
    acts.Jr   = Jr   * unit.from_Kpc_kms;
    acts.Jz   = Jz   * unit.from_Kpc_kms;
    acts.Jphi = Jphi * unit.from_Kpc_kms;
    actions::ActionMapperTorus mapper(*pot, acts);
    actions::ActionFinderAxisymFudge finder(*pot);
    allok &= test_actions(*pot, finder, mapper, acts);
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    delete pot;
    return 0;
}