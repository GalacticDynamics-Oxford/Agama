/** \file    test_actions_nbody.cpp
    \author  Eugene Vasiliev
    \date    August 2015

    This example demonstrates the use of action finder (Staeckel fudge approximation)
    to compute actions for particles from an N-body simulation.
    The N-body system consists of a disk and a halo, the two components being
    stored in separate GADGET files.
    The potential is computed from the snapshot itself, by creating a suitable
    potential expansion for each component: SplineExp for the halo and CylSplineExp
    for the disk. This actually takes most of the time.
    Then we compute actions for all particles from the disk component,
    and store them in a text file.

    An equivalent example in Python is located in pytests folder;
    it uses the same machinery through a Python extension of the C++ library.
*/
#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include "units.h"
#include <iostream>
#include <fstream>
#include <ctime>

int main() {
    // #1. Set up units.
    // some arbitrary internal units (note: the end result should not depend on their choice)
    const units::InternalUnits unit(units::Kpc, units::Myr);
    // input snapshot is in standard GADGET units
    const units::ExternalUnits extUnits(unit, units::Kpc, units::kms, 1e10*units::Msun);

    // #2. Get in N-body snapshots
    clock_t tbegin=std::clock();
    particles::PointMassArrayCar diskparticles, haloparticles;
    readSnapshot("../temp/disk.gadget", extUnits, diskparticles);
    readSnapshot("../temp/halo.gadget", extUnits, haloparticles);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to load snapshots;  "
        "disk mass=" << diskparticles.totalMass()*unit.to_Msun << " Msun, N=" << diskparticles.size() << ";  "
        "halo mass=" << haloparticles.totalMass()*unit.to_Msun << " Msun, N=" << haloparticles.size() <<"\n";

    // #3. Initialize potential approximations from these particles
    tbegin=std::clock();
    const potential::BasePotential* halo = new potential::SplineExp
        (20, 2, haloparticles, potential::ST_AXISYMMETRIC, 1.0 /*default smoothfactor*/);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init halo potential;  "
        "value at origin=" << halo->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2\n";
    tbegin=std::clock();
    const potential::BasePotential* disk = new potential::CylSplineExp
        (20, 20, 0, diskparticles, potential::ST_AXISYMMETRIC);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init disk potential;  "
        "value at origin=" << disk->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2\n";
    // not necessary, but we may store the potential coefs into a file and then load them back to speed up process
    writePotentialCoefs(std::string("disk") + getCoefFileExtension(*disk), *disk);
    writePotentialCoefs(std::string("halo") + getCoefFileExtension(*halo), *halo);

    // #3a. Combine the two components
    std::vector<const potential::BasePotential*> components(2);
    components[0] = halo;
    components[1] = disk;
    const potential::CompositeCyl poten(components);

    // #4. Compute actions
    tbegin=std::clock();
    actions::ActionFinderAxisymFudge actFinder(poten);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init action finder\n";
    std::ofstream strm("disk_actions.txt");
    strm << "# R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]\n";
    tbegin=std::clock();
    unsigned int numBadPoints = 0;
    for(size_t i=0; i<diskparticles.size(); i++) {
        try{
            const coord::PosVelCyl point = toPosVelCyl(diskparticles[i].first);
            actions::Actions acts = actFinder.actions(point);
            strm << point.R*unit.to_Kpc << "\t" << point.z*unit.to_Kpc << "\t" <<
                acts.Jr*unit.to_Kpc_kms << "\t" << acts.Jz*unit.to_Kpc_kms << "\t" << acts.Jphi*unit.to_Kpc_kms << "\t" << 
                totalEnergy(poten, point)*pow_2(unit.to_kms) << "\n";
        }
        catch(...){
            numBadPoints++;  // probably because energy is positive
        }
    }
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to compute actions  ("<<
        diskparticles.size() * 1.0*CLOCKS_PER_SEC / (std::clock()-tbegin) << " actions per second)";
    if(numBadPoints>0)
        std::cout << ";  " << numBadPoints << " points skipped";
    std::cout << "\nALL TESTS PASSED\n";
    return 0;
}