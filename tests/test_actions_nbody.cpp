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
    // some arbitrary internal units (note: the end result should not depend on their choice)
    const units::InternalUnits unit(0.2*units::Kpc, 0.2345*units::Myr);
    // input snapshot is in standard GADGET units
    const units::ExternalUnits extUnits(unit, units::Kpc, units::kms, 1e10*units::Msun);
    clock_t tbegin=std::clock();
    particles::PointMassArrayCar diskparticles, haloparticles;
    readSnapshot("temp/disk.gadget", extUnits, diskparticles);
    readSnapshot("temp/halo.gadget", extUnits, haloparticles);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to load snapshots;  "
        "disk mass=" << diskparticles.totalMass()*unit.to_Msun << " Msun, N=" << diskparticles.size() << ";  "
        "halo mass=" << haloparticles.totalMass()*unit.to_Msun << " Msun, N=" << haloparticles.size() <<"\n";
    tbegin=std::clock();
    const potential::BasePotential* halo = new potential::SplineExp(20, 2, haloparticles, potential::ST_AXISYMMETRIC);
    const potential::BasePotential* disk = new potential::CylSplineExp(20, 20, 0, diskparticles, potential::ST_AXISYMMETRIC);
    writePotential(std::string("disk") + getCoefFileExtension(*disk), *disk);
    writePotential(std::string("halo") + getCoefFileExtension(*halo), *halo);
    std::vector<const potential::BasePotential*> components(2);
    components[0] = halo;
    components[1] = disk;
    const potential::CompositeCyl poten(components);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init potential;  "
        "Potential at origin:  "
        "disk=" << disk->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2, "
        "halo=" << halo->value(coord::PosCar(0,0,0)) * pow_2(unit.to_kms) << " (km/s)^2\n";
    tbegin=std::clock();
    actions::ActionFinderAxisymFudge actFinder(poten);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init action finder\n";
    std::ofstream strm("disk_actions.txt");
    strm << "R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]\n";
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