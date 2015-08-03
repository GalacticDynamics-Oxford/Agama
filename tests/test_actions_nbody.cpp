#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_composite.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include <iostream>
#include <fstream>
#include <ctime>

//#include "units.h"
//const units::Units unit(0.2*units::Kpc, 100*units::Myr);
int main() {
    clock_t tbegin=std::clock();
    particles::PointMassArrayCar diskparticles, haloparticles;
    particles::readSnapshot("temp/halo.nemo", haloparticles);
    particles::readSnapshot("temp/disk.txt", diskparticles);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to load snapshots;  "
        "Ndisk=" << diskparticles.size() << ", Nhalo=" << haloparticles.size() <<" particles\n";
    tbegin=std::clock();
    const potential::BasePotential* halo = new potential::SplineExp(20, 2, haloparticles, potential::ST_AXISYMMETRIC);
    const potential::BasePotential* disk = new potential::CylSplineExp(20, 20, 0, diskparticles, potential::ST_AXISYMMETRIC);
    std::vector<const potential::BasePotential*> components(2);
    components[0] = halo;
    components[1] = disk;
    const potential::CompositeCyl poten(components);
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to init potential;  "
        "Potential at origin: "
        "  disk=" << potential::value(*disk, coord::PosCar(0,0,0)) <<
        ", halo=" << potential::value(*halo, coord::PosCar(0,0,0)) << "\n";
    actions::ActionFinderAxisymFudge actfinder(poten);
    std::ofstream strm("disk_actions.txt");
    strm << "R\tz\tJ_r\tJ_z\tJ_phi\tE\n";
    tbegin=std::clock();
    for(size_t i=0; i<diskparticles.size(); i++) {
        const coord::PosVelCyl point = coord::toPosVelCyl(diskparticles[i].first);
        actions::Actions acts = actfinder.actions(point);
        strm << point.R << "\t" << point.z << "\t" <<
            acts.Jr << "\t" << acts.Jz << "\t" << acts.Jphi << "\t" << potential::totalEnergy(poten, point) << "\n";
    }
    std::cout << (std::clock()-tbegin)*1.0/CLOCKS_PER_SEC << " s to compute actions  ("<<
        diskparticles.size() * 1.0*CLOCKS_PER_SEC / (std::clock()-tbegin) << " actions per second)\n";
    return 0;
}