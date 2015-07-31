#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_composite.h"
#include "particles_io.h"
#include "actions_staeckel.h"
#include <iostream>
#include <fstream>
//#include "units.h"
//const units::Units unit(0.2*units::Kpc, 100*units::Myr);
int main() {
    particles::PointMassArrayCar diskparticles, haloparticles;
    particles::readSnapshot("temp/halo.nemo", haloparticles);
    particles::readSnapshot("temp/disk.txt", diskparticles);
    std::cout << "Disk: " << diskparticles.size() << ", halo: " << haloparticles.size() <<" particles\n";
    const potential::BasePotential* halo = new potential::SplineExp(20, 2, haloparticles, potential::ST_AXISYMMETRIC);
    const potential::BasePotential* disk = new potential::CylSplineExp(20, 20, 0, diskparticles, potential::ST_AXISYMMETRIC);
    std::vector<const potential::BasePotential*> components(2);
    components[0] = halo;
    components[1] = disk;
    const potential::CompositeCyl poten(components);
    std::cout << "Potential at origin: "
        "  disk=" << potential::value(*disk, coord::PosCar(0,0,0)) <<
        ", halo=" << potential::value(*halo, coord::PosCar(0,0,0)) << "\n";
    std::ofstream strm("disk_actions.txt");
    strm << "R\tz\tE\tJ_r\tJ_z\tJ_phi\n";
    for(size_t i=0; i<diskparticles.size(); i++) {
        const coord::PosVelCyl point = coord::toPosVelCyl(diskparticles[i].first);
        actions::Actions acts = actions::axisymFudgeActions(poten, point);
        strm << point.R << "\t" << point.z << "\t" << potential::totalEnergy(poten, point) << "\t" <<
            acts.Jr << "\t" << acts.Jz << "\t" << acts.Jphi << "\n";
    }
    return 0;
}