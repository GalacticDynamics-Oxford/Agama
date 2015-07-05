#include "orbit.h"
#include "potential_analytic.h"
#include "potential_galpot.h"
#include "units.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const double integr_eps=1e-6;  // integration accuracy parameter
const double eps=1e-6;  // accuracy of comparison

template<typename coordSysT>
bool test_orbit_int(const potential::BasePotential& potential,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep, const bool output)
{
    std::vector<coord::PosVelT<coordSysT> > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    double avgH=0, avgLz=0, dispH=0, dispLz=0;
    for(size_t i=0; i<traj.size(); i++) {
        double H =potential::totalEnergy(potential, traj[i]);
        double Lz=coord::Lz(traj[i]);
        avgH +=H;  dispH +=pow_2(H);
        avgLz+=Lz; dispLz+=pow_2(Lz);
        if(output) {
            double xv[6];
            coord::toPosVelCar(traj[i]).unpack_to(xv);
            std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   "<<H<<" "<<Lz<<" "<<potential.density(traj[i])<<"\n";
        }
    }
    avgH/=traj.size();
    avgLz/=traj.size();
    dispH/=traj.size();
    dispLz/=traj.size();
    dispH= sqrt(std::max<double>(0, dispH -pow_2(avgH)));
    dispLz=sqrt(std::max<double>(0, dispLz-pow_2(avgLz)));
    std::cout << coord::CoordSysName<coordSysT>() << 
    ":  E=" <<avgH <<" +- "<<dispH<<
    ",  Lz="<<avgLz<<" +- "<<dispLz<<"\n";
    return dispH<eps && dispLz<eps;
}
bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelCar& initcond,
    const double total_time, const double timestep, const bool output)
{
    bool ok=true;
    ok&= test_orbit_int(potential, coord::toPosVelCar(initcond), total_time, timestep, output);
//    ok&= test_orbit_int(potential, coord::toPosVelCyl(initcond), total_time, timestep, output);
//    ok&= test_orbit_int(potential, coord::toPosVelSph(initcond), total_time, timestep, output);
    return ok;
}

int main() {
    std::cout<<std::setprecision(10);
    const coord::PosVelCar initcond(1, 0.5, 0.2, 0.1, 0.2, 0.3);
    const double total_time=1000.;
    const double timestep=1.;
    bool allok=true;
    bool output=true;
//    allok &= test_potential(potential::Logarithmic(1.,0.,1,.6), initcond, total_time, timestep, output);
    //potential::DiskPar disk1={ .1, 3., 0.3, 0, 0 };
    //potential::SphrPar sphr1={ .01, 0.8, 1., 4., 5., 0 };
    //potential::GalaxyPotential gp(std::vector<potential::DiskPar>(1, disk1), std::vector<potential::SphrPar>(1, sphr1));
    const potential::BasePotential* gp = potential::readGalaxyPotential("GSM_potential.pot", units::galactic_Myr);
    if(gp==NULL) { 
        std::cout<<"Potential not created\n";
        return 0;
    }
    allok &= test_potential(*gp, initcond, total_time, timestep, output);
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}