#include "potential_staeckel.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include <iostream>
#include <cmath>

const double integr_eps=1e-6;  // integration accuracy parameter
const double eps=1e-6;  // accuracy of comparison

template<typename coordSysT>
bool test_oblate_staeckel(const potential::StaeckelOblatePerfectEllipsoid& potential,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep, const bool output)
{
    std::vector<coord::PosVelT<coordSysT> > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    actions::AxisymIntegrals intAvg, intDisp;
    intAvg.H=intAvg.Lz=intAvg.I3=intDisp.H=intDisp.Lz=intDisp.I3=0;
    for(size_t i=0; i<traj.size(); i++) {
        const actions::AxisymIntegrals ints =
            actions::findIntegralsOfMotionOblatePerfectEllipsoid(potential, coord::toPosVelCyl(traj[i]));
        intAvg.H +=ints.H;  intDisp.H +=pow_2(ints.H);
        intAvg.Lz+=ints.Lz; intDisp.Lz+=pow_2(ints.Lz);
        intAvg.I3+=ints.I3; intDisp.I3+=pow_2(ints.I3);
        if(output) {
            double xv[6];
            coord::toPosVelCar(traj[i]).unpack_to(xv);
            std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   "<<
                ints.H<<" "<<ints.Lz<<" "<<ints.I3<<"\n";
        }
    }
    intAvg.H/=traj.size();
    intAvg.Lz/=traj.size();
    intAvg.I3/=traj.size();
    intDisp.H/=traj.size();
    intDisp.Lz/=traj.size();
    intDisp.I3/=traj.size();
    intDisp.H= sqrt(std::max<double>(0, intDisp.H -pow_2(intAvg.H)));
    intDisp.Lz=sqrt(std::max<double>(0, intDisp.Lz-pow_2(intAvg.Lz)));
    intDisp.I3=sqrt(std::max<double>(0, intDisp.I3-pow_2(intAvg.I3)));
    std::cout << coord::CoordSysName<coordSysT>() << 
    ":  E=" <<intAvg.H <<" +- "<<intDisp.H<<
    ",  Lz="<<intAvg.Lz<<" +- "<<intDisp.Lz<<
    ",  I3="<<intAvg.I3<<" +- "<<intDisp.I3<<"\n";
    return intDisp.H<eps && intDisp.Lz<eps && intDisp.I3<eps;
}

int main() {
    const potential::StaeckelOblatePerfectEllipsoid poten(1.0, 1.6, 0.8);
    const coord::PosVelCar initcond(1, 0.5, 0.2, 0.1, 0.2, 0.3);
    const double total_time=100.;
    const double timestep=1./8;
    bool allok=true;
    bool output=false;
    allok &= test_oblate_staeckel(poten, coord::toPosVelCar(initcond), total_time, timestep, output);
    allok &= test_oblate_staeckel(poten, coord::toPosVelCyl(initcond), total_time, timestep, output);
    allok &= test_oblate_staeckel(poten, coord::toPosVelSph(initcond), total_time, timestep, output);
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}