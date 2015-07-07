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
    actions::ActionFinderAxisymmetricStaeckel actionfinder(potential);
    actions::Actions actAvg, actDisp;
    actAvg.Jr=actAvg.Jz=actAvg.Jphi=actDisp.Jr=actDisp.Jz=actDisp.Jphi=0;
    for(size_t i=0; i<traj.size(); i++) {
        const actions::Actions acts = actionfinder.actions(coord::toPosVelCar(traj[i]));
        actAvg.Jr  +=acts.Jr;   actDisp.Jr  +=pow_2(acts.Jr);
        actAvg.Jz  +=acts.Jz;   actDisp.Jz  +=pow_2(acts.Jz);
        actAvg.Jphi+=acts.Jphi; actDisp.Jphi+=pow_2(acts.Jphi);
        if(output) {
            double xv[6];
            coord::toPosVelCar(traj[i]).unpack_to(xv);
            std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   "<<
                acts.Jr<<" "<<acts.Jz<<" "<<acts.Jphi<<"\n";
        }
    }
    actAvg.Jr/=traj.size();
    actAvg.Jz/=traj.size();
    actAvg.Jphi/=traj.size();
    actDisp.Jr/=traj.size();
    actDisp.Jz/=traj.size();
    actDisp.Jphi/=traj.size();
    actDisp.Jr  =sqrt(std::max<double>(0, actDisp.Jr  -pow_2(actAvg.Jr)));
    actDisp.Jz  =sqrt(std::max<double>(0, actDisp.Jz  -pow_2(actAvg.Jz)));
    actDisp.Jphi=sqrt(std::max<double>(0, actDisp.Jphi-pow_2(actAvg.Jphi)));
    std::cout << coord::CoordSysName<coordSysT>() << 
    ":  Jr="  <<actAvg.Jr  <<" +- "<<actDisp.Jr<<
    ",  Jz="  <<actAvg.Jz  <<" +- "<<actDisp.Jz<<
    ",  Jphi="<<actAvg.Jphi<<" +- "<<actDisp.Jphi<<"\n";
    return actDisp.Jr<eps && actDisp.Jz<eps && actDisp.Jphi<eps;
}

int main() {
    const potential::StaeckelOblatePerfectEllipsoid poten(1.0, 1.6, 1.0);
    //const coord::PosVelCar initcond(1, 0.5, 0.2, 0.1, 0.2, 0.3);
    //const coord::PosVelCar initcond(1, 0.3, 0.1, 0.1, 0.4, 0.1);
    const coord::PosVelCar initcond(1, 0.3, 0., 0.1, 0.4, 1e-4);
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