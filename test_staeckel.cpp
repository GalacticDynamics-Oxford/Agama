#include "potential_staeckel.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include <iostream>
#include <cmath>

int main() {

    const potential::StaeckelOblatePerfectEllipsoid poten(1.0, 1.6, 0.8);
    const double total_time=100.;
    const double timestep=1.;
    const coord::PosVelCar initcond(1, 0.5, 0.2, 0.1, 0.2, 0.3);
#if 1
    std::vector<coord::PosVelCar> traj;
    orbit::integrate(poten, initcond, total_time, timestep, traj);
#else
    std::vector<coord::PosVelCyl> traj;   // integration in cylindrical coords doesn't work yet
    orbit::integrate(poten, coord::toPosVelCyl(initcond), total_time, timestep, traj);
#endif
    for(size_t i=0; i<traj.size(); i++) {
        double xv[6];
        traj[i].unpack_to(xv);
        const actions::AxisymIntegrals Ints = actions::findIntegralsOfMotionOblatePerfectEllipsoid(poten, 
            coord::toPosVelCyl(traj[i]));
        std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   "<<
           Ints.H<<" "<<Ints.Lz<<" "<<Ints.I3<<"\n";
    }
    return 0;
}