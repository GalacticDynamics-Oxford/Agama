/** \file    test_staeckel.cpp
    \author  Eugene Vasiliev
    \date    July 2015

    This example shows the correctness of action/angle determination by the Staeckel fudge.

    We create an instance of Staeckel potential (in this example, oblate perfect ellipsoid),
    perform numerical orbit integration in this potential, and for each point
    on the trajectory, compute the values of actions and angles.
    Two related methods are employed - the first is valid only for a Staeckel potential,
    and the second (fudge) is an approximation which essentially turns out to be exact
    for this potential.
    We test that the values of actions are nearly constant (to the limit of orbit
    integration accuracy), and that angles increase linearly with time.
    We use several initial conditions corresponding to various generic and extreme cases
    (e.g., an in-plane orbit, or an orbit with very small radial action, or an orbit
    close to the separatrix between box and tube orbits in x-z plane).
*/
#include "potential_perfect_ellipsoid.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include "math_core.h"
#include "debug_utils.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

const double integr_eps=1e-8;        // integration accuracy parameter
const double eps=1e-7;               // accuracy of comparison
const double axis_a=1.6, axis_c=1.0; // axes of perfect ellipsoid
const bool output=utils::verbosityLevel >= utils::VL_VERBOSE;  // whether to create text files with orbits

template<typename coordSysT>
bool test_oblate_staeckel(const potential::OblatePerfectEllipsoid& potential,
    const actions::InterfocalDistanceFinder& ifdfinder,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep)
{
    std::vector<coord::PosVelT<coordSysT> > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    actions::ActionStat stats, statf;
    actions::Angles angf;
    bool ex_afs=false, ex_aff=false;
    std::ofstream strm;
    if(output) {
        std::ostringstream s;
        double x[6];
        initial_conditions.unpack_to(x);
        s<<coordSysT::name()<<"_"<<x[0]<<x[1]<<x[2]<<x[3]<<x[4]<<x[5];
        strm.open(s.str().c_str());
    }
    // two estimates of interfocal distance: from the trajectory
    // (sensible unless z==0 everywhere, in which case its value doesn't matter anyway)
    // and from the interpolator
    double ifd_p = actions::estimateInterfocalDistancePoints(potential, traj);
    double ifd_i = ifdfinder.value(totalEnergy(potential, traj[0]), Lz(traj[0]));
    for(size_t i=0; i<traj.size(); i++) {
        const coord::PosVelCyl p = coord::toPosVelCyl(traj[i]);
        try {
            actions::ActionAngles a = actions::actionAnglesAxisymStaeckel(potential, p);
            stats.add(a);
        }
        catch(std::exception &e) {
            if(!ex_afs) std::cout << "Exception in Staeckel at i="<<i<<": "<<e.what()<<"\n";
            ex_afs=true;
        }
        try {
            actions::ActionAngles a = actions::actionAnglesAxisymFudge(potential, p, ifd_p);
            statf.add(a);
            if(1 || i==0) angf=a;  // 1 to disable unwrapping
            else {
                angf.thetar   = math::unwrapAngle(a.thetar, angf.thetar);
                angf.thetaz   = math::unwrapAngle(a.thetaz, angf.thetaz);
                angf.thetaphi = math::unwrapAngle(a.thetaphi, angf.thetaphi);
            }
            if(output) {
                const coord::PosVelCyl pc = coord::toPosVelCyl(traj[i]);
                const coord::PosVelProlSph pp = coord::toPosVel<coord::Cyl,coord::ProlSph>
                    (pc, potential.coordsys());
                strm << i*timestep<<"   "<<
                    pc.phi<<" "<<pc.vphi<<" "<<pp.lambda<<" "<<pp.nu<<" "<<pp.lambdadot<<" "<<pp.nudot<<"  "<<
                    angf.thetar<<" "<<angf.thetaz<<" "<<angf.thetaphi<<"  "<<
                "\n";
            }
        }
        catch(std::exception &e) {
            if(!ex_aff) std::cout << "Exception in Fudge at i="<<i<<": "<<e.what()<<"\n";
            ex_aff=true;
        }
    }
    stats.finish();
    statf.finish();
    bool ok= stats.rms.Jr<eps && stats.rms.Jz<eps && stats.rms.Jphi<eps && !ex_afs
          && statf.rms.Jr<eps && statf.rms.Jz<eps && statf.rms.Jphi<eps && !ex_aff
          && fabs(stats.avg.Jr-statf.avg.Jr)<eps
          && fabs(stats.avg.Jz-statf.avg.Jz)<eps
          && fabs(stats.avg.Jphi-statf.avg.Jphi)<eps
          && (stats.avg.Jz==0 || fabs(ifd_p - ifd_i)<1e-5);
    std::cout << coordSysT::name() << ", Exact"
    ":  Jr="  <<stats.avg.Jr  <<" +- "<<stats.rms.Jr<<
    ",  Jz="  <<stats.avg.Jz  <<" +- "<<stats.rms.Jz<<
    ",  Jphi="<<stats.avg.Jphi<<" +- "<<stats.rms.Jphi<<
    (ex_afs ? ",  \033[1;33mCAUGHT EXCEPTION\033[0m\n":"\n");
    std::cout << coordSysT::name() << ", Fudge"
    ":  Jr="  <<statf.avg.Jr  <<" +- "<<statf.rms.Jr<<
    ",  Jz="  <<statf.avg.Jz  <<" +- "<<statf.rms.Jz<<
    ",  Jphi="<<statf.avg.Jphi<<" +- "<<statf.rms.Jphi<<
    (ok?"":" \033[1;31m**\033[0m")<<
    (ex_aff ? ",  \033[1;33mCAUGHT EXCEPTION\033[0m\n":"\n");
    return ok;
}

bool test_three_cs(const potential::OblatePerfectEllipsoid& potential, 
    const actions::InterfocalDistanceFinder& ifdfinder,
    const coord::PosVelCar& initcond, const char* title)
{
    const double total_time=100.;
    const double timestep=1./8;
    bool ok=true;
    std::cout << "\033[1;39m   ===== "<<title<<" =====\033[0m\n";
    ok &= test_oblate_staeckel(potential, ifdfinder, coord::toPosVelCar(initcond), total_time, timestep);
    ok &= test_oblate_staeckel(potential, ifdfinder, coord::toPosVelCyl(initcond), total_time, timestep);
    ok &= test_oblate_staeckel(potential, ifdfinder, coord::toPosVelSph(initcond), total_time, timestep);
    return ok;
}

int main() {
    const potential::OblatePerfectEllipsoid pot(1.0, axis_a, axis_c);
    const actions::InterfocalDistanceFinder ifi(pot);
    bool allok=true;
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0.3, 0.1, 0.1, 0.4, 0.1   ), "ordinary case");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0  , 0.2, 0.3486), "thin orbit (Jr~0)");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4732), "thin orbit in x-z plane (Jr~0, Jphi=0)");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4097), "tube orbit in x-z plane near separatrix");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0. , 0. , 0.4096), "box orbit in x-z plane near separatrix");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0. ,1e-8, 0.4097), "orbit with Jphi<<J_z");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0.3, 0. , 0.1, 0.4, 1e-4  ), "almost in-plane orbit (Jz~0)");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0.3, 0. , 0.1, 0.4, 0.    ), "exactly in-plane orbit (Jz=0)");
    allok &= test_three_cs(pot, ifi, coord::PosVelCar(1, 0. , 0. , 0. ,.296, 0.    ), "almost circular in-plane orbit (Jz=0,Jr~0)");
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}