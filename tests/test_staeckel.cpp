#include "potential_staeckel.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include <iostream>
#include <cmath>

const double integr_eps=1e-8;  // integration accuracy parameter
const double eps=1e-6;  // accuracy of comparison

// helper class to compute scatter in actions
class actionstat{
public:
    actions::Actions avg, disp;
    int N;
    actionstat() { avg.Jr=avg.Jz=avg.Jphi=0; disp=avg; N=0; }
    void add(const actions::Actions act) {
        avg.Jr  +=act.Jr;   disp.Jr  +=pow_2(act.Jr);
        avg.Jz  +=act.Jz;   disp.Jz  +=pow_2(act.Jz);
        avg.Jphi+=act.Jphi; disp.Jphi+=pow_2(act.Jphi);
        N++;
    }
    void finish() {
        avg.Jr/=N;
        avg.Jz/=N;
        avg.Jphi/=N;
        disp.Jr  =sqrt(std::max<double>(0, disp.Jr/N  -pow_2(avg.Jr)));
        disp.Jz  =sqrt(std::max<double>(0, disp.Jz/N  -pow_2(avg.Jz)));
        disp.Jphi=sqrt(std::max<double>(0, disp.Jphi/N-pow_2(avg.Jphi)));
    }
};

template<typename coordSysT>
bool test_oblate_staeckel(const potential::StaeckelOblatePerfectEllipsoid& potential,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep, const bool output)
{
    std::vector<coord::PosVelT<coordSysT> > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    actions::ActionFinderAxisymmetricStaeckel afs(potential);
    actions::ActionFinderAxisymmetricFudgeJS aff(potential);
    actionstat stats, statf;
    bool ex_afs=false, ex_aff=false;
    for(size_t i=0; i<traj.size(); i++) {
        try{
            stats.add(afs.actions(coord::toPosVelCar(traj[i])));
        }
        catch(std::exception &e) {
            ex_afs=true;
//            std::cout << "Exception in Staeckel at i="<<i<<": "<<e.what()<<"\n";
        }
        try{
            statf.add(aff.actions(coord::toPosVelCar(traj[i])));
        }
        catch(std::exception &e) {
            ex_aff=true;
//            std::cout << "Exception in Fudge at i="<<i<<": "<<e.what()<<"\n";
        }
        if(output) {
            double xv[6];
            coord::toPosVelCar(traj[i]).unpack_to(xv);
            std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"\n";
        }
    }
    stats.finish();
    statf.finish();
    std::cout << coord::CoordSysName<coordSysT>() << ", Exact"
    ":  Jr="  <<stats.avg.Jr  <<" +- "<<stats.disp.Jr<<
    ",  Jz="  <<stats.avg.Jz  <<" +- "<<stats.disp.Jz<<
    ",  Jphi="<<stats.avg.Jphi<<" +- "<<stats.disp.Jphi<< (ex_afs ? ",  CAUGHT EXCEPTION\n":"\n");
    std::cout << coord::CoordSysName<coordSysT>() << ", Fudge"
    ":  Jr="  <<statf.avg.Jr  <<" +- "<<statf.disp.Jr<<
    ",  Jz="  <<statf.avg.Jz  <<" +- "<<statf.disp.Jz<<
    ",  Jphi="<<statf.avg.Jphi<<" +- "<<statf.disp.Jphi<< (ex_afs ? ",  CAUGHT EXCEPTION\n":"\n");
    return stats.disp.Jr<eps && stats.disp.Jz<eps && stats.disp.Jphi<eps;
}

bool test_three_cs(const potential::StaeckelOblatePerfectEllipsoid& potential, const coord::PosVelCar& initcond, const char* title)
{
    const double total_time=100.;
    const double timestep=1./8;
    bool output=false;
    bool ok=true;
    std::cout << "   ===== "<<title<<" =====\n";
    ok &= test_oblate_staeckel(potential, coord::toPosVelCar(initcond), total_time, timestep, output);
    ok &= test_oblate_staeckel(potential, coord::toPosVelCyl(initcond), total_time, timestep, output);
    ok &= test_oblate_staeckel(potential, coord::toPosVelSph(initcond), total_time, timestep, output);
    return ok;
}

int main() {
    const potential::StaeckelOblatePerfectEllipsoid potential(1.0, 1.6, 1.0);
    bool allok=true;
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.3, 0.1, 0.1, 0.4, 0.1), "ordinary case");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.0, 0.0, 0, 0.2, 0.3486), "thin orbit (Jr~0)");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.0, 0.0, 0.0, 0, 0.4732), "thin orbit in x-z plane (Jr~0, Jphi=0)");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.0, 0.0, 0.0, 0, 0.4097), "tube orbit in x-z plane near separatrix");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.0, 0.0, 0.0, 0, 0.4096), "box orbit in x-z plane near separatrix");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.3, 0., 0.1, 0.4, 1e-3), "almost in-plane orbit (Jz~0)");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0.3, 0., 0.1, 0.4, 0), "exactly in-plane orbit (Jz=0)");
    allok &= test_three_cs(potential, coord::PosVelCar(1, 0, 0, 0, 0.296, 0), "almost circular in-plane orbit (Jz=0,Jr~0)");
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}