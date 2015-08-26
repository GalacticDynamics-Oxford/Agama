#include "orbit.h"
#include "actions_staeckel.h"
#include "potential_factory.h"
#include "units.h"
#include "debug_utils.h"
#include "utils_config.h"
#include "math_core.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

const double integr_eps = 1e-8;  // integration accuracy parameter
const double eps = 1e-6;  // accuracy of comparison
const units::InternalUnits unit(units::galactic_Myr);//(0.2*units::Kpc, 100*units::Myr);
int numActionEval = 0;
bool headerPrinted = false;

class BestIFDFinder: public math::IFunctionNoDeriv {
public:
    BestIFDFinder(const potential::BasePotential& p, const std::vector<coord::PosVelCar>& t,
                  actions::Actions& a, actions::Actions& d) :
        potential(p), traj(t), avg(a), disp(d) {};
    virtual double value(double ifd) const {
        actions::ActionStat acts;
        for(size_t i=0; i<traj.size(); i++)
            acts.add(actions::axisymFudgeActions(potential, coord::toPosVelCyl(traj[i]), ifd));
        numActionEval += traj.size();
        acts.finish();
        avg = acts.avg;
        disp= acts.disp;
        return acts.disp.Jr + acts.disp.Jz;
    }
private:
    const potential::BasePotential& potential;
    const std::vector<coord::PosVelCar>& traj;
    actions::Actions& avg;
    actions::Actions& disp;
};

bool test_actions(const potential::BasePotential& potential,
    const coord::PosVelCar& initial_conditions,
    const double total_time, const double timestep, double ifd)
{
    std::vector<coord::PosVelCar> traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    actions::Actions avg, disp;
    if(ifd<=0) 
        ifd = actions::estimateInterfocalDistancePoints(potential, traj);

    BestIFDFinder fnc(potential, traj, avg, disp);
    //ifd = math::findMin(fnc, 0.1, 10, NAN, 0.001);
    fnc.value(ifd);
    double dim = unit.to_Kpc*unit.to_Kpc/unit.to_Myr; //unit.to_Kpc_kms;
    double scatter = (disp.Jr+disp.Jz) / (avg.Jr+avg.Jz);
    double scatterNorm = 0.33 * sqrt( (avg.Jr+avg.Jz) / (avg.Jr+avg.Jz+fabs(avg.Jphi)) );
    bool tolerable = scatter < scatterNorm ; 
    double E = totalEnergy(potential, initial_conditions);
    if(!headerPrinted) {
        std::cout << "E         Lcirc       Jphi       Jr        Jr_err      Jz        Jz_err      Interfocal_distance\n";
        headerPrinted = true;
    }
    std::cout << 
        E*pow_2(unit.to_Kpc/unit.to_Myr) <<" "<<
        L_circ(potential, E)*dim << "  " << 
        avg.Jphi*dim <<" "<<
        avg.Jr*dim <<" "<< disp.Jr*dim <<" "<< 
        avg.Jz*dim <<" "<< disp.Jz*dim <<" "<<
        //acts.avg.Jphi*dim <<" "<< acts.disp.Jphi*dim <<"  "<< 
        //angs.freqr <<" "<< angs.freqz <<" "<< angs.freqphi <<"  "<<
        //angs.dispr <<" "<< angs.dispz <<" "<< angs.dispphi <<"  "<<
        //maxJr*dim << " " << maxJz*dim << "  "<<
        ifd*unit.to_Kpc; 
        //        (tolerable?"":" \033[1;31m **\033[0m") << std::endl;
    //std::cout <<"   "<< R <<" "<<ic[3]<<" "<<ic[4]<<" "<<ic[5]<< " "<<total_time<< std::endl;
    return tolerable;
}

const potential::BasePotential* make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    const potential::BasePotential* gp = potential::readGalaxyPotential(params_file, unit);
    std::remove(params_file);
    if(gp==NULL)
        std::cout<<"Potential not created\n";
    return gp;
}

const char* test_galpot_params =
// BestFitPotential.Tpot
"3\n"
"5.63482e+08 2.6771 0.1974 0 0\n"
"2.51529e+08 2.6771 0.7050 0 0\n"
"9.34513e+07 5.3542 0.04 4 0\n"
"2\n"
"9.49e+10    0.5  0  1.8  0.075   2.1\n"
"1.85884e+07 1.0  1  3    14.2825 250.\n";

/* McMillan2011, convenient
"2\n"
"7.52975e+08 3   0.3 0 0\n"
"1.81982e+08 3.5 0.9 0 0\n"
"2\n"
"9.41496e+10 0.5 0 1.8 0.075 2.1\n"
"1.25339e+07 1   1 3   17    0\n"; */
/* McMillan2011, best
"2\n"
"8.1663e+08  2.89769 0.3 0 0\n"
"2.09476e+08 3.30618 0.9 0 0\n"
"2\n"
"9.55712e+10 0.5 0 1.8 0.075  2.1\n"
"8.45559e+06 1   1 3   20.222 0\n";*/

int main(int argc, const char* argv[]) {
    bool allok = true;
    const potential::BasePotential* pot;
    pot = make_galpot(test_galpot_params);
    clock_t clockbegin = std::clock();
    actions::InterfocalDistanceFinder ifdFinder(*pot);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to init Delta-finder\n";
    clockbegin=std::clock();
    for(double E=pot->value(coord::PosCyl(0,0,0))*0.95, dE=-E*0.063; E<0; E+=dE) {
        double Rc = R_circ(*pot, E);
        double k,n,o;
        epicycleFreqs(*pot, Rc, k, n, o);
        double totalTime = 2*M_PI/k * 50;
        double timeStep  = totalTime / 500;
        double Lc = v_circ(*pot, Rc) * Rc;
        for(int iLz=0; iLz<16; iLz++) {
            double Lz   = (iLz+0.5)/16 * Lc;
            double R;   // radius of shell orbit
            double IFD  = actions::estimateInterfocalDistanceShellOrbit(*pot, E, Lz, &R);
            double vphi = Lz/R;
            double vmer = sqrt(2*(E-pot->value(coord::PosCyl(R,0,0)))-vphi*vphi);
            if(vmer!=vmer) {
                std::cerr << "Can't assign ICs!\n";
                continue;
            }
            for(int a=0; a<8; a++) {
                double ang=(a+0.01)/7.02 * M_PI/2;
                coord::PosVelCar ic(R, 0, 0, vmer*cos(ang), vphi, vmer*sin(ang));
                double ifd = ifdFinder.value(coord::toPosVelCyl(ic));
                allok &= test_actions(*pot, ic, totalTime, timeStep, ifd);
                std::cout << " "<<IFD<<"\n";
            }
        }
    }
    std::cout << numActionEval * 1.0*CLOCKS_PER_SEC / (std::clock()-clockbegin) << " actions per second\n";
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    delete pot;
    return 0;
}