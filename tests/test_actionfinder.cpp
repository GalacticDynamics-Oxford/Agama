#include "orbit.h"
#include "actions_staeckel.h"
#include "potential_factory.h"
#include "units.h"
#include "debug_utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <vector>

const double integr_eps=1e-8;  // integration accuracy parameter
const double eps=1e-6;  // accuracy of comparison
const units::InternalUnits unit(units::galactic_Myr);//(0.2*units::Kpc, 100*units::Myr);
int numActionEval=0;

//#define SINGLEORBIT
#define INPUTFILE

bool test_actions(const potential::BasePotential& potential,
    const coord::PosVelCar& initial_conditions,
    const double total_time, const double timestep, double ifd_est)
{
    std::vector<coord::PosVelCar > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    //double ifd_est  = ifdFinder.value(coord::toPosVelCyl(initial_conditions));
    double ifd_traj = sqrt( fmax(pow_2(ifd_est), 
        actions::estimateSquaredInterfocalDistancePoints(potential, traj) ) );
    double dim = unit.to_Kpc*unit.to_Kpc/unit.to_Myr; //unit.to_Kpc_kms;
#ifdef SINGLEORBIT
    std::ofstream fout("orbit.dat");
#endif
    actions::ActionStat acts;
    actions::AngleStat angs;
    for(size_t i=0; i<traj.size(); i++) {
        const coord::PosVelCyl p=coord::toPosVelCyl(traj[i]);
//        actions::ActionAngles a = actions::axisymFudgeActionAngles(potential, p, ifd_traj);
        actions::Actions a = actions::axisymFudgeActions(potential, p, ifd_traj);
        //angs.add(i*timestep*unit.to_Kpc/unit.to_kms, a);
        acts.add(a);
        numActionEval++;
#ifdef SINGLEORBIT
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(p, NULL, &grad, &hess);
        fout << (i*timestep*unit.to_Kpc/unit.to_kms)<<"  "<<
            p.R*unit.to_Kpc<<" "<<p.z*unit.to_Kpc<<"  "<<
            p.vR*unit.to_kms<<" "<<p.vz*unit.to_kms<<"   "<<
            a.Jr*dim<<" "<<a.Jz*dim<<"  "<<
            (3*p.z*grad.dR - 3*p.R*grad.dz + p.R*p.z*(hess.dR2-hess.dz2)
            + hess.dRdz*(p.z*p.z-p.R*p.R)) * pow_2(unit.to_Kpc) <<" "<< hess.dRdz <<"  "<<
            angs.thetar.back()<<" "<<angs.thetaz.back()<<" "<<angs.thetaphi.back()<<
            "\n";
#endif
    }
    acts.finish();
    angs.finish();
    double scatter = (acts.disp.Jr+acts.disp.Jz) / (acts.avg.Jr+acts.avg.Jz);
    double scatterNorm = 0.33 * sqrt( (acts.avg.Jr+acts.avg.Jz) / (acts.avg.Jr+acts.avg.Jz+fabs(acts.avg.Jphi)) );
    bool tolerable = scatter < scatterNorm ; 
    //    && angs.dispr < 0.1 && angs.dispz < 1.0 && angs.dispphi < 0.05;
    double E = totalEnergy(potential, initial_conditions);
    std::cout << 
        E*pow_2(unit.to_Kpc/unit.to_Myr) <<" "<<
        L_circ(potential, E)*dim << "  " << 
        acts.avg.Jphi*dim <<" "<<
        acts.avg.Jr*dim <<" "<< acts.disp.Jr*dim <<" "<< 
        acts.avg.Jz*dim <<" "<< acts.disp.Jz*dim <<" ";
        //acts.avg.Jphi*dim <<" "<< acts.disp.Jphi*dim <<"  "<< 
        //angs.freqr <<" "<< angs.freqz <<" "<< angs.freqphi <<"  "<<
        //angs.dispr <<" "<< angs.dispz <<" "<< angs.dispphi <<"  "<<
        //maxJr*dim << " " << maxJz*dim << "  "<<
        //ifd_traj*unit.to_Kpc <<" "<< ifd_est*unit.to_Kpc; 
        //        (tolerable?"":" \033[1;31m **\033[0m") << std::endl;
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

//double ic[6] = { 6, 0, 0.0, -100, 200, 185 };  // fish orbit R=4-9.5, |z|=5.5
//double ic[6] = { 6, 0, 0.0, -110, 200, 175 };  // similar but non-resonant
//double ic[6] = { 7.2, 0, 0.05, 35, 212, 85 };
//double ic[6] = { 7.2, 0, 0.0, -100, 200, 0 };
//double ic[6] = {8.00209, 0, -0.0901381, -4.44468, 244.38, -9.7863};  //Jr=Jz=0.001,Jphi=2
//double ic[6] = {8.372, 0, -0.753, 78.07, 233.6, -88.87};             // Jz=Jr=0.1,Jphi=2  in PJM11_best
//double ic[6] = {11.3472, 0, 1.88637, -103.99, 172.338, 64.0454};     // Jr=Jz=0.2,Jphi=2
//double ic[6] = {11.2429, 0, -4.34075, -183.643, 173.937, -42.1339};  // fish orbit R=6-20, z<=12, in PJM11_best, Jr=Jz=0.5,Jphi=2
//double ic[6] = {5.04304, 0, 3.72763, -232.44, 193.886, -2.35769};    // Jr=Jz=0.5, Jphi=1
double ic[6] = {3.46726, 0, -0.133605, -84.7102, -282.002, 31.8277};

double ics[30][6] = {
{5.06616, 0, -0.312879, 71.3196, 193.001, 1.10671},
{8.0978, 0, 0.547581, 6.45421, -241.492, 10.9483},
{8.44679, 0, -1.20714, -9.07666, 231.514, -71.0561},
{8.62508, 0, 0.0394613, -10.2208, 226.728, 10.8331},
{8.05243, 0, -1.02207, 16.3575, 242.852, 44.3512},
{6.94181, 0, 1.28152, 51.1929, 281.706, -62.0895},
{12.3392, 0, -0.127845, -6.18614, 237.723, 17.5905},
{12.639, 0, -1.70895, 10.5543, 232.085, 23.3847},
{11.8427, 0, -0.159802, 2.25277, 247.69, 6.02108},
{13.2383, 0, -0.0160772, 10.4629, 221.58, 7.788},
{13.556, 0, -1.79882, 22.9815, 216.386, 47.1126},
{11.6142, 0, -0.486307, 44.6349, 252.564, 4.06178},
{14.717, 0, -2.12204, -49.0004, 199.316, 23.7802},
{17.2579, 0, -0.321815, 3.08573, 226.627, 0.909792},
{16.9765, 0, -0.964899, -6.48608, 230.383, -7.2927},
{17.0621, 0, -2.38288, -6.0803, 229.228, -39.6974},
{16.9946, 0, 3.63905, 12.1067, 230.138, 4.75884},
{16.8612, 0, -2.24878, 15.3281, 231.958, -19.258},
{15.3825, 0, 0.00231117, -27.6264, 254.256, -7.00216},
{11.7606, 0, 10.2523, -123.451, 83.1396, -86.112},
{34.4939, 0, -14.9654, -35.3946, 28.3464, 183.061},
{14.6888, 0, -14.8857, 188.128, 66.5662, -271.958},
{51.3709, 0, 83.1065, -38.9198, 19.0337, 46.0276},
{14.7266, 0, 2.91052, -173.04, 132.79, -170.743},
{23.7377, 0, 10.6812, 92.0017, 123.573, -52.4921},
{31.2967, 0, -4.05745, 108.374, 93.7265, 217.896},
{60.7817, 0, 13.4328, 144.552, 48.2601, 4.20559},
{43.6769, 0, -46.8767, -53.9903, 67.1596, 208.135},
{28.0207, 0, -19.3931, -101.342, -139.579, 30.7491},
{36.3541, 0, -58.4024, 37.6475, 107.584, -191.881} };

int main(int argc, const char* argv[]) {
    bool allok = true;
    const potential::BasePotential* pot;
    if(argc>1) {
        utils::KeyValueMap params(argc, argv);
        potential::ConfigPotential config;
        potential::parseConfigPotential(params, config);
        pot = potential::createPotential(config);
    } else
        pot = make_galpot(test_galpot_params);
    double total_time = 4. * unit.from_Kpc/unit.from_kms;
    int numsteps = 1000;
    double timestep = total_time/numsteps;
    clock_t clockbegin = std::clock();
    actions::InterfocalDistanceFinder ifdFinder(*pot, 100);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to init Delta-finder\n";
    clockbegin=std::clock();
/*
#ifndef SINGLEORBIT
#ifdef INPUTFILE
    std::ifstream icfile("ic.dat");
    while(!icfile.eof()) {
        double J[3],ic[6];
        icfile >> ic[0]>>ic[1]>>ic[2]>>ic[3]>>ic[4]>>ic[5]>>J[0]>>J[1]>>J[2];
        std::cout<<J[0]<<" "<<J[1]<<" "<<J[2]<<"  ";
                
#else
    for(int k=0; k<30; k++) {
        for(int i=0; i<6; i++)
            ic[i]=ics[k][i];
#endif
#else
    {
#endif
 */
        for(double E=pot->value(coord::PosCyl(0,0,0))*0.95, dE=-E*0.063; E<0; E+=dE) {
            double Rc = R_circ(*pot, E);
            double k,n,o;
            epicycleFreqs(*pot, Rc, k, n, o);
            total_time = 2*M_PI/k * 50;
            timestep = total_time/500;
            double Lc = v_circ(*pot, Rc) * Rc;
            for(int iLz=0; iLz<16; iLz++) {
                double Lz   = (iLz+0.5)/16 * Lc;
                double maxJr, maxJz, R, IFD;
                actions::findClosedOrbitRZplane(*pot, E, Lz, maxJr, maxJz, R, IFD);
                double vphi = Lz/R;
                double vmer = sqrt(2*(E-pot->value(coord::PosCyl(R,0,0)))-vphi*vphi);
                if(vmer!=vmer) {
                    std::cout << "Can't assign ICs!\n";
                    continue;
                }
                for(int a=0; a<8; a++) {
                    double ang=(a+0.01)/7.02 * M_PI/2;
                    ic[0]=R;
                    ic[1]=0;
                    ic[2]=0;
                    ic[3]=vmer*cos(ang);
                    ic[5]=vmer*sin(ang);
                    ic[4]=vphi;
        /*for(int i=0; i<3; i++) {
            ic[i]   *= unit.from_Kpc;
            ic[i+3] *= unit.from_kms;
        }*/
        allok &= test_actions(*pot, coord::PosVelCar(ic), total_time, timestep, IFD);
                    std::cout << maxJr*unit.to_Kpc*unit.to_Kpc/unit.to_Myr << " " <<
                    maxJz*unit.to_Kpc*unit.to_Kpc/unit.to_Myr << std::endl;
                }
            }
    }
    std::cout << numActionEval * 1.0*CLOCKS_PER_SEC / (std::clock()-clockbegin) << " actions per second\n";
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    delete pot;
    return 0;
}