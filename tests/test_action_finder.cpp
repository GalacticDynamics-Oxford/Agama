/** \file    test_action_finder.cpp
    \author  Eugene Vasiliev
    \date    July 2015

    This test demonstrates the accuracy of action/angle determination using
    the Staeckel fudge approximation.

    We create an instance of realistic galactic potential,
    and scan the entire phase space by looping over energy, angular momentum
    and the third integral (the latter is explored by varying the direction
    of velocity in meridional plane as the orbit crosses the equatorial plane).
    For each initial condition, we numerically compute the orbit,
    and for each point on the trajectory, determine the values of actions
    and angles using the Staeckel fudge.
    The accuracy of action determination is assessed by the scatter in
    the values reported by the routine for different points on the same orbit;
    the quality of angle determination is assessed by checking how closely
    the angles follow a linear trend with time.

    In general, the approximation works fairly well for orbits with either
    J_r or J_z being small compared to the other two actions,
    but even for less favourable cases it is typically accurate to within
    a few percent. Except when the orbit appears to be in or near a resonance,
    in which case the rms error may be larger than 10%.
*/
#include "orbit.h"
#include "actions_staeckel.h"
#include "potential_factory.h"
#include "units.h"
#include "debug_utils.h"
#include "utils.h"
#include "math_core.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>

const double toler = 1e-8;  // integration accuracy parameter
const units::InternalUnits unit(units::galactic_Myr);
const double dim = unit.to_Kpc*unit.to_Kpc/unit.to_Myr;
int numActionEval  = 0;

bool isResonance(const std::vector<coord::PosVelCar>& traj)
{
    // determine if an orbit is a vertical 1:1 resonance, i.e. is not symmetric w.r.t z-reflection:
    // compare the average value of F(z) with its rms, where F(z) = z when R < average R, or -z otherwise
    math::Averager avgR, avgz;
    for(unsigned int i=0; i<traj.size(); i++)
        avgR.add(sqrt(pow_2(traj[i].x)+pow_2(traj[i].y)));
    for(unsigned int i=0; i<traj.size(); i++)
        avgz.add(traj[i].z * (pow_2(traj[i].x)+pow_2(traj[i].y) < pow_2(avgR.mean()) ? 1 : -1));
    return avgz.mean() / sqrt(avgz.disp());
}

bool test_actions(const potential::BasePotential& potential,
    const coord::PosVelCar& initial_conditions,
    const double total_time, const double timestep,
    const actions::ActionFinderAxisymFudge& actfinder,
    std::string& output)
{
    std::vector<coord::PosVelCar> traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, toler);
    double ifd = fmax(
        //actions::estimateFocalDistancePoints(potential, traj);
        actfinder.focalDistance(toPosVelCyl(initial_conditions)),
        initial_conditions.x*1e-4);

    std::ofstream strm;
    actions::ActionStat actF, actI;
    math::Averager avgI3;
    const coord::ProlSph coordsys(pow_2(ifd));
    for(size_t i=0; i<traj.size(); i++) {
        const coord::PosVelCyl point = toPosVelCyl(traj[i]);
        const coord::PosVelProlSph pprol = coord::toPosVel<coord::Cyl, coord::ProlSph>(point, coordsys);
        const double
        Phi  = potential.value(point),
        Phi0 = potential.value(coord::PosCyl(sqrt(pprol.lambda-coordsys.Delta2), 0, point.phi)),
        I3   = pprol.lambda * (Phi - Phi0) + 0.5 * (
            pow_2(point.z * point.vphi) +
            pow_2(point.R * point.vz - point.z * point.vR) +
            pow_2(point.vz) * coordsys.Delta2 );
        actions::Actions actsF = actions::actionsAxisymFudge(potential, toPosVelCyl(traj[i]), ifd);
        actions::Actions actsI = actfinder.actions(toPosVelCyl(traj[i]));
        strm << utils::pp(point.R, 8)+' '+utils::pp(point.z, 8)+' '+
            utils::pp(pprol.lambda, 8)+' '+utils::pp(pprol.nu, 8)+' '+
            utils::pp(I3, 8)+' '+
            utils::pp(actsF.Jr, 8)+' '+utils::pp(actsI.Jr, 8)+' '+
            utils::pp(actsF.Jz, 8)+' '+utils::pp(actsI.Jz, 8)+'\n';
        avgI3.add(I3);
        actF.add(actsF);
        actI.add(actsI);
    }
    numActionEval += traj.size();
    actF.finish();
    actI.finish();
    double scatter = (actF.rms.Jr+actF.rms.Jz) / (actF.avg.Jr+actF.avg.Jz);
    double scatterNorm = 0.33 * sqrt( (actF.avg.Jr+actF.avg.Jz) /
        (actF.avg.Jr+actF.avg.Jz+fabs(actF.avg.Jphi)) );
    bool tolerable = scatter < scatterNorm || isResonance(traj);
    double E = totalEnergy(potential, initial_conditions);
    output =
        utils::pp(E*pow_2(unit.to_Kpc/unit.to_Myr),7) +'\t'+
        utils::pp(L_circ(potential, E)*dim,7) +'\t'+
        utils::pp(actF.avg.Jphi*dim,7) +'\t'+
        utils::pp(actF.avg.Jr*dim,7) +'\t'+ utils::pp(actF.rms.Jr*dim,7) +'\t'+
        utils::pp(actF.avg.Jz*dim,7) +'\t'+ utils::pp(actF.rms.Jz*dim,7) +'\t'+
        utils::pp(actI.avg.Jr*dim,7) +'\t'+ utils::pp(actI.rms.Jr*dim,7) +'\t'+
        utils::pp(actI.avg.Jz*dim,7) +'\t'+ utils::pp(actI.rms.Jz*dim,7) +'\t'+
        utils::pp(avgI3.mean(),7) +'\t'+ utils::pp(sqrt(avgI3.disp()),7) +'\t'+
        //utils::pp(ifd*unit.to_Kpc,7) +'\t'+
        (tolerable?"":" **");
    return tolerable;
}

potential::PtrPotential make_galpot(const char* params)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    potential::PtrPotential gp = potential::readGalaxyPotential(params_file, unit);
    std::remove(params_file);
    if(gp.get()==NULL)
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

int main()
{
    bool allok = true;
    potential::PtrPotential pot = make_galpot(test_galpot_params);
    clock_t clockbegin = std::clock();
    actions::ActionFinderAxisymFudge actfinder(pot);
    std::cout << (std::clock()-clockbegin)*1.0/CLOCKS_PER_SEC << " seconds to init action interpolator\n";

    // prepare room for storing the output
    std::vector<double> Evalues;
    for(double E=pot->value(coord::PosCyl(0,0,0))*0.95, dE=-E*0.063; E<0; E+=dE)
        Evalues.push_back(E);
    const int NE   = Evalues.size();
    const int NLZ  = 16;
    const int NDIR = 8;
    std::vector<std::string> results(NE * NLZ * NDIR);
    clockbegin=std::clock();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(int iE=0; iE<NE; iE++) {
        double E = Evalues[iE];
        double Rc = R_circ(*pot, E);
        double k,n,o;
        epicycleFreqs(*pot, Rc, k, n, o);  // an estimate for orbit frequencies
        double totalTime = 2*M_PI/k * 50;  // needed to assign the integration time interval
        double timeStep  = totalTime / 500;
        double Lc = v_circ(*pot, Rc) * Rc;
        for(int iLz=0; iLz<NLZ; iLz++) {   // explore the range of angular momentum for a fixed energy
            double Lz   = (iLz+0.5)/NLZ * Lc;
            double R;                      // radius of a shell orbit
            actions::estimateFocalDistanceShellOrbit(*pot, E, Lz, &R);
            double vphi = Lz/R;
            double vmer = sqrt(2*(E-pot->value(coord::PosCyl(R,0,0)))-vphi*vphi);
            if(vmer!=vmer) {
                std::cerr << "Can't assign ICs for Rc="<<Rc<<", Rthin="<<R<<"!\n";
                allok=false;
                continue;
            }
            for(int a=0; a<NDIR; a++) {   // explore the range of third integral by varying the direction
                double ang=(a+0.01)/(NDIR-0.98) * M_PI/2;  // of velocity in the meridional plane
                coord::PosVelCar ic(R, 0, 0, vmer*cos(ang), vphi, vmer*sin(ang));
                int index = a + NDIR * (iLz + NLZ * iE);
                allok &= test_actions(*pot, ic, totalTime, timeStep, actfinder, results[index]);
            }
        }
    }
    std::cout << numActionEval * 1.0*CLOCKS_PER_SEC / (std::clock()-clockbegin) << " actions per second\n";
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream output("test_action_finder.dat");
        output << "E\tLcirc\tJphi\tJr\tJr_err\tJz\tJz_err\tiJr\tiJr_err\tiJz\tiJz_err\tI3\tI3_err\n";
        for(unsigned int l=0; l<results.size(); l++)
            output << results[l] << '\n';
    }
    if(allok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
