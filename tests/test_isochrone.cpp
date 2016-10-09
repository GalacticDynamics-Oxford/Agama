/** \file    test_isochrone.cpp
    \author  Eugene Vasiliev
    \date    February 2016

    This test checks the correctness of (exact) action-angle determination for Isochrone potential
    (and additionally for an arbitrary spherical potential)
*/
#include "potential_analytic.h"
#include "actions_isochrone.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "orbit.h"
#include "debug_utils.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>

//#define TEST_OLD_TORUS
#ifdef TEST_OLD_TORUS
#include "torus/Toy_Isochrone.h"
#endif

// whether to do performance test
//#define PERFTEST

bool test_isochrone(const coord::PosVelCyl& initial_conditions, const char* title)
{
    const bool output = utils::verbosityLevel >= utils::VL_VERBOSE; // whether to write a text file
    const double epsr = 2e-4;  // accuracy of comparison for radial action found with different methods
    const double epsd = 1e-7;  // accuracy of action conservation along the orbit for each method
    const double epst = 1e-9;  // accuracy of reverse transformation (pv=>aa=>pv) for isochrone
    const double epss = 1e-7;  // accuracy of reverse transformation for spherical a/a mapping
    const double epsf = 1e-7;  // accuracy of frequency determination
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    const double total_time=50;// integration time
    const double timestep=1./8;// sampling rate of trajectory
    std::cout << "\033[1;39m"<<title<<"\033[0m\n";
    std::vector<coord::PosVelCyl > traj;
    potential::Isochrone pot(M, b);
    orbit::integrate(pot, initial_conditions, total_time, timestep, traj, 1e-15);
    actions::ActionFinderSpherical actGrid(pot);  // interpolation-based action finder/mapper
    actions::ActionStat statI, statS, statF, statG;
    actions::ActionAngles aaI, aaF, aaS, aaG;
    actions::Frequencies frI, frF, frS, frG, frIinv, frSinv;
    math::Averager statfrIr, statfrIz, statH, statE;
    actions::Angles aoldF(0,0,0), aoldI(0,0,0), aoldS(0,0,0);
    bool anglesMonotonic= true;   // angle determination is reasonable
    bool reversible_iso = true;   // forward-reverse transform for isochrone gives the original point
    bool reversible_sph = true;   // same for spherical a/a finder/mapper
    bool reversible_grid= true;   // same for grid-interpolated spherical a/a finder/mapper
    bool deriv_iso_ok   = true;   // finite-difference derivs agree with analytic ones
    bool deriv_grid_ok  = true;   // same for the derivs of grid-interpolated a/a mapper
    std::ofstream strm;
    if(output) {
        strm.open("test_isochrone.dat");
        strm << std::setprecision(15);
    }
    double ifd = 1e-5;
    int numWarnings = 0;
#ifdef TEST_OLD_TORUS
    Torus::IsoPar toypar;
    toypar[0] = sqrt(M);
    toypar[1] = sqrt(b);
    toypar[2] = Lz(initial_conditions);
    Torus::ToyIsochrone toy(toypar);
#endif
    for(size_t i=0; i<traj.size(); i++) {
        statE.add(totalEnergy(pot, traj[i]));
        traj[i].phi = math::wrapAngle(traj[i].phi);
        aaI = actions::actionAnglesIsochrone(M, b,  traj[i], &frI);
        aaF = actions::actionAnglesAxisymFudge(pot, traj[i], ifd, &frF);
        aaS = actions::actionAnglesSpherical(pot, traj[i], &frS);
        aaG = actGrid. actionAngles(traj[i], &frG);
        statH.add(actions::computeHamiltonianSpherical(pot, aaI));  // find H(J)
        statI.add(aaI);
        statF.add(aaF);
        statS.add(aaS);
        statG.add(aaG);
        statfrIr.add(frI.Omegar);
        statfrIz.add(frI.Omegaz);
        actions::Angles anewF, anewI, anewS;
        anewF.thetar   = math::unwrapAngle(aaF.thetar,   aoldF.thetar);
        anewF.thetaz   = math::unwrapAngle(aaF.thetaz,   aoldF.thetaz);
        anewF.thetaphi = math::unwrapAngle(aaF.thetaphi, aoldF.thetaphi);
        anewI.thetar   = math::unwrapAngle(aaI.thetar,   aoldI.thetar);
        anewI.thetaz   = math::unwrapAngle(aaI.thetaz,   aoldI.thetaz);
        anewI.thetaphi = math::unwrapAngle(aaI.thetaphi, aoldI.thetaphi);
        anewS.thetar   = math::unwrapAngle(aaS.thetar,   aoldS.thetar);
        anewS.thetaz   = math::unwrapAngle(aaS.thetaz,   aoldS.thetaz);
        anewS.thetaphi = math::unwrapAngle(aaS.thetaphi, aoldS.thetaphi);
        anglesMonotonic &= i==0 || (
            anewI.thetar >= aoldI.thetar && anewS.thetar >= aoldS.thetar &&
           (anewF.thetar >= aoldF.thetar || aaF.Jr<1e-10) &&
            anewI.thetaz >= aoldI.thetaz && anewS.thetaz >= aoldS.thetaz &&
            anewF.thetaz >= aoldF.thetaz &&
            math::sign(aaI.Jphi) * anewI.thetaphi >= math::sign(aaI.Jphi) * aoldI.thetaphi &&
            math::sign(aaS.Jphi) * anewS.thetaphi >= math::sign(aaS.Jphi) * aoldS.thetaphi &&
            math::sign(aaF.Jphi) * anewF.thetaphi >= math::sign(aaF.Jphi) * aoldF.thetaphi);
        aoldI = anewI;
        aoldS = anewS;
        aoldF = anewF;
#ifdef TEST_OLD_TORUS
        coord::PosVelSph ps(toPosVelSph(traj[i]));
        Torus::PSPT pvs;
        pvs[0]=ps.r; pvs[1]=M_PI/2-ps.theta; pvs[2]=ps.phi;
        pvs[3]=ps.vr; pvs[4]=-ps.vtheta*ps.r; pvs[5]=ps.vphi*traj[i].R;
        Torus::PSPT aaT = toy.Backward3D(pvs); // (J_r,J_z,J_phi,theta_r,theta_z,theta_phi)
        Torus::PSPT pvi = toy.Forward3D(aaT);
        reversible_iso &= 
            math::fcmp(pvs[0], pvi[0], eps) == 0 && math::fcmp(pvs[1], pvi[1], eps) == 0 &&
            math::fcmp(pvs[2], pvi[2], eps) == 0 && math::fcmp(pvs[3]+1, pvi[3]+1, eps) == 0 &&
            math::fcmp(pvs[4]+1, pvi[4]+1, eps) == 0 && math::fcmp(pvs[5]+1, pvi[5]+1, eps) == 0;
        reversible_iso &=
            math::fcmp(aaT[0]+1, aaI.Jr+1, eps) == 0 &&  // ok when Jr<<1
            math::fcmp(aaT[1], aaI.Jz, eps) == 0 &&
            math::fcmp(aaT[2], aaI.Jphi, eps) == 0 &&
            (aaI.Jr<epsd || math::fcmp(aaT[3], aaI.thetar, eps) == 0) &&
            (aaI.Jz==0 || math::fcmp(math::wrapAngle(aaT[4]+1), math::wrapAngle(aaI.thetaz+1), eps) == 0) &&
            math::fcmp(aaT[5], aaI.thetaphi, eps) == 0;
        if(!reversible_iso)
            std::cout << aaT <<'\t' <<aaI << '\n';
#endif
        // inverse transformation for spherical potential
        coord::PosVelCyl pinv = actions::mapSpherical(pot, aaS, &frSinv);
        reversible_sph &= equalPosVel(pinv, traj[i], epss) && 
            math::fcmp(frS.Omegar, frSinv.Omegar, epss) == 0 &&
            math::fcmp(frS.Omegaz, frSinv.Omegaz, epss) == 0 &&
            math::fcmp(frS.Omegaphi, frSinv.Omegaphi, epss) == 0;
        // inverse transformation for interpolated spherical action finder, with derivatives
        actions::DerivAct<coord::SphMod> der_sph;
        pinv = toPosVelCyl(actGrid.map(aaG, &frSinv, &der_sph));
        reversible_grid &= equalPosVel(pinv, traj[i], epss) && 
            math::fcmp(frG.Omegar, frSinv.Omegar, epss) == 0 &&
            math::fcmp(frG.Omegaz, frSinv.Omegaz, epss) == 0 &&
            math::fcmp(frG.Omegaphi, frSinv.Omegaphi, epss) == 0;
        
        // inverse transformation for Isochrone with derivs
        actions::DerivAct<coord::SphMod> ac;
        coord::PosVelSphMod pd[2];
        coord::PosVelSphMod pp = actions::ToyMapIsochrone(M, b).map(aaI, &frIinv, &ac, NULL, pd);
        reversible_iso &= equalPosVel(toPosVelCyl(pp), traj[i], epst) && 
            math::fcmp(frI.Omegar, frIinv.Omegar, epst) == 0 &&
            math::fcmp(frI.Omegaz, frIinv.Omegaz, epst) == 0 &&
            math::fcmp(frI.Omegaphi, frIinv.Omegaphi, epst) == 0;
        // check derivs w.r.t. potential params
        coord::PosVelSphMod pM = actions::ToyMapIsochrone(M*(1+epsd), b).map(aaI);
        coord::PosVelSphMod pb = actions::ToyMapIsochrone(M, b*(1+epsd)).map(aaI);
        pM.r   = (pM.r   - pp.r)   / (M*epsd);
        pM.tau = (pM.tau - pp.tau) / (M*epsd);
        pM.phi = (pM.phi - pp.phi) / (M*epsd);
        pM.pr  = (pM.pr  - pp.pr)  / (M*epsd);
        pM.ptau= (pM.ptau- pp.ptau)/ (M*epsd);
        pM.pphi= (pM.pphi- pp.pphi)/ (M*epsd);
        pb.r   = (pb.r   - pp.r)   / (b*epsd);
        pb.tau = (pb.tau - pp.tau) / (b*epsd);
        pb.phi = (pb.phi - pp.phi) / (b*epsd);
        pb.pr  = (pb.pr  - pp.pr)  / (b*epsd);
        pb.ptau= (pb.ptau- pp.ptau)/ (b*epsd);
        pb.pphi= (pb.pphi- pp.pphi)/ (b*epsd);
        if(!equalPosVel(pM, pd[0], 1e-4) && ++numWarnings<10) {
            deriv_iso_ok = false;
            std::cout << "d/dM: " << pM << pd[0] << '\n';
        }
        if(!equalPosVel(pb, pd[1], 1e-4) && ++numWarnings<10) {
            deriv_iso_ok = false;
            std::cout << "d/db: " << pb << pd[1] << '\n';
        }
        // check derivs w.r.t. actions
        actions::ActionAngles aaT = aaI; aaT.Jr += epsd;
        coord::PosVelSphMod pJr = actions::ToyMapIsochrone(M, b).map(aaT);
        pJr.r   = (pJr.r   - pp.r)   / epsd;
        pJr.tau = (pJr.tau - pp.tau) / epsd;
        pJr.phi = (pJr.phi - pp.phi) / epsd;
        pJr.pr  = (pJr.pr  - pp.pr)  / epsd;
        pJr.ptau= (pJr.ptau- pp.ptau)/ epsd;
        pJr.pphi= (pJr.pphi- pp.pphi)/ epsd;
        aaT = aaI; aaT.Jz += epsd;
        coord::PosVelSphMod pJz = actions::ToyMapIsochrone(M, b).map(aaT);
        pJz.r   = (pJz.r   - pp.r)   / epsd;
        pJz.tau = (pJz.tau - pp.tau) / epsd;
        pJz.phi = (pJz.phi - pp.phi) / epsd;
        pJz.pr  = (pJz.pr  - pp.pr)  / epsd;
        pJz.ptau= (pJz.ptau- pp.ptau)/ epsd;
        pJz.pphi= (pJz.pphi- pp.pphi)/ epsd;
        if(aaI.Jz==0) {
            deriv_iso_ok &= !isFinite(ac.dbyJz.tau+ac.dbyJz.ptau);  // should be infinite
            // exclude from comparison
            pJz.tau=pJz.ptau=ac.dbyJz.tau=ac.dbyJz.ptau=der_sph.dbyJz.tau=der_sph.dbyJz.ptau=0;
        }
        aaT = aaI; aaT.Jphi += epsd;
        coord::PosVelSphMod pJp = actions::ToyMapIsochrone(M, b).map(aaT);
        pJp.r   = (pJp.r   - pp.r)   / epsd;
        pJp.tau = (pJp.tau - pp.tau) / epsd;
        pJp.phi = (pJp.phi - pp.phi) / epsd;
        pJp.pr  = (pJp.pr  - pp.pr)  / epsd;
        pJp.ptau= (pJp.ptau- pp.ptau)/ epsd;
        pJp.pphi= (pJp.pphi- pp.pphi)/ epsd;
        // compare grid-a/a derivs with the analytic ones from the isochrone
        //deriv_grid_ok &= equalPosVel(der_sph.dbyJr, ac.dbyJr, 1e-2);
        //deriv_grid_ok &= equalPosVel(der_sph.dbyJz, ac.dbyJz, 1e-2);
        //deriv_grid_ok &= equalPosVel(der_sph.dbyJphi, ac.dbyJphi, 1e-2);
        // compare finite-difference derivs with the analytic ones from the isochrone
        if(!equalPosVel(pJr, ac.dbyJr, 1e-4) && ++numWarnings<10) {
            deriv_iso_ok = false;
            std::cout << "d/dJr: " << pJr << ac.dbyJr << '\n';
        }
        if(!equalPosVel(pJz, ac.dbyJz, 1e-4) && ++numWarnings<10) {
            deriv_iso_ok = false;
            std::cout << "d/dJz: " << pJz << ac.dbyJz << '\n';
        }
        if(!equalPosVel(pJp, ac.dbyJphi, 1e-3) && ++numWarnings<10) {
            deriv_iso_ok = false;
            std::cout << "d/dJphi: " << pJp << ac.dbyJphi << '\n';
        }
        if(output) {
            strm << i*timestep<<"   "<<traj[i].R<<" "<<traj[i].z<<" "<<traj[i].phi<<"  "<<
                toPosVelCyl(pp).R<<" "<<toPosVelCyl(pp).z<<" "<<pp.phi<<"   "<<
                aaI.thetar<<" "<<aaI.thetaz<<" "<<aaI.thetaphi<<"  "<<
                aaS.thetar<<" "<<aaS.thetaz<<" "<<aaS.thetaphi<<"  "<<
                aaG.thetar<<" "<<aaG.thetaz<<" "<<aaG.thetaphi<<"  "<<
                aaF.thetar<<" "<<aaF.thetaz<<" "<<aaF.thetaphi<<"  "<<
            "\n";
        }
    }
    statI.finish();
    statS.finish();
    statG.finish();
    statF.finish();

    bool dispI_ok = statI.rms.Jr<epsd && statI.rms.Jz<epsd && statI.rms.Jphi<epsd;
    bool dispS_ok = statS.rms.Jr<epsd && statS.rms.Jz<epsd && statS.rms.Jphi<epsd;
    bool dispG_ok = statG.rms.Jr<epsd && statG.rms.Jz<epsd && statG.rms.Jphi<epsd;
    bool dispF_ok = statF.rms.Jr<epsd && statF.rms.Jz<epsd && statF.rms.Jphi<epsd;
    bool compareIF =
             fabs(statI.avg.Jr-statF.avg.Jr)<epsr
          && fabs(statI.avg.Jz-statF.avg.Jz)<epsr
          && fabs(statI.avg.Jphi-statF.avg.Jphi)<epsd;
    bool freq_ok = statfrIr.disp() < epsf*epsf && statfrIz.disp() < epsf*epsf;
    bool HofJ_ok = statH.disp() < pow_2(epsf*statH.mean());

    std::cout << "Isochrone"
    ":  Jr="  <<utils::pp(statI.avg.Jr,  14)<<" +- "<<utils::pp(statI.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statI.avg.Jz,  14)<<" +- "<<utils::pp(statI.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statI.avg.Jphi, 6)<<" +- "<<utils::pp(statI.rms.Jphi, 7)<<
    (dispI_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_iso?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<
    (deriv_iso_ok?"":" \033[1;31mDERIVS INCONSISTENT\033[0m ")<<std::endl;

#ifdef PERFTEST
    size_t ncycles=100, npoints=traj.size();
    clock_t clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionsIsochrone(M, b,  traj[i/ncycles]);
    double t_iso_act = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionAnglesIsochrone(M, b,  traj[i/ncycles]);
    double t_iso_ang = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++) {
        actions::ActionAngles aa(statI.avg, actions::Angles(i*0.12345,i*0.23456,i*0.34567));
        actions::mapIsochrone(M, b, aa);
    }
    double t_iso_map = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/t_iso_act, 4)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/t_iso_ang, 4)<<
    ",  map="<<utils::pp(npoints*ncycles/t_iso_map, 4)<<std::endl;
#endif

    std::cout << "Spherical"
    ":  Jr="  <<utils::pp(statS.avg.Jr,  14)<<" +- "<<utils::pp(statS.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statS.avg.Jz,  14)<<" +- "<<utils::pp(statS.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statS.avg.Jphi, 6)<<" +- "<<utils::pp(statS.rms.Jphi, 7)<<
    (dispS_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_sph?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<std::endl;

#ifdef PERFTEST
    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionsSpherical(pot, traj[i/ncycles]);
    double t_sph_act = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionAnglesSpherical(pot, traj[i/ncycles]);
    double t_sph_ang = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++) {
        actions::ActionAngles aa(statS.avg, actions::Angles(i*0.12345,i*0.23456,i*0.34567));
        actions::mapSpherical(pot, aa);
    }
    double t_sph_map = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/t_sph_act, 4)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/t_sph_ang, 4)<<
    ",  map="<<utils::pp(npoints*ncycles/t_sph_map, 4)<<std::endl;
#endif

    std::cout << "Interpol."
    ":  Jr="  <<utils::pp(statG.avg.Jr,  14)<<" +- "<<utils::pp(statG.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statG.avg.Jz,  14)<<" +- "<<utils::pp(statG.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statG.avg.Jphi, 6)<<" +- "<<utils::pp(statG.rms.Jphi, 7)<<
    (dispG_ok?"":" \033[1;31m**\033[0m")<<
    (reversible_grid?"":" \033[1;31mNOT INVERTIBLE\033[0m ")<<
    (deriv_grid_ok?"":" \033[1;31mDERIVS INCONSISTENT\033[0m ")<<std::endl;

#ifdef PERFTEST
    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actGrid.actions(traj[i/ncycles]);
    double t_grid_act = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actGrid.actionAngles(traj[i/ncycles]);
    double t_grid_ang = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++) {
        actions::ActionAngles aa(statS.avg, actions::Angles(i*0.12345,i*0.23456,i*0.34567));
        actGrid.map(aa);
    }
    double t_grid_map = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/t_grid_act, 4)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/t_grid_ang, 4)<<
    ",  map="<<utils::pp(npoints*ncycles/t_grid_map, 4)<<std::endl;
#endif

    std::cout << "Axi.Fudge"
    ":  Jr="  <<utils::pp(statF.avg.Jr,  14)<<" +- "<<utils::pp(statF.rms.Jr,   7)<<
    ",  Jz="  <<utils::pp(statF.avg.Jz,  14)<<" +- "<<utils::pp(statF.rms.Jz,   7)<<
    ",  Jphi="<<utils::pp(statF.avg.Jphi, 6)<<" +- "<<utils::pp(statF.rms.Jphi, 7)<<
    (dispF_ok?"":" \033[1;31m**\033[0m")<<std::endl;

#ifdef PERFTEST
    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionsAxisymFudge(pot, traj[i/ncycles], ifd);
    double t_fudge_act = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    clock = std::clock();
    for(size_t i=0; i<npoints*ncycles; i++)
        actions::actionAnglesAxisymFudge(pot, traj[i/ncycles], ifd);
    double t_fudge_ang = (std::clock()-clock)*1.0/CLOCKS_PER_SEC;

    std::cout << "eval/s:  actions="<<utils::pp(npoints*ncycles/t_fudge_act, 4)<<
    ",  act+ang="<<utils::pp(npoints*ncycles/t_fudge_ang, 4)<<std::endl;
#endif

    std::cout << 
    "Hamiltonian H(J)="<<utils::pp(statH.mean(), 14)<<" +- "<<utils::pp(sqrt(statH.disp()), 7)<<
    ",  H(x,v)="<<utils::pp(statE.mean(), 14)<<" +- "<<utils::pp(sqrt(statE.disp()), 7)<<
    (HofJ_ok?"":" \033[1;31m**\033[0m") <<
    (compareIF?"":" \033[1;31mNOT EQUAL\033[0m ")<<
    (freq_ok?"":" \033[1;31mFREQS NOT CONST\033[0m ")<<
    (anglesMonotonic?"":" \033[1;31mANGLES NON-MONOTONIC\033[0m ")<<std::endl;
    return dispI_ok && dispS_ok && dispG_ok && dispF_ok
        && reversible_iso && reversible_sph && reversible_grid
        && HofJ_ok && compareIF && freq_ok && deriv_iso_ok && deriv_grid_ok && anglesMonotonic;
}

void test_sph_iso()
{
    const double M = 2.7;      // mass and
    const double b = 0.6;      // scale radius of Isochrone potential
    potential::Isochrone potential(M, b);
    actions::ActionFinderSpherical af(potential);
    std::ofstream strm ("test_actions_isochrone_spherical.dat");
    strm << std::setprecision(15);
    for(double lr=-13; lr<=24; lr+=.25) {
        double r = pow(2., lr);
        double vc= v_circ(potential, r);
        double Lc= vc * r;
        double E = 0.5*pow_2(vc) +  // have to cope with ambiguous overloaded member function...
            dynamic_cast<const potential::BasePotential&>(potential).value(coord::PosCyl(r,0,0));
        for(double ll=0; ll<1; ll+=1./128) {
            double L = Lc * pow_2(sin(M_PI_2*ll));
            double Omegar, Omegaz, Jr= af.Jr(E, L, &Omegar, &Omegaz);
            actions::Frequencies fi, fs;
            coord::PosVelCyl point(r, 0, 0, sqrt(vc*vc-pow_2(L/r)), 0, L/r);
            actions::Actions as = actions::actionAnglesSpherical(potential, point, &fs);
            actions::Actions ai = actions::actionAnglesIsochrone(M, b, point, &fi);
            strm << E << ' ' << L/Lc << ' ' <<
                ai.Jr/(Lc-L) << ' ' << as.Jr/(Lc-L) << ' ' << Jr/(Lc-L) << ' ' <<
                fi.Omegar << ' ' << fs.Omegar << ' ' << Omegar << ' ' <<
                fi.Omegaz << ' ' << fs.Omegaz << ' ' << Omegaz << ' ' <<'\n'; 
        }
        strm <<'\n';
    }
}

int main()
{
    //test_sph_iso();
    bool ok=true;
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.3, 1.1, 0.1, 0.4,  0.1), "ordinary case");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 2.2, 1.0, 0.0,  0.5), "Jz==0");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 3.3, 0.0, 0.21, 0.9), "Jr small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0, 4.4, 0.6, 1.0, 1e-4), "Jphi small");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.5, 5.5, 0.5, 0.7, -0.5), "Jphi negative");
    ok &= test_isochrone(coord::PosVelCyl(1.0, 0.0,M_PI, 0.0, 0.0, -0.5), "Jz==0, Jphi<0");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
