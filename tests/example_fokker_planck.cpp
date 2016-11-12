/** \name   example_fokker_planck.cpp
    \author Eugene Vasiliev
    \date   Nov 2016

*/
#include "potential_factory.h"
#include "potential_analytic.h"
#include "galaxymodel_spherical.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>


/// write a text file with various quantities in a spherical model
void exportTable(const char* filename, const galaxymodel::FokkerPlanckSolver& fp)
{
    const potential::Interpolator& pot = fp.getPotential();
    const potential::PhaseVolume&  pv  = fp.getPhaseVolume();
    const std::vector<double> gridh = fp.getGridH();
    const std::vector<double> gridf = fp.getGridF();    
    std::vector<double> gridr(gridh.size()), gridPhi(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        gridPhi[i] = pv. E(gridh[i]);
        gridr  [i] = pot.R_max(gridPhi[i]);
    }
    math::LogLogSpline df(gridh, gridf);
    std::vector<double> gridrho = galaxymodel::computeDensity(df, pv, gridPhi);
    galaxymodel::SphericalModel model(pv, df);
    double mult = 16*M_PI*M_PI * model.cumulMass();
    
    std::ofstream strm(filename);
    strm << "r       \tM       \tPhi     \trho     \tf       \tg       \th       \t"
    "D_h      \tD_hh     \tFlux     \tdf/dh\n";
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h=gridh[i], f=gridf[i], Phi=gridPhi[i], rho=gridrho[i], r=gridr[i], g, dPhidr, dfdh;
        df.evalDeriv(h, NULL, &dfdh);
        pv.E(h, &g);
        pot.evalDeriv(r, NULL, &dPhidr);
        double
        M     = dPhidr * r*r,
        logh  = log(h),
        intf  = model.I0(logh),
        intfg = model.cumulMass(logh),
        intfh = model.cumulEkin(logh) * (2./3),
        D_h   = mult * intfg,                   // drift coefficient D_h
        D_hh  = mult * g * (h * intf + intfh),  // diffusion coefficient D_hh
        Flux  = -(D_hh * dfdh + D_h * f);
        strm <<
            utils::pp(r,   14) + '\t' +
            utils::pp(M,   14) + '\t' +
            utils::pp(Phi, 14) + '\t' +
            utils::pp(rho, 14) + '\t' +
            utils::pp(f,   14) + '\t' +
            utils::pp(g,   14) + '\t' +
            utils::pp(h,   14) + '\t' +
            utils::pp(D_h, 14) + '\t' +
            utils::pp(D_hh,14) + '\t' +
            utils::pp(Flux,14) + '\t' +
            utils::pp(dfdh,14) + '\n';
    }
}


void help()
{
    std::cout << "PhaseFlow Fokker-Planck solver v.0, "__DATE__"\n"
    "Command-line arguments (optional are marked by a default value in brackets):\n"
    "density=...      name of the density model (Plummer, Dehnen, SpheroidDensity, etc.)\n"
    "mass=(1)         mass of the density profile\n"
    "scaleradius=(1)  scale radius of the density profile\n"
    "gamma=(1)        inner power-law slope (for Dehnen and SpheroidDensity)\n"
    "beta=(4)         outer power-law slope (for SpheroidDensity only; "
    "it also has other parameters, see readme)\n"
    "mbh=(0)          additional central mass (black hole)\n"
    "time=...         total evolution time\n"
    "eps=(0.01)       max relative change of DF in one timestep of the FP solver "
    "(determines the timestep)\n"
    "dtmin=(0)        minimum length of FP timestep\n"
    "dtmax=(inf)      maximum length of FP timestep\n"
    "nsubstep=(8)     # of FP timesteps between each recomputation of potential and diffusion coefs\n"
    "timeout=(0)      (maximum) time interval between storing the output profiles (0 means unlimited)\n"
    "nstepout=(0)     maximum number of FP steps between outputs (0 means unlimited)\n";
    exit(0);
}

int main(int argc, char* argv[])
{
    if(argc<=1)
        help();
    utils::KeyValueMap args(argc-1, argv+1);
    double mbh  = args.getDouble("Mbh", 0);
    double time = args.getDouble("time", 0);
    double eps  = args.getDouble("eps", 1e-3);
    double dtmin= args.getDouble("dtmin", 0);
    double dtmax= args.getDouble("dtmax", INFINITY);
    int nsubstep= args.getInt("nsubstep", 8);
    int nstepout= args.getInt("nstepout", 0);
    double timeout=args.getDouble("timeout", 0);
    std::string fileout = args.getString("fileout");
    if(fileout.empty())
        timeout = nstepout = 0;
    potential::PtrPotential extPot(mbh>0 ? new potential::Plummer(mbh, 0) : NULL);
    if(time<=0 || !args.contains("density"))
        help();
    potential::PtrDensity initModel = potential::createDensity(args);
    galaxymodel::FokkerPlanckSolver fp(potential::DensityWrapper(*initModel), extPot);
    
    double timesim = 0, dt = (dtmin>0 ? dtmin : 1e-8), prevtimeout = -INFINITY;
    int nstep = 0, prevnstepout = -nstepout;
    while(timesim <= time && nstep < 1e6) {
        if(nstep % nsubstep == 0)
            std::cout <<
            "time: " + utils::pp(timesim, 9) + "\t"
            "Phi0: " + utils::pp(fp.Phi0, 9) + "\t"
            "Mass: " + utils::pp(fp.Mass, 9) + "\t"
            "Etot: " + utils::pp(fp.Etot + (mbh!=0 ? mbh * fp.Phi0 : 0), 9) + "\t"
            "Ekin: " + utils::pp(fp.Ekin, 9) + "\n";
        if( (timeout  > 0 && timesim >= prevtimeout  + timeout) || 
            (nstepout > 0 && nstep   >= prevnstepout + nstepout) )
        {
            std::cerr << "Writing output file at time " << timesim << '\n';
            exportTable((fileout+utils::toString(timesim)).c_str(), fp);
            prevtimeout = timesim;
            prevnstepout= nstep;
        }
        double relChange = fp.doStep(dt);
        timesim += dt;
        dt = fmin(dtmax, fmax(dtmin, dt * eps / relChange));
        nstep++;
        if(nstep % nsubstep == 0) {
            fp.reinitPotential();
            fp.reinitDifCoefs();
        }
    }
    if(timeout>0 || nstepout>0) { // final output
        std::cerr << "Writing output file at time " << timesim << '\n';
        exportTable((fileout+utils::toString(timesim)).c_str(), fp);
    }
}