/** \name   example_fokker_planck.cpp
    \author Eugene Vasiliev
    \date   Nov 2016

    This program follows the evolution of a spherical isotropic stellar system
    driven by two-body relaxation and described by a coupled system of equations:
    the Fokker--Planck (FP) equation for the distribution function (DF),
    and the Poisson equation for the potential.
    Initial conditions may be provided either as a text file with a mass model
    (pairs of values -- radius and enclosed mass), or as a built-in density profile;
    in either case, the initial DF is constructed via the Eddington inversion formula.
    The evolution of the DF is driven by relaxation, and the coefficients of the FP
    equation are computed from the DF itself for the given potential, which corresponds
    to the density profile of the evolving system, plus optionally an external component
    (in the present case, it may be a central point mass).
    We assume a single-mass stellar system in computing the relaxation coefficients.
    The time variable is measured in units of "relaxation time", defined as
    the N-body time unit of the system multiplied by the factor N / \ln\Lambda,
    where N is the number of stars in the system.
    The FP equation for f(h) is solved using a finite-difference scheme
    on a grid in phase space, where the argument of DF is the phase volume h.
    The density profile of the system and the FP coefficients are recomputed from
    the DF every few timesteps.
    The output profiles of density, potential and f are written to text files after
    user-specified intervals of time, or after a given number of timesteps;
    these files may also serve as input mass models.

    A classical example of core collapse for an initial Plummer model in virial units
    (the simulation progresses to t~1.444 until a pure power-law cusp forms,
    and bails out when the core radius shrinks to the innermost grid segment):
    ~~~~
    example_fokker_planck.exe time=1.5 density=plummer scaleradius=0.589 eps=1e-4 nstepout=10000 fileout=plum
    ~~~~
*/
#include "potential_factory.h"
#include "potential_analytic.h"
#include "potential_multipole.h"
#include "galaxymodel_spherical.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>

/// read a text file with the mass profile
/// (two columns -- radius and the enclosed mass within this radius),
/// and construct an interpolated density profile
potential::PtrDensity createModelFromFile(const char* filename, double& Mbh)
{
    std::ifstream strm(filename);
    if(!strm) {
        std::cout << "Can't read input file " << filename << "\n";
        exit(0);
    }
    std::vector<double> radius, mass;
    const std::string validDigits = "0123456789.-+";
    double m0 = 0;   // possible central mass to be subtracted from all other elements of input table
    while(strm) {
        std::string str;
        std::getline(strm, str);
        std::vector<std::string> elems = utils::splitString(str, " \t,;");
        if(elems.size() < 2 || validDigits.find(elems[0][0]) == std::string::npos)
            continue;
        double r = utils::toDouble(elems[0]),  m = utils::toDouble(elems[1]);
        if(r>0) {
            radius.push_back(r);
            mass.push_back(m-m0);
        } else {      // a nonzero mass at r=0 representing a central black hole is allowed
            m0 = m;   // in the input file, but this point will not be stored in the arrays
        }
    }
    Mbh += m0;   // central point mass stored in the file will be added to Mbh
    return potential::PtrDensity(new potential::DensitySphericalHarmonic(radius,
        std::vector<std::vector<double> >(1, galaxymodel::densityFromCumulativeMass(radius, mass))));
}

/// write a text file with various quantities in a spherical model
void exportTable(const char* filename, const galaxymodel::FokkerPlanckSolver& fp)
{
    const potential::Interpolator& pot = fp.getPotential();
    const potential::PhaseVolume&  pv  = fp.getPhaseVolume();
    const std::vector<double>   gridh  = fp.getGridH();
    const std::vector<double>   gridf  = fp.getGridF();    
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

/// print command-line options and exit
void help()
{
    std::cout << "PhaseFlow Fokker-Planck solver v.01, "__DATE__"\n"
    "Command-line arguments (optional are marked by a default value in brackets):\n"
    "filein=...       name of the input file with two columns: radius and enclosed mass, "
    "which specifies the initial density profile; alternatively, a builtin model may be used instead\n"
    "density=...      name of the builtin density model (Plummer, Dehnen, SpheroidDensity, etc.) -- "
    "either filein or density should be provided\n"
    "mass=(1)         mass of the density profile\n"
    "scaleradius=(1)  scale radius of the density profile\n"
    "gamma=(1)        inner power-law slope (for Dehnen and SpheroidDensity)\n"
    "beta=(4)         outer power-law slope (for SpheroidDensity only; "
    "it also has other parameters, see readme)\n"
    "mbh=(0)          additional central mass (black hole)\n"
    "time=...         total evolution time (required)\n"
    "eps=(0.001)      max relative change of DF in one timestep of the FP solver "
    "(determines the timestep)\n"
    "dtmin=(0)        minimum length of FP timestep\n"
    "dtmax=(inf)      maximum length of FP timestep (both these limits may modify the timestep "
    "that was computed using the criterion of max relative DF change)\n"
    "nsubstep=(8)     # of FP timesteps between each recomputation of potential and diffusion coefs\n"
    "hmin=(),hmax=()  the extent of the grid in phase volume h; the grid is logarithmically spaced "
    "between hmin and hmax, and by default encompasses a fairly large range of h "
    "(depending on the potential, but typically hmin~1e-10, hmax~1e10)\n"
    "gridsize=(200)   number of grid points in h\n"
    "fileout=()       name (common prefix) of output files, time is appended to the name; "
    "if not provided, don't output anything\n"
    "timeout=(0)      (maximum) time interval between storing the output profiles (0 means unlimited)\n"
    "nstepout=(0)     maximum number of FP steps between outputs (0 means unlimited; "
    "if neither of the two parameters is set, will not produce any output files)\n";
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
    double hmin = args.getDouble("hmin", 0);
    double hmax = args.getDouble("hmax", 0);
    int gridsize= args.getInt("gridsize", 200);
    int nsubstep= args.getInt("nsubstep", 8);
    int nstepout= args.getInt("nstepout", 0);
    double timeout=args.getDouble("timeout", 0);
    std::string fileout = args.getString("fileout");
    if(fileout.empty())
        timeout = nstepout = 0;
    if(time<=0 || !(args.contains("density") ^ args.contains("filein")) )
        help();
    std::vector<double> gridh;
    if(hmin>0 && hmax>hmin && gridsize>3)
        gridh = math::createExpGrid(gridsize, hmin, hmax);
    potential::PtrDensity initModel = args.contains("density") ?
        potential::createDensity(args) :
        createModelFromFile(args.getString("filein").c_str(), mbh);
    potential::PtrPotential extPot(mbh>0 ? new potential::Plummer(mbh, 0) : NULL);
    galaxymodel::FokkerPlanckSolver fp(potential::DensityWrapper(*initModel), extPot, gridh);
    
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
            prevtimeout  = timesim;
            prevnstepout = nstep;
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