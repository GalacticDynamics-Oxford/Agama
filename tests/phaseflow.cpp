/** \name   phaseflow.cpp
    \brief  PhaseFlow - Fokker-Planck solver for spherical isotropic stellar systems
    \author Eugene Vasiliev
    \date   2016-2017

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
    The FP equation for f(h) is solved using a finite-difference scheme on a grid
    in phase space, where the argument of DF is the phase volume h instead of
    the more commonly used energy (hence the name of the program).
    The density profile of the system and the FP coefficients are recomputed from
    the DF every few timesteps.
    The output profiles of density, potential and f are written to text files after
    user-specified intervals of time, or after a given number of timesteps;
    these files may also serve as input mass models.

    A classical example of core collapse for an initial Plummer model in virial units
    (the simulation progresses to t~1.444 until a pure power-law cusp forms,
    and bails out when the core radius shrinks to the innermost grid segment):
    ~~~~
    phaseflow.exe time=1.5 density=plummer scaleradius=0.589 eps=1e-4 nstepout=10000 fileout=plum
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

const char* usage =
    "PhaseFlow Fokker-Planck solver v.01 build " __DATE__ "\n"
    "Command-line arguments (optional are marked by a default value in brackets):\n"
    "density=...      either (a) the name of a built-in density model "
    "(Plummer, Dehnen, SpheroidDensity, etc.), in which case additional parameters "
    "of the density profile may be provided, see below;\n"
    "or (b) the name of the input file with two columns: radius and enclosed mass within this radius, "
    "which specifies the initial density profile\n"
    "mass=(1)         mass of the density profile (in case of a built-in model, option 'a')\n"
    "scaleradius=(1)  scale radius of the density profile (option 'a')\n"
    "gamma=(1)        inner power-law slope (for Dehnen and SpheroidDensity, option 'a')\n"
    "beta=(4)         outer power-law slope (for SpheroidDensity only, option 'a'; "
    "it also has other parameters, see readme.pdf)\n"
    "mbh=(0)          additional central mass (black hole) - may be used in both 'a' and 'b' cases\n"
    "time=...         total evolution time (required)\n"
    "eps=(0.001)      max relative change of DF in one timestep of the FP solver "
    "(determines the timestep)\n"
    "dtmin=(0)        minimum length of FP timestep\n"
    "dtmax=(inf)      maximum length of FP timestep (both these limits may modify the timestep "
    "that was computed using the criterion of max relative DF change)\n"
    "nsubstep=(8)     # of FP timesteps between each recomputation of potential and diffusion coefs\n"
    "updatepot=(true) whether to update the gravitational potential self-consistently "
    "(if false, the potential will be fixed to its initial profile, but the diffusion coefficients "
    "are still recomputed after every 'nsubstep' FP steps)\n"
    "selfgravity=(true) whether the density profile of the evolving system contributes to "
    "the total potential; if false, then an external potential must be provided "
    "(currently only the black hole), and updatepot has no effect\n"
    "zeroin=(false)   specifies the boundary condition at the innermost grid point: "
    "true means f(h_min)=0 (absorbing boundary), false (default) means no flux through the boundary\n"
    "zeroout=(false)  same for the outer boundary condition\n"
    "hmin=(),hmax=()  the extent of the grid in phase volume h; the grid is logarithmically spaced "
    "between hmin and hmax, and by default encompasses a fairly large range of h "
    "(depending on the potential, but typically hmin~1e-10, hmax~1e10)\n"
    "gridsize=(200)   number of grid points in h\n"
    "fileout=()       name (common prefix) of output files, time is appended to the name; "
    "if not provided, don't output anything\n"
    "timeout=(0)      (maximum) time interval between storing the output profiles (0 means unlimited)\n"
    "nstepout=(0)     maximum number of FP steps between outputs (0 means unlimited; "
    "if neither of the two parameters is set, will not produce any output files)\n";

/// write a text file with various quantities in a spherical model
void exportTable(const std::string& filename, const double timesim,
    const galaxymodel::FokkerPlanckSolver& fp)
{
    std::cerr << "Writing output file at time " << timesim << '\n';
    galaxymodel::writeSphericalModel(filename + utils::toString(timesim),
        /*model*/     galaxymodel::SphericalModel(
            /*h(E)*/  fp.getPhaseVolume(),
            /*f(h)*/  math::LogLogSpline(fp.getGridH(), fp.getGridF(), NAN, NAN, true),
            /*gridh*/ fp.getGridH()),
        /*potential*/ potential::FunctionToPotentialWrapper(fp.getPotential()),
        /*density*/   NULL,
        /*gridh*/     fp.getGridH());
}

int main(int argc, char* argv[])
{
    if(argc<=1) {  // print command-line options and exit
        std::cout << usage;
        return 0;
    }
    utils::KeyValueMap args(argc-1, argv+1);
    double mbh     = args.getDouble("Mbh", 0);
    double time    = args.getDouble("time", 0);
    double eps     = args.getDouble("eps", 1e-3);
    double dtmin   = args.getDouble("dtmin", 0);
    double dtmax   = args.getDouble("dtmax", INFINITY);
    double hmin    = args.getDouble("hmin", 0);
    double hmax    = args.getDouble("hmax", 0);
    int gridsize   = args.getInt   ("gridsize", 200);
    int nsubstep   = args.getInt   ("nsubstep", 8);
    int nstepout   = args.getInt   ("nstepout", 0);
    double timeout = args.getDouble("timeout", 0);
    bool updatepot = args.getBool  ("updatepot", true);
    std::string density = args.getString("density");
    std::string fileout = args.getString("fileout");
    if(fileout.empty())
        timeout = nstepout = 0;
    if(time<=0 || density.empty()) {
        std::cout << usage;
        return 0;
    }
    std::vector<double> gridh;
    if(hmin>0 && hmax>hmin && gridsize>3)
        gridh = math::createExpGrid(gridsize, hmin, hmax);         // user-defined grid extent
    potential::PtrDensity initModel = utils::fileExists(density) ? // if this refers to an existing file,
        galaxymodel::readMassProfile(density, &mbh) :              // read the cumulative mass profile,
        potential::createDensity(args);                            // otherwise create a built-in model
    potential::PtrPotential extPot(mbh>0 ? new potential::Plummer(mbh, 0) : NULL);
    galaxymodel::FokkerPlanckSolver fp(
        potential::DensityWrapper(*initModel),  // initial density profile used to construct the DF
        extPot,   // optional external potential (in the form of a central black hole)
        gridh,    // optional grid in phase volume (if none provided, will construct an automatic one)
        static_cast<galaxymodel::FokkerPlanckSolver::Options>(   // additional options
        (args.getBool("selfgravity", true) ? 0 : galaxymodel::FokkerPlanckSolver::FP_NO_SELF_GRAVITY) |
        (args.getBool("zeroin",  false) ? galaxymodel::FokkerPlanckSolver::FP_ZERO_DF_INNER : 0) |
        (args.getBool("zeroout", false) ? galaxymodel::FokkerPlanckSolver::FP_ZERO_DF_OUTER : 0) ) );

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
            exportTable(fileout, timesim, fp);
            prevtimeout  = timesim;
            prevnstepout = nstep;
        }
        double relChange = fp.doStep(dt);
        timesim += dt;
        dt = fmin(dtmax, fmax(dtmin, dt * eps / relChange));
        nstep++;
        if(nstep % nsubstep == 0) {
            if(updatepot)
                fp.reinitPotential();
            fp.reinitDifCoefs();
        }
    }
    if(timeout>0 || nstepout>0)  // final output
        exportTable(fileout, timesim, fp);
}
