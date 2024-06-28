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
#include "galaxymodel_fokkerplanck.h"
#include "galaxymodel_spherical.h"
#include "utils.h"
#include "utils_config.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>

const char* usage =
    "PhaseFlow Fokker-Planck solver v.02 build " __DATE__ "\n"
    "Parameters of the simulation should be provided either as command-line arguments "
    "(have the form param=value), or put into an INI file under the section [PhaseFlow] "
    "(each parameter on a separate line, in the same form), and the name of this file "
    "given as the only command-line argument. \n"
    "In case of multi-component models, INI file remains the only possibility: "
    "parameters for each species, including the initial density profile, should be listed "
    "in separate sections with arbitrary names, whereas global parameters (marked as [G]) "
    "should remain in the [PhaseFlow] section. \n"
    "Possible parameters (case-insensitive, optional are marked by a default value in brackets):\n"
    "  ==== Initial conditions ====\n"
    "  density=...     either (a) the name of a built-in density model "
    "(Plummer, Dehnen, Spheroid, Sersic, etc.), in which case additional parameters "
    "of the density profile may be provided, see below;\n"
    "or (b) the name of the input file with two columns: radius and enclosed mass within this radius, "
    "which specifies the initial density profile\n"
    "  mass=(1)        mass of the density profile (in case of a built-in model, option 'a')\n"
    "  scaleRadius=(1) scale radius of the density profile (option 'a')\n"
    "  gamma=(1)       inner power-law slope (for Dehnen and Spheroid, option 'a')\n"
    "  beta=(4)        outer power-law slope (for Spheroid only, option 'a'; "
    "it also has other parameters, see readme.pdf)\n"
    "  Mstar=(1)       mass of a single star; the number of stars in the system is thus "
    "equal to mass/Mstar, and it determines the relaxation rate\n"
    "  Mbh=(0)         [G] additional central mass (black hole, BH) - "
    "may be used both in 'a' and 'b' cases\n"
    "  initBH=(true)   [G] determines whether the BH was present in the system initially (true) "
    "or is added adiabatically (false). In the former case the initial distribution function is "
    "constructed in the combined potential of stars and the BH. In the opposite case it is "
    "initialized from the stellar model only, and then the potential is adiabatically modified "
    "by adding the BH while keeping the DF unchanged\n"
    "  ==== Poisson solver ====\n"
    "  updatePotential=(true)  [G] whether to update the gravitational potential self-consistently "
    "(if false, the potential will be fixed to its initial profile, but the diffusion coefficients "
    "are still recomputed after every FP step)\n"
    "  selfGravity=(true)  [G] whether the density profile of the evolving system contributes to the "
    "total potential; if false, then an external potential must be provided (currently only the BH)\n"
    "  ==== Fokker-Planck solver: grid and time integration ====\n"
    "  coulombLog=...   [G] Coulomb logarithm that enters the expression for the relaxation rate\n"
    "  timeTotal=...    [G] total evolution time (required)\n"
    "  eps=(0.01)       [G] accuracy parameter for time integration: the timestep is eps times "
    "the geometric mean of the two characteristic timescales - f / (df/dt) and the relaxation time\n"
    "  dtmin=(0)        [G] minimum length of FP timestep\n"
    "  dtmax=(inf)      [G] maximum length of FP timestep (both these limits may modify the timestep "
    "that was computed using the eps parameter)\n"
    "  hmin=(),hmax=()  [G] the extent of the grid in phase volume h; the grid is logarithmically "
    "spaced between hmin and hmax, and by default encompasses a fairly large range of h "
    "(depending on the potential, but typically hmin~1e-10, hmax~1e10)\n"
    "  rmax=()          [G] alternative specification of the outer grid boundary in terms of radius, "
    "not h (if provided, overrides the value of hmax)\n"
    "  gridSizeDF=(200) [G] number of grid points in h\n"
    "  method=(0)       [G] the choice of discretization method (0: Chang&Cooper, 1-3: finite element)\n"
    "  ==== Central black hole (sink) ====\n"
    "  captureRadius=(0)  in the case of a central BH, specifies the capture radius and turns on "
    "the absorbing boundary condition f(hmin)=0. In this case hmin is determined by the energy at "
    "which the capture occurs from a circular orbit, i.e. there should be no stars at lower energies. "
    "Setting captureRadius=0 implies a zero-flux boundary condition, even in the presense of the BH. "
    "It may grow along with the BH mass, as described by the following parameter. "
    "In the multi-component case, these numbers should all be either zero or non-zero for all species\n"
    "  captureRadiusScalingExp=(0)  is the power-law index of the dependence of capture radius "
    "on the central BH mass (1/3 for tidally disrupted stars, 1 for directly captured compact objects)\n"
    "  captureMassFraction=(1)  in the case of non-zero capture radius, the fraction of flux through "
    "the capture boundary that is added to the BH mass\n"
    "  lossCone=(true)  [G] in the case of a central BH and a non-zero capture radius, "
    "further turns on loss-cone draining, i.e., the decay of DF at all energies with a rate "
    "that corresponds to the steady-state solution for the diffusion in the angular momentum "
    "direction with an appropriate boundary condition (empty or full loss cone) determined from "
    "the capture radius and the relaxation rate\n"
    "  speedOfLight=(0) [G] if set to nonzero, account for energy loss due to emission of "
    "gravitational waves\n"
    "  ==== Star formation (source) ====\n"
    "  sourceRate=(0)   source term: total mass injected per unit time\n"
    "  sourceRadius=(0) radius within which the mass injection occurs\n"
    "  ==== Output ====\n"
    "  fileOut=()    [G] name (common prefix) of output files, time is appended to the name; "
    "if not provided, don't output anything\n"
    "  timeOut=(0)   [G] (maximum) time interval between storing the output profiles "
    "(0 means unlimited)\n"
    "  nstepOut=(0)  [G] maximum number of FP steps between outputs (0 means unlimited; " 
    "if neither of the two parameters is set, will not produce any output files)\n"
    "  fileLog=(fileOut+\".log\")  [G] name of the file with overall diagnostic information "
    "printed every timestep\n";

/// upper limit on the number of steps (arbitrary)
const int MAXNSTEP = 1e8;

/// write text file(s) with various quantities in a spherical model
void exportTable(const std::string& filename, const std::string& header, const double timeSim,
    const galaxymodel::FokkerPlanckSolver& fp)
{
    std::cerr << "Writing output file at time " << timeSim << '\n';
    unsigned int numComp = fp.numComp();  // number of components
    for(unsigned int comp=0; comp<numComp; comp++) {
        std::string fullfilename = filename + utils::toString(timeSim);
        if(numComp>1 && numComp<=26)
            fullfilename += char(comp+97); /* suffix a,b,... */
        else if(numComp>26)
            fullfilename += "_"+utils::toString(comp); /* numerical suffix */
        galaxymodel::writeSphericalIsotropicModel(fullfilename, header,
        /*f(h)*/      *fp.df(comp),
        /*potential*/ potential::FunctionToPotentialWrapper(*fp.potential()),
        /*gridh*/      fp.gridh(),
        /*drain time*/ fp.drainTime(comp));
    }
}

/// print out diagnostic information
void printInfo(std::ofstream& strmLog, const double timeSim, const galaxymodel::FokkerPlanckSolver& fp)
{
    std::cerr << "time: " << utils::pp(timeSim, 9) << '\r';
    strmLog <<
    utils::pp(timeSim,          11) + ' ' +
    utils::pp(fp.Phi0(),        11) + ' ' +
    utils::pp(fp.Etot(),        11) + ' ' +
    utils::pp(fp.Ekin(),        11) + ' ' +
    utils::pp(fp.sourceEnergy(),11) + ' ' +
    utils::pp(fp.drainEnergy(), 11) + ' ' +
    utils::pp(fp.Mbh(),         11);
    for(unsigned int c=0; c<fp.numComp(); c++)
        strmLog << ' ' + utils::pp(fp.Mass(c),       11);
    for(unsigned int c=0; c<fp.numComp(); c++)
        strmLog << ' ' + utils::pp(fp.sourceMass(c), 11);
    for(unsigned int c=0; c<fp.numComp(); c++)
        strmLog << ' ' + utils::pp(fp.drainMass(c),  11);
    strmLog << '\n';
}

typedef shared_ptr<utils::KeyValueMap> PtrKeyValueMap;

/// initial density profiles of model components
std::vector<potential::PtrDensity> densities;

/// construct the density of a single component and initialize its other parameters
galaxymodel::FokkerPlanckComponent initComponent(utils::KeyValueMap& args)
{
    galaxymodel::FokkerPlanckComponent comp;
    comp.Mstar                  = args.popDouble("Mstar", comp.Mstar);
    comp.captureRadius          = args.popDouble("captureRadius", comp.captureRadius);
    comp.captureMassFraction    = args.popDouble("captureMassFraction", comp.captureMassFraction);
    comp.captureRadiusScalingExp= args.popDouble("captureRadiusScalingExp", comp.captureRadiusScalingExp);
    comp.sourceRate             = args.popDouble("sourceRate", comp.sourceRate);
    comp.sourceRadius           = args.popDouble("sourceRadius", comp.sourceRadius);
    // remaining arguments must specify the density profile or a file
    std::string density         = args.getString("density");
    if(density.empty())
        throw std::invalid_argument("Need to provide density=... parameter");
    if(utils::fileExists(density))
        // if the density=... argument refers to an existing file, read the cumulative mass profile
        comp.initDensity.reset(new math::LogLogSpline(potential::readMassProfile(density)));
    else {
        // otherwise create a built-in model
        densities.push_back(potential::createDensity(args));
        // DensityWrapper only contains a reference to a BaseDensity object, therefore we need
        // to keep the actual density object alive until the FokkerPlanckSolver is constructed -
        // this is why there is a global array of PtrDensity pointers
        comp.initDensity.reset(new potential::Sphericalized<potential::BaseDensity>(*densities.back()));
    }
    return comp;
}

int main(int argc, char* argv[])
{
    if(argc<=1) {  // print command-line options and exit
        std::cout << usage;
        return 0;
    }

    // parse command-line arguments:
    // if only one argument was provided and it doesn't have a '=' symbol,
    // it's assumed to be the name of an INI file with all parameters stored in one or more sections,
    // otherwise all parameters are provided as command-line arguments (should have the form key=value),
    // and in this case they are put into a single section.
    std::vector<PtrKeyValueMap> sections;

    // section containing the global parameters, should be named [PhaseFlow] in the INI file
    PtrKeyValueMap globalSection;

    // parameters for each species of a multi-component model:
    // if there is only one, they may be kept in the same section as the global params,
    // or if there is no ini file, just provided among other command-line arguments;
    // alternatively, the INI file should contain one or more arbitrarily named sections
    // with a 'density=...' line
    std::vector<PtrKeyValueMap> compSections;

    // all parameters are combined into a single string forming the header written into the output files
    std::string header="phaseflow";

    if(argc>=2 && std::string(argv[1]).find('=')==std::string::npos) {
        // parse an INI file
        utils::ConfigFile config(argv[1]);
        std::vector<std::string> secNames = config.listSections();
        for(unsigned int i=0; i<secNames.size(); i++) {
            if(secNames[i].empty())
                continue;
            sections.push_back(PtrKeyValueMap(new utils::KeyValueMap(config.findSection(secNames[i]))));
            if(utils::stringsEqual(secNames[i], "PhaseFlow"))
                globalSection = sections.back();
            if(sections.back()->contains("density"))  // not mutually exclusive with the above condition
                compSections.push_back(sections.back());
            header += " " + sections.back()->dumpSingleLine();
        }
        if(!globalSection || compSections.empty())
            throw std::invalid_argument(
                "INI file should contain a section named [PhaseFlow] with global parameters, "
                "and one or more sections with a density=... parameter");
    } else {
        // take the command-line arguments themselves (excluding the 0th)
        sections.push_back(PtrKeyValueMap(new utils::KeyValueMap(argc-1, argv+1)));
        globalSection = sections[0];
        if(sections[0]->contains("density"))
            compSections.push_back(sections[0]);
        else
            throw std::invalid_argument("Should provide a density=... parameter");
        header += " " + globalSection->dumpSingleLine();
    }

    // first parse the global parameters, removing the known ones from the KeyValueMap;
    // if the same section is used for specifying the parameters for the single component,
    // the remaining parameters should describe its density profile and have no unknown keys.
    double eps          = globalSection->popDouble   ("eps", 1e-2);
    double dtmin        = globalSection->popDouble   ("dtmin", 0);
    double dtmax        = globalSection->popDouble   ("dtmax", INFINITY);
    bool initBH         = globalSection->popBool     ("initBH", true);
    double Mbh          = globalSection->popDouble   ("Mbh", 0);
    std::string fileOut = globalSection->popStringAlt("fileOut", "fileOutput");
    std::string fileLog = globalSection->popString   ("fileLog", fileOut.empty() ? "" : fileOut+".log");
    double timeOut      = globalSection->popDoubleAlt("timeOut", "outputInterval", 0);
    int nstepOut        = globalSection->popInt      ("nstepOut", 0);
    double timeTotal    = globalSection->popDoubleAlt("time", "timeTotal", 0);
    if(fileOut.empty())
        timeOut = nstepOut = 0;
    if(timeTotal <= 0)
        throw std::invalid_argument("Need to provide timeTotal=... parameter");
    galaxymodel::FokkerPlanckParams params;   // global parameters passed to the FokkerPlanckSolver
    params.method  = (galaxymodel::FokkerPlanckMethod)globalSection->popInt("method", params.method);
    params.hmin            = globalSection->popDouble("hmin", params.hmin);
    params.hmax            = globalSection->popDouble("hmax", params.hmax);
    params.rmax            = globalSection->popDouble("rmax", params.rmax);
    params.gridSize        = globalSection->popIntAlt("gridsize", "gridSizeDF", params.gridSize);
    params.coulombLog      = globalSection->popDouble("coulombLog", params.coulombLog);
    if(params.coulombLog <= 0)
        throw std::invalid_argument("Need to provide coulombLog=... parameter "
            "controlling the relaxation rate (10 is a reasonable fiducial value)");
    params.selfGravity     = globalSection->popBool  ("selfGravity", params.selfGravity);
    params.updatePotential = globalSection->popBool  ("updatePotential", params.updatePotential);
    params.lossConeDrain   = globalSection->popBool  ("lossCone", params.lossConeDrain);
    params.speedOfLight    = globalSection->popDouble("speedOfLight", params.speedOfLight);

    // init all model components
    std::vector<galaxymodel::FokkerPlanckComponent> components;
    for(unsigned int i=0; i<compSections.size(); i++)
        components.push_back(initComponent(*compSections[i]));

    // output log file
    std::ofstream strmLog;
    if(!fileLog.empty()) {
        strmLog.open(fileLog.c_str());
        strmLog << "#" + header + "\n"
        "#Time       Phi_star(0) Etotal      Ekin        Esource     Esink       Mbh         ";
        if(components.size() == 1) {
            strmLog << "Mass        Msource     Msink       ";
        } else {
            for(unsigned int c=0; c<components.size(); c++)
                strmLog << "Mass" + utils::toString(c) + "       ";
            for(unsigned int c=0; c<components.size(); c++)
                strmLog << "Msource" + utils::toString(c) + "    ";
            for(unsigned int c=0; c<components.size(); c++)
                strmLog << "Msink" + utils::toString(c) + "      ";
        }
        strmLog << "\n" << std::flush;
    }

    // if the central black hole is grown adiabatically, start from a zero initial BH mass
    params.Mbh = initBH ? Mbh : 0;

    // create the Fokker-Planck solver
    galaxymodel::FokkerPlanckSolver fp(params, components);

    if(!initBH)  // set the initial BH mass and modify the potential adiabatically
        fp.setMbh(Mbh);

    // begin the simulation
    double timeSim = 0, prevTimeOut = -INFINITY;
    double dt = fmin(dtmax, fmax(dtmin, 0.01 * eps * fp.relaxationTime()));
    int nstep = 0, prevNstepOut = -nstepOut;
    while(timeSim < timeTotal && nstep < MAXNSTEP) {
        // print out the diagnostic information
        printInfo(strmLog, timeSim, fp);

        // store output files once in a while
        if( (timeOut  > 0 && timeSim >= prevTimeOut  + timeOut) ||
            (nstepOut > 0 && nstep   >= prevNstepOut + nstepOut) )
        {
            exportTable(fileOut, header, timeSim, fp);
            prevTimeOut  = timeSim;
            prevNstepOut = nstep;
        }

        // adjust the length of the upcoming timestep
        if(timeSim + dt >= timeTotal) {
            dt = timeTotal - timeSim;
            timeSim = timeTotal;
        } else if(timeOut > 0 && timeSim + dt > prevTimeOut + timeOut)
        {   // end the current timestep exactly at the next export time
            dt = prevTimeOut + timeOut - timeSim;
            timeSim = prevTimeOut + timeOut;
        } else
            timeSim += dt;

        // perform one timestep of the Fokker-Planck solver
        double relChange = fp.evolve(dt);

        // adjust the length of the next timestep considering the characteristic evolution timescale
        dt = fmin(dtmax, fmax(dtmin, eps * sqrt(fp.relaxationTime() * dt / relChange)));
        nstep++;
    }

    // last output
    if((timeOut>0 || nstepOut>0) && prevTimeOut < timeSim) {
        printInfo(strmLog, timeSim, fp);
        exportTable(fileOut, header, timeSim, fp);
    }
}
