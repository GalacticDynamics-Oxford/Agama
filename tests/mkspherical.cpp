/** \file    mkspherical.cpp
    \brief   Manipulation of spherical mass models
    \author  Eugene Vasiliev
    \date    2010-2017

This program constructs spherical mass models from several possible sources:
either from a table with r, m(r) values specifying enclosed mass as a function of radius,
or a built-in analytic density profile,
or an N-body snapshot, in which case the smooth spherically-symmetric density profile
is computed from particle positions and masses.

The spherical model, in turn, may be used to construct a table with several variables
(potential, density, distribution function, etc.) as functions of radius or energy,
or to generate an N-body snapshot, in which particles are distributed
according to the given density profile and isotropically in velocities.

In short, this is a generalization of tools such as `halogen' or `spherICs'
for creating a spherical isotropic model with a given arbitrary density profile
(and optionally a different potential),
and at the same time a useful tool to study dynamical properties of a given N-body system
(or, rather, its spherically-symmetric isotropic counterpart).

*/

#include "utils_config.h"
#include "utils.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "df_spherical.h"
#include "galaxymodel_spherical.h"
#include "particles_io.h"
#include "math_core.h"
#include <iostream>
#include <fstream>
#include <cstdlib>


/// usage description
const char* usage =
    "mkspherical - a tool for creating and analyzing spherical isotropic models.\n"
    "These models may be constructed in three different (mutually exclusive) ways:\n"
    "(a)  by providing the name of the density profile (one of built-in models, e.g., Plummer);\n"
    "(b)  by providing the cumulative mass profile M(r) in a text file;\n"
    "(c)  from an N-body snapshot containing particle positions and velocities.\n"
    "The output, in turn, may consist of a table with several variables "
    "(density, distribution function, etc.) as functions of radius or energy, "
    "and/or an N-body snapshot representing random samples from the spherical model.\n\n"
    "Command-line arguments (case-insensitive, default values in brackets):\n"
    "density=...   either (a) the name of the density model, "
    "or (b) the file name with the cumulative mass profile (text file should contain at least "
    "two columns: radius and enclosed mass; for instance, it may be produced by mkspherical itself).\n"
    "in=...        (c) file with the input N-body snapshot.\n"
    "extractdf=(false) in case of input N-body snapshot, the distribution function may be "
    "extracted from particle energies (if true) or constructed using the Eddington inversion formula "
    "(default; same approach as for an input analytic density profile).\n"
    "potential=(none)  if provided, specifies the name of the potential model "
    "that may be different from the density profile - in this case the density corresponds to "
    "a tracer population which does not contribute to the total potential, and must be given "
    "by a text file (option b). Alternatively, the parameters of the potential (possibly composite) "
    "may be provided in an INI file, and the file name be given in this argument.\n"
    "mbh=(0)       the mass of a central black hole (if nonzero, it is added to the overall potential; "
    "moreover, if any particle in the input N-body snapshot is at origin, "
    "its mass is added to the central black hole and excluded from the density model).\n"
    "mass=(1), scaleRadius=(1), etc. are additional parameters of the density profile (option a), "
    "see the full list in readme.pdf; if a potential is specified, they refer to the potential "
    "instead of the tracer density, which in this case must be given by a text file (option b).\n"
    "tab=...       the name of output text file "
    "(its format allows it to serve as an input mass profile as well).\n" 
    "out=...       the name of output N-body snapshot generated from this spherical model.\n"
    "nbody=...     number of particles in the output snapshot.\n"
    "format=(text) format of the output N-body snapshot "
    "(Text, Nemo, Gadget - only the first letter matters).\n"
    "seed=(0)      random seed (0 means randomize).\n"
    "VERSION=3.0   " __DATE__ "\n";

// print a message and exit
inline void printfail(const char* msg)
{
    std::cerr << msg << '\n';
    exit(1);
}

// construct the distribution function from an N-body snapshot
math::LogLogSpline fitSphericalIsotropicDF(const particles::ParticleArrayCar& bodies,
    const potential::BasePotential& pot, const potential::PhaseVolume& phasevol, int gridsize)
{
    size_t nbody = bodies.size(), nactive = 0;
    std::vector<double> hvalues(nbody), masses(nbody);
    for(size_t i=0; i<nbody; i++) {
        // assemble the array of masses and phase volumes of particles (only finite nonzero values)
        double m = bodies.data[i].second;
        double h = phasevol(totalEnergy(pot, bodies.data[i].first));
        if(isFinite(h) && h > 0 && m > 0) {
            hvalues[nactive] = h;
            masses [nactive] = m;
            nactive++;
        }
    }
    utils::msg(utils::VL_VERBOSE, "fitSphericalIsotropicDF",
        utils::toString(nactive)+" out of "+utils::toString(nbody)+" particles used in DF");
    hvalues.resize(nactive);
    masses. resize(nactive);
    return df::fitSphericalIsotropicDF(hvalues, masses, gridsize);
}

// main program begins here
int main(int argc, char* argv[])
{
    if(argc<=1) {  // print help and exit
        std::cout << usage;
        return 0;
    }

    // combine all command-line arguments to form the output snapshot header
    utils::KeyValueMap args(argc-1, argv+1);
    std::string header="mkspherical " + args.dumpSingleLine();

    // parse command-line arguments, removing (popping) the processed ones from the key-value map
    std::string inputdensity   = args.getString("density");
    std::string inputpotential = args.popString("potential");
    std::string inputsnap      = args.popString("in");
    std::string outputsnap     = args.popString("out");
    std::string outputformat   = args.popString("format", "Text");
    std::string outputtab      = args.popString("tab");
    int seed      = args.popInt("seed");
    int nbody     = args.popInt("nbody");
    int gridsize  = args.popInt("gridsizer", 50);
    double rmin   = args.popDouble("rmin", 0.);
    double rmax   = args.popDouble("rmax", 0.);
    double mbh    = args.popDouble("mbh",  0.);
    bool extractdf= args.popBool("extractdf", false);
    if(!(inputsnap.empty() ^ inputdensity.empty()))
        printfail("Must provide either density=... or in=... as input");
    if((!outputsnap.empty() && nbody==0) || (outputsnap.empty() && nbody>0))
        printfail("Must provide both output snapshot filename (out=...) and number of bodies (nbody=...)");
    if(outputsnap.empty() && outputtab.empty())
        printfail("Must provide output snapshot filename (out=...) and/or output table filename (tab=...)");

    math::LogLogSpline densInterp;       // interpolated density profile constructed from a table
    potential::PtrDensity dens;          // the density profile (analytic or interpolated)
    potential::PtrPotential pot;         // the potential (may be different from the density)
    particles::ParticleArrayCar bodies;  // particles from the input N-body snapshot (if provided)

    // input is a name of a density profile or a file with the cumulative mass or density profile
    if(!inputdensity.empty()) {
        // the choice is made based on whether 'density=...' specifies an existing file name
        if(utils::fileExists(inputdensity)) {
            try {
                // try to read the input file as if it contained a cumulative mass profile
                densInterp = potential::readMassProfile(inputdensity);
                dens.reset(new potential::FunctionToDensityWrapper(densInterp));
            }
            catch(std::exception&) {
                // try to read the input file as if it contained a DensitySphericalHarmonic model
                dens = potential::readDensity(inputdensity);
                // if succeeded, remove the 'density' argument from the list
                args.unset("density");
            }
        } else
            dens = potential::createDensity(args);

        // check if a separate potential was also provided
        if(inputpotential.empty()) {
            pot = potential::Multipole::create(*dens, coord::ST_SPHERICAL, 0, 0, gridsize, rmin, rmax);
        } else if(utils::fileExists(inputpotential)) {
            // create a (possibly composite) potential from the parameters provided in an INI file
            pot = potential::readPotential(inputpotential);
        } else {
            // the createPotential() routine reads the potential name from the 'type=...' parameter
            args.set("type", inputpotential);
            pot = potential::createPotential(args);
        }

        if(!isSpherical(*dens) || !isSpherical(*pot))
            printfail("Density and potential models must be spherical");
        if(!isFinite(dens->totalMass()))
            printfail("Total mass must be finite");
    }

    // otherwise the input must be an N-body snapshot
    if(!inputsnap.empty()) {
        if(!utils::fileExists(inputsnap))
            printfail("Input file does not exist");
        bodies = particles::readSnapshot(inputsnap);
        if(bodies.data.empty())
            printfail("Input N-body snapshot is empty");

        // check that there are no particles at origin - if yes, attribute their mass
        // to the central black hole and reset the mass to zero, otherwise the Multipole will fail
        for(size_t i=0; i<bodies.size(); i++) {
            particles::ParticleArrayCar::ElemType& body = bodies.data[i];
            if(body.first.x == 0 && body.first.y == 0 && body.first.z == 0) {
                mbh += body.second;
                body.second = 0;
            }
        }

        // create the potential, which is the same as the density in this case
        pot  = potential::Multipole::create(bodies, coord::ST_SPHERICAL, 0, 0, gridsize, rmin, rmax);
        dens = pot;
    }

    // if a central massive black hole was provided, construct a two-component potential
    if(mbh>0) {
        std::vector<potential::PtrPotential> components(2);
        components[0] = potential::PtrPotential(new potential::Plummer(mbh, 0));
        components[1] = pot;
        pot = potential::PtrPotential(new potential::Composite(components));
    }

    // construct the mapping between energy and phase volume
    potential::PhaseVolume phasevol((potential::Sphericalized<potential::BasePotential>(*pot)));

    // compute the distribution function either from the density (using the Eddington inversion formula),
    // or from the input N-body snapshot (using the same approach as for density estimation,
    // but this time for the distribution in phase volume; in this case the density
    // reported in the output tab file will be computed from the DF, not from particle positions --
    // it may lead to a different result if the snapshot was not in equilibrium, or the velocities were
    // not in N-body units, or the real DF is not isotropic)
    math::LogLogSpline df = inputsnap.empty() || !extractdf ?
        df::createSphericalIsotropicDF(
            potential::Sphericalized<potential::BaseDensity>(*dens),
            potential::Sphericalized<potential::BasePotential>(*pot)) :
        fitSphericalIsotropicDF(bodies, *pot, phasevol, gridsize);

    // now ripe the fruit: create an output table and/or
    // generate an N-body representation of the spherical model
    if(!outputtab.empty()) {
        galaxymodel::writeSphericalIsotropicModel(
            outputtab, header, df, potential::Sphericalized<potential::BasePotential>(*pot));
    }

    if(!outputsnap.empty()) {
        // generate the samples
        math::randomize(seed);
        particles::ParticleArraySph bodies = galaxymodel::samplePosVel(
            potential::Sphericalized<potential::BasePotential>(*pot), df, mbh? nbody-1 : nbody);
        if(mbh)  // add the central black hole as the 0th particle
            bodies.data.insert(bodies.data.begin(),
                particles::ParticleArraySph::ElemType(coord::PosVelSph(0,0,0,0,0,0), mbh));

        // write the snapshot
        writeSnapshot(outputsnap, bodies, outputformat, units::ExternalUnits(), header);
    }
}
