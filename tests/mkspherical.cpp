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
    "or (b) the file name with the cumulative mass profile (text file should contain "
    "at least two columns: radius and enclosed mass).\n"
    "in=...        (c) file with the input N-body snapshot.\n"
    "potential=(none)  if provided, specifies the name of the potential model "
    "that may be different from the density profile - in this case the density corresponds to "
    "a tracer population which does not contribute to the total potential, "
    "and must be given by a text file (option b).\n"
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
    "VERSION=3.0   " __DATE__ "\n";

// print a message and exit
void printfail(const char* msg)
{
    std::cerr << msg << '\n';
    exit(0);
}

// construct the distribution function from an N-body snapshot
math::LogLogSpline fitSphericalDF(const particles::ParticleArrayCar& bodies,
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
    hvalues.resize(nactive);
    masses. resize(nactive);
    return galaxymodel::fitSphericalDF(hvalues, masses, gridsize);
}

// main program begins here
int main(int argc, char* argv[])
{
    try{
    if(argc<=1)  // print help and exit
        printfail(usage);

    // parse command-line parameters
    utils::KeyValueMap args(argc-1, argv+1);
    std::string inputdensity   = args.getString("density");
    std::string inputpotential = args.getString("potential");
    std::string inputsnap      = args.getString("in");
    std::string outputsnap     = args.getString("out");
    std::string outputformat   = args.getString("format", "Text");
    std::string outputtab      = args.getString("tab");
    int nbody    = args.getInt("nbody");
    int gridsize = args.getInt("gridsizer", 50);
    double mbh   = args.getDouble("mbh", 0.);
    if(!(inputsnap.empty() ^ inputdensity.empty()))
        printfail("Must provide either density=... or in=... as input");
    if((!outputsnap.empty() && nbody==0) || (outputsnap.empty() && nbody>0))
        printfail("Must provide both output snapshot filename (out=...) and number of bodies (nbody=...)");
    if(outputsnap.empty() && outputtab.empty())
        printfail("Must provide output snapshot filename (out=...) and/or output table filename (tab=...)");

    potential::PtrDensity dens;          // the density profile
    potential::PtrPotential pot;         // the potential (may be different from the density)
    particles::ParticleArrayCar bodies;  // particles from the input N-body snapshot (if provided)

    // input is a name of a density profile or a file with the cumulative mass profile
    if(!inputdensity.empty()) {
        // the choice is made based on whether 'density=...' specifies an existing file name
        if(utils::fileExists(inputdensity))
            dens = galaxymodel::readMassProfile(inputdensity, &mbh);
        else
            dens = potential::createDensity(args);

        // check if a separate potential was also provided
        if(inputpotential.empty()) {
            pot = potential::Multipole::create(*dens, 0, 0, gridsize);
        } else {
            // the createPotential() routine reads the potential name from the 'type=...' parameter
            args.set("type", inputpotential);
            pot = potential::createPotential(args);
        }

        if(!isSpherical(*dens) || !isSpherical(*pot))
            printfail("Density and potential models must be spherical");
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
        pot  = potential::Multipole::create(bodies, coord::ST_SPHERICAL, 0, 0, gridsize);
        dens = pot;
    }

    // if a central massive black hole was provided, construct a two-component potential
    if(mbh>0) {
        std::vector<potential::PtrPotential> components(2);
        components[0] = potential::PtrPotential(new potential::Plummer(mbh, 0));
        components[1] = pot;
        pot = potential::PtrPotential(new potential::CompositeCyl(components));
    }

    // construct the mapping between energy and phase volume
    potential::PhaseVolume phasevol((potential::PotentialWrapper(*pot)));

    // compute the distribution function either from the density (using the Eddington inversion formula),
    // or from the input N-body snapshot (using the same approach as for density estimation,
    // but this time for the distribution in phase volume)
    math::LogLogSpline df = inputsnap.empty() ?
        galaxymodel::makeEddingtonDF(potential::DensityWrapper(*dens), potential::PotentialWrapper(*pot)) :
        fitSphericalDF(bodies, *pot, phasevol, gridsize);

    // now ripe the fruit: create an output table and/or
    // generate an N-body representation of the spherical model
    if(!outputtab.empty()) {
        galaxymodel::writeSphericalModel(outputtab,
            galaxymodel::SphericalModel(phasevol, df), *pot, dens.get());
    }

    if(!outputsnap.empty()) {
        // combine all command-line arguments to form the output snapshot header
        std::string header="mkspherical";
        char** argv0 = argv+1;
        for(char *arg = *argv0; (arg = *argv0); argv0 ++ ){
            header += ' ' + std::string(arg);
        }

        // generate the samples
        math::randomize();
        particles::ParticleArraySph bodies = galaxymodel::generatePosVelSamples(
            potential::PotentialWrapper(*pot), df, mbh? nbody-1 : nbody);
        if(mbh)  // add the central black hole as the 0th particle
            bodies.data.insert(bodies.data.begin(),
                particles::ParticleArraySph::ElemType(coord::PosVelSph(0,0,0,0,0,0), mbh));

        // write the snapshot
        particles::createIOSnapshotWrite(outputsnap, outputformat, units::ExternalUnits(), header)->
            writeSnapshot(bodies);
    }

    }
    catch (std::exception& ex) {
        printfail(ex.what());
    }
}
