/** \file    test_selfconsistentmodel.cpp
    \author  Eugene Vasiliev
    \date    November 2015

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a two-component galaxy with disk and halo components, using a two-stage approach:
    first, we take a static potential/density profile for the disk, and find a self-consistent
    density profile of the halo component in the presence of the disk potential;
    second, we replace the static disk with a DF-based component and find the overall self-consistent
    model for both components. The rationale is that a reasonable guess for the total potential
    is already needed before constructing the DF for the disk component, since the latter relies
    upon plausible radially-varying epicyclic frequencies.
    Both stages require a few iterations to converge.
    Finally, we create N-body representations of both components.

    An equivalent Python example is given in pytests/example_self_consistent_model.py
*/
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "df_factory.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "particles_io.h"
#include "math_core.h"
#include "math_spline.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>

using potential::PtrDensity;
using potential::PtrPotential;

// define internal unit system - arbitrary numbers here!
//const units::InternalUnits intUnits(6.666*units::Kpc, 42*units::Myr);
const units::InternalUnits intUnits(2.7183*units::Kpc, 3.1416*units::Myr);

// define external unit system describing the data (including the parameters in INI file)
const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main()

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const std::string& fileName, const PtrPotential& potential)
{
    writePotential(fileName, *potential, extUnits);
    PtrPotential comp = potential->name()==potential::CompositeCyl::myName() ? potential :
        PtrPotential(new potential::CompositeCyl(std::vector<PtrPotential>(1, potential)));
    const potential::CompositeCyl& pot = dynamic_cast<const potential::CompositeCyl&>(*comp);
    std::ofstream strm(fileName.c_str());
    strm << "#radius";
    for(unsigned int i=0; i<pot.size(); i++) {
        strm << "\t"<<pot.component(i)->name();
    }
    if(pot.size()>1)
        strm << "\ttotal\n";
    // print values at certain radii, expressed in units of Kpc
    std::vector<double> radii = math::createExpGrid(71, 0.0316227766016838, 100);
    for(unsigned int i=0; i<radii.size(); i++) {
        strm << radii[i];  // output radius in kpc
        double v2sum = 0;  // accumulate squared velocity in internal units
        double r_int = radii[i] * intUnits.from_Kpc;  // radius in internal units
        for(unsigned int i=0; i<pot.size(); i++) {
            coord::GradCyl deriv;  // potential derivatives in internal units
            pot.component(i)->eval(coord::PosCyl(r_int, 0, 0), NULL, &deriv);
            double v2comp = r_int*deriv.dR;
            v2sum += v2comp;
            strm << '\t' << (sqrt(v2comp) * intUnits.to_kms);  // output in km/s
        }
        if(pot.size()>1)
            strm << '\t' << (sqrt(v2sum) * intUnits.to_kms);
        strm << '\n';
    }
}

/// generate an N-body representation of a density profile (without velocities) and write to a file
void writeNbodyDensity(const std::string& fileName, const potential::BaseDensity& dens)
{
    std::cout << "Writing N-body sampled density profile to " << fileName << '\n';
    particles::ParticleArray<coord::PosCyl> points = galaxymodel::generateDensitySamples(dens, 1e5);
    // assign the units for exporting the N-body snapshot so that G=1 again,
    // and mass & velocity scale is reasonable
    units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 977.8*units::kms, 2.223e+11*units::Msun);
    writeSnapshot(fileName+".nemo", points, "Nemo", extUnits);
}

/// generate an N-body representation of the entire model specified by its DF, and write to a file
void writeNbodyModel(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing a complete DF-based N-body model to " << fileName << '\n';
    particles::ParticleArrayCyl points = galaxymodel::generatePosVelSamples(model, 1e5);
    units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 977.8*units::kms, 2.223e+11*units::Msun);
    writeSnapshot(fileName+".nemo", points, "Nemo", extUnits);
}

/// print profiles of surface density to a file
void writeSurfaceDensityProfile(const std::string& fileName, const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing surface density profile to " << fileName+".surfdens" << '\n';
    std::vector<double> radii;
    // convert radii to internal units
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
        radii.push_back(r * intUnits.from_Kpc);
    int nr = radii.size();
    std::vector<double> surfDens(nr);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ir=0; ir<nr; ir++) {
        double dummy;
        computeProjectedMoments(model, radii[ir], 1e-3, 1e6, surfDens[ir], dummy);
    }

    std::ofstream strm((fileName+".surfdens").c_str());
    strm << "#Radius\tsurfaceDensity\n";
    for(int ir=0; ir<nr; ir++)
        strm << (radii[ir] * intUnits.to_Kpc) << '\t' << 
        (surfDens[ir] * intUnits.to_Msun_per_pc2) << '\n';
}

/// print vertical density profile for several sub-components of the stellar DF
void writeVerticalDensityProfile(const std::string& fileName,
    const potential::BasePotential& pot,
    const actions::BaseActionFinder& af,
    const std::vector<df::PtrDistributionFunction>& DFcomponents,
    double RsolarInKpc)
{
    std::cout << "Writing vertical density profile to " << fileName+".vertical" << '\n';
    std::vector<double> heights;
    // convert height to internal units
    for(double h=0; h<=8; h<1.5 ? h+=0.125 : h+=0.5)
        heights.push_back(h * intUnits.from_Kpc);
    double R = RsolarInKpc * intUnits.from_Kpc;
    int nh = heights.size();
    int nc = DFcomponents.size();
    int n = nh*nc;
    std::vector<double> dens(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i=0; i<n; i++) {
        int ih = i/nc, ic = i%nc;
        computeMoments(galaxymodel::GalaxyModel(pot, af, *DFcomponents[ic]),
            coord::PosCyl(R,heights[ih],0), 1e-3, 1e5, &dens[ih*nc+ic],
            NULL, NULL, NULL, NULL, NULL);
    }

    std::ofstream strm((fileName+".vertical").c_str());
    strm << "#z";
    for(int ic=0; ic<nc; ic++)
        strm << "\tdensity_comp"<<ic;
    strm << '\n';
    for(int ih=0; ih<nh; ih++) {
        strm << (heights[ih] * intUnits.to_Kpc);
        for(int ic=0; ic<nc; ic++)
            strm << '\t' << (dens[ih*nc+ic] * intUnits.to_Msun_per_pc3);
        strm << '\n';
    }
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model, const std::string& iterationStr)
{
    const potential::BaseDensity& compHalo = *model.components[0]->getDensity();
    const potential::BaseDensity& compDisc = *model.components[1]->getDensity();
    coord::PosCyl pt0(8.3 * intUnits.from_Kpc, 0, 0);
    coord::PosCyl pt1(8.3 * intUnits.from_Kpc, 1 * intUnits.from_Kpc, 0);
    std::cout <<
        "Disc total mass="      << (compDisc.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compDisc.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compDisc.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Halo total mass="      << (compHalo.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compHalo.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compHalo.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3"
        ", inner density slope="<< getInnerDensitySlope(compHalo) << "\n"
        "Potential at origin=-("<<
        (sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2"
        ", total mass=" << (model.totalPotential->totalMass() * intUnits.to_Msun) << " Msun\n";
    writeDensity("dens_disc_iter"+iterationStr, compDisc, extUnits);
    writeDensity("dens_halo_iter"+iterationStr, compHalo, extUnits);
    writeRotationCurve("rotcurve_iter"+iterationStr, model.totalPotential);
}

int main()
{
    // read parameters from the INI file
    const std::string iniFileName = "../data/SCM.ini";
    utils::ConfigFile ini(iniFileName);
    utils::KeyValueMap
        iniPotenThinDisc = ini.findSection("Potential thin disc"),
        iniPotenThickDisc= ini.findSection("Potential thick disc"),
        iniPotenGasDisc  = ini.findSection("Potential gas disc"),
        iniPotenBulge    = ini.findSection("Potential bulge"),
        iniPotenDarkHalo = ini.findSection("Potential dark halo"),
        iniDFThinDisc    = ini.findSection("DF thin disc"),
        iniDFThickDisc   = ini.findSection("DF thick disc"),
        iniDFStellarHalo = ini.findSection("DF stellar halo"),
        iniDFDarkHalo    = ini.findSection("DF dark halo"),
        iniSCMHalo       = ini.findSection("SelfConsistentModel halo"),
        iniSCMDisc       = ini.findSection("SelfConsistentModel disc"),
        iniSCM           = ini.findSection("SelfConsistentModel");
    if(!iniSCM.contains("rminSph")) {  // most likely file doesn't exist
        std::cout << "Invalid INI file " << iniFileName << "\n";
        return -1;
    }

    // set up parameters of the entire Self-Consistent Model
    galaxymodel::SelfConsistentModel model;
    model.rminSph         = iniSCM.getDouble("rminSph") * extUnits.lengthUnit;
    model.rmaxSph         = iniSCM.getDouble("rmaxSph") * extUnits.lengthUnit;
    model.sizeRadialSph   = iniSCM.getInt("sizeRadialSph");
    model.lmaxAngularSph  = iniSCM.getInt("lmaxAngularSph");
    model.RminCyl         = iniSCM.getDouble("RminCyl") * extUnits.lengthUnit;
    model.RmaxCyl         = iniSCM.getDouble("RmaxCyl") * extUnits.lengthUnit;
    model.zminCyl         = iniSCM.getDouble("zminCyl") * extUnits.lengthUnit;
    model.zmaxCyl         = iniSCM.getDouble("zmaxCyl") * extUnits.lengthUnit;
    model.sizeRadialCyl   = iniSCM.getInt("sizeRadialCyl");
    model.sizeVerticalCyl = iniSCM.getInt("sizeVerticalCyl");

    // initialize density profiles of various components
    std::vector<PtrDensity> densityStellarDisc(2);
    PtrDensity densityBulge    = potential::createDensity(iniPotenBulge,    extUnits);
    PtrDensity densityDarkHalo = potential::createDensity(iniPotenDarkHalo, extUnits);
    densityStellarDisc[0]      = potential::createDensity(iniPotenThinDisc, extUnits);
    densityStellarDisc[1]      = potential::createDensity(iniPotenThickDisc,extUnits);
    PtrDensity densityGasDisc  = potential::createDensity(iniPotenGasDisc,  extUnits);

    // add components to SCM - at first, all of them are static density profiles
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityDarkHalo, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(PtrDensity(
        new potential::CompositeDensity(densityStellarDisc)), true)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityBulge, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityGasDisc, true)));

    // initialize total potential of the model (first guess)
    updateTotalPotential(model);
    writeRotationCurve("rotcurve_init", model.totalPotential);

    std::cout << "**** STARTING ONE-COMPONENT MODELLING ****\nMasses are:  "
        "Mbulge=" << (densityBulge->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mgas="   << (densityGasDisc->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mdisc="  << (model.components[1]->getDensity()->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mhalo="  << (densityDarkHalo->totalMass() * intUnits.to_Msun) << " Msun\n";

    // create the dark halo DF from the parameters in INI file;
    // here the initial potential is only used to create epicyclic frequency interpolation table
    df::PtrDistributionFunction dfHalo = df::createDistributionFunction(
        iniDFDarkHalo, model.totalPotential.get(), extUnits);

    // replace the halo SCM component with the DF-based one
    model.components[0] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfHalo, potential::PtrDensity(),
        iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
        iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit,
        iniSCMHalo.getInt("sizeRadialSph"),
        iniSCMHalo.getInt("lmaxAngularSph") ));

    // do a few iterations to determine the self-consistent density profile of the halo
    int iteration=0;
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << ++iteration << '\n';
        doIteration(model);
        printoutInfo(model, utils::toString(iteration));
    }

    // now that we have a reasonable guess for the total potential,
    // we may initialize the DF of the stellar components
    std::vector<df::PtrDistributionFunction> dfStellarArray;
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThinDisc, model.totalPotential.get(), extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThickDisc, model.totalPotential.get(), extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFStellarHalo, model.totalPotential.get(), extUnits));
    // composite DF of all stellar components except the bulge
    df::PtrDistributionFunction dfStellar(new df::CompositeDF(dfStellarArray));

    // we can compute the masses even though we don't know the density profile yet
    std::cout << "**** STARTING TWO-COMPONENT MODELLING ****\n"
        "Masses are: Mdisc=" <<    (dfStellar->totalMass() * intUnits.to_Msun) <<
        " Msun (Mthin=" << (dfStellarArray[0]->totalMass() * intUnits.to_Msun) <<
        ", Mthick="     << (dfStellarArray[1]->totalMass() * intUnits.to_Msun) <<
        ", Mstel.halo=" << (dfStellarArray[2]->totalMass() * intUnits.to_Msun) <<
        "); Mdarkhalo=" <<            (dfHalo->totalMass() * intUnits.to_Msun) << " Msun\n";

    // prepare parameters for the density grid of the stellar component
    std::vector<double> gridRadialCyl = math::createNonuniformGrid(
        iniSCMDisc.getInt("sizeRadialCyl"),
        iniSCMDisc.getDouble("RminCyl") * extUnits.lengthUnit,
        iniSCMDisc.getDouble("RmaxCyl") * extUnits.lengthUnit, true);
    std::vector<double> gridVerticalCyl = math::createNonuniformGrid(
        iniSCMDisc.getInt("sizeVerticalCyl"),
        iniSCMDisc.getDouble("zminCyl") * extUnits.lengthUnit,
        iniSCMDisc.getDouble("zmaxCyl") * extUnits.lengthUnit, true);

    // replace the static disc density component of SCM with a DF-based disc component
    model.components[1] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(
        dfStellar, PtrDensity(), gridRadialCyl, gridVerticalCyl));

    // do a few more iterations to obtain the self-consistent density profile for both discs
    for(int i=0; i<5; i++) {
        std::cout << "Starting iteration #" << ++iteration << '\n';
        doIteration(model);
        printoutInfo(model, utils::toString(iteration));
    }

    // export model to an N-body snapshot
    std::cout << "Creating an N-body representation of the model\n";
    std::string iterationStr(utils::toString(iteration));

    // first create a representation of density profiles without velocities
    // (just for demonstration), by drawing samples from the density distribution
    writeNbodyDensity("dens_halo_iter"+iterationStr,
        *model.components[0]->getDensity());
    writeNbodyDensity("dens_disc_iter"+iterationStr,
        *model.components[1]->getDensity());

    writeSurfaceDensityProfile("model_disc_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar));
    writeVerticalDensityProfile("model_disc_iter_R8"+iterationStr,
        *model.totalPotential, *model.actionFinder, dfStellarArray, 8.3 /*kpc*/);
    writeVerticalDensityProfile("model_disc_iter_R0"+iterationStr,
        *model.totalPotential, *model.actionFinder, dfStellarArray, 0.01 /*kpc*/);

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    writeNbodyModel("model_halo_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfHalo));
    writeNbodyModel("model_disc_iter"+iterationStr,
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar));

    return 0;
}
