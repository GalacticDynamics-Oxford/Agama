/** \file    example_self_consistent_model.cpp
    \author  Eugene Vasiliev
    \date    2015-2017

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a four-component galaxy with disk, bulge and halo components defined by their DFs,
    and a static density profile of gas disk.
    Then we perform several iterations of recomputing the density profiles of components from their DFs
    and recomputing the total potential.
    Finally, we create N-body representations of all mass components: dark matter halo,
    stars (bulge, thin and thick disks and stellar halo combined), and gas disk.

    An equivalent Python example is given in pytests/example_self_consistent_model.py
*/
#include "galaxymodel_base.h"
#include "galaxymodel_selfconsistent.h"
#include "galaxymodel_velocitysampler.h"
#include "df_factory.h"
#include "potential_composite.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "potential_utils.h"
#include "particles_io.h"
#include "math_core.h"
#include "math_spline.h"
#include "units.h"
#include "utils.h"
#include "utils_config.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

using potential::PtrDensity;
using potential::PtrPotential;

// define internal unit system - arbitrary numbers here! the result should not depend on their choice
const units::InternalUnits intUnits(2.7183*units::Kpc, 3.1416*units::Myr);

// define external unit system describing the data (including the parameters in INI file)
const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

// used for outputting the velocity distribution (the value is read from the ini file)
double solarRadius = NAN;

// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main()

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const std::string& filename, const std::vector<PtrPotential>& potentials)
{
    std::ofstream strm(filename.c_str());
    strm << "# radius[Kpc]\tv_circ,total[km/s]\tdisk\tbulge\thalo\n";
    // print values at certain radii, expressed in units of Kpc
    std::vector<double> radii = math::createExpGrid(81, 0.01, 100);
    for(unsigned int i=0; i<radii.size(); i++) {
        strm << radii[i];  // output radius in kpc
        double v2sum = 0;  // accumulate squared velocity in internal units
        double r_int = radii[i] * intUnits.from_Kpc;  // radius in internal units
        std::string str;
        for(unsigned int i=0; i<potentials.size(); i++) {
            double vc = v_circ(*potentials[i], r_int);
            v2sum += pow_2(vc);
            str += "\t" + utils::toString(vc * intUnits.to_kms);  // output in km/s
        }
        strm << '\t' << (sqrt(v2sum) * intUnits.to_kms) << str << '\n';
    }
}

/// print surface density profiles to a file
void writeSurfaceDensityProfile(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing surface density profile\n";
    std::vector<double> radii;
    // convert radii to internal units
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
        radii.push_back(r * intUnits.from_Kpc);
    int nr = radii.size();
    int nc = model.distrFunc.numValues();  // number of DF components
    std::vector<double> surfDens(nr*nc), rmsHeight(nr*nc), rmsVel(nr*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ir=0; ir<nr; ir++) {
        computeProjectedMoments(model, radii[ir], &surfDens[ir*nc], &rmsHeight[ir*nc], &rmsVel[ir*nc],
            NULL, NULL, NULL, /*separate*/ true);
    }

    std::ofstream strm(filename.c_str());
    strm << "# Radius[Kpc]\tsurfaceDensity[Msun/pc^2]\n";
    for(int ir=0; ir<nr; ir++) {
        strm << radii[ir] * intUnits.to_Kpc;
        for(int ic=0; ic<nc; ic++)
            strm << '\t' << surfDens[ir*nc+ic] * intUnits.to_Msun_per_pc2;
        strm << '\n';
    }
}

/// print vertical density profile for several sub-components of the stellar DF
void writeVerticalDensityProfile(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing vertical density profile\n";
    std::vector<double> heights;
    // convert height to internal units
    for(double h=0; h<=8; h<1.5 ? h+=0.125 : h+=0.5)
        heights.push_back(h * intUnits.from_Kpc);
    double R = solarRadius * intUnits.from_Kpc;
    int nh = heights.size();
    int nc = model.distrFunc.numValues();
    std::vector<double> dens(nh*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ih=0; ih<nh; ih++) {
        computeMoments(model, coord::PosCyl(R,heights[ih],0), &dens[ih*nc], NULL, NULL,
            NULL, NULL, NULL, /*separate*/ true);
    }

    std::ofstream strm(filename.c_str());
    strm << "# z[Kpc]\tThinDisk\tThickDisk\tStellarHalo[Msun/pc^3]\n";
    for(int ih=0; ih<nh; ih++) {
        strm << heights[ih] * intUnits.to_Kpc;
        for(int ic=0; ic<nc; ic++)
            strm << '\t' << dens[ih*nc+ic] * intUnits.to_Msun_per_pc3;
        strm << '\n';
    }
}

/// print velocity distributions at the given point to a file
void writeVelocityDistributions(const std::string& filename, const galaxymodel::GalaxyModel& model)
{
    const coord::PosCyl point(solarRadius * intUnits.from_Kpc, 0.1 * intUnits.from_Kpc, 0);
    std::cout << "Writing velocity distributions at "
        "(R=" << point.R * intUnits.to_Kpc << ", z=" << point.z * intUnits.to_Kpc << ")\n";
    // create grids in velocity space
    double v_max = 360 * intUnits.from_kms;
    std::vector<double> gridvR   = math::createUniformGrid(75, -v_max, v_max);
    std::vector<double> gridvz   = gridvR;  // for simplicity use the same grid for all three dimensions
    std::vector<double> gridvphi = gridvR;
    std::vector<double> amplvR, amplvz, amplvphi;
    double density;
    // compute the distributions
    const int ORDER = 3;
    math::BsplineInterpolator1d<ORDER> intvR(gridvR), intvz(gridvz), intvphi(gridvphi);
    galaxymodel::computeVelocityDistribution<ORDER>(model, point, false /*not projected*/,
        gridvR, gridvz, gridvphi, /*output*/ &density, &amplvR, &amplvz, &amplvphi);

    std::ofstream strm(filename.c_str());
    strm << "# V\tf(V_R)\tf(V_z)\tf(V_phi) [1/(km/s)]\n";
    for(int i=-100; i<=100; i++) {
        double v = i*v_max/100;
        // unit conversion: the VDF has a dimension 1/V, so that \int f(V) dV = 1;
        // therefore we need to multiply it by 1/velocityUnit
        strm << utils::toString(v * intUnits.to_kms)+'\t'+
            utils::toString(intvR.  interpolate(v, amplvR)   / intUnits.to_kms)+'\t'+
            utils::toString(intvz.  interpolate(v, amplvz)   / intUnits.to_kms)+'\t'+
            utils::toString(intvphi.interpolate(v, amplvphi) / intUnits.to_kms)+'\n';
    }
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model, const std::string& iteration)
{
    const potential::BaseDensity& compDisk = *model.components[0]->getDensity();
    const potential::BaseDensity& compBulge= *model.components[1]->getDensity();
    const potential::BaseDensity& compHalo = *model.components[2]->getDensity();
    coord::PosCyl pt0(solarRadius * intUnits.from_Kpc, 0, 0);
    coord::PosCyl pt1(solarRadius * intUnits.from_Kpc, 1 * intUnits.from_Kpc, 0);
    std::cout <<
        "Disk total mass="      << (compDisk.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compDisk.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compDisk.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Halo total mass="      << (compHalo.totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compHalo.density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compHalo.density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Potential at origin=-("<<
        (sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2"
        ", total mass=" << (model.totalPotential->totalMass() * intUnits.to_Msun) << " Msun\n";
    writeDensity("dens_disk_"+iteration, compDisk, extUnits);
    writeDensity("dens_halo_"+iteration, compHalo, extUnits);
    writePotential("potential_"+iteration, *model.totalPotential, extUnits);
    std::vector<PtrPotential> potentials(3);
    potentials[0] = dynamic_cast<const potential::CompositeCyl&>(*model.totalPotential).component(1);
    potentials[1] = potential::Multipole::create(compBulge, /*lmax*/6, /*mmax*/0, /*gridsize*/25);
    potentials[2] = potential::Multipole::create(compHalo,  /*lmax*/6, /*mmax*/0, /*gridsize*/25);
    writeRotationCurve("rotcurve_"+iteration, potentials);
}

/// perform one iteration of the model
void doIteration(galaxymodel::SelfConsistentModel& model, int iterationIndex)
{
    std::cout << "\033[1;37mStarting iteration #" << iterationIndex << "\033[0m\n";
    bool error=false;
    try {
        doIteration(model);
    }
    catch(std::exception& ex) {
        error=true;  // report the error and allow to save the results of the last iteration
        std::cout << "\033[1;31m==== Exception occurred: \033[0m\n" << ex.what();
    }
    printoutInfo(model, "iter"+utils::toString(iterationIndex));
    if(error)
        exit(1);  // abort in case of problems
}

int main()
{
    // read parameters from the INI file
    const std::string iniFileName = "../data/SCM.ini";
    utils::ConfigFile ini(iniFileName);
    utils::KeyValueMap
        iniPotenThinDisk = ini.findSection("Potential thin disk"),
        iniPotenThickDisk= ini.findSection("Potential thick disk"),
        iniPotenGasDisk  = ini.findSection("Potential gas disk"),
        iniPotenBulge    = ini.findSection("Potential bulge"),
        iniPotenDarkHalo = ini.findSection("Potential dark halo"),
        iniDFThinDisk    = ini.findSection("DF thin disk"),
        iniDFThickDisk   = ini.findSection("DF thick disk"),
        iniDFStellarHalo = ini.findSection("DF stellar halo"),
        iniDFBulge       = ini.findSection("DF bulge"),
        iniDFDarkHalo    = ini.findSection("DF dark halo"),
        iniSCMDisk       = ini.findSection("SelfConsistentModel disk"),
        iniSCMBulge      = ini.findSection("SelfConsistentModel bulge"),
        iniSCMHalo       = ini.findSection("SelfConsistentModel halo"),
        iniSCM           = ini.findSection("SelfConsistentModel");
    if(!iniSCM.contains("rminSph")) {  // most likely file doesn't exist
        std::cout << "Invalid INI file " << iniFileName << "\n";
        return -1;
    }
    solarRadius = ini.findSection("Data").getDouble("SolarRadius", solarRadius);

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
    model.useActionInterpolation = iniSCM.getBool("useActionInterpolation");

    // initialize density profiles of various components
    std::vector<PtrDensity> densityStellarDisk(2);
    PtrDensity densityBulge    = potential::createDensity(iniPotenBulge,    extUnits);
    PtrDensity densityDarkHalo = potential::createDensity(iniPotenDarkHalo, extUnits);
    densityStellarDisk[0]      = potential::createDensity(iniPotenThinDisk, extUnits);
    densityStellarDisk[1]      = potential::createDensity(iniPotenThickDisk,extUnits);
    PtrDensity densityGasDisk  = potential::createDensity(iniPotenGasDisk,  extUnits);

    // add components to SCM - at first, all of them are static density profiles
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(PtrDensity(
        new potential::CompositeDensity(densityStellarDisk)), true)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityBulge, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityDarkHalo, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityGasDisk, true)));

    // initialize total potential of the model (first guess)
    updateTotalPotential(model);
    printoutInfo(model, "init");

    std::cout << "\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: "
        "Mdisk="  << (model.components[0]->getDensity()->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mbulge=" << (densityBulge   ->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mhalo="  << (densityDarkHalo->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mgas="   << (densityGasDisk ->totalMass() * intUnits.to_Msun) << " Msun\n";

    // create the dark halo DF
    df::PtrDistributionFunction dfHalo = df::createDistributionFunction(
        iniDFDarkHalo, model.totalPotential.get(), /*density not needed*/NULL, extUnits);
    // same for the bulge
    df::PtrDistributionFunction dfBulge = df::createDistributionFunction(
        iniDFBulge, model.totalPotential.get(), NULL, extUnits);
    // same for the stellar components (thin/thick disks and stellar halo)
    std::vector<df::PtrDistributionFunction> dfStellarArray;
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThinDisk, model.totalPotential.get(), NULL, extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFThickDisk, model.totalPotential.get(), NULL, extUnits));
    dfStellarArray.push_back(df::createDistributionFunction(
        iniDFStellarHalo, model.totalPotential.get(), NULL, extUnits));
    // composite DF of all stellar components except the bulge
    df::PtrDistributionFunction dfStellar(new df::CompositeDF(dfStellarArray));

    // replace the static disk density component of SCM with a DF-based disk component
    model.components[0] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(dfStellar, PtrDensity(),
        iniSCMDisk.getInt("mmaxAngularCyl"),
        iniSCMDisk.getInt("sizeRadialCyl"),
        iniSCMDisk.getDouble("RminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("RmaxCyl") * extUnits.lengthUnit,
        iniSCMDisk.getInt("sizeVerticalCyl"),
        iniSCMDisk.getDouble("zminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("zmaxCyl") * extUnits.lengthUnit));
    // same for the bulge
    model.components[1] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfBulge, potential::PtrDensity(),
        iniSCMBulge.getInt("lmaxAngularSph"),
        iniSCMBulge.getInt("mmaxAngularSph"),
        iniSCMBulge.getInt("sizeRadialSph"),
        iniSCMBulge.getDouble("rminSph") * extUnits.lengthUnit,
        iniSCMBulge.getDouble("rmaxSph") * extUnits.lengthUnit));
    // same for the halo
    model.components[2] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfHalo, potential::PtrDensity(),
        iniSCMHalo.getInt("lmaxAngularSph"),
        iniSCMHalo.getInt("mmaxAngularSph"),
        iniSCMHalo.getInt("sizeRadialSph"),
        iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
        iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit));


    // we can compute the masses even though we don't know the density profile yet
    std::cout <<
        "Masses of DF components:"
        " Mdisk="       <<         (dfStellar->totalMass() * intUnits.to_Msun) <<
        " Msun (Mthin=" << (dfStellarArray[0]->totalMass() * intUnits.to_Msun) <<
        ", Mthick="     << (dfStellarArray[1]->totalMass() * intUnits.to_Msun) <<
        ", Mstel.halo=" << (dfStellarArray[2]->totalMass() * intUnits.to_Msun) <<
        "); Mbulge="    <<           (dfBulge->totalMass() * intUnits.to_Msun) << " Msun"
        "; Mdarkhalo="  <<           (dfHalo ->totalMass() * intUnits.to_Msun) << " Msun\n";


    // do a few more iterations to obtain the self-consistent density profile for both disks
    for(int iteration=1; iteration<=5; iteration++)
        doIteration(model, iteration);

    // output various profiles (only for stellar components)
    std::cout << "\033[1;33mComputing density profiles and velocity distribution\033[0m\n";
    galaxymodel::GalaxyModel modelStars(*model.totalPotential, *model.actionFinder, *dfStellar);
    writeSurfaceDensityProfile ("model_stars_final.surfdens", modelStars);
    writeVerticalDensityProfile("model_stars_final.vertical", modelStars);
    writeVelocityDistributions ("model_stars_final.veldist",  modelStars);

    // export model to an N-body snapshot
    std::cout << "\033[1;33mCreating an N-body representation of the model\033[0m\n";
    std::string format = "text";   // could use "text", "nemo" or "gadget" here

    // first create a representation of density profiles without velocities
    // (just for demonstration), by drawing samples from the density distribution
    std::cout << "Writing N-body sampled density profile for the dark matter halo\n";
    particles::writeSnapshot("dens_dm_final", galaxymodel::sampleDensity(
        *model.components[2]->getDensity(), 800000), format, extUnits);
    std::cout << "Writing N-body sampled density profile for the stellar bulge, disk and halo\n";
    std::vector<PtrDensity> densityStars(2);
    densityStars[0] = model.components[0]->getDensity();  // stellar disks and halo
    densityStars[1] = model.components[1]->getDensity();  // bulge
    particles::writeSnapshot("dens_stars_final", galaxymodel::sampleDensity(
        potential::CompositeDensity(densityStars), 200000), format, extUnits);

    // now create genuinely self-consistent models of both components,
    // by drawing positions and velocities from the DF in the given (self-consistent) potential
    std::cout << "Writing a complete DF-based N-body model for the dark matter halo\n";
    particles::writeSnapshot("model_dm_final", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfHalo), 800000),
        format, extUnits);
    std::cout << "Writing a complete DF-based N-body model for the stellar bulge, disk and halo\n";
    dfStellarArray.push_back(dfBulge);
    dfStellar.reset(new df::CompositeDF(dfStellarArray));  // all stellar components incl. bulge
    particles::writeSnapshot("model_stars_final", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar), 200000),
        format, extUnits);
    // we didn't use an action-based DF for the gas disk, leaving it as a static component;
    // to create an N-body representation, we sample the density profile and assign velocities
    // from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    std::cout << "Writing an N-body model for the gas disk\n";
    particles::writeSnapshot("model_gas_final", galaxymodel::assignVelocity(
        galaxymodel::sampleDensity(*model.components[3]->getDensity(), 24000),
        /*parameters for the axisymmetric Jeans velocity sampler*/
        *model.components[3]->getDensity(), *model.totalPotential, /*beta*/ 0., /*kappa*/ 1.),
        format, extUnits);
}
