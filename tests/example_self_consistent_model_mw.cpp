/*  \file    example_self_consistent_model_mw.cpp
    \author  James Binney, Eugene Vasiliev
    \date    2015-2022

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a Milky Way model with four disks, bulge, stellar and dark halo components
    defined by their DFs, and a static density profile of gas disk.
    The thin disk is split into 3 age groups, and there is a separate thick disk.
    Then we perform several iterations of recomputing the density profiles of components from
    their DFs and recomputing the total potential.
    Finally, we create N-body representations of all mass components:
    dark matter halo, stars (bulge, several disks and stellar halo combined), and gas disk,
    and compute various diagnostic quantities written into text files.
    The DFs for the disky and spheroidal components used here differ from the built-in DF types, and
    are defined in the first part of the file; their parameters are contained in a separate INI file.
    The DF parameters are optimized to fit Gaia DR2 data, as described in Binney&Vasiliev 2023.
    An equivalent Python program example_self_consistent_model_mw.py introduces the same user-defined
    DFs implemented as Python functions, and for this reason is less computationally efficient.
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
#include "actions_staeckel.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

namespace df{

// user-defined modifications of spheroidal (DoublePowerLaw) and disky (Exponential) DFs

struct NewDoublePowerLawParam{
    double
        norm,      ///< normalization factor with the dimension of mass
        J0,        ///< break action (defines the transition between inner and outer regions)
        Jcutoff,   ///< cutoff action (sets exponential suppression at J>Jcutoff, 0 to disable)
        Jphi0,     ///< controls the steepness of rotation and the size of non-rotating core
        Jcore,     ///< central core size for a Cole&Binney-type modified double-power-law halo
        L0,        ///< helps define E surrogate jt
        slopeIn,   ///< power-law index for actions below the break action (Gamma)
        slopeOut,  ///< power-law index for actions above the break action (Beta)
        cutoffStrength, ///< steepness of exponential suppression at J>Jcutoff (zeta)
        alpha,     ///< helps define xi which goes 0 -> 1 inside -> out
        beta,      ///< induces radial bias via sin
        Fin,       ///< sets coeffs a,b that determin cL
        Fout,      ///< reduces cost of low incl orbits
        rotFrac;   ///< relative amplitude of the odd-Jphi component (-1 to 1, 0 means no rotation)
    NewDoublePowerLawParam() :  ///< set default values for all fields (NAN means that it must be set manually)
        norm(NAN), J0(NAN), Jcutoff(0), Jphi0(0), Jcore(0), L0(0),
        slopeIn(NAN), slopeOut(NAN), cutoffStrength(2),
        alpha(0.6), beta(NAN), Fin(NAN), Fout(NAN), rotFrac(0) {}
};

class NewDoublePowerLaw: public BaseDistributionFunction{
    const NewDoublePowerLawParam par;  ///< parameters of DF
    const double zeta;              ///< auxiliary coefficient for the case of a central core
public:
    /** Create an instance of double-power-law distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    NewDoublePowerLaw(const NewDoublePowerLawParam &params);

    /** return value of DF for the given set of actions */
    virtual void evalDeriv(const actions::Actions &J, double *f,
        df::DerivByActions *deriv=NULL) const;
};

/// helper class used in the root-finder to determine the auxiliary coefficient zeta for a cored halo
class NewDoublePowerLawZetaFinder: public math::IFunctionNoDeriv{
    const NewDoublePowerLawParam& par;

    // return the difference between the non-modified and modified DF as a function of zeta and
    // the appropriately scaled action variable (t -> hJ), weighted by d(hJ)/dt for the integration in t
    double deltaf(const double t, const double zeta) const
    {
        // integration is performed in a scaled variable t, ranging from 0 to 1,
        // which is remapped to hJ ranging from 0 to infinity as follows:
        double hJ    = par.Jcore * t*t*(3-2*t) / pow_2(1-t) / (1+2*t); 
        double dhJdt = par.Jcore * 6*t / pow_3(1-t) / pow_2(1+2*t);
        return hJ * hJ * dhJdt *
                math::pow(1 + par.J0 / hJ,  par.slopeIn) *
                math::pow(1 + hJ / par.J0, -par.slopeOut) *
                (math::pow(1 + par.Jcore/hJ * (par.Jcore/hJ - zeta), -0.5*par.slopeIn) - 1);
    }

    public:
        NewDoublePowerLawZetaFinder(const NewDoublePowerLawParam& _par) : par(_par) {}

        virtual double value(const double zeta) const
        {
            double result = 0;
            // use a fixed-order GL quadrature to compute the integrated difference in normalization between
            // unmodified and core-modified DF, which is sought to be zero by choosing an appropriate beta
            static const int GLORDER = 20;  // should be even, to avoid singularity in the integrand at t=0.5
            for(int i=0; i<GLORDER; i++)
                result += math::GLWEIGHTS[GLORDER][i] * deltaf(math::GLPOINTS[GLORDER][i], zeta);
            return result;
        }
};

NewDoublePowerLaw::NewDoublePowerLaw(const NewDoublePowerLawParam &inparams) :
    par(inparams),
    zeta(math::findRoot(NewDoublePowerLawZetaFinder(par), 0.0, 2.0, /*root-finder tolerance*/ SQRT_DBL_EPSILON))
{
    // sanity checks on parameters
    if(!(par.norm>0))
        throw std::invalid_argument("NewDoublePowerLaw: normalization must be positive");
    if(!(par.J0>0))
        throw std::invalid_argument("NewDoublePowerLaw: break action J0 must be positive");
    if(!(par.Jcore>=0 && zeta>=0))
        throw std::invalid_argument("NewDoublePowerLaw: core action Jcore is invalid");
    if(!(par.Jcutoff>=0))
        throw std::invalid_argument("NewDoublePowerLaw: truncation action Jcutoff must be non-negative");
    if(!(par.slopeOut>3) && par.Jcutoff==0)
        throw std::invalid_argument("NewDoublePowerLaw: mass diverges at large J (outer slope must be > 3)");
    if(!(par.slopeIn<3))
        throw std::invalid_argument("NewDoublePowerLaw: mass diverges at J->0 (inner slope must be < 3)");
    if(!(par.cutoffStrength>0))
        throw std::invalid_argument("NewDoublePowerLaw: cutoff strength parameter must be positive");
    if(!(fabs(par.rotFrac)<=1))
        throw std::invalid_argument("NewDoublePowerLaw: amplitude of odd-Jphi component must be between -1 and 1");
}

void NewDoublePowerLaw::evalDeriv(const actions::Actions &J, double *val, df::DerivByActions*) const
{
    double modJphi=fabs(J.Jphi);
    double L=J.Jz+modJphi;
    double c=L/(L+J.Jr);
    double jt=(1.5*J.Jr+L)/par.L0;
    double jta=pow(jt,par.alpha), xi=jta/(1+jta);
    double rat=(1-xi)*par.Fin+xi*par.Fout;
    double a=.5*(rat+1), b=.5*(rat-1);
    double cL= L>0? J.Jz*(a+b*modJphi/L)+modJphi: 0;
    double fac=exp(par.beta*sin(0.5*M_PI*c));
    double hJ=J.Jr/fac + .5*(1+c*xi)*fac*cL;
    double gJ=hJ;
    *val = par.norm / pow_3(2*M_PI * par.J0) *
        math::pow(1 + par.J0 / hJ,  par.slopeIn) *
        math::pow(1 + gJ / par.J0, -par.slopeOut);
    if(par.Jcutoff>0){   // exponential cutoff at large J
        *val *= exp(-math::pow(gJ / par.Jcutoff, par.cutoffStrength));
    }
    if(par.Jcore>0) {   // central core of nearly-constant f(J) at small J
        if(hJ==0)
            *val = par.norm / pow_3(2*M_PI * par.J0);
        else
            *val *= math::pow(1 + par.Jcore/hJ * (par.Jcore/hJ - zeta), -0.5*par.slopeIn);
    }
    if(par.rotFrac!=0)  // add the odd part
        *val *= 1 + par.rotFrac * tanh(J.Jphi / par.Jphi0);
}

struct NewExponentialParam{
    double
        norm,       ///< overall normalization factor with the dimension of mass (NOT the actual mass)
        Jr0,        ///< scale action setting the radial velocity dispersion
        Jz0,        ///< scale action setting the disk thickness and the vertical velocity dispersion
        Jphi0,      ///< scale action setting the disk radius
        pr,        ///< power of radial variation of sigR
        pz,        ///< power of radial variation of sigz
        addJden,    ///< additional contribution to the sum of actions that affects the density profile
        addJvel;    ///< same for the part that affects the velocity dispersion profiles
    NewExponentialParam() :  ///< set default values for all fields
        norm(NAN), Jr0(NAN), Jz0(NAN), Jphi0(NAN), pr(0.5), pz(0.5), addJden(0), addJvel(0) {}
};

class NewExponential: public df::BaseDistributionFunction{
    const NewExponentialParam par;     ///< parameters of the DF
public:
    NewExponential(const NewExponentialParam& params);
    virtual void evalDeriv(const actions::Actions &J, double *f,
        df::DerivByActions *deriv=NULL) const;
};

NewExponential::NewExponential(const NewExponentialParam& params) :
    par(params)
{
    if(!(par.norm>0))
        throw std::invalid_argument("NewExponential: norm must be positive");
    if(!(par.Jr0>0) || !(par.Jz0>0) || !(par.Jphi0>0))
        throw std::invalid_argument("NewExponential: scale actions must be positive");
    if(par.addJden<0 || par.addJden >= par.Jphi0)
        throw std::invalid_argument("NewExponential: addJden must be in (0, Jphi0)");
}

void NewExponential::evalDeriv(const actions::Actions &J, double *val, df::DerivByActions*) const
{
    double Jp = J.Jphi<=0 ? 0 : J.Jphi;
    if(Jp==0) {
        *val = 0;
        return;
    }
    double Jvel = Jp + par.addJvel;
    double Jden = Jp + par.addJden;
    double xr = pow(Jvel/par.Jphi0,par.pr)/par.Jr0;
    double xz = pow(Jvel/par.Jphi0,par.pz)/par.Jz0;
    double fr = xr * exp(-xr*J.Jr), fz = xz * exp(-xz*J.Jz);
    double xp = Jden / par.Jphi0;
    double fp = par.norm/par.Jphi0 * fabs(J.Jphi) / par.Jphi0 * exp(-xp);
    *val = fr * fz * fp;
}

PtrDistributionFunction createNewDoublePowerLawDF(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    if(!utils::stringsEqual(kvmap.getString("type"), "NewDoublePowerLaw"))
        throw std::runtime_error("invalid DF type");
    NewDoublePowerLawParam par;
    par.norm      = kvmap.getDouble("norm",      par.norm)    * conv.massUnit;
    par.J0        = kvmap.getDouble("J0",        par.J0)      * conv.lengthUnit * conv.velocityUnit;
    par.Jcutoff   = kvmap.getDouble("Jcutoff",   par.Jcutoff) * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0     = kvmap.getDouble("Jphi0",     par.Jphi0)   * conv.lengthUnit * conv.velocityUnit;
    par.Jcore     = kvmap.getDouble("Jcore",     par.Jcore)   * conv.lengthUnit * conv.velocityUnit;
    par.L0        = kvmap.getDouble("L0",        par.L0)      * conv.lengthUnit * conv.velocityUnit;
    par.slopeIn   = kvmap.getDouble("slopeIn",   par.slopeIn);
    par.slopeOut  = kvmap.getDouble("slopeOut",  par.slopeOut);
    par.cutoffStrength = kvmap.getDouble("cutoffStrength", par.cutoffStrength);
    par.alpha     = kvmap.getDouble("alpha",     par.alpha);
    par.beta      = kvmap.getDouble("beta",      par.beta);
    par.Fin       = kvmap.getDouble("Fin",       par.Fin);
    par.Fout      = kvmap.getDouble("Fout",      par.Fout);
    par.rotFrac   = kvmap.getDouble("rotFrac",   par.rotFrac);
    double mass = kvmap.getDouble("mass", NAN)* conv.massUnit;
    if(mass>0) {
        par.norm = 1.0;
        par.norm = mass / NewDoublePowerLaw(par).totalMass();
    }
    return PtrDistributionFunction(new NewDoublePowerLaw(par));
}

PtrDistributionFunction createNewExponentialDF(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    if(!utils::stringsEqual(kvmap.getString("type"), "NewExponential"))
        throw std::runtime_error("invalid DF type");
    NewExponentialParam par;
    par.norm   = kvmap.getDouble("norm",   par.norm)   * conv.massUnit;
    par.Jr0    = kvmap.getDouble("Jr0")    * conv.lengthUnit * conv.velocityUnit;
    par.Jz0    = kvmap.getDouble("Jz0")    * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0  = kvmap.getDouble("Jphi0")  * conv.lengthUnit * conv.velocityUnit;
    par.pr     = kvmap.getDouble("pr");
    par.pz     = kvmap.getDouble("pz");
    par.addJden= kvmap.getDouble("addJden")* conv.lengthUnit * conv.velocityUnit;
    par.addJvel= kvmap.getDouble("addJvel")* conv.lengthUnit * conv.velocityUnit;
    double mass = kvmap.getDouble("mass", NAN)* conv.massUnit;
    if(mass>0) {
        par.norm = 1.0;
        par.norm = mass / NewExponential(par).totalMass();
    }
    return PtrDistributionFunction(new NewExponential(par));
}

}  // namespace df


// define internal unit system - arbitrary numbers here! the result should not depend on their choice
const units::InternalUnits intUnits(2.7183*units::Kpc, 3.1416*units::Myr);

// define external unit system describing the data (including the parameters in INI file)
const units::ExternalUnits extUnits(intUnits, 1.*units::Kpc, 1.*units::kms, 1.*units::Msun);

// used for outputting the velocity distribution (the value is read from the ini file)
double solarRadius = NAN;

// header line for diagnostic outputs split into individual stellar components
const std::string componentNames = "bulge\tthin,young\tthin,middle\tthin,old\tthick\tstellarhalo";


// various auxiliary functions for printing out information are non-essential
// for the modelling itself; the essential workflow is contained in main()

/// print the rotation curve for a collection of potential components into a text file
void writeRotationCurve(const potential::Composite& potential)
{
    std::ofstream strm("mwmodel_rotcurve.txt");
    strm << "# radius[Kpc]\tv_circ,total[km/s]\tdarkmatter\tstars+gas\n";
    // print values at certain radii, expressed in units of Kpc
    std::vector<double> radii = math::createExpGrid(81, 0.01, 100);
    for(unsigned int i=0; i<radii.size(); i++) {
        strm << radii[i];  // output radius in kpc
        double v2sum = 0;  // accumulate squared velocity in internal units
        double r_int = radii[i] * intUnits.from_Kpc;  // radius in internal units
        std::string str;
        for(unsigned int c=0; c<potential.size(); c++) {
            double vc = v_circ(*potential.component(c), r_int);
            if(vc>0) v2sum += pow_2(vc); else v2sum -= pow_2(vc);
            str += "\t" + utils::toString(vc * intUnits.to_kms);  // output in km/s
        }
        double x = v2sum>=0? sqrt(v2sum) : -sqrt(fabs(v2sum));
        strm << '\t' << (x * intUnits.to_kms) << str << '\n';
    }
}

/// print velocity dispersion, in-plane and surface density profiles of each stellar component
void writeRadialProfile(const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing radial density and velocity profiles\n";
    std::vector<double> radii;
    // convert radii to internal units
    for(double r=1./8; r<=30; r<1 ? r*=2 : r<16 ? r+=0.5 : r+=2)
        radii.push_back(r * intUnits.from_Kpc);
    int nr = radii.size();
    int nc = model.distrFunc.numValues();  // number of DF components
    std::vector<double> surfDens(nr*nc), z0dens(nr*nc);
    std::vector<coord::VelCar> meanv(nr*nc);
    std::vector<coord::Vel2Car> meanv2(nr*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ir=0; ir<nr; ir++) {
        // in-plane density and velocity moments
        computeMoments(model, coord::PosCar(radii[ir], 0, 0),
            &z0dens[ir*nc], &meanv[ir*nc], &meanv2[ir*nc], /*separate*/ true);
        // projected density
        computeMoments(model, coord::PosProj(radii[ir],0),
            &surfDens[ir*nc], NULL, NULL, /*separate*/ true);
    }

    std::ofstream
    strmsd("mwmodel_surface_density.txt"),
    strmvd("mwmodel_volume_density.txt"),
    strmsR("mwmodel_sigmaR.txt"),
    strmsz("mwmodel_sigmaz.txt"),
    strmsp("mwmodel_sigmaphi.txt"),
    strmvp("mwmodel_meanvphi.txt");
    strmsd << "# Radius[Kpc]\t" << componentNames << "[Msun/pc^2]\n";
    strmvd << "# Radius[Kpc]\t" << componentNames << "[Msun/pc^3]\n";
    strmsR << "# Radius[Kpc]\t" << componentNames << "[km/s]\n";
    strmsz << "# Radius[Kpc]\t" << componentNames << "[km/s]\n";
    strmsp << "# Radius[Kpc]\t" << componentNames << "[km/s]\n";
    strmvp << "# Radius[Kpc]\t" << componentNames << "[km/s]\n";
    for(int ir=0; ir<nr; ir++){
        strmsd << radii[ir] * intUnits.to_Kpc;
        strmvd << radii[ir] * intUnits.to_Kpc;
        strmsR << radii[ir] * intUnits.to_Kpc;
        strmsz << radii[ir] * intUnits.to_Kpc;
        strmsp << radii[ir] * intUnits.to_Kpc;
        strmvp << radii[ir] * intUnits.to_Kpc;
        for(int ic=0; ic<nc; ic++) {
            strmsd << '\t' <<    surfDens[ir*nc+ic]      * intUnits.to_Msun_per_pc2;
            strmvd << '\t' <<      z0dens[ir*nc+ic]      * intUnits.to_Msun_per_pc3;
            strmsR << '\t' << sqrt(meanv2[ir*nc+ic].vx2) * intUnits.to_kms;
            strmsz << '\t' << sqrt(meanv2[ir*nc+ic].vz2) * intUnits.to_kms;
            strmsp << '\t' << sqrt(meanv2[ir*nc+ic].vy2 - pow_2(meanv[ir*nc+ic].vy)) * intUnits.to_kms;
            strmvp << '\t' <<      meanv [ir*nc+ic].vy   * intUnits.to_kms;
        }
        strmsd << '\n';
        strmvd << '\n';
        strmsR << '\n';
        strmsz << '\n';
        strmsp << '\n';
        strmvp << '\n';
    }
}

/// print vertical density profile for several sub-components of the stellar DF
void writeVerticalDensityProfile(const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing vertical density profile\n";
    std::vector<double> heights;
    // convert height to internal units
    for(double h=0; h<=5.01; h<0.49 ? h+=0.05 : h+=0.25)
        heights.push_back(h * intUnits.from_Kpc);
    int nh = heights.size();
    int nc = model.distrFunc.numValues();
    std::vector<double> dens(nh*nc);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ih=0; ih<nh; ih++) {
        computeMoments(model, coord::PosCar(solarRadius * intUnits.from_Kpc, 0, heights[ih]),
            &dens[ih*nc], NULL, NULL, /*separate*/ true);
    }

    std::ofstream strm("mwmodel_vertical_density.txt");
    strm << "# z[Kpc]\t" << componentNames << "[Msun/pc^3]\n";
    for(int ih=0; ih<nh; ih++) {
        strm << (heights[ih] * intUnits.to_Kpc);
        for(int ic=0; ic<nc; ic++)
            strm << '\t' << (dens[ih*nc+ic] * intUnits.to_Msun_per_pc3);
        strm << '\n';
    }
}

/// compute velocity distributions of each stellar component at several points and write them to files
void writeVelocityDistributions(const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing velocity distributions\n";
    const int numPoints = 4;
    const double R[numPoints] = {solarRadius-2.0, solarRadius, solarRadius+2.0, solarRadius};
    const double z[numPoints] = {0, 0, 0, 2.0};
    // create grid in velocity space for representing the spline-interpolated VDFs
    std::vector<double> gridv_spl =
        math::createSymmetricGrid(75, 6.0 * intUnits.from_kms, 400.0 * intUnits.from_kms);
    // store each component (thin/thick/bulge/etc) separately
    const int numComp = model.distrFunc.numValues();
    std::vector< std::vector<double> > dens(numPoints);
    // indexing scheme: [directionIndex, pointIndex, dfComponentIndex, amplIndex],
    // where directionIndex is {vx, vy, vz}, and amplIndex enumerates B-spline amplitudes
    std::vector< std::vector< std::vector<double> > > ampl[3];
    for(int d=0; d<3; d++)
        ampl[d].resize(numPoints, std::vector< std::vector<double> >(numComp));
    const int ORDER = 3;
    math::BsplineInterpolator1d<ORDER> interp(gridv_spl);

    // loop over all points in parallel
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int p=0; p<numPoints; p++) {
        dens[p].resize(numComp);
        galaxymodel::computeVelocityDistribution<ORDER>(model,
            coord::PosCar(R[p] * intUnits.from_Kpc, 0, z[p] * intUnits.from_Kpc),
            gridv_spl, gridv_spl, gridv_spl,
            /*density*/ &dens[p].front(), &ampl[0][p].front(), &ampl[1][p].front(), &ampl[2][p].front(),
            /*separate*/ true);
    }

    // now write out the results sequentially
    std::ofstream strm;
    // unit conversion: here the VDF is normalized so that the integral \int f(v) dv = rho,
    // the density of stars in each component at the given point;
    // therefore we need to multiply it by massUnit/lengthUnit^3/velocityUnit
    double conv = intUnits.to_Msun_per_pc3 / intUnits.to_kms;
    for(int p=0; p<numPoints; p++) {
        for(int d=0; d<3; d++) {
            strm.open(("mwmodel_vdf_R" + utils::toString(R[p]) + "_z" + utils::toString(z[p]) + "_v" +
            (d==0 ? "R" : d==1 ? "phi" : "z") + ".txt").c_str());
            strm << "# v[km/s]\t" << componentNames << "[Msun/kpc^3/(km/s)]\n";
            for(double v=-400; v<=400; v+=2) {
                strm << v;
                for(int c=0; c<numComp; c++)
                    strm << '\t' << conv * dens[p][c] *
                        interp.interpolate(v * intUnits.from_kms, ampl[d][p][c]);
                strm << '\n';
            }
            strm.close();
        }
    }
}

/// report progress after an iteration
void printoutInfo(const galaxymodel::SelfConsistentModel& model)
{
    potential::PtrDensity compStars = model.components[0]->getDensity();
    potential::PtrDensity compDark  = model.components[1]->getDensity();
    coord::PosCyl pt0(solarRadius * intUnits.from_Kpc, 0, 0);
    coord::PosCyl pt1(solarRadius * intUnits.from_Kpc, 1 * intUnits.from_Kpc, 0);
    std::cout <<
        "Disk total mass="      << (compStars->totalMass()  * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compStars->density(pt0) * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compStars->density(pt1) * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Halo total mass="      << (compDark->totalMass()   * intUnits.to_Msun) << " Msun"
        ", rho(Rsolar,z=0)="    << (compDark->density(pt0)  * intUnits.to_Msun_per_pc3) <<
        ", rho(Rsolar,z=1kpc)=" << (compDark->density(pt1)  * intUnits.to_Msun_per_pc3) << " Msun/pc^3\n"
        "Potential at origin=-("<<
        (sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2"
        ", total mass=" << (model.totalPotential->totalMass() * intUnits.to_Msun) << " Msun\n";
    writeDensity("mwmodel_density_stars.ini", *compStars, extUnits);
    writeDensity("mwmodel_density_dark.ini",  *compDark,  extUnits);
    writePotential("mwmodel_potential.ini", *model.totalPotential, extUnits);
    writeRotationCurve(dynamic_cast<const potential::Composite&>(*model.totalPotential));
}

int main()
{
    int numIterations = 4;
    // read parameters from the INI file
    const std::string iniFileName = "../data/SCM_MW.ini";
    utils::ConfigFile ini(iniFileName);
    utils::KeyValueMap
    iniPotenGasDisk  = ini.findSection("Potential gas disk"),
    iniDFyoungDisk   = ini.findSection("DF young disk"),
    iniDFmiddleDisk  = ini.findSection("DF middle disk"),
    iniDFoldDisk     = ini.findSection("DF old disk"),
    iniDFhighADisk   = ini.findSection("DF highA disk"),
    iniDFStellarHalo = ini.findSection("DF stellar halo"),
    iniDFBulge       = ini.findSection("DF bulge"),
    iniDFDarkHalo    = ini.findSection("DF dark halo"),
    iniSCMDisk       = ini.findSection("SelfConsistentModel disk"),
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

    // create the initial potential from all sections of the INI file starting with "[Potential..."
    model.totalPotential = potential::readPotential(iniFileName, extUnits);

    // create the dark halo DF
    df::PtrDistributionFunction dfHalo = df::createNewDoublePowerLawDF(iniDFDarkHalo, extUnits);
    // same for the stellar components (bulge, four disks, and stellar halo)
    std::vector<df::PtrDistributionFunction> dfStellarArray;
    dfStellarArray.push_back(df::createNewDoublePowerLawDF(iniDFBulge,      extUnits));
    dfStellarArray.push_back(df::createNewExponentialDF   (iniDFyoungDisk,  extUnits));
    dfStellarArray.push_back(df::createNewExponentialDF   (iniDFmiddleDisk, extUnits));
    dfStellarArray.push_back(df::createNewExponentialDF   (iniDFoldDisk,    extUnits));
    dfStellarArray.push_back(df::createNewExponentialDF   (iniDFhighADisk,  extUnits));
    dfStellarArray.push_back(df::createNewDoublePowerLawDF(iniDFStellarHalo,extUnits));
    // composite DF of all stellar components
    df::PtrDistributionFunction dfStellar(new df::CompositeDF(dfStellarArray));
    // replace the static disk density component of SCM with a DF-based disk component
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(dfStellar, potential::PtrDensity(),
        iniSCMDisk.getInt("mmaxAngularCyl"),
        iniSCMDisk.getInt("sizeRadialCyl"),
        iniSCMDisk.getDouble("RminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("RmaxCyl") * extUnits.lengthUnit,
        iniSCMDisk.getInt("sizeVerticalCyl"),
        iniSCMDisk.getDouble("zminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("zmaxCyl") * extUnits.lengthUnit)));
    // same for the dark halo
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfHalo, potential::PtrDensity(),
        iniSCMHalo.getInt("lmaxAngularSph"),
        iniSCMHalo.getInt("mmaxAngularSph"),
        iniSCMHalo.getInt("sizeRadialSph"),
        iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
        iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit)));
    // gas component is a fixed density profile
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(potential::createDensity(iniPotenGasDisk, extUnits), true)));
    
    utils::Timer timer1;
    // do a few iterations to obtain the self-consistent density profile for both disks
    for(int iteration=1; iteration<=numIterations; iteration++) {
        std::cout << "\033[1;37mStarting iteration #" << iteration << "\033[0m\n";
        doIteration(model);
        printoutInfo(model);
    }
    std::cout << utils::toString(timer1.deltaSeconds(), 3) + " seconds to build the model\n";

    // output various profiles
    std::cout << "\033[1;37mComputing diagnostics\033[0m\n";
    utils::Timer timer2;
    galaxymodel::GalaxyModel modelStars(*model.totalPotential, *model.actionFinder, *dfStellar);
    writeRadialProfile(modelStars);
    writeVerticalDensityProfile(modelStars);
    writeVelocityDistributions(modelStars);
    std::cout << utils::toString(timer2.deltaSeconds(), 3) + " seconds to compute diagnostics\n";

    // export model to an N-body snapshot
    std::cout << "\033[1;37mCreating an N-body representation of the model\033[0m\n";
    utils::Timer timer3;
    std::string format = "nemo";   // could use "text", "nemo" or "gadget" here
    particles::writeSnapshot("mwmodel_dm_final.nbody", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfHalo), 750000),
        format, extUnits);
    particles::writeSnapshot("mwmodel_stars_final.nbody", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar), 200000),
        format, extUnits);
    // we didn't use an action-based DF for the gas disk, leaving it as a static component;
    // to create an N-body representation, we sample the density profile and assign velocities
    // from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    particles::writeSnapshot("mwmodel_gas_final.nbody", galaxymodel::assignVelocity(
        galaxymodel::sampleDensity(*model.components[2]->getDensity(), 50000),
        /*parameters for the axisymmetric Jeans velocity sampler*/
        *model.components[2]->getDensity(), *model.totalPotential, /*beta*/ 0., /*kappa*/ 1.),
        format, extUnits);
    std::cout << utils::toString(timer3.deltaSeconds(), 3) + " seconds to create an N-body snapshot\n";
}
