/*  \file    example_self_consistent_model_mw.cpp
    \author  James Binney, Eugene Vasiliev
    \date    2015-2022

    This example demonstrates the machinery for constructing multicomponent self-consistent models
    specified by distribution functions in terms of actions.
    We create a multicomponent galaxy with four disks, bulge stellar and dark halo components
    defined by their DFs, and a static density profile of gas disk. The thin disk is split by age
    into 3 groups, and there is a separate thick disk.
    Then we perform several iterations of recomputing the density profiles of components from
    their DFs and recomputing the total potential.
    Finally, we create N-body representations of all mass components:
    dark matter halo, stars (bulge, thin and thick disks and stellar halo combined), and gas disk,
    and compute various diagnostic quantities written into text files.
    The DFs for the disky and spheroidal components used here differ from the built-in DF types, and
    are defined in the first part of the file; their parameters are contained in a separate INI file.
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
#include <ctime>

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
    const double beta;              ///< auxiliary coefficient for the case of a central core
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

/// helper class used in the root-finder to determine the auxiliary coefficient beta for a cored halo
class NewBetaFinder: public math::IFunctionNoDeriv{
    const NewDoublePowerLawParam& par;

    // return the difference between the non-modified and modified DF as a function of beta and
    // the appropriately scaled action variable (t -> hJ), weighted by d(hJ)/dt for the integration in t
    double deltaf(const double t, const double beta) const
    {
        // integration is performed in a scaled variable t, ranging from 0 to 1,
        // which is remapped to hJ ranging from 0 to infinity as follows:
        double hJ    = par.Jcore * t*t*(3-2*t) / pow_2(1-t) / (1+2*t); 
        double dhJdt = par.Jcore * 6*t / pow_3(1-t) / pow_2(1+2*t);
        return hJ * hJ * dhJdt *
                math::pow(1 + par.J0 / hJ,  par.slopeIn) *
                math::pow(1 + hJ / par.J0, -par.slopeOut) *
                (math::pow(1 + par.Jcore/hJ * (par.Jcore/hJ - beta), -0.5*par.slopeIn) - 1);
    }

    public:
        NewBetaFinder(const NewDoublePowerLawParam& _par) : par(_par) {}

        virtual double value(const double beta) const
        {
            double result = 0;
            // use a fixed-order GL quadrature to compute the integrated difference in normalization between
            // unmodified and core-modified DF, which is sought to be zero by choosing an appropriate beta
            static const int GLORDER = 20;  // should be even, to avoid singularity in the integrand at t=0.5
            for(int i=0; i<GLORDER; i++)
                result += math::GLWEIGHTS[GLORDER][i] * deltaf(math::GLPOINTS[GLORDER][i], beta);
            return result;
        }
};

NewDoublePowerLaw::NewDoublePowerLaw(const NewDoublePowerLawParam &inparams) :
    par(inparams), beta(math::findRoot(NewBetaFinder(par), 0.0, 2.0, /*root-finder tolerance*/ SQRT_DBL_EPSILON))
{
    // sanity checks on parameters
    if(!(par.norm>0))
        throw std::invalid_argument("NewDoublePowerLaw: normalization must be positive");
    if(!(par.J0>0))
        throw std::invalid_argument("NewDoublePowerLaw: break action J0 must be positive");
    if(!(par.Jcore>=0 && beta>=0))
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
        double fac=math::pow(gJ / par.Jcutoff, par.cutoffStrength);
        if(fac>25)
            *val = 0;
        else
            *val *= exp(-fac);
    }
    if(par.Jcore>0) {   // central core of nearly-constant f(J) at small J
        if(hJ==0)
            *val = par.norm / pow_3(2*M_PI * par.J0);
        else
            *val *= math::pow(1 + par.Jcore/hJ * (par.Jcore/hJ - beta), -0.5*par.slopeIn);
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
    double Jvel = fabs(Jp) + par.addJvel;
    double xr = pow(Jvel/par.Jphi0,par.pr)/par.Jr0;
    double xz = pow(Jvel/par.Jphi0,par.pz)/par.Jz0;
    double fr = xr * exp(-xr*J.Jr), fz = xz * exp(-xz*J.Jz);
    double Jden = Jp + par.addJden;
    double xp = Jden / par.Jphi0;
    double fp = par.norm/par.Jphi0 * fabs(J.Jphi) / par.Jphi0 * exp(-xp);
    *val = fr * fz * fp;
    if(J.Jphi < 0) {
        double x=J.Jphi/par.addJden;
        *val *= exp(x*(1-x));
    }
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
const std::string componentNames = "bulge\tthin,young\tthin,middle\tthin,old\tthick\tstel.halo";


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
void writeRadialDensityProfile(const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing surface density profile\n";
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
            strmsd << '\t' << surfDens[ir*nc+ic] * intUnits.to_Msun_per_pc2;
            strmvd << '\t' << z0dens  [ir*nc+ic] * intUnits.to_Msun_per_pc3;
            strmsR << '\t' << sqrt(meanv2[ir*nc+ic].vx2) * intUnits.to_kms;
            strmsz << '\t' << sqrt(meanv2[ir*nc+ic].vz2) * intUnits.to_kms;
            strmsp << '\t' << sqrt(meanv2[ir*nc+ic].vy2 - pow_2(meanv[ir*nc+ic].vy)) * intUnits.to_kms;
            strmvp << '\t' << sqrt(meanv [ir*nc+ic].vy)  * intUnits.to_kms;
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
    for(double h=0; h<=3; h<0.50001 ? h+=0.05 : h+=0.25)
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
    strm << "# z[Kpc]\t" << componentNames << "\n";
    for(int ih=0; ih<nh; ih++) {
        strm << (heights[ih] * intUnits.to_Kpc);
        for(int ic=0; ic<nc; ic++)
            strm << '\t' << (dens[ih*nc+ic] * intUnits.to_Msun_per_pc3);
        strm << '\n';
    }
}

/// compute velocity distributions of each stellar component at several points and write them to respective files
void writeVelocityDistributions(const galaxymodel::GalaxyModel& model)
{
    std::cout << "Writing velocity distributions\n";
    const int numPoints = 4;
    const double R[numPoints] = {solarRadius-2.0, solarRadius, solarRadius+2.0, solarRadius};
    const double z[numPoints] = {0, 0, 0, 2.0};
    // create grids in velocity space
    double vR_max = 120 * intUnits.from_kms;
    double vp_max = 360 * intUnits.from_kms;
    double vp_min = -90 * intUnits.from_kms;
    std::vector<double> gridvR   = math::createUniformGrid(50, -vR_max, vR_max);
    std::vector<double> gridvz   = gridvR;  // for simplicity use the same grid for two dimensions
    std::vector<double> gridvphi = math::createUniformGrid(50, vp_min, vp_max);
    // store each component (thin/thick/bulge/etc) separately
    const int numComp = model.distrFunc.numValues();
    std::vector< std::vector<double> > dens(numPoints);
    std::vector< std::vector< std::vector<double> > > amplvx(numPoints), amplvy(numPoints), amplvz(numPoints);
    const int ORDER = 3;
    math::BsplineInterpolator1d<ORDER> intvR(gridvR), intvz(gridvz), intvphi(gridvphi);

    // loop over all points in parallel
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int p=0; p<numPoints; p++) {
        // allocate output vectors for all components at this point
        dens  [p].resize(numComp);
        amplvx[p].resize(numComp);
        amplvy[p].resize(numComp);
        amplvz[p].resize(numComp);
        galaxymodel::computeVelocityDistribution<ORDER>(model,
            coord::PosCar(R[p] * intUnits.from_Kpc, 0, z[p] * intUnits.from_Kpc),
            gridvR, gridvphi, gridvz,
            /*density*/ &dens[p].front(), &amplvx[p].front(), &amplvy[p].front(), &amplvz[p].front(),
            /*separate*/ true);
    }

    // now write out the results sequentially
    std::ofstream strm;
    // unit conversion: here the VDF is normalized so that the integral \int f(v) dv = rho,
    // the density of stars in each component at the given point;
    // therefore we need to multiply it by massUnit/lengthUnit^3/velocityUnit
    double conv = intUnits.to_Msun_per_pc3 / intUnits.to_kms;
    for(int p=0; p<numPoints; p++) {
        strm.open(("mwmodel_vdf_R"+utils::toString(R[p])+"_z"+utils::toString(z[p])+"_vR.txt").c_str());
        strm << "# v[km/s]\t" << componentNames << "[Msun/kpc^3/(km/s)]\n";
        for(int i=-100; i<=100; i++) {
            double v = i*vR_max/100;
            strm << v * intUnits.to_kms;
            for(int c=0; c<numComp; c++)
                strm << '\t' << dens[p][c] * intvR.interpolate(v, amplvx[p][c]) * conv;
            strm << '\n';
        }
        strm.close();
        strm.open(("mwmodel_vdf_R"+utils::toString(R[p])+"_z"+utils::toString(z[p])+"_vz.txt").c_str());
        strm << "# v[km/s]\t" << componentNames << "[Msun/kpc^3/(km/s)]\n";
        for(int i=-100; i<=100; i++) {
            double v = i*vR_max/100;
            strm << v * intUnits.to_kms << '\t';
            for(int c=0; c<numComp; c++)
                strm << '\t' << dens[p][c] * intvz.interpolate(v, amplvz[p][c]) * conv;
            strm << '\n';
        }
        strm.close();
        strm.open(("mwmodel_vdf_R"+utils::toString(R[p])+"_z"+utils::toString(z[p])+"_vphi.txt").c_str());
        strm << "# v[km/s]\t" << componentNames << "[Msun/kpc^3/(km/s)]\n";
        for(int i=0; i<=200; i++) {
            double v = i*1.0/200 * (vp_max-vp_min) + vp_min;
            strm << v * intUnits.to_kms << '\t';
            for(int c=0; c<numComp; c++)
                strm << '\t' << dens[p][c] * intvphi.interpolate(v, amplvy[p][c]) * conv;
            strm << '\n';
        }
        strm.close();
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
    int numIterations=5;
    std::time_t start_t = std::time(NULL);
    // read parameters from the INI file
    const std::string iniFileName = "../data/SCM_MW.ini";
    utils::ConfigFile ini(iniFileName);
    utils::KeyValueMap
    iniPotenThinDisk = ini.findSection("Potential thin disk"),
    iniPotenThickDisk= ini.findSection("Potential thick disk"),
    iniPotenGasDisk  = ini.findSection("Potential gas disk"),
    iniPotenBulge    = ini.findSection("Potential bulge"),
    iniPotenDarkHalo = ini.findSection("Potential dark halo"),
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

    // initialize density profiles of various components
    std::vector<potential::PtrDensity> densityStars(3);
    potential::PtrDensity
        densityDark = potential::createDensity(iniPotenDarkHalo, extUnits),
        densityGas  = potential::createDensity(iniPotenGasDisk,  extUnits);
    densityStars[0] = potential::createDensity(iniPotenBulge,    extUnits);
    densityStars[1] = potential::createDensity(iniPotenThinDisk, extUnits);
    densityStars[2] = potential::createDensity(iniPotenThickDisk,extUnits);

    // add components to SCM - at first, all of them are static density profiles
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(potential::PtrDensity(
        new potential::CompositeDensity(densityStars)), true)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityDark, false)));
    model.components.push_back(galaxymodel::PtrComponent(
        new galaxymodel::ComponentStatic(densityGas, true)));

    // initialize total potential of the model (first guess)
    updateTotalPotential(model);
    printoutInfo(model);

    std::cout << "**** STARTING MODELLING ****\nInitial masses of density components: "
        "Mbulge=" << (densityStars[0]->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mthin="  << (densityStars[1]->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mthick=" << (densityStars[2]->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mdark="  << (densityDark    ->totalMass() * intUnits.to_Msun) << " Msun, "
        "Mgas="   << (densityGas     ->totalMass() * intUnits.to_Msun) << " Msun\n";

    // create the dark halo DF
    df::PtrDistributionFunction dfHalo = df::createNewDoublePowerLawDF(iniDFDarkHalo, extUnits);
    // same for the stellar components (bulge, thin/thick disks, and stellar halo)
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
    model.components[0] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithDisklikeDF(dfStellar, potential::PtrDensity(),
        iniSCMDisk.getInt("mmaxAngularCyl"),
        iniSCMDisk.getInt("sizeRadialCyl"),
        iniSCMDisk.getDouble("RminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("RmaxCyl") * extUnits.lengthUnit,
        iniSCMDisk.getInt("sizeVerticalCyl"),
        iniSCMDisk.getDouble("zminCyl") * extUnits.lengthUnit,
        iniSCMDisk.getDouble("zmaxCyl") * extUnits.lengthUnit));
    // same for the dark halo
    model.components[1] = galaxymodel::PtrComponent(
        new galaxymodel::ComponentWithSpheroidalDF(dfHalo, potential::PtrDensity(),
        iniSCMHalo.getInt("lmaxAngularSph"),
        iniSCMHalo.getInt("mmaxAngularSph"),
        iniSCMHalo.getInt("sizeRadialSph"),
        iniSCMHalo.getDouble("rminSph") * extUnits.lengthUnit,
        iniSCMHalo.getDouble("rmaxSph") * extUnits.lengthUnit));
    // gas component is left as it is
    
    // we can compute the masses even though we don't know the density profile yet
    std::cout <<
        "Masses of DF components:"
            "\nMstars="    << (dfStellar->totalMass() * intUnits.to_Msun) << " Msun" <<
            " (Mbulge="    << (dfStellarArray[0]->totalMass() * intUnits.to_Msun) <<
            ", Myoung="    << (dfStellarArray[1]->totalMass() * intUnits.to_Msun) <<
            ", Mmiddle="   << (dfStellarArray[2]->totalMass() * intUnits.to_Msun) <<
            ", Mold="      << (dfStellarArray[3]->totalMass() * intUnits.to_Msun) <<
            ", Mthick="    << (dfStellarArray[4]->totalMass() * intUnits.to_Msun) <<
            ", Mstel.halo="<< (dfStellarArray[5]->totalMass() * intUnits.to_Msun) <<
            "); Mdark="    << (dfHalo ->totalMass() * intUnits.to_Msun) << " Msun\n";
    std::cout << "Potential value at origin=-("<<
        (sqrt(-model.totalPotential->value(coord::PosCyl(0,0,0))) * intUnits.to_kms) << " km/s)^2\n";
    // update the action finder
    std::cout << "Updating action finder..."<<std::flush;
    model.actionFinder.reset(new actions::ActionFinderAxisymFudge(model.totalPotential,
        model.useActionInterpolation));
    std::cout << "done" << std::endl;
    // do a few more iterations to obtain the self-consistent density profile for both disks
    for(int iteration=1; iteration<=numIterations; iteration++) {
        std::cout << "Starting iteration #" << iteration << "\n";
        doIteration(model);
        printoutInfo(model);
    }
    std::time_t finish_t = std::time(NULL);
    std::cout << finish_t - start_t << " secs to build the model\n";

    // output various profiles
    galaxymodel::GalaxyModel modelStars(*model.totalPotential, *model.actionFinder, *dfStellar);
    writeRadialDensityProfile(modelStars);
    writeVerticalDensityProfile(modelStars);
    writeVelocityDistributions(modelStars);

    // export model to an N-body snapshot
    std::string format = "nemo";   // could use "text", "nemo" or "gadget" here
    std::cout << "Writing a complete DF-based N-body model for the dark matter halo\n";
    particles::writeSnapshot("mwmodel_dm_final.nbody", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfHalo), 750000),
        format, extUnits);
    std::cout << "Writing a complete DF-based N-body model for the stellar bulge, disk and halo\n";
    particles::writeSnapshot("mwmodel_stars_final.nbody", galaxymodel::samplePosVel(
        galaxymodel::GalaxyModel(*model.totalPotential, *model.actionFinder, *dfStellar), 200000),
        format, extUnits);
    // we didn't use an action-based DF for the gas disk, leaving it as a static component;
    // to create an N-body representation, we sample the density profile and assign velocities
    // from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    std::cout << "Writing an N-body model for the gas disk\n";
    particles::writeSnapshot("mwmodel_gas_final.nbody", galaxymodel::assignVelocity(
        galaxymodel::sampleDensity(*model.components[2]->getDensity(), 50000),
        /*parameters for the axisymmetric Jeans velocity sampler*/
        *model.components[2]->getDensity(), *model.totalPotential, /*beta*/ 0., /*kappa*/ 1.),
        format, extUnits);

    std::cout << std::time(NULL) - finish_t << " secs to compute diagnostics\n";
}
