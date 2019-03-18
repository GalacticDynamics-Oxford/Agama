/** \file    example_doublepowerlaw.cpp
    \author  Eugene Vasiliev
    \date    2017

    This example program helps to find the parameters of a DoublePowerLaw DF that best correspond
    to the given spherical isotropic model in the given spherical potential.
*/
#include <iostream>
#include <cmath>
#include "df_halo.h"
#include "df_spherical.h"
#include "galaxymodel_base.h"
#include "galaxymodel_spherical.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "math_core.h"
#include "math_fit.h"
#include "utils.h"
#include "utils_config.h"


/// helper class for computing the Kullback-Leibler distance betweeen DFs f and g
class DFIntegrandKLD: public math::IFunctionNdim {
    const df::BaseDistributionFunction &f, &g;      ///< two DF instances
    const df::ActionSpaceScalingTriangLog scaling;  ///< action-space scaling transformation
public:
    DFIntegrandKLD(const df::BaseDistributionFunction& _f, const df::BaseDistributionFunction& _g) :
        f(_f), g(_g), scaling() {}

    virtual void eval(const double vars[], double values[]) const
    {
        double jac;  // will be initialized by the following call
        const actions::Actions act = scaling.toActions(vars, &jac);
        if(jac!=0) {
            jac *= TWO_PI_CUBE;   // integral over three angles
            double valf = f.value(act), valg = g.value(act);
            if(!isFinite(valf))
                valf = 0;
            if(!isFinite(valg))
                valg = 0;
            values[0] = valf * jac;
            values[1] = valg * jac;
            values[2] = valf>0 ? valf * jac * log(valf / valg) : 0;
        } else {
            // we're (almost) at zero or infinity in terms of magnitude of J
            // at infinity we expect that f(J) tends to zero,
            // while at J->0 the jacobian of transformation is exponentially small.
            values[0] = values[1] = values[2] = 0;
        }
    }

    /// number of variables (3 actions)
    virtual unsigned int numVars()   const { return 3; }
    /// number of values to compute: f, g, f * ln(f/g)
    virtual unsigned int numValues() const { return 3; }
};

double kullbackLeiblerDistance(const df::BaseDistributionFunction& f, const df::BaseDistributionFunction& g)
{
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result[3];   // integrals of f, g, and f*ln(f/g)
    math::integrateNdim(DFIntegrandKLD(f, g),
        xlower, xupper, /*reqRelError*/ 1e-4, /*maxNumEval*/ 1e6, result);
    return result[2] / result[0] + log(result[1] / result[0]);
}

/// max number of free parameters
const unsigned int NPARAMS = 6;

df::DoublePowerLawParam defaultParams;

inline double choose(double x, double y) { return x==x?x:y; }

/// convert from parameter space to DF params
df::DoublePowerLawParam dfparams(const double vars[])
{
    df::DoublePowerLawParam params;
    params.J0        = exp(vars[0]);
    params.slopeIn   = choose(defaultParams.slopeIn,   vars[1]);
    params.slopeOut  = choose(defaultParams.slopeOut,  vars[2]);
    params.steepness = choose(defaultParams.steepness, vars[3]);
    params.coefJrIn  = choose(defaultParams.coefJrIn,  vars[4]);
    params.coefJzIn  = (3-params.coefJrIn)/2;  // fix g_z=g_phi taking into account that g_r+g_z+g_phi=3
    params.coefJrOut = choose(defaultParams.coefJrOut, vars[5]);
    params.coefJzOut = (3-params.coefJrOut)/2; // same for h_z
    params.norm      = 1.;
    return params;
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
    const df::BaseDistributionFunction& f;
public:
    ModelSearchFnc(const df::BaseDistributionFunction& _f) : f(_f) {};
    virtual void eval(const double vars[], double values[]) const
    {
        df::DoublePowerLawParam params = dfparams(vars);
        std::cout <<
            "J0="         << utils::pp(params.J0,       7) <<
            " slopeIn="   << utils::pp(params.slopeIn,  7) <<
            " slopeOut="  << utils::pp(params.slopeOut, 7) <<
            " steepness=" << utils::pp(params.steepness,7) <<
            " coefJrIn="  << utils::pp(params.coefJrIn, 7) <<
            " coefJrOut=" << utils::pp(params.coefJrOut,7) << ": ";
        try{
            if(params.slopeIn<0) params.slopeIn=0;
            double kld = params.slopeOut>12 || params.slopeOut<=3 ||
                params.coefJrIn<0.1 || params.coefJrIn>2.9 ? INFINITY :
                kullbackLeiblerDistance(f, df::DoublePowerLaw(params));
            std::cout << "KLD="       << utils::pp(kld, 8) << std::endl;
            values[0] = kld;
        }
        catch(std::exception&e) {
            std::cout << e.what() << std::endl;
            values[0] = INFINITY;
        }
    }
    virtual unsigned int numVars() const { return NPARAMS; }
    virtual unsigned int numValues() const { return 1; }
};

int main(int argc, char* argv[])
{

    if(argc<=1) {
        std::cout << "Find parameters of a double-power-law action-based DF "
        "that best correspond to the given spherical isotropic model.\n"
        "The spherical model is expressed as a QuasiSpherical DF f(J) (constructed numerically), "
        "and approximated by a DoublePowerLaw DF g(J), by minimizing the Kullback-Leibler distance "
        "between f(J) and g(J).\n"
        "After the best-fit parameters are found, print the radial profiles of density, "
        "circular velocity, and velocity dispersions of the original and the approximate models.\n"
        "Arguments:\n"
        "  density=... - name of the density model or a file with cumulative mass profile;\n"
        "  scaleRadius=..., gamma=..., etc. - parameters of the density model;\n"
        "  potential=... - if provided, describes the potential that may be different from "
        "the density profile; in this case the density model must be given by a file with "
        "cumulative mass profile M(r), and other command-line parameters refer to the potential.\n"
        "  slopeIn, slopeOut, steepness, coefJrIn, coefJrOut - if provided, fix the corresponding "
        "parameters of the double-power-law DF to the given value; otherwise the best-fit value "
        "will be found during the optimization procedure\n";
        return 0;
    }

    // parse command-line parameters
    utils::KeyValueMap args(argc-1, argv+1);
    std::string inputdensity   = args.getString("density");
    std::string inputpotential = args.getString("potential");
    defaultParams.slopeIn      = args.getDouble("slopeIn",   NAN);
    defaultParams.slopeOut     = args.getDouble("slopeOut",  NAN);
    defaultParams.steepness    = args.getDouble("steepness", NAN);
    defaultParams.coefJrIn     = args.getDouble("coefJrIn",  NAN);
    defaultParams.coefJrOut    = args.getDouble("coefJrOut", NAN);

    math::LogLogSpline densInterp;       // interpolated density profile constructed from a table
    potential::PtrDensity dens;          // the density profile (analytic or interpolated)
    potential::PtrPotential pot;         // the potential (may be different from the density)

    // input is a name of a density profile or a file with the cumulative mass profile;
    // the choice is made based on whether 'density=...' specifies an existing file name
    if(!inputdensity.empty() && utils::fileExists(inputdensity)) {
        densInterp = potential::readMassProfile(inputdensity);
        dens.reset(new potential::FunctionToDensityWrapper(densInterp));
    } else
        dens = potential::createDensity(args);

    // check if a separate potential was also provided
    if(inputpotential.empty()) {
        pot = potential::Multipole::create(*dens, 0, 0, /*gridsize*/ 40);
    } else {
        // the createPotential() routine reads the potential name from the 'type=...' parameter
        args.set("type", inputpotential);
        pot = potential::createPotential(args);
    }

    if(!isSpherical(*dens) || !isSpherical(*pot)) {
        std::cout << "Density and potential models must be spherical\n";
        return 0;
    }

    // construct the mapping between energy and phase volume
    potential::PhaseVolume phasevol((potential::PotentialWrapper(*pot)));

    // compute the distribution function from the density (using the Eddington inversion formula)
    math::LogLogSpline eddf = df::createSphericalIsotropicDF(
        potential::DensityWrapper(*dens), potential::PotentialWrapper(*pot));
    df::QuasiSphericalIsotropic dfsph(eddf, *pot);

    const double initparams[NPARAMS] = {0.0, 1.0, 6.0, 1.0, 1.0, 1.0};
    const double stepsizes [NPARAMS] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    const int maxNumIter = 1000;
    const double toler   = 1e-4;
    double bestparams[NPARAMS];
    ModelSearchFnc fnc(dfsph);
    int numIter = math::findMinNdim(fnc, initparams, stepsizes, toler, maxNumIter, bestparams);
    std::cout << numIter << " iterations\n";

    // plot the profiles of the final model
    const actions::ActionFinderSpherical af(*pot);
    galaxymodel::GalaxyModel mod(*pot, af, df::DoublePowerLaw(dfparams(bestparams)));
    galaxymodel::SphericalIsotropicModelLocal spm(phasevol, eddf);
    std::vector<double> rad = math::createExpGrid(61, 1e-3, 1e3);
    std::vector<double> rho(rad), sigR(rad), sigT(rad);
    double Mtot = mod.distrFunc.totalMass();
    int s=rad.size();
#pragma omp parallel for
    for(int i=0; i<s; i++) {
        coord::Vel2Cyl vel2;
        computeMoments(mod, coord::PosCyl(rad[i],0,0), &rho[i], NULL, &vel2, NULL, NULL, NULL);
        sigR[i] = sqrt(vel2.vR2);
        sigT[i] = sqrt(vel2.vz2);
        rho[i] /= Mtot;
    }
    potential::PtrPotential pot_appr = potential::Multipole::create(
        potential::DensitySphericalHarmonic(rad, std::vector< std::vector<double> >(1, rho)),
        0, 0, 40, 0.5e-3, 2e3);
    std::cout << "#R      rho_orig  rho_appr  vcirc_o   vcirc_a   sigma_o   sig_R     sig_t\n";
    for(int i=0; i<s; i++) {
        double Phi=pot->value(coord::PosCyl(rad[i],0,0));
        std::cout <<
            utils::pp(rad[i], 7) << ' ' <<
            utils::pp(spm.density(Phi), 9) << ' ' <<
            utils::pp(rho[i], 9) << ' ' <<
            utils::pp(v_circ(*pot, rad[i]), 9) << ' ' <<
            utils::pp(v_circ(*pot_appr, rad[i]), 9) << ' ' <<
            utils::pp(spm.velDisp(Phi), 9) << ' ' <<
            utils::pp(sigR[i], 9) << ' ' <<
            utils::pp(sigT[i], 9) << '\n';
    }
}
