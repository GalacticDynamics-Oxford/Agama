/** \file    example_doublepowerlaw.cpp
    \author  Eugene Vasiliev
    \date    2017-2025

    This example program helps to find the parameters of a DoublePowerLaw DF that best correspond
    to the given spherical isotropic model in the given spherical potential.
    See example_doublepowerlaw.py for another method for finding a DoublePowerLaw DF
    approximating the given density profile, which can also deal with flattened systems.
*/
#include <iostream>
#include <cmath>
#include "df_factory.h"
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


/// max number of free parameters
const unsigned int NPARAMS = 8;

/// parameters that are provided (fixed) by the user
df::DoublePowerLawParam defaultParams;

inline double choose(double x, double y) { return x==x?x:y; }

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
        if(jac!=0 && isFinite(jac)) {
            jac *= TWO_PI_CUBE;   // integral over three angles
            double valf = f.value(act), valg = g.value(act);
            if(!isFinite(valf))
                valf = 0;
            if(!isFinite(valg))
                valg = 0;
            values[0] = valf * jac;
            values[1] = valg * jac;
            values[2] = valf * jac * math::clip(log(valf / valg), -100., 100.);
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

double kullbackLeiblerDistance(
    const df::BaseDistributionFunction& f, const df::BaseDistributionFunction& g)
{
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result[3];   // integrals of f, g, and f*ln(f/g)
    math::integrateNdim(DFIntegrandKLD(f, g),
        xlower, xupper, /*reqRelError*/ 1e-4, /*maxNumEval*/ 1e6, result);
    return result[2] / result[0] + log(result[1] / result[0]);
}

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
    params.Jcutoff   = choose(defaultParams.Jcutoff,   exp(vars[6]));
    params.cutoffStrength = choose(defaultParams.cutoffStrength, vars[7]);
    params.norm      = 1.;
    return params;
}

/// function to be minimized
class ModelSearchFnc: public math::IFunctionNdim{
    const df::BaseDistributionFunction& f;
public:
    ModelSearchFnc(const df::BaseDistributionFunction& _f) : f(_f)
    {
        std::cout << "J0        slopeIn   slopeOut  steepness coefJrIn  "
            "coefJrOut Jcutoff cutoffStrength :  KLD\n";
    }

    virtual void eval(const double vars[], double values[]) const
    {
        df::DoublePowerLawParam params = dfparams(vars);
        try{
            double kld =
                params.slopeIn  <0   || params.slopeIn >=3   ||
                /*params.slopeOut<=3   ||*/ params.slopeOut>12   ||
                params.steepness<0.2 || params.steepness>5   ||
                params.coefJrIn <0.1 || params.coefJrIn >2.8 ||
                params.coefJzOut<0.1 || params.coefJrOut>2.8 ||
                params.cutoffStrength<0.2 || params.cutoffStrength>5 ?
                /*INFINITY*/ 1e100 :
                kullbackLeiblerDistance(f, df::DoublePowerLaw(params));
            std::cout <<
            utils::pp(params.J0,       8) + "  " +
            utils::pp(params.slopeIn,  8) + "  " +
            utils::pp(params.slopeOut, 8) + "  " +
            utils::pp(params.steepness,8) + "  " +
            utils::pp(params.coefJrIn, 8) + "  " +
            utils::pp(params.coefJrOut,8) + "  " +
            utils::pp(params.Jcutoff,  8) + "  " +
            utils::pp(params.cutoffStrength, 8) + " : " +
            utils::pp(kld, 8) + "\r";
            values[0] = kld;
        }
        catch(std::exception&) {
            values[0] = /*INFINITY*/ 1e100;
        }
    }
    virtual unsigned int numVars() const { return NPARAMS; }
    virtual unsigned int numValues() const { return 1; }
};

int main(int argc, char* argv[])
{

    if(argc<=1) {
        std::cout << "Find parameters of a double-power-law action-based DF "
        "that best correspond to the given spherical (an)isotropic model.\n"
        "The spherical model is expressed as a QuasiSpherical DF f(J) (constructed numerically), "
        "and approximated by a DoublePowerLaw DF g(J), by minimizing the Kullback-Leibler distance "
        "between f(J) and g(J).\n"
        "After the best-fit parameters are found, print the radial profiles of density, "
        "circular velocity, and velocity dispersions of the original and the approximate models.\n"
        "Arguments:\n"
        "  density=... - name of the density model or the name of an INI file with density parameters.\n"
        "  scaleRadius=..., gamma=..., etc. - other parameters of the density model (if given by name).\n"
        "  potential=... - if provided, specifies and INI file with parameters of the potential "
        "that may be different from the density profile.\n"
        "  beta0, r_a - parameters controlling the velocity anisotropy of the Cuddeford-Osipkov-Merritt "
        "model (central value of the anisotropy and the radius beyond which it transitions to unity); "
        "default values (0, infinity) generate isotropic models.\n"
        "  slopeIn, slopeOut, steepness, coefJrIn, coefJrOut, Jcutoff, cutoffStrength - "
        "if provided, fix the corresponding parameters of the DoublePowerLaw DF to the given values; "
        "otherwise the best-fit value will be found during the optimization procedure.\n";
        return 0;
    }

    // parse command-line parameters
    utils::KeyValueMap args(argc-1, argv+1);

    std::string inputdensity   = args.popString("density");
    std::string inputpotential = args.popString("potential");
    defaultParams.slopeIn      = args.popDouble("slopeIn",   NAN);
    defaultParams.slopeOut     = args.popDouble("slopeOut",  NAN);
    defaultParams.steepness    = args.popDouble("steepness", NAN);
    defaultParams.coefJrIn     = args.popDouble("coefJrIn",  NAN);
    defaultParams.coefJrOut    = args.popDouble("coefJrOut", NAN);
    defaultParams.Jcutoff      = args.popDouble("Jcutoff",   NAN);
    defaultParams.cutoffStrength=args.popDouble("cutoffStrength", NAN);
    double beta0               = args.popDouble("beta0", 0);
    double r_a                 = args.popDouble("r_a", INFINITY);

    math::LogLogSpline densInterp;  // interpolated density profile constructed from a table
    potential::PtrDensity dens;     // the density profile (analytic or interpolated)
    potential::PtrPotential pot;    // the potential (may be different from the density)

    // input is a name of a density profile or an INI file with its parameters;
    // the choice is made based on whether 'density=...' specifies an existing file name
    if(!inputdensity.empty() && utils::fileExists(inputdensity)) {
        dens = potential::readDensity(inputdensity);
    } else {
        args.set("type", inputdensity);
        dens = potential::createDensity(args);
    }

    // check if a separate potential was also provided
    if(inputpotential.empty()) {
        pot = potential::Multipole::create(*dens, coord::ST_SPHERICAL, 0, 0, /*gridsize*/ 40);
    } else if(utils::fileExists(inputpotential)) {
        // create a (possibly composite) potential from the parameters provided in an INI file
        pot = potential::readPotential(inputpotential);
    } else {
        std::cout << "Argument 'potential', if provided, must refer to an INI file "
            "with potential parameters\n";
        return 1;
    }

    if(!isSpherical(*dens) || !isSpherical(*pot)) {
        std::cout << "Density and potential models must be spherical\n";
        return 1;
    }

    // construct the mapping between energy and phase volume
    potential::PhaseVolume phasevol((potential::Sphericalized<potential::BasePotential>(*pot)));

    // [0]: original DF, [1]: approximate DoublePowerLaw DF
    std::vector<df::PtrDistributionFunction> dfs(2);

    // construct the distribution function from the density,
    // using the anisotropic generalization of the Eddington inversion formula
    dfs[0].reset(new df::QuasiSphericalCOM(
        potential::Sphericalized<potential::BaseDensity>(*dens),
        potential::Sphericalized<potential::BasePotential>(*pot),
        beta0, r_a));
    double bestparams[NPARAMS] = {0.0, 1.0, 6.0, 1.0, 1.0, 1.0, 20.0, 2.0};
    double stepsizes [NPARAMS] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 10.0, 0.1};
    const int maxNumIter = 500, maxNumLoop = 5;
    const double toler   = 1e-4;
    ModelSearchFnc fnc(*dfs[0]);
    double bestScore = INFINITY, currScore;
    // run the Nelder-Mead search a few times, because it can get stuck in a local minimum
    for(int loop=0; loop<maxNumLoop; loop++) {
        math::findMinNdim(fnc, bestparams, stepsizes, toler, maxNumIter, bestparams);
        fnc.eval(bestparams, &currScore);
        std::cout << '\n';
        if(currScore >= 0.9*bestScore)  // doesn't improve - hopefully found the global minimum
            break;
        bestScore = fmin(currScore, bestScore);
    }

    // compare the density and velocity dispersion profiles of the final model with the original ones
    const actions::ActionFinderSpherical af(*pot);
    df::DoublePowerLawParam bestParams = dfparams(bestparams);
    dfs[1].reset(new df::DoublePowerLaw(bestParams));
    df::CompositeDF dfcomp(dfs);
    galaxymodel::GalaxyModel mod(*pot, af, dfcomp);
    std::vector<double> rad = math::createExpGrid(61, 1e-3, 1e3);
    int size = rad.size();
    std::vector<double> rho_o(size), rho_a(size);
    std::vector<coord::Vel2Car> sigma2(size*2);
    double totalMass = dens->totalMass(), dfMass = dfs[1]->totalMass();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++) {
        double rho[2];
        computeMoments(mod, coord::PosCar(rad[i],0,0), rho, NULL, &sigma2[i*2], /*separate*/true);
        rho_o[i] = rho[0];
        rho_a[i] = rho[1] * totalMass / dfMass;
    }
    potential::PtrPotential pot_a = potential::Multipole::create(
        potential::DensitySphericalHarmonic(rad, std::vector< std::vector<double> >(1, rho_a)),
        coord::ST_SPHERICAL, 0, 0, rad.size(), rad.front(), rad.back());
    std::cout << "#R      rho_orig  rho_appr  vcirc_o   vcirc_a   sigmaR_o  sigmaR_a  sigmaT_o  sigmaT_a\n";
    for(int i=0; i<size; i++) {
        std::cout <<
            utils::pp(rad[i], 7) << ' ' <<
            utils::pp(rho_o[i], 9) << ' ' <<
            utils::pp(rho_a[i], 9) << ' ' <<
            utils::pp(v_circ(*pot  , rad[i]), 9) << ' ' <<
            utils::pp(v_circ(*pot_a, rad[i]), 9) << ' ' <<
            utils::pp(sqrt(sigma2[i*2  ].vx2), 9) << ' ' <<
            utils::pp(sqrt(sigma2[i*2+1].vx2), 9) << ' ' <<
            utils::pp(sqrt(sigma2[i*2  ].vy2), 9) << ' ' <<
            utils::pp(sqrt(sigma2[i*2+1].vy2), 9) << '\n';
    }
    std::cout << "Best-fit parameters of DoublePowerLaw DF "
    "(Kullback-Leibler distance=" << bestScore << "):\n"
    "norm          = " << utils::pp(totalMass / dfMass,       7) << "\n"
    "J0            = " << utils::pp(bestParams.J0,            7) << "\n"
    "slopeIn       = " << utils::pp(bestParams.slopeIn,       7) << "\n"
    "slopeOut      = " << utils::pp(bestParams.slopeOut,      7) << "\n"
    "steepness     = " << utils::pp(bestParams.steepness,     7) << "\n"
    "coefJrIn      = " << utils::pp(bestParams.coefJrIn,      7) << "\n"
    "coefJzIn      = " << utils::pp(bestParams.coefJzIn,      7) << "\n"
    "coefJrOut     = " << utils::pp(bestParams.coefJrOut,     7) << "\n"
    "coefJzOut     = " << utils::pp(bestParams.coefJzOut,     7) << "\n"
    "Jcutoff       = " << utils::pp(bestParams.Jcutoff,       7) << "\n"
    "cutoffStrength= " << utils::pp(bestParams.cutoffStrength,7) << "\n";
}
