/** \file    test_df_halo.cpp
    \author  Eugene Vasiliev
    \date    August 2015

    This test demonstrates that the action-based double-power-law distribution function
    corresponds rather well to the ergodic distribution function obtained by
    the Eddington inversion formula for the known spherically-symmetric isotropic model.
    We create an instance of DF and compute several quantities (such as density and
    velocity dispersion), comparing them to the standard analytical expressions.
*/
#include <iostream>
#include <fstream>
#include "potential_dehnen.h"
#include "actions_spherical.h"
#include "df_halo.h"
#include "galaxymodel_base.h"
#include "particles_io.h"
#include "math_random.h"
#include "math_specfunc.h"
#include "math_spline.h"
#include "debug_utils.h"
#include "utils.h"

const double reqRelError = 1e-5;
const int maxNumEval = 1e5;
const char* errmsg = "\033[1;31m **\033[0m";
std::string histograms;

bool testTotalMass(const galaxymodel::GalaxyModel& galmod, double massExact)
{
    std::cout << "\033[1;33mTesting " << galmod.potential.name() << "\033[0m\n";
    // Calculate the total mass
    double err;
    int numEval;
    double mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval, &err, &numEval);
    if(err > mass*0.01) {
        std::cout << "Mass=" << mass << " +- " << utils::pp(err,7) <<
        " in " << numEval << " DF evaluations\n";
        mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval*10, &err, &numEval);
    }
    bool ok = math::fcmp(mass, massExact, 0.05) == 0; // 5% relative error allowed because ar!=az
    std::cout <<
        "Mass=" << mass << " +- " << utils::pp(err,7) << " in " << numEval << " DF evaluations"
        " (analytic value=" << massExact << (ok?"":errmsg) <<")\n";
    return ok;
}

// return a random number or 0 or 1, to test all possibilities including the boundary cases
inline double rnd(int i) { return (i<=1) ? i*1. : math::random(); }
bool testActionSpaceScaling(const df::BaseActionSpaceScaling& s)
{
    for(int i=0; i<1000; i++) {
        double v[3] = { rnd(i%10), rnd(i/10%10), rnd(i/100) };
        actions::Actions J = s.toActions(v);
        if(J.Jr!=J.Jr || J.Jz!=J.Jz || J.Jphi!=J.Jphi) {
            std::cout << "v=[" << v[0] << "," << v[1] << "," << v[2] << "]: " << J << "\n";
            return false;
        }
        double w[3];
        s.toScaled(J, w);
        if(!(w[0]>=0 && w[0]<=1 && w[1]>=0 && w[1]<=1 && w[2]>=0 && w[2]<=1)) {
            std::cout << J << ": w=[" << w[0] << "," << w[1] << "," << w[2] << "\n";
            return false;
        }
        actions::Actions J1 = s.toActions(w);
        if( J1.Jr!=J1.Jr || J1.Jz!=J1.Jz || J1.Jphi!=J1.Jphi ||
            (isFinite(J1.Jr+J1.Jz+fabs(J1.Jphi)) && (
            math::fcmp(J.Jr, J1.Jr, 2e-10)!=0 ||
            math::fcmp(J.Jz, J1.Jz, 2e-10)!=0 ||
            math::fcmp(J.Jphi, J1.Jphi, 2e-10)!=0) ) )
        {
            std::cout << J << " != " << J1 << "\n";
            return false;
        }
    }
    return true;
}

bool testDFmoments(const galaxymodel::GalaxyModel& galmod, const coord::PosVelCyl& point,
    double dfExact, double densExact, double sigmaExact)
{
    std::cout << "\033[1mAt point " << point << "\033[0m we have\n";
    // compare the action-based distribution function f(J) with the analytic one
    actions::Actions J = galmod.actFinder.actions(point);
    // we only compare the even part of f(J), taking the average of its values at J_phi and -J_phi
    double dfValue = galmod.distrFunc.value(J);
    J.Jphi *= -1;
    dfValue = 0.5 * (galmod.distrFunc.value(J) + dfValue);
    double energy = totalEnergy(galmod.potential, point);
    bool dfok = math::fcmp(dfValue, dfExact, 0.05) == 0;
    std::cout <<
        "f(J)=" << dfValue << " for actions " << J << "\n"
        "f(E)=" << dfExact << " for energy E=" << energy << (dfok?"":errmsg) <<"\n";

    // compute density and velocity moments
    double density, densityErr;
    double velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    computeMoments(galmod, point,
        &density,    &velocityFirstMoment,    &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr,
        false, reqRelError, maxNumEval);
    bool densok = math::fcmp(density, densExact, 0.05) == 0;
    bool sigmaok =
        math::fcmp(velocitySecondMoment.vR2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vz2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vphi2, sigmaExact, 0.05) == 0;

    // compute velocity distributions
    double vmax = sqrt(-2*galmod.potential.value(point));
    std::vector<double> gridVR   = math::createUniformGrid(41, -vmax, vmax);
    std::vector<double> gridVz   = math::createUniformGrid(41, -vmax, vmax);
    std::vector<double> gridVphi = math::createUniformGrid(41, -vmax, vmax);
    std::vector<double> amplVR, amplVz, amplVphi/*, projVR, projVz, projVphi*/;
    double densvdf;
    const int ORDER = 3;
    math::BsplineInterpolator1d<ORDER> intVR(gridVR), intVz(gridVz), intVphi(gridVphi);
    galaxymodel::computeVelocityDistribution<ORDER>(galmod, point, false,
        gridVR, gridVz, gridVphi, &densvdf, &amplVR, &amplVz, &amplVphi);
    // skip the projected ones for the moment -- they are more expensive
    //galaxymodel::computeVelocityDistribution<ORDER>(galmod, point, true,
    //    gridVR, gridVz, gridVphi, projVR, projVz, projVphi, false, /*accuracy*/1e-2, /*maxNumEval*/1e6);
    // output the profiles
    for(int i=-100; i<=100; i++) {
        double v = i*vmax/100;
        histograms +=
        utils::pp(v, 9)+'\t'+
        utils::pp(intVR.  interpolate(v, amplVR),   9)+' '+
        utils::pp(intVz.  interpolate(v, amplVz),   9)+' '+
        utils::pp(intVphi.interpolate(v, amplVphi), 9)+'\n';
        //utils::pp(intVR.  interpolate(v, projVR),   9)+' '+
        //utils::pp(intVz.  interpolate(v, projVz),   9)+' '+
        //utils::pp(intVphi.interpolate(v, projVphi), 9)+'\n';
    }
    histograms+='\n';
    // compute the dispersions from the VDF (integrate with the weight factor v^2)
    double sigmaR  = intVR.  integrate(-vmax, vmax, amplVR, 2 /*power index for the weighting*/);
    double sigmaz  = intVz.  integrate(-vmax, vmax, amplVz, 2);
    double sigmaphi= intVphi.integrate(-vmax, vmax, amplVphi, 2);
    // they should agree with the velocity moments computed above
    bool densvdfok = math::fcmp(densvdf, density, 2e-5) == 0;
    bool sigmavdfok=
        math::fcmp(sigmaR,   velocitySecondMoment.vR2,   2e-5) == 0 &&
        math::fcmp(sigmaz,   velocitySecondMoment.vz2,   2e-5) == 0 &&
        math::fcmp(sigmaphi, velocitySecondMoment.vphi2, 2e-5) == 0;

    std::cout <<
        "density=" << density << " +- " << utils::pp(densityErr,7) << (densvdfok?"":errmsg) <<
        "  compared to analytic value " << densExact << (densok?"":errmsg) <<"\n"
        "mean velocity  vphi=" << velocityFirstMoment << " +- " << utils::pp(velocityFirstMomentErr, 7) << "\n"
        "velocity dispersion"
        "  vR2="    << velocitySecondMoment.vR2    << " +- " << utils::pp(velocitySecondMomentErr.vR2,  7) <<
        ", vz2="    << velocitySecondMoment.vz2    << " +- " << utils::pp(velocitySecondMomentErr.vz2,  7) <<
        ", vphi2="  << velocitySecondMoment.vphi2  << " +- " << utils::pp(velocitySecondMomentErr.vphi2,7) <<
        (sigmavdfok?"":errmsg) <<
        "   compared to analytic value " << sigmaExact << (sigmaok?"":errmsg) <<"\n";
    return dfok && densok && sigmaok && densvdfok && sigmavdfok;
}

/// analytic expression for the ergodic distribution function f(E)
/// in a Hernquist model with mass m, scale radius a, at energy E.
double dfHernquist(double m, double a, double E)
{
    double q = sqrt(-E*a/m);
    return m / (4 * pow(2 * m * a * M_PI*M_PI, 1.5) ) * pow(1-q*q, -2.5) *
        (3*asin(q) + q * sqrt(1-q*q) * (1-2*q*q) * (8*q*q*q*q - 8*q*q - 3) );
}

/// analytic expression for isotropic 1d velocity dispersion sigma^2
/// in a Hernquist model with mass m, scale radius a, at a radius r
double sigmaHernquist(double m, double a, double r)
{
    double x = r/a;
    return m/a * ( x * pow_3(x + 1) * log(1 + 1/x) -
        x/(x+1) * ( 25./12 + 13./3*x + 7./2*x*x + x*x*x) );
}

const int NUM_POINTS_H = 3;
const double testPointsH[NUM_POINTS_H][6] = {
    {0,   1, 0, 0, 0, 0.5},
    {0.2, 0, 0, 0, 0, 0.2},
    {5.0, 2, 9, 0, 0, 0.3} };

int main(){
    bool ok = true;
    ok &= testActionSpaceScaling(df::ActionSpaceScalingTriangLog());
    ok &= testActionSpaceScaling(df::ActionSpaceScalingRect(0.765432,0.234567));
    if(!ok) std::cout << "Scaling transformation in action space failed\n";

    // test double-power-law distribution function in a spherical Hernquist potential
    // NB: parameters obtained by fitting (example_df_fit.cpp)
    df::DoublePowerLawParam paramDPL;
    paramDPL.slopeIn   = 1.55;
    paramDPL.slopeOut  = 5.25;
    paramDPL.steepness = 1.24;
    paramDPL.J0        = 1.53;
    paramDPL.coefJrIn  = 1.56;
    paramDPL.coefJzIn  = (3-paramDPL.coefJrIn)/2;
    paramDPL.coefJrOut = 1.0;
    paramDPL.coefJzOut = 1.0;
    paramDPL.norm      = 2.7;
    paramDPL.rotFrac   = 0.5;  // add some rotation (odd-Jphi component),
    paramDPL.Jphi0     = 0.7;  // it shouldn't affect the even-order moments
    const potential::Dehnen potH(1., 1., 1., 1., 1.);        // potential
    const actions::ActionFinderSpherical actH(potH);         // action finder
    const df::DoublePowerLaw dfH(paramDPL);                  // distribution function
    const galaxymodel::GalaxyModel galmodH(potH, actH, dfH); // all together - the mighty triad

    ok &= testTotalMass(galmodH, 1.);

    for(int i=0; i<NUM_POINTS_H; i++) {
        const coord::PosVelCyl point(testPointsH[i]);
        double dfExact    = dfHernquist(1, 1, totalEnergy(potH, point));     // f(E) for the Hernquist model
        double densExact  = potH.density(point);                             // analytical value of density
        double sigmaExact = sigmaHernquist(1, 1, coord::toPosSph(point).r);  // analytical value of sigma^2
        ok &= testDFmoments(galmodH, point, dfExact, densExact, sigmaExact);
    }

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("test_df_halo.dat");
        strm << "#v       \tf(v_R)    f(v_z)    f(v_phi)\n";
        strm << histograms;
    }

    // test the model with a central core: the introduction of a core shouldn't change
    // the overall normalization, at least when the coefficients in the linear combination
    // of actions are the same in the inner and outer part of the halo, and there is no exp-cutoff
    paramDPL.coefJrOut = paramDPL.coefJrIn;
    paramDPL.coefJzOut = paramDPL.coefJzIn;
    paramDPL.norm = 2.28;
    // first compute the total mass for a model without core
    ok &= testTotalMass(galaxymodel::GalaxyModel(potH, actH, df::DoublePowerLaw(paramDPL)), 1.);
    // then set up a sizable core, create another DF and repeat the mass computation
    paramDPL.Jcore = 0.8;
    ok &= testTotalMass(galaxymodel::GalaxyModel(potH, actH, df::DoublePowerLaw(paramDPL)), 1.);


    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
