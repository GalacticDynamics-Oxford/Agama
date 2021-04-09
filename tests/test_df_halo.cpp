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

const char* errmsg = "\033[1;31m **\033[0m";
std::string histograms;

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
    double dens;
    coord::VelCar  vel;
    coord::Vel2Car vel2;
    computeMoments(galmod, toPosCar(point), &dens, &vel, &vel2);
    bool densok = math::fcmp(dens, densExact, 0.01) == 0;
    bool sigmaok =
        math::fcmp(vel2.vx2, sigmaExact, 0.05) == 0 &&
        math::fcmp(vel2.vy2, sigmaExact, 0.05) == 0 &&
        math::fcmp(vel2.vz2, sigmaExact, 0.05) == 0;

    // compute velocity distributions
    std::vector<double> gridv, amplvX, amplvY, amplvZ;  // will be initialized by the routine
    double densvdf;
    const int ORDER = 3, GRIDSIZE = 41;
    galaxymodel::computeVelocityDistribution<ORDER>(galmod, toPosCar(point),
        GRIDSIZE, /*output*/gridv, &densvdf, &amplvX, &amplvY, &amplvZ);
    math::BsplineInterpolator1d<ORDER> interp(gridv);
    double vmax = interp.xmax();
    // output the profiles
    for(int i=-100; i<=100; i++) {
        double v = i*vmax/100;
        histograms +=
        utils::pp(v, 9)+'\t'+
        utils::pp(interp.interpolate(v, amplvX), 9)+' '+
        utils::pp(interp.interpolate(v, amplvY), 9)+' '+
        utils::pp(interp.interpolate(v, amplvZ), 9)+'\n';
    }
    histograms+='\n';
    // compute the dispersions from the VDF (integrate with the weight factor v^2)
    double sigmaX = interp.integrate(-vmax, vmax, amplvX, 2 /*power index for the weighting*/);
    double sigmaY = interp.integrate(-vmax, vmax, amplvY, 2);
    double sigmaZ = interp.integrate(-vmax, vmax, amplvZ, 2);
    // they should agree with the velocity moments computed above
    bool densvdfok = math::fcmp(densvdf, dens, 5e-5) == 0;
    bool sigmavdfok=
        math::fcmp(sigmaX, vel2.vx2, 1e-4) == 0 &&
        math::fcmp(sigmaY, vel2.vy2, 1e-4) == 0 &&
        math::fcmp(sigmaZ, vel2.vz2, 1e-4) == 0;

    std::cout <<
        "density=" << dens << " = " << densvdf << (densvdfok?"":errmsg) <<
        "  compared to analytic value " << densExact << (densok?"":errmsg) <<"\n"
        "mean velocity  vphi=" << vel.vy << "\n"
        "velocity dispersion"
        "  vx2=" << vel2.vx2 << " = " << sigmaX <<
        ", vy2=" << vel2.vy2 << " = " << sigmaY <<
        ", vz2=" << vel2.vz2 << " = " << sigmaZ <<
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
    if(!ok) std::cout << "Scaling transformation in action space failed" << errmsg <<"\n";

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
    paramDPL.norm      = 1.0;
    paramDPL.rotFrac   = 0.5;  // add some rotation (odd-Jphi component),
    paramDPL.Jphi0     = 0.7;  // it shouldn't affect the even-order moments
    paramDPL.norm /= df::DoublePowerLaw(paramDPL).totalMass(); // normalize the total mass to unity
    const potential::Dehnen potH(1., 1., 1., 1., 1.);          // potential
    const actions::ActionFinderSpherical actH(potH);           // action finder
    const df::DoublePowerLaw dfH(paramDPL);                    // distribution function
    const galaxymodel::GalaxyModel galmodH(potH, actH, dfH);   // all together - the mighty triad

    for(int i=0; i<NUM_POINTS_H; i++) {
        const coord::PosVelCyl point(testPointsH[i]);
        double dfExact    = dfHernquist(1, 1, totalEnergy(potH, point));     // f(E) for the Hernquist model
        double densExact  = potH.density(point);                             // analytical value of density
        double sigmaExact = sigmaHernquist(1, 1, coord::toPosSph(point).r);  // analytical value of sigma^2
        ok &= testDFmoments(galmodH, point, dfExact, densExact, sigmaExact);
    }

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("test_df_halo.dat");
        strm << "#v       \tf(v_x)    f(v_y)    f(v_z)\n";
        strm << histograms;
    }

    // test the model with a central core: the introduction of a core shouldn't change
    // the overall normalization, at least when the coefficients in the linear combination
    // of actions are the same in the inner and outer part of the halo, and there is no exp-cutoff
    std::cout << "\033[1mTesting total mass in cored vs. non-cored models\033[0m\n";
    paramDPL.coefJrOut = paramDPL.coefJrIn;
    paramDPL.coefJzOut = paramDPL.coefJzIn;
    // first normalize the mass of a non-cored model to unity
    paramDPL.norm /= df::DoublePowerLaw(paramDPL).totalMass();
    // set up a sizable core, create another DF and repeat the mass computation
    paramDPL.Jcore = 0.8;
    double massCore = df::DoublePowerLaw(paramDPL).totalMass();
    std::cout << "Mass of a cored model: " << utils::pp(massCore,8) << ", non-cored: 1";
    if(math::fcmp(massCore, 1.0, 1e-5)!=0) {
        std::cout << ", difference: " << utils::toString(massCore-1., 3) << errmsg << "\n";
        ok = false;
    } else
        std::cout << "\n";

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
