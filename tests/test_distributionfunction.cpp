/** \file    test_distributionfunction.cpp
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
#include "actions_staeckel.h"
#include "df_halo.h"
#include "galaxymodel.h"
#include "particles_io.h"
#include "debug_utils.h"

const double reqRelError = 1e-4;
const int maxNumEval = 1e5;
const char* errmsg = "\033[1;31m **\033[0m";

bool testTotalMass(const galaxymodel::GalaxyModel& galmod, double massExact)
{
    std::cout << "\033[1;33mTesting " << galmod.potential.name() << "\033[0m\n";
    // Calculate the total mass
    double err;
    int numEval;
    double mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval, &err, &numEval);
    if(err > mass*0.01) {
        std::cout << "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations\n";
        mass = galmod.distrFunc.totalMass(reqRelError, maxNumEval*10, &err, &numEval);
    }
    bool ok = math::fcmp(mass, massExact, 0.05) == 0; // 5% relative error allowed because ar!=az
    std::cout <<
        "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations"
        " (analytic value=" << massExact << (ok?"":errmsg) <<")\n";
    return ok;
}

bool testDFmoments(const galaxymodel::GalaxyModel& galmod, const coord::PosVelCyl& point,
    double dfExact, double densExact, double sigmaExact)
{
    std::cout << "\033[1mAt point " << point << "\033[0m we have\n";
    // compare the action-based distribution function f(J) with the analytic one
    const actions::Actions J = galmod.actFinder.actions(point);
    double dfValue = galmod.distrFunc.value(J);
    double energy = totalEnergy(galmod.potential, point);
    bool dfok = math::fcmp(dfValue, dfExact, 0.05) == 0;
    std::cout <<
        "f(J)=" << dfValue << " for actions " << J << "\n"
        "f(E)=" << dfExact << " for energy E=" << energy << (dfok?"":errmsg) <<"\n";

    // compute density and velocity moments
    double density, densityErr;
    coord::VelCyl velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    computeMoments(galmod, point, reqRelError, maxNumEval,
        &density, &velocityFirstMoment, &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr);
    bool densok = math::fcmp(density, densExact, 0.05) == 0;
    bool sigmaok =
        math::fcmp(velocitySecondMoment.vR2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vz2,   sigmaExact, 0.05) == 0 &&
        math::fcmp(velocitySecondMoment.vphi2, sigmaExact, 0.05) == 0;

    std::cout << 
        "density=" << density << " +- " << densityErr << 
        "  compared to analytic value " << densExact << (densok?"":errmsg) <<"\n"
        "velocity"
        "  vR=" << velocityFirstMoment.vR << " +- " << velocityFirstMomentErr.vR <<
        ", vz=" << velocityFirstMoment.vz << " +- " << velocityFirstMomentErr.vz <<
        ", vphi=" << velocityFirstMoment.vphi << " +- " << velocityFirstMomentErr.vphi << "\n"
        "2nd moment of velocity"
        "  vR2="    << velocitySecondMoment.vR2    << " +- " << velocitySecondMomentErr.vR2 <<
        ", vz2="    << velocitySecondMoment.vz2    << " +- " << velocitySecondMomentErr.vz2 <<
        ", vphi2="  << velocitySecondMoment.vphi2  << " +- " << velocitySecondMomentErr.vphi2 <<
        ", vRvz="   << velocitySecondMoment.vRvz   << " +- " << velocitySecondMomentErr.vRvz <<
        ", vRvphi=" << velocitySecondMoment.vRvphi << " +- " << velocitySecondMomentErr.vRvphi <<
        ", vzvphi=" << velocitySecondMoment.vzvphi << " +- " << velocitySecondMomentErr.vzvphi <<
        "   compared to analytic value " << sigmaExact << (sigmaok?"":errmsg) <<"\n";
    return dfok && densok && sigmaok;
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

    // test double-power-law distribution function in a spherical Hernquist potential
    // NB: parameters obtained by fitting (test_df_fit.cpp)
    double norm  = 0.956;
    double alpha = 1.407;
    double beta  = 5.628;
    double j0    = 1.745;
    double jcore = 0.;
    double ar    = 1.614;
    double az    = (3-ar)/2;
    double aphi  = az;
    double br    = 1.0;
    double bz    = 1.0;
    double bphi  = 1.0;
    const df::DoublePowerLawParam paramDPL = {norm,j0,jcore,alpha,beta,ar,az,aphi,br,bz,bphi};
    const potential::Dehnen potH(1., 1., 1., 1., 1.);        // potential
    const actions::ActionFinderAxisymFudge actH(potH);       // action finder
    const df::DoublePowerLaw dfH(paramDPL);                  // distribution function
    const galaxymodel::GalaxyModel galmodH(potH, actH, dfH); // all together - the mighty triad

    // the analytic value of total mass for the case ar=az=aphi=br=bz=bphi=1 and jcore=0 is just 'norm'
    ok &= testTotalMass(galmodH, norm);

    for(int i=0; i<NUM_POINTS_H; i++) {
        const coord::PosVelCyl point(testPointsH[i]);
        double dfExact    = dfHernquist(1, 1, totalEnergy(potH, point));     // f(E) for the Hernquist model
        double densExact  = potH.density(point);                             // analytical value of density
        double sigmaExact = sigmaHernquist(1, 1, coord::toPosSph(point).r);  // analytical value of sigma^2
        ok &= testDFmoments(galmodH, point, dfExact, densExact, sigmaExact);
    }

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
