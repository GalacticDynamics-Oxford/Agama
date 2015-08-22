#include <iostream>
#include "potential_dehnen.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "galaxymodel.h"
#include "particles_io.h"
#include "math_specfunc.h"
#include "debug_utils.h"

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

int main(){
    bool ok = true;
    /// Create an instance of distribution function
    double inner_slope = 1.0; // inner density slope
    //double outer_slope = 4.0; // outer density slope
    double jcore = 0.;
    // NB: parameters obtained by fitting (test_df_fit.cpp)
    double alpha = 1.407;  // orig: (6-inner_slope) / (4-inner_slope);
    double beta  = 5.628;  // orig: 2 * outer_slope - 3;
    double h0    = 1.745;
    double ar    = 1.614;
    double az    = 0.693;
    double aphi  = az;
    double br    = 1.0;
    double bz    = 1.0;
    double bphi  = 1.0;
    const df::DoublePowerLawParam params={jcore,alpha,beta,h0,ar,az,aphi,br,bz,bphi};
    const df::DoublePowerLaw dpl(params);                     // distribution function
    const potential::Dehnen pot(1., 1., 1., 1., inner_slope); // potential
    const actions::ActionFinderAxisymFudge actf(pot);         // action finder
    const galaxymodel::GalaxyModel galmod(pot, actf, dpl);    // all together - the mighty triad

    // Calculate the total mass
    const double reqRelError = 1.e-04;
    const int maxNumEval = 1e05;
    double err;
    int numEval;
    double mass = dpl.totalMass(reqRelError, maxNumEval, &err, &numEval);
    // the analytic value of total mass for the case ar=az=aphi=br=bz=bphi=1 and jcore=0
    double mass_exact = pow_3(2*M_PI) * math::gamma(3-alpha) * math::gamma(beta-3) / math::gamma(beta-alpha);
    ok &= math::fcmp(mass, mass_exact, 0.05) == 0; // 5% relative error allowed because ar!=az

    std::cout <<
        "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations"
        " (analytic value=" << mass_exact << ")\n";

    // compare standard f(E) for the Hernquist model and the action-based df f(J),
    // normalized by total mass equal to unity
    coord::PosVelCyl point(.0, 1, 0, 0, 0, 0.5);  // some point
    const actions::Actions J = actf.actions(point);
    double energy = totalEnergy(pot, point);
    double df_value_J = dpl.value(J)/mass;   // raw value of f(J) divided by normalizing constant
    double df_value_E = dfHernquist(1, 1, energy);  // f(E) for the Hernquist model with unit total mass
    std::cout <<
        "f(J)=" << df_value_J << " for actions " << J << "\n"
        "f(E)=" << df_value_E << " for energy E=" << energy << "\n";
    ok &= math::fcmp(df_value_J, df_value_E, 0.05) == 0;

    // compute density and velocity moments; density again needs to be normalized by total mass of DF
    double density, densityErr;
    coord::VelCyl velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    computeMoments(galmod, point, reqRelError, maxNumEval,
        &density, &velocityFirstMoment, &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr);
    double dens_an  = pot.density(point);  // analytical value of density
    double sigma_an = sigmaHernquist(1, 1, coord::toPosSph(point).r);  // analytical value of sigma^2
    std::cout << "At point " << point << "we have "
        "density=" << (density/mass) << " +- " << (densityErr/mass) << 
        "  compared to analytic value " << dens_an << "\n"
        "velocity  vR=" << velocityFirstMoment.vR << " +- " << velocityFirstMomentErr.vR <<
        ", vz=" << velocityFirstMoment.vz << " +- " << velocityFirstMomentErr.vz <<
        ", vphi=" << velocityFirstMoment.vphi << " +- " << velocityFirstMomentErr.vphi << "\n"
        "2nd moment of velocity  vR2=" << velocitySecondMoment.vR2 << " +- " << velocitySecondMomentErr.vR2 <<
        ", vz2="    << velocitySecondMoment.vz2    << " +- " << velocitySecondMomentErr.vz2 <<
        ", vphi2="  << velocitySecondMoment.vphi2  << " +- " << velocitySecondMomentErr.vphi2 <<
        ", vRvz="   << velocitySecondMoment.vRvz   << " +- " << velocitySecondMomentErr.vRvz <<
        ", vRvphi=" << velocitySecondMoment.vRvphi << " +- " << velocitySecondMomentErr.vRvphi <<
        ", vzvphi=" << velocitySecondMoment.vzvphi << " +- " << velocitySecondMomentErr.vzvphi <<
        "   compared to analytic value " << sigma_an << "\n";
    ok &= math::fcmp(density/mass, dens_an, 0.02) == 0 &&
          math::fcmp(velocitySecondMoment.vR2,   sigma_an, 0.01) == 0 &&
          math::fcmp(velocitySecondMoment.vz2,   sigma_an, 0.01) == 0 &&
          math::fcmp(velocitySecondMoment.vphi2, sigma_an, 0.01) == 0;

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
#if 0
    particles::PointMassArrayCar points;
    generatePosVelSamples(galmod, 1e5, points);
    particles::BaseIOSnapshot* snap = particles::createIOSnapshotWrite(
        "Text", "sampled_actions.txt", units::ExternalUnits());
    snap->writeSnapshot(points);
    delete snap;
#endif
    return 0;
}
