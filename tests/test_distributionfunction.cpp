#include <iostream>
#include "potential_dehnen.h"
#include "actions_staeckel.h"
#include "df_halo.h"
#include "galaxymodel.h"
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
    /// Create an instance of distribution function
    double inner_slope = 1.0; // inner density slope
    double outer_slope = 4.0; // outer density slope
    double jcore = 0.;
    double alpha = (6-inner_slope) / (4-inner_slope);
    double beta  = 2 * outer_slope - 3;
    double h0    = 1.0;
    double ar    = 1.0;
    double az    = 1.;//0.5;
    double aphi  = 1.;//0.5;
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
    std::cout <<
        "Mass=" << mass << " +- " << err << " in " << numEval << " DF evaluations"
        " (analytic value=" << mass_exact << ")\n";

    // compare standard f(E) for the Hernquist model and the action-based df f(J),
    // normalized by total mass equal to unity
    coord::PosVelCyl point(6, 6, 0, 0, 0, 0.1);  // some point
    const actions::Actions J = actf.actions(point);
    double energy = totalEnergy(pot, point);
    std::cout <<
        "f(J)=" << (dpl.value(J)/mass) << " for actions " << J << "\n"
        "f(E)=" << dfHernquist(1, 1, energy) << " for energy E=" << energy << "\n";

    // compute density and velocity moments; density again needs to be normalized by total mass of DF
    double density, densityErr;
    coord::VelCyl velocityFirstMoment, velocityFirstMomentErr;
    coord::Vel2Cyl velocitySecondMoment, velocitySecondMomentErr;
    galmod.computeMoments(point, reqRelError, maxNumEval,
        &density, &velocityFirstMoment, &velocitySecondMoment,
        &densityErr, &velocityFirstMomentErr, &velocitySecondMomentErr);

    std::cout << "At point " << point << "we have "
        "density=" << (density/mass) << " +- " << (densityErr/mass) << 
        "  compared to analytic value " << pot.density(point) << "\n"
        "velocity  vR=" << velocityFirstMoment.vR << " +- " << velocityFirstMomentErr.vR <<
        ", vz=" << velocityFirstMoment.vz << " +- " << velocityFirstMomentErr.vz <<
        ", vphi=" << velocityFirstMoment.vphi << " +- " << velocityFirstMomentErr.vphi << "\n"
        "2nd moment of velocity  vR2=" << velocitySecondMoment.vR2 << " +- " << velocitySecondMomentErr.vR2 <<
        ", vz2="    << velocitySecondMoment.vz2    << " +- " << velocitySecondMomentErr.vz2 <<
        ", vphi2="  << velocitySecondMoment.vphi2  << " +- " << velocitySecondMomentErr.vphi2 <<
        ", vRvz="   << velocitySecondMoment.vRvz   << " +- " << velocitySecondMomentErr.vRvz <<
        ", vRvphi=" << velocitySecondMoment.vRvphi << " +- " << velocitySecondMomentErr.vRvphi <<
        ", vzvphi=" << velocitySecondMoment.vzvphi << " +- " << velocitySecondMomentErr.vzvphi <<
        "   compared to analytic value " << sigmaHernquist(1, 1, coord::toPosSph(point).r) << "\n";
    return 0;
}
