#include "potential_perfect_ellipsoid.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential{

OblatePerfectEllipsoid::OblatePerfectEllipsoid
    (double _mass, double major_axis, double minor_axis) :
    mass(_mass), coordSys(sqrt(pow_2(major_axis)-pow_2(minor_axis))), minorAxis(minor_axis)
{
    if(minor_axis<=0 || minor_axis>=major_axis)
        throw std::invalid_argument("Error in OblatePerfectEllipsoid: "
            "minor axis must be positive and strictly smaller than major axis");
}

void OblatePerfectEllipsoid::evalScalar(const coord::PosProlSph& pos,
    double* val, coord::GradProlSph* deriv, coord::HessProlSph* deriv2) const
{
    assert(&(pos.coordsys)==&coordSys);  // make sure we're not bullshited
    double absnu = fabs(pos.nu);
    double signu = pos.nu>=0 ? 1 : -1;
    double lmn = pos.lambda-absnu;
    if(absnu>coordSys.Delta2 || pos.lambda<coordSys.Delta2)
        throw std::invalid_argument("Error in OblatePerfectEllipsoid: "
            "incorrect values of spheroidal coordinates");
    if(!(pos.lambda < 1e100 && absnu < 1e100)) {
        if(val)
            *val = 0;
        if(deriv)
            deriv->dlambda = deriv->dnu = deriv->dphi = 0;
        if(deriv2)
            deriv2->dlambda2 = deriv2->dnu2 = deriv2->dlambdadnu = 0;
        return;
    }
    double Glambda, dGdlambda, d2Gdlambda2, Gnu, dGdnu, d2Gdnu2;
    // values and derivatives of G(lambda) and G(|nu|)
    evalDeriv(pos.lambda, &Glambda, &dGdlambda, &d2Gdlambda2);
    evalDeriv(absnu,      &Gnu,     &dGdnu,     &d2Gdnu2);
    double coef=(Glambda-Gnu)/pow_2(lmn);  // common subexpression
    if(val!=NULL) 
        *val = (absnu*Gnu - pos.lambda*Glambda) / lmn;
    if(deriv!=NULL) {
        deriv->dlambda =   coef*absnu      - dGdlambda*pos.lambda / lmn;
        deriv->dnu     = (-coef*pos.lambda + dGdnu*absnu          / lmn) * signu;
        deriv->dphi    = 0;
    }
    if(deriv2!=NULL) {
        deriv2->dlambda2   = (2*absnu     *(-coef + dGdlambda/lmn) - d2Gdlambda2*pos.lambda) / lmn;
        deriv2->dnu2       = (2*pos.lambda*(-coef + dGdnu    /lmn) + d2Gdnu2    *absnu     ) / lmn;
        deriv2->dlambdadnu = ((pos.lambda+absnu)*coef - (pos.lambda*dGdlambda + absnu*dGdnu) / lmn ) / lmn * signu;
    }
}

void OblatePerfectEllipsoid::evalDeriv(double tau, double* G, double* deriv, double* deriv2) const
{
    // G is defined by eq.27 in de Zeeuw(1985), except that we use 
    // tau = {tau_deZeeuw}+{gamma_deZeeuw}, which ranges from 0 to inf.
    if(tau<0)
        throw std::invalid_argument("Error in OblatePerfectEllipsoid: "
            "incorrect value of tau");
    double c2   = pow_2(minorAxis);
    double fac  = mass/minorAxis*(2./M_PI);
    double tauc = tau/c2;
    double arct = 1;  // value for the limiting case tau==0
    if(tauc > 1e-16) {
        double sqtc = sqrt(tauc);
        arct = atan(sqtc)/sqtc;
    }
    if(G)
        *G = fac * arct;
    if(deriv)
        *deriv = tauc > 1e-8 ?
            fac * 0.5 * (1 / (1+tauc) - arct) / tau :
            fac * (-1./3 + 2./5 * tauc) / c2;    // asymptotic expansion for tau->0
    if(deriv2!=NULL)
        *deriv2 = tauc > 1e-5 ?
            fac * 0.75 * (arct - (1+(5./3)*tauc) / pow_2(1+tauc)) / pow_2(tau) :
            fac * (2./5 - 6./7 * tauc) / pow_2(c2);
}

}  // namespace potential