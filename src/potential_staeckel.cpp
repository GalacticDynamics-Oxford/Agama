#include "potential_staeckel.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential{

StaeckelOblatePerfectEllipsoid::StaeckelOblatePerfectEllipsoid
    (double _mass, double major_axis, double minor_axis) :
    mass(_mass), coordSys(pow_2(major_axis)-pow_2(minor_axis)), minorAxis(minor_axis)
{
    if(minor_axis<=0 || minor_axis>=major_axis)
        throw std::invalid_argument("Error in StaeckelOblatePerfectEllipsoid: "
            "minor axis must be positive and strictly smaller than major axis");
}

void StaeckelOblatePerfectEllipsoid::evalScalar(const coord::PosProlSph& pos,
    double* val, coord::GradProlSph* deriv, coord::HessProlSph* deriv2) const
{
    assert(&(pos.coordsys)==&coordSys);  // make sure we're not bullshited
    double absnu = fabs(pos.nu);
    double signu = pos.nu>0 ? 1 : -1;
    double lmn = pos.lambda-absnu;
    if(absnu>coordSys.delta || pos.lambda<coordSys.delta)
        throw std::invalid_argument("Error in StaeckelOblatePerfectEllipsoid: "
            "incorrect values of spheroidal coordinates");
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

void StaeckelOblatePerfectEllipsoid::evalDeriv(double tau, double* G, double* deriv, double* deriv2) const
{
    // G is defined by eq.27 in de Zeeuw(1985), except that we use 
    // tau = {tau_deZeeuw}+{gamma_deZeeuw}, which ranges from 0 to inf.
    double c2 = pow_2(minorAxis);
    if(tau<0)
        throw std::invalid_argument("Error in StaeckelOblatePerfectEllipsoid: "
            "incorrect value of tau");
    if(tau==0) {  // handling a special case
        double val = mass*2./M_PI/minorAxis;
        if(G     !=NULL) *G     = val;
        if(deriv !=NULL) *deriv = val*(-1./3)/c2;
        if(deriv2!=NULL) *deriv2= val*( 2./5)/pow_2(c2);
    } else {
        double sqrttau = sqrt(tau);
        double arct = atan(sqrttau/minorAxis)/sqrttau;
        if(G     !=NULL)
            *G     = mass*2./M_PI * arct;
        if(deriv !=NULL)
            *deriv = mass*1./M_PI * (minorAxis / (tau+c2) - arct) / tau;
        if(deriv2!=NULL)
            *deriv2= mass*.5/M_PI * (-minorAxis * (5*tau+3*c2) / pow_2(tau+c2) + 3*arct) / pow_2(tau);
    }
}

}  // namespace potential