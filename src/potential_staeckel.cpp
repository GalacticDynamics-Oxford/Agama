#include "potential_staeckel.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential{

    StaeckelOblatePerfectEllipsoid::StaeckelOblatePerfectEllipsoid
        (double _mass, double major_axis, double minor_axis) :
        mass(_mass), coordSys(-pow_2(major_axis), -pow_2(minor_axis))
    {
        if(minor_axis<=0 || minor_axis>=major_axis)
            throw std::runtime_error("Error in StaeckelOblatePerfectEllipsoid: "
                "minor axis must be positive and strictly smaller than major axis");
    }

    void StaeckelOblatePerfectEllipsoid::evaluate(const coord::PosProlSph& pos,
        double* value, coord::GradProlSph* deriv, coord::HessProlSph* deriv2) const
    {
        assert(&(pos.coordsys)==&coordSys);  // make sure we're not bullshited
        double lmn = pos.lambda-pos.nu;
        double lpg = pos.lambda+coordSys.gamma;
        double npg = pos.nu+coordSys.gamma;
        if(npg<0 || lpg<=npg)
            throw std::runtime_error("Error in StaeckelOblatePerfectEllipsoid: "
                "incorrect values of spheroidal coordinates");
        double Glambda, dGdlambda, d2Gdlambda2, Gnu, dGdnu, d2Gdnu2;
        // values and derivatives of G(lambda) and G(nu)
        Glambda = eval_G(pos.lambda, &dGdlambda, &d2Gdlambda2);
        Gnu     = eval_G(pos.nu, &dGdnu, &d2Gdnu2);
        double coef=(Glambda-Gnu)/pow_2(lmn);  // common subexpression
        if(value!=NULL) 
            *value = (npg*Gnu - lpg*Glambda) / lmn;
        if(deriv!=NULL) {
            deriv->dlambda =  coef*npg - dGdlambda*lpg/lmn;
            deriv->dnu     = -coef*lpg + dGdnu*npg/lmn;
        }
        if(deriv2!=NULL) {
            deriv2->dlambda2   = (2*npg*(-coef + dGdlambda/lmn) - d2Gdlambda2*lpg) / lmn;
            deriv2->dnu2       = (2*lpg*(-coef + dGdnu    /lmn) + d2Gdnu2    *npg) / lmn;
            deriv2->dlambdadnu = ((lpg+npg)*coef - (lpg*dGdlambda+npg*dGdnu)/lmn ) / lmn;
        }
    }

    double StaeckelOblatePerfectEllipsoid::eval_G(double tau, double* deriv, double* deriv2) const
    {
        // G is defined by eq.27 in de Zeeuw(1985)
        double sqmg = sqrt(-coordSys.gamma), tpg=tau+coordSys.gamma;
        assert(tpg>=0);
        if(tpg==0) {  // handling a special case
            if(deriv!=NULL)  *deriv = mass*2./(3*M_PI)*sqmg/coordSys.gamma;
            if(deriv2!=NULL) *deriv2= mass*4./(5*M_PI)*sqmg/pow_2(coordSys.gamma);
            return mass*2./M_PI/sqmg;
        } else {
            double sqtpg = sqrt(tpg);
            double arct = atan(sqtpg/sqmg)/sqtpg;
            if(deriv!=NULL)  *deriv = mass  / M_PI * (sqmg/tau-arct)/tpg;
            if(deriv2!=NULL) *deriv2= mass*.5/M_PI * (-sqmg*(3*tau+2*tpg)/pow_2(tau) + 3*arct)/pow_2(tpg);
            return mass*2./M_PI * arct;
        }
    }

}  // namespace potential