#include "potential_staeckel.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential{

    StaeckelOblatePerfectEllipsoid::StaeckelOblatePerfectEllipsoid
        (double _mass, double major_axis, double minor_axis) :
        mass(_mass), CS(-major_axis*major_axis, -minor_axis*minor_axis)
    {
        if(minor_axis>=major_axis)
            throw std::runtime_error("Error in StaeckelOblatePerfectEllipsoid: "
                "minor axis must be strictly smaller than major axis");
    }

    void StaeckelOblatePerfectEllipsoid::eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
    {
        coord::PosDerivProlSph coordDerivs;
        coord::PosDeriv2ProlSph coordDerivs2;
        const coord::PosProlSph coords = coord::toPosProlSph(pos, CS, 
            deriv!=NULL || deriv2!=NULL ? &coordDerivs : NULL,
            deriv2!=NULL ? &coordDerivs2 : NULL);   // compute only necessary derivatives
        double lmn = coords.lambda-coords.nu;
        double lpg = coords.lambda+CS.gamma;
        double npg = coords.nu+CS.gamma;
        if(npg<0 || lpg<=npg)
            throw std::runtime_error("Error in StaeckelOblatePerfectEllipsoid: "
                "incorrect values of spheroidal coordinates");
        double Glambda, dGdlambda, d2Gdlambda2, Gnu, dGdnu, d2Gdnu2;
        // values and derivatives of G(lambda) and G(nu)
        Glambda = eval_G(coords.lambda, &dGdlambda, &d2Gdlambda2);
        Gnu     = eval_G(coords.nu, &dGdnu, &d2Gdnu2);
        if(potential!=NULL) {
            *potential = (npg*Gnu - lpg*Glambda) / lmn;
        }
        if(deriv==NULL && deriv2==NULL) return;

        double coef=(Glambda-Gnu)/pow_2(lmn);  // common subexpression
        double dPhidlambda =  coef*npg - dGdlambda*lpg/lmn;
        double dPhidnu     = -coef*lpg + dGdnu*npg/lmn;
        if(deriv!=NULL) {
            double dPhidR = dPhidlambda*coordDerivs.dlambdadR + dPhidnu*coordDerivs.dnudR;
            double dPhidz = dPhidlambda*coordDerivs.dlambdadz + dPhidnu*coordDerivs.dnudz;
            deriv->dR     = dPhidR;
            deriv->dz     = dPhidz;
            deriv->dphi   = 0;
        }
        if(deriv2!=NULL) {
            double d2Phidlambda2   = (2*npg*(-coef + dGdlambda/lmn) - d2Gdlambda2*lpg) / lmn;
            double d2Phidnu2       = (2*lpg*(-coef + dGdnu    /lmn) + d2Gdnu2    *npg) / lmn;
            double d2Phidlambdadnu = ((lpg+npg)*coef - (lpg*dGdlambda+npg*dGdnu)/lmn ) / lmn;
            deriv2->dR2 = 
                d2Phidlambda2   * pow_2(coordDerivs.dlambdadR) + 
                d2Phidnu2       * pow_2(coordDerivs.dnudR) +
                d2Phidlambdadnu * 2*coordDerivs.dlambdadR*coordDerivs.dnudR +
                dPhidlambda     * coordDerivs2.d2lambdadR2 +
                dPhidnu         * coordDerivs2.d2nudR2;
            deriv2->dz2 = 
                d2Phidlambda2   * pow_2(coordDerivs.dlambdadz) + 
                d2Phidnu2       * pow_2(coordDerivs.dnudz) +
                d2Phidlambdadnu * 2*coordDerivs.dlambdadz*coordDerivs.dnudz +
                dPhidlambda     * coordDerivs2.d2lambdadz2 +
                dPhidnu         * coordDerivs2.d2nudz2;
            deriv2->dRdz = 
                d2Phidlambda2   * coordDerivs.dlambdadR*coordDerivs.dlambdadz + 
                d2Phidnu2       * coordDerivs.dnudR    *coordDerivs.dnudz +
                d2Phidlambdadnu *(coordDerivs.dlambdadR*coordDerivs.dnudz + coordDerivs.dlambdadz*coordDerivs.dnudR) +
                dPhidlambda     * coordDerivs2.d2lambdadRdz +
                dPhidnu         * coordDerivs2.d2nudRdz;
            deriv2->dRdphi = deriv2->dzdphi = deriv2->dphi2 = 0;
        }
    }

    double StaeckelOblatePerfectEllipsoid::eval_G(double tau, double* deriv, double* deriv2) const
    {
        // G is defined by eq.27 in de Zeeuw(1985)
        double sqmg = sqrt(-CS.gamma), tpg=tau+CS.gamma;
        assert(tpg>=0);
        if(tpg==0) {  // handling a special case
            if(deriv!=NULL)  *deriv = mass*2./(3*M_PI)*sqmg/CS.gamma;
            if(deriv2!=NULL) *deriv2= mass*4./(5*M_PI)*sqmg/pow_2(CS.gamma);
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