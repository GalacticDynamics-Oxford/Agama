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
    double* val, coord::GradProlSph* deriv, coord::HessProlSph* deriv2, double /*time*/) const
{
    assert(pos.coordsys.Delta2 == coordSys.Delta2);  // make sure we're not bullshited
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


void PerfectEllipsoid::evalAxi(const coord::PosAxi& pos,
    double* value, coord::GradAxi* deriv, coord::HessAxi* deriv2) const
{
    double psi, M, Fv,
        norm  = -2/M_PI * mass,
        B     = scaleRadiusB,
        D2    = pos.cs.Delta2,
        D     = sqrt(fabs(D2)),
        sinnu = 1 / sqrt(1 + pow_2(pos.cotnu)),
        cosnu = sinnu!=0 ? pos.cotnu * sinnu : (pos.cotnu>0 ? 1 : -1);
    if(D2 >= 0) {  // oblate or spherical
        psi = sqrt(pow_2(pos.rho) + D2);
        M   = 1 / (pow_2(pos.rho) + D2 * pow_2(sinnu));
        Fv  = -D * atan(D / B * cosnu);
    } else {  // prolate
        psi = pos.rho;
        M   = 1 / (pow_2(pos.rho) - D2 * pow_2(cosnu));
        Fv  = D * atanh(D / B * cosnu);
    }
    double Fu = atan(psi / B);
    // Phi = norm * M(rho,nu) * [Grho(rho) + Gnu(nu)]
    double Phi = norm * M * (psi * Fu + cosnu * Fv);
    if(value)
        *value = psi == INFINITY ? -0.0 : Phi;
    if(!deriv && !deriv2)
        return;
    double
    dpsidrho   = D2 <= 0 ? 1 : pos.rho / psi,
    dcosnudnu  = -sinnu,
    dFudpsi    =  B / (pow_2(B) + pow_2(psi)),
    dFvdcosnu  = -B / (pow_2(B) + pow_2(cosnu) * D2),  // actually this is dFv/dcosv / D^2
    dGrhodpsi  = Fu + dFudpsi   * psi,
    dGnudcosnu = Fv + dFvdcosnu * cosnu * D2,
    dGrho      = dGrhodpsi  * dpsidrho,
    dGnu       = dGnudcosnu * dcosnudnu,
    dlnMdrho   =-2 * M * pos.rho,
    dlnMdnu    = 2 * M * cosnu * D2 * dcosnudnu;
    if(deriv) {
        deriv->drho = Phi * dlnMdrho + norm * M * dGrho;
        deriv->dnu  = Phi * dlnMdnu  + norm * M * dGnu;
        deriv->dphi = 0;
    }
    if(deriv2) {
        double
        d2psidrho2  = D2 <= 0 ? 0 : D2 / pow_3(psi),
        d2cosnudnu2 = -cosnu,
        d2Grho = dGrhodpsi  * d2psidrho2  + 2 * B * pow_2(dFudpsi   * dpsidrho),
        d2Gnu  = dGnudcosnu * d2cosnudnu2 - 2 * B * pow_2(dFvdcosnu * dcosnudnu) * D2,
        d2Mdrho2_over_M = 2 * pow_2(dlnMdrho) - 2 * M,
        d2Mdnu2_over_M  = 2 * pow_2(dlnMdnu)  + 2 * M * D2 * (pow_2(dcosnudnu) + cosnu * d2cosnudnu2),
        d2Mdrhodnu_over_M = 2 * dlnMdrho * dlnMdnu;
        deriv2->drho2   = Phi * d2Mdrho2_over_M   + norm * M * (2 * dlnMdrho * dGrho + d2Grho);
        deriv2->dnu2    = Phi * d2Mdnu2_over_M    + norm * M * (2 * dlnMdnu  * dGnu  + d2Gnu);
        deriv2->drhodnu = Phi * d2Mdrhodnu_over_M + norm * M * (dlnMdrho * dGnu + dlnMdnu * dGrho);
    }
}


void PerfectEllipsoid::evalCyl(const coord::PosCyl &pos,
    double* value, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const
{
    double D2 = pow_2(scaleRadiusA) - pow_2(scaleRadiusB);
    double D  = sqrt(fabs(D2)), A = scaleRadiusA, B = scaleRadiusB;
    // offset along the direction between origin and the focal point (away from origin)
    double p = (D2<=0 ? pos.R : fabs(pos.z)) - D;
    double signp = (D<=0 || pos.z>=0 ? 1 : -1);  // flip the sign of derivative if D2>0 and z<0
    // offset in the perpendicular direction
    double q = (D2<=0 ? pos.z : pos.R);
    // extent of the region where the Taylor expansion is more accurate than the actual potential:
    // ~DBL_EPSILON^(1/5) for the oblate potential (prolate spheroidal coordinates, D2>0),
    // ~DBL_EPSILON^(1/4) for the prolate potential (oblate spheroidal coordinates, D2<0).
    // the expansion itself has roughly the same accuracy in both cases, but the actual potential
    // suffers more severely from cancellation in the case D2>0, extending the region further out.
    double eps = D2<=0 ? 1e-4 : 6e-4;
    if(pow_2(p) + pow_2(q) < pow_2(fmax(A, B) * eps)) {
        // accurate treatment around focal points, expanding Phi to third order in dR,dz
        double K0, K1, K2, K3;  // coefs of Taylor expansion of the potential
        if(D2 <= 0) {
            K0  = 1/B;
            K1  = -1./3 * K0 / pow_2(B);
            K2  = -3./5 * K1 / pow_2(B);
            K3  = -5./7 * K2 / pow_2(B);
        } else {
            K0  = 0.5 * (atan(D / B) / D + B / pow_2(A));
            K1  = (-1./4 * K0 + 1./4 * pow_3(B) / pow_2(pow_2(A)) ) / D2;
            K2  = (-3./6 * K1 - 1./6 * pow_3(B/pow_2(A))          ) / D2;
            K3  = (-5./8 * K2 + 1./8 * pow_3(B/pow_2(A))/pow_2(A) ) / D2;
        }
        double norm = -2/M_PI * mass,
        Kp  = 2 *K1*D,
        Kpp = 2 *K1 + 8*K2*D*D,
        Kqq = 2 *K1 + 2*K2*D*D,
        Kppp= 24*K2*D + 48*K3*D*D*D,
        Kpqq= 8 *K2*D + 8 *K3*D*D*D;
        if(value)
            *value = norm * (K0 + Kp * p + 0.5 * Kpp * p*p + 0.5 * Kqq * q*q +
                1./6 * Kppp * p*p*p + 0.5 * Kpqq * p*q*q);
        if(deriv) {
            (D2<=0 ? deriv->dR : deriv->dz) =  // direction from origin to the focal point
                norm * (Kp + Kpp * p + 0.5 * Kppp * p*p + 0.5 * Kpqq * q*q) * signp;
            (D2<=0 ? deriv->dz : deriv->dR) =  // direction perpendicular to the previous one
                norm * (Kqq * q + Kpqq * p*q);
            deriv->dphi = 0;
        }
        if(deriv2) {
            (D2<=0 ? deriv2->dR2 : deriv2->dz2) = norm * (Kpp + Kppp * p);
            (D2<=0 ? deriv2->dz2 : deriv2->dR2) = norm * (Kqq + Kpqq * p);
            deriv2->dRdz  = norm * Kpqq * q * signp;
            deriv2->dphi2 = deriv2->dRdphi = deriv2->dzdphi = 0;
        }
    } else {
        // duplicate the code from coord::evalAndConvert(), since for technical reasons
        // this class cannot expose a coord::IScalarFunction<coord::Axi> interface
        bool needDeriv = deriv!=NULL || deriv2!=NULL;
        bool needDeriv2= deriv2!=NULL;
        coord::GradAxi evalGrad;
        coord::HessAxi evalHess;
        coord::PosDerivT <coord::Cyl, coord::Axi> coordDeriv;
        coord::PosDeriv2T<coord::Cyl, coord::Axi> coordDeriv2;
        const coord::PosAxi evalPos = needDeriv ?
            coord::toPosDeriv(pos, &coordDeriv, needDeriv2 ? &coordDeriv2 : NULL, coord::Axi(D2)) :
            coord::toPos(pos, coord::Axi(D2));
        // compute the function in transformed coordinates
        evalAxi(evalPos, value, needDeriv ? &evalGrad : NULL, needDeriv2 ? &evalHess : NULL);
        if(deriv)  // ... and convert gradient/hessian back to output coords if necessary.
            *deriv  = coord::toGrad(evalGrad, coordDeriv);
        if(deriv2)
            *deriv2 = coord::toHess(evalGrad, evalHess, coordDeriv, coordDeriv2);
    }
}

double PerfectEllipsoid::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    return 1/M_PI/M_PI * mass / (scaleRadiusB * pow_2(scaleRadiusA) *
        pow_2(pow_2(pos.R / scaleRadiusA) + pow_2(pos.z / scaleRadiusB) + 1));
}

}  // namespace potential
