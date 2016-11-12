#include "potential_analytic.h"
#include <cmath>
#include <cassert>

namespace potential{

void Plummer::evalDeriv(double r,
    double* potential, double* deriv, double* deriv2) const
{
    double rsq = pow_2(r) + pow_2(scaleRadius);
    double pot = -mass/sqrt(rsq);
    if(potential)
        *potential = pot;
    if(deriv)
        *deriv = -pot*r/rsq;
    if(deriv2)
        *deriv2 = pot*(2*pow_2(r)-pow_2(scaleRadius))/pow_2(rsq);
}

double Plummer::enclosedMass(double r) const
{
    // should give correct result in all limiting cases
    return mass / pow_3(sqrt(pow_2(scaleRadius/r) + 1));
}

void Isochrone::evalDeriv(double r,
    double* potential, double* deriv, double* deriv2) const
{
    double rb  = sqrt(pow_2(r) + pow_2(scaleRadius));
    double brb = scaleRadius + rb;
    double pot = -mass / brb;
    if(potential)
        *potential = pot;
    if(deriv)
        *deriv = -pot * r / (rb * brb);
    if(deriv2)
        *deriv2 = pot * (2*pow_2(r) - pow_2(scaleRadius) * (1 + scaleRadius / rb)) / pow_2(rb * brb);
}

void NFW::evalDeriv(double r,
    double* potential, double* deriv, double* deriv2) const
{
    double ln_over_r = r==INFINITY ? 0 :
        r > scaleRadius*1e-4 ? log(1 + r/scaleRadius) / r :
        // accurate asymptotic expansion at r->0
        (1 - 0.5 * r/scaleRadius * (1 - 2./3 * r/scaleRadius)) / scaleRadius;
    if(potential)
        *potential = -mass * ln_over_r;
    if(deriv)
        *deriv = mass * (r==0 ? 0.5/pow_2(scaleRadius) :
            (ln_over_r - 1/(r+scaleRadius))/r );
    if(deriv2)
        *deriv2 = -mass * (r==0 ? 2./(3*scaleRadius*pow_2(scaleRadius)) :
            (2*ln_over_r - (2*scaleRadius+3*r)/pow_2(scaleRadius+r) )/pow_2(r) );
}

void MiyamotoNagai::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double zb=sqrt(pow_2(pos.z)+pow_2(scaleRadiusB));
    double azb2=pow_2(scaleRadiusA+zb);
    double denom=1/sqrt(pow_2(pos.R) + azb2);
    if(potential)
        *potential = -mass*denom;
    if(deriv) {
        double denom3=mass*denom*denom*denom;
        deriv->dR = pos.R * denom3;
        deriv->dz = pos.z * denom3 * (1 + scaleRadiusA/zb);
        deriv->dphi = 0;
    }
    if(deriv2) {
        double denom5=mass*pow_2(pow_2(denom))*denom;
        deriv2->dR2 = denom5 * (azb2 - 2*pow_2(pos.R));
        deriv2->dz2 = denom5 *( (pow_2(pos.R) - 2*azb2) * pow_2(pos.z/zb) +
            pow_2(scaleRadiusB) * (scaleRadiusA/zb+1) * (pow_2(pos.R) + azb2) / pow_2(zb) );
        deriv2->dRdz= denom5 * -3*pos.R*pos.z * (scaleRadiusA/zb + 1);
        deriv2->dRdphi = deriv2->dzdphi = deriv2->dphi2 = 0;
    }
}

void Logarithmic::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
{
    double m2 = coreRadius2 + pow_2(pos.x) + pow_2(pos.y)/q2 + pow_2(pos.z)/p2;
    if(potential)
        *potential = sigma2*log(m2)*0.5;
    if(deriv) {
        deriv->dx = pos.x*sigma2/m2;
        deriv->dy = pos.y*sigma2/m2/q2;
        deriv->dz = pos.z*sigma2/m2/p2;
    }
    if(deriv2) {
        deriv2->dx2 = sigma2*(1/m2    - 2*pow_2(pos.x/m2));
        deriv2->dy2 = sigma2*(1/m2/q2 - 2*pow_2(pos.y/(m2*q2)));
        deriv2->dz2 = sigma2*(1/m2/p2 - 2*pow_2(pos.z/(m2*p2)));
        deriv2->dxdy=-sigma2*pos.x*pos.y * 2/(pow_2(m2)*q2);
        deriv2->dydz=-sigma2*pos.y*pos.z * 2/(pow_2(m2)*q2*p2);
        deriv2->dxdz=-sigma2*pos.z*pos.x * 2/(pow_2(m2)*p2);
    }
}

void Harmonic::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
{
    if(potential)
        *potential = 0.5*Omega2 * (pow_2(pos.x) + pow_2(pos.y)/q2 + pow_2(pos.z)/p2);
    if(deriv) {
        deriv->dx = pos.x*Omega2;
        deriv->dy = pos.y*Omega2/q2;
        deriv->dz = pos.z*Omega2/p2;
    }
    if(deriv2) {
        deriv2->dx2 = Omega2;
        deriv2->dy2 = Omega2/q2;
        deriv2->dz2 = Omega2/p2;
        deriv2->dxdy=deriv2->dydz=deriv2->dxdz=0;
    }
}

}  // namespace potential