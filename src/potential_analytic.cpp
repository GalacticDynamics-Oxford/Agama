#include "potential_analytic.h"
#include <cmath>

namespace potential{

void Plummer::evalDeriv(double r,
    double* potential, double* deriv, double* deriv2) const
{
    double invrsq = mass?  1. / (pow_2(r) + pow_2(scaleRadius)) : 0;  // if mass=0, output 0
    double pot = -mass * sqrt(invrsq);
    if(potential)
        *potential = pot;
    if(deriv)
        *deriv = -pot * r * invrsq;
    if(deriv2)
        *deriv2 = pot * (2 * pow_2(r * invrsq) - pow_2(scaleRadius * invrsq));
}

double Plummer::densitySph(const coord::PosSph &pos) const
{
    double invrsq = 1. / (pow_2(pos.r) + pow_2(scaleRadius));
    return 0.75/M_PI * mass * pow_2(scaleRadius * invrsq) * sqrt(invrsq);
}

double Plummer::enclosedMass(double r) const
{
    if(scaleRadius==0)
        return mass;
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
        *deriv2 = pot * (2*pow_2(r / (rb * brb)) - pow_2(scaleRadius / (rb * brb)) * (1 + scaleRadius / rb));
}

void NFW::evalDeriv(double r,
    double* potential, double* deriv, double* deriv2) const
{
    double rrel = r / scaleRadius;
    double ln_over_r = r==INFINITY ? 0 :
        rrel > 1e-3 ? log(1 + rrel) / r :
        // accurate asymptotic expansion at r->0
        (1 - 0.5 * rrel * (1 - 2./3 * rrel * (1 - 3./4 * rrel))) / scaleRadius;
    if(potential)
        *potential = -mass * ln_over_r;
    if(deriv)
        *deriv = mass * (r==0 ? 0.5 / pow_2(scaleRadius) :
            (ln_over_r - 1/(r+scaleRadius)) / r );
    if(deriv2)
        *deriv2 = -mass * (r==0 ? 2./3 / pow_3(scaleRadius) :
            (2*ln_over_r - (2*scaleRadius + 3*r) / pow_2(scaleRadius+r) ) / pow_2(r) );
}

void MiyamotoNagai::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double zb    = sqrt(pow_2(pos.z) + pow_2(scaleRadiusB));
    double azb2  = pow_2(scaleRadiusA + zb);
    double den2  = 1. / (pow_2(pos.R) + azb2);
    double denom = sqrt(den2);
    double Rsc   = pos.R * denom;
    double zsc   = pos.z * denom;
    if(potential)
        *potential = -mass * denom;
    if(deriv) {
        deriv->dR  = mass * den2 * Rsc;
        deriv->dz  = mass * den2 * zsc * (1 + scaleRadiusA/zb);
        deriv->dphi= 0;
    }
    if(deriv2) {
        double mden3 = mass * denom * den2;
        deriv2->dR2  = mden3 * (azb2*den2 - 2*pow_2(Rsc));
        deriv2->dz2  = mden3 * ( (pow_2(Rsc) - 2*azb2*den2) * pow_2(pos.z/zb) +
            pow_2(scaleRadiusB) * (scaleRadiusA/zb + 1) * (pow_2(Rsc) + azb2*den2) / pow_2(zb) );
        deriv2->dRdz = mden3 * -3 * Rsc * zsc * (scaleRadiusA/zb + 1);
        deriv2->dRdphi = deriv2->dzdphi = deriv2->dphi2 = 0;
    }
}

void Logarithmic::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
{
    double m2 = coreRadius2 + pow_2(pos.x) + pow_2(pos.y)/p2 + pow_2(pos.z)/q2;
    if(potential)
        *potential = 0.5 * sigma2 * log(m2);
    if(deriv) {
        deriv->dx = pos.x * sigma2/m2;
        deriv->dy = pos.y * sigma2/m2/p2;
        deriv->dz = pos.z * sigma2/m2/q2;
    }
    if(deriv2) {
        deriv2->dx2 = sigma2 * (1/m2    - 2 * pow_2(pos.x / m2));
        deriv2->dy2 = sigma2 * (1/m2/p2 - 2 * pow_2(pos.y / (m2 * p2)));
        deriv2->dz2 = sigma2 * (1/m2/q2 - 2 * pow_2(pos.z / (m2 * q2)));
        deriv2->dxdy=-sigma2 * pos.x * pos.y * 2 / (pow_2(m2) * p2);
        deriv2->dydz=-sigma2 * pos.y * pos.z * 2 / (pow_2(m2) * p2 * q2);
        deriv2->dxdz=-sigma2 * pos.z * pos.x * 2 / (pow_2(m2) * q2);
    }
}

void Harmonic::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
{
    if(potential)
        *potential = 0.5*Omega2 * (pow_2(pos.x) + pow_2(pos.y)/p2 + pow_2(pos.z)/q2);
    if(deriv) {
        deriv->dx = pos.x*Omega2;
        deriv->dy = pos.y*Omega2/p2;
        deriv->dz = pos.z*Omega2/q2;
    }
    if(deriv2) {
        deriv2->dx2 = Omega2;
        deriv2->dy2 = Omega2/p2;
        deriv2->dz2 = Omega2/q2;
        deriv2->dxdy=deriv2->dydz=deriv2->dxdz=0;
    }
}

}  // namespace potential
