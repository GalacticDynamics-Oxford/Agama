#include "potential_analytic.h"
#include "math_core.h"
#include "math_specfunc.h"
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

double Plummer::densitySph(const coord::PosSph &pos, double /*time*/) const
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
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const
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
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double /*time*/) const
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
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double /*time*/) const
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


void KeplerBinaryParams::keplerOrbit(double t, double bhX[], double bhY[], double bhVX[], double bhVY[]) const
{
    if(sma!=0) {
        double omegabh = sqrt(mass/pow_3(sma));
        double eta = math::solveKepler(ecc, omegabh * t + phase);
        double sineta, coseta;
        math::sincos(eta, sineta, coseta);
        double ell = sqrt(1 - ecc * ecc);
        double a   = -sma / (1 + q);
        double doteta = omegabh / (1 - ecc * coseta);
        bhX [1] = (coseta - ecc) * a;
        bhY [1] =  sineta * a * ell;
        bhVX[1] = -doteta * a * sineta;
        bhVY[1] =  doteta * a * coseta * ell;
        bhX [0] = -q * bhX [1];
        bhY [0] = -q * bhY [1];
        bhVX[0] = -q * bhVX[1];
        bhVY[0] = -q * bhVY[1];
    } else {
        bhX[0] = bhY[0] = bhVX[0] = bhVY[0] = bhX[1] = bhY[1] = bhVX[1] = bhVY[1] = 0;
    }
}

double KeplerBinaryParams::potential(const coord::PosCar& point, double time) const
{
    double bhX[2], bhY[2], bhVX[2], bhVY[2], result = 0;
    keplerOrbit(time, bhX, bhY, bhVX, bhVY);
    int numbh = (sma != 0 && q != 0) ? 2 : 1;
    double Mbh[2] = {
        numbh==1 ? mass : mass / (1 + q),
        mass * q / (1 + q) };
    for(int b=0; b<numbh; b++) {
        double x = point.x - bhX[b], y = point.y - bhY[b], z = point.z;
        double invr2 = 1 / (pow_2(x) + pow_2(y) + pow_2(z));
        result -= Mbh[b] * sqrt(invr2);
    }
    return result;
}

void KeplerBinary::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const
{
    double bhX[2], bhY[2], bhVX[2], bhVY[2];
    params.keplerOrbit(time, bhX, bhY, bhVX, bhVY);
    int numbh = (params.sma != 0 && params.q != 0) ? 2 : 1;
    if(potential) *potential=0;
    if(deriv)  deriv->dx=deriv->dy=deriv->dz=0;
    if(deriv2) deriv2->dx2=deriv2->dy2=deriv2->dz2=deriv2->dxdy=deriv2->dxdz=deriv2->dydz=0;
    double Mbh[2] = {
        numbh==1 ? params.mass : params.mass / (1 + params.q),
        params.mass * params.q / (1 + params.q) };
    for(int b=0; b<numbh; b++) {
        double x = pos.x - bhX[b], y = pos.y - bhY[b], z = pos.z;
        double invr2 = 1 / (pow_2(x) + pow_2(y) + pow_2(z));
        double minvr = Mbh[b] * sqrt(invr2);
        if(potential)
            *potential -= minvr;
        if(deriv) {
            deriv->dx += x * minvr * invr2;
            deriv->dy += y * minvr * invr2;
            deriv->dz += z * minvr * invr2;
        }
        if(deriv2) {
            deriv2->dx2  += minvr * invr2 * (3 * pow_2(x) * invr2 - 1);
            deriv2->dy2  += minvr * invr2 * (3 * pow_2(y) * invr2 - 1);
            deriv2->dz2  += minvr * invr2 * (3 * pow_2(z) * invr2 - 1);
            deriv2->dxdy += minvr * pow_2(invr2) * 3 * x * y;
            deriv2->dydz += minvr * pow_2(invr2) * 3 * y * z;
            deriv2->dxdz += minvr * pow_2(invr2) * 3 * x * z;
        }
    }
}

}  // namespace potential
