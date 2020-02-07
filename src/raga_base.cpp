#include "raga_base.h"
#include "potential_base.h"
#include "math_ode.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cmath>

namespace raga {

void BHParams::keplerOrbit(double t, double bhX[], double bhY[], double bhVX[], double bhVY[]) const
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

double BHParams::potential(double time, const coord::PosCar& point) const
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

static inline void evalRagaPotential(
    const potential::BasePotential& pot, const BHParams& bh,
    const double time, const coord::PosCar &pos,
    double* value, coord::GradCar* deriv, coord::HessCar* deriv2)
{
    pot.eval(pos, value, deriv, deriv2);
    if(bh.mass == 0)
        return;
    double bhX[2], bhY[2], bhVX[2], bhVY[2];
    bh.keplerOrbit(time, bhX, bhY, bhVX, bhVY);
    int numbh = (bh.sma != 0 && bh.q != 0) ? 2 : 1;
    double Mbh[2] = {
        numbh==1 ? bh.mass : bh.mass / (1 + bh.q),
        bh.mass * bh.q / (1 + bh.q) };
    for(int b=0; b<numbh; b++) {
        double x = pos.x - bhX[b], y = pos.y - bhY[b], z = pos.z;
        double invr2 = 1 / (pow_2(x) + pow_2(y) + pow_2(z));
        double minvr = Mbh[b] * sqrt(invr2);
        if(value)
            *value -= minvr;
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

void RagaOrbitIntegrator::eval(const double t, const double x[], double dxdt[]) const
{
    coord::GradCar grad;
    evalRagaPotential(pot, bh, t, coord::PosCar(x[0], x[1], x[2]), NULL, &grad, NULL);
    dxdt[0] = x[3];
    dxdt[1] = x[4];
    dxdt[2] = x[5];
    dxdt[3] = -grad.dx;
    dxdt[4] = -grad.dy;
    dxdt[5] = -grad.dz;
}

double RagaOrbitIntegrator::getAccuracyFactor(const double t, const double x[]) const
{
    double Epot, Ekin = 0.5 * (x[3]*x[3] + x[4]*x[4] + x[5]*x[5]);
    evalRagaPotential(pot, bh, t, coord::PosCar(x[0], x[1], x[2]), &Epot, NULL, NULL);
    return fmin(1, fabs(Epot + Ekin) / fmax(fabs(Epot), Ekin));
}

}  // namespace raga