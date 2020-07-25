#include "potential_dehnen.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential {

/// relative accuracy of potential computation (integration tolerance parameter)
static const double EPSREL_POTENTIAL_INT = 1e-6;

// Note that we use an integration rule with fixed maximum order, so it may not always achieve
// the required relative accuracy (in particular, at large radii the error is fairly large).
// There are several possible ways out:
// 1) use an adaptive-order rule (replace 'integrate' with 'integrateAdaptive') -
//    would considerably slow down calculations that are already not too fast;
// 2) replace the integration with a spherical-harmonic expansion up to l=2 at large radii
//    (like in Ferrers potential, but will need to take into account that density is non-zero
//    at all radii) - possible but would require extra work;
// 3) change variables in the integration - would be a clean solution but also requires extra work;
// 4) use a Multipole potential expansion instead - a preferred solution.
//    For this to work properly, one needs to supply the *density* profile, not the potential,
//    so that the Multipole potential will solve the Poisson equation itself.
//    This may be done by down-casting an instance of Dehnen class to BaseDensity,
//    or by using a SpheroidDensity object instead.

Dehnen::Dehnen(double _mass, double _scalerad, double _gamma, double _axisRatioY, double _axisRatioZ): 
    BasePotentialCar(), mass(_mass), scalerad(_scalerad),
    gamma(_gamma), axisRatioY(_axisRatioY), axisRatioZ(_axisRatioZ)
{
    if(scalerad<=0)
        throw std::invalid_argument("Dehnen potential: scale radius must be positive");
    if(gamma<0 || gamma>2)
        throw std::invalid_argument("Dehnen potential: gamma must lie in the range [0:2]");
}

double Dehnen::densityCar(const coord::PosCar& pos, double /*time*/) const
{
    double m = sqrt(pow_2(pos.x) + pow_2(pos.y/axisRatioY) + pow_2(pos.z/axisRatioZ));
    return mass * scalerad * (3-gamma) / (4*M_PI*axisRatioY*axisRatioZ) *
        math::pow(m, -gamma) * math::pow(scalerad+m, gamma-4);
}

namespace{  // internal

class DehnenIntegrandPhi: public math::IFunctionNoDeriv {
    const double X2, Y2, Z2, gamma, q2, p2;
public:
    DehnenIntegrandPhi(const coord::PosCar& pt, double gam, double q, double p, double scalerad) :
        X2(pow_2(pt.x/scalerad)), Y2(pow_2(pt.y/scalerad)), Z2(pow_2(pt.z/scalerad)),
        gamma(gam), q2(q*q), p2(p*p) {}
    virtual double value(double s) const {
        const double s2 = s*s;
        const double m = s * sqrt( X2 + Y2/(1-(1-q2)*s2) + Z2/(1-(1-p2)*s2) );
        const double numerator = (gamma==2)? (log((1+m)/m) - 1/(1+m)) :
            (1 - math::pow(m/(m+1), 2-gamma) * (3-gamma+m) / (1+m)) / (2-gamma);
        return -numerator / sqrt( (1-(1-q2)*s2) * (1-(1-p2)*s2) );
    }
};

class DehnenIntegrandForce: public math::IFunctionNoDeriv {
    const double X2, Y2, Z2, gamma;
public:
    double a2;  // squared scale radius in i-th coordinate
    double C1, C2, C3, C4, C5;  // coefficients in computation as in Merritt&Fridman 1996
    enum { FORCE, DERIV, DERIV_MIXED } mode;
    DehnenIntegrandForce(const coord::PosCar& pt, double _gamma) :
        X2(pt.x*pt.x), Y2(pt.y*pt.y), Z2(pt.z*pt.z), gamma(_gamma) {};
    double value(double s) const {
        const double s2=s*s;
        const double m = s * sqrt(X2 / (a2 + C1*s2) + Y2 / (a2 + C2*s2) + Z2 / (a2 + C3*s2) );
        double result = s2 * math::pow(m/(1+m), -gamma);
        switch(mode) {
            case FORCE:
                return result / pow_2(pow_2(1+m)) /
                sqrt( (a2 + C1*s2) * (a2 + C2*s2) * (a2 + C3*s2) );
            case DERIV:
                return result * s2 * (gamma + 4*m) / (pow_2(m * pow_2(1+m)) * (1+m) * 
                sqrt( (a2 + C1*s2) * (a2 + C2*s2) * (a2 + C3*s2) ) );
            case DERIV_MIXED:
                return result * s2 * (gamma + 4*m) / (pow_2(m * pow_2(1+m)) * (1+m) * 
                sqrt( (a2 + C5*s2) * pow_3(a2 + C4*s2) ) );
            default:  // shouldn't happen
                assert(!"Incorrect integration mode in Dehnen potential");
        }
    }
};

} // internal namespace

void Dehnen::evalCar(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double /*time*/) const
{
    if(axisRatioY==1 && axisRatioZ==1) {  // analytical expression for spherical potential
        double r = sqrt(pos.x*pos.x + pos.y*pos.y + pos.z*pos.z);
        double s = scalerad / r;
        if(potential!=NULL) {
            if(s > 2e-3/(4-gamma))
                *potential = mass/scalerad *
                    (gamma==2 ? -log(1+s) : (1 - math::pow(1+s, gamma-2)) / (gamma-2) );
            else  // asymptotic expansion for r-->infinity, or equivalently s-->0
                *potential = mass/scalerad *
                    -s * (1 + s * (gamma-3)/2 * (1 + s * (gamma-4)/3 * (1 + s * (gamma-5)/4)));
        }
        if(deriv==NULL && deriv2==NULL)
            return;
        double xr = pos.x/r, yr = pos.y/r, zr = pos.z/r;
        double val = mass * math::pow(r, 1-gamma) * math::pow(r+scalerad, gamma-3);
        if(deriv!=NULL) {
            deriv->dx = r>0 ? val*xr : 0;
            deriv->dy = r>0 ? val*yr : 0;
            deriv->dz = r>0 ? val*zr : 0;
        }
        if(deriv2!=NULL) {
            double val2  = -val * (gamma * s + 3) / (r+scalerad);
            deriv2->dx2  = val2 * xr * xr + val/r;
            deriv2->dy2  = val2 * yr * yr + val/r;
            deriv2->dz2  = val2 * zr * zr + val/r;
            deriv2->dxdy = val2 * xr * yr;
            deriv2->dydz = val2 * yr * zr;
            deriv2->dxdz = val2 * zr * xr;
        }
        return;
    }
    if(potential) {
        DehnenIntegrandPhi fnc(pos, gamma, axisRatioY, axisRatioZ, scalerad);
        *potential = math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT) * mass/scalerad;
    }
    if(deriv==NULL && deriv2==NULL)
        return;
    if(pos.x==0 && pos.y==0 && pos.z==0) {
        if(deriv)
            deriv->dx = deriv->dy = deriv->dz = 0;
        if(deriv2)
            deriv2->dx2 = deriv2->dy2 = deriv2->dz2 = deriv2->dxdy = deriv2->dxdz = deriv2->dydz = NAN;
        return;
    }
    double scalerad2 = pow_2(scalerad);
    DehnenIntegrandForce fnc(pos, gamma);
    fnc.a2 = scalerad2;
    fnc.C1 = 0;
    fnc.C2 = (pow_2(axisRatioY)-1)*scalerad2;
    fnc.C3 = (pow_2(axisRatioZ)-1)*scalerad2;
    fnc.mode = fnc.FORCE;
    double result = math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
    if(deriv)
        deriv->dx = (3-gamma)*mass * pos.x * result;
    if(deriv2) {
        fnc.mode = fnc.DERIV;
        deriv2->dx2  = (3-gamma)*mass *
            (result - pow_2(pos.x/scalerad) * math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT) );
        fnc.C4 = fnc.C2;
        fnc.C5 = fnc.C3;
        fnc.mode = fnc.DERIV_MIXED;
        deriv2->dxdy = -(3-gamma)*mass * pos.x*pos.y/scalerad *
            math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
        fnc.mode = fnc.FORCE;
    }

    fnc.a2 = scalerad2*pow_2(axisRatioY);
    fnc.C1 = (1-pow_2(axisRatioY))*scalerad2;
    fnc.C2 = 0;
    fnc.C3 = (pow_2(axisRatioZ)-pow_2(axisRatioY))*scalerad2;
    result = math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
    if(deriv)
        deriv->dy = (3-gamma)*mass * pos.y * result;
    if(deriv2) {
        fnc.mode = fnc.DERIV;
        deriv2->dy2  = (3-gamma)*mass *
            (result - pow_2(pos.y/axisRatioY/scalerad) * math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT) );
        fnc.C4 = fnc.C3;
        fnc.C5 = fnc.C1;
        fnc.mode = fnc.DERIV_MIXED;
        deriv2->dydz = -(3-gamma)*mass * pos.y*pos.z/axisRatioY/scalerad *
            math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
        fnc.mode = fnc.FORCE;
    }

    fnc.a2 = scalerad2*pow_2(axisRatioZ);
    fnc.C1 = (1-pow_2(axisRatioZ))*scalerad2;
    fnc.C2 = (pow_2(axisRatioY)-pow_2(axisRatioZ))*scalerad2;
    fnc.C3 = 0;
    result = math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
    if(deriv)
        deriv->dz = (3-gamma)*mass * pos.z * result;
    if(deriv2) {
        fnc.mode = fnc.DERIV;
        deriv2->dz2  = (3-gamma)*mass *
            (result - pow_2(pos.z/axisRatioZ/scalerad) * math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT) );
        fnc.C4 = fnc.C1;
        fnc.C5 = fnc.C2;
        fnc.mode = fnc.DERIV_MIXED;
        deriv2->dxdz = -(3-gamma)*mass * pos.x*pos.z/axisRatioZ/scalerad *
            math::integrate(fnc, 0, 1, EPSREL_POTENTIAL_INT);
    }
}

}  // namespace potential
