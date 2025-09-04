#include "actions_isochrone.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cmath>

namespace actions{

void evalIsochrone(
    const double M, const double b, const coord::PosVelCyl& pointCyl,
    Actions* act, Angles* ang, Frequencies* freq)
{
    coord::PosVelSph point(toPosVelSph(pointCyl));
    double rb = sqrt(b*b +pow_2(point.r));
    double L  = Ltotal(point);
    double L1 = sqrt(L*L + 4*M*b);
    double Lz = pointCyl.R * pointCyl.vphi;
    double E  = -M / (b+rb) + 0.5 * (pow_2(point.vr) + pow_2(point.vtheta) + pow_2(point.vphi));
    double J0 = M / sqrt(-2*E);
    if(act) {
        act->Jphi = Lz;
        act->Jz   = pointCyl.z==0 && pointCyl.vz==0 ? 0 : fmax(0, L - fabs(Lz));
        act->Jr   = E>=0 ? NAN : fmax(0, J0 - 0.5 * (L + L1));   // note: possible loss of precision!
    }
    if(ang) {
        double j0invsq = M*b / pow_2(J0);
        double ecc  = sqrt(pow_2(1-j0invsq) - pow_2(L/J0));
        double fac1 = (1 + ecc - j0invsq) * J0 / L;
        double fac2 = (1 + ecc + j0invsq) * J0 / L1;
        // below are quantities that depend on position along the orbit
        double k1   = point.r * point.vr;
        double k2   = J0 - M * rb / J0;
        double psi  = math::atan2(pointCyl.z * L,  -pointCyl.R * point.vtheta * point.r);
        double chi  = math::atan2(pointCyl.z * point.vphi, -point.vtheta * point.r);
        double eta  = math::atan2(k1, k2);                         // eccentric anomaly
        double sineta     = k1 / sqrt(k1*k1 + k2*k2);        // sin(eta)
        double tanhalfeta = eta==0 ? 0 : -k2/k1 + 1/sineta;  // tan(eta/2)
        ang->thetar   = math::wrapAngle(eta - sineta * ecc );
        ang->thetaz   = math::wrapAngle(psi + 0.5 * (1 + L/L1) * (ang->thetar - (eta<0 ? 2*M_PI : 0))
                      - math::atan(fac1 * tanhalfeta) - math::atan(fac2 * tanhalfeta) * L/L1 );
        ang->thetaphi = math::wrapAngle(point.phi - chi + math::sign(Lz) * ang->thetaz);
        if(pointCyl.z==0 && pointCyl.vz==0)  // if Jz=0, the value of theta_z is meaningless
            ang->thetaz = 0;
        if(E>=0)
            ang->thetar = ang->thetaz = ang->thetaphi = NAN;
    }
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * 0.5 * (1 + L/L1);
        freq->Omegaphi = math::sign(Lz) * freq->Omegaz;
    }
}

coord::PosVelCyl mapIsochrone(
    const double M, const double b,
    const ActionAngles& aa, Frequencies* freq)
{
    if(aa.Jr<0 || aa.Jz<0) {
        if(freq)
            freq->Omegar = freq->Omegaz = freq->Omegaphi = NAN;
        return coord::PosVelCyl(NAN, NAN, NAN, NAN, NAN, NAN);
    }
    double signJphi = math::sign(aa.Jphi);
    double absJphi  = signJphi * aa.Jphi;
    double L    = aa.Jz + absJphi;
    double L1   = sqrt(L*L + 4*M*b);
    double LL1  = 0.5 + 0.5 * L/L1;
    double J0   = aa.Jr + 0.5 * (L + L1);  // combined magnitude of actions
    double j0invsq = M*b / pow_2(J0);
    // x1,x2 are roots of equation  x^2 - 2*(x-1)*M*b/J0^2 + (L/J0)^2-1 = 0:
    // x1 = j0invsq - ecc, x2 = j0invsq + ecc;  -1 <= x1 <= x2 <= 1.
    double ecc  = fmin(sqrt(aa.Jr * (aa.Jr+L) * (aa.Jr+L1) * (aa.Jr+L+L1)) / pow_2(J0), 1.);
    double fac1 = (1 + ecc - j0invsq) * J0 / L;   // sqrt( (1-x1) / (1-x2) )
    double fac2 = (1 + ecc + j0invsq) * J0 / L1;  // sqrt( (1+x2) / (1+x1) )

    // quantities below depend on angles
    double eta = math::solveKepler(ecc, aa.thetar); // thetar = eta - ecc * sin(eta)
    double sineta, coseta, sinpsi, cospsi;
    math::sincos(eta, sineta, coseta);
    double ra = 1 - ecc * coseta;   // Kepler problem:  r / a = 1 - e cos(eta)
    double tanhalfeta = coseta==-1 ? INFINITY : sineta / (1 + coseta);
    double thetar   = aa.thetar - (eta>M_PI ? 2*M_PI : 0);
    double psi1     = math::atan(fac1 * tanhalfeta);
    double psi2     = math::atan(fac2 * tanhalfeta);
    double psi      = aa.thetaz - LL1 * thetar + psi1 + psi2 * L/L1;
    math::sincos(psi, sinpsi, cospsi);
    double chi      = aa.Jz != 0 ? math::atan2(absJphi * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    double r        = sqrt(fmax(0, pow_2(ra * J0*J0/M) - b*b));
    double vr       = J0 * ecc * sineta / r;
    double vtheta   = -L * sini * cospsi / (r * sintheta);
    coord::PosVelCyl point;
    point.R    = r * sintheta;
    point.z    = r * costheta;
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * signJphi);
    point.vR   = vr * sintheta + vtheta * costheta;
    point.vz   = vr * costheta - vtheta * sintheta;
    point.vphi = aa.Jphi / point.R;
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * LL1;
        freq->Omegaphi = freq->Omegaz * signJphi;
    }
    return point;
}

std::string ActionFinderIsochrone::name() const
{
    return "Isochrone(mass=" + utils::toString(mass) + ", radius=" + utils::toString(radius) + ")";
}

}  // namespace actions
