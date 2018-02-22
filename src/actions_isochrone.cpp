#include "actions_isochrone.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <stdexcept>
#include <cmath>

namespace actions{

Actions actionsIsochrone(
    const double M, const double b,
    const coord::PosVelCyl& point)
{
    double L = Ltotal(point);
    double E = -M / (b + sqrt(b*b + pow_2(point.R) + pow_2(point.z))) +
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    Actions acts;
    acts.Jphi = Lz(point);
    acts.Jz   = point.z==0 && point.vz==0 ? 0 : fmax(0, L - fabs(acts.Jphi));
    // note: the expression below may suffer from loss of precision when E -> Phi(0)!
    acts.Jr   = fmax(0, M / sqrt(-2*E) - 0.5 * (L + sqrt(L*L + 4*M*b)) );
    return acts;
}

ActionAngles actionAnglesIsochrone(
    const double M, const double b,
    const coord::PosVelCyl& pointCyl,
    Frequencies* freq)
{
    coord::PosVelSph point(toPosVelSph(pointCyl));
    ActionAngles aa;
    double rb   = sqrt(b*b +pow_2(point.r));
    double L    = Ltotal(point);
    double L1   = sqrt(L*L + 4*M*b);
    double J0   = M / sqrt(2*M / (b+rb) - pow_2(point.vr) - pow_2(point.vtheta) - pow_2(point.vphi));
    double j0invsq = M*b / pow_2(J0);
    // J0 is related to total energy via  J0 = M / sqrt(-2*E)
    aa.Jphi     = Lz(pointCyl);
    aa.Jz       = pointCyl.z==0 && pointCyl.vz==0 ? 0 : fmax(0, L - fabs(aa.Jphi));
    aa.Jr       = fmax(0, J0 - 0.5 * (L + L1));   // note: loss of precision is possible!
    double ecc  = sqrt(pow_2(1-j0invsq) - pow_2(L/J0));
    double fac1 = (1 + ecc - j0invsq) * J0 / L;
    double fac2 = (1 + ecc + j0invsq) * J0 / L1;
    // below are quantities that depend on position along the orbit
    double k1   = point.r * point.vr;
    double k2   = J0 - M * rb / J0;
    double eta  = atan2(k1, k2);                         // eccentric anomaly
    double sineta     = k1 / sqrt(k1*k1 + k2*k2);        // sin(eta)
    double tanhalfeta = eta==0 ? 0 : -k2/k1 + 1/sineta;  // tan(eta/2)
    double psi  = atan2(pointCyl.z * L,  -pointCyl.R * point.vtheta * point.r);
    double chi  = atan2(pointCyl.z * point.vphi, -point.vtheta * point.r);
    aa.thetar   = math::wrapAngle(eta - sineta * ecc );
    aa.thetaz   = math::wrapAngle(psi + 0.5 * (1 + L/L1) * (aa.thetar - (eta<0 ? 2*M_PI : 0))
                - atan(fac1 * tanhalfeta) - atan(fac2 * tanhalfeta) * L/L1 );
    aa.thetaphi = math::wrapAngle(point.phi - chi + math::sign(aa.Jphi) * aa.thetaz);
    if(aa.Jz == 0)  // in this case the value of theta_z is meaningless
        aa.thetaz = 0;
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * 0.5 * (1 + L/L1);
        freq->Omegaphi = math::sign(aa.Jphi) * freq->Omegaz;
    }
    return aa;
}

coord::PosVelCyl mapIsochrone(
    const double M, const double b,
    const ActionAngles& aa, Frequencies* freq)
{
    return toPosVelCyl(ToyMapIsochrone(M, b).map(aa, freq));
}

coord::PosVelSphMod ToyMapIsochrone::map(
    const ActionAngles& aa,
    Frequencies* freq,
    DerivAct<coord::SphMod>* derivAct,
    DerivAng<coord::SphMod>* derivAng,
    coord::PosVelSphMod* derivParam) const
{
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
    double psi1     = atan(fac1 * tanhalfeta);
    double psi2     = atan(fac2 * tanhalfeta);
    double psi      = aa.thetaz - LL1 * thetar + psi1 + psi2 * L/L1;
    math::sincos(psi, sinpsi, cospsi);
    double chi      = aa.Jz != 0 ? atan2(absJphi * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    coord::PosVelSphMod point;
    point.r    = sqrt(fmax(0, pow_2(ra * J0*J0/M) - b*b));
    point.pr   = J0 * ecc * sineta / point.r;
    point.tau  = costheta / (1+sintheta);
    point.ptau = L * sini * cospsi * (1/sintheta + 1);
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * signJphi);
    point.pphi = aa.Jphi;
    if(freq) {
        freq->Omegar   = pow_2(M) / pow_3(J0);
        freq->Omegaz   = freq->Omegar * LL1;
        freq->Omegaphi = freq->Omegaz * signJphi;
    }
    if(!derivParam && !derivAct && !derivAng)
        return point;
    // common terms
    double dtan1 = 1 / (1/tanhalfeta + tanhalfeta*pow_2(fac1));
    double dtan2 = 1 / (1/tanhalfeta + tanhalfeta*pow_2(fac2)) * L/L1;
    double mult_chi = 1 / (pow_2(aa.Jphi * sinpsi) + pow_2(L * cospsi));
    double  dtau = sini * cospsi / (sintheta * (1 + sintheta));
    double dptau = -L * sini * sinpsi * (pow_2(aa.Jphi/L) / pow_3(sintheta) + 1);
    if(derivParam) {
        // common terms - derivs w.r.t. (M*b)  !!!NOTE: the case b==0 is not supported!!!
        double tmp       = 1 + j0invsq - L1/J0;
        double  decc_dMb = (tmp * (1-j0invsq) / ecc - ecc) / (J0*L1);
        double dfac1_dMb = tmp * ((1-j0invsq) / ecc + 1)   / (L *L1);
        double dfac2_dMb = (ecc/(1+j0invsq) + 1) * (J0 * decc_dMb + (1-2*J0/L1)/L1 * ecc) / L1;
        double  dpsi_dMb = (thetar - 2*psi2) * L/pow_3(L1) + 
            (dfac1_dMb + fac1 * decc_dMb / ra) * dtan1 +
            (dfac2_dMb + fac2 * decc_dMb / ra) * dtan2;
        double  dchi_dMb = aa.Jz == 0 ? dpsi_dMb :
            absJphi * L * dpsi_dMb * mult_chi;
        double drpr_dMb  = sineta * (ecc/L1 + J0/ra * decc_dMb);        // d(r*pr) / d(M*b)
        double  drb_dMb  = b/point.r / pow_2(j0invsq) * 
            (decc_dMb * (ecc-coseta) + (2/(J0*L1) - 1/(M*b)) * ra*ra);  // d(r/b)  / d(M*b)
        // derivs w.r.t. M: dX/dM = b * dX/d(M*b)
        derivParam[0].r    = b*b * drb_dMb;
        derivParam[0].pr   = b/point.r * (drpr_dMb - b*point.pr * drb_dMb);
        derivParam[0].tau  = b *  dtau * dpsi_dMb;
        derivParam[0].ptau = b * dptau * dpsi_dMb;
        derivParam[0].phi  = b * dchi_dMb * signJphi;
        derivParam[0].pphi = 0;
        // derivs w.r.t. b
        derivParam[1].r    = M*b * drb_dMb + point.r/b;
        derivParam[1].pr   = M/b * derivParam[0].pr - point.pr/b;
        derivParam[1].tau  = M *  dtau * dpsi_dMb;
        derivParam[1].ptau = M * dptau * dpsi_dMb;
        derivParam[1].phi  = M * dchi_dMb * signJphi;
        derivParam[1].pphi = 0;
    }
    if(derivAct) {  // TODO: rewrite to make it less prone to cancellation errors!!!
        double  decc_dJr = ( (1 - pow_2(j0invsq)) / ecc - ecc) / J0;
        double  decc_add = -L / (J0*J0*ecc);
        double  decc_dL  = decc_dJr * LL1 + decc_add;
        double dfac1_dJr = fac1 / J0 + (decc_dJr*J0 + 2*j0invsq) / L;
        double dfac1_dL  = dfac1_dJr * LL1 -  fac1 / L  - 1 / (J0*ecc);
        double dfac2_dJr = fac2 / J0 + (decc_dJr*J0 - 2*j0invsq) / L1;
        double dfac2_dL  = dfac2_dJr * LL1 - (fac2 / L1 + 1 / (J0*ecc)) * L/L1;
        // derivs of intermediate angle vars and other common factors
        double  dpsi_dJr = 
            (dfac1_dJr + fac1 * decc_dJr / ra) * dtan1 +
            (dfac2_dJr + fac2 * decc_dJr / ra) * dtan2;
        double  dpsi_dL  = (2*psi2 - thetar) * 2*M*b/pow_3(L1) +
            (dfac1_dL  + fac1 * decc_dL  / ra) * dtan1 +
            (dfac2_dL  + fac2 * decc_dL  / ra) * dtan2;
        double  dchi_dJr = aa.Jz == 0 ? dpsi_dJr :
            absJphi * L * dpsi_dJr * mult_chi;
        double  dchi_dJz = //aa.Jz == 0 ? dpsi_dL  :
            absJphi * (L * dpsi_dL - sinpsi * cospsi) * mult_chi;
        double dchi_dJphi= aa.Jz == 0 ? dpsi_dL  :
            (absJphi * L * dpsi_dL + sinpsi * cospsi * (L - absJphi)) * mult_chi;
        double aoverr   = pow_2(J0*J0/M) / point.r;
        double drpr_dJr = sineta * (J0 * decc_dJr / ra + ecc);
        double drpr_dL  = sineta * (J0 * decc_dL  / ra + ecc * LL1);
        double ptau1 = dptau * dpsi_dL - cospsi * sini * pow_2(costheta) / pow_3(sintheta);
        double ptau2 = (1/pow_3(sintheta) + 1) * cospsi;
        double Ll    = L / (L + absJphi);
        // d/dJr
        derivAct->dbyJr.r   = aoverr * (decc_dJr * (ecc-coseta) + 2*ra*ra / J0);
        derivAct->dbyJr.pr  = (drpr_dJr - point.pr * derivAct->dbyJr.r) / point.r;
        derivAct->dbyJr.tau =  dtau * dpsi_dJr;
        derivAct->dbyJr.ptau= dptau * dpsi_dJr;
        derivAct->dbyJr.phi = dchi_dJr * signJphi;
        derivAct->dbyJr.pphi= 0;
        // d/dJz
        derivAct->dbyJz.r   = aoverr * (ecc-coseta) * decc_add + derivAct->dbyJr.r * LL1;
        derivAct->dbyJz.pr  = (drpr_dL - point.pr * derivAct->dbyJz.r) / point.r;
        derivAct->dbyJz.tau =  dtau * dpsi_dL + point.tau / (L * sintheta) * (1/pow_2(sini)-1);
        derivAct->dbyJz.ptau= ptau1 + ptau2 / sini;
        derivAct->dbyJz.phi = dchi_dJz * signJphi;
        derivAct->dbyJz.pphi= 0;
        // d/dJphi
        derivAct->dbyJphi.r   = derivAct->dbyJz.r  * signJphi;
        derivAct->dbyJphi.pr  = derivAct->dbyJz.pr * signJphi;
        derivAct->dbyJphi.tau = (dtau * dpsi_dL - point.tau / (L * sintheta) * (1-Ll) ) * signJphi;
        derivAct->dbyJphi.ptau= (ptau1 + ptau2 * sini * Ll) * signJphi;
        derivAct->dbyJphi.phi = dchi_dJphi;
        derivAct->dbyJphi.pphi= 1;
    }
    return point;
}

}  // namespace actions
