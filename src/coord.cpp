#include "coord.h"
#include "math_core.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace coord{

ProlSph::ProlSph(double Delta) :
    Delta2(Delta*Delta)
{
    if(Delta<=0)
        throw std::invalid_argument("Invalid parameters for Prolate Spheroidal coordinate system");
}

//--------  angular momentum functions --------//

template<> double Ltotal(const PosVelCar& p) {
    return sqrt(pow_2(p.y*p.vz-p.z*p.vy) + pow_2(p.z*p.vx-p.x*p.vz) + pow_2(p.x*p.vy-p.y*p.vx));
}
template<> double Ltotal(const PosVelCyl& p) {
    return sqrt((pow_2(p.R) + pow_2(p.z)) * pow_2(p.vphi) + pow_2(p.R*p.vz-p.z*p.vR));
}
template<> double Ltotal(const PosVelSph& p) {
    return sqrt(pow_2(p.vtheta) + pow_2(p.vphi)) * p.r;
}
template<> double Ltotal(const PosVelAxi& p) {
    return Ltotal(toPosVelCyl(p));
}

template<> double Lz(const PosVelCar& p) { return p.x * p.vy - p.y * p.vx; }
template<> double Lz(const PosVelCyl& p) { return p.R * p.vphi; }
template<> double Lz(const PosVelSph& p) {
    double sintheta, costheta;
    math::sincos(p.theta, sintheta, costheta);  // this gives exactly sintheta=0 for theta=M_PI
    return p.r * sintheta * p.vphi;
}
template<> double Lz(const PosVelAxi& p) {
    double chi = p.cs.Delta2>=0 ? p.rho : sqrt(pow_2(p.rho) - p.cs.Delta2);
    double sinnu = 1 / sqrt(1 + pow_2(p.cotnu));
    return p.vphi * chi * sinnu;
}

// multiply two numbers, replacing {anything including INFINITY} * 0 with 0;
// the same result may be achieved by nan2num(x*y), but with two comparisons instead of one
inline double mul(double x, double y) { return y==0 ? 0 : x*y; }

//--------  position conversion functions ---------//

template<> PosCar toPos(const PosCyl& p, const Car) {
    double sinphi, cosphi;
    math::sincos(p.phi, sinphi, cosphi);
    return PosCar(mul(p.R, cosphi), mul(p.R, sinphi), p.z);
}
template<> PosCar toPos(const PosSph& p, const Car) {
    double sintheta, costheta, sinphi, cosphi;
    math::sincos(p.theta, sintheta, costheta);
    math::sincos(p.phi, sinphi, cosphi);
    return PosCar(mul(p.r, sintheta*cosphi), mul(p.r, sintheta*sinphi), mul(p.r, costheta));
}
template<> PosCyl toPos(const PosCar& p, const Cyl) {
    return PosCyl(sqrt(pow_2(p.x) + pow_2(p.y)), p.z, math::atan2(p.y, p.x));
}
template<> PosCyl toPos(const PosSph& p, const Cyl) {
    double sintheta, costheta;
    math::sincos(p.theta, sintheta, costheta);
    return PosCyl(mul(p.r, sintheta), mul(p.r, costheta), p.phi);
}
template<> PosSph toPos(const PosCar& p, const Sph) {
    return PosSph(sqrt(pow_2(p.x)+pow_2(p.y)+pow_2(p.z)),
        math::atan2(sqrt(pow_2(p.x) + pow_2(p.y)), p.z), math::atan2(p.y, p.x));
}
template<> PosSph toPos(const PosCyl& p, const Sph) {
    return PosSph(sqrt(pow_2(p.R) + pow_2(p.z)), math::atan2(p.R, p.z), p.phi);
}
template<> PosCyl toPos(const PosProlSph& p, const Cyl) {
    if(fabs(p.nu)>p.coordsys.Delta2 || p.lambda<p.coordsys.Delta2)
        throw std::invalid_argument("Incorrect ProlSph coordinates");
    const double R = sqrt( (p.lambda-p.coordsys.Delta2) * (1 - fabs(p.nu) / p.coordsys.Delta2) );
    const double z = sqrt( p.lambda * fabs(p.nu) / p.coordsys.Delta2) * (p.nu>=0 ? 1 : -1);
    return PosCyl(R, z, p.phi);
}
// declare an instantiation which will be defined later
template<> PosProlSph toPosDeriv(const PosCyl& from,
    PosDerivT<Cyl, ProlSph>* derivs, PosDeriv2T<Cyl, ProlSph>* derivs2, const ProlSph cs);
template<> PosProlSph toPos(const PosCyl& from, const ProlSph cs) {
    return toPosDeriv<Cyl,ProlSph>(from, NULL, NULL, cs);
}


//-------- position conversion with derivatives --------//

template<>
PosCyl toPosDeriv(const PosCar& p, PosDerivT<Car, Cyl>* deriv, PosDeriv2T<Car, Cyl>* deriv2, const Cyl)
{
    const double R2=pow_2(p.x)+pow_2(p.y), R=sqrt(R2);
    if(R==0) {
        // degenerate case, but provide something meaningful nevertheless,
        // assuming that these numbers will be multiplied by 0 anyway
        if(deriv!=NULL)
            deriv->dRdx=deriv->dRdy=deriv->dphidx=deriv->dphidy=1.;
        if(deriv2!=NULL)
            deriv2->d2Rdx2=deriv2->d2Rdy2=deriv2->d2Rdxdy=
            deriv2->d2phidx2=deriv2->d2phidy2=deriv2->d2phidxdy=1.;
        return PosCyl(0, p.z, 0);
    }
    const double cosphi=p.x/R, sinphi=p.y/R;
    if(deriv!=NULL) {
        deriv->dRdx=cosphi;
        deriv->dRdy=sinphi;
        deriv->dphidx=-sinphi/R;
        deriv->dphidy=cosphi/R;
    }
    if(deriv2!=NULL) {
        deriv2->d2Rdx2 =pow_2(sinphi)/R;
        deriv2->d2Rdy2 =pow_2(cosphi)/R;
        deriv2->d2Rdxdy=-sinphi*cosphi/R;
        deriv2->d2phidx2 =2*sinphi*cosphi/R2;
        deriv2->d2phidy2 =-deriv2->d2phidx2;
        deriv2->d2phidxdy=(pow_2(sinphi)-pow_2(cosphi))/R2;
    }
    return PosCyl(R, p.z, math::atan2(p.y, p.x));
}

template<>
PosSph toPosDeriv(const PosCar& p, PosDerivT<Car, Sph>* deriv, PosDeriv2T<Car, Sph>* deriv2, const Sph)
{
    const double x2=pow_2(p.x), y2=pow_2(p.y), z2=pow_2(p.z);
    const double R2=x2+y2, R=sqrt(R2);
    const double r2=R2+z2, r=sqrt(r2), invr=1/r;
    if(deriv!=NULL) {
        deriv->drdx=p.x*invr;
        deriv->drdy=p.y*invr;
        deriv->drdz=p.z*invr;
        const double temp=p.z/(R*r2);
        deriv->dthetadx=p.x*temp;
        deriv->dthetady=p.y*temp;
        deriv->dthetadz=-R/r2;
        deriv->dphidx=-p.y/R2;
        deriv->dphidy=p.x/R2;
    }
    if(deriv2!=NULL) {
        const double invr3=invr/r2;
        deriv2->d2rdx2=(r2-x2)*invr3;
        deriv2->d2rdy2=(r2-y2)*invr3;
        deriv2->d2rdz2=R2*invr3;
        deriv2->d2rdxdy=-p.x*p.y*invr3;
        deriv2->d2rdxdz=-p.x*p.z*invr3;
        deriv2->d2rdydz=-p.y*p.z*invr3;
        const double invr4=1/(r2*r2);
        const double temp=p.z*invr4/(R*R2);
        deriv2->d2thetadx2=(r2*y2-2*R2*x2)*temp;
        deriv2->d2thetady2=(r2*x2-2*R2*y2)*temp;
        deriv2->d2thetadz2=2*R*p.z*invr4;
        deriv2->d2thetadxdy=-p.x*p.y*(r2+2*R2)*temp;
        const double temp2=(R2-z2)*invr4/R;
        deriv2->d2thetadxdz=p.x*temp2;
        deriv2->d2thetadydz=p.y*temp2;
        deriv2->d2phidx2=2*p.x*p.y/pow_2(R2);
        deriv2->d2phidy2=-deriv2->d2phidx2;
        deriv2->d2phidxdy=(y2-x2)/pow_2(R2);
    }
    return PosSph(r, math::atan2(R, p.z), math::atan2(p.y, p.x));
}

template<>
PosCar toPosDeriv(const PosCyl& p, PosDerivT<Cyl, Car>* deriv, PosDeriv2T<Cyl, Car>* deriv2, const Car)
{
    double sinphi, cosphi;
    math::sincos(p.phi, sinphi, cosphi);
    const double x=mul(p.R, cosphi), y=mul(p.R, sinphi);
    if(deriv!=NULL) {
        deriv->dxdR=cosphi;
        deriv->dydR=sinphi;
        deriv->dxdphi=-y;
        deriv->dydphi= x;
    }
    if(deriv2!=NULL) {
        deriv2->d2xdRdphi=-sinphi;
        deriv2->d2ydRdphi=cosphi;
        deriv2->d2xdphi2=-x;
        deriv2->d2ydphi2=-y;
    }
    return PosCar(x, y, p.z);
}

template<>
PosSph toPosDeriv(const PosCyl& p, PosDerivT<Cyl, Sph>* deriv, PosDeriv2T<Cyl, Sph>* deriv2, const Sph)
{
    const double r = sqrt(pow_2(p.R) + pow_2(p.z));
    const double rinv= 1./r;
    const double costheta=p.z*rinv, sintheta=p.R*rinv;
    if(deriv!=NULL) {
        deriv->drdR=sintheta;
        deriv->drdz=costheta;
        deriv->dthetadR=costheta*rinv;
        deriv->dthetadz=-sintheta*rinv;
    }
    if(deriv2!=NULL) {
        deriv2->d2rdR2=pow_2(costheta)*rinv;
        deriv2->d2rdz2=pow_2(sintheta)*rinv;
        deriv2->d2rdRdz=-costheta*sintheta*rinv;
        deriv2->d2thetadR2=-2*costheta*sintheta*pow_2(rinv);
        deriv2->d2thetadz2=-deriv2->d2thetadR2;
        deriv2->d2thetadRdz=(pow_2(sintheta)-pow_2(costheta))*pow_2(rinv);
    }
    return PosSph(r, math::atan2(p.R, p.z), p.phi);
}

template<>
PosCar toPosDeriv(const PosSph& p, PosDerivT<Sph, Car>* deriv, PosDeriv2T<Sph, Car>* deriv2, const Car)
{
    double sintheta, costheta, sinphi, cosphi;
    math::sincos(p.theta, sintheta, costheta);
    math::sincos(p.phi, sinphi, cosphi);
    const double R=mul(p.r, sintheta), x=mul(R, cosphi), y=mul(R, sinphi), z=mul(p.r, costheta);
    if(deriv!=NULL) {
        deriv->dxdr=sintheta*cosphi;
        deriv->dydr=sintheta*sinphi;
        deriv->dzdr=costheta;
        deriv->dxdtheta=z*cosphi;
        deriv->dydtheta=z*sinphi;
        deriv->dzdtheta=-R;
        deriv->dxdphi=-y;
        deriv->dydphi= x;
    }
    if(deriv2!=NULL) {
        deriv2->d2xdrdtheta=costheta*cosphi;
        deriv2->d2ydrdtheta=costheta*sinphi;
        deriv2->d2zdrdtheta=-sintheta;
        deriv2->d2xdrdphi=-sintheta*sinphi;
        deriv2->d2ydrdphi= sintheta*cosphi;
        deriv2->d2xdtheta2=-x;
        deriv2->d2ydtheta2=-y;
        deriv2->d2zdtheta2=-z;
        deriv2->d2xdthetadphi=-z*sinphi;
        deriv2->d2ydthetadphi= z*cosphi;
        deriv2->d2xdphi2=-x;
        deriv2->d2ydphi2=-y;
    }
    return PosCar(x, y, z);
}

template<>
PosCyl toPosDeriv(const PosSph& p, PosDerivT<Sph, Cyl>* deriv, PosDeriv2T<Sph, Cyl>* deriv2, const Cyl)
{
    double sintheta, costheta;
    math::sincos(p.theta, sintheta, costheta);
    const double R=mul(p.r, sintheta), z=mul(p.r, costheta);
    if(deriv!=NULL) {
        deriv->dRdr=sintheta;
        deriv->dRdtheta=z;
        deriv->dzdr=costheta;
        deriv->dzdtheta=-R;
    }
    if(deriv2!=NULL) {
        deriv2->d2Rdrdtheta=costheta;
        deriv2->d2Rdtheta2=-p.r*sintheta;
        deriv2->d2zdrdtheta=-sintheta;
        deriv2->d2zdtheta2=-p.r*costheta;
    }
    return PosCyl(R, z, p.phi);
}

template<>
PosCyl toPosDeriv(const PosProlSph& p, PosDerivT<ProlSph, Cyl>* deriv, PosDeriv2T<ProlSph, Cyl>* deriv2, const Cyl)
{
    const double absnu = fabs(p.nu);
    const double sign = p.nu>=0 ? 1 : -1;
    const double lminusd = p.lambda-p.coordsys.Delta2;
    const double nminusd = absnu-p.coordsys.Delta2;  // note: |nu|<=Delta^2
    if(nminusd>0 || lminusd<0)
        throw std::invalid_argument("Incorrect ProlSph coordinates");
    const double R = sqrt( lminusd * (1 - absnu / p.coordsys.Delta2) );
    const double z = sqrt( p.lambda * absnu /  p.coordsys.Delta2 ) * (p.nu>=0 ? 1 : -1);
    if(deriv!=NULL) {
        deriv->dRdlambda = 0.5*R/lminusd;
        deriv->dRdnu     = 0.5*R/nminusd * sign;
        deriv->dzdlambda = 0.5*z/p.lambda;
        deriv->dzdnu     = 0.5*z/p.nu;
    }
    if(deriv2!=NULL) {
        deriv2->d2Rdlambda2   = -0.25*R / pow_2(lminusd);
        deriv2->d2Rdnu2       = -0.25*R / pow_2(nminusd);
        deriv2->d2Rdlambdadnu = -0.25*R / (lminusd * nminusd * sign);
        deriv2->d2zdlambda2   = -0.25*z / pow_2(p.lambda);
        deriv2->d2zdnu2       = -0.25*z / pow_2(p.nu);
        deriv2->d2zdlambdadnu = -0.25*z / (p.lambda * p.nu);
    }
    return PosCyl(R, z, p.phi);
}

template<>
PosProlSph toPosDeriv(const PosCyl& from, PosDerivT<Cyl, ProlSph>* deriv, PosDeriv2T<Cyl, ProlSph>* deriv2, const ProlSph cs)
{
    // lambda and nu are roots "t" of equation  R^2/(t-Delta^2) + z^2/t = 1
    double R2     = pow_2(from.R), z2 = pow_2(from.z);
    double signz  = from.z>=0 ? 1 : -1;   // nu will have the same sign as z
    double sum    = R2+z2+cs.Delta2;
    double dif    = R2+z2-cs.Delta2;
    double sqD    = sqrt(pow_2(dif) + 4*R2*cs.Delta2);   // determinant is always non-negative
    if(z2==0) sqD = sum;
    if(R2==0) sqD = fabs(dif);
    double lmd, dmn;  // lambda-Delta^2, Delta^2-|nu| - separately from lambda and nu, to avoid roundoffs
    if(dif >= 0) {
        lmd       = 0.5 * (sqD + dif);
        dmn       = R2>0 ? cs.Delta2 * R2 / lmd : 0;
    } else {
        dmn       = 0.5 * (sqD - dif);
        lmd       = cs.Delta2 * R2 / dmn;
    }
    double lambda = cs.Delta2 + lmd;
    double absnu  = 2 * cs.Delta2 / (sum + sqD) * z2;
    if(absnu*2 > cs.Delta2)             // compare |nu| and Delta^2-|nu|
        absnu     = cs.Delta2 - dmn;    // avoid roundoff errors when Delta^2-|nu| is small
    else
        dmn       = cs.Delta2 - absnu;  // same in the opposite case, when |nu| is small
    if(deriv!=NULL || deriv2!=NULL) {
        if(sqD==0)
            throw std::runtime_error("Error in coordinate conversion Cyl=>ProlSph: "
                "the special case lambda = nu = Delta^2 is not implemented");
        if(deriv!=NULL) {  // accurate expressions valid for arbitrary large/small values (no cancellations)
            deriv->dlambdadR = from.R * 2*lambda / sqD;
            deriv->dlambdadz = from.z * 2*lmd    / sqD;
            deriv->dnudR     = from.R * 2*-absnu / sqD * signz;
            deriv->dnudz     = from.z * 2*dmn    / sqD * signz;
        }
        if(deriv2!=NULL) {  // here no attempts were made to avoid cancellation errors
            double common = 8 * cs.Delta2 * R2 * z2 / pow_3(sqD);
            deriv2->d2lambdadR2 = 1 + sum/sqD - common;
            deriv2->d2lambdadz2 = 1 + dif/sqD + common;
            deriv2->d2nudR2     =(1 - sum/sqD + common) * signz;
            deriv2->d2nudz2     =(1 - dif/sqD - common) * signz;
            deriv2->d2lambdadRdz= 2 * from.R * from.z * (1 - sum * dif / pow_2(sqD)) / sqD;
            deriv2->d2nudRdz    = -deriv2->d2lambdadRdz * signz;
        }
    }
    return PosProlSph(lambda, absnu*signz, from.phi, cs);
}

template<>
PosCyl toPosDeriv(const PosAxi& p,
    PosDerivT<Axi, Cyl> *deriv, PosDeriv2T<Axi, Cyl> *deriv2, const Cyl)
{
    double
        eta      = sqrt(pow_2(p.rho) + fabs(p.cs.Delta2)),
        chi      = p.cs.Delta2 >= 0  ?  p.rho  :    eta,
        psi      = p.cs.Delta2 >= 0  ?    eta  :  p.rho,
        sinnu    = 1 / sqrt(1 + pow_2(p.cotnu)),
        cosnu    = sinnu!=0 ? p.cotnu * sinnu : (p.cotnu>0 ? 1 : -1),
        R        = mul(chi, sinnu),
        z        = mul(psi, cosnu),
        dchidrho = p.cs.Delta2 >= 0  ?  1  :  p.rho / chi,
        dpsidrho = p.cs.Delta2 <= 0  ?  1  :  p.rho / psi;
    if(deriv) {
        deriv->dRdrho = dchidrho * sinnu;
        deriv->dRdnu  = chi * cosnu;
        deriv->dzdrho = dpsidrho * cosnu;
        deriv->dzdnu  =-psi * sinnu;
    }
    if(deriv2) {
        deriv2->d2Rdrho2   = p.cs.Delta2 >= 0  ?  0  : -p.cs.Delta2 / pow_3(chi) * sinnu;
        deriv2->d2Rdrhodnu = dchidrho * cosnu;
        deriv2->d2Rdnu2    = -R;
        deriv2->d2zdrho2   = p.cs.Delta2 <= 0  ?  0  :  p.cs.Delta2 / pow_3(psi) * cosnu;
        deriv2->d2zdrhodnu = -dpsidrho * sinnu;
        deriv2->d2zdnu2    = -z;
    }
    return coord::PosCyl(R, z, p.phi);
}

// common fragment shared between toPosDeriv<Cyl, Axi> and toPosVel<Cyl, Axi>
inline void getPosAxi(const PosCyl& p, const Axi cs,
    double& chi, double& psi, double& cosnu, double& sinnu)
{
    double r2 = pow_2(p.R) + pow_2(p.z);
    double sum = 0.5 * (r2 + cs.Delta2);
    double dif = 0.5 * (r2 - cs.Delta2);
    // these branches select the more accurate way of computing quantities without cancellation,
    // but formally any choice is mathematically correct
    double det = cs.Delta2 >= 0 ?
        sqrt(pow_2(dif) + mul(pow_2(p.R), cs.Delta2)):
        sqrt(pow_2(sum) - mul(pow_2(p.z), cs.Delta2));
    // 2*det = (chi * cosnu)^2 + (psi * sinnu)^2 = chi^2 + D^2 sinnu^2 = psi^2 - D^2 cosnu^2
    if(det == INFINITY) {
        chi = psi = INFINITY;
        if(p.R == INFINITY) { cosnu = 0; sinnu = 1; }
        else { cosnu = p.z > 0 ? 1 : -1; sinnu = 0; }
        return;
    }
    if(sum >= 0) {
        psi   = sqrt(det + sum);
        cosnu = p.z!=0 ? p.z / psi : 0;
    } else {  // implies Delta^2 < 0, r < |Delta|
        cosnu = sqrt((det - sum) / -cs.Delta2) * (p.z>=0 ? 1 : -1);
        psi   = p.z / cosnu;
    }
    if(dif >= 0) {
        chi   = sqrt(det + dif);
        sinnu = p.R!=0 ? p.R / chi : 0;
    } else {  // implies Delta^2 > 0, r < Delta
        sinnu = sqrt((det - dif) / cs.Delta2);
        chi   = p.R / sinnu;
    }
}

template<>
PosAxi toPosDeriv(const PosCyl& p,
    PosDerivT<Cyl, Axi> *deriv, PosDeriv2T<Cyl, Axi> *deriv2, const Axi cs)
{
    double rho, eta, chi, psi, cosnu, sinnu;
    getPosAxi(p, cs, chi, psi, cosnu, sinnu);
    // this branch, by contrast, critically distinguishes between prolate and oblate cases;
    // rho = min(chi, psi) and eta = max(chi, psi)
    if(cs.Delta2 >= 0) {
        rho = chi;
        eta = psi;
    } else {
        eta = chi;
        rho = psi;
    }
    double M = 1 / (pow_2(chi * cosnu) + pow_2(psi * sinnu));  // = 1 / (2*det)
    if(deriv) {
        deriv->drhodR = M * psi * sinnu * eta;
        deriv->drhodz = M * chi * cosnu * eta;
        deriv->dnudR  = M * chi * cosnu;
        deriv->dnudz  =-M * psi * sinnu;
    }
    if(deriv2) {
        double common = pow_2(chi * chi * cosnu) + pow_2(psi * psi * sinnu) - 3 * pow_2(chi * psi);
        deriv2->d2rhodR2 = pow_2(M) * psi * (
            chi * eta * pow_2(cosnu) * (1 - 4*M * cs.Delta2 * pow_2(sinnu)) +
            (cs.Delta2 >= 0 ? 0 : cs.Delta2 * pow_2(sinnu) ) );
        deriv2->d2rhodz2 = pow_2(M) * chi * (
            psi * eta * pow_2(sinnu) * (1 + 4*M * cs.Delta2 * pow_2(cosnu)) -
            (cs.Delta2 <= 0 ? 0 : cs.Delta2 * pow_2(cosnu) ) );
        deriv2->d2rhodRdz= pow_2(M) * cosnu * sinnu * eta * (pow_2(rho) + M * common);
        deriv2->d2nudR2  = pow_3(M) * cosnu * sinnu * common;
        deriv2->d2nudz2  = -deriv2->d2nudR2;
        deriv2->d2nudRdz = pow_3(M) * psi * chi *
            (pow_2(psi * sinnu) * (3 - 2*pow_2(sinnu)) - pow_2(chi * cosnu) * (3 - 2*pow_2(cosnu)));
    }
    return PosAxi(rho, cosnu / sinnu, p.phi, cs);
}

// shortcuts for coordinate conversions without derivatives
template<> PosCyl toPos(const PosAxi& from, const Cyl) {
    return toPosDeriv<Axi, Cyl>(from, NULL);
}

template<> PosAxi toPos(const PosCyl& from, const Axi cs) {
    return toPosDeriv<Cyl, Axi>(from, NULL, NULL, cs);
}

//--------  position+velocity conversion functions  ---------//

template<> PosVelCar toPosVel(const PosVelCyl& p, const Car) {
    double sinphi, cosphi;
    math::sincos(p.phi, sinphi, cosphi);
    const double vx = p.vR * cosphi - p.vphi * sinphi;
    const double vy = p.vR * sinphi + p.vphi * cosphi;
    return PosVelCar(p.R * cosphi, p.R * sinphi, p.z, vx, vy, p.vz);
}

template<> PosVelCar toPosVel(const PosVelSph& p, const Car) {
    double sintheta, costheta, sinphi, cosphi;
    math::sincos(p.theta, sintheta, costheta);
    math::sincos(p.phi, sinphi, cosphi);
    const double R = p.r * sintheta, vR = p.vr * sintheta + p.vtheta * costheta;
    const double vx = vR * cosphi - p.vphi * sinphi;
    const double vy = vR * sinphi + p.vphi * cosphi;
    const double vz = p.vr * costheta - p.vtheta * sintheta;
    return PosVelCar(R * cosphi, R * sinphi, p.r * costheta, vx, vy, vz);
}

template<> PosVelCyl toPosVel(const PosVelCar& p, const Cyl) {
    const double R=sqrt(pow_2(p.x) + pow_2(p.y));
    if(R==0)  // determine phi from vy/vx rather than y/x
        return PosVelCyl(R, p.z, math::atan2(p.vy, p.vx), sqrt(pow_2(p.vx) + pow_2(p.vy)), p.vz, 0);
    const double cosphi = p.x / R, sinphi = p.y / R;
    const double vR   = p.vx * cosphi + p.vy * sinphi;
    const double vphi =-p.vx * sinphi + p.vy * cosphi;
    return PosVelCyl(R, p.z, math::atan2(p.y, p.x), vR, p.vz, vphi);
}

template<> PosVelCyl toPosVel(const PosVelSph& p, const Cyl) {
    double sintheta, costheta;
    math::sincos(p.theta, sintheta, costheta);
    const double R  = p.r  * sintheta, z = p.r * costheta;
    const double vR = p.vr * sintheta + p.vtheta * costheta;
    const double vz = p.vr * costheta - p.vtheta * sintheta;
    return PosVelCyl(R, z, p.phi, vR, vz, p.vphi);
}

template<> PosVelSph toPosVel(const PosVelCar& p, const Sph) {
    const double R2 = pow_2(p.x) + pow_2(p.y), R = sqrt(R2), invR = 1/R;
    const double r2 = R2 + pow_2(p.z), r = sqrt(r2), invr = 1/r;
    if(R==0) {  // point along the z axis - determine phi from velocity rather than position
        const double vR = sqrt(pow_2(p.vx) + pow_2(p.vy));
        const double phi = math::atan2(p.vy, p.vx);
        if(p.z==0)  // point at origin - an even more special case
            return PosVelSph(0, math::atan2(vR, p.vz), phi, sqrt(pow_2(vR) + pow_2(p.vz)), 0, 0);
        return PosVelSph(r, p.z>=0 ? 0 : M_PI, phi,
            p.vz * (p.z>=0 ? 1 : -1), vR * (p.z>=0 ? 1 : -1), 0);
    }
    const double temp   = p.x * p.vx + p.y * p.vy;
    const double vr     = (temp + p.z * p.vz) * invr;
    const double vtheta = (temp * p.z * invR - p.vz * R) * invr;
    const double vphi   = (p.x * p.vy - p.y * p.vx) * invR;
    return PosVelSph(r, math::atan2(R, p.z), math::atan2(p.y, p.x), vr, vtheta, vphi);
}

template<> PosVelSph toPosVel(const PosVelCyl& p, const Sph) {
    const double r=sqrt(pow_2(p.R) + pow_2(p.z));
    if(r==0) {
        return PosVelSph(0, math::atan2(p.vR, p.vz), p.phi, sqrt(pow_2(p.vR) + pow_2(p.vz)), 0, 0);
    }
    const double invr = 1./r;
    const double costheta = p.z * invr, sintheta = p.R * invr;
    const double vr = p.vR * sintheta + p.vz * costheta;
    const double vtheta = p.vR * costheta - p.vz * sintheta;
    return PosVelSph(r, math::atan2(p.R, p.z), p.phi, vr, vtheta, p.vphi);
}

template<> PosVelProlSph toPosVel(const PosVelCyl& from, const ProlSph cs) {
    PosDerivT<Cyl, ProlSph> derivs;
    const PosProlSph pprol = toPosDeriv<Cyl, ProlSph> (from, &derivs, NULL, cs);
    double lambdadot = derivs.dlambdadR*from.vR + derivs.dlambdadz*from.vz;
    double nudot     = derivs.dnudR    *from.vR + derivs.dnudz    *from.vz;
    double phidot    = from.vphi!=0 ? from.vphi/from.R : 0;
    return PosVelProlSph(pprol, lambdadot, nudot, phidot);
}

template<> PosVelAxi toPosVel(const PosVelCyl& pc, const Axi cs) {
    if(cs.Delta2 == 0) {  // shortcut for the spherical case
        if(pc.R == 0 && pc.z == 0) {
            // degenerate case - cannot determine nu  from the position alone, use velocity instead
            double vel   = sqrt(pow_2(pc.vR) + pow_2(pc.vz));
            double cotnu = vel>0 ? pc.vz / pc.vR : 0;
            return PosVelAxi(PosAxi(0, cotnu, pc.phi, cs), VelAxi(vel, 0, pc.vphi));
        }
        double rho  = sqrt(pow_2(pc.R) + pow_2(pc.z));
        double cosnu= pc.z / rho, sinnu = pc.R / rho;
        double vrho = sinnu * pc.vR + cosnu * pc.vz;
        double vnu  = cosnu * pc.vR - sinnu * pc.vz;
        return PosVelAxi(PosAxi(rho, cosnu / sinnu, pc.phi, cs), VelAxi(vrho, vnu, pc.vphi));
    }
    double chi, psi, cosnu, sinnu;
    getPosAxi(pc, cs, chi, psi, cosnu, sinnu);
    double
    rho  = cs.Delta2 >= 0  ? chi : psi,
    den  = 1 / sqrt(pow_2(chi * cosnu) + pow_2(psi * sinnu)),
    sinxi= den != INFINITY ? den * psi * sinnu : 1.0,
    cosxi= den != INFINITY ? den * chi * cosnu : 0.0,
    vrho = sinxi * pc.vR + cosxi * pc.vz,
    vnu  = cosxi * pc.vR - sinxi * pc.vz;
    return PosVelAxi(PosAxi(rho, cosnu / sinnu, pc.phi, cs), VelAxi(vrho, vnu, pc.vphi));
}

template<> PosVelCyl toPosVel(const PosVelAxi& ps, const Cyl) {
    double
    sinnu = 1 / sqrt(1 + pow_2(ps.cotnu)),
    cosnu = sinnu!=0 ? ps.cotnu * sinnu : (ps.cotnu>0 ? 1 : -1);
    if(ps.cs.Delta2 == 0) {  // shortcut for the spherical case
        if(ps.rho == 0)  // assume that vnu = 0
            return PosVelCyl(0, 0, ps.phi, ps.vrho * sinnu, ps.vrho * cosnu, ps.vphi);
        double R  = ps.rho  * sinnu, z = ps.rho * cosnu;
        double vR = ps.vrho * sinnu + ps.vnu * cosnu;
        double vz = ps.vrho * cosnu - ps.vnu * sinnu;
        return PosVelCyl(R, z, ps.phi, vR, vz, ps.vphi);
    }
    double
    eta  = sqrt(pow_2(ps.rho) + fabs(ps.cs.Delta2)),
    chi  = ps.cs.Delta2 >= 0  ?  ps.rho  :     eta,
    psi  = ps.cs.Delta2 >= 0  ?     eta  :  ps.rho,
    R    = chi * sinnu,
    z    = psi * cosnu,
    den  = 1 / sqrt(pow_2(chi * cosnu) + pow_2(psi * sinnu)),
    sinxi= den != INFINITY ? den * psi * sinnu : 1.0,
    cosxi= den != INFINITY ? den * chi * cosnu : 0.0,
    vR   = sinxi * ps.vrho + cosxi * ps.vnu,
    vz   = cosxi * ps.vrho - sinxi * ps.vnu;
    return PosVelCyl(R, z, ps.phi, vR, vz, ps.vphi);
}

void PosVelSph::momenta(double& pr, double& ptheta, double& pphi) const
{
    pr     = vr;
    ptheta = vtheta*r;
    pphi   = Lz(*this);
}

void PosVelAxi::momenta(double& prho, double& pnu, double& pphi) const
{
    double
    sinnu= 1 / sqrt(1 + pow_2(cotnu)),
    eta  = sqrt(pow_2(rho) + fabs(cs.Delta2)),
    chi  = cs.Delta2 >= 0  ?  rho  :  eta,
    mul  = sqrt(pow_2(chi) + cs.Delta2 * pow_2(sinnu));
    prho = (cs.Delta2 == 0 ? 1 : mul / eta) * vrho;
    pnu  = mul * vnu;
    pphi = chi * sinnu * vphi;
}

//-------- implementations of functions that convert gradients --------//

template<>
GradCar toGrad(const GradCyl& src, const PosDerivT<Car, Cyl>& deriv) {
    GradCar dest;
    dest.dx = src.dR*deriv.dRdx + src.dphi*deriv.dphidx;
    dest.dy = src.dR*deriv.dRdy + src.dphi*deriv.dphidy;
    dest.dz = src.dz;
    return dest;
}

template<>
GradCar toGrad(const GradSph& src, const PosDerivT<Car, Sph>& deriv) {
    GradCar dest;
    dest.dx = src.dr*deriv.drdx + src.dtheta*deriv.dthetadx + src.dphi*deriv.dphidx;
    dest.dy = src.dr*deriv.drdy + src.dtheta*deriv.dthetady + src.dphi*deriv.dphidy;
    dest.dz = src.dr*deriv.drdz + src.dtheta*deriv.dthetadz;
    return dest;
}

template<>
GradCyl toGrad(const GradCar& src, const PosDerivT<Cyl, Car>& deriv) {
    GradCyl dest;
    dest.dR = src.dx*deriv.dxdR + src.dy*deriv.dydR;
    dest.dz = src.dz;
    dest.dphi = src.dx*deriv.dxdphi + src.dy*deriv.dydphi;
    return dest;
}

template<>
GradCyl toGrad(const GradSph& src, const PosDerivT<Cyl, Sph>& deriv) {
    GradCyl dest;
    dest.dR = src.dr*deriv.drdR + src.dtheta*deriv.dthetadR;
    dest.dz = src.dr*deriv.drdz + src.dtheta*deriv.dthetadz;
    dest.dphi = src.dphi;
    return dest;
}

template<>
GradSph toGrad(const GradCar& src, const PosDerivT<Sph, Car>& deriv) {
    GradSph dest;
    dest.dr     = src.dx*deriv.dxdr     + src.dy*deriv.dydr     + src.dz*deriv.dzdr;
    dest.dtheta = src.dx*deriv.dxdtheta + src.dy*deriv.dydtheta + src.dz*deriv.dzdtheta;
    dest.dphi   = src.dx*deriv.dxdphi   + src.dy*deriv.dydphi;
    return dest;
}

template<>
GradSph toGrad(const GradCyl& src, const PosDerivT<Sph, Cyl>& deriv) {
    GradSph dest;
    dest.dr     = src.dR*deriv.dRdr     + src.dz*deriv.dzdr;
    dest.dtheta = src.dR*deriv.dRdtheta + src.dz*deriv.dzdtheta;
    dest.dphi   = src.dphi;
    return dest;
}

template<>
GradCyl toGrad(const GradProlSph& src, const PosDerivT<Cyl, ProlSph>& deriv) {
    GradCyl dest;
    dest.dR   = src.dlambda*deriv.dlambdadR + src.dnu*deriv.dnudR;
    dest.dz   = src.dlambda*deriv.dlambdadz + src.dnu*deriv.dnudz;
    dest.dphi = src.dphi;
    return dest;
}

template<>
GradProlSph toGrad(const GradCyl& src, const PosDerivT<ProlSph, Cyl>& deriv) {
    GradProlSph dest;
    dest.dlambda = src.dR*deriv.dRdlambda + src.dz*deriv.dzdlambda;
    dest.dnu     = src.dR*deriv.dRdnu     + src.dz*deriv.dzdnu;
    dest.dphi    = src.dphi;
    return dest;
}

template<>
GradCyl toGrad(const GradAxi& src, const PosDerivT<Cyl, Axi>& deriv) {
    GradCyl dest;
    dest.dR   = src.drho*deriv.drhodR + src.dnu*deriv.dnudR;
    dest.dz   = src.drho*deriv.drhodz + src.dnu*deriv.dnudz;
    dest.dphi = src.dphi;
    return dest;
}

template<>
GradAxi toGrad(const GradCyl& src, const PosDerivT<Axi, Cyl>& deriv) {
    GradAxi dest;
    dest.drho = src.dR*deriv.dRdrho + src.dz*deriv.dzdrho;
    dest.dnu  = src.dR*deriv.dRdnu  + src.dz*deriv.dzdnu;
    dest.dphi = src.dphi;
    return dest;
}

//-------- implementations of functions that convert hessians --------//

template<>
HessCar toHess(const GradCyl& srcGrad, const HessCyl& srcHess,
    const PosDerivT<Car, Cyl>& deriv, const PosDeriv2T<Car, Cyl>& deriv2) {
    HessCar dest;
    dest.dx2 =
        (srcHess.dR2   *deriv.dRdx + srcHess.dRdphi*deriv.dphidx) * deriv.dRdx +
        (srcHess.dRdphi*deriv.dRdx + srcHess.dphi2 *deriv.dphidx) * deriv.dphidx +
        srcGrad.dR*deriv2.d2Rdx2   + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy =
        (srcHess.dR2   *deriv.dRdy + srcHess.dRdphi*deriv.dphidy) * deriv.dRdx +
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2 *deriv.dphidy) * deriv.dphidx +
        srcGrad.dR*deriv2.d2Rdxdy  + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dy2 =
        (srcHess.dR2   *deriv.dRdy + srcHess.dRdphi*deriv.dphidy) * deriv.dRdy +
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2 *deriv.dphidy) * deriv.dphidy +
        srcGrad.dR*deriv2.d2Rdy2   + srcGrad.dphi*deriv2.d2phidy2;
    dest.dxdz = srcHess.dRdz*deriv.dRdx + srcHess.dzdphi*deriv.dphidx;
    dest.dydz = srcHess.dRdz*deriv.dRdy + srcHess.dzdphi*deriv.dphidy;
    dest.dz2  = srcHess.dz2;
    return dest;
}

template<>
HessCar toHess(const GradSph& srcGrad, const HessSph& srcHess,
    const PosDerivT<Car, Sph>& deriv, const PosDeriv2T<Car, Sph>& deriv2) {
    HessCar dest;
    dest.dx2 =
        (srcHess.dr2     *deriv.drdx + srcHess.drdtheta  *deriv.dthetadx + srcHess.drdphi    *deriv.dphidx) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdx + srcHess.dtheta2   *deriv.dthetadx + srcHess.dthetadphi*deriv.dphidx) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdx + srcHess.dthetadphi*deriv.dthetadx + srcHess.dphi2     *deriv.dphidx) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdx2     + srcGrad.dtheta*deriv2.d2thetadx2  + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy =
        (srcHess.dr2     *deriv.drdy + srcHess.drdtheta  *deriv.dthetady + srcHess.drdphi    *deriv.dphidy) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2   *deriv.dthetady + srcHess.dthetadphi*deriv.dphidy) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2     *deriv.dphidy) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdxdy    + srcGrad.dtheta*deriv2.d2thetadxdy + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dxdz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdx +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetadx +
        (srcHess.drdphi  *deriv.drdz + srcHess.dthetadphi*deriv.dthetadz) * deriv.dphidx +
        srcGrad.dr*deriv2.d2rdxdz    + srcGrad.dtheta*deriv2.d2thetadxdz;
    dest.dy2 =
        (srcHess.dr2     *deriv.drdy + srcHess.drdtheta  *deriv.dthetady + srcHess.drdphi    *deriv.dphidy) * deriv.drdy +
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2   *deriv.dthetady + srcHess.dthetadphi*deriv.dphidy) * deriv.dthetady +
        (srcHess.drdphi  *deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2     *deriv.dphidy) * deriv.dphidy +
        srcGrad.dr*deriv2.d2rdy2     + srcGrad.dtheta*deriv2.d2thetady2  + srcGrad.dphi*deriv2.d2phidy2;
    dest.dydz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdy +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetady +
        (srcHess.drdphi  *deriv.drdz + srcHess.dthetadphi*deriv.dthetadz) * deriv.dphidy +
        srcGrad.dr*deriv2.d2rdydz    + srcGrad.dtheta*deriv2.d2thetadydz;
    dest.dz2 =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta  *deriv.dthetadz) * deriv.drdz +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2   *deriv.dthetadz) * deriv.dthetadz +
        srcGrad.dr*deriv2.d2rdz2     + srcGrad.dtheta*deriv2.d2thetadz2;
    return dest;
}

template<>
HessCyl toHess(const GradCar& srcGrad, const HessCar& srcHess,
    const PosDerivT<Cyl, Car>& deriv, const PosDeriv2T<Cyl, Car>& deriv2) {
    HessCyl dest;
    dest.dR2 =
        (srcHess.dx2 *deriv.dxdR + srcHess.dxdy*deriv.dydR) * deriv.dxdR +
        (srcHess.dxdy*deriv.dxdR + srcHess.dy2 *deriv.dydR) * deriv.dydR;
    dest.dRdz = srcHess.dxdz*deriv.dxdR + srcHess.dydz*deriv.dydR;
    dest.dRdphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdR +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydR +
        srcGrad.dx*deriv2.d2xdRdphi + srcGrad.dy*deriv2.d2ydRdphi;
    dest.dz2 = srcHess.dz2;
    dest.dzdphi = (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi);
    dest.dphi2 =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdphi +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydphi +
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

template<>
HessCyl toHess(const GradSph& srcGrad, const HessSph& srcHess,
    const PosDerivT<Cyl, Sph>& deriv, const PosDeriv2T<Cyl, Sph>& deriv2) {
    HessCyl dest;
    dest.dR2 =
        (srcHess.dr2     *deriv.drdR + srcHess.drdtheta*deriv.dthetadR) * deriv.drdR +
        (srcHess.drdtheta*deriv.drdR + srcHess.dtheta2 *deriv.dthetadR) * deriv.dthetadR +
        srcGrad.dr*deriv2.d2rdR2 + srcGrad.dtheta*deriv2.d2thetadR2;
    dest.dRdz =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta*deriv.dthetadz) * deriv.drdR +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2 *deriv.dthetadz) * deriv.dthetadR +
        srcGrad.dr*deriv2.d2rdRdz + srcGrad.dtheta*deriv2.d2thetadRdz;
    dest.dz2 =
        (srcHess.dr2     *deriv.drdz + srcHess.drdtheta*deriv.dthetadz) * deriv.drdz +
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2 *deriv.dthetadz) * deriv.dthetadz +
        srcGrad.dr*deriv2.d2rdz2 + srcGrad.dtheta*deriv2.d2thetadz2;
    dest.dRdphi = srcHess.drdphi*deriv.drdR + srcHess.dthetadphi*deriv.dthetadR;
    dest.dzdphi = srcHess.drdphi*deriv.drdz + srcHess.dthetadphi*deriv.dthetadz;
    dest.dphi2  = srcHess.dphi2;
    return dest;
}

template<>
HessSph toHess(const GradCar& srcGrad, const HessCar& srcHess,
    const PosDerivT<Sph, Car>& deriv, const PosDeriv2T<Sph, Car>& deriv2) {
    HessSph dest;
    dest.dr2 =
        (srcHess.dx2 *deriv.dxdr + srcHess.dxdy*deriv.dydr + srcHess.dxdz*deriv.dzdr) * deriv.dxdr +
        (srcHess.dxdy*deriv.dxdr + srcHess.dy2 *deriv.dydr + srcHess.dydz*deriv.dzdr) * deriv.dydr +
        (srcHess.dxdz*deriv.dxdr + srcHess.dydz*deriv.dydr + srcHess.dz2 *deriv.dzdr) * deriv.dzdr;
    dest.drdtheta =
        (srcHess.dx2 *deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta) * deriv.dxdr +
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2 *deriv.dydtheta + srcHess.dydz*deriv.dzdtheta) * deriv.dydr +
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdr +
        srcGrad.dx*deriv2.d2xdrdtheta + srcGrad.dy*deriv2.d2ydrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.drdphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdr +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi)*deriv.dydr +
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi)*deriv.dzdr +
        srcGrad.dx*deriv2.d2xdrdphi + srcGrad.dy*deriv2.d2ydrdphi;
    dest.dtheta2 =
        (srcHess.dx2 *deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta) * deriv.dxdtheta +
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2 *deriv.dydtheta + srcHess.dydz*deriv.dzdtheta) * deriv.dydtheta +
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdtheta +
        srcGrad.dx*deriv2.d2xdtheta2 + srcGrad.dy*deriv2.d2ydtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.dthetadphi =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdtheta +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydtheta +
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi) * deriv.dzdtheta +
        srcGrad.dx*deriv2.d2xdthetadphi + srcGrad.dy*deriv2.d2ydthetadphi;
    dest.dphi2 =
        (srcHess.dx2 *deriv.dxdphi + srcHess.dxdy*deriv.dydphi) * deriv.dxdphi +
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2 *deriv.dydphi) * deriv.dydphi +
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

template<>
HessSph toHess(const GradCyl& srcGrad, const HessCyl& srcHess,
    const PosDerivT<Sph, Cyl>& deriv, const PosDeriv2T<Sph, Cyl>& deriv2) {
    HessSph dest;
    dest.dr2 =
        (srcHess.dR2 *deriv.dRdr + srcHess.dRdz*deriv.dzdr) * deriv.dRdr +
        (srcHess.dRdz*deriv.dRdr + srcHess.dz2 *deriv.dzdr) * deriv.dzdr;
    dest.drdtheta =
        (srcHess.dR2 *deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta) * deriv.dRdr +
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdr +
        srcGrad.dR*deriv2.d2Rdrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.dtheta2 =
        (srcHess.dR2 *deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta) * deriv.dRdtheta +
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2 *deriv.dzdtheta) * deriv.dzdtheta +
        srcGrad.dR*deriv2.d2Rdtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.drdphi     = srcHess.dRdphi*deriv.dRdr     + srcHess.dzdphi*deriv.dzdr;
    dest.dthetadphi = srcHess.dRdphi*deriv.dRdtheta + srcHess.dzdphi*deriv.dzdtheta;
    dest.dphi2      = srcHess.dphi2;
    return dest;
}

//TODO// remove
template<>
HessCyl toHess(const GradProlSph& srcGrad, const HessProlSph& srcHess,
    const PosDerivT<Cyl, ProlSph>& deriv, const PosDeriv2T<Cyl, ProlSph>& deriv2) {
    HessCyl dest;
    dest.dR2 =
        (srcHess.dlambda2*deriv.dlambdadR + srcHess.dlambdadnu*deriv.dnudR)*deriv.dlambdadR +
        (srcHess.dlambdadnu*deriv.dlambdadR + srcHess.dnu2*deriv.dnudR)*deriv.dnudR +
        srcGrad.dlambda*deriv2.d2lambdadR2 + srcGrad.dnu*deriv2.d2nudR2;
    dest.dRdz =
        (srcHess.dlambda2*deriv.dlambdadz + srcHess.dlambdadnu*deriv.dnudz)*deriv.dlambdadR +
        (srcHess.dlambdadnu*deriv.dlambdadz + srcHess.dnu2*deriv.dnudz)*deriv.dnudR +
        srcGrad.dlambda*deriv2.d2lambdadRdz + srcGrad.dnu*deriv2.d2nudRdz;
    dest.dz2 =
        (srcHess.dlambda2*deriv.dlambdadz + srcHess.dlambdadnu*deriv.dnudz)*deriv.dlambdadz +
        (srcHess.dlambdadnu*deriv.dlambdadz + srcHess.dnu2*deriv.dnudz)*deriv.dnudz +
        srcGrad.dlambda*deriv2.d2lambdadz2 + srcGrad.dnu*deriv2.d2nudz2;
    dest.dRdphi = dest.dzdphi = dest.dphi2 = 0;  //TODO// assuming no dependence on phi
    return dest;
}

template<>
HessCyl toHess(const GradAxi& srcGrad, const HessAxi& srcHess,
    const PosDerivT<Cyl, Axi>& deriv, const PosDeriv2T<Cyl, Axi>& deriv2) {
    HessCyl dest;
    dest.dR2 =
        (srcHess.drho2  *deriv.drhodR + srcHess.drhodnu*deriv.dnudR) * deriv.drhodR +
        (srcHess.drhodnu*deriv.drhodR + srcHess.dnu2   *deriv.dnudR) * deriv.dnudR  +
        srcGrad.drho *deriv2.d2rhodR2 + srcGrad.dnu * deriv2.d2nudR2;
    dest.dRdz =
        (srcHess.drho2  *deriv.drhodz + srcHess.drhodnu*deriv.dnudz) * deriv.drhodR +
        (srcHess.drhodnu*deriv.drhodz + srcHess.dnu2   *deriv.dnudz) * deriv.dnudR  +
        srcGrad.drho *deriv2.d2rhodRdz+ srcGrad.dnu * deriv2.d2nudRdz;
    dest.dz2 =
        (srcHess.drho2  *deriv.drhodz + srcHess.drhodnu*deriv.dnudz) * deriv.drhodz +
        (srcHess.drhodnu*deriv.drhodz + srcHess.dnu2   *deriv.dnudz) * deriv.dnudz  +
        srcGrad.drho *deriv2.d2rhodz2 + srcGrad.dnu * deriv2.d2nudz2;
    dest.dRdphi = srcHess.drhodphi*deriv.drhodR + srcHess.dnudphi*deriv.dnudR;
    dest.dzdphi = srcHess.drhodphi*deriv.drhodz + srcHess.dnudphi*deriv.dnudz;
    dest.dphi2  = srcHess.dphi2;
    return dest;
}

template<>
HessAxi toHess(const GradCyl& srcGrad, const HessCyl& srcHess,
    const PosDerivT<Axi, Cyl>& deriv, const PosDeriv2T<Axi, Cyl>& deriv2) {
    HessAxi dest;
    dest.drho2 =
        (srcHess.dR2 *deriv.dRdrho + srcHess.dRdz*deriv.dzdrho) * deriv.dRdrho +
        (srcHess.dRdz*deriv.dRdrho + srcHess.dz2 *deriv.dzdrho) * deriv.dzdrho +
        srcGrad.dR*deriv2.d2Rdrho2 + srcGrad.dz*deriv2.d2zdrho2;
    dest.drhodnu =
        (srcHess.dR2 *deriv.dRdnu + srcHess.dRdz*deriv.dzdnu) * deriv.dRdrho +
        (srcHess.dRdz*deriv.dRdnu + srcHess.dz2 *deriv.dzdnu) * deriv.dzdrho +
        srcGrad.dR*deriv2.d2Rdrhodnu + srcGrad.dz*deriv2.d2zdrhodnu;
    dest.dnu2 =
        (srcHess.dR2 *deriv.dRdnu + srcHess.dRdz*deriv.dzdnu) * deriv.dRdnu +
        (srcHess.dRdz*deriv.dRdnu + srcHess.dz2 *deriv.dzdnu) * deriv.dzdnu +
        srcGrad.dR*deriv2.d2Rdnu2 + srcGrad.dz*deriv2.d2zdnu2;
    dest.drhodphi = srcHess.dRdphi*deriv.dRdrho + srcHess.dzdphi*deriv.dzdrho;
    dest.dnudphi  = srcHess.dRdphi*deriv.dRdnu  + srcHess.dzdphi*deriv.dzdnu;
    dest.dphi2    = srcHess.dphi2;
    return dest;
}

//------ conversion of derivatives of f(r) into gradients/hessians in different coord.sys. ------//
template<>
void evalAndConvertSph(const math::IFunction& F,
    const PosCar& pos, double* value, GradCar* deriv, HessCar* deriv2)
{
    assert(F.numDerivs()>=2);
    const double r=sqrt(pow_2(pos.x)+pow_2(pos.y)+pow_2(pos.z));
    if(deriv==NULL && deriv2==NULL) {
        F.evalDeriv(r, value, NULL, NULL);
        return;
    }
    double der, der2;
    F.evalDeriv(r, value, &der, deriv2!=NULL ? &der2 : NULL);
    double x_over_r=pos.x/r, y_over_r=pos.y/r, z_over_r=pos.z/r;
    if(r==0) {
        x_over_r=y_over_r=z_over_r=0;
    }
    if(deriv) {
        deriv->dx = x_over_r*der;
        deriv->dy = y_over_r*der;
        deriv->dz = z_over_r*der;
    }
    if(deriv2) {
        double der_over_r=der/r, dd=der2-der_over_r;
        if(r==0) {
            dd=0;
            if(der==0) der_over_r=der2;
        }
        deriv2->dx2 = pow_2(x_over_r)*dd + der_over_r;
        deriv2->dy2 = pow_2(y_over_r)*dd + der_over_r;
        deriv2->dz2 = pow_2(z_over_r)*dd + der_over_r;
        deriv2->dxdy= x_over_r*y_over_r*dd;
        deriv2->dydz= y_over_r*z_over_r*dd;
        deriv2->dxdz= x_over_r*z_over_r*dd;
    }
}

template<>
void evalAndConvertSph(const math::IFunction& F,
    const PosCyl& pos, double* value, GradCyl* deriv, HessCyl* deriv2)
{
    assert(F.numDerivs()>=2);
    const double r=sqrt(pow_2(pos.R)+pow_2(pos.z));
    if(deriv==NULL && deriv2==NULL) {
        F.evalDeriv(r, value, NULL, NULL);
        return;
    }
    double der, der2;
    F.evalDeriv(r, value, &der, deriv2!=NULL ? &der2 : NULL);
    double R_over_r=pos.R/r, z_over_r=pos.z/r;
    if(r==0) {
        R_over_r=z_over_r=0;
    }
    if(deriv) {
        deriv->dR = R_over_r*der;
        deriv->dz = z_over_r*der;
        deriv->dphi = 0;
    }
    if(deriv2) {
        double der_over_r=der/r, dd=der2-der_over_r;
        if(r==0) {
            dd=0;
            if(der==0) der_over_r=der2;
        }
        deriv2->dR2 = pow_2(R_over_r)*dd + der_over_r;
        deriv2->dz2 = pow_2(z_over_r)*dd + der_over_r;
        deriv2->dRdz= R_over_r*z_over_r*dd;
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

template<>
void evalAndConvertSph(const math::IFunction& F,
    const PosSph& pos, double* value, GradSph* deriv, HessSph* deriv2)
{
    assert(F.numDerivs()>=2);
    double der, der2;
    F.evalDeriv(pos.r, value, deriv!=NULL ? &der : NULL, deriv2!=NULL ? &der2 : NULL);
    if(deriv) {
        deriv->dr = der;
        deriv->dtheta = deriv->dphi = 0;
    }
    if(deriv2) {
        deriv2->dr2 = der2;
        deriv2->dtheta2 = deriv2->dphi2 = deriv2->drdtheta = deriv2->drdphi = deriv2->dthetadphi = 0;
    }
}


//------ 3x3 matrix representing a [passive] rotation specified by Euler angles ------//

Orientation::Orientation(double alpha, double beta, double gamma)
{
    double sa, ca, sb, cb, sc, cc;
    math::sincos(alpha, sa, ca);
    math::sincos(beta,  sb, cb);
    math::sincos(gamma, sc, cc);
    mat[0] =  ca * cc - sa * cb * sc;
    mat[1] =  sa * cc + ca * cb * sc;
    mat[2] =  sb * sc;
    mat[3] = -ca * sc - sa * cb * cc;
    mat[4] = -sa * sc + ca * cb * cc;
    mat[5] =  sb * cc;
    mat[6] =  sa * sb;
    mat[7] = -ca * sb;
    mat[8] =  cb;
}

void Orientation::toEulerAngles(double& alpha, double& beta, double& gamma) const
{
    // beta is between 0 and pi; sin(beta) >= 0,
    // and while beta=acos(mat[8]) is mathematically correct, it has poor accuracy
    // when beta is close to 0 or pi; the alternative formula with atan2 works in all cases.
    beta  = math::atan2(sqrt(pow_2(mat[2]) + pow_2(mat[5])), mat[8]);
    if(mat[2] == 0 && mat[5] == 0 && mat[6] == 0 && mat[7] == 0) {
        // degenerate case: beta=0 or beta=pi;
        // in this case we can determine only the sum or the difference of the other two angles
        alpha = 0;
        gamma = math::atan2(mat[1], mat[0]) * (mat[8]>0 ? 1 : -1);
    } else {
        gamma = math::atan2(mat[2], mat[5]);
        alpha = math::atan2(mat[6],-mat[7]);
    }
}

}  // namespace coord
