#include "coord.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace coord{

//--------  angular momentum functions --------//

template<> double Ltotal(const PosVelCar& p) {
    return sqrt(pow_2(p.y*p.vz-p.z*p.vy) + pow_2(p.z*p.vx-p.x*p.vz) + pow_2(p.x*p.vy-p.y*p.vx));
}
template<> double Ltotal(const PosVelCyl& p) {
    return sqrt((pow_2(p.R)+pow_2(p.z))*pow_2(p.vphi) + pow_2(p.R*p.vz-p.z*p.vR));
}
template<> double Ltotal(const PosVelSph& p) {
    return sqrt(pow_2(p.vtheta)+pow_2(p.vphi))*p.r;
}
template<> double Lz(const PosVelCar& p) { return p.x*p.vy-p.y*p.vx; }
template<> double Lz(const PosVelCyl& p) { return p.R*p.vphi; }
template<> double Lz(const PosVelSph& p) { return p.r*sin(p.theta)*p.vphi; }
// now the stupid part: spell out these names once again but without <>,
// to make sure they are included in the obj file and you don't get linker errors:
// the above were template specializations, and now we make template instantiations...
template double Ltotal(const PosVelCar& p);
template double Ltotal(const PosVelCyl& p);
template double Ltotal(const PosVelSph& p);
template double Lz(const PosVelCar& p);
template double Lz(const PosVelCyl& p);
template double Lz(const PosVelSph& p);

//--------  position conversion functions ---------//

template<> PosCar toPos(const PosCyl& pos) {
    return PosCar(pos.R*cos(pos.phi), pos.R*sin(pos.phi), pos.z);
}
template<> PosCar toPos(const PosSph& pos) {
    const double R=pos.r*sin(pos.theta);
    return PosCar(R*cos(pos.phi), R*sin(pos.phi), pos.r*cos(pos.theta)); 
}
template<> PosCyl toPos(const PosCar& pos) {
    return PosCyl(sqrt(pow_2(pos.x)+pow_2(pos.y)), pos.z, atan2(pos.y, pos.x));
}
template<> PosCyl toPos(const PosSph& pos) {
    return PosCyl(pos.r*sin(pos.theta), pos.r*cos(pos.theta), pos.phi);
}
template<> PosSph toPos(const PosCar& pos) {
    return PosSph(sqrt(pow_2(pos.x)+pow_2(pos.y)+pow_2(pos.z)), 
        atan2(sqrt(pow_2(pos.x)+pow_2(pos.y)), pos.z), atan2(pos.y, pos.x));
}
template<> PosSph toPos(const PosCyl& pos) {
    return PosSph(sqrt(pow_2(pos.R)+pow_2(pos.z)), atan2(pos.R, pos.z), pos.phi);
}
template<> PosCyl toPos(const PosProlSph& p) {
    const double R = sqrt((p.lambda+p.coordsys.alpha)*(p.nu+p.coordsys.alpha)/
        (p.coordsys.alpha-p.coordsys.gamma));
    const double z = sqrt((p.lambda+p.coordsys.gamma)*(p.nu+p.coordsys.gamma)/
        (p.coordsys.gamma-p.coordsys.alpha));
    return PosCyl(R, z, p.phi);
}
template<> PosProlSph toPos(const PosCyl& from, const ProlSph& cs) {
    // lambda and mu are roots "t" of equation  R^2/(t+alpha) + z^2/(t+gamma) = 1
    double R2=pow_2(from.R), z2=pow_2(from.z);
    double b = cs.alpha+cs.gamma - R2 - z2;
    double c = cs.alpha*cs.gamma - R2*cs.gamma - z2*cs.alpha;
    double det = b*b-4*c;
    if(det<=0)
        throw std::invalid_argument("Error in coordinate conversion Cyl=>ProlSph: det<=0");
    double sqD=sqrt(det);
    // lambda and mu are roots of quadratic equation  t^2+b*t+c=0
    double lambda = 0.5*(-b+sqD);
    double nu     = 0.5*(-b-sqD);
    lambda=fmax(-cs.alpha, lambda);
    nu=fmin(-cs.alpha, fmax(nu, -cs.gamma));  // prevent roundoff errors
    return PosProlSph(lambda, nu, from.phi, cs);
}
// instantiations...
template PosCar toPos(const PosCyl&);
template PosCar toPos(const PosCyl&);
template PosCyl toPos(const PosCar&);
template PosCyl toPos(const PosSph&);
template PosSph toPos(const PosCar&);
template PosSph toPos(const PosCyl&);
template PosCyl toPos(const PosProlSph&);
template PosProlSph toPos(const PosCyl&, const ProlSph&);

//-------- position conversion with derivatives --------//

template<>
PosCyl toPosDeriv(const PosCar& p, PosDerivT<Car, Cyl>* deriv, PosDeriv2T<Car, Cyl>* deriv2) 
{
    const double R2=pow_2(p.x)+pow_2(p.y), R=sqrt(R2);
    if(R==0)
        throw std::runtime_error("PosDeriv Car=>Cyl: R=0, degenerate case!");
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
    return PosCyl(R, p.z, atan2(p.y, p.x));
};

template<>
PosSph toPosDeriv(const PosCar& p, PosDerivT<Car, Sph>* deriv, PosDeriv2T<Car, Sph>* deriv2) {
    const double x2=pow_2(p.x), y2=pow_2(p.y), z2=pow_2(p.z);
    const double R2=x2+y2, R=sqrt(R2);
    const double r2=R2+z2, r=sqrt(r2), invr=1/r;
    if(R==0)
        throw std::runtime_error("PosDeriv Car=>Sph: R=0, degenerate case!");
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
    return PosSph(r, atan2(R, p.z), atan2(p.y, p.x));
}

template<>
PosCar toPosDeriv(const PosCyl& p, PosDerivT<Cyl, Car>* deriv, PosDeriv2T<Cyl, Car>* deriv2) {
    const double cosphi=cos(p.phi), sinphi=sin(p.phi);
    const double x=p.R*cosphi, y=p.R*sinphi;
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
PosSph toPosDeriv(const PosCyl& p, PosDerivT<Cyl, Sph>* deriv, PosDeriv2T<Cyl, Sph>* deriv2) {
    const double r=sqrt(pow_2(p.R)+pow_2(p.z));
    if(r==0)
        throw std::runtime_error("PosDeriv Cyl=>Sph: r=0, degenerate case!");
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
    return PosSph(r, atan2(p.R, p.z), p.phi);
};

template<>
PosCar toPosDeriv(const PosSph& p, PosDerivT<Sph, Car>* deriv, PosDeriv2T<Sph, Car>* deriv2) {
    const double sintheta=sin(p.theta), costheta=cos(p.theta);
    const double sinphi=sin(p.phi), cosphi=cos(p.phi);
    const double R=p.r*sintheta, x=R*cosphi, y=R*sinphi, z=p.r*costheta;
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
};

template<>
PosCyl toPosDeriv(const PosSph& p, PosDerivT<Sph, Cyl>* deriv, PosDeriv2T<Sph, Cyl>* deriv2) {
    const double costheta=cos(p.theta), sintheta=sin(p.theta);
    const double R=p.r*sintheta, z=p.r*costheta;
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
PosProlSph toPosDeriv(const PosCyl& from, const ProlSph& cs,
    PosDerivT<Cyl, ProlSph>* derivs, PosDeriv2T<Cyl, ProlSph>* derivs2)
{
    // lambda and mu are roots "t" of equation  R^2/(t+alpha) + z^2/(t+gamma) = 1
    double R2=pow_2(from.R), z2=pow_2(from.z);
    double b = cs.alpha+cs.gamma - R2 - z2;
    double c = cs.alpha*cs.gamma - R2*cs.gamma - z2*cs.alpha;
    double det = b*b-4*c;
    if(det<=0)
        throw std::invalid_argument("Error in coordinate conversion Cyl=>ProlSph: det<=0");
    double sqD=sqrt(det);
    // lambda and mu are roots of quadratic equation  t^2+b*t+c=0
    double lambda = 0.5*(-b+sqD);
    double nu     = 0.5*(-b-sqD);
    lambda=fmax(-cs.alpha, lambda);
    nu=fmin(-cs.alpha, fmax(nu, -cs.gamma)); // prevent roundoff errors
    if(z2==0) nu=-cs.gamma;
    double kalpha = (2*cs.alpha-b)/sqD;      // intermediate coefs
    double kgamma = (2*cs.gamma-b)/sqD;
    if(derivs!=NULL) {
        derivs->dlambdadR = from.R*(1+kgamma);
        derivs->dlambdadz = from.z*(1+kalpha);
        derivs->dnudR     = from.R*(1-kgamma);
        derivs->dnudz     = from.z*(1-kalpha);
    }
    if(derivs2!=NULL) {
        double kR = 2*R2*(1-pow_2(kgamma))/sqD + kgamma;
        double kz = 2*z2*(1-pow_2(kalpha))/sqD + kalpha;
        derivs2->d2lambdadR2 = 1+kR;
        derivs2->d2lambdadz2 = 1+kz;
        derivs2->d2nudR2     = 1-kR;
        derivs2->d2nudz2     = 1-kz;
        derivs2->d2lambdadRdz= 2*from.R*from.z*(1-kalpha*kgamma)/sqD;
        derivs2->d2nudRdz    = -derivs2->d2lambdadRdz;
    }
    return PosProlSph(lambda, nu, from.phi, cs);
}

template PosCyl toPosDeriv(const PosCar&, PosDerivT<Car, Cyl>*, PosDeriv2T<Car, Cyl>*);
template PosSph toPosDeriv(const PosCar&, PosDerivT<Car, Sph>*, PosDeriv2T<Car, Sph>*);
template PosCar toPosDeriv(const PosCyl&, PosDerivT<Cyl, Car>*, PosDeriv2T<Cyl, Car>*);
template PosSph toPosDeriv(const PosCyl&, PosDerivT<Cyl, Sph>*, PosDeriv2T<Cyl, Sph>*);
template PosCar toPosDeriv(const PosSph&, PosDerivT<Sph, Car>*, PosDeriv2T<Sph, Car>*);
template PosCyl toPosDeriv(const PosSph&, PosDerivT<Sph, Cyl>*, PosDeriv2T<Sph, Cyl>*);
template PosProlSph toPosDeriv(const PosCyl&, const ProlSph&, PosDerivT<Cyl, ProlSph>*, PosDeriv2T<Cyl, ProlSph>*);

//--------  position+velocity conversion functions  ---------//

template<> PosVelCar toPosVel(const PosVelCyl& p) {
    const double cosphi=cos(p.phi), sinphi=sin(p.phi);
    const double vx=p.vR*cosphi-p.vphi*sinphi;
    const double vy=p.vR*sinphi+p.vphi*cosphi;
    return PosVelCar(p.R*cosphi, p.R*sinphi, p.z, vx, vy, p.vz);
}

template<> PosVelCar toPosVel(const PosVelSph& p) {
    const double sintheta=sin(p.theta), costheta=cos(p.theta);
    const double sinphi=sin(p.phi), cosphi=cos(p.phi);
    const double R=p.r*sintheta, vmer=p.vr*sintheta + p.vtheta*costheta;
    const double vx=vmer*cosphi - p.vphi*sinphi;
    const double vy=vmer*sinphi + p.vphi*cosphi;
    const double vz=p.vr*costheta - p.vtheta*sintheta;
    return PosVelCar(R*cosphi, R*sinphi, p.r*costheta, vx, vy, vz); 
}

template<> PosVelCyl toPosVel(const PosVelCar& p) {
    const double R=sqrt(pow_2(p.x)+pow_2(p.y));
    if(R==0)  // determine phi from vy/vx rather than y/x
        return PosVelCyl(R, p.z, atan2(p.vy, p.vx), sqrt(pow_2(p.vx)+pow_2(p.vy)), p.vz, 0);
    const double cosphi=p.x/R, sinphi=p.y/R;
    const double vR  = p.vx*cosphi+p.vy*sinphi;
    const double vphi=-p.vx*sinphi+p.vy*cosphi;
    return PosVelCyl(R, p.z, atan2(p.y, p.x), vR, p.vz, vphi);
}

template<> PosVelCyl toPosVel(const PosVelSph& p) {
    const double costheta=cos(p.theta), sintheta=sin(p.theta);
    const double R=p.r*sintheta, z=p.r*costheta;
    const double vR=p.vr*sintheta+p.vtheta*costheta;
    const double vz=p.vr*costheta-p.vtheta*sintheta;
    return PosVelCyl(R, z, p.phi, vR, vz, p.vphi);
}

template<> PosVelSph toPosVel(const PosVelCar& p) {
    const double R2=pow_2(p.x)+pow_2(p.y), R=sqrt(R2);
    const double r2=R2+pow_2(p.z), r=sqrt(r2), invr=1/r;
    if(R==0) {
        const double vR=sqrt(pow_2(p.vx)+pow_2(p.vy));
        const double phi=atan2(p.vy, p.vx);
        if(p.z==0) 
            return PosVelSph(0, atan2(vR, p.vz), phi, sqrt(vR*vR+p.vz*p.vz), 0, 0);
        return PosVelSph(r, p.z>0?0:M_PI, phi, p.vz*(p.z>0?1:-1), vR*(p.z>0?1:-1), 0);
    }
    const double temp   = p.x*p.vx+p.y*p.vy;
    const double vr     = (temp+p.z*p.vz)*invr;
    const double vtheta = (temp*p.z/R-p.vz*R)*invr;
    const double vphi   = (p.x*p.vy-p.y*p.vx)/R;
    return PosVelSph(r, atan2(R, p.z), atan2(p.y, p.x), vr, vtheta, vphi);
}

template<> PosVelSph toPosVel(const PosVelCyl& p) {
    const double r=sqrt(pow_2(p.R)+pow_2(p.z));
    if(r==0) {
        return PosVelSph(0, atan2(p.vR, p.vz), p.phi, sqrt(p.vR*p.vR+p.vz*p.vz), 0, 0);
    }
    const double rinv= 1./r;
    const double costheta=p.z*rinv, sintheta=p.R*rinv;
    const double vr=p.vR*sintheta+p.vz*costheta;
    const double vtheta=p.vR*costheta-p.vz*sintheta;
    return PosVelSph(r, atan2(p.R, p.z), p.phi, vr, vtheta, p.vphi);
}

template<> PosVelProlSph toPosVel(const PosVelCyl& from, const ProlSph& cs) {
    PosDerivT<Cyl, ProlSph> derivs;
    const PosProlSph pprol = toPosDeriv<Cyl, ProlSph> (from, cs, &derivs);
    double lambdadot = derivs.dlambdadR*from.vR + derivs.dlambdadz*from.vz;
    double nudot     = derivs.dnudR    *from.vR + derivs.dnudz    *from.vz;
    double phidot    = from.vphi!=0 ? from.vphi/from.R : 0;
    return PosVelProlSph(pprol, lambdadot, nudot, phidot);
}

template PosVelCar toPosVel(const PosVelCyl&);
template PosVelCar toPosVel(const PosVelCyl&);
template PosVelCyl toPosVel(const PosVelCar&);
template PosVelCyl toPosVel(const PosVelSph&);
template PosVelSph toPosVel(const PosVelCar&);
template PosVelSph toPosVel(const PosVelCyl&);
template PosVelProlSph toPosVel(const PosVelCyl&, const ProlSph&);

//-------- implementations of functions that convert gradients --------//
// note: the code below is machine-generated

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
    dest.dr = src.dx*deriv.dxdr + src.dy*deriv.dydr + src.dz*deriv.dzdr;
    dest.dtheta = src.dx*deriv.dxdtheta + src.dy*deriv.dydtheta + src.dz*deriv.dzdtheta;
    dest.dphi = src.dx*deriv.dxdphi + src.dy*deriv.dydphi;
    return dest;
}

template<>
GradSph toGrad(const GradCyl& src, const PosDerivT<Sph, Cyl>& deriv) {
    GradSph dest;
    dest.dr = src.dR*deriv.dRdr + src.dz*deriv.dzdr;
    dest.dtheta = src.dR*deriv.dRdtheta + src.dz*deriv.dzdtheta;
    dest.dphi = src.dphi;
    return dest;
}

template<>
GradCyl toGrad(const GradProlSph& src, const PosDerivT<Cyl, ProlSph>& deriv) {
    GradCyl dest;
    dest.dR = src.dlambda*deriv.dlambdadR + src.dnu*deriv.dnudR;
    dest.dz = src.dlambda*deriv.dlambdadz + src.dnu*deriv.dnudz;
    dest.dphi = 0;
    return dest;
}

template GradCar toGrad(const GradCyl&, const PosDerivT<Car, Cyl>&);
template GradCar toGrad(const GradSph&, const PosDerivT<Car, Sph>&);
template GradCyl toGrad(const GradCar&, const PosDerivT<Cyl, Car>&);
template GradCyl toGrad(const GradSph&, const PosDerivT<Cyl, Sph>&);
template GradSph toGrad(const GradCar&, const PosDerivT<Sph, Car>&);
template GradSph toGrad(const GradCyl&, const PosDerivT<Sph, Cyl>&);
template GradCyl toGrad(const GradProlSph&, const PosDerivT<Cyl, ProlSph>&);

//-------- implementations of functions that convert hessians --------//
// note: the code below is machine-generated and is not intended to be human-readable

template<>
HessCar toHess(const GradCyl& srcGrad, const HessCyl& srcHess,
    const PosDerivT<Car, Cyl>& deriv, const PosDeriv2T<Car, Cyl>& deriv2) {
    HessCar dest;
    dest.dx2 = 
        (srcHess.dR2*deriv.dRdx + srcHess.dRdphi*deriv.dphidx)*deriv.dRdx + 
        (srcHess.dRdphi*deriv.dRdx + srcHess.dphi2*deriv.dphidx)*deriv.dphidx + 
        srcGrad.dR*deriv2.d2Rdx2 + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy = 
        (srcHess.dR2*deriv.dRdy + srcHess.dRdphi*deriv.dphidy)*deriv.dRdx + 
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2*deriv.dphidy)*deriv.dphidx + 
        srcGrad.dR*deriv2.d2Rdxdy + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dxdz = 
        srcHess.dRdz*deriv.dRdx + 
        srcHess.dzdphi*deriv.dphidx;
    dest.dy2 = 
        (srcHess.dR2*deriv.dRdy + srcHess.dRdphi*deriv.dphidy)*deriv.dRdy + 
        (srcHess.dRdphi*deriv.dRdy + srcHess.dphi2*deriv.dphidy)*deriv.dphidy + 
        srcGrad.dR*deriv2.d2Rdy2 + srcGrad.dphi*deriv2.d2phidy2;
    dest.dydz = 
        srcHess.dRdz*deriv.dRdy + 
        srcHess.dzdphi*deriv.dphidy;
    dest.dz2 = srcHess.dz2;
    return dest;
}

template<>
HessCar toHess(const GradSph& srcGrad, const HessSph& srcHess,
    const PosDerivT<Car, Sph>& deriv, const PosDeriv2T<Car, Sph>& deriv2) {
    HessCar dest;
    dest.dx2 = 
        (srcHess.dr2*deriv.drdx + srcHess.drdtheta*deriv.dthetadx + srcHess.drdphi*deriv.dphidx)*deriv.drdx + 
        (srcHess.drdtheta*deriv.drdx + srcHess.dtheta2*deriv.dthetadx + srcHess.dthetadphi*deriv.dphidx)*deriv.dthetadx + 
        (srcHess.drdphi*deriv.drdx + srcHess.dthetadphi*deriv.dthetadx + srcHess.dphi2*deriv.dphidx)*deriv.dphidx + 
        srcGrad.dr*deriv2.d2rdx2 + srcGrad.dtheta*deriv2.d2thetadx2 + srcGrad.dphi*deriv2.d2phidx2;
    dest.dxdy = 
        (srcHess.dr2*deriv.drdy + srcHess.drdtheta*deriv.dthetady + srcHess.drdphi*deriv.dphidy)*deriv.drdx + 
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2*deriv.dthetady + srcHess.dthetadphi*deriv.dphidy)*deriv.dthetadx + 
        (srcHess.drdphi*deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2*deriv.dphidy)*deriv.dphidx + 
        srcGrad.dr*deriv2.d2rdxdy + srcGrad.dtheta*deriv2.d2thetadxdy + srcGrad.dphi*deriv2.d2phidxdy;
    dest.dxdz = 
        (srcHess.dr2*deriv.drdz + srcHess.drdtheta*deriv.dthetadz)*deriv.drdx + 
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2*deriv.dthetadz)*deriv.dthetadx + 
        (srcHess.drdphi*deriv.drdz + srcHess.dthetadphi*deriv.dthetadz)*deriv.dphidx + 
        srcGrad.dr*deriv2.d2rdxdz + srcGrad.dtheta*deriv2.d2thetadxdz;
    dest.dy2 = 
        (srcHess.dr2*deriv.drdy + srcHess.drdtheta*deriv.dthetady + srcHess.drdphi*deriv.dphidy)*deriv.drdy + 
        (srcHess.drdtheta*deriv.drdy + srcHess.dtheta2*deriv.dthetady + srcHess.dthetadphi*deriv.dphidy)*deriv.dthetady + 
        (srcHess.drdphi*deriv.drdy + srcHess.dthetadphi*deriv.dthetady + srcHess.dphi2*deriv.dphidy)*deriv.dphidy + 
        srcGrad.dr*deriv2.d2rdy2 + srcGrad.dtheta*deriv2.d2thetady2 + srcGrad.dphi*deriv2.d2phidy2;
    dest.dydz = 
        (srcHess.dr2*deriv.drdz + srcHess.drdtheta*deriv.dthetadz)*deriv.drdy + 
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2*deriv.dthetadz)*deriv.dthetady + 
        (srcHess.drdphi*deriv.drdz + srcHess.dthetadphi*deriv.dthetadz)*deriv.dphidy + 
        srcGrad.dr*deriv2.d2rdydz + srcGrad.dtheta*deriv2.d2thetadydz;
    dest.dz2 = 
        (srcHess.dr2*deriv.drdz + srcHess.drdtheta*deriv.dthetadz)*deriv.drdz + 
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2*deriv.dthetadz)*deriv.dthetadz + 
        srcGrad.dr*deriv2.d2rdz2 + srcGrad.dtheta*deriv2.d2thetadz2;
    return dest;
}

template<>
HessCyl toHess(const GradCar& srcGrad, const HessCar& srcHess,
    const PosDerivT<Cyl, Car>& deriv, const PosDeriv2T<Cyl, Car>& deriv2) {
    HessCyl dest;
    dest.dR2 = 
        (srcHess.dx2*deriv.dxdR + srcHess.dxdy*deriv.dydR)*deriv.dxdR + 
        (srcHess.dxdy*deriv.dxdR + srcHess.dy2*deriv.dydR)*deriv.dydR;
    dest.dRdz = 
        srcHess.dxdz*deriv.dxdR + 
        srcHess.dydz*deriv.dydR;
    dest.dRdphi = 
        (srcHess.dx2*deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdR + 
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2*deriv.dydphi)*deriv.dydR + 
        srcGrad.dx*deriv2.d2xdRdphi + srcGrad.dy*deriv2.d2ydRdphi;
    dest.dz2 = srcHess.dz2;
    dest.dzdphi = 
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi);
    dest.dphi2 = 
        (srcHess.dx2*deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdphi + 
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2*deriv.dydphi)*deriv.dydphi + 
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

template<>
HessCyl toHess(const GradSph& srcGrad, const HessSph& srcHess,
    const PosDerivT<Cyl, Sph>& deriv, const PosDeriv2T<Cyl, Sph>& deriv2) {
    HessCyl dest;
    dest.dR2 = 
        (srcHess.dr2*deriv.drdR + srcHess.drdtheta*deriv.dthetadR)*deriv.drdR + 
        (srcHess.drdtheta*deriv.drdR + srcHess.dtheta2*deriv.dthetadR)*deriv.dthetadR + 
        srcGrad.dr*deriv2.d2rdR2 + srcGrad.dtheta*deriv2.d2thetadR2;
    dest.dRdz = 
        (srcHess.dr2*deriv.drdz + srcHess.drdtheta*deriv.dthetadz)*deriv.drdR + 
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2*deriv.dthetadz)*deriv.dthetadR + 
        srcGrad.dr*deriv2.d2rdRdz + srcGrad.dtheta*deriv2.d2thetadRdz;
    dest.dRdphi = 
        srcHess.drdphi*deriv.drdR + 
        srcHess.dthetadphi*deriv.dthetadR;
    dest.dz2 = 
        (srcHess.dr2*deriv.drdz + srcHess.drdtheta*deriv.dthetadz)*deriv.drdz + 
        (srcHess.drdtheta*deriv.drdz + srcHess.dtheta2*deriv.dthetadz)*deriv.dthetadz + 
        srcGrad.dr*deriv2.d2rdz2 + srcGrad.dtheta*deriv2.d2thetadz2;
    dest.dzdphi = 
        srcHess.drdphi*deriv.drdz + 
        srcHess.dthetadphi*deriv.dthetadz;
    dest.dphi2 = srcHess.dphi2;
    return dest;
}

template<>
HessSph toHess(const GradCar& srcGrad, const HessCar& srcHess,
    const PosDerivT<Sph, Car>& deriv, const PosDeriv2T<Sph, Car>& deriv2) {
    HessSph dest;
    dest.dr2 = 
        (srcHess.dx2*deriv.dxdr + srcHess.dxdy*deriv.dydr + srcHess.dxdz*deriv.dzdr)*deriv.dxdr + 
        (srcHess.dxdy*deriv.dxdr + srcHess.dy2*deriv.dydr + srcHess.dydz*deriv.dzdr)*deriv.dydr + 
        (srcHess.dxdz*deriv.dxdr + srcHess.dydz*deriv.dydr + srcHess.dz2*deriv.dzdr)*deriv.dzdr;
    dest.drdtheta = 
        (srcHess.dx2*deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta)*deriv.dxdr + 
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2*deriv.dydtheta + srcHess.dydz*deriv.dzdtheta)*deriv.dydr + 
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2*deriv.dzdtheta)*deriv.dzdr + 
        srcGrad.dx*deriv2.d2xdrdtheta + srcGrad.dy*deriv2.d2ydrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.drdphi = 
        (srcHess.dx2*deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdr + 
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2*deriv.dydphi)*deriv.dydr + 
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi)*deriv.dzdr + 
        srcGrad.dx*deriv2.d2xdrdphi + srcGrad.dy*deriv2.d2ydrdphi;
    dest.dtheta2 = 
        (srcHess.dx2*deriv.dxdtheta + srcHess.dxdy*deriv.dydtheta + srcHess.dxdz*deriv.dzdtheta)*deriv.dxdtheta + 
        (srcHess.dxdy*deriv.dxdtheta + srcHess.dy2*deriv.dydtheta + srcHess.dydz*deriv.dzdtheta)*deriv.dydtheta + 
        (srcHess.dxdz*deriv.dxdtheta + srcHess.dydz*deriv.dydtheta + srcHess.dz2*deriv.dzdtheta)*deriv.dzdtheta + 
        srcGrad.dx*deriv2.d2xdtheta2 + srcGrad.dy*deriv2.d2ydtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.dthetadphi = 
        (srcHess.dx2*deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdtheta + 
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2*deriv.dydphi)*deriv.dydtheta + 
        (srcHess.dxdz*deriv.dxdphi + srcHess.dydz*deriv.dydphi)*deriv.dzdtheta + 
        srcGrad.dx*deriv2.d2xdthetadphi + srcGrad.dy*deriv2.d2ydthetadphi;
    dest.dphi2 = 
        (srcHess.dx2*deriv.dxdphi + srcHess.dxdy*deriv.dydphi)*deriv.dxdphi + 
        (srcHess.dxdy*deriv.dxdphi + srcHess.dy2*deriv.dydphi)*deriv.dydphi + 
        srcGrad.dx*deriv2.d2xdphi2 + srcGrad.dy*deriv2.d2ydphi2;
    return dest;
}

template<>
HessSph toHess(const GradCyl& srcGrad, const HessCyl& srcHess,
    const PosDerivT<Sph, Cyl>& deriv, const PosDeriv2T<Sph, Cyl>& deriv2) {
    HessSph dest;
    dest.dr2 = 
        (srcHess.dR2*deriv.dRdr + srcHess.dRdz*deriv.dzdr)*deriv.dRdr + 
        (srcHess.dRdz*deriv.dRdr + srcHess.dz2*deriv.dzdr)*deriv.dzdr;
    dest.drdtheta = 
        (srcHess.dR2*deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta)*deriv.dRdr + 
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2*deriv.dzdtheta)*deriv.dzdr + 
        srcGrad.dR*deriv2.d2Rdrdtheta + srcGrad.dz*deriv2.d2zdrdtheta;
    dest.drdphi = 
        srcHess.dRdphi*deriv.dRdr + 
        srcHess.dzdphi*deriv.dzdr;
    dest.dtheta2 = 
        (srcHess.dR2*deriv.dRdtheta + srcHess.dRdz*deriv.dzdtheta)*deriv.dRdtheta + 
        (srcHess.dRdz*deriv.dRdtheta + srcHess.dz2*deriv.dzdtheta)*deriv.dzdtheta + 
        srcGrad.dR*deriv2.d2Rdtheta2 + srcGrad.dz*deriv2.d2zdtheta2;
    dest.dthetadphi = 
        srcHess.dRdphi*deriv.dRdtheta + 
        srcHess.dzdphi*deriv.dzdtheta;
    dest.dphi2 = srcHess.dphi2;
    return dest;
}

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
    dest.dRdphi = dest.dzdphi = dest.dphi2 = 0;
    return dest;
}

template HessCar toHess(const GradCyl&, const HessCyl&, const PosDerivT<Car, Cyl>&, const PosDeriv2T<Car, Cyl>&);
template HessCar toHess(const GradSph&, const HessSph&, const PosDerivT<Car, Sph>&, const PosDeriv2T<Car, Sph>&);
template HessCyl toHess(const GradCar&, const HessCar&, const PosDerivT<Cyl, Car>&, const PosDeriv2T<Cyl, Car>&);
template HessCyl toHess(const GradSph&, const HessSph&, const PosDerivT<Cyl, Sph>&, const PosDeriv2T<Cyl, Sph>&);
template HessSph toHess(const GradCar&, const HessCar&, const PosDerivT<Sph, Car>&, const PosDeriv2T<Sph, Car>&);
template HessSph toHess(const GradCyl&, const HessCyl&, const PosDerivT<Sph, Cyl>&, const PosDeriv2T<Sph, Cyl>&);
template HessCyl toHess(const GradProlSph&, const HessProlSph&, const PosDerivT<Cyl, ProlSph>&, const PosDeriv2T<Cyl, ProlSph>&);

//------ conversion of derivatives of f(r) into gradients/hessians in different coord.sys. ------//
template<>
void eval_and_convert_sph(const ISimpleFunction& F,
    const PosCar& pos, double* value, GradCar* deriv, HessCar* deriv2)
{
    const double r=sqrt(pow_2(pos.x)+pow_2(pos.y)+pow_2(pos.z));
    if(deriv==NULL && deriv2==NULL) {
        F.eval_simple(r, value);
        return;
    }
    double der, der2;
    F.eval_simple(r, value, &der, deriv2!=NULL ? &der2 : 0);
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
void eval_and_convert_sph(const ISimpleFunction& F,
    const PosCyl& pos, double* value, GradCyl* deriv, HessCyl* deriv2)
{
    const double r=sqrt(pow_2(pos.R)+pow_2(pos.z));
    if(deriv==NULL && deriv2==NULL) {
        F.eval_simple(r, value);
        return;
    }
    double der, der2;
    F.eval_simple(r, value, &der, deriv2!=NULL ? &der2 : 0);
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
void eval_and_convert_sph(const ISimpleFunction& F,
    const PosSph& pos, double* value, GradSph* deriv, HessSph* deriv2)
{
    double der, der2;
    F.eval_simple(pos.r, value, deriv!=NULL ? &der : NULL, deriv2!=NULL ? &der2 : NULL);
    if(deriv) {
        deriv->dr = der;
        deriv->dtheta = deriv->dphi = 0;
    }
    if(deriv2) {
        deriv2->dr2 = der2;
        deriv2->dtheta2 = deriv2->dphi2 = deriv2->drdtheta = deriv2->drdphi = deriv2->dthetadphi = 0;
    }
}

/* explicit _instantiations_ of these templated functions 
   (as opposed to _specializations_ above), needed to make sure all of them get compiled.
   That's the extra boilerplate needed by C++ standard... */
template void eval_and_convert_sph(const ISimpleFunction& F,
    const PosCar& pos, double* value, GradCar* deriv, HessCar* deriv2);
template void eval_and_convert_sph(const ISimpleFunction& F,
    const PosCyl& pos, double* value, GradCyl* deriv, HessCyl* deriv2);
template void eval_and_convert_sph(const ISimpleFunction& F,
    const PosSph& pos, double* value, GradSph* deriv, HessSph* deriv2);
    
}  // namespace coord
