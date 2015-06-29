#include "coord.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace coord{

    template<> double Lz(const PosVelCar& p) { return p.x*p.vy-p.y*p.vx; }
    template<> double Lz(const PosVelCyl& p) { return p.R*p.vphi; }
    template<> double Lz(const PosVelSph& p) { return p.r*sin(p.theta)*p.vphi; }

    template<> PosCar toPos(const PosCyl& pos) {
        return PosCar(pos.R*cos(pos.phi), pos.R*sin(pos.phi), pos.z);
    }
    template<> PosCar toPos(const PosSph& pos) {
        double R=pos.r*sin(pos.theta);
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

    template<> PosVelCar toPosVel(const PosVelCyl& p) {
        const double cosphi=cos(p.phi), sinphi=sin(p.phi);
        const double vx=p.vR*cosphi-p.vphi*sinphi;
        const double vy=p.vR*sinphi+p.vphi*cosphi;
        return PosVelCar(p.R*cosphi, p.R*sinphi, p.z, vx, vy, p.vz);
    }
    template<> PosVelCar toPosVel(const PosVelSph& pos) {
        throw std::runtime_error("Sph=>Car: Not implemented");
//        double R=pos.r*sin(pos.theta);
//        return PosVelCar(R*cos(pos.phi), R*sin(pos.phi), pos.r*cos(pos.theta)); 
    }
    template<> PosVelCyl toPosVel(const PosVelCar& p) {
        const double R=sqrt(pow_2(p.x)+pow_2(p.y));
        if(R==0)
            throw std::runtime_error("Car=>Cyl: R=0, fixme!");
        const double cosphi=p.x/R, sinphi=p.y/R;
        const double vR  = p.vx*cosphi+p.vy*sinphi;
        const double vphi=-p.vx*sinphi+p.vy*cosphi;
        return PosVelCyl(R, p.z, atan2(p.y, p.x), vR, p.vz, vphi);
    }
    template<> PosVelCyl toPosVel(const PosVelSph& pos) {
        throw std::runtime_error("Sph=>Cyl: Not implemented");
//        return PosVelCyl(pos.r*sin(pos.theta), pos.r*cos(pos.theta), pos.phi);
    }
    template<> PosVelSph toPosVel(const PosVelCar& pos) {
        throw std::runtime_error("Car=>Sph: Not implemented");
//        return PosVelSph(sqrt(pow_2(pos.x)+pow_2(pos.y)+pow_2(pos.z)), 
//            atan2(sqrt(pow_2(pos.x)+pow_2(pos.y)), pos.z), atan2(pos.y, pos.x));
    }
    template<> PosVelSph toPosVel(const PosVelCyl& pos) {
        throw std::runtime_error("Cyl=>Sph: Not implemented");
//        return PosVelSph(sqrt(pow_2(pos.R)+pow_2(pos.z)), atan2(pos.R, pos.z), pos.phi);
    }

    template<>
    void toDeriv(const PosCyl& srcPos, const GradCyl* srcGrad, const HessCyl* srcHess, 
        GradCar* destGrad, HessCar* destHess)
    {
        const double cosphi=cos(srcPos.phi), sinphi=sin(srcPos.phi);
        if(destGrad!=0) {
            destGrad->dx = srcGrad->dR*cosphi - srcGrad->dphi*sinphi/srcPos.R;
            destGrad->dy = srcGrad->dR*sinphi + srcGrad->dphi*cosphi/srcPos.R;
            destGrad->dz = srcGrad->dz;
        }
        if(destHess!=0) {
            if(srcHess==0)
                throw std::runtime_error("Cyl=>Car: Hessian should be provided");
            throw std::runtime_error("Cyl=>Car: Hessian not implemented"); 
        }
    }

    template<>
    void toDeriv(const PosSph& srcPos, const GradSph* srcGrad, const HessSph* srcHess, 
        GradCar* destGrad, HessCar* destHess)
    { throw std::runtime_error("Sph=>Car: Not implemented"); }
    template<>
    void toDeriv(const PosCar& srcPos, const GradCar* srcGrad, const HessCar* srcHess, 
        GradSph* destGrad, HessSph* destHess)
    { throw std::runtime_error("Car=>Sph: Not implemented"); }
    template<>
    void toDeriv(const PosCyl& srcPos, const GradCyl* srcGrad, const HessCyl* srcHess, 
        GradSph* destGrad, HessSph* destHess)
    { throw std::runtime_error("Cyl=>Sph: Not implemented"); }
    template<>
    void toDeriv(const PosCar& srcPos, const GradCar* srcGrad, const HessCar* srcHess, 
        GradCyl* destGrad, HessCyl* destHess)
    { throw std::runtime_error("Car=>Cyl: Not implemented"); }

    template<>
    void toDeriv(const PosSph& srcPos, const GradSph* srcGrad, const HessSph* srcHess, 
        GradCyl* destGrad, HessCyl* destHess)
    {
        if(srcGrad==0)
            throw std::runtime_error("Sph=>Cyl: Gradient should be provided");
        const double costh=cos(srcPos.theta), sinth=sin(srcPos.theta);
        if(destGrad!=0) {
            destGrad->dR = srcGrad->dr*sinth + srcGrad->dtheta*costh/srcPos.r;
            destGrad->dz = srcGrad->dr*costh - srcGrad->dtheta*sinth/srcPos.r;
            destGrad->dphi = srcGrad->dphi;
        }
        if(destHess!=0) {
            if(srcHess==0)
                throw std::runtime_error("Sph=>Cyl: Hessian should be provided");
            double rminus2=1./pow_2(srcPos.r), cos2=pow_2(costh), sin2=pow_2(sinth), sincos=sinth*costh;
            double mixed=srcHess->drdtheta/srcPos.r - 2*srcGrad->dtheta*rminus2;
            destHess->dR2 = srcHess->dr2*sin2 + srcHess->dtheta2*cos2*rminus2 + 
                srcGrad->dr*cos2/srcPos.r + mixed*2*sincos;
            destHess->dz2 = srcHess->dr2*cos2 + srcHess->dtheta2*sin2*rminus2 +
                srcGrad->dr*sin2/srcPos.r - mixed*2*sincos;
            destHess->dRdz= (srcHess->dr2-srcHess->dtheta2*rminus2) * sincos +
                - srcGrad->dr*sincos/srcPos.r + mixed*(cos2-sin2);
            destHess->dphi2 = srcHess->dphi2;
            destHess->dRdphi= srcHess->drdphi*sinth + srcHess->dthetadphi*costh/srcPos.r;
            destHess->dzdphi= srcHess->drdphi*costh - srcHess->dthetadphi*sinth/srcPos.r;
        }
    }


    // Prolate spheroidal coordinate system
    PosCyl toPosCyl(const PosProlSph& from, const CoordSysProlateSpheroidal& cs)
    {
        double R = sqrt((from.lambda+cs.alpha)*(from.nu+cs.alpha)/(cs.alpha-cs.gamma));
        double z = sqrt((from.lambda+cs.gamma)*(from.nu+cs.gamma)/(cs.gamma-cs.alpha));
        return PosCyl(R, z, from.phi);
    }

    PosProlSph toPosProlSph(const PosCyl& from, const CoordSysProlateSpheroidal& cs,
        PosDerivProlSph* derivs, PosDeriv2ProlSph* derivs2)
    {
        // lambda and mu are roots "t" of equation  R^2/(t+alpha) + z^2/(t+gamma) = 1
        double R2=pow_2(from.R), z2=pow_2(from.z);
        double b = cs.alpha+cs.gamma - R2 - z2;
        double c = cs.alpha*cs.gamma - R2*cs.gamma - z2*cs.alpha;
        double det = b*b-4*c;
        if(det<=0)
            throw std::runtime_error("Error in coordinate conversion Cyl=>ProlSph: det<=0");
        double sqD=sqrt(det);
        // lambda and mu are roots of quadratic equation  t^2+b*t+c=0
        double lambda = 0.5*(-b+sqD);
        double nu     = 0.5*(-b-sqD);
        double kalpha = (2*cs.alpha-b)/sqD;  // intermediate coefs
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
        return PosProlSph(lambda, nu, from.phi);
    }

}  // namespace coord
