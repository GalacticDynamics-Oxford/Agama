#include "coord.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

const double eps=1e-12;  // accuracy of comparison
namespace coord{

// some test functions that compute values, gradients and hessians in various coord systems
template<> class ScalarFunction<Car> {
public:
    ScalarFunction() {};
    virtual ~ScalarFunction() {};
    virtual void evaluate(const PosCar& p, double* value=0, GradCar* deriv=0, HessCar* deriv2=0) const
    {  // this is loosely based on Henon-Heiles potential..
        if(value) *value=(p.x*p.x+p.y*p.y)/2+p.z*(p.x*p.x-p.y*p.y/3)*p.y;
        if(deriv) {
            deriv->dx=p.x*(1+2*p.z*p.y);
            deriv->dy=p.y+p.z*(p.x*p.x-p.y*p.y);
            deriv->dz=(p.x*p.x-p.y*p.y/3)*p.y;
        }
        if(deriv2) {
            deriv2->dx2=(1+2*p.z*p.y);
            deriv2->dxdy=2*p.z*p.x;
            deriv2->dxdz=2*p.y*p.x;
            deriv2->dy2=1-2*p.z*p.y;
            deriv2->dydz=p.x*p.x-p.y*p.y;
            deriv2->dz2=0;
        }
    }
};

template<> class ScalarFunction<Cyl> {
public:
    ScalarFunction() {};
    virtual ~ScalarFunction() {};
    virtual void evaluate(const PosCyl& p, double* value=0, GradCyl* deriv=0, HessCyl* deriv2=0) const
    {  // and this is just the stuff above with relabelled variables
        if(value) *value=(p.R*p.R+p.phi*p.phi)/2+p.z*(p.R*p.R-p.phi*p.phi/3)*cos(p.phi);
        if(deriv) {
            deriv->dR=p.R*(1+2*p.z*p.phi);
            deriv->dphi=p.phi+p.z*(p.R*p.R-p.phi*p.phi);
            deriv->dz=(p.R*p.R-p.phi*p.phi/3)*p.phi;
        }
        if(deriv2) {
            deriv2->dR2=(1+2*p.z*p.phi);
            deriv2->dRdphi=2*p.z*p.R;
            deriv2->dRdz=2*p.phi*p.R;
            deriv2->dphi2=1-2*p.z*p.phi;
            deriv2->dzdphi=p.R*p.R-p.phi*p.phi;
            deriv2->dz2=exp(p.z);
        }
    }
};

template<> class ScalarFunction<Sph> {
public:
    ScalarFunction() {};
    virtual ~ScalarFunction() {};
    virtual void evaluate(const PosSph& p, double* value=0, GradSph* deriv=0, HessSph* deriv2=0) const
    {  // and this is the same stuff with intentionally confounded variables
        if(value) *value=(p.r*p.r+p.phi*p.phi)/2+p.theta*(p.r*p.r-p.phi*p.phi/3)*sin(p.phi);
        if(deriv) {
            deriv->dtheta=p.r*(1+2*p.theta*p.phi);
            deriv->dphi=p.phi+p.theta*(p.r*p.r-p.phi*p.phi);
            deriv->dr=(p.r*p.r-p.phi*p.phi/3)*p.phi;
        }
        if(deriv2) {
            deriv2->dtheta2=(1+2*p.theta*p.phi);
            deriv2->dthetadphi=2*p.theta*p.r;
            deriv2->dr2=2*p.phi*p.r;
            deriv2->dphi2=1-2*p.theta*p.phi;
            deriv2->drdtheta=p.r*p.r-p.phi*p.phi;
            deriv2->drdphi=cos(p.theta+1.23456);
        }
    }
};

template<typename coordSys> 
bool equalGrad(const GradT<coordSys>& g1, const GradT<coordSys>& g2);
template<typename coordSys> 
bool equalHess(const HessT<coordSys>& g1, const HessT<coordSys>& g2);

template<> bool equalGrad(const GradCar& g1, const GradCar& g2) {
    return fabs(g1.dx-g2.dx)<eps && fabs(g1.dy-g2.dy)<eps && fabs(g1.dz-g2.dz)<eps; }
template<> bool equalGrad(const GradCyl& g1, const GradCyl& g2) {
    return fabs(g1.dR-g2.dR)<eps && fabs(g1.dphi-g2.dphi)<eps && fabs(g1.dz-g2.dz)<eps; }
template<> bool equalGrad(const GradSph& g1, const GradSph& g2) {
    return fabs(g1.dr-g2.dr)<eps && fabs(g1.dtheta-g2.dtheta)<eps && fabs(g1.dphi-g2.dphi)<eps; }

template<> bool equalHess(const HessCar& h1, const HessCar& h2) {
    return fabs(h1.dx2-h2.dx2)<eps && fabs(h1.dy2-h2.dy2)<eps && fabs(h1.dz2-h2.dz2)<eps &&
        fabs(h1.dxdy-h2.dxdy)<eps && fabs(h1.dydz-h2.dydz)<eps && fabs(h1.dxdz-h2.dxdz)<eps; }
template<> bool equalHess(const HessCyl& h1, const HessCyl& h2) {
    return fabs(h1.dR2-h2.dR2)<eps && fabs(h1.dphi2-h2.dphi2)<eps && fabs(h1.dz2-h2.dz2)<eps &&
        fabs(h1.dRdphi-h2.dRdphi)<eps && fabs(h1.dzdphi-h2.dzdphi)<eps && fabs(h1.dRdz-h2.dRdz)<eps; }
template<> bool equalHess(const HessSph& h1, const HessSph& h2) {
    return fabs(h1.dr2-h2.dr2)<eps && fabs(h1.dtheta2-h2.dtheta2)<eps && fabs(h1.dphi2-h2.dphi2)<eps &&
        fabs(h1.drdtheta-h2.drdtheta)<eps && fabs(h1.drdphi-h2.drdphi)<eps && fabs(h1.dthetadphi-h2.dthetadphi)<eps; }
}

template<typename srcCS, typename destCS, typename intermedCS>
bool test_conv_posvel(double x[])
{
    const coord::PosVelT<srcCS>  srcpoint (x);
    const coord::PosVelT<destCS> destpoint=coord::toPosVel<srcCS,destCS>(srcpoint);
    const coord::PosVelT<srcCS>  invpoint =coord::toPosVel<destCS,srcCS>(destpoint);
    coord::PosDerivT<srcCS,destCS> derivStoD;
    coord::PosDerivT<destCS,srcCS> derivDtoI;
    coord::PosDeriv2T<srcCS,destCS> deriv2StoD;
    coord::PosDeriv2T<destCS,srcCS> deriv2DtoI;
    try{
        coord::toPosDeriv<srcCS,destCS>(srcpoint, &derivStoD, &deriv2StoD);
    }
    catch(std::runtime_error& e) {
        std::cerr << "    toPosDeriv failed: " << e.what() << "\n";
    }
    try{
        coord::toPosDeriv<destCS,srcCS>(destpoint, &derivDtoI, &deriv2DtoI);
    }
    catch(std::runtime_error& e) {
        std::cerr << "    toPosDeriv failed: " << e.what() << "\n";
    }
    coord::GradT<srcCS> srcgrad;
    coord::HessT<srcCS> srchess;
    coord::GradT<destCS> destgrad2step;
    coord::HessT<destCS> desthess2step;
    coord::ScalarFunction<srcCS> Fnc;
    double srcvalue, destvalue;
    Fnc.evaluate(srcpoint, &srcvalue, &srcgrad, &srchess);
    const coord::GradT<destCS> destgrad=coord::toGrad<srcCS,destCS>(srcgrad, derivDtoI);
    const coord::HessT<destCS> desthess=coord::toHess<srcCS,destCS>(srcgrad, srchess, derivDtoI, deriv2DtoI);
    const coord::GradT<srcCS> invgrad=coord::toGrad<destCS,srcCS>(destgrad, derivStoD);
    const coord::HessT<srcCS> invhess=coord::toHess<destCS,srcCS>(destgrad, desthess, derivStoD, deriv2StoD);
    try{
        coord::eval_and_convert_twostep<srcCS,intermedCS,destCS>(Fnc, destpoint, &destvalue, &destgrad2step, &desthess2step);
    }
    catch(std::exception& e) {
        std::cerr << "    2-step conversion: " << e.what() << "\n";
    }

    double src[6],dest[6],inv[6];
    srcpoint.unpack_to(src);
    destpoint.unpack_to(dest);
    invpoint.unpack_to(inv);
    double Lzsrc=coord::Lz(srcpoint), Lzdest=coord::Lz(destpoint);
    double Ltsrc=coord::Ltotal(srcpoint), Ltdest=coord::Ltotal(destpoint);
    double V2src=pow_2(src[3])+pow_2(src[4])+pow_2(src[5]);
    double V2dest=pow_2(dest[3])+pow_2(dest[4])+pow_2(dest[5]);
    bool samepos=true;
    for(int i=0; i<3; i++)
        if(fabs(src[i]-inv[i])>eps) samepos=false;
    bool samevel=true;
    for(int i=3; i<6; i++)
        if(fabs(src[i]-inv[i])>eps) samevel=false;
    bool sameLz=(fabs(Lzsrc-Lzdest)<eps);
    bool sameLt=(fabs(Ltsrc-Ltdest)<eps);
    bool sameV2=(fabs(V2src-V2dest)<eps);
    bool samegrad=equalGrad(srcgrad, invgrad);
    bool samehess=equalHess(srchess, invhess);
    bool samevalue2step=(fabs(destvalue-srcvalue)<eps);
    bool samegrad2step=equalGrad(destgrad, destgrad2step);
    bool samehess2step=equalHess(desthess, desthess2step);
    bool ok=samepos && samevel && sameLz && sameLt && sameV2 && 
        samegrad && samehess && samevalue2step && samegrad2step && samehess2step;
    std::cerr << (ok?"OK  ":"FAILED  ");
    if(!samepos) std::cerr << "pos  ";
    if(!samevel) std::cerr << "vel  ";
    if(!sameLz) std::cerr << "L_z  ";
    if(!sameLt) std::cerr << "L_total  ";
    if(!sameV2) std::cerr << "v^2  ";
    if(!samegrad) std::cerr << "gradient  ";
    if(!samehess) std::cerr << "hessian  ";
    if(!samevalue2step) std::cerr << "2-step conversion value  ";
    if(!samegrad2step) std::cerr << "2-step gradient  ";
    if(!samehess2step) std::cerr << "2-step hessian  ";
    std::cerr<<coord::CoordSysName<srcCS>()<<" => "<<
        coord::CoordSysName<destCS>()<<" => "<<coord::CoordSysName<srcCS>()<<"\n";
    return ok;
}

int main() {
    bool passed=true;
    for(int n=0; n<1; n++) {
        double pv[6]={1,2,3,4,5,6};
        passed &= test_conv_posvel<coord::Car, coord::Cyl, coord::Sph>(pv);
        passed &= test_conv_posvel<coord::Car, coord::Sph, coord::Cyl>(pv);
        passed &= test_conv_posvel<coord::Cyl, coord::Car, coord::Sph>(pv);
        passed &= test_conv_posvel<coord::Cyl, coord::Sph, coord::Car>(pv);
        passed &= test_conv_posvel<coord::Sph, coord::Car, coord::Cyl>(pv);
        passed &= test_conv_posvel<coord::Sph, coord::Cyl, coord::Car>(pv);
    }
    if(passed) std::cerr << "ALL TESTS PASSED\n";
    return 0;
}
