/** Test conversion between spherical, cylindrical and cartesian coordinates
    1) positions/velocities,
    2) gradients and hessians.
    In both cases take a value in one coordinate system (source), convert to another (destination),
    then convert back and compare.
    In the second case, also check that two-staged conversion involving a third (intermediate) 
    coordinate system gives the identical result to a direct conversion.
*/
#include "coord.h"
#include <iostream>
#include <cmath>
#include <stdexcept>

const double eps=1e-10;  // accuracy of comparison
namespace coord{

/// some test functions that compute values, gradients and hessians in various coord systems
template<> class IScalarFunction<Car> {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    virtual void eval_scalar(const PosCar& p, double* value=0, GradCar* deriv=0, HessCar* deriv2=0) const
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

template<> class IScalarFunction<Cyl> {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    virtual void eval_scalar(const PosCyl& p, double* value=0, GradCyl* deriv=0, HessCyl* deriv2=0) const
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

template<> class IScalarFunction<Sph> {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    virtual void eval_scalar(const PosSph& p, double* value=0, GradSph* deriv=0, HessSph* deriv2=0) const
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

/// need to define templated comparison functions for positions, gradients and hessians
template<typename coordSys> 
bool equalPos(const PosT<coordSys>& p1, const PosT<coordSys>& p2);
template<typename coordSys> 
bool equalGrad(const GradT<coordSys>& g1, const GradT<coordSys>& g2);
template<typename coordSys> 
bool equalHess(const HessT<coordSys>& g1, const HessT<coordSys>& g2);

template<> bool equalPos(const PosCar& p1, const PosCar& p2) {
    return fabs(p1.x-p2.x)<eps && fabs(p1.y-p2.y)<eps && fabs(p1.z-p2.z)<eps; }
template<> bool equalPos(const PosCyl& p1, const PosCyl& p2) {
    return fabs(p1.R-p2.R)<eps && fabs(p1.z-p2.z)<eps && fabs(p1.phi-p2.phi)<eps; }
template<> bool equalPos(const PosSph& p1, const PosSph& p2) {
    return fabs(p1.r-p2.r)<eps && fabs(p1.theta-p2.theta)<eps && fabs(p1.phi-p2.phi)<eps; }

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

/** the test itself: perform conversion of position/velocity from one coord system to the other and back */
template<typename srcCS, typename destCS>
bool test_conv_posvel(const coord::PosVelT<srcCS>& srcpoint)
{
    const coord::PosVelT<destCS> destpoint=coord::toPosVel<srcCS,destCS>(srcpoint);
    const coord::PosVelT<srcCS>  invpoint =coord::toPosVel<destCS,srcCS>(destpoint);
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
    bool ok=samepos && samevel && sameLz && sameLt && sameV2;
    std::cout << (ok?"OK  ":"FAILED  ");
    if(!samepos) std::cout << "pos  ";
    if(!samevel) std::cout << "vel  ";
    if(!sameLz) std::cout << "L_z  ";
    if(!sameLt) std::cout << "L_total  ";
    if(!sameV2) std::cout << "v^2  ";
    std::cout<<coord::CoordSysName<srcCS>()<<" => "<<
        coord::CoordSysName<destCS>()<<" => "<<coord::CoordSysName<srcCS>();
    if(!ok) {
        for(int i=0; i<6; i++) std::cout<<" "<<src[i];
        std::cout << " => ";
        for(int i=0; i<6; i++) std::cout<<" "<<dest[i];
        std::cout << " => ";
        for(int i=0; i<6; i++) std::cout<<" "<<inv[i];
    }
    std::cout << "\n";
    return ok;
}

template<typename srcCS, typename destCS, typename intermedCS>
bool test_conv_deriv(const coord::PosT<srcCS>& srcpoint)
{
    coord::PosT<destCS> destpoint;
    coord::PosT<srcCS> invpoint;
    coord::PosDerivT<srcCS,destCS> derivStoD;
    coord::PosDerivT<destCS,srcCS> derivDtoI;
    coord::PosDeriv2T<srcCS,destCS> deriv2StoD;
    coord::PosDeriv2T<destCS,srcCS> deriv2DtoI;
    try{
        destpoint=coord::toPosDeriv<srcCS,destCS>(srcpoint, &derivStoD, &deriv2StoD);
    }
    catch(std::runtime_error& e) {
        std::cout << "    toPosDeriv failed: " << e.what() << "\n";
    }
    try{
        invpoint=coord::toPosDeriv<destCS,srcCS>(destpoint, &derivDtoI, &deriv2DtoI);
    }
    catch(std::runtime_error& e) {
        std::cout << "    toPosDeriv failed: " << e.what() << "\n";
    }
    coord::GradT<srcCS> srcgrad;
    coord::HessT<srcCS> srchess;
    coord::GradT<destCS> destgrad2step;
    coord::HessT<destCS> desthess2step;
    coord::IScalarFunction<srcCS> Fnc;
    double srcvalue, destvalue;
    Fnc.eval_scalar(srcpoint, &srcvalue, &srcgrad, &srchess);
    const coord::GradT<destCS> destgrad=coord::toGrad<srcCS,destCS>(srcgrad, derivDtoI);
    const coord::HessT<destCS> desthess=coord::toHess<srcCS,destCS>(srcgrad, srchess, derivDtoI, deriv2DtoI);
    const coord::GradT<srcCS> invgrad=coord::toGrad<destCS,srcCS>(destgrad, derivStoD);
    const coord::HessT<srcCS> invhess=coord::toHess<destCS,srcCS>(destgrad, desthess, derivStoD, deriv2StoD);
    try{
        coord::eval_and_convert_twostep<srcCS,intermedCS,destCS>(Fnc, destpoint, &destvalue, &destgrad2step, &desthess2step);
    }
    catch(std::exception& e) {
        std::cout << "    2-step conversion: " << e.what() << "\n";
    }

    bool samepos=equalPos(srcpoint,invpoint);
    bool samegrad=equalGrad(srcgrad, invgrad);
    bool samehess=equalHess(srchess, invhess);
    bool samevalue2step=(fabs(destvalue-srcvalue)<eps);
    bool samegrad2step=equalGrad(destgrad, destgrad2step);
    bool samehess2step=equalHess(desthess, desthess2step);
    bool ok=samepos && samegrad && samehess && samevalue2step && samegrad2step && samehess2step;
    std::cout << (ok?"OK  ":"FAILED  ");
    if(!samepos) std::cout << "pos  ";
    if(!samegrad) std::cout << "gradient  ";
    if(!samehess) std::cout << "hessian  ";
    if(!samevalue2step) std::cout << "2-step conversion value  ";
    if(!samegrad2step) std::cout << "2-step gradient  ";
    if(!samehess2step) std::cout << "2-step hessian  ";
    std::cout<<coord::CoordSysName<srcCS>()<<" => "<<
        coord::CoordSysName<destCS>()<<" => "<<coord::CoordSysName<srcCS>()<<"\n";
    return ok;
}

/// define test suite in terms of points for various coord systems
const int numtestpoints=5;
const double PI=M_PI;
const double posvel_car[numtestpoints][6] = {
    {1, 2, 3, 4, 5, 6},   // ordinary point
    {0,-1, 2,-3, 4,-5},   // point in y-z plane 
    {2, 0,-1, 0, 3,-4},   // point in x-z plane
    {0, 0, 1, 2, 3, 4},   // point along z axis
    {0, 0, 0,-1,-2,-3}};  // point at origin with nonzero velocity
const double posvel_cyl[numtestpoints][6] = {   // order: R, z, phi
    {1, 2, 3, 4, 5, 6},   // ordinary point
    {2,-1, 0,-3, 4,-5},   // point in x-z plane
    {0, 2, 0, 0,-1, 0},   // point along z axis, vphi must be zero
    {0,-1, 2, 1, 2, 0},   // point along z axis, vphi must be zero, but vR is non-zero
    {0, 0, 0, 1,-2, 0}};  // point at origin with nonzero velocity in R and z
const double posvel_sph[numtestpoints][6] = {   // order: R, theta, phi
    {1, 2, 3, 4, 5, 6},   // ordinary point
    {2, 1, 0,-3, 4,-5},   // point in x-z plane
    {1, 0, 0,-1, 0, 0},   // point along z axis, vphi must be zero
    {1,PI, 2, 1, 2, 0},   // point along z axis, vphi must be zero, but vtheta is non-zero
    {0, 2,-1, 2, 0, 0}};  // point at origin with nonzero velocity in R

int main() {
    bool passed=true;
    std::cout << " ======= Testing conversion of position/velocity points =======\n";
    for(int n=0; n<numtestpoints; n++) {
        std::cout << " :::Cartesian point::: ";     for(int d=0; d<6; d++) std::cout << posvel_car[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_posvel<coord::Car, coord::Cyl>(coord::PosVelCar(posvel_car[n]));
        passed &= test_conv_posvel<coord::Car, coord::Sph>(coord::PosVelCar(posvel_car[n]));
        std::cout << " :::Cylindrical point::: ";   for(int d=0; d<6; d++) std::cout << posvel_cyl[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_posvel<coord::Cyl, coord::Car>(coord::PosVelCyl(posvel_cyl[n]));
        passed &= test_conv_posvel<coord::Cyl, coord::Sph>(coord::PosVelCyl(posvel_cyl[n]));
        std::cout << " :::Spherical point::: ";     for(int d=0; d<6; d++) std::cout << posvel_sph[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_posvel<coord::Sph, coord::Car>(coord::PosVelSph(posvel_sph[n]));
        passed &= test_conv_posvel<coord::Sph, coord::Cyl>(coord::PosVelSph(posvel_sph[n]));
    }
    std::cout << " ======= Testing conversion of gradients and hessians =======\n";
    for(int n=0; n<numtestpoints; n++) {
        std::cout << " :::Cartesian point::: ";     for(int d=0; d<6; d++) std::cout << posvel_car[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_deriv<coord::Car, coord::Cyl, coord::Sph>(coord::PosCar(posvel_car[n][0], posvel_car[n][1], posvel_car[n][2]));
        passed &= test_conv_deriv<coord::Car, coord::Sph, coord::Cyl>(coord::PosCar(posvel_car[n][0], posvel_car[n][1], posvel_car[n][2]));
        std::cout << " :::Cylindrical point::: ";   for(int d=0; d<6; d++) std::cout << posvel_cyl[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_deriv<coord::Cyl, coord::Car, coord::Sph>(coord::PosCyl(posvel_cyl[n][0], posvel_cyl[n][1], posvel_cyl[n][2]));
        passed &= test_conv_deriv<coord::Cyl, coord::Sph, coord::Car>(coord::PosCyl(posvel_cyl[n][0], posvel_cyl[n][1], posvel_cyl[n][2]));
        std::cout << " :::Spherical point::: ";     for(int d=0; d<6; d++) std::cout << posvel_sph[n][d]<<" ";  std::cout<<"\n";
        passed &= test_conv_deriv<coord::Sph, coord::Car, coord::Cyl>(coord::PosSph(posvel_sph[n][0], posvel_sph[n][1], posvel_sph[n][2]));
        passed &= test_conv_deriv<coord::Sph, coord::Cyl, coord::Car>(coord::PosSph(posvel_sph[n][0], posvel_sph[n][1], posvel_sph[n][2]));
    }
    if(passed) std::cout << "ALL TESTS PASSED\n";
    return 0;
}
