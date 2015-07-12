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
#include <iomanip>
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
    {  // this is loosely based on Henon-Heiles potential, shifted from origin..
        double x=p.x-0.5, y=p.y+1.5, z=p.z+0.25;
        if(value) 
            *value = (x*x+y*y)/2+z*(x*x-y*y/3)*y;
        if(deriv) {
            deriv->dx = x*(1+2*z*y);
            deriv->dy = y+z*(x*x-y*y);
            deriv->dz = (x*x-y*y/3)*y;
        }
        if(deriv2) {
            deriv2->dx2 = (1+2*z*y);
            deriv2->dxdy= 2*z*x;
            deriv2->dxdz= 2*y*x;
            deriv2->dy2 = 1-2*z*y;
            deriv2->dydz= x*x-y*y;
            deriv2->dz2 = 0;
        }
    }
};

template<> class IScalarFunction<Cyl> {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    virtual void eval_scalar(const PosCyl& p, double* value=0, GradCyl* deriv=0, HessCyl* deriv2=0) const
    {  // same potential expressed in different coordinates
        double sinphi = sin(p.phi), cosphi = cos(p.phi), sin2=pow_2(sinphi), R2=pow_2(p.R), R3=p.R*R2;
        if(value) 
            *value = R2*(3+p.R*p.z*sinphi*(6-8*sin2))/6;
        if(deriv) {
            deriv->dR  = p.R*(1+p.R*p.z*sinphi*(3-4*sin2));
            deriv->dphi= R3*p.z*cosphi*(1-4*sin2);
            deriv->dz  = R3*sinphi*(3-4*sin2)/3;
        }
        if(deriv2) {
            deriv2->dR2   = 1+2*p.R*p.z*sinphi*(3-4*sin2);
            deriv2->dRdphi= R2*p.z*cosphi*(3-12*sin2);
            deriv2->dRdz  = R2*sinphi*(3-4*sin2);
            deriv2->dphi2 = R3*p.z*sinphi*(-9+12*sin2);
            deriv2->dzdphi= R3*cosphi*(1-4*sin2);
            deriv2->dz2   = 0;
        }
    }
};

template<> class IScalarFunction<Sph> {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    virtual void eval_scalar(const PosSph& p, double* value=0, GradSph* deriv=0, HessSph* deriv2=0) const
    {  // some obscure combination of spherical harmonics
        if(value) 
            *value = pow(p.r,2.5)*sin(p.theta)*sin(p.phi+2) - pow_2(p.r)*pow_2(sin(p.theta))*cos(2*p.phi-3);
        if(deriv) {
            deriv->dr    = 2.5*pow(p.r,1.5)*sin(p.theta)*sin(p.phi+2) - 2*p.r*pow_2(sin(p.theta))*cos(2*p.phi-3);
            deriv->dtheta= pow(p.r,2.5)*cos(p.theta)*sin(p.phi+2) - pow_2(p.r)*sin(2*p.theta)*cos(2*p.phi-3);
            deriv->dphi  = pow(p.r,2.5)*sin(p.theta)*cos(p.phi+2) + pow_2(p.r)*pow_2(sin(p.theta))*2*sin(2*p.phi-3);;
        }
        if(deriv2) {
            deriv2->dr2       = 3.75*sqrt(p.r)*sin(p.theta)*sin(p.phi+2) - 2*pow_2(sin(p.theta))*cos(2*p.phi-3);
            deriv2->dtheta2   = pow(p.r,2.5)*sin(-p.theta)*sin(p.phi+2) - pow_2(p.r)*2*cos(2*p.theta)*cos(2*p.phi-3);
            deriv2->dphi2     = pow(p.r,2.5)*sin(-p.theta)*sin(p.phi+2) + pow_2(p.r)*pow_2(sin(p.theta))*4*cos(2*p.phi-3);
            deriv2->drdtheta  = 2.5*pow(p.r,1.5)*cos(p.theta)*sin(p.phi+2) - 2*p.r*sin(2*p.theta)*cos(2*p.phi-3);
            deriv2->drdphi    = 2.5*pow(p.r,1.5)*sin(p.theta)*cos(p.phi+2) + 4*p.r*pow_2(sin(p.theta))*sin(2*p.phi-3);
            deriv2->dthetadphi= pow(p.r,2.5)*cos(p.theta)*cos(p.phi+2) + pow_2(p.r)*sin(2*p.theta)*2*sin(2*p.phi-3);
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

/** check if we expect singularities in coordinate transformations */
template<typename coordSys> bool isSingular(const coord::PosT<coordSys>& p);
template<> bool isSingular(const coord::PosCar& p) {
    return (p.x==0&&p.y==0); }
template<> bool isSingular(const coord::PosCyl& p) {
    return (p.R==0); }
template<> bool isSingular(const coord::PosSph& p) {
    return (p.r==0||sin(p.theta)==0); }

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
    std::cout<<srcCS::name()<<" => "<<
        destCS::name()<<" => "<<srcCS::name();
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
    double srcvalue, destvalue=0;
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
    std::cout << (ok?"OK  ": isSingular(srcpoint)?"EXPECTEDLY FAILED  ":"FAILED  ");
    if(!samepos) 
        std::cout << "pos  ";
    if(!samegrad) 
        std::cout << "gradient  ";
    if(!samehess) 
        std::cout << "hessian  ";
    if(!samevalue2step) 
        std::cout << "2-step conversion value  ";
    if(!samegrad2step) 
        std::cout << "2-step gradient  ";
    if(!samehess2step) 
        std::cout << "2-step hessian  ";
    std::cout<<srcCS::name()<<" => "<<" [=> "<<intermedCS::name()<<"] => "<<
        destCS::name()<<" => "<<srcCS::name()<<"\n";
    return ok||isSingular(srcpoint);
}

bool test_prol(double lambda, double nu, double alpha, double gamma) {
    const coord::ProlSph cs(alpha, gamma);
    const coord::PosProlSph pp(lambda, nu, 0, cs);
    const coord::PosCyl pc=coord::toPosCyl(pp);
    const coord::PosProlSph ppnew=coord::toPos<coord::Cyl,coord::ProlSph>(pc, cs);
    //std::cout << std::setprecision(16) <<ppnew.lambda<<","<<ppnew.nu<<"\n";
    return fabs(ppnew.lambda-lambda)<1e-16 && fabs(ppnew.nu-nu)<1e-16;
}

/// define test suite in terms of points for various coord systems
const int numtestpoints=5;
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
    {1,3.14159, 2, 1, 2, 1e-4},   // point almost along z axis, vphi must be small, but vtheta is non-zero
    {0, 2,-1, 2, 0, 0}};  // point at origin with nonzero velocity in R

int main() {
    bool passed=true;
    passed &= test_prol(2.5600000000780003, 2.5599999701470493, -(1.6*1.6), -1);  // testing a certain bugfix
    if(!passed) std::cout << "ProlSph => Cyl => ProlSph failed for a nearly-degenerate case\n";

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
