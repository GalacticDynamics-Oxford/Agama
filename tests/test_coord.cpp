/** \name   test_coord.cpp
    \author Eugene Vasiliev
    \date   2015

    Test conversion between spherical, cylindrical and cartesian coordinates
    1) positions/velocities,
    2) gradients and hessians.
    In both cases take a value in one coordinate system (source), convert to another (destination),
    then convert back and compare.
    In the second case, also check that two-staged conversion involving a third (intermediate)
    coordinate system gives the identical result to a direct conversion.
*/
#include "coord.h"
#include "debug_utils.h"
#include <iostream>
#include <stdexcept>


const double eps=1e-14;  // accuracy of comparison

/// some test functions that compute values, gradients and hessians in various coord systems
template<typename CoordT>
class MyScalarFunction;

template<> class MyScalarFunction<coord::Car>: public coord::IScalarFunction<coord::Car> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosCar& p, double* value=0, coord::GradCar* deriv=0, coord::HessCar* deriv2=0) const
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

template<> class MyScalarFunction<coord::Cyl>: public coord::IScalarFunction<coord::Cyl> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosCyl& p, double* value=0, coord::GradCyl* deriv=0, coord::HessCyl* deriv2=0) const
    {   // same potential expressed in different coordinates
        double sinphi,cosphi;
        math::sincos(p.phi, sinphi, cosphi);
        double sin2=pow_2(sinphi), R2=pow_2(p.R), R3=p.R*R2;
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

template<> class MyScalarFunction<coord::Sph>: public coord::IScalarFunction<coord::Sph> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosSph& p, double* value=0, coord::GradSph* deriv=0, coord::HessSph* deriv2=0) const
    {   // some obscure combination of spherical harmonics
        double st, ct, sa, ca, sb, cb, sr=sqrt(p.r), r2=p.r*p.r;
        math::sincos(  p.theta, st, ct);
        math::sincos(  p.phi+2, sa, ca);
        math::sincos(2*p.phi-3, sb, cb);
        if(value)
            *value = r2*sr*st*sa - r2*st*st*cb;
        if(deriv) {
            deriv->dr    = 2.5*p.r*sr*st*sa - 2*p.r*st*st*cb;
            deriv->dtheta=      r2*sr*ct*sa - 2*r2 *st*ct*cb;
            deriv->dphi  =      r2*sr*st*ca + 2*r2 *st*st*sb;;
        }
        if(deriv2) {
            deriv2->dr2       =    3.75*sr*st*sa - 2*st*st*cb;
            deriv2->dtheta2   =     -r2*sr*st*sa - 2*r2*(ct*ct-st*st)*cb;
            deriv2->dphi2     =     -r2*sr*st*sa + 4*r2 *st*st*cb;
            deriv2->drdtheta  = 2.5*p.r*sr*ct*sa - 4*p.r*st*ct*cb;
            deriv2->drdphi    = 2.5*p.r*sr*st*ca + 4*p.r*st*st*sb;
            deriv2->dthetadphi=      r2*sr*ct*ca + 4*r2 *st*ct*sb;
        }
    }
};

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
    const coord::PosVelT<destCS> dstpoint=coord::toPosVel<srcCS,destCS>(srcpoint);
    const coord::PosVelT<srcCS>  invpoint=coord::toPosVel<destCS,srcCS>(dstpoint);
    double src[6],dst[6],inv[6];
    srcpoint.unpack_to(src);
    dstpoint.unpack_to(dst);
    invpoint.unpack_to(inv);
    double Lsrc =coord::Ltotal(srcpoint), Ldst=coord::Ltotal(dstpoint);
    double vsrc =pow_2(src[3])+pow_2(src[4])+pow_2(src[5]);
    double vdst =pow_2(dst[3])+pow_2(dst[4])+pow_2(dst[5]);
    bool samepos=equalPos(srcpoint, invpoint, eps);
    bool samevel=equalVel(srcpoint, invpoint, eps);
    bool sameL  =math::fcmp(Lsrc, Ldst, eps)==0;
    bool samev2 =math::fcmp(vsrc, vdst, eps)==0;
    bool ok=samepos && samevel && sameL && samev2;
    std::cout << (ok?"OK  ":"\033[1;31mFAILED\033[0m  ");
    if(!samepos) std::cout << "pos  ";
    if(!samevel) std::cout << "vel  " << src[3] << " " << (inv[3]-src[3]) << ", " << src[4] << " " << (inv[4]-src[4]) << ", " << src[5] << " " << (inv[5]-src[5]);
    if(!sameL)   std::cout << "L_total  ";
    if(!samev2)  std::cout << "v^2  ";
    std::cout<<srcCS::name()<<" => "<<
        destCS::name()<<" => "<<srcCS::name();
    if(!ok)
        std::cout << srcpoint << " => " << dstpoint << " => " << invpoint;
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
    MyScalarFunction<srcCS> Fnc;
    double srcvalue, destvalue=0;
    Fnc.evalScalar(srcpoint, &srcvalue, &srcgrad, &srchess);
    const coord::GradT<destCS> destgrad=coord::toGrad<srcCS,destCS>(srcgrad, derivDtoI);
    const coord::HessT<destCS> desthess=coord::toHess<srcCS,destCS>(srcgrad, srchess, derivDtoI, deriv2DtoI);
    const coord::GradT<srcCS> invgrad=coord::toGrad<destCS,srcCS>(destgrad, derivStoD);
    const coord::HessT<srcCS> invhess=coord::toHess<destCS,srcCS>(destgrad, desthess, derivStoD, deriv2StoD);
    try{
        coord::evalAndConvertTwoStep<srcCS,intermedCS,destCS>(Fnc, destpoint, &destvalue, &destgrad2step, &desthess2step);
    }
    catch(std::exception& e) {
        std::cout << "    2-step conversion: " << e.what() << "\n";
    }

    // not all components of the hessian can be accurately converted in the following nearly-degenerate case
    double epsh = eps;
    if(srcCS::name() == coord::Sph::name() && ((const double*)(&srcpoint))[1]/*theta*/ >= 3.1415)
        epsh = 1e-10;  // loosen the tolerance

    bool samepos  = coord::equalPos(srcpoint,invpoint, eps);
    bool samegrad = coord::equalGrad(srcgrad, invgrad, eps);
    bool samehess = coord::equalHess(srchess, invhess, epsh);
    bool samevalue2step= math::fcmp(destvalue, srcvalue, eps)==0;
    bool samegrad2step = coord::equalGrad(destgrad, destgrad2step, eps);
    bool samehess2step = coord::equalHess(desthess, desthess2step, epsh);
    bool ok=samepos && samegrad && samehess && samevalue2step && samegrad2step && samehess2step;
    std::cout << (ok?"OK  ": isSingular(srcpoint)?"EXPECTEDLY FAILED  ":"\033[1;31mFAILED\033[0m  ");
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

bool test_prol()
{
    const coord::ProlSph cs(sqrt(1.6*1.6-1));
    double lambda=1.5600000000780003, nu=-1.5599999701470495;
    const coord::PosProlSph pp(lambda, nu, 0, cs);
    const coord::PosCyl pc=coord::toPosCyl(pp);
    const coord::PosProlSph ppnew=coord::toPos<coord::Cyl,coord::ProlSph>(pc, cs);
    return fabs(ppnew.lambda-lambda)<3e-16 && fabs(ppnew.nu-nu)<3e-16;
}

bool test_prol2(const coord::PosCyl& src)
{
    const coord::ProlSph cs(sqrt(1.6*1.6-1));
    coord::PosDerivT <coord::Cyl, coord::ProlSph> derivCtoP;
    coord::PosDeriv2T<coord::Cyl, coord::ProlSph> deriv2CtoP;
    const coord::PosProlSph pp = coord::toPosDeriv<coord::Cyl, coord::ProlSph>(src, cs, &derivCtoP, &deriv2CtoP);
    coord::PosDerivT <coord::ProlSph, coord::Cyl> derivPtoC;
    coord::PosDeriv2T<coord::ProlSph, coord::Cyl> deriv2PtoC;
    const coord::PosCyl pc = coord::toPosDeriv<coord::ProlSph, coord::Cyl>(pp, &derivPtoC, &deriv2PtoC);
    coord::GradCyl gradCyl, gradCylNew;
    coord::HessCyl hessCyl;
    coord::GradProlSph gradProl;
    MyScalarFunction<coord::Cyl> Fnc;
    Fnc.evalScalar(src, NULL, &gradCyl, &hessCyl);
    gradProl   = coord::toGrad<coord::Cyl, coord::ProlSph> (gradCyl, derivPtoC);
    gradCylNew = coord::toGrad<coord::ProlSph, coord::Cyl> (gradProl, derivCtoP);
    bool samepos  = coord::equalPos(src, pc, eps);
    bool samegrad = coord::equalGrad(gradCyl, gradCylNew, eps);
    return samepos && samegrad;
}

bool test_prolmod()
{
    // gradient conversion
    const coord::ProlMod cs(1.23456);
    const double rho = cs.D-1e-1, tau = .5;
    const coord::PosProlMod pp(rho, tau, 0, cs);
    coord::PosDerivT<coord::ProlMod, coord::Cyl> derivPtoC;
    const coord::PosCyl pc=coord::toPosDeriv<coord::ProlMod, coord::Cyl>(pp, &derivPtoC);
    coord::PosDerivT<coord::Cyl, coord::ProlMod> derivCtoP;
    const coord::PosProlMod ppnew=coord::toPosDeriv<coord::Cyl,coord::ProlMod>(pc, cs, &derivCtoP);
    coord::GradCyl gradC;
    gradC.dR = 0.1234; gradC.dz = 1.2345; gradC.dphi = -2.3456;
    coord::GradProlMod gradP = coord::toGrad<coord::Cyl, coord::ProlMod> (gradC, derivPtoC);
    coord::GradCyl  gradCnew = coord::toGrad<coord::ProlMod, coord::Cyl> (gradP, derivCtoP);
    bool samepos  = fabs(pp.rho-ppnew.rho)<1e-15 && fabs(pp.tau-ppnew.tau)<1e-15;
    bool samegrad = coord::equalGrad(gradC, gradCnew, eps);
    // velocity conversion
    coord::PosVelCyl pvc(1.234, -0.2356, 4.568, 2.0846, -1.3563, 3.4531);
    coord::PosVelProlMod pvp(coord::toPosVel<coord::Cyl, coord::ProlMod>(pvc, cs));
    coord::PosVelCyl pvc1(coord::toPosVelCyl(pvp));
    bool samepv = coord::equalPosVel(pvc, pvc1, eps);
    // sph.mod.
    coord::PosVelSphMod psm(1.765432,0.3456,3.76543,0.213456,-2.123456,0.76543);
    coord::PosVelCyl pcy = toPosVelCyl(psm);
    coord::PosVelSphMod psm1 = coord::toPosVel<coord::Cyl, coord::SphMod>(pcy);
    return samepos && samegrad && samepv && equalPosVel(psm, psm1, eps);
}

template<typename CS> bool isNotNan(const coord::PosT<CS>& p);
template<> bool isNotNan(const coord::PosCar& p) { return p.x==p.x && p.y==p.y && p.z==p.z; }
template<> bool isNotNan(const coord::PosCyl& p) { return p.R==p.R && p.phi==p.phi && p.z==p.z; }
template<> bool isNotNan(const coord::PosSph& p) { return p.r==p.r && p.phi==p.phi && p.theta==p.theta; }

template<typename srcCS, typename destCS>
bool test_inf(double a, double b, double c)
{
    coord::PosT<srcCS> pos(a, b, c);
    return isNotNan(coord::toPos<srcCS, destCS>(pos)) &&
      isNotNan(coord::toPosDeriv<srcCS, destCS>(pos, NULL));
}
bool test_inf()
{
    return
    test_inf<coord::Cyl, coord::Car>(INFINITY, 0, 0) &&
    test_inf<coord::Cyl, coord::Car>(INFINITY, 1, 0) &&
    test_inf<coord::Cyl, coord::Car>(INFINITY, 0, 1) &&
    test_inf<coord::Sph, coord::Car>(INFINITY, 0, 0) &&
    test_inf<coord::Sph, coord::Car>(INFINITY, 1, 0) &&
    test_inf<coord::Sph, coord::Car>(INFINITY, 0, 1) &&
    test_inf<coord::Sph, coord::Cyl>(INFINITY, 0, 0) &&
    test_inf<coord::Sph, coord::Cyl>(INFINITY, 1, 0) &&
    test_inf<coord::Sph, coord::Cyl>(INFINITY, 0, 1) &&
    test_inf<coord::Car, coord::Cyl>(INFINITY, 0, 0) &&
    test_inf<coord::Car, coord::Cyl>(INFINITY, 1, 0) &&
    test_inf<coord::Car, coord::Sph>(INFINITY, 0, 0) &&
    test_inf<coord::Car, coord::Sph>(0, 0, INFINITY);
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
    passed &= test_prol();  // testing a certain bugfix
    if(!passed) std::cout << "ProlSph => Cyl => ProlSph failed for a nearly-degenerate case\n";
    passed &= test_prol2(coord::PosCyl(1.2,-2.3,3.4));  // testing negative z
    if(!passed) std::cout << "ProlSph => Cyl => ProlSph failed for z<0\n";
    passed &= test_prolmod();
    if(!passed) std::cout << "ProlMod <=> Cyl failed\n";
    passed &= test_inf();
    if(!passed) std::cout << "Coordinate conversion at infinity failed\n";

    std::cout << " ======= Testing conversion of position/velocity points =======\n";
    for(int n=0; n<numtestpoints; n++) {
        coord::PosVelCar pvcar(posvel_car[n]);
        std::cout << " :::Cartesian point::: " << pvcar << "\n";
        passed &= test_conv_posvel<coord::Car, coord::Cyl>(pvcar);
        passed &= test_conv_posvel<coord::Car, coord::Sph>(pvcar);
        coord::PosVelCyl pvcyl(posvel_cyl[n]);
        std::cout << " :::Cylindrical point::: " << pvcyl << "\n";
        passed &= test_conv_posvel<coord::Cyl, coord::Car>(pvcyl);
        passed &= test_conv_posvel<coord::Cyl, coord::Sph>(pvcyl);
        coord::PosVelSph pvsph(posvel_sph[n]);
        std::cout << " :::Spherical point::: " << pvsph << "\n";
        passed &= test_conv_posvel<coord::Sph, coord::Car>(pvsph);
        passed &= test_conv_posvel<coord::Sph, coord::Cyl>(pvsph);
    }
    std::cout << " ======= Testing conversion of gradients and hessians =======\n";
    for(int n=0; n<numtestpoints; n++) {
        coord::PosCar pcar = coord::PosVelCar(posvel_car[n]);
        std::cout << " :::Cartesian point::: " << pcar << "\n";
        passed &= test_conv_deriv<coord::Car, coord::Cyl, coord::Sph>(pcar);
        passed &= test_conv_deriv<coord::Car, coord::Sph, coord::Cyl>(pcar);
        coord::PosCyl pcyl = coord::PosVelCyl(posvel_cyl[n]);
        std::cout << " :::Cylindrical point::: " << pcyl << "\n";
        passed &= test_conv_deriv<coord::Cyl, coord::Car, coord::Sph>(pcyl);
        passed &= test_conv_deriv<coord::Cyl, coord::Sph, coord::Car>(pcyl);
        coord::PosSph psph = coord::PosVelSph(posvel_sph[n]);
        std::cout << " :::Spherical point::: " << psph << "\n";
        passed &= test_conv_deriv<coord::Sph, coord::Car, coord::Cyl>(psph);
        passed &= test_conv_deriv<coord::Sph, coord::Cyl, coord::Car>(psph);
    }
    if(passed)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
