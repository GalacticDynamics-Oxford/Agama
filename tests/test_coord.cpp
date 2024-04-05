/** \name   test_coord.cpp
    \author Eugene Vasiliev
    \date   2015-2024

    Test conversion between spherical, cylindrical and cartesian coordinates
    1) positions/velocities,
    2) gradients and hessians.
    In both cases take a value in one coordinate system (source), convert to another (destination),
    then convert back and compare.
*/
#include "coord.h"
#include "debug_utils.h"
#include "utils.h"
#include "math_random.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

const double eps=1e-14;  // accuracy of comparison

/// some test functions that compute values, gradients and hessians in various coord systems
template<typename CoordT>
class MyScalarFunction;

template<> class MyScalarFunction<coord::Car>: public coord::IScalarFunction<coord::Car> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosCar& p,
        double* value=NULL, coord::GradCar* deriv=NULL, coord::HessCar* deriv2=NULL, double /*time*/=0) const
    {  // this is loosely based on Henon-Heiles potential, shifted from origin and adding a z^2 term
        double x=p.x-0.5, y=p.y+1.5, z=p.z+0.25;
        if(value) 
            *value = (x*x+y*y)/2+z*(x*x-y*y/3)*y+z*z;
        if(deriv) {
            deriv->dx = x*(1+2*z*y);
            deriv->dy = y+z*(x*x-y*y);
            deriv->dz = (x*x-y*y/3)*y+2*z;
        }
        if(deriv2) {
            deriv2->dx2 = (1+2*z*y);
            deriv2->dxdy= 2*z*x;
            deriv2->dxdz= 2*y*x;
            deriv2->dy2 = 1-2*z*y;
            deriv2->dydz= x*x-y*y;
            deriv2->dz2 = 2;
        }
    }
};

template<> class MyScalarFunction<coord::Cyl>: public coord::IScalarFunction<coord::Cyl> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosCyl& p,
        double* value=NULL, coord::GradCyl* deriv=NULL, coord::HessCyl* deriv2=NULL, double /*time*/=0) const
    {   // same potential expressed in different coordinates
        double sinphi,cosphi;
        math::sincos(p.phi, sinphi, cosphi);
        double sin2=pow_2(sinphi), R2=pow_2(p.R), R3=p.R*R2;
        if(value)
            *value = R2*(3+p.R*p.z*sinphi*(6-8*sin2))/6+p.z*p.z;
        if(deriv) {
            deriv->dR  = p.R*(1+p.R*p.z*sinphi*(3-4*sin2));
            deriv->dphi= R3*p.z*cosphi*(1-4*sin2);
            deriv->dz  = R3*sinphi*(3-4*sin2)/3+2*p.z;
        }
        if(deriv2) {
            deriv2->dR2   = 1+2*p.R*p.z*sinphi*(3-4*sin2);
            deriv2->dRdphi= R2*p.z*cosphi*(3-12*sin2);
            deriv2->dRdz  = R2*sinphi*(3-4*sin2);
            deriv2->dphi2 = R3*p.z*sinphi*(-9+12*sin2);
            deriv2->dzdphi= R3*cosphi*(1-4*sin2);
            deriv2->dz2   = 2;
        }
    }
};

template<> class MyScalarFunction<coord::Sph>: public coord::IScalarFunction<coord::Sph> {
public:
    MyScalarFunction() {};
    virtual ~MyScalarFunction() {};
    virtual void evalScalar(const coord::PosSph& p,
        double* value=NULL, coord::GradSph* deriv=NULL, coord::HessSph* deriv2=NULL, double /*time*/=0) const
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
    return p.x==0 && p.y==0; }
template<> bool isSingular(const coord::PosCyl& p) {
    return p.R==0; }
template<> bool isSingular(const coord::PosSph& p) {
    return p.r==0 || p.theta==0; }
template<> bool isSingular(const coord::PosAxi& p) {
    return p.rho==0 && (p.cs.Delta2>0 ? 1/p.cotnu==0 : p.cs.Delta2<0 ? p.cotnu==0 : true); }

template<typename coordSys> std::string name(coordSys cs) { return cs.name(); }
template<> std::string name(coord::Axi cs) {
    return std::string(cs.name()) + " (D^2=" + utils::toString(cs.Delta2) + ")";
}

/** the test itself: perform conversion of position/velocity from one coord system to the other and back */
template<typename srcCS, typename destCS>
bool test_conv(const coord::PosVelT<srcCS>& srcpoint, const destCS& coordsys=destCS())
{
    const coord::PosVelT<destCS> dstpoint = coord::toPosVel<srcCS,destCS>(srcpoint, coordsys);
    const coord::PosVelT<srcCS>  invpoint = coord::toPosVel<destCS,srcCS>(dstpoint);
    double src[6],dst[6],inv[6];
    srcpoint.unpack_to(src);
    dstpoint.unpack_to(dst);
    invpoint.unpack_to(inv);
    double Lsrc = coord::Ltotal(srcpoint), Ldst=coord::Ltotal(dstpoint);
    double vsrc = pow_2(src[3])+pow_2(src[4])+pow_2(src[5]);
    double vdst = pow_2(dst[3])+pow_2(dst[4])+pow_2(dst[5]);
    bool samepos= equalPos(srcpoint, invpoint, eps);
    bool samevel= equalVel(srcpoint, invpoint, eps);
    bool sameL  = math::fcmp(Lsrc, Ldst, eps)==0;
    bool samev2 = math::fcmp(vsrc, vdst, eps)==0;

    coord::PosDerivT<srcCS,destCS> derivStoD, derivStoDa, derivStoDb;
    coord::PosDerivT<destCS,srcCS> derivDtoI;
    coord::PosDeriv2T<srcCS,destCS> deriv2StoD;
    coord::PosDeriv2T<destCS,srcCS> deriv2DtoI;
    coord::toPosDeriv<srcCS,destCS>(srcpoint, &derivStoD, &deriv2StoD, coordsys);
    coord::toPosDeriv<destCS,srcCS>(dstpoint, &derivDtoI, &deriv2DtoI);
    coord::toPosDeriv<srcCS,destCS>(coord::PosT<srcCS>(src[0]+eps, src[1], src[2]), &derivStoDa, NULL, coordsys);
    coord::toPosDeriv<srcCS,destCS>(coord::PosT<srcCS>(src[0], src[1]+eps, src[2]), &derivStoDb, NULL, coordsys);
    coord::GradT<srcCS> srcgrad;
    coord::HessT<srcCS> srchess;
    MyScalarFunction<srcCS> Fnc;
    Fnc.evalScalar(srcpoint, NULL, &srcgrad, &srchess);
    const coord::GradT<destCS> dstgrad = coord::toGrad<srcCS,destCS>(srcgrad, derivDtoI);
    const coord::HessT<destCS> dsthess = coord::toHess<srcCS,destCS>(srcgrad, srchess, derivDtoI, deriv2DtoI);
    const coord::GradT<srcCS>  invgrad = coord::toGrad<destCS,srcCS>(dstgrad, derivStoD);
    const coord::HessT<srcCS>  invhess = coord::toHess<destCS,srcCS>(dstgrad, dsthess, derivStoD, deriv2StoD);
    // not all components of the Hessian can be accurately converted for the Spherical system,
    // when the angle theta is close to pi, the relative precision of the difference deteriorates
    double epsh = eps;
    if(srcCS::name() == coord::Sph::name() && src[1]/*theta*/ >= 3.1415)
        epsh *= M_PI / (M_PI - src[1]);  // loosen the tolerance
    // for the spheroidal system, the accuracy of the Hessian also deteriorates near the focal points,
    // a quick-and-dirty scaling factor below takes this into account
    if(destCS::name() == coord::Axi::name())
        epsh /= fmin(1, pow_2(dst[0]) + fmin(pow_2(dst[1]), pow_2(1/dst[1])));
    bool samegrad = coord::equalGrad(srcgrad, invgrad, eps);
    bool samehess = coord::equalHess(srchess, invhess, epsh);
    bool isSing   = isSingular(dstpoint);
    bool ok = samepos && sameL && samevel && samev2 &&
        ((samegrad && samehess) || isSing);   // in the singular case, grad/hess may be undefined
    std::cout << (ok?"OK  ":"\033[1;31mFAILED\033[0m  ");
    std::cout << srcCS::name() << " => " <<
        name<destCS>(coordsys) << " => " << srcCS::name();
    if(!samepos) std::cout << " pos";
    if(!samevel) std::cout << " vel " << (inv[3]-src[3]) << ", " << (inv[4]-src[4]) << ", " << (inv[5]-src[5]);
    if(!sameL)   std::cout << " L_total";
    if(!samev2)  std::cout << " v^2";
    if(isSing && (!samegrad || !samehess)) std::cout << "  singular";  // show that it expectedly failed
    if(!samegrad)std::cout << " gradient";
    if(!samehess)std::cout << " hessian";
    std::cout << "\n";
    return ok;
}

// test the equivalence of two angles, ignoring possible offsets of 2pi
inline bool angles_equal(double a, double b)
{
    double sa, ca, sb, cb;
    math::sincos(a, sa, ca);
    math::sincos(b, sb, cb);
    return fabs(sa-sb) < eps && fabs(ca-cb) < eps;
}

// test of rotation matrices and their inverse
bool test_rotation()
{
    const coord::Orientation orientation(1.47, 2.58, 3.69);
    double xyz[3] = {1, 2, -3}, rot[3], inv[3];
    orientation.toRotated(xyz, rot);
    orientation.fromRotated(rot, inv);
    coord::Vel2Car a, b, c;
    a.vx2  = 2.34; a.vy2  =  3.45; a.vz2  =  4.56;
    a.vxvy = 1.23; a.vxvz = -0.98; a.vyvz = -2.34;
    b = orientation.toRotated(a);
    c = orientation.fromRotated(b);
    bool ok =
        math::fcmp(xyz[0], inv[0], eps) == 0 &&
        math::fcmp(xyz[1], inv[1], eps) == 0 &&
        math::fcmp(xyz[2], inv[2], eps) == 0 &&
        math::fcmp(a.vx2 , c.vx2 , eps) == 0 &&
        math::fcmp(a.vy2 , c.vy2 , eps) == 0 &&
        math::fcmp(a.vz2 , c.vz2 , eps) == 0 &&
        math::fcmp(a.vxvy, c.vxvy, eps) == 0 &&
        math::fcmp(a.vxvz, c.vxvz, eps) == 0 &&
        math::fcmp(a.vyvz, c.vyvz, eps) == 0;

    // test the conversion between Euler angles and rotation matrix in both directions
    for(int i=0; i<1000; i++) {
        // a mixture of random angles and various degenerate situations
        double alpha1, beta1, gamma1,
            alpha0 = i%4>1 ? M_PI * (math::random()*2-1) : (i%4==0 ? M_PI : -M_PI),
            beta0  = i%5>1 ? M_PI *  math::random()      : (i%5==0 ? M_PI : 0),
            gamma0 = i%7>1 ? M_PI * (math::random()*2-1) : (i%7==0 ? M_PI : -M_PI);
        coord::Orientation ori0(alpha0, beta0, gamma0);
        ori0.toEulerAngles(alpha1, beta1, gamma1);
        ok &= fabs(beta0 - beta1) < eps;
        // in the case of beta==0 or pi, only the sum or the difference of the angles can be recovered
        if(beta0 == 0)
            ok &= angles_equal(alpha0 + gamma0, alpha1 + gamma1);
        else if(beta0 == M_PI)
            ok &= angles_equal(alpha0 - gamma0, alpha1 - gamma1);
        else
            ok &= angles_equal(alpha0, alpha1) && angles_equal(gamma0, gamma1);
        // in all cases, the rotation matrix reconstructed from the returned angles should be identical
        coord::Orientation ori1(alpha1, beta1, gamma1);
        for(int k=0; k<9; k++)
            ok &= fabs(ori0.mat[k] - ori1.mat[k]) < eps;
    }
    return ok;
}

template<typename CS> bool isNotNan(const coord::PosT<CS>& p);
template<> bool isNotNan(const coord::PosCar& p) { return p.x==p.x && p.y==p.y && p.z==p.z; }
template<> bool isNotNan(const coord::PosCyl& p) { return p.R==p.R && p.phi==p.phi && p.z==p.z; }
template<> bool isNotNan(const coord::PosSph& p) { return p.r==p.r && p.phi==p.phi && p.theta==p.theta; }
template<> bool isNotNan(const coord::PosAxi& p) { return p.rho==p.rho && p.phi==p.phi && p.cotnu==p.cotnu; }

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
    test_inf<coord::Car, coord::Sph>(0, 0, INFINITY) &&
    test_inf<coord::Cyl, coord::Axi>(INFINITY, 0, 0) &&
    test_inf<coord::Cyl, coord::Axi>(INFINITY, 1, 0) &&
    test_inf<coord::Axi, coord::Cyl>(INFINITY, 0, 0) &&
    test_inf<coord::Axi, coord::Cyl>(INFINITY, 1, 0);
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
    {3, 2, 1, 4, 5, 6},   // ordinary point
    {1, 0, 2, 1, 2, 3},   // point in x-y plane
    {0, 1, 0, 0,-1, 0},   // point along z axis, vphi must be zero
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
    if(!test_inf()) {
        std::cout << "Coordinate conversion at infinity \033[1;31mFAILED\033[0m\n";
        passed=false;
    }
    if(!test_rotation()) {
        std::cout << "Rotation test \033[1;31mFAILED\033[0m\n";
        passed=false;
    }

    for(int n=0; n<numtestpoints; n++) {
        coord::PosVelCar pvcar(posvel_car[n]);
        std::cout << " :::Cartesian point::: " << pvcar << "\n";
        passed &= test_conv<coord::Car, coord::Cyl>(pvcar);
        passed &= test_conv<coord::Car, coord::Sph>(pvcar);
        coord::PosVelCyl pvcyl(posvel_cyl[n]);
        std::cout << " :::Cylindrical point::: " << pvcyl << "\n";
        passed &= test_conv<coord::Cyl, coord::Car>(pvcyl);
        passed &= test_conv<coord::Cyl, coord::Sph>(pvcyl);
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl);
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(+0.9999));
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(+1.0001));
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(-0.9999));
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(-1.0001));
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(+1.0));
        passed &= test_conv<coord::Cyl, coord::Axi>(pvcyl, coord::Axi(-1.0));
        coord::PosVelSph pvsph(posvel_sph[n]);
        std::cout << " :::Spherical point::: " << pvsph << "\n";
        passed &= test_conv<coord::Sph, coord::Car>(pvsph);
        passed &= test_conv<coord::Sph, coord::Cyl>(pvsph);
    }
    if(passed)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
