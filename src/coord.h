/** \file    coord.h 
    \brief   General-purpose coordinate types and routines
    \author  Eugene Vasiliev
    \date    2015-2023

This module provides the general framework for working with different coordinate systems.

It is heavily templated but this shouldn't intimidate the end user, because
the most important data structures and routines have dedicated non-templated aliases.

The fundamental data types are the following:

- coordinate systems (the simplest ones have no parameters at all);
- positions and position-velocity pairs in different coordinate systems;
- an abstract class for a scalar function defined in a particular coordinate system;
- gradients and hessians of scalar functions in different coordinate systems;
- coefficients of coordinate transformations between different systems:
  derivatives of destination coords by source coords (i.e. the jacobian matrix) and
  second derivatives of destination coords by source coords.

The fundamental routines operating on these structures are the following:

- conversion of position and position-velocity from one coordinate system to another;
- computation of coefficients of coordinate transformation (first/second derivatives);
- transformation of gradients and hessians;
- the "all-mighty function" that uses the above primitives to perform the following task:
  suppose we have a class that computes the value, gradient and hessian of a scalar function
  in a particular coordinate system ("evaluation CS"), and we need these quantities
  in a different system ("output CS").
  The routine transforms the input coordinates from outputCS to evalCS, along with their
  derivatives; computes the value, gradient and hessian in evalCS, transforms them back
  to outputCS. A modification of this routine uses another intermediate CS for the situation
  when a direct transformation is not implemented.
  The main application of this routine is the computation of potentials and forces
  in different coordinate systems.
*/
#pragma once
#include "math_base.h"

/** Classes and routines for representing position/velocity points,
    gradients and hessians of scalar functions (e.g., gravitational potential),
    and transformations between coordinate systems.
*/
namespace coord {

/// \name   Primitive data types: symmetry in 3d space
///@{

/** defines the symmetry properties of a function in three-dimensional space */
enum SymmetryType{
    ST_NONE         = 0, ///< no symmetry whatsoever
    // basic symmetries:
    ST_XREFLECTION  = 1, ///< change of sign in x (flip about yz plane)
    ST_YREFLECTION  = 2, ///< change of sign in y
    ST_ZREFLECTION  = 4, ///< change of sign in z
    ST_REFLECTION   = 8, ///< mirror reflection about origin (change of sign of all coordinates simultaneously)
    ST_ZROTATION    =16, ///< rotation about z axis
    ST_ROTATION     =32, ///< rotation about arbitrary axis
    // special flag indicating lack of information
    ST_UNKNOWN      =64, ///< unspecified - decided automatically
    // composite symmetries:
    /// bisymmetric: a combination of z-reflection and a mirror symmetry about origin,
    /// resulting in a mirror symmetry in xy-plane - change of sign in x and y simultaneously;
    /// this is suitable, for instance, to describe a spiral pattern containing only even-m modes
    /// (both cosine and sine terms) and symmetric w.r.t sign change in z coordinate.
    ST_BISYMMETRIC  = ST_ZREFLECTION | ST_REFLECTION,
    /// triaxial - reflection about principal planes (change of sign of any coordinate):
    /// note that while the combination of reflection symmetries about all three principal planes
    /// implies the reflection symmetry about origin (mirroring), the converse is not true,
    /// that's why these are separate concepts; if all three plane-reflection symmetries are present,
    /// then mirror-reflection is implied, and this all is encoded in the ST_TRIAXIAL value
    ST_TRIAXIAL     = ST_XREFLECTION | ST_YREFLECTION | ST_ZREFLECTION | ST_REFLECTION,
    ST_AXISYMMETRIC = ST_TRIAXIAL | ST_ZROTATION,    ///< axial symmetry combined with plane symmetry
    ST_SPHERICAL    = ST_AXISYMMETRIC | ST_ROTATION, ///< spherical symmetry
};

/** test for symmetry w.r.t.change of sign in x */
inline bool isXReflSymmetric(const SymmetryType sym) {
    return (sym & ST_XREFLECTION) == ST_XREFLECTION;
}

/** test for symmetry w.r.t.change of sign in y */
inline bool isYReflSymmetric(const SymmetryType sym) {
    return (sym & ST_YREFLECTION) == ST_YREFLECTION;
}

/** test for symmetry w.r.t.change of sign in z */
inline bool isZReflSymmetric(const SymmetryType sym) {
    return (sym & ST_ZREFLECTION) == ST_ZREFLECTION;
}

/** test for symmetry w.r.t.mirror reflection */
inline bool isReflSymmetric(const SymmetryType sym) {
    return (sym & ST_REFLECTION) == ST_REFLECTION;
}

/** test for rotational symmetry about z axis */
inline bool isZRotSymmetric(const SymmetryType sym) {
    return (sym & ST_ZROTATION) == ST_ZROTATION;
}

/** test for rotational symmetry about any axis (implies spherical symmetry) */
inline bool isRotSymmetric(const SymmetryType sym) {
    return (sym & ST_ROTATION) == ST_ROTATION;
}

/** test for symmetry under xy-reflection */
inline bool isBisymmetric(const SymmetryType sym) {
    return (sym & ST_BISYMMETRIC) == ST_BISYMMETRIC;
}

/** test for triaxial symmetry (reflection about any of the three principal planes) */
inline bool isTriaxial(const SymmetryType sym) {
    return (sym & ST_TRIAXIAL) == ST_TRIAXIAL;
}

/** test for axisymmetry in the 'common definition'
    (i.e., invariance under rotation about z axis and under change of sign in z) */
inline bool isAxisymmetric(const SymmetryType sym) {
    return (sym & ST_AXISYMMETRIC) == ST_AXISYMMETRIC;
}

/** test for spherical symmetry */
inline bool isSpherical(const SymmetryType sym) {
    return (sym & ST_SPHERICAL) == ST_SPHERICAL;
}

/** test for unknown/invalid symmetry */
inline bool isUnknown(const SymmetryType sym) {
    return sym<0 || sym>ST_SPHERICAL;
}

///@}
/// \name   Primitive data types: coordinate systems
///@{

/// trivial coordinate systems don't have any parameters,
/// their class names are simply used as tags in the rest of the code

/// cartesian coordinate system (galactocentric)
struct Car{
    static const char* name() { return "Cartesian"; }
};

/// cylindrical coordinate system (galactocentric)
struct Cyl{
    static const char* name() { return "Cylindrical"; }
};

/// spherical coordinate system (galactocentric)
struct Sph{
    static const char* name() { return "Spherical"; }
};

//  less trivial:
/** prolate spheroidal coordinate system, defined by a single parameter
    Delta>0 (focal distance).
    The traditionally used two parameters alpha and gamma (e.g., de Zeeuw 1985)
    are not independent, so we define  Delta^2 = gamma - alpha.
    TODO: remove since it is superseded by Axi (currently still needed by Staeckel Fudge)
*/
struct ProlSph{
    const double Delta2;     ///< Delta^2 = gamma - alpha > 0
    explicit ProlSph(double Delta);   ///< Delta is the focal distance
    static const char* name() { return "Prolate spheroidal"; }
};

/** Universal axisymmetric spheroidal coordinate system,
    depending on the squared focal distance Delta^2:
    prolate (>0), spherical (=0) or oblate (<0).
    In the prolate case, the focal points are located at R=0, z=+-Delta;
    in the oblate case, the focal circle is at z=0, R=|Delta|.
*/
struct Axi{
    const double Delta2;     ///< Delta^2>0: prolate, <0: oblate, =0: spherical
    explicit Axi(double _Delta2=0) : Delta2(_Delta2) {}
    static const char* name() { return "Axisymmetric spheroidal"; }
};

///@}
/// \name   Primitive data types: position in different coordinate systems
///@{

/// position in arbitrary coordinates:
/// the data types are defined as templates with the template parameter
/// being any of the coordinate system names defined above
template<typename CoordT> struct PosT;

/// position in cartesian coordinates
template<> struct PosT<Car>{
    double x, y, z;   ///< three cartesian coordinates
    PosT() {}
    PosT(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}
};
/// an alias to templated type specialization of position in cartesian coordinates
typedef struct PosT<Car> PosCar;

/// position in cylindrical coordinates
template<> struct PosT<Cyl>{
    double R;   ///< cylindrical radius = sqrt(x^2+y^2)
    double z;   ///< z coordinate
    double phi; ///< azimuthal angle in x-y plane [0:2pi)
    PosT() {}
    PosT(double _R, double _z, double _phi) : R(_R), z(_z), phi(_phi) {}
};
typedef struct PosT<Cyl> PosCyl;

/// position in spherical coordinates
template<> struct PosT<Sph>{
    double r;     ///< spherical radius
    double theta; ///< polar angle [0:pi] - 0 means along z axis in positive direction,
                  ///< pi is along z in negative direction, pi/2 is in x-y plane
    double phi;   ///< azimuthal angle in x-y plane [0:2pi)
    PosT() {}
    PosT(double _r, double _theta, double _phi) : r(_r), theta(_theta), phi(_phi) {}
};
typedef struct PosT<Sph> PosSph;

/** position in prolate spheroidal coordinates.
    We use a somewhat different definition from de Zeeuw 1985, namely:
    the value of `nu` keeps track of the sign of z, so that the conversion between cylindrical
    and prolate spheroidal coordinates is invertible. */
template<> struct PosT<ProlSph>{
    double lambda;  ///< lies in the range [delta:infinity)
    double nu;      ///< lies in the range [-delta:delta]; negative for z<0
    double phi;     ///< usual azimuthal angle
    const ProlSph coordsys;  ///< a point means nothing without specifying its coordinate system
    PosT(double _lambda, double _nu, double _phi, const ProlSph& _coordsys):
        lambda(_lambda), nu(_nu), phi(_phi), coordsys(_coordsys) {}
};
typedef struct PosT<ProlSph> PosProlSph;

/** position in universal axisymmetric spheroidal coordinates.
    "rho" is the analogue of the spherical radius and equivalent to it in the spherical case (Delta=0).
    In the prolate spheroidal case (Delta^2>0), it is related to the two other possible coordinate
    definitions as rho = D sinh(u) = sqrt(lambda^2 - Delta^2), and is zero when R=0, |z|<=Delta.
    In the oblate spheroidal case (Delta^2<0), it is defined as rho = D cosh(u)
    and is zero when z=0, R<=|Delta|.
    "nu" is the analogue of the polar angle theta and equivalent to it in the spherical case.
    nu=0  when R=0, z>= sqrt(max(0, Delta^2)), i.e. along the z axis beyond the focal points,
    nu=pi when R=0, z<=-sqrt(max(0, Delta^2)), and
    nu=pi/2 in the z=0 plane, except when R<|Delta| in the oblate system:
    in this case, there is a discontinuity in v when one crosses the z=0 plane,
    and it is defined to be <pi/2 on the crossing line.
    In practice, we use cotnu = cos(nu)/sin(nu) instead of nu itself, for two reasons:
    1) to avoid trigonometric functions when transforming between cylindrical and spheroidal coords,
    2) to avoid loss of precision when nu approaches pi/2 (i.e. along the R axis)
    or pi (i.e. along the negative z axis). In these cases cotnu approaches zero or -infinity,
    while close to the positive z axis it approaches +infinity, in all these situations
    the full precision mantissa is available.
    However, the first and second derivatives are still computed w.r.t. nu, not cotnu,
    to avoid divergence along the z axis.
    "phi" is the usual azimuthal angle.
    The transformation between (rho,nu) and (R,z) is given by
    R = chi * sin(nu) = chi / sqrt(1 + cotnu^2),
    z = psi * cos(nu) = psi / sqrt(1 + cotnu^2) * cotnu,
    where chi = rho, psi = sqrt(rho^2 + Delta^2) in the prolate spheroidal case (Delta>0),
    or    psi = rho, chi = sqrt(rho^2 - Delta^2) in the  oblate spheroidal case (Delta<0),
    or    chi = psi = r  in the spherical case.
*/
template<> struct PosT<Axi>{
    double rho;    ///< analogous to the spherical radius, lies in the range [0..infinity)
    double cotnu;  ///< cotangent of the analogue of the polar angle theta, [-inf..inf]
    double phi;    ///< usual azimuthal angle
    Axi cs;        ///< coordinate system
    PosT(double _rho, double _cotnu, double _phi, Axi _cs=Axi(0)) :
        rho(_rho), cotnu(_cotnu), phi(_phi), cs(_cs) {}
};
typedef struct PosT<Axi> PosAxi;

/** projected position (two Cartesian coordinates on the sky plane) */
struct PosProj{
    double X, Y;
    PosProj() {};
    PosProj(double _X, double _Y) : X(_X), Y(_Y) {}
    explicit PosProj(const PosCar& pos) : X(pos.x), Y(pos.y) {}  // discard the z component
};

///@}
/// \name   Primitive data types: velocity in different coordinate systems
///@{

/// velocity in arbitrary coordinates
template<typename CoordT> struct VelT;

/// velocity in cartesian coordinates
template<> struct VelT<Car> {
    double vx, vy, vz;   ///< components of velocity along three cartesian axes
    VelT() {}
    VelT(double _vx, double _vy, double _vz) : vx(_vx), vy(_vy), vz(_vz) {}
};
/// an alias to templated type specialization of velocity for cartesian coordinates
typedef struct VelT<Car> VelCar;

/// velocity in cylindrical coordinates
/// (this is not the same as time derivative of position in these coordinates!)
template<> struct VelT<Cyl> {
    double vR, vz, vphi;
    VelT() {}
    VelT(double _vR, double _vz, double _vphi) : vR(_vR), vz(_vz), vphi(_vphi) {}
};
typedef struct VelT<Cyl> VelCyl;

/// velocity in spherical coordinates
/// (this is not the same as time derivative of position in these coordinates!)
template<> struct VelT<Sph> {
    double vr, vtheta, vphi;
    VelT() {}
    VelT(double _vr, double _vtheta, double _vphi) : vr(_vr), vtheta(_vtheta), vphi(_vphi) {}
};
typedef struct VelT<Sph> VelSph;

/// velocity components (not momenta) in universal spheroidal coordinates
template<> struct VelT<Axi> {
    double vrho, vnu, vphi;
    VelT(double _vrho, double _vnu, double _vphi) : vrho(_vrho), vnu(_vnu), vphi(_vphi) {}
};
typedef struct VelT<Axi> VelAxi;

///@}
/// \name   Primitive data types: second moments of velocity in different coordinate systems
///@{

/// second moment of velocity in arbitrary coordinates
template<typename CoordT> struct Vel2T;

/// velocity in cartesian coordinates
template<> struct Vel2T<Car> {
    double vx2, vy2, vz2, vxvy, vxvz, vyvz;
};
typedef struct Vel2T<Car> Vel2Car;

/// second moment of velocity in cylindrical coordinates
template<> struct Vel2T<Cyl> {
    double vR2, vz2, vphi2, vRvz, vRvphi, vzvphi;
};
typedef struct Vel2T<Cyl> Vel2Cyl;

/// second moment of velocity in spherical coordinates
template<> struct Vel2T<Sph> {
    double vr2, vtheta2, vphi2, vrvtheta, vrvphi, vthetavphi;
};
typedef struct Vel2T<Sph> Vel2Sph;

///@}
/// \name   Primitive data types: position-velocity pairs in different coordinate systems
///@{

/// combined position and velocity in arbitrary coordinates
template<typename CoordT> struct PosVelT;

/// combined position and velocity in cartesian coordinates
template<> struct PosVelT<Car>: public PosCar, public VelCar {
    PosVelT() {}

    /// initialize from position and velocity
    PosVelT(const PosCar& pos, const VelCar& vel) : PosCar(pos), VelCar(vel) {}

    /// initialize from explicitly given numbers
    PosVelT(double _x, double _y, double _z, double _vx, double _vy, double _vz) :
        PosCar(_x, _y, _z), VelCar(_vx, _vy, _vz) {}

    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT(const double p[]) :
        PosCar(p[0], p[1], p[2]), VelCar(p[3], p[4], p[5]) {}

    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=x; out[1]=y; out[2]=z; out[3]=vx; out[4]=vy; out[5]=vz; }

    /// convert velocities to momenta (canonically conjugate to positions) - trivial in this case
    void momenta(double& px, double& py, double& pz) const { px=vx; py=vy; pz=vz; }
};
/// an alias to templated type specialization of position and velocity for cartesian coordinates
typedef struct PosVelT<Car> PosVelCar;

/// combined position and velocity in cylindrical coordinates
template<> struct PosVelT<Cyl>: public PosCyl, public VelCyl {
    PosVelT() {}

    /// initialize from position and velocity
    PosVelT(const PosCyl& pos, const VelCyl& vel) : PosCyl(pos), VelCyl(vel) {}

    /// initialize from explicitly given numbers
    PosVelT(double _R, double _z, double _phi, double _vR, double _vz, double _vphi) :
        PosCyl(_R, _z, _phi), VelCyl(_vR, _vz, _vphi) {};

    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT(const double p[]) :
        PosCyl(p[0], p[1], p[2]), VelCyl(p[3], p[4], p[5]) {};

    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=R; out[1]=z; out[2]=phi; out[3]=vR; out[4]=vz; out[5]=vphi; }

    /// convert velocities to momenta (canonically conjugate to positions)
    void momenta(double& pR, double& pz, double& pphi) const { pR=vR; pz=vz; pphi=vphi*R; }
};
typedef struct PosVelT<Cyl> PosVelCyl;

/// combined position and velocity in spherical coordinates
template<> struct PosVelT<Sph>: public PosSph, public VelSph {
    PosVelT() {}

    /// initialize from position and velocity
    PosVelT(const PosSph& pos, const VelSph& vel) : PosSph(pos), VelSph(vel) {}

    /// initialize from explicitly given numbers
    PosVelT(double _r, double _theta, double _phi, double _vr, double _vtheta, double _vphi) :
        PosSph(_r, _theta, _phi), VelSph(_vr, _vtheta, _vphi) {};

    /// initialize from an array of 6 floats (i.e., from a serialized array)
    PosVelT(const double p[]) :
        PosSph(p[0], p[1], p[2]), VelSph(p[3], p[4], p[5]) {};

    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=r; out[1]=theta; out[2]=phi; out[3]=vr; out[4]=vtheta; out[5]=vphi; }

    /// convert velocities to momenta (canonically conjugate to positions)
    void momenta(double& pr, double& ptheta, double& pphi) const;
};
typedef struct PosVelT<Sph> PosVelSph;

/// position and velocity in prolate spheroidal coordinates
/// TODO: remove once Staeckel Fudge is reimplemented using Axi
template<> struct PosVelT<ProlSph>: public PosProlSph{
    double lambdadot, nudot, phidot;  ///< time derivatives of position variables
    PosVelT(const PosProlSph& pos, double _lambdadot, double _nudot, double _phidot):
        PosProlSph(pos), lambdadot(_lambdadot), nudot(_nudot), phidot(_phidot) {}
    void unpack_to(double *out) const {
        out[0]=lambda; out[1]=nu; out[2]=phi; out[3]=lambdadot; out[4]=nudot; out[5]=phidot; }
};
typedef struct PosVelT<ProlSph> PosVelProlSph;

/// position and velocity in universal spheroidal coordinates
template<> struct PosVelT<Axi>: public PosAxi, public VelAxi {
    PosVelT(const PosAxi& pos, const VelAxi& vel) : PosAxi(pos), VelAxi(vel) {}

    /// serialize into an array of 6 floating-point numbers
    void unpack_to(double *out) const {
        out[0]=rho; out[1]=cotnu; out[2]=phi; out[3]=vrho; out[4]=vnu; out[5]=vphi; }

    /// convert velocities to momenta (canonically conjugate to positions)
    void momenta(double& prho, double& pnu, double& pphi) const;
};
typedef struct PosVelT<Axi> PosVelAxi;

///@}
/// \name   Primitive data types: gradient of a scalar function in different coordinate systems
///@{

/// components of a gradient in a given coordinate system
template<typename CoordT> struct GradT;

/// gradient of scalar function in cartesian coordinates
template<> struct GradT<Car>{
    double dx, dy, dz;
};
/// an alias to templated type specialization of gradient for cartesian coordinates
typedef struct GradT<Car> GradCar;

/// gradient of scalar function in cylindrical coordinates
template<> struct GradT<Cyl>{
    double dR, dz, dphi;
};
typedef struct GradT<Cyl> GradCyl;

/// gradient of scalar function in spherical coordinates
template<> struct GradT<Sph>{
    double dr, dtheta, dphi;
};
typedef struct GradT<Sph> GradSph;

/// gradient of scalar function in prolate spheroidal coordinates
template<> struct GradT<ProlSph>{
    double dlambda, dnu, dphi;
};
typedef struct GradT<ProlSph> GradProlSph;

/// gradient of scalar function in universal spheroidal coordinates
template<> struct GradT<Axi>{
    double drho, dnu, dphi;
};
typedef struct GradT<Axi> GradAxi;

///@}
/// \name   Primitive data types: hessian of a scalar function in different coordinate systems
///@{

/// components of a hessian of a scalar function (matrix of its second derivatives)
template<typename CoordT> struct HessT;

/// Hessian of scalar function F in cartesian coordinates: d2F/dx^2, d2F/dxdy, etc
template<> struct HessT<Car>{
    double dx2, dy2, dz2, dxdy, dydz, dxdz;
};
typedef struct HessT<Car> HessCar;

/// Hessian of scalar function in cylindrical coordinates
template<> struct HessT<Cyl>{
    double dR2, dz2, dphi2, dRdz, dzdphi, dRdphi;
};
typedef struct HessT<Cyl> HessCyl;

/// Hessian of scalar function in spherical coordinates
template<> struct HessT<Sph>{
    double dr2, dtheta2, dphi2, drdtheta, dthetadphi, drdphi;
};
typedef struct HessT<Sph> HessSph;

/// Hessian of scalar function in prolate spheroidal coordinates
template<> struct HessT<ProlSph>{
    double dlambda2, dnu2, dlambdadnu;  ///< note: derivatives by phi are assumed to be zero
};
typedef struct HessT<ProlSph> HessProlSph;

/// Hessian of scalar function in universal spheroidal coordinates
template<> struct HessT<Axi>{
    double drho2, dnu2, dphi2, drhodnu, drhodphi, dnudphi;
};
typedef struct HessT<Axi> HessAxi;

///@}
/// \name   Abstract interface classes for scalar functions
///@{

/** Prototype of a scalar function which is computed in a particular coordinate system */
template<typename CoordT>
class IScalarFunction {
public:
    IScalarFunction() {};
    virtual ~IScalarFunction() {};
    /** Evaluate any combination of value, gradient and hessian of the function at a given point.
        Each of these quantities is computed and stored in the output pointer if it was not NULL. */
    virtual void evalScalar(const PosT<CoordT>& x,
        double* value=NULL,
        GradT<CoordT>* deriv=NULL,
        HessT<CoordT>* deriv2=NULL,
        double time=0) const=0;
};

///@}
/// \name   Data types containing conversion coefficients between different coordinate systems
///@{

/** derivatives of coordinate transformation from source to destination
    coordinate systems (srcCS=>destCS): derivatives of destination variables
    w.r.t.source variables, aka Jacobian */
template<typename srcCS, typename destCS> struct PosDerivT;

/** instantiations of the general template for derivatives of coordinate transformations
    are separate structures for each pair of coordinate systems */
template<> struct PosDerivT<Car, Cyl> {
    double dRdx, dRdy, dphidx, dphidy;
};
template<> struct PosDerivT<Car, Sph> {
    double drdx, drdy, drdz, dthetadx, dthetady, dthetadz, dphidx, dphidy;
};
template<> struct PosDerivT<Cyl, Car> {
    double dxdR, dxdphi, dydR, dydphi;
};
template<> struct PosDerivT<Cyl, Sph> {
    double drdR, drdz, dthetadR, dthetadz;
};
template<> struct PosDerivT<Sph, Car> {
    double dxdr, dxdtheta, dxdphi, dydr, dydtheta, dydphi, dzdr, dzdtheta;
};
template<> struct PosDerivT<Sph, Cyl> {
    double dRdr, dRdtheta, dzdr, dzdtheta;
};
template<> struct PosDerivT<Cyl, ProlSph> {
    double dlambdadR, dlambdadz, dnudR, dnudz;
};
template<> struct PosDerivT<ProlSph, Cyl> {
    double dRdlambda, dRdnu, dzdlambda, dzdnu;
};
template<> struct PosDerivT<Cyl, Axi> {
    double drhodR, drhodz, dnudR, dnudz;
};
template<> struct PosDerivT<Axi, Cyl> {
    double dRdrho, dRdnu, dzdrho, dzdnu;
};

/** second derivatives of coordinate transformation from source to destination
    coordinate systems (srcCS=>destCS): d^2(dest_coord)/d(source_coord1)d(source_coord2) */
template<typename srcCS, typename destCS> struct PosDeriv2T;

/** instantiations of the general template for second derivatives of coordinate transformations */
template<> struct PosDeriv2T<Cyl, Car> {
    double d2xdRdphi, d2xdphi2, d2ydRdphi, d2ydphi2;
};
template<> struct PosDeriv2T<Sph, Car> {
    double d2xdrdtheta, d2xdrdphi, d2xdtheta2, d2xdthetadphi, d2xdphi2,
           d2ydrdtheta, d2ydrdphi, d2ydtheta2, d2ydthetadphi, d2ydphi2,
           d2zdrdtheta, d2zdtheta2;
};
template<> struct PosDeriv2T<Car, Cyl> {
    double d2Rdx2, d2Rdxdy, d2Rdy2, d2phidx2, d2phidxdy, d2phidy2;
};
template<> struct PosDeriv2T<Sph, Cyl> {
    double d2Rdrdtheta, d2Rdtheta2, d2zdrdtheta, d2zdtheta2;
};
template<> struct PosDeriv2T<Car, Sph> {
    double d2rdx2, d2rdxdy, d2rdxdz, d2rdy2, d2rdydz, d2rdz2,
        d2thetadx2, d2thetadxdy, d2thetadxdz, d2thetady2, d2thetadydz, d2thetadz2,
        d2phidx2, d2phidxdy, d2phidy2;
};
template<> struct PosDeriv2T<Cyl, Sph> {
    double d2rdR2, d2rdRdz, d2rdz2, d2thetadR2, d2thetadRdz, d2thetadz2;
};
template<> struct PosDeriv2T<Cyl, ProlSph> {
    double d2lambdadR2, d2lambdadRdz, d2lambdadz2, d2nudR2, d2nudRdz, d2nudz2;
};
template<> struct PosDeriv2T<ProlSph, Cyl> {
    double d2Rdlambda2, d2Rdlambdadnu, d2Rdnu2, d2zdlambda2, d2zdlambdadnu, d2zdnu2;
};
template<> struct PosDeriv2T<Cyl, Axi>{
    double d2rhodR2, d2rhodRdz, d2rhodz2, d2nudR2, d2nudRdz, d2nudz2;
};
template<> struct PosDeriv2T<Axi, Cyl>{
    double d2Rdrho2, d2Rdrhodnu, d2Rdnu2, d2zdrho2, d2zdrhodnu, d2zdnu2;
};

///@}
/// \name   Convenience arithmetic routines for data types in various coordinate systems
///@{

/// clear the content of a gradient
template<typename CoordT>
void clear(GradT<CoordT>& grad);

/// clear the content of a hessian
template<typename CoordT>
void clear(HessT<CoordT>& hess);

/// linear combination of two position/velocity points:  A := a*A + b*B
template<typename CoordT>
void combine(PosVelT<CoordT>& A, const PosVelT<CoordT>& B, double a=1, double b=1);

/// linear combination of two gradients:  A := a*A + b*B
template<typename CoordT>
void combine(GradT<CoordT>& A, const GradT<CoordT>& B, double a=1, double b=1);

/// linear combination of two hessians:  A := a*A + b*B
template<typename CoordT>
void combine(HessT<CoordT>& A, const HessT<CoordT>& B, double a=1, double b=1);

template<>
inline void clear(GradCar& grad) { grad.dx = grad.dy = grad.dz = 0; }

template<>
inline void clear(GradCyl& grad) { grad.dR = grad.dz = grad.dphi = 0; }

template<>
inline void clear(GradSph& grad) { grad.dr = grad.dtheta = grad.dphi = 0; }

template<>
inline void clear(HessCar& hess)
{ hess.dx2 = hess.dy2 = hess.dz2 = hess.dxdy = hess.dxdz = hess.dydz = 0; }

template<>
inline void clear(HessCyl& hess)
{ hess.dR2 = hess.dz2 = hess.dphi2 = hess.dRdz = hess.dRdphi = hess.dzdphi = 0; }

template<>
inline void clear(HessSph& hess)
{ hess.dr2 = hess.dtheta2 = hess.dphi2 = hess.drdtheta = hess.drdphi = hess.dthetadphi = 0; }

template<>
inline void combine(PosVelCar& A, const PosVelCar& B, double a, double b)
{
    A.x  = a * A.x  + b * B.x;
    A.y  = a * A.y  + b * B.y;
    A.z  = a * A.z  + b * B.z;
    A.vx = a * A.vx + b * B.vx;
    A.vy = a * A.vy + b * B.vy;
    A.vz = a * A.vz + b * B.vz;
}

template<>
inline void combine(GradCar& A, const GradCar& B, double a, double b)
{
    A.dx = a * A.dx + b * B.dx;
    A.dy = a * A.dy + b * B.dy;
    A.dz = a * A.dz + b * B.dz;
}

template<>
inline void combine(HessCar& A, const HessCar& B, double a, double b)
{
    A.dx2  = a * A.dx2  + b * B.dx2;
    A.dy2  = a * A.dy2  + b * B.dy2;
    A.dz2  = a * A.dz2  + b * B.dz2;
    A.dxdy = a * A.dxdy + b * B.dxdy;
    A.dxdz = a * A.dxdz + b * B.dxdz;
    A.dydz = a * A.dydz + b * B.dydz;
}

template<>
inline void combine(GradCyl& A, const GradCyl& B, double a, double b)
{
    A.dR   = a * A.dR   + b * B.dR;
    A.dz   = a * A.dz   + b * B.dz;
    A.dphi = a * A.dphi + b * B.dphi;
}

template<>
inline void combine(HessCyl& A, const HessCyl& B, double a, double b)
{
    A.dR2    = a * A.dR2    + b * B.dR2;
    A.dz2    = a * A.dz2    + b * B.dz2;
    A.dphi2  = a * A.dphi2  + b * B.dphi2;
    A.dRdz   = a * A.dRdz   + b * B.dRdz;
    A.dRdphi = a * A.dRdphi + b * B.dRdphi;
    A.dzdphi = a * A.dzdphi + b * B.dzdphi;
}

template<>
inline void combine(GradSph& A, const GradSph& B, double a, double b)
{
    A.dr     = a * A.dr     + b * B.dr;
    A.dtheta = a * A.dtheta + b * B.dtheta;
    A.dphi   = a * A.dphi   + b * B.dphi;
}

template<>
inline void combine(HessSph& A, const HessSph& B, double a, double b)
{
    A.dr2        = a * A.dr2        + b * B.dr2;
    A.dtheta2    = a * A.dtheta2    + b * B.dtheta2;
    A.dphi2      = a * A.dphi2      + b * B.dphi2;
    A.drdtheta   = a * A.drdtheta   + b * B.drdtheta;
    A.drdphi     = a * A.drdphi     + b * B.drdphi;
    A.dthetadphi = a * A.dthetadphi + b * B.dthetadphi;
}

///@}
/// \name   Routines for conversion between position/velocity in different coordinate systems
///@{

/** universal templated conversion function for positions:
    template parameters srcCS and destCS may be any of the coordinate system names.
    This template function shouldn't be used directly, because the return type depends
    on the template and hence cannot be automatically inferred by the compiler.
    Instead, named functions for each target coordinate system are defined below.
    This and related conversions are implemented bidirectionally between Cyl and
    three other coordinate systems (Car, Sph and Axi) and between Sph and Car,
    but not between Car<->Axi or Sph<->Axi.
    \param[in] from  is the position in the source coordinate system;
    \param[in] coordsys  is the destination coordinate system
    (needed if it has some parameters, otherwise may use the default trivial instantiation);
    \return  the position in the destination coordinate system.
*/
template<typename srcCS, typename destCS>
PosT<destCS> toPos(const PosT<srcCS>& from, const destCS coordsys=destCS());

/** templated conversion functions for positions
    with names reflecting the target coordinate system. */
template<typename srcCS>
inline PosCar toPosCar(const PosT<srcCS>& from) { return toPos<srcCS, Car>(from); }
template<typename srcCS>
inline PosCyl toPosCyl(const PosT<srcCS>& from) { return toPos<srcCS, Cyl>(from); }
template<typename srcCS>
inline PosSph toPosSph(const PosT<srcCS>& from) { return toPos<srcCS, Sph>(from); }

/** universal templated conversion function for coordinates and velocities:
    template parameters srcCS and destCS may be any of the coordinate system names.
    \param[in] from  is the position and velocity in the source coordinate system;
    \param[in] coordsys  is the destination coordinate system
    (needed if it has some parameters, otherwise may use the default trivial instantiation);
    \return  the position and velocity in the destination coordinate system.
*/
template<typename srcCS, typename destCS>
PosVelT<destCS> toPosVel(const PosVelT<srcCS>& from, const destCS coordsys=destCS());

/** templated conversion functions for coordinates and velocities
    with names reflecting the target coordinate system. */
template<typename srcCS>
inline PosVelCar toPosVelCar(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Car>(from); }
template<typename srcCS>
inline PosVelCyl toPosVelCyl(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Cyl>(from); }
template<typename srcCS>
inline PosVelSph toPosVelSph(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Sph>(from); }


/** trivial conversions */
template<> inline PosCar    toPos   <Car,Car>(const PosCar&    p, const Car) { return p; }
template<> inline PosCyl    toPos   <Cyl,Cyl>(const PosCyl&    p, const Cyl) { return p; }
template<> inline PosSph    toPos   <Sph,Sph>(const PosSph&    p, const Sph) { return p; }
template<> inline PosVelCar toPosVel<Car,Car>(const PosVelCar& p, const Car) { return p; }
template<> inline PosVelCyl toPosVel<Cyl,Cyl>(const PosVelCyl& p, const Cyl) { return p; }
template<> inline PosVelSph toPosVel<Sph,Sph>(const PosVelSph& p, const Sph) { return p; }

///@}
/// \name   Routines for conversion between position in different coordinate systems with derivatives
///@{

/** universal templated function for coordinate conversion that provides derivatives of transformation.
    Template parameters srcCS and destCS may be any of the coordinate system names;
    \param[in]  from specifies the point in srcCS coordinate system;
    \param[out] deriv will contain derivatives of the transformation
                (destination coords over source coords);
    \param[out] deriv2 if not NULL, will contain second derivatives of the coordinate transformation;
    \param[in]  coordsys  is the destination coordinate system (only needed if it has some parameters,
                otherwise may use the default trivial instantiation);
    \return     point in destCS coordinate system. */
template<typename srcCS, typename destCS>
PosT<destCS> toPosDeriv(const PosT<srcCS>& from,
    PosDerivT<srcCS, destCS>* deriv, PosDeriv2T<srcCS, destCS>* deriv2=NULL,
    const destCS coordsys=destCS());

///@}
/// \name   Routines for conversion of gradients and hessians between coordinate systems
///@{

/** templated function for transforming a gradient to a different coordinate system */
template<typename srcCS, typename destCS>
GradT<destCS> toGrad(const GradT<srcCS>& src, const PosDerivT<destCS, srcCS>& deriv);

/** templated function for transforming a hessian to a different coordinate system */
template<typename srcCS, typename destCS>
HessT<destCS> toHess(const GradT<srcCS>& srcGrad, const HessT<srcCS>& srcHess,
    const PosDerivT<destCS, srcCS>& deriv, const PosDeriv2T<destCS, srcCS>& deriv2);

/** All-mighty routine for evaluating the value of a scalar function and its derivatives
    in a different coordinate system (evalCS), and converting them to the target
    coordinate system (outputCS).
*/
template<typename evalCS, typename outputCS>
void evalAndConvert(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos,
    double* value=NULL,
    GradT<outputCS>* deriv=NULL,
    HessT<outputCS>* deriv2=NULL,
    double time=0,
    const evalCS coordsys=evalCS())
{
    bool needDeriv = deriv!=NULL || deriv2!=NULL;
    bool needDeriv2= deriv2!=NULL;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    PosDerivT <outputCS, evalCS> coordDeriv;
    PosDeriv2T<outputCS, evalCS> coordDeriv2;
    const PosT<evalCS> evalPos = needDeriv ?
        toPosDeriv<outputCS, evalCS>(pos, &coordDeriv, needDeriv2 ? &coordDeriv2 : NULL, coordsys) :
        toPos<outputCS, evalCS>(pos, coordsys);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : NULL, needDeriv2 ? &evalHess : NULL, time);
    if(deriv)  // ... and convert gradient/hessian back to output coords if necessary.
        *deriv  = toGrad<evalCS, outputCS> (evalGrad, coordDeriv);
    if(deriv2)
        *deriv2 = toHess<evalCS, outputCS> (evalGrad, evalHess, coordDeriv, coordDeriv2);
}

/// trivial instantiation of the above function for the case that conversion is not necessary
template<typename CS>
void evalAndConvert(const IScalarFunction<CS>& F, const PosT<CS>& pos,
    double* value, GradT<CS>* deriv, HessT<CS>* deriv2, const CS)
{  F.evalScalar(pos, value, deriv, deriv2); }

/** An even mightier routine for evaluating the value of a scalar function,
    its gradient and hessian, in a different coordinate system (evalCS),
    and converting them to the target coordinate system (outputCS)
    through an intermediate coordinate system (intermedCS),
    for the situation when a direct transformation is not available.
    TODO: remove as unnecessary when OblatePerfectEllipsoid is removed
*/
template<typename evalCS, typename intermedCS, typename outputCS>
void evalAndConvertTwoStep(const IScalarFunction<evalCS>& F,
    const PosT<outputCS>& pos,
    const evalCS& coordsys,
    double* value=NULL,
    GradT<outputCS>* deriv=NULL,
    HessT<outputCS>* deriv2=NULL,
    double time=0)
{
    bool needDeriv = deriv!=NULL || deriv2!=NULL;
    bool needDeriv2= deriv2!=NULL;
    GradT<evalCS> evalGrad;
    HessT<evalCS> evalHess;
    GradT<intermedCS> intermedGrad;
    HessT<intermedCS> intermedHess;
    PosDerivT <outputCS, intermedCS> coordDerivOI;
    PosDeriv2T<outputCS, intermedCS> coordDeriv2OI;
    PosDerivT <intermedCS, evalCS> coordDerivIE;
    PosDeriv2T<intermedCS, evalCS> coordDeriv2IE;
    const PosT<intermedCS> intermedPos = needDeriv ?
        toPosDeriv<outputCS, intermedCS>(pos, &coordDerivOI, needDeriv2 ? &coordDeriv2OI : NULL) :
        toPos<outputCS, intermedCS>(pos);
    const PosT<evalCS> evalPos = needDeriv ?
        toPosDeriv<intermedCS, evalCS>(intermedPos,
            &coordDerivIE, needDeriv2 ? &coordDeriv2IE : NULL, coordsys) :
        toPos<intermedCS, evalCS>(intermedPos, coordsys);
    // compute the function in transformed coordinates
    F.evalScalar(evalPos, value, needDeriv ? &evalGrad : NULL, needDeriv2 ? &evalHess : NULL, time);
    if(needDeriv)  // may be needed for either grad or hess (or both)
        intermedGrad = toGrad<evalCS, intermedCS> (evalGrad, coordDerivIE);
    if(deriv)
        *deriv  = toGrad<intermedCS, outputCS> (intermedGrad, coordDerivOI);
    if(deriv2) {
        intermedHess = toHess<evalCS, intermedCS> (evalGrad, evalHess, coordDerivIE, coordDeriv2IE);
        *deriv2 = toHess<intermedCS, outputCS> (intermedGrad, intermedHess, coordDerivOI, coordDeriv2OI);
    }
}

/** Specialized conversion routine for spherically-symmetric functions.
    Convert the derivatives of a simple function that only depends on the spherical radius
    into gradients and hessians in a target coordinate system (outputCS).
*/
template<typename outputCS>
void evalAndConvertSph(const math::IFunction& F,
    const PosT<outputCS>& pos,
    double* value=NULL,
    GradT<outputCS>* deriv=NULL,
    HessT<outputCS>* deriv2=NULL);

///@}
/// \name  Miscellaneous routines
///@{

/// compute the total angular momentum for a point in the given coordinate system CoordT
template<typename CoordT> double Ltotal(const PosVelT<CoordT> &p);

/// compute the z-component of angular momentum for a point in the given coordinate system CoordT
template<typename CoordT> double Lz(const PosVelT<CoordT> &p);

///@}
/// \name  3d rotations
///@{

/** 3d rotational transformation between two Cartesian coordinate systems with the same origin,
    parametrized by three Euler angles.
    Let (x,y,z) be the source reference frame, and (X,Y,Z) be the rotated target frame.
    The first rotation by angle alpha about the z axis creates an intermediate reference frame
    (x',y',z'), where the axis x' points along the line of nodes of the overall transformation.
    The second rotation by angle beta about the x' axis tilts the (x',y') plane by angle beta,
    creating a second intermediate reference frame (x'', y'', z'').
    The third rotation by angle gamma about the z'' axis does not change the orientation of z'',
    hence the final axis Z is the same as z'', and beta is the angle between Z and z.
    The composition of three rotations is described by an orthogonal rotation matrix R,
    such that a point with coordinates (x,y,z) in the original reference frame will have
    coordinates (X,Y,Z) in the rotated frame, specified by
    \code
    | X |   | mat[0]  mat[1]  mat[2] |   | x |
    | Y | = | mat[3]  mat[4]  mat[5] | * | y |
    | Z |   | mat[6]  mat[7]  mat[8] |   | z |
    \endcode
    where mat is the flattened matrix R in a row-major order.
    In other words, the point remains geometrically fixed, only the reference frame changes
    and the coordinates of this point change accordingly (passive rotation).
    The inverse rotation is produced by a triplet of angles (-gamma,-beta,-alpha),
    and its rotation matrix is simply the transpose of the forward rotation matrix.
*/
class Orientation {
public:
    double mat[9];  ///< the rotation matrix R

    /** Construct the rotation matrix: the Euler angles alpha, beta, gamma describe the sequence
        of elementary rotations needed to transform the coordinates from the 'original' system
        to the 'rotated' one (in particular, beta is the inclination angle).
        When all three angles are zero (default initialization), the two systems coincide. */
    explicit Orientation(double alpha=0, double beta=0, double gamma=0);

    /** Check if the orientation is face-on (i.e. inclination is zero or pi, or |cos(beta)|==1) */
    bool isFaceOn() const { return mat[8]==1 || mat[8]==-1; }

    /** Convert the rotation matrix into Euler angles,
        normalized to the range alpha, gamma: [-pi..pi], beta: [0..pi].
        When beta=0 or beta=pi, only one of the remaining angles can be determined uniquely,
        so alpha is set to zero and gamma conveys the information about orientation.
        In any case, a rotation matrix reconstructed from these angles should coincide
        with the original matrix up to floating-point errors. */
    void toEulerAngles(double& alpha, double& beta, double& gamma) const;

    /** Transform the coordinates of a point from the 'original' to the 'rotated' system
        (the point remains geometrically fixed, only the reference frame changes) */
    void toRotated(const double vec[3], double result[3]) const
    {
        result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
        result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
        result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
    }

    /** Transform the coordinates of a point from the 'rotated' to the 'original' reference frame */
    void fromRotated(const double vec[3], double result[3]) const
    {
        // the inverse transformation uses the transposed rotation matrix
        result[0] = mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2];
        result[1] = mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2];
        result[2] = mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2];
    }

    /** Transform the position in cartesian coordinates from the 'original' to the 'rotated' frame
        (convenience overload for PosCar) */
    PosCar toRotated(const PosCar& pos) const
    {
        double vec[3] = {pos.x, pos.y, pos.z}, result[3];
        toRotated(vec, result);
        return coord::PosCar(result[0], result[1], result[2]);
    }

    /** Transform the position in cartesian coordinates from the 'rotated' to the 'original' frame
        (convenience overload for PosCar) */
    PosCar fromRotated(const PosCar& pos) const
    {
        double vec[3] = {pos.x, pos.y, pos.z}, result[3];
        fromRotated(vec, result);
        return coord::PosCar(result[0], result[1], result[2]);
    }

    /** Transform the velocity in cartesian coordinates from the 'original' to the 'rotated' frame
        (convenience overload for VelCar) */
    VelCar toRotated(const VelCar& vel) const
    {
        double vec[3] = {vel.vx, vel.vy, vel.vz}, result[3];
        toRotated(vec, result);
        return coord::VelCar(result[0], result[1], result[2]);
    }

    /** Transform the velocity in cartesian coordinates from the 'rotated' to the 'original' frame
        (convenience overload for VelCar) */
    VelCar fromRotated(const VelCar& vel) const
    {
        double vec[3] = {vel.vx, vel.vy, vel.vz}, result[3];
        fromRotated(vec, result);
        return coord::VelCar(result[0], result[1], result[2]);
    }

    /** Transform the position and velocity from the 'original' to the 'rotated' frame
        (convenience overload for PosVelCar) */
    PosVelCar toRotated(const PosVelCar& posvel) const
    {
        double vec[6], result[6];
        posvel.unpack_to(vec);
        toRotated(vec,   result);
        toRotated(vec+3, result+3);
        return PosVelCar(result);
    }

    /** Transform the position and velocity from the 'rotated' to the 'original' frame
        (convenience overload for PosVelCar) */
    PosVelCar fromRotated(const PosVelCar& posvel) const
    {
        double vec[6], result[6];
        posvel.unpack_to(vec);
        fromRotated(vec,   result);
        fromRotated(vec+3, result+3);
        return PosVelCar(result);
    }

    /** Transform the second moment of velocity in cartesian coordinates
        from the 'original' to the 'rotated' frame */
    Vel2Car toRotated(const Vel2Car& vel2) const
    {
        double
        vel2x[3] = {vel2.vx2,  vel2.vxvy, vel2.vxvz}, res2x[3],
        vel2y[3] = {vel2.vxvy, vel2.vy2,  vel2.vyvz}, res2y[3],
        vel2z[3] = {vel2.vxvz, vel2.vyvz, vel2.vz2 }, res2z[3];
        toRotated(vel2x, res2x);
        toRotated(vel2y, res2y);
        toRotated(vel2z, res2z);
        Vel2Car result;
        result.vx2  = mat[0] * res2x[0] + mat[1] * res2y[0] + mat[2] * res2z[0];
        result.vy2  = mat[3] * res2x[1] + mat[4] * res2y[1] + mat[5] * res2z[1];
        result.vz2  = mat[6] * res2x[2] + mat[7] * res2y[2] + mat[8] * res2z[2];
        result.vxvy = mat[3] * res2x[0] + mat[4] * res2y[0] + mat[5] * res2z[0];
        result.vyvz = mat[6] * res2x[1] + mat[7] * res2y[1] + mat[8] * res2z[1];
        result.vxvz = mat[0] * res2x[2] + mat[1] * res2y[2] + mat[2] * res2z[2];
        return result;
    }

    /** Transform the second moment of velocity in cartesian coordinates
        from the 'rotated' to the 'original' frame */
    Vel2Car fromRotated(const Vel2Car& vel2) const
    {
        double
        vel2x[3] = {vel2.vx2,  vel2.vxvy, vel2.vxvz}, res2x[3],
        vel2y[3] = {vel2.vxvy, vel2.vy2,  vel2.vyvz}, res2y[3],
        vel2z[3] = {vel2.vxvz, vel2.vyvz, vel2.vz2 }, res2z[3];
        fromRotated(vel2x, res2x);
        fromRotated(vel2y, res2y);
        fromRotated(vel2z, res2z);
        Vel2Car result;
        result.vx2  = mat[0] * res2x[0] + mat[3] * res2y[0] + mat[6] * res2z[0];
        result.vy2  = mat[1] * res2x[1] + mat[4] * res2y[1] + mat[7] * res2z[1];
        result.vz2  = mat[2] * res2x[2] + mat[5] * res2y[2] + mat[8] * res2z[2];
        result.vxvy = mat[1] * res2x[0] + mat[4] * res2y[0] + mat[7] * res2z[0];
        result.vyvz = mat[2] * res2x[1] + mat[5] * res2y[1] + mat[8] * res2z[1];
        result.vxvz = mat[0] * res2x[2] + mat[3] * res2y[2] + mat[6] * res2z[2];
        return result;
    }

    /** Transform the gradient in cartesian coordinates from the 'original' to the 'rotated' frame */
    GradCar toRotated(const GradCar& grad) const
    {
        double vec[3] = {grad.dx, grad.dy, grad.dz}, res[3];
        toRotated(vec, res);
        coord::GradCar result;
        result.dx = res[0];
        result.dy = res[1];
        result.dz = res[2];
        return result;
    }

    /** Transform the gradient in cartesian coordinates from the 'rotated' to the 'original' frame */
    GradCar fromRotated(const GradCar& grad) const
    {
        double vec[3] = {grad.dx, grad.dy, grad.dz}, res[3];
        fromRotated(vec, res);
        coord::GradCar result;
        result.dx = res[0];
        result.dy = res[1];
        result.dz = res[2];
        return result;
    }

    /** Transform the hessian in cartesian coordinates from the 'original' to the 'rotated' frame */
    HessCar toRotated(const HessCar& hess) const
    {
        double
        hessx[3] = {hess.dx2,  hess.dxdy, hess.dxdz}, resx[3],
        hessy[3] = {hess.dxdy, hess.dy2,  hess.dydz}, resy[3],
        hessz[3] = {hess.dxdz, hess.dydz, hess.dz2 }, resz[3];
        toRotated(hessx, resx);
        toRotated(hessy, resy);
        toRotated(hessz, resz);
        HessCar result;
        result.dx2  = mat[0] * resx[0] + mat[1] * resy[0] + mat[2] * resz[0];
        result.dy2  = mat[3] * resx[1] + mat[4] * resy[1] + mat[5] * resz[1];
        result.dz2  = mat[6] * resx[2] + mat[7] * resy[2] + mat[8] * resz[2];
        result.dxdy = mat[3] * resx[0] + mat[4] * resy[0] + mat[5] * resz[0];
        result.dydz = mat[6] * resx[1] + mat[7] * resy[1] + mat[8] * resz[1];
        result.dxdz = mat[0] * resx[2] + mat[1] * resy[2] + mat[2] * resz[2];
        return result;
    }

    /** Transform the hessian in cartesian coordinates from the 'rotated' to the 'original' frame */
    HessCar fromRotated(const HessCar& hess) const
    {
        double
        hessx[3] = {hess.dx2,  hess.dxdy, hess.dxdz}, resx[3],
        hessy[3] = {hess.dxdy, hess.dy2,  hess.dydz}, resy[3],
        hessz[3] = {hess.dxdz, hess.dydz, hess.dz2 }, resz[3];
        fromRotated(hessx, resx);
        fromRotated(hessy, resy);
        fromRotated(hessz, resz);
        HessCar result;
        result.dx2  = mat[0] * resx[0] + mat[3] * resy[0] + mat[6] * resz[0];
        result.dy2  = mat[1] * resx[1] + mat[4] * resy[1] + mat[7] * resz[1];
        result.dz2  = mat[2] * resx[2] + mat[5] * resy[2] + mat[8] * resz[2];
        result.dxdy = mat[1] * resx[0] + mat[4] * resy[0] + mat[7] * resz[0];
        result.dydz = mat[2] * resx[1] + mat[5] * resy[1] + mat[8] * resz[1];
        result.dxdz = mat[0] * resx[2] + mat[3] * resy[2] + mat[6] * resz[2];
        return result;
    }

};

///@}

}  // namespace coord
