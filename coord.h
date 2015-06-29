#pragma once

/// convenience function for squaring a number, used in many places
inline double pow_2(double x) { return x*x; }

namespace coord{

    /// position in cartesian coordinates
    struct PosCar{
        double x, y, z;
        PosCar(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {};
    };

    /// position in cylindrical coordinates
    struct PosCyl{
        double R, z, phi;
        PosCyl(double _R, double _z, double _phi) : R(_R), z(_z), phi(_phi) {};
    };

    /// position in spherical coordinates
    struct PosSph{
        double r, theta, phi;
        PosSph(double _r, double _theta, double _phi) : r(_r), theta(_theta), phi(_phi) {};
    };

    /// combined position and velocity in arbitrary coordinates
    template<typename T> struct PosVelT;

    /// combined position and velocity in cartesian coordinates
    template<> struct PosVelT<PosCar>: public PosCar{
        double vx, vy, vz;
        PosVelT<PosCar>(double _x, double _y, double _z, double _vx, double _vy, double _vz) :
            PosCar(_x, _y, _z), vx(_vx), vy(_vy), vz(_vz) {};
        void unpack_to(double *out) const {
            out[0]=x; out[1]=y; out[2]=z; out[3]=vx; out[4]=vy; out[5]=vz; }
    };
    /// an alias to templated type specialization of position and velocity for cartesian coordinates
    typedef struct PosVelT<PosCar> PosVelCar;

    /// combined position and velocity in cylindrical coordinates
    template<> struct PosVelT<PosCyl>: public PosCyl{
        ///< velocity in cylindrical coordinates
        /// (this is not the same as time derivative of position in these coordinates!)
        double vR, vz, vphi;
        PosVelT<PosCyl>(double _R, double _z, double _phi, double _vR, double _vz, double _vphi) :
            PosCyl(_R, _z, _phi), vR(_vR), vz(_vz), vphi(_vphi) {};
        void unpack_to(double *out) const {
            out[0]=R; out[1]=z; out[2]=phi; out[3]=vR; out[4]=vz; out[5]=vphi; }
    };
    typedef struct PosVelT<PosCyl> PosVelCyl;

    /// combined position and velocity in spherical coordinates
    template<> struct PosVelT<PosSph>: public PosSph{
        /// velocity in spherical coordinates
        /// (this is not the same as time derivative of position in these coordinates!)
        double vr, vtheta, vphi;
        PosVelT<PosSph>(double _r, double _theta, double _phi, double _vr, double _vtheta, double _vphi) :
            PosSph(_r, _theta, _phi), vr(_vr), vtheta(_vtheta), vphi(_vphi) {};
        void unpack_to(double *out) const {
            out[0]=r; out[1]=theta; out[2]=phi; out[3]=vr; out[4]=vtheta; out[5]=vphi; }
    };
    typedef struct PosVelT<PosSph> PosVelSph;


    /// convenience function to extract the value of angular momentum
    template<typename coordT> double Lz(const PosVelT<coordT>& p);


    /// link between coordinate type and its corresponding gradient type
    template<typename T> struct GradT;

    /// gradient of scalar function in cartesian coordinates
    template<> struct GradT<PosCar>{
        double dx, dy, dz;
    };
    /// an alias to templated type specialization of gradient for cartesian coordinates
    typedef struct GradT<PosCar> GradCar;

    /// gradient of scalar function in cylindrical coordinates
    template<> struct GradT<PosCyl>{
        double dR, dz, dphi;
    };
    typedef struct GradT<PosCyl> GradCyl;

    /// gradient of scalar function in spherical coordinates
    template<> struct GradT<PosSph>{
        double dr, dtheta, dphi;
    };
    typedef struct GradT<PosSph> GradSph;


    /// link between coordinate type and its corresponding hessian type
    template<typename T> struct HessT;

    /// Hessian of scalar function in cartesian coordinates
    template<> struct HessT<PosCar>{
        double dx2, dy2, dz2, dxdy, dydz, dxdz;
    };
    typedef struct HessT<PosCar> HessCar;

    /// Hessian of scalar function in cylindrical coordinates
    template<> struct HessT<PosCyl>{
        double dR2, dz2, dphi2, dRdz, dzdphi, dRdphi;
    };
    typedef struct HessT<PosCyl> HessCyl;

    /// Hessian of scalar function in spherical coordinates
    template<> struct HessT<PosSph>{
        double dr2, dtheta2, dphi2, drdtheta, dthetadphi, drdphi;
    };
    typedef struct HessT<PosSph> HessSph;

    /** universal templated conversion function for coordinates:
        template parameters srcT and destT may be PosCar, PosCyl or PosSph, in any combination. */
    template<typename srcT, typename destT>
    destT toPos(const srcT& from);

    /** templated conversion functions for coordinates 
        with names reflecting the target coordinate system. */
    template<typename srcT>
    PosCar toPosCar(const srcT& from) { return toPos<srcT, PosCar>(from); }
    template<typename srcT>
    PosCyl toPosCyl(const srcT& from) { return toPos<srcT, PosCyl>(from); }
    template<typename srcT>
    PosSph toPosSph(const srcT& from) { return toPos<srcT, PosSph>(from); }

    /** universal templated conversion function for coordinates and velocities:
        template parameters srcT and destT may be PosCar, PosCyl or PosSph. */
    template<typename srcT, typename destT>
    PosVelT<destT> toPosVel(const PosVelT<srcT>& from);

    /** templated conversion functions for coordinates and velocities
        with names reflecting the target coordinate system. */
    template<typename srcT>
    PosVelCar toPosVelCar(const PosVelT<srcT>& from) { return toPosVel<srcT, PosCar>(from); }
    template<typename srcT>
    PosVelCyl toPosVelCyl(const PosVelT<srcT>& from) { return toPosVel<srcT, PosCyl>(from); }
    template<typename srcT>
    PosVelSph toPosVelSph(const PosVelT<srcT>& from) { return toPosVel<srcT, PosSph>(from); }

    /** trivial conversions */
    template<> inline PosCar toPosCar(const PosCar& p) { return p;}
    template<> inline PosCyl toPosCyl(const PosCyl& p) { return p;}
    template<> inline PosSph toPosSph(const PosSph& p) { return p;}
    template<> inline PosVelCar toPosVelCar(const PosVelCar& p) { return p;}
    template<> inline PosVelCyl toPosVelCyl(const PosVelCyl& p) { return p;}
    template<> inline PosVelSph toPosVelSph(const PosVelSph& p) { return p;}


    /** universal templated conversion functions for gradients and hessians:
        template parameters srcT and destT may be PosCar, PosCyl or PosSph;
        \param[in] srcPos    is the position in the source coordinate system;
        \param[in] srcGrad   is the gradient in the source coordinate system (must not be NULL);
        \param[in] srcHess   is the Hessian in the source coordinate system (may be NULL);
        \param[out] destGrad if not NULL, output will contain the gradient in the destination coordinates;
        \param[out] destHess if not NULL, output will contain the Hessian in the destination coordinates, 
                    if destHess!=NULL, srcHess must be provided too. */
    template<typename srcT, typename destT>
    void toDeriv(const srcT& srcPos,
        const GradT<srcT>* srcGrad, const HessT<srcT>* srcHess,
        GradT<destT>* destGrad, HessT<destT>* destHess);


    /** Prolate spheroidal coordinate system */
    struct CoordSysProlateSpheroidal{
        double alpha;
        double gamma;
        CoordSysProlateSpheroidal(double _alpha, double _gamma): alpha(_alpha), gamma(_gamma) {};
    };

    /** position in prolate spheroidal coordinates */
    struct PosProlSph{
        double lambda, nu, phi;
        PosProlSph(double _lambda, double _nu, double _phi): lambda(_lambda), nu(_nu), phi(_phi) {};
    };

    /** derivatives of coordinate transformation from cylindrical to prolate spheroidal coords */
    struct PosDerivProlSph{
        double dlambdadR, dlambdadz, dnudR, dnudz;
    };

    /** second derivatives of coordinate transformation from cylindrical to prolate spheroidal coords */
    struct PosDeriv2ProlSph{
        double d2lambdadR2, d2lambdadz2, d2lambdadRdz, d2nudR2, d2nudz2, d2nudRdz;
    };

    /** convert from prolate spheroidal to cylindrical coordinates */
    PosCyl toPosCyl(const PosProlSph& from, const CoordSysProlateSpheroidal& coordsys);

    /** convert from cylindrical to prolate spheroidal coordinates;
        if derivs!=NULL, it will contain derivatives of coordinate transformation;
        if derivs2!=NULL, it will contain its second derivatives. */
    PosProlSph toPosProlSph(const PosCyl& from, const CoordSysProlateSpheroidal& coordsys,
        PosDerivProlSph* derivs=0, PosDeriv2ProlSph* derivs2=0);

}  // namespace coord
