#pragma once

/// convenience function for squaring a number, used in many places
inline double pow_2(double x) { return x*x; }

namespace coord{

/// \name   Primitive data types: coordinate systems
///@{

    /// trivial coordinate systems don't have any parameters, 
    /// their class names are simply used as tags in the rest of the code

    /// cartesian coordinate system
    struct Car{};

    /// cylindrical coordinate system
    struct Cyl{};

    /// spherical coordinate system
    struct Sph{};

    //  less trivial:
    /// prolate spheroidal coordinate system
    struct ProlSph{
        double alpha;  ///< alpha=-a^2, where a is major axis
        double gamma;  ///< gamma=-c^2, where c is minor axis
        ProlSph(double _alpha, double _gamma): alpha(_alpha), gamma(_gamma) {};
    };

    /// templated routine that tells the name of coordinate system
    template<typename coordSysT> const char* CoordSysName();
    template<> static const char* CoordSysName<Car>() { return "Cartesian"; }
    template<> static const char* CoordSysName<Cyl>() { return "Cylindrical"; }
    template<> static const char* CoordSysName<Sph>() { return "Spherical"; }
    template<> static const char* CoordSysName<ProlSph>() { return "Prolate spheroidal"; }

///@}
/// \name   Primitive data types: position in different coordinate systems
///@{

    /// position in arbitrary coordinates:
    /// the data types are defined as templates with the template parameter
    /// being any of the coordinate system names defined above
    template<typename coordSysT> struct PosT;

    /// position in cartesian coordinates
    template<> struct PosT<Car>{
        double x, y, z;
        PosT<Car>(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {};
    };
    /// an alias to templated type specialization of position in cartesian coordinates
    typedef struct PosT<Car> PosCar;

    /// position in cylindrical coordinates
    template<> struct PosT<Cyl>{
        double R, z, phi;
        PosT<Cyl>(double _R, double _z, double _phi) : R(_R), z(_z), phi(_phi) {};
    };
    typedef struct PosT<Cyl> PosCyl;

    /// position in spherical coordinates
    template<> struct PosT<Sph>{
        double r, theta, phi;
        PosT<Sph>(double _r, double _theta, double _phi) : r(_r), theta(_theta), phi(_phi) {};
    };
    typedef struct PosT<Sph> PosSph;

    /// position in prolate spheroidal coordinates
    template<> struct PosT<ProlSph>{
        double lambda, nu, phi;
        const ProlSph& CS;    ///< a point means nothing without specifying its coordinate system
        PosT<ProlSph>(double _lambda, double _nu, double _phi, const ProlSph& _CS):
            lambda(_lambda), nu(_nu), phi(_phi), CS(_CS) {};
    };
    typedef struct PosT<ProlSph> PosProlSph;

///@}
/// \name   Primitive data types: position-velocity pairs in different coordinate systems
///@{

    /// combined position and velocity in arbitrary coordinates
    template<typename coordSysT> struct PosVelT;

    /// combined position and velocity in cartesian coordinates
    template<> struct PosVelT<Car>: public PosCar{
        double vx, vy, vz;
        PosVelT<Car>(double _x, double _y, double _z, double _vx, double _vy, double _vz) :
            PosCar(_x, _y, _z), vx(_vx), vy(_vy), vz(_vz) {};
        void unpack_to(double *out) const {
            out[0]=x; out[1]=y; out[2]=z; out[3]=vx; out[4]=vy; out[5]=vz; }
    };
    /// an alias to templated type specialization of position and velocity for cartesian coordinates
    typedef struct PosVelT<Car> PosVelCar;

    /// combined position and velocity in cylindrical coordinates
    template<> struct PosVelT<Cyl>: public PosCyl{
        ///< velocity in cylindrical coordinates
        /// (this is not the same as time derivative of position in these coordinates!)
        double vR, vz, vphi;
        PosVelT<Cyl>(double _R, double _z, double _phi, double _vR, double _vz, double _vphi) :
            PosCyl(_R, _z, _phi), vR(_vR), vz(_vz), vphi(_vphi) {};
        void unpack_to(double *out) const {
            out[0]=R; out[1]=z; out[2]=phi; out[3]=vR; out[4]=vz; out[5]=vphi; }
    };
    typedef struct PosVelT<Cyl> PosVelCyl;

    /// combined position and velocity in spherical coordinates
    template<> struct PosVelT<Sph>: public PosSph{
        /// velocity in spherical coordinates
        /// (this is not the same as time derivative of position in these coordinates!)
        double vr, vtheta, vphi;
        PosVelT<Sph>(double _r, double _theta, double _phi, double _vr, double _vtheta, double _vphi) :
            PosSph(_r, _theta, _phi), vr(_vr), vtheta(_vtheta), vphi(_vphi) {};
        void unpack_to(double *out) const {
            out[0]=r; out[1]=theta; out[2]=phi; out[3]=vr; out[4]=vtheta; out[5]=vphi; }
    };
    typedef struct PosVelT<Sph> PosVelSph;

///@}
/// \name   Primitive data types: gradient of a scalar function in different coordinate systems
///@{

    /// components of a gradient in a given coordinate system
    template<typename coordSysT> struct GradT;

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

///@}
/// \name   Primitive data types: hessian of a scalar function in different coordinate systems
///@{

    /// components of a hessian of a scalar function (matrix of its second derivatives)
    template<typename coordSysT> struct HessT;

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

///@}
/// \name   Abstract interface class: a scalar function evaluated in a particular coordinate systems
///@{

    template<typename coordSysT>
    class ScalarFunction {
    public:
        ScalarFunction() {};
        virtual ~ScalarFunction() {};
        virtual void evaluate(const PosT<coordSysT>& x,
            double* value=0,
            GradT<coordSysT>* deriv=0,
            HessT<coordSysT>* deriv2=0) const=0;
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
    typedef struct PosDerivT<Cyl, ProlSph> PosDerivCylProlSph;  ///< human-readable alias


    /** second derivatives of coordinate transformation from source to destination 
        coordinate systems (srcCS=>destCS): d^2(dest_coord)/d(source_coord1)d(source_coord2) */
    template<typename srcCS, typename destCS> struct PosDeriv2T;

    /** instantiations of the general template for second derivatives of coordinate transformations */
    template<> struct PosDeriv2T<Cyl, Car> {
        double d2xdR2, d2xdRdphi, d2xdphi2, d2ydR2, d2ydRdphi, d2ydphi2;
    };
    template<> struct PosDeriv2T<Sph, Car> {
        double d2xdr2, d2xdrdtheta, d2xdrdphi, d2xdtheta2, d2xdthetadphi, d2xdphi2,
            d2ydr2, d2ydrdtheta, d2ydrdphi, d2ydtheta2, d2ydthetadphi, d2ydphi2,
            d2zdr2, d2zdrdtheta, d2zdtheta2;
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
        double d2lambdadR2, d2lambdadz2, d2lambdadRdz, d2nudR2, d2nudz2, d2nudRdz;
    };
    typedef struct PosDeriv2T<Cyl, ProlSph> PosDeriv2CylProlSph; ///< alias

///@}
/// \name   Routines for conversion between position/velocity in different coordinate systems
///@{

    /** universal templated conversion function for positions:
        template parameters srcCS and destCS may be any of the coordinate system names.
        This template function shouldn't be used directly, because the return type depends 
        on the template and hence cannot be automatically inferred by the compiler. 
        Instead, named functions for each target coordinate system are defined below. */
    template<typename srcCS, typename destCS>
    PosT<destCS> toPos(const PosT<srcCS>& from);

    /** templated conversion taking the parameters of coordinate system into account */
    //template<typename srcCS, typename destCS>
    //PosT<destCS> toPos(const PosT<srcCS>& from, const srcCS& coordsys);

    /** templated conversion functions for positions 
        with names reflecting the target coordinate system. */
    template<typename srcCS>
    inline PosCar toPosCar(const PosT<srcCS>& from) { return toPos<srcCS, Car>(from); }
    template<typename srcCS>
    inline PosCyl toPosCyl(const PosT<srcCS>& from) { return toPos<srcCS, Cyl>(from); }
    template<typename srcCS>
    inline PosSph toPosSph(const PosT<srcCS>& from) { return toPos<srcCS, Sph>(from); }

    /** convert position from prolate spheroidal to cylindrical coordinates */
    //PosCyl toPosCyl(const PosProlSph& from, const ProlSph& coordsys);

    /** universal templated conversion function for coordinates and velocities:
        template parameters srcCS and destCS may be any of the coordinate system names */
    template<typename srcCS, typename destCS>
    PosVelT<destCS> toPosVel(const PosVelT<srcCS>& from);

    /** templated conversion functions for coordinates and velocities
        with names reflecting the target coordinate system. */
    template<typename srcCS>
    inline PosVelCar toPosVelCar(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Car>(from); }
    template<typename srcCS>
    inline PosVelCyl toPosVelCyl(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Cyl>(from); }
    template<typename srcCS>
    inline PosVelSph toPosVelSph(const PosVelT<srcCS>& from) { return toPosVel<srcCS, Sph>(from); }

    /** trivial conversions */
    template<> inline PosCar toPosCar(const PosCar& p) { return p;}
    template<> inline PosCyl toPosCyl(const PosCyl& p) { return p;}
    template<> inline PosSph toPosSph(const PosSph& p) { return p;}
    template<> inline PosVelCar toPosVelCar(const PosVelCar& p) { return p;}
    template<> inline PosVelCyl toPosVelCyl(const PosVelCyl& p) { return p;}
    template<> inline PosVelSph toPosVelSph(const PosVelSph& p) { return p;}

///@}
/// \name   Routines for conversion between position in different coordinate systems with derivatives
///@{

    /** universal templated function for coordinate conversion that provides derivatives of transformation.
        Template parameters srcCS and destCS may be any of the coordinate system names;
        \param[in]  from specifies the point in srcCS coordinate system;
        \param[out] deriv will contain derivatives of the transformation 
                    (destination coords over source coords);
        \param[out] deriv2 if not NULL, will contain second derivatives of the coordinate transformation;
        \return     point in destCS coordinate system. */
    template<typename srcCS, typename destCS>
    PosT<destCS> toPosDeriv(const PosT<srcCS>& from, 
        PosDerivT<srcCS, destCS>* deriv, PosDeriv2T<srcCS, destCS>* deriv2=0);

    /** a special case of transformation that needs the parameters of coordinate system:
        convert from cylindrical to prolate spheroidal coordinates;
        if derivs!=NULL, it will contain derivatives of coordinate transformation;
        if derivs2!=NULL, it will contain its second derivatives. */
    PosProlSph toPosDerivProlSph(const PosCyl& from, const ProlSph& coordsys,
        PosDerivCylProlSph* deriv, PosDeriv2CylProlSph* deriv2=0);

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
        in a different coordinate system (evalCS), and converting them to the target coordinate system (outputCS). */
    template<typename evalCS, typename outputCS>
    inline void eval_and_convert(const ScalarFunction<evalCS>& F,
        const PosT<outputCS>& pos, double* value, GradT<outputCS>* deriv, HessT<outputCS>* deriv2=0)
    {
        GradT<evalCS> evalGrad;
        HessT<evalCS> evalHess;
        PosDerivT <outputCS, evalCS> coordDeriv;
        PosDeriv2T<outputCS, evalCS> coordDeriv2;
        const PosT<evalCS> evalPos = toPosDeriv<outputCS, evalCS>(pos, 
            &coordDeriv, deriv2!=0 ? &coordDeriv2 : 0);
        // compute the function in transformed coordinates
        F.evaluate(evalPos, value, &evalGrad, deriv2!=0 ? &evalHess : 0);
        if(deriv)  // ... and convert gradient/hessian back to output coords if necessary.
            *deriv  = toGrad<evalCS, outputCS> (evalGrad, coordDeriv);
        if(deriv2)
            *deriv2 = toHess<evalCS, outputCS> (evalGrad, evalHess, coordDeriv, coordDeriv2);
    };

    /// trivial instantiation of the above function for the case that conversion is not necessary
    template<typename CS> 
    inline void eval_and_convert(const ScalarFunction<CS>& F, const PosT<CS>& pos, 
        double* value, GradT<CS>* deriv, HessT<CS>* deriv2)
    {  F.evaluate(pos, value, deriv, deriv2); }

    /** A less-mighty function that only computes the value of scalar function in a different coordinate system */
    template<typename evalCS, typename outputCS>
    inline void eval_and_convert(const ScalarFunction<evalCS>& F, 
        const PosT<outputCS>& pos, double* value) {
        F.evaluate(toPos<outputCS,evalCS>(pos), value);
    };

///@}

    /// convenience functions to extract the value of angular momentum and its z-component
    template<typename coordT> double Ltotal(const PosVelT<coordT>& p);
    template<typename coordT> double Lz(const PosVelT<coordT>& p);

}  // namespace coord
