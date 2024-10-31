/** \file    potential_base.h
    \brief   Base density and potential classes
    \author  Eugene Vasiliev
    \date    2009-2023
*/
#pragma once
#include "coord.h"
#include <string>

/** Classes and auxiliary routines related to creation and manipulation of 
    density models and gravitational potential models.

    These two concepts are related in such a way that a density model does not need 
    to provide potential and forces, while a potential model does. 
    Thus the latter is derived from the former.
    General-purpose potential expansions (Multipole, CylSpline)
    can be constructed both from density or from potential classes.
*/
namespace potential{

/// \name  Base class for all density models
///@{

/** Abstract class defining a density profile without a corresponding potential. 
    It provides overloaded functions for computing density in three different coordinate systems 
    ([Car]tesian, [Cyl]indrical and [Sph]erical); the derived classes typically should 
    implement the actual computation in one of them (in the most suitable coordinate system), 
    and provide a 'redirection' to it in the other two functions, by converting the input 
    coordinates to the most suitable system.
    Note that this class and its derivative BasePotential should represent constant objects,
    i.e. once created, they cannot be modified, and all their methods are const.
    Typically the derived classes are passed as const references to the base class
    (density or potential):

        const DerivedPotential pot1;     // statically typed object
        const BaseDensity& dens = pot1;  // polymorphic reference, here downgraded to the base class
        double mass = dens.totalMass();  // call virtual method of the base class
        // call a non-member function that accepts a reference to the base class
        double halfMassRadius = getRadiusByMass(dens, mass*0.5);

    Now these usage rules break down if we do not simply pass these objects to
    a call-and-return function, but rather create another object (e.g. an action finder
    or a composite potential) that holds the reference to the object throughout its lifetime,
    which may exceed that of the original object. In this case we must use dynamically
    created objects wrapped into a shared_ptr (typedef'ed as PtrDensity and PtrPotential).
*/
class BaseDensity{
public:

    /** Explicitly declare a virtual destructor in a class with virtual functions */
    virtual ~BaseDensity() {};

    /** Evaluate density at the position in a specified coordinate system (Car, Cyl, or Sph)
        at the given moment of time (optional, default 0).
        The actual computation is implemented in separately-named protected virtual functions. */
    double density(const coord::PosCar &pos, double time=0) const {
        return densityCar(pos, time); }
    double density(const coord::PosCyl &pos, double time=0) const {
        return densityCyl(pos, time); }
    double density(const coord::PosSph &pos, double time=0) const {
        return densitySph(pos, time); }

    /// returns the symmetry type of this density or potential
    virtual coord::SymmetryType symmetry() const = 0;

    /// return the name of density or potential model
    virtual std::string name() const = 0;

    /** estimate the mass enclosed within a given spherical radius;
        default implementation integrates density over volume, but derived classes
        may provide a cheaper alternative (not necessarily a very precise one).
    */
    virtual double enclosedMass(const double radius) const;

    /** return the total mass of the density model (possibly infinite);
        default implementation estimates the asymptotic behaviour of density at large radii,
        but derived classes may instead return a specific value.
    */
    virtual double totalMass() const;

    /** Vectorized evaluation of the density for several input points at once.
        \param[in]  npoints - size of the input array;
        \param[in]  pos - array of positions in the given coordinate system, with length npoints;
        \param[out] values - output array of length npoints that will be filled with density values;
        \param[in]  time (optional, default 0) - time at which the density is computed.
    */
    virtual void evalmanyDensityCar(const size_t npoints, const coord::PosCar pos[],
        /*output*/ double values[], /*input*/ double time=0) const
    {
        // default implementation just loops over input points one by one
        for(size_t p=0; p<npoints; p++)
            values[p] = densityCar(pos[p], time);
    }
    virtual void evalmanyDensityCyl(const size_t npoints, const coord::PosCyl pos[],
        /*output*/ double values[], /*input*/ double time=0) const
    {
        for(size_t p=0; p<npoints; p++)
            values[p] = densityCyl(pos[p], time);
    }
    virtual void evalmanyDensitySph(const size_t npoints, const coord::PosSph pos[],
        /*output*/ double values[], /*input*/ double time=0) const
    {
        for(size_t p=0; p<npoints; p++)
            values[p] = densitySph(pos[p], time);
    }

protected:

    /** evaluate density at the position specified in cartesian coordinates */
    virtual double densityCar(const coord::PosCar &pos, double time) const = 0;

    /** evaluate density at the position specified in cylindrical coordinates */
    virtual double densityCyl(const coord::PosCyl &pos, double time) const = 0;

    /** Evaluate density at the position specified in spherical coordinates */
    virtual double densitySph(const coord::PosSph &pos, double time) const = 0;
};  // class BaseDensity

///@}
/// \name   Base class for all potentials
///@{

/** Abstract class defining the gravitational potential.

    It provides public non-virtual functions for computing potential and 
    up to two its derivatives in three standard coordinate systems:
    [Car]tesian, [Cyl]indrical, and [Sph]erical. 
    These three functions share the same name `eval`, i.e. are overloaded 
    on the type of input coordinates. 
    They internally call three protected virtual functions, named after 
    each coordinate system. These functions are implemented in derived classes.
    Typically the potential and its derivatives are most easily computed in 
    one particular coordinate system, and the implementation of other two functions 
    simply convert the input coordinates to the most suitable system, and likewise 
    transforms the output values back to the requested coordinates, using 
    the `coord::evalAndConvert` function.
    Density is computed from Laplacian in each coordinate system, but derived 
    classes may override this behaviour and provide the density explicitly.
*/
class BasePotential: public BaseDensity{
public:
    /** Evaluate potential and up to two its derivatives in a specified coordinate system.
        \param[in]  pos is the position in the given coordinates.
        \param[out] potential - if not NULL, store the value of potential
                    in the variable addressed by this pointer.
        \param[out] deriv - if not NULL, store the gradient of potential
                    in the variable addressed by this pointer.
        \param[out] deriv2 - if not NULL, store the Hessian (matrix of second derivatives)
                    of potential in the variable addressed by this pointer.
        \param[in]  time (optional, default 0) - specifies the time at which the potential and
                    its derivatives are computed (only relevant for a time-dependent potential).
    */
    void eval(const coord::PosCar &pos,
        double* potential=NULL, coord::GradCar* deriv=NULL, coord::HessCar* deriv2=NULL,
        double time=0) const {
        return evalCar(pos, potential, deriv, deriv2, time); }
    void eval(const coord::PosCyl &pos,
        double* potential=NULL, coord::GradCyl* deriv=NULL, coord::HessCyl* deriv2=NULL,
        double time=0) const {
        return evalCyl(pos, potential, deriv, deriv2, time); }
    void eval(const coord::PosSph &pos,
        double* potential=NULL, coord::GradSph* deriv=NULL, coord::HessSph* deriv2=NULL,
        double time=0) const {
        return evalSph(pos, potential, deriv, deriv2, time); }

    /** Shorthand for evaluating the value of potential at a given point in any coordinate system */
    template<typename CoordT>
    inline double value(const coord::PosT<CoordT>& point, double time=0) const {
        double val;
        eval(point, &val, NULL, NULL, time);
        return val;
    }

    /** estimate the mass enclosed within a given radius from the radial component of force */
    virtual double enclosedMass(const double radius) const;

protected:
    /** evaluate potential and up to two its derivatives in cartesian coordinates;
        must be implemented in derived classes */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const = 0;

    /** evaluate potential and up to two its derivatives in cylindrical coordinates */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const = 0;

    /** evaluate potential and up to two its derivatives in spherical coordinates */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const = 0;

    /** Default implementation computes the density from Laplacian of the potential,
        but the derived classes may instead provide an explicit expression for it. */
    virtual double densityCar(const coord::PosCar &pos, double time) const;
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};  // class BasePotential

///@}
/// \name   Base classes for potentials that implement the computations in a particular coordinate system
///@{

/** Parent class for potentials that are evaluated in cartesian coordinates.
    It leaves the implementation of `evalCar` member function for cartesian coordinates undefined, 
    but provides the conversion from cartesian to cylindrical and spherical coordinates
    in `evalCyl` and `evalSph`. */
class BasePotentialCar: public BasePotential, coord::IScalarFunction<coord::Car>{

    /** evaluate potential and up to two its derivatives in cylindrical coordinates. */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const {
        coord::evalAndConvert<coord::Car, coord::Cyl>(*this, pos, potential, deriv, deriv2, time);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const {
        coord::evalAndConvert<coord::Car, coord::Sph>(*this, pos, potential, deriv, deriv2, time);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cartesian) coordinate system. */
    virtual void evalScalar(const coord::PosCar& pos,
        double* val=NULL, coord::GradCar* deriv=NULL, coord::HessCar* deriv2=NULL, double time=0) const
    {  evalCar(pos, val, deriv, deriv2, time);  }

    /** redirect density computation to a Laplacian in more suitable coordinates */
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {  return densityCar(toPosCar(pos), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    {  return densityCar(toPosCar(pos), time); }
};  // class BasePotentialCar


/** Parent class for potentials that are evaluated in cylindrical coordinates.
    It leaves the implementation of `evalCyl` member function for cylindrical coordinates undefined, 
    but provides the conversion from cylindrical to cartesian and spherical coordinates. */
class BasePotentialCyl: public BasePotential, coord::IScalarFunction<coord::Cyl>{

    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const {
        coord::evalAndConvert<coord::Cyl, coord::Car>(*this, pos, potential, deriv, deriv2, time);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double time) const {
        coord::evalAndConvert<coord::Cyl, coord::Sph>(*this, pos, potential, deriv, deriv2, time);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cylindrical) coordinate system. */
    virtual void evalScalar(const coord::PosCyl& pos,
        double* val=NULL, coord::GradCyl* deriv=NULL, coord::HessCyl* deriv2=NULL, double time=0) const
    {  evalCyl(pos, val, deriv, deriv2, time); }

    /** redirect density computation to more suitable coordinates */
    virtual double densityCar(const coord::PosCar &pos, double time) const
    {  return densityCyl(toPosCyl(pos), time); }
    virtual double densitySph(const coord::PosSph &pos, double time) const
    {  return densityCyl(toPosCyl(pos), time); }
};  // class BasePotentialCyl


/** Parent class for potentials that are evaluated in spherical coordinates.
    It leaves the implementation of `evalSph member` function for spherical coordinates undefined, 
    but provides the conversion from spherical to cartesian and cylindrical coordinates. */
class BasePotentialSph: public BasePotential, coord::IScalarFunction<coord::Sph>{

    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const {
        coord::evalAndConvert<coord::Sph, coord::Car>(*this, pos, potential, deriv, deriv2, time);
    }

    /** evaluate potential and up to two its derivatives in cylindrical coordinates. */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const {
        coord::evalAndConvert<coord::Sph, coord::Cyl>(*this, pos, potential, deriv, deriv2, time);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Spherical) coordinate system. */
    virtual void evalScalar(const coord::PosSph& pos,
        double* val=NULL, coord::GradSph* deriv=NULL, coord::HessSph* deriv2=NULL, double time=0) const
    {  evalSph(pos, val, deriv, deriv2, time); }

    /** redirect density computation to more suitable coordinates */
    virtual double densityCar(const coord::PosCar &pos, double time) const
    {  return densitySph(toPosSph(pos), time); }
    virtual double densityCyl(const coord::PosCyl &pos, double time) const
    {  return densitySph(toPosSph(pos), time); }
};  // class BasePotentialSph


/** Parent class for analytic spherically-symmetric potentials.
    Derived classes should implement a single function defined in 
    the `math::IFunction::evalDeriv` interface, that computes
    the potential and up to two its derivatives as functions of spherical radius.
    Conversion into other coordinate systems is implemented in this class. */
class BasePotentialSphericallySymmetric: public BasePotential, public math::IFunction{
public:
    using math::IFunction::value;
    using BasePotential::value;

    virtual coord::SymmetryType symmetry() const { return coord::ST_SPHERICAL; }

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double /*time*/) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double /*time*/) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    /** redirect density computation to spherical coordinates */
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const
    {  return densitySph(toPosSph(pos), /*time*/ 0); }

    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const
    {  return densitySph(toPosSph(pos), /*time*/ 0); }

    virtual unsigned int numDerivs() const { return 2; }
};

///@}
/// \name   Wrapper classes used to construct temporary objects
///@{

/** A wrapper class providing a BasePotential interface for an arbitrary function of radius
    which computes a spherically-symmetric potential and its derivatives;
    should only be used to construct temporary objects passed as arguments to some routines.
*/
class FunctionToPotentialWrapper: public BasePotentialSphericallySymmetric{
    const math::IFunction &fnc;  ///< function representing the radial dependence of potential
public:
    virtual std::string name() const { return "FunctionToPotentialWrapper"; }
    virtual void evalDeriv(double r,
        double* val=NULL, double* deriv=NULL, double* deriv2=NULL) const {
        fnc.evalDeriv(r, val, deriv, deriv2); }
    explicit FunctionToPotentialWrapper(const math::IFunction &f) : fnc(f) {}
};

/** A wrapper class providing a BaseDensity interface for an arbitrary function of radius
    should only be used to construct temporary objects passed as arguments to some routines.
*/
class FunctionToDensityWrapper: public BaseDensity{
    const math::IFunction &fnc;  ///< function representing the spherically-symmetric density profile
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const
    { return fnc(toPosSph(pos).r); }
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const
    { return fnc(toPosSph(pos).r); }
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return fnc(pos.r); }
public:
    virtual coord::SymmetryType symmetry() const { return coord::ST_SPHERICAL; }
    virtual std::string name() const { return "FunctionToDensityWrapper"; }
    explicit FunctionToDensityWrapper(const math::IFunction &f) : fnc(f) {}
};

// classes for on-the-fly symmetrization
template<class BaseDensityOrPotential> class Sphericalized;
template<class BaseDensityOrPotential> class Axisymmetrized;

/** Wrapper class for on-the-fly spherical averaging of a given density profile.
    It can be used to represent a BaseDensity instance as a 1d IFunctionNoDeriv.
    If the underlying density is already spherical, it does no extra work,
    otherwise it averages the underlying density over two spherical angles,
    using a hard-coded fixed-order quadrature rule (so the error is worse for more strongly
    flattened models, but is usually below 1e-3).
    This class should be instantiated only as a temporary object;
    use the `DensitySphericalHarmonic` class with lmax=0 to create a regular spherically-averaged
    version of an arbitrary density profile, which is more efficient to evaluate repeatedly.
*/
template<> class Sphericalized<BaseDensity>: public BaseDensity, public math::IFunctionNoDeriv {
    const BaseDensity& dens;
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const
    { return value(toPosSph(pos).r); }
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const
    { return value(toPosSph(pos).r); }
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return value(pos.r); }
public:
    Sphericalized(const BaseDensity& _dens);
    virtual coord::SymmetryType symmetry() const { return coord::ST_SPHERICAL; }
    virtual std::string name() const
    { return isSpherical(dens.symmetry()) ? dens.name() : "Sphericalized " + dens.name(); }
    virtual double value(double r) const;
};

/** Wrapper class for on-the-fly spherical averaging of a given potential profile.
    It can be used to represent a BasePotential instance as a 1d IFunction with two derivatives.
    If the underlying potential is already spherical, it does no extra work,
    otherwise it averages the underlying potential and its derivatives over two spherical angles,
    using a hard-coded fixed-order quadrature rule (so the error is worse for more strongly
    flattened models, but is usually below 1e-3).
    This class should be instantiated only as a temporary object;
    use the `Multipole` class with lmax=0 to create a regular spherically-averaged
    version of an arbitrary potential profile, which is more efficient to evaluate repeatedly.
*/
template<> class Sphericalized<BasePotential>: public BasePotentialSphericallySymmetric {
    const BasePotential& pot;
public:
    Sphericalized(const BasePotential& _pot);
    virtual std::string name() const
    { return isSpherical(pot.symmetry()) ? pot.name() : "Sphericalized " + pot.name(); }
    virtual void evalDeriv(double r, double* val=NULL, double* der=NULL, double* der2=NULL) const;
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const;
};

/** Wrapper class for on-the-fly azimuthal averaging of a given density profile.
    If the underlying density is already Z-rotation symmetric, it does no extra work,
    otherwise it averages the underlying density over the phi angle,
    using a regularly spaced fixed-order quadrature.
    This class should be instantiated only as a temporary object;
    use the `DensityAzimuthalHarmonic` or `DensitySphericalHarmonic` classes with mmax=0
    to create a regular axisymmetrized version of an arbitrary density profile,
    which is more efficient to evaluate repeatedly.
*/
template<> class Axisymmetrized<BaseDensity>: public BaseDensity {
    const BaseDensity& dens;
    virtual double densityCar(const coord::PosCar &pos, double /*time*/) const
    { return densityCyl(toPosCyl(pos), /*time*/ 0); }
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return densityCyl(toPosCyl(pos), /*time*/ 0); }
public:
    Axisymmetrized(const BaseDensity& _dens);
    virtual coord::SymmetryType symmetry() const
    { return static_cast<coord::SymmetryType>(dens.symmetry() | coord::ST_ZROTATION); }
    virtual std::string name() const
    { return isZRotSymmetric(dens.symmetry()) ? dens.name() : "Axisymmetrized " + dens.name(); }
};

/** Wrapper class for on-the-fly azimuthal averaging of a given potential profile.
    It can be used to represent a BasePotential instance as a 1d IFunction with two derivatives,
    evaluating the azimuthally-averaged potential in the equatorial plane.
    If the underlying potential is already Z-rotation symmetric, it does no extra work,
    otherwise it averages the underlying potential and its derivatives over the phi angle,
    using a regularly spaced fixed-order quadrature.
    This class should be instantiated only as a temporary object;
    use the `CylSpline` or `Multipole` classes with mmax=0 to create a regular axisymmetrized
    version of an arbitrary potential profile, which is more efficient to evaluate repeatedly.
*/
template<> class Axisymmetrized<BasePotential>: public BasePotentialCyl, public math::IFunction {
    const BasePotential& pot;
    virtual void evalCyl(const coord::PosCyl &pos,
        double* value, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;
public:
    Axisymmetrized(const BasePotential& _pot);
    virtual coord::SymmetryType symmetry() const
    { return static_cast<coord::SymmetryType>(pot.symmetry() | coord::ST_ZROTATION); }
    virtual std::string name() const
    { return isZRotSymmetric(pot.symmetry()) ? pot.name() : "Axisymmetrized " + pot.name(); }
    // IFunction interface: evaluate the potential and its radial derivatives in the equatorial plane
    virtual void evalDeriv(const double R, double *val=NULL, double *der=NULL, double *der2=NULL) const
    {
        coord::GradCyl grad;
        coord::HessCyl hess;
        evalCyl(coord::PosCyl(R,0,0), val, der? &grad : 0, der2? &hess : 0, 0);
        if(der)
            *der = grad.dR;
        if(der2)
            *der2 = hess.dR2;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

///@}
/// \name   Non-member functions for all potential classes
///@{

/** Convenience functions for evaluating the total energy of a given position/velocity pair */
inline double totalEnergy(const BasePotential& potential, const coord::PosVelCar& p, double time=0)
{  return potential.value(p, time) + 0.5*(p.vx*p.vx+p.vy*p.vy+p.vz*p.vz); }

inline double totalEnergy(const BasePotential& potential, const coord::PosVelCyl& p, double time=0)
{  return potential.value(p, time) + 0.5*(pow_2(p.vR)+pow_2(p.vz)+pow_2(p.vphi)); }

inline double totalEnergy(const BasePotential& potential, const coord::PosVelSph& p, double time=0)
{  return potential.value(p, time) + 0.5*(pow_2(p.vr)+pow_2(p.vtheta)+pow_2(p.vphi)); }


// duplication of the symmetry testing functions from coord:: namespace
inline bool isXReflSymmetric(const BaseDensity& dens) {
    return  isXReflSymmetric(dens.symmetry()); }
inline bool isYReflSymmetric(const BaseDensity& dens) {
    return  isYReflSymmetric(dens.symmetry()); }
inline bool isZReflSymmetric(const BaseDensity& dens) {
    return  isZReflSymmetric(dens.symmetry()); }
inline bool isReflSymmetric(const BaseDensity& dens) {
    return  isReflSymmetric(dens.symmetry()); }
inline bool isZRotSymmetric(const BaseDensity& dens) {
    return  isZRotSymmetric(dens.symmetry()); }
inline bool isRotSymmetric(const BaseDensity& dens) {
    return  isRotSymmetric(dens.symmetry()); }
inline bool isBisymmetric(const BaseDensity& dens) {
    return  isBisymmetric(dens.symmetry()); }
inline bool isTriaxial(const BaseDensity& dens) {
    return  isTriaxial(dens.symmetry()); }
inline bool isAxisymmetric(const BaseDensity& dens) {
    return  isAxisymmetric(dens.symmetry()); }
inline bool isSpherical(const BaseDensity& dens) {
    return  isSpherical(dens.symmetry()); }


/** Find (spherical) radius corresponding to the given enclosed mass */
double getRadiusByMass(const BaseDensity& dens, const double enclosedMass);


/** Scaling transformation for 3-dimensional integration over volume:
    \param[in]  vars are three scaled variables that lie in the range [0:1];
    \param[out] jac (optional) is the jacobian of transformation (if set to NULL, it is not computed);
    \return  the un-scaled coordinates corresponding to the scaled variables.
*/
coord::PosCyl unscaleCoords(const double vars[], double* jac=NULL);

/// helper class for integrating density over volume
class DensityIntegrandNdim: public math::IFunctionNdim {
public:
    DensityIntegrandNdim(const BaseDensity& _dens, bool _nonnegative = false) :
        dens(_dens), axisym(isZRotSymmetric(_dens)), nonnegative(_nonnegative) {}

    /// evaluate the integrand for the density at one input point (scaled R,z,phi)
    virtual void eval(const double vars[], double values[]) const {
        evalmany(1, vars, values);
    }

    /// evaluate the integrand for many input points (scaled R,z,phi) at once
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const;

    /// dimensions of integration: only integrate in phi if density is not axisymmetric
    virtual unsigned int numVars() const { return axisym ? 2 : 3; }

    /// output a single value (the density)
    virtual unsigned int numValues() const { return 1; }

    const BaseDensity& dens;  ///< the density model to be integrated over
    const bool axisym;        ///< flag determining if the density is axisymmetric
    const bool nonnegative;   ///< flag determining whether to return zero if density was negative
};

///@}

}  // namespace potential
