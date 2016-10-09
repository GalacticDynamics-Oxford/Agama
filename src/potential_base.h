/** \file    potential_base.h
    \brief   Base density and potential classes
    \author  Eugene Vasiliev
    \date    2009-2015
*/
#pragma once
#include "coord.h"

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
    These objects also cannot be copied by value, thus if one needs to pass them around,
    one should use either references to the base class (density or potential):
 
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
        The actual computation is implemented in separately-named protected virtual functions. */
    double density(const coord::PosCar &pos) const {
        return densityCar(pos); }
    double density(const coord::PosCyl &pos) const {
        return densityCyl(pos); }
    double density(const coord::PosSph &pos) const {
        return densitySph(pos); }

    /// returns the symmetry type of this density or potential
    virtual coord::SymmetryType symmetry() const = 0;

    /** return the name of density or potential model;
        this also serves as a unique pointer that distinguishes between different classes,
        so the derived classes should return a pointer to a static const char* variable.
    */
    virtual const char* name() const = 0;

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

protected:

    /** evaluate density at the position specified in cartesian coordinates */
    virtual double densityCar(const coord::PosCar &pos) const = 0;

    /** evaluate density at the position specified in cylindrical coordinates */
    virtual double densityCyl(const coord::PosCyl &pos) const = 0;

    /** Evaluate density at the position specified in spherical coordinates */
    virtual double densitySph(const coord::PosSph &pos) const = 0;

    /** Empty constructor is needed explicitly since we have disabled the default copy constructor */
    BaseDensity() {};

private:
/** Copy constructor and assignment operators are not allowed, because their inadvertent
    application (slicing) would lead to a complex derived class being assigned to 
    a variable of base class, thus destroying its internal state.
    Thus these polymorphic objects are simply non-copyable, period.
    If one needs to pass a reference to an object that may possibly outlive the scope
    of the object being passed, then one should use create this object dynamically 
    and place it into a shared_ptr.
*/
    BaseDensity(const BaseDensity&);
    BaseDensity& operator=(const BaseDensity&);
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
                    of potential in the variable addressed by this pointer.  */
    void eval(const coord::PosCar &pos,
        double* potential=NULL, coord::GradCar* deriv=NULL, coord::HessCar* deriv2=NULL) const {
        return evalCar(pos, potential, deriv, deriv2); }
    void eval(const coord::PosCyl &pos,
        double* potential=NULL, coord::GradCyl* deriv=NULL, coord::HessCyl* deriv2=NULL) const {
        return evalCyl(pos, potential, deriv, deriv2); }
    void eval(const coord::PosSph &pos,
        double* potential=NULL, coord::GradSph* deriv=NULL, coord::HessSph* deriv2=NULL) const {
        return evalSph(pos, potential, deriv, deriv2); }

    /** Shorthand for evaluating the value of potential at a given point in any coordinate system */
    template<typename coordSysT>
    inline double value(const coord::PosT<coordSysT>& point) const {
        double val;
        eval(point, &val);
        return val;
    }

protected:
    /** evaluate potential and up to two its derivatives in cartesian coordinates;
        must be implemented in derived classes */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const = 0;

    /** evaluate potential and up to two its derivatives in cylindrical coordinates */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const = 0;

    /** evaluate potential and up to two its derivatives in spherical coordinates */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const = 0;

    /** Default implementation computes the density from Laplacian of the potential,
        but the derived classes may instead provide an explicit expression for it. */
    virtual double densityCar(const coord::PosCar &pos) const;
    virtual double densityCyl(const coord::PosCyl &pos) const;
    virtual double densitySph(const coord::PosSph &pos) const;
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
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvert<coord::Car, coord::Cyl>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvert<coord::Car, coord::Sph>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cartesian) coordinate system. */
    virtual void evalScalar(const coord::PosCar& pos,
        double* val=NULL, coord::GradCar* deriv=NULL, coord::HessCar* deriv2=NULL) const
    { evalCar(pos, val, deriv, deriv2); }

    /** redirect density computation to a Laplacian in more suitable coordinates */
    virtual double densityCyl(const coord::PosCyl &pos) const
    {  return densityCar(toPosCar(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const
    {  return densityCar(toPosCar(pos)); }
};  // class BasePotentialCar


/** Parent class for potentials that are evaluated in cylindrical coordinates.
    It leaves the implementation of `evalCyl` member function for cylindrical coordinates undefined, 
    but provides the conversion from cylindrical to cartesian and spherical coordinates. */
class BasePotentialCyl: public BasePotential, coord::IScalarFunction<coord::Cyl>{

    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvert<coord::Cyl, coord::Car>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvert<coord::Cyl, coord::Sph>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cylindrical) coordinate system. */
    virtual void evalScalar(const coord::PosCyl& pos,
        double* val=NULL, coord::GradCyl* deriv=NULL, coord::HessCyl* deriv2=NULL) const {
        evalCyl(pos, val, deriv, deriv2);
    }

    /** redirect density computation to more suitable coordinates */
    virtual double densityCar(const coord::PosCar &pos) const
    {  return densityCyl(toPosCyl(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const
    {  return densityCyl(toPosCyl(pos)); }
};  // class BasePotentialCyl


/** Parent class for potentials that are evaluated in spherical coordinates.
    It leaves the implementation of `evalSph member` function for spherical coordinates undefined, 
    but provides the conversion from spherical to cartesian and cylindrical coordinates. */
class BasePotentialSph: public BasePotential, coord::IScalarFunction<coord::Sph>{

    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvert<coord::Sph, coord::Car>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in cylindrical coordinates. */
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvert<coord::Sph, coord::Cyl>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Spherical) coordinate system. */
    virtual void evalScalar(const coord::PosSph& pos,
        double* val=NULL, coord::GradSph* deriv=NULL, coord::HessSph* deriv2=NULL) const { 
        evalSph(pos, val, deriv, deriv2); 
    }

    /** redirect density computation to more suitable coordinates */
    virtual double densityCar(const coord::PosCar &pos) const
    {  return densitySph(toPosSph(pos)); }
    virtual double densityCyl(const coord::PosCyl &pos) const
    {  return densitySph(toPosSph(pos)); }
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

    /** find the mass enclosed within a given radius from the radial component of force */
    virtual double enclosedMass(const double radius) const;

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    /** redirect density computation to spherical coordinates */
    virtual double densityCar(const coord::PosCar &pos) const
    {  return densitySph(toPosSph(pos)); }

    virtual double densityCyl(const coord::PosCyl &pos) const
    {  return densitySph(toPosSph(pos)); }

    virtual unsigned int numDerivs() const { return 2; }
};

///@}
/// \name   Wrapper classes used to construct temporary objects
///@{

/** A wrapper class providing a BasePotential interface for an arbitrary function of radius
    which computes a spherically-symmetric potential and its derivatives;
    should only be used to construct temporary objects passed as arguments to some routines.
*/
class FunctionWrapper: public BasePotentialSphericallySymmetric{
    const math::IFunction &fnc;  ///< function representing the radial dependence of potential
    virtual const char* name() const { return "FunctionWrapper"; }
    virtual void evalDeriv(double r, double* val, double* deriv, double* deriv2) const {
        fnc.evalDeriv(r, val, deriv, deriv2); }
public:
    explicit FunctionWrapper(const math::IFunction &f) : fnc(f) {}
};

/** A wrapper class providing a IFunction interface to a potential */
class PotentialWrapper: public math::IFunction {
    const BasePotential& potential;  ///< potential that is evaluated along x-axis
    virtual void evalDeriv(const double R, double *val, double *der, double *der2) const {
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(R,0,0), val, der? &grad : 0, der2? &hess : 0);
        if(der)
            *der = grad.dR;
        if(der2)
            *der2 = hess.dR2;
    }
    virtual unsigned int numDerivs() const { return 2; }
public:
    explicit PotentialWrapper(const BasePotential &p) : potential(p) {};
};

/** A wrapper class providing a IFunction interface to a spherically-symmetric density */
class DensityWrapper: public math::IFunctionNoDeriv {
    const BaseDensity& density;  ///< density that is evaluated along x-axis
    virtual double value(const double r) const {
        return density.density(coord::PosCyl(r, 0, 0));
    }
public:
    explicit DensityWrapper(const BaseDensity &d) : density(d) {};
};

///@}
/// \name   Non-member functions for all potential classes
///@{

/** Convenience functions for evaluating the total energy of a given position/velocity pair */
inline double totalEnergy(const BasePotential& potential, const coord::PosVelCar& p)
{  return potential.value(p) + 0.5*(p.vx*p.vx+p.vy*p.vy+p.vz*p.vz); }

inline double totalEnergy(const BasePotential& potential, const coord::PosVelCyl& p)
{  return potential.value(p) + 0.5*(pow_2(p.vR)+pow_2(p.vz)+pow_2(p.vphi)); }

inline double totalEnergy(const BasePotential& potential, const coord::PosVelSph& p)
{  return potential.value(p) + 0.5*(pow_2(p.vr)+pow_2(p.vtheta)+pow_2(p.vphi)); }


// duplication of the symmetry testing functionlets from coord:: namespace
inline bool isXReflSymmetric(const BaseDensity& dens) {
    return isXReflSymmetric(dens.symmetry()); }
inline bool isYReflSymmetric(const BaseDensity& dens) {
    return isYReflSymmetric(dens.symmetry()); }
inline bool isZReflSymmetric(const BaseDensity& dens) {
    return isZReflSymmetric(dens.symmetry()); }
inline bool isReflSymmetric(const BaseDensity& dens) {
    return isReflSymmetric(dens.symmetry()); }
inline bool isZRotSymmetric(const BaseDensity& dens) {
    return isZRotSymmetric(dens.symmetry()); }
inline bool isTriaxial(const BaseDensity& dens) {
    return isTriaxial(dens.symmetry()); }
inline bool isAxisymmetric(const BaseDensity& dens) {
    return isAxisymmetric(dens.symmetry()); }
inline bool isSpherical(const BaseDensity& dens) {
    return isSpherical(dens.symmetry()); }


/** Find (spherical) radius corresponding to the given enclosed mass */
double getRadiusByMass(const BaseDensity& dens, const double enclosedMass);

/** Find the asymptotic power-law index of density profile at r->0 */
double getInnerDensitySlope(const BaseDensity& dens);

/** Compute m-th azimuthal harmonic of density profile by averaging the density over angle phi 
    with weight factor cos(m phi) or sin(m phi), at the given point in (R,z) plane */
double computeRho_m(const BaseDensity& dens, double R, double z, int m);


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

    /// integrand for the density at a given point (R,z,phi) with appropriate coordinate scaling
    virtual void eval(const double vars[], double values[]) const;

    /// dimensions of integration: only integrate in phi if density is not axisymmetric
    virtual unsigned int numVars() const { return axisym ? 2 : 3; }

    /// output a single value (the density)
    virtual unsigned int numValues() const { return 1; }

    /// convert from scaled variables to the real position;
    /// optionally compute the jacobian of transformation if jac!=NULL
    inline coord::PosCyl unscaleVars(const double vars[], double* jac=NULL) const {
        return unscaleCoords(vars, jac); }
private:
    const BaseDensity& dens;  ///< the density model to be integrated over
    const bool axisym;        ///< flag determining if the density is axisymmetric
    const bool nonnegative;   ///< flag determining whether to return zero if density was negative
};

///@}
}  // namespace potential
