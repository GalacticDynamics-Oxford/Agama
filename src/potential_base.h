#pragma once
#include "coord.h"

namespace potential{

/** defines the symmetry properties of density or potential */
enum SymmetryType{ 
    ST_NONE         = 0, ///< no symmetry whatsoever
    ST_REFLECTION   = 1, ///< reflection about origin (change of sign of all coordinates simultaneously)
    ST_PLANESYM     = 2, ///< reflection about principal planes (change of sign of any coordinate)
    ST_ZROTSYM      = 4, ///< rotation about z axis
    ST_SPHSYM       = 8, ///< rotation about arbitrary axis
    ST_TRIAXIAL     = ST_REFLECTION | ST_PLANESYM, ///< triaxial symmetry
    ST_AXISYMMETRIC = ST_TRIAXIAL | ST_ZROTSYM,    ///< axial symmetry
    ST_SPHERICAL    = ST_AXISYMMETRIC | ST_SPHSYM, ///< spherical symmetry
    ST_DEFAULT      = ST_TRIAXIAL                  ///< a default value
};

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;

/// absolute error in potential computation (supercedes the relative error in case of very small coefficients)
const double EPSABS_POTENTIAL_INT = 1e-15;

/// relative accuracy of density computation
const double EPSREL_DENSITY_INT = 1e-4;

/// \name  Base class for all density models
///@{

/** Abstract class defining a density profile without a corresponding potential. */
class BaseDensity{
public:

    BaseDensity() {};
    virtual ~BaseDensity() {};

    /** Evaluate density at the position in a specified coordinate system.
        Template parameter may be Car, Cyl, or Sph, and the actual computation 
        is implemented in separately-named protected virtual functions. */
    template<typename coordSysT>
    double density(const coord::PosT<coordSysT> &pos) const;

    /// returns the symmetry type of this density or potential
    virtual SymmetryType symmetry() const=0;

    /// return the name of density or potential model
    virtual const char* name() const=0;

    /** estimate the mass enclosed within a given spherical radius;
        default implementation integrates density over volume, but derived classes
        may provide a cheaper alternative (not necessarily a very precise one). */
    virtual double enclosedMass(const double radius) const;
    
    /** return the total mass of the density model (possibly infinite);
        default implementation estimates the asymptotic behaviour of density at large radii,
        but derived classes may instead return a specific value. */
    virtual double totalMass() const;

protected:
//  Protected members: virtual methods for `density` in different coordinate systems
    /** evaluate density at the position specified in cartesian coordinates */
    virtual double density_car(const coord::PosCar &pos) const=0;

    /** evaluate density at the position specified in cylindrical coordinates */
    virtual double density_cyl(const coord::PosCyl &pos) const=0;

    /** Evaluate density at the position specified in spherical coordinates */
    virtual double density_sph(const coord::PosSph &pos) const=0;

/** Copy constructor and assignment operators are not allowed, because 
    their inadvertent application (slicing) would lead to a complex derived class 
    being assigned to a variable of base class, thus destroying its internal state. */
    BaseDensity(const BaseDensity& src);
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
    Density is computed from Laplacian in each coordinate system.
*/
class BasePotential: public BaseDensity{
public:
    BasePotential() : BaseDensity() {};
    virtual ~BasePotential() {};

    /** Evaluate potential and up to two its derivatives in a specified coordinate system.
        \param[in]  pos is the position in the given coordinates.
        \param[out] potential - if not NULL, store the value of potential
                    in the variable addressed by this pointer.
        \param[out] deriv - if not NULL, store the gradient of potential
                    in the variable addressed by this pointer.
        \param[out] deriv2 - if not NULL, store the Hessian (matrix of second derivatives)
                    of potential in the variable addressed by this pointer.  */
    template<typename coordSysT>
    void eval(const coord::PosT<coordSysT> &pos,
        double* potential=0, 
        coord::GradT<coordSysT>* deriv=0, 
        coord::HessT<coordSysT>* deriv2=0) const;

protected:
    /** evaluate potential and up to two its derivatives in cartesian coordinates;
        must be implemented in derived classes */
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const=0;

    /** evaluate potential and up to two its derivatives in cylindrical coordinates */
    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const=0;

    /** evaluate potential and up to two its derivatives in spherical coordinates */
    virtual void eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const=0;

    /** Default implementation computes the density from Laplacian of the potential,
        but the derived classes may instead provide an explicit expression for it. */
    virtual double density_car(const coord::PosCar &pos) const;
    virtual double density_cyl(const coord::PosCyl &pos) const;
    virtual double density_sph(const coord::PosSph &pos) const;
};  // class BasePotential

// Template specializations for `BaseDensity::density` and `BasePotential::eval` 
// in particular coordinate systems 
// (need to be declared outside the scope of class definition)

/// Evaluate density at the position specified in cartesian coordinates
template<> inline double BaseDensity::density(const coord::PosCar &pos) const
{  return density_car(pos); };

/// Evaluate density at the position specified in cylindrical coordinates
template<> inline double BaseDensity::density(const coord::PosCyl &pos) const
{  return density_cyl(pos); };

/// Evaluate density at the position specified in spherical coordinates
template<> inline double BaseDensity::density(const coord::PosSph &pos) const
{  return density_sph(pos); };

/// Evaluate potential and up to two its derivatives in cartesian coordinates
template<> inline void BasePotential::eval(const coord::PosCar &pos,
    double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const
{  eval_car(pos, potential, deriv, deriv2); };

/// Evaluate potential and up to two its derivatives in cylindrical coordinates
template<> inline void BasePotential::eval(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{  eval_cyl(pos, potential, deriv, deriv2); };

/// Evaluate potential and up to two its derivatives in spherical coordinates
template<> inline void BasePotential::eval(const coord::PosSph &pos,
    double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const
{  eval_sph(pos, potential, deriv, deriv2); };

///@}
/// \name   Base classes for potentials that implement the computations in a particular coordinate system
///@{

/** Parent class for potentials that are evaluated in cartesian coordinates.
    It leaves the implementation of `eval_car` member function for cartesian coordinates undefined, 
    but provides the conversion from cartesian to cylindrical and spherical coordinates in eval_cyl and eval_sph. */
class BasePotentialCar: public BasePotential, coord::IScalarFunction<coord::Car>{
public:
    BasePotentialCar() : BasePotential() {}

private:
    /** evaluate potential and up to two its derivatives in cylindrical coordinates. */
    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvert<coord::Car, coord::Cyl>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvert<coord::Car, coord::Sph>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cartesian) coordinate system. */
    virtual void evalScalar(const coord::PosCar& pos,
        double* value=0, coord::GradCar* deriv=0, coord::HessCar* deriv2=0) const
    { eval_car(pos, value, deriv, deriv2); }
#if 0  // testing
    /** reimplement density computation via Laplacian in more suitable coordinates */
    virtual double density_cyl(const coord::PosCyl &pos) const
    {  return density_car(coord::toPosCar(pos)); }
    virtual double density_sph(const coord::PosSph &pos) const
    {  return density_car(coord::toPosCar(pos)); }
#endif
};  // class BasePotentialCar


/** Parent class for potentials that are evaluated in cylindrical coordinates.
    It leaves the implementation of `eval_cyl` member function for cylindrical coordinates undefined, 
    but provides the conversion from cylindrical to cartesian and spherical coordinates. */
class BasePotentialCyl: public BasePotential, coord::IScalarFunction<coord::Cyl>{
public:
    BasePotentialCyl() : BasePotential() {}

private:
    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvert<coord::Cyl, coord::Car>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in spherical coordinates. */
    virtual void eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvert<coord::Cyl, coord::Sph>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Cylindrical) coordinate system. */
    virtual void evalScalar(const coord::PosCyl& pos,
        double* value=0, coord::GradCyl* deriv=0, coord::HessCyl* deriv2=0) const {
        eval_cyl(pos, value, deriv, deriv2);
    }
#if 0  // testing
    /** reimplement density computation in more suitable coordinates */
    virtual double density_car(const coord::PosCar &pos) const
    {  return density_cyl(coord::toPosCyl(pos)); }
    virtual double density_sph(const coord::PosSph &pos) const
    {  return density_cyl(coord::toPosCyl(pos)); }
#endif
};  // class BasePotentialCyl


/** Parent class for potentials that are evaluated in spherical coordinates.
    It leaves the implementation of `eval_sph member` function for spherical coordinates undefined, 
    but provides the conversion from spherical to cartesian and cylindrical coordinates. */
class BasePotentialSph: public BasePotential, coord::IScalarFunction<coord::Sph>{
public:
    BasePotentialSph() : BasePotential() {}

private:
    /** evaluate potential and up to two its derivatives in cartesian coordinates. */
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvert<coord::Sph, coord::Car>(*this, pos, potential, deriv, deriv2);
    }

    /** evaluate potential and up to two its derivatives in cylindrical coordinates. */
    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvert<coord::Sph, coord::Cyl>(*this, pos, potential, deriv, deriv2);
    }

    /** implements the IScalarFunction interface for evaluating the potential and its derivatives 
        in the preferred (Spherical) coordinate system. */
    virtual void evalScalar(const coord::PosSph& pos,
        double* value=0, coord::GradSph* deriv=0, coord::HessSph* deriv2=0) const { 
        eval_sph(pos, value, deriv, deriv2); 
    }
#if 0  // testing
    /** reimplement density computation in more suitable coordinates */
    virtual double density_car(const coord::PosCar &pos) const
    {  return density_sph(coord::toPosSph(pos)); }
    virtual double density_cyl(const coord::PosCyl &pos) const
    {  return density_sph(coord::toPosSph(pos)); }
#endif
};  // class BasePotentialSph


/** Parent class for analytic spherically-symmetric potentials.
    Derived classes should implement a single function defined in 
    the `math::IFunction::evalDeriv` interface, that computes
    the potential and up to two its derivatives as functions of spherical radius.
    Conversion into other coordinate systems is implemented in this class. */
class BasePotentialSphericallySymmetric: public BasePotential, math::IFunction{
public:
    BasePotentialSphericallySymmetric() : BasePotential() {}

    virtual SymmetryType symmetry() const { return ST_SPHERICAL; }

    /** find the mass enclosed within a given radius from Poisson equation */
    virtual double enclosedMass(double radius) const;

private:
    virtual void eval_car(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void eval_cyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual void eval_sph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2) const {
        coord::evalAndConvertSph(*this, pos, potential, deriv, deriv2); }

    virtual int numDerivs() const { return 2; }
};

///@}
/// \name   Convenience functions
///@{

/** Shorthand for evaluating the value of potential at a given point */
template<typename coordT>
double Phi(const BasePotential& potential, const coordT& point) {
    double val;
    potential.eval(point, &val);
    return val;
}

/** Convenience function for evaluating total energy of a given position/velocity pair */
template<typename coordT>
double totalEnergy(const BasePotential& potential, const coord::PosVelT<coordT>& posvel);

template<>
inline double totalEnergy(const BasePotential& potential, const coord::PosVelCar& p)
{  return Phi(potential, p) + 0.5*(p.vx*p.vx+p.vy*p.vy+p.vz*p.vz); }

template<>
inline double totalEnergy(const BasePotential& potential, const coord::PosVelCyl& p)
{  return Phi(potential, p) + 0.5*(pow_2(p.vR)+pow_2(p.vz)+pow_2(p.vphi)); }

template<>
inline double totalEnergy(const BasePotential& potential, const coord::PosVelSph& p)
{  return Phi(potential, p) + 0.5*(pow_2(p.vr)+pow_2(p.vtheta)+pow_2(p.vphi)); }


/** Compute circular velocity at a given radius in equatorial plane */
double v_circ(const BasePotential& potential, double radius);

/** Find radius corresponding to the given enclosed mass */
double getRadiusByMass(const BaseDensity& dens, const double mass);

/** Find the asymptotic power-law index of density profile at r->0 */
double getInnerDensitySlope(const BaseDensity& dens);

///@}
}  // namespace potential
