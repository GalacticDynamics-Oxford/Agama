/** \file    potential_analytic.h
    \brief   Several common analytic potential models
    \author  Eugene Vasiliev
    \date    2009-2015
*/
#pragma once
#include "potential_base.h"

namespace potential{

/** Spherical Plummer potential:
    \f$  \Phi(r) = - M / \sqrt{r^2 + b^2}  \f$. */
class Plummer: public BasePotentialSphericallySymmetric{
public:
    Plummer(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Plummer"; }
    virtual double enclosedMass(const double radius) const;
    virtual double totalMass() const { return mass; }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadius;  ///< scale radius of the Plummer model  (b)

    /** Evaluate potential and up to two its derivatives by spherical radius. */
    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;

    /** explicitly define the density function, instead of relying on the potential derivatives
        (they suffer from cancellation errors already at r>1e5) */
    virtual double densitySph(const coord::PosSph &pos, double time) const;
};

/** Spherical Isochrone potential:
    \f$  \Phi(r) = - M / (b + \sqrt{r^2 + b^2})  \f$. */
class Isochrone: public BasePotentialSphericallySymmetric{
public:
    Isochrone(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Isochrone"; }
    virtual double totalMass() const { return mass; }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadius;  ///< scale radius of the Isochrone model  (b)
    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
};

/** Spherical Navarro-Frenk-White potential:
    \f$  \Phi(r) = - M \ln{1 + (r/r_s)} / r  \f$  (note that total mass is infinite and not M). */
class NFW: public BasePotentialSphericallySymmetric{
public:
    NFW(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "NFW"; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double mass;         ///< normalization factor  (M);  equals to mass enclosed within ~5.3r_s
    const double scaleRadius;  ///< scale radius of the NFW model  (r_s)

    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const
    { return (1./4/M_PI) * mass / pos.r / pow_2(pos.r + scaleRadius); }
};

/** Axisymmetric Miyamoto-Nagai potential:
    \f$  \Phi(r) = - M / \sqrt{ R^2 + (A + \sqrt{z^2+b^2})^2 }  \f$. */
class MiyamotoNagai: public BasePotentialCyl{
public:
    MiyamotoNagai(double _mass, double _scaleRadiusA, double _scaleRadiusB) :
        mass(_mass), scaleRadiusA(_scaleRadiusA), scaleRadiusB(_scaleRadiusB) {};
    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "MiyamotoNagai"; }
    virtual double totalMass() const { return mass; }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadiusA; ///< first scale radius  (A),  determines the extent in the disk plane
    const double scaleRadiusB; ///< second scale radius (B),  determines the vertical extent

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double time) const;
};

/** Triaxial logarithmic potential:
    \f$  \Phi(r) = (1/2) \sigma^2 \ln[ r_c^2 + x^2 + (y/p)^2 + (z/q)^2 ]  \f$. */
class Logarithmic: public BasePotentialCar{
public:
    Logarithmic(double sigma, double coreRadius=0, double axisRatioYtoX=1, double axisRatioZtoX=1) :
        sigma2(pow_2(sigma)), coreRadius2(pow_2(coreRadius)),
        p2(pow_2(axisRatioYtoX)), q2(pow_2(axisRatioZtoX)) {}
    virtual coord::SymmetryType symmetry() const {
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Logarithmic"; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double sigma2;       ///< squared asymptotic circular velocity (sigma)
    const double coreRadius2;  ///< squared core radius (r_c)
    const double p2;           ///< squared y/x axis ratio (p)
    const double q2;           ///< squared z/x axis ratio (q)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
};

/** Triaxial harmonic potential:
    \f$  \Phi(r) = (1/2) \Omega^2 [ x^2 + (y/p)^2 + (z/q)^2 ]  \f$. */
class Harmonic: public BasePotentialCar{
public:
    Harmonic(double Omega, double axisRatioYtoX=1, double axisRatioZtoX=1) :
        Omega2(pow_2(Omega)), p2(pow_2(axisRatioYtoX)), q2(pow_2(axisRatioZtoX)) {}
    virtual coord::SymmetryType symmetry() const {
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Harmonic"; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double Omega2;       ///< squared oscillation frequency (Omega)
    const double p2;           ///< squared y/x axis ratio (p)
    const double q2;           ///< squared z/x axis ratio (q)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
};


//------ Potential of a binary black hole ------//

/** Parameters describing the central single or binary supermassive black hole (BH) */
struct KeplerBinaryParams {
    double mass;  ///< mass of the central black hole or total mass of the binary
    double q;     ///< binary BH mass ratio (0<=q<=1)
    double sma;   ///< binary BH semimajor axis
    double ecc;   ///< binary BH eccentricity (0<=ecc<1)
    double phase; ///< binary BH orbital phase (0<=phase<2*pi)

    /// set defaults
    KeplerBinaryParams(double _mass=0, double _q=0, double _sma=0, double _ecc=0, double _phase=0) :
        mass(_mass), q(_q), sma(_sma), ecc(_ecc), phase(_phase) {}

    /** Compute the position and velocity of the two components of the binary black hole
        at the time 't'.
        if this is a single black hole, it stays at origin with zero velocity,
        and in case of a binary, its center of mass is fixed at origin.
        Output arrays of length 2 each will contain the x- and y-coordinates and
        corresponding velocities of both components of the binary at time 't';
        its orbit is assumed to lie in the x-y plane and directed along the x axis.
    */
    void keplerOrbit(double t, double bhX[], double bhY[], double bhVX[], double bhVY[]) const;

    /** Compute the time-dependent potential of the two black holes at the given point */
    double potential(const coord::PosCar& point, double time) const;
};

/** Time-dependent potential of a binary BH on a Kepler orbit in the x-y plane, centered at origin */
class KeplerBinary: public BasePotentialCar {
public:
    KeplerBinary(const KeplerBinaryParams& _params) : params(_params) {}
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "KeplerBinary"; }
    virtual coord::SymmetryType symmetry() const
    { return (params.sma==0 || params.q==0 ? coord::ST_SPHERICAL : coord::ST_NONE); }
    virtual double totalMass() const { return params.mass; }
    virtual double enclosedMass(const double radius) const
    { return radius>=params.sma ? params.mass : 0; }
private:
    const KeplerBinaryParams params;   ///< parameters of the binary
    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2, double time) const;
    virtual double densityCar(const coord::PosCar &, double) const { return 0; }
};

}