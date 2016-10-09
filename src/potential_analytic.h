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
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "Plummer"; return text; }
    virtual double enclosedMass(const double radius) const;
    virtual double totalMass() const { return mass; }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadius;  ///< scale radius of the Plummer model  (b)

    /** Evaluate potential and up to two its derivatives by spherical radius. */
    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
};

/** Spherical Isochrone potential:
    \f$  \Phi(r) = - M / (b + \sqrt{r^2 + b^2})  \f$. */
class Isochrone: public BasePotentialSphericallySymmetric{
public:
    Isochrone(double _mass, double _scaleRadius) :
        mass(_mass), scaleRadius(_scaleRadius) {}
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "Isochrone"; return text; }
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
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "NFW"; return text; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double mass;         ///< normalization factor  (M);  equals to mass enclosed within ~5.3r_s
    const double scaleRadius;  ///< scale radius of the NFW model  (r_s)

    virtual void evalDeriv(double r,
        double* potential, double* deriv, double* deriv2) const;
};

/** Axisymmetric Miyamoto-Nagai potential:
    \f$  \Phi(r) = - M / \sqrt{ R^2 + (A + \sqrt{z^2+b^2})^2 }  \f$. */
class MiyamotoNagai: public BasePotentialCyl{
public:
    MiyamotoNagai(double _mass, double _scaleRadiusA, double _scaleRadiusB) :
        mass(_mass), scaleRadiusA(_scaleRadiusA), scaleRadiusB(_scaleRadiusB) {};
    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "MiyamotoNagai"; return text; }
    virtual double totalMass() const { return mass; }
private:
    const double mass;         ///< total mass  (M)
    const double scaleRadiusA; ///< first scale radius  (A),  determines the extent in the disk plane
    const double scaleRadiusB; ///< second scale radius (B),  determines the vertical extent

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

/** Triaxial logarithmic potential:
    \f$  \Phi(r) = \sigma^2 \ln[ r_c^2 + x^2 + (y/q)^2 + (z/p)^2 ]  \f$. */
class Logarithmic: public BasePotentialCar{
public:
    Logarithmic(double sigma, double coreRadius=0, double axisRatioYtoX=1, double axisRatioZtoX=1) :
        sigma2(pow_2(sigma)), coreRadius2(pow_2(coreRadius)),
        q2(pow_2(axisRatioYtoX)), p2(pow_2(axisRatioZtoX)) {}
    virtual coord::SymmetryType symmetry() const { 
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "Logarithmic"; return text; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double sigma2;       ///< squared asymptotic circular velocity (sigma)
    const double coreRadius2;  ///< squared core radius (r_c)
    const double q2;           ///< squared y/x axis ratio (q)
    const double p2;           ///< squared z/x axis ratio (p)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;
};

/** Triaxial harmonic potential:
 \f$  \Phi(r) = \Omega^2 [ x^2 + (y/q)^2 + (z/p)^2 ]/2  \f$. */
class Harmonic: public BasePotentialCar{
public:
    Harmonic(double Omega, double axisRatioYtoX=1, double axisRatioZtoX=1) :
        Omega2(pow_2(Omega)), q2(pow_2(axisRatioYtoX)), p2(pow_2(axisRatioZtoX)) {}
    virtual coord::SymmetryType symmetry() const { 
        return p2==1 ? (q2==1 ? coord::ST_SPHERICAL : coord::ST_AXISYMMETRIC) : coord::ST_TRIAXIAL; }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "Harmonic"; return text; }
    virtual double totalMass() const { return INFINITY; }
private:
    const double Omega2;       ///< squared oscillation frequency (Omega)
    const double q2;           ///< squared y/x axis ratio (q)
    const double p2;           ///< squared z/x axis ratio (p)

    virtual void evalCar(const coord::PosCar &pos,
        double* potential, coord::GradCar* deriv, coord::HessCar* deriv2) const;
};

}