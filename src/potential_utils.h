/** \file    potential_utils.h
    \brief   General routines for various tasks associated with potential classes
    \author  Eugene Vasiliev
    \date    2009-2016
*/
#pragma once
#include "potential_base.h"
#include "math_spline.h"

namespace potential{

/** Compute circular velocity at a given (cylindrical) radius R in equatorial plane */
double v_circ(const BasePotential& potential, double R);

/** Compute angular momentum of a circular orbit in equatorial plane for a given value of energy */
double L_circ(const BasePotential& potential, double E);

/** Compute cylindrical radius of a circular orbit in equatorial plane for a given value of energy */
double R_circ(const BasePotential& potential, double E);

/** Compute cylindrical radius of an orbit in equatorial plane for a given value of
    z-component of angular momentum */
double R_from_Lz(const BasePotential& potential, double Lz);

/** Compute radius in the equatorial plane corresponding to the given value of potential */
double R_max(const BasePotential& potential, double Phi);

/** Compute epicycle frequencies for a circular orbit in the equatorial plane with radius R.
    \param[in]  potential is the instance of potential (must have axial symmetry)
    \param[in]  R     is the cylindrical radius 
    \param[out] kappa is the epicycle frequency of radial oscillations
    \param[out] nu    is the frequency of vertical oscillations
    \param[out] Omega is the azimuthal angular frequency (essentially v_circ/R)
*/
void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega);

/** Determine the behavior of potential at origin.
    The potential is approximated as a power-law with slope s>=-1:
    Phi(r) = Phi0 + coef * r^s  if s!=0,  or  Phi0 + coef * ln(r) if s==0;
    the corresponding density behaves as rho ~ r^(s-2),
    and if s>0, the value of potential at r=0 is finite and equals Phi0.
    \param[in]  potential  is the instance of potential (will be evaluated along x axis);
    \param[out] Phi0,coef  if not NULL, will contain the extrapolation coefficients;
    \returns the slope `s` of potential extrapolation law.
*/
double innerSlope(const BasePotential& potential, double* Phi0=NULL, double* coef=NULL);

/** Determine the behavior of potential at infinity.
    The potential is approximated as the Newtonian fall-off plus a power-law with slope s<0:
    Phi(r) = -M/r + coef * r^s  if s!=-1,  or  -M/r + coef * ln(r) / r  if s==-1;
    the corresponding density behaves as rho ~ r^(s-2),
    and if s<-1, the total mass is finite and equals M.
    \param[in]  potential  is the instance of potential (will be evaluated along x axis);
    \param[out] M,coef  if not NULL, will contain the extrapolation coefficients;
    \returns the slope `s` of potential extrapolation law.
*/
double outerSlope(const BasePotential& potential, double* M=NULL, double* coef=NULL);

/** Find the minimum and maximum radii of an orbit in the equatorial plane
    with given energy and angular momentum (which are the roots of equation
    \f$  2 (E - \Phi(R,z=0)) - L^2/R^2 = 0  \f$ ).
    \param[in]  potential  is the instance of axisymmetric potential;
    \param[in]  E is the total energy of the orbit;
    \param[in]  L is the angular momentum of the orbit;
    \param[out] R1 will contain the pericenter radius;
    \param[out] R2 will contain the apocenter radius;
    \throw std::invalid_argument if the potential is not axisymmetric, or energy is outside
    the allowed range, or angular momentum is not compatible with energy.
*/
void findPlanarOrbitExtent(const potential::BasePotential& potential, double E, double L,
    double& R1, double& R2);


/** Interpolator class for faster evaluation potential and related quantities --
    radius and angular momentum of a circular orbit as functions of energy,
    epicyclic frequencies as functions of radius.
    It is applicable to any axisymmetric potential that tends to zero at infinity,
    and may be regular or singular at origin; the dynamical quantities refer to orbits
    in the equatorial plane.
*/
class Interpolator: public math::IFunction {
public:
    /** The potential passed as parameter is only used to initialize the internal
        interpolation tables in the constructor, and is not used afterwards
        when interpolation is needed. */
    explicit Interpolator(const BasePotential& potential);

    /// compute the potential and its derivatives at the given cylindrical radius
    virtual void evalDeriv(const double R, double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

    /// provide up to 2 derivatives of potential
    virtual unsigned int numDerivs() const { return 2; }

    /// return L_circ(E) and optionally its first derivative
    double L_circ(const double E, double* deriv=NULL) const;

    /// return R_circ(E) and optionally its first derivative
    double R_circ(const double E, double* deriv=NULL) const;

    /// return radius of a circular orbit with the given angular momentum, optionally with derivative
    double R_from_Lz(const double Lz, double* deriv=NULL) const;

    /// return the radius corresponding to the given potential, optionally with derivative
    double R_max(const double Phi, double* deriv=NULL) const;

    /** return interpolated values of epicyclic frequencies at the given radius
        \param[in]  R     is the cylindrical radius 
        \param[out] kappa is the epicycle frequency of radial oscillations
        \param[out] nu    is the frequency of vertical oscillations
        \param[out] Omega is the azimuthal angular frequency (essentially v_circ/R)
    */
    void epicycleFreqs(const double R, double& kappa, double& nu, double& Omega) const;

    /** return the slope of potential near r=0 (same as the standalone function `innerSlope`).
        \param[out] Phi0,coef  if not NULL, will contain the extrapolation coefficients;
    */
    double innerSlope(double* Phi0=NULL, double* coef=NULL) const;
    
    /** return the slope of potential near r=0 (same as the standalone function `outerSlope`).
        \param[out] M,coef  if not NULL, will contain the extrapolation coefficients;
    */
    double outerSlope(double* M=NULL, double* coef=NULL) const {
        if(M)    *M = Mtot;
        if(coef) *coef = coefOut;
        return slopeOut;
    }
    
private:
    double Phi0;                ///< value of potential at r=0 (possibly -inf)
    double slopeOut, Mtot, coefOut; ///< coefficients for power-law potential at large radii
    math::QuinticSpline LofE;   ///< spline-interpolated scaled function for L_circ(E)
    math::QuinticSpline RofL;   ///< spline-interpolated scaled R_circ(L)
    math::QuinticSpline Phi;    ///< interpolated potential as a function of radius
    math::QuinticSpline RofPhi; ///< inverse function (radius as a function of potential)
    math::CubicSpline   freqNu; ///< ratio of squared epicyclic frequencies nu to Omega
};

/** Two-dimensional interpolator for peri/apocenter radii as functions of energy and angular momentum.
    This class is a further development of one-dimensional interpolator and works with
    axisymmetric potentials that have a finite value at r=0 and tend to zero at r=infinity.
    The accuracy of peri/apocenter radii interpolation is at the level of 1e-10 or better
    for almost all orbits; however, if the density profile is not decaying fast enough at infinity
    (i.e. r^-3 or shallower), the accuracy is rapidly deteriorating for very loosely bound orbits.
*/
class Interpolator2d {
public:
    /** Create internal interpolation tables for the given potential,
        which itself is not used afterwards */
    explicit Interpolator2d(const BasePotential& potential);

    /** Compute parameters of an orbit in the equatorial plane with the given energy and ang.momentum.
        \param[in]  E is the energy, which must be in the range Phi(0) <= E < 0;
        \param[in]  L is the angular momentum;
        \param[out] R1 will contain the pericenter radius,
        or NAN if the energy is outside the allowed range;
        \param[out] R2 same for the apocenter radius;
    */
    void findPlanarOrbitExtent(double E, double L, double& R1, double& R2) const;

    /** Compute scaled peri/apocenter radii (normalized to the radius of a circular orbit
        with the given energy), as functions of energy and relative angular momentum.
        This routine is similar to `findPlanarOrbitExtent` (actually, is a first step in that routine),
        except that it returns scaled radii, is applicable even in the limit r=0 or r=infinity,
        and does not perform a refinement step that improves the accuracy of the interpolation.
        \param[in]  E is the energy, which must be in the range Phi(0) <= E <= 0;
        \param[in]  Lrel is the relative angular momentum (scaled to the ang.mom.of a circular orbit),
        must be in the range 0 <= Lrel <= 1.
        \param[out] R1rel will contain the pericenter radius, normalized to the radius of a circular
        orbit (lies in the range 0 <= R1rel <= 1), or NAN if E or Lrel are outside the allowed range;
        \param[out] R2rel same for the apocenter radius (lies in the range 1 <= R2rel < infinity).
    */
    void findScaledOrbitExtent(double E, double Lrel, double &R1rel, double &R2rel) const;

    /// 1d interpolator for potential and other quantities in the equatorial plane
    const Interpolator pot;

private:
    math::QuinticSpline2d intR1, intR2; ///< 2d interpolators for scaled peri/apocenter radii
};

}  // namespace potential
