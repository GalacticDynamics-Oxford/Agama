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


/** Interpolator class for faster evaluation of potential and related quantities --
    radius and angular momentum of a circular orbit as functions of energy,
    epicyclic frequencies as functions of radius (all in the equatorial plane).
    It is applicable to any axisymmetric potential that tends to zero at infinity,
    is monotonic with radius, and may be regular or singular at origin.
*/
class Interpolator: public math::IFunction {
public:
    /** The potential passed as parameter is only used to initialize the internal
        interpolation tables in the constructor, and is not used afterwards
        when interpolation is needed. */
    explicit Interpolator(const BasePotential& potential);

    /// compute the potential and its derivatives at the given cylindrical radius in the z=0 plane
    virtual void evalDeriv(const double R,
        double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const;

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
    double invPhi0;             ///< 1/(value of potential at r=0), or 0 if the potential is singular
    double slopeOut, Mtot, coefOut; ///< coefficients for power-law potential at large radii
    math::QuinticSpline LofE;   ///< spline-interpolated scaled function for L_circ(E)
    math::QuinticSpline RofL;   ///< spline-interpolated scaled R_circ(L)
    math::QuinticSpline PhiofR; ///< interpolated potential as a function of radius
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

/** Computation of phase volume for a spherical potential.
    Phase volume h(E) is defined as
    \f$  h(E) = 16\pi^2/3 \int_0^{r_{max}(E)} r^2 v^3 dr = \int_{\Phi(0)}^E g(E') dE'  \f$;
    its derivative g(E) is called the density of states and is given by
    \f$  g(E) = 16\pi^2 \int_0^{r_{max}(E)} r^2 v dr = 4\pi^2 L_{circ}^2(E) T_{rad}(E)  \f$,
    where  \f$  v = \sqrt{2(E-\Phi(r))}  \f$,  L_circ is the angular momentum of a circular orbit,
    and  \f$  T_{rad}(E) = 2 \int_0^{r_{max}(E)} dr/v  \f$  is the radial period.
    These quantities are computed for the given potential Phi(E), which is provided through an
    `IFunction` interface (i.e., may be an instance of `potential::Interpolator`, or
    `potential::PotentialWrapper`),
    and stored as interpolating splines; the potential itself is not stored (only its value at r=0).
    This class provides methods for computing h(E) and E(h), together with the derivative g(E),
    and the inverse transformation E(h) and g(h).
*/
class PhaseVolume: public math::IFunction {
public:
    /// Construct the interpolator for h(E) from an arbitrary function Phi(r),
    /// which must be monotonic in radius (otherwise a std::runtime_error exception is thrown)
    explicit PhaseVolume(const math::IFunction& potential);

    /** compute h and g from E.
        \param[in]  E is the energy (may be arbitrary;
        if E>=0, return infinity, if E<=Phi(0), return 0);
        \param[out] h is the phase volume corresponding to the given energy;
        \param[out] g is the density of states (dh/dE), if this pointer is NULL then it's not computed.
    */
    virtual void evalDeriv(const double E, double* h=NULL, double* g=NULL, double* =NULL) const;

    /** compute E, g and dg/dh from h.
        \param[in]  h  is the phase volume [0..infinity]
        \param[out] g  is the density of states, g=1/(dE/dh), if NULL then it's not computed;
        \param[out] dgdh is the derivative dg/dh, if NULL then it's not computed; 
        \return  the energy E corresponding to the given phase volume h.
    */
    double E(const double h, double* g=NULL, double* dgdh=NULL) const;

    /// compute deltaE = E(h1) - E(h2) in a way that does not suffer from cancellation;
    /// h1 and h2 are specified through their logarithms
    double deltaE(const double logh1, const double logh2, double* g1=NULL) const;

    /// provides one derivative of h(E)
    virtual unsigned int numDerivs() const { return 1; }

private:
    double invPhi0;             ///< 1/Phi(0), where Phi(0) is the value of potential at r=0
    math::QuinticSpline HofE;   ///< phase volume h(E), suitably scaled
    math::QuinticSpline EofH;   ///< inverse of the above
};

}  // namespace potential
