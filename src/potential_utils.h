/** \file    potential_utils.h
    \brief   General routines for various tasks associated with potential classes
    \author  Eugene Vasiliev
    \date    2009-2016
*/
#pragma once
#include "potential_base.h"
#include "math_spline.h"

namespace potential{

/** Compute the circular velocity \$ v_{circ} = \sqrt{R d\Phi/dR} \$ 
    at the given cylindrical radius R in the equatorial plane;
    the potential is axisymmetrized if necessary.
*/
double v_circ(const BasePotential& potential, double R);

/** Compute the cylindrical radius of a circular orbit in the equatorial plane
    for a given value of energy; the potential is axisymmetrized if necessary.
*/
double R_circ(const BasePotential& potential, double E);

/** Compute the characteristic orbital period 2 pi R_circ / v_circ as a function of energy;
    the potential is axisymmetrized if necessary.
*/
inline double T_circ(const BasePotential& potential, double E)
{
    double R = R_circ(potential, E);
    return R / v_circ(potential, R) * 2*M_PI;
}

/** Compute the angular momentum of a circular orbit in the equatorial plane
    for the given value of energy; the potential is axisymmetrized if necessary.
*/
inline double L_circ(const BasePotential& potential, double E)
{
    double R = R_circ(potential, E);
    return R * v_circ(potential, R);
}

/** Compute cylindrical radius of an orbit in the equatorial plane for a given value of
    z-component of angular momentum; the potential is axisymmetrized if necessary.
*/
double R_from_Lz(const BasePotential& potential, double Lz);

/** Compute the radius of a radial orbit in the equatorial plane with the given energy,
    i.e. the root of Phi(R)=E; the potential is axisymmetrized if necessary.
*/
double R_max(const BasePotential& potential, double Phi);

/** Compute epicycle frequencies for a circular orbit in the equatorial plane with radius R.
    \param[in]  potential is the instance of potential (axisymmetrized if necessary);
    \param[in]  R     is the cylindrical radius;
    \param[out] kappa is the epicycle frequency of radial oscillations;
    \param[out] nu    is the frequency of vertical oscillations;
    \param[out] Omega is the azimuthal angular frequency (essentially v_circ/R).
*/
void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega);

/** Determine the behavior of potential at origin.
    The potential is approximated as a power-law with slope s>=-1:
    Phi(r) = Phi0 + coef * r^s  if s!=0,  or  Phi0 + coef * ln(r) if s==0;
    the corresponding density behaves as rho ~ r^(s-2),
    and if s>0, the value of potential at r=0 is finite and equals Phi0.
    \param[in]  potential  is the instance of potential represented by a 1d function;
    \param[out] Phi0,coef  if not NULL, will contain the extrapolation coefficients;
    \returns the slope `s` of potential extrapolation law.
*/
double innerSlope(const math::IFunction& potential, double* Phi0=NULL, double* coef=NULL);

/** Find the minimum and maximum radii of an orbit in the equatorial plane with a given
    energy and angular momentum; for an axisymmetric potential, these are the roots of
    \f$  2 (E - \Phi(R,z=0)) - L^2/R^2 = 0  \f$.
    \param[in]  potential  is the instance of potential (axisymmetrized if necessary);
    \param[in]  E is the total energy of the orbit;
    \param[in]  L is the angular momentum of the orbit;
    \param[out] R1 will contain the pericenter radius;
    \param[out] R2 will contain the apocenter radius;
    \return NAN for both output parameters in case of other errors (e.g., if the energy is
    outside the allowed range); if L is greater than Lcirc(E), return Rcirc(E) for both R1,R2.
*/
void findPlanarOrbitExtent(const BasePotential& potential, double E, double L,
    double& R1, double& R2);

/** Find the two points Z1,Z2 where the potential Phi(X,Y,Z) equals the target value E.
    The line of sight is specified by two coordinates in the X,Y plane of the 'observed'
    coordinate system and its orientation with respect to the 'intrinsic' coordinates of the potential.
    \param[in]  potential  is the instance of potential (with arbitrary symmetry);
    \param[in]  E  is the required target value of the potential;
    \param[in]  X,Y  are two coordinates in the 'observed' reference frame, with the third coordinate Z
    (the line of sight) being the one searched for.
    \param[in] orientation  specifies the orientation of the XYZ (observed) coordinate system
    with respect to xyz (intrinsic coorinate system for the potential).
    \param[out] Zm  will contain the value of Z corresponding to the minimum of the potential.
    \param[out] Z1,Z2  will contain the two roots of Phi(X,Y,Z)=E, such that Z1 <= Zm <= Z2,
    or NAN if the potential is everywhere higher than E.
*/
void findRoots(const BasePotential& potential, double E,
    double X, double Y, const coord::Orientation& orientation,
    /*output*/ double &Zm, double &Z1, double &Z2);


/** Create a grid in radius suitable for interpolation of various quantities depending on the potential.
    The grid spacing is determined by the variation of the logarithmic derivative of the potential
    (becomes more sparse when the potential approaches an asymptotic power-law regime).
    \param[in]  potential  is the instance of potential;
    \param[in]  accuracy   is the parameter determining the grid spacing
    (it is proportional to accuracy^(1/4) and, of course, depends on potential derivatives);
    \returns  the grid in radius, typically containing from few dozen to a couple of hundred nodes.
*/
std::vector<double> createInterpolationGrid(const BasePotential& potential, double accuracy);


/** Interpolator class for faster evaluation of potential and related quantities --
    radius and angular momentum of a circular orbit as functions of energy,
    epicyclic frequencies as functions of radius (all in the equatorial plane).
    It is applicable to any spherical or axisymmetric potential that tends to zero at infinity,
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

private:
    const double invPhi0;       ///< 1/(value of potential at r=0), or 0 if the potential is singular
    double slopeOut, massOut, coefOut; ///< coefficients for power-law potential at large radii
    math::QuinticSpline LofE;   ///< spline-interpolated scaled function for L_circ(E)
    math::QuinticSpline RofL;   ///< spline-interpolated scaled R_circ(L)
    math::QuinticSpline PhiofR; ///< interpolated potential as a function of radius
    math::QuinticSpline RofPhi; ///< inverse function (radius as a function of potential)
    math::CubicSpline   freqNu; ///< ratio of squared epicyclic frequencies nu to Omega
};


/** Two-dimensional interpolator for peri/apocenter radii as functions of energy and angular momentum.
    This class is a further development of one-dimensional interpolator and works with spherical or
    axisymmetric potentials that tend to zero at r=infinity, and may be regular or singular at origin.
    The accuracy of peri/apocenter radii interpolation is typically at the level of 1e-9 or better.
*/
class Interpolator2d: public Interpolator {
public:
    /** Create internal interpolation tables for the given potential,
        which itself is not used afterwards.
        \note OpenMP-parallelized loop over the energy grid.
    */
    explicit Interpolator2d(const BasePotential& potential);

    /** Compute parameters of an orbit in the equatorial plane with the given energy and ang.momentum.
        \param[in]  E is the energy, which must be in the range Phi(0) <= E < 0;
        \param[in]  L is the angular momentum;
        \param[out] R1 will contain the pericenter radius,
        or NAN if the energy is outside the allowed range;
        \param[out] R2 same for the apocenter radius;
    */
    void findPlanarOrbitExtent(double E, double L, double& R1, double& R2) const;

private:
    const double invPhi0;                      ///< 1/(value of potential at r=0)
    math::QuinticSpline2d intR1, intR2;  ///< 2d interpolators for scaled peri/apocenter radii
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
