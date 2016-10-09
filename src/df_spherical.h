/** \file    df_spherical.h
    \brief   Spherical isotropic distribution functions
    \author  EV
    \date    2010-2016

    This module deals with isotropic distribution funcions for spherical systems.
    They are traditionally expressed as functions of energy f(E), but it appears that
    another quantity -- the phase volume h -- is better suited as the argument of f.
    Phase volume h(E) is defined as the volume of the region {x,v} phase space where 
    Phi(x) + (1/2) v^2 <= E. In spherical symmetry, it is computed as
    \f$  h(E) = \int_0^{r_{max}(E)} 4\pi r^2 dr \int_0^{v_{max}(E,r)} 4\pi v^2 dv =
    (16/3) \pi^2 \int_0^{r_{max}(E)} r^2 [2(E-\Phi(r))]^{3/2} \f$, with E=Phi(r_max).
    The derivative dh(E)/dE = g(E) is called the density of states.
    The distribution function formulated in terms of h has the advantage that
    the mass per unit interval of h is given simply by  dM = f(h) dh, 
    whereas in terms of E it is given by the familiar expression dM = f(E) g(E) dE.
    Moreover, an adiabatic modification of potential leaves f(h) unchanged,
    although the mapping h(E), of course, depends on the potential.

    This module provides the class `PhaseVolume`, which manages the transformation
    between E and h for any potential. Any function of a single variable may serve as
    a distribution function f(h); the class `SphericalIsotropic` defines one particular
    representation in terms of log(f) being a spline-interpolated function of log(h).
    The routines `makeEddingtonDF` and `fitSphericalDF` construct such interpolators
    from a pair of density and potential models (using the Eddington inversion formula)
    or from an array of particles with known masses and values of h (using penalized
    spline log-density fit).
    The class `DiffusionCoefs` constructs interpolation tables to provide coefficients
    of drift and diffusion describing the standard two-body relaxation process.
*/

#pragma once
#include "math_spline.h"

namespace df {

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
    explicit PhaseVolume(const math::IFunction& potential);

    /// return the phase volume h for the given energy, and optionally its derivative (density of states)
    virtual void evalDeriv(const double E, double* h=NULL, double* g=NULL, double* =NULL) const;

    /// return the energy corresponding to the given phase volume h,
    /// and optionally the derivative g = dh(E) / dE if the second argument is not NULL
    double E(const double h, double* g=NULL) const;

    /// compute deltaE = E(h1) - E(h2) in a way that does not suffer from cancellation;
    /// h1 and h2 are specified through their logarithms
    double deltaE(const double logh1, const double logh2, double* g1=NULL) const;

    /// provides one derivative of h(E)
    virtual unsigned int numDerivs() const { return 1; }

    /// return the extent of the logarithmic grid used for interpolation;
    /// g(h) has a power-law asymptotic behaviour outside the grid
    double logHmin() const { return EofH.xmin(); }
    double logHmax() const { return EofH.xmax(); }
private:
    double invPhi0;             ///< 1/Phi(0), where Phi(0) is the value of potential at r=0
    math::QuinticSpline HofE;   ///< phase volume h(E), suitably scaled
    math::QuinticSpline EofH;   ///< inverse of the above
};


/** Representation of a spherical isotropic distribution function f(h) expressed in terms of
    phase volume h(E), using spline interpolation for log(f) as a function of log(h).
*/
class SphericalIsotropic: public math::IFunctionNoDeriv {
public:
    /** construct an interpolator in scaled coordinates from the provided arrays of h and f(h).
    \param[in]  gridh is the array of grid nodes h; must be positive and in order of increase.
    \param[in]  gridf is the array of DF values f(h) at grid nodes; must be positive.
    \param[in]  slopeIn  is the power-law index of the asymptotic behaviour of f(h) at small h:
    f(h) ~ h^slopeIn; default value NAN means that it will be determined by the spline interpolator,
    any non-default value should be > -1 for the integral  f(h) dh  to be finite.
    \param[in]  slopeOut is the same parameter for the asymptotic behaviour of f(h) at large h;
    default value NAN means auto-detect, and a user-provided value should be < -1.
    \return  an object that provides the interpolated f(h) for any h; values outside
    the extent of the original grid are extrapolated as power-law asymptotes.
    \throw  std::invalid_argument exception if the input data is incorrect
    (e.g., array sizes not equal), or std::runtime_error if the asymptotic behaviour leads to
    a divergent total mass (i.e., f(h) rises too steeply as h-->0 or falls too slowly as h-->inf).
    */
    SphericalIsotropic(const std::vector<double>& gridh, const std::vector<double>& gridf,
        double slopeIn=NAN, double slopeOut=NAN);

    /// return the value of distribution function f(h)
    virtual double value(const double h) const;
private:
    math::CubicSpline spl; ///< spline-interpolated log(f(h)) as a function of log(h)
};


/** Construct a spherical isotropic distribution function of phase volume h for the given pair
    of density and potential profiles (which need not be related through the Poisson equation),
    using the Eddington inversion formula.
    \param[in]  density   is any one-dimensional function returning rho(r); may be constructed
    from a spherically-symmetric `BaseDensity` object with a wrapper class `DensityWrapper`.
*/
SphericalIsotropic makeEddingtonDF(const math::IFunction& density, const math::IFunction& potential);


/** Construct a spherical isotropic distribution function f(h) from an array of particles.
    \param[in]  hvalues  is the array of values of h (phase volume) for each particle;
    \param[in]  masses   is the array of particle masses;
    \param[in]  gridSize is the number of nodes in the interpolated function
    (20-40 is a reasonable choice); the grid nodes are assigned automatically.
    \return     an instance of SphericalIsotropic function f(h).
    \throw  std::invalid_argument exception if the input data is bad (e.g., masses are negative,
    or array sizes are not equal, etc.)
*/
SphericalIsotropic fitSphericalDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize);


/** Diffusion coefficients for two-body relaxation */
class DiffusionCoefs {
public:
    /** construct the internal interpolators for diffusion coefficients.
    \param[in] phasevol  is the object providing the correspondence between E and h;
    \param[in] df  is the distribution function expressed in terms of h.
    \throw std::runtime_error in case of incorrect asymptotic behaviour of E(h) or f(h)
    (e.g., the distribution function has infinite mass, or the potential is non-monotonic, etc.),
    or any other inconsistency detected in the input data or constructed interpolators.
    */
    DiffusionCoefs(const PhaseVolume& phasevol, const math::IFunction& df);

    /** compute the orbit-averaged drift and diffusion coefficients in energy.
    The returned values should be multiplied by N^{-1} \ln\Lambda.
    \param[in]  E   is the energy; should lie in the range from Phi(0) to 0
    (otherwise the motion is unbound and orbit-averaging does not have sense);
    \param[out] DE  will contain the drift coefficient <Delta E>;
    \param[out] DEE will contain the diffusion coefficient <Delta E^2>.
    */
    void evalOrbitAvg(double E, double &DE, double &DEE) const;
    
    /** compute the local drift and diffusion coefficients in velocity,
    as defined, e.g., by eq.7.88 or L.26 in Binney&Tremaine(2008);
    the returned values should be multiplied by N^{-1} \ln\Lambda.
    \param[in]  Phi    is the potential at the given point;
    \param[in]  E      is the energy of the moving particle Phi + (1/2) v^2,
    should be >= Phi, and may be positive;
    \param[out] dvpar  will contain  <v Delta v_par>,
    where Delta v_par is the drag coefficient in the direction parallel to the particle velocity;
    \param[out] dv2par will contain  <Delta v^2_par>,
    the diffusion coefficient in the parallel component of velocity;
    \param[out] dv2per will contain  <Delta v^2_per>,
    the diffusion coefficient in the perpendicular component of velocity;
    \throw std::invalid_argument if E<Phi or Phi>=0.
    */
    void evalLocal(double Phi, double E, double &dvpar, double &dv2par, double &dv2per) const;

    /// cumulative mass as a function of h, i.e., \f$ \int_0^h f(h') dh' \f$;
    /// default value of argument (infinity) returns the total mass of the distribution function
    double cumulMass(const double h=INFINITY) const;

    /// inverse of the above: return the value of h corresponding to the given cumulative mass
    double findh(const double cumulMass) const;

    /// return the reference to the phase volume object
    const PhaseVolume& phaseVolume() const { return phasevol; }

    /// return the phase volume corresponding to the given energy (shortcut)
    double phaseVolume(double E) const { return phasevol(E); }
private:
    /// the object providing the correspondence between phase volume and energy
    const PhaseVolume phasevol;

    /// 1d interpolators for various weighted integrals of f(h), represented in log-log coordinates:
    /// \f$ \int_E^0 f(E') dE' = \int_{h(E)}^\infty f(h') / g(h') dh' \f$,
    /// \f$ \int_{\Phi(0)}^E f(E') g(E') dE' = \int_0^{h(E)} f(h') dh' \f$,
    /// \f$ \int_{\Phi(0)}^E f(E') h(E') dE' = \int_0^{h(E)} f(h') h' / g(h') dh' \f$.
    math::QuinticSpline intf, intfg, intfh;

    /// 2d interpolators for scaled velocity diffusion coefficients
    math::CubicSpline2d intv2par, intv2per;

    /// total mass associated with the DF, same as intfg(infinity)
    double totalMass;
};

std::vector<double> sampleSphericalDF(const DiffusionCoefs& model, unsigned int npoints);

}; // namespace
