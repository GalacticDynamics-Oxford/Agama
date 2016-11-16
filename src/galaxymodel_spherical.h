/** \file    galaxymodel_spherical.h
    \brief   Spherically-symmetric models
    \date    2010-2016
    \author  Eugene Vasiliev

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

    The class `potential::PhaseVolume`, defined in potential_utils.h, establishes
    the bi-directional correspondence between E and h for any potential.
    Any function of a single variable may serve as a distribution function f(h);
    the class `math::LogLogSpline` can be used for representing a log-scaled interpolated f.
    The routines `makeEddingtonDF` and `fitSphericalDF` construct such interpolators
    from a pair of density and potential models (using the Eddington inversion formula)
    or from an array of particles with known masses and values of h (using penalized
    spline log-density fit).
    The classes `SphericalModel` and `DiffusionCoefs` construct interpolation tables
    to provide coefficients of drift and diffusion describing the standard two-body
    relaxation process (the former - for the 1d orbit-averaged Fokker-Planck equation,
    the latter - for position-dependent relaxation coefficients as functions of r and v).
    The class `FokkerPlanckSolver` manages the self-consistent evolution of a DF f(h)
    driven by the relaxation, together with the changes in the density/potential.
*/
#pragma once
#include "potential_utils.h"
#include "particles_base.h"
#include "smart.h"

namespace galaxymodel{

/** A combination of a spherical isotropic DF f(h) with a spherical potential.
    The DF is given by an arbitrary function (typically a LogLogSpline) of one argument
    (the phase volume h), and the potential is implicitly given by the mapping between Phi and h,
    represented by the `potential::PhaseVolume` class.
    This class constructs interpolators for various dynamical quantities (I0, K_g, K_h)
    that enter the expressions for two-body relaxation coefficients.
*/
class SphericalModel: math::IFunctionNoDeriv{

    /// 1d interpolators for various weighted integrals of f(h), represented in log-log coordinates:
    math::QuinticSpline
    intf,  ///< \f$ I_0 = \int_E^0 f(E') dE' = \int_{h(E)}^\infty f(h') / g(h') dh' \f$
    intfg, ///< \f$ K_g = \int_{\Phi(0)}^E f(E') g(E') dE' = \int_0^{h(E)} f(h') dh' \f$
    intfh; ///< \f$ K_h = \int_{\Phi(0)}^E f(E') h(E') dE' = \int_0^{h(E)} f(h') h' / g(h') dh' \f$

    double totalMass;    ///< total mass associated with the DF, same as cumulMass(INFINITY)
    double totalEnergy;  ///< total energy of the model: \f$ \int_0^\infty f(h) E(h) dh \f$.
public:
    SphericalModel(const potential::PhaseVolume& phasevol, const math::IFunction& df);

    /// return the value of DF f(h), obtained by differentiating one of the interpolating splines
    virtual double value(const double h) const;

    /// return \f$ I_0 = \int_h^\infty f(h') / g(h') dh' \f$
    double I0(const double logh) const;

    /// cumulative mass as a function of log(h), i.e., \f$ M(h) = \int_0^h f(h') dh' \f$;
    /// default value of argument (infinity) returns the total mass of the distribution function
    double cumulMass(const double logh=INFINITY) const;

    /// cumulative kinetic energy as a function of log(h):
    /// \f$ Ekin(h) = 3/2 \int_0^h f(h') h' / g(h') dh' \f$
    double cumulEkin(const double logh=INFINITY) const;

    /// total energy of the model: \f$ Etot = \int_0^\infty f(h) E(h) dh \f$.
    double cumulEtotal() const { return totalEnergy; }

    /// the object providing the correspondence between phase volume h and energy E
    const potential::PhaseVolume phasevol;
};


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
    DiffusionCoefs(const potential::PhaseVolume& phasevol, const math::IFunction& df);

    /** compute the orbit-averaged drift and diffusion coefficients in energy.
    The returned values should be multiplied by  \f$ N^{-1} \ln\Lambda \f$.
    \param[in]  E   is the energy; should lie in the range from Phi(0) to 0
    (otherwise the motion is unbound and orbit-averaging does not have sense);
    \param[out] DeltaE  will contain the drift coefficient <Delta E>;
    \param[out] DeltaE2 will contain the diffusion coefficient <Delta E^2>.
    */
    void evalOrbitAvg(double E, double &DeltaE, double &DeltaE2) const;

    /** compute the local drift and diffusion coefficients in velocity,
    as defined, e.g., by eq.7.88 or L.26 in Binney&Tremaine(2008);
    the returned values should be multiplied by  \f$ N^{-1} \ln\Lambda \f$.
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

    const SphericalModel model;
private:

    /// 2d interpolators for scaled velocity diffusion coefficients
    math::CubicSpline2d intv2par, intv2per;
};


/** The class that solves the one-dimensional Fokker-Planck equation for the evolution of
    a spherical isotropic distribution function (DF) f(h) driven by two-body relaxation.
    The diffusion coefficients are computed from the DF itself, and the gravitational potential
    evolves according to the changing density profile of the model, with an optional contribution
    from an external potential (e.g., a central massive object).
    Initially the DF is computed from the provided density profile via the Eddington inversion formula;
    after one or several Fokker-Planck steps the density should be recomputed by integrating the DF
    over velocity in the current potential, and then the potential itself is updated (the Poisson step),
    followed by the recomputation of diffusion coefficients.
    At all stages the DF is represented on a fixed grid in phase volume (h);
    the mapping between h and E changes after each potential update.
*/
class FokkerPlanckSolver {
public:
    potential::PtrPotential extPot;     ///< external potential (if present)
    potential::Interpolator totalPot;   ///< total potential
    potential::PhaseVolume  phasevol;   ///< mapping between energy and phase volume
    std::vector<double> gridh;          ///< grid in h (phase volume), stays fixed throughout the evolution
    std::vector<double> gridf;          ///< values of distribution function at grid nodes
    std::vector<double> diag, above, below; ///< coefficients of the tridiagonal system solved at each step
public:
    /** Construct the Fokker-Planck model with the given density profile,
        optionally embedded in an external potential.
        \param[in]  initDensity  is a function that provides the density profile;
        the initial distribution function will be constructed using the Eddington inversion formula.
        \param[in]  externalPotential  (optional) is an additional spherically-symmetric potential
        (e.g., a central black hole), which will be summed with the self-consistently generated
        potential of the evolving model; may be omitted.
    */
    FokkerPlanckSolver(const math::IFunction& initDensity,
        const potential::PtrPotential& externalPotential = potential::PtrPotential(),
        const std::vector<double>& gridh = std::vector<double>());

    /** Recompute the potential and the phase volume mapping (h <-> E) by integrating the DF over velocity,
        and solving the Poisson equation (adding the external potential if present).
    */
    void reinitPotential();

    /** Recompute the diffusion coefficients following the update of the potential */
    void reinitDifCoefs();

    /** Perform one Fokker-Planck step with the timestep dt (recompute the values of f on the grid);
        return the maximum relative change of f across the grid |log(f_new/f_old)|. */
    double doStep(double dt);

    const potential::Interpolator& getPotential()   const { return totalPot; }
    const potential::PhaseVolume&  getPhaseVolume() const { return phasevol; }
    const std::vector<double>& getGridH() const { return gridh; }
    const std::vector<double>& getGridF() const { return gridf; }

    /// diagnostic quantities: total mass, stellar potential at origin, total energy and kinetic energy
    double Mass, Phi0, Etot, Ekin;

};


/** Construct a spherical isotropic distribution function of phase volume h for the given pair
    of density and potential profiles (which need not be related through the Poisson equation),
    using the Eddington inversion formula.
    \param[in]  density   is any one-dimensional function returning rho(r); may be constructed
    from a spherically-symmetric `BaseDensity` object using the wrapper class `DensityWrapper`.
    \param[in]  potential  is any one-dimensional function representing the spherically-symmetric
    potential (may be constructed using the `PotentialWrapper` class).
    \param[in,out]  gridh  is the array of phase volumes at which the DF is defined;
    if this array is empty on input, a plausible grid will be created automatically,
    but in any case it may later be modified by this routine, to eliminate negative DF values.
    \param[out]     gridf  is the array of DF values at corresponding nodes of gridh,
    guaranteed to contain only positive values (thus a LogLogSpline may be constructed from it).
    \throw  std::runtime_error if no valid DF can be constructed.
*/
void makeEddingtonDF(const math::IFunction& density, const math::IFunction& potential,
    /*output*/ std::vector<double>& gridh, std::vector<double>& gridf);

/** Construct a spherical isotropic distribution function using the Eddington formula.
    This is a convenience overloaded routine that first computes the values of f at a grid in h,
    using the previous function with the same name, and then creates an instance of LogLogSpline
    interpolator to represent log(f) as a function of log(h) in terms of a cubic spline.
    \param[in]  density    is any one-dimensional function returning rho(r);
    \param[in]  potential  is any one-dimensional function representing the potential;
    \return  an instance of math::LogLogSpline interpolator for the distribution function.
*/
inline math::LogLogSpline makeEddingtonDF(
    const math::IFunction& density, const math::IFunction& potential)
{
    std::vector<double> gridh, gridf;
    makeEddingtonDF(density, potential, gridh, gridf);
    return math::LogLogSpline(gridh, gridf);
}


/** Construct a spherical isotropic distribution function f(h) from an array of particles.
    \param[in]  hvalues  is the array of values of h (phase volume) for each particle;
    \param[in]  masses   is the array of particle masses;
    \param[in]  gridSize is the number of nodes in the interpolated function
    (20-40 is a reasonable choice); the grid nodes are assigned automatically.
    \return     an instance of SphericalIsotropic function f(h).
    \throw  std::invalid_argument exception if the input data is bad (e.g., masses are negative,
    or array sizes are not equal, etc.)
*/
math::LogLogSpline fitSphericalDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize);


/** Construct an interpolated spherical density profile from two arrays -- radii and
    enclosed mass M(<r).
    First a suitably scaled interpolator is constructed for M(r);
    if it is found to have a finite limiting value at r --> infinity, the asymptotic power-law
    behaviour of density at large radii will be correctly represented.
    Then the density at each point of the radial grid is computed from the derivative of
    this interpolator. The returned array may be used to construct a LogLogSpline interpolator
    or a DensitySphericalHarmonic object (obviously, with only one harmonic).
    \param[in]  gridr  is the grid in radius (must have positive values sorted in order of increase);
    typically the radial grid should be exponentially spaced with r[i+1]/r[i] ~ 1.2 - 2.
    \param[in]  gridm  is the array of enclosed mass at each radius (must be positive and monotonic);
    \return  an array of density values at the given radii.
    \throw   std::invalid_argument if the input arrays were incorrect
    (incompatible sizes, non-monotinic or negative values), or
    std::runtime_error if the interpolator failed to produce a positive-definite density.
*/
std::vector<double> densityFromCumulativeMass(
    const std::vector<double>& gridr, const std::vector<double>& gridm);


/** Generate N-body samples of the spherical isotropic distribution function in the given potential.
    This is the analog of a more general eponimous function that samples from an arbitrary DF
    in an arbitrary (axisymmetric) potential.
    \param[in]  pot is the spherical potential;
    \param[in]  df  is the isotropic distribution function model f(h);
    \param[in]  numPoints  is the required number of samples;
    \returns    a new array of particles (position/velocity/mass)
    sampled from the distribution function;
*/
particles::ParticleArraySph generatePosVelSamples(
    const math::IFunction& pot, const math::IFunction& df, const unsigned int numPoints);


/** Compute the density generated by a spherical distribution function in the given potential.
    \param[in]  df  is the distribution function f(h);
    \param[in]  pv  is the instance of phase volume class that provides mapping from E to h;
    \param[in]  gridPhi is the array of Phi(r) - the values potential at the radial points
    where the density should be computed (it is cheaper and more accurate to do it for many
    values of radius at once);
    \return  the density  \f$ \rho(\Phi) = \int_{\Phi}^0 dE  f(h(E))  4\pi \sqrt{ 2 [E - Phi] }  \f$
    at the nodes of the radial grid.
*/
std::vector<double> computeDensity(const math::IFunction& df, const potential::PhaseVolume& pv,
    const std::vector<double> &gridPhi);

}  // namespace