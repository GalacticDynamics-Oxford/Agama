/** \file    galaxymodel_spherical.h
    \brief   Spherically-symmetric isotropic models
    \date    2010-2017
    \author  Eugene Vasiliev

    This module deals with isotropic distribution funcions for spherical systems.
    They are traditionally expressed as functions of energy f(E), but it appears that
    another quantity -- the phase volume h -- is better suited as the argument of f.
    Phase volume h(E) is defined as the volume of the region {x,v} phase space where 
    Phi(x) + (1/2) v^2 <= E. In spherical symmetry, it is computed as
    \f[
    h(E) = \int_0^{r_{max}(E)} 4\pi r^2 dr \int_0^{v_{max}(E,r)} 4\pi v^2 dv =
    (16/3) \pi^2 \int_0^{r_{max}(E)} r^2 [2(E-\Phi(r))]^{3/2} ,
    \f]
    with E=Phi(r_max).
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
    The routines `df::createSphericalIsotropicDF` and `df::fitSphericalIsotropicDF`,
    defined in df_spherical.h, construct such interpolators from a pair of density and
    potential models (using the Eddington inversion formula) or from an array of particles
    with given masses and values of h (using penalized spline log-density fit).
    The classes `SphericalIsotropicModel` and `SphericalIsotropicModelLocal`, defined in
    this file, construct interpolation tables to provide coefficients of drift and diffusion
    describing the two-body relaxation process (the former - for the 1d orbit-averaged
    Fokker-Planck equation, the latter - for position-dependent relaxation coefficients
    as functions of r and v).
    These models are used in the galaxymodel_fokkerplanck module (the FP code PhaseFlow)
    and in the raga_*** modules (the Monte Carlo stellar-dynamical code Raga).
*/
#pragma once
#include "particles_base.h"
#include "potential_utils.h"
#include <string>

namespace galaxymodel{

/** A combination of a spherical isotropic DF f(h) with a spherical potential.
    The DF is given by an arbitrary function (typically a LogLogSpline) of one argument
    (the phase volume h), and the potential is implicitly given by the mapping between Phi and h,
    represented by the `potential::PhaseVolume` class.
    This class constructs interpolators for various dynamical quantities (I0, K_g, K_h)
    that enter the expressions for two-body relaxation coefficients.
    This class also provides the interpolated value of f(h) through the IFunction interface
    (it does not store the original function provided in the constructor, but computes f(h)
    from one of its interpolators).
*/
class SphericalIsotropicModel: public math::IFunction{

    /// 1d interpolators for various weighted integrals of f(h), represented in log-log coordinates:
    math::LogLogSpline
    intf,  ///< \f$ I_0 = \int_E^0 f(E') dE' = \int_{h(E)}^\infty f(h') / g(h') dh' \f$
    intfg, ///< \f$ K_g = \int_{\Phi(0)}^E f(E') g(E') dE' = \int_0^{h(E)} f(h') dh' \f$
    intfh, ///< \f$ K_h = \int_{\Phi(0)}^E f(E') h(E') dE' = \int_0^{h(E)} f(h') h' / g(h') dh' \f$
    intfE; ///< \f$ K_E = \int_{\Phi(0)}^E f(E') g(E') E' dE' = \int_0^{h(E)} f(h') E(h') dh' \f$ 

    double totalMass;    ///< total mass associated with the DF, same as cumulMass(INFINITY)
    double htransition;  ///< value of h separating two regimes of computing f(h) (using intf or intfg)
public:
    /** Construct the model for the given h(E) and f(h).
        \param[in]  phasevol  is the instance of phase volume h(E); a copy of it is stored internally.
        \param[in]  df  is the instance of distribution function f(h).
        \param[in]  gridh  (optional) if provided, will use this grid in phase volume for interpolation
        (this makes sense if f(h) is represented by a spline interpolator itself, to ensure
        1:1 correspondence between grids); otherwise will construct a suitable grid automatically.
        \throw  std::runtime_error  in case of various problems detected with the data
        (e.g., incorrect asymptotic behaviour).
    */
    SphericalIsotropicModel(
        const potential::PhaseVolume& phasevol,
        const math::IFunction& df,
        const std::vector<double>& gridh = std::vector<double>());

    /// return the value of DF f(h), and optionally its first derivative,
    /// obtained by differentiating one of the interpolating splines
    virtual void evalDeriv(const double h, double* f=NULL, double* dfdh=NULL, double* =NULL) const;

    /// may provide up to one derivative of f(h)
    virtual unsigned int numDerivs() const { return 1; }

    /// return \f$ I_0 = \int_h^\infty f(h') / g(h') dh' \f$
    double I0(const double h) const;

    /// cumulative mass as a function of h, i.e., \f$ M(h) = \int_0^h f(h') dh' \f$;
    /// default value of argument (infinity) returns the total mass of the distribution function
    double cumulMass(const double h=INFINITY) const;

    /// cumulative kinetic energy as a function of h:
    /// \f$ Ekin(h) = 3/2 \int_0^h f(h') h' / g(h') dh' \f$
    double cumulEkin(const double h=INFINITY) const;

    /// cumulative total energy of the model as a function of h:
    /// \f$ Etot(h) = \int_0^h f(h') E(h') dh' \f$
    double cumulEtotal(const double h=INFINITY) const;

    /// the object providing the correspondence between phase volume h and energy E
    const potential::PhaseVolume phasevol;
};


/** An augmented spherical model providing various quantities as functions of position.
    This class is a further extension of SphericalIsotropicModel, providing two-dimensional
    interpolators for local (position-dependent) quantities (advection and diffusion coefficients
    in velocity, density, velocity dispersion, etc.)
    As a by-product, the same 2d interpolators can be used to compute the density and draw samples
    from the velocity distribution at any position specified by the potential.
    Hence this class performs (some of) the same tasks for spherical isotropic DFs
    as the more general routines in galaxymodel_base.h working with the GalaxyModel structure.
*/
class SphericalIsotropicModelLocal: public SphericalIsotropicModel {

    /// 2d interpolators for scaled integrals over distribution function
    math::CubicSpline2d intJ1, intJ3;

    /// perform actual initialization of interpolators
    void init(const math::IFunction& df, const std::vector<double>& gridh);

public:

    /** Construct the internal interpolators for diffusion coefficients.
        \param[in]  phasevol  is the object providing the correspondence between E and h.
        \param[in]  df  is the distribution function expressed in terms of h.
        \param[in]  gridh  (optional) grid in phase volume for the interpolators
        (if not provided, will construct a suitable one automatically).
        \throw std::runtime_error in case of incorrect asymptotic behaviour of E(h) or f(h)
        (e.g., the distribution function has infinite mass, or the potential is non-monotonic, etc.),
        or any other inconsistency detected in the input data or constructed interpolators.
    */
    SphericalIsotropicModelLocal(
        const potential::PhaseVolume& phasevol,
        const math::IFunction& df,
        const std::vector<double>& gridh = std::vector<double>()) 
    :
        SphericalIsotropicModel(phasevol, df, gridh) { init(df, gridh); }

    /** Construct the interpolators for diffusion coefficients from a SphericalIsotropicModel.
        \param[in]  model  is an instance of SphericalIsotropicModel,
        which provides both the phase volume and the expression for f(h).
        \param[in]  gridh  (optional) grid in phase volume for the interpolators
        (if not provided, will construct a suitable one automatically).
        \throw std::runtime_error in case of inconsistencies in the input data.
    */
    SphericalIsotropicModelLocal(
        const SphericalIsotropicModel& model,
        const std::vector<double>& gridh = std::vector<double>())
    :
        SphericalIsotropicModel(model) { init(*this, gridh); }

    /** Compute the local drift and diffusion coefficients in velocity,
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

    /** compute the density as a function of potential */
    double density(double Phi) const;

    /** compute the velocity dispersion (sigma, not sigma^2) as a function of potential */
    double velDisp(double Phi) const;

    /** draw a sample from the velocity distribution at the radius corresponding to the given potential */
    double sampleVelocity(double Phi) const;
};


/** Compute the orbit-averaged drift and diffusion coefficients in energy.
    The returned values should be multiplied by  \f$ N^{-1} \ln\Lambda \f$.
    \param[in]  model  is the instance of SphericalIsotropicModel that provides the necessary quantities
    \param[in]  E   is the energy; should lie in the range from Phi(0) to 0
    (otherwise the motion is unbound and orbit-averaging does not have sense);
    \param[out] DeltaE  will contain the drift coefficient <Delta E>;
    \param[out] DeltaE2 will contain the diffusion coefficient <Delta E^2>.
*/
void difCoefEnergy(const SphericalIsotropicModel& model, double E, double &DeltaE, double &DeltaE2);

/** Compute the diffusion coefficient entering the expression for the loss-cone flux.
    \param[in]  model   is the spherical model providing the DF and the phase volume.
    \param[in]  pot  represents the spherically-symmetric gravitational potential.
    \param[in]  E    is the energy for which the coefficient should be computed.
    \return  the limiting value D of the orbit-averaged diffusion coefficient in
    scaled squared angular momentum for nearly-radial orbits. It is defined as
    \f$  D = (1/2) \lim_{R \to 0}  \langle \Delta R^2 \rangle  / R  \f$,  where
    \f$  R = L^2 / L_{circ}^2(E)  \f$  is the normalized squared angular momentum, and
    the averaging is performed for a radial orbit with the given energy.
    The expression for D is [Merritt 2013, eq. 6.31]
    \f$  D = 8\pi^2 g^{-1}(E)  \int_0^{r_{max}(E)}  dr  \langle \Delta v_\bot^2 \rangle  r^2 / v  \f$.
    The returned value should be multiplied by  \f$ N^{-1} \ln\Lambda \f$.
*/
double difCoefLosscone(const SphericalIsotropicModel& model, const math::IFunction& pot, double E);


/** Generate N-body samples of the spherical isotropic distribution function in the given potential.
    This is the analog of a more general eponimous function that samples from an arbitrary DF
    in an arbitrary (axisymmetric) potential.
    \param[in]  pot is the spherical potential;
    \param[in]  df  is the isotropic distribution function model f(h);
    \param[in]  numPoints  is the required number of samples;
    \returns    a new array of particles (position/velocity/mass)
    sampled from the distribution function;
*/
particles::ParticleArraySph samplePosVel(
    const math::IFunction& pot,
    const math::IFunction& df,
    const unsigned int numPoints);


/** Compute the density generated by a spherical distribution function in the given potential.
    \param[in]  df  is the distribution function f(h);
    \param[in]  pv  is the instance of phase volume class that provides mapping from E to h;
    \param[in]  gridPhi is the array of Phi(r) - the values potential at the radial points
    where the density should be computed (it is cheaper and more accurate to do it for many
    values of radius at once);
    \param[out]  gridVelDisp (optional) is the array to be filled with the values of velocity
    dispersion at the same radii as the returned density values; if NULL it is not computed.
    \return  the density  \f$ \rho(\Phi) = \int_{\Phi}^0 dE  f(h(E))  4\pi \sqrt{ 2 [E - \Phi] }  \f$
    at the nodes of the radial grid.
*/
std::vector<double> computeDensity(
    const math::IFunction& df,
    const potential::PhaseVolume& pv,
    const std::vector<double> &gridPhi,
    std::vector<double> *gridVelDisp = NULL);


/** Compute the projected density and line-of-sight velocity dispersion at nodes of a grid
    in projected radius R (all at once - more efficient than computing it at each radius separately).
    \param[in]  dens  is the function that computes the density at the given radius r;
    \param[in]  velDisp  is the function that computes the 1d velocity dispersion at the radius r;
    \param[in]  gridR  is the grid in projected radii where the output should be computed;
    \param[out] gridProjDensity  will be filled with values of projected density at the nodes of gridR;
    \param[out] gridProjVelDisp  same for line-of-sight velocity dispersion.
*/
void computeProjectedDensity(
    const math::IFunction& dens,
    const math::IFunction& velDisp,
    const std::vector<double> &gridR,
    std::vector<double>& gridProjDensity,
    std::vector<double>& gridProjVelDisp);


/** Write a text file with a table of several variables as functions of radius or energy.
    These variables are extracted from a spherical model described by a combination of
    potential, phase volume and distribution function.
    The following variables are printed:
    - radius                 r
    - enclosed mass          M(r),   obtained by integrating the density profile corresponding to the DF
    - potential              Phi(r), serves as the energy E in arguments of other variables
    - density                rho(r), obtained by integrating the DF, not differentiating the potential
    - distribution function  f(E),   or rather f(h(E))
    - energy-enclosed mass   M(E),   i.e. mass of particles with energies less than E
    - phase volume           h(E)
    - average radial period  T(E) = g(E) / (4 pi^2 Lcirc(E)),  where g(E) = dh(E)/dE
    - circular orbit radius  rcirc(E)
    - its angular momentum   Lcirc(E) = rcirc * sqrt(dPhi/dr (rcirc))
    - velocity dispersion    sigma(r) - characteristic 1d velocity at the radius r
    - line-of-sight vel.disp sigma_los(R), where R is the projected radius equal to r
    - surface density        Sigma(R), or projected density ( = \int_R^infinity rho(R,z) dz )
    - diffusion coefficient  < Delta E^2 >
    - drift coefficient      < Delta E >
    - phase-space mass flux  F(h)
    - loss-cone dif.coef.    D_RR/R at R=0  (only if there is a central black hole)

    \param[in]  fileName  is the output file name.
    \param[in]  header  is the text written in the first line of the file
    \param[in]  model  is the SphericalIsotropicModel instance that provides both f(E) and h(E).
    \param[in]  pot  is the potential Phi(r) - it is not contained in the spherical model
    (even though it may be derived from the phase volume), so should be provided separately.
    \param[in]  gridh  (optional)  the grid in phase volume (h) for the output table;
    if not provided, a suitable grid that encompasses the region of significant variation
    of f(h) is constructed automatically.
*/
void writeSphericalIsotropicModel(
    const std::string& fileName,
    const std::string& header,
    const SphericalIsotropicModel& model,
    const math::IFunction& pot,
    const std::vector<double>& gridh = std::vector<double>());

}  // namespace
