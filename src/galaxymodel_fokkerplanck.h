/** \file    galaxymodel_fokkerplanck.h
    \brief   Fokker-Planck solver for spherically-symmetric models
    \date    2016-2017
    \author  Eugene Vasiliev, Aleksey Generozov

    This module contains the `FokkerPlanckSolver` class that manages
    the self-consistent evolution of a DF f(h) driven by the relaxation,
    together with the changes in the density/potential.
*/
#pragma once
#include "potential_utils.h"
#include "smart.h"
#include <vector>

namespace galaxymodel{

/** Variants of the numerical implementation of the Fokker-Planck solver */
enum FokkerPlanckMethod {
    FP_CHANGCOOPER = 0,  ///< finite-difference scheme of Chang&Cooper(1970)
    FP_FEM1 = 1,         ///< finite-element method with piecewise-linear basis functions
    FP_FEM2 = 2,         ///< finite-element method with quadratic basis functions
    FP_FEM3 = 3          ///< finite-element method with cubic splines
};

/** Description of a single component in the Fokker-Planck model (initial density and other parameters) */
struct FokkerPlanckComponent {

    /** 1d function providing the initial density profile of this species;
        the initial DF is obtained from this density via the Eddington inversion formula. */
    math::PtrFunction initDensity;

    /** Mass of a single star of this species; determines the relaxation rate. */
    double Mstar;

    /** Capture radius in the case of a central black hole.
        If set to a positive value, this means two things.
        First, the innermost boundary hmin is assigned to the phase volume corresponding to the energy
        at which the radius of a circular orbit is twice the capture radius.
        Second, it turns on the absorbing boundary condition at hmin: the DF value is fixed to a very
        small number; the default alternative in case of captureRadius=0 is a zero-flux boundary. */
    double captureRadius;

    /** Index of the power-law scaling of capture radius with black hole mass.
        Zero means that the capture radius stays constant;
        1/3 is for tidal disruptions of stars outside horizon;
        1 is for objects (NS, BH) that are captured entirely as they cross the horizon. */
    double captureRadiusScalingExp;

    /** fraction of flux going into the loss cone that is added to the black hole mass (between 0 and 1). */
    double captureMassFraction;

    /** turns on additional source term: specifies the increase of mass per unit time. */
    double sourceRate;

    /** radius within which the injected mass is deposited. */
    double sourceRadius;

    /** set default values in the constructor */
    FokkerPlanckComponent() :
        Mstar(1.),
        captureRadius(0.),
        captureRadiusScalingExp(0.),
        captureMassFraction(1.),
        sourceRate(0.),
        sourceRadius(0.)
        {}
};

/** Parameters passed to the constructor of FokkerPlanckSolver. */
struct FokkerPlanckParams {

    /** Choice of discretization and time evolution scheme for the Fokker-Planck solver */
    FokkerPlanckMethod method;

    /** size of the logarithmic grid in phase volume. */
    size_t gridSize;

    /** lower and upper boundaries of the grid; 0 means autodetect. */
    double hmin, hmax;

    /** outer radius of grid (if provided, overrides the value of hmax) */
    double rmax;

    /** the mass of the central black hole (if present, default 0). */
    double Mbh;

    /** Coulomb logarithm that enters the expressions for diffusion coefs;
        a rule of thumb is to take  ln(Mbh / Mstar)  if there is a central black hole,
        or ln(0.1 * Mtotal / Mstar) in the opposite case. */
    double coulombLog;

    /** indicates whether the density of evolving system contributes to the potential (on by default;
        turning it off makes sense only if an external potential - central black hole - was provided). */
    bool selfGravity;

    /** indicated whether the stellar potential is updated in the course of simulation (default yes) */
    bool updatePotential;

    /** applicable when captureRadius>0; turns on a sink term at all energies (not just at hmin),
        which mimics the effect of diffusion along the angular momentum axis and leads to a steady-state
        flux into the loss cone. The magnitude of this flux is determined from the diffusion coefficient
        in angular momentum, taking into accound the appropriate regime (empty or full loss cone);
        ultimately, everything is determined by captureRadius and relaxationRate. */
    bool lossConeDrain;

    /** speed of light in N-body units: if nonzero, account for the energy loss due to GW emission */
    double speedOfLight;

    /** set default values in the constructor */
    FokkerPlanckParams() :
        method(FP_CHANGCOOPER),
        gridSize(0),         // use default grid size
        hmin(0.), hmax(0.),  // autodetect
        rmax(0.),
        Mbh(0.),
        coulombLog(0.),
        selfGravity(true),
        updatePotential(true),
        lossConeDrain(true),
        speedOfLight(0)
    {}
};


/// opaque internal implementation of the Fokker-Planck discretization scheme
class FokkerPlanckImpl;

/// opaque internal data for the FokkerPlanckSolver that changes with time
class FokkerPlanckData;


/** The class that solves the one-dimensional Fokker-Planck equation for the evolution of
    one or several spherical isotropic distribution functions (DF) f(h) driven by two-body relaxation.
    The diffusion coefficients are computed from the DF itself, and the gravitational potential
    evolves according to the changing density profile of the model, with an optional contribution
    from an external potential (e.g., a central massive object).
    Initially the DF is computed from the provided density profile via the Eddington inversion formula;
    after one or several Fokker-Planck steps the density should be recomputed by integrating the DF
    over velocity in the current potential, and then the potential itself is updated (the Poisson step),
    followed by the recomputation of diffusion coefficients.
    At all stages the DF is represented on a fixed grid in phase volume (h);
    the mapping between h and E changes after each potential update.
    The system may consist of one or two components, each with its own DF.
    The latter case corresponds to a system of light and heavy particles, evolving in the common
    potential and exchanging the energy between themselves.
    In the two-component case, one needs to specify the fraction of total mass attributed to
    the second component (its initial DF is the same as the first, differing only in normalization),
    and the ratio of stellar masses between the second and the first component.
*/
class FokkerPlanckSolver {
public:

    /** Construct the Fokker-Planck model with the given density profile.
        \param[in]  params  specifies all parameters of the solver;
        \param[in]  components  is the array of individual components (species) of the model
        (initial density profiles that determine the DF, and other related parameters).
        \throw  std::runtime_error or some other exception if the parameters are incorrect.
    */
    FokkerPlanckSolver(const FokkerPlanckParams& params,
        const std::vector<FokkerPlanckComponent>& components);

    /** evolve the DF using the Fokker-Planck equation for a time deltat,
        followed by recomputation of the potential (if required) and the relaxation coefficients;
        return the maximum relative change of f across the grid |log(f_new/f_old)|. */
    double evolve(double deltat);

    /// return the total potential
    math::PtrFunction potential() const;

    /// return the phase volume
    potential::PtrPhaseVolume phaseVolume() const;

    /// return the grid in phase volume
    std::vector<double> gridh() const;

    /// return the given DF component represented by an appropriate interpolator
    math::PtrFunction df(unsigned int indexComp) const;

    /// return the number of DF components
    unsigned int numComp() const;

    /// return the estimate of the shortest relaxation time across the grid
    double relaxationTime() const;

    /// return mass of the central black hole (BH) which contributes to the total potential; 
    /// it may change internally in the course of evolution due to accretion of stars
    double Mbh() const;

    /// assign a new mass for the central black hole (this forces the recomputation
    /// of the total potential while keeping the DF fixed)
    void setMbh(double Mbh);


    //  diagnostic quantities updated in the course of evolution:

    /// total mass of each stellar component (not including the BH)
    double Mass(unsigned int indexComp) const;

    /// stellar potential at origin (not including the BH)
    double Phi0() const;

    /// total energy of the entire system (including the BH, sum for all components)
    double Etot() const;

    /// kinetic energy (sum for all components)
    double Ekin() const;

    /// total mass added to each component due to star formation
    double sourceMass(unsigned int indexComp) const;

    /// energy associated with the added mass (sum for all components)
    double sourceEnergy() const;

    /// total change of mass of each stellar component due to capture/disruption by the BH (negative)
    double drainMass(unsigned int indexComp) const;

    /// change in total energy associated with the removed mass (sum for all components)
    double drainEnergy() const;

    /// draining time of the given species (inverse of the rate of change due to capture/disruption
    /// by the BH and the gravitational-wave energy loss), same length as gridh
    std::vector<double> drainTime(unsigned int indexComp) const;

private:
    /// opaque structure containing the initial parameters and all internal data that evolves with time
    shared_ptr<FokkerPlanckData> data;

    /// opaque internal implementation of the discretization scheme
    shared_ptr<const FokkerPlanckImpl> impl;

    /** Recompute the potential and the phase volume mapping (h <-> E)
        by first computing the density by integrating the DF over velocity,
        and then solving the Poisson equation (adding the central black hole if present). */
    void reinitPotential(double deltat);

    /** Update the advection and diffusion coefficients using the current DFs of all components */
    void reinitAdvDifCoefs();

};

}  // namespace
