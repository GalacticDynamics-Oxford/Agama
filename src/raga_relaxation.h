/** \file    raga_relaxation.h
    \brief   Simulate the effect of two-body relaxation (part of the Raga code)
    \author  Eugene Vasiliev
    \date    2013-2020

    This module simulates the effect of two-body (collisional) relaxation,
    i.e., is the main component of the Monte Carlo approach to stellar dynamics.

    We use the Spitzer's approach to model the two-body relaxation:
    as independent and uncorrelated velocity perturbations applied to each particle
    while it moves in the global smooth potential. These perturbations are described
    by the classical expressions for the drift and diffusion coefficients for velocity
    (dating back to Chandrasekhar's work). The functional form of these coefficients
    is determined by the distribution function of 'background' stars (scatterers),
    which in the present implementation are identified with the actual particles
    in the simulation. On the other hand, the amplitude of the coefficients is a free
    parameter, which is related to the number of stars in the physical system that is
    being modelled. For instance, we may represent a nuclear star cluster having 10^8
    stars with an N-body system having only 10^6 particles, but set the level of
    relaxation equal to that of the target physical system, and recover the correct
    evolution timescale. Of course, this does not take into account other numerical
    perturbations, mainly arising due to discreteness noise during the recomputation
    of potential and distribution function at the end of each episode; this introduces
    a practical limit on the lowest possible relaxation level that can be simulated
    reliably, in fact roughly two orders of magnitude below the inherent two-body
    relaxation level of a conventional N-body system with the same number of particles
    (hence the choice of the parameters in the above example). On the other hand,
    in physical systems with that many stars the collisional relaxation is typically
    unimportant, so that the Monte Carlo method bridges the gap between collisional
    and nearly collisionless systems in a reliable way.

    The computation of diffusion coefficients starts from the construction of
    the distribution function (DF) of stars, which is assumed to be isotropic
    (depends only on energy, or equivalently on the phase volume h, which is
    a monotonic function of energy in any spherically-symmetric potential).
    Of course, the actual N-body system may be non-spherical and non-isotropic;
    the motivation for this approximation, apart from computational simplification,
    is that the diffusion coefficients are integrals over the DF, so that details
    shouldn't matter.

    Thus within this module, the stellar system is represented in two complementary
    ways: with the actual non-spherical potential, and its sphericalized version
    that determines the correspondence between energy and phase volume, and ultimately
    the diffusion coefficients.
    The runtime function performs two independent tasks:
    - adds perturbations to the particle velocity after each timestep, which depend
    on the potential and kinetic energy through the model for diffusion coefficients.
    - collects samples of particle energy (or, rather, phase volume h(E)) at regular
    intervals of time during the episode, which are used to recompute the DF and
    the diffusion model at the end of episode. This is analogous to the way that
    the trajectory samples are collected for recomputing the density and hence
    the potential (in raga_potential.cpp), and for the same reason of reducing
    discreteness noise, we may take more than one sample per particle during each
    episode. Note that the potential and the DF are updated independently --
    in fact, either of them may be turned off, but if the relaxation rate is non-zero,
    then the DF must be updated to ensure the (approximate) energy conservation.

    The collected samples are used to recompute the DF f(h) at the end of each episode,
    and at the beginning of the first one (in this case only the instantaneous values
    of particle energies in the initial snapshot are used).
    Note that particles with positive energy cannot contribute to the DF
    by construction (they have infinite phase volume), but in practice their number
    shouldn't be significant.
    The relaxation model (radial dependence of various quantities -- spherical
    potential, density, DF, etc.) may be written to a text file at regular intervals
    of time (an integer multiple of episode length).
*/
#pragma once
#include "raga_base.h"
#include "particles_base.h"
#include "potential_analytic.h"

// forward declaration (definitions are in galaxymodel_spherical.h)
namespace galaxymodel {
class SphericalIsotropicModelLocal;
typedef shared_ptr<SphericalIsotropicModelLocal> PtrSphericalIsotropicModelLocal;
}

namespace raga {

/** The runtime function that applies velocity perturbations to particle orbits and
    collects samples of particle energies (or, rather, phase volumes) during the episode
*/
class RuntimeRelaxation: public orbit::BaseRuntimeFnc {
public:
    RuntimeRelaxation(
        orbit::BaseOrbitIntegrator& orbint,
        const potential::BasePotential& _potentialSph,
        const galaxymodel::SphericalIsotropicModelLocal& _relaxationModel,
        double _coulombLog,
        double _mass,
        double _outputTimestep,
        const std::vector<double>::iterator& _outputFirst,
        const std::vector<double>::iterator& _outputLast,
        unsigned int _seed)
    :
        BaseRuntimeFnc(orbint),
        potentialSph(_potentialSph),
        relaxationModel(_relaxationModel),
        coulombLog(_coulombLog),
        mass(_mass),
        outputTimestep(_outputTimestep),
        outputFirst(_outputFirst),
        outputLast (_outputLast),
        outputIter (_outputFirst),
        seed(_seed)
    {}
    virtual bool processTimestep(double tbegin, double timestep);
private:
    /** The spherically-symmetric approximation to the actual total potential,
        including the central black hole (in case of a binary BH, it is represented by
        a single point mass at origin).
        It is used to convert the actual particle position/velocity to the energy 
        (in this approximate potential); both the potential and the kinetic energy
        enter the expressions for diffusion coefficients.
        Furthermore, at regular intervals of time, the energy is converted to the phase
        volume, and its samples are stored in the external output array.
    */
    const potential::BasePotential& potentialSph;

    /** The diffusion model that provides the expression for the local (position-dependent)
        drift and diffusion coefficients for the components of velocity parallel and
        perpendicular to the instantaneous direction of motion.
        Moreover, this model contains the PhaseVolume object that transforms the energy to
        phase volume.
    */
    const galaxymodel::SphericalIsotropicModelLocal& relaxationModel;

    /** Coulomb logarithm \ln\Lambda ~ \ln N_\star */
    const double coulombLog;

    /** Stellar mass associated with this particle (the strength of dynamical friction is
        proportional to this mass, while relaxation rate is independent of it) */
    const double mass;

    /** Interval between storing the samples of phase volume taken from this orbit during
        the current episode (counting from the beginning of the episode) */
    const double outputTimestep;

    /** location in the external array for storing the samples from this orbit
        (the two iterators point to the first and the last array elements reserved for this orbit) */
    const std::vector<double>::const_iterator outputFirst, outputLast;

    /** pointer to the current array element where the upcoming sample will be placed */
    std::vector<double>::iterator outputIter;

    /** seed for the orbit-local pseudo-random number generator */
    unsigned int seed;
};

/** Fixed global parameters of this task */
struct ParamsRelaxation {
    /// number of subsamples collected for each orbit during an episode
    unsigned int numSamplesPerEpisode;

    /// Coulomb logarithm (ln Lambda), if set to 0 then this task is not activated
    double coulombLog;

    /// size of the grid in energy space for constructing the spherical model
    unsigned int gridSizeDF;

    /// [base] file name for outputting the properties of the spherical model;
    /// the current simulation time is appended to the file name
    std::string outputFilename;

    /// interval between writing out a text file with the spherical model
    double outputInterval;

    /// optional header written in the output file
    std::string header;

    /// set defaults
    ParamsRelaxation() :
        numSamplesPerEpisode(1), coulombLog(0), gridSizeDF(25), outputInterval(0)
    {}
};

/** The driver class for simulating two-body relaxation */
class RagaTaskRelaxation: public BaseRagaTask {
public:
    RagaTaskRelaxation(
        const ParamsRelaxation& params,
        const particles::ParticleArrayAux& particles,
        const potential::PtrPotential& ptrPot,
        const potential::KeplerBinaryParams& bh);
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex);
    virtual void startEpisode(double timeStart, double episodeLength);
    virtual void finishEpisode();
    virtual const char* name() const { return "Relaxation     "; }

private:
    /** fixed parameters of this task  */
    const ParamsRelaxation params;

    /** read-only reference to the list of particles
        (only their current masses are used in constructing the DF)  */
    const particles::ParticleArrayAux& particles;

    /** read-only pointer to the stellar potential of the system
        which is used to construct a sphericalized potential and its associated
        mapping between phase volume and energy; together with the DF, they are used
        to compute the relaxation coefficients at the beginning of the simulation
        and recompute them after each episode
    */
    const potential::PtrPotential& ptrPot;

    /** read-only reference to the parameters (essentially, only the mass)
        of the central black hole(s) which also contribute to the sphericalized potential
        (if this is a binary black hole, it is represented by a single point mass
        at origin in the sphericalized potential)
    */
    const potential::KeplerBinaryParams& bh;

    /** last time when the spherical model was written into a text file  */
    double prevOutputTime;

    /** beginning and duration of the current episode  */
    double episodeStart, episodeLength;

    /** internally constructed sphericalized total potential
        (it is essentially the l=0 term in the Multipole representatin of the actual
        stellar potential, plus the Newtonian contribution of the central black hole
    */
    potential::PtrPotential ptrPotSph;

    /** internally constructed relaxation model for the spherical potential,
        which provides the diffusion coefficients for perturbing the particle velocity
        during orbit integration
    */
    galaxymodel::PtrSphericalIsotropicModelLocal ptrRelaxationModel;

    /** place for storing the phase volume h(E) (essentially a function of energy)
        sampled from particle trajectories during the episode
        (each particle is allocated a block of numSamplesPerEpisode elements);
        these samples, together with particle masses divided by numSamplesPerEpisode,
        are used to re-construct the DF f(h) at the end of an episode
    */
    std::vector<double> particle_h;
};

}  // namespace raga