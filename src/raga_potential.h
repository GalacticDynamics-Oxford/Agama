/** \file    raga_potential.h
    \brief   Recomputation of gravitational potential (part of the Raga code)
    \author  Eugene Vasiliev
    \date    2013-2017

    This module handles the evolution of stellar gravitational potential during the simulation.

    The total potential is composed of the smooth (mean-field) potential of the stellar system,
    plus optionally the potential of the central massive black hole (or a BH binary).
    The stellar potential is created by particles, and is assumed to evolve only slowly
    (on a timescale much longer than the duration of a single episode), thus it remains fixed
    throughout the episode (hence all particles move in a common global potential independently,
    and do not directly interact with each other).
    This potential is represented by a spherical-harmonic expansion, implemented in the Multipole
    class of the Agama library. One may adjust the resolution of the radial grid and the angular
    order of this expansion, and in particular force a certain degree of symmetry.
    At the beginning of the simulation, the potential is computed from the initial positions
    of all particles, which is performed by the RagaCore class in the main module.
    The present module serves the purpose of recomputing the potential throughout the simulation
    (at the end of each episode); this may be turned off in the global settings, in which case
    the instance of RagaTaskPotential is not created).
    To reduce the impact of discreteness noise, the potential is constructed not only from
    the positions of particles at the end of the episode, but taking into account their
    entire trajectories during the episode (in other words, taking more than one sub-sample
    from each orbit, although one is also possible). The RuntimePotential class is responsible
    for storing these samples at regular intervals during the orbit integration in the global
    array variable which belongs to the RagaTaskPotential class. The latter class performs
    re-computation of the potential at the end of the episode, and optionally stores
    the potential expansion coefficients into a text file at pre-defined intervals of time
    (should be an integer number of episodes).
*/
#pragma once
#include "raga_base.h"
#include "particles_base.h"

namespace raga {

/// Array of particle coordinates in the cylindrical system and their masses
typedef particles::ParticleArray<coord::PosCyl>::ArrayType ParticleArrayType;

/** The runtime function that collects samples from each particle's trajectory during the episode
    and stores them in the external array in the pre-allocated block */
class RuntimePotential: public orbit::BaseRuntimeFnc {
public:
    RuntimePotential(
        orbit::BaseOrbitIntegrator& orbint,
        double _outputTimestep,
        const ParticleArrayType::iterator& _outputFirst,
        const ParticleArrayType::iterator& _outputLast)
    :
        BaseRuntimeFnc(orbint),
        outputTimestep(_outputTimestep),
        outputFirst(_outputFirst),
        outputLast (_outputLast),
        outputIter (_outputFirst)
    {}
    virtual bool processTimestep(double tbegin, double timestep);
private:
    /// interval between taking samples from the trajectory (counting from the beginning of the episode)
    const double outputTimestep;

    /// location in the external array for storing the samples from this orbit
    /// (the two iterators point to the first and the last array elements reserved for this orbit)
    const ParticleArrayType::const_iterator outputFirst, outputLast;

    /// pointer to the current array element where the upcoming sample will be placed
    ParticleArrayType::iterator outputIter;
};

/** Fixed global parameters of this task */
struct ParamsPotential {
    /// imposed symmetry of the potential
    coord::SymmetryType symmetry;

    /// order of spherical-harmonic expansion of the potential
    /// (0 means spherical symmetry, and generally only even values should be used)
    unsigned int lmax;

    /// min/max grid radii (0 means autodetect)
    double rmin, rmax;

    /// number of points in the radial grid for the potential;
    /// all these parameters describe the Multipole potential whether constructed
    /// at the beginning of the simulation or updated at the end of an episode
    unsigned int gridSizeR;

    /// number of subsamples collected for each orbit during an episode
    unsigned int numSamplesPerEpisode;

    /// [base] file name for outputting the potential expansion coefficients;
    /// the simulation time is appended to the filename
    std::string outputFilename;

    /// interval between outputting the potential (should be a multiple of the episode length)
    double outputInterval;

    /// set defaults
    ParamsPotential() :
        symmetry(coord::ST_TRIAXIAL),
        lmax(0), rmin(0), rmax(0), gridSizeR(25),
        numSamplesPerEpisode(1), outputInterval(0)
    {}
};

/** The driver class performing the task of potential update and output */
class RagaTaskPotential: public BaseRagaTask {
public:
    RagaTaskPotential(
        const ParamsPotential& params,
        const particles::ParticleArrayAux& particles,
        potential::PtrPotential& ptrPot);
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex);
    virtual void startEpisode(double timeStart, double episodeLength);
    virtual void finishEpisode();
    virtual const char* name() const { return "PotentialUpdate"; }
private:

    /** fixed parameters of this task  */
    const ParamsPotential params;

    /** read-only reference to the list of particles (only their current masses are used)  */
    const particles::ParticleArrayAux& particles;

    /** reference to the global shared pointer containing the stellar potential,
        the original shared pointer is owned by the RagaCore class;
        it is re-assigned by this task at the end of episode
    */
    potential::PtrPotential& ptrPot;

    /** last time when the potential was written out into a text file  */
    double prevOutputTime;

    /** beginning and duration of the current episode  */
    double episodeStart, episodeLength;

    /** place for storing the coordinates of particle subsamples recorded
        during the episode and used to reinitialize the potential
        (each particle is allocated a block of numSamplesPerEpisode elements)
    */
    particles::ParticleArray<coord::PosCyl> particleTrajectories;
};

}  // namespace raga