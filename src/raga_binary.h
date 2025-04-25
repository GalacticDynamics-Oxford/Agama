/** \file    raga_binary.h
    \brief   Binary supermassive black hole evolution (part of the Raga code)
    \author  Eugene Vasiliev
    \date    2015-2017

    This module implements the evolution of a binary supermassive black hole
    sitting at the center of the stellar system.

    The binary is assumed to be 'hard', i.e. its components move on a Keplerian orbit around
    the common center of mass, which itself is pinned down at origin.
    Furthermore its orbit lies in the x-y plane and is elongated in the x direction.
    The particles of the N-body system move in the time-dependent potential of this binary,
    and as a result, they gain or lose energy and angular momentum; by conservation laws,
    this should result in corresponding changes in the orbital parameters of the binary.
    The assumption is that they change very slowly, so these parameters are fixed
    during each episode, and updated at the end of an episode, using the accumulated
    changes in energy and angular momentum of all relevant particles during this episode.
    An 'encounter' is recorded when a particle approaches to within a given distance
    from the binary. The difference between energy and momentum of a particle
    at the beginning and the end of an encounter is recorded in a `BinaryEncounterData'
    structure; and the sequence of encounters for each particle during the current episode
    is then used to modify the parameters of the binary at the finalization stage.
    If the potential is non-axisymmetric, the change in angular momentum arising from
    the torque due to the stellar potential is subtracted from the overall change recorded
    during the encounter.
    Additionally, the binary black hole may shrink due to the emission of gravitational waves.
    This is also taken into account when adjusting its parameters at the end of an episode.
    The binary orbit parameters (semimajor axis and eccentricity) are written to a text file
    after each episode.
*/
#pragma once
#include "raga_base.h"
#include "particles_base.h"
#include "potential_analytic.h"

namespace raga {

/** Information about an encounter of a particle with the binary supermassive black hole */
struct BinaryEncounterData {
    double Tbegin, Tlength; ///< start time of encounter and its duration
    double Ebegin, Lbegin;  ///< energy and total angular momentum at the beginning of encounter
    double deltaE, deltaLz; ///< accumulated change in energy and z-angular momentum during encounter
    double costheta, phi;   ///< direction of velocity at the end of encounter (angles in spherical coord)
    BinaryEncounterData(double Tb=0, double Eb=0, double Lb=0) :
        Tbegin(Tb), Tlength(0), Ebegin(Eb), Lbegin(Lb), deltaE(0), deltaLz(0), costheta(NAN), phi(NAN) {}
};

/** List of encounters for each particle during an episode */
typedef std::vector<BinaryEncounterData> BinaryEncounterList;

/** Runtime function responsible for tracking the encounters with the binary black hole */
class RuntimeBinary: public orbit::BaseRuntimeFnc {
    const potential::BasePotential& potstars;///< stellar potential
    const potential::KeplerBinaryParams& bh; ///< parameters of the binary BH
    BinaryEncounterList& encountersList;     ///< place for storing the information about encounters
public:
    RuntimeBinary(
        orbit::BaseOrbitIntegrator& orbint,
        const potential::BasePotential& _potstars,
        const potential::KeplerBinaryParams& _bh,
        BinaryEncounterList& _encountersList)
    :
        BaseRuntimeFnc(orbint),
        potstars(_potstars),
        bh(_bh),
        encountersList(_encountersList)
    {}

    virtual bool processTimestep(double tbegin, double timestep);
};

/** Fixed global parameters for handling the evolution of binary supermassive black hole */
struct ParamsBinary {
    /// file name for storing the binary BH evolution history (if empty then no output is performed)
    std::string outputFilename;

    /// speed of light in model units, used to compute the losses due to gravitational-wave emission
    double speedOfLight;

    /// set defaults
    ParamsBinary() : speedOfLight(INFINITY) {}
};

/** The driver class implementing the evolution of the binary supermassive black hole.
    The parameters of the binary are stored elsewhere and used during orbit integration
    by the `integrateOrbit' routine. This class holds a non-const reference to these parameters,
    through which it modifies the orbital parameters at the end of each episode.
*/
class RagaTaskBinary: public BaseRagaTask {
public:
    RagaTaskBinary(
        const ParamsBinary& params,
        const particles::ParticleArrayAux& particles,
        const potential::PtrPotential& ptrPot,
        potential::KeplerBinaryParams& bh);
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex);
    virtual void startEpisode(double timeStart, double episodeLength);
    virtual void finishEpisode();
    virtual const char* name() const { return "BinaryBH       "; }

private:
    /// fixed parameters of this task
    const ParamsBinary params;

    /// read-only reference to the array of particles in the simulation (only the masses are used)
    const particles::ParticleArrayAux& particles;

    /// read-only pointer to the stellar potential of the system (it is used to compute
    /// the changes in particle energy and momentum during an encounter)
    const potential::PtrPotential& ptrPot;

    /// non-const reference to the parameters of the binary BH;
    /// its orbital parameters are updated at the end of the episode
    potential::KeplerBinaryParams& bh;

    /// beginning and duration of the current episode
    double episodeStart, episodeLength;

    /// storage for recording the encounters for each particle by its associated runtime function
    std::vector<BinaryEncounterList> encounters;

    /// whether this is the first episode in the simulation (if yes then print the file header)
    bool firstEpisode;
};

}  // namespace raga