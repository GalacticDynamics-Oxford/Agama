/** \file    raga_trajectory.h
    \brief   Record the evolution of the N-body system (part of the Raga code)
    \author  Eugene Vasiliev
    \date    2013-2017

    This module performs the simple task of storing the output of Monte Carlo simulation
    as N-body snapshots.

    To simplify matters, the output is only possible between episodes,
    hence there is no corresponding runtime function (it would do nothing anyway),
    and the positions/velocities of particles are taken from the global `particles' array
    at the end of an episode.
    It may be possible to store the output more frequently, recording trajectories of particles
    during the episode, hence the name of the task; however, this is considered impractical.
*/
#pragma once
#include "raga_base.h"
#include "particles_base.h"

namespace raga {

/** Global parameters of this task */
struct ParamsTrajectory {
    /// [base] file name for writing the output snapshots;
    /// if the output format is Nemo, then all snapshots are stored in a single file with this name,
    /// otherwise the simulation time is appended to the filename and each snapshot is written
    /// to a different file
    std::string outputFilename;

    /// format of the output snapshot
    std::string outputFormat;

    /// interval between output (should be an integer multiple of the episode length)
    double outputInterval;

    /// optional header written in the output file
    std::string header;

    /// set defaults
    ParamsTrajectory() : outputFormat("text"), outputInterval(0) {}
};

/** The task that stores the output of the simulation as a series of N-body snapshots */
class RagaTaskTrajectory: public BaseRagaTask {
public:
    RagaTaskTrajectory(
        const ParamsTrajectory& params,
        const particles::ParticleArrayAux& particles);
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator&, unsigned int) {}  // does nothing
    virtual void startEpisode(double timeStart, double episodeLength);
    virtual void finishEpisode();
    virtual const char* name() const { return "SnapshotOutput "; }
private:
    /// perform the actual output and update the last output time
    void outputParticles(double time);

    /// fixed parameters of this task
    const ParamsTrajectory params;

    /// read-only reference to the list of particles in the simulation
    /// (they are written into the output file when the time comes)
    const particles::ParticleArrayAux& particles;

    /// last time the output file was written
    double prevOutputTime;

    /// beginning and duration of the current episode
    double episodeStart, episodeLength;
};

}  // namespace raga