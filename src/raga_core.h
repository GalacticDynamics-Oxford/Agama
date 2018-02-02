/** \file    raga_core.h
    \brief   The driver class for the Raga stellar-dynamical code
    \author  Eugene Vasiliev
    \date    2013-2017

    This module implements the driver class for the Monte Carlo stellar-dynamical code Raga.

    As described in raga_base.h, the approach consists of the following ingredients:
    - A collection of particles (N-body system) -- coordinates, velocities and masses.
    - The smooth gravitational potential that these particles move in
    (it consists of the stellar potential, computed from the particles themselves
    by the Multipole class, plus optionally the central single or binary massive black hole).
    - Zero or more 'tasks' that deal with various physical processes in the simulation
    (e.g., two-body relaxation or the evolution of the binary black hole).

    The RagaCore class binds together these ingredients and performs the main simulation loop
    (a series of episodes):
    - each task prepares for the upcoming episode;
    - the RagaCore class computes the orbits of all particles during the episode;
    - each task processes the data collected during the episode and possibly modifies
    the state of the entire system.

    In the second phase (orbit integration), each particle is processed independently from
    the others, so the loop is trivially parallelized. Each orbit has an attached array of
    runtime functions, created by their corresponding tasks for each particle;
    the data collected by these functions is passed directly to their parent tasks, and
    they are also allowed to change the state of the orbit integration (or even terminate it).

    The global properties of the system are:
    - the stellar potential,
    - the parameters of the central black hole(s),
    - the particle masses and phase-space coordinates.
    These properties are stored in member variables of the RagaCore class,
    and each task is granted access to some of them (it could be either read-only access or
    the right to modify the variable; in both cases the tasks store the reference
    to the original variable which is owned by RagaCore). The order of updates is important,
    so that the tasks are created in a particular sequence at the beginning.
    - Additionally, there are fixed global parameters for each task and for RagaCore itself;
    these parameters are read from the INI file and do not change during the simulation.
    They also determine which tasks are created for the particular simulation.
*/
#pragma once
#include "raga_base.h"
#include "raga_binary.h"
#include "raga_losscone.h"
#include "raga_potential.h"
#include "raga_relaxation.h"
#include "raga_trajectory.h"
#include "utils_config.h"

namespace raga{

/// parameters of the entire simulation, not attributed to any Raga task
struct ParamsRaga {
    double integratorAccuracy;  ///< accuracy parameter for the orbit integrator
    bool   updatePotential;     ///< flag specifying whether to update the stellar potential
    double timeCurr;            ///< current sumulation time
    double timeEnd;             ///< total (maximum) simulation time
    double episodeLength;       ///< duration of one episode
    std::string fileInput;      ///< input file name (initial conditions for the simulation)
    std::string fileLog;        ///< file name for logging the global parameters of the simulation
};

/// the driver class performing the actual simulation
class RagaCore {
    ParamsRaga       paramsRaga;           ///< global parameters of the simulation
    ParamsPotential  paramsPotential;      ///< parameters of the stellar potential
    ParamsRelaxation paramsRelaxation;     ///< parameters of two-body perturbations
    ParamsTrajectory paramsTrajectory;     ///< parameters of trajectory output
    ParamsLosscone   paramsLosscone;       ///< parameters of loss-cone treatment
    ParamsBinary     paramsBinary;         ///< parameters of the binary BH evolution
    potential::PtrPotential ptrPot;        ///< stellar potential used in orbit integration
    BHParams bh;                           ///< parameters of the central black hole(s)
    particles::ParticleArrayCar particles; ///< particles (masses and phase-space coordinates)
    std::vector<PtrRagaTask> tasks;        ///< array of runtime tasks

    /** parse the configuration parameters stored in the key=value dictionary */
    void loadSettings(const utils::KeyValueMap& config);

    /** perform one complete episode */
    void doEpisode();

public:

    /** initialize the simulation using the parameters provided in the dictionary */
    explicit RagaCore(const utils::KeyValueMap& config);

    /** run the simulation (perform one or several episodes) */
    void run();
};

}  // namespace
