/** \file    raga_base.h
    \brief   Common classes for the Raga stellar-dynamical code
    \author  Eugene Vasiliev
    \date    2013-2017

    This is the common header included by all parts of the Raga code.

    The evolution of the N-particle system is followed through a series of 'episodes'.
    During each episode, the global properties of the system are fixed
    (the smooth gravitational potential, parameters of the central [binary] black hole, etc.),
    and each particle moves independently from others (the 'orbit integration' phase).
    During this phase, one or more global tasks perform additional operations on each orbit;
    the list of tasks depends on the options selected for each particular simulation
    (e.g. whether it includes two-body relaxation or the loss-cone treatment).
    At the beginning of an episode each task prepares the ground, and after all particle orbits
    have been computed, the episode is finished by invoking the corresponding finalization
    routines of each task; the order *is* important. During these phases (between the episodes),
    the global properties of the system may be modified, but particle positions stay put
    (although the mass of a particle may be set to zero to indicate that it is no longer active).

    Each task is represented by three closely related classes:
    - the global instance of the task which persists for the entire simulation
    (derived from BaseRagaTask);
    - an instance of a 'runtime function' attached to each orbit for the duration of an episode,
    that performs specific operations and possibly collects some data into variables owned by
    the parent task (because an instance of a runtime function only exists during orbit
    integration, and is deleted once the particle reaches the end of the current episode
    and the control flow moves to the next particle).
    - the set of control parameters represented by a data-only structure, initialized at
    the beginning of the simulation by parsing an INI file (these parameters are fixed, but
    other global properties stored in member variables of a task may be changed between episodes).
    These triplets of related classes (the task, the runtime function, and the parameters)
    are grouped together in a separate source file for each task.

    The top-level workflow is managed by the class RagaCore in raga_core.cpp, which is responsible
    for setting up the simulation (reading the INI file and constructing the list of tasks) and
    cycling through the episodes (invoking initialization and finalization methods of each task,
    and running the orbit integration in between).
*/
#pragma once
#include "orbit.h"

/** The Monte Carlo stellar-dynamical code Raga */
namespace raga {

/** Prototype of a RAGA task that handles a specific aspect of the evolution.
    Instances of derived classess are created at the beginning of the simulation
    and exist for its entire duration, unlike the runtime functions whose lifetime
    is restricted to each episode.
    Each task corresponds to a particular runtime function class;
    the instance of such function is created for each orbit in each episode,
    and the function may collect the data for its parent task and/or modify
    the behaviour of the orbit during the current episode.
    At the end of the episode, the collected data (which must be stored externally
    to the runtime function, i.e. in a member variable of its parent task) is used
    to possibly modify the global state of the simulation (e.g. the total potential);
    each task is granted access to its specific set of global parameters.
*/
class BaseRagaTask {
public:
    virtual ~BaseRagaTask() {}

    /** Create an instance of a runtime function for the given particle
        and attach it to the orbit integrator for that particle */
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex) = 0;

    /** Prepare for the upcoming episode that begins at timeStart and lasts for episodeLength */
    virtual void startEpisode(double timeStart, double episodeLength) = 0;

    /** Finalize the episode and possibly change the global state of the simulation */
    virtual void finishEpisode() = 0;

    /** Return a human-readable task name */
    virtual const char* name() const = 0;
};

/** Shared pointer to a RAGA task */
typedef shared_ptr<BaseRagaTask> PtrRagaTask;

}
