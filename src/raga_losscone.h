/** \file    raga_losscone.h
    \brief   Loss-cone treatment (part of the Raga code)
    \author  Eugene Vasiliev
    \date    2013-2017

    This module deals with captures of stars by the central supermassive black hole(s).

    A particle is considered to be captured or tidally disrupted if it approaches to within
    the given distance (loss-cone radius) from the black hole, while passing the pericenter
    of its orbit. Such particle is excluded from subsequent evolution, its orbit integration
    is terminated for the current episode, and its mass is set to zero at the end of
    the episode, rendering it an inactive particle. The mass of the captured particles,
    or a given fraction of it, is then added to the mass of the black hole at the end of
    the episode. If there is a binary black hole, each component may capture stars within
    its own loss-cone radius. In this case, captured particles do not contribute
    to the binary hardening, since their evolution terminates at the moment of capture.
    Since it may happen that a star passes the pericenter during the current timestep,
    and is outside the capture radius both at the beginning and the end of timestep,
    but still briefly plunges inside this radius in between, we determine the exact
    time of pericenter passage and the distance r from the black hole at this time,
    by locating the root of dr(t)/dt=0 when it changes from negative to positive
    within the current timestep. In case of two black holes, the distance to each one
    is tracked separately.
*/
#pragma once
#include "raga_base.h"
#include "particles_base.h"
#include "potential_analytic.h"

namespace raga {

/** Information about a particle captured by a supermassive black hole */
struct CaptureData {
    /// time (from the beginning of the episode) at which the particle was captured (NAN if it wasn't)
    double tcapt;

    /// energy of the particle w.r.t. this black hole (i.e. considering only its gravitational
    /// potential plus the kinetic energy) at the moment of capture
    double E;

    /// distance to the black hole at the moment of capture
    double rperi;

    /// capture radius for the given star and the given black hole
    double rcapt;

    /// which of the two black holes in the binary has captured this particle
    /// (always 0 in the case of a single black hole)
    int indexBH;

    CaptureData() : tcapt(NAN), E(0), rperi(0), rcapt(0), indexBH(0) {}
};

/** Runtime function responsible for tracking the capture of particles by the central black hole(s) */
class RuntimeLosscone: public orbit::BaseRuntimeFnc {

    /// parameters of the binary BH
    const potential::KeplerBinaryParams& bh;

    /// place for storing the information for the captured particle:
    /// this points to an element of a vector in the parent task, which by default indicates
    /// no capture (tcapt=-1)
    const std::vector<CaptureData>::iterator output;

    /// capture radii for the first and optionally the second black hole (array of two numbers)
    double captureRadius[2];

    /// time derivative of the distance to the first and the second black hole:
    /// when its sign changes from - to +, then this timestep contains the pericenter passage
    double drdt[2];

public:
    RuntimeLosscone(
        orbit::BaseOrbitIntegrator& orbint,
        const potential::KeplerBinaryParams& _bh,
        const std::vector<CaptureData>::iterator& _output,
        const double _captureRadius[])
    :
        BaseRuntimeFnc(orbint),
        bh(_bh),
        output(_output)
    {
        drdt[0] = drdt[1] = 0.;
        captureRadius[0] = _captureRadius[0];
        captureRadius[1] = _captureRadius[1];
    }

    virtual bool processTimestep(double tbegin, double timestep);
};

/** Fixed global parameters of the loss-cone module */
struct ParamsLosscone {
    /// file name for storing the capture history (if empty then no output is stored)
    std::string outputFilename;

    /// fraction of the mass of captured stars to be added to the BH mass (between 0 and 1)
    double captureMassFraction;

    /// speed of light in N-body units, used to compute the Schwarzschild radius of the BHs
    double speedOfLight;

    /// set defaults
    ParamsLosscone() : captureMassFraction(1), speedOfLight(INFINITY) {}
};

/** The driver class implementing the capture of particles by the central supermassive black hole(s).
    The parameters of the black hole(s) are stored in the external BHParams variable,
    and a non-const reference to it is used to update the masses of the black holes
    at the end of the episode, if any particles were captured.
    A non-const reference to the array of particles is used to set the masses of captured particles
    to zero, which excludes them from the subsequent evolution (their positions remain frozen at
    the moment of capture).
    The list of captured particles, sorted by the capture time, is written to a text file.
*/
class RagaTaskLosscone: public BaseRagaTask {
public:
    RagaTaskLosscone(
        const ParamsLosscone& params,
        particles::ParticleArrayAux& particles,
        potential::KeplerBinaryParams& bh);
    virtual void createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex);
    virtual void startEpisode(double timeStart, double episodeLength);
    virtual void finishEpisode();
    virtual const char* name() const { return "LossCone       "; }
private:
    /// fixed parameters of this task
    const ParamsLosscone params;

    /// reference to the array of particles in the simulation;
    /// the masses of captured particles are set to zero at the end of the episode
    particles::ParticleArrayAux& particles;

    /// non-const reference to the parameters of the black hole(s);
    /// the BH mass is increased when particles are captured
    potential::KeplerBinaryParams& bh;

    /// beginning and end of the current episode
    double episodeStart, episodeEnd;

    /// storage for recording the capture events for each particle
    /// (each runtime function is given an element in this vector)
    std::vector<CaptureData> captures;

    /// total # of captured particles since the beginning of the simulation
    /// (indicates whether one needs to print the file header on the first recorded capture)
    unsigned int totalNumCaptured;
};

}  // namespace raga