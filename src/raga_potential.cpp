#include "raga_potential.h"
#include "potential_multipole.h"
#include "potential_factory.h"
#include "utils.h"
#include "math_core.h"
#include <cmath>

namespace raga {

bool RuntimePotential::processTimestep(double timeBegin, double timeStep)
{
    double timeOffset;
    while(timeOffset = outputTimestep * (outputIter - outputFirst + 1) - timeBegin,
        timeOffset>0 && timeOffset<=timeStep && outputIter != outputLast)
    {
        (outputIter++)->first = toPosCyl(orbint.getSol(timeOffset));
    }
    return true;
}


RagaTaskPotential::RagaTaskPotential(
    const ParamsPotential& _params,
    const particles::ParticleArrayAux& _particles,
    potential::PtrPotential& _ptrPot)
:
    params(_params),
    particles(_particles),
    ptrPot(_ptrPot),
    prevOutputTime(-INFINITY)
{
    FILTERMSG(utils::VL_DEBUG, "RagaTaskPotential", "Potential update is enabled");
}

void RagaTaskPotential::createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int index)
{
    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new RuntimePotential(
        orbint,
        episodeLength / params.numSamplesPerEpisode,
        particleTrajectories.data.begin() + params.numSamplesPerEpisode * index,
        particleTrajectories.data.begin() + params.numSamplesPerEpisode * (index+1) )));
}

void RagaTaskPotential::startEpisode(double timeStart, double length)
{
    episodeStart  = timeStart;
    episodeLength = length;
    unsigned int nbody = particles.size();
    particleTrajectories.data.assign(nbody * params.numSamplesPerEpisode,
        particles::ParticleArray<coord::PosCyl>::ElemType(coord::PosCyl(NAN, NAN, NAN), NAN));
    // output the potential (if needed) at the start of the first episode
    if(prevOutputTime == -INFINITY && !params.outputFilename.empty()) {
        prevOutputTime = episodeStart;
        writePotential(params.outputFilename + utils::toString(episodeStart), *ptrPot);
    }
}

void RagaTaskPotential::finishEpisode()
{
    // assign mass to trajectory samples
    unsigned int nbody = particles.size();
    for(unsigned int i=0; i<nbody; i++) {
        double mass = particles.mass(i) / params.numSamplesPerEpisode;
        for(unsigned int j=0; j<params.numSamplesPerEpisode; j++)
            particleTrajectories[i * params.numSamplesPerEpisode + j].second = mass;
    }

    // eliminate samples with zero mass or undetermined coordinates:
    // scan the array and squeeze it towards the head
    ParticleArrayType::iterator src = particleTrajectories.data.begin(); // where to take elements from
    ParticleArrayType::iterator dest = src;  // where to store the elements (dest <= src always)
    while(src != particleTrajectories.data.end()) {
        if(isFinite(src->first.R) && src->second > 0)
            *(dest++) = *src;
        ++src;
    }
    // shrink the array to retain only valid samples
    particleTrajectories.data.erase(dest, particleTrajectories.data.end());
    FILTERMSG(utils::VL_DEBUG, "RagaTaskPotential",
        "Retained "+utils::toString(particleTrajectories.size())+" samples");

    // update the potential
    ptrPot = potential::Multipole::create(particleTrajectories,
        params.symmetry, params.lmax, params.lmax, params.gridSizeR, params.rmin, params.rmax);

    // write out the new potential (if needed)
    double time = episodeStart+episodeLength;
    if(!params.outputFilename.empty() && time >= prevOutputTime + params.outputInterval*0.999999) {
        prevOutputTime = time;
        writePotential(params.outputFilename + utils::toString(time), *ptrPot);
    }
}

}  // namespace raga