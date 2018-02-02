#include "raga_potential.h"
#include "potential_multipole.h"
#include "potential_factory.h"
#include "utils.h"
#include "math_core.h"
#include <cmath>

namespace raga {

orbit::StepResult RuntimePotential::processTimestep(
    const math::BaseOdeSolver& sol, const double tbegin, const double tend, double[])
{
    double t;
    while(t = outputTimestep * (outputIter - outputFirst + 1),
        t>tbegin && t<=tend && outputIter != outputLast)
    {
        (outputIter++)->first = toPosCyl(
            coord::PosCar(sol.getSol(t, 0), sol.getSol(t, 1), sol.getSol(t, 2)));
    }
    return orbit::SR_CONTINUE;
}


RagaTaskPotential::RagaTaskPotential(
    const ParamsPotential& _params,
    const particles::ParticleArrayCar& _particles,
    potential::PtrPotential& _ptrPot)
:
    params(_params),
    particles(_particles),
    ptrPot(_ptrPot),
    prevOutputTime(-INFINITY)
{
    utils::msg(utils::VL_DEBUG, "RagaTaskPotential", "Potential update is enabled");
}

orbit::PtrRuntimeFnc RagaTaskPotential::createRuntimeFnc(unsigned int index)
{
    return orbit::PtrRuntimeFnc(new RuntimePotential(
        episodeLength / params.numSamplesPerEpisode,
        particleTrajectories.data.begin() + params.numSamplesPerEpisode * index,
        particleTrajectories.data.begin() + params.numSamplesPerEpisode * (index+1) ));
}

void RagaTaskPotential::startEpisode(double timeStart, double length)
{
    episodeStart  = timeStart;
    episodeLength = length;
    unsigned int nbody = particles.size();
    particleTrajectories.data.assign(nbody * params.numSamplesPerEpisode,
        particles::ParticleArray<coord::PosCyl>::ElemType(coord::PosCyl(NAN, NAN, NAN), NAN));
    outputPotential(episodeStart);
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
    utils::msg(utils::VL_DEBUG, "RagaTaskPotential",
        "Retained "+utils::toString(particleTrajectories.size())+" samples");

    // update the potential
    ptrPot = potential::Multipole::create(particleTrajectories,
        params.symmetry, params.lmax, params.lmax, params.gridSizeR);

    // write out the new potential
    outputPotential(episodeStart+episodeLength);
}

void RagaTaskPotential::outputPotential(double time)
{
    if(!params.outputFilename.empty() && time >= prevOutputTime + params.outputInterval) {
        prevOutputTime = time;
        writePotential(params.outputFilename + utils::toString(time), *ptrPot);
    }
}

}  // namespace raga