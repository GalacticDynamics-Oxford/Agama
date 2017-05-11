#include "raga_trajectory.h"
#include "particles_io.h"
#include "utils.h"

namespace raga {

//---------- Trajectory output ----------//

RagaTaskTrajectory::RagaTaskTrajectory(
    const ParamsTrajectory& _params,
    const particles::ParticleArrayCar& _particles)
:
    params(_params),
    particles(_particles),
    prevOutputTime(-INFINITY)
{
    utils::msg(utils::VL_DEBUG, "RagaTaskTrajectory",
        "Output interval="+utils::toString(params.outputInterval));
}

void RagaTaskTrajectory::outputParticles(double time)
{
    if(!params.outputFilename.empty() && time >= prevOutputTime + params.outputInterval) {
        std::string filename = params.outputFilename;
        // on the first output, create the file, otherwise append to the existing file (only for nemo format)
        bool append = prevOutputTime != -INFINITY;
        // if outputFormat=='nemo', it is written into a single file,
        // otherwise each snapshot is written into a separate file
        if(params.outputFormat.empty() || (params.outputFormat[0]!='N' && params.outputFormat[0]!='n')) {
            filename += utils::toString(time);
            append = false;
        }
        particles::PtrIOSnapshot snap = particles::createIOSnapshotWrite(
            filename, params.outputFormat, units::ExternalUnits(), params.header, time, append);
        snap->writeSnapshot(particles);
        prevOutputTime = time;
    }
}

PtrRuntimeFnc RagaTaskTrajectory::createRuntimeFnc(unsigned int)
{
    return PtrRuntimeFnc(new RuntimeTrajectory());
}

void RagaTaskTrajectory::startEpisode(double timeStart, double length)
{
    episodeStart  = timeStart;
    episodeLength = length;
    outputParticles(episodeStart);
}

void RagaTaskTrajectory::finishEpisode()
{
    outputParticles(episodeStart+episodeLength);
}

}  // namespace raga