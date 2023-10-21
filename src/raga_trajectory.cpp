#include "raga_trajectory.h"
#include "particles_io.h"
#include "utils.h"
#include <fstream>
#include <stdexcept>

namespace raga {

//---------- Trajectory output ----------//

RagaTaskTrajectory::RagaTaskTrajectory(
    const ParamsTrajectory& _params,
    const particles::ParticleArrayAux& _particles)
:
    params(_params),
    particles(_particles),
    prevOutputTime(-INFINITY)
{
    FILTERMSG(utils::VL_DEBUG, "RagaTaskTrajectory",
        "Output interval="+utils::toString(params.outputInterval));
}

void RagaTaskTrajectory::outputParticles(double time)
{
    if(!params.outputFilename.empty() && time >= prevOutputTime + params.outputInterval*0.999999) {
        std::string filename = params.outputFilename;
        // on the first output, create the file, otherwise append to the existing file (only for nemo)
        bool append = prevOutputTime != -INFINITY;
        // if outputFormat=='nemo', it is written into a single file,
        // otherwise each snapshot is written into a separate file
        if(params.outputFormat.empty() || tolower(params.outputFormat[0])!='n') {
            filename += utils::toString(time);
            append = false;
        }
        particles::writeSnapshot(filename, particles,
            params.outputFormat, units::ExternalUnits(), params.header, time, append);
        prevOutputTime = time;
    }
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