#include "raga_losscone.h"
#include "math_core.h"
#include "utils.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>

namespace raga {

//---------- Loss-cone handling ----------//
namespace{

/** Helper class for finding the pericenter passage:
    compute the time derivative of the (squared) distance to the black hole;
    when it passes through zero, we are at the peri/apocenter
*/
class PericenterFinder: public math::IFunctionNoDeriv {
    const potential::KeplerBinaryParams& bh;
    const orbit::BaseOrbitIntegrator& orbint;
    const int bhindex;
    const double tbegin;
public:
    PericenterFinder(const potential::KeplerBinaryParams& _bh,
        const orbit::BaseOrbitIntegrator& _orbint, int _bhindex, double _tbegin) :
        bh(_bh), orbint(_orbint), bhindex(_bhindex), tbegin(_tbegin)  {}

    /// return the time derivative of the squared distance to the given black hole
    virtual double value(const double timeOffset) const
    {
        double bhX[2], bhY[2], bhVX[2], bhVY[2];
        bh.keplerOrbit(tbegin+timeOffset, bhX, bhY, bhVX, bhVY);
        coord::PosVelCar pv = orbint.getSol(timeOffset);
        return
            (pv.x - bhX[bhindex]) * (pv.vx - bhVX[bhindex]) +
            (pv.y - bhY[bhindex]) * (pv.vy - bhVY[bhindex]) +
             pv.z * pv.vz;
    }
};

} // anonymous namespace

bool RuntimeLosscone::processTimestep(double tbegin, double timestep)
{
    // first check if this particle has already been captured
    // (actually then the orbit integration should have been terminated, so this check is redundant)
    if(output->tcapt >= 0)
        return false;

    int numBH = bh.sma>0 ? 2 : 1;
    for(int b=0; b<numBH; b++) {
        const PericenterFinder pf(bh, orbint, b, tbegin);

        // if this is the first timestep, need to compute d(r^2)/dt at the beginning
        // of the timestep, otherwise copy the stored value from the previous timestep
        // w.r.t the central black hole(s) at the beginning of the timestep
        double prevdrdt = tbegin == 0 ? pf(0) : drdt[b];

        // now compute the same quantity at the end of the current timestep
        drdt[b] = pf(timestep);

        // check if we just passed a pericenter w.r.t. one of the black hole(s),
        // i.e. r^2 was decreasing at the beginning of this timestep,
        // and is now increasing at the end of the timestep
        if(! (prevdrdt <= 0 && drdt[b] > 0))
            continue;

        // if we did, then find the exact time of pericenter passage (offset relative to tbegin)
        double tperiOffset = math::findRoot(pf, 0, timestep, 1e-4);
        if(!isFinite(tperiOffset))  // the root-finder failed:
            // this may happen if the velocity has changed unfavourably between
            // the end of the previous timestep and the beginning of the current one,
            // due to two-body perturbation applied between the timesteps;
            // in this case set assume the pericenter passage happened at the timestep boundary.
            tperiOffset = 0;

        // compute the pericenter distance
        double bhX[2], bhY[2], bhVX[2], bhVY[2];
        bh.keplerOrbit(tperiOffset + tbegin, bhX, bhY, bhVX, bhVY);
        coord::PosVelCar posvel = orbint.getSol(tperiOffset);
        double rperi = sqrt(pow_2(posvel.x-bhX[b]) + pow_2(posvel.y-bhY[b]) + pow_2(posvel.z));

        // compare it with the capture radius
        if(rperi > captureRadius[b])
            continue;

        // record the capture event
        output->tcapt   = tperiOffset + tbegin;
        double Mbh      = bh.mass * (numBH==1 ? 1 : (b==0 ? 1 : bh.q) / (1 + bh.q));
        output->E       = -Mbh / rperi +
            0.5 * (pow_2(posvel.vx-bhVX[b]) + pow_2(posvel.vy-bhVY[b]) + pow_2(posvel.vz));
        output->rperi   = rperi;
        output->rcapt   = captureRadius[b];
        output->indexBH = b;

        // store the position/velocity at the moment of capture
        orbint.init(posvel, tperiOffset + tbegin);

        // inform the ODE integrator that this particle is no more...
        return false;
    }

    // if the particle wasn't captured, continue as usual
    return true;
}


RagaTaskLosscone::RagaTaskLosscone(
    const ParamsLosscone& _params,
    particles::ParticleArrayAux& _particles,
    potential::KeplerBinaryParams& _bh)
:
    params(_params),
    particles(_particles),
    bh(_bh),
    totalNumCaptured(0)
{
    FILTERMSG(utils::VL_DEBUG, "RagaTaskLosscone",
        std::string(bh.sma==0 ? "One black hole" : "Two black holes") +
        ", c=" + utils::toString(params.speedOfLight) +
        ", accreted mass fraction=" + utils::toString(params.captureMassFraction));
}

void RagaTaskLosscone::createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int particleIndex)
{
    double Mbh0 = bh.sma==0 ? bh.mass / (1 + bh.q) : bh.mass;
    double Mbh1 = bh.sma==0 ? bh.mass / (1 + bh.q) * bh.q : 0;
    double captureRadius[2] = {
        fmax(8 * Mbh0 / pow_2(params.speedOfLight),
            particles.point(particleIndex).stellarRadius *
            pow(Mbh0 / particles.point(particleIndex).stellarMass, 1./3) ),
        fmax(8 * Mbh1 / pow_2(params.speedOfLight),
            particles.point(particleIndex).stellarRadius *
            pow(Mbh1 / particles.point(particleIndex).stellarMass, 1./3) ) };

    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new RuntimeLosscone(
        orbint, bh, captures.begin() + particleIndex, captureRadius)));
}

void RagaTaskLosscone::startEpisode(double timeStart, double length)
{
    episodeStart  = timeStart;
    episodeEnd    = timeStart + length;
    captures.assign(particles.size(), CaptureData());  // reserve room for recording capture events
}

void RagaTaskLosscone::finishEpisode()
{
    // handle captured particles
    double capturedMass[2] = {0};
    int numBH = bh.sma>0 ? 2 : 1;
    std::vector< std::pair<double, size_t> > sortedCaptures;
    assert(particles.size() == captures.size());
    for(size_t ip=0; ip<particles.size(); ip++) {
        double mass = particles.mass(ip);
        double tcapt= captures [ip].tcapt;
        if(mass == 0 || !(tcapt >= 0))
            continue;   // nothing happened to this particle
        assert(captures[ip].indexBH < numBH);
        capturedMass[captures[ip].indexBH] += mass;
        sortedCaptures.push_back(std::make_pair(tcapt, ip));
    }
    if(sortedCaptures.size()>0) {
        std::sort(sortedCaptures.begin(), sortedCaptures.end());
        std::ofstream strm;
        if(!params.outputFilename.empty()) {
            if(totalNumCaptured == 0) {
                // this is the first time the file is opened (i.e. is created), so print out the header
                strm.open(params.outputFilename.c_str());
                strm << "#Time   \tParticleMass\tPericenterRad\tEnergy  \t"
                "ParticleIndex\tBHindex\tCaptureRadius\tStellarRadius\tStellarMass\n";
            } else  // append to the file
                strm.open(params.outputFilename.c_str(), std::ios_base::app);
        }
        for(size_t c=0; c<sortedCaptures.size(); c++) {
            size_t ip = sortedCaptures[c].second;  // index of the captured particle
            std::string strParticleIndex = utils::toString(ip);
            if(strParticleIndex.size()<8)  // padding to at least one tab-length
                strParticleIndex.insert(strParticleIndex.end(), 8-strParticleIndex.size(), ' ');
            strm <<
                utils::pp(episodeStart + captures[ip].tcapt, 12) + '\t' +
                utils::pp(particles.mass(ip),   12) + '\t' +  // particle mass
                utils::pp(captures[ip].rperi,   12) + '\t' +  // pericenter distance
                utils::pp(captures[ip].E,       12) + '\t' +  // energy at the moment of capture
                strParticleIndex                    + '\t' +  // index of the particle that was captured
                utils::toString(captures[ip].indexBH)+'\t' +  // index of the black hole that captured it
                utils::pp(captures[ip].rcapt,   12) + '\t' +  // capture radius for this event
                utils::pp(particles.point(ip).stellarRadius, 12) + '\t' +  // radius and
                utils::pp(particles.point(ip).stellarMass,   12) + '\n';   // mass of the star
            // set the particle mass to zero to indicate that it no longer exists
            particles[ip].second = 0;
        }
        utils::msg(utils::VL_MESSAGE, "RagaTaskLosscone",
            "By time " + utils::toString(episodeEnd) +
            " captured " + utils::toString(sortedCaptures.size()) +
            " particles, total mass=" + utils::toString(capturedMass[0]) +
            (numBH>1 ? "+" + utils::toString(capturedMass[1]) : ""));

        // add the mass of captured particles to the mass(es) of the black hole(s)
        if(numBH>1) {  // adjust the mass ratio of the binary
            bh.q =
                (bh.mass * bh.q + capturedMass[1] * (1 + bh.q)) /
                (bh.mass        + capturedMass[0] * (1 + bh.q));
        }
        bh.mass += (capturedMass[0] + capturedMass[1]) * params.captureMassFraction;
        totalNumCaptured += sortedCaptures.size();
    }
}

}  // namespace raga