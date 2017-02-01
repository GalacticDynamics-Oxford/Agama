#include "raga_losscone.h"
#include "math_core.h"
#include "utils.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <algorithm>

namespace raga {

//---------- Loss-cone handling ----------//

/** Helper class for finding the pericenter passage:
    compute the time derivative of the (squared) distance to the black hole;
    when it passes through zero, we are at the peri/apocenter
*/
class PericenterFinder: public math::IFunctionNoDeriv {
    const BHParams& bh;
    const math::BaseOdeSolver& sol;
    const int bhindex;
public:
    PericenterFinder(const BHParams& _bh, const math::BaseOdeSolver& _sol, const int _bhindex) :
        bh(_bh), sol(_sol), bhindex(_bhindex)  {}

    /// return the time derivative of the squared distance to the given black hole
    virtual double value(const double time) const
    {
        double bhX[2], bhY[2], bhVX[2], bhVY[2];
        double posvel[6];
        bh.keplerOrbit(time, bhX, bhY, bhVX, bhVY);
        sol.getSol(time, posvel);
        return
            (posvel[0]-bhX[bhindex]) * (posvel[3]-bhVX[bhindex]) +
            (posvel[1]-bhY[bhindex]) * (posvel[4]-bhVY[bhindex]) +
             posvel[2] * posvel[5];
    }
};

StepResult RuntimeLosscone::processTimestep(
    const math::BaseOdeSolver& sol, const double tbegin, const double tend, double vars[])
{
    // first check if this particle has already been captured
    // (actually then the orbit integration should have been terminated, so this check is redundant)
    if(output->tcapt >= 0)
        return SR_TERMINATE;

    int numBH = bh.sma>0 ? 2 : 1;
    for(int b=0; b<numBH; b++) {
        const PericenterFinder pf(bh, sol, b);

        // if this is the first timestep, need to compute d(r^2)/dt at the beginning
        // of the timestep, otherwise copy the stored value from the previous timestep
        // w.r.t the central black hole(s) at the beginning of the timestep
        double prevdrdt = tbegin == 0 ? pf(tbegin) : drdt[b];

        // now compute the same quantity at the end of the current timestep
        drdt[b] = pf(tend);

        // check if we just passed a pericenter w.r.t. one of the black hole(s),
        // i.e. r^2 was decreasing at the beginning of this timestep,
        // and is now increasing at the end of the timestep
        if(! (prevdrdt <= 0 && drdt[b] > 0))
            continue;

        // if we did, then find the exact time of pericenter passage
        double tperi = math::findRoot(pf, tbegin, tend, 1e-4);
        if(!isFinite(tperi))  // the root-finder failed:
            // this may happen if the velocity has changed unfavourably between
            // the end of the previous timestep and the beginning of the current one,
            // due to two-body perturbation applied between the timesteps;
            // in this case set assume the pericenter passage happened at the timestep boundary.
            tperi = tbegin;

        // compute the pericenter distance
        double bhX[2], bhY[2], bhVX[2], bhVY[2];
        double posvel[6];
        bh.keplerOrbit(tperi, bhX, bhY, bhVX, bhVY);
        sol.getSol(tperi, posvel);
        double rperi = sqrt(pow_2(posvel[0]-bhX[b]) + pow_2(posvel[1]-bhY[b]) + pow_2(posvel[2]));

        // compare it with the capture radius
        if(rperi > captureRadius[b])
            continue;

        // record the capture event
        output->tcapt   = tperi;
        double Mbh      = bh.mass * (numBH==1 ? 1 : (b==0 ? 1 : bh.q) / (1 + bh.q));
        output->E       = -Mbh / rperi +
            0.5 * (pow_2(posvel[3]) + pow_2(posvel[4]) + pow_2(posvel[5]));
        output->rperi   = rperi;
        output->indexBH = b;
        /*output->L2 =
            pow_2((posvel[1]-bhY[b]) * (posvel[5]        ) - (posvel[2]       ) * (posvel[4]-bhVY[b])) +
            pow_2((posvel[2]       ) * (posvel[3]-bhVX[b]) - (posvel[0]-bhX[b]) * (posvel[5]        )) +
            pow_2((posvel[0]-bhX[b]) * (posvel[4]-bhVY[b]) - (posvel[1]-bhY[b]) * (posvel[3]-bhVX[b]));*/

        // store the position/velocity at the moment of capture
        for(int i=0; i<6; i++)
            vars[i] = posvel[i];

        // inform the ODE integrator that this particle is no more...
        return SR_TERMINATE;
    }

    // if the particle wasn't captured, continue as usual
    return SR_CONTINUE;
}


RagaTaskLosscone::RagaTaskLosscone(
    const ParamsLosscone& _params,
    particles::ParticleArrayCar& _particles,
    BHParams& _bh)
:
    params(_params),
    particles(_particles),
    bh(_bh),
    totalNumCaptured(0)
{
    utils::msg(utils::VL_DEBUG, "RagaTaskLosscone",
        "Capture radius=" + utils::toString(params.captureRadius[0]) +
        (  bh.sma>0 ? "," + utils::toString(params.captureRadius[1]) : "") +
        ", accreted mass fraction=" + utils::toString(params.captureMassFraction));
}

PtrRuntimeFnc RagaTaskLosscone::createRuntimeFnc(unsigned int particleIndex)
{
    return PtrRuntimeFnc(new RuntimeLosscone(
        bh, captures.begin() + particleIndex, params.captureRadius));
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
        double mass = particles[ip].second;
        double tcapt= captures [ip].tcapt;
        if(mass == 0 || tcapt < 0)
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
                strm << "Time    \tParticleMass\tPericenterRad\tEnergy  \tParticleIndex\tBHindex\n";
            } else  // append to the file
                strm.open(params.outputFilename.c_str(), std::ios_base::app);
        }
        for(size_t c=0; c<sortedCaptures.size(); c++) {
            size_t ip = sortedCaptures[c].second;  // index of the captured particle
            std::string strParticleIndex = utils::toString(ip);
            if(strParticleIndex.size()<8)  // padding to at least one tab-length
                strParticleIndex.insert(strParticleIndex.end(), 8-strParticleIndex.size(), ' ');
            strm <<
                utils::pp(episodeStart + captures[ip].tcapt, 10) + '\t' +
                utils::pp(particles[ip].second, 12) + '\t' +   // particle mass
                utils::pp(captures [ip].rperi,  12) + '\t' +   // pericenter distance
                utils::pp(captures [ip].E,      12) + '\t' +   // energy at the moment of capture
                strParticleIndex                    + '\t' +   // index of the particle that was captured
                utils::toString(captures[ip].indexBH) + '\n';  // index of the black hole that captured it
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