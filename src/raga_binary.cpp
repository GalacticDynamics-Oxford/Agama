#include "raga_binary.h"
#include "math_core.h"
#include "potential_base.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <utility>

namespace raga {

//---------- Three-body encounters of particles with the central binary black hole ----------//

/// a particle is considered to be in a three-body encounter if its distance from origin
/// is less than the binary semimajor axis multiplied by this factor
const double BINARY_ENCOUNTER_RADIUS = 5.0;

/** Helper class to find the exact moment of crossing the critical radius */
class EncounterFinder: public math::IFunctionNoDeriv {
    const math::BaseOdeSolver& sol;
    const double r2crit;
public:
    EncounterFinder(const math::BaseOdeSolver& _sol, const double _r2crit) :
        sol(_sol), r2crit(_r2crit)  {}
    
    /// return the difference between the radius at the given time and the critical radius (both squared)
    virtual double value(const double time) const
    {
        double point[6];
        sol.getSol(time, point);
        return pow_2(point[0]) + pow_2(point[1]) + pow_2(point[2]) - r2crit;
    }
};

/// helper function to find the total energy in the time-dependent potential
static inline double totalEnergy(double time,
    const potential::BasePotential& pot, const BHParams& bh, double posvel[6])
{
    const coord::PosCar pos(posvel[0], posvel[1], posvel[2]);
    return pot.value(pos) + bh.potential(time, pos) +
        0.5 * (pow_2(posvel[3]) + pow_2(posvel[4]) + pow_2(posvel[5]));
}

/// compute the change in L_z due to the torque from the stellar potential for the orbit segment
/// on the interval of time [t1:t2] provided by the ODE solver
static double computeLztorque(
    const potential::BasePotential& potential, const math::BaseOdeSolver& sol, double t1, double t2)
{
    if(isZRotSymmetric(potential))
        return 0;
    // integrate the torque over the interval [t1:t2] using 2-point Gauss rule
    const double delta = 0.211324865; // (1-sqrt(1./3))/2
    double point[6];
    coord::GradCar grad;
    sol.getSol(t1 * delta + t2 * (1-delta), point);
    potential.eval(coord::PosCar(point[0], point[1], point[2]), NULL, &grad);
    double torque = point[0] * grad.dy - point[1] * grad.dx;
    sol.getSol(t2 * delta + t1 * (1-delta), point);
    potential.eval(coord::PosCar(point[0], point[1], point[2]), NULL, &grad);
    torque += point[0] * grad.dy - point[1] * grad.dx;
    if(isFinite(torque))
        return torque * 0.5 * (t2-t1);
    else {
        utils::msg(utils::VL_WARNING, "RuntimeBinary", "Cannot compute torque due to stellar potential");
        return 0;
    }
}

orbit::StepResult RuntimeBinary::processTimestep(
    const math::BaseOdeSolver& sol, const double tbegin, const double tend, double [])
{
    // first determine whether the particle experiences an encounter during this timestep
    double ptbegin[6], ptend[6];  // position/velocity at the beginning and the end of encounter
    sol.getSol(tbegin, ptbegin);  // initially assigned to the beginning/end of timestep
    sol.getSol(tend,   ptend);
    double r2begin = pow_2(ptbegin[0]) + pow_2(ptbegin[1]) + pow_2(ptbegin[2]);
    double r2end   = pow_2(ptend  [0]) + pow_2(ptend  [1]) + pow_2(ptend  [2]);
    double r2crit  = pow_2(bh.sma * BINARY_ENCOUNTER_RADIUS);
    if(r2begin >= r2crit && r2end >= r2crit)  // the entire timestep is outside the critical radius:
        return orbit::SR_CONTINUE;                   // no further action required

    // if during the timestep the particle spends some time inside the critical radius,
    // we need to determine exactly the time when it enters and exits the sphere with this radius
    // (it may well be the beginning or the end of the timestep)
    double tbeginEnc = tbegin, tendEnc = tend;

    // if the particle has crossed the critical radius during the timestep,
    // we need to find the exact time this happened
    if((r2end < r2crit) ^ (r2begin < r2crit)) {
        double tcross = math::findRoot(EncounterFinder(sol, r2crit), tbegin, tend, 1e-4);
        assert(tcross >= tbegin && tcross <= tend);
        if(r2begin >= r2crit) {
            tbeginEnc = tcross;
            sol.getSol(tcross, ptbegin);
        } else {
            tendEnc = tcross;
            sol.getSol(tcross, ptend);
        }
    }
    
    // the encounter has started during this timestep
    // if the particle was outside the critical radius at the beginning, but moved in,
    // or if this is the first timestep in the entire episode and it was already inside rcrit
    bool newEncounter =
        (r2end < r2crit && r2begin >= r2crit) ||
        (tbegin==0 && r2begin < r2crit);

    // if there are no recorded encounters for this orbit,
    // then this timestep must be a beginning of the first one
    assert(!encountersList.empty() || newEncounter);

    // compute the changes in particle energy and z-component of angular momentum during the time
    // that it spends inside the critical radius (it may be the entire timestep or its fraction)
    double Ebegin  = totalEnergy(tbeginEnc, pot, bh, ptbegin);
    double Eend    = totalEnergy(tendEnc,   pot, bh, ptend);
    double Lzbegin = ptbegin[0] * ptbegin[4] - ptbegin[1] * ptbegin[3];
    double Lzend   = ptend  [0] * ptend  [4] - ptend  [1] * ptend  [3];
    // if the stellar potential itself is non-axisymmetric, the z-component of angular momentum
    // also changes due to the torque from the stellar potential, and we need to take this into account
    double Lztorque = computeLztorque(pot, sol, tbeginEnc, tendEnc);

    // if the encounter started during the current timestep,
    // we add a new record to the list of encounters for this orbit,
    // otherwise we update the most recent encounter record
    if(newEncounter) {
        double Lbegin = sqrt(pow_2(ptbegin[1] * ptbegin[5] - ptbegin[2] * ptbegin[4]) +
            pow_2(Lzbegin) + pow_2(ptbegin[2] * ptbegin[3] - ptbegin[0] * ptbegin[5]) );
        encountersList.push_back(BinaryEncounterData(tbeginEnc, Ebegin, Lbegin));
    }
    BinaryEncounterData& enc = encountersList.back();
    enc.Tlength += tendEnc - tbeginEnc;
    enc.deltaE  += Eend  - Ebegin;
    enc.deltaLz += Lzend - Lzbegin - Lztorque;
    enc.costheta = ptend[5] / sqrt(pow_2(ptend[3]) + pow_2(ptend[4]) + pow_2(ptend[5]));
    enc.phi      = atan2(ptend[4], ptend[3]);

    return orbit::SR_CONTINUE;
}


RagaTaskBinary::RagaTaskBinary(
    const ParamsBinary& _params,
    const particles::ParticleArrayCar& _particles,
    const potential::PtrPotential& _ptrPot,
    BHParams& _bh)
:
    params(_params),
    particles(_particles),
    ptrPot(_ptrPot),
    bh(_bh),
    firstEpisode(true)
{
    utils::msg(utils::VL_DEBUG, "RagaTaskBinary",
        "Initial semimajor axis=" + utils::toString(bh.sma) +
        ", eccentricity=" + utils::toString(bh.ecc));        
}

orbit::PtrRuntimeFnc RagaTaskBinary::createRuntimeFnc(unsigned int particleIndex)
{
    return orbit::PtrRuntimeFnc(new RuntimeBinary(*ptrPot, bh, encounters[particleIndex]));
}

void RagaTaskBinary::startEpisode(double timeStart, double length)
{
    episodeStart  = timeStart;
    episodeLength = length;
    encounters.assign(particles.size(), BinaryEncounterList());  // reserve room for storing the encounters
    if(!params.outputFilename.empty() && firstEpisode) {
        std::ofstream strm(params.outputFilename.c_str());
        strm << "time    \tsemimajor_axis\teccentricity\tBH_mass \tq(mass_ratio)\t"
            "hardening_star\thardening_gw\n" +
            utils::pp(timeStart, 10) + '\t' +
            utils::pp(bh.sma,    10) + '\t' +
            utils::pp(bh.ecc,    10) + '\t' +
            utils::pp(bh.mass,   10) + '\t' +
            utils::pp(bh.q,      10) + "\t0       \t0\n";
        strm.close();
        strm.open((params.outputFilename+"_enc").c_str());
        strm << "timeStart    duration Ebegin   Lbegin   deltaE   deltaLz  costheta  phi mass     index\n";
    }
    firstEpisode = false;
}

/** Helper routine to evolve the orbital parameters of a binary black hole which
    changes its energy and eccentricity due to a combination of stellar-dynamical hardening and
    gravitational-wave (GW) emission.
    \param[in]  bh  are the initial parameters of the binary;
    \param[in]  H = d(1/a)/dt is the stellar-dynamical hardening rate;
    \param[in]  K = d(e^2)/dt is the stellar-dynamical eccentricity growth rate;
    \param[in]  speedOfLight  is the speed of light in model units, which determines the GW loss rate;
    \param[in,out]  maxTime   is the duration of episode (on input), and if the binary coalesces
    due to GW emission on a shorter time than maxTime, then this variable will contain
    the coalescence time on output.
    \return  the new set of binary orbit parameters.
*/
static BHParams evolveBH(const BHParams& oldbh, double H, double K, double speedOfLight, double& maxTime)
{
    BHParams bh = oldbh;
    bh.phase = math::wrapAngle(bh.phase + sqrt(bh.mass/pow_3(bh.sma)) * maxTime);
    double timePassed = 0;
    // evolve semimajor axis (sma) and eccentricity (ecc), in several sub-steps if necessary
    do {
        // compute the hardening and eccentricity growth rate due to GW emission,
        // given the current values of semimajor axis and eccentricity
        double mult = pow_3(bh.mass) * math::pow(speedOfLight * bh.sma, -5) * bh.sma *
            bh.q / pow_2(1+bh.q) * std::pow(1 - pow_2(bh.ecc), -3.5);   // common multiple
        double Hgw  =  mult * (192 + pow_2(bh.ecc) * (584 + 74 * pow_2(bh.ecc))) / 15 / bh.sma;
        double Kgw  = -mult * (608 + pow_2(bh.ecc) * 242) / 15 * (1 - pow_2(bh.ecc)) * pow_2(bh.ecc);
        double tstep = 0.01 * fmin(
            fmin(1 - pow_2(bh.ecc), pow_2(bh.ecc) + 1e-3) / fabs(K + Kgw),
            1 / bh.sma / fabs(H + Hgw) );
         if(!isFinite(tstep) || tstep > maxTime - timePassed)
             tstep = maxTime - timePassed;
         if(0.25 / bh.sma / Hgw < maxTime - timePassed  &&  bh.ecc < 0.01) {
             // imminent coalescence due to GW emission
             timePassed += 0.25 / bh.sma / Hgw;
             bh.sma  = 0;
             bh.ecc  = 0;
             maxTime = timePassed;  // will exit the loop and record the exact coalescence time
         } else {
             bh.sma  = 1 / (1 / bh.sma + tstep * (H + Hgw));
             bh.ecc  = fmin(0.999, sqrt(fmax(0, pow_2(bh.ecc) + tstep * (K + Kgw))));
             timePassed += tstep;
         }
    } while(timePassed < maxTime * (1-1e-10));
    return bh;
}

void RagaTaskBinary::finishEpisode()
{
    // if the binary has already coalesced, nothing happens anymore
    if(bh.sma == 0 || bh.mass == 0)
        return;
    assert(particles.size() == encounters.size());

    // assemble the list of all encounters sorted by start time
    std::vector<std::pair<double, std::pair<size_t, size_t> > > allEncounters;
    // sum up the total energy and angular momentum gained by all particles in the simulation;
    // these changes must be reciprocally imposed on the binary BH
    unsigned int numEnc=0, numPart=0;
    double deltaE=0, deltaLz=0;
    for(size_t ip=0; ip<particles.size(); ip++) {
        double mass = particles[ip].second;
        if(mass == 0 || encounters[ip].empty())
            continue;   // nothing happened to this particle
        for(size_t ie=0; ie<encounters[ip].size(); ie++) {
            deltaE  += mass * encounters[ip][ie].deltaE;
            deltaLz += mass * encounters[ip][ie].deltaLz;
            allEncounters.push_back(std::pair<double, std::pair<size_t, size_t> >(
                encounters[ip][ie].Tbegin, std::pair<size_t, size_t>(ip, ie) ) );
        }
        numEnc += encounters[ip].size();
        numPart++;
    }

    // compute the hardening rate H=d(1/a)/dt and eccentricity growth rate K=d(e^2)/dt in this episode
    double mult = 2 * pow_2(1+bh.q) / bh.q / pow_2(bh.mass) / episodeLength;  // common factor
    double H = mult * deltaE;
    double K = mult *
        ( sqrt((1 - pow_2(bh.ecc)) * bh.mass / bh.sma) * deltaLz - (1 - pow_2(bh.ecc)) * deltaE * bh.sma);

    // gradually modify the orbital parameters of the binary taking into account both
    // the stellar-dynamical and the gravitational-wave-induced hardening and eccentricity evolution
    double oldsma = bh.sma;
    bh = evolveBH(bh, H, K, params.speedOfLight, episodeLength);

    // infer the average GW-induced hardening rate during this episode
    double Hgw = isFinite(params.speedOfLight) ? (1/bh.sma - 1/oldsma) / episodeLength - H : 0;

    utils::msg(utils::VL_MESSAGE, "RagaTaskBinary",
        "By time " + utils::toString(episodeStart+episodeLength) + ", " +
        utils::toString(numPart) + " particles had " +
        utils::toString(numEnc)  + " encounters; "
        "hardening rate=" + utils::toString(H) + '+' + utils::toString(Hgw) +
        ", new a=" + utils::toString(bh.sma) +
        ", ecc=" + utils::toString(bh.ecc));
    
    // record the new parameters to the output file
    if(!params.outputFilename.empty()) {
        std::ofstream strm(params.outputFilename.c_str(), std::ios_base::app);
        strm <<
            utils::pp(episodeStart+episodeLength, 10) + '\t' +
            utils::pp(bh.sma,  10) + '\t' +
            utils::pp(bh.ecc,  10) + '\t' +
            utils::pp(bh.mass, 10) + '\t' +
            utils::pp(bh.q,    10) + '\t' +
            utils::pp(H,       10) + '\t' +
            utils::pp(Hgw,     10) + '\n';
        strm.close();
        std::sort(allEncounters.begin(), allEncounters.end());
        strm.open((params.outputFilename+"_enc").c_str(), std::ios_base::app);
        for(size_t k=0; k<allEncounters.size(); k++) {
            size_t ip = allEncounters[k].second.first, ie = allEncounters[k].second.second;
            const BinaryEncounterData& enc = encounters[ip][ie];
            strm <<
            utils::pp(episodeStart + enc.Tbegin,  12) + ' ' +
            utils::pp(enc.Tlength, 8) + ' ' +
            utils::pp(enc.Ebegin,  8) + ' ' +
            utils::pp(enc.Lbegin,  8) + ' ' +
            utils::pp(enc.deltaE,  8) + ' ' +
            utils::pp(enc.deltaLz, 8) + ' ' +
            utils::pp(enc.costheta,6) + ' ' +
            utils::pp(enc.phi,     6) + ' ' +
            utils::pp(particles[ip].second, 8) /*mass*/ + ' ' +
            utils::toString(ip) + '\n';
        }
    }
}

}  // namespace raga