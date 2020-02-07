#include "raga_core.h"
#include "utils.h"
#include "utils_config.h"
#include "potential_multipole.h"
#include "potential_factory.h"
#include "math_core.h"
#include "particles_io.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <ctime>

namespace raga {

void computeTotalEnergyModel(
    const potential::BasePotential& pot,
    const BHParams& bh,
    const particles::ParticleArrayAux& particles,
    double time, double& resultEtot, double& resultEsum)
{
    double Etot=0, Esum=0;
    // potential at origin excluding BH
    double stellarPotentialCenter = pot.value(coord::PosCyl(0,0,0));
    // add the energy of BH in the stellar potential 
    Etot += bh.mass * stellarPotentialCenter * 0.5;
    Esum += bh.mass * stellarPotentialCenter;
    // add the internal energy of binary BH
    if(bh.sma>0) {
        double Ebin = 0.5 * pow_2(bh.mass) / bh.sma *
            bh.q / pow_2(1+bh.q);
        Etot -= Ebin;
        Esum -= Ebin*2;
    }
    // add energies of all particles
    ptrdiff_t nbody = particles.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:Etot,Esum)
#endif
    for(ptrdiff_t ip=0; ip<nbody; ip++) {
        const coord::PosVelCar& point = particles.point(ip);
        double Epot = pot.value(point) + bh.potential(time, point);
        double Ekin = (pow_2(point.vx) + pow_2(point.vy) + pow_2(point.vz)) * 0.5;
        Etot += particles.mass(ip) * (Ekin+Epot*0.5);
        Esum += particles.mass(ip) * (Ekin+Epot);
    }
    resultEtot = Etot;
    resultEsum = Esum;
}

std::string printLog(
    const potential::BasePotential& pot,
    const BHParams& bh,
    const particles::ParticleArrayAux& particles,
    const double time,
    const std::string& taskName,
    /*output diagnostic quantities*/ double &Ekin, double &Epot)
{
    double Etot, Esum;
    computeTotalEnergyModel(pot, bh, particles, 0, Etot, Esum);
    Ekin = 2*Etot-Esum;
    Epot =  -Etot+Esum;
    utils::msg(utils::VL_MESSAGE, "Raga",
        taskName + " done at t=" + utils::toString(time)+
        ", total energy=" + utils::toString(Etot) + ", sumE=" + utils::toString(Esum));
    return
        utils::pp(time, 10) + '\t' + taskName + '\t' +
        utils::pp(Etot, 12) + '\t' +
        utils::pp(Esum, 12) + '\t' +
        utils::pp(pot.value(coord::PosCar(0,0,0)), 12) + '\t' +
        utils::pp(particles.totalMass(), 12) +'\n';
}

void RagaCore::initPotentialFromParticles()
{
    if(!paramsRaga.initPotentialExternal) {
        // create a Multipole potential from input particles
        utils::msg(utils::VL_MESSAGE, "Raga", "Creating multipole potential from "+
            utils::toString(particles.size())+" particles");
        ptrPot = potential::Multipole::create(particles, paramsPotential.symmetry,
            paramsPotential.lmax, paramsPotential.lmax, paramsPotential.gridSizeR,
            paramsPotential.rmin, paramsPotential.rmax);
    }
    std::ofstream strmLog;
    if(!paramsRaga.fileLog.empty())
        strmLog.open(paramsRaga.fileLog.c_str(), std::ios::app);
    strmLog << printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, "Initialization ",
        /*output*/Ekin, Epot);
}

void RagaCore::doEpisode(double episodeLength)
{
    if(!ptrPot)
        throw std::runtime_error("Raga: potential is not initialized");
    if(!(episodeLength > 0))
        return;
    utils::msg(utils::VL_MESSAGE, "Raga",
        "Starting episode at time " + utils::toString(paramsRaga.timeCurr));
    std::time_t wallClockStartEpisode = std::time(NULL);

    int numtasks = tasks.size();
    for(int task=0; task<numtasks; task++)
        tasks[task]->startEpisode(paramsRaga.timeCurr, episodeLength);

    // compute energies of all particles
    ptrdiff_t nbody = particles.size();
    std::vector< std::pair<double, ptrdiff_t> > particleOrder(nbody);
    for(ptrdiff_t ip=0; ip<nbody; ip++) {
        const coord::PosVelCar& point = particles.point(ip);
        double E = ptrPot->value(point) + bh.potential(paramsRaga.timeCurr, point) +
            (pow_2(point.vx) + pow_2(point.vy) + pow_2(point.vz)) * 0.5;
        particleOrder[ip].first = E;
        particleOrder[ip].second = ip;
    }
    // sort particles in energy, so that the most tightly bound ones
    // are processed first, because it may take longer
    std::sort(particleOrder.begin(), particleOrder.end());

    orbit::OrbitIntParams orbitIntParams;
    orbitIntParams.accuracy = paramsRaga.accuracy;
    orbitIntParams.maxNumSteps = paramsRaga.maxNumSteps;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(ptrdiff_t ip=0; ip<nbody; ip++) {
        ptrdiff_t index = particleOrder[ip].second;
        if(particles.mass(index) != 0) {   // run only non-zero-mass particles
            orbit::RuntimeFncArray timestepFncs(numtasks);
            for(int task=0; task<numtasks; task++)
                timestepFncs[task] = tasks[task]->createRuntimeFnc(index);
            particles[index].first = particles::ParticleAux(
                /* replace the initial position/velocity with that at the end of the episode */
                orbit::integrate(
                particles.point(index), episodeLength,
                RagaOrbitIntegrator(*ptrPot, bh),
                timestepFncs, orbitIntParams),
                /* keep the original extended particle attributes */
                particles.point(index).stellarMass,
                particles.point(index).stellarRadius);
        }
    }   // end parallel for

    double wallClockDurationEpisode = std::max(1., difftime(std::time(NULL), wallClockStartEpisode));
    utils::msg(utils::VL_MESSAGE, "Raga",
        utils::toString(nbody) + " particles, " +
        utils::toString(nbody / wallClockDurationEpisode) + " orbits/s");

    std::ofstream strmLog;
    if(!paramsRaga.fileLog.empty())
        strmLog.open(paramsRaga.fileLog.c_str(), std::ios::app);

    paramsRaga.timeCurr += episodeLength;
    strmLog << printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, "Episode        ",
        /*output*/Ekin, Epot);

    // finish episode by calling corresponding function for each task
    for(int task=0; task<numtasks; task++) {
        tasks[task]->finishEpisode();
        strmLog << printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, tasks[task]->name(),
            /*output*/Ekin, Epot);
    }
}

void RagaCore::init(const utils::KeyValueMap& config)
{
    // a header line written in the output file contains all parameters from the ini file
    paramsTrajectory.header = paramsRelaxation.header = "Raga " + config.dumpSingleLine();

    // parameters of the stellar potential and the central black hole(s)
    paramsPotential.symmetry  = potential::getSymmetryTypeByName(config.getString("Symmetry"));
    paramsPotential.gridSizeR = config.getInt   ("gridSizeR", paramsPotential.gridSizeR);
    paramsPotential.rmin      = config.getDouble("rmin", paramsPotential.rmin);
    paramsPotential.rmax      = config.getDouble("rmax", paramsPotential.rmax);
    paramsPotential.lmax      = config.getInt   ("lmax", paramsPotential.lmax);
    bh.mass  = config.getDouble("Mbh",          bh.mass);
    bh.q     = config.getDouble("binary_q",     bh.q);
    bh.sma   = config.getDouble("binary_sma",   bh.sma);
    bh.ecc   = config.getDouble("binary_ecc",   bh.ecc);
    bh.phase = config.getDouble("binary_phase", bh.phase);

    // global parameters of the simulation
    paramsRaga.accuracy       = config.getDouble("accuracy",    orbit::OrbitIntParams().accuracy);
    paramsRaga.maxNumSteps    = config.getDouble("maxNumSteps", orbit::OrbitIntParams().maxNumSteps);
    paramsRaga.fileInput      = config.getString("fileInput");
    paramsRaga.fileLog        = config.getString("fileLog",
        paramsRaga.fileInput.empty() ? "" : paramsRaga.fileInput+".log");
    paramsRaga.timeEnd        = config.getDouble("timeTotal", paramsRaga.timeEnd);
    paramsRaga.timeCurr       = config.getDouble("timeInit", paramsRaga.timeCurr);
    paramsRaga.episodeLength  = config.getDouble("episodeLength", paramsRaga.timeEnd-paramsRaga.timeCurr);
    paramsRaga.updatePotential= config.getBool  ("updatePotential", paramsRaga.updatePotential);
    if(!paramsRaga.updatePotential)
        utils::msg(utils::VL_MESSAGE, "Raga", "Potential update is disabled ([Raga]/updatePotential)");
    if(!paramsRaga.fileLog.empty()) {
        std::ofstream strmLog(paramsRaga.fileLog.c_str());
        strmLog << "#Time   \tTaskName\tTotalEnergy\tSumEnergy\tPhi_star(0)\tTotalMass\n";
    }

    // parameters of individual tasks
    paramsTrajectory.outputFilename = config.getString("fileOutput");
    paramsPotential. outputFilename = config.getString("fileOutputPotential");
    paramsRelaxation.outputFilename = config.getString("fileOutputRelaxation");
    paramsLosscone.  outputFilename = config.getString("fileOutputLosscone");
    paramsBinary.    outputFilename = config.getString("fileOutputBinary");
    paramsTrajectory.outputInterval =
    paramsPotential. outputInterval =
    paramsRelaxation.outputInterval = config.getDouble("outputInterval");
    paramsTrajectory.outputFormat   = config.getString("fileOutputFormat", paramsTrajectory.outputFormat);
    if(paramsTrajectory.outputInterval<=0 || paramsTrajectory.outputFilename.empty())
        utils::msg(utils::VL_MESSAGE, "Raga",
            "Snapshot output is disabled ([Raga]/outputInterval, [Raga]/fileOutput)");
    paramsPotential. numSamplesPerEpisode =
    paramsRelaxation.numSamplesPerEpisode =
        std::max(1, config.getInt("numSamplesPerEpisode", paramsRelaxation.numSamplesPerEpisode));
    paramsRelaxation.coulombLog       = config.getDouble("coulombLog", paramsRelaxation.coulombLog);
    paramsRelaxation.gridSizeDF       = config.getInt   ("gridSizeDF", paramsRelaxation.gridSizeDF);
    paramsLosscone.captureMassFraction= config.getDouble("captureMassFraction", paramsLosscone.captureMassFraction);
    paramsLosscone.speedOfLight       =
    paramsBinary.  speedOfLight       = config.getDouble("speedOfLight", paramsBinary.speedOfLight);
    if((bh.sma>0 || paramsLosscone.speedOfLight<INFINITY) && bh.mass==0)
        // binary semimajor axis or speed of light were set without a black hole, so it has no effect
        utils::msg(utils::VL_MESSAGE, "Raga", "No central black hole!");

    // initialize various tasks, depending on the parameters
    tasks.clear();
    // Order *IS* important!
    if(bh.mass>0)
    {   // capture of stars by a central black hole and regularization of eccentric orbits near pericenter
        tasks.push_back(PtrRagaTask(new RagaTaskLosscone  (paramsLosscone,   particles, bh)));
    }
    if(bh.sma>0 && bh.q>0 && bh.mass>0)
    {   // binary black hole evolution
        tasks.push_back(PtrRagaTask(new RagaTaskBinary    (paramsBinary,     particles, ptrPot, bh)));
    }
    if(paramsRaga.updatePotential)
    {   // recomputation of stellar potential
        tasks.push_back(PtrRagaTask(new RagaTaskPotential (paramsPotential,  particles, ptrPot)));
    }
    if(paramsRelaxation.coulombLog>0)
    {   // relaxation (velocity perturbation)
        tasks.push_back(PtrRagaTask(new RagaTaskRelaxation(paramsRelaxation, particles, ptrPot, bh)));
    }
    if(!paramsTrajectory.outputFilename.empty())
    {   // output trajectories of particles
        tasks.push_back(PtrRagaTask(new RagaTaskTrajectory(paramsTrajectory, particles)));
    }

    if(!paramsRaga.fileInput.empty())
    {   // if the input file is provided, load the particle snapshot
        particles = particles::readSnapshot(paramsRaga.fileInput);
    }

    if(config.contains("type") || config.contains("file"))
    {   // create a user-defined potential
        paramsRaga.initPotentialExternal = true;
        if(particles.size() > 0) {
            try {
                // if potential type is Multipole or CylSpline, construct it from the input particles
                ptrPot = potential::createPotential(config, particles);
                utils::msg(utils::VL_MESSAGE, "Raga", "Creating "+std::string(ptrPot->name())+
                    " potential from "+utils::toString(particles.size())+" particles and erasing them");
                particles.data.clear();
            }
            catch(std::exception&) {
                ptrPot = potential::createPotential(config);
                utils::msg(utils::VL_MESSAGE, "Raga", "Creating analytic potential ignoring "+
                   utils::toString(particles.size())+" particles");
            }
        } else {
            ptrPot = potential::createPotential(config);
            utils::msg(utils::VL_MESSAGE, "Raga", "Creating analytic potential - no particles");
        }
    } else {
        // create a self-consistent potential from the input snapshot, if it was provided
        paramsRaga.initPotentialExternal = false;
        if(particles.size() > 0) {
            initPotentialFromParticles();
        } else {
            utils::msg(utils::VL_MESSAGE, "Raga", "No potential created in loadSettings - no particles");
        }
    }
}

}  // namespace
