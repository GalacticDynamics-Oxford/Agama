#include "raga_core.h"
#include "utils.h"
#include "utils_config.h"
#include "particles_io.h"
#include "potential_multipole.h"
#include "potential_factory.h"
#include <fstream>
#include <stdexcept>
#include <ctime>

namespace raga {

void computeTotalEnergyModel(
    const potential::BasePotential& pot,
    const BHParams& bh,
    const particles::ParticleArrayCar& particles,
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
    int nbody = particles.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:Etot,Esum)
#endif
    for(int ip=0; ip<nbody; ip++) {
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
    const particles::ParticleArrayCar& particles,
    const double time,
    const std::string& taskName)
{
    double Etot, Esum;
    computeTotalEnergyModel(pot, bh, particles, 0, Etot, Esum);
    utils::msg(utils::VL_MESSAGE, "RagaEpisode",
        taskName + " done at t=" + utils::toString(time)+
        ", total energy=" + utils::toString(Etot) + ", sumE=" + utils::toString(Esum));
    return
        utils::pp(time, 10) + '\t' + taskName + '\t' +
        utils::pp(Etot, 12) + '\t' +
        utils::pp(Esum, 12) + '\t' +
        utils::pp(pot.value(coord::PosCar(0,0,0)), 12) + '\n';
}

RagaCore::RagaCore(const utils::KeyValueMap& config)
{
    // parse the configuration and check the validity of parameters
    loadSettings(config);

    // read input snapshot
    particles = particles::readSnapshot(paramsRaga.fileInput);
    if(particles.size()==0)
        throw std::runtime_error("Error reading initial snapshot "+paramsRaga.fileInput);

    ptrPot = potential::Multipole::create(particles, paramsPotential.symmetry,
        paramsPotential.lmax, paramsPotential.lmax, paramsPotential.gridSizeR);

    // initialize various tasks, depending on the parameters
    // Order *IS* important!
    if(paramsLosscone.captureRadius[0]>0 && bh.mass>0) 
    {   // capture of stars by a central black hole
        tasks.push_back(PtrRagaTask(new RagaTaskLosscone(
            paramsLosscone, particles, bh)));
    }
    if(bh.sma>0 && bh.q>0 && bh.mass>0)
    {   // binary black hole evolution
        tasks.push_back(PtrRagaTask(new RagaTaskBinary(
            paramsBinary, particles, ptrPot, bh)));
    }
    if(paramsRaga.updatePotential)
    {   // potential recomputation
        tasks.push_back(PtrRagaTask(new RagaTaskPotential(
            paramsPotential, particles, ptrPot)));
    }
    if(paramsRelaxation.relaxationRate>0)
    {   // relaxation (velocity perturbation)
        tasks.push_back(PtrRagaTask(new RagaTaskRelaxation(
            paramsRelaxation, particles, ptrPot, bh)));
    }
    if(paramsTrajectory.outputInterval>0 && !paramsTrajectory.outputFilename.empty()) 
    {   // output trajectories of particles
        tasks.push_back(PtrRagaTask(new RagaTaskTrajectory(
            paramsTrajectory, particles)));
    }
}

void RagaCore::run()
{
    if(!paramsRaga.fileLog.empty()) {
        std::ofstream strmLog(paramsRaga.fileLog.c_str());
        strmLog << "Time    \tTaskName\tTotalEnergy\tSumEnergy\tPhi_star(0)\n" +
            printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, "Initialization");
    }
    while(paramsRaga.timeCurr < paramsRaga.timeEnd)
        doEpisode();
}

void RagaCore::doEpisode()
{
    utils::msg(utils::VL_MESSAGE, "RagaEpisode",
        "Starting episode at time " + utils::toString(paramsRaga.timeCurr));
    std::time_t wallClockStartEpisode = std::time(NULL);

    int numtasks = tasks.size();
    double episodeLength = paramsRaga.episodeLength;  // duration of this episode
    for(int task=0; task<numtasks; task++)
        tasks[task]->startEpisode(paramsRaga.timeCurr, episodeLength);

    // loop over the particles
    int nbody = particles.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for(int index=0; index<nbody; index++) {
        if(particles.mass(index) != 0) {   // run only non-zero-mass particles
            RuntimeFncArray tsfnc(numtasks);
            for(int task=0; task<numtasks; task++)
                tsfnc[task] = tasks[task]->createRuntimeFnc(index);
            particles[index].first = integrateOrbit(*ptrPot, bh,
                particles.point(index), episodeLength, tsfnc, paramsRaga.integratorAccuracy);
        }
    }   // end parallel for

    double wallClockDurationEpisode = std::max<double>(1., std::time(NULL) - wallClockStartEpisode);
    utils::msg(utils::VL_MESSAGE, "RagaEpisode",
        utils::toString(nbody) + " particles, " +
        utils::toString(nbody / wallClockDurationEpisode) + " orbits/s");

    std::ofstream strmLog;
    if(!paramsRaga.fileLog.empty())
        strmLog.open(paramsRaga.fileLog.c_str(), std::ios::app);

    paramsRaga.timeCurr += episodeLength;
    strmLog << printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, "Episode ");

    // finish episode by calling corresponding function for each task
    for(int task=0; task<numtasks; task++) {
        tasks[task]->finishEpisode();
        strmLog << printLog(*ptrPot, bh, particles, paramsRaga.timeCurr, tasks[task]->name());
    }
}

void RagaCore::loadSettings(const utils::KeyValueMap& config)
{
    // a header line written in the output file contains all parameters from the ini file
    paramsTrajectory.header = paramsRelaxation.header = "Raga " + config.dumpSingleLine();

    // parameters of the stellar potential and the central black hole(s)
    paramsPotential.symmetry  = potential::getSymmetryTypeByName(config.getString("Symmetry"));
    paramsPotential.gridSizeR = config.getInt("gridSizeR", 25);
    paramsPotential.lmax      = config.getInt("lmax", 0);
    bh.mass  = config.getDouble("Mbh", 0);
    bh.q     = config.getDouble("binary_q", 0);
    bh.sma   = config.getDouble("binary_sma", 0);
    bh.ecc   = config.getDouble("binary_ecc", 0);
    bh.phase = config.getDouble("binary_phase", 0);

    // global parameters of the simulation
    paramsRaga.fileInput = config.getString("fileInput");
    if(paramsRaga.fileInput=="" || !utils::fileExists(paramsRaga.fileInput))
        throw std::runtime_error("Input file "+paramsRaga.fileInput+" does not exist ([Raga]/fileInput)");
    paramsRaga.integratorAccuracy  = config.getDouble("accuracy", 1e-8);
    paramsRaga.fileLog  = config.getString("fileLog", paramsRaga.fileInput+".log");
    paramsRaga.timeEnd  = config.getDouble("timeTotal");
    paramsRaga.timeCurr = config.getDouble("timeInit");
    paramsRaga.episodeLength = config.getDouble("episodeLength", paramsRaga.timeEnd-paramsRaga.timeCurr);
    if(paramsRaga.timeEnd <= paramsRaga.timeCurr || paramsRaga.episodeLength <= 0)
        throw std::runtime_error("Total simulation time and episode length should be positive "
            "([Raga]/timeTotal, [Raga]/episodeLength)");
    paramsRaga.updatePotential = config.getBool("updatePotential", true);
    if(!paramsRaga.updatePotential)
        utils::msg(utils::VL_MESSAGE, "RagaLoadSettings",
            "Potential update is disabled ([Raga]/updatePotential)");
    
    // parameters of individual tasks
    paramsTrajectory.outputFilename = config.getString("fileOutput");
    paramsPotential. outputFilename = config.getString("fileOutputPotential");
    paramsRelaxation.outputFilename = config.getString("fileOutputRelaxation");
    paramsLosscone.  outputFilename = config.getString("fileOutputLosscone");
    paramsBinary.    outputFilename = config.getString("fileOutputBinary");
    paramsTrajectory.outputInterval = paramsPotential.outputInterval = 
    paramsRelaxation.outputInterval = config.getDoubleAlt("outputInterval", "timestepOutput");
    paramsTrajectory.outputFormat   = config.getString("fileOutputFormat", "t");
    if(paramsTrajectory.outputInterval<=0 || paramsTrajectory.outputFilename.empty())
        utils::msg(utils::VL_MESSAGE, "RagaLoadSettings",
            "Warning, output snapshots disabled ([Raga]/timestepOutput, [Raga]/fileOutput)");
    paramsPotential.numSamplesPerEpisode = paramsRelaxation.numSamplesPerEpisode = 
        std::max(1, config.getInt("numSamplesPerEpisode", 1));
    paramsRelaxation.relaxationRate   = config.getDouble("relaxationRate", 0);
    paramsRelaxation.gridSizeDF       = config.getInt("gridSizeDF", 25);
    paramsLosscone.captureRadius[0]   = config.getDoubleAlt("captureRadius", "captureRadius1", 0);
    paramsLosscone.captureRadius[1]   = config.getDouble("captureRadius2", paramsLosscone.captureRadius[0]);
    paramsLosscone.captureMassFraction= config.getDouble("captureMassFraction", 1);
    paramsBinary.speedOfLight         = config.getDouble("speedOfLight", 0);
    if(paramsBinary.speedOfLight == 0)
        paramsBinary.speedOfLight = INFINITY;
    if((bh.sma>0 || paramsLosscone.captureRadius[0]>0) && bh.mass==0)
        // binary semimajor axis or capture radius were set without a black hole, so it has no effect
        utils::msg(utils::VL_MESSAGE, "RagaLoadSettings", "No central black hole!");
}

}  // namespace
