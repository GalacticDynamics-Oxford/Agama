#include "raga_core.h"
#include "utils.h"
#include "utils_config.h"
#include "potential_multipole.h"
#include "potential_cylspline.h"
#include "potential_factory.h"
#include "potential_composite.h"
#include "math_core.h"
#include "particles_io.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>

namespace raga {

void computeTotalEnergyModel(
    const potential::BasePotential& pot,
    const potential::KeplerBinaryParams& bh,
    const particles::ParticleArrayAux& particles,
    double time, double& resultEtot, double& resultEsum)
{
    double Etot=0, Esum=0;
    // potential at origin excluding BH
    double stellarPotentialCenter = pot.value(coord::PosCyl(0,0,0));
    // add the energy of BH in the stellar potential
    if(bh.mass>0) {
        Etot += bh.mass * stellarPotentialCenter * 0.5;
        Esum += bh.mass * stellarPotentialCenter;
    }
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
        double Epot = pot.value(point, time) + bh.potential(point, time);
        double Ekin = (pow_2(point.vx) + pow_2(point.vy) + pow_2(point.vz)) * 0.5;
        Etot += particles.mass(ip) * (Ekin+Epot*0.5);
        Esum += particles.mass(ip) * (Ekin+Epot);
    }
    resultEtot = Etot;
    resultEsum = Esum;
}

std::string printLog(
    const potential::BasePotential& pot,
    const potential::KeplerBinaryParams& bh,
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
    utils::Timer timer;

    int numtasks = tasks.size();
    for(int task=0; task<numtasks; task++)
        tasks[task]->startEpisode(paramsRaga.timeCurr, episodeLength);

    // compute energies of all particles
    ptrdiff_t nbody = particles.size();
    std::vector< std::pair<double, ptrdiff_t> > particleOrder(nbody);
    for(ptrdiff_t ip=0; ip<nbody; ip++) {
        const coord::PosVelCar& point = particles.point(ip);
        double E = ptrPot->value(point, paramsRaga.timeCurr) +
            bh.potential(point, paramsRaga.timeCurr) +
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

    // if needed, construct a composite potential (stars + BH)
    std::vector<potential::PtrPotential> potComponents(bh.mass!=0 ? 2 : 1);
    potComponents[0] = ptrPot;  // stellar potential
    if(bh.mass!=0) potComponents[1].reset(new potential::KeplerBinary(bh));  // BH potential
    potential::PtrPotential ptrTotalPot = bh.mass!=0 ?
        potential::PtrPotential(new potential::Composite(potComponents)) :
        ptrPot;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1)
#endif
    for(ptrdiff_t ip=0; ip<nbody; ip++) {
        ptrdiff_t index = particleOrder[ip].second;
        if(particles.mass(index) != 0) {   // run only non-zero-mass particles
            orbit::OrbitIntegrator<coord::Car> orbint(*ptrTotalPot, /*Omega*/0, orbitIntParams);
            for(int task=0; task<numtasks; task++)
                tasks[task]->createRuntimeFnc(orbint, index);
            orbint.init(particles.point(index));
            coord::PosVelCar endposvel = orbint.run(episodeLength);
            particles[index].first = particles::ParticleAux(
                /* replace the initial position/velocity with that at the end of the episode */
                endposvel,
                /* keep the original extended particle attributes */
                particles.point(index).stellarMass,
                particles.point(index).stellarRadius);
        }
    }   // end parallel for

    utils::msg(utils::VL_MESSAGE, "Raga",
        utils::toString(nbody) + " particles, " +
        utils::toString(nbody / timer.deltaSeconds()) + " orbits/s");

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

// note that "config" is passed by value, since it is modified inside this method
void RagaCore::init(utils::KeyValueMap config)
{
    // a header line written in the output file contains all parameters from the ini file
    paramsTrajectory.header = paramsRelaxation.header = "Raga " + config.dumpSingleLine();

    // parameters of the stellar potential and the central black hole(s)
    paramsPotential.symmetry  = potential::getSymmetryTypeByName(config.getString("Symmetry"));
    paramsPotential.gridSizeR = config.getInt   ("gridSizeR", paramsPotential.gridSizeR);
    paramsPotential.rmin      = config.getDouble("rmin", paramsPotential.rmin);
    paramsPotential.rmax      = config.getDouble("rmax", paramsPotential.rmax);
    paramsPotential.lmax      = config.getInt   ("lmax", paramsPotential.lmax);
    bh.mass  = config.popDouble("Mbh",          bh.mass);
    bh.q     = config.popDouble("binary_q",     bh.q);
    bh.sma   = config.popDouble("binary_sma",   bh.sma);
    bh.ecc   = config.popDouble("binary_ecc",   bh.ecc);
    bh.phase = config.popDouble("binary_phase", bh.phase);

    // global parameters of the simulation
    paramsRaga.accuracy       = config.popDouble("accuracy",    orbit::OrbitIntParams().accuracy);
    paramsRaga.maxNumSteps    = config.popDouble("maxNumSteps", orbit::OrbitIntParams().maxNumSteps);
    paramsRaga.fileInput      = config.popString("fileInput");
    paramsRaga.fileLog        = config.popString("fileLog",
        paramsRaga.fileInput.empty() ? "" : paramsRaga.fileInput+".log");
    paramsRaga.timeEnd        = config.popDouble("timeTotal", paramsRaga.timeEnd);
    paramsRaga.timeCurr       = config.popDouble("timeInit", paramsRaga.timeCurr);
    paramsRaga.episodeLength  = config.popDouble("episodeLength", paramsRaga.timeEnd-paramsRaga.timeCurr);
    paramsRaga.updatePotential= config.popBool  ("updatePotential", paramsRaga.updatePotential);
    if(!paramsRaga.updatePotential)
        utils::msg(utils::VL_MESSAGE, "Raga", "Potential update is disabled ([Raga]/updatePotential)");
    if(!paramsRaga.fileLog.empty()) {
        std::ofstream strmLog(paramsRaga.fileLog.c_str());
        strmLog << "#Time   \tTaskName\tTotalEnergy\tSumEnergy\tPhi_star(0)\tTotalMass\n";
    }

    // parameters of individual tasks
    paramsTrajectory.outputFilename = config.popString("fileOutput");
    paramsPotential. outputFilename = config.popString("fileOutputPotential");
    paramsRelaxation.outputFilename = config.popString("fileOutputRelaxation");
    paramsLosscone.  outputFilename = config.popString("fileOutputLosscone");
    paramsBinary.    outputFilename = config.popString("fileOutputBinary");
    paramsTrajectory.outputInterval =
    paramsPotential. outputInterval =
    paramsRelaxation.outputInterval = config.popDouble("outputInterval");
    paramsTrajectory.outputFormat   = config.popString("fileOutputFormat", paramsTrajectory.outputFormat);
    if(paramsTrajectory.outputInterval<=0 || paramsTrajectory.outputFilename.empty())
        utils::msg(utils::VL_MESSAGE, "Raga",
            "Snapshot output is disabled ([Raga]/outputInterval, [Raga]/fileOutput)");
    paramsPotential. numSamplesPerEpisode =
    paramsRelaxation.numSamplesPerEpisode =
        std::max(1, config.popInt("numSamplesPerEpisode", paramsRelaxation.numSamplesPerEpisode));
    paramsRelaxation.coulombLog       = config.popDouble("coulombLog", paramsRelaxation.coulombLog);
    paramsRelaxation.gridSizeDF       = config.popInt   ("gridSizeDF", paramsRelaxation.gridSizeDF);
    paramsLosscone.captureMassFraction= config.popDouble("captureMassFraction", paramsLosscone.captureMassFraction);
    paramsLosscone.speedOfLight       =
    paramsBinary.  speedOfLight       = config.popDouble("speedOfLight", paramsBinary.speedOfLight);
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
        std::string type = config.getString("type");
        if((utils::stringsEqual(type, potential::Multipole::myName()) ||
            utils::stringsEqual(type, potential::BasisSet ::myName()) ||
            utils::stringsEqual(type, potential::CylSpline::myName())) &&
            particles.size() > 0)
        {
            utils::msg(utils::VL_MESSAGE, "Raga", "Creating "+type+" potential from "+
                utils::toString(particles.size())+" particles and erasing them");
            ptrPot = potential::createPotential(config, particles);
            particles.data.clear();
        } else {
            utils::msg(utils::VL_MESSAGE, "Raga", "Creating "+type+" potential");
            ptrPot = potential::createPotential(config);
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
