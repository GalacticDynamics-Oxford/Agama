#include "raga_relaxation.h"
#include "potential_multipole.h"
#include "galaxymodel_spherical.h"
#include "utils.h"
#include "math_core.h"
#include "potential_utils.h"
#include <cassert>
#include <cmath>
#include <fstream>

namespace raga {

orbit::StepResult RuntimeRelaxation::processTimestep(
    const math::BaseOdeSolver& sol, const double tbegin, const double tend, double currentState[6])
{
    // 1. collect samples of phase volume corresponding to particle energy
    // taken at regular intervals of time
    double timestep = tend-tbegin, tsamp;
    double data[6];
    while(tsamp = outputTimestep * (outputIter - outputFirst + 1),
        tsamp>tbegin && tsamp<=tend && outputIter != outputLast)
    {
        sol.getSol(tsamp, data);
        double E = totalEnergy(potentialSph, coord::PosVelCar(data));
        *(outputIter++) = relaxationModel.phasevol(E);
    }

    // 2. simulate the two-body relaxation

    // 2a. obtain the position, velocity and potential at the middle of the timestep
    tsamp = (tbegin+tend) * 0.5;
    sol.getSol(tsamp, data);
    coord::PosVelCar posvel(data);
    double vel = sqrt(pow_2(posvel.vx) + pow_2(posvel.vy) + pow_2(posvel.vz));
    if(vel==0)  // can't do anything meaningful for a non-moving particle
        return orbit::SR_CONTINUE;
    double Phi = potentialSph.value(posvel);
    double E   = Phi + 0.5 * pow_2(vel);

    // 2b. compute the diffusion coefs at the middle of the timestep
    double dvpar, dv2par, dv2per;
    relaxationModel.evalLocal(Phi, E, dvpar, dv2par, dv2per);
    if(!isFinite(dvpar+dv2par+dv2per) || dv2par<0 || dv2per<0) {
        utils::msg(utils::VL_WARNING, FUNCNAME,
            "Cannot compute diffusion coefficients at t="+utils::toString(tsamp)+
            ", r="+utils::toString(sqrt(pow_2(data[0])+pow_2(data[1])+pow_2(data[2])))+
            ", Phi="+utils::toString(Phi,10)+", E="+utils::toString(E,10));
        return orbit::SR_CONTINUE;
    }

    // 2c. scale the diffusion coefs
    dvpar  *= relaxationRate;
    dv2par *= relaxationRate;
    dv2per *= relaxationRate;
    double dEdt = dvpar + 0.5 * (dv2par + dv2per);
    if(dEdt * timestep < -0.5 * pow_2(vel))
        utils::msg(utils::VL_WARNING, FUNCNAME,
            "Energy perturbation is larger than its value: "
            "Phi="+utils::toString(Phi)+", v="+utils::toString(vel)+
            "; dt="+utils::toString(timestep)+", dE="+utils::toString(dEdt * timestep) );

    // 2d. assign the random (gaussian) velocity perturbation
    double rand1, rand2;
    math::getNormalRandomNumbers(rand1, rand2);
    double deltavpar = rand1 * sqrt(dv2par * timestep) + dvpar / vel * timestep;
    double deltavper = rand2 * sqrt(dv2per * timestep);

    // 2e. add the perturbations to the velocity
    double uper[3];  // unit vector perpendicular to velocity
    double vmag =    // magnitude of the current velocity vector
        math::getRandomPerpendicularVector(/*input: 3 components of velocity*/ currentState+3,
        /*output: a random unit vector*/ uper);
    for(int d=0; d<3; d++)
        currentState[d+3] +=
            // first term is the component of unit vector parallel to v: v[d]/|v|
            (currentState[d+3] / vmag) * deltavpar +
            uper[d] * deltavper;
    return orbit::SR_REINIT;
}

//----- RagaTaskRelaxation -----//

namespace{  // internal

/// modified log-log spline with a restriction on the inner slope:
class CautiousLogLogSpline: public math::IFunction {
    /// spline in log-log-scaled coordinates
    math::LogLogSpline spl;
    /// innermost grid point, corresponding function value and inner slope for extrapolation
    double xmin, sval, slope;
public:
    CautiousLogLogSpline(const math::LogLogSpline& S, double minslope) :
        spl(S), xmin(spl.xmin())
    {
        spl.evalDeriv(xmin, &sval, &slope);
        slope *= xmin / sval;
        if(!(slope >= minslope)) {
            utils::msg(utils::VL_WARNING, "RagaTaskRelaxation", "Adjusted the inner slope of f(h) "
                "from "+utils::toString(slope)+" to "+utils::toString(minslope)+" to keep Etotal finite");
            slope = minslope;
        }
    }

    virtual void evalDeriv(const double x,
        double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const
    {
        // extrapolation at small x is a power-law with the given slope;
        // if the correction in the constructor was not applied, this is the same as the original spline,
        // otherwise this results in a shallower inner power-law profile
        if(x<xmin) {
            double ratio = pow(x/xmin, slope);
            if(value)
                *value = sval * ratio;
            if(deriv)
                *deriv = sval * ratio * slope / x;
            if(deriv2)
                *deriv2= sval * ratio * slope * (slope-1) / pow_2(x);
        } else
            spl.evalDeriv(x, value, deriv, deriv2);
    }

    virtual unsigned int numDerivs() const { return 2; }
};

// eliminate samples with zero mass or positive energy (i.e. non-existent h):
void eliminateBadSamples(std::vector<double>& particle_h, std::vector<double>& particle_m)
{
    // scan the array and squeeze it towards the head
    std::vector<double>::iterator src_h = particle_h.begin(); // where to take elements from
    std::vector<double>::iterator src_m = particle_m.begin(); // where to take elements from
    std::vector<double>::iterator dest_h = src_h;  // where to store the elements (dest <= src always)
    std::vector<double>::iterator dest_m = src_m;
    while(src_h != particle_h.end()) {
        if(isFinite(*src_h) && *src_h>0 && *src_m>0) {
            *(dest_h++) = *src_h;
            *(dest_m++) = *src_m;
        }
        ++src_h;
        ++src_m;
    }
    assert(src_m == particle_m.end());  // two arrays must have the same size

    // shrink the array to retain only valid samples
    particle_h.erase(dest_h, particle_h.end());
    particle_m.erase(dest_m, particle_m.end());
    utils::msg(utils::VL_DEBUG, "RagaTaskRelaxation",
        "Retained "+utils::toString((unsigned int)particle_h.size())+" samples");
}

// prepare a spherical version of the total potential (with a single black hole at origin)
potential::PtrPotential createSphericalPotential(
    const potential::BasePotential& potential, double Mbh)
{
    // obtain the sph.-harm. coefficients of the stellar potential
    // (here we assume that it is represented as a Multipole class!)
    const potential::Multipole& pot =
        dynamic_cast<const potential::Multipole&>(potential);
    std::vector<double> rad;
    std::vector<std::vector<double> > Phi, dPhi;
    pot.getCoefs(rad, Phi, dPhi);

    // safety check: ensure that the potential is finite at origin
    // (more specifically, extrapolated as Phi(0) + C r^s with s>=0.05) -
    // this is needed for well-behaved diffusion coefs
    const double MINSLOPE = 0.05;
    double lnr1r0= log(rad[1]/rad[0]);
    double ratio = (Phi[0][1] - Phi[0][0]) / (dPhi[0][0] * rad[0] * lnr1r0);
    // ratio = [(r1/r0)^s - 1] / s / ln(r1/r0), and is  >= 1 + s/2 * ln(r1/r0)  if s>0
    double slope = (ratio - 1) / lnr1r0 * 2;  // this approximately holds if s is near 0
    if(slope < MINSLOPE) {
        // modify the derivative at the innermost grid point to correct the slope
        utils::msg(utils::VL_WARNING, "RagaTaskRelaxation", "Adjusted the inner slope of the potential "
            "from "+utils::toString(slope)+" to "+utils::toString(MINSLOPE)+" to keep Phi(0) finite");
        ratio = MINSLOPE * 0.5 * lnr1r0 + 1;
        dPhi[0][0] = (Phi[0][1] - Phi[0][0]) / (ratio * rad[0] * lnr1r0);
    }

    // retain only the l=0 terms and add the contribution from the central black hole
    Phi.resize(1);
    dPhi.resize(1);
    for(unsigned int i=0; i<rad.size(); i++) {
        Phi [0][i] -= Mbh / rad[i];
        dPhi[0][i] += Mbh / pow_2(rad[i]);
    }

    // construct the spherical potential
    return potential::PtrPotential(new potential::Multipole(rad, Phi, dPhi));
}

// prepare the relaxation model (diffusion coefficients) for the spherical potential
galaxymodel::PtrSphericalModelLocal createRelaxationModel(
    const potential::BasePotential& sphPot,
    std::vector<double>& particle_h,
    std::vector<double>& particle_m,
    const unsigned int numbins)
{
    // establish the correspondence between phase volume <=> energy
    potential::PhaseVolume phasevol((potential::PotentialWrapper(sphPot)));

    // eliminate particles with zero mass or positive energy
    eliminateBadSamples(particle_h, particle_m);

    // the fitting procedure guarantees that f(h) grows slower than h^-1 as h -> 0,
    // but to ensure that the total energy is finite, a stricter condition must be satisfied,
    // which depends on the innermost slope of the potential
    // (only relevant if the potential is singular, i.e. its innerSlope<0)
    double minSlope = -1 - 1 / (1.5 + 3/innerSlope(potential::PotentialWrapper(sphPot))) + 0.05;

    // determine the distribution function from the particle samples and represent it as a log-log spline
    CautiousLogLogSpline df(galaxymodel::fitSphericalDF(particle_h, particle_m, numbins), minSlope);

    // compute diffusion coefficients
    return galaxymodel::PtrSphericalModelLocal(new galaxymodel::SphericalModelLocal(phasevol, df));
}

}  // internal ns

RagaTaskRelaxation::RagaTaskRelaxation(
    const ParamsRelaxation& _params,
    const particles::ParticleArrayCar& _particles,
    const potential::PtrPotential& _ptrPot,
    const BHParams& _bh)
:
    params(_params),
    particles(_particles),
    ptrPot(_ptrPot),
    bh(_bh),
    prevOutputTime(-INFINITY)
{
    ptrPotSph = createSphericalPotential(*ptrPot, bh.mass);
    potential::PhaseVolume phasevol((potential::PotentialWrapper(*ptrPotSph)));
    ptrdiff_t nbody = particles.size();
    std::vector<double> particle_m(nbody);
    particle_h.resize(nbody);
#pragma omp parallel for schedule(static)
    for(ptrdiff_t i=0; i<nbody; i++) {
        particle_h[i] = phasevol(totalEnergy(*ptrPotSph, particles.point(i)));
        particle_m[i] = particles.mass(i);
    }
    ptrRelaxationModel = createRelaxationModel(*ptrPotSph,
        particle_h, particle_m, params.gridSizeDF);
    
    utils::msg(utils::VL_DEBUG, "RagaTaskRelaxation",
        "Initialized with relaxation rate="+utils::toString(params.relaxationRate));
}

orbit::PtrRuntimeFnc RagaTaskRelaxation::createRuntimeFnc(unsigned int index)
{
    return orbit::PtrRuntimeFnc(new RuntimeRelaxation(
        *ptrPotSph,
        *ptrRelaxationModel,
        params.relaxationRate,
        episodeLength / params.numSamplesPerEpisode,   // interval of time between storing the output samples
        particle_h.begin() + params.numSamplesPerEpisode * index,  // first and last index of the output sample
        particle_h.begin() + params.numSamplesPerEpisode * (index+1) ));
}

void RagaTaskRelaxation::startEpisode(double timeStart, double length)
{
    // at the beginning of the first episode, write out the spherical model file
    if(!params.outputFilename.empty() && prevOutputTime == -INFINITY) {
        prevOutputTime = timeStart;
        galaxymodel::writeSphericalModel(
            params.outputFilename + utils::toString(timeStart), params.header,
            *ptrRelaxationModel, potential::PotentialWrapper(*ptrPotSph));
    }
    episodeStart  = timeStart;
    episodeLength = length;
    // prepare space for storing samples of phase volume of all particles during the upcoming episode
    particle_h.assign(particles.size() * params.numSamplesPerEpisode, NAN);    
}

void RagaTaskRelaxation::finishEpisode()
{
    // assign mass to trajectory samples
    size_t nbody = particles.size();
    std::vector<double> particle_m(nbody * params.numSamplesPerEpisode);
    for(size_t i=0; i<nbody; i++) {
        double mass = particles.mass(i) / params.numSamplesPerEpisode;
        for(unsigned int j=0; j<params.numSamplesPerEpisode; j++)
            particle_m[i * params.numSamplesPerEpisode + j] = mass;
    }

    // create a new relaxation model for a sphericalized version of the current potential
    ptrPotSph = createSphericalPotential(*ptrPot, bh.mass);
    ptrRelaxationModel = createRelaxationModel(*ptrPotSph,
        particle_h, particle_m, params.gridSizeDF);

    // check if we need to output the relaxation model to a file
    double currentTime = episodeStart+episodeLength;
    if(!params.outputFilename.empty() && currentTime >= prevOutputTime + params.outputInterval) {
        prevOutputTime = currentTime;
        galaxymodel::writeSphericalModel(
            params.outputFilename + utils::toString(currentTime), params.header,
            *ptrRelaxationModel, potential::PotentialWrapper(*ptrPotSph));
    }
}

}  // namespace raga