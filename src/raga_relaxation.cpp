#include "raga_relaxation.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_multipole.h"
#include "galaxymodel_spherical.h"
#include "df_spherical.h"
#include "utils.h"
#include "math_core.h"
#include "math_random.h"
#include <cassert>
#include <cmath>
#include <fstream>

namespace raga {

bool RuntimeRelaxation::processTimestep(double tbegin, double timestep)
{
    // 1. collect samples of phase volume corresponding to particle energy
    // taken at regular intervals of time
    double tsamp;
    while(tsamp = outputTimestep * (outputIter - outputFirst + 1) - tbegin,
        tsamp>0 && tsamp<=timestep && outputIter != outputLast)
    {
        double E = totalEnergy(potentialSph, orbint.getSol(tsamp));
        *(outputIter++) = relaxationModel.phasevol(E);
    }

    // 2. simulate the two-body relaxation

    // 2a. obtain the position, velocity and potential at the end of the timestep
    coord::PosVelCar posvel = orbint.getSol(timestep);
    double vel = sqrt(pow_2(posvel.vx) + pow_2(posvel.vy) + pow_2(posvel.vz));
    if(vel==0)  // can't do anything meaningful for a non-moving particle
        return true;
    double Phi = potentialSph.value(posvel);
    double E   = Phi + 0.5 * pow_2(vel);

    // 2b. compute the diffusion coefs at the end of the timestep
    double dvpar, dv2par, dv2per;
    relaxationModel.evalLocal(Phi, E, mass, dvpar, dv2par, dv2per);
    if(!isFinite(dvpar+dv2par+dv2per) || dv2par<0 || dv2per<0) {
        FILTERMSG(utils::VL_WARNING, "RagaTaskRelaxation",
            "Cannot compute diffusion coefficients at t="+utils::toString(tbegin+timestep)+
            ", r="+utils::toString(sqrt(pow_2(posvel.x)+pow_2(posvel.y)+pow_2(posvel.z)))+
            ", Phi="+utils::toString(Phi,10)+", E="+utils::toString(E,10));
        return true;
    }

    // 2c. scale the diffusion coefs
    dvpar  *= coulombLog;
    dv2par *= coulombLog;
    dv2per *= coulombLog;
    double dEdt = dvpar + 0.5 * (dv2par + dv2per);
    if(dEdt * timestep < -0.5 * pow_2(vel))
        FILTERMSG(utils::VL_WARNING, "RagaTaskRelaxation",
            "Energy perturbation is larger than its value: "
            "Phi="+utils::toString(Phi)+", v="+utils::toString(vel)+
            "; dt="+utils::toString(timestep)+", dE="+utils::toString(dEdt * timestep) );

    // 2d. assign the random (gaussian) velocity perturbation
    // initialize the PRNG state vector, using the current position-velocity
    // as the source of "randomness", with an unique seed for each orbit
    double data[6];
    posvel.unpack_to(data);
    math::PRNGState state = math::hash(data, 6, seed);
    double rand1, rand2;  // two normally distributed numbers
    math::getNormalRandomNumbers(/*output*/ rand1, rand2, /*PRNGState*/ &state);
    double deltavpar = rand1 * sqrt(dv2par * timestep) + dvpar / vel * timestep;
    double deltavper = rand2 * sqrt(dv2per * timestep);

    // 2e. add the perturbations to the velocity
    double uper[3];  // unit vector perpendicular to velocity
    double vmag =    // magnitude of the current velocity vector
        math::getRandomPerpendicularVector(
        /*input:  3 components of velocity*/ data+3,
        /*output: a random unit vector*/ uper,
        /*input/output: PRNG seed*/ &state);
    for(int d=0; d<3; d++)
        data[d+3] +=
            // first term is the component of unit vector parallel to v: v[d]/|v|
            (data[d+3] / vmag) * deltavpar +
            uper[d] * deltavper;

    // 2f. update the internal state of the orbit integrator with the new velocity
    orbint.init(coord::PosVelCar(data));
    return true;  // integration may continue
}

//----- RagaTaskRelaxation -----//

namespace{  // internal

/// modified log-log spline with a restriction on the inner and outer slopes
class CautiousLogLogSpline: public math::IFunction {
    /// spline in log-log-scaled coordinates
    math::LogLogSpline spl;
    /// inner- and outermost grid points, corresponding function values and slopes for extrapolation
    double xin, xout, fin, fout, slopein, slopeout;
public:
    CautiousLogLogSpline(const math::LogLogSpline& S, double minslope) :
        spl(S), xin(spl.xmin()), xout(spl.xmax())
    {
        spl.evalDeriv(xin, &fin, &slopein);
        slopein *= xin / fin;
        if(!(slopein >= minslope)) {
            FILTERMSG(utils::VL_WARNING, "RagaTaskRelaxation", "Adjusted the inner slope of f(h) "
                "from "+utils::toString(slopein)+" to "+utils::toString(minslope)+
                " to keep Etotal finite");
            slopein = minslope;
        }
        // outer slope must be steeper than -1 to keep the total mass finite
        spl.evalDeriv(xout, &fout, &slopeout);
        slopeout = fmin(slopeout * xout / fout, -1.05);
    }

    virtual void evalDeriv(const double x,
        double* value=NULL, double* deriv=NULL, double* deriv2=NULL) const
    {
        // extrapolation at small x is a power-law with the given slope;
        // if the correction in the constructor was not applied, this is the same as the original spline,
        // otherwise this results in a shallower inner power-law profile
        if(x<xin) {
            double ratio = pow(x/xin, slopein);
            if(value)
                *value = fin * ratio;
            if(deriv)
                *deriv = fin * ratio * slopein / x;
            if(deriv2)
                *deriv2= fin * ratio * slopein * (slopein-1) / pow_2(x);
        } else if(x>xout) {
            double ratio = pow(x/xout, slopeout);
            if(value)
                *value = fout * ratio;
            if(deriv)
                *deriv = fout * ratio * slopeout / x;
            if(deriv2)
                *deriv2= fout * ratio * slopeout * (slopeout-1) / pow_2(x);
        } else
            spl.evalDeriv(x, value, deriv, deriv2);
    }

    virtual unsigned int numDerivs() const { return 2; }
};

// eliminate samples with zero mass or positive energy (i.e. non-existent h):
void eliminateBadSamples(std::vector<double>& particle_h,
    std::vector<double>& particle_m, std::vector<double>& stellar_m)
{
    // scan the array and squeeze it towards the head
    std::vector<double>::iterator src_h = particle_h.begin(); // where to take elements from
    std::vector<double>::iterator src_m = particle_m.begin(); // where to take elements from
    std::vector<double>::iterator src_s = stellar_m.begin();  // where to take elements from
    std::vector<double>::iterator dest_h = src_h;  // where to store the elements (dest <= src always)
    std::vector<double>::iterator dest_m = src_m;
    std::vector<double>::iterator dest_s = src_s;
    while(src_h != particle_h.end()) {
        if(isFinite(*src_h) && *src_h>0 && *src_m>0) {
            *(dest_h++) = *src_h;
            *(dest_m++) = *src_m;
            *(dest_s++) = *src_s;
        }
        ++src_h;
        ++src_m;
        ++src_s;
    }
    assert(src_m == particle_m.end() && src_s == stellar_m.end());  // two arrays must have the same size

    // shrink the array to retain only valid samples
    particle_h.erase(dest_h, particle_h.end());
    particle_m.erase(dest_m, particle_m.end());
    stellar_m .erase(dest_s, stellar_m.end());
    FILTERMSG(utils::VL_DEBUG, "RagaTaskRelaxation",
        "Retained "+utils::toString((unsigned int)particle_h.size())+" samples");
}

// prepare a spherical version of the total potential (with a single black hole at origin)
potential::PtrPotential createSphericalPotential(
    const potential::BasePotential& potential, double Mbh)
{
    // obtain the sph.-harm. coefficients of the stellar potential
    std::vector<double> rad;
    std::vector<std::vector<double> > Phi, dPhi;
    const potential::Multipole* potmul = dynamic_cast<const potential::Multipole*>(&potential);
    if(potmul) {
        // if the potential is an instance of Multipole class, take the coefs directly from it
        potmul->getCoefs(rad, Phi, dPhi);
    } else {
        // otherwise construct a temporary instance of Multipole and take the coefs from it
        potential::Multipole::create(potential,
            coord::ST_SPHERICAL, /*lmax*/0, /*mmax*/0, /*gridSizeR*/50)->getCoefs(rad, Phi, dPhi);
    }

    // safety check: ensure that the potential is finite at origin
    // (more specifically, extrapolated as Phi(0) + C r^s with s>=0.25, i.e. rho(r) no steeper
    // than the Bahcall-Wolf profile) - this is needed for well-behaved diffusion coefs
    const double MINSLOPE = 1./4;
    double lnr1r0= log(rad[1]/rad[0]);
    double ratio = (Phi[0][1] - Phi[0][0]) / (dPhi[0][0] * rad[0] * lnr1r0);
    // ratio = [(r1/r0)^s - 1] / s / ln(r1/r0), and is  >= 1 + s/2 * ln(r1/r0)  if s>0
    double slope = (ratio - 1) / lnr1r0 * 2;  // this approximately holds if s is near 0
    if(slope < MINSLOPE) {
        // modify the derivative at the innermost grid point to correct the slope
        FILTERMSG(utils::VL_WARNING, "RagaTaskRelaxation", "Adjusted the inner slope of the potential "
            "from "+utils::toString(slope)+" to "+utils::toString(MINSLOPE)+" to keep Phi(0) finite");
        ratio = MINSLOPE * 0.5 * lnr1r0 + 1;
        dPhi[0][0] = (Phi[0][1] - Phi[0][0]) / (ratio * rad[0] * lnr1r0);
    }

    // retain only the l=0 terms and add the contribution from the central black hole
    Phi.resize(1);
    dPhi.resize(1);
    potential::PtrPotential potSph(new potential::Multipole(rad, Phi, dPhi));
    if(Mbh==0)
        return potSph;
    else {
        // create a composite potential out of the stellar potential and the singular BH potential
        std::vector<potential::PtrPotential> comp(2);
        comp[0] = potSph;
        comp[1] = potential::PtrPotential(new potential::Plummer(Mbh, 0));
        return potential::PtrPotential(new potential::Composite(comp));
    }
}

// prepare the relaxation model (diffusion coefficients) for the spherical potential
galaxymodel::PtrSphericalIsotropicModelLocal createAndWriteRelaxationModel(
    const potential::PhaseVolume& phasevol,
    const potential::BasePotential& sphPot,
    std::vector<double>& particle_h,
    std::vector<double>& particle_m,
    std::vector<double>& stellar_m,
    const unsigned int numbins,
    const std::string& filename,
    const std::string& header)
{
    // eliminate particles with zero mass or positive energy
    eliminateBadSamples(particle_h, particle_m, stellar_m);

    // the fitting procedure guarantees that f(h) grows slower than h^-1 as h -> 0,
    // but we impose a stricter restriction that  f(h)  does not diverge as h -> 0.
    double minSlope = 0;

    // determine the distribution function from the particle samples
    // and represent it as a log-log spline.
    // there are two distinct kinds of DFs: one is "unweighted" (df)
    // (corresponds to the gravitating mass density in the phase space),
    // the other (DF) is additionally weighted by stellar mass of each particle,
    // and determines the relaxation rate
    CautiousLogLogSpline df(df::fitSphericalIsotropicDF(particle_h, particle_m, numbins), minSlope);
    CautiousLogLogSpline DF(df::fitSphericalIsotropicDF(particle_h, stellar_m,  numbins), minSlope);

    // write out the model to a text file, if needed
    if(!filename.empty())
        galaxymodel::writeSphericalIsotropicModel(filename, header, df, sphPot);

    // compute diffusion coefficients
    return  galaxymodel::PtrSphericalIsotropicModelLocal(
        new galaxymodel::SphericalIsotropicModelLocal(phasevol, /*unweighted*/df, /*mass-weighted*/DF));
}

}  // internal ns

RagaTaskRelaxation::RagaTaskRelaxation(
    const ParamsRelaxation& _params,
    const particles::ParticleArrayAux& _particles,
    const potential::PtrPotential& _ptrPot,
    const potential::KeplerBinaryParams& _bh)
:
    params(_params),
    particles(_particles),
    ptrPot(_ptrPot),
    bh(_bh),
    prevOutputTime(-INFINITY)
{
    FILTERMSG(utils::VL_DEBUG, "RagaTaskRelaxation",
        "Initialized with ln Lambda="+utils::toString(params.coulombLog));
}

void RagaTaskRelaxation::createRuntimeFnc(orbit::BaseOrbitIntegrator& orbint, unsigned int index)
{
    orbint.addRuntimeFnc(orbit::PtrRuntimeFnc(new RuntimeRelaxation(
        orbint,
        *ptrPotSph,
        *ptrRelaxationModel,
        params.coulombLog,
        particles.point(index).stellarMass,
        // interval of time between storing the output samples
        episodeLength / params.numSamplesPerEpisode,
        // first and last index of the output sample
        particle_h.begin() + params.numSamplesPerEpisode * index,
        particle_h.begin() + params.numSamplesPerEpisode * (index+1),
        index  // seed for the orbit-local PRNG
    )));
}

void RagaTaskRelaxation::startEpisode(double timeStart, double length)
{
    // at the beginning of the first episode, create the spherical model and write it to a file
    if(!ptrRelaxationModel) {
        // create the sphericalized version of the true potential (including the central BH)
        ptrPotSph = createSphericalPotential(*ptrPot, bh.mass);
        // establish the correspondence between phase volume <=> energy
        potential::PhaseVolume phasevol((potential::Sphericalized<potential::BasePotential>(*ptrPotSph)));
        // compute the values of phase volume h for each particle
        ptrdiff_t nbody = particles.size();
        std::vector<double> particle_m(nbody);
        std::vector<double> stellar_m (nbody);
        particle_h.resize(nbody);
        for(ptrdiff_t i=0; i<nbody; i++) {
            particle_h[i] = phasevol(totalEnergy(*ptrPotSph, particles.point(i)));
            particle_m[i] = particles.mass(i);
            stellar_m [i] = particles.mass(i) * particles.point(i).stellarMass;
        }
        // create the relaxation model and write it to a file (if needed)
        ptrRelaxationModel = createAndWriteRelaxationModel(
            phasevol, *ptrPotSph, particle_h, particle_m, stellar_m, params.gridSizeDF,
            params.outputFilename.empty() ? "" : params.outputFilename + utils::toString(timeStart),
            params.header);
        prevOutputTime = timeStart;
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
    std::vector<double> stellar_m (nbody * params.numSamplesPerEpisode);
    for(size_t i=0; i<nbody; i++) {
        double mass = particles.mass(i) / params.numSamplesPerEpisode,
            mstar = particles.point(i).stellarMass;
        for(unsigned int j=0; j<params.numSamplesPerEpisode; j++) {
            particle_m[i * params.numSamplesPerEpisode + j] = mass;
            stellar_m [i * params.numSamplesPerEpisode + j] = mass * mstar;
        }
    }

    // check if need to write out the relaxation model
    double time = episodeStart+episodeLength;
    std::string outputFilename;
    if(!params.outputFilename.empty() && time >= prevOutputTime + params.outputInterval*0.999999) {
        prevOutputTime = time;
        outputFilename = params.outputFilename + utils::toString(time);
    }

    // create a new relaxation model for the sphericalized version of the current potential
    ptrPotSph = createSphericalPotential(*ptrPot, bh.mass);
    potential::PhaseVolume phasevol((potential::Sphericalized<potential::BasePotential>(*ptrPotSph)));
    ptrRelaxationModel = createAndWriteRelaxationModel(
        phasevol, *ptrPotSph, particle_h, particle_m, stellar_m, params.gridSizeDF,
        outputFilename, params.header);
}

}  // namespace raga