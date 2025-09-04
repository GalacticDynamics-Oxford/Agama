#include "galaxymodel_velocitysampler.h"
#include "galaxymodel_spherical.h"
#include "galaxymodel_jeans.h"
#include "potential_multipole.h"
#include "df_spherical.h"
#include "math_core.h"
#include "math_random.h"
#include "smart.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>

//----- velocity assignment -----//
namespace galaxymodel {

particles::ParticleArrayCar assignVelocityEdd(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const SphericalIsotropicModelLocal& sphModel)
{
    ptrdiff_t npoints = pointCoords.size();
    particles::ParticleArrayCar result;
    result.data.resize(npoints);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0; i<npoints; i++) {
        const coord::PosCyl& point = pointCoords.point(i);
        math::PRNGState state = math::hash(/*position is the source of randomness*/ &point, 3, /*seed*/ i);
        double Phi = pot.value(point);
        double v;
        int numAttempts = 0;  // prevent a lockup in troubled cases
        do {
            v = sphModel.sampleVelocity(Phi, &state);
        } while(Phi + 0.5*v*v >= 0 && ++numAttempts<100);
        if(numAttempts>=100)  // something went wrong
            v = 0;
        double vec[3], sinphi, cosphi;
        math::getRandomUnitVector(vec, &state);
        math::sincos(point.phi, sinphi, cosphi);
        result[i] = particles::ParticleArrayCar::ElemType(coord::PosVelCar(
            point.R * cosphi, point.R * sinphi, point.z,
            v * vec[0], v * vec[1], v * vec[2]),
            pointCoords.mass(i));
    }
    return result;
}

particles::ParticleArrayCar assignVelocityJeansSph(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const math::IFunction& jeansSphModel,
    const double beta)
{
    ptrdiff_t npoints = pointCoords.size();
    particles::ParticleArrayCar result;
    result.data.resize(npoints);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0; i<npoints; i++) {
        const coord::PosCyl& point = pointCoords.point(i);
        math::PRNGState state = math::hash(/*position is the source of randomness*/ &point, 3, /*seed*/ i);
        double r = hypot(point.R, point.z);
        double sigma_r = jeansSphModel.value(r);
        double sigma_t = sigma_r * sqrt(2-2*beta);  // vel.disp. in two tangential directions combined
        double Phi = pot.value(point);
        double vr, vt;
        int numAttempts = 0;
        do {
            math::getNormalRandomNumbers(vr, vt, &state);
            vr *= sigma_r;
            vt *= sigma_t;
        } while(Phi + 0.5 * (vr*vr + vt*vt) > 0 && ++numAttempts<100);
        double sinphi, cosphi;
        math::sincos(point.phi, sinphi, cosphi);
        double xyz[3] = { point.R * cosphi, point.R * sinphi, point.z };
        double vper[3];
        math::getRandomPerpendicularVector(xyz, vper, &state);
        if(r==0) r=1.;  // avoid indeterminacy
        result[i] = particles::ParticleArrayCar::ElemType(coord::PosVelCar(
            xyz[0], xyz[1], xyz[2],
            vr * xyz[0] / r + vt * vper[0],
            vr * xyz[1] / r + vt * vper[1],
            vr * xyz[2] / r + vt * vper[2]),
            pointCoords.mass(i));
    }
    return result;
}

particles::ParticleArrayCar assignVelocityJeansAxi(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const JeansAxi& jeansAxiModel)
{
    ptrdiff_t npoints = pointCoords.size();
    particles::ParticleArrayCar result;
    result.data.resize(npoints);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(ptrdiff_t i=0; i<npoints; i++) {
        const coord::PosCyl& point = pointCoords.point(i);
        math::PRNGState state = math::hash(/*position is the source of randomness*/ &point, 3, /*seed*/ i);
        coord::VelCyl  vel;
        coord::Vel2Cyl vel2;
        jeansAxiModel.moments(point, vel, vel2);
        double sigma_z   = sqrt(vel2.vz2);
        double sigma_R   = sqrt(vel2.vR2);
        double sigma_phi = sqrt(fmax(0., vel2.vphi2 - pow_2(vel.vphi)));
        double Phi = pot.value(point);
        double vR, vz, vphi, sphi, devnull;
        int numAttempts = 0;
        do {
            math::getNormalRandomNumbers(sphi, /*ignored - need only one number here*/ devnull, &state);  
            vphi = vel.vphi + sphi * sigma_phi;
            math::getNormalRandomNumbers(vR, vz, &state);
            vR *= sigma_R;
            vz *= sigma_z;
        } while(Phi + 0.5 * (vR*vR + vz*vz + vphi*vphi) > 0 && ++numAttempts<100);
        result[i] = particles::ParticleArrayCar::ElemType(
            toPosVelCar(coord::PosVelCyl(point, coord::VelCyl(vR, vz, vphi))),
            pointCoords.mass(i));
    }
    return result;
}

particles::ParticleArrayCar assignVelocity(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BaseDensity& dens,
    const potential::BasePotential& pot,
    const double beta,
    const double kappa)
{
    /// type of velocity sampling procedure
    enum SamplingMethod {
        SD_EDDINGTON, ///< assign velocities from the Eddington DF
        SD_JEANSSPH,  ///< assign velocities from a spherical Jeans model
        SD_JEANSAXI   ///< assign velocities from an axisymmetric Jeans anisotropic model
    };

    // depending on the provided arguments, use different methods for assigning the velocities
    const SamplingMethod method =
        beta !=beta  ? SD_EDDINGTON :
        kappa!=kappa ? SD_JEANSSPH  : SD_JEANSAXI;

    if(method == SD_EDDINGTON || method == SD_JEANSSPH) {
        // sphericalized versions of density and potential
        // (if the input functions are already spherical, this adds no extra cost)
        potential::Sphericalized<potential::BaseDensity>   sphDens(dens);
        potential::Sphericalized<potential::BasePotential> sphPot (pot);
        if(method == SD_EDDINGTON) {
            FILTERMSG(utils::VL_DEBUG, "assignVelocity", "Using Eddington DF for " +
                sphDens.name() + " in " + sphPot.name());
            const potential::PhaseVolume phasevol(sphPot);
            const math::LogLogSpline df = df::createSphericalIsotropicDF(sphDens, sphPot);
            const SphericalIsotropicModelLocal model(phasevol, df, df);
            return assignVelocityEdd(pointCoords, pot, model);
        } else if(method == SD_JEANSSPH) {
            FILTERMSG(utils::VL_DEBUG, "assignVelocity", "Using spherical Jeans for " +
                sphDens.name() + " in " + sphPot.name());
            math::LogLogSpline model = createJeansSphModel(sphDens, sphPot, beta);
            return assignVelocityJeansSph(pointCoords, pot, model, beta);
        }
    } else if(method == SD_JEANSAXI) {
        FILTERMSG(utils::VL_DEBUG, "assignVelocity", "Using axisymmetric Jeans for " +
            potential::Axisymmetrized<potential::BaseDensity>  (dens).name() + " in " +
            potential::Axisymmetrized<potential::BasePotential>(pot) .name() );
        JeansAxi model(dens, pot, beta, kappa);
        return assignVelocityJeansAxi(pointCoords, pot, model);
    }
    assert(!"assignVelocity: unknown method");
    return particles::ParticleArrayCar();
}

}
