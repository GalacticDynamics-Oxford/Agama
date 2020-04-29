#include "galaxymodel_jeans.h"
#include "math_core.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <fstream>

namespace galaxymodel{

namespace {

/// tolerance parameter for grid generation
static const double EPSDER2 = 1e-6;

/// accuracy parameter for integration in constructing the spherical Jeans model
static const double EPSINT  = 1e-6;

/// fraction of mass enclosed by the innermost grid cell or excluded by the outermost one
static const double MIN_MASS_FRAC  = 1e-4;

/// grid size for the axisymmetric Jeans eq
static const int NPOINTS_GRID = 64;

/// whether to use finite-element method for integrating the Jeans equation in z
#define USE_FEM

/// the product of density, radial force, and an appropriate power of radius, which enter the Jeans eq
class JeansSphIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction& pot;
    const math::IFunction& dens;
    const double beta;
public:
    JeansSphIntegrand(const math::IFunction& _pot, const math::IFunction& _dens, const double _beta) :
        pot(_pot), dens(_dens), beta(_beta) {}
    virtual double value(double r) const
    {
        double dPhidr;
        pot.evalDeriv(r, NULL, &dPhidr);
        return math::pow(r, beta*2) * dens(r) * dPhidr;
    }
};

}  // internal namespace

math::LogLogSpline createJeansSphModel(
    const math::IFunction &dens, const math::IFunction &pot, double beta)
{
    JeansSphIntegrand fnc(pot, dens, beta);
    // create an appropriate grid in log(r) and convert it to grid in r
    std::vector<double> gridr = math::createInterpolationGrid(math::LogLogScaledFnc(fnc), EPSDER2);
    for(size_t i=0, size=gridr.size(); i<size; i++)
        gridr[i] = exp(gridr[i]);
    // eliminate points from the tail where the density is zero or dominated by roundoff errors
    for(int i=gridr.size()-1; i>=0; i--) {
        math::PointNeighborhood pnDens(dens, gridr[i]);
        if(pnDens.f0 <= 0 || pnDens.fder == 0)
            gridr.pop_back();
        else
            break;
    }
    size_t npoints = gridr.size();
    if(npoints<3)
        throw std::runtime_error("createJeansSphModel: density seems to be non-positive");
    std::vector<double> integr(npoints);
    // start from the outermost interval - from rmax to infinity
    math::PointNeighborhood pnPot (pot,  gridr.back());
    math::PointNeighborhood pnDens(dens, gridr.back());
    double powPot  = pnPot. fder * gridr.back() / pnPot. f0;  // assuming that Phi(infinity)==0
    double powDens = pnDens.fder * gridr.back() / pnDens.f0;
    double power = 2*beta + powPot-1 + powDens;
    if(!(power < -1))
        throw std::runtime_error("createJeansSphModel: integrals do not converge at large radii");
    integr.back() = pnPot.f0 * pnDens.f0 * math::pow(gridr.back(), 2*beta) * powPot / (-1-power);
    // compute integrals over all radial intervals from outside in
    for(int i=npoints-2; i>=0; i--)
        integr[i] = integr[i+1] + math::integrate(fnc, gridr[i], gridr[i+1], EPSINT);
    // convert the integrals into the velocity dispersion at each radius
    for(int i=npoints-1; i>=0; i--) {
        double rho = dens(gridr[i]);
        if(rho!=0 && integr[i]>0) {
            integr[i] = fmin(
                sqrt(integr[i] / (rho * math::pow(gridr[i], 2*beta))),
                // safety measure: do not let the computed value be larger than the escape velocity
                sqrt(-2 * pot(gridr[i])) );
        } else {  // eliminate this point altogether
            gridr. erase(gridr. begin()+i);
            integr.erase(integr.begin()+i);
        }
    }

    // write out the results for debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("JeansSph.log");
        strm << "R\tsigma_r\n";
        for(unsigned int i=0; i<gridr.size(); i++)
            strm << utils::pp(gridr[i], 7) + '\t' + utils::pp(integr[i], 7) + '\n';
    }

    return math::LogLogSpline(gridr, integr);
}

JeansAxi::JeansAxi(const potential::BaseDensity &dens, const potential::BasePotential &pot,
    double beta, double _kappa) :
    bcoef(1. / (1-beta)), kappa(_kappa)
{
    if(!(beta < 1.))
        throw std::invalid_argument("JeansAxi: beta_m should be less than unity");
    // construct a suitable grid in R,z
    double Mtotal = dens.totalMass();
    double rmin = getRadiusByMass(dens, Mtotal * MIN_MASS_FRAC);
    double rmax = getRadiusByMass(dens, Mtotal * (1.-MIN_MASS_FRAC));
    if(!isFinite(rmin+rmax))
        throw std::runtime_error("JeansAxi: cannot construct grid (Mtotal=" + utils::toString(Mtotal) +
            ", rmin=" + utils::toString(rmin) + ", rmax=" + utils::toString(rmax) + ")");
    utils::msg(utils::VL_DEBUG, "JeansAxi", "Created grid in R,z: [" +
        utils::toString(rmin) + ":" + utils::toString(rmax) + "]");
    std::vector<double> gridR = math::createNonuniformGrid(NPOINTS_GRID, rmin, rmax, true);
    std::vector<double> gridz = gridR;  // for simplicity
    const int gridzsize = gridR.size();
    const int gridRsize = gridz.size();
    math::Matrix<double> vphi2(gridRsize, gridzsize, 0.), vz2(gridRsize, gridzsize);
    math::Matrix<double> rhosigmaz2(gridRsize, gridzsize), rhoval(gridRsize, gridzsize);
    std::vector<double> epicycleRatioVal(gridRsize);

#ifndef USE_FEM
    // perform the integration in z direction at each point of the radial grid
    const int GLORDER = 6;  // order of Gauss-Legendre integration
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iR=0; iR<gridRsize; iR++) {
        double R = gridR[iR];
        double result = 0.;
        // sum up integrals over the segments of the grid in z, starting from the outermost one
        // (from the last point to infinity)
        for(int iz=gridzsize-1; iz>=0; iz--) {
            for(int k=0; k<GLORDER; k++) {
                double z, w;  // z-coordinate of the k-th point in the quadrature rule and its weight
                if(iz<gridzsize-1) {
                    z = gridz[iz] + (gridz[iz+1]-gridz[iz]) * glnodes[k];
                    w = (gridz[iz+1]-gridz[iz]) * glweights[k];
                } else {  // scaling transformation on the last segment: z = zmax/t, 0<t<1
                    z = gridz[gridzsize-1] / glnodes[k];
                    w = gridz[gridzsize-1] / pow_2(glnodes[k]) * glweights[k];
                }
                double rho    = potential::azimuthalAverage<potential::AV_RHO>(dens, R, z);
                double dPhidz = potential::azimuthalAverage<potential::AV_DZ> (pot,  R, z);
                result += rho * dPhidz * w;
            }
            rhosigmaz2(iR, iz) = result;
            rhoval(iR, iz)     = potential::azimuthalAverage<potential::AV_RHO>(dens, R, gridz[iz]);
        }
    }

#else

    double maxz = gridz.back();
    // grid in `t` (scaled z coordinate)
    std::vector<double> gridfemz(gridzsize+4);
    for(int i=0; i<gridzsize; i++)
        gridfemz[i] = gridz[i] / (gridz[i] + maxz);
    // last point of gridz corresponds to t=0.5, we add a few more points spanning zmax..infinity
    gridfemz[gridzsize+0] = 0.55;
    gridfemz[gridzsize+1] = 0.65;
    gridfemz[gridzsize+2] = 0.80;
    gridfemz[gridzsize+3] = 1.00;
    // we first represent the relevant quantities -- rho  and  d(rho sigma_z^2) = -rho dPhi/dz --
    // as B-spline interpolants in scaled z coordinate, using the finite-element framework,
    // and then integrate the interpolant to obtain  rho sigma_z^2
    const math::FiniteElement1d<2> femz(gridfemz);
    std::vector<double> femzpoints = femz.integrPoints();    // obtain the integration points in `t`
    const int femzsize = femzpoints.size();
    std::vector<double> femzweights(femzsize, 1.);
    for(int kz=0; kz<femzsize; kz++) {
        femzweights[kz] = maxz / pow_2(1. - femzpoints[kz]); // dz/dt
        femzpoints [kz] = maxz / (1./femzpoints[kz] - 1.);   // convert scaled var `t` into `z`
    }
    const math::BandMatrix<double> mat = femz.computeProjMatrix();
    // this will represent the integral of r.h.s.
    const math::BsplineInterpolator1d<3> fint(gridfemz);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iR=0; iR<gridRsize; iR++) {
        double R = gridR[iR];
        std::vector<double> pointrho(femzsize), pointrhs(femzsize);
        // collect values of two functions at integration points
        for(int kz=0; kz<femzsize; kz++) {
            double z      = femzpoints[kz];
            double rho    = potential::azimuthalAverage<potential::AV_RHO>(dens, R, z);
            double dPhidz = potential::azimuthalAverage<potential::AV_DZ> (pot,  R, z);
            pointrho[kz]  = rho;
            pointrhs[kz]  = -rho * dPhidz * femzweights[kz];  // multiply by dz/dt
        }
        // construct FEM representations of both rho and  rhs = -rho dPhi/dt
        std::vector<double> projrhs = femz.computeProjVector(pointrhs);
        std::vector<double> projrho = femz.computeProjVector(pointrho);
        std::vector<double> ampldEdt= math::solveBand(mat, projrhs);
        std::vector<double> amplrho = math::solveBand(mat, projrho);
        // integrate the B-spline representation of r.h.s. to obtain the amplitudes
        // of B-spline representation of  E = rho sigma_z^2
        std::vector<double> amplE   = femz.interp.antideriv(ampldEdt);
        // and subtract the last element (the boundary condition is that E(infinity)=0)
        const double maxamplE = fabs(amplE.back()) * 1e-12;
        for(int iz=0; iz<(int)amplE.size(); iz++) {
            amplE[iz] -= amplE.back();
            // also zero out the elements of negligible magnitude -
            // this will trigger extrapolation at large radii later on
            if(fabs(amplE[iz]) < maxamplE)
                amplE[iz] = 0.;
        }

        // if the density is not computed accurately enough (e.g. it comes from a Multipole or CylSpline
        // potential approximation), then the entire high-z tail of the solution is unreliable;
        // we try to detect these cases and extrapolate both rho and rho sigma_z^2 to higher z
        // as power-laws (rho may be completely wrong but having sigma_z^2 ~ 1/z is quite reasonable)
        int iz = 0;
        for(; iz<gridzsize; iz++) {
            // same as femz.interp.integrate(gridfemz[iz], gridfemz.back(), ampldEdz);
            rhosigmaz2(iR, iz) =    fint.interpolate(gridfemz[iz], amplE);
            rhoval(iR, iz) = femz.interp.interpolate(gridfemz[iz], amplrho);
            // check if things get wrong
            if(iz>2 &&
                (amplE[iz] <= 0 || amplrho[iz] <= 0 || rhoval(iR, iz) <= 0 || rhosigmaz2(iR, iz) <= 0)) {
                // assume that the next-to-last point was still ok (the last one might be already wrong),
                // and restart in the regime of extrapolation
                iz -= 1;
                break;
            }
        }
        // check if we haven't processed all points in the normal interpolation regime
        // and need to extrapolate points starting from iz
        // (so that iz-1 is the index of the last reliable point)
        for(int ez=iz; ez<gridzsize; ez++) {
            // extrapolate to high z as power laws
            const double OUTER_GAMMA = -5.;   // power-law index for density - quite arbitrary
            rhoval    (iR, ez) = rhoval(iR, iz-1) * pow(gridz[ez] / gridz[iz-1], OUTER_GAMMA);
            // we wish that sigma_z^2 drops as 1/r in absense of more reliable information
            rhosigmaz2(iR, ez) = rhosigmaz2(iR, iz-1) * pow(gridz[ez] / gridz[iz-1], OUTER_GAMMA-1);
        }
    }
#endif

    for(int iz=0; iz<gridzsize; iz++) {
        for(int iR=0; iR<gridRsize; iR++) {
            double R = gridR[iR], z = gridz[iz];
            // compute  d (rho sigma_z^2) / dR  using simple-minded finite-differences
            int iR1 = std::max(0, iR-1), iR2 = std::min(iR+1, gridRsize-1);
            double derR   = (rhosigmaz2(iR2, iz) - rhosigmaz2(iR1, iz)) / (gridR[iR2] - gridR[iR1]);
            double rho    = rhoval(iR, iz);
            double Phi    = potential::azimuthalAverage<potential::AV_PHI>(pot,  R, z);
            double dPhidR = potential::azimuthalAverage<potential::AV_DR> (pot,  R, z);
            double vesc2  = -Phi;   // half the squared escape velocity - maximum allowed sigma^2
            vphi2(iR, iz) = math::clip(
                bcoef * (R * derR + rhosigmaz2(iR, iz)) / rho + R * dPhidR,  0., vesc2 );
            vz2  (iR, iz) = math::clip(rhosigmaz2(iR, iz) / rho,  0., vesc2);
        }
    }

    for(int iR=0; iR<gridRsize; iR++) {
        coord::GradCyl grad;
        coord::HessCyl hess;
        pot.eval(coord::PosCyl(gridR[iR], 0, 0), NULL, &grad, &hess);
        epicycleRatioVal[iR] = (iR==0 && grad.dR==0) ? 1 : 0.75 + 0.25 * gridR[iR] * hess.dR2 / grad.dR;
    }

    // write out the results for debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("JeansAxi.log");
        strm << "R\tz\tvz2\tvphi2\trho\trho*vz2\n";
        for(int iR=0; iR<gridRsize; iR++) {
            for(int iz=0; iz<gridzsize; iz++)
                strm << utils::pp(gridR[iR], 7) + '\t' + utils::pp(gridz[iz], 7) + '\t' +
                    utils::pp(vz2(iR,iz),    7) + '\t' + utils::pp(vphi2(iR,iz), 7) + '\t' +
                    utils::pp(rhoval(iR,iz), 7) + '\t' + utils::pp(rhosigmaz2(iR,iz), 7) + '\n';
            strm << '\n';
        }
    }

    intvphi2 = math::LinearInterpolator2d(gridR, gridz, vphi2);
    intvz2   = math::LinearInterpolator2d(gridR, gridz, vz2);
    epicycleRatio = math::CubicSpline(gridR, epicycleRatioVal);
}

void JeansAxi::moments(const coord::PosCyl& point, coord::VelCyl &vel, coord::Vel2Cyl &vel2) const
{
    double R = point.R, z = fabs(point.z), mult = 1.;
    if(R > intvphi2.xmax() || z > intvphi2.ymax()) {
        // assume that sigma(r) ~ 1/sqrt(r) at large distances
        mult = fmin(intvphi2.xmax() / R, intvphi2.ymax() / z);
        R = fmin(intvphi2.xmax(), R * mult);
        z = fmin(intvphi2.ymax(), z * mult);
    }
    vel2.vphi2 = intvphi2.value(R, z) * mult;
    vel2.vz2   = intvz2.  value(R, z) * mult;
    vel2.vR2   = vel2.vz2 * bcoef;
    vel2.vRvz  = vel2.vRvphi = vel2.vzvphi = 0.;
    vel.vR     = vel.vz = 0;
    vel.vphi   = kappa * sqrt(fmax(0, vel2.vphi2 - vel2.vR2 * epicycleRatio(R) /*sigma_phi^2*/));
}

}  // namespace