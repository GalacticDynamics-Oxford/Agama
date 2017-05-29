#include "galaxymodel_target.h"
#include "galaxymodel_densitygrid.h"
#include "galaxymodel_jeans.h"
#include "math_core.h"
#include "potential_multipole.h"
#include "potential_utils.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <alloca.h>
#include <fstream>

namespace galaxymodel{

namespace {  // internal

/// Orbit runtime function that collects the values of a given N-dimensional function
/// for each point on the trajectory, weighted by the amount of time spent at this point
class RuntimeFncSchw: public orbit::BaseRuntimeFnc {
    /// number of points taken from the trajectory during each timestep of the ODE solver
    static const int NUM_SAMPLES_PER_STEP = 10;

    /// the function that collects some data for a given point
    /// (takes the position/velocity in Cartesian coordinates as input)
    const math::IFunctionNdim& grid;

    /// where the data for this orbit will be ultimately stored (points to an external array)
    StorageNumT* output;

    /** intermediate storage for the data collected during orbit integration,
        weighted by the time chunk associated with each sub-step on the trajectory;
        internally accumulated in double precision, and at the end of integration normalized
        by the integration time and written in the output array converted to StorageNumT
    */
    std::vector<double> data;

    /// total integration time - will be used to normalize the collected data
    /// at the end of orbit integration
    double time;

public:
    RuntimeFncSchw(const math::IFunctionNdim& _grid, StorageNumT* _output) :
        grid(_grid), output(_output), data(grid.numValues(), 0.), time(0.) {}

    /// normalize the collected data by the total integration time,
    /// and convert to the numerical type used in the output storage
    virtual ~RuntimeFncSchw()
    {
        if(time==0) return;
        const unsigned int size = grid.numValues();
        for(unsigned int i=0; i<size; i++)
            output[i] = static_cast<StorageNumT>(data[i] / time);
    }

    /// collect the data returned by the function for each point sub-sampled from the trajectory
    /// on the current timestep, and add it to the temporary storage array,
    /// weighted by the duration of the substep
    virtual orbit::StepResult processTimestep(
        const math::BaseOdeSolver& solver, const double tbegin, const double tend, double[])
    {
        time += tend-tbegin;
        double substep = (tend-tbegin) / NUM_SAMPLES_PER_STEP;  // duration of each sub-step
        const unsigned int size = grid.numValues();
        // temporary array for storing the values of grid basis functions at each sub-step
        double* values = static_cast<double*>(alloca(size * sizeof(double)));
        for(int s=0; s<NUM_SAMPLES_PER_STEP; s++) {
            double point[6];  // position and velocity in cartesian coordinates at the current sub-step
            double tsubstep = tbegin + substep * (s+0.5);  // equally-spaced samples in time
            solver.getSol(tsubstep, point);
            grid.eval(point, values);
            for(unsigned int i=0; i<size; i++)
                data[i] += values[i] * substep;
        }
        return orbit::SR_CONTINUE;
    }
};

//---- auxiliary grid construction routines ----//

/// relative accuracy for computing the mass enclosed in a grid segment
static const double EPSREL_MASS_INT = 1e-3;
/// max # of density evaluations for computing the enclosed mass
static const int MAX_NUM_EVAL = 10000;
/// relative accuracy for root-finder to determine the radius enclosing the given mass
static const double EPSREL_MASS_ROOT = 1e-4;

/// 3d integration of density over a region aligned with cylindrical coordinates
class DensityIntegrandCylNdim: public math::IFunctionNdim {
    const potential::BaseDensity& dens;  ///< the density model to be integrated over
    const double rscale;                 ///< characteristic radius for scaling transformation
public:
    DensityIntegrandCylNdim(const potential::BaseDensity& _dens, double _rscale) :
        dens(_dens), rscale(_rscale){}

    /// integrand for the density at a given point (R,z) with an appropriate scaling
    virtual void eval(const double vars[], double values[]) const {
        double
        R   = rscale / (1. / vars[0] - 1.),
        z   = rscale / (1. / vars[1] - 1.),
        jac = 4*M_PI * R * pow_2( rscale / (1. - vars[0]) / (1. - vars[1]) ),
        val = potential::azimuthalAverage<potential::AV_RHO>(dens, R, z);
        values[0] =  val!=0 && isFinite(val+jac) ?  val*jac  :  0.;
    }

    virtual unsigned int numVars() const { return 2; }    // two input values (scaled R and z)

    virtual unsigned int numValues() const { return 1; }  // one output value (rho times jacobian)
};

template<int DIRECTION_Z>
class CylMassRootFinder: public math::IFunctionNoDeriv {
    const math::IFunctionNdim& fnc;  ///< N-dimensional function to integrate
    const double* xlower;            ///< lower boundaries of the integration region
    const double  target;            ///< required value for the integral
public:
    CylMassRootFinder(const math::IFunctionNdim& _fnc, const double* _xlower, double _target) :
        fnc(_fnc), xlower(_xlower), target(_target) {}
    virtual double value(double x) const {
        if(x == xlower[DIRECTION_Z])
            return -target;
        double xupper[2] = {1., 1.};
        xupper[DIRECTION_Z] = x;
        double result;
        math::integrateNdim(fnc, xlower, xupper, EPSREL_MASS_INT, MAX_NUM_EVAL, &result);
        return result-target;
    }
};

template<int DIRECTION_Z>
std::vector<double> getCylRzByMass(
    const potential::BaseDensity& density, const std::vector<double>& gridMass)
{
    // characteristic spatial scale
    double rscale = getRadiusByMass(density, gridMass[gridMass.size()/2]);
    if(rscale==0 || !isFinite(rscale))
        throw std::runtime_error("TargetDensity: cannot assign grid radii");
    DensityIntegrandCylNdim fnc(density, rscale);
    double xlower[2] = {0., 0.};
    unsigned int npoints = gridMass.size();
    std::vector<double> result(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        // required mass inside this grid segment
        double target = gridMass[i] - (i>0 ? gridMass[i-1] : 0.);
        double xroot  = math::findRoot(CylMassRootFinder<DIRECTION_Z>(fnc, xlower, target),
            xlower[DIRECTION_Z], 1., EPSREL_MASS_ROOT);
        if(!isFinite(xroot) || xroot<=0 || xroot>=1.)
            throw std::runtime_error("TargetDensity: cannot assign grid radii");
        xlower[DIRECTION_Z] = xroot;  // move the lower boundary of the next segment
        result[i] = rscale / (1. / xroot - 1.);
    }
    return result;
}

}  // internal ns

//----- Density discretization scheme -----//

TargetDensity::TargetDensity(const potential::BaseDensity& density, const DensityGridParams& params)
{
    if(!isTriaxial(density))
        throw std::runtime_error("TargetDensity: density must have at least triaxial symmetry");
    // first determine the grid radii that enclose the specified fractions of mass
    double totalMass = density.totalMass();
    if(!isFinite(totalMass))
        throw std::runtime_error("TargetDensity: total mass must be finite");
    unsigned int gridSizeR = params.gridSizeR;
    double outerShellMass  = params.outerShellMass ?: gridSizeR / (gridSizeR + 1.);
    double innerShellMass  = params.innerShellMass ?: outerShellMass / gridSizeR;
    if( gridSizeR == 0 || innerShellMass <= 0 || outerShellMass <= innerShellMass || outerShellMass >= 1)
        throw std::invalid_argument("TargetDensity: invalid grid parameters");
    std::vector<double> gridMass = math::createNonuniformGrid(
        gridSizeR, innerShellMass * totalMass, outerShellMass * totalMass, false);
    if(params.type == DG_CYLINDRICAL_TOPHAT || params.type == DG_CYLINDRICAL_LINEAR) {
        gridR = getCylRzByMass<0>(density, gridMass);
        gridMass = math::createNonuniformGrid(
            params.gridSizez, innerShellMass * totalMass, outerShellMass * totalMass, false);
        gridz = getCylRzByMass<1>(density, gridMass);
    } else {
        gridR.resize(gridSizeR);
        for(unsigned int s=0; s<gridSizeR; s++)
            gridR[s] = getRadiusByMass(density, gridMass[s]);
    }

    utils::msg(utils::VL_DEBUG, "TargetDensity", "Grid in radius: [" +
        utils::toString(gridR[0]) + ":" + utils::toString(gridR.back()) + (gridz.empty() ? "]" :
        "], in z: [" + utils::toString(gridz[0]) + ":" + utils::toString(gridz.back()) + "]") );
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("TargetDensity.log");
        strm << "#Density type="<<params.type<<"\n";
        for(unsigned int s=0; s<gridR.size(); s++) strm<<gridR[s]<<"\n";
        for(unsigned int s=0; s<gridz.size(); s++) strm<<gridz[s]<<"\n";
    }

    // then construct the appropriate density grid object
    switch(params.type) {
        case DG_CLASSIC_TOPHAT:
            grid.reset(new DensityGridClassic<0>(
                params.stripsPerPane, gridR, params.axisRatioY, params.axisRatioZ));
            break;
        case DG_CLASSIC_LINEAR:
            grid.reset(new DensityGridClassic<1>(
                params.stripsPerPane, gridR, params.axisRatioY, params.axisRatioZ));
            break;
        case DG_SPH_HARM:
            grid.reset(new DensityGridSphHarm(params.lmax, params.mmax, gridR));
            break;
        case DG_CYLINDRICAL_TOPHAT:
            grid.reset(new DensityGridCylindrical<0>(params.mmax, gridR, gridz));
            break;
        case DG_CYLINDRICAL_LINEAR:
            grid.reset(new DensityGridCylindrical<1>(params.mmax, gridR, gridz));
            break;
        default:
            throw std::invalid_argument("TargetDensity: unknown grid type");
    }

    // finally, compute the projection of the input density onto the grid
    constraintValues = grid->computeProjVector(density);
    constraintPenalties.assign(constraintValues.size(), 1.);
    // add the last constraint specifying the total mass
    constraintPenalties.push_back(1.);
    constraintValues.push_back(totalMass);
}

orbit::PtrRuntimeFnc TargetDensity::getOrbitRuntimeFnc(StorageNumT* output) const
{
    output[constraintValues.size()-1] = 1.;  // contribution of the orbit to the total mass
    return orbit::PtrRuntimeFnc(new RuntimeFncSchw(*grid, output));
}

const char* TargetDensity::name() const { return grid->name(); }

std::string TargetDensity::elemName(unsigned int index) const
{
    if(index+1 < constraintValues.size())
        return grid->elemName(index);
    else
        return "Total mass";
}


//----- Kinematic discretization scheme -----//

/// N-dimensional function that computes the amplitudes of B-spline representation of
/// squared radial and tangential velocity dispersions
template<int DEGREE>
class KinemJeansGrid: public math::IFunctionNdim {
    const math::BsplineInterpolator1d<DEGREE> bspl;
public:
    explicit KinemJeansGrid(const math::BsplineInterpolator1d<DEGREE>& _bspl) : bspl(_bspl) {}

    virtual void eval(const double point[6], double values[]) const
    {
        double r2  = pow_2(point[0]) + pow_2(point[1]) + pow_2(point[2]), r = sqrt(r2);
        double vr2 = pow_2(point[0] * point[3] + point[1] * point[4] + point[2] * point[5]) / r2;
        double vt2 = pow_2(point[3]) + pow_2(point[4]) + pow_2(point[5]) - vr2;
        bspl.eval(&r, values);
        unsigned int numBasisFnc = bspl.numValues();
        for(unsigned int b=0; b<numBasisFnc; b++) {
            values[b + numBasisFnc] = values[b] * vt2;
            values[b] *= vr2;
        }
    }
    virtual unsigned int numVars() const { return 6; }
    virtual unsigned int numValues() const { return bspl.numValues() * 2; }
};

/// auxiliary function for initializing the B-spline representation of the velocity dispersion
template<int DEGREE>
void setupKinemJeans(const std::vector<double>& gridr,
    const math::IFunction& dens, const math::IFunction& sigmar,
    /*output*/ std::vector<double>& constraintValues, shared_ptr<const math::IFunctionNdim>& grid)
{
    // compute the finite-element decomposition of the function  4pi rho(r) [r sigma_r(r)]^2
    math::FiniteElement1d<DEGREE> fem(gridr);
    grid.reset(new KinemJeansGrid<DEGREE>(fem.interp));
    
    std::vector<double> integrPoints = fem.integrPoints();
    std::vector<double> fncval(integrPoints.size());
    for(unsigned int i=0; i<integrPoints.size(); i++) {
        double r  = integrPoints[i];
        fncval[i] = dens(r) * 4*M_PI * pow_2(r * sigmar(r));
    }
    // amplitudes of FEM representation of the function rho * r^2 * sigma_r^2
    constraintValues = fem.computeProjVector(fncval);
    assert(constraintValues.size() == fem.interp.numValues());
}

TargetKinemJeans::TargetKinemJeans(
    const potential::BaseDensity& dens,
    const potential::BasePotential& pot,
    double beta,
    unsigned int gridSizeR,
    unsigned int degree)
{
    // first determine the grid radii that enclose the specified fractions of mass
    double totalMass = dens.totalMass();
    if(!isFinite(totalMass))
        throw std::runtime_error("TargetKinemJeans: total mass must be finite");
    std::vector<double> gridMass = math::createUniformGrid(
        gridSizeR, 0., totalMass * (1 - 1./gridSizeR));
    std::vector<double> gridr(gridSizeR);
    for(unsigned int s=1; s<gridSizeR; s++) {
        gridr[s] = getRadiusByMass(dens, gridMass[s]);
        if(!isFinite(gridr[s]) || gridr[s] <= gridr[s-1])
            throw std::runtime_error("TargetKinemJeans: cannot assign grid radii");
    }

    // sphericalized versions of density and potential (temporary objects created if needed)
    potential::PtrDensity sphDens;
    potential::PtrPotential sphPot;
    // wrapper functions for the original or the sphericalized density/potential
    math::PtrFunction fncDens;
    math::PtrFunction fncPot;
    if(isSpherical(pot)) {
        fncPot.reset(new potential::PotentialWrapper(pot));
    } else {
        sphPot = potential::Multipole::create(pot, /*lmax*/0, /*mmax*/0, /*gridSizeR*/50);
        fncPot.reset(new potential::PotentialWrapper(*sphPot));
    }
    if(isSpherical(dens)) {
        fncDens.reset(new potential::DensityWrapper(dens));
    } else {
        sphDens = potential::DensitySphericalHarmonic::create(dens,
            /*lmax*/0, /*mmax*/0, /*gridSizeR*/50, gridr[1]*0.1, gridr.back()*10.);
        fncDens.reset(new potential::DensityWrapper(*sphDens));
    }

    // construct the spherical Jeans model and compute the velocity dispersion as a function of radius
    math::LogLogSpline sigmar = createJeansSphModel(*fncDens, *fncPot, beta);

    // the choice of actual B-spline representation is done at runtime by the `degree` argument,
    // but it corresponds to different template specializations, therefore some setup tasks are
    // delegated to a dedicated templated routine
    switch(degree) {
        case 0: setupKinemJeans<0>(gridr, *fncDens, sigmar, /*output*/ constraintValues, grid); break;
        case 1: setupKinemJeans<1>(gridr, *fncDens, sigmar, /*output*/ constraintValues, grid); break;
        case 2: setupKinemJeans<2>(gridr, *fncDens, sigmar, /*output*/ constraintValues, grid); break;
        case 3: setupKinemJeans<3>(gridr, *fncDens, sigmar, /*output*/ constraintValues, grid); break;
        default: throw std::invalid_argument("TargetKinemJeans: degree of interpolation may not exceed 3");
    }

    // amplitudes of FEM representation for rho * r^2 * sigma_t^2, where sigma_t^2 = 2 (1-beta) sigma_r^2
    unsigned int numBasisFnc = constraintValues.size();
    constraintValues.resize(2 * numBasisFnc);
    for(unsigned int b=0; b<numBasisFnc; b++)
        constraintValues[b + numBasisFnc] = constraintValues[b] * 2 * (1-beta);
    // penalties
    constraintPenalties.assign(2 * numBasisFnc, 1.);
}

orbit::PtrRuntimeFnc TargetKinemJeans::getOrbitRuntimeFnc(StorageNumT* output) const
{
    return orbit::PtrRuntimeFnc(new RuntimeFncSchw(*grid, output));
}

std::string TargetKinemJeans::elemName(unsigned int index) const
{
    unsigned int numBasisFnc = constraintValues.size() / 2;
    if(index >= numBasisFnc)
        return "sigmat[" + utils::toString(index-numBasisFnc) + "]";
    else
        return "sigmar[" + utils::toString(index) + "]";
}

}  // namespace