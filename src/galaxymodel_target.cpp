#include "galaxymodel_target.h"
#include "galaxymodel_densitygrid.h"
#include "galaxymodel_jeans.h"
#include "galaxymodel_losvd.h"
#include "galaxymodel.h"
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

/// number of points taken from the trajectory during each timestep of the ODE solver
static const int NUM_SAMPLES_PER_STEP = 10;

/// helper routines for the templated class RuntimeFncSchw,
/// where FncType is the function that collects some data for each point sampled from trajectory.

/// allocate a new instance of (temporary) data storage
template<typename FncType>
math::Matrix<double> makeDatacube(const FncType& fnc);

/// convert the temporarily collected data to another array that is actually stored for each orbit
template<typename FncType>
math::Matrix<double> finalizeData(const FncType& fnc, const math::Matrix<double>& datacube);

/// generic implementation of the two above defined helper routines
template<>
inline math::Matrix<double> makeDatacube(const math::IFunctionNdimAdd& fnc)
{ return math::Matrix<double>(1, fnc.numValues(), 0.); }

template<>
inline math::Matrix<double> finalizeData(
    const math::IFunctionNdimAdd&, const math::Matrix<double>& datacube)
{ return datacube; }  // trivial copy

/// specialized implementation for the case of LOSVD grid, in which
/// the intermediate data collection array is different from the final storage array
template<>
inline math::Matrix<double> makeDatacube(const BaseLOSVDGrid& fnc)
{ return fnc.newDatacube(); }

template<>
inline math::Matrix<double> finalizeData(
    const BaseLOSVDGrid& fnc, const math::Matrix<double>& datacube)
{ return fnc.getAmplitudes(datacube); }  // non-trivial finalization


/// Orbit runtime function that collects the values of a given N-dimensional function
/// for each point on the trajectory, weighted by the amount of time spent at this point
/// \tparam  FncType  is a generic IFunctionNdimAdd or a specific descendant class such as LOSVDGrid
template<typename FncType>
class RuntimeFncSchw: public orbit::BaseRuntimeFnc {

    /// the function that collects some data for a given point
    /// (takes the position/velocity in Cartesian coordinates as input)
    const FncType& fnc;

    /// where the data for this orbit will be ultimately stored (points to an external array)
    StorageNumT* output;

    /** intermediate storage for the data collected during orbit integration,
        weighted by the time chunk associated with each sub-step on the trajectory;
        internally accumulated in double precision, and at the end of integration normalized
        by the integration time and written in the output array converted to StorageNumT
    */
    math::Matrix<double> datacube;

    /// total integration time - will be used to normalize the collected data
    /// at the end of orbit integration
    double time;

public:
    RuntimeFncSchw(const FncType& _fnc, StorageNumT* _output) :
        fnc(_fnc), output(_output), datacube(makeDatacube(fnc)), time(0.) {}

    /// finalize data collection, normalize the array by the total integration time,
    /// and convert to the numerical type used in the output storage
    virtual ~RuntimeFncSchw()
    {
        if(time==0) return;
        math::Matrix<double> data = finalizeData(fnc, datacube);
        const double *dataptr = data.data(), invtime = 1./time;
        for(size_t i=0, size = data.size(); i<size; i++)
            output[i] = static_cast<StorageNumT>(dataptr[i] * invtime);
        
    }

    /// collect the data returned by the function for each point sub-sampled from the trajectory
    /// on the current timestep, and add it to the temporary storage array,
    /// weighted by the duration of the substep
    virtual orbit::StepResult processTimestep(
        const math::BaseOdeSolver& solver, const double tbegin, const double tend, double[])
    {
        time += tend-tbegin;
        double substep = (tend-tbegin) / NUM_SAMPLES_PER_STEP;  // duration of each sub-step
        double *dataptr = datacube.data();
        for(int s=0; s<NUM_SAMPLES_PER_STEP; s++) {
            double point[6];  // position and velocity in cartesian coordinates at the current sub-step
            double tsubstep = tbegin + substep * (s+0.5);  // equally-spaced samples in time
            solver.getSol(tsubstep, point);
            fnc.addPoint(point, substep, dataptr);
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

//---- moments of a multi-component DF ----//

// wrapper function for computing a multi-component density by integrating each DF component over velocity
class MulticomponentDensityFromDF: public math::IFunctionNdim {
    const GalaxyModel& model;
public:
    explicit MulticomponentDensityFromDF(const GalaxyModel& _model) : model(_model) {}
    virtual unsigned int numVars() const { return 3; }  // a triplet of cylindrical coordinates
    virtual unsigned int numValues() const { return model.distrFunc.numValues(); }

    /// input: position in cylindrical coordinates, output: integrals of all DF components
    virtual void eval(const double vars[], double values[]) const {
        computeMoments(model, coord::PosCyl(vars[0], vars[1], vars[2]), values, NULL, NULL);
    }
};

}  // internal ns

//----- Density discretization scheme -----//

TargetDensity::TargetDensity(const potential::BaseDensity& density, const DensityGridParams& params) :
    lmax(0), mmax(0)   // will be initialized later
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
            lmax = mmax = params.stripsPerPane * 4;
            break;
        case DG_CLASSIC_LINEAR:
            grid.reset(new DensityGridClassic<1>(
                params.stripsPerPane, gridR, params.axisRatioY, params.axisRatioZ));
            lmax = mmax = params.stripsPerPane * 4;
            break;
        case DG_SPH_HARM:
            grid.reset(new DensityGridSphHarm(params.lmax, params.mmax, gridR));
            lmax = params.lmax;  mmax = params.mmax;
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
    if(isAxisymmetric(density)) mmax = 0;
    if(isSpherical   (density)) lmax = 0;

    // finally, compute the projection of the input density onto the grid
    constraintValues = grid->computeProjVector(density);
    // add the last constraint specifying the total mass
    constraintValues.push_back(totalMass);
}

orbit::PtrRuntimeFnc TargetDensity::getOrbitRuntimeFnc(StorageNumT* output) const
{
    output[constraintValues.size()-1] = 1.;  // contribution of the orbit to the total mass
    return orbit::PtrRuntimeFnc(new RuntimeFncSchw<math::IFunctionNdimAdd>(*grid, output));
}

const char* TargetDensity::name() const { return grid->name(); }

std::string TargetDensity::constraintName(size_t index) const
{
    if(index+1 < constraintValues.size())
        return grid->elemName(index);
    else
        return "Total mass";
}

void TargetDensity::computeDFProjection(const GalaxyModel& model, StorageNumT* output) const
{
    // BaseDensityGrid provides a method for computing the projection of a density onto the grid
    // by weighted integration of the density multiplied by each basis function.
    // The density, in turn, is given by an integral of the DF over velocities.
    // However, it would be too expensive to perform this integral at each point used in the projection;
    // instead, we compute the density on a coarser grid, roughly coinciding with the grid used
    // for density discretization, and then construct a suitable interpolator and provide it 
    // as the input density for the projection.
    // If the discretization scheme is based on a spherical grid (DensityGridClassic, DensityGridSphHarm),
    // we use spherical-harmonic expansion with linearly interpolated coefficients in radius;
    // otherwise for DensityGridCylindrical we use azimuthal Fourier expansion with linearly
    // interpolated coefficients in meridional plane (R,z).

    const MulticomponentDensityFromDF densWrapper(model);   // temporary density wrapper object

    // array of density interpolators for each component of the DF
    int numComp = model.distrFunc.numValues();
    std::vector<potential::PtrDensity> densityComponents(numComp);
    if(gridz.empty()) {
        // compute the spherical-harmonic expansion for the density of each DF component
        math::SphHarmIndices ind(lmax, mmax, coord::ST_TRIAXIAL);
        std::vector< std::vector< std::vector<double> > > sphHarmCoefs;
        potential::computeDensityCoefsSph(densWrapper, ind, gridR, /*output*/sphHarmCoefs);
        for(int c=0; c<numComp; c++)
            densityComponents[c].reset(new potential::DensitySphericalHarmonic(gridR, sphHarmCoefs[c]));
    } else {
        // TODO!!
        //       computeDensityComponentsCylindrical(df, mmax, gridR, gridz);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int c=0; c<numComp; c++) {
        std::vector<double> data = grid->computeProjVector(*densityComponents[c]);
        std::copy(data.begin(), data.end(), output + c * data.size());
    }
}

void TargetDensity::getMatrix(const math::IMatrixDense<StorageNumT>& recordedDatacube,
    math::IMatrixDense<StorageNumT>& result) const
{
    if( recordedDatacube.rows() != result.rows() ||
        recordedDatacube.cols() != datacubeSize() ||
        result.cols() != constraintsSize() )
        throw std::length_error("TargetDensity: invalid array sizes");
    utils::msg(utils::VL_MESSAGE, "TargetDensity",
        "No need to invoke the `matrix()` method, just use the original datacube");
    std::copy(recordedDatacube.data(), recordedDatacube.data() + recordedDatacube.size(), result.data());
}

//----- Kinematic discretization scheme -----//

/// N-dimensional function that computes the amplitudes of B-spline representation of
/// squared radial and tangential velocity dispersions
template<int DEGREE>
class KinemJeansGrid: public math::IFunctionNdimAdd {
    const math::BsplineInterpolator1d<DEGREE> bspl;
public:
    explicit KinemJeansGrid(const std::vector<double>& grid) : bspl(grid) {}

    virtual void addPoint(const double point[6], double mult, double output[]) const
    {
        double r2  = pow_2(point[0]) + pow_2(point[1]) + pow_2(point[2]), r = sqrt(r2);
        double vr2 = pow_2(point[0] * point[3] + point[1] * point[4] + point[2] * point[5]) / r2;
        double vt2 = pow_2(point[3]) + pow_2(point[4]) + pow_2(point[5]) - vr2;
        bspl.addPoint(&r, mult * vr2, output);
        bspl.addPoint(&r, mult * vt2, output + bspl.numValues());
    }
    virtual unsigned int numVars() const { return 6; }
    virtual unsigned int numValues() const { return bspl.numValues() * 2; }
};

TargetKinemJeans::TargetKinemJeans(const potential::BaseDensity& dens,
    unsigned int degree, unsigned int gridSizeR, double beta) :
    multbeta(static_cast<StorageNumT>(2*(1-beta)))
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
    // construct the appropriate finite-element grid
    switch(degree) {
        case 0: grid.reset(new KinemJeansGrid<0>(gridr)); break;
        case 1: grid.reset(new KinemJeansGrid<1>(gridr)); break;
        case 2: grid.reset(new KinemJeansGrid<2>(gridr)); break;
        case 3: grid.reset(new KinemJeansGrid<3>(gridr)); break;
        default: throw std::invalid_argument("TargetKinemJeans: degree of interpolation may not exceed 3");
    }
    numConstraints = grid->numValues() / 2;
}

orbit::PtrRuntimeFnc TargetKinemJeans::getOrbitRuntimeFnc(StorageNumT* output) const
{
    return orbit::PtrRuntimeFnc(new RuntimeFncSchw<math::IFunctionNdimAdd>(*grid, output));
}

std::string TargetKinemJeans::constraintName(size_t index) const
{
    return "beta[" + utils::toString(index) + "]";
}

void TargetKinemJeans::getMatrix(const math::IMatrixDense<StorageNumT>& recordedDatacube,
    math::IMatrixDense<StorageNumT>& result) const
{
    if( recordedDatacube.rows() != result.rows() ||
        recordedDatacube.cols() != datacubeSize() ||
        result.cols() != constraintsSize() )
        throw std::length_error("TargetKinemJeans: invalid array sizes");
    // get raw pointers for direct access to the matrix entries
    const StorageNumT* recordedDatacubePtr = recordedDatacube.data();
    StorageNumT* resultPtr = result.data();
    for(size_t i=0; i<result.rows(); i++)
        for(size_t c=0; c<numConstraints; c++)
            resultPtr[i * numConstraints + c] =
                recordedDatacubePtr[ i * 2 * numConstraints + c] * multbeta -
                recordedDatacubePtr[(i*2+1)* numConstraints + c];
}

//------//

TargetKinemLOSVD::TargetKinemLOSVD(const LOSVDGridParams& params, unsigned int degree,
    const std::vector<GaussHermiteExpansion>& ghexp) : GHexp(ghexp)
{
    switch(degree) {
        case 0: grid.reset(new LOSVDGrid<0>(params)); break;
        case 1: grid.reset(new LOSVDGrid<1>(params)); break;
        case 2: grid.reset(new LOSVDGrid<2>(params)); break;
        case 3: grid.reset(new LOSVDGrid<3>(params)); break;
        default: throw std::invalid_argument(
            "TargetKinemLOSVD: degree of interpolation may not exceed 3");
    }
    // obtain the grid dimensions
    math::Matrix<double> ampl = grid->getAmplitudes(grid->newDatacube());
    numApertures = ampl.rows();
    numBasisFncV = ampl.cols();
    assert(numApertures == params.apertures.size());
    // check the correctness of provided constraints (array of Gauss-Hermite expansions)
    if(ghexp.size() != numApertures)
        throw std::invalid_argument("TargetKinemLOSVD: "
            "number of provided Gauss-Hermite moments does not match the number of apertures");
    // determine the (max) order of Gauss-Hermite expansion
    numGHmoments = 0;
    for(size_t a=0; a<numApertures; a++)
        numGHmoments = std::max(numGHmoments, GHexp[a].coefs().size());
    if(numGHmoments <= 2)
        throw std::invalid_argument("TargetKinemLOSVD: "
            "order of Gauss-Hermite expansion should be at least two");
}

orbit::PtrRuntimeFnc TargetKinemLOSVD::getOrbitRuntimeFnc(StorageNumT* output) const
{
    return orbit::PtrRuntimeFnc(new RuntimeFncSchw<BaseLOSVDGrid>(*grid, output));
}

std::string TargetKinemLOSVD::constraintName(size_t index) const
{
    return "aperture[" + utils::toString(index / numGHmoments) +
        "], h" + utils::toString(index % numGHmoments);
}

double TargetKinemLOSVD::constraintValue(size_t index) const
{
    return GHexp.at(index / numGHmoments).coefs().at(index % numGHmoments);
}

void TargetKinemLOSVD::getMatrix(const math::IMatrixDense<StorageNumT>& recordedDatacube,
    math::IMatrixDense<StorageNumT>& result) const
{
    if( recordedDatacube.rows() != result.rows() ||
        recordedDatacube.cols() != datacubeSize() ||
        result.cols() != constraintsSize() )
        throw std::length_error("TargetKinemLOSVD: invalid array sizes");
    // get raw pointers for direct access to the matrix entries
    const StorageNumT* recordedDatacubePtr = recordedDatacube.data();
    StorageNumT* resultPtr = result.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int a=0; a<(int)numApertures; a++) {
        // obtain the matrix that converts the B-spline amplitudes into Gauss-Hermite moments
        math::Matrix<double> ghmat = grid->getGaussHermiteMatrix(
            numGHmoments-1, GHexp[a].gamma(), GHexp[a].center(), GHexp[a].sigma());
        std::vector<double> srcrow(numBasisFncV), dstrow(numGHmoments);  // temp storage
        // loop over all orbits
        for(size_t r=0; r<result.rows(); r++) {
            // convert the section of one row of the input array, corresponding to one aperture
            // and one orbit, from StorageNumT to double
            const StorageNumT* srcptr = recordedDatacubePtr + numBasisFncV * (r * numApertures + a);
            std::copy(srcptr, srcptr + numBasisFncV, srcrow.begin());
            // multiply the array of amplitudes by the conversion matrix
            math::blas_dgemv(math::CblasNoTrans, 1., ghmat, srcrow, 0., dstrow);
            // convert back to StorageNumT and write to a section of one row of the result array
            std::copy(dstrow.begin(), dstrow.end(), resultPtr + (r * numApertures + a) * numGHmoments);
        }
    }
}

}  // namespace