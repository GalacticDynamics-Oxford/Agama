/** \file    galaxymodel_losvd.h
    \brief   Line-of-sight velocity distribution in Schwarzschild/FEM models
    \date    2017-2019
    \author  Eugene Vasiliev

*/
#pragma once
#include "math_geometry.h"
#include "galaxymodel_target.h"

namespace galaxymodel{

/** Definition of a Gaussian point-spread function with the given width and amplitude */
struct GaussianPSF {
    /// width of the gaussian
    double width;
    /// amplitude (in case of several Gaussian components, their amplitudes are expected to sum up to unity)
    double ampl;
    GaussianPSF(double _width=NAN, double _ampl=1.) : width(_width), ampl(_ampl) {}
};

/** Parameters for handling the line-of-sight velocity distributions */
struct LOSVDParams {
    double alpha, beta, gamma;   ///< viewing angles for transforming the intrinsic to projected coords
    std::vector<double> gridx, gridy;     ///< internal grids in X',Y' (image plane coords)
    std::vector<double> gridv;            ///< grid in line-of-sight velocity
    std::vector<GaussianPSF> spatialPSF;  ///< array of spatial point-spread functions
    double velocityPSF;                   ///< width of the gaussian velocity smoothing kernel
    std::vector<math::Polygon> apertures; ///< array of apertures on the image plane
    coord::SymmetryType symmetry;         ///< symmetry of the potential and the orbital shape

    /// set (unreasonable) default values
    LOSVDParams() :
        alpha(0.), beta(0.), gamma(0.), velocityPSF(0.), symmetry(coord::ST_TRIAXIAL) {}
};


/** The class for recording the line-of-sight velocity distribution.
    It is represented in terms of a B-spline interpolator of degree N for each of spatial apertures
    (regions on the image plane delineated by arbitrary polygons).
    For each aperture we thus have a vector of B-spline amplitudes.
    The LOSVD is constructed in two steps:
    - first the raw data for an orbit is stored in an external array,
    by adding each point on the trajectory, weighted by the time spent at this point.
    The 6d phase-space point is first converted to a 2d point in the image plane and the velocity
    perpendicular to this plane, using the rotation matrix provided to the constructor.
    Then the contribution of this point to each of the basis functions in the internally managed
    3d tensor-product B-spline interpolator is computed. The maximum number of basis functions affected
    by a single point is (N+1)^3. This contribution, weighted with the provided multiplicative factor,
    is stored in an external 2d matrix.
    - at the end of orbit integration, this datacube is converted to a 2d matrix of amplitudes,
    where each row represents the data for a single aperture.
    The last step uses two auxiliary matrices for the spatial and velocity directions, correspondingly,
    which are initialized in the constructor.
    \tparam N  is the degree of B-spline interpolators (0,1,2 or 3).
    Higher-degree interpolators are more accurate and allow a larger pixel size
    (fewer expansion coefficients).
*/
template<int N>
class TargetLOSVD: public BaseTarget {
    const coord::Orientation orientation; ///< transforming between intrinsic and observed coords
    const math::BsplineInterpolator1d<N> bsplx, bsply, bsplv;  ///< basis-set interpolators
    math::Matrix<double> apertureConvolutionMatrix;  ///< spatial convolution and rebinning matrix
    math::Matrix<double> velocityConvolutionMatrix;  ///< velocity convolution matrix
    const coord::SymmetryType symmetry;   ///< symmetry of the potential and the orbital shape
    bool symmetricGrids;                  ///< whether the input grids are reflection-symmetric
public:
    /// construct the grid with given parameters.
    /// \throw std::invalid_argument if the parameters are incorrect.
    /// \note OpenMP-parallelized loop over params.apertures and over internal arrays.
    TargetLOSVD(const LOSVDParams& params);

    virtual const char* name() const;
    virtual std::string coefName(unsigned int index) const;

    /// return the total number of points in the flattened datacube
    virtual unsigned int numValues() const {
        return bsplx.numValues() * bsply.numValues() * bsplv.numValues();
    }

    /// return the number of coefficients in the output array:
    /// the number of apertures times the number of amplitudes of B-spline expansion of LOSVD
    virtual unsigned int numCoefs() const {
        return apertureConvolutionMatrix.rows() * bsplv.numValues();
    }

    /// allocate a new internal 3d data cube stored in a 2d matrix of the appropriate shape
    virtual math::Matrix<double> newDatacube() const {
        return math::Matrix<double>(bsplx.numValues() * bsply.numValues(), bsplv.numValues(), 0.);
    }

    /// add a weighted point to the datacube.
    /// \param[in]  point  is the 6d point in phase space: x,y,z,vx,vy,vz, which is
    /// internally converted to the sky plane position x', y' and the line-of-sight velocity v_los;
    /// \param[in]  mult  is the weight of input point;
    /// \param[in,out]  datacube points to the external array storing the flattened 3d data cube;
    /// all its elements that have a contribution from the input point are incremented by
    /// the weights of corresponding basis functions multiplied by the input factor 'mult'.
    virtual void addPoint(const double point[6], const double mult, double* datacube) const;

    /// convert the intermediate data stored in the regular 3d data cube
    /// into the array of basis function amplitudes for the LOSVD in each aperture
    virtual void finalizeDatacube(math::Matrix<double> &datacube, StorageNumT* output) const;

    /// compute the array of LOSVD in each apertures from a DF-based model.
    /// \note OpenMP-parallelized loop over the 2d grid in the image plane X,Y.
    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const;

    /// compute the normalizations of the LOSVD (total mass in each aperture, i.e., integral of
    /// surface density over the aperture, convolved with the spatial PSF).
    /// \note OpenMP-parallelized loop over the 2d grid in the image plane X,Y.
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const;
};


/** A simple class for recording radial and tangential velocity dispersions in spherical shells,
    represented by 1d B-spline interpolators in radius.
    \tparam N  is the degree of B-spline interpolators (0,1,2 or 3).
*/
template<int N>
class TargetKinemShell: public BaseTarget {
    const math::BsplineInterpolator1d<N> bspl;  ///< B-spline for representing rho * sigma^2
public:
    /// construct the target from the provided grid in spherical radius (should start at r=0)
    TargetKinemShell(const std::vector<double>& gridr) : bspl(gridr) {}
    virtual const char* name() const;
    virtual std::string coefName(unsigned int index) const;
    virtual void addPoint(const double point[6], double mult, double output[]) const;
    virtual unsigned int numVars() const { return 6; }
    virtual unsigned int numValues() const { return bspl.numValues() * 2; }

    /// compute the velocity dispersion profile from a DF-based model.
    /// \note OpenMP-parallelized loop over the radial grid.
    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const;

    /// this does not make sense for this target - throws a std::runtime_error
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity&) const;
};

}  // namespace
