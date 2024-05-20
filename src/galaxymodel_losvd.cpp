#include "galaxymodel_losvd.h"
#include "galaxymodel_base.h"
#include "math_core.h"
#include "math_random.h"
#include "potential_base.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace galaxymodel{

namespace {  // internal

/// relative tolerance for computing the pixel masses multiplied by B-spline basis functions
static const double EPSREL_PIXEL_MASS = 1e-3;

/// max number of density evaluations per each pixel in the above integrals
static const int MAX_NUM_EVAL_PIXEL_MASS = 1e4;

/// max number of DF evaluations for constructing the LOSVD
static const int MAX_NUM_EVAL_LOSVD_DF = 1e6;

std::vector<GaussianPSF> checkPSF(const std::vector<GaussianPSF>& gaussianPSF)
{
    // if there are no input PSFs provided, create a trivial one (zero-width and unit amplitude)
    if(gaussianPSF.empty())
        return std::vector<GaussianPSF>(1, GaussianPSF(0., 1.));
    double sumAmpl = 0.;
    for(size_t i=0; i<gaussianPSF.size(); i++)
        sumAmpl += gaussianPSF[i].ampl;
    if(fabs(sumAmpl-1.) > 1e-3)  // show a warning
        utils::msg(utils::VL_MESSAGE, "TargetLOSVD", "Amplitudes of input PSFs do not sum up to unity");
    return gaussianPSF;
}


template<int N>
math::Matrix<double> getConvolutionMatrix(
    const math::BsplineInterpolator1d<N> &bspl, const GaussianPSF &psf)
{
    size_t size = bspl.numValues();
    math::FiniteElement1d<N> fem(bspl);
    // matrix P - integrals of products of basis functions
    math::BandMatrix<double> proj = fem.computeProjMatrix();
    // matrix C is the convolution matrix if the PSF width is positive, otherwise equal to P
    math::Matrix<double> conv(psf.width == 0 ?
        math::Matrix<double>(proj) :
        fem.computeConvMatrix(math::Gaussian(psf.width)));
    // multiply by the PSF amplitude: since it represents a 2d PSF, i.e. convolved in both coordinates,
    // we use the square root of the amplitude for each coordinate
    math::blas_dmul(sqrt(psf.ampl), conv);
    // product P^{-T} C^T
    math::Matrix<double> tmpd(size, size);
    // first compute D = P^{-T} C^T
    for(size_t i = 0; i < size; i++) {
        std::vector<double> row(&conv(i, 0), &conv(i, 0) + size);
        std::vector<double> vec = solveBand(proj, row);
        for(size_t j = 0; j < size; j++)
            tmpd(j, i) = vec[j];
    }
    // then compute P^{-1} D^T
    for(size_t i = 0; i < size; i++) {
        std::vector<double> col(&tmpd(i, 0), &tmpd(i, 0) + size);
        std::vector<double> vec = solveBand(proj, col);
        for(size_t j = 0; j < size; j++)
            conv(i, j) = vec[j];
    }
    return conv;
}


/// helper class for computing the surface density in the image plane, multiplied by
/// basis functions of a 2d tensor-product B-spline expansion.
template<int N>
class ApertureMassIntegrand: public math::IFunctionNdim {
    const potential::BaseDensity& density;        ///< density model 
    const double* mat;                            ///< orthogonal matrix for coordinate transformation
    const math::BsplineInterpolator1d<N>& bsplx;  ///< B-spline for the X coordinate in the image plane
    const math::BsplineInterpolator1d<N>& bsply;  ///< same for the Y coordinate
    const double scaleRadius;   ///< scaling radius for mapping the infinite interval in Z into [0:1]
public:
    ApertureMassIntegrand(
        const potential::BaseDensity& _density,
        const double* _transformMatrix,
        const math::BsplineInterpolator1d<N>& _bsplx,
        const math::BsplineInterpolator1d<N>& _bsply)
    :
        density(_density),
        mat(_transformMatrix),
        bsplx(_bsplx),
        bsply(_bsply),
        scaleRadius(fmax(
            fmax(fabs(bsplx.xmin()), fabs(bsplx.xmax())),
            fmax(fabs(bsply.xmin()), fabs(bsply.xmax()))
        ))
    {}

    /// input variables are rotation-transformed X, Y and Z
    virtual unsigned int numVars() const { return 3; }

    /// output values are the 2d tensor-product B-spline basis functions, multiplied by density
    virtual unsigned int numValues() const { return pow_2(N+1); }

    /// compute the density times all non-trivial B-spline basis functions at the given point X,Y,Z
    virtual void eval(const double vars[], double values[]) const
    {
        const double X = vars[0], Y = vars[1], w = vars[2],
        // transform the scaled variable w in the range [0:1] into Z
        Z   = scaleRadius * (1 / (1-w) - 1 / w),
        jac = scaleRadius * (1 / pow_2(1-w) + 1 / pow_2(w));
        // transform the rotated coords X,Y,Z back into the reference (un-rotated) frame,
        // multiplying this vector by the transposed rotation matrix
        // (since the inverse of an orthogonal matrix is just its transpose),
        // and compute the density at this point, multiplied by the jacobian of w -> Z mapping
        double val = jac * density.density(coord::PosCar(
            mat[0] * X + mat[3] * Y + mat[6] * Z,
            mat[1] * X + mat[4] * Y + mat[7] * Z,
            mat[2] * X + mat[5] * Y + mat[8] * Z));
        if(!isFinite(val))
            val = 0.;   // prevent failure in the case of error in density computation (e.g., at origin)
        // obtain the values of B-spline basis functions
        double weightx[N+1], weighty[N+1];
        bsplx.nonzeroComponents(X, 0, weightx);
        bsply.nonzeroComponents(Y, 0, weighty);
        // add the contribution of this point to the integrals
        for(int ky=0; ky<=N; ky++)
            for(int kx=0; kx<=N; kx++)
                values[ ky * (N+1) + kx ] = val * weightx[kx] * weighty[ky];
    }
};

template<int N>
class ApertureLOSVDIntegrand: public math::IFunctionNdim {
    const math::BsplineInterpolator1d<N>& bsplx;  ///< B-spline for the X coordinate in the image plane
    const math::BsplineInterpolator1d<N>& bsply;  ///< same for the Y coordinate
    const math::BsplineInterpolator1d<N>& bsplv;  ///< same for the V_Z coordinate
public:
    ApertureLOSVDIntegrand(
        const math::BsplineInterpolator1d<N>& _bsplx,
        const math::BsplineInterpolator1d<N>& _bsply,
        const math::BsplineInterpolator1d<N>& _bsplv) :
    bsplx(_bsplx), bsply(_bsply), bsplv(_bsplv)  {}

    /// input variables are cartesian position and velocity in the observationally-aligned frame
    virtual unsigned int numVars() const { return 6; }

    virtual unsigned int numValues() const { return pow_2(N+1) * bsplv.numValues(); }

    /// compute the density times all non-trivial B-spline basis functions at the given point X,Y,Z
    virtual void eval(const double vars[], double values[]) const
    {
        // find the index of grid segment in each dimension that this points belongs to,
        // and evaluate all nontrivial basis functions at this point in each dimension
        double weightx[N+1], weighty[N+1], weightv[N+1];
        /*      */ bsplx.nonzeroComponents(vars[0], 0, weightx);
        /*      */ bsply.nonzeroComponents(vars[1], 0, weighty);
        int indv = bsplv.nonzeroComponents(vars[5], 0, weightv);
        int nv   = bsplv.numValues();
        std::fill(values, values+numValues(), 0.);
        // add the contribution of this point to the integrals
        for(int ky=0; ky<=N; ky++)
            for(int kx=0; kx<=N; kx++)
                for(int kv=0; kv<=N; kv++)
                    values[ (ky * (N+1) + kx) * nv + indv + kv ] =
                        weightx[kx] * weighty[ky] * weightv[kv];
    }
};

}  // internal ns

//----- TargetLOSVD -----//

template<int N>
TargetLOSVD<N>::TargetLOSVD(const LOSVDParams& params) :
    orientation(
        // for axisymmetric systems, alpha has no effect; setting it to 0 simplifies bisymmetrization
        isAxisymmetric(params.symmetry) ? 0 : params.alpha,
        params.beta, params.gamma),
    bsplx(params.gridx), bsply(params.gridy), bsplv(params.gridv),
    apertureConvolutionMatrix(params.apertures.size(), bsplx.numValues() * bsply.numValues(), 0.),
    symmetry(params.symmetry), symmetricGrids(true)
{
    const size_t
        numApertures = params.apertures.size(),
        numBasisFncX = bsplx.numValues(),
        numBasisFncY = bsply.numValues(),
        numBasisFnc  = numBasisFncX * numBasisFncY,
        numBasisFncX2= pow_2(numBasisFncX);

    if(numApertures <= 0)
        throw std::invalid_argument("TargetLOSVD: no apertures defined");

    // check if input grids are reflection-symmetric:
    // if yes, will symmetrize the datacube after it has been fully assembled (often somewhat faster),
    // if not, will add the symmetric counterpart to each input point
    for(unsigned int i=0, size=bsplx.xvalues().size(); i<size/2; i++)
        symmetricGrids &= math::fcmp(bsplx.xvalues()[i], -bsplx.xvalues()[size-1-i], 1e-12) == 0;
    for(unsigned int i=0, size=bsply.xvalues().size(); i<size/2; i++)
        symmetricGrids &= math::fcmp(bsply.xvalues()[i], -bsply.xvalues()[size-1-i], 1e-12) == 0;
    for(unsigned int i=0, size=bsplv.xvalues().size(); i<size/2; i++)
        symmetricGrids &= math::fcmp(bsplv.xvalues()[i], -bsplv.xvalues()[size-1-i], 1e-12) == 0;

    // construct the spatial rebinning matrix
    math::Matrix<double> apertureMatrix(numApertures, numBasisFnc, 0.);
    volatile bool outOfBounds = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i = 0; i < (int)numApertures; i++) {
        bool apOutOfBounds = math::computeBsplineIntegralsOverPolygon(
            params.apertures[i], bsplx, bsply, &apertureMatrix(i, 0));
        outOfBounds |= apOutOfBounds;
    }
    if(outOfBounds)
        throw std::invalid_argument("TargetLOSVD: datacube does not cover all apertures");

    // ensure that there is at least one PSF, even with a zero width
    std::vector<GaussianPSF> spatialPSF = checkPSF(params.spatialPSF);

    // construct the combined aperture rebinning + spatial convolution matrix
    for(size_t g = 0; g < spatialPSF.size(); g++) {
        const math::Matrix<double> convx = getConvolutionMatrix(bsplx, spatialPSF[g]);
        const math::Matrix<double> convy = getConvolutionMatrix(bsply, spatialPSF[g]);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {   // define thread-local intermediate matrices
            math::Matrix<double> block(numBasisFnc, numBasisFncX), tmpprod(numApertures, numBasisFncX);
            // faster access to matrix elements in row-major order through pointers to flattened data
            const double *dconvx = convx.data(), *dprod = tmpprod.data();
            double *dconva = apertureConvolutionMatrix.data(), *dblock = block.data();

            // we need to compute the product Q = A L  of the matrix A (apertureMatrix)
            // having Na (numApertures) rows and Nx * Ny (numBasisFncX * numBasisFncY) columns
            // by a matrix L formed by outer product of two convolution matrices
            // Lx (convx) and Ly (convy):  L_{uw} = Lx_{lk} Ly_{ji}, where the combined indices
            // are u = Nx j + l, w = Nx i + k;   0 <= k,l < Nx,  0 <= i,j < Ny.
            // It would be impractical to assemble the entire matrix L (it may not even fit into memory),
            // so we multiply A by Ny vertically elongated sub-blocks of the matrix L, each with Nx
            // columns and Nx * Ny rows, and store the result in corresponding vertical stripes
            // of the matrix Q (Nx columns and Na rows). The outermost loop over the sub-blocks
            // is OpenMP-parallelized, as the destination regions do not overlap.
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
            for(int i = 0; i < (int)numBasisFncY; i++) {
                // assemble the sub-block with index i of the matrix L:
                // it consists of Ny copies of matrix Lx, stacked vertically,
                // each one multiplied by one element of matrix Ly from j-th row, i-th column
                for(size_t j = 0; j < numBasisFncY; j++) {
                    double convYji = convy(j, i);
                    // copy the entire matrix Lx multiplied by a constant (element [j,i] of matrix Ly)
                    // into a contiguous chunk of memory of length Nx^2
                    // (Nx consecutive rows of temporary matrix block, starting from row j)
                    for(size_t lk = 0, dest = j * numBasisFncX2; lk < numBasisFncX2; lk++, dest++)
                        dblock[dest] = dconvx[lk] * convYji;
                }
                // multiply the matrix A by the block and store the result in temporary product matrix
                math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans, 1.,
                    apertureMatrix, block, 0., tmpprod);
                // copy-add the result into the vertical stripe of destination matrix Q,
                // spanning all rows and Nx consecutive columns starting from i * Nx
                for(size_t a = 0; a < numApertures; a++) {
                    // 'a' is the row index in both the destination matrix Q and the temporary
                    // product matrix; copy Nx contiguous elements
                    for(size_t l = 0, src = a * numBasisFncX, dest = a * numBasisFnc + i * numBasisFncX;
                        l < numBasisFncX; l++, src++, dest++)
                        dconva[dest] += dprod[src];
                }
            }
        }
    }

    // construct the velocity convolution matrix
    velocityConvolutionMatrix = getConvolutionMatrix(bsplv, GaussianPSF(params.velocityPSF, 1.));
}

namespace{
// common fragment for adding a point
template<int N>
inline void reallyAddPoint(const math::BsplineInterpolator1d<N>& bsplx,
    const math::BsplineInterpolator1d<N>& bsply, const math::BsplineInterpolator1d<N>& bsplv,
    const double X, const double Y, const double V, const double mult, double* datacube)
{
    // quick check if the point is inside the grids at all
    if( X < bsplx.xmin() || X > bsplx.xmax() ||
        Y < bsply.xmin() || Y > bsply.xmax() ||
        V < bsplv.xmin() || V > bsplv.xmax() )
        return;   // don't waste time computing B-splines

    // find the index of grid segment in each dimension that this points belongs to,
    // and evaluate all nontrivial basis functions at this point in each dimension
    double weightx[N+1], weighty[N+1], weightv[N+1];
    int indx = bsplx.nonzeroComponents(X, 0, weightx),
        indy = bsply.nonzeroComponents(Y, 0, weighty),
        indv = bsplv.nonzeroComponents(V, 0, weightv);

    // add the contribution of this point to the datacube
    const int nx = bsplx.numValues(), nv = bsplv.numValues();
    for(int ky=0; ky<=N; ky++)
        for(int kx=0; kx<=N; kx++)
            for(int kv=0; kv<=N; kv++)
                //datacube((indy + ky) * nx + indx + kx, indv + kv) +=
                datacube[ ((indy + ky) * nx + indx + kx) * nv + indv + kv ] +=
                    mult * weightx[kx] * weighty[ky] * weightv[kv];
}
}

/*  Add a point sampled from the orbit during the current timestep to the datacube,
    performing various symmetrization procedures depending on the properties of the potential.
    \param[in]  point  is a 6d pos/vel point on the actual orbit;
    \param[in]  _mult  is its weight (interval of time associated with this orbital segment);
    \param[in/out]  datacube  stores the flattened 3d datacube (X, Y, V_Z).
    The symmetrization procedure involves several stages:
    0) if the potential is axisymmetric, the input point is rotated about the z axis by
    a random angle, and likewise if it is spherical, the point is rotated about a random axis
    in 3d by a random angle. The random number actually deterministically depends on the 6d point
    itself (i.e. is reproducible regardless of the order in which the orbits are integrated).
    1) the resulting point "pt" now needs to be transformed into the observed coordinate system,
    whose orientation is specified by the rotation matrix. Pos/vel in the intrinsic coord.sys.
    of the model are labelled by lowercase letters, and in the observed system - by uppercase.
    A point at x,y,z,vx,vy,vz and another one at -x,-y,-z,-vx,-vy,-vz (point-symmetric)
    belong to the same physical orbit (i.e. could have been sampled from the same trajectory
    at a different time), regardless of the orbit type and figure rotation.
    More symmetries are possible for triaxial systems (4 with figure rotation and 8 without),
    but these are different depending on the orbit type (short/long axis tube or box),
    so cannot applied at this stage when the orbit type is not known.
    However, in axisymmetric potentials all orbits are z-axis tubes, and one can apply a further
    symmetrization step to enforce that the _projected_ datacube is symmetric w.r.t. projected
    major axis (line of nodes); this corresponds to adding a point whose intrisic coordinates
    x',y',z' are related to the original point x,y,z in a rather nontrivial way.
*/
template<int N>
void TargetLOSVD<N>::addPoint(const double point[6], double _mult, double* datacube) const
{
    double mult=_mult;  // for some strange reason, Intel compiler complains about modifying _mult
    double pt[6];
    // construct initial state for the PRNG, using the input point position/velocity
    // as the source of "randomness"
    math::PRNGState state = math::hash(point, 6);
    if(isSpherical(symmetry)) {
        // if spherically-symmetric, randomize the orientation of the point on the 2d sphere
        coord::Orientation rot;
        math::getRandomRotationMatrix(/*output*/ rot.mat, /*input/output*/ &state);
        rot.toRotated(point+0, pt+0);  // position
        rot.toRotated(point+3, pt+3);  // velocity
    } else if(isAxisymmetric(symmetry)) {
        // if symmetric w.r.t rotation in phi, rotate the point about z axis by a random angle
        double ang = math::random(/*input/output*/ &state) * 2*M_PI, sa, ca;
        math::sincos(ang, sa, ca);
        pt[0] = point[0] * ca - point[1] * sa;
        pt[1] = point[0] * sa + point[1] * ca;
        pt[2] = point[2];
        pt[3] = point[3] * ca - point[4] * sa;
        pt[4] = point[3] * sa + point[4] * ca;
        pt[5] = point[5];
    } else {
        // copy the input point
        for(int i=0; i<6; i++)
            pt[i] = point[i];
    }

    // impose various additional symmetries by adding more than one point to the datacube
    double
    X0 = orientation.mat[0] * pt[0], X1 = orientation.mat[1] * pt[1], X2 = orientation.mat[2] * pt[2],
    Y0 = orientation.mat[3] * pt[0], Y1 = orientation.mat[4] * pt[1], Y2 = orientation.mat[5] * pt[2],
    V01= orientation.mat[6] * pt[3]   +   orientation.mat[7] * pt[4], V2 = orientation.mat[8] * pt[5];

    // 0. when have mirror symmetry (x,y,z <-> -x,-y,-z), and the datacube grids are symmetric,
    // no need to add the mirror point, because the symmetrization will be done afterwards
    // (it also corresponds to X,Y,V <-> -X,-Y,-V, i.e. point-symmetry of the kinematic dataset)
    bool addMirrorPoint = isReflSymmetric(symmetry) && !symmetricGrids;
    if(addMirrorPoint) mult *= 0.5;

    // 1. symmetrization w.r.t. z,vz <=> -z,-vz: this is applicable to boxes and short-axis tubes,
    // but will enforce long-axis tubes in non-axisymmetric potentials to have no net rotation,
    // thereby eliminating the possibility of kinematic twists. OTOH for rotating barred potentials,
    // un-symmetrized LAT are tilted w.r.t. the principal plane, so would break the triaxial
    // symmetry of the density profile. A proper solution is to decouple spatial and kinematic
    // symmetrization and make the latter dependent on the orbit type (i.e. LAT would be flipped
    // in x,vx, whereas SAT -- in z,vz), but since the orbit type is not known in advance before
    // it is completed, this would require storing several differently symmetrized datacubes and
    // combining some of them at the end of the orbit integration after determining the orbit class.
    // For now, we forfeit the possibility of modelling kinematic twists in triaxial systems and
    // always add a copy of input point at x,y,-z,vx,vy,-vz (it projects to different X,Y,V than
    // the original point).
    bool flipZ = isBisymmetric(symmetry);
    if(flipZ) mult *= 0.5;

    // 2. in an axisymmetric system, also ensure a symmetry w.r.t. change of inclination angle
    // beta <-> pi-beta, or flipping the observed datacube about the line of nodes - together with
    // the point-symmetry of step 0, this corresponds to fourfold (bi-symmetrization) of image plane.
    // it's equivalent to flipping the signs of 1,4 and 8-th elements of transformMatrix, or X1,Y1,V2;
    // there is no trivial relation between the original point x,y,z and this new point x',y',z'
    if(isAxisymmetric(symmetry)) mult *= 0.5;

    if(true) {
        if(true) {
            if(true)                      //  x,  y,  z
                reallyAddPoint(bsplx, bsply, bsplv, X0+X1+X2, Y0+Y1+Y2, V01+V2, mult, datacube);
            if(isAxisymmetric(symmetry))  //  x', y', z'
                reallyAddPoint(bsplx, bsply, bsplv, X0-X1+X2, Y0-Y1+Y2, V01-V2, mult, datacube);
        }
        if(flipZ) {
            if(true)                      //  x,  y, -z
                reallyAddPoint(bsplx, bsply, bsplv, X0+X1-X2, Y0+Y1-Y2, V01-V2, mult, datacube);
            if(isAxisymmetric(symmetry))  //  x', y',-z'
                reallyAddPoint(bsplx, bsply, bsplv, X0-X1-X2, Y0-Y1-Y2, V01+V2, mult, datacube);
        }
    }
    if(addMirrorPoint) {  // only if grids are not symmetric but we do need to point-symmetrize
        if(true) {
            if(true)                      // -x, -y, -z
                reallyAddPoint(bsplx, bsply, bsplv,-X0-X1-X2,-Y0-Y1-Y2,-V01-V2, mult, datacube);
            if(isAxisymmetric(symmetry))  // -x',-y',-z'
                reallyAddPoint(bsplx, bsply, bsplv,-X0+X1-X2,-Y0+Y1-Y2,-V01+V2, mult, datacube);
        }
        if(flipZ) {
            if(true)                      // -x, -y,  z
                reallyAddPoint(bsplx, bsply, bsplv,-X0-X1+X2,-Y0-Y1+Y2,-V01+V2, mult, datacube);
            if(isAxisymmetric(symmetry))  // -x',-y', z'
                reallyAddPoint(bsplx, bsply, bsplv,-X0+X1+X2,-Y0+Y1+Y2,-V01-V2, mult, datacube);
        }
    }
}

template<int N>
void TargetLOSVD<N>::finalizeDatacube(math::Matrix<double> &datacube, StorageNumT* output) const
{
    // 0th stage: mirror-symmetrization (if the model is invariant under x,y,z <-> -x,-y,-z,
    // and grids are reflection-symmetric, this can be done now for the entire datacube,
    // otherwise it was done for each added point individually)
    if(isReflSymmetric(symmetry) && symmetricGrids) {
        double* data = datacube.data();
        // average the symmetric elements from the head and tail of the flattened array
        for(size_t i=0, size = datacube.size(); i<size/2; i++)
            data[i] = data[size-1-i] = 0.5 * (data[i] + data[size-1-i]);
    }
    // 1st stage: spatial convolution and rebinning
    math::Matrix<double> tmpmat(apertureConvolutionMatrix.rows(), bsplv.numValues());
    math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans,
        1., apertureConvolutionMatrix, datacube, 0., tmpmat);
    // 2nd stage: velocity convolution
    math::Matrix<double> result(apertureConvolutionMatrix.rows(), bsplv.numValues());
    math::blas_dgemm(math::CblasNoTrans, math::CblasTrans,
        1., tmpmat, velocityConvolutionMatrix, 0., result);
    // store the matrix in the flattened output array
    const double *data = result.data();
    for(size_t i=0, size=result.size(); i<size; i++)
        output[i] = static_cast<StorageNumT>(data[i]);
}

template<int N>
void TargetLOSVD<N>::computeDFProjection(const GalaxyModel& model, StorageNumT* output) const
{
    // 1st stage: compute the integrals of the DF, weighted by the B-spline basis functions,
    // over each pixel of the regular 2d grid in the image plane (projections onto the B-spline basis)
    ApertureLOSVDIntegrand<N> fnc(bsplx, bsply, bsplv);
    math::Matrix<double> datacube = newDatacube();
    double* cubedata = datacube.data();
    int numPixels = (bsplx.xvalues().size()-1) * (bsply.xvalues().size()-1);
    // loop over pixels of the 2d B-spline grid in the image plane
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int p=0; p<numPixels; p++) {
        const int indx = p % (bsplx.xvalues().size()-1), indy = p / (bsplx.xvalues().size()-1),
        nx = bsplx.numValues(), nv = bsplv.numValues();
        // integration in a 2d rectangular pixel: X, Y are projected coords in the image plane
        double Xlim[2] = { bsplx.xvalues()[indx], bsplx.xvalues()[indx+1] };
        double Ylim[2] = { bsply.xvalues()[indy], bsply.xvalues()[indy+1] };
        std::vector<double> result(fnc.numValues());
        computeProjection(model, fnc, Xlim, Ylim, orientation,
            &result[0], EPSREL_PIXEL_MASS, MAX_NUM_EVAL_LOSVD_DF);
        // add the computed integrals to the output array
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for(int ky=0; ky<=N; ky++)
                for(int kx=0; kx<=N; kx++)
                    for(int iv=0; iv<nv; iv++)
                        cubedata[ ((indy + ky) * nx + indx + kx) * nv + iv ] +=
                            result[ (ky * (N+1) + kx) * nv + iv ];
        }
    }

    // 2nd stage: convert the collected datacube into the output array of LOSVDs in each aperture
    finalizeDatacube(datacube, output);
}

template<int N>
std::vector<double> TargetLOSVD<N>::computeDensityProjection(const potential::BaseDensity& density) const
{
    // 1st stage: compute the integrals of surface density, weighted by the B-spline basis functions,
    // over each pixel of the regular 2d grid in the image plane (projections onto the B-spline basis)
    ApertureMassIntegrand<N> fnc(density, orientation.mat, bsplx, bsply);
    std::vector<double> pixelMasses(bsplx.numValues() * bsply.numValues());
    int numPixels = (bsplx.xvalues().size()-1) * (bsply.xvalues().size()-1);
    // loop over pixels of the 2d B-spline grid in the image plane
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int p=0; p<numPixels; p++) {
        int ix = p % (bsplx.xvalues().size()-1), iy = p / (bsplx.xvalues().size()-1);
        // integration in a 3d slab: X, Y are projected coords in the image plane
        // (inside the current pixel), and w is the scaled Z coordinate (mapped onto the interval [0:1])
        double xywlow[3] = { bsplx.xvalues()[ix  ], bsply.xvalues()[iy  ], 0 };
        double xywupp[3] = { bsplx.xvalues()[ix+1], bsply.xvalues()[iy+1], 1 };
        double result[ (N+1) * (N+1) ];  // number of nonzero 2d basis elements in each pixel
        // compute the integrals
        math::integrateNdim(fnc, xywlow, xywupp, EPSREL_PIXEL_MASS, MAX_NUM_EVAL_PIXEL_MASS, result);
        // get the offset of the B-spline values in the output array
        double dummy[N+1];
        int indx = bsplx.nonzeroComponents(xywlow[0], 0, dummy),
            indy = bsply.nonzeroComponents(xywlow[1], 0, dummy);
        assert(indx == ix && indy == iy &&
            ix + N < (int)bsplx.numValues() && iy + N < (int)bsply.numValues());
        const int nx = bsplx.numValues();
        // add the computed integrals to the output array
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            for(int kx=0; kx<=N; kx++)
                for(int ky=0; ky<=N; ky++)
                    pixelMasses[(iy + ky) * nx + ix + kx] += result[ky * (N+1) + kx];
        }
    }

    // 2nd stage: convert these projections to the aperture masses (simultaneously convolving with PSF)
    std::vector<double> result(apertureConvolutionMatrix.rows());
    math::blas_dgemv(math::CblasNoTrans, 1., apertureConvolutionMatrix, pixelMasses, 0., result);
    return result;
}

template<int N>
std::string TargetLOSVD<N>::coefName(unsigned int index) const
{
    if(index >= numCoefs())
        throw std::out_of_range("TargetLOSVD: index out of range");
    return "Aperture[" + utils::toString(index / bsplv.numValues()) +
        "], velocity[" + utils::toString(index % bsplv.numValues()) + "]";
}

template<> const char* TargetLOSVD<0>::name() const { return "LOSVD0"; }
template<> const char* TargetLOSVD<1>::name() const { return "LOSVD1"; }
template<> const char* TargetLOSVD<2>::name() const { return "LOSVD2"; }
template<> const char* TargetLOSVD<3>::name() const { return "LOSVD3"; }

// template instantiations
template class TargetLOSVD<0>;
template class TargetLOSVD<1>;
template class TargetLOSVD<2>;
template class TargetLOSVD<3>;


//----- TargetKinemShell -----//

template<int N>
std::string TargetKinemShell<N>::coefName(unsigned int index) const
{
    return (index < bspl.numValues() ? "Vr[" : "Vt[") + utils::toString(index % bspl.numValues()) + "]";
}

template<int N>
void TargetKinemShell<N>::addPoint(const double point[6], double mult, double output[]) const
{
    double r2  = pow_2(point[0]) + pow_2(point[1]) + pow_2(point[2]), r = sqrt(r2);
    double vr2 = pow_2(point[0] * point[3] + point[1] * point[4] + point[2] * point[5]) / r2;
    double vt2 = pow_2(point[3]) + pow_2(point[4]) + pow_2(point[5]) - vr2;
    bspl.addPoint(&r, mult * vr2, output);
    bspl.addPoint(&r, mult * vt2, output + bspl.numValues());
}

template<int N>
void TargetKinemShell<N>::computeDFProjection(const GalaxyModel& model, StorageNumT* output) const
{
    // compute the moments of the DF at this many GL points on each segment of the radial grid
    static const int GLORDER = 4;
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
    math::Matrix<double> datacube = newDatacube();
    const int size = (bspl.xvalues().size()-1) * GLORDER;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int n=0; n<size; n++) {
        int i=n/GLORDER, k=n%GLORDER;  // decompose the combined index into the grid segment and offset
        double r = bspl.xvalues()[i] * (1-glnodes[k]) + bspl.xvalues()[i+1] * glnodes[k];
        double dens;
        coord::Vel2Car vel2;
        computeMoments(model, coord::PosCar(r,0,0), &dens, NULL, &vel2);
        double mult = dens * glweights[k] * 4*M_PI*r*r * (bspl.xvalues()[i+1] - bspl.xvalues()[i]);
#ifdef _OPENMP
#pragma omp critical
#endif
        {
            bspl.addPoint(&r, mult *  vel2.vx2, datacube.data());
            bspl.addPoint(&r, mult * (vel2.vy2 + vel2.vz2), datacube.data() + bspl.numValues());
        }
    }
    finalizeDatacube(datacube, output);
}

template<int N>
std::vector<double> TargetKinemShell<N>::computeDensityProjection(const potential::BaseDensity&) const
{
    // this operation does not make sense
    throw std::runtime_error("TargetKinemShell cannot be applied to a density object");
}

template<> const char* TargetKinemShell<0>::name() const { return "KinemShell0"; }
template<> const char* TargetKinemShell<1>::name() const { return "KinemShell1"; }
template<> const char* TargetKinemShell<2>::name() const { return "KinemShell2"; }
template<> const char* TargetKinemShell<3>::name() const { return "KinemShell3"; }

// template instantiations
template class TargetKinemShell<0>;
template class TargetKinemShell<1>;
template class TargetKinemShell<2>;
template class TargetKinemShell<3>;

} // namespace