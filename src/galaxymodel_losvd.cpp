#include "galaxymodel_losvd.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_fit.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <alloca.h>

namespace galaxymodel{

namespace {  // internal

class GaussianPSFfnc: public math::IFunctionNoDeriv, public GaussianPSF {
public:
    GaussianPSFfnc(const GaussianPSF& psf) : GaussianPSF(psf) {}
    virtual double value(const double x) const {
        // this represents a 2d PSF, i.e. convolved in both coordinates,
        // hence we use square root of the amplitude for each coordinate
        return sqrt(ampl) / (M_SQRT2 * M_SQRTPI * width) * exp(-0.5 * pow_2(x / width));
    }
};

template<int N>
std::vector<GaussianPSF> checkPSF(const std::vector<GaussianPSF>& gaussianPSF)
{
    // if there are no input PSFs provided, create a trivial one (zero-width and unit amplitude)
    if(gaussianPSF.empty())
        return std::vector<GaussianPSF>(1, GaussianPSF(0., 1.));
    double sumAmpl = 0.;
    for(size_t i=0; i<gaussianPSF.size(); i++)
        sumAmpl += gaussianPSF[i].ampl;
    if(fabs(sumAmpl-1.) > 1e-3)  // show a warning
        utils::msg(utils::VL_MESSAGE, "LOSVDGrid", "Amplitudes of input PSFs do not sum up to unity");
    return gaussianPSF;
}

template<int N>
math::Matrix<double> getConvolutionMatrix(
    const math::BsplineInterpolator1d<N> &bspl, const GaussianPSF &kernel)
{
    size_t size = bspl.numValues();
    math::FiniteElement1d<N> fem(bspl.xvalues());
    // matrix P - integrals of products of basis functions
    math::BandMatrix<double> proj = fem.computeProjMatrix();
    // matrix C is the convolution matrix if the PSF width is positive, otherwise equal to P
    math::Matrix<double> conv = kernel.width > 0 ?
        fem.computeConvMatrix(GaussianPSFfnc(kernel)) :
        math::Matrix<double>(proj);
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

//--------- VELOCITY MOMENTS ----------//

/** Accuracy parameter for integrating the product f(x)*exp(-x^2) over the entire real axis.
    When f is a polynomial, this integral can be exactly computed using the Gauss-Hermite
    quarature rule, but in our applications f(x) is only piecewise-polynomial, and there seems
    to be no easy-to-use generalization of this quadrature rule for finite intervals.
    Therefore we use a very simple-minded but surprisingly efficient approach:
    integration nodes are 2N^2+1 equally spaced points
    -N, ..., -1/N, 0, 1/N, 2/N, ..., N,
    and the integral is approximated as
    \f$   \int_{-\infty}^{\infty}  f(x) \exp(-x^2) dx  \approx
    (1/N) \sum_{i=-N^2}^{N^2}  f(i/N) \exp(-(i/N)^2)   \f$.
*/
static const int QUADORDER = 7;  // N=7, i.e. 99 integration nodes

void hermiteArray(const int nmax, const double x, double* result)
{
    // This is neither "probabilist's" nor "physicist's" definition of Hermite polynomials,
    // but rather "astrophysicist's" (with a different normalization).
    // dH_n/dx = \sqrt{2n} H_{n-1};
    // \int_{-\infty}^\infty dx H_n(x) H_m(x) \exp(-x^2) / (2\pi) = \delta_{mn} / (2 \sqrt{\pi});
    // \int_{-\infty}^\infty dx H_n(x) \exp(-x^2/2) / \sqrt{2\pi} = \sqrt{n!} / n!!  for even n.
    if(nmax<1)
        return;
    result[0] = 1.;
    if(nmax>=1)
        result[1] = M_SQRT2 * x;
    static const double sqroots[8] =
        { 1., sqrt(2.), sqrt(3.), 2., sqrt(5.), sqrt(6.), sqrt(7.), sqrt(8.) };
    double sqrtn = 1.;
    for(int n=1; n<nmax; n++) {
        double sqrtnplus1 = n<8 ? sqroots[n] : sqrt(n+1.);
        result[n+1] = (M_SQRT2 * x * result[n] - sqrtn * result[n-1]) / sqrtnplus1;
        sqrtn = sqrtnplus1;
    }
}

inline std::vector<double> computeGaussHermiteMoments( 
    unsigned int order, const math::IFunction& fnc, double gamma, double center, double sigma)
{
    std::vector<double> hpoly(order+1);   // temp.storage for Hermite polynomials
    std::vector<double> result(order+1);
    for(int p=0; p<=pow_2(QUADORDER); p++) {
        double y = p * (1./QUADORDER);    // equally-spaced points (only positive half of real axis)
        double mult = M_SQRT2 * sigma / gamma / QUADORDER * exp(-0.5*y*y);
        double fp = fnc(center + sigma * y);
        double fm = p==0 ? 0. : fnc(center - sigma * y);  // fnc value at symmetrically negative point
        hermiteArray(order, y, &hpoly[0]);
        for(unsigned int i=0; i<=order; i++)
            result[i] += mult * (fp + /*odd/even*/ (i%2 ? -1 : 1) * fm) * hpoly[i];
    }
    return result;
}

template<int N>
inline math::Matrix<double> computeGaussHermiteMatrix( 
    const unsigned int order, double gamma, double center, double sigma,
    const math::BsplineInterpolator1d<N>& interp)
{
    // the product of B-spline of degree N and a Hermite polynomial of degree 'order' is a polynomial
    // of degree N+order multiplied by an exponential function; we don't try to integrate it exactly,
    // but use a GL quadrature with Nnodes per each segment of the B-spline grid
    const int NnodesGL = std::max<int>((N+order+1)/2+1, 3);
    std::vector<double> glnodes(NnodesGL), glweights(NnodesGL);
    math::prepareIntegrationTableGL(0, 1, NnodesGL, &glnodes[0], &glweights[0]);
    std::vector<double> hpoly(order+1);   // temp.storage for Hermite polynomials
    double bspl[N+1];                     // temp.storage for B-splines
    const int gridSize = interp.xvalues().size(), numBsplines = interp.numValues();
    math::Matrix<double> result(order+1, numBsplines, 0.);
    double* dresult = result.data();      // shortcut for raw matrix storage
    for(int n=0; n<gridSize-1; n++) {
        const double x1 = interp.xvalues()[n], x2 = interp.xvalues()[n+1], dx = x2-x1;
        for(int k=0; k<NnodesGL; k++) {
            // evaluate the possibly non-zero B-splines and keep track of the index of the leftmost one
            const double x = x1 + dx * glnodes[k];
            unsigned int leftInd = interp.nonzeroComponents(x, /*derivOrder*/0, bspl);
            // evaluate the Hermite polynomials
            const double y = (x - center) / sigma;
            hermiteArray(order, y, &hpoly[0]);
            // overall multiplicative factor
            const double mult = M_SQRT2 / gamma * dx * glweights[k] * exp(-0.5*y*y);
            // add the contribution of this GL point to the integrals of H_m(x) * B_j(x),
            // where the index j runs from leftInd to leftInd+N
            for(unsigned int m=0; m<=order; m++)
                for(int b=0; b<=N; b++)
                    //result(m, b+leftInd) = ...
                    dresult[ m * numBsplines + b + leftInd ] += mult * hpoly[m] * bspl[b];
        }
    }
    return result;
}

class GaussianFitter: public math::IFunctionNdimDeriv {
    unsigned int order;
    const math::IFunction& fnc;
public:
    GaussianFitter(unsigned int _order, const math::IFunction& _fnc) : order(_order), fnc(_fnc) {}
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double gamma = vars[0], center = vars[1], sigma = vars[2], sqsigma = sqrt(sigma);
        std::vector<double> hpoly(order+1);
        for(int p=0; p <= 2*pow_2(QUADORDER); p++) {
            double y = (1./QUADORDER) * (p - pow_2(QUADORDER));  // equally spaced points
            double x = center + sigma * y;
            double f = fnc(x);
            hermiteArray(order, y, &hpoly[0]);
            double sum = 1.;
            for(unsigned int n=3; n<=order; n++)
                sum += vars[n] * hpoly[n];
            double mult = 1./M_SQRT2/M_SQRTPI/sqrt(QUADORDER) * exp(-0.5*y*y) * sum / sqsigma;
            if(values)
                values[p] = (1/sqrt(QUADORDER)) * sqsigma * f - mult * gamma;
            if(derivs) {
                derivs[p*(order+1)  ] = -mult;
                derivs[p*(order+1)+1] = -mult * gamma / sigma * y;
                derivs[p*(order+1)+2] =  mult * gamma / sigma * (1-y*y);
                for(unsigned int n=3; n<=order; n++)
                    derivs[p*(order+1)+n] = -mult * gamma / sum * hpoly[n];
            }
        }
    }
    virtual unsigned int numVars() const { return order+1; }
    virtual unsigned int numValues() const { return 2*pow_2(QUADORDER)+1; }
};

class MomentIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction& fnc;
    int n;
public:
    MomentIntegrand(const math::IFunction& _fnc, int _n) : fnc(_fnc), n(_n) {}
    virtual double value(const double x) const {
        return fnc(x) * math::pow(x, n);
    }
};

}  // internal ns

GaussHermiteExpansion::GaussHermiteExpansion(
    unsigned int order, const math::IFunction& fnc, double gamma, double center, double sigma) :
    Gamma(gamma), Center(center), Sigma(sigma)
{
    if(!isFinite(gamma + center + sigma)) {
        GaussianFitter fit(/*either "2" or "order"*/ 2, fnc);
        std::vector<double> params(order+1);
        // start the fit from a (hopefully) reasonable estimate based on the moments of the input function
        params[0] = math::integrateAdaptive(fnc, -INFINITY, INFINITY, 1e-3);      // normalization
        params[1] = math::integrateAdaptive(MomentIntegrand(fnc, 1),              // mean value
            -INFINITY, INFINITY, 1e-3) / params[0];
        params[2] = sqrt(fmax(0, math::integrateAdaptive(MomentIntegrand(fnc, 2), // dispersion
            -INFINITY, INFINITY, 1e-3) / params[0] - pow_2(params[1])));
        math::nonlinearMultiFit(fit, /*init*/&params[0], 1e-6, 100, /*output*/&params[0]);
        Gamma  = params[0];
        Center = params[1];
        Sigma  = params[2];
    }
    moments = computeGaussHermiteMoments(order, fnc, Gamma, Center, Sigma);
}

double GaussHermiteExpansion::value(const double x) const
{
    unsigned int ncoefs = moments.size();
    if(ncoefs==0) return 0.;
    double xscaled= (x - Center) / Sigma;
    double norm   = (1./M_SQRT2/M_SQRTPI) * Gamma / Sigma * exp(-0.5 * pow_2(xscaled));
    double* hpoly = static_cast<double*>(alloca(ncoefs * sizeof(double)));
    hermiteArray(ncoefs-1, xscaled, hpoly);
    double result = 0.;
    for(unsigned int i=0; i<ncoefs; i++)
        result += moments[i] * hpoly[i];
    return result * norm;
}

double GaussHermiteExpansion::normn(unsigned int n)
{
    if(n%2 == 1) return 0;  // odd GH function integrate to zero over the entire real axis
    switch(n) {
        case 0: return 1.;
        case 2: return 1./M_SQRT2;
        case 4: return 0.6123724356957945;  // sqrt(6)/4
        case 6: return 0.5590169943749474;  // sqrt(5)/4
        case 8: return 0.5229125165837972;  // sqrt(70)/16
        default: return sqrt(math::factorial(n)) / math::dfactorial(n);
    }
}

double GaussHermiteExpansion::norm() const {
    double result = 0;
    for(size_t n=0; n<moments.size(); n+=2)
        result += moments[n] * normn(n);
    return result * Gamma;
}

//----- LOSVDGrid -----//

template<int N>
LOSVDGrid<N>::LOSVDGrid(const LOSVDGridParams& params) :
    bsplx(params.gridx), bsply(params.gridy), bsplv(params.gridv),
    apertureConvolutionMatrix(params.apertures.size(), bsplx.numValues() * bsply.numValues(), 0.)
{
    const size_t
        numApertures = params.apertures.size(),
        numBasisFncX = bsplx.numValues(),  numBasisFncY = bsply.numValues(),
        numBasisFnc  = numBasisFncX * numBasisFncY,  numBasisFncX2 = pow_2(numBasisFncX);

    if(numApertures <= 0)
        throw std::invalid_argument("LOSVDGrid: no apertures defined");

    // construct the projection matrix for transforming the position/velocity in the
    // intrinsic 3d coordinate system into image plane coordinates and line-of-sight velocity
    math::makeRotationMatrix(params.theta, params.phi, params.chi, transformMatrix);

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
        utils::msg(utils::VL_MESSAGE, "LOSVDGrid", "Datacube does not cover all apertures");

    // ensure that there is at least one PSF, even with a zero width
    std::vector<GaussianPSF> spatialPSF = checkPSF<N>(params.spatialPSF);

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

template<int N>
void LOSVDGrid<N>::addPoint(const double point[6], const double mult, double* datacube) const
{
    double xp =   transformMatrix[0] * point[0] + transformMatrix[1] * point[1] + transformMatrix[2] * point[2];
    double yp =   transformMatrix[3] * point[0] + transformMatrix[4] * point[1] + transformMatrix[5] * point[2];
    // z' axis points towards the observer, so we have a minus sign for v_los
    double vl = -(transformMatrix[6] * point[3] + transformMatrix[7] * point[4] + transformMatrix[8] * point[5]);

    // find the index of grid segment in each dimension that this points belongs to,
    // and evaluate all nontrivial basis functions at this point in each dimension
    double weightx[N+1], weighty[N+1], weightv[N+1];
    int indx = bsplx.nonzeroComponents(xp, 0, weightx),
        indy = bsply.nonzeroComponents(yp, 0, weighty),
        indv = bsplv.nonzeroComponents(vl, 0, weightv);

    // add the contribution of this point to the datacube
    const size_t nx = bsplx.numValues(), nv = bsplv.numValues();
    for(int i=0; i<=N; i++)
        for(int j=0; j<=N; j++)
            for(int k=0; k<=N; k++)
                //datacube((indy + j) * nx + indx + i, indv + k) +=
                datacube[ ((indy + j) * nx + indx + i) * nv + indv + k ] +=
                    mult * weightx[i] * weighty[j] * weightv[k];
}

template<int N>
math::Matrix<double> LOSVDGrid<N>::getAmplitudes(const math::Matrix<double> &datacube) const
{
    // 1st stage: spatial convolution and rebinning
    math::Matrix<double> tmpmat(apertureConvolutionMatrix.rows(), bsplv.numValues());
    math::blas_dgemm(math::CblasNoTrans, math::CblasNoTrans,
        1., apertureConvolutionMatrix, datacube, 0., tmpmat);
    // 2nd stage: velocity convolution
    math::Matrix<double> result(apertureConvolutionMatrix.rows(), bsplv.numValues());
    math::blas_dgemm(math::CblasNoTrans, math::CblasTrans,
        1., tmpmat, velocityConvolutionMatrix, 0., result);
    return result;
}

template<int N>
math::Matrix<double> LOSVDGrid<N>::getGaussHermiteMatrix(
    unsigned int order, double gamma, double center, double sigma) const
{
    return computeGaussHermiteMatrix(order, gamma, center, sigma, bsplv);
}

// template instantiations
template class LOSVDGrid<0>;
template class LOSVDGrid<1>;
template class LOSVDGrid<2>;
template class LOSVDGrid<3>;

} // namespace