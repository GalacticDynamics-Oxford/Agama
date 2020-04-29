#include "math_gausshermite.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>
#include <alloca.h>

namespace math{

namespace {  // internal

/// relative accuracy in computing the moments of LOSVD (total normalization, mean value and dispersion)
static const double EPSREL_MOMENTS = 1e-3;


/// helper class for computing the integrals of f(x) times 1,x,x^2, using scaled integration variable
class MomentsIntegrand: public IFunctionNdim {
    const IFunction& fnc;
public:
    explicit MomentsIntegrand(const IFunction& _fnc) : fnc(_fnc) {}
    virtual unsigned int numVars()   const { return 1; }
    virtual unsigned int numValues() const { return 3; }
    virtual void eval(const double vars[], double values[]) const {
        // input scaled variable z ranges from 0 to 1, and maps to x as follows:
        double z = vars[0], x = exp(1/(1-z) - 1/z), j = x * (1/pow_2(1-z) + 1/pow_2(z));
        double fp = fnc(x), fm = fnc(-x);
        if((fp==0 && fm==0) || j==INFINITY) {
            values[0] = values[1] = values[2] = 0;
        } else {
            values[0] = (fp + fm) * j;
            values[1] = (fp - fm) * j * x;
            values[2] = (fp + fm) * j * x * x;
        }
    }
};

/// compute the 0th, 1st and 2nd moments of a probability distribution function:
/// f0 =   \int_{-\infty}^{\infty} f(x) dx                          (overall normalization)
/// f1 =  (\int_{-\infty}^{\infty} f(x) x dx) / f0                  (mean x)
/// f2 = ((\int_{-\infty}^{\infty} f(x) x^2 dx) / f0 - f1^2)^{1/2}  (standard deviation of x)
std::vector<double> computeClassicMoments(const IFunction& fnc)
{
    double result[3], zlower[1]={0.0}, zupper[1]={+1.0};
    integrateNdim(MomentsIntegrand(fnc),
        zlower, zupper, EPSREL_MOMENTS, /*maxNumEval*/1000, /*output*/result);
    std::vector<double> moments(3);
    moments[0] = result[0];
    moments[1] = result[0] != 0 ? result[1] / result[0] : 0;
    moments[2] = result[0] != 0 ? sqrt(fmax(0, result[2] / result[0] - pow_2(moments[1]))) : 0;
    return moments;
}


/** Accuracy parameter for integrating the product f(x)*exp(-x^2) over the entire real axis.
    When f is a polynomial, this integral can be exactly computed using the Gauss-Hermite
    quadrature rule, but for a general function f(x) there is another very simple-minded but
    surprisingly efficient approach: integration nodes are 2N^2+1 equally spaced points
    -N, ..., -1/N, 0, 1/N, 2/N, ..., N,
    and the integral is approximated as
    \f$   \int_{-\infty}^{\infty}  f(x) \exp(-x^2) dx  \approx
    (1/N) \sum_{i=-N^2}^{N^2}  f(i/N) \exp(-(i/N)^2)   \f$.
*/
static const int QUADORDER = 7;  // N=7, i.e. 99 integration nodes


/// compute the array of Hermite polynomials up to and including degree nmax at the given point
void hermiteArray(const int nmax, const double x, double* result)
{
    // This is neither "probabilist's" nor "physicist's" definition of Hermite polynomials,
    // but rather "astrophysicist's" (with a different normalization).
    // dH_n/dx = \sqrt{2n} H_{n-1};
    // \int_{-\infty}^\infty dx H_n(x) H_m(x) \exp(-x^2) / (2\pi) = \delta_{mn} / (2 \sqrt{\pi});
    // \int_{-\infty}^\infty dx H_n(x) \exp(-x^2/2) / \sqrt{2\pi} = \sqrt{n!} / n!!  for even n.
    result[0] = 1.;
    if(nmax<1)
        return;
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


/// compute the coefficients of GH expansion for an arbitrary function f(x)
inline std::vector<double> computeGaussHermiteMoments(const IFunction& fnc,
    unsigned int order, double ampl, double center, double width)
{
    std::vector<double> hpoly(order+1);   // temp.storage for Hermite polynomials
    std::vector<double> result(order+1);
    for(int p=0; p<=pow_2(QUADORDER); p++) {
        double y = p * (1./QUADORDER);    // equally-spaced points (only nonnegative half of real axis)
        double mult = M_SQRT2 * width / ampl / QUADORDER * exp(-0.5*y*y);
        double fp = fnc(center + width * y);
        double fm = p==0 ? 0. : fnc(center - width * y);  // fnc value at symmetrically negative point
        hermiteArray(order, y, &hpoly[0]);
        for(unsigned int m=0; m<=order; m++)
            result[m] += mult * (fp + /*odd/even*/ (m%2 ? -1 : 1) * fm) * hpoly[m];
    }
    return result;
}


/// compute the coefficients of GH expansion for a function f(x) that is a B-spline of degree N
template<int N>
inline std::vector<double> computeGaussHermiteMoments(const BsplineWrapper<N>& fnc,
    unsigned int order, double ampl, double center, double width)
{
    const std::vector<double>& grid = fnc.bspl.xvalues();
    const int NnodesGL = 4;  // number of nodes per segment of the B-spline grid
    const double *glnodes = GLPOINTS[NnodesGL], *glweights = GLWEIGHTS[NnodesGL];
    std::vector<double> hpoly(order+1);    // temp.storage for Hermite polynomials
    std::vector<double> result(order+1);
    for(size_t i=0; i<grid.size()-1; i++) {
        double x1= grid[i], x2=grid[i+1];  // endpoints of the current grid segment
        for(int k=0; k<NnodesGL; k++) {
            double
            x = glnodes[k] * x2 + (1-glnodes[k]) * x1,  // GL integration node within the segment
            y = (x-center) / width,
            mult = M_SQRT2 / ampl * exp(-0.5*y*y) * glweights[k] * (x2-x1) * fnc(x);
            hermiteArray(order, y, &hpoly[0]);
            for(unsigned int m=0; m<=order; m++)
                result[m] += mult * hpoly[m];
        }
    }
    return result;
}


/** compute the coefs of GH expansion for an array of B-spline basis functions of degree N.
    A function f(x) represented as a B-spline expansion with an array of amplitudes A_k
    f(x) = \sum_{j=1}^J A_j B_j(x)   (where B_j(x) are N-th degree B-splines over some grid)
    has the Gauss-Hermite coefficients given by h_m = C_{mj} A_j, where C_{mj} is the matrix
    returned by this routine.
*/
template<int N>
Matrix<double> computeGaussHermiteMatrix(const BsplineInterpolator1d<N>& interp, 
    unsigned int order, double ampl, double center, double width)
{
    // the product of B-spline of degree N and a Hermite polynomial of degree 'order' is a polynomial
    // of degree N+order multiplied by an exponential function; we don't try to integrate it exactly,
    // but use a Gauss-Legendre quadrature with Nnodes per each segment of the B-spline grid
    const int NnodesGL = std::min<int>(6, std::max<int>(3, (N+order+1)/2+1));
    const double *glnodes = GLPOINTS[NnodesGL], *glweights = GLWEIGHTS[NnodesGL];
    std::vector<double> hpoly(order+1);   // temp.storage for Hermite polynomials
    double bspl[N+1];                     // temp.storage for B-splines
    const int gridSize = interp.xvalues().size(), numBsplines = interp.numValues();
    Matrix<double> result(order+1, numBsplines, 0.);
    double* dresult = result.data();      // shortcut for raw matrix storage
    for(int n=0; n<gridSize-1; n++) {
        const double x1 = interp.xvalues()[n], x2 = interp.xvalues()[n+1], dx = x2-x1;
        for(int k=0; k<NnodesGL; k++) {
            // evaluate the possibly non-zero B-splines and keep track of the index of the leftmost one
            const double x = x1 + dx * glnodes[k];
            unsigned int leftInd = interp.nonzeroComponents(x, /*derivOrder*/0, /*output*/ bspl);
            // evaluate the Hermite polynomials
            const double y = (x - center) / width;
            hermiteArray(order, y, &hpoly[0]);
            // overall multiplicative factor
            const double mult = M_SQRT2 / ampl * dx * glweights[k] * exp(-0.5*y*y);
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


/** A helper class to be used in multidimensional minimization with the Levenberg-Marquardt method,
    when constructing the best-fit Gauss-Hermite approximation of a given function f(x).
    The GH expansion of the given order M has M+1 free parameters:
    amplitude, center and width of the base gaussian,
    and M-2 GH coefficients h_3, h_4, ..., h_M, with the convention that h_0=1, h_1=h_2=0.
    The fit minimizes the rms deviation between f(x) and the GH expansion g(x) specified by these 
    parameters over the set of Q points (Q = 2*QUADORDER^2+1 is fixed, and the points are equally spaced,
    but their location of these points depends on the mean and width of the current set of parameters).
    The evalDeriv() method returns the difference between f(x_k) and g(x_k) for each of these points x_k,
    and its partial derivatives w.r.t. all parameters of GH expansion, all used in the Levenberg-Marquardt
    fitting routine.
*/
class GaussHermiteFitterFnc: public IFunctionNdimDeriv {
    const unsigned int order;  ///< order of GH expansion
    const IFunction& fnc;      ///< function to be approximated
public:
    GaussHermiteFitterFnc(unsigned int _order, const IFunction& _fnc) : order(_order), fnc(_fnc) {}
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double ampl = vars[0], center = vars[1], width = vars[2], sqwidth = sqrt(width);
        double* hpoly = static_cast<double*>(alloca((order+1) * sizeof(double)));
        for(int p=0; p <= 2*pow_2(QUADORDER); p++) {
            double y = (1./QUADORDER) * (p - pow_2(QUADORDER));  // equally spaced points
            double x = center + width * y;
            double sum = 1.;
            if(order>2) {
                hermiteArray(order, y, hpoly);
                for(unsigned int n=3; n<=order; n++)
                    sum += vars[n] * hpoly[n];
            }
            double mult = 1./M_SQRT2/M_SQRTPI * exp(-0.5*y*y) * sum / sqwidth;
            if(values)
                values[p] = sqwidth * fnc(x) - mult * ampl;
            if(derivs) {
                derivs[p*(order+1)  ] = -mult;
                derivs[p*(order+1)+1] = -mult * ampl / width * y;
                derivs[p*(order+1)+2] =  mult * ampl / width * (1-y*y);
                for(unsigned int n=3; n<=order; n++)
                    derivs[p*(order+1)+n] = -mult * ampl / sum * hpoly[n];
            }
        }
    }
    virtual unsigned int numVars()   const { return order+1; }
    virtual unsigned int numValues() const { return 2*pow_2(QUADORDER)+1; };
};

/** Another helper class performing the same task (Levenberg-Marquardt fitting) but specialized
    to the case of input function represented by a B-spline interpolator.
    The difference lies in the method for computing the integral  \int |f(x)-g(x)|^2 dx:
    for a general function, we use a fixed uniform grid in the scaled variable y (argument of the
    exponent in GH expansion), while for a B-spline we use a fixed grid in the unscaled variable x
    tailored to the B-spline grid, to improve the integration accuracy for a non-smooth f(x).
    \tparam N is the degree of B-spline
*/
template<int N>
class GaussHermiteFitter: public IFunctionNdimDeriv {
    const unsigned int order;  ///< order of GH expansion
    /// nodes and weights of integration grid for B-spline, and function values at these points
    std::vector<double> nodes, weights, fvalues;
public:
    GaussHermiteFitter(unsigned int _order, const BsplineWrapper<N>& fnc) :
        order(_order)
    {
        const std::vector<double>& grid = fnc.bspl.xvalues();
        const int NnodesGL = 3;  // integration points per one segment of B-spline grid
        const double *glnodes = GLPOINTS[NnodesGL], *glweights = GLWEIGHTS[NnodesGL];
        nodes.  resize((grid.size()-1) * NnodesGL);
        weights.resize(nodes.size());
        fvalues.resize(nodes.size());
        for(size_t i=0; i<grid.size()-1; i++) {
            double x1= grid[i], x2=grid[i+1];  // endpoints of the current grid segment
            for(int k=0; k<NnodesGL; k++) {
                nodes  [i*NnodesGL+k] = glnodes[k] * x2 + (1-glnodes[k]) * x1;
                weights[i*NnodesGL+k] = sqrt(glweights[k] * (x2 - x1));
                fvalues[i*NnodesGL+k] = fnc(nodes[i*NnodesGL+k]);
            }
        }
    }
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double ampl = vars[0], center = vars[1], width = vars[2];
        double* hpoly = static_cast<double*>(alloca((order+1) * sizeof(double)));
        for(size_t p=0; p<nodes.size(); p++) {
            double y = (nodes[p] - center) / width;
            double sum = 1.;
            if(order>2) {
                hermiteArray(order, y, hpoly);
                for(unsigned int n=3; n<=order; n++)
                    sum += vars[n] * hpoly[n];
            }
            double mult = 1./M_SQRT2/M_SQRTPI * exp(-0.5*y*y) * weights[p] * sum / width;
            if(values)
                values[p] = weights[p] * fvalues[p] - mult * ampl;
            if(derivs) {
                derivs[p*(order+1)  ] = -mult;
                derivs[p*(order+1)+1] = -mult * ampl / width * y;
                derivs[p*(order+1)+2] =  mult * ampl / width * (1-y*y);
                for(unsigned int n=3; n<=order; n++)
                    derivs[p*(order+1)+n] = -mult * ampl / sum * hpoly[n];
            }
        }
    }
    virtual unsigned int numVars()   const { return order+1; }
    virtual unsigned int numValues() const { return nodes.size(); };
};

}  // internal ns

// constructor from a generic function
GaussHermiteExpansion::GaussHermiteExpansion(const IFunction& fnc,
    unsigned int order, double ampl, double center, double width) :
    Ampl(ampl), Center(center), Width(width)
{
    if(order<2)
        throw std::invalid_argument("GaussHermiteExpansion: order must be >=2");
    if(!isFinite(ampl + center + width)) {
        // estimate the first 3 moments of the function, which are used as starting values in the fit
        std::vector<double> params = computeClassicMoments(fnc);
        // now that we have a reasonable initial values for the moments of the input function,
        // perform a Levenberg-Marquardt optimization to find the best-fit parameters of the GH expansion.
        // Note that there are two conceptually different ways of fitting these parameters:
        // 1) determine only the overall amplitude, center and width of the best-fit Gaussian,
        // fixing h_0=1, h_1=h_2=0 and not considering higher-order terms.
        // 2) determine simultaneously the parameters ampl, center, width, h_3, ... h_M, while still
        // fixing h_0=1, h_1=h_2=0.
        // The latter approach, although seemingly natural, does not, in fact, fit a GH expansion:
        // if one computes all GH coefficients for the best-fit values, it turns out that h_1,h_2 != 0,
        // but they were ignored during the fit. Moreover, the best-fit values of center and sigma
        // (and hence all GH moments) depend on the chosen order of expansion.
        // By contrast, in the first case, the 0th basis function (the gaussian) is always the same,
        // and increasing the order of expansion does not change the values of previous terms.
        // This first choice also produces h_1=h_2=0, as is typically implied.
        // Note, however, that this is not the best-fit approximation at the given order
        // (neither is var.2 -- to obtain the absolute best fit, one would need to freely adjust
        // h_1 and h_2 during the fit).
        const unsigned int fitorder = 2;  // either "2" for the 1st var or "order" for the 2nd var
        params.resize(fitorder+1);
        nonlinearMultiFit(GaussHermiteFitterFnc(fitorder, fnc),
            /*init*/ &params[0], /*accuracy*/ 1e-6, /*max.num.fnc.eval.*/ 100, /*output*/ &params[0]);
        Ampl   = params[0];
        Center = params[1];
        Width  = params[2];
    }
    moments = computeGaussHermiteMoments(fnc, order, Ampl, Center, Width);
}

// constructor specialized to B-spline functions - same as above, just with a different implementations
// of GaussHermiteFitter and computeGaussHermiteMoments
template<int N>
GaussHermiteExpansion::GaussHermiteExpansion(const BsplineWrapper<N>& fnc,
    unsigned int order, double ampl, double center, double width) :
    Ampl(ampl), Center(center), Width(width)
{
    if(order<2)
        throw std::invalid_argument("GaussHermiteExpansion: order must be >=2");
    if(!isFinite(ampl + center + width)) {
        std::vector<double> params = computeClassicMoments(fnc);
        const unsigned int fitorder = 2;
        params.resize(fitorder+1);
        nonlinearMultiFit(GaussHermiteFitter<N>(fitorder, fnc),
            /*init*/ &params[0], /*accuracy*/ 1e-6, /*max.num.fnc.eval.*/ 100, /*output*/ &params[0]);
        Ampl   = params[0];
        Center = params[1];
        Width  = params[2];
    }
    moments = computeGaussHermiteMoments(fnc, order, Ampl, Center, Width);
}

// explicit template instantiations
template GaussHermiteExpansion::GaussHermiteExpansion(const BsplineWrapper<0>&, unsigned int, double, double, double);
template GaussHermiteExpansion::GaussHermiteExpansion(const BsplineWrapper<1>&, unsigned int, double, double, double);
template GaussHermiteExpansion::GaussHermiteExpansion(const BsplineWrapper<2>&, unsigned int, double, double, double);
template GaussHermiteExpansion::GaussHermiteExpansion(const BsplineWrapper<3>&, unsigned int, double, double, double);


double GaussHermiteExpansion::value(const double x) const
{
    unsigned int ncoefs = moments.size();
    if(ncoefs==0) return 0.;
    double xscaled= (x - Center) / Width;
    double norm   = (1./M_SQRT2/M_SQRTPI) * Ampl / Width * exp(-0.5 * pow_2(xscaled));
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
        default: return sqrt(factorial(n)) / dfactorial(n);
    }
}

double GaussHermiteExpansion::norm() const {
    double result = 0;
    for(size_t n=0; n<moments.size(); n+=2)
        result += moments[n] * normn(n);
    return result * Ampl;
}

Matrix<double> computeGaussHermiteMatrix(int N, const std::vector<double>& grid,
    unsigned int order, double ampl, double center, double width)
{
    switch(N) {
        case 0: return computeGaussHermiteMatrix(
            BsplineInterpolator1d<0>(grid), order, ampl, center, width);
        case 1: return computeGaussHermiteMatrix(
            BsplineInterpolator1d<1>(grid), order, ampl, center, width);
        case 2: return computeGaussHermiteMatrix(
            BsplineInterpolator1d<2>(grid), order, ampl, center, width);
        case 3: return computeGaussHermiteMatrix(
            BsplineInterpolator1d<3>(grid), order, ampl, center, width);
        default:
            throw std::invalid_argument("computeGaussHermiteMatrix: wrong B-spline degree");
    }
}

} // namespace