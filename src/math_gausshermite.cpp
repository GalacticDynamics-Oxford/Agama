#include "math_gausshermite.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace math{

namespace {  // internal

/// relative accuracy in computing the moments of LOSVD (total normalization, mean value and width)
/// for the initial guess of the parameters of the Gauss-Hermite expansion
/// (these are later refined by finding a true best-fit Gaussian approximation to the input LOSVD,
/// so do not need to be computed very accurately).
static const double EPSREL_MOMENTS = 1e-3;

/// relative accuracy of finding a best-fit Gaussian for the given function
static const double EPSREL_GAUSSIAN_FIT = 1e-8;

/** Accuracy parameter for integrating the product f(x) * exp(-x^2) over the entire real axis.
    When f is a polynomial, this integral can be exactly computed using the Gauss-Hermite
    quadrature rule, but for a general function f(x) there is another very simple-minded but
    surprisingly efficient approach: integration nodes are 2N^2+1 equally spaced points
    -N, ..., -1/N, 0, 1/N, 2/N, ..., N,
    and the integral is approximated as
    \f$   \int_{-\infty}^{\infty}  f(x) \exp(-x^2) dx  \approx
    (1/N) \sum_{i=-N^2}^{N^2}  f(i/N) \exp(-(i/N)^2)   \f$.
*/
static const int QUADORDER = 7;  // N=7, i.e. 99 integration nodes


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
        values[0] = nan2num((fp + fm) * j);
        values[1] = nan2num((fp - fm) * j * x);
        values[2] = nan2num((fp + fm) * j * x * x);
    }
};

/// compute the 0th, 1st and 2nd moments of a probability distribution function:
/// f0 =   \int_{-\infty}^{\infty} f(x) dx                          (overall normalization)
/// f1 =  (\int_{-\infty}^{\infty} f(x) x dx) / f0                  (mean x)
/// f2 = ((\int_{-\infty}^{\infty} f(x) x^2 dx) / f0 - f1^2)^{1/2}  (standard deviation of x)
/// Note: this routine may fail to find the region of non-zero input function
/// if it is significantly offset from origin
void computeClassicMoments(const IFunction& fnc, /*output*/ double moments[3])
{
    double zlower[1]={0.0}, zupper[1]={+1.0};
    integrateNdim(MomentsIntegrand(fnc),
        zlower, zupper, EPSREL_MOMENTS, /*maxNumEval*/ 1000, /*output*/ moments);
    if(moments[0] != 0) {
        moments[1] /= moments[0];
        moments[2] = sqrt(fmax(0, moments[2] / moments[0] - pow_2(moments[1])));
    }
}

/// same as above, but for a function f(x) that is a piecewise-polynomial
/// (robust w.r.t. shifts from origin, unlike the above function)
void computeClassicMoments(const BaseInterpolator1d& fnc, /*output*/ double moments[3])
{
    for(int m=0; m<=2; m++)
        moments[m] = fnc.integrate(fnc.xmin(), fnc.xmax(), m);
    if(moments[0] != 0) {
        moments[1] /= moments[0];
        double disp = moments[2] / moments[0] - pow_2(moments[1]);
        // safety check: for a narrowly peaked function, the second moment may be dominated
        // by the noise in the tails, rather than the width of the peak itself.
        // To prevent it from becoming very small or even negative, impose the lower limit
        // of std.dev. to be the width of one grid cell near the peak of the function.
        ptrdiff_t ind = std::max((ptrdiff_t)0, std::min((ptrdiff_t)fnc.xvalues().size()-2,
            binSearch(moments[1], &fnc.xvalues()[0], fnc.xvalues().size())));
        moments[2] = sqrt(fmax(pow_2(fnc.xvalues()[ind+1]-fnc.xvalues()[ind]), disp));
    }
}


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
std::vector<double> computeGaussHermiteMoments(const IFunction& fnc,
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

/** compute the coefficients of GH expansion for a function f(x) that is a piecewise-polynomial
    with a maximum degree N on each segment. The expression for the GH moment h_m is
    \f$  h_m = \sqrt{2} / \Xi  \int_{-\infty}^{\infty}  f(x)  \exp(-y^2/2) H_m(y)  \f$,  where
    H_m(y)  are Hermite polynomials, y = (x-m)/s  is the scaled variable,
    m is the center, s is the width, and Xi is the amplidude of the base Gaussian.
    For obvious reasons, the integration is carried out separately in each segment of the grid
    of the input interpolator, and since the product of H_m(y) f(x(y)) is a polynomial of degree
    P = M + N  (where M is the order of GH expansion), the integral on each segment y_i..y_{i+1}
    can be computed analytically, using the Gaussian.integrate() method that provides the integral
    of the base Gaussian times y^p.
    To decompose the polynomial  H_m(y) * f(y * s + m)  into monomials on each segment for each m,
    we record its values at P+1 equally-spaced points on this segment, and then convert them into
    \f$  \sum_{p=0}^P  c_p y^p  \f$  using the vandermonde function.
*/
std::vector<double> computeGaussHermiteMoments(const BaseInterpolator1d& fnc,
    unsigned int order, double ampl, double center, double width)
{
    const std::vector<double>& grid = fnc.xvalues();
    std::vector<double> result(order+1);
    const Gaussian gaussian(1.0);        // unit Gaussian in scaled variable y
    const int P = order + fnc.degree();  // max degree of monomials
    const double mult = 2*M_SQRTPI / ampl * width;
    // coordinates of equally-spaced points on the current grid segment (in the scaled variable y)
    double* ynodes = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // values of  f(x(y)) * H_m(y)  for all these points and all m<=M Hermite polynomials
    double* yvalues = static_cast<double*>(alloca((P+1) * (order+1) * sizeof(double)));
    // values of all Hermite polynomials H_m(y), m<=M, at the current point y
    double* hpoly = static_cast<double*>(alloca((order+1) * sizeof(double)));
    // coefficients  c_m  of the decomposition of  f(x(y)) * H_m(y)  into monomials y^p
    double* monomialCoefs = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // integrals of  exp(-y^2/2) * y^p  for all p<=P on the current grid segment
    double* monomialIntegrals = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // integration is carried out separately on each grid segment of the input interpolator
    for(size_t i=0; i<grid.size()-1; i++) {
        // first, loop over P+1 equally-spaced points on this grid segment, collecting
        // the values of f(x) and all M+1 Hermite polynomials H_m at each point y_p, 0<=p<=P,
        // as well as the integrals of  exp(-y^2/2) * y^p  over this segment.
        for(int p=0; p<=P; p++) {
            monomialIntegrals[p] = gaussian.integrate(
                (grid[i] - center) / width, (grid[i+1] - center) / width, p);
            double x  = grid[i] + (grid[i+1] - grid[i]) * (p+0.5) / (P+1);
            ynodes[p] = (x-center) / width;
            double fx = fnc(x);
            hermiteArray(order, ynodes[p], /*output*/ hpoly);
            for(unsigned int m=0; m<=order; m++)
                yvalues[m * (P+1) + p] = fx * hpoly[m];
        }
        // second, loop over the M+1 Hermite polynomials, determining the coefficients c_m
        // of the decomposition of  f(x(y)) * H_m(y)  into monomials y^p on this grid segment,
        // and then collect the contribution of each monomial to the integral for h_m.
        for(unsigned int m=0; m<=order; m++) {
            vandermonde(P, ynodes, yvalues + m * (P+1), /*output*/ monomialCoefs);
            // finally
            for(int p=0; p<=P; p++)
                result[m] += mult * monomialCoefs[p] * monomialIntegrals[p];
        }
    }
    return result;
}


/** compute the coefs of GH expansion for an array of B-spline basis functions of degree N.
    A function f(x) represented as a B-spline expansion with an array of amplitudes A_k
    \f$  f(x) = \sum_{j=1}^J A_j B_j(x)  \f$,
    where B_j(x) are N-th degree B-splines over some grid,
    has the Gauss-Hermite coefficients given by  \f$  h_m = C_{mj} A_j  \f$,
    where C_{mj} is the matrix returned by this routine.
    The approach for computing the coefficients C_{mj} is similar to the previous routine:
    add up the integrals over each grid segment of the B-spline,
    which are calculated analytically by decomposing the integrand into monomials.
*/
template<int N>
Matrix<double> computeGaussHermiteMatrix(const BsplineInterpolator1d<N>& interp, 
    unsigned int order, double ampl, double center, double width)
{
    const std::vector<double>& grid = interp.xvalues();
    const int numBsplines = interp.numValues();
    Matrix<double> result(order+1, numBsplines, 0.);
    double* dresult = result.data();  // shortcut for raw matrix storage
    double bspl[N+1];                 // temp.storage for B-splines
    const Gaussian gaussian(1.0);     // unit Gaussian in scaled variable y
    const int P = order + N;          // max degree of monomials
    const double mult = 2*M_SQRTPI / ampl * width;
    // coordinates of equally-spaced points on the current grid segment (in the scaled variable y)
    double* ynodes = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // values of  H_m(y) * B_j(x(y))  for all these points, all m<=M Hermite polynomials
    // and all b<=N nonzero B-spline basis functions
    double* yvalues = static_cast<double*>(alloca((P+1) * (order+1) * (N+1) * sizeof(double)));
    // values of all Hermite polynomials H_m(y), m<=M, at the current point y
    double* hpoly = static_cast<double*>(alloca((order+1) * sizeof(double)));
    // coefficients  c_m  of the decomposition of  B_b(x(y)) * H_m(y)  into monomials y^p
    double* monomialCoefs = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // integrals of  exp(-y^2/2) * y^p  for all p<=P on the current grid segment
    double* monomialIntegrals = static_cast<double*>(alloca((P+1) * sizeof(double)));
    // integration is carried out separately on each grid segment of the input B-spline interpolator
    for(size_t i=0; i<grid.size()-1; i++) {
        unsigned int leftInd = i;  // index of the first nonzero B-spline basis function on this segment
        // first, loop over P+1 equally-spaced points y_p on this grid segment, collecting
        // the values of all N+1 B-spline basis functions and all M+1 Hermite polynomials H_m,
        // as well as the integrals of  exp(-y^2/2) * y^p  over this segment.
        for(int p=0; p<=P; p++) {
            // integrals of exp(-y^2/2) * y^p over this grid segment for all monomial degrees p<=P
            monomialIntegrals[p] = gaussian.integrate(
                (grid[i] - center) / width, (grid[i+1] - center) / width, p);
            double x  = grid[i] + (grid[i+1] - grid[i]) * (p+0.5) / (P+1);
            ynodes[p] = (x-center) / width;
            // evaluate all N+1 possibly non-zero B-spline basis functions B_b(x(y)) at this point y_p
            leftInd = interp.nonzeroComponents(x, /*derivOrder*/0, /*output*/ bspl);
            // evaluate all M+1 Hermite polynomials at this point y_p
            hermiteArray(order, ynodes[p], hpoly);
            // store the products of all combinations of B-spline basis functions and Hermite polynomials
            for(unsigned int m=0; m<=order; m++)
                for(int b=0; b<=N; b++)
                    yvalues[(m * (N+1) + b) * (P+1) + p] = bspl[b] * hpoly[m];
        }
        // second, loop over the M+1 Hermite polynomials and N+1 B-spline basis functions,
        // determining the coefficients c_m of the decomposition of  B_b(x(y)) * H_m(y)
        // into monomials y^p on this grid segment, and then collect the contribution of
        // each monomial to the integral for H_m(x) * B_j(x), where the index j = leftInd + b
        // and  0<=b<=N  enumerates all nonzero B-spline basis functions on this segment.
        for(unsigned int m=0; m<=order; m++) {
            for(int b=0; b<=N; b++) {
                vandermonde(P, ynodes, yvalues + (m * (N+1) + b) * (P+1), /*output*/ monomialCoefs);
                for(int p=0; p<=P; p++)
                    //result(m, b+leftInd) = ...
                    dresult[ m * numBsplines + b + leftInd ] +=
                        mult * monomialCoefs[p] * monomialIntegrals[p];
            }
        }
    }
    return result;
}


/** A helper class to be used in multidimensional minimization with the Levenberg-Marquardt method,
    when constructing the best-fit Gaussian approximation g(x) of a given function f(x),
    which has three free parameters:  amplitude, center and width.
    The fit minimizes the rms deviation between f(x) and the Gaussian expansion g(x)
    over the set of Q points (Q = 2*QUADORDER^2+1 is fixed, and the points are equally spaced,
    but their location of these points depends on the mean and width of the Gaussian).
    The evalDeriv() method returns the difference between f(x_k) and g(x_k) for each of
    these points x_k, and its partial derivatives w.r.t. parameters the Gaussian,
    all used in the Levenberg-Marquardt fitting routine.
    The free parameters in the fit are renormalized to make them of order unity,
    as explained in the docstring of GaussianFitterInterp.
*/
class GaussianFitterFnc: public IFunctionNdimDeriv {
    const IFunction& fnc;  ///< function to be approximated
    const double a0, s0;   ///< renormalization factors
public:
    GaussianFitterFnc(const IFunction& _fnc, double _a0, double _s0) :
        fnc(_fnc), a0(_a0), s0(_s0)
    {}

    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double ampl = vars[0], center = vars[1], width = vars[2], sqwidth = sqrt(width);
        for(int p=0; p <= 2*pow_2(QUADORDER); p++) {
            double y = (1./QUADORDER) * (p - pow_2(QUADORDER));  // equally spaced points
            double x = center + width * y;
            double mult = 1./M_SQRT2/M_SQRTPI * exp(-0.5*y*y) / sqwidth;
            if(values)
                values[p] = sqwidth * fnc(x * s0) * s0 / a0 - mult * ampl;
            if(derivs) {
                derivs[p*3  ] = -mult;
                derivs[p*3+1] = -mult * ampl / width * y;
                derivs[p*3+2] =  mult * ampl / width * (1-y*y);
            }
        }
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 2*pow_2(QUADORDER)+1; };
};

/** Another helper class performing the same task: find the best-fit Gaussian approximation
    \f$  g(x) = a / \sqrt(2\pi) / s * \exp(-0.5 * (x-m)^2 / s^2)  \f$
    of an input function  f(x), specialized to the case of a piecewise-polynomial function f(x)
    (e.g., a B-spline).
    Here  a = amplitude of the Gaussian, m = center, s = width  are the parameters to be optimized.
    Unlike the previous class, this one is used with the derivative-assisted multidimensional
    minimization routine rather than the Levenberg-Marquardt routine, and for each choice of these
    three parameters of the Gaussian, returns the integral of the squared difference between f and g:
    \f$  E = (1/2) \int_{-\infty}^\infty { |f(x) - g(x)|^2  -  f(x)^2 }  dx  \f$.
    The last term in the integrand, f(x)^2, is a constant that does not depend on the parameters,
    so is subtracted for simplicity, hence the integrand is  (1/2) g(x)^2 - f(x) * g(x).
    The integration is carried out analytically using the same approach as in the routines
    computeGaussHermiteMoments() and computeGaussHermiteMatrix(), namely, splitting the integral
    into the segments in which the input function f(x) is decomposed into monomials.
    An additional complication arises from the fact that the multidimensional minimization routine
    uses an absolute tolerance on the function gradient as the stopping criterion.
    To make the procedure truly scale-invariant, the fit is carried out for renormalized parameters:
    a / a0, m / s0, s / s0, where a0 is the initial guess for the overall amplitude (0th moment of f)
    and s0 is its characteristic scale (square root of the full 2nd moment of f).
    Thus the renormalized parameters, as well as the integrated error E, should all be of order unity.
*/
class GaussianFitterInterp: public IFunctionNdimDeriv {
    /// polynomial degree of the input function
    const int N;
    /// grid of the input piecewise-polynomial function, renormalized by s0
    std::vector<double> grid;
    /// values of the input function at N+1 points on each grid segment, renormalized by a0/s0
    std::vector<double> fncValues;
public:
    GaussianFitterInterp(const BaseInterpolator1d& fnc, double a0, double s0) :
        N(fnc.degree())
    {
        const std::vector<double>& origGrid = fnc.xvalues();
        grid.resize(origGrid.size());
        fncValues.resize((origGrid.size()-1) * (N+1));
        for(size_t i=0; i<origGrid.size(); i++) {
            grid[i] = origGrid[i] / s0;
            if(i == origGrid.size()-1) continue;
            for(int n=0; n<=N; n++)
                fncValues[i * (N+1) + n] =
                    fnc(origGrid[i] + (origGrid[i+1] - origGrid[i]) * (n+0.5) / (N+1)) / a0 * s0;
        }
    }

    /// computes the integral E, and optionally its first and second derivatives w.r.t. a,m,s.
    void evalDeriv2(const double vars[], double values[], double derivs[], double derivs2[]) const
    {
        double a = vars[0], m = vars[1], s = vars[2];  // parameters of the Gaussian g(x)
        if(values)  // contribution of the integral of 0.5 * g(x)^2 over the real axis
            values[0] = 0.25 / M_SQRTPI * a * a / fabs(s);
        // derivatives of E w.r.t. a, m, s
        if(derivs) {  // derivatives of the same integral
            derivs[0] = 0.5 / M_SQRTPI * a / fabs(s);
            derivs[1] = 0;
            derivs[2] = -0.5 * derivs[0] * a / s;
        }
        if(derivs2) {  // second derivatives
            derivs2[0] = 0.5 / M_SQRTPI / fabs(s);  // d2E / da2
            derivs2[1] = 0;                         // d2E / da dm;  derivs[3] is the same
            derivs2[2] = -derivs2[0] * a / s;       // d2E / da ds;  derivs[6] is the same
            derivs2[4] = 0;                         // d2E / dm2
            derivs2[5] = 0;                         // d2E / dm ds;  derivs[7] is the same
            derivs2[8] = -derivs2[2] * a / s;       // d2E / ds2
        }
        // the input function is a polynomial of degree at most N on each segment,
        // so to decompose it into monomials, its values at N+1 equally-spaced points
        // on each segment were recorded in the constructor;
        // here we only need to convert the coordinates of these points into the scaled variable y.
        double* ynodes = static_cast<double*>(alloca((N+1) * sizeof(double)));
        // coefficients c_n of the input function expressed as \f$  \sum_{n=0}^N c_n y^n  \f$.
        double* monomialCoefs = static_cast<double*>(alloca((N+1) * sizeof(double)));
        // integrals of  \f$  1/\sqrt(2\pi) \exp[-(1/2) y^2] y^p  \f$  over the current grid segment
        // for all monomials with p<=P  (the max degree P<=N+4 is higher than the input function,
        // since the first and second derivatives of E involve additional powers of y).
        const int P = derivs2 ? N+4 : derivs ? N+2 : N;
        double* gaussianIntegrals = static_cast<double*>(alloca((P+1) * sizeof(double)));
        Gaussian gaussian(1);
        // integration is carried out separately on each grid segment of the input function
        for(size_t i=0; i<grid.size()-1; i++) {
            // endpoints of the current segment in the scaled variable  y = (x-m) / s
            double yleft = (grid[i] - m) / s, yright = (grid[i+1] - m) / s;
            // analytic integrals of the unit Gaussian times various monomials on this segment
            for(int p=0; p<=P; p++)
                gaussianIntegrals[p] = gaussian.integrate(yleft, yright, p);
            // decompose the input function into the sum of monomial terms in y on this segment
            for(int n=0; n<=N; n++)
                ynodes[n] = yleft + (yright - yleft) * (n+0.5) / (N+1);
            vandermonde(N, ynodes, &(fncValues[i * (N+1)]), /*output*/ monomialCoefs);
            // add the contribution of each monomial term to the total integral and its derivatives
            for(int n=0; n<=N; n++) {
                if(values) {
                    values[0] += -a * monomialCoefs[n] * gaussianIntegrals[n];
                }
                if(derivs) {
                    derivs[0] += -monomialCoefs[n] * gaussianIntegrals[n];
                    derivs[1] += -a / s * monomialCoefs[n] * gaussianIntegrals[n+1];
                    derivs[2] +=  a / s * monomialCoefs[n] *
                        (gaussianIntegrals[n] - gaussianIntegrals[n+2]);
                }
               if(derivs2) {
                    // d2E / da2  has no contribution from F
                    // d2E / da dm
                    derivs2[1] += -1 / s * monomialCoefs[n] * gaussianIntegrals[n+1];
                    // d2E / da ds
                    derivs2[2] +=  1 / s * monomialCoefs[n] *
                        (gaussianIntegrals[n] - gaussianIntegrals[n+2]);
                    // d2E / dm2
                    derivs2[4] += a / pow_2(s) * monomialCoefs[n] *
                        (gaussianIntegrals[n] - gaussianIntegrals[n+2]);
                    // d2E / dm ds
                    derivs2[5] += a / pow_2(s) * monomialCoefs[n] *
                        (3 * gaussianIntegrals[n+1] - gaussianIntegrals[n+3]);
                    // d2E / ds2
                    derivs2[8] += a / pow_2(s) * monomialCoefs[n] *
                        (5 * gaussianIntegrals[n+2] - 2 * gaussianIntegrals[n] - gaussianIntegrals[n+4]);
                }
            }
        }
        // second derivatives are symmetric, so copy the upper half of this matrix into the lower half
        if(derivs2) {
            derivs2[3] = derivs2[1];
            derivs2[6] = derivs2[2];
            derivs2[7] = derivs2[5];
        }
    }

    /// this is the method used in the minimization, which only needs values and first derivatives
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        evalDeriv2(vars, values, derivs, NULL);
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
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
        double params[3];
        computeClassicMoments(fnc, params);

        // now that we have a reasonable initial values for the moments of the input function,
        // perform a Levenberg-Marquardt optimization to find the best-fit parameters of
        // the Gaussian approximation to this function.
        // This results in the first three GH coefficients being h_0=1, h_1=h_2=0.
        // For numerical stability and to make the procedure invariant w.r.t. changes in
        // the magnitude and spatial size of the input function, the fitted parameters
        // are renormalized: amplitude divided by a0 and mean&width divided by s0, defined below
        double a0 = params[0], s0 = sqrt(pow_2(params[1]) + pow_2(params[2]));
        params[0] /= a0;
        params[1] /= s0;
        params[2] /= s0;

        nonlinearMultiFit(GaussianFitterFnc(fnc, a0, s0),
            /*init*/ params,
            /*accuracy*/ EPSREL_GAUSSIAN_FIT,
            /*maxNumIter*/ 100,
            /*output(updated)*/ params);

        // renormalize back the fitted parameters
        Ampl   = params[0] * a0;
        Center = params[1] * s0;
        Width  = params[2] * s0;
    }
    moments = computeGaussHermiteMoments(fnc, order, Ampl, Center, Width);
}

// constructor specialized to piecewise-polynomial functions - same as above, just with a different
// implementations of GaussianFitter, computeClassicMoments and computeGaussHermiteMoments
GaussHermiteExpansion::GaussHermiteExpansion(const BaseInterpolator1d& fnc,
    unsigned int order, double ampl, double center, double width) :
    Ampl(ampl), Center(center), Width(width)
{
    if(order<2)
        throw std::invalid_argument("GaussHermiteExpansion: order must be >=2");
    if(!isFinite(ampl + center + width)) {
        // estimate the first 3 moments of the function, which are used as starting values in the fit
        double params[3];
        computeClassicMoments(fnc, params);

        // the next step (finding the best-fit Gaussian approximation) is carried out for
        // renormalized parameters: amplitude divided by a0 and mean&width divided by s0, defined as
        double a0 = params[0], s0 = sqrt(pow_2(params[1]) + pow_2(params[2]));
        params[0] /= a0;
        params[1] /= s0;
        params[2] /= s0;
        // this ensures that the procedure remains invariant w.r.t. the magnitude of these values,
        // despite using an absolute threshold on the gradient as a stopping criterion

        GaussianFitterInterp fitter(fnc, a0, s0);
        findMinNdimDeriv(fitter,
            /*init*/ params,
            /*step*/ 0.5,
            /*accuracy*/ EPSREL_GAUSSIAN_FIT, 
            /*maxNumIter*/ 100,
            /*output(updated)*/ params);

        // the above call should have found a very good approximation to the best-fit Gaussian,
        // but it can be further improved by one Newton correction step involving second derivs
        std::vector<double> derivs(3);  // derivs should be close to zero, but not exactly zero
        Matrix<double> derivs2(3, 3);   // second derivs are also computed analytically
        fitter.evalDeriv2(&params[0], NULL, &derivs[0], derivs2.data());
        std::vector<double> correction = SVDecomp(derivs2).solve(derivs);
        for(int i=0; i<=2; i++)
            params[i] -= correction[i];
         fitter.evalDeriv(&params[0], NULL, &derivs[0]);  // should now be zero to machine precision

        // renormalize back the fitted parameters
        Ampl   = params[0] * a0;
        Center = params[1] * s0;
        Width  = params[2] * s0;

        // the fitted width may turn out to be negative (this is mathematically allowed,
        // but should be corrected by flipping the signs of both amplitude and width, but not center).
        if(Width < 0) {
            Ampl   = -Ampl;
            Width  = -Width;
        }
    }
    moments = computeGaussHermiteMoments(fnc, order, Ampl, Center, Width);
}

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
            throw std::invalid_argument("computeGaussHermiteMatrix: invalid B-spline degree");
    }
}

} // namespace