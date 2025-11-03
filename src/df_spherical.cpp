#include "df_spherical.h"
#include "potential_utils.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_fit.h"
#include "utils.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <fstream>

namespace df{

namespace{

/// tolerance on the 2nd derivative of the auxiliary function of log(r) used in grid generation
/// ~eps^(1/3), because this is the typical accuracy of the finite-difference estimate
static const double ACCURACY_INTERP = ROOT3_DBL_EPSILON;

/// step size for numerical differentiation using 4th order scheme with 5 points separated by this step
static const double STEP_DERIV = 1e-3;   // ~eps^(1/5), to get a relative accuracy of eps^(4/5)

/// fixed order of Gauss-Legendre quadrature on each segment of the grid
static const int GLORDER = 8;

/// minimum relative difference between two adjacent values of potential (to reduce roundoff errors)
static const double MIN_REL_DIFFERENCE = DBL_EPSILON / ROOT3_DBL_EPSILON;  // ~eps^(2/3) ~ 4e-11


/** Fit a polynomial approximation of the given degree N to the first M>N points of the input function
    y(x), which is assumed to be computed with some numerical error (whose magnitude is unknown a priori).
    We attempt to find a suitable approximation of y(x) in terms of its Taylor series at x=0,
    in the range of x where this approximation is "better" than the original function.
    For a fixed number of M leftmost points, we construct a best-fit polynomial,
    and compute the rms residuals in this fit (RMS).
    The truncation error of the Taylor series is expected to be ~ (d^{N+1}y / dx^{N+1}) * (x_M)^{N+1},
    but for small enough x_M the actual error is likely dominated by numerical errors on the input values,
    and is nearly constant (or at least grows slower than x_M^{N+1}).
    So we try to find the number of points M beyond which the truncation error takes over.
    As the figure of merit, we use score=RMS / x_M^N;  this quantity may first decrease as long as
    the numerical errors dominate, and then starts to increase when the truncation error becomes dominant.
    \tparam     N  is the order of polynomial approximation;
    \param[in]  x  is the vector of x-coordinates (supposed to increase);
    \param[in]  y  is the vector of function values at these coordinates;
    \param[out] bestFitCoefs  will containt the N+1 coefficients of the best-fit polynomial approximation;
    \return     the number of points for which this optimal approximation should produce a better
    accuracy than the actual values of y(x).
*/
template<int N>
size_t fitPolynomial(const std::vector<double>& x, const std::vector<double>& y,
    /*output*/ std::vector<double>& bestFitCoefs)
{
    size_t size =  x.size();
    assert(size == y.size() && size > N+1);
    std::vector<double> mat((N+1) * size);  // flattened matrix of the linear least-square fit
    std::vector<double> rhs(size);          // rhs vector
    // assemble the Vandermonde matrix - each row is 1, x_i, x_i^2, ... x_i^N
    for(size_t i=0; i<size; i++)
        for(int p=0; p<=N; p++)
            mat[i * (N+1) + p] = math::pow(x[i], p);
    // storage for the coefficients of polynomial fit for each trial value of M, flattened into one array
    std::vector<double> coefs((N+1) * size);
    std::vector<double> result(N+1);         // temporary results for a single fit
    double bestScore = INFINITY;
    size_t bestM = N+2;
    for(size_t M = bestM; M<size; M++) {
        rhs.assign(y.begin(), y.begin()+M);  // copy the first M values of y vector
        // perform the fit for the first M points
        double rms;
        linearMultiFit(math::MatrixView<double>(M, N+1, &mat.front()), rhs, /*weights*/NULL,
            /*output: fit coefs*/ result, /*output: rms error*/ &rms);
        // copy the result into the summary table
        for(int p=0; p<=N; p++)
            coefs[M * (N+1) + p] = result[p];
        // estimate of the approximation error arising from truncating the Taylor expansion at order N:
        // d^{N+1}y / dx^{N+1} * x_M^{N+1}.
        double score = rms / math::pow(x[M-1], N);
        if(score < bestScore) {
            bestScore = score;
            bestM = M;
        } else if(M >= bestM+2)
            break;  // stop if the score didn't improve for two consecutive points
    }
    // output the stored fit coefficients for the value of M that produced the best score
    bestFitCoefs.assign(coefs.begin() + bestM * (N+1), coefs.begin() + (bestM+1) * (N+1));
    return bestM;
}


/** The augmented density as a function of radius, with accurately computed two derivatives.
    This function returns the value of a user-defined function which computes the density at
    the given radius, multiplied by extra terms r^{2 beta0} * [1 + (r/r_a)^2]^{1-beta0}.
    More importantly, it provides two derivatives of this composite function w.r.t. potential,
    either by using finite differences on the original density profile, or a Taylor series
    approximation to it at Phi --> Phi(r=0), to improve the overall accuracy;
    the extra terms in the augmented density are differentiated analytically.
    The eval() method computes the potential, its two derivatives by radius, the density, and
    its two derivatives by potential.
    It also provides the IFunction interface, with the method evalDeriv() returning the value
    and two derivatives by log(r) of an auxiliary function which represents the variation of
    both density and potential with radius:
    F = log[ Phi(r)^2 / Phi(0) - Phi(r) ] + log[ rho(r) ].
    This function is used to construct a suitable radial grid via createInterpolationGrid().
*/
class AugmentedDensity: public math::IFunction {
    const math::IFunction& density;   ///< original density
    const math::IFunction& potential; ///< the potential
    const double beta0, r_a;          ///< parameters of the augmented density part
    const double Phi0;                ///< potential at origin
    double rmin;                      ///< minimum radius below which we use Taylor series (if possible)
    std::vector<double> coefs;        ///< coefficients of Taylor series of the original density at Phi0
public:
    AugmentedDensity(
        const math::IFunction& _density, const math::IFunction& _potential, double _beta0, double _r_a)
    :
        density(_density), potential(_potential), beta0(_beta0), r_a(_r_a), Phi0(potential(0)), rmin(0)
    {
        // if the original function provided derivatives, use them without any further hassle;
        // also skip any further refinements if the potential is singular at origin
        if(density.numDerivs() >= 2 || Phi0 == -INFINITY)
            return;

        // otherwise examine the variation of rho(r) with radius (only in a sensible range though)
        std::vector<double> gridr, gridPhi, gridrho;
        // start from an arbitrary initial radius, but make sure it's large enough
        double r = 1.;
        while(potential(r) < 0.5*Phi0)  r *= 2;
        double Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2;
        do{
            eval(r, /*output*/ Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2);
            if(isFinite(rho) && rho>0 && Phi < 0.5*Phi0 && Phi > Phi0 * (1-MIN_REL_DIFFERENCE)) {
                gridr.  push_back(r);
                gridPhi.push_back(Phi-Phi0);
                gridrho.push_back(rho);
            }
            r *= 1/M_SQRT2;
        } while(Phi > Phi0 * (1-MIN_REL_DIFFERENCE));
        if(gridr.size() <= 5)
            return;
        std::reverse(gridr.begin(),   gridr.end());
        std::reverse(gridPhi.begin(), gridPhi.end());
        std::reverse(gridrho.begin(), gridrho.end());

        // assuming that rho(Phi) has a finite limiting value at Phi0 and is reasonably smooth,
        // construct Taylor series up to 2nd or 3rd order, approximating it up to certain radius
        size_t nfit = fitPolynomial<2>(gridPhi, gridrho, coefs);
        coefs.push_back(0);  // the x^3 term is disabled for the moment

        // check if the fit is actually reasonable, in that the relative error is small enough
        double sumRelErr2 = 0;
        for(size_t i=0; i<nfit; i++) {
            sumRelErr2 += pow_2(1 - (coefs[0] + gridPhi[i] * (coefs[1] + gridPhi[i] *
                (coefs[2] + gridPhi[i] * coefs[3]))) / gridrho[i]);
        }
        if(!(sqrt(sumRelErr2 / nfit) < 1e-3))   // the fit is not acceptable
            return;

        rmin = gridr[nfit-1];   // will use analytic approximation to rho(Phi) below this radius
        FILTERMSG(utils::VL_DEBUG, "SphericalDF",
            "Augmented density rho=" + utils::toString(coefs[0], 12) +
            (coefs[1]>=0 ? "+" : "") + utils::toString(coefs[1], 12) + "*x" +
            (coefs[2]>=0 ? "+" : "") + utils::toString(coefs[2], 12) + "*x^2" +
            (coefs[3]>=0 ? "+" : "") + utils::toString(coefs[3], 12) + "*x^3" +
            ", where 0 <= x=Phi+" + utils::toString(-Phi0, 16) +
            " <= " + utils::toString(gridPhi[nfit]) + ", or r <= " + utils::toString(rmin) +
            " (" + utils::toString(nfit) + " points)");
    }

    /// compute the potential Phi at the given radius, its two derivatives by radius,
    /// the augmented density rho^{augmented}(r),  and its two derivatives by potential
    void eval(double r, /*output*/ double& Phi, double& dPhidr, double& d2Phidr2,
        double& rho, double& drhodPhi, double& d2rhodPhi2) const
    {
        potential.evalDeriv(r, &Phi, &dPhidr, &d2Phidr2);
        if(r<rmin) {  // use Taylor series expansion around Phi=Phi0
            double dif = Phi-Phi0;
            rho        = coefs[0] + dif * (coefs[1] + dif * (coefs[2] + dif * coefs[3]));
            drhodPhi   = coefs[1] + dif * (2 * coefs[2] + dif * 3 * coefs[3]);
            d2rhodPhi2 = coefs[2] * 2 + dif * 6 * coefs[3];
            return;
        }
        // compute the original density and its derivatives
        double drhodr, d2rhodr2;
        if(density.numDerivs() >= 2) {  // use the derivatives provided by the original function
            density.evalDeriv(r, &rho, &drhodr, &d2rhodr2);
        } else {    // use finite differences to obtain the derivatives of the original function
            double dr= STEP_DERIV * r,
            rhom2    = fmax(0, density(r-dr*2)),
            rhom1    = fmax(0, density(r-dr)),
            rhop1    = fmax(0, density(r+dr)),
            rhop2    = fmax(0, density(r+dr*2));
            rho      = fmax(0, density(r));
            drhodr   = (2./3 * (rhop1 - rhom1) - 1./12 * (rhop2 - rhom2) ) / dr;
            d2rhodr2 = (4./3 * (rhop1 + rhom1) - 1./12 * (rhop2 + rhom2) - 2.5 * rho) / pow_2(dr);
        }
        // compute the augmented density multiplier and its derivatives
        if(beta0!=0 || r_a!=INFINITY) {
            double
            rra2     = pow_2(r / r_a),
            aug      = math::pow(1 + rra2, 1-beta0) * math::pow(r, 2*beta0),
            daugdr   = aug * 2 * (beta0 + rra2) / (r * (1 + rra2)),
            d2augdr2 = aug * 2 * (beta0 * (2*beta0-1) + (beta0+1) * rra2 + rra2*rra2) /
                pow_2(r * (1 + rra2));
            d2rhodr2 = d2rhodr2 * aug + 2 * drhodr * daugdr + rho * d2augdr2;
            drhodr   = drhodr * aug + rho * daugdr;
            rho     *= aug;
        }
        // convert the radial derivatives of (augmented) density to derivatives w.r.t. Phi
        drhodPhi     = drhodr / dPhidr;
        d2rhodPhi2   = (d2rhodr2 - d2Phidr2 / dPhidr * drhodr) / pow_2(dPhidr);
    }

    /// the auxiliary function whose second derivative represents the log-curvature of rho(Phi),
    /// passed to createInterpolationGrid to set up a grid in log(r)
    virtual void evalDeriv(const double logr, double* val, double* der, double* der2) const
    {
        double r=exp(logr), Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2;
        eval(r,  /*output*/ Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2);
        double logrho = rho>0 ? log(rho) : 0,
            dlogrho   = rho>0 ? drhodPhi / rho : 0,
            d2logrho  = rho>0 ? d2rhodPhi2 / rho - pow_2(dlogrho) : 0,
            Y = 1 / (1-Phi/Phi0),  // ranges from 0 infinity
            Z = dPhidr * r * ( (2-Y) / Phi + dlogrho );
        if(val)
            *val = log(-Phi / Y) + logrho;
        if(der)
            *der = Z;
        if(der2) {
            if(Phi < Phi0 * (1-MIN_REL_DIFFERENCE))
                // in case of a finite potential at r=0,
                // we avoid approaching too close to 0 to avoid roundoff errors in Phi
                *der2 = 0;
            else
                *der2 = Z + r * r * ( d2Phidr2 * ( (2-Y) / Phi + dlogrho) +
                    (d2logrho - (2-2*Y+Y*Y) / pow_2(Phi)) * pow_2(dPhidr) );
        }
    }

    virtual unsigned int numDerivs() const { return 2; }
};


/// compute the DF of a spherical isotropic (Eddington) or anisotropic (Cuddeford-Osipkov-Merritt) model
void createSphericalDF(
    const math::IFunction& density, const math::IFunction& potential, double beta0, double r_a,
    /*input/output*/ std::vector<double>& gridPhi, /*output*/ std::vector<double>& gridf)
{
    // 1. check input parameters
    if(!(beta0>=-0.5 && beta0<1 && r_a>0))
       throw std::invalid_argument("SphericalDF: beta0 must be between -0.5 and 1, "
            "and r_a should be positive (possibly +inf)");
    const double Phi0 = potential(0);
    if(!(Phi0 < 0))    // Phi0 may be -inf, but shouldn't be NaN or, even worse, >=0
        throw std::runtime_error("SphericalDF: potential must be negative at r=0");
    int derivOrder = beta0>=0.5 ? 1 : 2;   // n, the order of derivative d^(n) rho / d Phi^(n)
    double fracpow = 2.5-beta0-derivOrder; // fracpow=1 for half-integer beta, otherwise between 0 and 1
    assert(fracpow>0 && fracpow<=1);
    double prefact = fracpow==1 ?          // C, beta-dependent prefactor in the Abel inversion formula
        0.5/M_PI/M_PI :    // special case of half-integer beta = +-1/2
        pow(2., beta0) / (2*M_SQRT2*M_SQRTPI*M_PI * math::gamma(1-beta0) * math::gamma(1-fracpow));

    // the magic function computing rho(Phi) and its derivatives, using some tricks to improve accuracy
    AugmentedDensity augdensity(density, potential, beta0, r_a);

    std::vector<double> gridr;
    double prevPhi = Phi0;
    if(gridPhi.empty()) {
        // 2a. prepare grid in log-scaled radius, ranging from -inf to +inf
        gridr = math::createInterpolationGrid(augdensity, ACCURACY_INTERP);  // so far it's log(r) not r

        // 2b. check the grid and convert it into unscaled Phi and r
        for(size_t i=0; i<gridr.size(); ) {
            gridr[i]   = exp(gridr[i]);   // unscale from log(r) to r
            double Phi = potential.value(gridr[i]);
            // throw away grid nodes if they are too closely spaced (if the difference between adjacent
            // potential values is dominated by roundoff / cancellation errors)
            if(Phi > prevPhi * (1-MIN_REL_DIFFERENCE)) {
                gridPhi.push_back(Phi);
                prevPhi = Phi;
                i++;
            } else {
                gridr.erase(gridr.begin()+i);
            }
        }
    } else {
        // 2c. input grid in potential was provided, convert it to the grid in radius
        gridr.resize(gridPhi.size());
        potential::FunctionToPotentialWrapper pw(potential);
        for(size_t i=0; i<gridPhi.size(); i++)
            gridr[i] = potential::R_max(pw, gridPhi[i]);
    }

    size_t gridsize = gridPhi.size();
    if(gridsize < 3)
        throw std::runtime_error("SphericalDF: failed to determine radial grid");

    gridf.resize(gridsize);
    double Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2;
    if(fracpow == 1) {
        // 3a. for half-integer beta0 we do not need to compute the integrals,
        // in this case f_j = C * d^(n)\rho / d\Phi^(n) (r_j)
        for(size_t j=0; j<gridsize; j++) {
            double r = gridr[j];
            augdensity.eval(r, /*output*/ Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2);
            gridf[j] = prefact * (derivOrder == 1 ? -drhodPhi : d2rhodPhi2);
        }
    } else {
        // 3b. compute the asymptotic slope of augmented density as a function of Phi,
        // needed to calculate the integrals over the last grid segment, Phi(rmax) to 0
        augdensity.eval(gridr.back(), /*output*/ Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2);
        // log-slope of the augmented density profile, rho^ ~ |Phi|^s as Phi --> 0
        double slope = Phi / rho * drhodPhi;
        if(rho == 0)
            slope = 0;  // prevent 0/0 indeterminacy
        FILTERMSG(utils::VL_DEBUG, "SphericalDF",
            "Augmented density is ~ |Phi|^"+utils::toString(slope)+" at large r");

        // 3c. the general case when we do actually need the Abel inversion.
        // First we compute analytically the values of all integrals f_j over the last segment
        // [Phi_{gridsize-1} to 0]
        double num = prefact * rho * (derivOrder==1 ? slope : slope*(1-slope) / Phi);
        // the integrand is d^(n)
        slope -= derivOrder;
        for(unsigned int j=gridsize-3; j<gridsize; j++) {
            double factor = j==gridsize-1 ?
                math::gamma(1-fracpow) * exp(math::lngamma(1+slope) - math::lngamma(2+slope-fracpow)) :
                // all other grid nodes except the last one will have contributions from other segments,
                // which are anyway much larger than this last segment, so it doesn't hurt even if it
                // cannot be computed accurately (e.g. for extreme values of slope);
                // for this reason and to save effort, we limit the loop to the last three nodes only
                math::hypergeom2F1(fracpow, 1+slope, 2+slope, gridPhi.back() / gridPhi[j]) / (1+slope);
            if(isFinite(factor))
                gridf[j] = num * factor / pow(-gridPhi[j], fracpow);
            // do nothing (keep f=0) in case of arithmetic overflow (outer slope too steep)
        }

        // use pre-computed integration tables
        const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];

        // 3d. the integration for
        // f_j = \int_{\Phi_j}^0 d\Phi  F(\Phi) /  (\Phi - \Phi_j)^{fracpow},  where
        // F(\Phi) = C * d^(n) \rho( \Phi ) / d\Phi^(n),
        // is carried over each grid segment Phi_i .. Phi_{i+1}, i=0..gridsize-1,
        // and for each i we add the contribution of this segment to all integrals I_j, j<=i.
        prevPhi = Phi0;
        for(unsigned int i=0; i<gridsize-1; i++) {
            // Since we need to integrate the (augmented) density, which is a function of r,
            // it is more convenient to use log(r) instead of Phi as the integration variable;
            // hence \int d\Phi * [...]  = \int d(log r) r dPhi/dr * [...].
            // Moreover, the integrand has a singularity at Phi=Phi_j, which is treated as follows:
            // for j=i (the first segment for each f_j, which contains the singularity at the left end),
            // we integrate analytically the singular part
            // \int_{\Phi_j}^{\Phi_{j+1}} d\Phi  F(\Phi_j)  /  (\Phi - \Phi_j)^{fracpow}
            // (with constant numerator equal to the value of augmented density at the left end),
            // and numerically integrate the residual part  (F(\Phi) - F(\Phi_j)) / [...].
            // The denominator is still "nasty", so we further change the integration variable to  y,
            // ranging from 0 to 1, defined so that the integration nodes are clustered at the left end:
            // log(r) = log(r_i) + (log(r_{i+1}) - log(r_i)) * y^2m,  with a corresponding weight factor.
            // For all other grid segments i>j, the integrand has no singularities.
            double r1    = gridr[i];
            double logr1 = log(r1), dlogr = log(gridr[i+1]) - logr1;
            double num1  = NAN;

            // The numerical integration uses a fixed-order Gauss-Legendre quadrature on each segment
            for(int k=-1; k<GLORDER; k++) {
                // node of Gauss-Legendre quadrature within the current segment (logr[i] .. logr[i+1]);
                // the integration variable y ranges from 0 to 1, and logr(y) is defined below
                double y = k==-1 ? 0 : glnodes[k];
                double r = exp(logr1 + dlogr * y*y);
                augdensity.eval(r, /*output*/ Phi, dPhidr, d2Phidr2, rho, drhodPhi, d2rhodPhi2);
                // keep an eye on the potential - it must be monotonic with radius
                if(!(dPhidr >= 0 && Phi >= prevPhi))
                    throw std::runtime_error("SphericalDF: potential is not monotonic "
                        "with radius at r="+utils::toString(r));
                prevPhi = Phi;
                double num = derivOrder == 1 ? -drhodPhi : d2rhodPhi2;
                if(k==-1) {
                    // the singular part for i=j
                    num1 = num;
                    gridf[i] += prefact * num1 *
                        math::pow(gridPhi[i+1] - gridPhi[i], 1-fracpow) / (1-fracpow);
                } else {
                    // contribution of this point to each integral on the current segment,
                    // taking into account the transformation of variables and the GL quadrature weight
                    double factor = prefact * dPhidr * (/* dr/dy */ r * 2*y * dlogr) * glweights[k];
                    // now add a contribution to the integrals f_j for all j <= i
                    // (with a singular part subtracted for j=i)
                    for(unsigned int j=0; j<i; j++)  // all except the last
                        gridf[j] += factor * num * math::pow(Phi - gridPhi[j], -fracpow);
                    // the last with singular part subtracted
                    gridf[i] += factor * (num - num1) * math::pow(Phi - gridPhi[i], -fracpow);
                }
            }
        }
    }

    // 4a. output debugging information, if needed
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("SphericalDF.log");
        strm << "# r            \trho(r)         \tQ=Phi(r)       \tf(Q)           \n";
        for(unsigned int i=0; i<gridsize; i++) {
            strm <<
            utils::pp(gridr[i], 15) << '\t' <<
            utils::pp(density(gridr[i]), 15) << '\t' <<
            utils::pp(gridPhi[i], 15) << '\t' <<
            utils::pp(gridf[i], 15) << '\n';
        }
    }

    // 4b. tidy up
    for(unsigned int i=0; i<gridf.size(); ) {
        // make sure that the values are non-negative, and don't leave spurious orphaned points
        // at both ends, which may appear to have f>0 due to various inaccuracies,
        // even if adjacent points are negative
        if(gridf[i] < 0 || (i==0 && gridf[i+1] < 0) || (i==gridf.size()-1 && gridf[gridf.size()-2] <= 0))
            gridf[i] = 0;
        // eliminate invalid values
        if(!isFinite(gridf[i])) {
            gridf.erase(gridf.begin()+i);
            gridr.erase(gridr.begin()+i);
            gridPhi.erase(gridPhi.begin()+i);
        } else {
            i++;
        }
    }

    // 4c. return the results in two arrays: gridPhi, gridf
}

// a thin wrapper on top of the above routine which takes together the two returned arrays,
// gridPhi and gridf, and creates a log-log-scaled spline in scaled Phi
math::LogLogSpline createSphericalDF(
    const math::IFunction& density, const math::IFunction& potential, double beta0, double r_a)
{
    std::vector<double> gridPhi, gridf;
    createSphericalDF(density, potential, beta0, r_a, /*output*/ gridPhi, gridf);
    // construct a log-log-scaled spline interpolator for f(P), where  P = 1/Phi0 - 1/Q.
    // Both f and P will be converted to log-scaled values by LogLogSpline,
    // so we need to provide the un-log-scaled coordinate, reusing the gridPhi array for P.
    double invPhi0 = 1/potential(0);
    for(unsigned int i=0; i<gridPhi.size(); i++)
        gridPhi[i] = invPhi0 - 1/gridPhi[i];
    return math::LogLogSpline(gridPhi, gridf);
}

}  // internal namespace

// another thin wrapper on top of the main routine createSphericalDF for the case of
// isotropic models (classical Eddington inversion formula), which converts the energy grid
// into the grid in phase volume h, and ensures that the asymptotic log-slopes are reasonable
void createSphericalIsotropicDF(
    const math::IFunction& density,
    const math::IFunction& potential,
    /*in/out*/ std::vector<double>& gridh,
    /*output*/ std::vector<double>& gridf)
{
    // construct a phase-volume interpolator to convert between E and h
    potential::PhaseVolume phasevol(potential);
    std::vector<double> gridPhi(gridh.size());

    // if the input grid in h was provided, convert it into a grid in Phi
    for(size_t i=0; i<gridh.size(); i++)
        gridPhi[i] = phasevol.E(gridh[i]);

    // construct the DF
    createSphericalDF(density, potential, /*beta*/ 0, /*r_a*/ INFINITY, /*output*/ gridPhi, gridf);

    // convert the grid in Phi back to the grid in h
    gridh.resize(gridPhi.size());
    for(size_t i=0; i<gridPhi.size(); i++)
        gridh[i] = phasevol(gridPhi[i]);

    // check that the computed values describe a DF that doesn't grow too fast as h-->0
    // and decays fast enough as h-->infinity;  this still does not guarantee that it's reasonable,
    // but at least allows to construct a valid interpolator
    do {
        math::LogLogSpline spl(gridh, gridf);
        double valIn, valOut, derIn, derOut;
        spl.evalDeriv(gridh.front(), &valIn,  &derIn);
        spl.evalDeriv(gridh.back(),  &valOut, &derOut);
        double slopeIn  = valIn ==0 ?  0  :  gridh.front() / valIn  * derIn;
        double slopeOut = valOut==0 ?  0  :  gridh.back()  / valOut * derOut;
        if(slopeIn > -1 && (slopeOut < -1 || valOut==0)) {
            FILTERMSG(utils::VL_DEBUG, "createSphericalIsotropicDF",
                "f(h) " + (valIn>0 ? "~ h^" + utils::toString(slopeIn) : "is zero") + " at small h "
                "and " + (valOut>0 ? "~ h^" + utils::toString(slopeOut): "is zero") + " at large h");

            // results are returned in the two arrays, gridh and gridf
            return;
        }
        // otherwise remove the innermost and/or outermost point and repeat
        if(slopeIn < -1) {
            gridf.erase(gridf.begin());
            gridh.erase(gridh.begin());
        }
        if(slopeOut > -1) {
            gridf.erase(gridf.end()-1);
            gridh.erase(gridh.end()-1);
        }
    } while(gridf.size() > 2);
    throw std::runtime_error("createSphericalIsotropicDF: could not construct a valid non-negative DF");
}


//---- Construction of f(h) from an N-body snapshot ----//

math::LogLogSpline fitSphericalIsotropicDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize)
{
    const unsigned int nbody = hvalues.size();
    if(masses.size() != nbody)
        throw std::invalid_argument("fitSphericalIsotropicDF: array sizes are not equal");

    // 1. collect the log-scaled values of phase volume
    std::vector<double> logh(nbody);
    for(unsigned int i=0; i<nbody; i++) {
        logh[i] = log(hvalues[i]);
        if(!isFinite(logh[i]+masses[i]) || masses[i]<0)
            throw std::invalid_argument("fitSphericalIsotropicDF: incorrect input data");
    }

    // 2. create a reasonable grid in log(h)
    int Nmin = static_cast<int>(fmax(1, log(nbody+1)/log(2)));
    std::nth_element(logh.begin(), logh.begin() + Nmin, logh.end());
    double loghmin = logh[Nmin];
    std::nth_element(logh.begin(), logh.end() - Nmin, logh.end());
    double loghmax = logh[nbody-Nmin];
    std::vector<double> gridh = math::createUniformGrid(gridSize, loghmin, loghmax);
    FILTERMSG(utils::VL_DEBUG, "fitSphericalIsotropicDF",
        "Grid in h=["+utils::toString(exp(gridh.front()))+":"+utils::toString(exp(gridh.back()))+"]");

    // 3a. perform spline log-density fit, and
    // 3b. initialize a cubic spline for log(f) as a function of log(h)
    math::CubicSpline fitfnc(gridh,
        math::splineLogDensity<3>(gridh, logh, masses,
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));

    // 4. store the values of cubic spline at grid nodes, together with two endpoint derivatives --
    // this data is sufficient to reconstruct it exactly later in the LogLogSpline constructor
    double derLeft, derRight;
    fitfnc.evalDeriv(gridh.front(), NULL, &derLeft);
    fitfnc.evalDeriv(gridh.back(),  NULL, &derRight);
    assert(derLeft > 0 && derRight < 0);  // a condition for a valid fit (total mass should be finite)
    // endpoint derivatives of fitfnc are  d [log (h f(h) ) ] / d [log h] = 1 + d [log f] / d [log h],
    // so we subtract unity to obtain the logarithmic derivs of f(h)
    derLeft  -= 1;
    derRight -= 1;

    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h = exp(gridh[i]);
        // the fit provides log( dM/d(log h) ) = log( h dM/dh ) = log( h f(h) )
        gridf[i] = exp(fitfnc(gridh[i])) / h;
        gridh[i] = h;
    }

    // debugging output
    FILTERMSG(utils::VL_DEBUG, "fitSphericalIsotropicDF",
        "f(h) ~ h^" + utils::toString(derLeft)  + " at small h"
        " and ~ h^" + utils::toString(derRight) + " at large h");
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("FitSphericalIsotropicDF.log");
        strm << "#h      \tf(h)    \tinner slope: " << derLeft << ", outer slope: " << derRight << '\n';
        for(unsigned int i=0; i<gridh.size(); i++)
            strm << utils::pp(gridh[i], 12) << '\t' << utils::pp(gridf[i],12) << '\n';
    }

    // the derivatives provided to LogLogSpline are df/dh = (f/h) d[log f] / d[log h]
    derLeft  = derLeft  * gridf.front() / gridh.front();
    derRight = derRight * gridf.back () / gridh.back ();

    // 5. construct an interpolating spline that matches exactly our fitfnc (it's also a cubic spline
    // in the same scaled variables), including the correct slopes for extrapolation outside the grid
    return math::LogLogSpline(gridh, gridf, derLeft, derRight);
}


//------ QuasiSphericalCOM DF class for Cuddeford-Osipkov-Merritt models -------//

QuasiSphericalCOM::QuasiSphericalCOM(const math::IFunction& density, const math::IFunction& potential,
    double _beta0, double _r_a, double _rotFrac, double _Jphi0)
:
    QuasiSpherical(potential), invPhi0(1./potential(0)),
    beta0(_beta0), r_a(_r_a), rotFrac(_rotFrac), Jphi0(_Jphi0),
    df(createSphericalDF(density, potential, beta0, r_a))
{
    if(!(fabs(rotFrac)<=1))
        throw std::invalid_argument("QuasiSphericalCOM: rotFrac must be between -1 and 1");
}

void QuasiSphericalCOM::evalDeriv(const ClassicalIntegrals& ints,
    double *value, DerivByClassicalIntegrals *deriv) const
{
    double Q = ints.E + 0.5 * pow_2(ints.L/r_a), P = invPhi0 - 1./Q, dfdP = 0;
    if(Q>=0)
        *value = 0;  // DF is zero for Q>=0, which may happen even for a valid combination of E and L
    else
        df.evalDeriv(P, value, deriv ? &dfdP : NULL);
    // note that we don't (and cannot) check that L <= Lcirc(E)
    double Lpow = math::pow(ints.L, -2*beta0);
    *value *= Lpow;
     // add the odd part if necessary
    double rot = rotFrac!=0 && Jphi0!=INFINITY ? rotFrac * tanh(ints.Lz / Jphi0) : 0;
    *value *= (1+rot);
    if(deriv) {
        double dfdQ = Q<0 ? dfdP / pow_2(Q) : 0;
        deriv->dbyE = (1+rot) * Lpow * dfdQ;
        deriv->dbyL = (1+rot) * Lpow * dfdQ * ints.L / pow_2(r_a) + (beta0 ? -2*beta0 * (*value) / ints.L : 0);
        deriv->dbyLz = 0;
        if(Jphi0!=0 && rotFrac!=0)
            deriv->dbyLz += *value * rotFrac * (1 - pow_2(rot / rotFrac)) / (1+rot) / Jphi0;
    }
    // check for invalid range - this shouldn't happen in normal circumstances, so be picky here
    if(ints.L<0 || fabs(ints.Lz)>ints.L || P<0)
        *value = NAN;
}

}  // namespace df
