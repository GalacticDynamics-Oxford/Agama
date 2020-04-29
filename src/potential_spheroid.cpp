#include "potential_spheroid.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_spline.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential{

namespace {  // internal

/** integrand for computing the total mass:  4pi r^2 rho(r) */
class SpheroidDensityIntegrand: public math::IFunctionNoDeriv {
public:
    SpheroidDensityIntegrand(const SpheroidParam& _params): params(_params) {};
private:
    const SpheroidParam params;
    virtual double value(double x) const {
        double rrel = exp( 1/(1-x) - 1/x );
        double result = 
            pow_3(rrel) * (1/pow_2(1-x) + 1/pow_2(x)) *
            math::pow(rrel, -params.gamma) *
            math::pow(1 + math::pow(rrel, params.alpha), (params.gamma-params.beta)/params.alpha) *
            exp(-math::pow(rrel * params.scaleRadius / params.outerCutoffRadius, params.cutoffStrength));
        return isFinite(result) ? result : 0;
    }
};

/** one-dimensional density profile described by a double-power-law model with an exponential cutoff */
class SpheroidDensityFnc: public math::IFunctionNoDeriv{
    const SpheroidParam params;
public:
    SpheroidDensityFnc(const SpheroidParam _params): params(_params) {}
    virtual double value(const double r) const
    {
        double  r0 = r/params.scaleRadius;
        double rho = params.densityNorm * math::pow(r0, -params.gamma) *
        math::pow(1 + math::pow(r0, params.alpha), (params.gamma-params.beta) / params.alpha);
        if(params.outerCutoffRadius<INFINITY)
            rho *= exp(-math::pow(r/params.outerCutoffRadius, params.cutoffStrength));
        return rho;
    }
};

/** helper function for computing the deprojected density in the Nuker model */
class NukerIntegrand: public math::IFunctionNoDeriv {
    const double alpha, beta, gamma, r;
public:
    NukerIntegrand(double _alpha, double _beta, double _gamma, double _r) :
        alpha(_alpha), beta(_beta), gamma(_gamma), r(_r) {}
    virtual double value(double t) const
    {   // 0<=t<1 is the scaled radial variable
        double u = sqrt(1 - 2 * t * (1-t)),  R = r * u / (1-t),  Ra = pow(R, alpha);
        return pow(R, -gamma-1) * pow(1 + Ra, (gamma-beta)/alpha-1) * (gamma + beta * Ra) / u / (1-t);
    }
};

/** helper function for finding the coefficient b in the Sersic model */
class SersicBRootFinder: public math::IFunctionNoDeriv {
    const double twon, g;
public:
    SersicBRootFinder(double n): twon(2*n), g(math::gamma(twon)) {};
    virtual double value(double b) const { return g - 2*math::gammainc(twon, b); }
};

/// cutoff value for the integral, defined so that exp(-MAX_EXP) is very small and may be neglected
static const double MAX_EXP = 100.;

/** helper function for computing the deprojected density profile in the Sersic model:
    \f$  \int_0^\infty  dx  (\cosh x)^{1/n-1}  \exp[ - k (\cosh x)^{1/n} ]  \f$,  k = b [r/Reff]^{1/n}.
    Since the argument of exponent rapidly rises with x, we integrate only up to xmax instead of infinity.
*/
class SersicIntegrand: public math::IFunctionNoDeriv {
    const double invn, k;
public:
    SersicIntegrand(double n, double kk) : invn(1/n), k(kk) {}
    virtual double value(double x) const
    {
        double y = cosh(x), z = pow(y, invn);
        if(k * z > MAX_EXP) return 0;
        return z / y * exp(-k * z);
    }
};

}  // internal ns

double SpheroidParam::mass() const
{
    if(beta<=3 && outerCutoffRadius==INFINITY)
        return INFINITY;
    return 4*M_PI * densityNorm * pow_3(scaleRadius) * axisRatioY * axisRatioZ *
        ( outerCutoffRadius==0 ?   // have an analytic expression
        math::gamma((beta-3)/alpha) * math::gamma((3-gamma)/alpha) /
        math::gamma((beta-gamma)/alpha) / alpha :
        math::integrateAdaptive(SpheroidDensityIntegrand(*this), 0, 1, 1e-6) );
}

double NukerParam::mass() const {
    return 2*M_PI * surfaceDensity * pow_2(scaleRadius) * axisRatioY * axisRatioZ *
        math::gamma((beta-2)/alpha) * math::gamma((2-gamma)/alpha) /
        math::gamma((beta-gamma)/alpha) / alpha * pow(2, (beta-gamma)/alpha);
}

double SersicParam::b() const {
    return math::findRoot(SersicBRootFinder(sersicIndex),
        fmax(2*sersicIndex-1, 1e-4), 2*sersicIndex, 1e-10);
}

double SersicParam::mass() const {
    return 2*M_PI * surfaceDensity * pow_2(scaleRadius) * axisRatioY * axisRatioZ *
        sersicIndex * math::gamma(2*sersicIndex) * pow(b(), -2*sersicIndex);
}

math::PtrFunction createSpheroidDensity(const SpheroidParam& params)
{
    if(!(params.scaleRadius > 0))
        throw std::invalid_argument("Spheroid scale radius must be positive");
    if(!(params.axisRatioY > 0 && params.axisRatioZ > 0))
        throw std::invalid_argument("Spheroid axis ratio must be positive");
    if(!(params.outerCutoffRadius > 0))
        throw std::invalid_argument("Spheroid outer cutoff radius must be positive (or infinite)");
    if(!(params.alpha > 0))
        throw std::invalid_argument("Spheroid parameter alpha must be positive");
    if(params.beta <= 2 && params.outerCutoffRadius == INFINITY)
        throw std::invalid_argument("Spheroid outer slope beta must be greater than 2, "
            "or a cutoff radius must be provided");
    if(!(params.gamma < 3))
        throw std::invalid_argument("Spheroid inner slope gamma must be less than 3");
    if(!(params.gamma <= params.beta))
        throw std::invalid_argument("Spheroid inner slope gamma should not exceed the outer slope beta");
    if(!(params.cutoffStrength > 0) && isFinite(params.outerCutoffRadius))
        throw std::invalid_argument("Spheroid cutoff strength must be positive");
    return math::PtrFunction(new SpheroidDensityFnc(params));
}

math::PtrFunction createNukerDensity(const NukerParam& params)
{
    if(!(params.scaleRadius > 0))
        throw std::invalid_argument("Nuker scale radius must be positive");
    if(!(params.axisRatioY > 0 && params.axisRatioZ > 0))
        throw std::invalid_argument("Nuker axis ratio must be positive");
    if(!(params.alpha > 0))
        throw std::invalid_argument("Nuker parameter alpha must be positive");
    if(!(params.beta  > 2))
        throw std::invalid_argument("Nuker outer slope beta must be greater than 2");
    if(!(params.gamma >= 0 && params.gamma < 2))
        throw std::invalid_argument("Nuker inner slope gamma must be between 0 and 2");

    // compute the deprojected density on a grid of points in log-scaled radius  y = ln(r/Rscale)
    const int NPOINTS = 41;
    double YMIN = 0.5 / sqrt(1 + pow_2(params.alpha)), YMAX = 15 * sqrt(1 + pow_2(1/params.alpha));
    std::vector<double> gridy = math::createSymmetricGrid(NPOINTS, YMIN, YMAX), gridr, gridrho;
    gridr.reserve(NPOINTS); gridrho.reserve(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        double r    = exp(gridy[i]);
        double coef = math::integrateAdaptive(
            NukerIntegrand(params.alpha, params.beta, params.gamma, r),
            0, 1, 1e-6) * pow(2, (params.beta - params.gamma) / params.alpha) / M_PI;
        if(coef>0 && coef<INFINITY) {  // avoid numerical issues if the integral failed to converge
            gridr.  push_back(params.scaleRadius * r);
            gridrho.push_back(params.surfaceDensity * coef / params.scaleRadius);
        }
    }
    double leftSlope =    // asymptotic power-law slope of the density profile at small radii
        params.gamma>0 ? -1-params.gamma :
        params.alpha<1 ? -1+params.alpha :
        params.alpha>1 ? 0 : NAN;
    double rightSlope = -1-params.beta;   // asymptotic power-law slope at large radii
    return math::PtrFunction(new math::LogLogSpline(gridr, gridrho,
        leftSlope * gridrho.front() / gridr.front(),
        rightSlope * gridrho.back() / gridr.back()));
}

math::PtrFunction createSersicDensity(const SersicParam& params)
{
    double b = params.b(), n = params.sersicIndex;
    if(!(n>=0.5) || !isFinite(b))
        throw std::invalid_argument("Sersic index n should be larger than 0.5");
    if(!(params.scaleRadius > 0))
        throw std::invalid_argument("Sersic scale radius must be positive");
    if(!(params.axisRatioY > 0 && params.axisRatioZ > 0))
        throw std::invalid_argument("Sersic axis ratio must be positive");

    // compute the deprojected density on a grid of points in log-scaled radius  y = 1/n ln(r/Reff)
    const int NPOINTS = 32;
    const double YMIN = -10., YMAX = log(MAX_EXP) - log(b);
    std::vector<double> gridr(NPOINTS), gridrho(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        double y    = YMIN + (YMAX-YMIN) * i / NPOINTS;
        double xmax = n * (YMAX - y);
        if(n>1)xmax = fmin(xmax, MAX_EXP * n / (n-1) );
        double coef = math::integrateAdaptive(SersicIntegrand(n, b * exp(y)), 0, xmax, 1e-6);
        gridr  [i]  = params.scaleRadius * exp(n * y);
        gridrho[i]  = params.surfaceDensity * b / (M_PI * n * params.scaleRadius) * exp((1-n) * y) * coef;
    }
    return math::PtrFunction(new math::LogLogSpline(gridr, gridrho));
}

double SpheroidDensity::densityCar(const coord::PosCar &pos) const
{
    return rho->value(sqrt(pow_2(pos.x) + pow_2(pos.y) / p2 + pow_2(pos.z) / q2));
}

double SpheroidDensity::densityCyl(const coord::PosCyl &pos) const
{
    double R2 = p2 == 1 ? pow_2(pos.R) :
        pow_2(pos.R) * (1 + pow_2(sin(pos.phi)) * (1 / p2 - 1));
    return rho->value(sqrt(R2 + pow_2(pos.z) / q2));
}

double SpheroidDensity::densitySph(const coord::PosSph &pos) const
{
    if(p2==1 && q2==1)  // spherically-symmetric profile
        return rho->value(pos.r);
    else
        return densityCar(toPosCar(pos));
}

} // namespace
