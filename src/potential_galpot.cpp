/*
Copyright Walter Dehnen, 1996-2004 
e-mail:   walter.dehnen@astro.le.ac.uk 
address:  Department of Physics and Astronomy, University of Leicester 
          University Road, Leicester LE1 7RH, United Kingdom 

------------------------------------------------------------------------
Version 0.0    15. July      1997 
Version 0.1    24. March     1998 
Version 0.2    22. September 1998 
Version 0.3    07. June      2001 
Version 0.4    22. April     2002 
Version 0.5    05. December  2002 
Version 0.6    05. February  2003 
Version 0.7    23. September 2004
Version 0.8    24. June      2005

----------------------------------------------------------------------
Modifications by Eugene Vasiliev, 2015-2017
(so extensive that almost nothing of the original code remains)

*/
#include "potential_galpot.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_spline.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential{

namespace{  // internal

//----- disk density and potential -----//

/** simple exponential radial density profile without inner hole or wiggles */
class DiskDensityRadialExp: public math::IFunction {
public:
    DiskDensityRadialExp(const DiskParam& params): 
        surfaceDensity(params.surfaceDensity),
        invScaleRadius(1./params.scaleRadius)
    {};
private:
    const double surfaceDensity, invScaleRadius;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        double val = surfaceDensity * exp(-R * invScaleRadius);
        if(f)
            *f = val;
        if(fprime)
            *fprime = -val * invScaleRadius;
        if(fpprime)
            *fpprime = val * pow_2(invScaleRadius);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** more convoluted radial density profile - exponential with possible inner hole and modulation */
class DiskDensityRadialRichExp: public math::IFunction {
public:
    DiskDensityRadialRichExp(const DiskParam& params):
        surfaceDensity     (params.surfaceDensity),
        invScaleRadius  (1./params.scaleRadius),
        innerCutoffRadius  (params.innerCutoffRadius),
        modulationAmplitude(params.modulationAmplitude)
    {};
private:
    const double surfaceDensity, invScaleRadius, innerCutoffRadius, modulationAmplitude;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        if(innerCutoffRadius && R==0.) {
            if(f) *f=0;
            if(fprime)  *fprime=0;
            if(fpprime) *fpprime=0;
            return;
        }
        const double
            Rinv = 1 / R,
            Rrel = R * invScaleRadius,
            Rcut = innerCutoffRadius ? innerCutoffRadius * Rinv : 0,
            cr   = modulationAmplitude ? modulationAmplitude * cos(Rrel) : 0,
            sr   = modulationAmplitude ? modulationAmplitude * sin(Rrel) : 0,
            val  = surfaceDensity * exp(-Rcut - Rrel + cr),
            fp   = Rcut * Rinv - (1+sr) * invScaleRadius;
        if(fpprime)
            *fpprime = val ? (fp*fp - 2*Rcut*pow_2(Rinv)
                - cr * pow_2(invScaleRadius)) * val : 0;  // if val==0, the bracket could be NaN
        if(fprime)
            *fprime  = val ? fp*val : 0;
        if(f) 
            *f = val;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** integrand for computing the total mass:  2pi R Sigma(R); x=R/scaleRadius */
class DiskDensityRadialRichExpIntegrand: public math::IFunctionNoDeriv {
public:
    DiskDensityRadialRichExpIntegrand(const DiskParam& _params): params(_params) {};
private:
    const DiskParam params;
    virtual double value(double x) const {
        if(x==0 || x==1) return 0;
        double Rrel = x/(1-x);
        return x / pow_3(1-x) *
            exp(-params.innerCutoffRadius/params.scaleRadius/Rrel - Rrel
                +params.modulationAmplitude*cos(Rrel));
    }
};

/** exponential vertical disk density profile */
class DiskDensityVerticalExp: public math::IFunction {
public:
    DiskDensityVerticalExp(double scaleHeight): invScaleHeight(1./scaleHeight) {};
private:
    const double invScaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z * invScaleHeight);
        double      h        = exp(-x);
        if(H)       *H       = 0.5 / invScaleHeight * (h-1+x);
        if(Hprime)  *Hprime  = 0.5 * math::sign(z) * (1.-h);
        if(Hpprime) *Hpprime = 0.5 * h * invScaleHeight;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** isothermal (sech^2) vertical disk density profile */
class DiskDensityVerticalIsothermal: public math::IFunction {
public:
    DiskDensityVerticalIsothermal(double scaleHeight): invScaleHeight(1./scaleHeight) {};
private:
    const double invScaleHeight;
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        double      x        = fabs(z * invScaleHeight);
        double      h        = exp(-x);
        double      sh1      = 1 + h,  invsh1 = 1./sh1;
        if(H)       *H       = 1./invScaleHeight * (0.5*x + log(0.5*sh1));
        if(Hprime)  *Hprime  = 0.5 * math::sign(z) * (1.-h) * invsh1;
        if(Hpprime) *Hpprime = h * invScaleHeight * pow_2(invsh1);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

/** vertically thin disk profile */
class DiskDensityVerticalThin: public math::IFunction {
public:
    DiskDensityVerticalThin() {};
private:
    /**  evaluate  H(z) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double z, double* H=NULL, double* Hprime=NULL, double* Hpprime=NULL) const {
        if(H)       *H       = 0.5 * fabs(z);
        if(Hprime)  *Hprime  = 0.5 * math::sign(z);
        if(Hpprime) *Hpprime = 0;
    }
    virtual unsigned int numDerivs() const { return 2; }
};

}  // internal ns

/** helper routine to create an instance of radial density function */
math::PtrFunction createRadialDiskFnc(const DiskParam& params) {
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Disk scale radius cannot be <=0");
    if(params.innerCutoffRadius<0)
        throw std::invalid_argument("Disk inner cutoff radius cannot be <0");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0)
        return math::PtrFunction(new DiskDensityRadialExp(params));
    else
        return math::PtrFunction(new DiskDensityRadialRichExp(params));
}

/** helper routine to create an instance of vertical density function */
math::PtrFunction createVerticalDiskFnc(const DiskParam& params) {
    if(params.scaleHeight>0)
        return math::PtrFunction(new DiskDensityVerticalExp(params.scaleHeight));
    if(params.scaleHeight<0)
        return math::PtrFunction(new DiskDensityVerticalIsothermal(-params.scaleHeight));
    else
        return math::PtrFunction(new DiskDensityVerticalThin());
}

double DiskParam::mass() const
{
    if(modulationAmplitude==0) {  // have an analytic expression
        if(innerCutoffRadius==0)
            return 2*M_PI * pow_2(scaleRadius) * surfaceDensity;
        else {
            double p = sqrt(innerCutoffRadius / scaleRadius);
            return 4*M_PI * pow_2(scaleRadius) * surfaceDensity *
                p * (p * math::besselK(0, 2*p) + math::besselK(1, 2*p));
        }
    }
    return 2*M_PI * pow_2(scaleRadius) * surfaceDensity *
        math::integrate(DiskDensityRadialRichExpIntegrand(*this), 0, 1, 1e-6);
}

double DiskDensity::densityCyl(const coord::PosCyl &pos) const
{
    double h;
    verticalFnc->evalDeriv(pos.z, NULL, NULL, &h);
    return radialFnc->value(pos.R) * h;
}

double DiskAnsatz::densityCyl(const coord::PosCyl &pos) const
{
    double h, H, Hp, f, fp, fpp, r=sqrt(pow_2(pos.R) + pow_2(pos.z));
    verticalFnc->evalDeriv(pos.z, &H, &Hp, &h);
    radialFnc  ->evalDeriv(r, &f, &fp, &fpp);
    return f*h + (pos.z!=0 ? 2*fp*(H+pos.z*Hp)/r : 0) + fpp*H;
}

void DiskAnsatz::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double r = sqrt(pow_2(pos.R) + pow_2(pos.z));
    double h=0, H=0, Hp=0, f=0, fp=0, fpp=0;
    bool deriv1 = deriv!=NULL || deriv2!=NULL;  // compute 1st derivative of f and H only if necessary
    verticalFnc->evalDeriv(pos.z, &H, deriv1? &Hp : NULL, deriv2? &h : NULL);
    radialFnc  ->evalDeriv(r,     &f, deriv1? &fp : NULL, deriv2? &fpp : NULL);
    f  *= 4*M_PI;
    fp *= 4*M_PI;
    fpp*= 4*M_PI;
    double rinv = r>0 ? 1./r : 1.;  // if r==0, avoid indeterminacy in 0/0
    double Rr   = pos.R * rinv;
    double zr   = pos.z * rinv;
    if(potential) {
        *potential = f * H;
    }
    if(deriv) {
        deriv->dR = H * Rr * fp;
        deriv->dz = H * zr * fp + Hp * f;
        deriv->dphi=0;
    }
    if(deriv2) {
        deriv2->dR2 = H * (fpp * pow_2(Rr) + fp * rinv * pow_2(zr));
        deriv2->dz2 = H * (fpp * pow_2(zr) + fp * rinv * pow_2(Rr)) + fp * Hp * zr * 2 + f * h;
        deriv2->dRdz= H * Rr * zr * (fpp - fp * rinv) + fp * Hp * Rr;
        deriv2->dRdphi=deriv2->dzdphi=deriv2->dphi2=0;
    }
}

//----- spheroid density -----//

namespace {  // internal

/** integrand for computing the total mass:  4pi r^2 rho(r) */
class SpheroidDensityIntegrand: public math::IFunctionNoDeriv {
public:
    SpheroidDensityIntegrand(const SphrParam& _params): params(_params) {};
private:
    const SphrParam params;
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
    const SphrParam params;
public:
    SpheroidDensityFnc(const SphrParam _params): params(_params) {}
    virtual double value(const double r) const
    {
        double  r0 = r/params.scaleRadius;
        double rho = params.densityNorm * math::pow(r0, -params.gamma) *
        math::pow(1 + math::pow(r0, params.alpha), (params.gamma-params.beta) / params.alpha);
        if(params.outerCutoffRadius)
            rho *= exp(-math::pow(r/params.outerCutoffRadius, params.cutoffStrength));
        return rho;
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
    SersicIntegrand(double n, double kk): invn(1/n), k(kk) {};
    virtual double value(double x) const
    {
        double y = cosh(x), z = pow(y, invn);
        if(k * z > MAX_EXP) return 0;
        return z / y * exp(-k * z);
    }
};

}  // internal ns

double SphrParam::mass() const
{
    if(beta<=3 && outerCutoffRadius==0)
        return INFINITY;
    return 4*M_PI * densityNorm * pow_3(scaleRadius) * axisRatioY * axisRatioZ *
        ( outerCutoffRadius==0 ?   // have an analytic expression
        math::gamma((beta-3)/alpha) * math::gamma((3-gamma)/alpha) /
        math::gamma((beta-gamma)/alpha) / alpha :
        math::integrateAdaptive(SpheroidDensityIntegrand(*this), 0, 1, 1e-6) );
}

double SersicParam::b() const {
    return math::findRoot(SersicBRootFinder(sersicIndex),
        fmax(2*sersicIndex-1, 1e-4), 2*sersicIndex, 1e-10);
}

double SersicParam::mass() const {
    return 2*M_PI * surfaceDensity * pow_2(scaleRadius) * axisRatioY * axisRatioZ *
        sersicIndex * math::gamma(2*sersicIndex) * pow(b(), -2*sersicIndex);
}

math::PtrFunction createSpheroidDensity(const SphrParam& params)
{
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Spheroid scale radius must be positive");
    if(params.axisRatioY<=0 || params.axisRatioZ<=0)
        throw std::invalid_argument("Spheroid axis ratio must be positive");
    if(params.outerCutoffRadius<0)
        throw std::invalid_argument("Spheroid outer cutoff radius cannot be negative");
    if(params.alpha<=0)
        throw std::invalid_argument("Spheroid parameter alpha must be positive");
    if(params.beta<=2 && params.outerCutoffRadius==0)
        throw std::invalid_argument("Spheroid outer slope beta must be greater than 2, "
            "or a positive cutoff radius must be provided");
    if(params.gamma>=3)
        throw std::invalid_argument("Spheroid inner slope gamma must be less than 3");
    if(params.gamma>params.beta)
        throw std::invalid_argument("Spheroid inner slope gamma should not exceed the outer slope beta");
    if(params.outerCutoffRadius>0 && params.cutoffStrength<=0)
        throw std::invalid_argument("Spheroid cutoff strength must be positive");
    return math::PtrFunction(new SpheroidDensityFnc(params));
}
    
math::PtrFunction createSersicDensity(const SersicParam& params)
{
    double b = params.b(), n = params.sersicIndex;
    if(!(n>0) || !isFinite(b))
        throw std::invalid_argument("Sersic index n should be larger than 0.5");
    if(params.scaleRadius <= 0)
        throw std::invalid_argument("Sersic scale radius must be positive");

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
