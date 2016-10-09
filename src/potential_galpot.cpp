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
Modifications by Eugene Vasiliev, 2015-2016
(so extensive that almost nothing of the original code remains)

*/
#include "potential_galpot.h"
#include "potential_composite.h"
#include "potential_multipole.h"
#include "math_core.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential{

/// order of the Multipole expansion
static const int GALPOT_LMAX = 32;

/// order of the azimuthal Fourier expansion in case of non-axisymmetric components
static const int GALPOT_MMAX = 6;

/// number of radial points in Multipole 
static const int GALPOT_NRAD = 60;

/// factors determining the radial extent of logarithmic grid in Multipole;
/// they are multiplied by the min/max scale radii of model components
static const double GALPOT_RMIN = 1e-4,  GALPOT_RMAX = 1e4;

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
    double r    = sqrt(pow_2(pos.R) + pow_2(pos.z));
    double h, H, Hp, f, fp, fpp;
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

/** integrand for computing the total mass:  4pi r^2 rho(r), x=r/scaleRadius */
class SpheroidDensityIntegrand: public math::IFunctionNoDeriv {
public:
    SpheroidDensityIntegrand(const SphrParam& _params): params(_params) {};
private:
    const SphrParam params;
    virtual double value(double x) const {
        if(x==0 || x==1) return 0;
        double rrel = x/(1-x);
        return 
            pow_2(x/pow_2(1-x)) * pow(rrel, -params.gamma) *
            pow(1 + pow(rrel, params.alpha), (params.gamma-params.beta)/params.alpha) *
            exp(-pow_2(rrel * params.scaleRadius / params.outerCutoffRadius));
    }
};

double SphrParam::mass() const
{
    if(beta<=3 && outerCutoffRadius==0)
        return INFINITY;
    return 4*M_PI * densityNorm * pow_3(scaleRadius) * axisRatioY * axisRatioZ *
        ( outerCutoffRadius==0 ?   // have an analytic expression
        math::gamma((beta-3)/alpha) * math::gamma((3-gamma)/alpha) /
        math::gamma((beta-gamma)/alpha) / alpha :
        math::integrate(SpheroidDensityIntegrand(*this), 0, 1, 1e-6) );
}

SpheroidDensity::SpheroidDensity (const SphrParam &_params) :
    BaseDensity(), params(_params)
{
    if(params.scaleRadius<=0)
        throw std::invalid_argument("Spheroid scale radius must be positive");
    if(params.axisRatioY<=0 || params.axisRatioZ<=0)
        throw std::invalid_argument("Spheroid axis ratio must be positive");
    if(params.outerCutoffRadius<0)
        throw std::invalid_argument("Spheroid outer cutoff radius cannot be <0");
    if(params.alpha<=0)
        throw std::invalid_argument("Spheroid parameter alpha must be positive");
    if(params.beta<=2 && params.outerCutoffRadius==0)
        throw std::invalid_argument("Spheroid outer slope beta must be greater than 2, "
            "or a positive cutoff radius must be provided");
    if(params.gamma>=3)
        throw std::invalid_argument("Spheroid inner slope gamma must be less than 3");
}

double SpheroidDensity::densityCyl(const coord::PosCyl &pos) const
{
    double  R2 = params.axisRatioY == 1 ? pow_2(pos.R) :
        pow_2(pos.R) * (1 + pow_2(sin(pos.phi)) * (1/pow_2(params.axisRatioY) - 1));
    double  r  = sqrt(R2 + pow_2(pos.z/params.axisRatioZ));
    double  r0 = r/params.scaleRadius;
    double rho = params.densityNorm;
    if(params.gamma==1.) rho /= r0;       else 
    if(params.gamma==2.) rho /= r0*r0;    else
    if(params.gamma==0.5)rho /= sqrt(r0); else
    if(params.gamma!=0.) rho /= pow(r0, params.gamma);
    if(params.alpha==2.)  r0 *= r0; else
    if(params.alpha!=1.)  r0  = pow(r0, params.alpha);
    r0 += 1;
    const double bga = (params.beta-params.gamma) / params.alpha;
    if(bga==1.) rho /= r0;       else
    if(bga==2.) rho /= r0*r0;    else
    if(bga==3.) rho /= r0*r0*r0; else
    rho *= pow(r0, -bga);
    if(params.outerCutoffRadius)
        rho *= exp(-pow_2(r/params.outerCutoffRadius));
    return rho;
}

//----- GalaxyPotential refactored into a Composite potential -----//
std::vector<PtrPotential> createGalaxyPotentialComponents(
    const std::vector<DiskParam>& diskParams, 
    const std::vector<SphrParam>& sphrParams)
{
    double lengthMin=INFINITY, lengthMax=0;  // keep track of length scales of all components
    bool isSpherical=diskParams.size()==0;   // whether there are any non-spherical components
    bool isAxisymmetric=true;                // same for non-axisymmetric spheroidal components

    // assemble the set of density components for the multipole
    // (all spheroids and residual part of disks),
    // and the complementary set of potential components
    // (the flattened part of disks, eventually to be joined by the multipole itself)
    std::vector<PtrDensity> componentsDens;
    std::vector<PtrPotential> componentsPot;
    for(unsigned int i=0; i<diskParams.size(); i++) {
        if(diskParams[i].surfaceDensity == 0)
            continue;
        // the two parts of disk profile: DiskAnsatz goes to the list of potentials...
        componentsPot.push_back(PtrPotential(new DiskAnsatz(diskParams[i])));
        // ...and gets subtracted from the entire DiskDensity for the list of density components
        componentsDens.push_back(PtrDensity(new DiskDensity(diskParams[i])));
        DiskParam negDisk(diskParams[i]);
        negDisk.surfaceDensity *= -1;  // subtract the density of DiskAnsatz
        componentsDens.push_back(PtrDensity(new DiskAnsatz(negDisk)));
        // keep track of characteristic lengths
        lengthMin = fmin(lengthMin, diskParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, diskParams[i].scaleRadius);
        if(diskParams[i].innerCutoffRadius>0)
            lengthMin = fmin(lengthMin, diskParams[i].innerCutoffRadius);
    }
    for(unsigned int i=0; i<sphrParams.size(); i++) {
        if(sphrParams[i].densityNorm == 0)
            continue;
        componentsDens.push_back(PtrDensity(new SpheroidDensity(sphrParams[i])));
        lengthMin = fmin(lengthMin, sphrParams[i].scaleRadius);
        lengthMax = fmax(lengthMax, sphrParams[i].scaleRadius);
        if(sphrParams[i].outerCutoffRadius) 
            lengthMax = fmax(lengthMax, sphrParams[i].outerCutoffRadius);
        isSpherical    &= sphrParams[i].axisRatioZ == 1;
        isAxisymmetric &= sphrParams[i].axisRatioY == 1;
    }
    if(componentsDens.size()==0)
        throw std::invalid_argument("Empty parameters in GalPot");
    // create an composite density object to be passed to Multipole;
    const CompositeDensity dens(componentsDens);

    // create multipole potential from this combined density
    double rmin = GALPOT_RMIN * lengthMin;
    double rmax = GALPOT_RMAX * lengthMax;
    int lmax = isSpherical    ? 0 : GALPOT_LMAX;
    int mmax = isAxisymmetric ? 0 : GALPOT_MMAX;
    componentsPot.push_back(Multipole::create(dens, lmax, mmax, GALPOT_NRAD, rmin, rmax));

    // the array of components to be passed to the constructor of CompositeCyl potential:
    // the multipole and the non-residual parts of disk potential
    return componentsPot;
}

PtrPotential createGalaxyPotential(
    const std::vector<DiskParam>& DiskParams,
    const std::vector<SphrParam>& SphrParams)
{
    return PtrPotential(new CompositeCyl(createGalaxyPotentialComponents(DiskParams, SphrParams)));
}

} // namespace
