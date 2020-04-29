/*
This is a new implementation of GalPot written by Eugene Vasiliev, 2015-2017.

The original GalPot code:
Copyright Walter Dehnen, 1996-2004
e-mail:   walter.dehnen@astro.le.ac.uk
address:  Department of Physics and Astronomy, University of Leicester
          University Road, Leicester LE1 7RH, United Kingdom

Version 0.0    15. July      1997
Version 0.1    24. March     1998
Version 0.2    22. September 1998
Version 0.3    07. June      2001
Version 0.4    22. April     2002
Version 0.5    05. December  2002
Version 0.6    05. February  2003
Version 0.7    23. September 2004
Version 0.8    24. June      2005
*/

#include "potential_disk.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <cassert>

namespace potential{

namespace{  // internal

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

/** more complex radial density profile - exponential/Sersic with possible inner hole and modulation */
class DiskDensityRadialRichExp: public math::IFunction {
public:
    DiskDensityRadialRichExp(const DiskParam& params):
        surfaceDensity     (params.surfaceDensity),
        invScaleRadius  (1./params.scaleRadius),
        innerCutoffRadius  (params.innerCutoffRadius),
        modulationAmplitude(params.modulationAmplitude),
        invSersicIndex  (1./params.sersicIndex)
    {};
private:
    const double surfaceDensity, invScaleRadius, innerCutoffRadius, modulationAmplitude, invSersicIndex;
    /**  evaluate  f(R) and optionally its two derivatives, if these arguments are not NULL  */
    virtual void evalDeriv(double R, double* f=NULL, double* fprime=NULL, double* fpprime=NULL) const {
        if((innerCutoffRadius && R==0.) || R==INFINITY) {
            if(f) *f=0;
            if(fprime)  *fprime=0;
            if(fpprime) *fpprime=0;
            return;
        }
        const double
            Rinv = 1 / R,
            Rrel = R * invScaleRadius,
            Rrn  = math::pow(Rrel, invSersicIndex),
            RrnR = R>0 ? Rrn * Rinv : invSersicIndex==1 ? 1. : invSersicIndex>1 ? 0. : INFINITY,
            Rcut = innerCutoffRadius ? innerCutoffRadius * Rinv : 0,
            cr   = modulationAmplitude ? modulationAmplitude * cos(Rrel) : 0,
            sr   = modulationAmplitude ? modulationAmplitude * sin(Rrel) : 0,
            val  = surfaceDensity * exp(-Rcut - Rrn + cr),
            fp   = Rcut * Rinv - invSersicIndex * RrnR - sr * invScaleRadius;
        if(fpprime)
            *fpprime = val ?
                val * (fp*fp - 2*Rcut*pow_2(Rinv) - cr * pow_2(invScaleRadius) +
                (invSersicIndex==1 ? 0 : RrnR * Rinv * invSersicIndex * (1-invSersicIndex)) ) :
                0;  // if val==0, the bracket could be NaN
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
            exp(-params.innerCutoffRadius/params.scaleRadius/Rrel - math::pow(Rrel, 1/params.sersicIndex)
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
        if(H)       *H       = 0.5 / invScaleHeight *  // use asymptotic expansion for small x
            (x>1e-5 ? h-1+x : x*x * (0.5 - 1./6*x));   // to avoid roundoff errors
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
        if(H)       *H       = 1./invScaleHeight *
            (x>1e-3 ? 0.5*x + log(0.5*sh1) : x*x * (1./8 - 1./192*x*x));
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
    if(params.sersicIndex<=0)
        throw std::invalid_argument("Disk Sersic index must be positive");
    if(params.innerCutoffRadius==0 && params.modulationAmplitude==0 && params.sersicIndex==1)
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
            return M_PI * pow_2(scaleRadius) * surfaceDensity * math::gamma(2*sersicIndex+1);
        else if(sersicIndex==1) {
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
    bool deriv1 = deriv!=NULL || deriv2!=NULL;  // compute derivatives of f and H only if necessary
    verticalFnc->evalDeriv(pos.z, &H, deriv1? &Hp : NULL, deriv2? &h   : NULL);
    radialFnc  ->evalDeriv(r,     &f, deriv1? &fp : NULL, deriv2? &fpp : NULL);
    f  *= 4*M_PI;
    fp *= 4*M_PI;
    fpp*= 4*M_PI;
    double rinv = r>0 ? 1./r : 1.;  // if r==0, avoid indeterminacy in 0/0
    double Rr   = pos.R * rinv;
    double zr   = pos.z * rinv;
    if(potential) {
        *potential = f==0 ? 0 : f * H;
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

} // namespace
