#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace potential{

namespace{  // internal routines

/// relative accuracy of root-finders for radius
static const double ACCURACY = 1e-10;

/// a number that is considered nearly infinity in log-scaled root-finders
static const double HUGE_NUMBER = 1e100;

// -------- routines for conversion between energy, radius and angular momentum --------- //

/** helper class to find the root of  Phi(R) = E */
class RmaxRootFinder: public math::IFunction {
    const BasePotential& poten;
    const double E;
public:
    RmaxRootFinder(const BasePotential& _poten, double _E) : poten(_poten), E(_E) {};
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* =0) const {
        double Phi;
        coord::GradCyl grad;
        double R = exp(logR);
        poten.eval(coord::PosCyl(R,0,0), &Phi, &grad);
        if(val) {
            if(R>=HUGE_NUMBER && !isFinite(Phi))
                Phi=0;
            *val = Phi-E;
        }
        if(deriv)
            *deriv = grad.dR * R;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  L_z^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given energy E).
*/
class RcircRootFinder: public math::IFunction {
    const BasePotential& poten;
    const double E;
public:
    RcircRootFinder(const BasePotential& _poten, double _E) : poten(_poten), E(_E) {};
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* deriv2=0) const {
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        double R = exp(logR);
        poten.eval(coord::PosCyl(R,0,0), &Phi, &grad, &hess);
        if(val) {
            if(R>=HUGE_NUMBER && !isFinite(Phi))
                *val = -1-fabs(E);  // safely negative value
            else
                *val = 2*(E-Phi) - (R>1./HUGE_NUMBER && R<HUGE_NUMBER ? R*grad.dR : 0);
        }
        if(deriv)
            *deriv = (-3*grad.dR - R*hess.dR2) * R;
        if(deriv2)
            *deriv2 = NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  L_z^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given angular momentum L_z).
    For the reason of accuracy, we multiply the equation by  1/(R+1), 
    which ensures that the value stays finite as R -> infinity or R -> 0.
*/
class RfromLzRootFinder: public math::IFunction {
    const BasePotential& poten;
    const double Lz2;
public:
    RfromLzRootFinder(const BasePotential& _poten, double _Lz) : poten(_poten), Lz2(_Lz*_Lz) {};
    virtual void evalDeriv(const double R, double* val=0, double* deriv=0, double* deriv2=0) const {
        coord::GradCyl grad;
        coord::HessCyl hess;
        // TODO: this is unsatisfactory, need to convert to log-scaling
        static const double UNREASONABLY_LARGE_VALUE = 1e10;
        if(R < UNREASONABLY_LARGE_VALUE) {
            poten.eval(coord::PosCyl(R,0,0), NULL, &grad, &hess);
            if(val)
                *val = ( Lz2 - (R>0 ? pow_3(R)*grad.dR : 0) ) / (R+1);
            if(deriv)
                *deriv = -(Lz2 + pow_2(R)*( (3+2*R)*grad.dR + R*(R+1)*hess.dR2) ) / pow_2(R+1);
        } else {   // at large R, Phi(R) ~ -M/R, we may use this asymptotic approximation even at infinity
            poten.eval(coord::PosCyl(UNREASONABLY_LARGE_VALUE,0,0), NULL, &grad);
            if(val)
                *val = Lz2/(R+1) - pow_2(UNREASONABLY_LARGE_VALUE) * grad.dR / (1+1/R);
            if(deriv)
                *deriv = NAN;
        } 
        if(deriv2)
            *deriv2 = NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** Helper function for finding the roots of (effective) potential in R direction */
class RPeriApoRootFinder: public math::IFunction {
    const BasePotential& potential;
    const double E, halfL2;
public:
    RPeriApoRootFinder(const BasePotential& p, double _E, double L) :
        potential(p), E(_E), halfL2(L*L/2) {};
    virtual unsigned int numDerivs() const { return 1; }
    virtual void evalDeriv(const double R, double* val=0, double* der=0, double* =0) const {
        double Phi=0;
        coord::GradCyl grad;
        potential.eval(coord::PosCyl(R,0,0), &Phi, der? &grad : NULL);
        if(val)
            *val = (E-Phi)*R*R - halfL2;
        if(der)
            *der = 2*R*(E-Phi) - R*R*grad.dR;
    }
};

/** Helper function for finding the roots of (effective) potential in R direction,
    for a power-law asymptotic form potential at small radii */
class RPeriApoRootFinderPowerLaw: public math::IFunction {
    const double s, v2;  // potential slope and squared relative ang.mom.(normalized to Lcirc)
public:
    RPeriApoRootFinderPowerLaw(double slope, double Lrel2) : s(slope), v2(Lrel2)
    { assert(s>=-1 && Lrel2>=0 && Lrel2<=1); }
    virtual unsigned int numDerivs() const { return 1; }
    virtual void evalDeriv(const double x, double* val=0, double* der=0, double* =0) const {
        if(val)
            *val = s!=0 ?  (2+s)*x*x - 2*pow(x, 2+s) - v2*s  :  (x==0 ? 0 : x*x * (1 - 2*log(x)) - v2);
        if(der)
            *der = s!=0 ?  (4+2*s) * (x - pow(x, 1+s))  :  -4*x*log(x);
    }
};

/// root polishing routine to improve the accuracy of peri/apocenter radii determination
static inline double refineRoot(const math::IFunction& pot, double R, double E, double L)
{
    double val, der, der2;
    pot.evalDeriv(R, &val, &der, &der2);
    // F = E - Phi(r) - L^2/(2r^2), refine the root of F=0 using Halley's method with two derivatives
    double F  = E - val - 0.5*pow_2(L/R);
    double Fp = pow_2(L/R)/R - der;
    double Fpp= -3*pow_2(L/(R*R)) - der2;
    double dR = -F / (Fp - 0.5 * F * Fpp / Fp);
    return fabs(dR) < R ? R+dR : R;  // precaution to avoid unpredictably large corrections
}

/// scaling transformations for energy or potential: the input energy ranges from Phi0 to 0,
/// the output scaled variable - from -inf to +inf.
static inline double scaledE(const double E, const double Phi0) {
    return log(1/Phi0 - 1/E);
}

/// inverse scaling transformation for energy or potential
static inline double unscaledE(const double scaledE, const double Phi0) {
    return 1 / (1/Phi0 - exp(scaledE));
}

/// derivative of scaling transformation: dE/d{scaledE}
static inline double scaledEder(const double E, const double Phi0) {
    return E * (E/Phi0 - 1);
}

}  // internal namespace

double v_circ(const BasePotential& potential, double radius)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular velocity is possible");
    if(radius==0)
        return 0;  // this is not quite true for a singular potential at origin..
    coord::GradCyl deriv;
    potential.eval(coord::PosCyl(radius, 0, 0), NULL, &deriv);
    return sqrt(radius*deriv.dR);
}

double R_circ(const BasePotential& potential, double energy) {
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return exp(math::findRoot(RcircRootFinder(potential, energy), -INFINITY, INFINITY, ACCURACY));
}

double L_circ(const BasePotential& potential, double energy) {
    double R = R_circ(potential, energy);
    return R * v_circ(potential, R);
}

double R_from_Lz(const BasePotential& potential, double Lz) {
    if(Lz==0)
        return 0;
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    return math::findRoot(RfromLzRootFinder(potential, Lz), 0, INFINITY, ACCURACY);
}

double R_max(const BasePotential& potential, double energy) {
    return exp(math::findRoot(RmaxRootFinder(potential, energy), -INFINITY, INFINITY, ACCURACY));
}

void epicycleFreqs(const BasePotential& potential, const double R,
    double& kappa, double& nu, double& Omega)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Potential is not axisymmetric, "
            "no meaningful definition of circular orbit is possible");
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(R, 0, 0), NULL, &grad, &hess);
    double gradR_over_R = (R==0 && grad.dR==0) ? hess.dR2 : grad.dR/R;
    //!!! no attempt to check if the expressions under sqrt are non-negative - 
    // they could well be for a physically plausible potential of a flat disk with an inner hole
    kappa = sqrt(hess.dR2 + 3*gradR_over_R);
    nu    = sqrt(hess.dz2);
    Omega = sqrt(gradR_over_R);
}

double innerSlope(const BasePotential& potential, double* Phi0, double* coef)
{
    // the choice of r must balance two opposite requirements:
    // it should be close enough to origin so that we really probe the inner slope,
    // but not too close so that Phi(r)-Phi(0) has enough significant digits;
    // in the case of a constant-density core the latter quantity is proportional to r^2,
    // so with r = double_epsilon^(1/3) ~ 1e-5 the values of Phi(r) and Phi(0) 
    // may coincide in the first 10 digits, leaving 5 significant digits when subtracted.
    double r = 1e-5;
    double val;
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(r,0,0), &val, &grad, &hess);
    double  s = 1 + r * hess.dR2 / grad.dR;
    if(coef)
        *coef = s==0 ?  grad.dR * r  :  grad.dR / s * pow(r, 1-s);
    if(Phi0)
        *Phi0 = s==0 ?  val - r * grad.dR * log(r)  :  val - r * grad.dR / s;
    return s;
}

double outerSlope(const BasePotential& potential, double* M, double* coef)
{
    // TODO here and in the previous routine:  make the choice of r scale-invariant;
    // add checks for roundoff errors and possibly iterate with adjusted value of r
    double r = 1e+5;
    double val;
    coord::GradCyl grad;
    coord::HessCyl hess;
    potential.eval(coord::PosCyl(r,0,0), &val, &grad, &hess);
    double  s = (2*grad.dR + r*hess.dR2) / (grad.dR + val/r);
    if(coef)
        *coef = s==-1 ?  (val + r*grad.dR) * r  :  (val + r*grad.dR) * pow(r, -s) / (s+1);
    if(M)
        *M    = s==-1 ?  (log(r) * (val + r*grad.dR) - val) * r  :  (r*grad.dR - s*val) * r / (s+1);
    return s;
}

void findPlanarOrbitExtent(const BasePotential& potential, double E, double L, double& R1, double& R2)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("findPlanarOrbitExtent only works for axisymmetric potentials");
    double Phi0, coef, slope;
    slope = innerSlope(potential, &Phi0, &coef);
    // accurate treatment close to the origin assuming a power-law asymptotic behavior of potential
    bool asympt   = (slope>0 && E>=Phi0 && E-Phi0 < fabs(Phi0)*1e-8);
    double Rcirc  = !asympt ?  R_circ(potential, E) :
        slope==0 ?  exp((E-Phi0) / coef - 0.5)  :  pow((E-Phi0) / (coef * (1+0.5*slope)), 1/slope);
    double Lcirc2 = !asympt ?  2 * (E - potential.value(coord::PosCyl(Rcirc,0,0))) * pow_2(Rcirc) :
        slope==0 ?  coef * pow_2(Rcirc)  :  (E-Phi0) / (1/slope+0.5) * pow_2(Rcirc);
    if(!isFinite(Lcirc2))
        throw std::invalid_argument("Error in findPlanarOrbitExtent: cannot determine Rcirc(E)");
    double Lrel2  = L*L / Lcirc2;
    if(Lrel2>=1) {
        if(Lrel2<1+1e-8) {  // assuming a roundoff error and not an intentional foul
            R1 = R2 = Rcirc;
            return;
        } else
            throw std::invalid_argument("Error in findPlanarOrbitExtent: E="+
                utils::toString(E,16)+" and L="+utils::toString(L,16)+
                " have incompatible values (Lcirc="+utils::toString(sqrt(Lcirc2),16)+")");
    }
    if(asympt) {
        RPeriApoRootFinderPowerLaw fnc(slope, Lrel2);
        R1 = Rcirc * math::findRoot(fnc, 0, 1, ACCURACY);
        R2 = Rcirc * math::findRoot(fnc, 1, 2, ACCURACY);
    } else {
        RPeriApoRootFinder fnc(potential, E, L);
        R1 = math::findRoot(fnc, 0, Rcirc, ACCURACY);
        R2 = math::findRoot(fnc, Rcirc, 3*Rcirc, ACCURACY);
        // for a reasonable potential, 2*Rcirc is actually an upper limit,
        // but in case of trouble, repeat with a safely larger value 
        if(!isFinite(R2))
            R2 = math::findRoot(fnc, Rcirc, (1+1e-8)*R_max(potential, E), ACCURACY);
    }
}

// -------- Same tasks implemented as an interpolation interface -------- //

Interpolator::Interpolator(const BasePotential& potential)
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Interpolator: can only work with axisymmetric potentials");
    double Phiinf = potential.value(coord::PosCyl(INFINITY,1.,1.));
    // not every potential returns a valid value at infinity, but if it does, make sure that it's zero
    if(Phiinf==Phiinf && Phiinf!=0)  
        throw std::runtime_error("Interpolator: can only work with potentials "
            "that tend to zero as r->infinity");   // otherwise assume Phiinf==0
    slopeOut = potential::outerSlope(potential, &Mtot, &coefOut);
    Phi0 = potential.value(coord::PosCyl(0,0,0));
    if(Phi0!=Phi0)   // well-behaved potential must not be NAN at 0, but is allowed to be -INFINITY
        throw std::runtime_error("Interpolator: potential cannot be computed at r=0");

    const double dlogR = .0625;  // grid spacing in log radius
    const double EPS   = 1e-6; // required tolerance on the 2nd deriv to declare the asymptotic limit
    // TODO: make the choice of initial radius more scale-invariant!
    const double logRinit = 0; // initial value of log radius (rather arbitrary but doesn't matter)
    const int NUM_ARRAYS = 8;  // 1d arrays of various quantities:
    std::vector<double> grids[NUM_ARRAYS];
    std::vector<double>   // assign a proper name to each of these arrays:
    &gridR  =grids[0],    // log(r)
    &gridPhi=grids[1],    // scaled Phi(r)
    &gridE  =grids[2],    // scaled Ecirc(Rcirc) where Rcirc=r
    &gridL  =grids[3],    // log(Lcirc)
    &gridNu =grids[4],    // ratio of squared epicyclic frequencies nu^2/Omega^2
    &gridPhider=grids[5], // d(scaled Phi)/ d(log r)
    &gridRder  =grids[6], // d(log Rcirc) / d(log Lcirc)
    &gridLder  =grids[7]; // d(log Lcirc) / d(scaled Ecirc)

    double logR = logRinit;
    int   stage = 0;   // 0 means scan inward, 1 - outward, 2 - done
    while(stage<2) {   // first scan inward in radius, then outward, then stop
        double R = exp(logR);
        double Phival;
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(R, 0, 0), &Phival, &grad, &hess);
        double Ecirc = Phival + 0.5*R*grad.dR;
        // epicyclic frequencies
        double kappa = sqrt(hess.dR2 + 3*grad.dR/R);
        double Omega = sqrt(grad.dR/R);
        double nu2Om = hess.dz2 / grad.dR * R;   // ratio of nu^2/Omega^2 - allowed to be negative
        double Lcirc = Omega * R*R;
        gridPhi.push_back(scaledE(Phival, Phi0));// log-scaled potential at the radius
        gridR.  push_back(logR);                 // log-scaled radius
        gridE.  push_back(scaledE(Ecirc, Phi0)); // log-scaled energy of a circular orbit at the radius
        gridL.  push_back(log(Lcirc));           // log-scaled ang.mom. of a circular orbit
        gridNu. push_back(nu2Om);                // ratio of nu^2/Omega^2 
        // also compute the scaled derivatives for the quintic splines
        double dRdL = 2*Omega / (pow_2(kappa) * R);
        double dLdE = 1/Omega;
        gridRder.push_back(dRdL * Lcirc / R);  // extra factors are from conversion to log-derivatives
        gridLder.push_back(dLdE * scaledEder(Ecirc, Phi0) / Lcirc);
        gridPhider.push_back(grad.dR * R / scaledEder(Phival, Phi0));
        if(!(grad.dR>=0 && Ecirc<0))  // guard against weird values of circular velocity, incl. NaN
            throw std::runtime_error("Interpolator: cannot determine circular velocity at r="+
                utils::toString(R));
        // check if we have reached an asymptotic regime,
        // by examining the curvature (2nd derivative) of relation between scaled Rcirc, Lcirc and E.
        double dlR = dlogR;
        unsigned int np = gridR.size();
        if(np>=3 && fabs(logR - logRinit)>=2) {
            double der2R = math::deriv2(gridE[np-3], gridE[np-2], gridE[np-1],
                gridR[np-3], gridR[np-2], gridR[np-1], gridRder[np-3], gridRder[np-2], gridRder[np-1]);
            double der2L = math::deriv2(gridE[np-3], gridE[np-2], gridE[np-1],
                gridL[np-3], gridL[np-2], gridL[np-1], gridLder[np-3], gridLder[np-2], gridLder[np-1]);
            // check if converged, or if the covered range of radii is too large (>1e6),
            // or we are suffering from loss of precision (too close to the origin and potential is too flat)
            if((fabs(der2R) < EPS && fabs(der2L) < EPS) || 
                fabs(logR - logRinit) >= 15 ||
                (isFinite(Phi0) && Phival-Phi0 < -Phi0*1e-12) )
            {
                if(stage==0) {   // we've been assembling the arrays inward, now need to reverse them
                    for(int i=0; i<NUM_ARRAYS; i++)
                        std::reverse(grids[i].begin(), grids[i].end());
                }
                logR = logRinit;  // restart from the middle
                ++stage;          // switch direction in scanning, or finish
            } else {
                // if we are close to the asymptotic regime but not yet there, we may afford to increase
                // the spacing between grid nodes without deteriorating the accuracy of interpolation
                if(fabs(der2R) < EPS*10)
                    dlR = dlogR*4;
                else if(fabs(der2R) < EPS*100)
                    dlR = dlogR*2;
            }
        }
        if(stage==0)
            logR -= dlR;
        else
            logR += dlR;
    }

    // init various 1d splines
    freqNu = math::CubicSpline(gridR, gridNu, 0, 0);  // set endpoint derivatives to zero
    Phi  = math::QuinticSpline(gridR, gridPhi, gridPhider);
    LofE = math::QuinticSpline(gridE, gridL, gridLder);
    RofL = math::QuinticSpline(gridL, gridR, gridRder);
    // inverse relation between R and Phi - the derivative is reciprocal
    for(unsigned int i=0; i<gridR.size(); i++)
        gridPhider[i] = 1/gridPhider[i];
    RofPhi = math::QuinticSpline(gridPhi, gridR, gridPhider);
}

void Interpolator::evalDeriv(const double R, double* val, double* deriv, double* deriv2) const
{
    double logR = log(R);
    if(logR > Phi.xvalues().back()) {  // extrapolation at large r
        double Rs = exp(logR * slopeOut);
        if(val)
            *val = -Mtot/R + coefOut * (slopeOut==-1 ? logR/R : Rs);
        if(deriv)
            *deriv = slopeOut==-1 ?
                (Mtot + (1-logR) * coefOut) / pow_2(R) :
                (Mtot/R + coefOut * Rs * slopeOut) / R;
        if(deriv2)
            *deriv2 = slopeOut==-1 ?
                (-2*Mtot + (2*logR-3) * coefOut) / pow_3(R) :
                (-2*Mtot/R + coefOut * Rs * slopeOut * (slopeOut-1)) / pow_2(R);
        return;
    }
    double Phival, dscaledPhidlogR;
    Phi.evalDeriv(logR, &Phival, deriv2!=0||deriv!=0? &dscaledPhidlogR : NULL, deriv2);
    Phival = unscaledE(Phival, Phi0);
    double dPhidscaledPhi = scaledEder(Phival, Phi0);
    if(val)
        *val    = Phival;
    if(deriv)
        *deriv  = dPhidscaledPhi * dscaledPhidlogR / R;
    if(deriv2)
        *deriv2 = (*deriv2 - dscaledPhidlogR * (1 + (1 - 2*Phival/Phi0) * dscaledPhidlogR) )
            * dPhidscaledPhi / pow_2(R);
}

double Interpolator::innerSlope(double* Phi0_, double* coef) const
{
    double val, der, logr = Phi.xvalues().front();
    Phi.evalDeriv(logr, &val, &der);
    double Phival = unscaledE(val, Phi0);
    if(isFinite(Phi0)) {
        double slope = der * Phival / Phi0;
        if(Phi0_)
            *Phi0_ = Phi0;
        if(coef)
            *coef = (Phival - Phi0) * exp(-logr * slope);
        return slope;
    } else {
        if(Phi0_)
            *Phi0_ = 0;  // we don't have a more accurate approximation in this case
        if(coef)
            *coef = Phival * exp(logr * der);
        return -der;
    }
}

double Interpolator::R_max(const double E, double* deriv) const
{
    double val;
    RofPhi.evalDeriv(scaledE(E, Phi0), &val, deriv);
    val = exp(val);
    if(deriv)
        *deriv *= val / scaledEder(E, Phi0);
    return val;
}

double Interpolator::L_circ(const double E, double* der) const
{
    if(E<Phi0 || E>=0)
        throw std::invalid_argument("Interpolator: energy outside the allowed range");
    double splVal, splDer;
    LofE.evalDeriv(scaledE(E, Phi0), &splVal, der!=NULL ? &splDer : NULL);
    double Lcirc = exp(splVal);
    if(der)
        *der = splDer / scaledEder(E, Phi0) * Lcirc;
    return Lcirc;
}

double Interpolator::R_from_Lz(const double Lz, double* der) const
{
    double splVal, splDer;
    RofL.evalDeriv(log(fabs(Lz)), &splVal, der!=NULL ? &splDer : NULL);
    double Rcirc = exp(splVal);
    if(der) {
        *der = splDer * Rcirc / Lz;
    }
    return Rcirc;
}

double Interpolator::R_circ(const double E, double* der) const
{
    if(E<Phi0 || E>=0)
        throw std::invalid_argument("Interpolator: energy outside the allowed range");
    double logL, logLder, logR, logRder;
    LofE.evalDeriv(scaledE(E, Phi0), &logL, der!=NULL ? &logLder : NULL);
    RofL.evalDeriv(logL,    &logR, der!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(der)
        *der = logLder * logRder / scaledEder(E, Phi0) * Rcirc;
    return Rcirc;
}

void Interpolator::epicycleFreqs(double R, double& kappa, double& nu, double& Omega) const
{
    double dPhi, d2Phi;
    evalDeriv(R, NULL, &dPhi, &d2Phi);
    double dPhidR_over_R = R>0 || dPhi!=0 ? dPhi/R : d2Phi;  // correct limit at r->0 if dPhi/dr->0 too
    kappa = sqrt(d2Phi + 3*dPhidR_over_R);
    Omega = sqrt(dPhidR_over_R);
    nu    = sqrt(freqNu(log(R)) * dPhidR_over_R);  // nu^2 = Omega^2 * spline-interpolated fnc
}

// --------- 2d interpolation of peri/apocenter radii in equatorial plane --------- //

Interpolator2d::Interpolator2d(const BasePotential& potential) :
    pot(potential)
{
    // for computing the asymptotic values at E=Phi(0), we assume a power-law behavior of potential:
    // Phi = Phi0 + coef * r^s;  potential must be finite at r=0 for this implementation to work
    double Phi0, slope = pot.innerSlope(&Phi0);
    if(!isFinite(Phi0) || slope<=0)
        throw std::runtime_error("Interpolator2d: can only deal with potentials that are finite at r->0");
    const unsigned int sizeE = 50;
    const unsigned int sizeL = 40;

    // create a grid in energy
    std::vector<double> gridE(sizeE);
    for(unsigned int i=0; i<sizeE; i++) {
        double x = 1.*i/(sizeE-1);
        gridE[i] = (1 - pow_3(x) * (10+x*(-15+x*6))) * Phi0; // see below
    }

    // create a grid in L/Lcirc(E)
    std::vector<double> gridL(sizeL);
    for(unsigned int i=0; i<sizeL; i++) {
        double x = 1.*i/(sizeL-1);
        // transformation of interval [0:1] onto itself that places more grid points near the edges:
        // a function with zero 1st and 2nd derivs at x=0 and x=1
        gridL[i] = pow_3(x) * (10+x*(-15+x*6));
        //pow_2(x*x) * (35+x*(-84+x*(70-x*20))); // <-- that would give three zero derivs
    }

    // fill 2d grids for scaled peri/apocenter radii R1, R2 and their derivatives in {E, L/Lcirc}:
    math::Matrix<double> gridR1  (sizeE, sizeL), gridR2  (sizeE, sizeL);
    math::Matrix<double> gridR1dE(sizeE, sizeL), gridR1dL(sizeE, sizeL);
    math::Matrix<double> gridR2dE(sizeE, sizeL), gridR2dL(sizeE, sizeL);

    // loop over values of energy strictly inside the interval [Ein:0];
    // the boundary values will be treated separately
    for(unsigned int iE=1; iE<sizeE-1; iE++) {
        double E = gridE[iE];
        coord::GradCyl grad;
        coord::HessCyl hess;
        double Rc     = potential::R_circ(potential, E);  // radius of a circular orbit with this energy
        potential.eval(coord::PosCyl(Rc,0,0), NULL, &grad, &hess);
        double Lc     = Rc*sqrt(Rc*grad.dR);              // corresponding angular momentum
        double Om2kap2= grad.dR / (3*grad.dR + Rc*hess.dR2);  // ratio of epi.freqs (Omega / kappa)^2
        double dRcdE  = 2/grad.dR * Om2kap2;
        double dLcdE  = Rc*Rc/Lc;
        /** The actual quantities to interpolate are not peri/apocenter radii R1/R2 themselves,
            but rather scaled variables xi1/xi2 that are exactly zero at L=Lc,
            and vary linearly near the edges of the interval. */
        for(unsigned int iL=0; iL<sizeL-1; iL++) {
            double L = Lc * gridL[iL];
            double R1, R2, Phi;
            if(iL==0) {  // exact values for a radial orbit
                R1=0;
                R2=potential::R_max(potential, E);
            } else
                potential::findPlanarOrbitExtent(potential, E, L, R1, R2);
            gridR1(iE, iL) = pow_2((R1-Rc)/Rc);
            gridR2(iE, iL) = pow_2((R2-Rc)/Rc);
            // compute derivatives of Rperi/apo w.r.t. E and L/Lcirc
            potential.eval(coord::PosCyl(R1,0,0), &Phi, &grad);
            if(R1==0) grad.dR=0;   // it won't be used anyway, but prevents a possible NaN
            double dR1dE = (1 - 2*(E-Phi) * dLcdE / Lc) / (grad.dR - 2*(E-Phi) / R1);
            double dR1dZ = -Lc * sqrt(2*(E-Phi)) / (grad.dR * R1 - 2*(E-Phi));
            potential.eval(coord::PosCyl(R2,0,0), &Phi, &grad);
            double dR2dE = (1 - 2*(E-Phi) * dLcdE / Lc) / (grad.dR - 2*(E-Phi) / R2);
            double dR2dZ = -Lc * L / (grad.dR * pow_2(R2) - 2*(E-Phi) * R2);
            gridR1dE(iE, iL) = 2*(R1-Rc) / pow_2(Rc) * (dR1dE - R1/Rc * dRcdE);
            gridR1dL(iE, iL) = 2*(R1-Rc) / pow_2(Rc) *  dR1dZ;
            gridR2dE(iE, iL) = 2*(R2-Rc) / pow_2(Rc) * (dR2dE - R2/Rc * dRcdE);
            gridR2dL(iE, iL) = 2*(R2-Rc) / pow_2(Rc) *  dR2dZ;
        }
        // limiting values for a nearly circular orbit:
        // R{1,2} = Rcirc * (1 +- Omega/kappa * ecc),  where ecc = sqrt(1 - (L/Lcirc)^2)
        gridR1(iE, sizeL-1) = gridR2(iE, sizeL-1) = gridR1dE(iE, sizeL-1) = gridR2dE(iE, sizeL-1) = 0;
        gridR1dL(iE, sizeL-1) = gridR2dL(iE, sizeL-1) = -2*Om2kap2;
    }

    // asymptotic values at E -> Phi0
    double dE = gridE[1] - gridE[0];
    for(unsigned int iL=0; iL<sizeL-1; iL++) {
        double Z = gridL[iL];
        RPeriApoRootFinderPowerLaw fnc(slope, Z*Z);
        double R1overRc = math::findRoot(fnc, 0, 1, ACCURACY);
        double R2overRc = iL==0 ? pow(1+slope/2, 1/slope) : math::findRoot(fnc, 1, 2, ACCURACY);
        double dR1overRc_dZ = iL==0 ? sqrt(slope/(slope+2)) :
            slope*Z / ((slope+2) * (1-pow(R1overRc, slope)) * R1overRc);
        double dR2overRc_dZ = slope*Z / ((slope+2) * (1-pow(R2overRc, slope)) * R2overRc);
        gridR1  (0, iL) = pow_2(R1overRc-1);
        gridR2  (0, iL) = pow_2(R2overRc-1);
        gridR1dL(0, iL) =  2 * (R1overRc-1) * dR1overRc_dZ;
        gridR2dL(0, iL) =  2 * (R2overRc-1) * dR2overRc_dZ;
        // we cannot directly compute derivatives w.r.t energy at gridE[0],
        // so we use quadratic interpolation to obtain them
        // from the values at gridE[0], gridE[1] and derivs w.r.t E at gridE[1]
        gridR1dE(0, iL) = 2 * (gridR1(1, iL) - gridR1(0, iL)) / dE - gridR1dE(1, iL);
        gridR2dE(0, iL) = 2 * (gridR2(1, iL) - gridR2(0, iL)) / dE - gridR2dE(1, iL);
    }
    // limiting values for L=Lcirc
    gridR1  (0, sizeL-1) = gridR2  (0, sizeL-1) = gridR1dE(0, sizeL-1) = gridR2dE(0, sizeL-1) = 0;
    gridR1dL(0, sizeL-1) = gridR2dL(0, sizeL-1) = -2/(slope+2);

    // asymptotic values at E -> 0, assuming Keplerian regime
    dE = gridE[sizeE-1] - gridE[sizeE-2];
    for(unsigned int iL=0; iL<sizeL; iL++) {
        gridR1  (sizeE-1, iL) = gridR2  (sizeE-1, iL) = 1 - pow_2(gridL[iL]);
        gridR1dL(sizeE-1, iL) = gridR2dL(sizeE-1, iL) = -2*gridL[iL];
        gridR1dE(sizeE-1, iL) = -gridR1dE(sizeE-2, iL) +
            2 * (gridR1(sizeE-1, iL) - gridR1(sizeE-2, iL)) / dE;
        gridR2dE(sizeE-1, iL) = -gridR2dE(sizeE-2, iL) +
            2 * (gridR2(sizeE-1, iL) - gridR2(sizeE-2, iL)) / dE;
    }

#if 0  /// test code, to be removed
    std::ofstream strm("file.dat");
    strm << std::setprecision(15);
    for(unsigned int iE=0; iE<sizeE; iE++) {
        for(unsigned int iL=0; iL<sizeL; iL++) {
            strm << gridE[iE] << "\t" << gridL[iL] << "\t" << 
            gridR1  (iE, iL) << "\t" << gridR1dE(iE, iL) << "\t" << gridR1dL(iE, iL) << "\t" <<
            (iE>0 ? (gridR1(iE, iL)-gridR1(iE-1, iL)) / (gridE[iE]-gridE[iE-1]) : NAN) << "\t" <<
            (iL>0 ? (gridR1(iE, iL)-gridR1(iE, iL-1)) / (gridL[iL]-gridL[iL-1]) : NAN) << "\t" <<
            gridR2  (iE, iL) << "\t" << gridR2dE(iE, iL) << "\t" << gridR2dL(iE, iL) << "\t" <<
            (iE>0 ? (gridR2(iE, iL)-gridR2(iE-1, iL)) / (gridE[iE]-gridE[iE-1]) : NAN) << "\t" <<
            (iL>0 ? (gridR2(iE, iL)-gridR2(iE, iL-1)) / (gridL[iL]-gridL[iL-1]) : NAN) << "\n";
        }
        strm<<"\n";
    }
    strm.close();
#endif

    // create 2d interpolators
    intR1 = math::QuinticSpline2d(gridE, gridL, gridR1, gridR1dE, gridR1dL);
    intR2 = math::QuinticSpline2d(gridE, gridL, gridR2, gridR2dE, gridR2dL);
}

void Interpolator2d::findScaledOrbitExtent(double E, double Lrel,
    double &R1rel, double &R2rel) const
{
    R1rel = 1 - sqrt(intR1.value(E, Lrel));
    R2rel = 1 + sqrt(intR2.value(E, Lrel));
}

void Interpolator2d::findPlanarOrbitExtent(double E, double L,
    double &R1, double &R2) const
{
    double Lc   = pot.L_circ(E);
    double Rc   = pot.R_from_Lz(Lc);
    double Lrel = Lc>0 ? fmin(fabs(L/Lc), 1) : 0;
    findScaledOrbitExtent(E, Lrel, R1, R2);
    R1 = fmin(refineRoot(pot, R1*Rc, E, L), Rc);  // one iteration of root polishing
    R2 = fmax(refineRoot(pot, R2*Rc, E, L), Rc);
}

}  // namespace potential
