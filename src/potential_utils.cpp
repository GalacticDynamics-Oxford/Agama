#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <stdexcept>
#include <fstream>   // for writing debug info

namespace potential{

namespace{  // internal routines

/// relative accuracy of root-finders for radius
static const double ACCURACY_ROOT = 1e-10;

/// required tolerance on the 2nd deriv to declare the asymptotic limit
static const double ACCURACY_INTERP = 1e-6;

/// a number that is considered nearly infinity in log-scaled root-finders
static const double HUGE_NUMBER = 1e100;

/// safety factor to avoid roundoff errors in estimating the inner/outer asymptotic slopes
static const double ROUNDOFF_THRESHOLD = DBL_EPSILON / ROOT3_DBL_EPSILON;

/// minimum relative difference between two adjacent values of potential (to reduce roundoff errors)
static const double MIN_REL_DIFFERENCE = 1e-12;
    
/// fixed order of Gauss-Legendre integration of PhaseVolume on each segment of a log-grid
static const int GLORDER1 = 6;   // for shorter segments
static const int GLORDER2 = 10;  // for larger segments
/// the choice between short and long segments is determined by the ratio between consecutive nodes
static const double GLRATIO = 2.0;

// -------- routines for conversion between energy, radius and angular momentum --------- //

/** helper class to find the root of  Phi(R) = E */
class RmaxRootFinder: public math::IFunction {
    const math::IFunction& poten;
    const double E;
public:
    RmaxRootFinder(const math::IFunction& _poten, double _E) : poten(_poten), E(_E) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* =0) const
    {
        double Phi, dPhidR, R = exp(logR);
        poten.evalDeriv(R, &Phi, &dPhidR);
        if(val) {
            if(R==0 && Phi==-INFINITY)
                *val = +E;  // safely negative value
            else if(R>=HUGE_NUMBER && !isFinite(Phi))
                *val = -E;
            else
                *val = Phi-E;
        }
        if(deriv)
            *deriv = dPhidR * R;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  Phi(R) + 1/2 R dPhi/dR = E
    (i.e. the radius R of a circular orbit with the given energy E).
*/
class RcircRootFinder: public math::IFunction {
    const math::IFunction& poten;
    const double E;
public:
    RcircRootFinder(const math::IFunction& _poten, double _E) : poten(_poten), E(_E) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* deriv2=0) const
    {
        double Phi, dPhidR, d2PhidR2, R = exp(logR);
        poten.evalDeriv(R, &Phi, &dPhidR, &d2PhidR2);
        if(val) {
            if(R==0 && Phi==-INFINITY)
                *val = +1+fabs(E);  // safely positive value
            else if(R>=HUGE_NUMBER && !isFinite(Phi))
                *val = -1-fabs(E);  // safely negative value
            else {
                double v = R * dPhidR;
                *val = 2*(E-Phi) - (isFinite(v) ? v : 0);
            }
        }
        if(deriv)
            *deriv = (-3*dPhidR - R*d2PhidR2) * R;
        if(deriv2)
            *deriv2 = NAN;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** helper class to find the root of  L^2 - R^3 d\Phi(R)/dR = 0
    (i.e. the radius R of a circular orbit with the given angular momentum L).
*/
class RfromLRootFinder: public math::IFunction {
    const math::IFunction& poten;
    const double L2;
public:
    RfromLRootFinder(const math::IFunction& _poten, double _L) : poten(_poten), L2(_L*_L) {}
    virtual void evalDeriv(const double logR, double* val=0, double* deriv=0, double* deriv2=0) const
    {
        double dPhidR, d2PhidR2, R = exp(logR);
        poten.evalDeriv(R, NULL, &dPhidR, &d2PhidR2);
        double F = pow_3(R)*dPhidR; // Lz^2(R)
        if(!isFinite(F))            // this may happen if R --> 0 or R --> infinity,
            F = R<1 ? 0 : 2*L2;     // in these cases replace it with a finite number of a correct sign
        if(val)
            *val = F - L2;
        if(deriv)
            *deriv = pow_3(R) * (3*dPhidR + R*d2PhidR2);
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
            *val = s!=0 ?  (2+s)*x*x - 2*std::pow(x, 2+s) - v2*s  :  (x==0 ? 0 : x*x * (1 - 2*log(x)) - v2);
        if(der)
            *der = s!=0 ?  (4+2*s) * (x - std::pow(x, 1+s))  :  -4*x*log(x);
    }
};

/// root polishing routine to improve the accuracy of peri/apocenter radii determination
inline double refineRoot(const math::IFunction& pot, double R, double E, double L)
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

/// Scaling transformations for energy: the input energy ranges from Phi0 to 0,
/// the output scaled variable - from -inf to +inf. Here Phi0=Phi(0) may be finite or -inf.
/// The goal is to avoid cancellation errors when Phi0 is finite and E --> Phi0 --
/// in this case the scaled variable may achieve any value down to -inf, instead of
/// cramping into a few remaining bits of precision when E is almost equal to Phi0,
/// so that any function that depends on scaledE may be comfortably evaluated with full precision.
/// Additionally, this transformation is intended to achieve an asymptotic power-law behaviour
/// for any quantity whose logarithm is interpolated as a function of scaledE and linearly
/// extrapolated as its argument tends to +- infinity.

/// return scaledE and dE/d(scaledE) as functions of E and invPhi0 = 1/Phi(0)
inline void scaleE(const double E, const double invPhi0,
    /*output*/ double& scaledE, double& dEdscaledE)
{
    double expE = invPhi0 - 1/E;
    scaledE     = log(expE);
    dEdscaledE  = E * E * expE;
}

/// return E and dE/d(scaledE) as functions of scaledE
inline void unscaleE(const double scaledE, const double invPhi0,
    /*output*/ double& E, double& dEdscaledE, double& d2EdscaledE2)
{
    double expE = exp(scaledE);
    E           = 1 / (invPhi0 - expE);
    dEdscaledE  = E * E * expE;
    d2EdscaledE2= E * dEdscaledE * (invPhi0 + expE);
}

/// same as above, but for two separate values of E1 and E2;
/// in addition, compute the difference between E1 and E2 in a way that is not prone
/// to cancellation when both E1 and E2 are close to Phi0 and the latter is finite.
inline void unscaleDeltaE(const double scaledE1, const double scaledE2, const double invPhi0,
    /*output*/ double& E1, double& E2, double& E1minusE2, double& dE1dscaledE1)
{
    double exp1  = exp(scaledE1);
    double exp2  = exp(scaledE2);
    E1           = 1 / (invPhi0 - exp1);
    E2           = 1 / (invPhi0 - exp2);
    E1minusE2    = (exp1 - exp2) * E1 * E2;
    dE1dscaledE1 = E1 * E1 * exp1;
}

/** A specially designed function whose second derivative indicates the local variation of potential,
    used to determine the range and spacing between radial grid nodes for interpolation.
    Its second derivative is identically zero if the potential is a power-law in radius (e.g., -M/r).
*/
class ScalePhi: public math::IFunction {
    const math::IFunction& pot;
    const double invPhi0;
public:
    ScalePhi(const math::IFunction& _pot) : pot(_pot), invPhi0(1/pot(0)) {}
    virtual void evalDeriv(const double logr, double* val, double* der, double* der2) const {
        double r=exp(logr), Phi, dPhi, d2Phi;
        pot.evalDeriv(r, &Phi, &dPhi, &d2Phi);
        double expE = invPhi0 - 1/Phi;
        if(val)
            *val = log(expE) + 2 * log(-Phi);
        if(der)
            *der = dPhi * r / (Phi * Phi * expE) + 2 * dPhi/Phi*r;
        if(der2) {
            if(invPhi0!=0 && expE < -invPhi0 * MIN_REL_DIFFERENCE)
                // in case of a finite potential at r=0,
                // we avoid approaching too close to 0 to avoid roundoff errors in Phi
                *der2 = 0;
            else
                *der2 = pow_2(r/Phi) / expE * (dPhi * (1/r - dPhi/Phi * (2 + 1/Phi/expE)) + d2Phi)
                + 2 * r/Phi * (d2Phi*r + dPhi*(1-dPhi/Phi*r));
        }
    }
    virtual unsigned int numDerivs() const { return 2; }
};

}  // internal namespace


double v_circ(const math::IFunction& potential, double radius)
{
    if(radius==0)
        return isFinite(potential.value(0)) ? 0 : INFINITY;
    double dPhidr;
    potential.evalDeriv(radius, NULL, &dPhidr);
    return sqrt(radius * dPhidr);
}

double R_circ(const math::IFunction& potential, double energy) {
    return exp(math::findRoot(RcircRootFinder(potential, energy), math::ScalingInf(), ACCURACY_ROOT));
}

double R_from_L(const math::IFunction& potential, double L) {
    if(L==0)
        return 0;
    return exp(math::findRoot(RfromLRootFinder(potential, L), math::ScalingInf(), ACCURACY_ROOT));
}

double R_max(const math::IFunction& potential, double energy) {
    return exp(math::findRoot(RmaxRootFinder(potential, energy), math::ScalingInf(), ACCURACY_ROOT));
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

double innerSlope(const math::IFunction& potential, double* Phi0, double* coef)
{
    // this routine shouldn't suffer from cancellation errors, provided that
    // the potential and its derivatives are computed accurately,
    // thus we may use a fixed tiny radius at which the slope is estimated.
    double r = 1e-10;  // TODO: try making it more scale-invariant?
    double Phi, dPhidR, d2PhidR2;
    potential.evalDeriv(r, &Phi, &dPhidR, &d2PhidR2);
    double  s = 1 + r * d2PhidR2 / dPhidR;
    if(coef)
        *coef = s==0 ?  dPhidR * r  :  dPhidR / s * std::pow(r, 1-s);
    if(Phi0)
        *Phi0 = s==0 ?  Phi - r * dPhidR * log(r)  :  Phi - r * dPhidR / s;
    return s;
}

double outerSlope(const math::IFunction& potential, double* M, double* coef)
{
    double r = 1e+10;  // start reasonably far...
    double s, Phi, dPhidR, d2PhidR2;
    bool roundoff = false;
    int numiter = 0;
    do {
        if(roundoff)  // at each iteration, decrease the radius at which the estimates are made,
            r /= 10;  // if the computation was dominated by roundoff error at the previous iteration
        potential.evalDeriv(r, &Phi, &dPhidR, &d2PhidR2);
        double num1 = 2*dPhidR, num2 = -r*d2PhidR2, den1 = dPhidR, den2 = -Phi/r;
        s = (num1 - num2) / (den1 - den2);
        roundoff =    // check if the value of s is dominated by roundoff errors
            fabs(num1-num2) < fmax(fabs(num1), fabs(num2)) * ROUNDOFF_THRESHOLD ||
            fabs(den1-den2) < fmax(fabs(den1), fabs(den2)) * ROUNDOFF_THRESHOLD;
    } while(roundoff && ++numiter<10);
    if(roundoff || s>0) {    // not successful - return the total mass only
        if(coef) *coef=0;
        if(M)    *M = -potential.value(r) * r;
        return 0;
    }
    if(coef)
        *coef = s==-1 ?  (Phi + r*dPhidR) * r  :  (Phi + r*dPhidR) * std::pow(r, -s) / (s+1);
    if(M)
        *M    = s==-1 ?  (log(r) * (Phi + r*dPhidR) - Phi) * r  :  (r*dPhidR - s*Phi) * r / (s+1);
    return s;
}

void findPlanarOrbitExtent(const BasePotential& potential, double E, double L, double& R1, double& R2)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("findPlanarOrbitExtent only works for axisymmetric potentials");
    double Phi0, coef, slope = innerSlope(PotentialWrapper(potential), &Phi0, &coef);
    // accurate treatment close to the origin assuming a power-law asymptotic behavior of potential
    bool asympt  = slope>0 && E>=Phi0 && E-Phi0 < fabs(Phi0)*SQRT_DBL_EPSILON;
    double Rcirc = !asympt ?  R_circ(potential, E) :
        slope==0 ?  exp((E-Phi0) / coef - 0.5)  :  std::pow((E-Phi0) / (coef * (1+0.5*slope)), 1/slope);
    if(!isFinite(Rcirc))
        throw std::invalid_argument("Error in findPlanarOrbitExtent: cannot determine Rcirc(E="+
            utils::toString(E,16) + ")\n" + utils::stacktrace());
    if(asympt) {
        double vcirc = slope==0 ? sqrt(coef) : sqrt(coef*slope) * std::pow(Rcirc, 0.5*slope);
        RPeriApoRootFinderPowerLaw fnc(slope, pow_2(L / (Rcirc * vcirc)));
        R1 = Rcirc * math::findRoot(fnc, 0, 1, ACCURACY_ROOT);
        R2 = Rcirc * math::findRoot(fnc, 1, 2, ACCURACY_ROOT);
    } else {
        RPeriApoRootFinder fnc(potential, E, L);
        R1 = math::findRoot(fnc, 0, Rcirc, ACCURACY_ROOT);
        R2 = math::findRoot(fnc, Rcirc, 3*Rcirc, ACCURACY_ROOT);
        // for a reasonable potential, 2*Rcirc is actually an upper limit,
        // but in case of trouble, repeat with a safely larger value 
        if(!isFinite(R2))
            R2 = math::findRoot(fnc, Rcirc, (1+ACCURACY_ROOT)*R_max(potential, E), ACCURACY_ROOT);
    }
    if(!isFinite(R1+R2)) {
        // this may arise when the orbit is very nearly circular due to roundoff errors,
        // or due to intentional foul.
        // Assuming the former reason, do not crash but pretend to be on a circular orbit exactly.
        R1 = R2 = Rcirc;
        utils::msg(utils::VL_WARNING, FUNCNAME,
            "E=" + utils::toString(E,16) + " and L=" + utils::toString(L,16) +
            " have incompatible values (Rcirc=" + utils::toString(Rcirc,16) + ")");
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
    slopeOut = potential::outerSlope(PotentialWrapper(potential), &Mtot, &coefOut);
    double Phi0 = potential.value(coord::PosCyl(0,0,0));
    if(!(Phi0<0))  // well-behaved potential must not be NAN at 0, but is allowed to be -INFINITY
        throw std::runtime_error("Interpolator: potential cannot be computed at r=0");
    invPhi0 = 1./Phi0;

    std::vector<double> gridLogR = math::createInterpolationGrid(
        ScalePhi(PotentialWrapper(potential)), ACCURACY_INTERP);
    // erase innermost grid nodes where the value of potential is too close to Phi(0) (within roundoff)
    while(!gridLogR.empty() &&
        potential.value(coord::PosCyl(exp(gridLogR[0]), 0, 0)) < Phi0 * (1-MIN_REL_DIFFERENCE))
        gridLogR.erase(gridLogR.begin());

    unsigned int gridsize = gridLogR.size();
    std::vector<double>   // various arrays:
    gridPhi(gridsize),    // scaled Phi(r)
    gridE(gridsize),      // scaled Ecirc(Rcirc) where Rcirc=r
    gridL(gridsize),      // log(Lcirc)
    gridNu(gridsize),     // ratio of squared epicyclic frequencies nu^2/Omega^2
    gridPhider(gridsize), // d(scaled Phi)/ d(log r)
    gridRder(gridsize),   // d(log Rcirc) / d(log Lcirc)
    gridLder(gridsize);   // d(log Lcirc) / d(scaled Ecirc)

    std::ofstream strm;   // debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open("potentialInterpolator.log");
        strm << "#R      \tPhi(R)  \tdPhi/dR \td2Phi/dR2\td2Phi/dz2\tEcirc   \tLcirc\n";
    }

    for(unsigned int i=0; i<gridsize; i++) {
        double R = exp(gridLogR[i]);
        double Phival;
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(R, 0, 0), &Phival, &grad, &hess);
        // epicyclic frequencies
        double kappa2= hess.dR2 + 3*grad.dR/R;  // kappa^2
        double Omega = sqrt(grad.dR/R);         // Omega, always exists if potential is monotonic with R
        double nu2Om = hess.dz2 / grad.dR * R;  // ratio of nu^2/Omega^2 - allowed to be negative
        double Ecirc = Phival + 0.5*R*grad.dR;  // energy of a circular orbit at this radius
        double Lcirc = Omega * R*R;             // angular momentum of a circular orbit
        double scaledPhi, dPhidscaledPhi, scaledEcirc, dEcircdscaledEcirc;
        scaleE(Phival, invPhi0, scaledPhi,   dPhidscaledPhi);
        scaleE(Ecirc,  invPhi0, scaledEcirc, dEcircdscaledEcirc);
        gridPhi[i] = scaledPhi;    // log-scaled potential at the radius
        gridE  [i] = scaledEcirc;  // log-scaled energy of a circular orbit at the radius
        gridL  [i] = log(Lcirc);   // log-scaled ang.mom. of a circular orbit
        gridNu [i] = nu2Om;        // ratio of nu^2/Omega^2 
        // also compute the scaled derivatives for the quintic splines
        double dRdL = 2*Omega / (kappa2 * R);
        double dLdE = 1/Omega;
        gridRder  [i] = dRdL * Lcirc / R;  // extra factors are from conversion to log-derivatives
        gridLder  [i] = dLdE * dEcircdscaledEcirc / Lcirc;
        gridPhider[i] = grad.dR * R / dPhidscaledPhi;
        // debugging printout
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm << utils::pp(R, 15) + '\t' +
            utils::pp(Phival,    15) + '\t' +
            utils::pp(grad.dR,   15) + '\t' +
            utils::pp(hess.dR2,  15) + '\t' +
            utils::pp(hess.dz2,  15) + '\t' +
            utils::pp(Ecirc,     15) + '\t' +
            utils::pp(Lcirc,     15) + '\n' << std::flush;
        }
        // guard against weird behaviour of potential
        if(!(Phival<0 && grad.dR>=0 && (i==0 || gridPhi[i]>gridPhi[i-1])))
            throw std::runtime_error(
                "Interpolator: potential is not monotonically increasing with radius at R=" +
                utils::toString(R) + '\n' + utils::stacktrace());
        if(!(Ecirc<0 && Lcirc>=0 && (i==0 || (gridE[i]>gridE[i-1] && gridL[i]>gridL[i-1])) && dRdL>=0))
            throw std::runtime_error(
                "Interpolator: energy or angular momentum of a circular orbit are not monotonic "
                "with radius at R=" + utils::toString(R) + '\n' + utils::stacktrace());
        if(!(nu2Om>=0))  // not a critical error, but possibly a sign of problems
            utils::msg(utils::VL_WARNING, "Interpolator",
                "Vertical epicyclic frequency is negative at R=" + utils::toString(R));
    }

    // init various 1d splines
    freqNu = math::CubicSpline  (gridLogR, gridNu,   0, 0);  // set endpoint derivatives to zero
    LofE   = math::QuinticSpline(gridE,    gridL,    gridLder);
    RofL   = math::QuinticSpline(gridL,    gridLogR, gridRder);
    PhiofR = math::QuinticSpline(gridLogR, gridPhi,  gridPhider);
    // inverse relation between R and Phi - the derivative is reciprocal
    for(unsigned int i=0; i<gridsize; i++)
        gridPhider[i] = 1/gridPhider[i];
    RofPhi = math::QuinticSpline(gridPhi, gridLogR, gridPhider);
}

void Interpolator::evalDeriv(const double R, double* val, double* deriv, double* deriv2) const
{
    double logR = log(R);
    if(logR > PhiofR.xvalues().back()) {  // extrapolation at large r
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
    double scaledPhi, dscaledPhidlogR, Phival, dPhidscaledPhi, dummy;
    PhiofR.evalDeriv(logR, &scaledPhi, deriv2!=0||deriv!=0? &dscaledPhidlogR : NULL, deriv2);
    unscaleE(scaledPhi, invPhi0, Phival, dPhidscaledPhi, dummy);
    if(val)
        *val    = Phival;
    if(deriv)
        *deriv  = dPhidscaledPhi * dscaledPhidlogR / R;
    if(deriv2)
        *deriv2 = (*deriv2 - dscaledPhidlogR * (1 + (1 - 2 * Phival * invPhi0) * dscaledPhidlogR) )
            * dPhidscaledPhi / pow_2(R);
}

double Interpolator::innerSlope(double* Phi0, double* coef) const
{
    double val, der, logr = PhiofR.xvalues().front();
    PhiofR.evalDeriv(logr, &val, &der);
    double Phival, dummy1, dummy2;
    unscaleE(val, invPhi0, Phival, dummy1, dummy2);
    if(invPhi0!=0) {
        double slope = der * Phival * invPhi0;
        if(Phi0)
            *Phi0 = 1/invPhi0;
        if(coef)
            *coef = (Phival - 1/invPhi0) * exp(-logr * slope);
        return slope;
    } else {
        if(Phi0)
            *Phi0 = 0;  // we don't have a more accurate approximation in this case
        if(coef)
            *coef = Phival * exp(logr * der);
        return -der;
    }
}

double Interpolator::R_max(const double E, double* deriv) const
{
    double scaledE, dEdscaledE, val;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    RofPhi.evalDeriv(scaledE, &val, deriv);
    val = exp(val);
    if(deriv)
        *deriv *= val / dEdscaledE;
    return val;
}

double Interpolator::L_circ(const double E, double* der) const
{
    if(!(E*invPhi0<=1 && E<=0))
        throw std::invalid_argument("Interpolator: energy E="+utils::toString(E)+
            " is outside the allowed range");
    double scaledE, dEdscaledE, splVal, splDer;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &splVal, der!=NULL ? &splDer : NULL);
    double Lcirc = exp(splVal);
    if(der)
        *der = splDer / dEdscaledE * Lcirc;
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
    if(!(E*invPhi0<=1 && E<=0))
        throw std::invalid_argument("Interpolator: energy E="+utils::toString(E)+
            " is outside the allowed range");
    double scaledE, dEdscaledE, logL, logLder, logR, logRder;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &logL, der!=NULL ? &logLder : NULL);
    RofL.evalDeriv(logL,    &logR, der!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(der)
        *der = logLder * logRder / dEdscaledE * Rcirc;
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


//---- Correspondence between h and E ----//

PhaseVolume::PhaseVolume(const math::IFunction& pot)
{
    double Phi0 = pot(0);
    if(!(Phi0<0))
        throw std::invalid_argument("PhaseVolume: invalid value of Phi(r=0)");
    invPhi0 = 1/Phi0;

    // create grid in log(r)
    std::vector<double> gridr = math::createInterpolationGrid(ScalePhi(pot), ACCURACY_INTERP);
    std::transform(gridr.begin(), gridr.end(), gridr.begin(), exp);  // convert to grid in r
    std::vector<double> gridE;
    gridE.reserve(gridr.size());

    // compute the potential at each node of the radial grid, throwing away nodes that are
    // too closely spaced, such that the difference between adjacent potential values suffers from
    // roundoff/cancellation errors due to finite precision of floating-point arithmetic
    double prevPhi = Phi0;
    for(unsigned int i=0; i<gridr.size();) {
        double E = pot.value(gridr[i]);
        if(i>0 && !(E>=gridE[i-1]))
            throw std::invalid_argument(
                "PhaseVolume: potential is non-monotonic at r="+utils::toString(gridr[i]));
        if(E > prevPhi * (1-MIN_REL_DIFFERENCE)) {
            gridE.push_back(E);
            i++;
            prevPhi = E;
        } else {
            gridr.erase(gridr.begin()+i);
        }
    }
    unsigned int gridsize = gridr.size();
    if(gridsize == 0)
        throw std::runtime_error("PhaseVolume: cannot construct a suitable grid in radius");
    
    std::vector<double> gridH(gridsize), gridG(gridsize);
    double glnodes1[GLORDER1], glweights1[GLORDER1], glnodes2[GLORDER2], glweights2[GLORDER2];
    math::prepareIntegrationTableGL(0, 1, GLORDER1, glnodes1, glweights1);
    math::prepareIntegrationTableGL(0, 1, GLORDER2, glnodes2, glweights2);

    // loop through all grid segments, and in each segment add the contribution to integrals
    // in all other segments leftmost of the current one (thus the complexity is Ngrid^2,
    // but the number of potential evaluations is only Ngrid * GLORDER).
    for(unsigned int i=0; i<gridsize; i++) {
        double deltar = gridr[i] - (i>0 ? gridr[i-1] : 0);
        // choose a higher-order quadrature rule for longer grid segments
        int glorder = i>0 && gridr[i] < gridr[i-1]*GLRATIO ? GLORDER1 : GLORDER2;
        const double *glnodes   = glorder == GLORDER1 ? glnodes1   : glnodes2;
        const double *glweights = glorder == GLORDER1 ? glweights1 : glweights2;
        for(int k=0; k<glorder; k++) {
            // node of Gauss-Legendre quadrature within the current segment (r[i-1] .. r[i]);
            // the integration variable y ranges from 0 to 1, and r(y) is defined below
            double y = glnodes[k];
            double r = gridr[i] - pow_2(1-y) * deltar;
            double E = pot.value(r);
            // contribution of this point to each integral on the current segment, taking into account
            // the transformation of variable y -> r  and the common weight factor r^2
            double weight = glweights[k] * 2*(1-y) * deltar * pow_2(r);
            // add a contribution to the integrals expressing g(E_j) and h(E_j) for all E_j > Phi(r[i])
            for(unsigned int j=i; j<gridsize; j++) {
                double v  = sqrt(fmax(0, gridE[j] - E));
                gridG[j] += weight * v * 1.5;
                gridH[j] += weight * pow_3(v);
            }
        }
    }

    // debugging output: asymptotic slopes
    if(utils::verbosityLevel >= utils::VL_DEBUG) {
        double inner = gridH.front() / gridG.front() / (gridE.front() - (isFinite(Phi0) ? Phi0 : 0));
        double outer = gridH.back()  / gridG.back()  / gridE.back();
        utils::msg(utils::VL_DEBUG, "PhaseVolume", "Potential asymptotes: "
            "Phi(r) ~ r^" + utils::toString( 6 * inner / (2 - 3 * inner), 8) + " at small r, "
            "Phi(r) ~ r^" + utils::toString( 6 * outer / (2 - 3 * outer), 8) + " at large r.");
    }
    
    std::ofstream strm;   // debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open("potentialPhaseVolume.log");
        strm << "#r      \tE       \th       \tg=dh/dE\n";
    }
    
    // convert h, g and E to scaled coordinates
    for(unsigned int i=0; i<gridsize; i++) {
        double E = gridE[i], H = gridH[i], G = gridG[i], scaledE, dEdscaledE;
        scaleE(E, invPhi0, scaledE, dEdscaledE);
        gridE[i] = scaledE;
        gridH[i] = log(H) + log(16*M_PI*M_PI/3*2*M_SQRT2);
        gridG[i] = G / H * dEdscaledE;
        // debugging printout
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm << utils::pp(gridr[i], 15) + '\t' + utils::pp(E, 15) + '\t' +
                utils::pp(H, 15) + '\t' + utils::pp(G, 15) + '\n' << std::flush;
        }
    }

    HofE = math::QuinticSpline(gridE, gridH, gridG);
    // inverse relation between E and H - the derivative is reciprocal
    for(unsigned int i=0; i<gridG.size(); i++)
        gridG[i] = 1/gridG[i];
    EofH = math::QuinticSpline(gridH, gridE, gridG);
}

void PhaseVolume::evalDeriv(const double E, double* h, double* g, double*) const
{
    // out-of-bounds value of energy returns 0 or infinity, but not NAN
    if(!(E * invPhi0 < 1)) {
        if(h) *h=0;
        if(g) *g=0;
        return;
    }
    if(E>=0) {
        if(h) *h=INFINITY;
        if(g) *g=INFINITY;
        return;
    }
    double scaledE, dEdscaledE, val;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    HofE.evalDeriv(scaledE, &val, g);
    val = exp(val);
    if(h)
        *h = val;
    if(g)
        *g *= val / dEdscaledE;
}

double PhaseVolume::E(const double h, double* g, double* dgdh) const
{
    if(h==0) {
        if(g) *g=0;
        return invPhi0 == 0 ? -INFINITY : 1/invPhi0;
    }
    if(h==INFINITY) {
        if(g) *g=INFINITY;
        return 0;
    }
    double scaledE, dEdscaledE, d2EdscaledE2, realE, dscaledEdlogh, d2scaledEdlogh2;
    EofH.evalDeriv(log(h), &scaledE,
        g!=NULL || dgdh!=NULL ? &dscaledEdlogh : NULL,
        dgdh!=NULL ? &d2scaledEdlogh2 : NULL);
    unscaleE(scaledE, invPhi0, realE, dEdscaledE, d2EdscaledE2);
    if(g)
        *g = h / ( dEdscaledE * dscaledEdlogh );
    if(dgdh)
        *dgdh = ( (1 - d2scaledEdlogh2 / dscaledEdlogh) / dscaledEdlogh -
            d2EdscaledE2 / dEdscaledE ) / dEdscaledE;
    return realE;
}

double PhaseVolume::deltaE(const double logh1, const double logh2, double* g1) const
{
    //return E(exp(logh1), g1) - E(exp(logh2)); //<-- naive implementation
    double scaledE1, scaledE2, E1, E2, E1minusE2, scaledE1deriv;
    EofH.evalDeriv(logh1, &scaledE1, g1);
    EofH.evalDeriv(logh2, &scaledE2);
    unscaleDeltaE(scaledE1, scaledE2, invPhi0, E1, E2, E1minusE2, scaledE1deriv);
    if(g1)
        *g1 = exp(logh1) / *g1 / scaledE1deriv;
    return E1minusE2;
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
    if(slope <= 0.05)
        utils::msg(utils::VL_WARNING, "Interpolator2d",
            "Potential is nearly singular at origin, numerical problems are likely to occur");
    const unsigned int sizeE = 50;
    const unsigned int sizeL = 40;

    // transformation of interval [0:1] onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at x=0 and x=1
    math::ScalingQui scaling(0, 1);

    // create a grid in energy
    std::vector<double> gridE(sizeE);
    for(unsigned int i=0; i<sizeE; i++)
        gridE[i] = (1 - math::unscale(scaling, 1. * i / (sizeE-1))) * Phi0;

    // create a grid in L/Lcirc(E)
    std::vector<double> gridL(sizeL);
    for(unsigned int i=0; i<sizeL; i++)
        gridL[i] = math::unscale(scaling, 1. * i / (sizeL-1));

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
        double R1overRc = math::findRoot(fnc, 0, 1, ACCURACY_ROOT);
        double R2overRc = iL==0 ? std::pow(1+slope/2, 1/slope) : math::findRoot(fnc, 1, 2, ACCURACY_ROOT);
        double dR1overRc_dZ = iL==0 ? sqrt(slope/(slope+2)) :
            slope*Z / ((slope+2) * (1-std::pow(R1overRc, slope)) * R1overRc);
        double dR2overRc_dZ = slope*Z / ((slope+2) * (1-std::pow(R2overRc, slope)) * R2overRc);
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
    std::ofstream strm("potentialInterpolator2d.log");
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
