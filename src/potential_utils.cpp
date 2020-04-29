#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
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
static const double ACCURACY_INTERP2= 1e-4;

/// size of the interpolation grid in the dimension corresponding to relative angular momentum
static const unsigned int GRID_SIZE_L = 40;

/// a number that is considered nearly infinity in log-scaled root-finders
static const double HUGE_NUMBER = 1e100;

/// safety factor to avoid roundoff errors in estimating the inner/outer asymptotic slopes
static const double ROUNDOFF_THRESHOLD = DBL_EPSILON / ROOT3_DBL_EPSILON;  // eps^(2/3) ~ 4e-11

/// minimum relative difference between two adjacent values of potential (to reduce roundoff errors)
static const double MIN_REL_DIFFERENCE = ROUNDOFF_THRESHOLD;

/// maximum value for L/Lcirc above which approximate the peri/apocenter radii analytically
static const double LREL_NEARLY_CIRCULAR = 0.999999;

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
            *val = Phi - E;
            if(!isFinite(*val)) {  // take special measures
                if(E==-INFINITY)
                    *val = Phi==E ? 0 : +1.0;
                else if(Phi==-INFINITY)
                    *val = -1.0;  // safely negative value
                else if(R>=HUGE_NUMBER)
                    *val = +1.0;
            }
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
            double v = 0.5 * R * dPhidR;
            *val = Phi - E + (isFinite(v) ? v : 0);
            if(!isFinite(*val)) {  // special cases
                if(E==-INFINITY)
                    *val = Phi==E ? 0 : +1.0;
                else if(Phi==-INFINITY)
                    *val = -1.0;  // safely negative value
                else if(R>=HUGE_NUMBER)
                    *val = +1.0;  // safely positive value
            }
        }
        if(deriv)
            *deriv = (1.5 * dPhidR + 0.5 * R * d2PhidR2) * R;
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
        double F = pow_3(R) * dPhidR; // Lz^2(R)
        if(val)
            *val = isFinite(F) ? F - L2 :
            // this may fail if R --> 0 or R --> infinity,
            // in these cases replace it with a finite number of a correct sign
            R < 1 ? -L2 : +L2;
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
            *val = (R>0 ? (E-Phi)*R*R : 0) - halfL2;
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
    return fabs(dR) < 0.25*R ? R+dR : R;  // precaution to avoid unpredictably large corrections
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
    const BasePotential& pot;
    const double invPhi0;
public:
    ScalePhi(const BasePotential& _pot) : pot(_pot), invPhi0(1/pot.value(coord::PosCyl(0,0,0))) {}
    virtual void evalDeriv(const double logr, double* val, double* der, double* der2) const {
        double r=exp(logr), Phi;
        coord::GradCyl dPhi;
        coord::HessCyl d2Phi;
        pot.eval(coord::PosCyl(r,0,0), &Phi, &dPhi, &d2Phi);
        double expE  = invPhi0 - 1/Phi;
        double nu2Om = d2Phi.dz2 / dPhi.dR * r;  // ratio of nu^2/Omega^2
        double dnu2Om=0, d2nu2Om=0;  // 1st and 2nd derivatives of the above ratio w.r.t. log(r)
        if(der || der2) {
            // compute these derivatives by finite differences
            coord::GradCyl dPhiP, dPhiM;
            coord::HessCyl d2PhiP,d2PhiM;
            const double delta = 10*ROOT3_DBL_EPSILON;
            double rP = exp(logr+delta), rM = exp(logr-delta);
            pot.eval(coord::PosCyl(rP,0,0), NULL, &dPhiP, &d2PhiP);
            pot.eval(coord::PosCyl(rM,0,0), NULL, &dPhiM, &d2PhiM);
            double nu2OmP = d2PhiP.dz2 / dPhiP.dR * rP, nu2OmM = d2PhiM.dz2 / dPhiM.dR * rM;
            dnu2Om = (nu2OmP - nu2OmM) / (2*delta);
            d2nu2Om= (nu2OmP + nu2OmM - 2*nu2Om) / pow_2(delta);
            if(fabs(d2nu2Om) < delta)
                d2nu2Om = 0;
        }
        if(val)
            *val = log(expE) + 2 * log(-Phi) + 0.5*nu2Om;
        if(der)
            *der = dPhi.dR * r / (Phi * Phi * expE) + 2 * dPhi.dR/Phi*r + 0.5*dnu2Om;
        if(der2) {
            if(invPhi0!=0 && expE < -invPhi0 * MIN_REL_DIFFERENCE)
                // in case of a finite potential at r=0,
                // we avoid approaching too close to 0 to avoid roundoff errors in Phi
                *der2 = 0;
            else
                *der2 = pow_2(r/Phi) / expE * (dPhi.dR * (1/r - dPhi.dR/Phi * (2 + 1/Phi/expE)) + d2Phi.dR2)
                + 2 * r/Phi * (d2Phi.dR2*r + dPhi.dR*(1-dPhi.dR/Phi*r)) + 0.5*d2nu2Om;
        }
    }
    virtual unsigned int numDerivs() const { return 2; }
};

}  // internal namespace


std::vector<double> createInterpolationGrid(const BasePotential& potential, double accuracy)
{
    // create a grid in log-radius with spacing depending on the local variation of the potential
    std::vector<double> grid = math::createInterpolationGrid(ScalePhi(potential), accuracy);

    // convert to grid in radius
    for(size_t i=0, size=grid.size(); i<size; i++)
        grid[i] = exp(grid[i]);

    // erase innermost grid nodes where the value of potential is too close to Phi(0) (within roundoff)
    double Phi0 = potential.value(coord::PosCyl(0,0,0));
    while(grid.size() > 2  &&  potential.value(coord::PosCyl(grid[0],0,0)) < Phi0 * (1-MIN_REL_DIFFERENCE))
        grid.erase(grid.begin());

    return grid;
}

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
    if(fabs(L) == INFINITY)
        return INFINITY;
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
    // no attempt to check if the expressions under sqrt are non-negative - 
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

void findPlanarOrbitExtent(const BasePotential& potential, double E, double L, double& R1, double& R2)
{
    if(!isAxisymmetric(potential))
        throw std::invalid_argument("findPlanarOrbitExtent only works for axisymmetric potentials");
    double Phi0, coef, slope = innerSlope(PotentialWrapper(potential), &Phi0, &coef);
    
    if(slope>0  &&  E >= Phi0  &&  E < Phi0 * (1-MIN_REL_DIFFERENCE)) {
        // accurate treatment at origin to avoid roundoff errors when Phi -> Phi(r=0),
        // assuming a power-law asymptotic behavior of potential at r->0
        double Rcirc = slope==0 ?  exp((E-Phi0) / coef - 0.5)  :
            std::pow((E-Phi0) / (coef * (1+0.5*slope)), 1/slope);
        if(!isFinite(Rcirc)) {
            R1 = R2 = NAN;
            return;
        }
        if(Rcirc == 0) {  // energy exactly equals the potential at origin
            R1 = R2 = 0;
            return;
        }
        double Lcirc = Rcirc * (slope==0 ? sqrt(coef) : sqrt(coef*slope) * std::pow(Rcirc, 0.5*slope));
        if(L >= Lcirc) {
            R1 = R2 = Rcirc;
        } else {
            RPeriApoRootFinderPowerLaw fnc(slope, pow_2(L / Lcirc));
            R1 = Rcirc * fmin(1., math::findRoot(fnc, 0, 1, ACCURACY_ROOT));
            R2 = Rcirc * fmax(1., math::findRoot(fnc, 1, 2, ACCURACY_ROOT));
        }
    } else {  // normal scenario when we don't suffer from roundoff errors 
        double Rcirc = R_circ(potential, E);
        if(!isFinite(Rcirc)) {
            R1 = R2 = NAN;
            return;
        }
        if(Rcirc == 0) {  // energy exactly equals the potential at origin
            R1 = R2 = 0;
            return;
        }
        double Phi;
        coord::GradCyl grad;
        coord::HessCyl hess;
        potential.eval(coord::PosCyl(Rcirc, 0, 0), &Phi, &grad, &hess);
        double Lcirc = Rcirc * sqrt(Rcirc * grad.dR);
        if(L >= Lcirc || (!isFinite(Lcirc) && Rcirc <= 1./HUGE_NUMBER))
        {
            // assume an exactly circular orbit (to within roundoff error),
            // i.e., don't panic if the input E and L were incompatible
            R1 = R2 = Rcirc;
        } else if(L > Lcirc * LREL_NEARLY_CIRCULAR ||
            (E-Phi)*pow_2(Rcirc) <= 0.5*L*L /*in this case the root-finder would fail due to roundoff*/)
        {   // asymptotic expressions for nearly circular orbits, when the ordinary method is inefficient
            double offset = sqrt( (1 - pow_2(L/Lcirc)) * grad.dR / (3 * grad.dR + Rcirc * hess.dR2) );
            R1 = Rcirc * (1-offset);
            R2 = Rcirc * (1+offset);
            // root polishing to improve the accuracy of peri/apocenter radii determination
            R1 = fmin(Rcirc, refineRoot(PotentialWrapper(potential), R1, E, L));
            R2 = fmax(Rcirc, refineRoot(PotentialWrapper(potential), R2, E, L));
        } else {
            // normal case
            RPeriApoRootFinder fnc(potential, E, L);
            R1 = math::findRoot(fnc, 0, Rcirc, ACCURACY_ROOT);
            R2 = math::findRoot(fnc, Rcirc, 3*Rcirc, ACCURACY_ROOT);
            // for a reasonable potential, 2*Rcirc is actually an upper limit,
            // but in case of trouble, repeat with a safely larger value (+extra cost of computing Rmax)
            if(!isFinite(R2))
                R2 = math::findRoot(fnc, Rcirc, (1+ACCURACY_ROOT) * R_max(potential, E), ACCURACY_ROOT);
        }
    }
}

// -------- Same tasks implemented as an interpolation interface -------- //

Interpolator::Interpolator(const BasePotential& potential) :
    invPhi0(1./potential.value(coord::PosCyl(0,0,0)))
{
    if(!isZRotSymmetric(potential))
        throw std::invalid_argument("Interpolator: can only work with axisymmetric potentials");
    double Phiinf = potential.value(coord::PosCyl(INFINITY,1.,1.));
    // not every potential returns a valid value at infinity, but if it does, make sure that it's zero
    if(Phiinf==Phiinf && Phiinf!=0)
        throw std::runtime_error("Interpolator: can only work with potentials "
            "that tend to zero as r->infinity");   // otherwise assume Phiinf==0
    // well-behaved potential must be -INFINITY <= Phi0 < 0
    if(invPhi0 > 0 || !isFinite(invPhi0))
        throw std::runtime_error("Interpolator: potential must be negative at r=0");

    std::vector<double> gridR = createInterpolationGrid(potential, ACCURACY_INTERP);
    unsigned int gridsize = gridR.size();
    std::vector<double>   // various arrays:
    gridLogR(gridsize),   // ln(r)
    gridPhi(gridsize),    // scaled Phi(r)
    gridE(gridsize),      // scaled Ecirc(Rcirc) where Rcirc=r
    gridL(gridsize),      // log(Lcirc)
    gridNu(gridsize),     // ratio of squared epicyclic frequencies nu^2/Omega^2
    gridPhider(gridsize), // d(scaled Phi)/ d(log r)
    gridRder(gridsize),   // d(log Rcirc) / d(log Lcirc)
    gridLder(gridsize);   // d(log Lcirc) / d(scaled Ecirc)

    std::ofstream strm;   // debugging
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        strm.open("PotentialInterpolator.log");
        strm << "#R      \tPhi(R)  \tdPhi/dR \td2Phi/dR2\td2Phi/dz2\tEcirc   \tLcirc\n";
    }

    for(unsigned int i=0; i<gridsize; i++) {
        double R = gridR[i];
        gridLogR[i] = log(R);
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

        // estimate the outer asymptotic behaviour
        if(i==gridsize-1) {
            double num1 = 2*grad.dR, num2 = -R*hess.dR2, den1 = grad.dR, den2 = -Phival/R;
            slopeOut    = (num1 - num2) / (den1 - den2);
            bool roundoff =    // check if the value of slope is dominated by roundoff errors
                fabs(num1-num2) < fmax(fabs(num1), fabs(num2)) * ROUNDOFF_THRESHOLD ||
                fabs(den1-den2) < fmax(fabs(den1), fabs(den2)) * ROUNDOFF_THRESHOLD;
            if(roundoff || slopeOut>=0) {    // not successful - use the total mass only
                slopeOut= -1;
                coefOut = 0;
                massOut = -Phival * R;
            } else {
                if(fabs(slopeOut+1) < ROUNDOFF_THRESHOLD)
                    slopeOut = -1;   // value for a logarithmically-growing M(r), as in NFW
                coefOut = (Phival + R*grad.dR) * std::pow(R, -slopeOut);
                massOut = -R*Phival + coefOut *
                    (slopeOut==-1 ? log(R) : (std::pow(R, slopeOut+1) - 1) / (slopeOut+1));
            }
        }
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
    if(logR > PhiofR.xvalues().back() && coefOut!=0)
    {  // special care for extrapolation at large r
        double Rs = exp(logR * slopeOut);   // R^slopeOut
        double Phi= (-massOut + (slopeOut==-1 ? logR : (R*Rs-1) / (slopeOut+1)) * coefOut ) / R;
        if(val)
            *val = Phi;
        if(deriv)
            *deriv = (-Phi + coefOut * Rs) / R;
        if(deriv2)
            *deriv2 = (2 * Phi + coefOut * Rs * (slopeOut-2) ) / pow_2(R);
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
    double scaledE, dEdscaledE, logR;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    RofPhi.evalDeriv(scaledE, &logR, deriv);
    double R = exp(logR);
    if(logR > PhiofR.xvalues().back()) {
        // extra correction step at large r because of non-trivial extrapolation of potential
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(R, &Phi, &dPhidR, &d2PhidR2);
        R -= math::clip(   // cautionary measure to avoid too large corrections
            (Phi-E) / (dPhidR - 0.5 * (Phi-E) * d2PhidR2 / dPhidR),   // Halley correction
            -0.25*R, 0.25*R);
    }
    if(deriv)
        *deriv *= R / dEdscaledE;
    return R;
}

double Interpolator::L_circ(const double E, double* deriv) const
{
    if(!(E>=1./invPhi0 && E<=0)) {
        if(deriv)
            *deriv = NAN;
        return NAN;
    }
    double scaledE, dEdscaledE, logL, logLder;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &logL, deriv!=NULL ? &logLder : NULL);
    double Lcirc = exp(logL);
    if(scaledE > LofE.xvalues().back()) {
        // extra correction step at large radii
        double Rcirc = exp(RofL(logL));  // first get an approximation for Rcirc
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        double Ecirc = Phi + 0.5 * Rcirc * dPhidR;
        double denom = 1 - 0.5 * (Ecirc-E) * (Rcirc * d2PhidR2 - dPhidR) /
            ((Rcirc * d2PhidR2 + 3 * dPhidR) * Rcirc * dPhidR);
        Lcirc = math::clip(   // cautionary measure to avoid too large corrections
            sqrt(Rcirc * dPhidR) * (Rcirc - (Ecirc-E) / (dPhidR * denom)),   // Halley correction
            0.75*Lcirc, 1.25*Lcirc);
    }
    if(deriv)
        *deriv = logLder / dEdscaledE * Lcirc;
    return Lcirc;
}

double Interpolator::R_from_Lz(const double Lz, double* deriv) const
{
    double logL = log(fabs(Lz)), logR, logRder;
    RofL.evalDeriv(logL, &logR, deriv!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(logL > RofL.xvalues().back()) {
        // extra correction step at large radii
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        Rcirc -= math::clip(   // cautionary measure to avoid too large corrections
            (Rcirc * dPhidR - pow_2(Lz/Rcirc)) / (3 * dPhidR + Rcirc * d2PhidR2),   // Newton correction
            -0.25*Rcirc, 0.25*Rcirc);
        // even though this is Newton (1st order), not Halley (2nd order) correction,
        // it seems to be fairly accurate
    }
    if(deriv)
        *deriv = logRder * Rcirc / Lz;
    return Rcirc;
}

double Interpolator::R_circ(const double E, double* deriv) const
{
    if(!(E>=1./invPhi0 && E<=0)) {
        if(deriv)
            *deriv = NAN;
        return NAN;
    }
    double scaledE, dEdscaledE, logL, logLder, logR, logRder;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    LofE.evalDeriv(scaledE, &logL, deriv!=NULL ? &logLder : NULL);
    RofL.evalDeriv(logL,    &logR, deriv!=NULL ? &logRder : NULL);
    double Rcirc = exp(logR);
    if(logL > RofL.xvalues().back()) {
        // extra correction step at large radii
        double Phi, dPhidR, d2PhidR2;
        evalDeriv(Rcirc, &Phi, &dPhidR, &d2PhidR2);
        Rcirc -= math::clip(   // cautionary measure to avoid too large corrections
            ( 2*(Phi-E) + Rcirc * dPhidR ) / (3 * dPhidR + Rcirc * d2PhidR2),   // Newton correction
            -0.25*Rcirc, 0.25*Rcirc);
        // this is only 1st order correction, and could be improved by another iteration,
        // but we leave it as it is
    }
    if(deriv)
        *deriv = logLder * logRder / dEdscaledE * Rcirc;
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
    Interpolator(potential),
    invPhi0(1./potential.value(coord::PosCyl(0,0,0)))  // -infinity <= Phi(0) < 0
{
    std::vector<double> gridR = createInterpolationGrid(potential, ACCURACY_INTERP2);

    // interpolation grid in scaled variables: X = scaledE = log(1/Phi(0)-1/E), Y = L / Lcirc(E)
    const int sizeE = gridR.size();
    const int sizeL = GRID_SIZE_L;
    std::vector<double> gridX(sizeE), gridY(sizeL);

    // create a non-uniform grid in Y = L/Lcirc(E), using a transformation of interval [0:1]
    // onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at Y=0 and Y=1
    math::ScalingQui scaling(0, 1);
    for(int i=0; i<sizeL; i++)
        gridY[i] = math::unscale(scaling, 1. * i / (sizeL-1));

    // 2d grids for scaled peri/apocenter radii W1, W2 and their derivatives in {X,Y}:
    // W1 = (R1 / Rc - 1)^2, same for W2, where
    // R1 and R2 are the peri/apocenter radii, and Rc is the radius of a circular orbit;
    // W1 and W2 are exactly zero when L=Lcirc (equivalently Y=1), and vary linearly as Y -> 1
    math::Matrix<double> gridW1  (sizeE, sizeL), gridW2  (sizeE, sizeL);
    math::Matrix<double> gridW1dX(sizeE, sizeL), gridW1dY(sizeE, sizeL);
    math::Matrix<double> gridW2dX(sizeE, sizeL), gridW2dY(sizeE, sizeL);

    std::string errorMessage;  // store the error text in case of an exception in the openmp block
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iE=0; iE<sizeE; iE++) {
        try{
            double Rc = gridR[iE];
            double Phi;
            coord::GradCyl grad;
            coord::HessCyl hess;
            potential.eval(coord::PosCyl(Rc, 0, 0), &Phi, &grad, &hess);
            double E  = Phi + 0.5*Rc*grad.dR;   // energy of a circular orbit at this radius
            double Lc = Rc * sqrt(Rc*grad.dR);  // angular momentum of a circular orbit
            double dEdX;                        // dE / d scaledE
            scaleE(E, invPhi0, /*output*/gridX[iE], dEdX);
            double Om2kap2= grad.dR / (3*grad.dR + Rc*hess.dR2);  // ratio of epi.freqs (Omega / kappa)^2
            double dRcdE  = 2/grad.dR * Om2kap2;
            double dLcdE  = Rc*Rc/Lc;
            for(int iL=0; iL<sizeL-1; iL++) {
                double L = Lc * gridY[iL];
                double R1, R2, Phi;
                if(iL==0) {  // exact values for a radial orbit
                    R1=0;
                    R2=potential::R_max(potential, E);
                } else
                    potential::findPlanarOrbitExtent(potential, E, L, R1, R2);
                gridW1(iE, iL) = pow_2(R1 / Rc - 1);
                gridW2(iE, iL) = pow_2(R2 / Rc - 1);
                // compute derivatives of Rperi/apo w.r.t. E and L/Lcirc
                potential.eval(coord::PosCyl(R1,0,0), &Phi, &grad);
                if(R1==0) grad.dR=0;   // it won't be used anyway, but prevents a possible NaN
                double dW1dE = (1 / (E-Phi) - 2 * dLcdE / Lc) / (grad.dR / (E-Phi) - 2 / R1);
                double dW1dY = -Lc * M_SQRT2 / (grad.dR * R1 / sqrt(E-Phi) - 2*sqrt(E-Phi));
                potential.eval(coord::PosCyl(R2,0,0), &Phi, &grad);
                double dW2dE = (1 - 2*(E-Phi) * dLcdE / Lc) / (grad.dR - 2*(E-Phi) / R2);
                double dW2dY = -Lc * L / (grad.dR * pow_2(R2) - 2*(E-Phi) * R2);
                gridW1dX(iE, iL) = 2*(R1-Rc) / pow_2(Rc) * (dW1dE - R1/Rc * dRcdE) * dEdX;
                gridW1dY(iE, iL) = 2*(R1-Rc) / pow_2(Rc) *  dW1dY;
                gridW2dX(iE, iL) = 2*(R2-Rc) / pow_2(Rc) * (dW2dE - R2/Rc * dRcdE) * dEdX;
                gridW2dY(iE, iL) = 2*(R2-Rc) / pow_2(Rc) *  dW2dY;
            }
            // limiting values for a nearly circular orbit:
            // R{1,2} = Rcirc * (1 +- Omega/kappa * ecc),
            // where ecc = sqrt(1 - (L/Lcirc)^2) = sqrt(1-Y^2)
            gridW1  (iE, sizeL-1) = gridW2  (iE, sizeL-1) = 0;
            gridW1dX(iE, sizeL-1) = gridW2dX(iE, sizeL-1) = 0;
            gridW1dY(iE, sizeL-1) = gridW2dY(iE, sizeL-1) = -2*Om2kap2;
        }
        catch(std::exception& e) {
            errorMessage = e.what();
        }
    }

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {   // debugging output
        std::ofstream strm("PotentialInterpolator2d.log");
        strm << "# X=scaledE    \tY=L/Lcirc      \t"
            "W1=scaledRperi \tdW1/dX         \tdW1/dY         \t"
            "W2=scaledRapo  \tdW2/dX         \tdW2/dY         \n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                strm <<
                utils::pp(gridX[iE], 15) + "\t" +
                utils::pp(gridY[iL], 15) + "\t" +
                utils::pp(gridW1  (iE, iL), 15) + "\t" +
                utils::pp(gridW1dX(iE, iL), 15) + "\t" +
                utils::pp(gridW1dY(iE, iL), 15) + "\t" +
                utils::pp(gridW2  (iE, iL), 15) + "\t" +
                utils::pp(gridW2dX(iE, iL), 15) + "\t" +
                utils::pp(gridW2dY(iE, iL), 15) + "\n";
            }
            strm << "\n";
        }
    }

    if(!errorMessage.empty())
        throw std::runtime_error("Interpolator2d: "+errorMessage);

    // create 2d interpolators
    intR1 = math::QuinticSpline2d(gridX, gridY, gridW1, gridW1dX, gridW1dY);
    intR2 = math::QuinticSpline2d(gridX, gridY, gridW2, gridW2dX, gridW2dY);
}

void Interpolator2d::findPlanarOrbitExtent(double E, double L,
    double &R1, double &R2) const
{
    double Lc   = L_circ(E);
    double Rc   = R_from_Lz(Lc);
    double Lrel = Lc>0 ? math::clip(fabs(L/Lc), 0., 1.) : 0;
    double scaledE, dEdscaledE;
    scaleE(E, invPhi0, scaledE, dEdscaledE);
    scaledE = math::clip(scaledE, intR1.xmin(), intR1.xmax());
    R1 = (1 - sqrt(intR1.value(scaledE, Lrel))) * Rc;
    R2 = (1 + sqrt(intR2.value(scaledE, Lrel))) * Rc;
    // one iteration of root polishing
    R1 = fmin(refineRoot(*this, R1, E, L), Rc);
    R2 = fmax(refineRoot(*this, R2, E, L), Rc);
}


//---- Correspondence between h and E ----//

PhaseVolume::PhaseVolume(const math::IFunction& pot)
{
    double Phi0 = pot(0);
    if(!(Phi0<0))
        throw std::invalid_argument("PhaseVolume: invalid value of Phi(r=0)");
    invPhi0 = 1/Phi0;

    // create grid in log(r)
    std::vector<double> gridr = math::createInterpolationGrid(
        ScalePhi(FunctionToPotentialWrapper(pot)), ACCURACY_INTERP);
    for(size_t i=0; i<gridr.size(); i++)
        gridr[i] = exp(gridr[i]);  // convert to grid in r
    std::vector<double> gridE;
    gridE.reserve(gridr.size());

    // compute the potential at each node of the radial grid, throwing away nodes that are
    // too closely spaced, such that the difference between adjacent potential values suffers from
    // roundoff/cancellation errors due to finite precision of floating-point arithmetic
    double prevPhi = Phi0;
    for(size_t i=0; i<gridr.size(); ) {
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
    size_t gridsize = gridr.size();
    if(gridsize == 0)
        throw std::runtime_error("PhaseVolume: cannot construct a suitable grid in radius");

    std::vector<double> gridH(gridsize), gridG(gridsize);
    const double *glnodes1 = math::GLPOINTS[GLORDER1], *glweights1 = math::GLWEIGHTS[GLORDER1];
    const double *glnodes2 = math::GLPOINTS[GLORDER2], *glweights2 = math::GLWEIGHTS[GLORDER2];

    // loop through all grid segments, and in each segment add the contribution to integrals
    // in all other segments leftmost of the current one (thus the complexity is Ngrid^2,
    // but the number of potential evaluations is only Ngrid * GLORDER).
    for(size_t i=0; i<gridsize; i++) {
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
            for(size_t j=i; j<gridsize; j++) {
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
        strm.open("PhaseVolume.log");
        strm << "#r      \tE       \th       \tg=dh/dE\n";
    }

    // convert h, g and E to scaled coordinates
    for(size_t i=0; i<gridsize; i++) {
        double E = gridE[i], H = gridH[i], G = gridG[i], scaledE, dEdscaledE;
        scaleE(E, invPhi0, scaledE, dEdscaledE);
        gridE[i] = scaledE;
        gridH[i] = log(H) + log(16*M_PI*M_PI/3*2*M_SQRT2);
        gridG[i] = G / H * dEdscaledE;
        // debugging printout
        if(utils::verbosityLevel >= utils::VL_VERBOSE) {
            strm <<
                utils::pp(gridr[i], 15) + '\t' +
                utils::pp(E, 15) + '\t' +
                utils::pp(H, 15) + '\t' +
                utils::pp(G, 15) + '\n' << std::flush;
        }
    }

    HofE = math::QuinticSpline(gridE, gridH, gridG);
    // inverse relation between E and H - the derivative is reciprocal
    for(size_t i=0; i<gridG.size(); i++)
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

}  // namespace potential
