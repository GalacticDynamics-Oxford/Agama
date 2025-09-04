#include "actions_spherical.h"
#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>

namespace actions{

namespace {

/// use interpolation for computing E(Jr,L) - if not defined, use root-finding (slower)
#define INTERPOLATE_ENERGY

/// required tolerance on the value of Jr(E) in the root-finder
static const double ACCURACY_JR = 1e-6;

/// accuracy parameter determining the radial spacing of the 2d interpolation grid for Jr
static const double ACCURACY_INTERP2 = 1e-4;

/// size of the interpolation grid in the dimension corresponding to relative angular momentum
static const unsigned int GRID_SIZE_L = 25;

/// minimum order of Gauss-Legendre quadrature for actions, frequencies and angles
static const unsigned int INTEGR_ORDER = 8;

/** order of Gauss-Legendre quadrature for actions, frequencies and angles:
    use a higher order for more eccentric orbits, as indicated by the ratio
    of pericenter to apocenter radii (R1/R2).
    The heuristic recipe below keeps the accuracy at a nearly constant level down to R1/R2=0.001 */
inline unsigned int integrOrder(double R1overR2) {
    if(R1overR2==0)
        return math::MAX_GL_ORDER;
    return std::min<int>(math::MAX_GL_ORDER, INTEGR_ORDER - 4/M_LN2 * log(R1overR2));
}


/// return scaledE and optionally dE/d(scaledE) as functions of E and invPhi0 = 1/Phi(0)
inline double scaleE(const double E, const double invPhi0, /*output*/ double* dEdscaledE=NULL)
{
    double expX = invPhi0 - 1/E;
    if(dEdscaledE)
        *dEdscaledE = E * E * expX;
    return log(expX);
}

/// return E and optionally dE/d(scaledE) as a function of scaledE and invPhi0
inline double unscaleE(const double scaledE, const double invPhi0, /*output*/ double* dEdscaledE=NULL)
{
    double expX = exp(scaledE);
    double E = 1 / (invPhi0 - expX);
    if(dEdscaledE)
        *dEdscaledE = E * E * expX;
    return E;
}

/// compute the radial action from a 2d interpolation table as in ActionFinderSpherical
double computeJr(double E, double scaledE, double dEdscaledE, double L, 
    const potential::Interpolator& pot, const math::BaseInterpolator2d& intJr,
    /*optional output*/double *Omegar, double *Omegaz)
{
    bool needDeriv = Omegar!=NULL || Omegaz!=NULL;
    double dLcdE, Lc = pot.L_circ(E, needDeriv? &dLcdE : NULL);
    if(!isFinite(Lc)) {  // E>=0 or E<Phi(0)
        if(Omegar) *Omegar = NAN;
        if(Omegaz) *Omegaz = NAN;
        return NAN;
    }

    double X = math::clip(scaledE, intJr.xmin(), intJr.xmax());
    double Y = math::clip(fabs(L/Lc), 0., 1.);

    // obtain the value of scaled Jr as a function of scaled E and L, and unscale it
    double val, derX, derY;
    intJr.evalDeriv(X, Y, &val, needDeriv? &derX : NULL, needDeriv? &derY : NULL);
    if(needDeriv) {
        double dJrdL = derY * (1-Y) - val;
        double dJrdE = derX * (1-Y) * Lc / dEdscaledE - (derY * (1-Y) * Y - val) * dLcdE;
        if(Omegar)
            *Omegar  = 1 / dJrdE;
        if(Omegaz)
            *Omegaz  = -dJrdL / dJrdE;
    }

    // the interpolated value might turn out to be slightly negative due to approximation inaccuracies,
    // and is then replaced by zero
    return fmax(0, val * Lc * (1-Y));
}


/// different regimes for calculation of various integrals involving radial velocity
typedef enum { MODE_JR, MODE_OMEGAR, MODE_OMEGAZ } Operation;

/// Integrand for computing the radial action, frequencies and angles in a spherical potential
/// or a potential interpolator, both accessed through a uniform IFunction interface
template<Operation mode>
class Integrand: public math::IFunctionNoDeriv {
    const math::IFunction& potential;
    const double E, L; ///< integrals of motion (energy and total angular momentum)
    const double R1;   ///< lower limit of integration, used to subtract the singular term in MODE_OMEGAZ
public:
    Integrand(const math::IFunction& p, double _E, double _L, double _R1) :
        potential(p), E(_E), L(_L), R1(_R1) {};
    virtual double value(const double r) const {
        // r cannot be zero because the integrand is never evaluated at endpoints of the interval
        double Phi = potential.value(r);
        double vr2 = 2*(E-Phi) - pow_2(L/r);
        if(vr2<=0 || !isFinite(vr2) || r==R1) return 0;
        double vr  = sqrt(vr2);
        if(mode==MODE_JR)     return vr;
        if(mode==MODE_OMEGAR) return 1/vr;
        if(mode==MODE_OMEGAZ) return L/(r*r*vr) - 1/(sqrt(pow_2(r/R1)-1)*r);
        // in the latter case, a singular term with analytic antiderivative
        // is subtracted from the integrand
        assert(!"Invalid mode in action integrand");
        return 0;
    }
};

/** compute the integral involving radius and radial velocity on the interval from peri- to apocenter,
    using scaling transformation to remove singularities at the endpoints.
    \param[in] poten  is the original or interpolated potential, accessed through IFunction interface;
    \param[in] E, L   are the integrals of motion (energy and ang.mom.);
    \param[in] R1, R2 are the peri/apocenter radii corresponding to these E and L (computed elsewhere);
    \param[in] R      is the upper limit of integration (the lower limit is always R1), NAN means R2;
    \tparam mode      determines the quantity to compute:
    MODE_JR     ->    \int_{R1}^{R}  v_r(E,L,r) dr,
    MODE_OMEGAR ->    \int 1 / v_r dr,
    MODE_OMEGAZ ->    \int L / (v_r * r^2) dr,  in the latter case the integrand is split into
    two parts, one of them can be integrated analytically, and the other does not diverge as L->0,r->0.
    \return the value of integral.
*/
template<Operation mode>
inline double integr(const math::IFunction& poten,
    double E, double L, double R1, double R2, double R=NAN)
{
    if(R1==R2)
        return 0;
    if(R!=R)
        R=R2;                   // default upper limit for integration
    R = math::clip(R, R1, R2);  // roundoff errors might cause R to be outside the allowed interval
    Integrand<mode> integrand(poten, E, L, R1);
    math::ScalingCub scaling(R1, R2);
    double add = 0;  // part of the integral that is computed analytically
    if(mode==MODE_OMEGAZ) {
        if(R1==0) {
            // "add" is (half) the change in angle phi for a purely radial orbit
            // (Pi/2 for potentials that are regular at origin,
            // or larger for singular potentials, up to Pi in the Kepler case)
            double slope = potential::innerSlope(poten);
            add = M_PI / (2 + fmin(slope, 0));
        } else
            add = acos(R1/R);
    }
    return integrateGL(math::ScaledIntegrand<math::ScalingCub>(scaling, integrand),
        0, math::scale(scaling, R), integrOrder(R1/R2)) + add;
}

/// helper function to find the upper limit of integral for the radial phase,
/// such that its value equals the target
class RadiusFromPhaseFinder: public math::IFunction {
    const Integrand<MODE_OMEGAR> integrand;
    const math::ScaledIntegrand<math::ScalingCub> transf;
    const double target;  // target value of the integral
public:
    RadiusFromPhaseFinder(const math::IFunction &poten,
        double E, double L, double R1, double R2, double _target)
    :
        integrand(poten, E, L, R1),
        transf(math::ScalingCub(R1, R2), integrand),
        target(_target)
    {}
    virtual void evalDeriv(const double x, double *val, double *der, double*) const {
        if(val)
            *val = math::integrateGL(transf, 0, x, INTEGR_ORDER) - target;
        if(der)
            *der = transf(x);
    }
    virtual unsigned int numDerivs() const { return 1; }
    double findRoot() const {
        double rscaled = math::findRoot(*this, 0, 1, ACCURACY_JR);
        if(!isFinite(rscaled)) {
            // can happen close to apocenter, because the phase found by numerical integration
            // may not exactly correspond to the (inverse) frequency computed by another method
            rscaled = 1;
        }
        return math::unscale(transf.scaling, rscaled);
    }
};

/// helper class to find the energy corresponding to the given radial action
class HamiltonianFinderFnc: public math::IFunctionNoDeriv {
    /// the instance of potential
    const potential::BasePotential& pot;
    /// (inverse) value of potential at origin, used for scaling transformations
    const double invPhi0;
    /// the values of actions (Jr is assumed to be positive)
    const double Jr, L;
public:
    HamiltonianFinderFnc(const potential::BasePotential& _pot, double _Jr, double _L) :
        pot(_pot), invPhi0(1. / pot.value(coord::PosCyl(0,0,0))), Jr(_Jr), L(_L)  {};

    /// report the difference between target Jr and the one computed at the given (scaled) energy
    virtual double value(const double scaledE) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(scaledE==-INFINITY)
            return -Jr;   // left boundary of the interval is at a circular orbit, for which Jr=0
        else if(scaledE==INFINITY)
            return +Jr;   // right boundary is at infinity, return a safely positive value for E=0
        double R1, R2, E=unscaleE(scaledE, invPhi0);
        findPlanarOrbitExtent(pot, E, L, R1, R2);
        return integr<MODE_JR>(potential::Sphericalized<potential::BasePotential>(pot),
            E, L, R1, R2) / M_PI - Jr;
    }
};

/// same operation using the interpolated radial action finder
class HamiltonianFinderFncInterpolated: public math::IFunction {
    /// the instance of potential and peri/apocenter interpolator
    const potential::Interpolator2d& pot;
    /// the values of actions (Jr is assumed to be positive), and the inverse potential at origin
    const double Jr, L, invPhi0;
    /// the instance of 2d action interpolator
    const math::BaseInterpolator2d& intJr;
public:
    HamiltonianFinderFncInterpolated(
        const potential::Interpolator2d& _pot, double _Jr, double _L, const double _invPhi0,
        const math::BaseInterpolator2d& _intJr) :
        pot(_pot), Jr(_Jr), L(_L), invPhi0(_invPhi0), intJr(_intJr)  {};

    /// report the difference between target Jr and the one computed at the given (scaled) energy
    virtual void evalDeriv(const double scaledE, double *val=0, double *der=0, double* =0) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(fabs(scaledE)==INFINITY) {
            if(val)
                *val = scaledE<0 ? -Jr : +Jr;
            if(der)
                *der = NAN;
            return;
        }
        // convert the values of E and L into the scaled variables used for interpolation
        double dEdscaledE, E = unscaleE(scaledE, invPhi0, /*output*/ &dEdscaledE);
        // compute the trial action for the given E and L, and Omega_r which is (dJr/dE)^-1
        double trialJr = computeJr(E, scaledE, dEdscaledE, L, pot, intJr, /*output: Omega_r*/der, NULL);
        if(val)
            *val = trialJr - Jr;
        if(der)
            *der = dEdscaledE / *der;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** Compute angles for the given point.
    This routine is shared between the standalone function `evalSpherical`
    and the member function `eval` of the interpolated action finder.
    \param[in]  point is the input point;
    \param[in]  potential is the original or interpolated potential;
    \param[in]  E is the total energy;
    \param[in]  L is the angular momentum;
    \param[in]  R1, R2 are peri/apocenter radii (computed elsewhere);
    \param[in]  Omegar, Omegaz are the corresponding frequencies (computed elsewhere);
    \returns    angle variables.
*/
Angles computeAngles(const coord::PosVelCyl& point,
    const math::IFunction &potential, const double E, const double L,
    const double R1, const double R2, const double Omegar, const double Omegaz)
{
    double r = sqrt(pow_2(point.R)+pow_2(point.z));
    double vtheta = (point.vR * point.z - point.vz * point.R) / r;
    // aux angles:  sin(psi) = cos(theta) / sin(i),  sin(chi) = cot(i) cot(theta)
    double psi = math::atan2(point.z * L,  -point.R * vtheta * r);
    double chi = math::atan2(point.z * point.vphi, -vtheta * r);
    Angles ang;
    ang.thetar = integr<MODE_OMEGAR>(potential, E, L, R1, R2, r) * Omegar;
    double thr = ang.thetar;
    double thz = integr<MODE_OMEGAZ>(potential, E, L, R1, R2, r);
    if(point.R * point.vR + point.z * point.vz < 0) {
        // v_r<0 - we're on the second half of radial period
        ang.thetar = 2*M_PI - ang.thetar;
        thz        = -thz;
        thr        = ang.thetar - 2*M_PI;
    }
    ang.thetaz   = math::wrapAngle(psi + thr * Omegaz / Omegar - thz);
    ang.thetaphi = math::wrapAngle(point.phi - chi + math::sign(point.vphi) * ang.thetaz);
    if(point.z==0 && point.vz==0)  // in this case Jz==0, and the value of theta_z is meaningless
        ang.thetaz = 0;
    return ang;
}

/** Compute the position/velocity from action/angles.
    This routine is shared between standalone function `mapSpherical`
    and the member function `map` of the interpolated action mapper.
    \param[in]  aa are the actions and angles;
    \param[in]  potential is the original or interpolated potential;
    \param[in]  E is the total energy corresponding to these values of actions,
    computed earlier by a different routine;
    \param[in]  L is the angular momentum (computed earlier from these values of actions);
    \param[in]  R1, R2 are peri/apocenter radii (computed elsewhere);
    \param[in]  Omegar, Omegaz are the corresponding frequencies (computed elsewhere);
    \returns    the point (position+velocity)
*/
coord::PosVelCyl mapPointFromActionAngles(const ActionAngles &aa,
    const math::IFunction &potential, const double E, const double L,
    const double R1, const double R2, const double Omegar, const double Omegaz)
{
    // find r from theta_r: radial phase theta_r ranging from 0 (peri) to pi (apocenter)
    // to 2pi (back to pericenter); v_r>=0 if 0<=theta_r<=pi, otherwise v_r<0.
    double thr = math::wrapAngle(aa.thetar);
    if(thr>M_PI)
        thr   -= 2*M_PI;  // follow the convention that -pi < thr <= pi
    double r   = RadiusFromPhaseFinder(potential, E, L, R1, R2, fabs(thr)/Omegar).findRoot();
    double vr  = math::sign(thr) * sqrt(fmax(0, 2 * (E - potential(r)) - (L>0 ? pow_2(L/r) : 0) ));
    // find other auxiliary angles
    double thz = math::sign(thr) * integr<MODE_OMEGAZ>(potential, E, L, R1, R2, r);
    double psi = aa.thetaz + thz - thr * Omegaz / Omegar;
    double sinpsi, cospsi;
    math::sincos(psi, sinpsi, cospsi);
    double chi      = aa.Jz != 0 ? math::atan2(fabs(aa.Jphi) * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    double R        = r * sintheta;
    double vtheta   = L * sini * cospsi / R;
    // finally, output position/velocity
    return coord::PosVelCyl(R,
        r * costheta,  // z
        math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * math::sign(aa.Jphi)),  // phi
        vr * sintheta - vtheta * costheta,  // vR
        vr * costheta + vtheta * sintheta,  // vz
        aa.Jphi != 0 ? aa.Jphi / R : 0);    // vphi
}


/// construct the interpolating spline for scaled radial action W = Jr / (Lcirc-L)
/// as a function of E and L/Lcirc
math::QuinticSpline2d createActionInterpolator(const potential::Interpolator2d& pot)
{
    const double invPhi0 = 1. / pot.value(0);
    std::vector<double> gridR = potential::createInterpolationGrid(
        potential::FunctionToPotentialWrapper(pot), ACCURACY_INTERP2);
    // extend the grid a little bit at large radii
    gridR.push_back( exp(2.5 * log(gridR[gridR.size()-1]) - 1.5 * log(gridR[gridR.size()-2])) );

    // interpolation grid in scaled variables: X = scaledE = log(1/Phi(0)-1/E), Y = L / Lcirc(E)
    const int sizeE = gridR.size();
    const int sizeL = GRID_SIZE_L;
    std::vector<double> gridX(sizeE), gridY(sizeL);

    // create a non-uniform grid in Y = L/Lcirc(E), using a transformation of interval [0:1]
    // onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at x=0 and x=1
    math::ScalingQui scaling(0, 1);
    for(int i=0; i<sizeL; i++)
        gridY[i] = math::unscale(scaling, i / (sizeL-1.));

    // value of W=Jr/(Lcirc-L) and its derivatives w.r.t. X=scaledE and Y=L/Lcirc
    math::Matrix<double> gridW(sizeE, sizeL), gridWdX(sizeE, sizeL), gridWdY(sizeE, sizeL);
    std::vector<double> gridWatY1(sizeE);  // last column of the W matrix (values at Y=1 and any X)

    std::string errorMessage;  // store the error text in case of an exception in the OpenMP block
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iE=0; iE<sizeE; iE++) {
        try{
            double Rc = gridR[iE];
            double Phi, dPhi, d2Phi;
            pot.evalDeriv(Rc, &Phi, &dPhi, &d2Phi);
            double E  = Phi + 0.5 * Rc * dPhi;   // energy of a circular orbit at this radius
            double Lc = Rc  * sqrt( Rc * dPhi);  // angular momentum of a circular orbit
            double dEdX;                         // dE / d scaledE
            gridX[iE] = scaleE(E, invPhi0, /*output*/ &dEdX);
            double dLcdE = Rc*Rc/Lc;
            for(int iL=0; iL<sizeL-1; iL++) {
                double L = Lc * gridY[iL];
                double R1, R2;
                pot.findPlanarOrbitExtent(E, L, R1, R2);
                double Jr    = integr<MODE_JR>    (pot, E, L, R1, R2) / M_PI;
                double dJrdE = integr<MODE_OMEGAR>(pot, E, L, R1, R2) / M_PI;
                double dJrdL =-integr<MODE_OMEGAZ>(pot, E, L, R1, R2) / M_PI;
                gridW  (iE, iL) = Jr / (Lc - L);
                gridWdX(iE, iL) = (dJrdE + (L * dJrdL - Jr) * dLcdE / Lc) / (Lc - L) * dEdX;
                gridWdY(iE, iL) = (dJrdL + Jr / (Lc - L)) / (1 - gridY[iL]);
            }
            // limiting value for a nearly circular orbit (Y=1): Jr / (Lcirc-L) = Omega/kappa
            gridW  (iE, sizeL-1) = gridWatY1[iE] = sqrt(dPhi / (d2Phi * Rc + 3 * dPhi));
            // derivative w.r.t. Y is obtained by quardatic interpolation of finite-differences,
            // using value at the boundary node, and value+deriv at the next-to-boundary node
            gridWdY(iE, sizeL-1) = -gridWdY(iE, sizeL-2) +
                2 * (gridW(iE, sizeL-1) - gridW(iE, sizeL-2)) / (gridY[sizeL-1] - gridY[sizeL-2]);
            // derivative w.r.t. X will be obtained from 1d spline after all W values are computed
        }
        catch(std::exception& e) {
            errorMessage = e.what();
        }
    }

    // derivative dW/dX at Y=1 is computed by constructing an auxiliary 1d spline for W(X)|Y=1
    // and differentiating it
    try{
        math::CubicSpline intWatY1(gridX, gridWatY1);
        for(int iE=0; iE<sizeE; iE++)
            intWatY1.evalDeriv(gridX[iE], NULL, &gridWdX(iE, sizeL-1));
    }
    catch(std::exception& e) {
        errorMessage = e.what();
    }

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {   // debugging output
        std::ofstream strm("ActionFinderSpherical.log");
        strm << "# X=scaledE    \tY=L/Lcirc      \tW=Jr/(Lcirc-L) \tdW/dX          \tdW/dY          \n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                strm << 
                utils::pp(gridX  [iE],     15) + "\t" +
                utils::pp(gridY  [iL],     15) + "\t" +
                utils::pp(gridW  (iE, iL), 15) + "\t" +
                utils::pp(gridWdX(iE, iL), 15) + "\t" +
                utils::pp(gridWdY(iE, iL), 15) + "\n";
            }
            strm<<"\n";
        }
    }

    if(!errorMessage.empty())
        throw std::runtime_error("ActionFinderSpherical: "+errorMessage);

    return math::QuinticSpline2d(gridX, gridY, gridW, gridWdX, gridWdY);
}

/// construct the interpolating spline for scaled energy X as a function of log(Jr+L), L/(Jr+L)
math::QuinticSpline2d createEnergyInterpolator(const potential::Interpolator2d& pot,
    const math::BaseInterpolator2d& intJr)
{
    const double Phi0 = pot.value(0), invPhi0 = 1. / Phi0;
    std::vector<double> gridR = potential::createInterpolationGrid(
        potential::FunctionToPotentialWrapper(pot), ACCURACY_INTERP2);
    // extend the grid a little
    gridR.push_back( exp(2.5 * log(gridR[gridR.size()-1]) - 1.5 * log(gridR[gridR.size()-2])) );

    // interpolation grid for X = scaledE = log(1/Phi(0)-1/E) in scaled variables:
    // P = log(L+Jr), Q = L/(L+Jr)
    const int sizeP = gridR.size();
    const int sizeQ = GRID_SIZE_L;
    std::vector<double> gridP(sizeP), gridQ(sizeQ);

    // create a non-uniform grid in Q = Jr/(L+Jr), using a transformation of interval [0:1]
    // onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at x=0 and x=1
    math::ScalingQui scaling(0, 1);
    for(int i=0; i<sizeQ; i++)
        gridQ[i] = math::unscale(scaling, i / (sizeQ-1.));

    // value of X=scaledE and its derivatives w.r.t. P and Q
    math::Matrix<double> gridX(sizeP, sizeQ), gridXdP(sizeP, sizeQ), gridXdQ(sizeP, sizeQ);

    std::string errorMessage;  // store the error text in case of an exception in the OpenMP block
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iP=0; iP<sizeP; iP++) {
        try{
            double Rc = gridR[iP];
            double Phi, dPhi, d2Phi;
            pot.evalDeriv(Rc, &Phi, &dPhi, &d2Phi);
            double Lc = Rc  * sqrt( Rc * dPhi);  // exp(P) = Jr+L
            gridP[iP] = log(Lc);
            for(int iQ=0; iQ<sizeQ; iQ++) {
                double L = Lc * gridQ[iQ], Jr = Lc * (1-gridQ[iQ]);
                // radius of a circular orbit with angular momentum equal to L
                double Rcirc = iQ<sizeQ-1 ? pot.R_from_Lz(L) : Rc;
                // initial guess (more precisely, lower bound) for Hamiltonian
                double Elow  = pot.value(Rcirc) + (L>0 ? 0.5 * pow_2(L/Rcirc) : 0);
                double dEdX,X= scaleE(Elow, invPhi0, /*output*/ &dEdX);
                double dEdJr = sqrt(d2Phi + 3*dPhi/Rc);  // kappa - epicyclic frequency (when Jr=0)
                double dEdL  = sqrt(dPhi/Rc);            // Omega --"--
                // if the radial action Jr is zero, Elow = Ecirc is the correct result,
                // otherwise need to find E such that Jr(E, L) equals the target value
                if(Jr>0) {
                    HamiltonianFinderFncInterpolated fnc(pot, Jr, L, invPhi0, intJr);
                    // find E such that Jr(E, L) equals the target value.
                    // We use logarithmically-scaled variable X=scaledE, which technically may range
                    // from -inf to +inf, but in practice is likely to be within a range of +-few tens.
                    // Since this is still an unbound range, in the root-finder we employ another
                    // scaling transformation X <-> z, with 0<z<1.
                    math::ScalingInf scaling;
                    double zroot = math::findRoot(
                        math::ScaledFnc<math::ScalingInf>(scaling, fnc),
                        /*lower limit is Elow, which translates to*/ math::scale(scaling, X),
                        /*upper limit on scaledE is infinity, which corresponds to*/ 1, ACCURACY_JR);
                    if(zroot==zroot) {
                        // only if the root-finder was successful,
                        // otherwise leave Elow=Ecirc as for Jr=0
                        X = math::unscale(scaling, zroot);
                        double E = unscaleE(X, invPhi0, /*output*/ &dEdX);
                        // once again compute the radial action _and_frequencies_ for the given energy
                        // (return value is ignored because we assume that it is equal to Jr)
                        computeJr(E, X, dEdX, L, pot, intJr, /*output*/&dEdJr, &dEdL);
                    }
                }
                gridX  (iP, iQ) = X;
                gridXdP(iP, iQ) = (dEdJr * Jr + dEdL * L) / dEdX;
                gridXdQ(iP, iQ) = (Jr+L) * (dEdL - dEdJr) / dEdX;
            }
        }
        catch(std::exception& e) {
            errorMessage = e.what();
        }
    }
    if(!errorMessage.empty())
        throw std::runtime_error("ActionFinderSpherical: "+errorMessage);

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {   // debugging output
        std::ofstream strm("ActionFinderSphericalEnergy.log");
        strm << "# P=ln(Jr+L)   \tQ=L/(Jr+L)     \tX=scaledE      \tdX/dP          \tdX/dQ          \n";
        for(int iP=0; iP<sizeP; iP++) {
            for(int iQ=0; iQ<sizeQ; iQ++) {
                strm << 
                utils::pp(gridP  [iP],     15) + "\t" +
                utils::pp(gridQ  [iQ],     15) + "\t" +
                utils::pp(gridX  (iP, iQ), 15) + "\t" +
                utils::pp(gridXdP(iP, iQ), 15) + "\t" +
                utils::pp(gridXdQ(iP, iQ), 15) + "\n";
            }
            strm<<"\n";
        }
    }

    //return math::CubicSpline2d(gridP, gridQ, gridX);
    return math::QuinticSpline2d(gridP, gridQ, gridX, gridXdP, gridXdQ);
}

}  //internal namespace


double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts)
{
    if(acts.Jr<0 || acts.Jz<0)
        return NAN;
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
    // radius of a circular orbit with this angular momentum
    double rcirc = R_from_Lz(potential, L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Elow = potential.value(coord::PosCyl(rcirc, 0, 0)) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // if the radial action is zero, this is the result
    if(acts.Jr==0)
        return Elow;
    // find E such that Jr(E, L) equals the target value.
    // We use logarithmically-scaled variable scaledE, which technically ranges from -inf to +inf,
    // but is likely to be within a range of +-few tens. On top of that, in the root-finder
    // we employ another scaling transformation scaledE <-> z, with 0<z<1.
    double invPhi0 = 1. / potential.value(coord::PosCyl(0,0,0));
    math::ScalingInf scaling;
    HamiltonianFinderFnc fnc(potential, acts.Jr, L);
    double zroot   = math::findRoot(
        math::ScaledFnc<math::ScalingInf>(scaling, fnc),
        /*lower limit is Elow, which translates to*/ math::scale(scaling, scaleE(Elow, invPhi0)),
        /*upper limit on scaledE is infinity, which corresponds to*/ 1, ACCURACY_JR);
    return unscaleE(math::unscale(scaling, zroot), invPhi0);
}

coord::PosVelCyl mapSpherical(
    const potential::BasePotential &pot,
    const ActionAngles &aa, Frequencies* freqout)
{
    if(!isSpherical(pot))
        throw std::invalid_argument("mapSpherical: potential must be spherically symmetric");
    double E = computeHamiltonianSpherical(pot, aa);
    double L = aa.Jz + fabs(aa.Jphi);  // total angular momentum
    double R1, R2;
    findPlanarOrbitExtent(pot, E, L, R1, R2);
    Frequencies freq;
    potential::Sphericalized<potential::BasePotential> sphPot(pot);
    freq.Omegar = M_PI / integr<MODE_OMEGAR>(sphPot, E, L, R1, R2);
    freq.Omegaz = freq.Omegar * integr<MODE_OMEGAZ>(sphPot, E, L, R1, R2) / M_PI;
    freq.Omegaphi = freq.Omegaz * math::sign(aa.Jphi);
    if(freqout)  // freak out only if requested
        *freqout = freq;
    return mapPointFromActionAngles(aa, sphPot, E, L, R1, R2, freq.Omegar, freq.Omegaz);
}

void evalSpherical(const potential::BasePotential& pot, const coord::PosVelCyl& point,
    Actions* act, Angles* ang, Frequencies* freqout)
{
    if(!isSpherical(pot))
        throw std::invalid_argument("evalSpherical can only deal with spherical potentials");
    double E  = totalEnergy(pot, point);
    double Lz = point.R * point.vphi;
    double Lx2plusLy2 = pow_2(point.z * point.vphi) + pow_2(point.R * point.vz - point.z * point.vR);
    double L  = sqrt(Lz * Lz + Lx2plusLy2);
    if(act) {
        act->Jr = NAN;  // temporary assignment - will be replaced by the correct value unless E>=0
        act->Jz = L==0 ? 0 : Lx2plusLy2 / (L + fabs(Lz));  // a roundoff-safe way of computing L - |Lz|
        act->Jphi = Lz;
    }
    if(E>=0) {
        if(ang)
            ang->thetar = ang->thetaz = ang->thetaphi = NAN;
        if(freqout)
            freqout->Omegar = freqout->Omegaz = freqout->Omegaphi = NAN;
        return;
    }
    double R1, R2;   // peri/apocenter radii
    findPlanarOrbitExtent(pot, E, L, R1, R2);
    potential::Sphericalized<potential::BasePotential> sphPot(pot);
    if(act)
        act->Jr = integr<MODE_JR>(sphPot, E, L, R1, R2) / M_PI;
    // if angles are requested, frequencies need to be computed as well
    if(ang || freqout) {
        Frequencies freq;
        if(R2 > R1 * (1 + 1e-8)) {  // normal case
            freq.Omegar = M_PI / integr<MODE_OMEGAR>(sphPot, E, L, R1, R2);
            freq.Omegaz = freq.Omegar * integr<MODE_OMEGAZ>(sphPot, E, L, R1, R2) / M_PI;
        } else {  // degenerate case of a purely circular orbit
            epicycleFreqs(pot, point.R, freq.Omegar, freq.Omegaz, /*ignored*/ freq.Omegaphi);
        }
        freq.Omegaphi = freq.Omegaz * math::sign(point.vphi);
        if(freqout)  // freak out only if requested
            *freqout = freq;
        if(ang)
            *ang = computeAngles(point, sphPot, E, L, R1, R2, freq.Omegar, freq.Omegaz);
    }
}

ActionFinderSpherical::ActionFinderSpherical(const potential::BasePotential& potential) :
    invPhi0(1. / potential.value(coord::PosCyl(0,0,0))),
    pot(potential),
    intJr(createActionInterpolator(pot)),
#ifdef INTERPOLATE_ENERGY
    intE(createEnergyInterpolator(pot, intJr)),
#endif
    myName("Spherical(potential=" + potential.name() + ")")
{}

double ActionFinderSpherical::Jr(double E, double L, double *Omegar, double *Omegaz) const
{
    // convert the values of E and L into the scaled variables used for interpolation
    double dEdX, X = scaleE(E, invPhi0, /*output*/ &dEdX);
    return computeJr(E, X, dEdX, L, pot, intJr, /*optional output*/ Omegar, Omegaz);
}

void ActionFinderSpherical::eval(const coord::PosVelCyl& point,
    Actions* act, Angles* ang, Frequencies* freq) const
{
    Actions acts;
    double E  = pot.value(sqrt(pow_2(point.R) + pow_2(point.z))) + 
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    double Lz = point.R * point.vphi;
    double Lx2plusLy2 = pow_2(point.z * point.vphi) + pow_2(point.R * point.vz - point.z * point.vR);
    double L  = sqrt(Lz * Lz + Lx2plusLy2);
    double Omegar, Omegaz, Jr, R1, R2;
    if(freq || ang)
        Jr = this->Jr(E, L, &Omegar, &Omegaz);
    else
        Jr = this->Jr(E, L);
    if(act) {
        act->Jr = Jr;
        act->Jz = L==0 ? 0 : Lx2plusLy2 / (L + fabs(Lz));  // a roundoff-safe way of computing L - |Lz|
        act->Jphi = Lz;
    }
    if(ang) {
        if(E>=0)
            ang->thetar = ang->thetaz = ang->thetaphi = NAN;
        else {
            pot.findPlanarOrbitExtent(E, L, R1, R2);
            *ang = computeAngles(point, pot, E, L, R1, R2, Omegar, Omegaz);
        }
    }
    if(freq) {
        freq->Omegar = Omegar;
        freq->Omegaz = Omegaz;
        freq->Omegaphi = Omegaz * math::sign(Lz);
    }
}

double ActionFinderSpherical::E(const Actions& acts) const
{
    if(acts.Jr<0 || acts.Jz<0)
        return NAN;
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
#ifdef INTERPOLATE_ENERGY
    double scaledE, der,
    Jtot = acts.Jr + L,
    Lrel = L/Jtot,
    logJ = log(Jtot),
    logJeval = math::clip(logJ, intE.xmin(), intE.xmax());  // make sure it's within the 2d grid
    if(logJ != logJeval) {
        // compute the value at the edge of the 2d interpolation grid, and then linearly extrapolate
        intE.evalDeriv(logJeval, Lrel, &scaledE, &der);
        scaledE += der * (logJ - logJeval);
    } else
        scaledE = intE.value(logJ, Lrel);
    return unscaleE(scaledE, invPhi0);
#else
    // radius of a circular orbit with this angular momentum
    double rcirc = pot.R_from_Lz(L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Elow  = pot.value(rcirc) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    if(acts.Jr==0)
        return Elow;
    // find E such that Jr(E, L) equals the target value.
    // We use logarithmically-scaled variable scaledE, which technically ranges from -inf to +inf,
    // but is likely to be within a range of +-few tens. On top of that, in the root-finder
    // we employ another scaling transformation scaledE <-> z, with 0<z<1.
    math::ScalingInf scaling;
    HamiltonianFinderFncInterpolated fnc(pot, acts.Jr, L, invPhi0, intJr);
    double zroot   = math::findRoot(
        math::ScaledFnc<math::ScalingInf>(scaling, fnc),
        /*lower limit is Elow, which translates to*/ math::scale(scaling, scaleE(Elow, invPhi0)),
        /*upper limit on scaledE is infinity, which corresponds to*/ 1, ACCURACY_JR);
    return unscaleE(math::unscale(scaling, zroot), invPhi0);    
#endif
}

coord::PosVelCyl ActionFinderSpherical::map(const ActionAngles& aa, Frequencies* freq) const
{
    if(aa.Jr<0 || aa.Jz<0) {
        if(freq)
            freq->Omegar = freq->Omegaz = freq->Omegaphi = NAN;
        return coord::PosVelCyl(NAN, NAN, NAN, NAN, NAN, NAN);
    }
    double E = this->E(aa);
    double L = aa.Jz + fabs(aa.Jphi);  // total angular momentum
    // compute the frequencies
    double Omegar, Omegaz;
    double Jr = this->Jr(E, L, &Omegar, &Omegaz);
    double Omegaphi = Omegaz * math::sign(aa.Jphi);
    if(freq)  // output if requested
        *freq = Frequencies(Omegar, Omegaz, Omegaphi);
#ifdef INTERPOLATE_ENERGY
    E += Omegar * (aa.Jr-Jr);  // first-order correction for the interpolated E(Jr,L)
#endif
    // find peri/apocenter radii
    double R1, R2;
    pot.findPlanarOrbitExtent(E, L, R1, R2);
    // map the point from action/angles and frequencies
    return mapPointFromActionAngles(aa, pot, E, L, R1, R2, Omegar, Omegaz);
}

}  // namespace actions
