#include "actions_spherical.h"
#include "potential_utils.h"
#include "math_core.h"
#include "utils.h"
#include <string>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <fstream>   // for writing debug info

namespace actions{

namespace {

/// required tolerance on the value of Jr(E) in the root-finder
const double ACCURACY_JR = 1e-6;
    
/// accuracy parameter determining the radial spacing of the 2d interpolation grid for Jr
static const double ACCURACY_INTERP2 = 1e-4;

/// minimum order of Gauss-Legendre quadrature for actions, frequencies and angles
const unsigned int INTEGR_ORDER = 10;
    
/** order of Gauss-Legendre quadrature for actions, frequencies and angles:
    use a higher order for more eccentric orbits, as indicated by the ratio
    of pericenter to apocenter radii (R1/R2) */
inline unsigned int integrOrder(double R1overR2) {
    if(R1overR2==0)
        return math::MAX_GL_ORDER;
    int log2;  // base-2 logarithm of R1/R2
    frexp(R1overR2, &log2);
    return std::min<int>(math::MAX_GL_ORDER, INTEGR_ORDER - log2);
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
        if(vr2<=0 || r==R1) return 0;
        double vr  = sqrt(vr2);
        if(mode==MODE_JR)     return vr;
        if(mode==MODE_OMEGAR) return 1/vr;
        if(mode==MODE_OMEGAZ) return L/(r*r*vr) - 1/(sqrt(pow_2(r/R1)-1)*r);
        // in the latter case, a singular term with analytic antiderivative is subtracted from the integrand
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
    MODE_OMEGAZ ->    \int L / (v_r * r^2) dr,  in the latter case the integrand is split into two parts,
    one of them can be integrated analytically, and the other does not diverge as L->0,r->0.
    \return the value of integral.
*/
template<Operation mode>
inline double integr(const math::IFunction& poten,
    double E, double L, double R1, double R2, double R=NAN)
{
    if(R!=R) R=R2;               // default upper limit for integration
    R = math::clamp(R, R1, R2);  // roundoff errors might cause R to be outside the allowed interval
    Integrand<mode> integrand(poten, E, L, R1);
    math::ScalingCub scaling(R1, R2);
    double add = 0;  // part of the integral that is computed analytically
    if(mode==MODE_OMEGAZ) {
        if(R1==0) {
            // "add" is (half) the change in angle phi for a purely radial orbit (Pi/2 for potentials
            // that are regular at origin, or larger for singular potentials, up to Pi in the Kepler case)
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
    const potential::BasePotential& potential;
    /// the values of actions
    const double Jr, L;
    /// boundaries of the energy interval (to use in the first two calls from the root-finder)
    const double Emin, Emax;
public:
    HamiltonianFinderFnc(const potential::BasePotential& p,
        double _Jr, double _L, double _Emin, double _Emax) :
        potential(p), Jr(_Jr), L(_L), Emin(_Emin), Emax(_Emax) {};
    /// report the difference between target Jr and the one computed at the given energy
    virtual double value(const double E) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(E==Emin)
            return -Jr;
        if(E==Emax)
            return Jr+1e-10;  // at r==infinity should return some positive value
        double R1, R2;
        findPlanarOrbitExtent(potential, E, L, R1, R2);
        return integr<MODE_JR>(potential::PotentialWrapper(potential), E, L, R1, R2) / M_PI - Jr;
    }
};

/// same operation using the interpolated radial action finder
class HamiltonianFinderFncInt: public math::IFunction {
    /// the instance of interpolated action finder
    const ActionFinderSpherical& af;
    /// the values of actions
    const double Jr, L;
    /// boundaries of the energy interval (to use in the first two calls from the root-finder)
    const double Emin, Emax;
public:
    HamiltonianFinderFncInt(const ActionFinderSpherical& _af,
        double _Jr, double _L, double _Emin, double _Emax) :
        af(_af), Jr(_Jr), L(_L), Emin(_Emin), Emax(_Emax) {};
    /// report the difference between target Jr and the one computed at the given energy
    virtual void evalDeriv(const double E, double *val=0, double *der=0, double* =0) const {
        // first two calls in root-finder are for the boundary points, we already know the answer
        if(E==Emin || E==Emax) {
            if(val)
                *val = E==Emin ? -Jr : Jr+1e-10;
            if(der)
                *der = NAN;
            return;
        }
        double v = af.Jr(E, L, der);  // der now contains Omega_r = 1 / (dJr/dE)
        if(val)
            *val = v - Jr;
        if(der)
            *der = 1 / *der;
    }
    virtual unsigned int numDerivs() const { return 1; }
};

/** Compute actions and other orbit parameters for the given point and potential.
    This routine is shared between `actionsSpherical` and `actionsAnglesSpherical`,
    and in addition to actions, outputs other quantities that may be later used elsewhere.
    \param[in]  point  is the input point;
    \param[in]  potential  is the gravitational potential;
    \param[out] E  is the total energy;
    \param[out] L  is the angular momentum;
    \param[out] R1, R2  are the peri/apocenter radii;
    \return  the values of actions (or NAN in Jr if the energy is positive).
*/
Actions computeActions(const coord::PosVelCyl& point, const potential::BasePotential& pot,
    double &E, double &L, double &R1, double &R2)
{
    if(!isSpherical(pot))
        throw std::invalid_argument("actionsSpherical can only deal with spherical potentials");
    Actions act;
    E = totalEnergy(pot, point);
    L = Ltotal(point);
    act.Jphi = Lz(point);
    // avoid roundoff errors if Jz is close to 0 or exactly 0
    act.Jz = point.z==0 && point.vz==0 ? 0 : fmax(0, L - fabs(act.Jphi));
    if(E>=0) {
        act.Jr = NAN;
    } else {
        findPlanarOrbitExtent(pot, E, L, R1, R2);
        act.Jr = integr<MODE_JR>(potential::PotentialWrapper(pot), E, L, R1, R2) / M_PI;
    }
    return act;
}

/** Compute angles for the given point.
    This routine is shared between the standalone function `actionAnglesSpherical`
    and the member function `actionAngles` of the interpolated action finder.
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
    double psi = atan2(point.z * L,  -point.R * vtheta * r);
    double chi = atan2(point.z * point.vphi, -vtheta * r);
    Angles ang;
    ang.thetar = integr<MODE_OMEGAR>(potential, E, L, R1, R2, r) * Omegar;
    double thr = ang.thetar;
    double thz = integr<MODE_OMEGAZ>(potential, E, L, R1, R2, r);
    if(point.R * point.vR + point.z * point.vz < 0) {  // v_r<0 - we're on the second half of radial period
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
coord::PosVelSphMod mapPointFromActionAngles(const ActionAngles &aa,
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
    double chi      = aa.Jz != 0 ? atan2(fabs(aa.Jphi) * sinpsi, L * cospsi) : psi;
    double sini     = sqrt(1 - pow_2(aa.Jphi / L)); // inclination angle of the orbital plane
    double costheta = sini * sinpsi;                // z/r
    double sintheta = sqrt(1 - pow_2(costheta));    // R/r is always non-negative
    // finally, output position/velocity
    coord::PosVelSphMod point;
    point.r    = r;
    point.pr   = vr;
    point.tau  = costheta / (1+sintheta);
    point.ptau = L * sini * cospsi * (1/sintheta + 1);
    point.phi  = math::wrapAngle(aa.thetaphi + (chi-aa.thetaz) * math::sign(aa.Jphi));
    point.pphi = aa.Jphi;
    return point;
}

/** Compute the derivatives of pos/vel w.r.t. actions by finite differencing.
    \param[in]  aa  are the actions and angles at a point slightly offset from the original one;
    \param[in]  p0  is the original point;
    \param[in]  EPS is the magnitude of offset (in either of the three actions);
    \param[in]  af  is the instance of interpolated action finder (used to compute frequencies);
    \param[in]  pot is the interpolated potential;
    \param[in]  E   is the energy slightly offset from the original one;
    \param[in]  R1,R2  are the peri/apocenter radii, again slightly offset (all computed elsewhere);
    \return  the derivative of position/velocity point by the action that had this offset.
*/
coord::PosVelSphMod derivPointFromActions(
    const ActionAngles &aa, const coord::PosVelSphMod &p0, double EPS,
    const ActionFinderSpherical& af, const potential::Interpolator2d &pot,
    const double E, const double R1, const double R2)
{
    double Omegar, Omegaz, L = aa.Jz + fabs(aa.Jphi);
    af.Jr(E, L, &Omegar, &Omegaz);
    double Ra,Rb;
    pot.findPlanarOrbitExtent(E, L, Ra, Rb);
    coord::PosVelSphMod p = mapPointFromActionAngles(aa, pot, E, L, R1, R2, Omegar, Omegaz);
    p.r   = (p.r   - p0.r   )/EPS;
    p.pr  = (p.pr  - p0.pr  )/EPS;
    p.tau = (p.tau - p0.tau )/EPS;
    p.ptau= (p.ptau- p0.ptau)/EPS;
    p.phi = (p.phi - p0.phi )/EPS;
    p.pphi= (p.pphi- p0.pphi)/EPS;
    return p;
}

/// return scaledE and dE/d(scaledE) as functions of E and invPhi0 = 1/Phi(0)
inline void scaleE(const double E, const double invPhi0,
    /*output*/ double& scaledE, double& dEdscaledE)
{
    double expE = invPhi0 - 1/E;
    scaledE     = log(expE);
    dEdscaledE  = E * E * expE;
}

/// construct the interpolating spline for scaled radial action W = Jr / (Lcirc-L)
/// as a function of E and L/Lcirc
math::QuinticSpline2d makeActionInterpolator(const potential::Interpolator2d& pot)
{
    double invPhi0 = 1. / pot.value(0);
    std::vector<double> gridR = potential::createInterpolationGrid(
        potential::FunctionToPotentialWrapper(pot), ACCURACY_INTERP2);

    // interpolation grid in scaled variables: X = scaledE = log(1/Phi(0)-1/E), Y = L / Lcirc(E)
    const int sizeE = gridR.size();
    const int sizeL = 40;
    std::vector<double> gridX(sizeE), gridY(sizeL);

    // create a non-uniform grid in Y = L/Lcirc(E), using a transformation of interval [0:1]
    // onto itself that places more grid points near the edges:
    // a function with zero 1st and 2nd derivs at x=0 and x=1
    math::ScalingQui scaling(0, 1);
    for(int i=0; i<sizeL; i++)
        gridY[i] = math::unscale(scaling, 1. * i / (sizeL-1));

    // value of W=Jr/(Lcirc-L) and its derivatives w.r.t. X=scaledE and Y=L/Lcirc
    math::Matrix<double> gridW(sizeE, sizeL), gridWdX(sizeE, sizeL), gridWdY(sizeE, sizeL);
    std::vector<double> gridWatY1(sizeE);  // last column of the W matrix (values at Y=1 and any X)

    std::string errorMessage;  // store the error text in case of an exception in the openmp block
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
            scaleE(E, invPhi0, /*output*/gridX[iE], dEdX);
            double dLcdE  = Rc*Rc/Lc;
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
    if(!errorMessage.empty())
        throw std::runtime_error("ActionFinderSpherical: "+errorMessage);

    // derivative dW/dX at Y=1 is computed by constructing an auxiliary 1d spline for W(X)|Y=1
    // and differentiating it
    math::CubicSpline intWatY1(gridX, gridWatY1);
    for(int iE=0; iE<sizeE; iE++)
        intWatY1.evalDeriv(gridX[iE], NULL, &gridWdX(iE, sizeL-1));

    if(utils::verbosityLevel >= utils::VL_VERBOSE) {   // debugging output
        std::ofstream strm("ActionFinderSpherical.log");
        strm << "# X=scaledE    \tY=L/Lcirc      \tW=Jr/(Lcirc-L) \tdW/dX          \tdW/dY          \n";
        for(int iE=0; iE<sizeE; iE++) {
            for(int iL=0; iL<sizeL; iL++) {
                strm << 
                utils::pp(gridX[iE], 15) + "\t" +
                utils::pp(gridY[iL], 15) + "\t" +
                utils::pp(gridW  (iE, iL), 15) + "\t" +
                utils::pp(gridWdX(iE, iL), 15) + "\t" +
                utils::pp(gridWdY(iE, iL), 15) + "\n";
            }
            strm<<"\n";
        }
    }

    //return math::CubicSpline2d(gridX, gridY, gridW);
    return math::QuinticSpline2d(gridX, gridY, gridW, gridWdX, gridWdY);
}

}  //internal namespace


double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts)
{
    if(acts.Jr<0 || acts.Jz<0)
        throw std::invalid_argument("computeHamiltonianSpherical: input actions are negative");
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
    // radius of a circular orbit with this angular momentum
    double rcirc = R_from_Lz(potential, L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Ecirc = potential.value(coord::PosCyl(rcirc, 0, 0)) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // upper bound for Hamiltonian
    double Einf  = potential.value(coord::PosCyl(INFINITY, 0, 0));
    if(!isFinite(Einf) && Einf != INFINITY)  // some potentials may return NAN for r=infinity
        Einf = 0;  // assume the default value for potential at infinity
    // find E such that Jr(E, L) equals the target value
    HamiltonianFinderFnc fnc(potential, acts.Jr, L, Ecirc, Einf);
    return math::findRoot(fnc, Ecirc, Einf, ACCURACY_JR);
}

coord::PosVelCyl mapSpherical(
    const potential::BasePotential &pot,
    const ActionAngles &aa, Frequencies* freqout)
{
    if(!isSpherical(pot))
        throw std::invalid_argument("mapSpherical: potential must be spherically symmetric");
    if(aa.Jr<0 || aa.Jz<0)
        throw std::invalid_argument("mapSpherical: input actions are negative");
    double E = computeHamiltonianSpherical(pot, aa);
    double L = aa.Jz + fabs(aa.Jphi);  // total angular momentum
    double R1, R2;
    findPlanarOrbitExtent(pot, E, L, R1, R2);
    Frequencies freq;
    freq.Omegar = M_PI / integr<MODE_OMEGAR>(potential::PotentialWrapper(pot), E, L, R1, R2);
    freq.Omegaz = freq.Omegar * integr<MODE_OMEGAZ>(potential::PotentialWrapper(pot), E, L, R1, R2) / M_PI;
    freq.Omegaphi = freq.Omegaz * math::sign(aa.Jphi);
    if(freqout)  // freak out only if requested
        *freqout = freq;
    return toPosVelCyl(mapPointFromActionAngles(
        aa, potential::PotentialWrapper(pot), E, L, R1, R2, freq.Omegar, freq.Omegaz));
}


Actions actionsSpherical(
    const potential::BasePotential& potential, const coord::PosVelCyl& point)
{
    double E, L, R1, R2;
    return computeActions(point, potential, E, L, R1, R2);
}

ActionAngles actionAnglesSpherical(
    const potential::BasePotential& pot, const coord::PosVelCyl& point, Frequencies* freqout)
{
    double E, L, R1, R2;
    Actions acts = computeActions(point, pot, E, L, R1, R2);
    if(!isFinite(acts.Jr)) { // E>=0
        if(freqout) freqout->Omegar = freqout->Omegaz = freqout->Omegaphi = NAN;
        return ActionAngles(acts, Angles(NAN, NAN, NAN));
    }
    Frequencies freq;
    freq.Omegar = M_PI / integr<MODE_OMEGAR>(potential::PotentialWrapper(pot), E, L, R1, R2);
    freq.Omegaz = freq.Omegar * integr<MODE_OMEGAZ>(potential::PotentialWrapper(pot), E, L, R1, R2) / M_PI;
    freq.Omegaphi = freq.Omegaz * math::sign(point.vphi);
    if(freqout)  // freak out only if requested
        *freqout = freq;
    // may wish to add a special case of Jr==0 (output the epicyclic frequencies, but no angles?)
    Angles angs  = computeAngles(point, potential::PotentialWrapper(pot),
        E, L, R1, R2, freq.Omegar, freq.Omegaz);
    return ActionAngles(acts, angs);
}


ActionFinderSpherical::ActionFinderSpherical(const potential::BasePotential& potential) :
    invPhi0(1. / potential.value(coord::PosCyl(0,0,0))),
    pot(potential),
    intJr(makeActionInterpolator(pot))
{}

double ActionFinderSpherical::Jr(double E, double L, double *Omegar, double *Omegaz) const
{
    bool needDeriv = Omegar!=NULL || Omegaz!=NULL;
    double dLcdE, Lc = pot.L_circ(E, needDeriv? &dLcdE : NULL);
    if(!isFinite(Lc)) {  // E>=0 or E<Phi(0)
        if(Omegar) *Omegar = NAN;
        if(Omegaz) *Omegaz = NAN;
        return NAN;
    }

    // convert the values of E and L into the scaled variables used for interpolation
    double dEdX, X;
    scaleE(E, invPhi0, X, dEdX);
    X = math::clamp(X, intJr.xmin(), intJr.xmax());
    double Y = math::clamp(fabs(L/Lc), 0., 1.);

    // obtain the value of scaled Jr as a function of scaled E and L, and unscale it
    double val, derX, derY;
    intJr.evalDeriv(X, Y, &val, needDeriv? &derX : NULL, needDeriv? &derY : NULL);
    if(needDeriv) {
        double dJrdL = derY * (1-Y) - val;
        double dJrdE = derX * (1-Y) * Lc / dEdX - (derY * (1-Y) * Y - val) * dLcdE;
        if(Omegar)
            *Omegar  = 1 / dJrdE;
        if(Omegaz)
            *Omegaz  = -dJrdL / dJrdE;
    }
    return val * Lc * (1-Y);
}

Actions ActionFinderSpherical::actions(const coord::PosVelCyl& point) const
{
    Actions acts;
    double E  = pot.value(sqrt(pow_2(point.R) + pow_2(point.z))) + 
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    double L  = Ltotal(point);
    acts.Jphi = Lz(point);
    acts.Jz   = point.z==0 && point.vz==0 ? 0 : fmax(0, L - fabs(acts.Jphi));
    acts.Jr   = E<=0 ? Jr(E, L) : NAN;
    return acts;
}

ActionAngles ActionFinderSpherical::actionAngles(
    const coord::PosVelCyl& point, Frequencies* freq) const
{
    Actions acts;
    double E  = pot.value(sqrt(pow_2(point.R) + pow_2(point.z))) + 
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    double L  = Ltotal(point);
    double Omegar, Omegaz;
    acts.Jphi = Lz(point);
    acts.Jz   = point.z==0 && point.vz==0 ? 0 : fmax(0, L - fabs(acts.Jphi));
    acts.Jr   = Jr(E, L, &Omegar, &Omegaz);
    if(freq)
        *freq = Frequencies(Omegar, Omegaz, Omegaz * math::sign(acts.Jphi));
    double R1, R2;
    pot.findPlanarOrbitExtent(E, L, R1, R2);
    Angles angs = computeAngles(point, pot, E, L, R1, R2, Omegar, Omegaz);
    return ActionAngles(acts, angs);
}

double ActionFinderSpherical::E(const Actions& acts) const
{
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
    // radius of a circular orbit with this angular momentum
    double rcirc = pot.R_from_Lz(L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Ecirc = pot.value(rcirc) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // find E such that Jr(E, L) equals the target value
    return math::findRoot(HamiltonianFinderFncInt(*this, acts.Jr, L, Ecirc, 0),
        Ecirc, 0, ACCURACY_JR);
}

coord::PosVelSphMod ActionFinderSpherical::map(
    const ActionAngles& aa,
    Frequencies* freq,
    DerivAct<coord::SphMod>* derivAct,
    DerivAng<coord::SphMod>*,
    coord::PosVelSphMod*) const
{
    if(aa.Jr<0 || aa.Jz<0)
        throw std::invalid_argument("mapSpherical: input actions are negative");
    double E = this->E(aa);
    double L = aa.Jz + fabs(aa.Jphi);  // total angular momentum
    // compute the frequencies
    double Omegar, Omegaz;
    Jr(E, L, &Omegar, &Omegaz);
    double Omegaphi = Omegaz * math::sign(aa.Jphi);
    if(freq)  // output if requested
        *freq = Frequencies(Omegar, Omegaz, Omegaphi);
    // find peri/apocenter radii
    double R1, R2;
    pot.findPlanarOrbitExtent(E, L, R1, R2);
    // map the point from action/angles and frequencies
    coord::PosVelSphMod p0 = mapPointFromActionAngles(aa, pot, E, L, R1, R2, Omegar, Omegaz);
    if(derivAct) {
        // use the fact that dE/dJr = Omega_r, dE/dJz = Omega_z, etc, and find dR{1,2}/dJ{r,z}
        double dPhidR;
        pot.evalDeriv(R1, NULL, &dPhidR);
        double factR1 = pow_2(L) / pow_3(R1) - dPhidR;
        pot.evalDeriv(R2, NULL, &dPhidR);
        double factR2 = pow_2(L) / pow_3(R2) - dPhidR;
        double dR1dJr = -Omegar / factR1;
        double dR2dJr = -Omegar / factR2;
        double dR1dJz = (L / pow_2(R1) - Omegaz) / factR1;
        double dR2dJz = (L / pow_2(R2) - Omegaz) / factR2;
        // compute the derivs using finite-difference (silly approach, no error control)
        double EPS= 1e-8;   // no proper scaling attempted!! (TODO - do it properly or not at all)
        derivAct->dbyJr = derivPointFromActions(
            ActionAngles(Actions(aa.Jr + EPS, aa.Jz, aa.Jphi), aa), p0, EPS, *this, pot,
            E + EPS * Omegar, R1 + EPS * dR1dJr, R2 + EPS * dR2dJr);
        derivAct->dbyJz = derivPointFromActions(
            ActionAngles(Actions(aa.Jr, aa.Jz + EPS, aa.Jphi), aa), p0, EPS, *this, pot,
            E + EPS * Omegaz, R1 + EPS * dR1dJz, R2 + EPS * dR2dJz);
        derivAct->dbyJphi = derivPointFromActions(
            ActionAngles(Actions(aa.Jr, aa.Jz, aa.Jphi + EPS), aa), p0, EPS, *this, pot,
            E + EPS * Omegaphi, R1 + EPS * dR1dJz * math::sign(aa.Jphi),
            R2 + EPS * dR2dJz * math::sign(aa.Jphi));  // dX/dJphi = dX/dJz*sign(Jphi)
    }
    return p0;
}

}  // namespace actions
