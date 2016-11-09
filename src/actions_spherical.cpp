#include "actions_spherical.h"
#include "actions_interfocal_distance_finder.h"
#include "potential_utils.h"
#include "math_core.h"
#include <stdexcept>
#include <cassert>
#include <cmath>

namespace actions{

namespace {

/// required tolerance on the value of Jr(E) in the root-finder
const double ACCURACY_JR = 1e-6;

/// order of Gauss-Legendre quadrature for actions, frequencies and angles
const unsigned int INTEGR_ORDER = 10;

/** order of Gauss-Legendre quadrature for actions, frequencies and angles:
    use a higher order for more eccentric orbits, as indicated by the ratio
    of pericenter to apocenter radii (R1/R2) */
static inline unsigned int integrOrder(double R1overR2) {
    int order;  // base-2 logarithm of R1/R2
    frexp(R1overR2, &order);
    return INTEGR_ORDER + std::min<int>(-order, INTEGR_ORDER);
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
        assert("Invalid mode in action integrand"==0);
        return 0;
    }
};

/// helper class for computing the radial action and its derivatives
/// in the asymptotic limit of small radii (E -> Phi(0)) under the assumption of power-law potential;
template<Operation mode>
class IntegrandPowerLaw: public math::IFunctionNoDeriv {
    const double s, v;  ///< s is the potential slope, v is the relative ang.mom.(normalized to Lcirc)
    const double R1;    ///< lower limit of integration
public:
    IntegrandPowerLaw(double slope, double Lrel, double _R1) : s(slope), v(Lrel), R1(_R1) {}
    virtual double value(const double x) const {
        double t  = s!=0 ?  (pow(x, s) - 1) / s  :  log(x);
        double vr = sqrt(fmax(0, 1 - pow_2(v/x) - 2*t));
        if(mode==MODE_JR)     return vr;
        if(mode==MODE_OMEGAZ) return v/(x*x*vr) - 1/(sqrt(pow_2(x/R1)-1)*x);
        assert("Invalid mode in power-law action integrand"==0);
        return 0;
    }
};

/** compute the integral involving radius and radial velocity on the interval from peri- to apocenter,
    using scaling transformation to remove singularities at the endpoints.
    \param[in] poten  is the original or interpolated potential, accessed through IFunction interface;
    \param[in] E, L   are the integrals of motion (energy and ang.mom.);
    \param[in] R1, R2 are the peri/apocenter radii corresponding to these E and L (computed elsewhere);
    \param[in] R      is the upper limit of integration (the lower limit is always R1), -1 means R2;
    \tparam mode      determines the quantity to compute:
    MODE_JR     ->    \int_{R1}^{R}  v_r(E,L,r) dr,
    MODE_OMEGAR ->    \int 1 / v_r dr,
    MODE_OMEGAZ ->    \int L / (v_r * r^2) dr,  in the latter case the integrand is split into two parts,
    one of them can be integrated analytically, and the other does not diverge as L->0,r->0.
    \return the value of integral.
*/
template<Operation mode>
inline static double integr(const math::IFunction& poten,
    double E, double L, double R1, double R2, double R=-1)
{
    if(R==-1) R=R2;             // default upper limit for integration
    R = fmin(fmax(R, R1), R2);  // roundoff errors might cause R to be outside the allowed interval
    Integrand<mode> integrand(poten, E, L, R1);
    math::ScaledIntegrandEndpointSing transf(integrand, R1, R2);
    return math::integrateGL(transf, 0, transf.y_from_x(R), integrOrder(R1/R2))
        + (mode==MODE_OMEGAZ ? acos(R1/R) : 0);
}

/** same as above, but for the asymptotic limit of E -> Phi(0) in a power-law potential;
    \param[in]  slope defines the slope of the potential (Phi = Phi(0) + A * r^s);
    \param[in]  Lrel  is the relative angular momentum (normalized to the one of a circular orbit);
    \param[in]  R1, R2 are peri/apocenter radii normalized to the radius of a circular orbit.
*/
template<Operation mode>
inline static double integrPowerLaw(double slope, double Lrel, double R1, double R2)
{
    IntegrandPowerLaw<mode> integrand(slope, Lrel, R1);
    math::ScaledIntegrandEndpointSing transf(integrand, R1, R2);
    return math::integrateGL(transf, 0, 1, integrOrder(R1/R2)) + (mode==MODE_OMEGAZ ? acos(R1/R2) : 0);
}

/// helper function to find the upper limit of integral for the radial phase,
/// such that its value equals the target
class RadiusFromPhaseFinder: public math::IFunction {
    const Integrand<MODE_OMEGAR> integrand;
    const math::ScaledIntegrandEndpointSing transf;
    const double target;  // target value of the integral
public:
    RadiusFromPhaseFinder(const math::IFunction &poten,
        double E, double L, double R1, double R2, double _target) :
        integrand(poten, E, L, R1), transf(integrand, R1, R2), target(_target) {};
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
        return transf.x_from_y(rscaled);
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
static Actions computeActions(const coord::PosVelCyl& point, const potential::BasePotential& pot,
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
    This routine is shared between standalone function `actionAnglesSpherical`
    and the member function `actionAngles` of the interpolated action finder.
    \param[in]  point is the input point;
    \param[in]  potential is the original or interpolated potential;
    \param[in]  E is the total energy;
    \param[in]  L is the angular momentum;
    \param[in]  R1, R2 are peri/apocenter radii (computed elsewhere);
    \param[in]  Omegar, Omegaz are the corresponding frequencies (computed elsewhere);
    \returns    angle variables.
*/
static Angles computeAngles(const coord::PosVelCyl& point,
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
static coord::PosVelSphMod mapPointFromActionAngles(const ActionAngles &aa,
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
    double sinpsi   = sin(psi);
    double cospsi   = cos(psi);
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
    \param[in]  potential  is the interpolated potential;
    \param[in]  E   is the energy slightly offset from the original one;
    \param[in]  R1,R2  are the peri/apocenter radii, again slightly offset (all computed elsewhere);
    \return  the derivative of position/velocity point by the action that had this offset.
*/
static coord::PosVelSphMod derivPointFromActions(
    const ActionAngles &aa, const coord::PosVelSphMod &p0, double EPS,
    const ActionFinderSpherical& af, const potential::Interpolator2d &interp,
    const double E, const double R1, const double R2)
{
    double Omegar, Omegaz, L = aa.Jz + fabs(aa.Jphi);
    af.Jr(E, L, &Omegar, &Omegaz);
    double Ra,Rb;
    interp.findPlanarOrbitExtent(E, L, Ra, Rb);
    coord::PosVelSphMod p = mapPointFromActionAngles(aa, interp.pot, E, L, R1, R2, Omegar, Omegaz);
    p.r   = (p.r   - p0.r   )/EPS;
    p.pr  = (p.pr  - p0.pr  )/EPS;
    p.tau = (p.tau - p0.tau )/EPS;
    p.ptau= (p.ptau- p0.ptau)/EPS;
    p.phi = (p.phi - p0.phi )/EPS;
    p.pphi= (p.pphi- p0.pphi)/EPS;
    return p;
}

/// construct the interpolating spline for scaled radial action X = Jr / (Lcirc-L)
/// as a function of E and L/Lcirc
static math::CubicSpline2d makeActionInterpolator(const potential::Interpolator2d& interp)
{
    // for computing the asymptotic values at E=Phi(0), we assume a power-law behavior of potential:
    // Phi = Phi0 + coef * r^s
    double Phi0, slope = interp.pot.innerSlope(&Phi0);
    const int sizeE = 50;
    const int sizeL = 40;
    
    // create grids in energy and L/Lcirc(E), same as in Interpolator2d
    std::vector<double> gridE(sizeE), gridL(sizeL);
    for(int i=0; i<sizeE; i++) {
        double x = 1.*i/(sizeE-1);
        gridE[i] = (1 - pow_3(x) * (10+x*(-15+x*6))) * Phi0;
    }
    for(int i=0; i<sizeL; i++) {
        double x = 1.*i/(sizeL-1);
        gridL[i] = pow_3(x) * (10+x*(-15+x*6));
    }
    // value of Jr/(Lcirc-L) and its derivatives w.r.t. E and L/Lcirc
    math::Matrix<double> gridJr(sizeE, sizeL), gridJrdE(sizeE, sizeL), gridJrdL(sizeE, sizeL);

    // loop over values of energy strictly inside the interval [Phi0:0];
    // the boundary values will be treated separately
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int iE=1; iE<sizeE-1; iE++) {
        double E = gridE[iE];
        double dLcdE, Lc = interp.pot.L_circ(E, &dLcdE);
        for(int iL=0; iL<sizeL-1; iL++) {
            double L = gridL[iL] * Lc;
            double R1, R2;
            interp.findPlanarOrbitExtent(E, L, R1, R2);
            double Jr    = integr<MODE_JR>    (interp.pot, E, L, R1, R2) / M_PI;
            double dJrdE = integr<MODE_OMEGAR>(interp.pot, E, L, R1, R2) / M_PI;
            double dJrdL =-integr<MODE_OMEGAZ>(interp.pot, E, L, R1, R2) / M_PI;
            gridJr  (iE, iL) = Jr / (Lc - L);
            gridJrdE(iE, iL) = (dJrdE + (gridL[iL] * dJrdL - Jr / Lc) * dLcdE) / (Lc - L);
            gridJrdL(iE, iL) = (dJrdL + gridJr(iE, iL)) / (1 - gridL[iL]);
        }
        // limiting values for a nearly circular orbit
        // Jr = Omega/(2 kappa) * Lcirc * ecc,  where ecc = sqrt(1 - (L/Lcirc)^2).
        double kappa, nu, Omega;
        interp.pot.epicycleFreqs(interp.pot.R_circ(E), kappa, nu, Omega);
        gridJr(iE, sizeL-1) = Omega / kappa;
    }
    
    // asymptotic expressions for E -> Phi(0) assuming a power-law potential near origin
    for(int iL=0; iL<sizeL-1; iL++) {
        double R1, R2;   // these are scaled values, normalized to Rcirc
        interp.findScaledOrbitExtent(Phi0, gridL[iL], R1, R2);
        // integrations return the scaled value Jr/Lcirc and its derivative w.r.t. (L/Lcirc)
        double JroverLc = integrPowerLaw<MODE_JR>    (slope, gridL[iL], R1, R2) / M_PI;
        double dJrdL    =-integrPowerLaw<MODE_OMEGAZ>(slope, gridL[iL], R1, R2) / M_PI;
        gridJr  (0, iL) = JroverLc / (1 - gridL[iL]);
        gridJrdL(0, iL) = (dJrdL + gridJr(0, iL)) / (1 - gridL[iL]);
    }
    gridJr(0, sizeL-1) = sqrt(1/(slope+2));

    // asymptotic expressions for E -> 0 assuming Newtonian potential at infinity
    for(int iL=0; iL<sizeL; iL++) {
        gridJr  (sizeE-1, iL) = 1;
        gridJrdL(sizeE-1, iL) = 0;
    }
    
    // derivs wrt E for circular orbits cannot be obtained directly (involve 3rd deriv of potential),
    // thus they are computed by finite-differences (2nd order for interior nodes, 1st order at boundaries)
    for(int iE=1; iE<sizeE-1; iE++) {
        double difp = (gridJr(iE+1, sizeL-1) - gridJr(iE  , sizeL-1)) / (gridE[iE+1] - gridE[iE  ]);
        double difm = (gridJr(iE  , sizeL-1) - gridJr(iE-1, sizeL-1)) / (gridE[iE  ] - gridE[iE-1]);
        gridJrdE(iE, sizeL-1) = (difp * (gridE[iE] - gridE[iE-1]) + difm * (gridE[iE+1] - gridE[iE])) /
            (gridE[iE+1] - gridE[iE-1]);  // 2nd order accurate expression for the first derivative
        if(iE==1)
            gridJrdE(0, sizeL-1) = difm;  // at the endpoints use a 1st order expression
        if(iE==sizeE-2)
            gridJrdE(sizeE-1, sizeL-1) = difp;
    }

    // derivs wrt E at E=0 and E=Phi0 computed by quardatic interpolation of finite-differences,
    // using value at the boundary node, and value+deriv at the next-to-boundary node
    for(int iL=0; iL<sizeL-1; iL++) {
        gridJrdE(0, iL) = 2 * (gridJr(1, iL) - gridJr(0, iL)) / (gridE[1]-gridE[0]) - gridJrdE(1, iL);
        gridJrdE(sizeE-1, iL) = -gridJrdE(sizeE-2, iL) +
            2 * (gridJr(sizeE-1, iL) - gridJr(sizeE-2, iL)) / (gridE[sizeE-1]-gridE[sizeE-2]);
    }

    // same for derivs wrt (L/Lcirc) at L=Lcirc
    for(int iE=0; iE<sizeE; iE++)
        gridJrdL(iE, sizeL-1) = -gridJrdL(iE, sizeL-2) +
            2 * (gridJr(iE, sizeL-1) - gridJr(iE, sizeL-2)) / (gridL[sizeL-1]-gridL[sizeL-2]);

#if 0   // debugging output
    std::ofstream strm("filea.dat");
    strm << std::setprecision(15);
    for(unsigned int iE=0; iE<sizeE; iE++) {
        for(unsigned int iL=0; iL<sizeL; iL++) {
            strm << gridE[iE] << "\t" << gridL[iL] << "\t" << 
            gridJr  (iE, iL) << "\t" << gridJrdE(iE, iL) << "\t" << gridJrdL(iE, iL) << "\t" <<
            (iE>0 ? (gridJr(iE, iL)-gridJr(iE-1, iL)) / (gridE[iE]-gridE[iE-1]) : NAN) << "\t" <<
            (iL>0 ? (gridJr(iE, iL)-gridJr(iE, iL-1)) / (gridL[iL]-gridL[iL-1]) : NAN) << "\n";
        }
        strm<<"\n";
    }
    strm.close();
#endif

    // disappointingly, the use of derivatives for quintic spline interpolation does not seem
    // to give better accuracy than cubic interpolation...
    return math::CubicSpline2d(gridE, gridL, gridJr/*, gridJrdE, gridJrdL*/);
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
    if(!isFinite(acts.Jr))  // E>=0
        return ActionAngles(acts, Angles(NAN, NAN, NAN));
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
    interp(potential), intJr(makeActionInterpolator(interp)) {}

double ActionFinderSpherical::Jr(double E, double L, double *Omegar, double *Omegaz) const
{
    bool needDeriv = Omegar!=NULL || Omegaz!=NULL;
    double val, derE, derZ, dLcdE;
    double Lc = interp.pot.L_circ(E, needDeriv? &dLcdE : NULL);
    double Z  = Lc>0 ? fmin(fabs(L/Lc), 1) : 0;
    intJr.evalDeriv(E, Z, &val, needDeriv? &derE : NULL, needDeriv? &derZ : NULL);
    if(needDeriv) {
        double dJrdL = derZ * (1-Z) - val;
        double dJrdE = derE * (1-Z) * Lc - (derZ * (1-Z) * Z - val) * dLcdE;
        if(Omegar)
            *Omegar  = 1 / dJrdE;
        if(Omegaz)
            *Omegaz  = -dJrdL / dJrdE;
    }
    return val * Lc * (1-Z);
}

Actions ActionFinderSpherical::actions(const coord::PosVelCyl& point) const
{
    Actions acts;
    double E  = interp.pot.value(sqrt(pow_2(point.R) + pow_2(point.z))) + 
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
    double E  = interp.pot.value(sqrt(pow_2(point.R) + pow_2(point.z))) + 
        0.5 * (pow_2(point.vR) + pow_2(point.vz) + pow_2(point.vphi));
    double L  = Ltotal(point);
    double Omegar, Omegaz;
    acts.Jphi = Lz(point);
    acts.Jz   = point.z==0 && point.vz==0 ? 0 : fmax(0, L - fabs(acts.Jphi));
    acts.Jr   = Jr(E, L, &Omegar, &Omegaz);
    if(freq)
        *freq = Frequencies(Omegar, Omegaz, Omegaz * math::sign(acts.Jphi));
    double R1, R2;
    interp.findPlanarOrbitExtent(E, L, R1, R2);
    Angles angs = computeAngles(point, interp.pot, E, L, R1, R2, Omegar, Omegaz);
    return ActionAngles(acts, angs);
}

double ActionFinderSpherical::E(const Actions& acts) const
{
    double L = acts.Jz + fabs(acts.Jphi);  // total angular momentum
    // radius of a circular orbit with this angular momentum
    double rcirc = interp.pot.R_from_Lz(L);
    // initial guess (more precisely, lower bound) for Hamiltonian
    double Ecirc = interp.pot(rcirc) + (L>0 ? 0.5 * pow_2(L/rcirc) : 0);
    // find E such that Jr(E, L) equals the target value
    HamiltonianFinderFncInt fnc(*this, acts.Jr, L, Ecirc, 0);
    return math::findRoot(fnc, Ecirc, 0, ACCURACY_JR);
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
    interp.findPlanarOrbitExtent(E, L, R1, R2);
    // map the point from action/angles and frequencies
    coord::PosVelSphMod p0 = mapPointFromActionAngles(
        aa, interp.pot, E, L, R1, R2, Omegar, Omegaz);
    if(derivAct) {
        // use the fact that dE/dJr = Omega_r, dE/dJz = Omega_z, etc, and find dR{1,2}/dJ{r,z}
        double dPhidR;
        interp.pot.evalDeriv(R1, NULL, &dPhidR);
        double factR1 = pow_2(L) / pow_3(R1) - dPhidR;
        interp.pot.evalDeriv(R2, NULL, &dPhidR);
        double factR2 = pow_2(L) / pow_3(R2) - dPhidR;
        double dR1dJr = -Omegar / factR1;
        double dR2dJr = -Omegar / factR2;
        double dR1dJz = (L / pow_2(R1) - Omegaz) / factR1;
        double dR2dJz = (L / pow_2(R2) - Omegaz) / factR2;
        // compute the derivs using finite-difference (silly approach, no error control)
        double EPS= 1e-8;   // no proper scaling attempted!!
        derivAct->dbyJr = derivPointFromActions(
            ActionAngles(Actions(aa.Jr + EPS, aa.Jz, aa.Jphi), aa), p0, EPS, *this, interp,
            E + EPS * Omegar, R1 + EPS * dR1dJr, R2 + EPS * dR2dJr);
        derivAct->dbyJz = derivPointFromActions(
            ActionAngles(Actions(aa.Jr, aa.Jz + EPS, aa.Jphi), aa), p0, EPS, *this, interp,
            E + EPS * Omegaz, R1 + EPS * dR1dJz, R2 + EPS * dR2dJz);
        derivAct->dbyJphi = derivPointFromActions(
            ActionAngles(Actions(aa.Jr, aa.Jz, aa.Jphi + EPS), aa), p0, EPS, *this, interp,
            E + EPS * Omegaphi, R1 + EPS * dR1dJz * math::sign(aa.Jphi),
            R2 + EPS * dR2dJz * math::sign(aa.Jphi));  // dX/dJphi = dX/dJz*sign(Jphi)
    }
    return p0;
}

}  // namespace actions
