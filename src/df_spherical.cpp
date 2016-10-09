#include "df_spherical.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <algorithm>
// debugging
#include <fstream>

namespace df {
namespace{

/// relative accuracy in computing the integrals for phase-space volume
static const double ACCURACY = 1e-10;

/// default grid spacing in log radius or log phase volume
static const double DELTALOG = 0.125;

// required tolerance on the 2nd deriv to declare the asymptotic limit
static const double EPS2DER  = 1e-6;

/// helper function to find a root of fnc(x)=val  (TODO: augment math::findRoot with this feature!)
class RootFinder: public math::IFunction {
    const math::IFunction& fnc;
    double val;
public:
    RootFinder(const math::IFunction& _fnc, double _val) : fnc(_fnc), val(_val) {}
    virtual void evalDeriv(const double x, double *v, double *d, double *dd) const {
        fnc.evalDeriv(x, v, d, dd);
        if(v)
            *v -= val;
    }
    virtual unsigned int numDerivs() const { return fnc.numDerivs(); }
};

/// integrand for computing phase volume and density of states:
/// int_0^{rmax} dr r^2 v^n(E,r)  is transformed into
/// rmax^3 * int_0^1 ds rs(s)^2 v^n(E,rmax*rs) drs/ds, where rs=(3-2s)s^2;
/// this improves accuracy by regularizing the integrand at endpoints.
class PotIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction& pot;
    double rmax, n, E;
public:
    PotIntegrand(const math::IFunction& _pot, double _rmax, double _n) :
        pot(_pot), rmax(_rmax), n(_n), E(pot(rmax)) {}
    virtual double value(const double s) const {
        double rs = s*s*(3-2*s);
        return 6*s*(1-s) * pow_2(rs) * pow(fmax(0, E - pot(rmax*rs)), n);
    }
};

/// different regimes for calculation of various integrals involving f(h)
typedef enum { MODE_INTF, MODE_INTFG, MODE_INTFH, MODE_INTJ1, MODE_INTJ3 } Operation;

/// integrand for computing the product of f(h) and some other function, depending on the mode
template<Operation mode>
class DFIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction &df;
    const PhaseVolume &pv;
    const double logh0;
public:
    DFIntegrand(const math::IFunction& _df, const PhaseVolume& _pv, double _logh0=0) :
        df(_df), pv(_pv), logh0(_logh0) {}
    virtual double value(const double logh) const {
        double g, h = exp(logh);
        double deltaE = pv.deltaE(logh, logh0, &g);
        double val =   // the value of weighting function in the integrand
            mode==MODE_INTF  ? 1 :
            mode==MODE_INTFG ? g :
            mode==MODE_INTFH ? h :
            mode==MODE_INTJ1 ? sqrt(deltaE) :
            mode==MODE_INTJ3 ? pow_3(sqrt(deltaE)) : NAN;
        // the original integrals are formulated in terms of  \int f(E) val(E) dE,
        // and we replace  dE  by  d(log h) * [ dh / d(log h) ] / [ dh / dE ],
        // that's why there are extra factors h and 1/g below.
        return df(h) * h / g * val;
    }
};

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
static inline void scaleE(const double E, const double invPhi0,
    /*output*/ double& scaledE, double& dEdscaledE)
{
    double expE = invPhi0 - 1/E;
    scaledE     = log(expE);
    dEdscaledE  = E * E * expE;
}

/// return E and dE/d(scaledE) as functions of scaledE
static inline void unscaleE(const double scaledE, const double invPhi0,
    /*output*/ double& E, double& dEdscaledE)
{
    double expE = exp(scaledE);
    E           = 1 / (invPhi0 - expE);
    dEdscaledE  = E * E * expE;
}

/// same as above, but for two separate values of E1 and E2;
/// in addition, compute the difference between E1 and E2 in a way that is not prone
/// to cancellation when both E1 and E2 are close to Phi0 and the latter is finite.
static inline void unscaleDeltaE(const double scaledE1, const double scaledE2, const double invPhi0,
    /*output*/ double& E1, double& E2, double& E1minusE2, double& dE1dscaledE1)
{
    double exp1  = exp(scaledE1);
    double exp2  = exp(scaledE2);
    E1           = 1 / (invPhi0 - exp1);
    E2           = 1 / (invPhi0 - exp2);
    E1minusE2    = (exp1 - exp2) * E1 * E2;
    dE1dscaledE1 = E1 * E1 * exp1;
}

}  // internal namespace

//---- Correspondence between h and E ----//

PhaseVolume::PhaseVolume(const math::IFunction& pot)
{
    double Phi0 = pot(0);
    if(!(Phi0<0))
        throw std::invalid_argument("PhaseVolume: invalid value of Phi(r=0)");
    invPhi0 = 1/Phi0;

    // TODO: make the choice of initial radius more scale-invariant!
    const double logRinit = 0; // initial value of log radius (rather arbitrary but doesn't matter)
    const int NUM_ARRAYS = 3;  // 1d arrays of various quantities:
    std::vector<double> grids[NUM_ARRAYS];
    std::vector<double>   // assign a proper name to each of these arrays:
    &gridE  =grids[0],    // scaled E=Phi(r)
    &gridH  =grids[1],    // log(h(E))
    &gridG  =grids[2];    // log(g(E)), where g=dh/dE

    double logR = logRinit;
    int   stage = 0;   // 0 means scan inward, 1 - outward, 2 - done
    while(stage<2) {   // first scan inward in radius, then outward, then stop
        double R = exp(logR);
        double E = pot(R);
        double G = math::integrate(PotIntegrand(pot, R, 0.5), 0, 1, ACCURACY);
        double H = math::integrate(PotIntegrand(pot, R, 1.5), 0, 1, ACCURACY);
        double scaledE, dEdscaledE;
        scaleE(E, invPhi0, scaledE, dEdscaledE);
        gridE.push_back(scaledE);
        gridH.push_back(log(H) + log(16*M_PI*M_PI/3*2*M_SQRT2) + 3*logR);
        gridG.push_back(1.5*G/H * dEdscaledE);

        // check if we have reached an asymptotic regime,
        // by examining the curvature (2nd derivative) of relation between scaled H and E.
        unsigned int np = gridE.size();
        double dlogR = DELTALOG;
        if(np>=3 && fabs(logR - logRinit)>=2) {
            double der2 = math::deriv2(gridE[np-3], gridE[np-2], gridE[np-1],
                gridH[np-3], gridH[np-2], gridH[np-1], gridG[np-3], gridG[np-2], gridG[np-1]);
            // check if converged, or if the covered range of radii is too large (>1e6)
            if(fabs(der2) < EPS2DER || fabs(logR - logRinit)>=15) {
                if(stage==0) {   // we've been assembling the arrays inward, now need to reverse them
                    for(int i=0; i<NUM_ARRAYS; i++)
                        std::reverse(grids[i].begin(), grids[i].end());
                }
                logR = logRinit;  // restart from the middle
                ++stage;          // switch direction in scanning, or finish
            } else {
                // if we are close to the asymptotic regime but not yet there, we may afford to increase
                // the spacing between grid nodes without deteriorating the accuracy of interpolation
                if(fabs(der2) < EPS2DER*10)
                    dlogR *= 4;
                else if(fabs(der2) < EPS2DER*100)
                    dlogR *= 2;
            }
        }
        if(stage==0)
            logR -= dlogR;
        else
            logR += dlogR;
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

double PhaseVolume::E(const double h, double* g) const
{
    if(h==0) {
        if(g) *g=0;
        return invPhi0 == 0 ? -INFINITY : 1/invPhi0;
    }
    if(h==INFINITY) {
        if(g) *g=INFINITY;
        return 0;
    }
    double scaledE, dEdscaledE, realE;
    EofH.evalDeriv(log(h), &scaledE, g);
    unscaleE(scaledE, invPhi0, realE, dEdscaledE);
    if(g)
        *g = h / *g / dEdscaledE;
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


//---- Distribution functions f(h) ----//

SphericalIsotropic::SphericalIsotropic(const std::vector<double>& gridh, const std::vector<double>& gridf,
    double slopeIn, double slopeOut)
{
    if(gridh.size() != gridf.size())
        throw std::invalid_argument("SphericalIsotropic: array lengths are not equal");
    std::vector<double> sh(gridh.size()), sf(gridf.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        sh[i] = log(gridh[i]);
        sf[i] = log(gridf[i]);
        if((i>0 && sh[i]<=sh[i-1]) || !isFinite(sf[i]+sh[i]))
            throw std::invalid_argument("SphericalIsotropic: incorrect input data");
    }
    // construct the spline, optionally with the user-provided endpoint derivatives:
    // f(h) ~ h^slopeIn, df/dh = slopeIn * f / h = d(log f) / d(log h) * f / h
    spl = math::CubicSpline(sh, sf, slopeIn, slopeOut);
    // check correctness of asymptotic behaviour
    double der;
    spl.evalDeriv(sh.front(), NULL, &der);
    if(!(der > -1))
        throw std::runtime_error("SphericalIsotropic: f(h) rises too steeply as h-->0");
    spl.evalDeriv(sh.back(), NULL, &der);
    if(!(der < -1))
        throw std::runtime_error("SphericalIsotropic: f(h) falls too slowly as h-->infinity");
}

double SphericalIsotropic::value(const double h) const
{
    if(!(h>0))
        return 0;
    return exp(spl(log(h)));
}


SphericalIsotropic makeEddingtonDF(const math::IFunction& /*density*/, const math::IFunction& /*potential*/)
{
    throw std::runtime_error("makeEddingtonDF not implemented");
}

SphericalIsotropic fitSphericalDF(
    const std::vector<double>& hvalues, const std::vector<double>& masses, unsigned int gridSize)
{
    const unsigned int nbody = hvalues.size();
    if(masses.size() != nbody)
        throw std::invalid_argument("fitSphericalDF: array sizes are not equal");

    // 1. collect the log-scaled values of phase volume
    std::vector<double> logh(nbody);
    for(unsigned int i=0; i<nbody; i++) {
        logh[i] = log(hvalues[i]);
        if(!isFinite(logh[i]+masses[i]) || masses[i]<0)
            throw std::invalid_argument("fitSphericalDF: incorrect input data");
    }

    // 2. create a reasonable grid in log(h), with almost uniform spacing subject to the condition
    // that each segment contains at least "a few" particles (weakly depending on their total number)
    const int minParticlesPerBin  = std::max(1, static_cast<int>(log(nbody+1)/log(2)));
    std::vector<double> gridh = math::createAlmostUniformGrid(gridSize+2, logh, minParticlesPerBin);
    utils::msg(utils::VL_DEBUG, "fitSphericalDF",
        "Grid in h=["+utils::toString(exp(gridh[1]))+":"+utils::toString(exp(gridh[gridh.size()-2]))+"]"
        ", particles span h=["+utils::toString(exp(gridh[0]))+":"+utils::toString(exp(gridh.back()))+"]");
    gridh.erase(gridh.begin());
    gridh.pop_back();

    // 3a. perform spline log-density fit, and
    // 3b. initialize a cubic spline for log(f) as a function of log(h)
    math::CubicSpline fitfnc(gridh,
        math::splineLogDensity<3>(gridh, logh, masses,
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));

    // 4. store the values of cubic spline at grid nodes, together with two endpoint derivatives --
    // this data is sufficient to reconstruct it exactly later in the SphericalIsotropic constructor
    double slopeIn, slopeOut;
    fitfnc.evalDeriv(gridh.front(), NULL, &slopeIn);
    fitfnc.evalDeriv(gridh.back(),  NULL, &slopeOut);
    assert(slopeIn > 0 && slopeOut < 0);  // a condition for a valid fit (total mass should be finite)
    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h = exp(gridh[i]);
        // the fit provides log( dM/d(log h) ) = log( h dM/dh ) = log( h f(h) )
        gridf[i] = exp(fitfnc(gridh[i])) / h;
        gridh[i] = h;
    }
    // additional "-1" comes from the fact that slopes were computed as d [log (h f(h) ) ] / d [log h],
    // and the SphericalIsotropic interpolator takes the slopes for d[ log f(h) ] / d [log h]
    slopeIn  -= 1;
    slopeOut -= 1;

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("dffit");
        strm << "#slope_in\t" << slopeIn << "\n#slope_out\t" << slopeOut << '\n';
        for(unsigned int i=0; i<gridh.size(); i++)
            strm << gridh[i] << '\t' << gridf[i] << '\n';
    }

    // 5. construct an interpolating spline that matches exactly our fitfnc (it's also a cubic spline
    // in the same scaled variables), including the correct slopes for extrapolation outside the grid
    return SphericalIsotropic(gridh, gridf, slopeIn, slopeOut);
}


DiffusionCoefs::DiffusionCoefs(const PhaseVolume& _phasevol, const math::IFunction& df) :
    phasevol(_phasevol)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in X = log[h(Phi)] and Y = log[h(E)/h(Phi)]
    const double logHmin         = phasevol.logHmin(),  logHmax = phasevol.logHmax();
    const unsigned int npoints   = static_cast<unsigned int>(fmax(100, (logHmax-logHmin)/0.5));
    std::vector<double> gridLogH = math::createUniformGrid(npoints, logHmin, logHmax);
    const unsigned int npointsY  = 100;
    const double mindeltaY       = fmin(0.1, (logHmax-logHmin)/npointsY);
    std::vector<double> gridY    = math::createNonuniformGrid(npointsY, mindeltaY, logHmax-logHmin, true);

    // 2. store the values of f, g, h at grid nodes
    std::vector<double> gridF(npoints), gridG(npoints), gridH(npoints);
    std::vector<double> gridFint(npoints), gridFGint(npoints), gridFHint(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridH[i] = exp(gridLogH[i]);
        gridF[i] = df(gridH[i]);
        phasevol.E(gridH[i], &gridG[i]);
    }

    // 3a. determine the asymptotic behaviour of f(h):
    // f(h) ~ h^outerFslope as h-->inf  or  h^innerFslope as h-->0
    double innerFslope = log(gridF[1] / gridF[0]) / (gridLogH[1] - gridLogH[0]);
    double outerFslope = log(gridF[npoints-1] / gridF[npoints-2]) /
        (gridLogH[npoints-1] - gridLogH[npoints-2]);
    if(!(innerFslope > -1))
        throw std::runtime_error("DiffusionCoefs: f(h) rises too rapidly as h-->0");
    if(!(outerFslope < -1))
        throw std::runtime_error("DiffusionCoefs: f(h) falls off too slowly as h-->infinity");

    // 3b. determine the asymptotic behaviour of h(E), or rather, g(h) = dh/dE:
    // -E ~ h^outerEslope  and  g(h) ~ h^(1-outerEslope)  as  h-->inf,
    // and in the nearly Keplerian potential at large radii outerEslope should be ~ -2/3.
    // -E ~ h^innerEslope + const  and  g(h) ~ h^(1-innerEslope)  as  h-->0:
    // if innerEslope<0, Phi(r) --> -inf as r-->0, and we assume that |innerE| >> const;
    // otherwise Phi(0) is finite, and we assume that  innerE-Phi(0) << |Phi(0)|.
    // in general, if Phi ~ r^n + const at small r, then innerEslope = 2n / (6+3n);
    // innerEslope ranges from -2/3 for a Kepler potential to ~0 for a logarithmic potential,
    // to +1/3 for a harmonic (constant-density) core.
    double Phi0   = phasevol.E(0);  // Phi(r=0), may be -inf
    double innerE = phasevol.E(gridH.front());
    double outerE = phasevol.E(gridH.back());
    if(!(Phi0 < innerE && innerE < outerE && outerE < 0))
        throw std::runtime_error("DiffusionCoefs: weird behaviour of potential");
    if(Phi0 != -INFINITY)   // determination of inner slope depends on whether the potential is finite
        innerE -= Phi0;
    double innerEslope = gridH.front() / gridG.front() / innerE;
    double outerEslope = gridH.back()  / gridG.back()  / outerE;
    double outerRatio  = outerFslope  / outerEslope;
    if(!(outerRatio > 0 && innerEslope + innerFslope > -1))
        throw std::runtime_error("DiffusionCoefs: weird asymptotic behaviour of phase volume");

    // 4. construct 1d interpolating splines for integrals of f(E) dE, f(E) g(E) dE, and f(E) h(E) dE

    // 4a. integral of f(h) dE = f(h) / g(h) dh -- compute from outside in,
    // summing contributions from all intervals of h above its current value
    DFIntegrand<MODE_INTF> dfint(df, phasevol);
    // the outermost segment from h_max to infinity is integrated analytically
    gridFint.back() = -gridF.back() * outerE / (1 + outerRatio);
    for(int i=npoints-1; i>=1; i--) {
        gridFint[i-1] = gridFint[i] + math::integrate(dfint, gridLogH[i-1], gridLogH[i], ACCURACY);
    }

    // 4b. integrands of f*g dE  and  f*h dE;  note that g = dh/dE.
    // compute from inside out, summing contributions from all previous intervals of h
    DFIntegrand<MODE_INTFG> dfgint(df, phasevol);
    DFIntegrand<MODE_INTFH> dfhint(df, phasevol);
    // integrals over the first segment (0..gridH[0]) are computed analytically
    gridFGint[0] = gridF[0] * gridH[0] / (1 + innerFslope);
    gridFHint[0] = gridF[0] * pow_2(gridH[0]) / gridG[0] / (1 + innerEslope + innerFslope);
    for(unsigned int i=1; i<npoints; i++) {
        gridFGint[i] = gridFGint[i-1] + math::integrate(dfgint, gridLogH[i-1], gridLogH[i], ACCURACY);
        gridFHint[i] = gridFHint[i-1] + math::integrate(dfhint, gridLogH[i-1], gridLogH[i], ACCURACY);
    }
    // add the contribution of integrals from the last grid point up to infinity (very small anyway)
    gridFGint.back() -= gridF.back() * gridH.back() / (1 + outerFslope);
    gridFHint.back() -= gridF.back() * pow_2(gridH.back()) / gridG.back() / (1 + outerEslope + outerFslope);
    totalMass = gridFGint.back();  // cache the value of total mass which is used in evalLocal/evalOrbitAvg

    // 4c. log-scale the computed values and prepare derivatives for quintic spline
    std::vector<double> gridFder(npoints), gridFGder(npoints), gridFHder(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridFder[i]  = gridH[i] / -gridFint[i] * gridF[i] / gridG[i];
        gridFGder[i] = gridH[i] / gridFGint[i] * gridF[i];
        gridFHder[i] = gridH[i] / gridFHint[i] * gridF[i] * gridH[i] / gridG[i];
        gridFint[i]  = log(gridFint[i]);
        gridFGint[i] = log(gridFGint[i]);
        gridFHint[i] = log(gridFHint[i]);
        if(!(gridFder[i]<=0 && gridFGder[i]>=0 && gridFHder[i]>=0 && 
            isFinite(gridFint[i] + gridFGint[i] + gridFHint[i])))
            throw std::runtime_error("DiffusionCoefs: cannot construct valid interpolators");
    }
    // integrals of f*g and f*h have finite limit as h-->inf;
    // extrapolate them as constants beyond the last grid point
    gridFGder.back() = gridFHder.back() = 0;

    // 4d. initialize splines for log-scaled integrals
    intf  = math::QuinticSpline(gridLogH, gridFint,  gridFder);
    intfg = math::QuinticSpline(gridLogH, gridFGint, gridFGder);
    intfh = math::QuinticSpline(gridLogH, gridFHint, gridFHder);

    // 5. construct 2d interpolating splines for dv2par, dv2per as functions of Phi and E

    // 5a. asymptotic values for J1/J0 and J3/J0 as Phi --> 0 and (E/Phi) --> 0
    double outerJ1 = 0.5*M_SQRTPI * math::gamma(2 + outerRatio) / math::gamma(2.5 + outerRatio);
    double outerJ3 = outerJ1 * 1.5 / (2.5 + outerRatio);

    // 5b. compute the values of J1/J0 and J3/J0 at nodes of 2d grid in X=log(h(Phi)), Y=log(h(E)/h(Phi))
    math::Matrix<double> gridv2par(npoints, npointsY), gridv2per(npoints, npointsY);
    for(unsigned int i=0; i<npoints; i++)
    {
        // The first coordinate of the grid is X = log(h(Phi)), the second is Y = log(h(E)) - X.
        // For each pair of values of X and Y, we compute the following integrals:
        // J_n = \int_\Phi^E f(E') [(E'-\Phi) / (E-\Phi)]^{n/2}  dE';  n = 0, 1, 3.
        // Then the value of 2d interpolants are assigned as
        // \log[ J3 / J0 ], \log[ (3*J1-J3) / J0 ] .
        // In practice, we replace the integration over dE by integration over dy = d(log h),
        // and accumulate the values of modified integrals sequentially over each segment in Y.
        // Here the modified integrals are  J{n}acc = \int_X^Y f(y) (dE'/dy) (E'(y)-\Phi)^{n/2}  dy,
        // i.e., without the term [E(Y,X)-\Phi(X)]^{n/2} in the denominator,
        // which is invoked later when we assign the values to the 2d interpolants.
        double J0acc = 0, J1acc = 0, J3acc = 0;  // accumulators
        DFIntegrand<MODE_INTF>  intJ0(df, phasevol, gridLogH[i]);
        DFIntegrand<MODE_INTJ1> intJ1(df, phasevol, gridLogH[i]);
        DFIntegrand<MODE_INTJ3> intJ3(df, phasevol, gridLogH[i]);
        gridv2par(i, 0) = log(2./5);  // analytic limiting values for Phi=E
        gridv2per(i, 0) = log(8./5);
        for(unsigned int j=1; j<npointsY; j++) {
            double logHprev = gridLogH[i] + gridY[j-1];
            double logHcurr = gridLogH[i] + gridY[j];
            if(j==1) {
                // integration over the first segment uses a more accurate quadrature rule
                // to accounting for a possible endpoint singularity at Phi=E
                J0acc = math::integrate(math::ScaledIntegrandEndpointSing(
                    intJ0, logHprev, logHcurr), 0, 1, ACCURACY);
                J1acc = math::integrate(math::ScaledIntegrandEndpointSing(
                    intJ1, logHprev, logHcurr), 0, 1, ACCURACY);
                J3acc = math::integrate(math::ScaledIntegrandEndpointSing(
                    intJ3, logHprev, logHcurr), 0, 1, ACCURACY);
            } else {
                J0acc += math::integrate(intJ0, logHprev, logHcurr, ACCURACY);
                J1acc += math::integrate(intJ1, logHprev, logHcurr, ACCURACY);
                J3acc += math::integrate(intJ3, logHprev, logHcurr, ACCURACY);
            }
            if(i==npoints-1) {
                // last row: analytic limiting values for Phi-->0 and any E/Phi
                double EoverPhi = exp(gridY[j] * outerEslope);  // strictly < 1
                double oneMinusJ0overI0 = pow(EoverPhi, 1+outerRatio);  // < 1
                double Fval1 = math::hypergeom2F1(-0.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double Fval3 = math::hypergeom2F1(-1.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double I0    = exp(intf(gridLogH[i]));
                double sqPhi = sqrt(-phasevol.E(gridH[i]));
                if(isFinite(Fval1+Fval3)) {
                    J0acc = I0 * (1 - oneMinusJ0overI0);
                    J1acc = I0 * (outerJ1 - oneMinusJ0overI0 * Fval1) * sqPhi;
                    J3acc = I0 * (outerJ3 - oneMinusJ0overI0 * Fval3) * pow_3(sqPhi);
                } else {
                    // this procedure sometimes fails, since hypergeom2F1 is not very robust;
                    // in this case we simply keep the values computed by numerical integration
                    utils::msg(utils::VL_WARNING, "DiffusionCoefs", "Can't compute asymptotic value");
                }
            }
            double dv = sqrt(phasevol.deltaE(logHcurr, gridLogH[i]));
            double J1overJ0 = J1acc / J0acc / dv;
            double J3overJ0 = J3acc / J0acc / pow_3(dv);
            if(J1overJ0<=0 || J3overJ0<=0 || !isFinite(J1overJ0+J3overJ0)) {
                utils::msg(utils::VL_WARNING, "DiffusionCoefs", "Invalid value"
                    "  J0="+utils::toString(J0acc)+
                    ", J1="+utils::toString(J1acc)+
                    ", J3="+utils::toString(J3acc));
                J1overJ0 = 2./3;   // fail-safe values corresponding to E=Phi
                J3overJ0 = 2./5;
            }
            gridv2par(i, j) = log(J3overJ0);
            gridv2per(i, j) = log(3 * J1overJ0 - J3overJ0);
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("diffcoefs");
        for(unsigned int i=0; i<npoints; i++) {
            double Phi = phasevol.E(gridH[i]);
            for(unsigned int j=0; j<npointsY; j++) {
                double E = phasevol.E(exp(gridLogH[i] + gridY[j]));
                strm << utils::pp(gridLogH[i],10) +' '+ utils::pp(gridY[j],10) +'\t'+
                utils::pp(Phi,14) +' '+ utils::pp(E,14) +'\t'+
                utils::pp(exp(gridv2par(i, j)),10) +' '+ utils::pp(exp(gridv2per(i, j)),10)+'\n';
            }
            strm << '\n';
        }
    }

    // 5c. construct the 2d splines
    intv2par = math::CubicSpline2d(gridLogH, gridY, gridv2par);
    intv2per = math::CubicSpline2d(gridLogH, gridY, gridv2per);
}

void DiffusionCoefs::evalOrbitAvg(double E, double &DE, double &DEE) const
{
    double h, g;
    phasevol.evalDeriv(E, &h, &g);
    double
    logh = log(h),
    IF   = exp(intf(logh)),
    IFG  = exp(intfg(logh)),
    IFH  = exp(intfh(logh));
    DE   = 16*M_PI*M_PI * totalMass * (IF - IFG / g);
    DEE  = 32*M_PI*M_PI * totalMass * (IF * h + IFH) / g;
}

void DiffusionCoefs::evalLocal(double Phi, double E, double &dvpar, double &dv2par, double &dv2per) const
{
    double loghPhi = log(phasevol(Phi));
    double loghE   = log(phasevol(E));
    if(!(Phi<0 && loghE >= loghPhi))
        throw std::invalid_argument("DiffusionCoefs: incompatible values of E and Phi");

    // compute the 1d interpolators for I0, J0
    double I0 = exp(intf(loghE));
    double J0 = fmax(exp(intf(loghPhi)) - I0, 0);
    // restrict the arguments of 2d interpolators to the range covered by their grids
    double X = fmin(fmax(loghPhi, intv2par.xmin()), intv2par.xmax());
    double Y = fmin(fmax(loghE-loghPhi, intv2par.ymin()), intv2par.ymax());
    // compute the 2d interpolators for J1, J3
    double v2par = exp(intv2par.value(X, Y)) * J0;
    double v2per = exp(intv2per.value(X, Y)) * J0;
    if(E>=0) {  // in this case, the coefficients were computed for E=0, need to scale them to E>0
        double J1 = (v2par + v2per) / 3;
        double corr = 1 / sqrt(1 - E / Phi);  // correction factor <1
        J1    *= corr;
        v2par *= pow_3(corr);
        v2per  = 3 * J1 - v2par;
    }
    double mult = 32*M_PI*M_PI/3 * totalMass;
    dvpar  = -mult * (v2par + v2per);
    dv2par =  mult * (v2par + I0);
    dv2per =  mult * (v2per + I0 * 2);
    /*if(loghPhi<X)
        utils::msg(utils::VL_WARNING, "DiffusionCoefs",
        "Extrapolating to small h: log(h(Phi))="+utils::toString(loghPhi)+
        ", log(h(E))="+utils::toString(loghE)+
        ", I0="+utils::toString(I0)+", J0="+utils::toString(J0));*/
}

double DiffusionCoefs::cumulMass(const double h) const
{
    return exp(intfg(log(h)));
}

double DiffusionCoefs::findh(const double cm) const
{
    // solve the relation intfg(log(h)) = log(cm)  to find h for the given cm (cumulative mass)
    if(cm==0)
        return 0;
    double logcm = log(cm), loghmin = intfg.xmin(), loghmax = intfg.xmax();
    if(logcm > intfg(loghmax))
        return INFINITY;
    double valmin, dermin;
    intfg.evalDeriv(loghmin, &valmin, &dermin);
    if(logcm <= valmin) {
        // find the root (logh) using linear extrapolation:
        // log(cm) = intfg(logh) = valmin + dermin * (logh - loghmin)
        return exp((logcm - valmin) / dermin + loghmin);
    }
    return exp(findRoot(RootFinder(intfg, logcm), loghmin, loghmax, ACCURACY));
}


std::vector<double> sampleSphericalDF(const DiffusionCoefs& model, unsigned int npoints)
{
    std::vector<double> result(npoints);
    double totalMass = model.cumulMass();
    for(unsigned int i=0; i<npoints; i++)
        result[i] = model.findh(totalMass * math::random());
    return result;
}

}; // namespace
