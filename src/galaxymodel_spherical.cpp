#include "galaxymodel_spherical.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_sample.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <fstream>

namespace galaxymodel{

namespace{

/// required tolerance for the root-finder
static const double EPSROOT  = 1e-6;

/// tolerance on the 2nd derivative of a function of phase volume for grid generation
/// ~ (DBL_EPSILON)^(1/3), because this is the typical accuracy of the finite-difference estimate
static const double EPSDER2  = ROOT3_DBL_EPSILON;

/// fixed order of Gauss-Legendre quadrature on each segment of the grid
static const int GLORDER  = 8;    // default value for all segments, or, alternatively, two values:
static const int GLORDER1 = 6;    // for shorter segments
static const int GLORDER2 = 10;   // for larger segments
/// the choice between short and long segments is determined by the segment length in log(h)
static const double GLDELTA = 0.7;  // ln(2)

/// lower limit on the value of density or DF to be considered seriously
/// (this weird-looking threshold takes into account possible roundoff errors
/// in converting the values to/from log-scaled ones, 
static const double MIN_VALUE_ROUNDOFF = 0.9999999999999e-100;

/// minimum relative difference between two adjacent values of potential (to reduce roundoff errors)
static const double MIN_REL_DIFFERENCE = 1e-12;

/** helper function for finding the slope of asymptotic power-law behaviour of a certain function:
    if  f(x) ~ f0 * (1 + A * x^B)  as  x --> 0  or  x --> infinity,  then the slope B is given by
    solving the equation  [x0^B - x1^B] / [x1^B - x2^B] = [f(x0) - f(x1)] / [f(x1) - f(x2)],
    where x0, x1 and x2 are three consecutive points near the end of the interval.
    The arrays of log(x) and corresponding f(x) are passed as parameters to this function,
    and its value() method is used in the root-finding routine.
*/
class SlopeFinder: public math::IFunctionNoDeriv {
    const double logx0, logx1, logx2, ratio;
public:
    SlopeFinder(double _logx0, double _logx1, double _logx2, double f0, double f1, double f2) :
        logx0(_logx0), logx1(_logx1), logx2(_logx2), ratio( (f0-f1) / (f1-f2) ) {}

    virtual double value(const double B) const {
        if(B==0)
            return (logx0 - logx1) / (logx1 - logx2) - ratio;
        if(B==INFINITY || B==-INFINITY)  // in either case, assume that the ratio of (x0/x1)^B vanishes
            return -ratio;
        double x0B = exp(B*logx0), x1B = exp(B*logx1), x2B = exp(B*logx2);
        return (x0B - x1B) / (x1B - x2B) - ratio;
    }
};

/// helper class for interpolating the density as a function of phase volume,
/// used in the Eddington inversion routine,
/// optionally with an asymptotic expansion for a constant-density core.
class DensityInterp: public math::IFunction {
    const math::CubicSpline spl;
    double logrho0, A, B, loghmin;

    // evaluate log(rho) and its derivatives w.r.t. log(h) using the asymptotic expansion
    void asympt(const double logh, /*output*/ double* logrho, double* dlogrho, double* d2logrho) const
    {
        double AhB = A * exp(B * logh);
        if(logrho)
            *logrho   = logrho0 + log(1 + AhB);
        if(dlogrho)
            *dlogrho  = B / (1 + 1 / AhB);
        if(d2logrho)
            *d2logrho = pow_2(B / (1 + AhB)) * AhB;
    }

public:
    // initialize the spline for log(rho) as a function of log(h),
    // and check if the density is approaching a constant value as h-->0;
    /// if it does, then determines the coefficients of its asymptotic expansion
    /// rho(h) = rho0 * (1 + A * h^B)
    DensityInterp(const std::vector<double>& logh, const std::vector<double>& logrho) :
        spl(logh, logrho),
        logrho0(-INFINITY), A(0), B(0), loghmin(-INFINITY)
    {
        if(logh.size()<4)
            return;
        // try determining the exponent B from the first three grid points
        double B0 = math::findRoot(
            SlopeFinder(logh[0], logh[1], logh[2], logrho[0], logrho[1], logrho[2]), 0.05, 20, EPSROOT);
        if(!isFinite(B0))
            return;
        // now try the same using three points shifted by one
        double B1 = math::findRoot(
            SlopeFinder(logh[1], logh[2], logh[3], logrho[1], logrho[2], logrho[3]), 0.05, 20, EPSROOT);
        if(!isFinite(B1))
            return;
        // consistency check - if the two values differ significantly, we're getting nonsense
        if(B0 < B1*0.95 || B1 < B0*0.95)
            return;
        B = B0;
        A = (logrho[0] - logrho[1]) / (exp(B*logh[0]) - exp(B*logh[1]));
        logrho0 = logrho[0] - A * exp(B*logh[0]);
        if(!isFinite(logrho0) || fabs(logrho[0]-logrho0) > 0.1)
            return;

        // now need to determine the critical value of log(h)
        // below which we will use the asymptotic expansion
        for(unsigned int i=1; i<logh.size(); i++) {
            loghmin = logh[i];
            double corelogrho, coredlogrho, cored2logrho;  // values returned by the asymptotic expansion
            asympt(logh[i], &corelogrho, &coredlogrho, &cored2logrho);
            if(!(fabs((corelogrho-logrho[i]) / (corelogrho-logrho0)) < 0.01)) {
                utils::msg(utils::VL_DEBUG, "makeEddingtonDF",
                    "Density core: rho="+utils::toString(exp(logrho0))+"*(1"+(A>0?"+":"")+
                    utils::toString(A)+"*h^"+utils::toString(B)+") at h<"+utils::toString(exp(loghmin)));
                break;   // TODO: come up with a more rigorous method...
            }
        }
    }

    /// returns the interpolated value for log(rho) as a function of log(h),
    /// together with its two derivatives.
    /// It automatically switches to the asymptotic expansion if necessary.
    virtual void evalDeriv(const double logh,
        /*output*/ double* logrho, double* dlogrho, double* d2logrho) const
    {
        if(logh < loghmin)
            asympt(logh, logrho, dlogrho, d2logrho);
        else
            spl.evalDeriv(logh, logrho, dlogrho, d2logrho);
    }

    virtual unsigned int numDerivs() const { return 2; }
};

/// integrand for computing the product of f(E) and a weight function (E-Phi)^{P/2}
template<int P>
class DFIntegrand: public math::IFunctionNoDeriv {
    const math::IFunction &df;
    const potential::PhaseVolume &pv;
    const double logh0;
public:
    DFIntegrand(const math::IFunction& _df, const potential::PhaseVolume& _pv, double _logh0) :
        df(_df), pv(_pv), logh0(_logh0) {}

    virtual double value(const double logh) const
    {
        double h = exp(logh), g, w = sqrt(pv.deltaE(logh, logh0, &g));
        // the original integrals are formulated in terms of  \int f(E) weight(E) dE,
        // and we replace  dE  by  d(log h) * [ dh / d(log h) ] / [ dh / dE ],
        // that's why there are extra factors h and 1/g below.
        return df(h) * h / g * math::pow(w, P);
    }
};


/** helper class for integrating or sampling the isotropic DF in a given spherical potential */
class DFSphericalIntegrand: public math::IFunctionNdim {
    const math::IFunction& pot;
    const math::IFunction& df;
    const potential::PhaseVolume pv;
public:
    DFSphericalIntegrand(const math::IFunction& _pot, const math::IFunction& _df) :
        pot(_pot), df(_df), pv(pot) {}

    /// un-scale r and v and return the jacobian of this transformation
    double unscalerv(double scaledr, double scaledv, double& r, double& v, double& Phi) const {
        double drds;
        r   = math::unscale(math::ScalingSemiInf(), scaledr, &drds);
        Phi = pot(r);
        double vesc = sqrt(-2*Phi);
        v   = scaledv * vesc;
        return pow_2(4*M_PI) * pow_2(r * vesc * scaledv) * vesc * drds;
    }

    virtual void eval(const double vars[], double values[]) const
    {
        double r, v, Phi;
        double jac = unscalerv(vars[0], vars[1], r, v, Phi);
        values[0] = 0;
        if(isFinite(jac) && jac>1e-100 && jac<1e100) {
            double f = df(pv(Phi + 0.5 * v*v));
            if(isFinite(f))
                values[0] = f * jac;
        }
    }
    virtual unsigned int numVars()   const { return 2; }
    virtual unsigned int numValues() const { return 1; }
};


/** helper class for setting up a grid for log(rho) in log(h) used in the Eddington inversion */
class LogRhoOfLogH: public math::IFunction {
    const math::LogLogScaledFnc loglogdensity;
    const potential::PhaseVolume& phasevol;
    const math::IFunction& pot;
    const double PhiMin;   ///< minimum allowed value of potential at the innermost grid node
public:
    LogRhoOfLogH(const math::IFunction& density,
        const potential::PhaseVolume& _phasevol, const math::IFunction& _pot) :
        loglogdensity(density), phasevol(_phasevol), pot(_pot), PhiMin(pot(0) * (1-MIN_REL_DIFFERENCE))
    {}

    virtual void evalDeriv(double logh, double* logrho, double* der, double* der2) const
    {
        double h = exp(logh), g, dgdh;
        double E = phasevol.E(h, &g, &dgdh);
        double r = potential::R_max(pot, E);
        double Phi, dPhidr, d2Phidr2;
        pot.evalDeriv(r, &Phi, &dPhidr, &d2Phidr2);
        if(r <= 0 || !isFinite(dPhidr + d2Phidr2) || Phi<PhiMin) {
            // this may happen if E is too close to Phi(0), in which case don't do anything
            if(logrho)
                *logrho=NAN;
            if(der)
                *der=NAN;
            if(der2)
                *der2=0;  // this is the only thing that is really needed in grid generation
            return;
        }
        math::PointNeighborhood lrho(loglogdensity, log(r));  // derivs of log(rho) w.r.t. log(r)
        // now we have the full transformation chain h -> E -> r -> rho,
        // with two derivatives at each stage, and will combine them to obtain
        // log(rho) and its derivatives w.r.t. log(h)
        double dlogrdlogh = h / (r * dPhidr * g);
        if(logrho)
            *logrho = lrho.f0;
        if(der)
            *der  = lrho.fder * dlogrdlogh;
        if(der2) {
            double d2logrdlogh2 = dlogrdlogh *
                (1 - dlogrdlogh - (d2Phidr2 / pow_2(dPhidr) + dgdh) * h / g);
            *der2 = lrho.fder2 * pow_2(dlogrdlogh) + lrho.fder * d2logrdlogh2;
        }
    }

    virtual unsigned int numDerivs() const { return 2; }
};

}  // internal namespace

//---- Eddington inversion ----//

void makeEddingtonDF(const math::IFunction& density, const math::IFunction& potential,
    /* input/output */ std::vector<double>& gridh, /* output */ std::vector<double>& gridf)
{
    // 1. construct a phase-volume interpolator
    potential::PhaseVolume phasevol(potential);

    // 2. prepare grids
    std::vector<double> gridlogh, gridPhi, gridlogrho;
    if(gridh.empty()) {   // no input: estimate the grid extent
        gridlogh = math::createInterpolationGrid(
            LogRhoOfLogH(density, phasevol, potential), EPSDER2);
    } else {              // input grid in h was provided
        gridlogh.resize(gridh.size());
        for(size_t i=0; i<gridh.size(); i++)
            gridlogh[i] = log(gridh[i]);
    }

    // 2b. store the values of h, Phi and rho at the nodes of a grid
    double prevPhi = potential(0);
    for(unsigned int i=0; i<gridlogh.size();) {
        double Phi = phasevol.E(exp(gridlogh[i]));
        double R   = potential::R_max(potential, Phi);
        double rho = density(R);
        // throw away grid nodes with invalid or negligible values of density,
        // and also nodes too closely so that the difference between adjacent potential values
        // is dominated by roundoff / cancellation errors
        if( isFinite(R) && R > 0 &&
            isFinite(rho) && rho > MIN_VALUE_ROUNDOFF &&
            Phi > prevPhi * (1-MIN_REL_DIFFERENCE))
        {
            gridPhi.push_back(Phi);
            gridlogrho.push_back(log(rho));
            i++;
            prevPhi = Phi;
        } else {
            gridlogh.erase(gridlogh.begin()+i);
        }
    }
    unsigned int gridsize = gridlogh.size();
    if(gridsize < 3)
        throw std::runtime_error("makeEddingtonDF: invalid grid size");
    gridf.resize(gridsize);   // f(h_i) = int_{h[i]}^{infinity}
    gridh.resize(gridsize);
    for(size_t i=0; i<gridsize; i++)
        gridh[i] = exp(gridlogh[i]);

    // 3. construct a spline for log(rho) as a function of log(h),
    // optionally with an asymptotic expansion in the case of a constant-density core
    DensityInterp densityInterp(gridlogh, gridlogrho);

    // 4. compute the integrals for f(h) at each point in the h-grid.
    // The conventional Eddington inversion formula is
    // f(E) = 1/(\sqrt{8}\pi^2)  \int_{E(h)}^0  d\Phi  (d^2\rho / d\Phi^2)  /  \sqrt{\Phi-E} ,
    // we replace the integration variable by \ln h, and the density is also expressed as
    // \rho(h), or, rather, \ln \rho ( \ln h ), in the form of a spline constructed above.
    // Therefore, the expression for the Eddington integral changes to a somewhat lengthier one:
    // f(h) = 1/(\sqrt{8}\pi^2)  \int_{\ln h}^\infty  d \ln h'  /  \sqrt{\Phi(h') - \Phi(h)} *
    // ( d2rho(h') +  drho(h') * ( (drho(h') - 1) g(h') / h'  + dg(h')/dh') ) * \rho(h'),
    // where  drho = d(\ln\rho) / d(\ln h), d2rho = d^2(\ln\rho) / d(\ln h)^2, computed at h'.
    // These integrals for each value of h on the grid are computed by summing up
    // contributions from all segments of h-grid above the current one,
    // and on each segment we use a Gauss-Legendre quadrature with GLORDER points.
    // We start from the last interval from h_max to infinity, where the integrals are computed
    // analytically, and then proceed to lower values of h, each time accumulating the contribution
    // of the current segment to all integrals which contain this segment.
    // As a final remark, the integration variable on each segment is not \ln h', but rather
    // an auxiliary variable y, defined such that \ln h = \ln h[i-1] + (\ln h[i] - \ln h[i-1]) y^2.
    // This eliminates the singularity in the term  1 / \sqrt{\Phi(h')-\Phi(h)}  when h=h[i-1]. 

    // 4a. integrate from log(h_max) = logh[gridsize-1] to infinity, or equivalently, 
    // from Phi(r_max) to 0, assuming that Phi(r) ~ -1/r and h ~ r^(3/2) in the asymptotic regime.
    // First we find the slope d(logrho)/d(logh) at h_max
    double logrho, dlogrho, d2logrho;
    densityInterp.evalDeriv(gridlogh.back(), &logrho, &dlogrho, &d2logrho);
    double slope = -1.5*dlogrho;  // density at large radii is ~ r^(-slope)
    utils::msg(utils::VL_DEBUG, "makeEddingtonDF",
        "Density is ~r^"+utils::toString(-slope)+" at large r");
    double mult  = 0.5 / M_SQRT2 / M_PI / M_PI;
    // next we compute analytically the values of all integrals for f(h_j) on the segment h_max..infinity
    for(unsigned int j=0; j<gridsize; j++) {
        double factor = j==gridsize-1 ? M_SQRTPI * (slope < 10 ?
                math::gamma(slope) / math::gamma(slope-0.5) : /*exact expr. overflows at large slope*/
                sqrt(slope)*(1-3./8./slope*(1+7./48./slope))  /*approx. better than 1e-5*/ ) :
            math::hypergeom2F1(0.5, slope-1, slope, gridPhi.back() / gridPhi[j]);
        if(isFinite(factor))
            gridf[j] = mult * slope * exp(logrho) / -gridPhi.back() * factor / sqrt(-gridPhi[j]);
        // do nothing (keep f=0) in case of arithmetic overflow (outer slope too steep)
    }

    // 4b. use pre-computed integration tables
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];

    // 4c. integrate from logh[i-1] to logh[i] for all i=1..gridsize-1
    for(unsigned int i=gridsize-1; i>0; i--)
    {
        for(int k=0; k<GLORDER; k++)
        {
            // node of Gauss-Legendre quadrature within the current segment (logh[i-1] .. logh[i]);
            // the integration variable y ranges from 0 to 1, and logh(y) is defined below
            double y = glnodes[k];
            double logh = gridlogh[i-1] + (gridlogh[i]-gridlogh[i-1]) * y*y;
            // compute E, g, dg/dh at the current point h (GL node)
            double h = exp(logh), g, dgdh;
            double E = phasevol.E(h, &g, &dgdh);
            // compute log[rho](log[h]) and its derivatives
            densityInterp.evalDeriv(logh, &logrho, &dlogrho, &d2logrho);
            // GL weight -- contribution of this point to each integral on the current segment,
            // taking into account the transformation of variable y -> logh
            double weight = glweights[k] * 2*y * (gridlogh[i]-gridlogh[i-1]);
            // common factor for all integrals - the derivative d^2\rho / d\Phi^2,
            // expressed in scaled and transformed variables
            double factor = (d2logrho * g / h + dlogrho * ( (dlogrho-1) * g / h + dgdh) ) * exp(logrho);
            // now add a contribution to the integral expressing f(h_j) for all h_j <= h[i-1]
            for(unsigned int j=0; j<i; j++) {
                double denom2 = E - gridPhi[j];
                if(denom2 < fabs(gridPhi[i]) * SQRT_DBL_EPSILON)  // loss of precision is possible:
                    denom2 = phasevol.deltaE(logh, gridlogh[j]);  // use a more accurate expression
                gridf[j] += mult * weight * factor / sqrt(denom2);
            }
        }
    }

    // 4.5 write out the results
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("MakeEddingtonDF.log");
        strm << "#h      \tf(h) as computed\n";
        for(unsigned int i=0; i<gridh.size(); i++)
            strm << utils::pp(gridh[i], 12) << '\t' << utils::pp(gridf[i],12) << '\n';
    }

    // 5. check validity and remove negative values
    bool hasNegativeF = false;
    for(unsigned int i=0; i<gridf.size();) {
        if(gridf[i] <= MIN_VALUE_ROUNDOFF) {
            hasNegativeF |= gridf[i]<0;
            gridf.erase(gridf.begin() + i);
            gridh.erase(gridh.begin() + i);
        } else
            i++;
    }
    if(hasNegativeF)
        utils::msg(utils::VL_WARNING, "makeEddingtonDF", "Distribution function is negative");
    if(gridf.size() < 2)
        throw std::runtime_error("makeEddingtonDF: could not construct a valid non-negative DF");

    // also check that the remaining positive values describe a DF that doesn't grow too fast as h-->0
    // and decays fast enough as h-->infinity;  this still does not guarantee that it's reasonable,
    // but at least allows to construct a valid interpolator
    do {
        math::LogLogSpline spl(gridh, gridf);
        double derIn, derOut;
        spl.evalDeriv(gridh.front(), NULL, &derIn);
        spl.evalDeriv(gridh.back(),  NULL, &derOut);
        double slopeIn  = gridh.front() / gridf.front() * derIn;
        double slopeOut = gridh.back()  / gridf.back()  * derOut;
        if(slopeIn > -1 && slopeOut < -1) {
            utils::msg(utils::VL_DEBUG, "makeEddingtonDF",
                "f(h) ~ h^"+utils::toString(slopeIn)+
                " at small h and ~ h^"+utils::toString(slopeOut)+" at large h");

            // write out the results once more
            if(utils::verbosityLevel >= utils::VL_VERBOSE) {
                std::ofstream strm("MakeEddingtonDF.log", std::ofstream::app);
                strm << "\n#h      \tf(h) final; inner slope="<< slopeIn <<", outer="<< slopeOut <<"\n";
                for(unsigned int i=0; i<gridh.size(); i++)
                    strm << utils::pp(gridh[i], 12) << '\t' << utils::pp(gridf[i],12) << '\n';
            }

            // results are returned in the two arrays, gridh and gridf
            return;
        }
        // otherwise remove the innermost and/or outermost point and repeat
        if(slopeIn < -1) {
            gridf.erase(gridf.begin());
            gridh.erase(gridh.begin());
        }
        if(slopeOut > -1) {
            gridf.erase(gridf.end()-1);
            gridh.erase(gridh.end()-1);
        }
    } while(gridf.size() > 2);
    throw std::runtime_error("makeEddingtonDF: could not construct a valid non-negative DF");
}


//---- Construction of f(h) from an N-body snapshot ----//

math::LogLogSpline fitSphericalDF(
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
    // this data is sufficient to reconstruct it exactly later in the LogLogSpline constructor
    double derLeft, derRight;
    fitfnc.evalDeriv(gridh.front(), NULL, &derLeft);
    fitfnc.evalDeriv(gridh.back(),  NULL, &derRight);
    assert(derLeft > 0 && derRight < 0);  // a condition for a valid fit (total mass should be finite)
    // endpoint derivatives of fitfnc are  d [log (h f(h) ) ] / d [log h] = 1 + d [log f] / d [log h],
    // so we subtract unity to obtain the logarithmic derivs of f(h)
    derLeft  -= 1;
    derRight -= 1;

    std::vector<double> gridf(gridh.size());
    for(unsigned int i=0; i<gridh.size(); i++) {
        double h = exp(gridh[i]);
        // the fit provides log( dM/d(log h) ) = log( h dM/dh ) = log( h f(h) )
        gridf[i] = exp(fitfnc(gridh[i])) / h;
        gridh[i] = h;
    }

    // debugging output
    utils::msg(utils::VL_DEBUG, "fitSphericalDF",
        "f(h) ~ h^" + utils::toString(derLeft)  + " at small h"
        " and ~ h^" + utils::toString(derRight) + " at large h");
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("FitSphericalDF.log");
        strm << "#h      \tf(h)    \tinner slope: " << derLeft << ", outer slope: " << derRight << '\n';
        for(unsigned int i=0; i<gridh.size(); i++)
            strm << utils::pp(gridh[i], 12) << '\t' << utils::pp(gridf[i],12) << '\n';
    }

    // the derivatives provided to LogLogSpline are df/dh = (f/h) d[log f] / d[log h]
    derLeft  = derLeft  * gridf.front() / gridh.front();
    derRight = derRight * gridf.back () / gridh.back ();

    // 5. construct an interpolating spline that matches exactly our fitfnc (it's also a cubic spline
    // in the same scaled variables), including the correct slopes for extrapolation outside the grid
    return math::LogLogSpline(gridh, gridf, derLeft, derRight);
}


//---- create an N-body realization of a spherical model ----//

particles::ParticleArraySph samplePosVel(
    const math::IFunction& pot, const math::IFunction& df, const unsigned int numPoints)
{
    DFSphericalIntegrand fnc(pot, df);
    math::Matrix<double> result;  // sampled scaled coordinates/velocities
    double totalMass, errorMass;  // total normalization of the DF and its estimated error
    double xlower[2] = {0,0};     // boundaries of sampling region in scaled coordinates
    double xupper[2] = {1,1};
    math::sampleNdim(fnc, xlower, xupper, numPoints, result, NULL, &totalMass, &errorMass);
    const double pointMass = totalMass / result.rows();
    particles::ParticleArraySph points;
    points.data.reserve(result.rows());
    for(unsigned int i=0; i<result.rows(); i++) {
        double r, v, Phi, vdir[3],
        rtheta = acos(math::random()*2-1),
        rphi   = 2*M_PI * math::random();
        math::getRandomUnitVector(vdir);
        fnc.unscalerv(result(i, 0), result(i, 1), r, v, Phi);
        points.add(coord::PosVelSph(r, rtheta, rphi, v*vdir[0], v*vdir[1], v*vdir[2]), pointMass);
    }
    return points;
}


//---- construct a density profile from a cumulative mass profile ----//

std::vector<double> densityFromCumulativeMass(
    const std::vector<double>& gridr, const std::vector<double>& gridm)
{
    unsigned int size = gridr.size();
    if(size<3 || gridm.size()!=size)
        throw std::invalid_argument("densityFromCumulativeMass: invalid array sizes");
    // check monotonicity and convert to log-scaled radial grid
    std::vector<double> gridlogr(size), gridlogm(size), gridrho(size);
    for(unsigned int i=0; i<size; i++) {
        if(!(gridr[i] > 0 && gridm[i] > 0))
            throw std::invalid_argument("densityFromCumulativeMass: negative input values");
        if(i>0 && (gridr[i] <= gridr[i-1] || gridm[i] <= gridm[i-1]))
            throw std::invalid_argument("densityFromCumulativeMass: arrays are not monotonic");
        gridlogr[i] = log(gridr[i]);
    }
    // determine if the cumulative mass approaches a finite limit at large radii,
    // that is, M = Minf - A * r^B  with A>0, B<0
    double B = math::findRoot(SlopeFinder(
        gridlogr[size-1], gridlogr[size-2], gridlogr[size-3],
        gridm   [size-1], gridm   [size-2], gridm   [size-3] ), -100, 0, EPSROOT);
    double invMinf = 0;  // 1/Minf, or remain 0 if no finite limit is detected
    if(B<0) {
        double A =  (gridm[size-1] - gridm[size-2]) / 
            (exp(B * gridlogr[size-2]) - exp(B * gridlogr[size-1]));
        if(A>0) {  // viable extrapolation
            invMinf = 1 / (gridm[size-1] + A * exp(B * gridlogr[size-1]));
            utils::msg(utils::VL_DEBUG, "densityFromCumulativeMass",
                "Extrapolated total mass=" + utils::toString(1/invMinf) +
                ", rho(r)~r^" + utils::toString(B-3) + " at large radii" );
        }
    }
    // scaled mass to interpolate:  log[ M / (1 - M/Minf) ] as a function of log(r),
    // which has a linear asymptotic behaviour with slope -B as log(r) --> infinity;
    // if Minf = infinity, this additional term has no effect
    for(unsigned int i=0; i<size; i++)
        gridlogm[i] = log(gridm[i] / (1 - gridm[i]*invMinf));
    math::CubicSpline spl(gridlogr, gridlogm, true /*enforce monotonicity*/);
    if(!spl.isMonotonic())
        throw std::runtime_error("densityFromCumulativeMass: interpolated mass is not monotonic");
    // compute the density at each point of the input radial grid
    for(unsigned int i=0; i<size; i++) {
        double val, der;
        spl.evalDeriv(gridlogr[i], &val, &der);
        val = exp(val);
        gridrho[i] = der * val / (4*M_PI * pow_3(gridr[i]) * pow_2(1 + val * invMinf));
        if(gridrho[i] <= 0)   // shouldn't occur if the spline is (strictly) monotonic
            throw std::runtime_error("densityFromCumulativeMass: interpolated density is non-positive");
    }
    return gridrho;
}


//---- Compute density and optionally velocity dispersion from DF ----//

std::vector<double> computeDensity(const math::IFunction& df, const potential::PhaseVolume& pv,
    const std::vector<double> &gridPhi, std::vector<double> *gridVelDisp)
{
    unsigned int gridsize = gridPhi.size();
    std::vector<double> result(gridsize);
    if(gridVelDisp)
        gridVelDisp->assign(gridsize, 0);
    // assuming that the grid in Phi is sufficiently dense, use a fixed-order quadrature on each segment
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
    for(unsigned int i=0; i<gridsize; i++) {
        double deltaPhi = (i<gridsize-1 ? gridPhi[i+1] : 0) - gridPhi[i];
        if(deltaPhi<=0)
            throw std::runtime_error("computeDensity: grid in Phi must be monotonically increasing");
        for(int k=0; k<GLORDER; k++) {
            // node of Gauss-Legendre quadrature within the current segment (Phi[i] .. Phi[i+1]);
            // the integration variable y ranges from 0 to 1, and Phi(y) is defined below
            double y   = glnodes[k];
            double Phi = gridPhi[i] + y*y * deltaPhi;
            // contribution of this point to each integral on the current segment, taking into account
            // the transformation of variable y -> Phi, multiplied by the value of f(h(Phi))
            double weight = glweights[k] * 2*y * deltaPhi * df(pv(Phi)) * (4*M_PI*M_SQRT2);
            // add a contribution to the integrals expressing rho(Phi[j]) for all Phi[j] < Phi
            for(unsigned int j=0; j<=i; j++) {
                double dif = Phi - gridPhi[j];  // guaranteed to be positive (or zero due to roundoff)
                assert(dif>=0);
                double val = sqrt(dif) * weight;
                result[j] += val;
                if(gridVelDisp)
                    gridVelDisp->at(j) += val * dif;
            }
        }
    }
    if(gridVelDisp)
        for(unsigned int i=0; i<gridsize; i++)
            gridVelDisp->at(i) = sqrt(2./3 * gridVelDisp->at(i) / result[i]);
    return result;
}


//---- Compute projected density and velocity dispersion ----//

void computeProjectedDensity(const math::IFunction& dens, const math::IFunction& velDisp,
    const std::vector<double> &gridR,
    std::vector<double>& gridProjDensity, std::vector<double>& gridProjVelDisp)
{
    unsigned int gridsize = gridR.size();
    gridProjDensity.assign(gridsize, 0);
    gridProjVelDisp.assign(gridsize, 0);
    // assuming that the grid in R is sufficiently dense, use a fixed-order quadrature on each segment
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
    for(unsigned int i=0; i<gridsize; i++) {
        bool last = i==gridsize-1;
        double deltar = last ? gridR[i] : gridR[i+1]-gridR[i];
        if(deltar<=0)
            throw std::runtime_error("computeProjectedDensity: grid in R must be monotonically increasing");
        for(int k=0; k<GLORDER; k++) {
            // node of Gauss-Legendre quadrature within the current segment (R[i] .. R[i+1]);
            // the integration variable y ranges from 0 to 1, and r(y) is defined below
            // (differently for the last grid segment which extends to infinity)
            double y = glnodes[k];
            double r = last ? gridR[i] / (1 - y*y) : gridR[i] + y*y * deltar;
            // contribution of this point to each integral on the current segment, taking into account
            // the transformation of variable y -> r, multiplied by the value of rho(r)
            double weight = glweights[k] * (last ? 2*y / pow_2(1-y*y) : 2*y) * deltar * dens(r) * 2*r;
            double velsq  = pow_2(velDisp(r));
            // add a contribution to the integrals expressing Sigma(R) for all R[j] < r
            for(unsigned int j=0; j<=i; j++) {
                double dif = pow_2(r) - pow_2(gridR[j]);  // guaranteed to be positive
                assert(dif>0);
                double val = weight / sqrt(dif);
                gridProjDensity[j] += val;
                gridProjVelDisp[j] += val * velsq;
            }
        }
    }
    for(unsigned int i=0; i<gridsize; i++)
        gridProjVelDisp[i] = sqrt(gridProjVelDisp[i] / gridProjDensity[i]);
}


//---- Spherical model specified by a DF f(h) and phase volume h(E) ----//

SphericalModel::SphericalModel(const potential::PhaseVolume& _phasevol, const math::IFunction& df,
    const std::vector<double>& gridh) :
    phasevol(_phasevol)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in log[h(Phi)] if it wasn't provided
    std::vector<double> gridLogH;
    if(gridh.empty())
        gridLogH = math::createInterpolationGrid(math::LogLogScaledFnc(df), EPSDER2);
    else {
        gridLogH.resize(gridh.size());
        for(size_t i=0; i<gridh.size(); i++)
            gridLogH[i] = log(gridh[i]);
    }
    const unsigned int npoints = gridLogH.size();

    // 2. store the values of f, g, h at grid nodes (ensure to consider only positive values of f)
    std::vector<double> gridF(npoints), gridG(npoints), gridH(npoints), gridE(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        double h = exp(gridLogH[i]);
        double f = df(h);
        if(!(f>=0))
            throw std::runtime_error("SphericalModel: f("+utils::toString(h)+")="+utils::toString(f));
        gridF[i] = f;
        gridH[i] = h;
        gridE[i] = phasevol.E(h, &gridG[i]);
    }
    std::vector<double> gridFint(npoints), gridFGint(npoints), gridFHint(npoints), gridFEint(npoints);

    // 3a. determine the asymptotic behaviour of f(h):
    // f(h) ~ h^outerFslope as h-->inf  or  h^innerFslope as h-->0
    double innerFslope, outerFslope;
    if(df.numDerivs() >= 1) {
        double der;
        df.evalDeriv(gridH[0], NULL, &der);
        innerFslope = der / gridF[0] * gridH[0];
        df.evalDeriv(gridH[npoints-1], NULL, &der);
        outerFslope = der / gridF[npoints-1] * gridH[npoints-1];
    } else {
        innerFslope = log(gridF[1] / gridF[0]) / (gridLogH[1] - gridLogH[0]);
        outerFslope = log(gridF[npoints-1] / gridF[npoints-2]) /
            (gridLogH[npoints-1] - gridLogH[npoints-2]);
    }
    if(gridF[0] <= MIN_VALUE_ROUNDOFF) {
        gridF[0] = innerFslope = 0.;
    } else if(!(innerFslope > -1))
        throw std::runtime_error("SphericalModel: f(h) rises too rapidly as h-->0\n"
            "f(h="+utils::toString(gridH[0])+")="+utils::toString(gridF[0]) + "; "
            "f(h="+utils::toString(gridH[1])+")="+utils::toString(gridF[1]) + " => "
            "f ~ h^"+utils::toString(innerFslope));
    if(gridF[npoints-1] <= MIN_VALUE_ROUNDOFF) {
        gridF[npoints-1] = outerFslope = 0.;
    } else if(!(outerFslope < -1))
        throw std::runtime_error("SphericalModel: f(h) falls off too slowly as h-->infinity\n"
             "f(h="+utils::toString(gridH[npoints-1])+")="+utils::toString(gridF[npoints-1]) + "; "
             "f(h="+utils::toString(gridH[npoints-2])+")="+utils::toString(gridF[npoints-2]) + " => "
             "f ~ h^"+utils::toString(outerFslope));

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
    double innerE = gridE.front();
    double outerE = gridE.back();
    if(!(Phi0 < innerE && innerE < outerE && outerE < 0))
        throw std::runtime_error("SphericalModel: weird behaviour of potential\n"
            "Phi(0)="+utils::toString(Phi0)  +", "
            "innerE="+utils::toString(innerE)+", "
            "outerE="+utils::toString(outerE));
    if(Phi0 != -INFINITY)   // determination of inner slope depends on whether the potential is finite
        innerE -= Phi0;
    double innerEslope = gridH.front() / gridG.front() / innerE;
    double outerEslope = gridH.back()  / gridG.back()  / outerE;
    double outerRatio  = outerFslope  / outerEslope;
    if(!(outerEslope < 0))   // should be <0 if the potential tends to zero at infinity
        throw std::runtime_error("SphericalModel: weird behaviour of E(h) at infinity: "
            "E ~ h^" +utils::toString(outerEslope));
    if(!(innerEslope + innerFslope > -1))
        throw std::runtime_error("SphericalModel: weird behaviour of f(h) at origin: "
            "E ~ h^"+utils::toString(innerEslope)+", "
            "f ~ h^"+utils::toString(innerFslope)+", "
            "their product grows faster than h^-1 => total energy is infinite");

    // 4. compute integrals
    // \int f(E) dE        = \int f(h) / g(h) h d(log h),    [?]
    // \int f(E) g(E) dE   = \int f(h) h d(log h),           [mass]
    // \int f(E) h(E) dE   = \int f(h) / g(h) h^2 d(log h),  [kinetic energy]
    // \int f(E) g(E) E dE = \int f(h) E h d(log h)          [total energy]

    // 4a. integrate over all interior segments
    const double *glnodes1 = math::GLPOINTS[GLORDER1], *glweights1 = math::GLWEIGHTS[GLORDER1];
    const double *glnodes2 = math::GLPOINTS[GLORDER2], *glweights2 = math::GLWEIGHTS[GLORDER2];
    for(unsigned int i=1; i<npoints; i++) {
        double dlogh = gridLogH[i]-gridLogH[i-1];
        // choose a higher-order quadrature rule for longer grid segments
        int glorder  = dlogh < GLDELTA ? GLORDER1 : GLORDER2;
        const double *glnodes   = glorder == GLORDER1 ? glnodes1   : glnodes2;
        const double *glweights = glorder == GLORDER1 ? glweights1 : glweights2;
        for(int k=0; k<glorder; k++) {
            // node of Gauss-Legendre quadrature within the current segment (logh[i-1] .. logh[i]);
            double logh = gridLogH[i-1] + dlogh * glnodes[k];
            // GL weight -- contribution of this point to each integral on the current segment
            double weight = glweights[k] * dlogh;
            // compute E, f, g, h at the current point h (GL node)
            double h = exp(logh), g, E = phasevol.E(h, &g), f = df(h);
            if(!(f>=0))
                throw std::runtime_error("SphericalModel: f("+utils::toString(h)+")="+utils::toString(f));
            // the original integrals are formulated in terms of  \int f(E) weight(E) dE,
            // where weight = 1, g, h for the three integrals,
            // and we replace  dE  by  d(log h) * [ dh / d(log h) ] / [ dh / dE ],
            // that's why there are extra factors h and 1/g below.
            double integrand = f * h * weight;
            gridFint[i-1] += integrand / g;
            gridFGint[i]  += integrand;
            gridFHint[i]  += integrand / g * h;
            gridFEint[i]  -= integrand * E;
        }
    }

    // 4b. integral of f(h) dE = f(h) / g(h) dh -- compute from outside in,
    // summing contributions from all intervals of h above its current value
    // the outermost segment from h_max to infinity is integrated analytically
    gridFint.back() = -gridF.back() * outerE / (1 + outerRatio);
    for(int i=npoints-1; i>=1; i--) {
        gridFint[i-1] += gridFint[i];
    }

    // 4c. integrands of f*g dE,  f*h dE  and  f*g*E dE;  note that g = dh/dE.
    // compute from inside out, summing contributions from all previous intervals of h
    // integrals over the first segment (0..gridH[0]) are computed analytically
    gridFGint[0] = gridF[0] * gridH[0] / (1 + innerFslope);
    gridFHint[0] = gridF[0] * pow_2(gridH[0]) / gridG[0] / (1 + innerEslope + innerFslope);
    gridFEint[0] = gridF[0] * gridH[0] * (innerEslope >= 0 ?
        -Phi0   / (1 + innerFslope) :
        -innerE / (1 + innerFslope + innerEslope) );

    for(unsigned int i=1; i<npoints; i++) {
        gridFGint[i] += gridFGint[i-1];
        gridFHint[i] += gridFHint[i-1];
        gridFEint[i] += gridFEint[i-1];
    }
    // add the contribution of integrals from the last grid point up to infinity (very small anyway)
    gridFGint.back() -= gridF.back() * gridH.back() / (1 + outerFslope);
    gridFHint.back() -= gridF.back() * pow_2(gridH.back()) / gridG.back() / (1 + outerEslope + outerFslope);
    gridFEint.back() += gridF.back() * gridH.back() * outerE / (1 + outerEslope + outerFslope);
    totalMass = gridFGint.back();
    if(!(totalMass > 0))
        throw std::runtime_error("SphericalModel: f(h) is nowhere positive");

    // decide on the value of h separating two regimes of computing f(h) from interpolating splines:
    // if h is not too large, use intfg, otherwise use intf
    htransition = gridH[0];
    for(unsigned int i=1; i<npoints-1 && gridFGint[i+1] < totalMass * 0.999; i++)
        htransition = gridH[i];

    // 5. construct 1d interpolating splines for these integrals
    // 5a. prepare derivatives for quintic spline
    std::vector<double> gridFder(npoints), gridFGder(npoints), gridFHder(npoints), gridFEder(npoints);
    for(unsigned int i=0; i<npoints; i++) {
        gridFder [i] = -gridF[i] / gridG[i];
        gridFGder[i] =  gridF[i];
        gridFHder[i] =  gridF[i] * gridH[i] / gridG[i];
        gridFEder[i] = -gridF[i] * gridE[i];
        if(!(gridFder[i]<=0 && gridFGder[i]>=0 && gridFHder[i]>=0 && gridFEder[i]>=0 && 
            isFinite(gridFint[i] + gridFGint[i] + gridFHint[i] + gridFEint[i])))
            throw std::runtime_error("SphericalModel: cannot construct valid interpolators");
    }
    // integrals of f*g, f*h and f*g*E have finite limit as h-->inf;
    // extrapolate them as constants beyond the last grid point
    gridFGder.back() = gridFHder.back() = gridFEder.back() = 0;

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("SphericalModel.log");
        strm << "h             \tg             \tE             \tf(E)          \t"
            "int_E^0 f dE  \tint_Phi0^E f g\tint_Phi0^E f h\tint_Phi0^E f g E\n";
        for(unsigned int i=0; i<npoints; i++) {
            strm <<
            utils::pp(gridH[i],     14) + '\t' + utils::pp(gridG[i],     14) + '\t' +
            utils::pp(gridE[i],     14) + '\t' + utils::pp(gridF[i],     14) + '\t' +
            utils::pp(gridFint[i],  14) + '\t' + utils::pp(gridFGint[i], 14) + '\t' +
            utils::pp(gridFHint[i], 14) + '\t' + utils::pp(gridFEint[i], 14) + '\n';
        }
    }

    // 5b. initialize splines for log-scaled integrals
    intf  = math::LogLogSpline(gridH, gridFint,  gridFder);
    intfg = math::LogLogSpline(gridH, gridFGint, gridFGder);
    intfh = math::LogLogSpline(gridH, gridFHint, gridFHder);
    intfE = math::LogLogSpline(gridH, gridFEint, gridFEder);
}

void SphericalModel::evalDeriv(const double h, double* f, double* dfdh, double* /*ignored*/) const
{
    double der, der2, g, dgdh;
    // at large h, intfg reaches a limit (totalMass), thus its derivative may be inaccurate
    if(h < htransition) {   // still ok
        // f(h) = d[ int_0^h f(h') dh' ] / d h
        intfg.evalDeriv(h, NULL, &der, dfdh? &der2 : NULL);
        if(f)
            *f = der;
        if(dfdh)
            *dfdh = der2;
    } else {
        // otherwise we compute it from a different spline which tends to zero at large h:
        // f(h) = -g(h)  d[ int_h^\infty f(h') / g(h') dh' ] / d h
        intf.evalDeriv(h, NULL, &der, dfdh? &der2 : NULL);
        phasevol.E(h, &g, dfdh? &dgdh : NULL);
        if(f)
            *f = -der * g;
        if(dfdh)
            *dfdh = -der2 * g - der * dgdh;
    }
}

double SphericalModel::I0(const double h) const
{
    return intf(h);
}

double SphericalModel::cumulMass(const double h) const
{
    if(h==INFINITY)
        return totalMass;
    return intfg(h);
}

double SphericalModel::cumulEkin(const double h) const
{
    return 1.5 * intfh(h);
}

double SphericalModel::cumulEtotal(const double h) const
{
    return -intfE(h);
}

//---- Extended spherical model with 2d interpolation for position-dependent quantities ----//

void SphericalModelLocal::init(const math::IFunction& df, const std::vector<double>& gridh)
{
    // 1. determine the range of h that covers the region of interest
    // and construct the grid in X = log[h(Phi)] and Y = log[h(E)/h(Phi)]
    std::vector<double> gridLogH;
    if(gridh.empty())
        gridLogH = math::createInterpolationGrid(math::LogLogScaledFnc(df), EPSDER2);
    else {
        gridLogH.resize(gridh.size());
        for(size_t i=0; i<gridh.size(); i++)
            gridLogH[i] = log(gridh[i]);
    }
    while(!gridLogH.empty() && df(exp(gridLogH.back())) <= MIN_VALUE_ROUNDOFF)  // ensure that f(hmax)>0
        gridLogH.pop_back();
    if(gridLogH.size() < 3)
        throw std::runtime_error("SphericalModelLocal: f(h) is nowhere positive");
    const double logHmin        = gridLogH.front(),  logHmax = gridLogH.back();
    const unsigned int npoints  = gridLogH.size();
    const unsigned int npointsY = 100;
    const double mindeltaY      = fmin(0.1, (logHmax-logHmin)/npointsY);
    std::vector<double> gridY   = math::createNonuniformGrid(npointsY, mindeltaY, logHmax-logHmin, true);

    // 3. determine the asymptotic behaviour of f(h) and g(h):
    // f(h) ~ h^outerFslope as h-->inf and  g(h) ~ h^(1-outerEslope)
    double outerH = exp(gridLogH.back()), outerG;
    double outerE = phasevol.E(outerH, &outerG), outerFslope;
    if(df.numDerivs() >= 1) {
        double val, der;
        df.evalDeriv(outerH, &val, &der);
        outerFslope = der / val * outerH;
    } else {
        outerFslope = log(df(outerH) / df(exp(gridLogH[npoints-2]))) /
            (gridLogH[npoints-1] - gridLogH[npoints-2]);
    }
    if(!(outerFslope < -1))  // in this case SphericalModel would have already thrown the same exception
        throw std::runtime_error("SphericalModelLocal: f(h) falls off too slowly as h-->infinity");
    double outerEslope = outerH / outerG / outerE;
    double outerRatio  = outerFslope / outerEslope;
    if(!(outerRatio > 0))
        throw std::runtime_error("SphericalModelLocal: weird asymptotic behaviour of phase volume\n"
            "h(E="+utils::toString(outerE)+")="+utils::toString(outerH) +
            "; dh/dE="+utils::toString(outerG) + " => outerEslope="+utils::toString(outerEslope) +
            ", outerFslope="+utils::toString(outerFslope));

    // 5. construct 2d interpolating splines for dv2par, dv2per as functions of Phi and E

    // 5a. asymptotic values for J1/J0 and J3/J0 as Phi --> 0 and (E/Phi) --> 0
    double outerJ1 = 0.5*M_SQRTPI * math::gamma(2 + outerRatio) / math::gamma(2.5 + outerRatio);
    double outerJ3 = outerJ1 * 1.5 / (2.5 + outerRatio);

    // 5b. compute the values of J1/J0 and J3/J0 at nodes of 2d grid in X=log(h(Phi)), Y=log(h(E)/h(Phi))
    math::Matrix<double> gridJ1(npoints, npointsY), gridJ3(npoints, npointsY);
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
        DFIntegrand<0> intJ0(df, phasevol, gridLogH[i]);
        DFIntegrand<1> intJ1(df, phasevol, gridLogH[i]);
        DFIntegrand<3> intJ3(df, phasevol, gridLogH[i]);
        gridJ1(i, 0) = log(2./3);  // analytic limiting values for Phi=E
        gridJ3(i, 0) = log(2./5);
        for(unsigned int j=1; j<npointsY; j++) {
            double logHprev = gridLogH[i] + gridY[j-1];
            double logHcurr = gridLogH[i] + gridY[j];
            if(j==1) {
                // integration over the first segment uses a more accurate quadrature rule
                // to accounting for a possible endpoint singularity at Phi=E
                math::ScalingCub scaling(logHprev, logHcurr);
                J0acc = math::integrateGL(
                    math::ScaledIntegrand<math::ScalingCub>(scaling, intJ0), 0, 1, GLORDER);
                J1acc = math::integrateGL(
                    math::ScaledIntegrand<math::ScalingCub>(scaling, intJ1), 0, 1, GLORDER);
                J3acc = math::integrateGL(
                    math::ScaledIntegrand<math::ScalingCub>(scaling, intJ3), 0, 1, GLORDER);
            } else {
                J0acc += math::integrateGL(intJ0, logHprev, logHcurr, GLORDER);
                J1acc += math::integrateGL(intJ1, logHprev, logHcurr, GLORDER);
                J3acc += math::integrateGL(intJ3, logHprev, logHcurr, GLORDER);
            }
            if(i==npoints-1) {
                // last row: analytic limiting values for Phi-->0 and any E/Phi
                double EoverPhi = exp(gridY[j] * outerEslope);  // strictly < 1
                double oneMinusJ0overI0 = std::pow(EoverPhi, 1+outerRatio);  // < 1
                double Fval1 = math::hypergeom2F1(-0.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double Fval3 = math::hypergeom2F1(-1.5, 1+outerRatio, 2+outerRatio, EoverPhi);
                double I0    = this->I0(exp(gridLogH[i]));
                double sqPhi = sqrt(-outerE);
                if(isFinite(Fval1+Fval3)) {
                    J0acc = I0 * (1 - oneMinusJ0overI0);
                    J1acc = I0 * (outerJ1 - oneMinusJ0overI0 * Fval1) * sqPhi;
                    J3acc = I0 * (outerJ3 - oneMinusJ0overI0 * Fval3) * pow_3(sqPhi);
                } else {
                    // this procedure sometimes fails, since hypergeom2F1 is not very robust;
                    // in this case we simply keep the values computed by numerical integration
                    utils::msg(utils::VL_WARNING, "SphericalModelLocal", "Can't compute asymptotic value");
                }
            }
            double dv = sqrt(phasevol.deltaE(logHcurr, gridLogH[i]));
            double J1overJ0 = J1acc / J0acc / dv;
            double J3overJ0 = J3acc / J0acc / pow_3(dv);
            if(J1overJ0<=0 || J3overJ0<=0 || !isFinite(J1overJ0+J3overJ0)) {
                utils::msg(utils::VL_WARNING, "SphericalModelLocal", "Invalid value"
                    "  J0="+utils::toString(J0acc)+
                    ", J1="+utils::toString(J1acc)+
                    ", J3="+utils::toString(J3acc));
                J1overJ0 = 2./3;   // fail-safe values corresponding to E=Phi
                J3overJ0 = 2./5;
            }
            gridJ1(i, j) = log(J1overJ0);
            gridJ3(i, j) = log(J3overJ0);
        }
    }

    // debugging output
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream strm("SphericalModelLocal.log");
        strm << "ln[h(Phi)] ln[hE/hPhi]\tPhi            E             \tJ1         J3\n";
        for(unsigned int i=0; i<npoints; i++) {
            double Phi = phasevol.E(exp(gridLogH[i]));
            for(unsigned int j=0; j<npointsY; j++) {
                double E = phasevol.E(exp(gridLogH[i] + gridY[j]));
                strm << utils::pp(gridLogH[i],10) +' '+ utils::pp(gridY[j],10) +'\t'+
                utils::pp(Phi,14) +' '+ utils::pp(E,14) +'\t'+
                utils::pp(exp(gridJ1(i, j)),10) +' '+ utils::pp(exp(gridJ3(i, j)),10)+'\n';
            }
            strm << '\n';
        }
    }

    // 5c. construct the 2d splines
    intJ1 = math::CubicSpline2d(gridLogH, gridY, gridJ1);
    intJ3 = math::CubicSpline2d(gridLogH, gridY, gridJ3);
}

void SphericalModelLocal::evalLocal(
    double Phi, double E, double &dvpar, double &dv2par, double &dv2per) const
{
    double hPhi = phasevol(Phi);
    double hE   = phasevol(E);
    if(!(Phi<0 && hE >= hPhi))
        throw std::invalid_argument("SphericalModelLocal: incompatible values of E and Phi");

    // compute the 1d interpolators for I0, J0
    double I0 = this->I0(hE);
    double J0 = fmax(this->I0(hPhi) - I0, 0);
    // restrict the arguments of 2d interpolators to the range covered by their grids
    double X  = math::clamp(log(hPhi),    intJ1.xmin(), intJ1.xmax());
    double Y  = math::clamp(log(hE/hPhi), intJ1.ymin(), intJ1.ymax());
    // compute the 2d interpolators for J1, J3
    double J1 = exp(intJ1.value(X, Y)) * J0;
    double J3 = exp(intJ3.value(X, Y)) * J0;
    if(E>=0) {  // in this case, the coefficients were computed for E=0, need to scale them to E>0
        double corr = 1 / sqrt(1 - E / Phi);  // correction factor <1
        J1 *= corr;
        J3 *= pow_3(corr);
    }
    double mult = 32*M_PI*M_PI/3 * cumulMass();
    dvpar  = -mult *  J1 * 3;
    dv2par =  mult * (I0 + J3);
    dv2per =  mult * (I0 * 2 + J1 * 3 - J3);
    /*if(loghPhi<X)
        utils::msg(utils::VL_WARNING, "SphericalModelLocal",
        "Extrapolating to small h: log(h(Phi))="+utils::toString(loghPhi)+
        ", log(h(E))="+utils::toString(loghE)+
        ", I0="+utils::toString(I0)+", J0="+utils::toString(J0));*/
}

/// Helper class for finding the value of energy at which
/// the cumulative distribution function equals the target value
class VelocitySampleRootFinder: public math::IFunctionNoDeriv
{
    const SphericalModel& model;      ///< the model providing h(E) and J0(h)
    const math::CubicSpline2d& intJ1; ///< J1 as a function of h(E) and h(Phi)
    const double Phi;                 ///< Phi, the potential at the given radius
    const double loghPhi;             ///< log(h(Phi)) is cached to avoid its repeated evaluation
    const double I0plusJ0;            ///< I0(h(Phi))
    const double target;              ///< target value of the cumulative DF
public:
    VelocitySampleRootFinder(const SphericalModel& _model, const math::CubicSpline2d& _intJ1,
        const double _Phi, const double _loghPhi, const double _I0plusJ0, const double _target) :
        model(_model), intJ1(_intJ1), Phi(_Phi), loghPhi(_loghPhi), I0plusJ0(_I0plusJ0), target(_target)
    {}
    double value(const double loghEoverhPhi) const
    {
        double hE = exp(loghEoverhPhi + loghPhi);
        double E  = model.phasevol.E(hE);
        double J0 = I0plusJ0 - model.I0(hE);
        double J1 = exp(intJ1.value(loghPhi, loghEoverhPhi)) * J0;
        double val= J1 * sqrt(fmax(E-Phi, 0.));
        return val - target;
    }
};

double SphericalModelLocal::sampleVelocity(double Phi) const
{
    if(!(Phi<0))
        throw std::invalid_argument("SphericalModelLocal: invalid value of Phi");
    double hPhi     = phasevol(Phi);
    double loghPhi  = math::clamp(log(hPhi), intJ1.xmin(), intJ1.xmax());
    double I0plusJ0 = I0(hPhi);
    double maxJ1    = exp(intJ1.value(loghPhi, intJ1.ymax())) * I0plusJ0;
    double frac     = math::random();
    double target   = frac * maxJ1 * sqrt(-Phi);
    // find the value of E at which the cumulative distribution function equals the target
    double loghEoverhPhi = math::findRoot(
        VelocitySampleRootFinder(*this, intJ1, Phi, loghPhi, I0plusJ0, target),
        intJ1.ymin(), intJ1.ymax(), EPSROOT);
    if(!(loghEoverhPhi>=0))
       return 0.;  // might not be able to find the root in some perverse cases at very large radii
    double hE = exp(loghEoverhPhi + loghPhi);
    double E  = phasevol.E(hE);
    return sqrt(2. * (E - Phi));
}

double SphericalModelLocal::density(double Phi) const
{
    if(!(Phi<0))
        throw std::invalid_argument("SphericalModelLocal: invalid value of Phi");
    double hPhi     = phasevol(Phi);
    double loghPhi  = math::clamp(log(hPhi), intJ1.xmin(), intJ1.xmax());
    double J1overJ0 = exp(intJ1.value(loghPhi, intJ1.ymax()));
    double I0plusJ0 = I0(hPhi);  // in fact I0(E)=0 because E=0
    return 4*M_PI*M_SQRT2 * sqrt(-Phi) * J1overJ0 * I0plusJ0;
}

double SphericalModelLocal::velDisp(double Phi) const
{
    if(!(Phi<0))
        throw std::invalid_argument("SphericalModelLocal: invalid value of Phi");
    double hPhi     = phasevol(Phi);
    double loghPhi  = math::clamp(log(hPhi), intJ1.xmin(), intJ1.xmax());
    double J3overJ1 = exp(intJ3.value(loghPhi, intJ3.ymax()) - intJ1.value(loghPhi, intJ1.ymax()));
    return sqrt(-2./3 * Phi * J3overJ1);
}

//---- non-member functions for various diffusion coefficients ----//

void difCoefEnergy(const SphericalModel& model, double E, double &DeltaE, double &DeltaE2)
{
    double h, g;
    model.phasevol.evalDeriv(E, &h, &g);
    double totalMass = model.cumulMass(),
    IF   = model.I0(h),
    IFG  = model.cumulMass(h),
    IFH  = model.cumulEkin(h) * (2./3);
    DeltaE  = 16*M_PI*M_PI * totalMass * (IF - IFG / g);
    DeltaE2 = 32*M_PI*M_PI * totalMass * (IF * h + IFH) / g;
}

double difCoefLosscone(const SphericalModel& model, const math::IFunction& pot, double E)
{
    double h = model.phasevol(E), rmax = potential::R_max(pot, E), g, dgdh;
    model.phasevol.E(h, &g, &dgdh);
    // we are computing the orbit-averaged diffusion coefficient  < Delta v_per^2 >,
    // by integrating it over the radial range covered by the orbit.
    // D = [8 pi^2 / g(E)]  int_0^{rmax(E)}  dr  r^2 / v(E,r)  < Delta v_per^2 >,
    // where  < Delta v_per^2 > = 16 pi^2 Mtotal [ 4/3 I_0(E) + 2 J_{1/2}(E,r) - 2/3 J_{3/2}(E,r) ],
    // I_0     = int_E^0  f(E')  dE',
    // J_{n/2} = int_Phi(r)^E  f(E') (v'/v)^n  dE',
    // v(E,r)  = sqrt{ 2 [E - Phi(r)] },  v'(E',r) = sqrt{ 2 [E' - Phi(r)] }.
    // This is a double integral, and the inner integral consists of two parts:
    // (a)  I_0 does not depend on r and may be brought outside the orbit-averaging integral,
    // which itself is computed analytically:
    // int_0^{rmax(E)} dr  r^2 / v  =  1 / (16 pi^2)  dg(E)/dE,  and  dg/dE = g * dg/dh.
    double result = 2./3 * dgdh * model.I0(h);
    // (b)  the remaining terms need to be integrated numerically;
    // we use a fixed-order GL quadrature for both nested integrals
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
    for(int ir=0; ir<GLORDER; ir++) {
        // the outermost integral in scaled radial variable:  r/rmax 
        double r = glnodes[ir] * rmax, Phi = pot(r);
        double w = 8*M_PI*M_PI * rmax / g * pow_2(r) * glweights[ir];
        for(int iE=0; iE<GLORDER; iE++) {
            // the innermost integral in scaled energy variable:  (E'-Phi) / (E-Phi)
            double Ep  = E * glnodes[iE] + Phi * (1-glnodes[iE]);
            double fEp = model.value(model.phasevol(Ep));  // model.value is the value of DF
            double vp  = sqrt(2 * (Ep-Phi));
            result += glweights[iE] * w * fEp * vp * (1 - 1./3 * glnodes[iE] /*(Ep-Phi) / (E-Phi)*/);
        }
    }
    return result * 16*M_PI*M_PI * model.cumulMass();
}


// ------ Input/output of text tables describing spherical models ------ //

math::LogLogSpline readMassProfile(const std::string& filename)
{
    std::ifstream strm(filename.c_str());
    if(!strm)
        throw std::runtime_error("readMassProfile: can't read input file " + filename);
    std::vector<double> radius, mass;
    const std::string validDigits = "0123456789.-+";
    while(strm) {
        std::string str;
        std::getline(strm, str);
        std::vector<std::string> elems = utils::splitString(str, " \t,;");
        if(elems.size() < 2 || validDigits.find(elems[0][0]) == std::string::npos)
            continue;
        double r = utils::toDouble(elems[0]),  m = utils::toDouble(elems[1]);
        if(r<0)
            throw std::runtime_error("readMassProfile: radii should be positive");
        if(r==0 && m!=0)
            throw std::runtime_error("readMassProfile: M(r=0) should be zero");
        if(r>0) {
            radius.push_back(r);
            mass.push_back(m);
        }
    }
    return math::LogLogSpline(radius, densityFromCumulativeMass(radius, mass));
}

void writeSphericalModel(const std::string& fileName, const std::string& header,
    const SphericalModel& model, const math::IFunction& pot, const std::vector<double>& gridh)
{
    // construct a suitable grid in h, if not provided
    std::vector<double> gridH(gridh);
    const math::IFunction& df = model;
    if(gridh.empty()) {
        // estimate the range of log(h) where the DF varies considerably
        std::vector<double> gridLogH = math::createInterpolationGrid(math::LogLogScaledFnc(df), EPSDER2);
        gridH.resize(gridLogH.size());
        for(size_t i=0; i<gridLogH.size(); i++)
            gridH[i] = exp(gridLogH[i]);
    } else if(gridh.size()<2)
        throw std::runtime_error("writeSphericalModel: gridh is too small");

    // construct the corresponding grid in E and r
    double Phi0 = pot(0);
    std::vector<double> gridR, gridPhi, gridG;
    for(size_t i=0; i<gridH.size(); ) {
        double g, Phi = model.phasevol.E(gridH[i], &g);
        // avoid closely spaced potential values whose difference is dominated by roundoff errors
        if(Phi > (gridPhi.empty()? Phi0 : gridPhi.back()) * (1-MIN_VALUE_ROUNDOFF)) {
            gridPhi.push_back(Phi);
            gridG.  push_back(g);
            gridR.  push_back(potential::R_max(pot, Phi));
            i++;
        } else {
            gridH.erase(gridH.begin()+i);
        }
    }
    size_t npoints = gridH.size();

    // compute the density and 1d velocity dispersion by integrating over the DF
    std::vector<double> gridRho, gridVelDisp;
    gridRho = computeDensity(model, model.phasevol, gridPhi, &gridVelDisp);
    for(size_t i=0; i<npoints; i++)  // safety measure to avoid problems in log-log-spline
        if(!isFinite(gridRho[i]+gridVelDisp[i]) || gridRho[i]<=MIN_VALUE_ROUNDOFF)
            gridRho[i] = gridVelDisp[i] = MIN_VALUE_ROUNDOFF;

    // construct interpolators for the density and velocity dispersion profiles
    math::LogLogSpline density(gridR, gridRho);
    math::LogLogSpline veldisp(gridR, gridVelDisp);

    // and use them to compute the projected density and velocity dispersion
    std::vector<double> gridProjDensity, gridProjVelDisp;
    computeProjectedDensity(density, veldisp, gridR, gridProjDensity, gridProjVelDisp);

    double mult = 16*M_PI*M_PI * model.cumulMass();  // common factor for diffusion coefs

    // determine the central mass (check if it appears to be non-zero)
    double coef, slope = potential::innerSlope(pot, NULL, &coef);
    double Mbh = fabs(slope + 1.) < 1e-3 ? -coef : 0;

    // prepare for integrating the density in radius to obtain enclosed mass
    const double *glnodes = math::GLPOINTS[GLORDER], *glweights = math::GLWEIGHTS[GLORDER];
    double Mcumul = 0;

    // print the header and the first line for r=0 (commented out)
    std::ofstream strm(fileName.c_str());
    if(!header.empty())
        strm << "#" << header << "\n";
    strm <<
        "#r      \tM(r)    \tE=Phi(r)\trho(r)  \tf(E)    \tM(E)    \th(E)    \tTrad(E) \trcirc(E) \t"
        "Lcirc(E) \tVelDispersion\tVelDispProj\tSurfaceDensity\tDeltaE^2\tMassFlux\tEnergyFlux";
    if(Mbh>0)
        strm << "\tD_RR/R(0)\n#0        Mbh = " << utils::pp(Mbh, 14) << "\t-INFINITY\n";
    else
        strm << "\n#0      \t0       \t" << utils::pp(Phi0, 14) << '\n';

    // output various quantities as functions of r (or E) to the file
    for(unsigned int i=0; i<gridH.size(); i++) {
        double r = gridR[i], f, dfdh, g = gridG[i], h = gridH[i];
        df.evalDeriv(h, &f, &dfdh);
        // integrate the density on the previous segment
        double rprev = i==0 ? 0 : gridR[i-1];
        for(int k=0; k<GLORDER; k++) {
            double rk = rprev + glnodes[k] * (r-rprev);
            Mcumul += (4*M_PI) * (r-rprev) * glweights[k] * pow_2(rk) * density(rk);
        }
        double
        E        = gridPhi[i],
        rho      = gridRho[i],
        intfg    = model.cumulMass(h),    // mass of particles within phase volume < h
        intfh    = model.cumulEkin(h) * (2./3),
        intf     = model.I0(h),
        //DeltaE   = mult *  (intf - intfg / g),
        DeltaE2  = mult *  (intf * h + intfh) / g * 2.,
        FluxM    =-mult * ((intf * h + intfh) * g * dfdh + intfg * f),
        FluxE    = E * FluxM - mult * ( -(intf * h + intfh) * f + intfg * intf),
        rcirc    = potential::R_circ(pot, E),
        Lcirc    = rcirc * potential::v_circ(pot, rcirc),
        Tradial  = g / (4*M_PI*M_PI * pow_2(Lcirc)),
        veldisp  = gridVelDisp[i],
        veldproj = gridProjVelDisp[i],
        Sigma    = gridProjDensity[i],
        DRRoverR = difCoefLosscone(model, pot, E);

        strm << utils::pp(r,        14) +  // [ 1] radius
        '\t' +  utils::pp(Mcumul,   14) +  // [ 2] enclosed mass
        '\t' +  utils::pp(E,        14) +  // [ 3] Phi(r)=E
        '\t' +  utils::pp(rho,      14) +  // [ 4] rho(r)
        '\t' +  utils::pp(f,        14) +  // [ 5] distribution function f(E)
        '\t' +  utils::pp(intfg,    14) +  // [ 6] mass of particles having energy below E
        '\t' +  utils::pp(h,        14) +  // [ 7] phase volume
        '\t' +  utils::pp(Tradial,  14) +  // [ 8] average radial period at the energy E
        '\t' +  utils::pp(rcirc,    14) +  // [ 9] radius of a circular orbit with energy E
        '\t' +  utils::pp(Lcirc,    14) +  // [10] angular momentum of this circular orbit
        '\t' +  utils::pp(veldisp,  14) +  // [11] 1d velocity dispersion at r
        '\t' +  utils::pp(veldproj, 14) +  // [12] line-of-sight velocity dispersion at projected R
        '\t' +  utils::pp(Sigma,    14) +  // [13] surface density at projected R
        '\t' +  utils::pp(DeltaE2,  14) +  // [14] diffusion coefficient <Delta E^2>
        '\t' +  utils::pp(FluxM,    14) +  // [15] flux of particles through the phase volume
        '\t' +  utils::pp(FluxE,    14);   // [16] flux of energy through the phase volume
        if(Mbh>0)  strm <<                 //      in case of a central black hole:
        '\t' +  utils::pp(DRRoverR, 14);   // [17] loss-cone diffusion coef 
        strm << '\n';
    }
}

}
