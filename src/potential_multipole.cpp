#include "potential_multipole.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "math_spline.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <alloca.h>

namespace potential {

// internal definitions
namespace{

/// minimum number of terms in sph.-harm. expansion used to compute coefficients
/// of a non-spherical density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)
static const int LMIN_SPHHARM = 16;

/// choice between using 1d splines in radius for each (l,m) or 2d splines in (r,theta) for each m
/// is controlled by the order of expansion in theta:
/// if lmax <= LMAX_1D_SPLINE, use 1d splines, otherwise 2d
static const int LMAX_1D_SPLINE = 2;

/// minimum number of grid nodes
static const unsigned int MULTIPOLE_MIN_GRID_SIZE = 2;

/// order of Gauss-Legendre quadrature for computing the radial integrals in Multipole
static const unsigned int GLORDER_RAD = 10;

/// safety factor to avoid roundoff errors near grid boundaries
static const double SAFETY_FACTOR = 100*DBL_EPSILON;

// Helper function to deduce symmetry from the list of non-zero coefficients;
// combine the array of coefficients at different radii into a single array
// and then call the corresponding routine from math::.
// This routine is templated on the number of arrays that it handles:
// each one should have identical number of elements (# of harmonic terms - (lmax+1)^2),
// and each element of each array should have the same dimension (number of radial grid points).
template<int N>
math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> >* C[N])
{
    unsigned int numRadii=0, numCoefs=0;
    bool correct=true;
    for(int n=0; n<N; n++) {
        if(n==0) {
            numCoefs = C[n]->size();
            if(numCoefs>0)
                numRadii = C[n]->at(0).size();
        } else
            correct &= C[n]->size() == numCoefs;
    }
    std::vector<double> D(numCoefs);
    for(unsigned int c=0; correct && c<numCoefs; c++) {
        for(int n=0; n<N; n++) {
            if(C[n]->at(c).size() == numRadii)
                // if any of the elements is non-zero,
                // then the combined array will have non-zero c-th element too.
                for(unsigned int k=0; k<numRadii; k++)
                    D[c] += fabs(C[n]->at(c)[k]);
            else
                correct = false;
        }
    }
    if(!correct)
        throw std::invalid_argument("Error in SphHarmIndices: invalid size of input arrays");
    return math::getIndicesFromCoefs(D);
}

inline math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> > &C)
{
    const std::vector< std::vector<double> >* A = &C;
    return getIndicesFromCoefs<1>(&A);
}
inline math::SphHarmIndices getIndicesFromCoefs(
    const std::vector< std::vector<double> > &C1, const std::vector< std::vector<double> > &C2)
{
    const std::vector< std::vector<double> >* A[2] = {&C1, &C2};
    return getIndicesFromCoefs<2>(A);
}

// resize the array of coefficients down to the requested order;
// \param[in]  lmax  is the maximum order of angular expansion in cos(theta);
// \param[in]  mmax  is the maximum order of expansion in phi (may be less than lmax);
// \param[in,out]  coefs  is the 2d array of coefficients for each harmonic term (1st index)
// at each radius (2nd index), which may be sequestered if necessary.
void restrictSphHarmCoefs(int lmax, int mmax, std::vector<std::vector<double> >& coefs)
{
    coefs.resize(pow_2(lmax+1), std::vector<double>(coefs[0].size(), 0));
    math::SphHarmIndices ind(lmax, mmax, coord::ST_NONE);
    for(unsigned int c=0; c<coefs.size(); c++)
        if(abs(ind.index_m(c))>mmax)
            coefs[c].assign(coefs[c].size(), 0.);
}

// ------- Spherical-harmonic expansion of density or potential ------- //
// The routine `computeSphHarmCoefs` can work with density, potential, or generic N-dimensional
// functions, and computes the sph-harm expansion for either density (in the first case),
// potential and its r-derivative (in the second case), or all values returned by the function
// (in the third case). To avoid code duplication, the function that actually retrieves
// the relevant quantity is separated into a dedicated routine `storeValue`,
// which stores one or more values for each input point.
// The `computeSphHarmCoefsSph` routine is templated on the type of input data.

template<class BaseDensityOrPotential>
void storeValue(const BaseDensityOrPotential& src, const coord::PosCyl& pos, double values[]);

template<>
inline void storeValue(const BaseDensity& src, const coord::PosCyl& pos, double values[])
{
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src, const coord::PosCyl& pos, double values[])
{
    coord::GradCyl grad;
    src.eval(pos, values, &grad);
    double rinv = 1. / sqrt(pow_2(pos.R) + pow_2(pos.z));
    values[1] = grad.dR * pos.R * rinv + grad.dz * pos.z * rinv;
}

template<>
inline void storeValue(const math::IFunctionNdim& src, const coord::PosCyl& pos, double values[])
{
    double point[3] = {pos.R, pos.z, pos.phi};
    src.eval(point, values);
}

// number of quantities computed at each point
template<class BaseDensityOrPotential> int numQuantities(const BaseDensityOrPotential& src);
template<> int numQuantities(const BaseDensity&)   { return 1; }
template<> int numQuantities(const BasePotential&) { return 2; }
template<> int numQuantities(const math::IFunctionNdim& src) { return src.numValues(); }

// collect the values of input quantities at a 3d grid in (r,theta,phi)
template<class BaseDensityOrPotential>
inline std::vector<double> collectValuesSerial(const BaseDensityOrPotential& src,
    const math::SphHarmTransformForward& trans, const std::vector<double>& radii)
{
    int numValues        = numQuantities(src);
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    int numSamplesRadii  = radii.size();
    std::vector<double> values(numSamplesAngles * numSamplesRadii * numValues);
    for(int indR=0; indR<numSamplesRadii; indR++) {
        for(int indA=0; indA<numSamplesAngles; indA++) {
            double rad  = radii[indR];
            double z    = rad * trans.costheta(indA);
            double R    = sqrt(rad*rad - z*z);
            double phi  = trans.phi(indA);
            storeValue(src, coord::PosCyl(R, z, phi),
                &values[(indR * numSamplesAngles + indA) * numValues]);
        }
    }
    return values;
}

// same as above, but OpenMP-parallelizing the loop;
// this variant is primarily used in the context of DF-based self-consistent modelling,
// when the density (0th moment of the DF) is computed by expensive integration
template<class BaseDensityOrPotential>
inline std::vector<double> collectValuesParallel(const BaseDensityOrPotential& src,
    const math::SphHarmTransformForward& trans, const std::vector<double>& radii)
{
    int numValues        = numQuantities(src);
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    int numSamplesTotal  = numSamplesAngles * radii.size();
    std::vector<double> values(numSamplesTotal * numValues);
    std::string errorMsg;
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
    // loop over radii and angular directions, using a combined index variable for better load balancing
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numSamplesTotal; n++) {
        if(cbrk.triggered()) continue;
        try{
            int indR    = n / numSamplesAngles;  // index in radial grid
            int indA    = n % numSamplesAngles;  // combined index in angular direction (theta,phi)
            double rad  = radii[indR];
            double z    = rad * trans.costheta(indA);
            double R    = sqrt(rad*rad - z*z);
            double phi  = trans.phi(indA);
            storeValue(src, coord::PosCyl(R, z, phi),
                &values[(indR * numSamplesAngles + indA) * numValues]);
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error("Keyboard interrupt");
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computeSphHarmCoefs: "+errorMsg);
    return values;
}

template<class BaseDensityOrPotential>
void computeSphHarmCoefs(const BaseDensityOrPotential& src, 
    const math::SphHarmIndices& ind, const std::vector<double>& radii, bool parallel,
    std::vector< std::vector<double> > * coefs[])
{
    unsigned int numPointsRadius = radii.size();
    if(numPointsRadius<1)
        throw std::invalid_argument("computeSphHarmCoefs: radial grid size too small");
    //  initialize sph-harm transform
    const math::SphHarmTransformForward trans(ind);

    // 1st step: collect the values of input quantities at a 3d grid in (r,theta,phi)
    std::vector<double> values = parallel ?
        collectValuesParallel(src, trans, radii) :
        collectValuesSerial  (src, trans, radii);

    // 2nd step: transform these values to spherical-harmonic expansion coefficients at each radius
    std::vector<double> shcoefs(ind.size());
    int numValues        = numQuantities(src);
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    for(int q=0; q<numValues; q++) {
        coefs[q]->assign(ind.size(), std::vector<double>(numPointsRadius));
        for(unsigned int indR=0; indR<numPointsRadius; indR++) {
            trans.transform(&values[indR * numSamplesAngles * numValues + q],
                &shcoefs.front(), numValues);
            math::eliminateNearZeros(shcoefs);
            for(unsigned int c=0; c<ind.size(); c++)
                coefs[q]->at(c)[indR] = shcoefs[c];
        }
    }
}

// transform an N-body snapshot to an array of spherical-harmonic coefficients:
// for each k-th particle, the array of sph.-harm. functions Y_lm(theta_k, phi_k)
// is stored in the output array with the following indexing scheme:
// C_lm(particle_k) = coefs[SphHarmIndices::index(l,m)][k].
// This saves memory, since only the arrays for harmonic coefficients allowed
// by the indexing scheme are allocated and returned.
void computeSphericalHarmonicsFromParticles(
    const particles::ParticleArray<coord::PosCyl> &particles,
    const math::SphHarmIndices &ind,
    std::vector<double> &particleRadii,
    std::vector< std::vector<double> > &coefs)
{
    // allocate space for non-trivial harmonics only (depending on the symmetry)
    ptrdiff_t nbody = particles.size();
    particleRadii.resize(nbody);
    coefs.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
            coefs[ind.index(l, m)].resize(nbody);
    bool needSine = ind.mmin()<0;
    std::string errorMsg;
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // thread-local temporary arrays for Legendre and trigonometric functions
        std::vector<double> tmp(ind.lmax+2+2*ind.mmax);
        double *leg = &tmp[0], *trig = leg + ind.lmax+1;
        trig[0] = 1.;  // stores cos(0*phi), which is not computed by trigMultiAngle
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(ptrdiff_t i=0; i<nbody; i++) {
            if(cbrk.triggered()) continue;
            // compute Y_lm for each particle
            try{
                const coord::PosCyl& pos = particles.point(i);
                double r   = sqrt(pow_2(pos.R) + pow_2(pos.z));
                double tau = pos.z / (r + pos.R);
                particleRadii[i] = r;
                math::trigMultiAngle(pos.phi, ind.mmax, needSine, trig+1 /* start from m=1 */);
                for(int m=0; m<=ind.mmax; m++) {
                    double mult = 2*M_SQRTPI * (m==0 ? 1 : M_SQRT2);
                    math::sphHarmArray(ind.lmax, m, tau, leg);
                    for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
                        coefs[ind.index(l, m)][i] = mult * leg[l-m] * trig[m];
                    if(needSine && m>0)
                        for(int l=ind.lmin(-m); l<=ind.lmax; l+=ind.step)
                            coefs[ind.index(l, -m)][i] = mult * leg[l-m] * trig[ind.mmax+m];
                }
            }
            catch(std::exception& e) {
                errorMsg = e.what();
            }
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error("Keyboard interrupt");
    if(!errorMsg.empty())
        throw std::runtime_error("computeSphericalHarmonicsFromParticles: " + errorMsg);
}


/// auto-assign min/max radii of the grid if they were not provided, for a smooth density model
void chooseGridRadii(const BaseDensity& src, const unsigned int gridSizeR,
    double& rmin, double& rmax)
{
    if(rmax!=0 && rmin!=0) {
        utils::msg(utils::VL_DEBUG, "Multipole",
            "User-defined grid in r=["+utils::toString(rmin)+":"+utils::toString(rmax)+"]");
        return;
    }
    const double
    LOGSTEP = log(1 + sqrt(20./gridSizeR)), // log-spacing between consecutive grid nodes (rule of thumb)
    LOGRMAX = 100.,                         // do not consider |log r| larger than this number
    DELTA   = ROOT3_DBL_EPSILON,            // log-spacing for numerical differentiation
    MAXRHO  = 1e100,                        // upper/lower bounds on the magnitude of rho to consider
    MINRHO  = 1e-100;
    double logr = 0., maxcurv = 0., rcenter = 1.;
    unsigned int skipped = 0;
    std::vector<double> rad, rho;           // keep track of the points with reasonable density values
    // find the radius at which the density varies most considerably
    while(fabs(logr) < LOGRMAX) {
        double r = exp(logr), rho0 = sphericalAverage<AV_RHO>(src, r), curv = 0;
        // only consider points with density values within reasonable bounds (e.g. excluding zero)
        if(fabs(rho0) >= MINRHO && fabs(rho0) <= MAXRHO) {
            // add this point to the overall array, either at the end or at the beginning
            if(logr>0) {
                rad.push_back(r);
                rho.push_back(rho0);
            } else {
                rad.insert(rad.begin(), r);
                rho.insert(rho.begin(), rho0);
            }
            // estimate the second derivative d^2[log(rho)] / d[log(r)]^2
            double rhop = sphericalAverage<AV_RHO>(src, exp(logr+DELTA));
            double rhom = sphericalAverage<AV_RHO>(src, exp(logr-DELTA));
            double derp = log(rhop/rho0) / DELTA, derm = log(rho0/rhom) / DELTA;
            double der2 = fabs(derp-derm) / DELTA;   // estimate of the logarithmic second derivative,
            if(der2 < 10 * DELTA)  // computed to an accuracy ~eps^(1/3),
                der2 = 0;          // hence we declare it zero if it's smaller than 10x the roundoff error
            // density- and volume-weighted logarithmic curvature of density profile
            curv = der2 * rho0 * pow_2(r);
        }
        if(curv > maxcurv) {
            // the radius where the curvature is maximal is taken as the "center" of the profile
            skipped = 0;
            maxcurv = curv;
            rcenter = r;
        } else {
            // if the estimate of the center does not change for several steps in a row, stop the search
            skipped++;
            if(skipped > gridSizeR)
                break;
        }
        // sweep back and forth: logr = [0, +LOGSTEP, -LOGSTEP, +2*LOGSTEP, -2*LOGSTEP, +3*LOGSTEP, etc.]
        logr = -logr + (logr>0 ? 0. : LOGSTEP);
    }
    if(rad.size()<2) {  // density was nowhere within reasonable bounds, most likely identically zero
        rad.insert(rad.begin(), exp(-LOGSTEP));
        rho.insert(rho.begin(), 0.0);
        rad.push_back(exp(LOGSTEP));
        rho.push_back(0.0);
    }
    // by now we have an estimate of the "central" radius (where the density varies most rapidly),
    // and by default we set the min/max grid radii to be equally spaced (in logarithm) from rcenter,
    // but only considering the points where the density was within reasonable bounds (e.g. nonzero)
    if(rmax == 0) {
        rmax = fmin(rad.back(), rcenter * exp(LOGSTEP * 0.5*gridSizeR));
    }
    if(rmin == 0) {
        // default choice: take rmin as small as possible, given the maximum allowed grid spacing
        rmin = fmax(rad[0], rmax * exp(-LOGSTEP * gridSizeR));
        // this choice may be inefficient, because if the potential is reasonably flat near origin,
        // then its derivatives would not be computed reliably at small radii.
        // to determine the suitable inner radius, we demand that the potential at rmin is at least
        // DELTAPHI times larger than the potential at origin.
        const double DELTAPHI = pow_2(ROOT3_DBL_EPSILON);  // ~4e-11
        // 1. estimate the potential at origin (very crudely, linearly interpolating rho on each segment):
        // \Phi(0) = -4\pi \int_0^\infty \rho(r) r dr
        double Phi0 = 0;
        for(unsigned int i=1; i<rad.size(); i++) {
            Phi0 += -4*M_PI / 6 * (rad[i]-rad[i-1]) *
                (rho[i] * (2*rad[i] + rad[i-1]) + rho[i-1] * (rad[i] + 2*rad[i-1]));
        }
        // 2. estimate Phi(r) - Phi(0) at each radius, and shift rmin up if necessary
        double enclMass = 0, deltaPhi = 0;
        for(unsigned int i=1; i<rad.size() && rad[i] < rmax*0.999; i++) {
            enclMass += 4*M_PI / 12 * (rad[i]-rad[i-1]) *
                ((rho[i] + rho[i-1]) * pow_2(rad[i] + rad[i-1]) +
                2 * rho[i] * pow_2(rad[i]) + 2 * rho[i-1] * pow_2(rad[i-1]));
            deltaPhi += 4*M_PI / 6 * (rad[i]-rad[i-1]) *
                (rho[i] * (2*rad[i] + rad[i-1]) + rho[i-1] * (rad[i] + 2*rad[i-1]));
            double dPhi = enclMass / rad[i] + deltaPhi;  // Phi(r) - Phi(0)
            if(fabs(dPhi) < fabs(Phi0) * DELTAPHI)
                rmin = rad[i];
        }
    }
    utils::msg(utils::VL_DEBUG, "Multipole",
        "Grid in r=["+utils::toString(rmin)+":"+utils::toString(rmax)+"] ");
}

/// auto-assign min/max radii of the grid if they were not provided, for a discrete N-body model
void chooseGridRadii(const particles::ParticleArray<coord::PosCyl>& particles,
    unsigned int gridSizeR, double &rmin, double &rmax) 
{
    if(rmin!=0 && rmax!=0)
        return;
    std::vector<double> radii;
    radii.reserve(particles.size());
    double prmin=INFINITY, prmax=0;
    for(size_t i=0, size=particles.size(); i<size; i++) {
        double r = sqrt(pow_2(particles.point(i).R) + pow_2(particles.point(i).z));
        if(particles.mass(i) != 0) {   // only consider particles with non-zero mass
            if(r==0)
                throw std::runtime_error("Multipole: no massive particles at r=0 allowed");
            radii.push_back(r);
            prmin = std::min(prmin, r);
            prmax = std::max(prmax, r);
        }
    }
    size_t nbody = radii.size();
    if(nbody==0)
        throw std::runtime_error("Multipole: no particles provided as input");
    std::nth_element(radii.begin(), radii.begin() + nbody/2, radii.end());
    double rhalf = radii[nbody/2];   // half-mass radius (if all particles have equal mass)
    double spacing = 1 + sqrt(20./gridSizeR);  // ratio between two adjacent grid nodes
    // # of points inside the first or outside the last grid node
    int Nmin = static_cast<int>(log(nbody+1)/log(2));
    if(rmin==0) {
        std::nth_element(radii.begin(), radii.begin() + Nmin, radii.end());
        rmin = std::max(radii[Nmin], rhalf * std::pow(spacing, -0.5*gridSizeR));
    }
    if(rmax==0) {
        std::nth_element(radii.begin(), radii.end() - Nmin, radii.end());
        rmax = std::min(radii[nbody-Nmin], rhalf * std::pow(spacing, 0.5*gridSizeR));
    }
    utils::msg(utils::VL_DEBUG, "Multipole",
        "Grid in r=["+utils::toString(rmin)+":"+utils::toString(rmax)+"]"
        ", particles span r=["+utils::toString(prmin)+":"+utils::toString(prmax)+"]");
}

/** helper function to determine the coefficients for potential extrapolation:
    assuming that 
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s              if s!=v, or
        Phi(r) = W * (r/r1)^v + U * (r/r1)^s * ln(r/r1)   if s==v,
    and given v and the values of Phi(r1), Phi(r2) and dPhi/dr(r1),
    determine the coefficients s, U and W.
    Here v = l for the inward and v = -l-1 for the outward extrapolation.
    This corresponds to the density profile extrapolated as rho ~ r^(s-2).
*/
void computeExtrapolationCoefs(double Phi1, double Phi2, double dPhi1,
    double r1, double r2, int v, double& s, double& U, double& W)
{
    double lnr = log(r2/r1);
    double num1 = r1*dPhi1, num2 = v*Phi1, den1 = Phi1, den2 = Phi2 * exp(-v*lnr);
    double A = lnr * (num1 - num2) / (den1 - den2);
    bool roundoff =   // check if the value of A is dominated by roundoff errors
        fabs(num1-num2) < std::max(fabs(num1), fabs(num2)) * SAFETY_FACTOR ||
        fabs(den1-den2) < std::max(fabs(den1), fabs(den2)) * SAFETY_FACTOR;
    if(!isFinite(A) || A >= 0 || roundoff)
    {   // no solution - output only the main multipole component (with zero Laplacian)
        U = 0;
        s = 0;
        W = Phi1;
        return;
    }
    // find x(A) such that  x = A * (1 - exp(x)),  where  x = (s-v) * ln(r2/r1);
    // if A is close to -1 (the value for a logarithmic potential), assume it to be exactly -1
    // (otherwise the subsequent operations will incur large cancellation errors)
    s = fabs(A+1)<SQRT_DBL_EPSILON ? v :
        v + (A - math::lambertW(A * exp(A), /*choice of branch*/ A>-1)) / lnr;
    // safeguard against weird slope determination:
    // density may not grow faster than r^-3 at small r, or the potential would be steeper than r^-1
    if(v>=0 && (!isFinite(s)/* || s<=-1*/))
        s = 2;    // results in a constant-density core for the inward extrapolation
    // similarly, density must drop faster than r^-2 at large r,
    // or else the potential would grow indefinitely, even as slowly as log(r), which is not allowed
    if(v<0  && (!isFinite(s) || s>=0))
        s = -2;   // results in a r^-4 falloff for the outward density extrapolation
    if(s != v) {  // non-degenerate case of a power-law extrapolation
        U = (r1*dPhi1 - v*Phi1) / (s-v);
        W = (r1*dPhi1 - s*Phi1) / (v-s);
    } else {      // degenerate logarithmic case (but still permitted for inward extrapolation)
        U = r1*dPhi1 - v*Phi1;
        W = Phi1;
    }
}

/** construct asymptotic power-law potential for extrapolation to small or large radii
    from the given spherical-harmonic coefs and their derivatives at two consecutive radii.
    \param[in]  radii is the array of grid radii (only the first two or the last two elements
    are used, depending on the parameter 'inner');
    \param[in]  Phi  is the full array of potential coefficients (each element corresponds
    to a particular harmonic term and lists the coefficients of this harmonic at all radii);
    \param[in]  dPhi is the array of potential derivatives with the same indexing scheme;
    \param[in]  inner is the flag specifying whether to use the first two or the last two
    elements of these arrays.
    \return  an instance of PowerLawMultipole providing the extrapolation.
*/
PtrPotential initAsympt(const std::vector<double>& radii,
    const std::vector<std::vector<double> >& Phi,
    const std::vector<std::vector<double> >& dPhi, bool inner)
{
    unsigned int nc = Phi.size();  // the number of sph-harm terms in the input arrays
    // limit the number of terms to consider for extrapolation
    const unsigned int lmax = 8;
    nc = std::min<unsigned int>(nc, pow_2(lmax+1));
    std::vector<double> S(nc), U(nc), W(nc);
    // index1 is the endpoint element of each array (the first or the last element, depending on 'inner')
    // index2 is the next-to-endpoint element
    unsigned int index1 = inner? 0 : Phi[0].size()-1;
    unsigned int index2 = inner? 1 : Phi[0].size()-2;

    // determine the coefficients for potential extrapolation at small and large radii
    for(unsigned int c=0; c<nc; c++) {
        int l = math::SphHarmIndices::index_l(c);
        computeExtrapolationCoefs(Phi[c][index1], Phi[c][index2], dPhi[c][index1],
            radii[index1], radii[index2], inner ? l : -l-1,
            /*output*/ S[c], U[c], W[c]);
        // TODO: may need to constrain the slope of l>0 harmonics so that it doesn't exceed
        // that of the l=0 one; this is enforced in the constructor of PowerLawMultipole,
        // but the slope should already have been adjusted before computing the coefs U and W.
        if(l==0)
            utils::msg(utils::VL_DEBUG, "Multipole",
                std::string("Power-law index of ")+(inner?"inner":"outer")+
                " density profile: "+utils::toString(S[c]-2));
    }
    return PtrPotential(new PowerLawMultipole(radii[index1], inner, S, U, W));
}


/** transform Fourier components C_m(r, theta) and their derivs to the actual potential.
    C_m is an array of size nq*nm, where nm is the number of azimuthal harmonics 
    (either mmax+1, or 2*mmax+1, depending on the symmetry encoded in ind),
    and nq is the number of quantities to convert: either 1 (only the potential harmonics),
    3 (potential and its two derivs w.r.t. r and theta), or 6 (plus three second derivs w.r.t.
    r^2, r theta, and theta^2), all stored contiguously in the C_m array:
    first come the nm potential harmonics, then nm harmonics for dPhi/dr, and so on.
    How many quantities are processed is determined by grad and hess being NULL or non-NULL.
*/
void fourierTransformAzimuth(const math::SphHarmIndices& ind, const double phi,
    const double C_m[], double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    const int mmin = ind.mmin();
    const int nm = ind.mmax - mmin + 1;  // number of azimuthal harmonics in C_m array
    // first assign the m=0 harmonic, which is the only one in the axisymmetric case
    if(val)
        *val = C_m[-mmin];
    if(grad) {
        grad->dr      = C_m[-mmin+nm];
        grad->dtheta  = C_m[-mmin+nm*2];
        grad->dphi    = 0;
    }
    if(hess) {
        hess->dr2     = C_m[-mmin+nm*3];
        hess->drdtheta= C_m[-mmin+nm*4];
        hess->dtheta2 = C_m[-mmin+nm*5];
        hess->drdphi  = hess->dthetadphi = hess->dphi2 = 0;
    }
    if(ind.mmax == 0)
        return;
    const bool useSine = mmin<0 || numQuantities>1;
    // temporary storage for trigonometric functions - allocated on the stack, automatically freed
    double* trig_m = static_cast<double*>(alloca(ind.mmax*(1+useSine) * sizeof(double)));
    math::trigMultiAngle(phi, ind.mmax, useSine, trig_m);
    for(int mm=0; mm<nm; mm++) {
        int m = mm + mmin;
        if(m==0)
            continue;  // the m=0 terms were set at the beginning
        if(ind.lmin(m)>ind.lmax)
            continue;  // empty harmonic
        double trig  = m>0 ? trig_m[m-1] : trig_m[ind.mmax-m-1];  // cos or sin
        double dtrig = m>0 ? -m*trig_m[ind.mmax+m-1] : -m*trig_m[-m-1];
        double d2trig = -m*m*trig;
        if(val)
            *val += C_m[mm] * trig;
        if(grad) {
            grad->dr     += C_m[mm+nm  ] *  trig;
            grad->dtheta += C_m[mm+nm*2] *  trig;
            grad->dphi   += C_m[mm]      * dtrig;
        }
        if(hess) {
            hess->dr2       += C_m[mm+nm*3] *   trig;
            hess->drdtheta  += C_m[mm+nm*4] *   trig;
            hess->dtheta2   += C_m[mm+nm*5] *   trig;
            hess->drdphi    += C_m[mm+nm  ] *  dtrig;
            hess->dthetadphi+= C_m[mm+nm*2] *  dtrig;
            hess->dphi2     += C_m[mm]      * d2trig;
        }
    }
}

/** transform sph.-harm. coefs of potential (C_lm) and its first (dC_lm) and second (d2C_lm)
    derivatives w.r.t. arbitrary function of radius to the value, gradient and hessian of 
    potential in spherical coordinates (w.r.t. the same function of radius).
*/
void sphHarmTransformInverseDeriv(
    const math::SphHarmIndices& ind, const coord::PosCyl& pos,
    const double C_lm[], const double dC_lm[], const double d2C_lm[],
    double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    // temporary storage for coefficients - allocated on the stack, automatically freed
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    int sizeC = 6 * (2*ind.mmax+1), sizeP = ind.lmax+1;
    double*   C_m  = static_cast<double*>(alloca((sizeC + 3*sizeP) * sizeof(double)));
    double*   P_lm = C_m + sizeC;
    double*  dP_lm = numQuantities>=3 ? P_lm + sizeP   : NULL;
    double* d2P_lm = numQuantities==6 ? P_lm + sizeP*2 : NULL;
    const double tau = pos.z / (sqrt(pow_2(pos.R) + pow_2(pos.z)) + pos.R);
    const int nm = ind.mmax - ind.mmin() + 1;  // number of azimuthal harmonics in C_m array
    for(int mm=0; mm<nm; mm++) {
        int m = mm + ind.mmin();
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;  // extra factor sqrt{2} for m!=0 trig fncs
        for(int q=0; q<numQuantities; q++)
            C_m[mm + q*nm] = 0;
        int absm = abs(m);
        math::sphHarmArray(ind.lmax, absm, tau, P_lm, dP_lm, d2P_lm);
        for(int l=lmin; l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m), p = l-absm;
            C_m[mm] += P_lm[p] * C_lm[c] * mul;
            if(numQuantities>=3) {
                C_m[mm + nm  ] +=  P_lm[p] * dC_lm[c] * mul;   // dPhi_m/dr
                C_m[mm + nm*2] += dP_lm[p] *  C_lm[c] * mul;   // dPhi_m/dtheta
            }
            if(numQuantities==6) {
                C_m[mm + nm*3] +=   P_lm[p] * d2C_lm[c] * mul; // d2Phi_m/dr2
                C_m[mm + nm*4] +=  dP_lm[p] *  dC_lm[c] * mul; // d2Phi_m/drdtheta
                C_m[mm + nm*5] += d2P_lm[p] *   C_lm[c] * mul; // d2Phi_m/dtheta2
            }
        }
    }
    fourierTransformAzimuth(ind, pos.phi, C_m, val, grad, hess);
}

/** optimized version of sphHarmTransformInverseDeriv for the case lmax=2, mmin=0, step=2,
    processing only the {l=0,m=0}, {l=2,m=0} and optionally {l=2,m=2} terms (a common special case)
*/
void sphHarmTransformInverseDeriv2(
    const math::SphHarmIndices& ind, const coord::PosCyl& pos,
    const double C_lm[], const double dC_lm[], const double d2C_lm[],
    double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    static const int i00 = ind.index(0,0), i20 = ind.index(2,0), i22 = ind.index(2,2);
    static const double C2 = sqrt(1.25), D2 = sqrt(3.75);
    const double 
    tau = pos.z / (sqrt(pow_2(pos.R) + pow_2(pos.z)) + pos.R),
    ct  =      2 * tau  / (1 + tau*tau),  // cos(theta)
    st  = (1 - tau*tau) / (1 + tau*tau),  // sin(theta)
    cc  = ct * ct, cs = ct * st, ss = st * st,
    Y20       = (3*C2 * cc - C2),
    dY20      = -6*C2 * cs,
    d2Y20     =-12*C2 * cc + 6*C2,
    Phi0      =   Y20 *   C_lm[i20] +   C_lm[i00],
    dPhi0dr   =   Y20 *  dC_lm[i20] +  dC_lm[i00],
    dPhi0dt   =  dY20 *   C_lm[i20],
    d2Phi0dr2 =   Y20 * d2C_lm[i20] + d2C_lm[i00],
    d2Phi0drt =  dY20 *  dC_lm[i20],
    d2Phi0dt2 = d2Y20 *   C_lm[i20];
    if(val)
        *val = Phi0;
    if(grad) {
        grad->dr      = dPhi0dr;
        grad->dtheta  = dPhi0dt;
        grad->dphi    = 0;
    }
    if(hess) {
        hess->dr2     = d2Phi0dr2;
        hess->drdtheta= d2Phi0drt;
        hess->dtheta2 = d2Phi0dt2;
        hess->drdphi  = hess->dthetadphi = hess->dphi2 = 0;
    }
    if(ind.mmax == 2) {
        double sp, cp,
        Y22       =    D2 * ss,
        dY22      =  2*D2 * cs,
        d2Y22     =  4*D2 * cc - 2*D2,
        Phi2      =   Y22 *   C_lm[i22],
        dPhi2dr   =   Y22 *  dC_lm[i22],
        dPhi2dt   =  dY22 *   C_lm[i22],
        d2Phi2dr2 =   Y22 * d2C_lm[i22],
        d2Phi2drt =  dY22 *  dC_lm[i22],
        d2Phi2dt2 = d2Y22 *   C_lm[i22];
        math::sincos(2*pos.phi, sp, cp);
        if(val)
            *val += Phi2 * cp;
        if(grad) {
            grad->dr     += dPhi2dr *    cp;
            grad->dtheta += dPhi2dt *    cp;
            grad->dphi   +=  Phi2   * -2*sp;
        }
        if(hess) {
            hess->dr2       += d2Phi2dr2 *    cp;
            hess->drdtheta  += d2Phi2drt *    cp;
            hess->dtheta2   += d2Phi2dt2 *    cp;
            hess->drdphi    +=  dPhi2dr  * -2*sp;
            hess->dthetadphi+=  dPhi2dt  * -2*sp;
            hess->dphi2     +=   Phi2    * -4*cp;
        }
    }
}

// transform potential derivatives from {ln(r), theta} to {R, z}
void transformDerivsSphToCyl(const coord::PosCyl& pos,
    const coord::GradSph &gradSph, const coord::HessSph &hessSph,
    coord::GradCyl *gradCyl, coord::HessCyl *hessCyl)
{
    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // in the source grad/hess we have derivs w.r.t. ln(r) instead of r
    const double r2inv = 1 / (pow_2(pos.R) + pow_2(pos.z));
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = pos.R * r2inv;
    der.drdz = pos.z * r2inv;
    der.dthetadR =  der.drdz;
    der.dthetadz = -der.drdR;
    if(gradCyl)
        *gradCyl = toGrad(gradSph, der);
    if(hessCyl) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2      = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz     = -2 * der.drdR * der.drdz;
        der2.d2rdz2      = -der2.d2rdR2;
        der2.d2thetadR2  =  der2.d2rdRdz;
        der2.d2thetadRdz = -der2.d2rdR2;
        der2.d2thetadz2  = -der2.d2rdRdz;
        *hessCyl = toHess(gradSph, hessSph, der, der2);
    }
}

}  // end internal namespace


//---- driver functions for computing sph-harm coefficients of density or potential ----//

// density coefs from density
void computeDensityCoefsSph(const BaseDensity& src,
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> > &output)
{
    std::vector< std::vector<double> > *coefs = &output;
    computeSphHarmCoefs<BaseDensity>(src, ind, gridRadii, /*parallel*/ true, /*output*/ &coefs);
}

// density coefs from a multicomponent density
void computeDensityCoefsSph(const math::IFunctionNdim& src,
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector< std::vector<double> > > &coefs)
{
    const unsigned int N = src.numValues();
    coefs.resize(N);
    // prepare the array of pointers to the output vectors
    std::vector< std::vector< std::vector<double> > *> coefRefs(N);
    for(unsigned int i=0; i<N; i++)
        coefRefs[i] = &coefs[i];
    computeSphHarmCoefs<math::IFunctionNdim>(src, ind, gridRadii, /*parallel*/ true,
        /*output*/ &coefRefs.front());
}

// density coefs from N-body snapshot
void computeDensityCoefsSph(
    const particles::ParticleArray<coord::PosCyl> &particles,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &coefs,
    double smoothing)
{
    unsigned int gridSizeR = gridRadii.size();
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("computeDensityCoefsSph: radial grid size too small");
    for(unsigned int k=0; k<gridSizeR; k++)
        if(gridRadii[k] <= (k==0 ? 0 : gridRadii[k-1]))
            throw std::invalid_argument("computePotentialCoefs: "
                "radii of grid points must be positive and sorted in increasing order");

    // allocate the output arrays
    std::vector<double> gridLogRadii(gridSizeR);
    for(size_t i=0; i<gridSizeR; i++)
        gridLogRadii[i] = log(gridRadii[i]);
    coefs.assign(ind.size(), std::vector<double>(gridSizeR, 0.));

    // compute the sph-harm coefs at each particle's radius
    std::vector<std::vector<double> > harmonics(ind.size());
    std::vector<double> particleRadii;
    computeSphericalHarmonicsFromParticles(particles, ind, particleRadii, harmonics);
    size_t nbody = particleRadii.size();
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress

    // convert the radii to log-radii and store particle masses in the {l=0,m=0} harmonic
    for(size_t i=0; i<nbody; i++) {
        if((harmonics[0][i] = particles.mass(i)) == 0) continue;   // ignore particles with zero mass
        if(particleRadii[i] == 0)
            throw std::runtime_error("computeDensityCoefsSph: no massive particles at r=0 allowed");
        particleRadii[i] = log(particleRadii[i]);
    }

    // construct the l=0 harmonic using a penalized log-density estimate
    math::CubicSpline spl0(gridLogRadii, math::splineLogDensity<3>(
        gridLogRadii, particleRadii, /*weights*/harmonics[0],
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));
    for(unsigned int k=0; k<gridSizeR; k++)
        coefs[0][k] = exp(spl0(gridLogRadii[k])) / (4*M_PI*pow_3(gridRadii[k]));
    if(utils::verbosityLevel >= utils::VL_DEBUG) {
        double innerSlope, outerSlope;
        spl0.evalDeriv(gridLogRadii.front(), NULL, &innerSlope);
        spl0.evalDeriv(gridLogRadii.back(),  NULL, &outerSlope);
        innerSlope -= 3;  // we have computed the log of density of particles in log(r),
        outerSlope -= 3;  // which is equal to the true density multiplied by 4 pi r^3
        utils::msg(utils::VL_DEBUG, "Multipole",
            "Power-law index of density profile: inner="+utils::toString(innerSlope)+
            ", outer="+utils::toString(outerSlope));
    }
    if(ind.size()==1)
        return;

    // obtain the list of nontrivial l>0 harmonics
    // (TODO: this should be made part of math::SphHarmIndices interface!)
    std::vector<unsigned int> nonzeroCoefs;
    for(unsigned int c=1; c<ind.size(); c++)
        if(!harmonics[c].empty())
            nonzeroCoefs.push_back(c);
    int nonzeroCoefsSize = nonzeroCoefs.size();

    // construct the l>0 terms by fitting a penalized smoothing spline,
    // where the penalty is given for "wiggliness" of the curve.
    // The amount of smoothing is specified through the number of "equivalent degrees of freedom" (EDF),
    // which ranges between 2 (infinite penalty resulting in a straight-line fit)
    // to gridSizeR for the case of zero penalty.
    // we set edf roughly half-way between the two extremes for the default value of smoothing=1
    math::SplineApprox fitter(gridLogRadii, particleRadii, /*weights*/harmonics[0]);
    double edf = 2 + (gridSizeR-2) / (smoothing+1);
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int h=0; h<nonzeroCoefsSize; h++) {
        if(cbrk.triggered()) continue;
        try{
            math::CubicSpline splc(gridLogRadii, fitter.fit(harmonics[nonzeroCoefs[h]], edf));
            // multiply the coefs by the value of the l=0 term (which is the spherical density estimate)
            for(unsigned int k=0; k<gridSizeR; k++)
                coefs[nonzeroCoefs[h]][k] = splc(gridLogRadii[k]) * coefs[0][k];
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error("Keyboard interrupt");
    if(!errorMsg.empty())
        throw std::runtime_error("computeDensityCoefsSph: " + errorMsg);
}

// potential coefs from potential
void computePotentialCoefsSph(const BasePotential &src,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi)
{
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential>(src, ind, gridRadii, /*parallel*/ false, /*output*/ coefs);
}

// potential coefs from density:
// core function to solve Poisson equation in spherical harmonics for a smooth density profile
void computePotentialCoefsSph(const BaseDensity& dens,
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> >& Phi,
    std::vector< std::vector<double> >& dPhi)
{
    int gridSizeR = gridRadii.size();
    if(gridSizeR < (int)MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("computePotentialCoefsSph: radial grid size too small");
    for(int k=0; k<gridSizeR; k++)
        if(gridRadii[k] <= (k==0 ? 0 : gridRadii[k-1]))
            throw std::invalid_argument("computePotentialCoefsSph: "
                "radii of grid points must be positive and sorted in increasing order");

    // several intermediate arrays are aliased with the output arrays,
    // but are denoted by different names to clearly mark their identity
    std::vector< std::vector<double> >& Qint = Phi;
    std::vector< std::vector<double> >& Qext = dPhi;
    std::vector< std::vector<double> >& Pint = Phi;
    std::vector< std::vector<double> >& Pext = dPhi;
    Phi .assign(ind.size(), std::vector<double>(gridSizeR, 0.));
    dPhi.assign(ind.size(), std::vector<double>(gridSizeR, 0.));

    // pre-computed tables for (non-adaptive) Gauss-Legendre integration over radius
    const double *glnodes = math::GLPOINTS[GLORDER_RAD], *glweights = math::GLWEIGHTS[GLORDER_RAD];

    // prepare SH transformation
    math::SphHarmTransformForward trans(ind);

    // Loop over radial grid segments and compute integrals of rho_lm(r) times powers of radius,
    // for each interval of radii in the input grid (0 <= k < Nr):
    //   Qint[l,m][k] = \int_{r_{k-1}}^{r_k} \rho_{l,m}(r) (r/r_k)^{l+2} dr,  with r_{-1} = 0;
    //   Qext[l,m][k] = \int_{r_k}^{r_{k+1}} \rho_{l,m}(r) (r/r_k)^{1-l} dr,  with r_{Nr} = \infty.
    // Here \rho_{l,m}(r) are the sph.-harm. coefs for density at each radius.

    // The first loop below used to be OpenMP-parallelized, but now this is disabled
    // because the parallelization overheads do not compensate for speedup:
    // the computation of potential by 1d integration in radius is typically quite cheap by itself;
    // if the density computation is expensive, then one should construct an intermediate
    // DensitySphericalHarmonic interpolator and pass it to this routine.
    std::vector<double> densValues(trans.size());
    std::vector<double> tmpCoefs(ind.size());
    for(int k=0; k<=gridSizeR; k++) {
        double rkminus1 = (k>0 ? gridRadii[k-1] : 0);
        double deltaGridR = k<gridSizeR ?
            gridRadii[k] - rkminus1 :  // length of k-th radial segment
            gridRadii.back();          // last grid segment extends to infinity

        // loop over ORDER_RAD_INT nodes of GL quadrature for each radial grid segment
        for(unsigned int s=0; s<GLORDER_RAD; s++) {
            double r = k<gridSizeR ?
                rkminus1 + glnodes[s] * deltaGridR :  // radius inside ordinary k-th segment
                // special treatment for the last segment which extends to infinity:
                // the integration variable is t = r_{Nr-1} / r
                gridRadii.back() / glnodes[s];

            // collect the values of density at all points of angular grid at the given radius
            for(unsigned int i=0; i<densValues.size(); i++)
                densValues[i] = dens.density(coord::PosCyl(
                    r * sqrt(1-pow_2(trans.costheta(i))), r * trans.costheta(i), trans.phi(i)));

            // compute density SH coefs
            trans.transform(&densValues.front(), &tmpCoefs.front());
            math::eliminateNearZeros(tmpCoefs);

            // accumulate integrals over density times radius in the Qint and Qext arrays
            for(int m=ind.mmin(); m<=ind.mmax; m++) {
                for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    if(k<gridSizeR)
                        // accumulate Qint for all segments except the one extending to infinity
                        Qint[c][k] += tmpCoefs[c] * glweights[s] * deltaGridR *
                            math::pow(r / gridRadii[k], l+2);
                    if(k>0)
                        // accumulate Qext for all segments except the innermost one
                        // (which starts from zero), with a special treatment for last segment
                        // that extends to infinity and has a different integration variable
                        Qext[c][k-1] += tmpCoefs[c] * glweights[s] * deltaGridR *
                            (k==gridSizeR ? 1 / pow_2(glnodes[s]) : 1) * // jacobian of 1/r transform
                            math::pow(r / gridRadii[k-1], 1-l);
                }
            }
        }
    }

    // Run the summation loop, replacing the intermediate values Qint, Qext
    // with the interior and exterior potential coefficients (stored in the same arrays):
    //   Pint_{l,m}(r) = r^{-l-1} \int_0^r \rho_{l,m}(s) s^{l+2} ds ,
    //   Pext_{l,m}(r) = r^l \int_r^\infty \rho_{l,m}(s) s^{1-l} ds ,
    // In doing so, we use a recurrent relation that avoids over/underflows when
    // dealing with large powers of r, by replacing r^n with (r/r_prev)^n.
    // Finally, compute the total potential and its radial derivative for each SH term.
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);

            // Compute Pint by summing from inside out, using the recurrent relation
            // Pint(r_{k+1}) r_{k+1}^{l+1} = Pint(r_k) r_k^{l+1} + Qint[k] * r_k^{l+2}
            double val = 0;
            for(int k=0; k<gridSizeR; k++) {
                if(k>0)
                    val *= math::pow(gridRadii[k-1] / gridRadii[k], l+1);
                val += gridRadii[k] * Qint[c][k];
                Pint[c][k] = val;
            }

            // Compute Pext by summing from outside in, using the recurrent relation
            // Pext(r_k) r_k^{-l} = Pext(r_{k+1}) r_{k+1}^{-l} + Qext[k] * r_k^{1-l}
            val = 0;
            for(int k=gridSizeR-1; k>=0; k--) {
                if(k<gridSizeR-1)
                    val *= math::pow(gridRadii[k] / gridRadii[k+1], l);
                val += gridRadii[k] * Qext[c][k];
                Pext[c][k] = val;
            }

            // Finally, put together the interior and exterior coefs to compute 
            // the potential and its radial derivative for each spherical-harmonic term
            double mul = -4*M_PI / (2*l+1);
            for(int k=0; k<gridSizeR; k++) {
                double tmpPhi = mul * (Pint[c][k] + Pext[c][k]);
                dPhi[c][k]    = mul * (-(l+1)*Pint[c][k] + l*Pext[c][k]) / gridRadii[k];
                // extra step needed because Phi/dPhi and Pint/Pext are aliased
                Phi [c][k]    = tmpPhi;
            }
        }
    }
}

//------ Spherical-harmonic expansion of density ------//

// static factory methods
PtrDensity DensitySphericalHarmonic::create(
    const BaseDensity& dens, int lmax, int mmax, 
    unsigned int gridSizeR, double rmin, double rmax, bool accurateIntegration)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<=0 || rmax<=rmin)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of min/max grid radii");
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");
    // to improve accuracy of SH coefficient computation, we increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(dens) ? 0 :
        accurateIntegration? std::max<int>(lmax, LMIN_SPHHARM) : lmax;
    int mmax_tmp = isZRotSymmetric(dens) ? 0 :
        accurateIntegration? std::max<int>(mmax, LMIN_SPHHARM) : mmax;
    std::vector<std::vector<double> > coefs;
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    computeDensityCoefsSph(dens,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, dens.symmetry()),
        gridRadii, coefs);
    // resize the coefficients back to the requested order
    restrictSphHarmCoefs(lmax, mmax, coefs);
    return PtrDensity(new DensitySphericalHarmonic(gridRadii, coefs));
}

PtrDensity DensitySphericalHarmonic::create(
    const particles::ParticleArray<coord::PosCyl> &particles,
    coord::SymmetryType sym, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax, double smoothing)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("DensitySphericalHarmonic: invalid grid parameters");
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");
    chooseGridRadii(particles, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    if(isSpherical(sym))
        lmax = 0;
    if(isZRotSymmetric(sym))
        mmax = 0;
    std::vector<std::vector<double> > coefs;
    computeDensityCoefsSph(particles, math::SphHarmIndices(lmax, mmax, sym), gridRadii, coefs, smoothing);
    return PtrDensity(new DensitySphericalHarmonic(gridRadii, coefs));
}

// the actual constructor
DensitySphericalHarmonic::DensitySphericalHarmonic(const std::vector<double> &_gridRadii,
    const std::vector< std::vector<double> > &coefs) :
    BaseDensity(),
    gridRadii(_gridRadii),
    ind(getIndicesFromCoefs(coefs)),
    logScaling(true)
{
    unsigned int gridSizeR = gridRadii.size();
    if(coefs.size() != ind.size() || gridSizeR < MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");
    for(unsigned int n=0; n<coefs.size(); n++)
        if(coefs[n].size() != gridSizeR)
            throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");

    spl.resize(ind.size());
    std::vector<double> gridLogR(gridSizeR), tmparr(gridSizeR);
    for(size_t i=0; i<gridSizeR; i++)
        gridLogR[i] = log(gridRadii[i]);

    // set up 1d splines in radius for each non-trivial (l,m) coefficient
    for(unsigned int c=0; c<ind.size(); c++) {
        int l = ind.index_l(c);
        int m = ind.index_m(c);
        if(ind.lmin(m) > ind.lmax || (l-ind.lmin(m)) % ind.step != 0)
            continue;  // skip identically zero harmonics
        if(l==0) {
            // for the l=0 coefficient (main spherically-symmetric component), we may use
            // logarithmic scaling of its amplitude, if it is everywhere positive
            for(unsigned int k=0; k<gridSizeR; k++)
                logScaling &= coefs[0][k] > 0;
            for(unsigned int k=0; k<gridSizeR; k++)
                tmparr[k] = logScaling ? log(coefs[0][k]) : coefs[0][k];
            spl[0].reset(new math::CubicSpline(gridLogR, tmparr));
            // when using log-scaling, the endpoint derivatives of spline are simply
            // d[log(rho(log(r)))] / d[log(r)] = power-law indices of the inner/outer slope;
            // without log-scaling, the endpoint derivatives of spline are
            // d[rho(log(r))] / d[log(r)] = power-law indices multiplied by the values of rho.
            spl[0]->evalDeriv(gridLogR.front(), NULL, &innerSlope);
            spl[0]->evalDeriv(gridLogR.back(),  NULL, &outerSlope);
            if(!logScaling) {
                if(coefs[0].front() != 0)
                    innerSlope /= coefs[0].front();
                else
                    innerSlope = 0;
                if(coefs[0].back() != 0)
                    outerSlope /= coefs[0].back();
                else
                    outerSlope = 0;
            }

            // We check (and correct if necessary) the logarithmic slope of density profile
            // at the innermost and outermost grid radii, used in power-law extrapolation.
            // slope = (1/rho) d(rho)/d(logr), is usually negative (at least at large radii).
            // Note that the inner slope less than -2 leads to a divergent potential at origin,
            // but the enclosed mass is still finite if slope is greater than -3;
            // similarly, outer slope greater than -3 leads to a divergent total mass,
            // but the potential tends to a finite limit as long as the slope is less than -2.
            // Both these 'dangerous' semi-infinite regimes are allowed here,
            // but likely may result in problems elsewhere.
            if(!isFinite(innerSlope))
                innerSlope = 0;
            if(!isFinite(outerSlope))
                outerSlope = coefs[0].back()==0 ? 0 : -4.;
            innerSlope = std::max(innerSlope, -2.8);
            outerSlope = std::min(outerSlope, -2.2);
        } else {
            // values of l!=0 components are normalized to the value of l=0 component at each radius
            // and are extrapolated as constants beyond the extent of the grid
            // (with zero endpoint derivatives)
            for(unsigned int k=0; k<gridSizeR; k++)
                tmparr[k] = coefs[0][k]!=0 ? coefs[c][k] / coefs[0][k] : 0;
            spl[c].reset(new math::CubicSpline(gridLogR, tmparr, /*regularize*/false, 0, 0));
        }
    }
}

void DensitySphericalHarmonic::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefs) const
{
    radii = gridRadii;
    computeDensityCoefsSph(*this, ind, radii, coefs);
}

double DensitySphericalHarmonic::densityCyl(const coord::PosCyl &pos) const
{
    assert(spl[0]);  // 0th harmonic should always be present
    // temporary array allocated on the stack
    double* coefs = static_cast<double*>(alloca(ind.size() * sizeof(double)));
    double r = sqrt(pow_2(pos.R) + pow_2(pos.z) );
    double rmin = gridRadii.front(), rmax = gridRadii.back();
    double logr = log(math::clip(r, rmin, rmax));  // the argument of spline functions
    // first compute the l=0 coefficient, possibly log-unscaled
    coefs[0] = spl[0]->value(logr);
    if(logScaling)
        coefs[0] = exp(coefs[0]);
    // extrapolate if necessary
    if(r < rmin)
        coefs[0] *= pow(r / rmin, innerSlope);
    if(r > rmax)
        coefs[0] *= pow(r / rmax, outerSlope);
    // then compute other coefs, which are scaled by the value of l=0 coef
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            if(c!=0)
                coefs[c] = spl[c]->value(logr) * coefs[0];
        }
    double tau = pos.z / (sqrt(pow_2(pos.R) + pow_2(pos.z)) + pos.R);
    return math::sphHarmTransformInverse(ind, coefs, tau, pos.phi);
}


//----- declarations of two multipole potential interpolators -----//

class MultipoleInterp1d: public BasePotentialCyl {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp1d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp1d"; };
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// interpolation splines in log(r) for each {l,m} sph.-harm. component of potential
    std::vector<math::QuinticSpline> spl;
    /// whether to perform log-scaling on the l=0 component
    bool logScaling;
    /// the inverse of the value of potential at origin (if using log-scaling), may be zero
    double invPhi0;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

class MultipoleInterp2d: public BasePotentialCyl {
public:
    /** construct interpolating splines from the values and derivatives of harmonic coefficients */
    MultipoleInterp2d(
        const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return "MultipoleInterp2d"; }
private:
    /// indexing scheme for sph.-harm. coefficients
    math::SphHarmIndices ind;
    /// 2d interpolation splines in meridional plane for each azimuthal harmonic (m) component
    std::vector<math::QuinticSpline2d> spl;
    /// whether to perform log-scaling on the m=0 component
    bool logScaling;
    /// the inverse of the value of potential at origin (if using log-scaling), may be zero
    double invPhi0;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

template<class BaseDensityOrPotential>
PtrPotential createMultipole(
    const BaseDensityOrPotential& src,
    int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("Multipole: invalid grid parameters");
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");
    chooseGridRadii(src, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    if(isSpherical(src))
        lmax = 0;
    if(isZRotSymmetric(src))
        mmax = 0;
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp=0, mmax_tmp=0;
    if(isSpherical(src))
        lmax = 0;   // don't waste effort on computing non-spherical harmonic terms
    else
        lmax_tmp = std::max<int>(lmax, LMIN_SPHHARM);
    if(isZRotSymmetric(src))
        mmax = 0;   // similarly for non-axisymmetric harmonic terms
    else
        mmax_tmp = std::max<int>(mmax, LMIN_SPHHARM);
    std::vector<std::vector<double> > Phi, dPhi;
    computePotentialCoefsSph(src,
        math::SphHarmIndices(lmax_tmp, mmax_tmp, src.symmetry()),
        gridRadii, Phi, dPhi);
    // resize the coefficients back to the requested order
    restrictSphHarmCoefs(lmax, mmax, Phi);
    restrictSphHarmCoefs(lmax, mmax, dPhi);
    return PtrPotential(new Multipole(gridRadii, Phi, dPhi));
}

//------ the driver class for multipole potential ------//

PtrPotential Multipole::create(
    const BaseDensity& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax);
}

PtrPotential Multipole::create(
    const BasePotential& src, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    return createMultipole(src, lmax, mmax, gridSizeR, rmin, rmax);
}

PtrPotential Multipole::create(
    const particles::ParticleArray<coord::PosCyl> &particles,
    coord::SymmetryType sym, int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax, double smoothing)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("Multipole: invalid grid parameters");
    if(lmax<0 || mmax<0 || mmax>lmax)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");
    // we first reduce the grid size by two, and then add two extra nodes at ends:
    // this is necessary for an accurate extrapolation that requires a robust estimate
    // of 2nd derivative of potential, using 0th and 1st radial node at each end.
    // The constructed density approximation has a power-law asymptotic behaviour
    // beyond its grid domain, and by creating additional grid point for the potential,
    // we robustly capture this power-law slope.
    gridSizeR = std::max<unsigned int>(gridSizeR-2, MULTIPOLE_MIN_GRID_SIZE);
    chooseGridRadii(particles, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    if(isSpherical(sym))
        lmax = 0;
    if(isZRotSymmetric(sym))
        mmax = 0;
    std::vector<std::vector<double> > coefDens, Phi, dPhi;
    math::SphHarmIndices ind(lmax, mmax, sym);
    // create an intermediate density approximation
    computeDensityCoefsSph(particles, ind, gridRadii, coefDens, smoothing);
    DensitySphericalHarmonic dens(gridRadii, coefDens);
    // add two extra points to the radial grid for the potential, for the reason explained above
    gridRadii.insert(gridRadii.begin(), pow_2(gridRadii[0])/gridRadii[1]);
    gridRadii.push_back(pow_2(gridRadii.back())/gridRadii[gridRadii.size()-2]);
    computePotentialCoefsSph(dens, ind, gridRadii, Phi, dPhi);
    return PtrPotential(new Multipole(gridRadii, Phi, dPhi));
}

// now the one and only 'proper' constructor
Multipole::Multipole(
    const std::vector<double> &_gridRadii,
    const std::vector<std::vector<double> > &Phi,
    const std::vector<std::vector<double> > &dPhi) :
    gridRadii(_gridRadii), ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = gridRadii.size();
    bool correct = gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        Phi.size() == ind.size() && dPhi.size() == ind.size();
    for(unsigned int c=0; correct && c<ind.size(); c++)
        correct &= Phi[c].size() == gridSizeR && dPhi[c].size() == gridSizeR;
    for(unsigned int k=1; correct && k<gridSizeR; k++)
        correct &= gridRadii[k] > gridRadii[k-1];
    if(!correct)
        throw std::invalid_argument("Multipole: invalid radial grid");

    // construct the interpolating splines:
    // choose between 1d or 2d splines, depending on the expected efficiency
    impl = ind.lmax <= LMAX_1D_SPLINE ?
        PtrPotential(new MultipoleInterp1d(gridRadii, Phi, dPhi)) :
        PtrPotential(new MultipoleInterp2d(gridRadii, Phi, dPhi));

    // determine asymptotic behaviour at small and large radii
    asymptInner = initAsympt(gridRadii, Phi, dPhi, true);
    asymptOuter = initAsympt(gridRadii, Phi, dPhi, false);
}

void Multipole::getCoefs(
    std::vector<double> &radii,
    std::vector<std::vector<double> > &Phi,
    std::vector<std::vector<double> > &dPhi) const
{
    radii = gridRadii;
    radii.front() *= 1+SAFETY_FACTOR;  // avoid the possibility of getting outside the of radii where
    radii.back()  *= 1-SAFETY_FACTOR;  // the interpolating splines are defined, due to roundoff errors
    // use the fact that the spherical-harmonic transform is invertible to machine precision:
    // take the values and derivatives of potential at grid nodes and apply forward transform to obtain
    // the coefficients (however, this may lose a few digits of precision for higher-order terms).
    computePotentialCoefsSph(*impl, ind, radii, Phi, dPhi);
    radii = gridRadii;  // restore the original values of radii
}

void Multipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const
{
    double rsq = pow_2(pos.R) + pow_2(pos.z);
    if(rsq < pow_2(gridRadii.front()) * (1+SAFETY_FACTOR))
        asymptInner->eval(pos, potential, deriv, deriv2);
    else if(rsq > pow_2(gridRadii.back()) * (1-SAFETY_FACTOR))
        asymptOuter->eval(pos, potential, deriv, deriv2);
    else
        impl->eval(pos, potential, deriv, deriv2);
}

double Multipole::densityCyl(const coord::PosCyl &pos) const
{
    double rsq = pow_2(pos.R) + pow_2(pos.z);
    if(rsq < pow_2(gridRadii.front()) * (1+SAFETY_FACTOR))
        return asymptInner->density(pos);
    else if(rsq > pow_2(gridRadii.back()) * (1-SAFETY_FACTOR))
        return asymptOuter->density(pos);  // gives a more accurate result than the default implementation
    else
        return impl->density(pos);
}

double Multipole::enclosedMass(double radius) const
{
    if(radius == 0)
        return 0;  // TODO: this may not be correct for a potential of a point mass!
    // use the l=0 harmonic term of dPhi/dr to estimate the spherically-averaged enclosed mass
    if(radius == INFINITY)
        radius = gridRadii.back() * 1e20;
    const BasePotential& pot =
        radius <= gridRadii.front()* (1+SAFETY_FACTOR) ? *asymptInner :
        radius >= gridRadii.back() * (1-SAFETY_FACTOR) ? *asymptOuter : *impl;
    std::vector< std::vector<double> > Phi, dPhi;
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential>(pot, ind,
        /*a single point in radius*/ std::vector<double>(1, radius),
        /*parallel*/ false, /*output*/ coefs);
    return pow_2(radius) * dPhi[0][0];
}

// ------- Implementations of multipole potential interpolators ------- //

// ------- PowerLawPotential ------- //

PowerLawMultipole::PowerLawMultipole(double _r0, bool _inner,
    const std::vector<double>& _S,
    const std::vector<double>& _U,
    const std::vector<double>& _W) :
    ind(math::getIndicesFromCoefs(_U)), r0sq(_r0*_r0), inner(_inner), S(_S), U(_U), W(_W) 
{
    // safeguard against errors in slope determination - 
    // ensure that all harmonics with l>0 do not asymptotically overtake the principal one (l=0)
    for(unsigned int c=1; c<S.size(); c++)
        if(U[c]!=0 && ((inner && S[c] < S[0]) || (!inner && S[c] > S[0])) )
            S[c] = S[0];
}

void PowerLawMultipole::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    bool needGrad = grad!=NULL || hess!=NULL;
    bool needHess = hess!=NULL;
    unsigned int size = ind.size() * (needHess ? 3 : needGrad ? 2 : 1);
    double*   Phi_lm = static_cast<double*>(alloca(size * sizeof(double)));
    double*  dPhi_lm = needGrad ?  Phi_lm + ind.size() : NULL;
    double* d2Phi_lm = needHess ? dPhi_lm + ind.size() : NULL;
    double rsq   = pow_2(pos.R) + pow_2(pos.z);
    double dlogr = log(rsq / r0sq) * 0.5;
    // simplified treatment in strongly asymptotic regime - retain only l==0 term
    int lmax = (inner && rsq < r0sq*1e-16) || (!inner && rsq > r0sq*1e16) ? 0 : ind.lmax;

    // define {v=l, r0=rmin} for the inner or {v=-l-1, r0=rmax} for the outer extrapolation;
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}}            + W_{l,m} * (r/r0)^v   if s!=v,
    // Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0) + W_{l,m} * (r/r0)^v   if s==v.
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            double s=S[c], u=U[c], w=W[c], v = inner ? l : -l-1;
            double rv  = v!=0 ? exp( dlogr * v ) : 1;                // (r/r0)^v
            double rs  = s!=v ? (s!=0 ? exp( dlogr * s ) : 1) : rv;  // (r/r0)^s
            double urs = u * rs * (s!=v || u==0 ? 1 : dlogr);  // if s==v, multiply by ln(r/r0)
            double wrv = w * rv;
            Phi_lm[c] = urs + wrv;
            if(needGrad)
                dPhi_lm[c] = urs*s + wrv*v + (s!=v ? 0 : u*rs);
            if(needHess)
                d2Phi_lm[c] = urs*s*s + wrv*v*v + (s!=v ? 0 : 2*s*u*rs);
        }
    if(lmax == 0) {  // fast track
        if(potential)
            *potential = Phi_lm[0];
        double rsqinv = rsq>0 ? 1/rsq : 0,
            Rr2 = rsq<INFINITY ? pos.R * rsqinv : 0,
            zr2 = rsq<INFINITY ? pos.z * rsqinv : 0;
        if(grad) {
            grad->dR = dPhi_lm[0] * Rr2;
            grad->dz = dPhi_lm[0] * zr2;
            grad->dphi = 0;
        }
        if(hess) {
            double d2 = d2Phi_lm[0] - 2 * dPhi_lm[0];
            hess->dR2 = d2 * pow_2(Rr2) + dPhi_lm[0] * rsqinv;
            hess->dz2 = d2 * pow_2(zr2) + dPhi_lm[0] * rsqinv;
            hess->dRdz= d2 * Rr2 * zr2;
            hess->dRdphi = hess->dzdphi = hess->dphi2 = 0;
        }
        return;
    }
    coord::GradSph gradSph;
    coord::HessSph hessSph;
    sphHarmTransformInverseDeriv(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, potential,
        needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    if(needGrad)
        transformDerivsSphToCyl(pos, gradSph, hessSph, grad, hess);
}

double PowerLawMultipole::densityCyl(const coord::PosCyl &pos) const
{
    double rsq   = pow_2(pos.R) + pow_2(pos.z);
    double dlogr = log(rsq / r0sq) * 0.5;
    // simplified treatment in strongly asymptotic regime - retain only l==0 term
    int lmax = (inner && rsq < r0sq*1e-16) || (!inner && rsq > r0sq*1e16) ? 0 : ind.lmax;
    unsigned int size = lmax==0 ? 1 : ind.size();
    double* rho_lm = static_cast<double*>(alloca(size * sizeof(double)));

    // compute all spherical-harmonic coefficients for the density expansion
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            double s=S[c], u=U[c], v = inner ? l : -l-1;
            double ursm2 = s!=2 ? u * exp( dlogr * (s-2) ) : u;  // u * (r/r0)^(s-2)
            if(s!=v)
                rho_lm[c] = ursm2 * (s*(s+1) - l*(l+1));
            else
                rho_lm[c] = ursm2 * (s*(s+1) * dlogr - s*(s-1) + 1);
        }

    if(lmax == 0) {  // fast track - just return the l=0 coef
        return 0.25/M_PI / r0sq * rho_lm[0];
    } else {  // perform inverse spherical-harmonic transform
        double tau = pos.z / (sqrt(pow_2(pos.R) + pow_2(pos.z)) + pos.R);
        return 0.25/M_PI / r0sq * math::sphHarmTransformInverse(ind, rho_lm, tau, pos.phi);
    }
}

// ------- Multipole potential with 1d interpolating splines for each SH harmonic ------- //

MultipoleInterp1d::MultipoleInterp1d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        ind.size() == Phi.size() && ind.size() == dPhi.size() &&
        Phi[0].size() == gridSizeR && ind.lmax >= 0 && ind.mmax <= ind.lmax);

    // whether to perform logarithmic scaling for the amplitude of l=0 term
    logScaling = true;  // will be enabled if all values of Phi_00(r) are negative
    // compute the extrapolation coefficients at small r;
    // if s>0, the potential is finite at r=0 and equal to W
    double s, U, W;
    computeExtrapolationCoefs(Phi[0][0], Phi[0][1], dPhi[0][0], radii[0], radii[1], 0, /*output*/s, U, W);
    invPhi0 = s>0 ? 1./W : 0;

    // set up a logarithmic radial grid
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++) {
        gridR[k] = log(radii[k]);
        // if the potential is everywhere negative, use some form of log-scaling
        logScaling &= Phi[0][k] < 0;
        // if the potential is non-monotonic, don't attempt to accurately follow its power-law
        // asymptotic behaviour at origin (setting 1/Phi(0)=0), but still use log-scaling if possible
        if(Phi[0][k]*invPhi0 >= 1)
            invPhi0 = 0;
    }
    std::vector<double> Phi_lm(gridSizeR), dPhi_lm(gridSizeR);  // temp.arrays

    // set up 1d quintic splines in radius for each non-trivial (l,m) coefficient
    spl.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(unsigned int k=0; k<gridSizeR; k++) {
                if(c==0) {
                    Phi_lm [k] =  logScaling ? log(invPhi0 - 1 / Phi[c][k]) : Phi[c][k];
                    dPhi_lm[k] = (logScaling ? 1 / (Phi[c][k] * (invPhi0 * Phi[c][k] - 1)) : 1) *
                        radii[k] * dPhi[c][k];
                } else if(Phi[0][k] != 0) {
                    Phi_lm [k] = Phi[c][k] / Phi[0][k];
                    dPhi_lm[k] = (dPhi[c][k] - Phi_lm[k] * dPhi[0][k]) * radii[k] / Phi[0][k];
                } else
                    Phi_lm [k] = dPhi_lm[k] = 0;  // don't pretend to be anywhere accurate in this case
            }
            spl[c] = math::QuinticSpline(gridR, Phi_lm, dPhi_lm);
        }
}

void MultipoleInterp1d::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    bool needGrad = grad!=NULL || hess!=NULL;
    bool needHess = hess!=NULL;
    double r = sqrt(pow_2(pos.R) + pow_2(pos.z)), logr = log(r);

    // temporary array created on the stack, without dynamic memory allocation.
    // it will be automatically freed upon return from this routine, just as any local stack variable.
    int ncoefs = pow_2(ind.lmax + 1);
    double*   Phi_lm = static_cast<double*>(alloca(3 * ncoefs * sizeof(double)));
    double*  dPhi_lm = Phi_lm + ncoefs;    // part of the temporary array
    double* d2Phi_lm = Phi_lm + 2*ncoefs;
    coord::GradSph gradSph;
    coord::HessSph hessSph;

    // first compute the l=0 coefficient, possibly log-unscaled
    spl[0].evalDeriv(logr, Phi_lm,
        needGrad ? dPhi_lm  : NULL,
        needHess ? d2Phi_lm : NULL);
    if(logScaling) {
        double expX = exp(Phi_lm[0]), Phi = 1 / (invPhi0 - expX);
        Phi_lm[0] = Phi;
        if(needGrad) {
            double dPhidX = pow_2(Phi) * expX;
            if(needHess)
                d2Phi_lm[0] = dPhidX * (d2Phi_lm[0] + pow_2(dPhi_lm[0]) * Phi * (invPhi0 + expX));
            dPhi_lm[0] *= dPhidX;
        }
    }
    if(ind.lmax == 0) {   // fast track in the spherical case
        if(potential)
            *potential = Phi_lm[0];
        if(needGrad) {
            gradSph.dr = dPhi_lm[0];
            gradSph.dtheta = gradSph.dphi = 0;
        }
        if(needHess) {
            hessSph.dr2 = d2Phi_lm[0];
            hessSph.dtheta2 = hessSph.dphi2 = hessSph.drdtheta = hessSph.drdphi = hessSph.dthetadphi = 0;
        }
    } else {
        // compute spherical-harmonic coefs
        for(int m=ind.mmin(); m<=ind.mmax; m++)
            for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                unsigned int c = ind.index(l, m);
                if(c==0)
                    continue;
                spl[c].evalDeriv(logr, &Phi_lm[c],
                    needGrad?  &dPhi_lm[c] : NULL,
                    needHess? &d2Phi_lm[c] : NULL);
                // scale by the value of l=0 coef
                if(needHess)
                    d2Phi_lm[c] = d2Phi_lm[c] * Phi_lm[0] + 2 * dPhi_lm[c] * dPhi_lm[0] +
                        Phi_lm[c] * d2Phi_lm[0];
                if(needGrad)
                    dPhi_lm[c] = dPhi_lm[c] * Phi_lm[0] + Phi_lm[c] * dPhi_lm[0];
                Phi_lm[c] *= Phi_lm[0];
            }
        if(ind.lmax==2 && ind.mmin()==0 && ind.step==2)   // an optimized special case
            sphHarmTransformInverseDeriv2(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, potential,
                needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
        else
            sphHarmTransformInverseDeriv(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, potential,
                needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    }
    if(needGrad)
        transformDerivsSphToCyl(pos, gradSph, hessSph, grad, hess);
}

// ------- Multipole potential with 2d interpolating splines for each azimuthal harmonic ------- //

/** Set up a grid in tau = cos(theta) / (sin(theta)+1).
    We want (some of) the nodes of the grid to coincide with the nodes of Gauss-Legendre
    quadrature on the interval -1 <= cos(theta) <= 1, which ensures that the values
    of 2d spline at these angles exactly equals the input values, thereby making
    the forward and reverse Legendre transformation invertible to machine precision.
    So we compute these nodes for the given order of sph.-harm. expansion lmax,
    augment them with two endpoints (+-1), and insert additional nodes between the original ones
    to increase the accuracy of approximating the Legendre polynomials by quintic splines.
*/
std::vector<double> createGridInTau(unsigned int lmax)
{
    unsigned int numPointsGL = lmax+1;
    std::vector<double> tau(numPointsGL+2), dummy(numPointsGL);
    math::prepareIntegrationTableGL(-1, 1, numPointsGL, &tau[1], &dummy.front());
    // convert GL nodes (cos theta) to tau = cos(theta)/(sin(theta)+1)
    for(unsigned int iGL=1; iGL<=numPointsGL; iGL++)
        tau[iGL] /= 1 + sqrt(1-pow_2(tau[iGL]));
    // add points at the ends of original interval (GL nodes are all interior)
    tau.back() = 1;
    tau.front()= -1;
    // split each interval between two successive GL nodes (or the ends of original interval)
    // into 3 grid segments (accuracy of Legendre function approximation is better than 1e-6)
    unsigned int oversampleFactor = 3;
    // number of grid points for spline in 0 <= theta) <= pi
    unsigned int gridSizeT = (numPointsGL+1) * oversampleFactor + 1;
    std::vector<double> gridT(gridSizeT);
    for(unsigned int iGL=0; iGL<=numPointsGL; iGL++)
        for(unsigned int iover=0; iover<oversampleFactor; iover++) {
            gridT[iGL * oversampleFactor + iover] =
                (tau[iGL] * (oversampleFactor-iover) + tau[iGL+1] * iover) / oversampleFactor;
        }
    gridT.back() = 1;
    return gridT;
}

MultipoleInterp2d::MultipoleInterp2d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi)),
    logScaling(true)
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        ind.size() == Phi.size() && ind.size() == dPhi.size() &&
        Phi[0].size() == gridSizeR && ind.lmax >= 0 && ind.mmax <= ind.lmax);

    // compute the extrapolation coefficients at small r;
    // if s>0, the potential is finite at r=0 and equal to W
    double s, U, W;
    computeExtrapolationCoefs(Phi[0][0], Phi[0][1], dPhi[0][0], radii[0], radii[1], 0, /*output*/s, U, W);
    invPhi0 = s>0 ? 1./W : 0;

    // set up a 2D grid in ln(r) and tau = cos(theta)/(sin(theta)+1):
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> gridT = createGridInTau(ind.lmax);
    unsigned int gridSizeT = gridT.size();

    // allocate temporary arrays for initialization of 2d splines
    math::Matrix<double> Phi_val (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dR  (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dT  (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dRdT(gridSizeR, gridSizeT);
    // copies of these matrices for the m=0 term
    std::vector<double>  Phi0_val, Phi0_dR, Phi0_dT, Phi0_dRdT;
    // convenience pointers to the raw matrix storage
    double *Phim_val = Phi_val.data(), *Phim_dR = Phi_dR.data(),
        *Phim_dT = Phi_dT.data(), *Phim_dRdT = Phi_dRdT.data();
    // temp.arrays for Legendre polynomials and their theta-derivatives
    std::vector<double>  Plm(ind.lmax+1), dPlm(ind.lmax+1);

    // loop over azimuthal harmonic indices (m)
    spl.resize(2*ind.mmax+1);
    for(int mm=0; mm<=ind.mmax-ind.mmin(); mm++) {
        // this weird order ensures that we first process the m=0 term even if there are m<0 terms
        int m = mm<=ind.mmax ? mm : ind.mmax-mm;
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        int absm = math::abs(m);
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;
        // assign Phi_m, dPhi_m/d(ln r) & dPhi_m/d(tau) at each node of 2d grid (r_k, tau_j)
        for(unsigned int j=0; j<gridSizeT; j++) {
            math::sphHarmArray(ind.lmax, absm, gridT[j], &Plm.front(), &dPlm.front());
            for(unsigned int k=0; k<gridSizeR; k++) {
                double val=0, dR=0, dT=0, dRdT=0;
                for(int l=lmin; l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    val +=  Phi[c][k] *  Plm[l-absm];   // Phi_{l,m}(r)
                    dR  += dPhi[c][k] *  Plm[l-absm];   // d Phi / d r
                    dT  +=  Phi[c][k] * dPlm[l-absm];   // d Phi / d theta
                    dRdT+= dPhi[c][k] * dPlm[l-absm];   // d2Phi / dr dtheta
                }
                if(m==0) {
                    logScaling &= val<0;
                    // a nonzero value of 1/Phi(0) is used to improve the accuracy of interpolation
                    // for potentials with finite central value, but it works only if the potential
                    // is everywhere larger than Phi(0); otherwise we set 1/Phi(0) to zero and
                    // and still use log-scaling (if possible) to improve accuracy at large radii
                    if(invPhi0 <= 1 / (mul * val))
                        invPhi0 = 0;
                }
                Phi_val (k, j) = mul * val;
                // transform d Phi / d r      to  d Phi / d ln(r)
                Phi_dR  (k, j) = mul * dR * radii[k];
                // transform d Phi / d theta  to  d Phi / d tau
                Phi_dT  (k, j) = mul * dT * -2 / (pow_2(gridT[j]) + 1);
                // transform d2Phi / dr dtheta to d2Phi / d ln(r) d tau
                Phi_dRdT(k, j) = mul * dRdT * radii[k] * -2 / (pow_2(gridT[j]) + 1);
            }
        }

        // further transform the amplitude and the derivatives:
        // for the m=0 term we use log-scaling if possible (i.e. if it is everywhere negative),
        // the other terms are normalized to the value of the m=0 term
        if(m==0) {
            // store the un-transformed m=0 term, which is later used to scale the other terms
            Phi0_val .assign(Phim_val , Phim_val + Phi_val. size());
            Phi0_dR  .assign(Phim_dR  , Phim_dR  + Phi_dR.  size());
            Phi0_dT  .assign(Phim_dT  , Phim_dT  + Phi_dT.  size());
            Phi0_dRdT.assign(Phim_dRdT, Phim_dRdT+ Phi_dRdT.size());
            if(logScaling) {
                for(unsigned int i=0; i<gridSizeT*gridSizeR; i++) {
                    double dScaledPhidPhi = 1 / (Phi0_val[i] * (invPhi0 * Phi0_val[i] - 1));
                    Phim_dR  [i] *= dScaledPhidPhi;
                    Phim_dT  [i] *= dScaledPhidPhi;
                    Phim_dRdT[i]  = dScaledPhidPhi * Phim_dRdT[i] +
                        (1 - 2 * invPhi0 * Phi0_val[i]) * Phim_dR[i] * Phim_dT[i];
                    Phim_val [i]  = log(invPhi0 - 1 / Phi0_val[i]);
                }
            }
        } else {
            for(unsigned int i=0; i<gridSizeT*gridSizeR; i++) {
                if(Phi0_val[i] == 0)  // don't attempt to scale by a zero value
                    continue;         // (assume that the m!=0 terms are zero too)
                double Phi_rel = Phim_val[i] / Phi0_val[i];
                Phim_val [i] = Phi_rel;
                Phim_dR  [i] = (Phim_dR[i] - Phi_rel * Phi0_dR[i]) / Phi0_val[i];
                Phim_dT  [i] = (Phim_dT[i] - Phi_rel * Phi0_dT[i]) / Phi0_val[i];
                Phim_dRdT[i] = (Phim_dRdT[i] - Phi_rel * Phi0_dRdT[i] -
                    Phim_dR[i] * Phi0_dT[i] - Phim_dT[i] * Phi0_dR[i]) / Phi0_val[i];
            }
        }

        // establish 2D quintic spline for Phi_m(ln(r), tau)
        spl[m+ind.mmax] = math::QuinticSpline2d(gridR, gridT, Phi_val, Phi_dR, Phi_dT, Phi_dRdT);
    }
}

void MultipoleInterp2d::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    const double
        r         = sqrt(pow_2(pos.R) + pow_2(pos.z)),
        logr      = log(r),
        rplusRinv = 1. / (r + fabs(pos.R)),
        tau       = pos.R==0 ? math::sign(pos.z) : pos.z * rplusRinv;

    // number of azimuthal harmonics to compute
    const int mmin = ind.mmin(), nm = ind.mmax - mmin + 1;

    // only compute those quantities that will be needed in output
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;

    // temporary array for storing coefficients: Phi, two first and three second derivs for each m
    // allocated on the stack and will be automatically freed upon leaving this routine
    double *C_m = static_cast<double*>(alloca(nm * numQuantities * sizeof(double)));
    double *Phi = C_m,    // assign proper names to these arrays
        *dlnr   = C_m+nm,
        *dtau   = C_m+nm*2,
        *dlnr2  = C_m+nm*3,
        *dlnrdtau=C_m+nm*4,
        *dtau2  = C_m+nm*5;

    // value, first and second derivs of scaled potential in scaled coordinates,
    // where 'r' stands for ln(r) and 'theta' - for tau
    double trPot;
    coord::GradSph trGrad;
    coord::HessSph trHess;

    // compute azimuthal harmonics
    for(int mm=0; mm<nm; mm++) {
        int m = mm + mmin;
        if(ind.lmin(m) > ind.lmax)
            continue;
        spl[m+ind.mmax].evalDeriv(logr, tau, &Phi[mm],
            numQuantities>=3 ? &dlnr    [mm] : NULL,
            numQuantities>=3 ? &dtau    [mm] : NULL,
            numQuantities==6 ? &dlnr2   [mm] : NULL,
            numQuantities==6 ? &dlnrdtau[mm] : NULL,
            numQuantities==6 ? &dtau2   [mm] : NULL);
    }

    // transform the amplitude: first perform the inverse log-scaling for the m=0 term,
    // which resides in the array elements with index mm = 0 - mmin
    if(logScaling) {
        double expX = exp(Phi[-mmin]), val = 1 / (invPhi0 - expX);
        Phi[-mmin]  = val;
        if(numQuantities>=3) {
            double dPhidX = pow_2(val) * expX;
            if(numQuantities==6) {
                double d2PhidX2 = dPhidX * val * (invPhi0 + expX);
                dlnr2   [-mmin] = dPhidX * dlnr2   [-mmin] + d2PhidX2 * dlnr[-mmin] * dlnr[-mmin];
                dtau2   [-mmin] = dPhidX * dtau2   [-mmin] + d2PhidX2 * dtau[-mmin] * dtau[-mmin];
                dlnrdtau[-mmin] = dPhidX * dlnrdtau[-mmin] + d2PhidX2 * dlnr[-mmin] * dtau[-mmin];
            }
            dlnr[-mmin] *= dPhidX;
            dtau[-mmin] *= dPhidX;
        }
    }

    // then multiply other terms by the value of the m=0 term, which resides in the [-mmin] element
    for(int mm=0; mm<nm; mm++) {
        int m = mm + mmin;
        if(m==0 || ind.lmin(m) > ind.lmax)
            continue;
        if(numQuantities==6) {
            dlnr2[mm] = dlnr2[mm] * Phi[-mmin] + Phi[mm] * dlnr2[-mmin] + 2 * dlnr[mm] * dlnr[-mmin];
            dtau2[mm] = dtau2[mm] * Phi[-mmin] + Phi[mm] * dtau2[-mmin] + 2 * dtau[mm] * dtau[-mmin];
            dlnrdtau[mm] = dlnrdtau[mm] * Phi[-mmin] + Phi[mm] * dlnrdtau[-mmin] +
                dlnr[mm] * dtau[-mmin] + dtau[mm] * dlnr[-mmin];
        }
        if(numQuantities>=3) {
            dlnr[mm] = dlnr[mm] * Phi[-mmin] + Phi[mm] * dlnr[-mmin];
            dtau[mm] = dtau[mm] * Phi[-mmin] + Phi[mm] * dtau[-mmin];
        }
        Phi[mm] *= Phi[-mmin];
    }

    // Fourier synthesis from azimuthal harmonics to actual quantities, still in scaled coords
    fourierTransformAzimuth(ind, pos.phi, C_m, &trPot,
        numQuantities>=3 ? &trGrad : NULL, numQuantities==6 ? &trHess : NULL);

    if(potential)
        *potential = trPot;
    if(numQuantities==1)
        return;   // nothing else needed

    // abuse the coordinate transformation framework (Sph -> Cyl), where actually
    // our source Sph coords are not (r, theta, phi), but (ln r, tau, phi)
    const double
        rinv  = 1/r,
        r2inv = pow_2(rinv);
    coord::PosDerivT<coord::Cyl, coord::Sph> der;
    der.drdR = pos.R * r2inv;
    der.drdz = pos.z * r2inv;
    der.dthetadR = -tau * rinv;
    der.dthetadz = rinv - rplusRinv;
    if(grad)
        *grad = toGrad(trGrad, der);
    if(hess) {
        coord::PosDeriv2T<coord::Cyl, coord::Sph> der2;
        der2.d2rdR2  = pow_2(der.drdz) - pow_2(der.drdR);
        der2.d2rdRdz = -2 * der.drdR * der.drdz;
        der2.d2rdz2  = -der2.d2rdR2;
        der2.d2thetadR2  = pos.z * r2inv * rinv;
        der2.d2thetadRdz = pow_2(der.dthetadR) - pow_2(der.drdR) * r * rplusRinv;
        der2.d2thetadz2  = -der2.d2thetadR2 - der.dthetadR * rplusRinv;
        *hess = toHess(trGrad, trHess, der, der2);
    }
}

}  // namespace potential
