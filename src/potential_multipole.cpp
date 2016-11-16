#include "potential_multipole.h"
#include "math_core.h"
#include "math_specfunc.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <cmath>
#include <cfloat>
#include <algorithm>

namespace potential {

// internal definitions
namespace{

/// minimum number of terms in sph.-harm. expansion used to compute coefficients
/// of a non-spherical density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)    
const int LMIN_SPHHARM = 16;

/// maximum allowed order of sph.-harm. expansion
const int LMAX_SPHHARM = 64;

/// minimum number of grid nodes
const unsigned int MULTIPOLE_MIN_GRID_SIZE = 2;

/// order of Gauss-Legendre quadrature for computing the radial integrals in Multipole
const unsigned int ORDER_RAD_INT = 15;

/// safety factor to avoid roundoff errors near grid boundaries
const double SAFETY_FACTOR = 100*DBL_EPSILON;

// Helper function to deduce symmetry from the list of non-zero coefficients;
// combine the array of coefficients at different radii into a single array
// and then call the corresponding routine from math::.
// This routine is templated on the number of arrays that it handles:
// each one should have identical number of elements (# of harmonic terms - (lmax+1)^2),
// and each element of each array should have the same dimension (number of radial grid points).
template<int N>
static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> >* C[N])
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

static math::SphHarmIndices getIndicesFromCoefs(const std::vector< std::vector<double> > &C)
{
    const std::vector< std::vector<double> >* A = &C;
    return getIndicesFromCoefs<1>(&A);
}
static math::SphHarmIndices getIndicesFromCoefs(
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
static void restrictSphHarmCoefs(int lmax, int mmax, std::vector<std::vector<double> >& coefs)
{
    coefs.resize(pow_2(lmax+1), std::vector<double>(coefs[0].size(), 0));
    math::SphHarmIndices ind(lmax, mmax, coord::ST_NONE);
    for(unsigned int c=0; c<coefs.size(); c++)
        if(abs(ind.index_m(c))>mmax)
            coefs[c].assign(coefs[c].size(), 0.);
}

// ------- Spherical-harmonic expansion of density or potential ------- //
// The routine `computeSphHarmCoefs` can work with both density and potential classes,
// computes the sph-harm expansion for either density (in the first case),
// or potential and its r-derivative (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine `storeValue`, which stores either one or two
// values for each input point. The `computeSphHarmCoefsSph` routine is templated on both
// the type of input data and the number of quantities stored for each point.

template<class BaseDensityOrPotential>
void storeValue(const BaseDensityOrPotential& src,
    const coord::PosCyl& pos, double values[], int arraySize);

template<>
inline void storeValue(const BaseDensity& src,
    const coord::PosCyl& pos, double values[], int) {
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src,
    const coord::PosCyl& pos, double values[], int arraySize) {
    coord::GradCyl grad;
    src.eval(pos, values, &grad);
    double rinv = 1. / sqrt(pow_2(pos.R) + pow_2(pos.z));
    values[arraySize] = grad.dR * pos.R * rinv + grad.dz * pos.z * rinv;
}

template<class BaseDensityOrPotential, int NQuantities>
static void computeSphHarmCoefs(const BaseDensityOrPotential& src, 
    const math::SphHarmIndices& ind, const std::vector<double>& radii,
    std::vector< std::vector<double> > * coefs[])
{
    unsigned int numPointsRadius = radii.size();
    if(numPointsRadius<1)
        throw std::invalid_argument("computeSphHarmCoefs: radial grid size too small");
    //  initialize sph-harm transform
    math::SphHarmTransformForward trans(ind);

    // 1st step: collect the values of input quantities at a 2d grid in (r,theta);
    // loop over radii and angular directions, using a combined index variable for better load balancing.
    int numSamplesAngles = trans.size();  // size of array of density values at each r
    int numSamplesTotal  = numSamplesAngles * numPointsRadius;
    std::vector<double> values(numSamplesTotal * NQuantities);
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numSamplesTotal; n++) {
        try{
            int indR    = n / numSamplesAngles;  // index in radial grid
            int indA    = n % numSamplesAngles;  // combined index in angular direction (theta,phi)
            double rad  = radii[indR];
            double z    = rad * trans.costheta(indA);
            double R    = sqrt(rad*rad - z*z);
            double phi  = trans.phi(indA);
            storeValue(src, coord::PosCyl(R, z, phi),
                &values[indR * numSamplesAngles + indA], numSamplesTotal);
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computeSphHarmCoefs: "+errorMsg);
    
    // 2nd step: transform these values to spherical-harmonic expansion coefficients at each radius
    std::vector<double> shcoefs(ind.size());
    for(int q=0; q<NQuantities; q++) {
        coefs[q]->assign(ind.size(), std::vector<double>(numPointsRadius));
        for(unsigned int indR=0; indR<numPointsRadius; indR++) {
            trans.transform(&values[indR * numSamplesAngles + q * numSamplesTotal], &shcoefs.front());
            math::eliminateNearZeros(shcoefs);
            for(unsigned int c=0; c<ind.size(); c++)
                coefs[q]->at(c)[indR] = shcoefs[c];
        }
    }
}

// transform an N-body snapshot to an array of spherical-harmonic coefficients:
// input particles are sorted in radius, and for each k-th particle the array of
// sph.-harm. functions Y_lm(theta_k, phi_k) times the particle mass is computed
// and stored in the output array with the following indexing scheme:
// C_lm(particle_k) = coefs[SphHarmIndices::index(l,m)][k].
// This saves memory, since only the arrays for harmonic coefficients allowed
// by the indexing scheme are allocated and returned.
static void computeSphericalHarmonicsFromParticles(
    const particles::ParticleArray<coord::PosCyl> &particles,
    const math::SphHarmIndices &ind,
    std::vector<double> &particleRadii,
    std::vector< std::vector<double> > &coefs)
{
    if((int)ind.size() > pow_2(1+LMAX_SPHHARM))
        throw std::invalid_argument(
            "computeSphericalHarmonicsFromParticles: order of expansion is too large");
    // allocate space
    int nbody = particles.size();
    particleRadii.resize(nbody);
    coefs.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
            coefs[ind.index(l, m)].resize(nbody);
    bool needSine = ind.mmin()<0;
    std::string errorMsg;

    // compute Y_lm for each particle
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0; i<nbody; i++) {
        try{
            // temporary arrays for Legendre and trigonometric functions, separate for each thread
            double leg[LMAX_SPHHARM+1], trig[2*LMAX_SPHHARM];
            const coord::PosCyl& pos = particles.point(i);
            double r   = sqrt(pow_2(pos.R) + pow_2(pos.z));
            double tau = pos.z / (r + pos.R);
            const double mass = particles.mass(i);
            if(r==0 && mass!=0)
                errorMsg = "no massive particles at r=0 allowed";
            particleRadii[i] = r;
            math::trigMultiAngle(pos.phi, ind.mmax, needSine, trig);
            for(int m=0; m<=ind.mmax; m++) {
                math::sphHarmArray(ind.lmax, m, tau, leg);
                for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step)
                    coefs[ind.index(l, m)][i] = mass * leg[l-m] * 2*M_SQRTPI *
                        (m==0 ? 1 : M_SQRT2 * trig[m-1]);
                if(needSine && m>0)
                    for(int l=ind.lmin(-m); l<=ind.lmax; l+=ind.step)
                        coefs[ind.index(l, -m)][i] = mass * leg[l-m] * 2*M_SQRTPI *
                            M_SQRT2 * trig[ind.mmax+m-1];
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("computeSphericalHarmonicsFromParticles: " + errorMsg);
}


/// auto-assign min/max radii of the grid if they were not provided, for a smooth density model
static void chooseGridRadii(const BaseDensity& src, const unsigned int gridSizeR,
    double& rmin, double& rmax)
{
    if(rmax!=0 && rmin!=0)
        return;
    double rhalf = getRadiusByMass(src, 0.5 * src.totalMass());
    if(!isFinite(rhalf))
        throw std::invalid_argument("Multipole: failed to automatically determine grid extent");
    double spacing = 1 + sqrt(20./gridSizeR);  // ratio between consecutive grid nodes
    if(rmax==0)
        rmax = rhalf * pow(spacing,  0.5*gridSizeR);
    if(rmin==0)
        rmin = rhalf * pow(spacing, -0.5*gridSizeR);
    utils::msg(utils::VL_DEBUG, "Multipole",
        "Grid in r=["+utils::toString(rmin)+":"+utils::toString(rmax)+"]");
}

/// auto-assign min/max radii of the grid if they were not provided, for a discrete N-body model
static void chooseGridRadii(const particles::ParticleArray<coord::PosCyl>& particles,
    unsigned int gridSizeR, double &rmin, double &rmax) 
{
    if(rmin!=0 && rmax!=0)
        return;
    unsigned int Npoints = particles.size();
    std::vector<double> radii(Npoints);
    double prmin=INFINITY, prmax=0;
    for(unsigned int i=0; i<Npoints; i++) {
        radii[i] = sqrt(pow_2(particles.point(i).R) + pow_2(particles.point(i).z));
        prmin = fmin(prmin, radii[i]);
        prmax = fmax(prmax, radii[i]);
    }
    std::nth_element(radii.begin(), radii.begin() + Npoints/2, radii.end());
    double rhalf = radii[Npoints/2];   // half-mass radius (if all particles have equal mass)
    double spacing = 1 + sqrt(20./gridSizeR);  // ratio between two adjacent grid nodes
    // # of points inside the first or outside the last grid node
    int Nmin = static_cast<int>(log(Npoints+1)/log(2));
    if(rmin==0) {
        std::nth_element(radii.begin(), radii.begin() + Nmin, radii.end());
        rmin = std::max(radii[Nmin], rhalf * pow(spacing, -0.5*gridSizeR));
    }
    if(rmax==0) {
        std::nth_element(radii.begin(), radii.end() - Nmin, radii.end());
        rmax = std::min(radii[Npoints-Nmin], rhalf * pow(spacing, 0.5*gridSizeR));
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
static void computeExtrapolationCoefs(double Phi1, double Phi2, double dPhi1,
    double r1, double r2, int v, double& s, double& U, double& W)
{
    double lnr = log(r2/r1);
    double num1 = r1*dPhi1, num2 = v*Phi1, den1 = Phi1, den2 = Phi2 * exp(-v*lnr);
    double A = lnr * (num1 - num2) / (den1 - den2);
    bool roundoff =   // check if the value of A is dominated by roundoff errors
        fabs(num1-num2) < fmax(fabs(num1), fabs(num2)) * SAFETY_FACTOR ||
        fabs(den1-den2) < fmax(fabs(den1), fabs(den2)) * SAFETY_FACTOR;
    if(!isFinite(A) || A >= 0 || roundoff)
    {   // no solution - output only the main multipole component (with zero Laplacian)
        U = 0;
        s = 0;
        W = Phi1;
        return;
    }
    // find x(A) such that  x = A * (1 - exp(x)),  where  x = (s-v) * ln(r2/r1)
    s = A==-1 ? v : v + (A - math::lambertW(A * exp(A), /*choice of branch*/ A>-1)) / lnr;
    // safeguard against weird slope determination
    if(v>=0 && (!isFinite(s) || s<=-1))
        s = 2;  // results in a constant-density core for the inward extrapolation
    if(v<0  && (!isFinite(s) || s>=0))
        s = -2; // results in a r^-4 falloff for the outward extrapolation
    if(s != v) {
        U = (r1*dPhi1 - v*Phi1) / (s-v);
        W = (r1*dPhi1 - s*Phi1) / (v-s);
    } else {
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
static PtrPotential initAsympt(const std::vector<double>& radii,
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
            utils::msg(utils::VL_VERBOSE, "Multipole",
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
static inline void fourierTransformAzimuth(const math::SphHarmIndices& ind, const double phi,
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
    double trig_m[2*LMAX_SPHHARM];
    const bool useSine = mmin<0 || numQuantities>1;
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
static inline void sphHarmTransformInverseDeriv(
    const math::SphHarmIndices& ind, const coord::PosCyl& pos,
    const double C_lm[], const double dC_lm[], const double d2C_lm[],
    double *val, coord::GradSph *grad, coord::HessSph *hess)
{
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;  // number of quantities in C_m
    const int nm = ind.mmax - ind.mmin() + 1;  // number of azimuthal harmonics in C_m array
    const double tau = pos.z / (sqrt(pow_2(pos.R) + pow_2(pos.z)) + pos.R);
    // temporary storage for coefficients
    double    C_m[(LMAX_SPHHARM*2+1) * 6];
    double    P_lm[LMAX_SPHHARM+1];
    double   dP_lm_arr[LMAX_SPHHARM+1];
    double  d2P_lm_arr[LMAX_SPHHARM+1];
    double*  dP_lm = numQuantities>=3 ?  dP_lm_arr : NULL;
    double* d2P_lm = numQuantities==6 ? d2P_lm_arr : NULL;
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

// transform potential derivatives from {ln(r), theta} to {R, z}
static inline void transformDerivsSphToCyl(const coord::PosCyl& pos,
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

// perform scaling transformation for the amplitude of potential and its derivatives:
// on entry, pot, grad and hess contain the value, gradient and hessian of an auxiliary quantity
// G[ln(r),...] = Phi[ln(r),...] * sqrt(r^2 + R0^2);
// on output, they are replaced with the value, gradient and hessian of Phi w.r.t. [ln(r),...];
// grad or hess may be NULL, if they are ultimately not needed.
// \tparam nonrad tells whether we need to handle non-radial components, or they are zero.
template<bool nonrad>
static inline void transformAmplitude(double r, double Rscale,
    double& pot, coord::GradSph *grad, coord::HessSph *hess)
{
    // additional scaling factor for the amplitude: 1 / sqrt(r^2 + R0^2)
    double amp = 1. / sqrt(pow_2(r) + pow_2(Rscale));
    pot *= amp;
    if(!grad)
        return;
    // unscale the amplitude of derivatives, i.e. transform from
    // d [scaledPhi(scaledCoords) * amp] / d[scaledCoords]  to  d[scaledPhi] / d[scaledCoords]
    double damp = -r*r*amp;  // d amp[ln(r)] / d[ln(r)] / amp^2
    grad->dr = (grad->dr + pot * damp) * amp;
    if(nonrad) {
        grad->dtheta *= amp;
        grad->dphi   *= amp;
    }
    if(hess) {
        hess->dr2 = (hess->dr2 + (2 * grad->dr + (2 - pow_2(r * amp)) * pot ) * damp) * amp;
        if(nonrad) {
            hess->drdtheta = (hess->drdtheta + grad->dtheta * damp) * amp;
            hess->drdphi   = (hess->drdphi   + grad->dphi   * damp) * amp;
            hess->dtheta2 *= amp;
            hess->dphi2   *= amp;
            hess->dthetadphi *= amp;
        }
    }
}

// find a suitable scaling radius R0 for transformation of potential amplitudes
// (TODO: not very robust, incorrect results for steep cusps?)
static double assignRscale(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi)
{
    // determine the characteristic radius from the condition that Phi(0) = -Mtotal/rscale
    double Rscale = radii.back() * Phi[0].back() / Phi[0].front();
    if(!(Rscale>0))   // something weird happened, set to a reasonable default value
        Rscale = 1.;
    // if the l=0 harmonic component of the potential is well-behaved, that is everywhere negative
    // and monotonically increasing with radius, then additionally ensure that the scaled potential
    // does the same (to preserve monotonicity), at least near the center; otherwise don't care
    for(unsigned int i=0; i<radii.size(); i++) {
        double ratio = -Phi[0][i] / dPhi[0][i] * radii[i];
        if(ratio >= 0 && Phi[0][i] / Phi[0][0] >= 0.5 && pow_2(Rscale) + pow_2(radii[i]) < ratio)
            Rscale = sqrt(ratio - pow_2(radii[i]));
    }
    utils::msg(utils::VL_VERBOSE, "Multipole", "Rscale="+utils::toString(Rscale));
    return Rscale;
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
    computeSphHarmCoefs<BaseDensity, 1>(src, ind, gridRadii, &coefs);
}

// density coefs from N-body snapshot
void computeDensityCoefsSph(
    const particles::ParticleArray<coord::PosCyl> &particles,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &coefs,
    double &innerSlope, double &outerSlope,
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
    std::transform(gridRadii.begin(), gridRadii.end(), gridLogRadii.begin(), log);
    coefs.assign(ind.size(), std::vector<double>(gridSizeR, 0.));

    // compute the sph-harm coefs at each particle's radius
    std::vector<std::vector<double> > harmonics(ind.size());
    std::vector<double> particleRadii;
    computeSphericalHarmonicsFromParticles(particles, ind, particleRadii, harmonics);

    // normalize all l>0 harmonics by the value of l=0 term
    for(unsigned int i=0; i<particleRadii.size(); i++) {
        particleRadii[i] = log(particleRadii[i]);
        for(unsigned int c=1; c<ind.size(); c++)
            if(!harmonics[c].empty())
                harmonics[c][i] /= harmonics[0][i];
    }

    // construct the l=0 harmonic using a penalized log-density estimate
    math::CubicSpline spl0(gridLogRadii, math::splineLogDensity<3>(
        gridLogRadii, particleRadii, harmonics[0],
        math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT | math::FO_PENALTY_3RD_DERIV)));
    spl0.evalDeriv(gridLogRadii.front(), NULL, &innerSlope);
    spl0.evalDeriv(gridLogRadii.back(),  NULL, &outerSlope);
    innerSlope -= 3;  // we have computed the log of density of particles in log(r),
    outerSlope -= 3;  // which is equal to the true density multiplied by 4 pi r^3
    for(unsigned int k=0; k<gridSizeR; k++)
        coefs[0][k] = exp(spl0(gridLogRadii[k])) / (4*M_PI*pow_3(gridRadii[k]));
    utils::msg(utils::VL_DEBUG, "Multipole",
        "Power-law index of density profile: inner="+utils::toString(innerSlope)+
        ", outer="+utils::toString(outerSlope));
    if(ind.size()==1)
        return;

    // construct the l>0 terms by fitting a penalized smoothing spline
    math::SplineApprox fitter(gridLogRadii, particleRadii);
    double edf = 2 + gridSizeR / (smoothing+1);
    for(unsigned int c=1; c<ind.size(); c++) {
        if(!harmonics[c].empty()) {
            math::CubicSpline splc(gridLogRadii, fitter.fit(harmonics[c], edf));
            for(unsigned int k=0; k<gridSizeR; k++)
                coefs[c][k] = splc(gridLogRadii[k]) * coefs[0][k];
        }
    }
}

// potential coefs from potential
void computePotentialCoefsSph(const BasePotential &src,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi)
{
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential, 2>(src, ind, gridRadii, coefs);
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

    // prepare tables for (non-adaptive) Gauss-Legendre integration over radius
    double glnodes[ORDER_RAD_INT], glweights[ORDER_RAD_INT];
    math::prepareIntegrationTableGL(0, 1, ORDER_RAD_INT, glnodes, glweights);

    // prepare SH transformation
    math::SphHarmTransformForward trans(ind);

    // Loop over radial grid segments and compute integrals of rho_lm(r) times powers of radius,
    // for each interval of radii in the input grid (0 <= k < Nr):
    //   Qint[l,m][k] = \int_{r_{k-1}}^{r_k} \rho_{l,m}(r) (r/r_k)^{l+2} dr,  with r_{-1} = 0;
    //   Qext[l,m][k] = \int_{r_k}^{r_{k+1}} \rho_{l,m}(r) (r/r_k)^{1-l} dr,  with r_{Nr} = \infty.
    // Here \rho_{l,m}(r) are the sph.-harm. coefs for density at each radius.
    std::string errorMsg;

    // in principle this may be OpenMP-parallelized,
    // but if the density computation is expensive, then one should construct an intermediate
    // DensitySphericalHarmonic interpolator and pass it to this routine.
    // Otherwise it's quite cheap, as it only does 1d integrals in radius.
//#ifdef _OPENMP
//#pragma omp parallel for schedule(dynamic)
//#endif
    for(int k=0; k<=gridSizeR; k++) {
        try{
            // local per-thread temporary arrays
            std::vector<double> densValues(trans.size());
            std::vector<double> tmpCoefs(ind.size());
            double rkminus1 = (k>0 ? gridRadii[k-1] : 0);
            double deltaGridR = k<gridSizeR ?
                gridRadii[k] - rkminus1 :  // length of k-th radial segment
                gridRadii.back();          // last grid segment extends to infinity

            // loop over ORDER_RAD_INT nodes of GL quadrature for each radial grid segment
            for(unsigned int s=0; s<ORDER_RAD_INT; s++) {
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
                                math::powInt(r / gridRadii[k], l+2);
                        if(k>0)
                            // accumulate Qext for all segments except the innermost one
                            // (which starts from zero), with a special treatment for last segment
                            // that extends to infinity and has a different integration variable
                            Qext[c][k-1] += tmpCoefs[c] * glweights[s] * deltaGridR *
                                (k==gridSizeR ? 1 / pow_2(glnodes[s]) : 1) * // jacobian of 1/r transform
                                math::powInt(r / gridRadii[k-1], 1-l);
                    }
                }
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsSph: "+errorMsg);

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
                    val *= math::powInt(gridRadii[k-1] / gridRadii[k], l+1);
                val += gridRadii[k] * Qint[c][k];
                Pint[c][k] = val;
            }

            // Compute Pext by summing from outside in, using the recurrent relation
            // Pext(r_k) r_k^{-l} = Pext(r_{k+1}) r_{k+1}^{-l} + Qext[k] * r_k^{1-l}
            val = 0;
            for(int k=gridSizeR-1; k>=0; k--) {
                if(k<gridSizeR-1)
                    val *= math::powInt(gridRadii[k] / gridRadii[k+1], l);
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
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<=0 || rmax<=rmin)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of min/max grid radii");
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(dens) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(dens) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
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
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");
    chooseGridRadii(particles, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    if(isSpherical(sym))
        lmax = 0;
    if(isZRotSymmetric(sym))
        mmax = 0;
    std::vector<std::vector<double> > coefs;
    double innerSlope, outerSlope;
    computeDensityCoefsSph(particles,
        math::SphHarmIndices(lmax, mmax, sym),
        gridRadii, coefs, innerSlope, outerSlope, smoothing);
    return PtrDensity(new DensitySphericalHarmonic(gridRadii, coefs, innerSlope, outerSlope));
}

// the actual constructor
DensitySphericalHarmonic::DensitySphericalHarmonic(const std::vector<double> &gridRadii,
    const std::vector< std::vector<double> > &coefs, double _innerSlope, double _outerSlope) :
    BaseDensity(), ind(getIndicesFromCoefs(coefs)),
    innerSlope(_innerSlope), outerSlope(_outerSlope), logScaling(true)
{
    unsigned int gridSizeR = gridRadii.size();
    if(coefs.size() != ind.size() || gridSizeR < MULTIPOLE_MIN_GRID_SIZE)
        throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");
    for(unsigned int n=0; n<coefs.size(); n++)
        if(coefs[n].size() != gridSizeR)
            throw std::invalid_argument("DensitySphericalHarmonic: incorrect size of coefficients array");
    if(ind.lmax>LMAX_SPHHARM)
        throw std::invalid_argument("DensitySphericalHarmonic: invalid choice of expansion order");

    spl.resize(ind.size());
    std::vector<double> gridLogR(gridSizeR), tmparr(gridSizeR);
    std::transform(gridRadii.begin(), gridRadii.end(), gridLogR.begin(), log);

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
            // when using log-scaling, the endpoint derivatives of spline are simply
            // d[log(rho(log(r)))] / d[log(r)] = power-law indices of the inner/outer slope;
            // without log-scaling, the endpoint derivatives of spline are
            // d[rho(log(r))] / d[log(r)] = power-law indices multiplied by the values of rho.
            spl[0] = math::CubicSpline(gridLogR, tmparr,
                innerSlope * (logScaling ? 1 : coefs[0].front()),
                outerSlope * (logScaling ? 1 : coefs[0].back()));
            spl[0].evalDeriv(gridLogR.front(), NULL, &innerSlope);
            spl[0].evalDeriv(gridLogR.back(),  NULL, &outerSlope);
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
            innerSlope = fmax(innerSlope, -2.8);
            outerSlope = fmin(outerSlope, -2.2);
        } else {
            // values of l!=0 components are normalized to the value of l=0 component at each radius
            // and are extrapolated as constants beyond the extent of the grid
            // (with zero endpoint derivatives)
            for(unsigned int k=0; k<gridSizeR; k++)
                tmparr[k] = coefs[0][k]!=0 ? coefs[c][k] / coefs[0][k] : 0;
            spl[c] = math::CubicSpline(gridLogR, tmparr, 0, 0);
        }
    }
}

void DensitySphericalHarmonic::getCoefs(
    std::vector<double> &radii, std::vector< std::vector<double> > &coefs,
    double &slopeInner, double &slopeOuter) const
{
    radii.resize(spl[0].xvalues().size());
    for(unsigned int k=0; k<radii.size(); k++)
        radii[k] = exp(spl[0].xvalues()[k]);
    computeDensityCoefsSph(*this, ind, radii, coefs);
    slopeInner = innerSlope;
    slopeOuter = outerSlope;
}

double DensitySphericalHarmonic::densityCyl(const coord::PosCyl &pos) const
{
    double coefs[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double logr = log( pow_2(pos.R) + pow_2(pos.z) ) * 0.5;
    double logrmin = spl[0].xmin(), logrmax = spl[0].xmax();
    double logrspl = fmax(logrmin, fmin(logrmax, logr));   // the argument of spline functions
    // first compute the l=0 coefficient, possibly log-unscaled
    coefs[0] = spl[0](logrspl);
    if(logScaling)
        coefs[0] = exp(coefs[0]);
    // extrapolate if necessary
    if(logr < logrmin)
        coefs[0] *= exp( (logr-logrmin) * innerSlope);
    if(logr > logrmax)
        coefs[0] *= exp( (logr-logrmax) * outerSlope);
    // then compute other coefs, which are scaled by the value of l=0 coef
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            if(c!=0)
                coefs[c] = spl[c](logrspl) * coefs[0];
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
    /// characteristic radius for amplitude scaling transformation
    double Rscale;

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
    /// characteristic radius for amplitude scaling transformation
    double Rscale;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};

template<class BaseDensityOrPotential>
static PtrPotential createMultipole(
    const BaseDensityOrPotential& src,
    int lmax, int mmax,
    unsigned int gridSizeR, double rmin, double rmax)
{
    if(gridSizeR < MULTIPOLE_MIN_GRID_SIZE || rmin<0 || (rmax!=0 && rmax<=rmin))
        throw std::invalid_argument("Multipole: invalid grid parameters");
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");
    chooseGridRadii(src, gridSizeR, rmin, rmax);
    std::vector<double> gridRadii = math::createExpGrid(gridSizeR, rmin, rmax);
    // to improve accuracy of SH coefficient computation, we may increase the order of expansion
    // that determines the number of integration points in angles
    int lmax_tmp =     isSpherical(src) ? 0 : std::max<int>(lmax, LMIN_SPHHARM);
    int mmax_tmp = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, LMIN_SPHHARM);
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
    if(lmax<0 || lmax>LMAX_SPHHARM || mmax<0 || mmax>lmax)
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
    double innerSlope, outerSlope;
    computeDensityCoefsSph(particles, ind, gridRadii, coefDens, innerSlope, outerSlope, smoothing);
    DensitySphericalHarmonic dens(gridRadii, coefDens, innerSlope, outerSlope);
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
    if(ind.lmax>LMAX_SPHHARM)
        throw std::invalid_argument("Multipole: invalid choice of expansion order");

    // construct the interpolating splines
    impl = ind.lmax<=2 ?   // choose between 1d or 2d splines, depending on the expected efficiency
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
    // take the values and derivatives of potential at grid nodes and apply forward transform
    // to obtain the coefficients.
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

double Multipole::enclosedMass(double radius) const
{
    if(radius==0)
        return 0;  // TODO: this may not be correct for a potential of a point mass!
    // use the l=0 harmonic term of dPhi/dr to estimate the spherically-averaged enclosed mass
    const BasePotential& pot =
        radius <= gridRadii.front()* (1+SAFETY_FACTOR) ? *asymptInner :
        radius >= gridRadii.back() * (1-SAFETY_FACTOR) ? *asymptOuter : *impl;
    std::vector< std::vector<double> > Phi, dPhi;
    std::vector< std::vector<double> > *coefs[2] = {&Phi, &dPhi};
    computeSphHarmCoefs<BasePotential, 2>(pot, ind, std::vector<double>(1, radius), coefs);
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
    double   Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double  dPhi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
    double d2Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
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
        double rsqinv = 1/rsq, Rr2 = pos.R * rsqinv, zr2 = pos.z * rsqinv;
        if(grad) {
            grad->dR = rsq>0 ? dPhi_lm[0] * Rr2 : S[0]>1 ? 0 : INFINITY;
            grad->dz = rsq>0 ? dPhi_lm[0] * zr2 : S[0]>1 ? 0 : INFINITY; 
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
    Rscale = assignRscale(radii, Phi, dPhi);

    // set up a logarithmic radial grid
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> Phi_lm(gridSizeR), dPhi_lm(gridSizeR);  // temp.arrays

    // set up 1d quintic splines in radius for each non-trivial (l,m) coefficient
    spl.resize(ind.size());
    for(int m=ind.mmin(); m<=ind.mmax; m++)
        for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
            unsigned int c = ind.index(l, m);
            for(unsigned int k=0; k<gridSizeR; k++) {
                double amp = sqrt(pow_2(Rscale) + pow_2(radii[k]));   // additional scaling multiplier
                Phi_lm[k]  =  amp *  Phi[c][k];
                // transform d Phi / d r  to  d (Phi * amp(r)) / d ln(r)
                dPhi_lm[k] = (amp * dPhi[c][k] + Phi[c][k] * radii[k] / amp) * radii[k];
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
    double pot;
    coord::GradSph gradSph;
    coord::HessSph hessSph;
    if(ind.lmax == 0) {   // fast track in the spherical case
        spl[0].evalDeriv(logr, &pot,
            needGrad ? &gradSph.dr  : NULL,
            needHess ? &hessSph.dr2 : NULL);
        if(needGrad)
            gradSph.dtheta = gradSph.dphi = 0;
        if(needHess)
            hessSph.dtheta2 = hessSph.dphi2 = hessSph.drdtheta = hessSph.drdphi = hessSph.dthetadphi = 0;
        transformAmplitude<false>(r, Rscale, pot,
            needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);

    } else {
        double   Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
        double  dPhi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
        double d2Phi_lm[(LMAX_SPHHARM+1)*(LMAX_SPHHARM+1)];
        // compute spherical-harmonic coefs
        for(int m=ind.mmin(); m<=ind.mmax; m++)
            for(int l=ind.lmin(m); l<=ind.lmax; l+=ind.step) {
                unsigned int c = ind.index(l, m);
                spl[c].evalDeriv(logr, &Phi_lm[c],
                    needGrad?  &dPhi_lm[c] : NULL,
                    needHess? &d2Phi_lm[c] : NULL);
            }
        sphHarmTransformInverseDeriv(ind, pos, Phi_lm, dPhi_lm, d2Phi_lm, &pot,
            needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
        transformAmplitude<true>(r, Rscale, pot,
            needGrad ? &gradSph : NULL, needHess ? &hessSph : NULL);
    }
    if(potential)
        *potential = pot;
    if(needGrad)
        transformDerivsSphToCyl(pos, gradSph, hessSph, grad, hess);
}

// ------- Multipole potential with 2d interpolating splines for each azimuthal harmonic ------- //

/** Set up non-uniform grid in cos(theta), with denser spacing close to z-axis.
    We want (some of) the nodes of the grid to coincide with the nodes of Gauss-Legendre
    quadrature on the interval -1 <= cos(theta) <= 1, which ensures that the values
    of 2d spline at these angles exactly equals the input values, thereby making
    the forward and reverse Legendre transformation invertible to machine precision.
    So we first compute these nodes for the given order of sph.-harm. expansion lmax,
    and then take only the non-negative half of them for the spline in cos(theta),
    plus one at theta=0.
    To achieve better accuracy in approximating the Legendre polynomials by quintic
    splines, we insert additional nodes in between the original ones.
*/
static std::vector<double> createGridInTheta(unsigned int lmax)
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
    // into this many grid points (accuracy of Legendre function approximation is better than 1e-6)
    unsigned int oversampleFactor = 3;
    // number of grid points for spline in 0 <= theta) <= pi
    unsigned int gridSizeT = (numPointsGL+1) * oversampleFactor + 1;
    std::vector<double> gridT(gridSizeT);
    for(unsigned int iGL=0; iGL<=numPointsGL; iGL++)
        for(unsigned int iover=0; iover<oversampleFactor; iover++) {
            gridT[/*gridT.size() - 1 -*/ (iGL * oversampleFactor + iover)] =
                (tau[iGL] * (oversampleFactor-iover) + tau[iGL+1] * iover) / oversampleFactor;
        }
    gridT.back() = 1;
    return gridT;
}

MultipoleInterp2d::MultipoleInterp2d(
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &Phi,
    const std::vector< std::vector<double> > &dPhi) :
    ind(getIndicesFromCoefs(Phi, dPhi))
{
    unsigned int gridSizeR = radii.size();
    assert(gridSizeR >= MULTIPOLE_MIN_GRID_SIZE &&
        ind.size() == Phi.size() && ind.size() == dPhi.size() &&
        Phi[0].size() == gridSizeR && ind.lmax >= 0 && ind.mmax <= ind.lmax);
    Rscale = assignRscale(radii, Phi, dPhi);

    // set up a 2D grid in ln(r) and tau = cos(theta)/(sin(theta)+1):
    std::vector<double> gridR(gridSizeR);
    for(unsigned int k=0; k<gridSizeR; k++)
        gridR[k] = log(radii[k]);
    std::vector<double> gridT = createGridInTheta(ind.lmax);
    unsigned int gridSizeT = gridT.size();

    // allocate temporary arrays for initialization of 2d splines
    math::Matrix<double> Phi_val(gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dR (gridSizeR, gridSizeT);
    math::Matrix<double> Phi_dT (gridSizeR, gridSizeT);
    std::vector<double>  Plm(ind.lmax+1), dPlm(ind.lmax+1);

    // loop over azimuthal harmonic indices (m)
    spl.resize(2*ind.mmax+1);
    for(int m=ind.mmin(); m<=ind.mmax; m++) {
        int lmin = ind.lmin(m);
        if(lmin > ind.lmax)
            continue;
        int absm = math::abs(m);
        double mul = m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2;
        // assign Phi_m, dPhi_m/d(ln r) & dPhi_m/d(tau) at each node of 2d grid (r_k, tau_j)
        for(unsigned int j=0; j<gridSizeT; j++) {
            math::sphHarmArray(ind.lmax, absm, gridT[j], &Plm.front(), &dPlm.front());
            for(unsigned int k=0; k<gridSizeR; k++) {
                double val=0, dR=0, dT=0;
                for(int l=lmin; l<=ind.lmax; l+=ind.step) {
                    unsigned int c = ind.index(l, m);
                    val +=  Phi[c][k] *  Plm[l-absm];   // Phi_{l,m}(r)
                    dR  += dPhi[c][k] *  Plm[l-absm];   // d Phi / d r
                    dT  +=  Phi[c][k] * dPlm[l-absm];   // d Phi / d theta
                }
                double amp = sqrt(pow_2(Rscale) + pow_2(radii[k]));   // additional scaling multiplier
                Phi_val(k, j) = mul *  amp * val;
                // transform d Phi / d r      to  d (Phi * amp(r)) / d ln(r)
                Phi_dR (k, j) = mul * (amp * dR + val * radii[k] / amp) * radii[k];
                // transform d Phi / d theta  to  d (Phi * amp) / d tau
                Phi_dT (k, j) = mul *  amp * dT * -2 / (pow_2(gridT[j]) + 1);
            }
        }
        // establish 2D quintic spline for Phi_m(ln(r), tau)
        spl[m+ind.mmax] = math::QuinticSpline2d(gridR, gridT, Phi_val, Phi_dR, Phi_dT);
    }
}

void MultipoleInterp2d::evalCyl(const coord::PosCyl &pos,
    double* potential, coord::GradCyl* grad, coord::HessCyl* hess) const
{
    const double
        r         = sqrt(pow_2(pos.R) + pow_2(pos.z)),
        logr      = log(r),
        rplusRinv = 1. / (r + pos.R),
        tau       = pos.z * rplusRinv;

    // number of azimuthal harmonics to compute
    const int nm = ind.mmax - ind.mmin() + 1;

    // temporary array for storing coefficients: Phi, two first and three second derivs for each m
    double C_m[(2*LMAX_SPHHARM+1) * 6];

    // value, first and second derivs of scaled potential in scaled coordinates,
    // where 'r' stands for ln(r) and 'theta' - for tau
    double trPot;
    coord::GradSph trGrad;
    coord::HessSph trHess;

    // only compute those quantities that will be needed in output
    const int numQuantities = hess!=NULL ? 6 : grad!=NULL ? 3 : 1;

    // compute azimuthal harmonics
    for(int mm=0; mm<nm; mm++) {
        int m = mm + ind.mmin();
        if(ind.lmin(m) > ind.lmax)
            continue;
        spl[m+ind.mmax].evalDeriv(logr, tau, &C_m[mm],
            numQuantities>=3 ? &C_m[mm+nm  ] : NULL,
            numQuantities>=3 ? &C_m[mm+nm*2] : NULL,
            numQuantities==6 ? &C_m[mm+nm*3] : NULL,
            numQuantities==6 ? &C_m[mm+nm*4] : NULL,
            numQuantities==6 ? &C_m[mm+nm*5] : NULL);
        if(pos.R==0) {  // zero out the derivatives w.r.t. tau that are pure roundoff errors
            C_m[mm+nm*2] = 0;  //  d / dtau
            C_m[mm+nm*4] = 0;  // d2 / dr dtau
        }
    }

    // Fourier synthesis from azimuthal harmonics to actual quantities, still in scaled coords
    fourierTransformAzimuth(ind, pos.phi, C_m, &trPot,
        numQuantities>=3 ? &trGrad : NULL, numQuantities==6 ? &trHess : NULL);

    // scaling transformation for the amplitude of interpolated potential
    transformAmplitude<true>(r, Rscale, trPot,
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

}; // namespace
