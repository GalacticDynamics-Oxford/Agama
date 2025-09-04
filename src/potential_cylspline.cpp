#include "potential_cylspline.h"
#include "potential_multipole.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_specfunc.h"
#include "math_sphharm.h"
#include "math_spline.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace potential {

// internal definitions
namespace{

/// minimum number of grid nodes
static const unsigned int CYLSPLINE_MIN_GRID_SIZE = 2;

/// minimum number of terms in Fourier expansion used to compute coefficients
/// of a non-axisymmetric density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)
static const unsigned int MMIN_FOURIER = 12;

/// the number of additional Fourier terms used to compute coefficients on top of
/// the requested number of output terms (again to improve the accuracy of integration)
static const unsigned int MADD_FOURIER = 6;

/// order of multipole extrapolation outside the grid
static const int LMAX_EXTRAPOLATION = 8;

/// max number of function evaluations in multidimensional integration
static const unsigned int MAX_NUM_EVAL = 10000;

/// relative accuracy of potential computation (integration tolerance parameter)
static const double EPSREL_POTENTIAL_INT = 1e-6;

/// eliminate Fourier terms whose relative amplitude is less than this number
static const double EPS_COEF = 1e-10;

// resize the array(s) of Fourier coefficients to the requested order and eliminate all-zero terms.
// \tparam  K  is the number of arrays (one for density, three for potential and its derivatives);
// \param[in]  mmax  is the maximum order of expansion in phi;
// \param[in]  sym   is the required symmetry of the expansion (may be more symmetric than input coefs);
// \param[in,out]  coefs  are K arrays of coefficients for each harmonic term.
// \param[in,out]  gridz  is the grid in vertical dimension:
// if the required symmetry includes Z-reflection, but gridz covers both positive and negative z,
// then replace it with only half of the grid and average the computed coefficients in +-z.
template<int K>
void restrictFourierCoefs(int mmax, coord::SymmetryType sym,
    std::vector< math::Matrix<double> > coefs[K], std::vector<double>& gridz)
{
    int mmaxSrc = coefs[0].size() / 2;  // input array(s) contain terms from -mmaxSrc to mmaxSrc
    int mmaxExp = 0;  // largest |m| that is present in the input and allowed by symmetry
    std::vector<int> indices = math::getIndicesAzimuthal(mmaxSrc, sym);
    for(int m=-mmaxSrc; m<=mmaxSrc; m++) {
        if(m==0)  // always present
            continue;
        bool allowed = false;
        for(unsigned int i=0; i<indices.size(); i++)
            allowed |= indices[i] == m;
        allowed &= abs(m) <= mmax;
        if(!allowed) {  // this m-harmonic should be zeroed out
            for(int k=0; k<K; k++)
                coefs[k][mmaxSrc+m] = math::Matrix<double>();
        } else {  // check if this m-harmonic is identically zero in the input arrays
            allowed = false;
            for(int k=0; k<K; k++)
                allowed |= !allZeros(coefs[k][mmaxSrc+m]);
        }
        if(allowed)  // keep track of the largest |m| that is allowed and non-empty
            mmaxExp = std::max(abs(m), mmaxExp);
    }

    // mmaxExp is the smallest of mmaxSrc and mmax(required), but can be even smaller
    // if the input arrays were identically zero for all |m|>mmaxExp
    if(mmaxSrc > mmaxExp) {
        // remove extra coefs: (mmaxSrc-mmaxExp) from both heads and tails of arrays
        for(int k=0; k<K; k++) {
            coefs[k].erase(coefs[k].begin() + mmaxSrc+mmaxExp+1, coefs[k].end());
            coefs[k].erase(coefs[k].begin(), coefs[k].begin() + mmaxSrc-mmaxExp);
        }
    }

    // if needed, enforce z-reflection symmetry
    if(gridz[0] != 0 && isZReflSymmetric(sym)) {
        size_t sizeR = coefs[0][mmaxExp].rows(), sizez2 = gridz.size() / 2;
        bool gridzsym = gridz.size() == sizez2*2+1 && gridz[sizez2] == 0;
        for(size_t i=0; i<sizez2; i++)
            gridzsym &= gridz[i] == -gridz[2*sizez2-i];
        if(!gridzsym)
            throw std::runtime_error("restrictFourierCoefs: non-symmetric gridz is not allowed");
        gridz.erase(gridz.begin(), gridz.begin()+sizez2);
        for(int k=0; k<K; k++) {
            double sign = k<2 ? 1 : -1;  // k==2 is dPhi/dz, need to anti-symmetrize it
            for(int m=0; m<=2*mmaxExp; m++) {
                if(coefs[k][m].size() == 0)
                    continue;
                math::Matrix<double> mat(sizeR, sizez2+1);
                for(size_t i=0; i<sizeR; i++)
                    for(size_t j=0; j<=sizez2; j++)
                        mat(i, j) = 0.5 * (coefs[k][m](i, sizez2+j) + sign * coefs[k][m](i, sizez2-j));
                coefs[k][m] = mat;
            }
        }
    }
}

// ------- Fourier expansion of density or potential ------- //
// The routine 'computeFourierCoefs' can work with both density and potential classes,
// computes the azimuthal Fourier expansion for either density (in the first case),
// or potential and its R- and z-derivatives (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine 'collectValues', which stores either one or three
// values for each input point, depending on the source function. The routine 'computeFourierCoefs'
// is templated on the type of source function (BaseDensity or BasePotential).

// number of quantities computed at each point
template<class BaseDensityOrPotential> int numQuantitiesAtPoint(const BaseDensityOrPotential& src);
template<> int numQuantitiesAtPoint(const BaseDensity&)   { return 1; }
template<> int numQuantitiesAtPoint(const BasePotential&) { return 3; }

template<class BaseDensityOrPotential>
void collectValues(const BaseDensityOrPotential& src, const std::vector<coord::PosCyl>& points,
    /*output*/ double values[]);

template<>
inline void collectValues(const BaseDensity& src, const std::vector<coord::PosCyl>& points,
    /*output array of length points.size()*/ double values[])
{
    src.evalmanyDensityCyl(points.size(), &points[0], values);   // vectorized evaluation at many points
}

template<>
inline void collectValues(const BasePotential& src, const std::vector<coord::PosCyl>& points,
    /*output array of length 3*points.size()*/ double values[])
{
    coord::GradCyl grad;
    for(size_t i=0, count=points.size(); i<count; i++) {
        src.eval(points[i], &values[i*3], &grad);
        values[i*3+1] = grad.dR;
        values[i*3+2] = grad.dz;
    }
}

// compute the coefficients of Fourier expansion of the source function (density or potential)
// at the 2d grid of points
template<class BaseDensityOrPotential>
void computeFourierCoefs(const BaseDensityOrPotential &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > coefs[])
{
    size_t sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE)
        throw std::invalid_argument("computeFourierCoefs: incorrect grid size");
    if(!isZReflSymmetric(src) && gridz[0]==0)
        throw std::invalid_argument("computeFourierCoefs: input density is not symmetric "
            "under z-reflection, the grid in z must cover both positive and negative z");

    // 0th step: set up the Fourier transform
    int mmin = isYReflSymmetric(src) ? 0 : -static_cast<int>(mmax);
    bool useSine = mmin<0;
    math::FourierTransformForward trans(mmax, useSine);
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, src.symmetry());
    size_t numHarmonicsComputed = indices.size(), sizephi = trans.size();
    int numPoints = sizeR * sizez * sizephi;
    int numQuantities = numQuantitiesAtPoint(src);  // 1 for density, 3 for potential
    for(int q=0; q<numQuantities; q++) {
        coefs[q].resize(mmax*2+1);
        for(size_t i=0; i<numHarmonicsComputed; i++)
            coefs[q].at(indices[i]+mmax) = math::Matrix<double>(sizeR, sizez);
    }

    // 1st step: prepare the 3d grid of points in (R,z,phi) where the input quantities are needed
    std::vector<coord::PosCyl> points(numPoints);
    for(size_t iR=0; iR<sizeR; iR++)
        for(size_t iz=0; iz<sizez; iz++)
            for(size_t iphi=0; iphi<sizephi; iphi++)
                points[(iR * sizez + iz) * sizephi + iphi] =
                    coord::PosCyl(gridR[iR], gridz[iz], trans.phi(iphi));

    // 2nd step: collect the values of input quantities at this 3d grid (specific to each src type)
    std::vector<double> values(numPoints * numQuantities);
    collectValues(src, points, &values[0]);

    // 3rd step: transform these values to Fourier expansion coefficients at each (R,z)
    std::vector<double> coefs_m(2*mmax+1);
    for(size_t iR=0; iR<sizeR; iR++)
        for(size_t iz=0; iz<sizez; iz++)
            for(int q=0; q<numQuantities; q++) {
                trans.transform(&values[((iR * sizez + iz) * sizephi) * numQuantities + q],
                    &coefs_m[0], /*stride*/ numQuantities);
                // eliminate Fourier terms smaller than EPS_COEF * sum(all m terms)
                double norm = 0;
                for(unsigned int i=0; i<numHarmonicsComputed; i++)
                    norm += fabs(coefs_m[indices[i] + (useSine ? mmax : 0)]);
                for(unsigned int i=0; i<numHarmonicsComputed; i++) {
                    int m = indices[i];
                    double val = coefs_m[useSine ? m+mmax : m];
                    coefs[q].at(m+mmax)(iR, iz) =
                        // at R=0, all non-axisymmetric harmonics must vanish
                        (iR==0 && m!=0) || (fabs(val) < norm * EPS_COEF) ? 0 :
                        val / (m==0 ? 2*M_PI : M_PI);
                }
            }
}

// ------- Computation of potential from density ------- //
// The routines below solve the Poisson equation by computing the Fourier harmonics
// of potential via direct 2d integration over (R,z) plane.
// If the input density is axisymmetric, then the values of density at phi=0 is taken,
// otherwise the density must first be Fourier-transformed itself and represented
// as an instance of DensityAzimuthalHarmonic class, which provides the member function
// returning the value of m-th harmonic at the given point (R,z).

inline void density_rho_m(const BaseDensity& dens, int m, size_t npoints, const coord::PosCyl pos[],
    /*output array of length npoints*/ double rho[])
{
    const DensityAzimuthalHarmonic* densazi = dynamic_cast<const DensityAzimuthalHarmonic*>(&dens);
    if(densazi) {  // dynamic_cast successful
        for(size_t p=0; p<npoints; p++)
            rho[p] = densazi->rho_m(m, pos[p].R, pos[p].z);
    } else {  // dynamic_cast failed - assume the input density is already axisymmetric
        if(m==0)
            dens.evalmanyDensityCyl(npoints, pos, rho);
        else
            std::fill(rho, rho+npoints, 0);  // m!=0 harmonics are zero in the axisymmetric case
    }
}

// Routine that computes the contribution to the m-th harmonic of potential at location (R0,z0)
// from the point at (R,z) with given 'mass' (or, rather, mass times trig(m phi)
// in the discrete case, or density times jacobian times trig(m phi) in the continuous case).
// This routine is used both in AzimuthalHarmonicIntegrand to compute the potential from
// a continuous density distribution, and in ComputePotentialCoefsFromPoints to obtain
// the potential from a discrete point mass collection.
void computePotentialHarmonicAtPoint(int m, double R, double z, double R0, double z0,
    double mass, bool useDerivs, /*output array - add to it*/double values[])
{
    // the contribution to the potential is given by
    // rho * \int_0^\infty dk J_m(k R) J_m(k R0) exp(-k|z-z0|)
    double t = R*R + R0*R0 + pow_2(z0-z);
    if(R > 0 && R0 > 0) {  // normal case
        double sq = 1 / (M_PI * sqrt(R*R0));
        double u  = t / (2*R*R0);   // u >= 1
        double dQ = 0, Q = math::legendreQ(math::abs(m)-0.5, u, useDerivs ? &dQ : NULL);
        if(!isFinite(Q+dQ)) return;
        values[0] += -sq * mass * Q;
        if(useDerivs) {
            values[1] += -sq * mass * (dQ/R - (Q/2 + u*dQ)/R0);
            values[2] += -sq * mass * dQ * (z0-z) / (R*R0);
        }
    } else if(m==0) {  // degenerate case (R or R0 are zero) - here only m=0 harmonic survives
        double s = 1 / sqrt(t);
        values[0] += -mass * s;
        if(useDerivs)
            values[2] += mass * s * (z0-z) / t;
    }
}

// multiply two numbers while avoiding a NAN indeterminacy in case of (weak INFINITY) * (strong 0)
inline double prod(double a, double b) { return b==0 ? 0 : a*b; }

// The N-dimensional integrand for computing the potential harmonics from density,
// which is either an instance of DensitySphericalHarmonic or an arbitrary axisymmetric density model.
// The integral over (R,z) is performed in scaled coordinates (xi,eta) in unit box,
// separately for each node (R0,z0) of the meridional grid, and separately for each harmonic m.
// To improve the accuracy of integrating the singular integrands for the potential derivatives,
// which diverge as 1/r near the grid nodes and change the sign (thus the divergences cancel out),
// we subtract the dominant singular terms and add back their analytic integrals.
// The integrand for the potential itself also has an integrable singularity ln(r),
// which does not present much difficulties (tried subtracting it, but this didn't have much effect).
class AzimuthalHarmonicIntegrand: public math::IFunctionNdim {
public:
    AzimuthalHarmonicIntegrand(const BaseDensity& _dens, int _m,
        double _R0, double _z0, double _Rmin, double _zmin, double _Rmax, double _zmax, bool _useDerivs) :
        dens(_dens), m(_m),
        R0(_R0), z0(_z0), Rmin(_Rmin), zmin(_zmin), Rmax(_Rmax), zmax(_zmax), useDerivs(_useDerivs),
        jac0(2*M_PI * log(1 + Rmax/Rmin) * Rmin * log(1 + zmax/zmin) * zmin)
    {
        assert(isZRotSymmetric(dens) || (bool)dynamic_cast<const DensityAzimuthalHarmonic*>(&dens));
        if(!useDerivs)
            return;
        // scaled coordinates xi,eta corresponding to R0,z0 (the node of the meridional grid)
        xi0    = log(1 + R0/Rmin) / log(1 + Rmax/Rmin);
        eta0   = log(1 + fabs(z0)/zmin) / log(1 + zmax/zmin);
        // derivatives of the real coordinates R,z w.r.t. the scaled ones xi,eta
        dRdxi  = log(1 + Rmax/Rmin) * (Rmin + R0);
        dzdeta = log(1 + zmax/zmin) * (zmin + fabs(z0));
        coord::PosCyl p0(R0, z0, 0);
        density_rho_m(dens, m, 1, &p0, &mul);
        mul *= -2 * dRdxi * dzdeta;
        // if the density at R0,z0 is infinite, the subtraction would not work and should be disabled
        if(!isFinite(mul))
            mul = 0;
        // analytic integrals of the subtracted singular terms over the entire box 0<xi<1, 0<eta<1
        dPhidR_add = mul * (
            0.5 / dRdxi * (
                + prod(log(pow_2(dRdxi * (0-xi0)) + pow_2(dzdeta * (0-eta0))), 0-eta0)
                - prod(log(pow_2(dRdxi * (1-xi0)) + pow_2(dzdeta * (0-eta0))), 0-eta0)
                - prod(log(pow_2(dRdxi * (0-xi0)) + pow_2(dzdeta * (1-eta0))), 1-eta0)
                + prod(log(pow_2(dRdxi * (1-xi0)) + pow_2(dzdeta * (1-eta0))), 1-eta0) ) +
            1.0 / dzdeta * (
                + prod(math::atan(dzdeta * (0-eta0) / dRdxi / (0-xi0)), 0-xi0)
                - prod(math::atan(dzdeta * (1-eta0) / dRdxi / (0-xi0)), 0-xi0)
                - prod(math::atan(dzdeta * (0-eta0) / dRdxi / (1-xi0)), 1-xi0)
                + prod(math::atan(dzdeta * (1-eta0) / dRdxi / (1-xi0)), 1-xi0) ) );
        dPhidz_add = mul * (
            0.5 / dzdeta * (
                + prod(log(pow_2(dRdxi * (0-xi0)) + pow_2(dzdeta * (0-eta0))), 0-xi0)
                - prod(log(pow_2(dRdxi * (0-xi0)) + pow_2(dzdeta * (1-eta0))), 0-xi0)
                - prod(log(pow_2(dRdxi * (1-xi0)) + pow_2(dzdeta * (0-eta0))), 1-xi0)
                + prod(log(pow_2(dRdxi * (1-xi0)) + pow_2(dzdeta * (1-eta0))), 1-xi0) ) +
            1.0 / dRdxi * (
                + prod(math::atan(dRdxi * (0-xi0) / dzdeta / (0-eta0)), 0-eta0)
                - prod(math::atan(dRdxi * (1-xi0) / dzdeta / (0-eta0)), 0-eta0)
                - prod(math::atan(dRdxi * (0-xi0) / dzdeta / (1-eta0)), 1-eta0)
                + prod(math::atan(dRdxi * (1-xi0) / dzdeta / (1-eta0)), 1-eta0) ) );
        if(R0==0) {  // the integral of the z-derivative has a different functional form on the z axis
            dPhidR_add = 0;  // and the R-derivative term is zero
            dPhidz_add = M_PI * mul / (dRdxi * dzdeta) * (
                + sqrt(pow_2(dRdxi) + pow_2(dzdeta * (0-eta0)))
                - sqrt(pow_2(dRdxi) + pow_2(dzdeta * (1-eta0)))
                + (1 - 2 * eta0) * dzdeta
            );
        }
    }

    // evaluate the integrand at a single input point
    virtual void eval(const double vars[], double values[]) const {
        evalmany(1, vars, values);
    }

    // vectorized evaluation at several input points (scaled R,z)
    virtual void evalmany(const size_t npoints, const double vars[], double values[]) const
    {
        // 1st step: unscale input coordinates
        coord::PosCyl* pos = static_cast<coord::PosCyl*>(alloca(npoints * sizeof(coord::PosCyl)));
        double* jac = static_cast<double*>(alloca(npoints * sizeof(double)));
        for(size_t p=0; p<npoints; p++) {
            const double xi  = vars[p*2  ], pR = pow(1 + Rmax/Rmin, xi ), R = Rmin * (pR - 1);
            const double eta = vars[p*2+1], pz = pow(1 + zmax/zmin, eta), z = zmin * (pz - 1);
            jac[p] = jac0 * R * pR * pz;
            pos[p] = coord::PosCyl(R, z, /*phi*/0);
        }

        // 2nd step: collect the values of m-th harmonic of the density at these points
        double* rhoplus = static_cast<double*>(alloca(npoints * sizeof(double)));
        density_rho_m(dens, m, npoints, pos, /*output*/ rhoplus);
        for(size_t p=0; p<npoints; p++)
            pos[p].z = -pos[p].z;
        // in the typical case of z-reflection symmetry we save effort by reusing
        // the same values of density at (R,z) and (R,-z)
        double* rhominus = rhoplus;
        if(!isZReflSymmetric(dens)) {  // otherwise also need to collect the density values at (R,-z)
            rhominus = static_cast<double*>(alloca(npoints * sizeof(double)));
            density_rho_m(dens, m, npoints, pos, /*output*/ rhominus);
        }

        // 3rd step: compute the potential and its derivs at these points
        size_t nvalues = numValues();
        std::fill(values, values + npoints * nvalues, 0);
        for(size_t p=0; p<npoints; p++) {
            computePotentialHarmonicAtPoint(m, pos[p].R, /*z was reflected*/ -pos[p].z, R0, z0,
                rhoplus [p] * jac[p], useDerivs, /*output*/values + p * nvalues);
            computePotentialHarmonicAtPoint(m, pos[p].R, /*z was reflected*/ +pos[p].z, R0, z0,
                rhominus[p] * jac[p], useDerivs, /*output*/values + p * nvalues);
            if(useDerivs) {
                // offsets from the grid node in scaled coordinates (xi,eta)
                double dxi = vars[p*2+0] - xi0, deta = vars[p*2+1] - eta0;
                // u2 is an approximation for the singular term in the denominator, (R-R0)^2 + (z-z0)^2
                double u2 = pow_2(dRdxi * dxi) + pow_2(dzdeta * deta);
                // the derivatives have the following asymptotic form as u2 -> 0:
                // dPhi_m/dR ~ (R-R0) / ((R-R0)^2 + (z-z0)^2),
                // dPhi_m/dz ~ (z-z0) / ((R-R0)^2 + (z-z0)^2),
                // and we approximate them by linearizing the mapping R <-> xi, z <-> eta around R0,z0
                double dPhidR_sub = prod(mul * dRdxi  / u2, dxi);
                double dPhidz_sub = prod(mul * dzdeta / u2, deta);
                // double Phi_sub = mul * -0.5 * log(u2);   // unused - no improvement in accuracy
                if(R0==0) {  // the subtracted singular term is different for nodes on the z axis
                    dPhidR_sub = 0;
                    dPhidz_sub *= M_PI * dRdxi * dxi / sqrt(u2);
                }
                // subtract the coordinate-dependent singular terms (dPhi_sub), and add back their
                // analytic integrals (dPhi_add) of these terms over the entire domain [xi,eta].
                // since the integration is performed over a unit box, adding these constant
                // contributions to the integrand at each point is equivalent to adding them only
                // once to the resulting integral after it is computed (simplifies bookkeeping).
                // One also needs to take into account the sign of z0 for dPhi/dz, 
                // and double the contribution to dPhi/dR in the equatorial plane (z0=0).
                values[p * nvalues + 1] += (dPhidR_add - dPhidR_sub) * (z0==0 ? 2 : 1);
                values[p * nvalues + 2] += (dPhidz_add - dPhidz_sub) * (z0>0 ? 1 : z0<0 ? -1 : 0);
            }
        }
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return useDerivs ? 3 : 1; }
private:
    const BaseDensity& dens;  ///< the density profile in the Poisson eqn
    const int m;              ///< azimuthal harmonic number
    const double R0, z0;      ///< the node in the meridional grid at which the integral is computed
    const double Rmin, zmin;  ///< smallest grid segment
    const double Rmax, zmax;  ///< extent of the integration domain
    const bool useDerivs;     ///< whether to compute only the potential or also its partial derivs by R,z
    const double jac0;        ///< constant prefactor for the jacobian
    // coefs for subtracting the singular part of the integrand and adding back its analytic integral
    double mul, xi0, eta0, dRdxi, dzdeta, dPhidR_add, dPhidz_add;
};

/** Compute the coefficients of azimuthal Fourier expansion of potential and optionally
    its derivatives from the given density profile, used for creating a CylSpline object.
    This function solves the Poisson equation in cylindrical coordinates,
    by first creating a Fourier expansion of density in azimuthal angle (phi), if necessary,
    and then using 2d numerical integration to compute the values and derivatives
    of each Fourier component of potential at the nodes of 2d grid in R,z plane.
    This is a rather costly calculation.
    The output is either one (if useDerivs==false) or three vectors of matrices,
    will be resized as needed.
    \note OpenMP-parallelized loop over the 2d grid in R,z.
*/
void computePotentialCoefsFromDensity(const BaseDensity &dens,
    int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> > output[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, dens.symmetry());
    unsigned int numQuantitiesOutput = useDerivs ? 3 : 1;  // Phi only, or Phi plus two derivs
    // the number of output coefficients is always a full set even if some of them are empty
    for(unsigned int q=0; q<numQuantitiesOutput; q++) {
        output[q].resize(2*mmax+1);
        for(unsigned int i=0; i<indices.size(); i++) {  // only allocate those coefs that will be used
            output[q].at(indices[i]+mmax)=math::Matrix<double>(sizeR, sizez, 0);
        }
    }

    double zmin = gridz[0]==0 ? gridz[1] : gridz[sizez/2]==0 ? gridz[sizez/2+1] : gridR[1];

    int numPoints = sizeR * sizez;
    std::string errorMsg;
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
    bool stop = false;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {  // combined index variable
        if(stop) continue;
        if(cbrk.triggered()) stop = true;
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        try{
            double boxmin[2]={0, 0}, boxmax[2]={1, 1};  // integration box in scaled coords
            double result[3], error[3];
            int numEval;
            for(unsigned int i=0; i<indices.size(); i++) {
                int m = indices[i];
                AzimuthalHarmonicIntegrand fnc(dens, m,
                    gridR[iR], gridz[iz], gridR[1], zmin, gridR.back(), gridz.back(), useDerivs);
                math::integrateNdim(fnc, boxmin, boxmax,
                    EPSREL_POTENTIAL_INT, MAX_NUM_EVAL,
                    result, error, &numEval);
                for(unsigned int q=0; q<numQuantitiesOutput; q++)
                    output[q].at(m+mmax)(iR,iz) += result[q];
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            stop = true;
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error(cbrk.message());
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsCyl: "+errorMsg);
}

// transform an N-body snapshot to an array of Fourier harmonic coefficients
void computeAzimuthalHarmonicsFromParticles(
    const particles::ParticleArray<coord::PosCyl>& particles,
    const std::vector<int>& indices,
    std::vector<std::vector<double> >& harmonics,
    std::vector<std::pair<double, double> > &Rz)
{
    assert(harmonics.size()>0 && indices.size()>0);
    size_t nbody = particles.size();
    unsigned int nind = indices.size();
    int mmax = (harmonics.size()-1)/2;
    bool needSine = false;
    for(unsigned int i=0; i<nind; i++) {
        needSine |= indices[i]<0;
        harmonics[indices[i]+mmax].resize(nbody);
    }
    Rz.resize(nbody);
    double* trig = static_cast<double*>(alloca(mmax*(1+needSine) * sizeof(double)));
    for(size_t b=0; b<nbody; b++) {
        const coord::PosCyl& pc = particles.point(b);
        Rz[b].first = pc.R;
        Rz[b].second= pc.z;
        if(pc.R == 0 && pc.z == 0 && particles.mass(b) != 0)
            // this check is only relevant for CylSpline, not for DensityAzimuthalHarmonic,
            // but currently there is no method for constructing the latter directly from a snapshot
            throw std::runtime_error("CylSpline: no massive particles at r=0 allowed");
        math::trigMultiAngle(pc.phi, mmax, needSine, trig);
        for(unsigned int i=0; i<nind; i++) {
            int m = indices[i];
            harmonics[m+mmax][b] = particles.mass(b) *
                (m==0 ? 1 : m>0 ? 2*trig[m-1] : 2*trig[mmax-m-1]);
        }
    }
}

/** Compute the coefficients of azimuthal Fourier expansion of potential and
    its derivatives from an N-body snapshot.
    \param[in] particles  is the array of particles.
    \param[in] sym  is the assumed symmetry of the input snapshot,
    which defines the list of angular harmonics to compute and to ignore
    (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd m terms are zeros).
    \param[in] mmax  is the order of angular expansion (if the symmetry includes
    coord::ST_ZROTATION flag, it doesn't make sense to have mmax>0).
    \param[in] gridR is the grid in cylindrical radius, which must start at 0.
    \param[in] gridz is the grid in vertical dimension - under coord::ST_ZREFLECTION
    symmetry it must start at 0, otherwise must cover both positive and negative z.
    \param[out] output is either one (if useDerivs==false) or three vectors of matrices
    (Phi, dPhidR, dPhidz), which will contain the arrays of computed coefficients
    for potential and its derivatives at the nodes of 2d grid for each angular harmonic,
    with the same convention as used in the constructor of `CylSpline`;
    will be resized as needed.
*/
void computePotentialCoefsFromParticles(
    const std::vector<int>& indices,
    const std::vector<std::vector<double> > &harmonics,
    const std::vector<std::pair<double, double> > &Rz,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> >* output[])
{
    assert(harmonics.size()>0 && indices.size()>0);
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    int mmax = (harmonics.size()-1)/2;
    bool zsym = gridz[0]==0;  // whether we assume z-reflection symmetry, deduced from the grid
    unsigned int numQuantitiesOutput = useDerivs ? 3 : 1;  // Phi only, or Phi plus two derivs
    for(unsigned int q=0; q<numQuantitiesOutput; q++) {
        output[q]->resize(2*mmax+1);
        for(unsigned int i=0; i<indices.size(); i++) {
            output[q]->at(indices[i]+mmax)=math::Matrix<double>(sizeR, sizez, 0);
        }
    }
    ptrdiff_t nbody = Rz.size();
    int numPoints = sizeR * sizez;
    std::string errorMsg;
    utils::CtrlBreakHandler cbrk;  // catch Ctrl-Break keypress
    bool stop = false;
    // parallelize the loop over the nodes of 2d grid, not the inner loop over particles
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        // thread-local temp storage for all harmonic terms at a single point in the (R,z) plane
        std::vector<double> temp(indices.size() * numQuantitiesOutput);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
        for(int ind=0; ind<numPoints; ind++) {
            if(stop) continue;
            if(cbrk.triggered()) stop = true;
            unsigned int iR = ind % sizeR;
            unsigned int iz = ind / sizeR;
            temp.assign(indices.size() * numQuantitiesOutput, 0);
            try{
                for(ptrdiff_t b=0; b<nbody; b++) {
                    if(Rz[b].first > gridR.back() || fabs(Rz[b].second) > gridz.back())
                        continue;   // skip particles that are outside the grid
                    for(unsigned int i=0; i<indices.size(); i++) {
                        int m = indices[i];
                        double values[3] = {0,0,0};
                        computePotentialHarmonicAtPoint(m, Rz[b].first, Rz[b].second,
                            gridR[iR], gridz[iz], harmonics[m+mmax][b], useDerivs, values);
                        if(zsym) {  // add symmetric contribution from -z
                            computePotentialHarmonicAtPoint(m, Rz[b].first, -Rz[b].second,
                                gridR[iR], gridz[iz], harmonics[m+mmax][b], useDerivs, values);
                            for(unsigned int q=0; q<numQuantitiesOutput; q++)
                                values[q] *= 0.5;  // average with the one from +z
                        }
                        // accumulate the computed quantities in a thread-local array
                        for(unsigned int q=0; q<numQuantitiesOutput; q++)
                            temp[i*numQuantitiesOutput+q] += values[q];
                    }
                }
            }
            catch(std::exception& e) {
                errorMsg = e.what();
                stop = true;
            }
            // copy the thread-local array into the global array of coefficients
            for(unsigned int i=0; i<indices.size(); i++)
                for(unsigned int q=0; q<numQuantitiesOutput; q++)
                    output[q]->at(indices[i]+mmax)(iR,iz) = temp[i*numQuantitiesOutput+q];
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error(cbrk.message());
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsFromParticles: "+errorMsg);
}

// This routine constructs a spherical-harmonic expansion describing
// asymptotic behaviour of the potential beyond the grid definition region.
// It takes the values of potential at the outer edge of the grid in (R,z) plane,
// and finds a combination of SH coefficients that approximate the theta-dependence
// of each m-th azimuthal harmonic term, in the least-square sense.
// In doing so, we must assume that the coefficients behave like C_{lm} ~ r^{-1-l},
// which is valid for empty space, but is not able to describe the residual density;
// thus this asymptotic form describes the potential and forces rather well,
// but returns zero density.
PtrPotential determineAsympt(
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    const std::vector< math::Matrix<double> > &Phi)
{
    bool zsym = gridz[0]==0;
    unsigned int sizeR = gridR.size();
    unsigned int sizez = gridz.size();
    std::vector<coord::PosCyl> points;     // coordinates of boundary points
    std::vector<unsigned int> indR, indz;  // indices of these points in the Phi array

    // assemble the boundary points and their indices
    for(unsigned int iR=0; iR<sizeR-1; iR++) {
        // first run along R at the max-z and min-z edges
        unsigned int iz=sizez-1;
        points.push_back(coord::PosCyl(gridR[iR], gridz[iz], 0));
        indR.push_back(iR);
        indz.push_back(iz);
        if(zsym) {  // min-z edge is the negative of max-z edge
            points.push_back(coord::PosCyl(gridR[iR], -gridz[iz], 0));
            indR.push_back(iR);
            indz.push_back(iz);
        } else {  // min-z edge must be at the beginning of the array
            iz = 0;
            points.push_back(coord::PosCyl(gridR[iR], gridz[iz], 0));
            indR.push_back(iR);
            indz.push_back(iz);
        }
    }
    for(unsigned int iz=0; iz<sizez; iz++) {
        // next run along z at max-R edge
        unsigned int iR=sizeR-1;
        points.push_back(coord::PosCyl(gridR[iR], gridz[iz], 0));
        indR.push_back(iR);
        indz.push_back(iz);
        if(zsym && iz>0) {
            points.push_back(coord::PosCyl(gridR[iR], -gridz[iz], 0));
            indR.push_back(iR);
            indz.push_back(iz);
        }
    }
    unsigned int npoints = points.size();
    int mmax = (Phi.size()-1)/2;        // # of angular(phi) harmonics in the original potential
    int lmax_fit = LMAX_EXTRAPOLATION;  // # of meridional harmonics to fit - don't set too large
    int mmax_fit = std::min<int>(lmax_fit, mmax);
    unsigned int ncoefs = pow_2(lmax_fit+1);
    std::vector<double> Plm(lmax_fit+1);     // temp.storage for sph-harm functions
    std::vector<double> W(ncoefs), zeros(ncoefs);
    double r0 = fmin(gridR.back(), gridz.back());

    // find values of spherical harmonic coefficients
    // that best match the potential at the array of boundary points
    for(int m=-mmax_fit; m<=mmax_fit; m++)
        if(Phi[m+mmax].cols()*Phi[m+mmax].rows()>0) {
            // for m-th harmonic, we may have lmax-m+1 different l-terms
            int absm = math::abs(m);
            math::Matrix<double> matr(npoints, lmax_fit-absm+1);
            std::vector<double>  rhs(npoints);
            std::vector<double>  sol;
            // The linear system to solve in the least-square sense is M_{p,l} * S_l = R_p,
            // where R_p = Phi at p-th boundary point (0<=p<npoints),
            // M_{l,p}   = value of l-th harmonic coefficient at p-th boundary point,
            // S_l       = the amplitude of l-th coefficient to be determined.
            for(unsigned int p=0; p<npoints; p++) {
                rhs[p] = Phi[m+mmax](indR[p], indz[p]);
                double r = sqrt(pow_2(points[p].R) + pow_2(points[p].z));
                double tau = points[p].z / (points[p].R + r);
                math::sphHarmArray(lmax_fit, absm, tau, &Plm.front());
                for(int l=absm; l<=lmax_fit; l++)
                    matr(p, l-absm) =
                        Plm[l-absm] * math::pow(r/r0, -l-1) *
                        (m==0 ? 2*M_SQRTPI : 2*M_SQRTPI*M_SQRT2);
            }
            math::linearMultiFit(matr, rhs, NULL, sol);
            for(int l=absm; l<=lmax_fit; l++) {
                unsigned int c = math::SphHarmIndices::index(l, m);
                W[c] = sol[l-absm];
            }
        }
    // safeguarding against possible problems
    if(!isFinite(W[0])) {
        // something went wrong - at least return a correct value for the l=0 term
        math::Averager avg;
        for(unsigned int p=0; p<npoints; p++)
            avg.add(Phi[mmax](indR[p], indz[p]) * sqrt(pow_2(points[p].R) + pow_2(points[p].z)) / r0);
        W.assign(ncoefs, 0);
        W[0] = avg.mean();
        FILTERMSG(utils::VL_WARNING, "CylSpline",
            "Failed to determine extrapolation coefficients, set W0=" + utils::toString(W[0]));
    }
    math::eliminateNearZeros(W);
    return PtrPotential(new PowerLawMultipole(r0, false /*not inner*/, zeros, zeros, W));
}

// Automatically choose reasonable grid extent if it was not provided
void chooseGridRadii(const BaseDensity& src,
    unsigned int gridSizeR, double &Rmin, double &Rmax,
    unsigned int gridSizez, double &zmin, double &zmax)
{
    // if the input is an instance of CylSpline or DensityAzimuthalHarmonic,
    // adopt its values for rmin/max unless given explicitly
    const DensityAzimuthalHarmonic* dah = dynamic_cast<const DensityAzimuthalHarmonic*>(&src);
    const CylSpline* cyl = dynamic_cast<const CylSpline*>(&src);
    double Rmin1=Rmin, Rmax1=Rmax, zmin1=zmin, zmax1=zmax;
    if(dah)
        dah->getGridExtent(Rmin1, Rmax1, zmin1, zmax1);
    else if(cyl)
        cyl->getGridExtent(Rmin1, Rmax1, zmin1, zmax1);
    if(Rmin==0)
        Rmin = Rmin1;
    if(Rmax==0)
        Rmax = Rmax1;
    if(zmin==0)
        zmin = zmin1;
    if(zmax==0)
        zmax = zmax1;

    // if the grid min/max radii is not provided, try to determine automatically
    if(Rmax==0 || Rmin==0) {
        double mass  = src.totalMass();
        if(mass==0) {   // safe default values for an empty model
            Rmin = zmin = 1.0;
            Rmax = zmax = Rmin * (gridSizeR-1);
            return;
        }
        double rhalf = getRadiusByMass(src, 0.5 * mass);
        if(!isFinite(rhalf))
            throw std::invalid_argument(
                std::string("CylSpline: failed to automatically determine grid extent ") +
                (mass==INFINITY ? "(total mass is infinite)" : "(cannot compute half-mass radius)"));
        double spacing = 1 + sqrt(10./sqrt(gridSizeR*gridSizez));  // ratio between consecutive grid nodes
        if(Rmax==0)
            Rmax = rhalf * std::pow(spacing,  0.5*gridSizeR);
        if(Rmin==0)
            Rmin = rhalf * std::pow(spacing, -0.5*gridSizeR);
        // TODO: introduce a more intelligent adaptive approach as in Multipole,
        // reducing the outer radius if the density drops to zero or decreases too rapidly,
        // and also allow for different radial/vertical scales?
    }
    // TODO: allow the z grid to be different from the R grid if warranted by the model
    if(zmax==0)
        zmax=Rmax;
    if(zmin==0)
        zmin=Rmin;
    FILTERMSG(utils::VL_DEBUG, "CylSpline",
        "Grid in R=["+utils::toString(Rmin)+":"+utils::toString(Rmax)+
             "], z=["+utils::toString(zmin)+":"+utils::toString(zmax)+"]");
}

void chooseGridRadii(const particles::ParticleArray<coord::PosCyl>& particles,
    unsigned int gridSizeR, double &Rmin, double &Rmax,
    unsigned int gridSizez, double &zmin, double &zmax)
{
    if(Rmin!=0 && Rmax!=0 && zmin!=0 && zmax!=0)
        return;

    std::vector<double> radii;
    radii.reserve(particles.size());
    for(size_t i=0; i<particles.size(); i++) {
        double r = sqrt(pow_2(particles.point(i).R) + pow_2(particles.point(i).z));
        if(particles.mass(i) != 0)  // only consider particles with non-zero mass
            radii.push_back(r);
    }
    size_t nbody = radii.size();
    if(nbody==0)
        throw std::invalid_argument("CylSpline: no particles provided as input");

    std::nth_element(radii.begin(), radii.begin() + nbody/2, radii.end());
    double gridSize = sqrt(gridSizeR*gridSizez);  // average of the two sizes
    double Rhalf = radii[nbody/2];   // half-mass radius (if all particles have equal mass)
    double spacing = 1 + sqrt(10./gridSize);
    int Nmin = static_cast<int>(log(nbody+1)/log(2));  // # of points in the inner cell
    if(Rmin==0) {
        std::nth_element(radii.begin(), radii.begin() + Nmin, radii.end());
        Rmin = std::max(radii[Nmin], Rhalf * std::pow(spacing, -0.5*gridSize));
    }
    if(Rmax==0) {
        std::nth_element(radii.begin(), radii.end() - Nmin, radii.end());
        Rmax = std::min(radii[nbody-Nmin], Rhalf * std::pow(spacing, 0.5*gridSize));
    }
    if(zmax==0)
        zmax=Rmax;
    if(zmin==0)
        zmin=Rmin;
    FILTERMSG(utils::VL_DEBUG, "CylSpline",
        "Grid in R=["+utils::toString(Rmin)+":"+utils::toString(Rmax)+
             "], z=["+utils::toString(zmin)+":"+utils::toString(zmax)+"]");
}

}  // internal namespace

// -------- public classes: DensityAzimuthalHarmonic --------- //

shared_ptr<const DensityAzimuthalHarmonic> DensityAzimuthalHarmonic::create(
    const BaseDensity& src, coord::SymmetryType symExp, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax,
    unsigned int gridSizez, double zmin, double zmax,
    bool fixOrder)
{
    coord::SymmetryType symSrc = src.symmetry();
    if(isUnknown(symSrc))
        throw std::invalid_argument(
            "DensityAzimuthalHarmonic: symmetry of the input density model is not specified");
    if(isUnknown(symExp))
        symExp = symSrc;
    else
        symExp = static_cast<coord::SymmetryType>(symSrc | symExp);  // inherit any symmetry from input
    // ensure the grid radii are set to some reasonable values
    chooseGridRadii(src, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("DensityAzimuthalHarmonic: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(symSrc))
        gridz = math::mirrorGrid(gridz);
    // when fixOrder==false, to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle:
    // the number of output harmonics remains the same, but the accuracy of approximation increases
    int mmaxSrc = isZRotSymmetric(symSrc) ? 0 :
        (fixOrder ? mmax : std::max<int>(mmax+MADD_FOURIER, MMIN_FOURIER));
    std::vector< math::Matrix<double> > coefs;
    computeFourierCoefs<BaseDensity>(src, mmaxSrc, gridR, gridz, &coefs);
    // the value at R=0,z=0 might be undefined, in which case we take it from nearby points
    for(unsigned int iz=0; iz<gridz.size(); iz++)
        if(gridz[iz] == 0 && !isFinite(coefs[mmaxSrc](0, iz))) {
            double d1 = coefs[mmaxSrc](0, iz+1);  // value at R=0,z>0
            double d2 = coefs[mmaxSrc](1, iz);    // value at R>0,z=0
            for(unsigned int mm=0; mm<coefs.size(); mm++)
                if(coefs[mm].cols()>0)  // loop over all non-empty harmonics
                    coefs[mm](0, iz) = (int)mm==mmaxSrc ? (d1+d2)/2 : 0;  // only m=0 survives
        }
    // resize the coefficients back to the requested order and symmetry
    restrictFourierCoefs<1>(mmax, symExp, &coefs, gridz);
    return shared_ptr<const DensityAzimuthalHarmonic>(new DensityAzimuthalHarmonic(gridR, gridz, coefs));
}

// the actual constructor
DensityAzimuthalHarmonic::DensityAzimuthalHarmonic(
    const std::vector<double>& gridR_orig, const std::vector<double>& gridz_orig,
    const std::vector< math::Matrix<double> > &coefs)
{
    unsigned int sizeR = gridR_orig.size(), sizez_orig = gridz_orig.size(), sizez = sizez_orig;
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE || coefs.size()%2 == 0)
        throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect grid size");
    int mysym = coord::ST_AXISYMMETRIC;
    // grid in z may only cover half-space z>=0 if the density is z-reflection symmetric
    std::vector<double> gridR=gridR_orig, gridz;
    if(gridz_orig[0] == 0) {
        gridz = math::mirrorGrid(gridz_orig);
        sizez = 2*sizez_orig-1;
    } else {  // if the original grid covered both z>0 and z<0, we assume that the symmetry is broken
        gridz = gridz_orig;
        mysym &= ~(coord::ST_ZREFLECTION | coord::ST_REFLECTION);
    }
    Rscale = gridR_orig[sizeR/2];  // take some reasonable value for scale radius for coord transformation
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = asinh(gridR[iR]/Rscale);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = asinh(gridz[iz]/Rscale);

    math::Matrix<double> val(sizeR, sizez);
    int mmax = (coefs.size()-1)/2;
    spl.resize(2*mmax+1);
    for(int mm=0; mm<=2*mmax; mm++) {
        if(coefs[mm].rows() == 0 && coefs[mm].cols() == 0)
            continue;
        if(coefs[mm].rows() != sizeR || coefs[mm].cols() != sizez_orig)
            throw std::invalid_argument("DensityAzimuthalHarmonic: incorrect coefs array size");
        double sum=0;
        for(unsigned int iR=0; iR<sizeR; iR++)
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                double value = coefs[mm](iR, iz);
                if((mysym & coord::ST_ZREFLECTION) == coord::ST_ZREFLECTION) {
                    val(iR, sizez_orig-1-iz) = value;
                    val(iR, sizez_orig-1+iz) = value;
                } else
                    val(iR, iz) = value;
                sum += fabs(value);
            }
        int m = mm-mmax;
        if(sum>0 || m==0) {
            spl[mm].reset(new math::CubicSpline2d(gridR, gridz, val));
            if(m!=0)  // no z-rotation symmetry because m!=0 coefs are non-zero
                mysym &= ~coord::ST_ZROTATION;
            if(m<0)
                mysym &= ~(coord::ST_YREFLECTION | coord::ST_REFLECTION);
            if((m<0) ^ (m%2 != 0))
                mysym &= ~(coord::ST_XREFLECTION | coord::ST_REFLECTION);
        }
    }
    sym = static_cast<coord::SymmetryType>(mysym);
}

double DensityAzimuthalHarmonic::rho_m(int m, double R, double z) const
{
    int mmax = (spl.size()-1)/2;
    double lR = asinh(R/Rscale), lz = asinh(z/Rscale);
    if( math::abs(m)>mmax || !spl[m+mmax] ||
        lR<spl[mmax]->xmin() || lz<spl[mmax]->ymin() ||
        lR>spl[mmax]->xmax() || lz>spl[mmax]->ymax() )
        return 0;
    return spl[m+mmax]->value(lR, lz);
}

double DensityAzimuthalHarmonic::densityCyl(const coord::PosCyl &pos, double /*time*/) const
{
    int mmax = (spl.size()-1)/2;
    double lR = asinh(pos.R/Rscale), lz = asinh(pos.z/Rscale);
    assert(spl[mmax]);  // the spline for m=0 must exist
    if( lR<spl[mmax]->xmin() || lz<spl[mmax]->ymin() ||
        lR>spl[mmax]->xmax() || lz>spl[mmax]->ymax() )
        return 0;
    double result = 0;

    bool needSine = !isYReflSymmetric(sym);
    // temporary stack-allocated array
    double* trig = static_cast<double*>(alloca(mmax*(1+needSine) * sizeof(double)));
    if(!isZRotSymmetric(sym))
        math::trigMultiAngle(pos.phi, mmax, needSine, trig);
    for(int m=-mmax; m<=mmax; m++) {
        if(spl[m+mmax]) {
            double trig_m = m==0 ? 1 : m>0 ? trig[m-1] : trig[mmax-1-m];
            result += spl[m+mmax]->value(lR, lz) * trig_m;
        }
    }
    return result;
}

void DensityAzimuthalHarmonic::getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(spl.size() == mmax*2+1 && !spl[mmax]->empty());
    const std::vector<double>& scaledR = spl[mmax]->xvalues();
    const std::vector<double>& scaledz = spl[mmax]->yvalues();
    Rmin = Rscale * sinh(scaledR[1]);
    Rmax = Rscale * sinh(scaledR.back());
    zmin = Rscale * sinh(scaledz[scaledz.size()/2+1]);  // first node above zero
    zmax = Rscale * sinh(scaledz.back());
}

void DensityAzimuthalHarmonic::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &coefs) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(/*mmax>=0 &&*/ spl.size() == mmax*2+1 && spl[mmax]);
    coefs.resize(2*mmax+1);
    unsigned int sizeR = spl[mmax]->xvalues().size();
    unsigned int sizez = spl[mmax]->yvalues().size();
    gridR = spl[mmax]->xvalues();
    if(isZReflSymmetric(sym)) {
        // output only coefs for half-space z>=0
        sizez = (sizez+1) / 2;
        gridz.assign(spl[mmax]->yvalues().begin() + sizez-1, spl[mmax]->yvalues().end());
    } else
        gridz = spl[mmax]->yvalues();
    for(unsigned int mm=0; mm<=2*mmax; mm++)
        if(spl[mm]) {
            coefs[mm]=math::Matrix<double>(sizeR, sizez);
            for(unsigned int iR=0; iR<sizeR; iR++)
                for(unsigned int iz=0; iz<sizez; iz++)
                    coefs[mm](iR, iz) = spl[mm]->value(gridR[iR], gridz[iz]);
            //math::eliminateNearZeros(coefs[mm]);
        }
    // unscale the grid coordinates
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = Rscale * sinh(gridR[iR]);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = Rscale * sinh(gridz[iz]);
}

// -------- public classes: CylSpline --------- //

shared_ptr<const CylSpline> CylSpline::create(
    const BaseDensity& src, coord::SymmetryType symExp, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax,
    unsigned int gridSizez, double zmin, double zmax,
    bool fixOrder, bool useDerivs)
{
    coord::SymmetryType symSrc = src.symmetry();
    if(isUnknown(symSrc))
        throw std::invalid_argument("CylSpline: symmetry of the input density model is not specified");
    if(isUnknown(symExp))
        symExp = symSrc;
    else
        symExp = static_cast<coord::SymmetryType>(symSrc | symExp);  // inherit any symmetry from input
    // ensure the grid radii are set to some reasonable values
    chooseGridRadii(src, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(symSrc))
        gridz = math::mirrorGrid(gridz);

    PtrDensity densInterp;  // pointer to an internally created interpolating object if it is needed
    const BaseDensity* dens = &src;  // pointer to either the original density or the interpolated one
    // For an axisymmetric potential we don't use interpolation,
    // as the Fourier expansion of density trivially has only one harmonic;
    // also, if the input density is already a Fourier expansion, use it directly.
    // Otherwise, we need to create a temporary DensityAzimuthalHarmonic interpolating object.
    if(!isZRotSymmetric(symSrc) && !dynamic_cast<const DensityAzimuthalHarmonic*>(&src)) {
        double Rmax = gridR.back() * 2;
        double Rmin = gridR[1] * 0.01;
        double zmax = gridz.back() * 2;
        double zmin = gridz[0]==0 ? gridz[1] * 0.01 :
            gridz[gridz.size()/2]==0 ? gridz[gridz.size()/2+1] * 0.01 : Rmin;
        double delta=0.1;  // relative difference between grid nodes = log(x[n+1]/x[n])
        // create a density interpolator; it will be automatically deleted upon return
        densInterp = DensityAzimuthalHarmonic::create(src, symExp, mmax,
            static_cast<unsigned int>(log(Rmax/Rmin)/delta), Rmin, Rmax,
            static_cast<unsigned int>(log(zmax/zmin)/delta), zmin, zmax, fixOrder);
        // and use it for computing the potential
        dens = densInterp.get();
    }

    std::vector< math::Matrix<double> > coefs[3]; // Phi, dPhidR, dPhidz
    computePotentialCoefsFromDensity(*dens, mmax, gridR, gridz, useDerivs, /*output*/coefs);
    // eliminate m-terms that are identically zero, and enforce z-reflection symmetry if needed
    if(useDerivs)
        restrictFourierCoefs<3>(mmax, symExp, coefs, gridz);
    else
        restrictFourierCoefs<1>(mmax, symExp, coefs, gridz);
    return shared_ptr<const CylSpline>(new CylSpline(gridR, gridz, coefs[0], coefs[1], coefs[2]));
}

shared_ptr<const CylSpline> CylSpline::create(
    const BasePotential& src, coord::SymmetryType symExp, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax,
    unsigned int gridSizez, double zmin, double zmax,
    bool fixOrder)
{
    coord::SymmetryType symSrc = src.symmetry();
    if(isUnknown(symSrc))
        throw std::invalid_argument("CylSpline: symmetry of the input potential model is not specified");
    if(isUnknown(symExp))
        symExp = symSrc;
    else
        symExp = static_cast<coord::SymmetryType>(symSrc | symExp);  // inherit any symmetry from input
    chooseGridRadii(src, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(symSrc))
        gridz = math::mirrorGrid(gridz);
    // when fixOrder==false, to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle:
    // the number of output harmonics remains the same, but the accuracy of approximation increases
    int mmaxSrc = isZRotSymmetric(src) ? 0 :
        (fixOrder ? mmax : std::max<int>(mmax+MADD_FOURIER, MMIN_FOURIER));
    std::vector< math::Matrix<double> > coefs[3],
    /*aliases*/ &Phi=coefs[0], &dPhidR=coefs[1], &dPhidz=coefs[2];
    computeFourierCoefs<BasePotential>(src, mmaxSrc, gridR, gridz, coefs);
    // assign potential derivatives at R=0 or z=0 to zero, depending on the symmetry
    for(unsigned int iz=0; iz<gridz.size(); iz++) {
        if(gridz[iz] == 0 && isZReflSymmetric(symSrc)) {
            for(unsigned int iR=0; iR<gridR.size(); iR++)
                for(unsigned int mm=0; mm<dPhidz.size(); mm++)
                    if(dPhidz[mm].cols()>0)  // loop over all non-empty harmonics
                        dPhidz[mm](iR, iz) = 0; // z-derivative is zero in the symmetry plane
        }
        for(unsigned int mm=0; mm<dPhidR.size(); mm++)
            if(dPhidR[mm].cols()>0)  // loop over all non-empty harmonics
                dPhidR[mm](0, iz) = 0;  // R-derivative at R=0 should always be zero
    }
    // resize the coefficients back to the requested order and symmetry
    restrictFourierCoefs<3>(mmax, symExp, coefs, gridz);
    return shared_ptr<const CylSpline>(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

shared_ptr<const CylSpline> CylSpline::create(
    const particles::ParticleArray<coord::PosCyl>& particles,
    coord::SymmetryType sym, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax,
    unsigned int gridSizez, double zmin, double zmax, bool useDerivs)
{
    if(isUnknown(sym))
        throw std::invalid_argument("CylSpline: symmetry is not specified");
    chooseGridRadii(particles, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(sym))
        gridz = math::mirrorGrid(gridz);
    if(isZRotSymmetric(sym))
        mmax = 0;
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    std::vector< math::Matrix<double> >* output[] = {&Phi, &dPhidR, &dPhidz};
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, sym);
    std::vector<std::vector<double> > harmonics(2*mmax+1);
    std::vector<std::pair<double, double> > Rz;
    computeAzimuthalHarmonicsFromParticles(particles, indices, harmonics, Rz);
    computePotentialCoefsFromParticles(indices, harmonics, Rz, gridR, gridz, useDerivs, output);
    return shared_ptr<const CylSpline>(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

// the actual constructor
CylSpline::CylSpline(
    const std::vector<double> &gridR_orig,
    const std::vector<double> &gridz_orig,
    const std::vector< math::Matrix<double> > &Phi,
    const std::vector< math::Matrix<double> > &dPhidR,
    const std::vector< math::Matrix<double> > &dPhidz)
{
    unsigned int sizeR = gridR_orig.size(), sizez = gridz_orig.size(), sizez_orig = sizez;
    int mmax  = (Phi.size()-1)/2;  // index of the m=0 term
    // use quintic splines if derivs are provided, otherwise cubic
    haveDerivs = dPhidR.size() > 0 && dPhidz.size() > 0;
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE ||
        gridR_orig[0]!=0 || Phi.size()%2 == 0 || Phi[mmax].size() == 0 ||
        (haveDerivs && (Phi.size() != dPhidR.size() || Phi.size() != dPhidz.size())) )
        throw std::invalid_argument("CylSpline: incorrect grid or coefs array size");
    spl.resize(2*mmax+1);
    int mysym = coord::ST_AXISYMMETRIC;
    bool zsym = true;
    double Phi0;  // potential at R=0,z=0
    // grid in z may only cover half-space z>=0 if the density is z-reflection symmetric:
    std::vector<double> gridR = gridR_orig, gridz;
    if(gridz_orig[0] == 0) {
        gridz = math::mirrorGrid(gridz_orig);
        sizez = 2*sizez_orig-1;
        Phi0  = Phi[mmax](0, 0);
    } else {  // if the original grid covered both z>0 and z<0, we assume that the symmetry is broken
        gridz = gridz_orig;
        mysym&= ~(coord::ST_ZREFLECTION | coord::ST_REFLECTION);
        zsym  = false;
        Phi0  = Phi[mmax](0, (sizez+1)/2);
    }

    asymptOuter = determineAsympt(gridR_orig, gridz_orig, Phi);
    // at large radii, Phi(r) ~= -Mtotal/r
    double Mtot = -(asymptOuter->value(coord::PosSph(gridR.back(), 0, 0)) * gridR.back());
    if(Phi0 < 0 && Mtot > 0)     // assign Rscale so that it approximately equals -Mtotal/Phi(r=0),
        Rscale  = -Mtot / Phi0;  // i.e. would equal the scale radius of a Plummer potential
    else
        Rscale  = gridR[sizeR/2];  // rather arbitrary

    // transform the grid to log-scaled coordinates
    for(unsigned int i=0; i<sizeR; i++) {
        gridR[i] = asinh(gridR[i]/Rscale);
    }
    for(unsigned int i=0; i<sizez; i++) {
        gridz[i] = asinh(gridz[i]/Rscale);
    }

    // check if we may use log-scaling of the m=0 term (i.e. if the potential is negative everywhere)
    logScaling = true;
    for(unsigned int i=0; i<Phi[mmax].size(); i++)
        logScaling &= Phi[mmax].data()[i] < 0;

    // temporary matrices of scaled potential and derivatives used to construct 2d splines
    math::Matrix<double> val(sizeR, sizez), derR(sizeR, sizez), derz(sizeR, sizez);
    math::Matrix<double> val0, derR0, derz0;   // copies of these matrices for the m=0 term

    // loop over azimuthal harmonic indices (m)
    for(int im=0; im<=2*mmax; im++) {
        // we process the terms in the following order:
        // first start with the m=0 term, which is contained in the arrays Phi[mmax], ...
        // then go over all positive m up to mmax inclusive (the last one is in Phi[mmax*2]),
        // and then to negative ones (contained in Phi[0]..Phi[mmax-1]).
        // This is done to ensure that we first consider the m=0 term,
        // whose amplitude is used to normalize the other terms.
        int mm = (im+mmax) % (2*mmax+1), m = mm-mmax;
        if(Phi[mm].rows() == 0 && Phi[mm].cols() == 0)
            continue;
        if((   Phi[mm].rows() != sizeR ||    Phi[mm].cols() != sizez_orig) || (haveDerivs &&
           (dPhidR[mm].rows() != sizeR || dPhidR[mm].cols() != sizez_orig  ||
            dPhidz[mm].rows() != sizeR || dPhidz[mm].cols() != sizez_orig)))
            throw std::invalid_argument("CylSpline: incorrect coefs array size");
        bool nontrivial = false;  // keep track if this term is identically zero or not
        for(unsigned int iR=0; iR<sizeR; iR++) {
            double R = gridR_orig[iR];
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                double z = gridz_orig[iz];
                nontrivial |= Phi[mm](iR, iz) != 0;
                unsigned int iz1 = zsym ? sizez_orig-1+iz : iz;  // index in the internal 2d grid
                // values of potential and its derivatives are represented as (optionally) scaled
                // 2d functions, and the derivatives are transformed to the asinh-scaled coordinates
                double dRdRscaled = sqrt(R*R + Rscale*Rscale), dzdzscaled = sqrt(z*z + Rscale*Rscale);
                if(logScaling) {
                    // if the potential is everywhere negative, the m=0 term is log-scaled,
                    // and other terms are normalized by the value of the m=0 term
                    if(m == 0) {
                        val(iR,iz1) = log(-Phi[mm](iR,iz));
                        if(haveDerivs) {
                            derR(iR,iz1) = dRdRscaled * dPhidR[mm](iR,iz) / Phi[mm](iR,iz);
                            derz(iR,iz1) = dzdzscaled * dPhidz[mm](iR,iz) / Phi[mm](iR,iz);
                        }
                    } else {   // normalize by the m=0 term contained in the [mmax] array element
                        double v0 = Phi[mmax](iR,iz), vm = Phi[mm](iR,iz) / v0;
                        val(iR,iz1) = vm;
                        if(haveDerivs) {
                            derR(iR,iz1) = dRdRscaled * (dPhidR[mm](iR,iz) - vm * dPhidR[mmax](iR,iz)) / v0;
                            derz(iR,iz1) = dzdzscaled * (dPhidz[mm](iR,iz) - vm * dPhidz[mmax](iR,iz)) / v0;
                        }
                    }
                } else {   // no scaling
                    val(iR,iz1) = Phi[mm](iR,iz);
                    if(haveDerivs) {
                        derR(iR,iz1) = dRdRscaled * dPhidR[mm](iR,iz);
                        derz(iR,iz1) = dzdzscaled * dPhidz[mm](iR,iz);
                    }
                }
                if(zsym) {
                    assert(gridz_orig[iz]>=0);           // source data only covers upper half-space
                    unsigned int iz2 = sizez_orig-1-iz;  // index of the reflected cell
                    val (iR, iz2) = val (iR, iz1);
                    derR(iR, iz2) = derR(iR, iz1);
                    derz(iR, iz2) =-derz(iR, iz1);
                }
            }
        }
        if(m == 0) {  // store the values and derivatives of the scaled m=0 term
            val0  = val;
            derR0 = derR;
            derz0 = derz;
        }
        if(nontrivial || m==0) {  // only construct splines if they are not identically zero or m=0
            spl[mm] = haveDerivs ?
                math::PtrInterpolator2d(new math::QuinticSpline2d(gridR, gridz, val, derR, derz)) :
                math::PtrInterpolator2d(new math::CubicSpline2d(gridR, gridz, val,
                    /*regularize*/false, /*dPhi/dR at R=0*/0, /*other derivs unspecified*/NAN, NAN, NAN));
            // check if this non-trivial harmonic breaks any symmetry
            if(m!=0)  // no z-rotation symmetry because m!=0 coefs are non-zero
                mysym &= ~coord::ST_ZROTATION;
            if(m<0)
                mysym &= ~(coord::ST_YREFLECTION | coord::ST_REFLECTION);
            if((m<0) ^ (m%2 != 0))
                mysym &= ~(coord::ST_XREFLECTION | coord::ST_REFLECTION);
        }
    }
    sym = static_cast<coord::SymmetryType>(mysym);
}

void CylSpline::evalCyl(const coord::PosCyl &pos,
    double* val, coord::GradCyl* der, coord::HessCyl* der2, double /*time*/) const
{
    int mmax = (spl.size()-1)/2;
    double Rscaled = asinh(pos.R/Rscale);
    double zscaled = asinh(pos.z/Rscale);
    if( Rscaled<spl[mmax]->xmin() || zscaled<spl[mmax]->ymin() ||
        Rscaled>spl[mmax]->xmax() || zscaled>spl[mmax]->ymax() ) {
        // outside the grid definition region, use the asymptotic expansion
        asymptOuter->eval(pos, val, der, der2);
        return;
    }
    double dRscaleddR   = 1 / sqrt(pow_2(pos.R) + pow_2(Rscale));
    double dzscaleddz   = 1 / sqrt(pow_2(pos.z) + pow_2(Rscale));
    double d2RscaleddR2 = -pos.R * pow_3(dRscaleddR);
    double d2zscaleddz2 = -pos.z * pow_3(dzscaleddz);

    // only compute those quantities that will be needed in output
    const bool needPhi  = true;
    const bool needGrad = der !=NULL || der2!=NULL;
    const bool needHess = der2!=NULL;

    // value and derivatives (in scaled coords) of the m=0 term, which are later used
    // to scale the other terms after we have performed the Fourier transform on all of them
    double Phi0, dPhi0dR, dPhi0dz, d2Phi0dR2, d2Phi0dRdz, d2Phi0dz2;
    spl[mmax]->evalDeriv(Rscaled, zscaled,
        needPhi  ?   &Phi0     : NULL,
        needGrad ?  &dPhi0dR   : NULL,
        needGrad ?  &dPhi0dz   : NULL,
        needHess ? &d2Phi0dR2  : NULL,
        needHess ? &d2Phi0dRdz : NULL,
        needHess ? &d2Phi0dz2  : NULL);
    if(logScaling) {
        Phi0 = -exp(Phi0);
        if(needHess) {
            d2Phi0dR2  = Phi0 * (d2Phi0dR2  + pow_2(dPhi0dR));
            d2Phi0dRdz = Phi0 * (d2Phi0dRdz + dPhi0dR * dPhi0dz);
            d2Phi0dz2  = Phi0 * (d2Phi0dz2  + pow_2(dPhi0dz));
        }
        if(needGrad) {
            dPhi0dR *= Phi0;
            dPhi0dz *= Phi0;
        }
    }

    // if the potential is axisymmetric, skip the Fourier transform and amplitude scaling
    if(mmax==0) {
        if(val)
            *val = Phi0;
        if(der) {
            der->dR   = dPhi0dR * dRscaleddR;
            der->dz   = dPhi0dz * dzscaleddz;
            der->dphi = 0;
        }
        if(der2) {
            der2->dR2 = d2Phi0dR2 * pow_2(dRscaleddR) + dPhi0dR * d2RscaleddR2;
            der2->dz2 = d2Phi0dz2 * pow_2(dzscaleddz) + dPhi0dz * d2zscaleddz2;
            der2->dRdz= d2Phi0dRdz * dRscaleddR * dzscaleddz;
            der2->dRdphi = der2->dzdphi = der2->dphi2 = 0;
        }
        return;
    }

    // total scaled potential, gradient and hessian in scaled coordinates:
    // if using log-scaling, the values of m!=0 coefs are multiplied by the value of the m=0 term,
    // which we do at the very end, so initialize the sum with the value of the m=0 term
    // scaled by itself, i.e. unity; otherwise we simply sum up all m terms without any scaling
    double Phi = logScaling ? 1 : Phi0;
    coord::GradCyl grad;
    coord::HessCyl hess;
    grad.dR  = grad.dz  = grad.dphi  = 0;
    hess.dR2 = hess.dz2 = hess.dphi2 = hess.dRdz = hess.dRdphi = hess.dzdphi = 0;

    bool needSine = needGrad || !isYReflSymmetric(sym);
    double* trig_arr = static_cast<double*>(alloca(mmax*(1+needSine) * sizeof(double)));
    math::trigMultiAngle(pos.phi, mmax, needSine, trig_arr);

    // loop over other (m!=0) azimuthal harmonics and compute the temporary (scaled) values
    for(int mm=0; mm<=2*mmax; mm++) {
        int m = mm-mmax;
        if(!spl[mm] || m==0)  // empty harmonic or the already computed m=0 one
            continue;
        // scaled value, gradient and hessian of m-th harmonic in scaled coordinates
        double Phi_m;
        coord::GradCyl dPhi_m;
        coord::HessCyl d2Phi_m;
        spl[mm]->evalDeriv(Rscaled, zscaled,
            needPhi  ?   &Phi_m      : NULL,
            needGrad ?  &dPhi_m.dR   : NULL,
            needGrad ?  &dPhi_m.dz   : NULL,
            needHess ? &d2Phi_m.dR2  : NULL,
            needHess ? &d2Phi_m.dRdz : NULL,
            needHess ? &d2Phi_m.dz2  : NULL);
        double trig  = m>0 ? trig_arr[m-1] : trig_arr[mmax-1-m];  // cos or sin
        double dtrig = m>0 ? -m*trig_arr[mmax+m-1] : -m*trig_arr[-m-1];
        double d2trig = -m*m*trig;
        Phi += Phi_m * trig;
        if(needGrad) {
            grad.dR   += dPhi_m.dR *  trig;
            grad.dz   += dPhi_m.dz *  trig;
            grad.dphi +=  Phi_m    * dtrig;
        }
        if(needHess) {
            hess.dR2    += d2Phi_m.dR2  *   trig;
            hess.dz2    += d2Phi_m.dz2  *   trig;
            hess.dRdz   += d2Phi_m.dRdz *   trig;
            hess.dRdphi +=  dPhi_m.dR   *  dtrig;
            hess.dzdphi +=  dPhi_m.dz   *  dtrig;
            hess.dphi2  +=   Phi_m      * d2trig;
        }
    }

    if(logScaling) {
        // unscale both amplitude of all quantities and their coordinate derivatives
        if(val)
            *val = Phi0 * Phi;
        if(der) {
            der->dR   = (Phi0 * grad.dR + dPhi0dR * Phi) * dRscaleddR;
            der->dz   = (Phi0 * grad.dz + dPhi0dz * Phi) * dzscaleddz;
            der->dphi =  Phi0 * grad.dphi;
        }
        if(der2) {
            der2->dR2 = (Phi0 * hess.dR2 + 2 * dPhi0dR * grad.dR + d2Phi0dR2 * Phi) *
                pow_2(dRscaleddR)  +  (Phi0 * grad.dR + dPhi0dR * Phi) * d2RscaleddR2;
            der2->dz2 = (Phi0 * hess.dz2 + 2 * dPhi0dz * grad.dz + d2Phi0dz2 * Phi) *
                pow_2(dzscaleddz)  +  (Phi0 * grad.dz + dPhi0dz * Phi) * d2zscaleddz2;
            der2->dRdz = (Phi0 * hess.dRdz + dPhi0dR * grad.dz + dPhi0dz * grad.dR + d2Phi0dRdz * Phi) *
                dRscaleddR * dzscaleddz;
            der2->dRdphi = (Phi0 * hess.dRdphi + dPhi0dR * grad.dphi) * dRscaleddR;
            der2->dzdphi = (Phi0 * hess.dzdphi + dPhi0dz * grad.dphi) * dzscaleddz;
            der2->dphi2  =  Phi0 * hess.dphi2;
        }
    } else {
        // unscale just the derivatives according to the coordinate transformation
        if(val)
            *val = Phi;
        if(der) {
            der->dR   = (grad.dR + dPhi0dR) * dRscaleddR;
            der->dz   = (grad.dz + dPhi0dz) * dzscaleddz;
            der->dphi = grad.dphi;
        }
        if(der2) {
            der2->dR2 = (hess.dR2 + d2Phi0dR2) * pow_2(dRscaleddR) + (grad.dR + dPhi0dR) * d2RscaleddR2;
            der2->dz2 = (hess.dz2 + d2Phi0dz2) * pow_2(dzscaleddz) + (grad.dz + dPhi0dz) * d2zscaleddz2;
            der2->dRdz = (hess.dRdz + d2Phi0dRdz) * dRscaleddR * dzscaleddz;
            der2->dRdphi = hess.dRdphi * dRscaleddR;
            der2->dzdphi = hess.dzdphi * dzscaleddz;
            der2->dphi2  = hess.dphi2;
        }
    }
}

void CylSpline::getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(spl.size() == mmax*2+1 && !spl[mmax]->empty());
    const std::vector<double>& scaledR = spl[mmax]->xvalues();
    const std::vector<double>& scaledz = spl[mmax]->yvalues();
    Rmin = Rscale * sinh(scaledR[1]);
    Rmax = Rscale * sinh(scaledR.back());
    zmin = Rscale * sinh(scaledz[scaledz.size()/2+1]);  // first node above zero
    zmax = Rscale * sinh(scaledz.back());
}

void CylSpline::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(/*mmax>=0 &&*/ spl.size() == mmax*2+1 && !spl[mmax]->empty());
    const std::vector<double>& scaledR = spl[mmax]->xvalues();
    const std::vector<double>& scaledz = spl[mmax]->yvalues();
    unsigned int sizeR = scaledR.size();
    unsigned int sizez = scaledz.size();
    unsigned int iz0   = 0;
    if(isZReflSymmetric(sym)) {
        // output only coefs for half-space z>=0
        sizez = (sizez+1) / 2;
        iz0   = sizez-1;  // index of z=0 value in the internal scaled coordinate axis array
    }
    // unscale the coordinates
    gridR.resize(sizeR);
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = Rscale * sinh(scaledR[iR]);
    gridz.resize(sizez);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = Rscale * sinh(scaledz[iz+iz0]);

    // output derivs only if they were provided as input (i.e. when using quintic splines)
    Phi.assign(2*mmax+1, math::Matrix<double>());
    dPhidR.assign(haveDerivs? 2*mmax+1 : 0, math::Matrix<double>());
    dPhidz.assign(haveDerivs? 2*mmax+1 : 0, math::Matrix<double>());
    for(unsigned int mm=0; mm<=2*mmax; mm++) {
        if(!spl[mm])
            continue;
        Phi[mm]=math::Matrix<double>(sizeR, sizez);
        if(haveDerivs) {
            dPhidR[mm]=math::Matrix<double>(sizeR, sizez);
            dPhidz[mm]=math::Matrix<double>(sizeR, sizez);
        }
    }
    for(unsigned int iR=0; iR<sizeR; iR++) {
        for(unsigned int iz=0; iz<sizez; iz++) {
            double Rscaled = scaledR[iR];     // coordinates in the internal scaled coords array
            double zscaled = scaledz[iz+iz0];
            double dRscaleddR = 1 / sqrt(pow_2(gridR[iR]) + pow_2(Rscale));
            double dzscaleddz = 1 / sqrt(pow_2(gridz[iz]) + pow_2(Rscale));
            if(logScaling) {
                // first retrieve the m=0 harmonic and un-log-scale it;
                // the derivatives are taken w.r.t. scaled R and z
                double Phi0, dPhi0dR, dPhi0dz;
                spl[mmax]->evalDeriv(Rscaled, zscaled, &Phi0, &dPhi0dR, &dPhi0dz);
                Phi0 = -exp(Phi0);
                dPhi0dR *= Phi0;
                dPhi0dz *= Phi0;
                // then loop over all harmonics, scaling the amplitude by that of the m=0 term,
                // and convert the derivatives from scaled to real R and z
                for(unsigned int mm=0; mm<=2*mmax; mm++) {
                    if(!spl[mm])
                        continue;
                    double val=1, derR=0, derz=0;
                    if(mm!=mmax)  // [mmax] is the m=0 term, "scaled by itself", i.e. identically unity
                        spl[mm]->evalDeriv(Rscaled, zscaled, &val, &derR, &derz);
                    Phi[mm](iR,iz) = Phi0 * val;
                    if(haveDerivs) {
                        dPhidR[mm](iR,iz) = (Phi0 * derR + dPhi0dR * val) * dRscaleddR;
                        dPhidz[mm](iR,iz) = (Phi0 * derz + dPhi0dz * val) * dzscaleddz;
                    }
                }
            } else {
                // the no-scaling case is simpler - just convert the derivatives for each term
                for(unsigned int mm=0; mm<=2*mmax; mm++) {
                    if(!spl[mm])
                        continue;
                    double derR, derz;
                    spl[mm]->evalDeriv(Rscaled, zscaled, &Phi[mm](iR,iz), &derR, &derz);
                    if(haveDerivs) {
                        dPhidR[mm](iR,iz) = derR * dRscaleddR;
                        dPhidz[mm](iR,iz) = derz * dzscaleddz;
                    }
                }
            }
        }
    }
}

double CylSpline::enclosedMass(double radius) const
{
    if(radius==0)
        return 0;
    int mmax = (spl.size()-1)/2;
    double Rscaled = asinh(radius/Rscale);
    if( Rscaled<spl[mmax]->xmin() || Rscaled>spl[mmax]->xmax() ) {
        // outside the grid definition region, use the asymptotic expansion
        if(radius == INFINITY)
            radius = sinh(spl[mmax]->xmax()) * 1e20;
        coord::GradCyl grad;
        asymptOuter->eval(coord::PosCyl(radius, 0, 0), NULL, &grad);
        return grad.dR * pow_2(radius);
    }
    // use the radial derivative of the m=0 harmonic to estimate the enclosed mass
    double dRscaleddR = 1 / sqrt(pow_2(radius) + pow_2(Rscale)), Phi0, dPhi0dRscaled;
    spl[mmax]->evalDeriv(Rscaled, 0, &Phi0, &dPhi0dRscaled);
    if(logScaling) {
        Phi0 = -exp(Phi0);
        dPhi0dRscaled *= Phi0;
    }
    return dPhi0dRscaled * dRscaleddR * pow_2(radius);
}

}  // namespace potential
