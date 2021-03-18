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
#include <alloca.h>

namespace potential {

// internal definitions
namespace{

/// minimum number of grid nodes
static const unsigned int CYLSPLINE_MIN_GRID_SIZE = 2;

/// minimum number of terms in Fourier expansion used to compute coefficients
/// of a non-axisymmetric density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)
static const unsigned int MMIN_AZIMUTHAL_FOURIER = 16;

/// order of multipole extrapolation outside the grid
static const int LMAX_EXTRAPOLATION = 8;

/// max number of function evaluations in multidimensional integration
static const unsigned int MAX_NUM_EVAL = 10000;

/// to avoid singularities in potential integration kernel, we add a small softening
/// (intended to be much less than typical grid spacing) - perhaps need to make it grid-dependent
static const double EPS2_SOFTENING = 1e-12;

/// relative accuracy of potential computation (integration tolerance parameter)
static const double EPSREL_POTENTIAL_INT = 1e-6;

/// eliminate Fourier terms whose relative amplitude is less than this number
static const double EPS_COEF = 1e-10;

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
    src.evalmanyDensityCyl(points.size(), &points[0], values);   // vectorized evaluation at many point
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
    std::vector< math::Matrix<double> >* coefs[])
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
        coefs[q]->resize(mmax*2+1);
        for(size_t i=0; i<numHarmonicsComputed; i++)
            coefs[q]->at(indices[i]+mmax) = math::Matrix<double>(sizeR, sizez);
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
                    coefs[q]->at(m+mmax)(iR, iz) =
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
    if(dens.name() == DensityAzimuthalHarmonic::myName()) {  // quickly compare char* pointers, not strings
        for(size_t p=0; p<npoints; p++)
            rho[p] = static_cast<const DensityAzimuthalHarmonic&>(dens).rho_m(m, pos[p].R, pos[p].z);
    } else {  // use the input density directly in the axisymmetric case
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
            // only soften the derivative, because it diverges as 1/|u-1|,
            // but the infinite contributions from z>z0 and z<z0 should nearly cancel anyway
            // when one approaches the singularity
            dQ = math::sign(dQ) / sqrt( 1/pow_2(dQ) + EPS2_SOFTENING);
            values[1] += -sq * mass * (dQ/R - (Q/2 + u*dQ)/R0);
            values[2] += -sq * mass * dQ * (z0-z) / (R*R0);
        }
    } else      // degenerate case
    if(m==0) {  // here only m=0 harmonic survives;
        double s = 1 / sqrt(t + EPS2_SOFTENING); // actually the integration never reaches R=0 anyway
        values[0] += -mass * s;
        if(useDerivs)
            values[2] += mass * s * (z0-z) / t;
    }
}

// the N-dimensional integrand for computing the potential harmonics from density
class AzimuthalHarmonicIntegrand: public math::IFunctionNdim {
public:
    AzimuthalHarmonicIntegrand(const BaseDensity& _dens, int _m,
        double _R0, double _z0, double _Rmin, double _zmin, double _Rmax, double _zmax, bool _useDerivs) :
        dens(_dens), m(_m),
        R0(_R0), z0(_z0), Rmin(_Rmin), zmin(_zmin), Rmax(_Rmax), zmax(_zmax), useDerivs(_useDerivs),
        jac0(2*M_PI * log(1 + Rmax/Rmin) * Rmin * log(1 + zmax/zmin) * zmin)
    {}

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
            const double pR = pow(1 + Rmax/Rmin, vars[p*2+0]), R = Rmin * (pR - 1);
            const double pz = pow(1 + zmax/zmin, vars[p*2+1]), z = zmin * (pz - 1);
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
        if(!isZReflSymmetric(dens)) {
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
        }
        
        // workaround for the n-dimensional quadrature routine: it seems to be unable
        // to properly handle cases when one of components of the integrand is identically zero,
        // that's why we output 1 instead, and zero it out later
        if(useDerivs && R0==0)
            for(size_t p=0; p<npoints; p++)
                values[p * nvalues + 1] = 1;
        if(useDerivs && isZReflSymmetric(dens) && z0==0)
            for(size_t p=0; p<npoints; p++)
                values[p * nvalues + 2] = 1;
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return useDerivs ? 3 : 1; }
private:
    const BaseDensity& dens;  ///< the density profile in the Poisson eqn
    const int m;              ///< azimuthal harmonic number
    const double R0, z0;      ///< the point at which the integral is computed
    const double Rmin, zmin;  ///< smallest grid segment
    const double Rmax, zmax;  ///< extent of the integration domain
    const bool useDerivs;     ///< whether to compute only the potential or also its partial derivs by R,z
    const double jac0;        ///< constant prefactor for the jacobian
};

void computePotentialCoefsFromDensity(const BaseDensity &src,
    unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> >* output[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE)
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    if(isZRotSymmetric(src))
        mmax = 0;
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, src.symmetry());
    unsigned int numQuantitiesOutput = useDerivs ? 3 : 1;  // Phi only, or Phi plus two derivs
    // the number of output coefficients is always a full set even if some of them are empty
    for(unsigned int q=0; q<numQuantitiesOutput; q++) {
        output[q]->resize(2*mmax+1);
        for(unsigned int i=0; i<indices.size(); i++) {  // only allocate those coefs that will be used
            output[q]->at(indices[i]+mmax)=math::Matrix<double>(sizeR, sizez, 0);
        }
    }

    PtrDensity densInterp;  // pointer to an internally created interpolating object if it is needed
    const BaseDensity* dens = &src;  // pointer to either the original density or the interpolated one
    // For an axisymmetric potential we don't use interpolation,
    // as the Fourier expansion of density trivially has only one harmonic;
    // also, if the input density is already a Fourier expansion, use it directly.
    // Otherwise, we need to create a temporary DensityAzimuthalHarmonic interpolating object.
    if(!isZRotSymmetric(src) && src.name() != DensityAzimuthalHarmonic::myName()) {
        double Rmax = gridR.back() * 2;
        double Rmin = gridR[1] * 0.01;
        double zmax = gridz.back() * 2;
        double zmin = gridz[0]==0 ? gridz[1] * 0.01 :
            gridz[sizez/2]==0 ? gridz[sizez/2+1] * 0.01 : Rmin;
        double delta=0.1;  // relative difference between grid nodes = log(x[n+1]/x[n])
        // create a density interpolator; it will be automatically deleted upon return
        densInterp = DensityAzimuthalHarmonic::create(src, mmax,
            static_cast<unsigned int>(log(Rmax/Rmin)/delta), Rmin, Rmax,
            static_cast<unsigned int>(log(zmax/zmin)/delta), zmin, zmax);
        // and use it for computing the potential
        dens = densInterp.get();
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
                AzimuthalHarmonicIntegrand fnc(*dens, m,
                    gridR[iR], gridz[iz], gridR[1], zmin, gridR.back(), gridz.back(), useDerivs);
                math::integrateNdim(fnc, boxmin, boxmax, 
                    EPSREL_POTENTIAL_INT, MAX_NUM_EVAL,
                    result, error, &numEval);
                if(isZReflSymmetric(*dens) && gridz[iz]==0)
                    result[2] = 0;
                if(gridR[iR]==0)
                    result[1] = 0;
                for(unsigned int q=0; q<numQuantitiesOutput; q++)
                    output[q]->at(m+mmax)(iR,iz) += result[q];
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            stop = true;
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error("Keyboard interrupt");
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
        math::trigMultiAngle(pc.phi, mmax, needSine, trig);
        for(unsigned int i=0; i<nind; i++) {
            int m = indices[i];
            harmonics[m+mmax][b] = particles.mass(b) *
                (m==0 ? 1 : m>0 ? 2*trig[m-1] : 2*trig[mmax-m-1]);
        }
    }
}

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
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {
        if(stop) continue;
        if(cbrk.triggered()) stop = true;
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
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
                    for(unsigned int q=0; q<numQuantitiesOutput; q++)
                        output[q]->at(m+mmax)(iR,iz) += values[q];
                }
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
            stop = true;
        }
    }
    if(cbrk.triggered())
        throw std::runtime_error("Keyboard interrupt");
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsFromParticles: "+errorMsg);
}

}  // internal namespace

// the driver functions that use the templated routines defined above

// density coefs from density
void computeDensityCoefsCyl(const BaseDensity& src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &output)
{
    std::vector< math::Matrix<double> > *coefs = &output;
    computeFourierCoefs<BaseDensity>(src, mmax, gridR, gridz, &coefs);
    // the value at R=0,z=0 might be undefined, in which case we take it from nearby points
    for(unsigned int iz=0; iz<gridz.size(); iz++)
        if(gridz[iz] == 0 && !isFinite(output[mmax](0, iz))) {
            double d1 = output[mmax](0, iz+1);  // value at R=0,z>0
            double d2 = output[mmax](1, iz);    // value at R>0,z=0
            for(unsigned int mm=0; mm<output.size(); mm++)
                if(output[mm].cols()>0)  // loop over all non-empty harmonics
                    output[mm](0, iz) = mm==mmax ? (d1+d2)/2 : 0;  // only m=0 survives
        }
}

// potential coefs from potential
void computePotentialCoefsCyl(const BasePotential &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    computeFourierCoefs<BasePotential>(src, mmax, gridR, gridz, coefs);
    // assign potential derivatives at R=0 or z=0 to zero, depending on the symmetry
    for(unsigned int iz=0; iz<gridz.size(); iz++) {
        if(gridz[iz] == 0 && isZReflSymmetric(src)) {
            for(unsigned int iR=0; iR<gridR.size(); iR++)
                for(unsigned int mm=0; mm<dPhidz.size(); mm++)
                    if(dPhidz[mm].cols()>0)  // loop over all non-empty harmonics
                        dPhidz[mm](iR, iz) = 0; // z-derivative is zero in the symmetry plane
        }
        for(unsigned int mm=0; mm<dPhidR.size(); mm++)
            if(dPhidR[mm].cols()>0)  // loop over all non-empty harmonics
                dPhidR[mm](0, iz) = 0;  // R-derivative at R=0 should always be zero
    }
}

// potential coefs from density, with derivatves
void computePotentialCoefsCyl(const BaseDensity &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    computePotentialCoefsFromDensity(src, mmax, gridR, gridz, true, coefs);
}

// potential coefs from density, without derivatves
void computePotentialCoefsCyl(const BaseDensity &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi)
{
    std::vector< math::Matrix<double> > *coefs = &Phi;
    computePotentialCoefsFromDensity(src, mmax, gridR, gridz, false, &coefs);
}

// potential coefs from N-body array, with derivatives
void computePotentialCoefsCyl(
    const particles::ParticleArray<coord::PosCyl>& particles,
    coord::SymmetryType sym,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz)
{
    if( gridR.size() < CYLSPLINE_MIN_GRID_SIZE ||
        gridz.size() < CYLSPLINE_MIN_GRID_SIZE ||
        (isZReflSymmetric(sym) && gridz[0] != 0) )
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, sym);
    std::vector<std::vector<double> > harmonics(2*mmax+1);
    std::vector<std::pair<double, double> > Rz;
    computeAzimuthalHarmonicsFromParticles(particles, indices, harmonics, Rz);
    std::vector< math::Matrix<double> >* output[] = {&Phi, &dPhidR, &dPhidz};
    computePotentialCoefsFromParticles(indices, harmonics, Rz, gridR, gridz, true, output);
}

// potential coefs from N-body array, without derivatives
void computePotentialCoefsCyl(
    const particles::ParticleArray<coord::PosCyl>& particles,
    coord::SymmetryType sym,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> > &Phi)
{
    if( gridR.size() < CYLSPLINE_MIN_GRID_SIZE ||
        gridz.size() < CYLSPLINE_MIN_GRID_SIZE ||
        (isZReflSymmetric(sym) && gridz[0] != 0) )
        throw std::invalid_argument("computePotentialCoefsCyl: invalid grid parameters");
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, sym);
    std::vector<std::vector<double> > harmonics(2*mmax+1);
    std::vector<std::pair<double, double> > Rz;
    computeAzimuthalHarmonicsFromParticles(particles, indices, harmonics, Rz);
    std::vector< math::Matrix<double> >* output = &Phi;
    computePotentialCoefsFromParticles(indices, harmonics, Rz, gridR, gridz, false, &output);
}

// -------- public classes: DensityAzimuthalHarmonic --------- //

PtrDensity DensityAzimuthalHarmonic::create(const BaseDensity& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax, bool accurateIntegration)
{
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("DensityAzimuthalHarmonic: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > coefs;
    // to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle:
    // the number of output harmonics remains the same, but the accuracy of approximation increases.
    int mmaxFourier = isZRotSymmetric(src) ? 0 :
        accurateIntegration ? std::max<int>(mmax, MMIN_AZIMUTHAL_FOURIER) : mmax;
    computeDensityCoefsCyl(src, mmaxFourier, gridR, gridz, coefs);
    if(mmaxFourier > (int)mmax) {
        // remove extra coefs: (mmaxFourier-mmax) from both heads and tails of arrays
        coefs.erase(coefs.begin() + mmaxFourier+mmax+1, coefs.end());
        coefs.erase(coefs.begin(), coefs.begin() + mmaxFourier-mmax);
    }
    return PtrDensity(new DensityAzimuthalHarmonic(gridR, gridz, coefs));
}

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
        mysym &= ~coord::ST_ZREFLECTION;
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
    for(int m=-mmax; m<=mmax; m++)
        if(spl[m+mmax]) {
            double trig_m = m==0 ? 1 : m>0 ? trig[m-1] : trig[mmax-1-m];
            result += spl[m+mmax]->value(lR, lz) * trig_m;
        }
    return result;
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

namespace {  // internal routines

// This routine constructs an spherical-harmonic expansion describing
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
        utils::msg(utils::VL_WARNING, "CylSpline",
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
                (isFinite(mass) ? "(total mass is infinite)" : "(cannot compute half-mass radius)"));
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
    utils::msg(utils::VL_DEBUG, "CylSpline",
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
    utils::msg(utils::VL_DEBUG, "CylSpline",
        "Grid in R=["+utils::toString(Rmin)+":"+utils::toString(Rmax)+
             "], z=["+utils::toString(zmin)+":"+utils::toString(zmax)+"]");
}

} // internal namespace

PtrPotential CylSpline::create(const BaseDensity& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax, bool useDerivs)
{
    // ensure the grid radii are set to some reasonable values
    chooseGridRadii(src, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("Error in CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    if(useDerivs)
        computePotentialCoefsCyl(src, mmax, gridR, gridz, /*output*/ Phi, dPhidR, dPhidz);
    else
        computePotentialCoefsCyl(src, mmax, gridR, gridz, /*output*/ Phi);
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

PtrPotential CylSpline::create(const BasePotential& src, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax,
    unsigned int gridSizez, double zmin, double zmax)
{
    chooseGridRadii(src, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("Error in CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(src))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    std::vector< math::Matrix<double> > *coefs[3] = {&Phi, &dPhidR, &dPhidz};
    // to improve accuracy of Fourier coefficient computation, we may increase
    // the order of expansion that determines the number of integration points in phi angle;
    int mmaxFourier = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, MMIN_AZIMUTHAL_FOURIER);
    computePotentialCoefsCyl(src, mmaxFourier, gridR, gridz, /*output*/ Phi, dPhidR, dPhidz);
    if(mmaxFourier > (int)mmax) {
        // remove extra coefs: (mmaxFourier-mmax) from both heads and tails of arrays
        for(int q=0; q<3; q++) {
            coefs[q]->erase(coefs[q]->begin() + mmaxFourier+mmax+1, coefs[q]->end());
            coefs[q]->erase(coefs[q]->begin(), coefs[q]->begin() + mmaxFourier-mmax);
        }
    }
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

PtrPotential CylSpline::create(
    const particles::ParticleArray<coord::PosCyl>& points,
    coord::SymmetryType sym, int mmax,
    unsigned int gridSizeR, double Rmin, double Rmax, 
    unsigned int gridSizez, double zmin, double zmax, bool useDerivs)
{
    chooseGridRadii(points, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax);
    if( gridSizeR<CYLSPLINE_MIN_GRID_SIZE || Rmin<=0 || Rmax<=Rmin ||
        gridSizez<CYLSPLINE_MIN_GRID_SIZE || zmin<=0 || zmax<=zmin)
        throw std::invalid_argument("Error in CylSpline: invalid grid parameters");
    std::vector<double> gridR = math::createNonuniformGrid(gridSizeR, Rmin, Rmax, true);
    std::vector<double> gridz = math::createNonuniformGrid(gridSizez, zmin, zmax, true);
    if(!isZReflSymmetric(sym))
        gridz = math::mirrorGrid(gridz);
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    if(useDerivs)
        computePotentialCoefsCyl(points, sym, mmax, gridR, gridz, /*output*/ Phi, dPhidR, dPhidz);
    else
        computePotentialCoefsCyl(points, sym, mmax, gridR, gridz, /*output*/ Phi);
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

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
        mysym&= ~coord::ST_ZREFLECTION;
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
                mysym &= ~coord::ST_YREFLECTION;
            if((m<0) ^ (m%2 != 0))
                mysym &= ~coord::ST_XREFLECTION;
            if(m%2 != 0)
                mysym &= ~coord::ST_REFLECTION;
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
