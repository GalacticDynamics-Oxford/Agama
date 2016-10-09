#include "potential_cylspline.h"
#include "potential_multipole.h"
#include "math_core.h"
#include "math_fit.h"
#include "math_spline.h"
#include "math_specfunc.h"
#include "math_sphharm.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace potential {

// internal definitions
namespace{

/// minimum number of grid nodes
const unsigned int CYLSPLINE_MIN_GRID_SIZE = 2;

/// minimum number of terms in Fourier expansion used to compute coefficients
/// of a non-axisymmetric density or potential model (it may be larger than
/// the requested number of output terms, to improve the accuracy of integration)
const unsigned int MMIN_AZIMUTHAL_FOURIER = 16;

/// maximum allowed order of azimuthal Fourier expansion
const unsigned int MMAX_AZIMUTHAL_FOURIER = 64;

/// lower cutoff in radius for Legendre Q function
const double MIN_R = 1e-10;

/// max number of function evaluations in multidimensional integration
const unsigned int MAX_NUM_EVAL = 10000;

/// relative accuracy of potential computation (integration tolerance parameter)
const double EPSREL_POTENTIAL_INT = 1e-6;

// ------- Fourier expansion of density or potential ------- //
// The routine 'computeFourierCoefs' can work with both density and potential classes,
// computes the azimuthal Fourier expansion for either density (in the first case),
// or potential and its R- and z-derivatives (in the second case).
// To avoid code duplication, the function that actually retrieves the relevant quantity
// is separated into a dedicated routine 'storeValue', which stores either one or three
// values for each input point. The 'computeFourierCoefs' routine is templated on both
// the type of input data and the number of quantities stored for each point.

template<class BaseDensityOrPotential>
static void storeValue(const BaseDensityOrPotential& src, const coord::PosCyl& pos, double values[]);

template<>
inline void storeValue(const BaseDensity& src, const coord::PosCyl& pos, double values[]) {
    *values = src.density(pos);
}

template<>
inline void storeValue(const BasePotential& src, const coord::PosCyl& pos, double values[]) {
    coord::GradCyl grad;
    src.eval(pos, values, &grad);
    values[(2*MMAX_AZIMUTHAL_FOURIER+1)]   = grad.dR;
    values[(2*MMAX_AZIMUTHAL_FOURIER+1)*2] = grad.dz;
}

template<class BaseDensityOrPotential, int NQuantities>
static void computeFourierCoefs(const BaseDensityOrPotential &src,
    const unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    std::vector< math::Matrix<double> >* coefs[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE)
        throw std::invalid_argument("computeFourierCoefs: incorrect grid size");
    if(mmax > MMAX_AZIMUTHAL_FOURIER)
        throw std::invalid_argument("computeFourierCoefs: mmax is too large");
    if(!isZReflSymmetric(src) && gridz[0]==0)
        throw std::invalid_argument("computeFourierCoefs: input density is not symmetric "
            "under z-reflection, the grid in z must cover both positive and negative z");
    int mmin = isYReflSymmetric(src) ? 0 : -static_cast<int>(mmax);
    bool useSine = mmin<0;
    math::FourierTransformForward trans(mmax, useSine);
    std::vector<int> indices = math::getIndicesAzimuthal(mmax, src.symmetry());
    unsigned int numHarmonicsComputed = indices.size();
    int numPoints = sizeR * sizez;
    for(int q=0; q<NQuantities; q++) {
        coefs[q]->resize(mmax*2+1);
        for(unsigned int i=0; i<numHarmonicsComputed; i++)
            coefs[q]->at(indices[i]+mmax)=math::Matrix<double>(sizeR, sizez);
    }
    std::string errorMsg;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int n=0; n<numPoints; n++) {
        int iR = n % sizeR;  // index in radial grid
        int iz = n / sizeR;  // index in vertical direction
        double values [(2*MMAX_AZIMUTHAL_FOURIER+1) * NQuantities];
        double coefs_m[(2*MMAX_AZIMUTHAL_FOURIER+1)];
        try{
            for(unsigned int i=0; i<trans.size(); i++)
                storeValue<BaseDensityOrPotential>(src,
                    coord::PosCyl(gridR[iR], gridz[iz], trans.phi(i)), values+i);
            for(int q=0; q<NQuantities; q++) {
                trans.transform(&values[q*(2*MMAX_AZIMUTHAL_FOURIER+1)], coefs_m);
                for(unsigned int i=0; i<numHarmonicsComputed; i++) {
                    int m = indices[i];
                    coefs[q]->at(m+mmax)(iR, iz) =
                        iR==0 && m!=0 ? 0 :   // at R=0, all non-axisymmetric harmonics must vanish
                        coefs_m[useSine ? m+mmax : m] / (m==0 ? 2*M_PI : M_PI);
                }
            }
        }
        catch(std::exception& e) {
            errorMsg = e.what();
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computeFourierCoefs: "+errorMsg);
}

// ------- Computation of potential from density ------- //
// The routines below solve the Poisson equation by computing the Fourier harmonics
// of potential via direct 2d integration over (R,z) plane. 
// If the input density is axisymmetric, then the value of density at phi=0 is taken,
// otherwise the density must first be Fourier-transformed itself and represented
// as an instance of DensityAzimuthalHarmonic class, which provides the member function
// returning the value of m-th harmonic at the given (R,z).

inline double density_rho_m(const BaseDensity& dens, int m, double R, double z) {
    if(dens.name() == DensityAzimuthalHarmonic::myName())  // quickly compare char* pointers, not strings
        return static_cast<const DensityAzimuthalHarmonic&>(dens).rho_m(m, R, z);
    return m==0 ? dens.density(coord::PosCyl(R, z, 0)) : 0;
}

// Routine that computes the contribution tothe  m-th harmonic potential at location (R0,z0)
// from the point at (R,z) with given 'mass' (or, rather, mass times trig(m phi)
// in the discrete case, or density times jacobian times trig(m phi) in the continuous case).
// This routine is used both in AzimuthalHarmonicIntegrand to compute the potential from
// a continuous density distribution, and in ComputePotentialCoefsFromPoints to obtain
// the potential from a discrete point mass collection.
static void computePotentialHarmonicAtPoint(int m, double R, double z, double R0, double z0,
    double mass, bool useDerivs, double values[])
{
    // the contribution to the potential is given by
    // rho * \int_0^\infty dk J_m(k R) J_m(k R0) exp(-k|z-z0|)
    double t = R*R + R0*R0 + pow_2(z0-z);
    if(R > MIN_R && R0 > MIN_R) {  // normal case
        double sq = 1 / (M_PI * sqrt(R*R0));
        double u  = t / (2*R*R0);
        if(u < 1+MIN_R)
            return;  // close to singularity
        double dQ;
        double Q  = math::legendreQ(m-0.5, u, useDerivs ? &dQ : NULL);
        values[0]+= -sq * mass * Q;
        if(useDerivs) {
            values[1]+= -sq * mass * (dQ/R - (Q/2 + u*dQ)/R0);
            values[2]+= -sq * mass * dQ * (z0-z) / (R*R0);
        }
    } else      // degenerate case
    if(m==0) {  // here only m=0 harmonic survives
        if(t < 1e-15)
            return;    // close to singularity
        double s  = 1/sqrt(t);
        values[0]+= -mass * s;
        if(useDerivs)
            values[2]+=  mass * s * (z0-z) / t;
    }
}

// the N-dimensional integrand for computing the potential harmonics from density
class AzimuthalHarmonicIntegrand: public math::IFunctionNdim {
public:
    AzimuthalHarmonicIntegrand(const BaseDensity& _dens, int _m,
        double _R0, double _z0, bool _useDerivs) :
        dens(_dens), m(_m), R0(_R0), z0(_z0), useDerivs(_useDerivs) {}
    // evaluate the function at a given (R,z) point (scaled)
    virtual void eval(const double pos[], double values[]) const
    {
        for(unsigned int c=0; c<numValues(); c++)
            values[c] = 0;
        // unscale input coordinates
        const double s = pos[0];
        const double r = exp( 1/(1-s) - 1/s );
        if(r<1e-100 || r>1e100)
            return;  // scaled coords point at 0 or infinity
        const double th= pos[1] * M_PI/2;
        const double R = r*cos(th);
        const double z = r*sin(th);
        const double jac = pow_2(M_PI*r) * R * (1/pow_2(1-s) + 1/pow_2(s));
        
        // get the values of density at (R,z) and (R,-z):
        // here the density evaluation may be a computational bottleneck,
        // so in the typical case of z-reflection symmetry we save on using
        // the same value of density for both positive and negative z1.
        double rho = jac * density_rho_m(dens, m, R, z);
        computePotentialHarmonicAtPoint(m, R, z, R0, z0, rho, useDerivs, values);
        if(!isZReflSymmetric(dens))
               rho = jac * density_rho_m(dens, m, R,-z);
        computePotentialHarmonicAtPoint(m, R,-z, R0, z0, rho, useDerivs, values);

#ifdef HAVE_CUBA
        // workaround for CUBA n-dimensional quadrature routine: it seems to be unable 
        // to properly handle cases when one of components of the integrand is identically zero,
        // that's why we output 1 instead, and zero it out later
        if(R0==0)
            values[1] = 1;
        if(isZReflSymmetric(dens) && z0==0)
            values[2] = 1;
#endif
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return useDerivs ? 3 : 1; }
private:
    const BaseDensity& dens;
    const int m;
    // the point at which the integral is computed, also defines the toroidal coordinate system
    const double R0, z0;
    const bool useDerivs;
};

static void computePotentialCoefsFromDensity(const BaseDensity &src,
    unsigned int mmax,
    const std::vector<double> &gridR,
    const std::vector<double> &gridz,
    bool useDerivs,
    std::vector< math::Matrix<double> >* output[])
{
    unsigned int sizeR = gridR.size(), sizez = gridz.size();
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE ||
        mmax<0 || mmax > MMAX_AZIMUTHAL_FOURIER)
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
        double Rmax = gridR.back() * 100;
        double Rmin = gridR[1] * 0.01;
        double zmax = gridz.back() * 100;
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

    int numPoints = sizeR * sizez;
    std::string errorMsg;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {  // combined index variable
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        try{
            // integration box in scaled coords - r range is slightly smaller than 0:1
            // due to exponential scaling (rscaled=0.045 corresponds to r<1e-9)
            double Rzmin[2]={0.045,0.}, Rzmax[2]={0.955,1.};
            double result[3], error[3];
            int numEval;
            for(unsigned int i=0; i<indices.size(); i++) {
                int m = indices[i];
                AzimuthalHarmonicIntegrand fnc(*dens, m, gridR[iR], gridz[iz], useDerivs);
                math::integrateNdim(fnc, Rzmin, Rzmax, 
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
        }
    }
    if(!errorMsg.empty())
        throw std::runtime_error("Error in computePotentialCoefsCyl: "+errorMsg);
}

// transform an N-body snapshot to an array of Fourier harmonic coefficients
static void computeAzimuthalHarmonicsFromParticles(
    const particles::ParticleArray<coord::PosCyl>& particles,
    const std::vector<int>& indices,
    std::vector<std::vector<double> >& harmonics,
    std::vector<std::pair<double, double> > &Rz)
{
    assert(harmonics.size()>0 && indices.size()>0);
    unsigned int nbody = particles.size();
    unsigned int nind  = indices.size();
    int mmax = (harmonics.size()-1)/2;
    bool needSine = false;
    for(unsigned int i=0; i<nind; i++) {
        needSine |= indices[i]<0;
        harmonics[indices[i]+mmax].resize(nbody);
    }
    Rz.resize(nbody);
    std::vector<double> tmpharm(2*mmax);
    for(unsigned int b=0; b<nbody; b++) {
        const coord::PosCyl& pc = particles.point(b);
        Rz[b].first = pc.R;
        Rz[b].second= pc.z;
        math::trigMultiAngle(pc.phi, mmax, needSine, &tmpharm.front());
        for(unsigned int i=0; i<nind; i++) {
            int m = indices[i];
            harmonics[m+mmax][b] = particles.mass(b) *
                (m==0 ? 1 : m>0 ? 2*tmpharm[m-1] : 2*tmpharm[mmax-m-1]);
        }
    }
}

static void computePotentialCoefsFromParticles(
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
    int nbody = Rz.size();
    int numPoints = sizeR * sizez;
    std::string errorMsg;
    // parallelize the loop over the nodes of 2d grid, not the inner loop over particles
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for(int ind=0; ind<numPoints; ind++) {
        unsigned int iR = ind % sizeR;
        unsigned int iz = ind / sizeR;
        try{
            for(int b=0; b<nbody; b++) {
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
        }
    }
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
    computeFourierCoefs<BaseDensity, 1>(src, mmax, gridR, gridz, &coefs);
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
    computeFourierCoefs<BasePotential, 3>(src, mmax, gridR, gridz, coefs);
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
        mmax<0 || mmax > MMAX_AZIMUTHAL_FOURIER ||
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
        mmax<0 || mmax > MMAX_AZIMUTHAL_FOURIER ||
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
    unsigned int gridSizez, double zmin, double zmax)
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
    int mmaxFourier = isZRotSymmetric(src) ? 0 : std::max<int>(mmax, MMIN_AZIMUTHAL_FOURIER);
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
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE || 
        coefs.size()%2 == 0 || coefs.size() > 2*MMAX_AZIMUTHAL_FOURIER+1)
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
        gridR[iR] = log(1+gridR[iR]/Rscale);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = log(1+fabs(gridz[iz])/Rscale)*math::sign(gridz[iz]);

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
        if(sum>0) {
            spl[mm] = math::CubicSpline2d(gridR, gridz, val);
            int m = mm-mmax;
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
    double lR = log(1+R/Rscale), lz = log(1+fabs(z)/Rscale)*math::sign(z);
    if( math::abs(m)>mmax || spl[m+mmax].isEmpty() ||
        lR<spl[mmax].xmin() || lz<spl[mmax].ymin() ||
        lR>spl[mmax].xmax() || lz>spl[mmax].ymax() )
        return 0;
    return spl[m+mmax].value(lR, lz);
}

double DensityAzimuthalHarmonic::densityCyl(const coord::PosCyl &pos) const
{
    int mmax = (spl.size()-1)/2;
    double lR = log(1+pos.R/Rscale), lz = log(1+fabs(pos.z)/Rscale)*math::sign(pos.z);
    if( lR<spl[mmax].xmin() || lz<spl[mmax].ymin() ||
        lR>spl[mmax].xmax() || lz>spl[mmax].ymax() )
        return 0;
    double result = 0;
    double trig[2*MMAX_AZIMUTHAL_FOURIER];
    if(!isZRotSymmetric(sym)) {
        bool needSine = !isYReflSymmetric(sym);
        math::trigMultiAngle(pos.phi, mmax, needSine, trig);
    }
    for(int m=-mmax; m<=mmax; m++)
        if(!spl[m+mmax].isEmpty()) {
            double trig_m = m==0 ? 1 : m>0 ? trig[m-1] : trig[mmax-1-m];
            result += spl[m+mmax].value(lR, lz) * trig_m;
        }
    return result;
}

void DensityAzimuthalHarmonic::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz, 
    std::vector< math::Matrix<double> > &coefs) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(mmax>=0 && spl.size() == mmax*2+1 && !spl[mmax].isEmpty());
    coefs.resize(2*mmax+1);
    unsigned int sizeR = spl[mmax].xvalues().size();
    unsigned int sizez = spl[mmax].yvalues().size();
    gridR = spl[mmax].xvalues();
    if(isZReflSymmetric(sym)) {
        // output only coefs for half-space z>=0
        sizez = (sizez+1) / 2;
        gridz.assign(spl[mmax].yvalues().begin() + sizez-1, spl[mmax].yvalues().end());
    } else
        gridz = spl[mmax].yvalues();
    for(unsigned int mm=0; mm<=2*mmax; mm++)
        if(!spl[mm].isEmpty()) {
            coefs[mm]=math::Matrix<double>(sizeR, sizez);
            for(unsigned int iR=0; iR<sizeR; iR++)
                for(unsigned int iz=0; iz<sizez; iz++)
                    coefs[mm](iR, iz) = spl[mm].value(gridR[iR], gridz[iz]);
            //math::eliminateNearZeros(coefs[mm]);
        }
    // unscale the grid coordinates
    for(unsigned int iR=0; iR<sizeR; iR++)
        gridR[iR] = Rscale * (exp(gridR[iR]) - 1);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = Rscale * (exp(fabs(gridz[iz])) - 1) * math::sign(gridz[iz]);
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
static PtrPotential determineAsympt(
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
    int mmax = (Phi.size()-1)/2;  // # of angular(phi) harmonics in the original potential
    int lmax_fit = 8;             // # of meridional harmonics to fit - don't set too large
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
                        Plm[l-absm] * math::powInt(r/r0, -l-1) *
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
    }
    math::eliminateNearZeros(W);
    return PtrPotential(new PowerLawMultipole(r0, false /*not inner*/, zeros, zeros, W));
}

// Automatically choose reasonable grid extent if it was not provided
static void chooseGridRadii(const BaseDensity& src,
    unsigned int gridSizeR, double &Rmin, double &Rmax, 
    unsigned int gridSizez, double &zmin, double &zmax)
{
    // if the grid min/max radii is not provided, try to determine automatically
    if(Rmax==0 || Rmin==0) {
        double rhalf = getRadiusByMass(src, 0.5 * src.totalMass());
        if(!isFinite(rhalf))
            throw std::invalid_argument("CylSpline: failed to automatically determine grid extent");
        double spacing = 1 + sqrt(10./sqrt(gridSizeR*gridSizez));  // ratio between consecutive grid nodes
        if(Rmax==0)
            Rmax = rhalf * pow(spacing,  0.5*gridSizeR);
       	if(Rmin==0)
            Rmin = rhalf * pow(spacing, -0.5*gridSizeR);
    }
    if(zmax==0)
        zmax=Rmax;
    if(zmin==0)
        zmin=Rmin;
    utils::msg(utils::VL_DEBUG, "CylSpline",
        "Grid in R=["+utils::toString(Rmin)+":"+utils::toString(Rmax)+
             "], z=["+utils::toString(zmin)+":"+utils::toString(zmax)+"]");
}

static void chooseGridRadii(const particles::ParticleArray<coord::PosCyl>& points,
    unsigned int gridSizeR, double &Rmin, double &Rmax, 
    unsigned int gridSizez, double &zmin, double &zmax)
{
    if(Rmin!=0 && Rmax!=0 && zmin!=0 && zmax!=0)
        return;
    unsigned int Npoints = points.size();
    std::vector<double> radii(Npoints);
    for(unsigned int i=0; i<Npoints; i++)
        radii[i] = sqrt(pow_2(points.point(i).R) + pow_2(points.point(i).z));
    std::nth_element(radii.begin(), radii.begin() + Npoints/2, radii.end());
    double gridSize = sqrt(gridSizeR*gridSizez);  // average of the two sizes
    double Rhalf = radii[Npoints/2];   // half-mass radius (if all particles have equal mass)
    double spacing = 1 + sqrt(10./gridSize);
    int Nmin = static_cast<int>(log(Npoints+1)/log(2));  // # of points in the inner cell
    if(Rmin==0) {
        std::nth_element(radii.begin(), radii.begin() + Nmin, radii.end());
        Rmin = std::max(radii[Nmin], Rhalf * pow(spacing, -0.5*gridSize));
    }
    if(Rmax==0) {
        std::nth_element(radii.begin(), radii.end() - Nmin, radii.end());
        Rmax = std::min(radii[Npoints-Nmin], Rhalf * pow(spacing, 0.5*gridSize));
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
        computePotentialCoefsCyl(src, mmax, gridR, gridz, Phi, dPhidR, dPhidz);
    else
        computePotentialCoefsCyl(src, mmax, gridR, gridz, Phi);
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
    computeFourierCoefs<BasePotential, 3>(src, mmaxFourier, gridR, gridz, coefs);
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
        computePotentialCoefsCyl(points, sym, mmax, gridR, gridz, Phi, dPhidR, dPhidz);
    else
        computePotentialCoefsCyl(points, sym, mmax, gridR, gridz, Phi);
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
    bool haveDerivs = dPhidR.size() > 0 && dPhidz.size() > 0;
    if(sizeR<CYLSPLINE_MIN_GRID_SIZE || sizez<CYLSPLINE_MIN_GRID_SIZE || gridR_orig[0]!=0 ||
        Phi.size()%2 == 0 || Phi.size() > 2*MMAX_AZIMUTHAL_FOURIER+1 ||
        (haveDerivs && (Phi.size() != dPhidR.size() || Phi.size() != dPhidz.size())) )
        throw std::invalid_argument("CylSpline: incorrect grid size");
    int mmax  = (Phi.size()-1)/2;
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
        Rscale  = 1.;  // rather arbitrary

    // transform the grid to log-scaled coordinates
    for(unsigned int i=0; i<sizeR; i++) {
        gridR[i] = log(1+gridR[i]/Rscale);
    }
    for(unsigned int i=0; i<sizez; i++) {
        gridz[i] = log(1+fabs(gridz[i])/Rscale) * math::sign(gridz[i]);
    }

    math::Matrix<double> val(sizeR, sizez), derR(sizeR, sizez), derz(sizeR, sizez);
    spl.resize(2*mmax+1);
    for(int mm=0; mm<=2*mmax; mm++) {
        if(Phi[mm].rows() == 0 && Phi[mm].cols() == 0)
            continue;
        if((   Phi[mm].rows() != sizeR ||    Phi[mm].cols() != sizez_orig) || (haveDerivs && 
           (dPhidR[mm].rows() != sizeR || dPhidR[mm].cols() != sizez_orig  ||
            dPhidz[mm].rows() != sizeR || dPhidz[mm].cols() != sizez_orig)))
            throw std::invalid_argument("CylSpline: incorrect coefs array size");
        double sum=0;
        for(unsigned int iR=0; iR<sizeR; iR++) {
            double R = gridR_orig[iR];
            for(unsigned int iz=0; iz<sizez_orig; iz++) {
                double z = gridz_orig[iz];
                unsigned int iz1 = zsym ? sizez_orig-1+iz : iz;  // index in the internal 2d grid
                // values of potential and its derivatives are represented as scaled 2d functions:
                // the amplitude is scaled by 'amp', while both coordinates are log-scaled.
                // thus the values passed to the constructor of 2d spline must be properly modified,
                // and the derivatives additionally transformed to the scaled coordinates.
                double amp = sqrt(pow_2(Rscale)+pow_2(R)+pow_2(z));
                val (iR, iz1) =  Phi[mm](iR,iz) * amp;
                if(haveDerivs) {
                    derR(iR, iz1) = (dPhidR[mm](iR,iz) * amp + Phi[mm](iR,iz) * R / amp) * (R+Rscale);
                    derz(iR, iz1) = (dPhidz[mm](iR,iz) * amp + Phi[mm](iR,iz) * z / amp) * (fabs(z)+Rscale);
                }
                if(zsym) {
                    assert(z>=0);  // source data only covers upper half-space
                    unsigned int iz2 = sizez_orig-1-iz;  // index of the reflected cell
                    val (iR, iz2) = val (iR, iz1);
                    derR(iR, iz2) = derR(iR, iz1);
                    derz(iR, iz2) =-derz(iR, iz1);
                }
                sum += fabs(Phi[mm](iR, iz));
            }
        }
        if(sum>0) {
            spl[mm] = haveDerivs ? 
                math::PtrInterpolator2d(new math::QuinticSpline2d(gridR, gridz, val, derR, derz)) :
                math::PtrInterpolator2d(new math::CubicSpline2d(gridR, gridz, val, 0, NAN, NAN, NAN));
            // check if this non-trivial harmonic breaks any symmetry
            int m = mm-mmax;
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
    double* val, coord::GradCyl* der, coord::HessCyl* der2) const
{
    int mmax = (spl.size()-1)/2;
    double Rscaled = log(1+pos.R/Rscale);
    double zscaled = log(1+fabs(pos.z)/Rscale) * math::sign(pos.z);
    if( Rscaled<spl[mmax]->xmin() || zscaled<spl[mmax]->ymin() ||
        Rscaled>spl[mmax]->xmax() || zscaled>spl[mmax]->ymax() ) {
        // outside the grid definition region, use the asymptotic expansion
        asymptOuter->eval(pos, val, der, der2);
        return;
    }

    // only compute those quantities that will be needed in output
    bool needPhi  = true;
    bool needGrad = der !=NULL || der2!=NULL;
    bool needHess = der2!=NULL;
    double trig_arr[2*MMAX_AZIMUTHAL_FOURIER];
    if(!isZRotSymmetric(sym)) {
        bool needSine = needGrad || !isYReflSymmetric(sym);
        math::trigMultiAngle(pos.phi, mmax, needSine, trig_arr);
    }

    // total scaled potential, gradient and hessian in scaled coordinates
    double Phi = 0;
    coord::GradCyl grad;
    coord::HessCyl hess;
    grad.dR  = grad.dz = grad.dphi = 0;
    hess.dR2 = hess.dz2 = hess.dphi2 = hess.dRdz = hess.dRdphi = hess.dzdphi = 0;

    // loop over azimuthal harmonics and compute the temporary (scaled) values
    for(int mm=0; mm<=2*mmax; mm++) {
        if(!spl[mm])  // empty harmonic
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
        int m = mm-mmax;
        double trig  = m==0 ? 1. : m>0 ? trig_arr[m-1] : trig_arr[mmax-1-m];  // cos or sin
        double dtrig = m==0 ? 0. : m>0 ? -m*trig_arr[mmax+m-1] : -m*trig_arr[-m-1];
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

    // unscale both amplitude of all quantities and their coordinate derivatives
    double r2 = pow_2(pos.R) + pow_2(pos.z);
    double S  = 1 / sqrt(pow_2(Rscale)+r2);  // scaling of the amplitude
    if(val)
        *val = S * Phi;
    if(!needGrad)
        return;
    double dSdr_over_r = -S*S*S;    
    double dRscaleddR = 1/(Rscale+pos.R);
    double dzscaleddz = 1/(Rscale+fabs(pos.z));
    if(der) {
        der->dR   = S * grad.dR * dRscaleddR + dSdr_over_r * Phi * pos.R;
        der->dz   = S * grad.dz * dzscaleddz + dSdr_over_r * Phi * pos.z;
        der->dphi = S * grad.dphi;
    }
    if(der2) {
        double d2RscaleddR2 = -pow_2(dRscaleddR);
        double d2zscaleddz2 = -pow_2(dzscaleddz) * math::sign(pos.z);
        double d2Sdr2 = (pow_2(Rscale) - 2 * r2) * dSdr_over_r * S * S;
        r2 += 1e-100;  // prevent 0/0 indeterminacy if r2==0
        der2->dR2 =
            (d2Sdr2 * pow_2(pos.R) + dSdr_over_r * pow_2(pos.z)) / r2 * Phi + 
            dSdr_over_r * 2 * pos.R * dRscaleddR * grad.dR +
            S * (hess.dR2 * pow_2(dRscaleddR) + grad.dR * d2RscaleddR2);
        der2->dz2 =
            (d2Sdr2 * pow_2(pos.z) + dSdr_over_r * pow_2(pos.R)) / r2 * Phi + 
            dSdr_over_r * 2 * pos.z * dzscaleddz * grad.dz +
            S * (hess.dz2 * pow_2(dzscaleddz) + grad.dz * d2zscaleddz2);
        der2->dRdz =
            (d2Sdr2 - dSdr_over_r) * pos.R * pos.z / r2 * Phi +
            dSdr_over_r * (pos.z * dRscaleddR * grad.dR + pos.R * dzscaleddz * grad.dz) +
            S * hess.dRdz * dRscaleddR * dzscaleddz;
        der2->dRdphi =
            dSdr_over_r * pos.R * grad.dphi +
            S * dRscaleddR * hess.dRdphi;
        der2->dzdphi =
            dSdr_over_r * pos.z * grad.dphi +
            S * dzscaleddz * hess.dzdphi;
        der2->dphi2 = S * hess.dphi2;
    }
}

void CylSpline::getCoefs(
    std::vector<double> &gridR, std::vector<double> &gridz, 
    std::vector< math::Matrix<double> > &Phi,
    std::vector< math::Matrix<double> > &dPhidR,
    std::vector< math::Matrix<double> > &dPhidz) const
{
    unsigned int mmax = (spl.size()-1)/2;
    assert(mmax>=0 && spl.size() == mmax*2+1 && !spl[mmax]->isEmpty());
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
        gridR[iR] = Rscale * (exp(scaledR[iR]) - 1);
    gridz.resize(sizez);
    for(unsigned int iz=0; iz<sizez; iz++)
        gridz[iz] = Rscale * (exp(fabs(scaledz[iz+iz0])) - 1) * math::sign(scaledz[iz+iz0]);

    Phi.resize(2*mmax+1);
    dPhidR.resize(2*mmax+1);
    dPhidz.resize(2*mmax+1);
    for(unsigned int mm=0; mm<=2*mmax; mm++) {
        if(!spl[mm])
            continue;
        Phi   [mm]=math::Matrix<double>(sizeR, sizez);
        dPhidR[mm]=math::Matrix<double>(sizeR, sizez);
        dPhidz[mm]=math::Matrix<double>(sizeR, sizez);
        for(unsigned int iR=0; iR<sizeR; iR++)
            for(unsigned int iz=0; iz<sizez; iz++) {
                double Rscaled = scaledR[iR];     // coordinates in the internal scaled coords array
                double zscaled = scaledz[iz+iz0];
                // scaling of the amplitude
                double S = 1 / sqrt(pow_2(Rscale) + pow_2(gridR[iR]) + pow_2(gridz[iz]));
                double dSdr_over_r = -S*S*S;
                // scaling of derivatives
                double dRscaleddR  = 1 / (Rscale + gridR[iR]);
                double dzscaleddz  = 1 / (Rscale + fabs(gridz[iz]));
                double val, dR, dz;
                spl[mm]->evalDeriv(Rscaled, zscaled, &val, &dR, &dz);
                Phi   [mm](iR,iz) = S * val;
                dPhidR[mm](iR,iz) = S * dR * dRscaleddR + dSdr_over_r * val * gridR[iR];
                dPhidz[mm](iR,iz) = S * dz * dzscaleddz + dSdr_over_r * val * gridz[iz];
            }
    }
}

}; // namespace
