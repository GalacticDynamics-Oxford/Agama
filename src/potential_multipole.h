/** \file    potential_multipole.h
    \brief   density and potential approximations based on spherical-harmonic expansion
    \author  Eugene Vasiliev
    \date    2010-2016

    This module provides representation of arbitrary density and potential profiles
    in terms of spherical-harmonic (or multipole) expansion, with coefficients being
    interpolated in radius.

    Mathematical background for spherical-harmonic transformation is provided in
    math_sphharm.h; in brief, the transformation is defined by two order parameters --
    lmax is the maximum index of Legendre polynomials in cos(theta), and mmax<=lmax is
    the maximum index of Fourier harmonics in phi. The choice of terms to be used in
    the expansion depends on symmetry properties of the model; for instance,
    in the triaxial case only terms with even and non-negative l,m are involved,
    and in the axisymmetric case only m=0 terms are non-zero.

    The density approximation uses cubic splines in log(r) for each sph.-harm. term,
    with power-law extrapolation to small and large radii.

    The potential approximation (Multipole) uses quintic splines in log(r), defined
    by their values and derivatives at grid nodes. Depending on the order of expansion,
    it employs either 1d splines for each term, or 2d splines in the (r,theta) plane
    for each azimuthal (m) Fourier harmonic, whichever is more efficient.
    In the second case, it is similar to CylSpline expansion, but the latter uses
    2d splines in scaled (R,z) coordinates for each m.
    Additionally, scaling transformations of coordinates and amplitudes are used
    to improve the accuracy of interpolation.
    The approximation is fairly accurate even with a rather sparse grid spacing:
    for instance, a grid with 20 nodes may cover a radial range spanning 6 orders
    of magnitude and still provide a good accuracy.
    Extrapolation of potential to small and large r (beyond the extent of the grid)
    is based on asymptotic power-law scaling of multipole coefficients, thus the density
    profile is generally well approximated even outside the grid.
    This asymptotic power-law extrapolation is also used in CylSpline.

    The potential expansion may be computed either from an existing potential
    (providing a computationally efficient approximation to a possibly expensive
    potential), from a density profile (thus solving the Poisson equation in spherical
    harmonics), or from an array of point masses (again solving the Poisson equation).
    Once constructed, the coefficients of expansion can be stored to and subsequently
    loaded from a text file, using routines `readPotential/writePotential` in
    potential_factory.h
*/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_spline.h"
#include "math_sphharm.h"
#include "smart.h"

namespace potential {


/** Spherical-harmonic expansion of density with coefficients being spline functions of radius */
class DensitySphericalHarmonic: public BaseDensity {
public:
    /** construct the object from the provided density model and grid parameters */
    static PtrDensity create(const BaseDensity& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin, double rmax);

    /** construct the density interpolator from an N-body snapshot.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
    */
    static PtrDensity create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** construct the object from stored coefficients */
    DensitySphericalHarmonic(const std::vector<double> &gridRadii,
        const std::vector< std::vector<double> > &coefs, double innerSlope=NAN, double outerSlope=NAN);

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "DensitySphericalHarmonic"; return text; }

    /** return the radii of spline nodes and the array of density expansion coefficients,
        together with power-law indices of extrapolation to small and large radii */
    void getCoefs(std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray,
        double &innerSlope, double &outerSlope) const;

private:
    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// radial dependence of each sph.-harm. expansion term
    std::vector<math::CubicSpline> spl;

    /// logarithmic density slope at small and large radii (rho ~ r^s)
    double innerSlope, outerSlope;

    /// whether the l=0 term is interpolated using log-scaling
    bool logScaling;

    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }

    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }

    virtual double densityCyl(const coord::PosCyl &pos) const;  // this is the implementation

};  // class DensitySphericalHarmonic


/** Auxiliary potential class that represents an array of multipole terms,
    with each term being a sum of two power-law profiles.
    It is internally used by spherical-harmonic and azimuthal Fourier expansion potential classes,
    as an extrapolation to small or large radii beyond the definition region of the main potential.
    Each term with index {l,m} is given by
    \f$  \Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}}            + W_{l,m} * (r/r0)^v  \f$  if s!=v,
    \f$  \Phi_{l,m}(r) = U_{l,m} * (r/r0)^{s_{l,m}} * ln(r/r0) + W_{l,m} * (r/r0)^v  \f$  if s==v.
    Here v=l for the inward extrapolation and v=-1-l for the outward extrapolation,
    so that the term W r^v represents the 'main' multipole component corresponding 
    to the Laplace equation, i.e. with zero density. The other term U r^s corresponds
    to a power-law density profile of the given harmonic component (rho ~ r^{s-2}). 
*/
class PowerLawMultipole: public BasePotentialCyl {
public:
    /** Create the potential from the three arrays: amplitudes of harmonic coefficients (U, W)
        and the power-law slope of the coefficient U with nonzero Laplacian (S),
        the reference radius r0 and the flag choosing between inward and outward extrapolation */
    PowerLawMultipole(double r0, bool inner,
        const std::vector<double>& S,
        const std::vector<double>& U,
        const std::vector<double>& W);
    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "PowerLaw"; return text; }
private:
    const math::SphHarmIndices ind; ///< indexing scheme for sph.-harm.coefficients
    double r0sq;                    ///< reference radius, squared
    bool inner;                     ///< whether this is an inward or outward extrapolation
    std::vector<double> S, U, W;    ///< sph.-harm.coefficients for extrapolation
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};


/// Multipole expansion for potentials
class Multipole: public BasePotentialCyl{
public:
    /** create the potential from the analytic density or potential model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        It exists in two variants: the first one takes a density model as input
        and solves Poisson equation to find the potential sph.-harm. coefficients;
        the second one takes a potential model and computes these coefs directly.
        \param[in]  src        is the input density or potential model;
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect
        (only works for models with finite mass).
    */
    static PtrPotential create(const BaseDensity& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0.);

    /** same as above, but takes a potential model as an input */
    static PtrPotential create(const BasePotential& src, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0.);

    /** create the potential from an N-body snapshot.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
    */
    static PtrPotential create(
        const particles::ParticleArray<coord::PosCyl> &points,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** construct the potential from the set of spherical-harmonic coefficients.
        \param[in]  radii  is the grid in radius;
        \param[in]  Phi  is the matrix of harmonic coefficients for the potential;
                    its first dimension is equal to the number of radial grid points,
                    and the second gives the number of coefficients (lmax+1)^2;
        \param[in]  dPhi  is the matrix of radial derivatives of harmonic coefs
                    (same size as Phi, each element is  d Phi_{l,m}(r) / dr ).
    */
    Multipole(const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);

    /** return the array of spherical-harmonic expansion coefficients.
        \param[out] radii will contain the radii of grid nodes;
        \param[out] Phi   will contain the spherical-harmonic expansion coefficients
                    for the potential at the given radii;
        \param[out] dPhi  will contain the radial derivatives of these coefs.
    */
    void getCoefs(std::vector<double> &radii,
        std::vector<std::vector<double> > &Phi,
        std::vector<std::vector<double> > &dPhi) const;

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual const char* name() const { return myName(); }
    static const char* myName() { static const char* text = "Multipole"; return text; }
    virtual double enclosedMass(const double radius) const;

private:
    /// radial grid
    const std::vector<double> gridRadii;

    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// actual potential implementation (based either on 1d or 2d interpolating splines)
    PtrPotential impl;

    /// asymptotic behaviour at small and large radii described by `PowerLawMultipole`
    PtrPotential asymptInner, asymptOuter;

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2) const;
};


/** Compute spherical-harmonic density expansion coefficients at the given radii.
    First it collects the values of density at a 3d grid in radii and angles,
    then applies sph.-harm. transform at each radius.
    The first step is OpenMP-parallelized, so that it may be efficiently used
    for an input density profile that is expensive to compute.
    \param[in]  dens - the input density profile.
    \param[in]  ind  - indexing scheme for spherical-harmonic coefficients,
                which determines the order of expansion and its symmetry properties.
    \param[in]  gridRadii - the array of radial points for the output coefficients;
                must form an increasing sequence and start from r>0.
    \param[out] coefs - the array of sph.-harm. coefficients:
                coefs[k][c] is the value of c-th coefficient (where c is a single index 
                combining both l and m) at the radius r_k; will be resized as needed.
    \throws std::invalid_argument if gridRadii are not correct.
*/
void computeDensityCoefsSph(const BaseDensity& dens,
    const math::SphHarmIndices& ind,
    const std::vector<double>& gridRadii,
    std::vector< std::vector<double> > &coefs);

/** Compute the coefficients of spherical-harmonic density expansion
    from an N-body snapshot.
    \param[in] particles  is the array of particles.
    \param[in] ind   is the coefficient indexing scheme (defines the order of expansion
    and its symmetries).
    \param[in] gridRadii is the grid in spherical radius.
    \param[in] smoothing is the amount of smoothing applied in penalized spline fitting procedure.
    \param[out] coefs  will contain the arrays of computed sph.-harm. coefficients
    that can be provided to the constructor of `DensitySphericalHarmonic` class;
    will be resized as needed.
*/
void computeDensityCoefsSph(
    const particles::ParticleArray<coord::PosCyl> &points,
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &coefs,
    double &innerSlope, double &outerSlope,
    double smoothing = 1.0);

/** Compute spherical-harmonic potential expansion coefficients,
    by first creating a sph.-harm.representation of the density profile,
    and then solving the Poisson equation.
    \param[in]  dens - the input density profile.
    \param[in]  ind  - indexing scheme for spherical-harmonic coefficients,
                which determines the order of expansion and its symmetry properties.
    \param[in]  gridRadii - the array of radial points for the output coefficients;
                must form an increasing sequence and start from r>0.
    \param[out] Phi  - the array of sph.-harm. coefficients for the potential:
                Phi[k][c] is the value of c-th coefficient (where c is a single index 
                combining both l and m) at the radius r_k; will be resized as needed.
    \param[out] dPhi - the array of radial derivatives of each sph.-harm. term:
                dPhi_{l,m}(r) = d(Phi_{l,m})/dr; will be resized as needed.
    \throws std::invalid_argument if gridRadii are not correct.
*/
void computePotentialCoefsSph(const BaseDensity &dens, 
    const math::SphHarmIndices &ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi);

/** Same as above, but compute coefficients from the potential directly,
    without solving Poisson equation */
void computePotentialCoefsSph(const BasePotential& pot,
    const math::SphHarmIndices& ind,
    const std::vector<double> &gridRadii,
    std::vector< std::vector<double> > &Phi,
    std::vector< std::vector<double> > &dPhi);

}  // namespace
