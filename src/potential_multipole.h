/** \file    potential_multipole.h
    \brief   density and potential approximations based on spherical-harmonic expansion
    \author  Eugene Vasiliev
    \date    2010-2025

    This module provides tools for representing arbitrary density and potential profiles
    in terms of spherical-harmonic (or multipole) expansion, with coefficients being
    interpolated in radius, and a related class for representing the potential in terms
    of a basis-set expansion (still spherical-harmonic in angles, but with a separate
    set of basis functions for the radial part).

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
#include "math_sphharm.h"
#include "smart.h"

namespace potential {


/** Spherical-harmonic expansion of density with coefficients being spline functions of radius */
class DensitySphericalHarmonic: public BaseDensity {
public:
    /** Construct the density interpolator from the provided density profile and grid parameters.
        This is not a constructor, but a static method returning a shared pointer to
        the newly created density object.
        \param[in]  src        is the input density model;
        \param[in]  sym        is the required symmetry of the density expansion
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect;
        \param[in]  fixOrder   whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const DensitySphericalHarmonic> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** Construct the density interpolator from an N-body snapshot.
        This is not a constructor, but a static method returning a shared pointer to
        the newly created density object.
        \param[in]  particles  is the array of particles.
        \param[in]  sym        is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
        \note OpenMP-parallelized loops over particles and over expansion coefficients.
    */
    static shared_ptr<const DensitySphericalHarmonic> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** Construct the object from previously computed coefficients.
        \param[in]  gridRadii  is the grid in radius (sorted in order of increase, first node > 0).
        \param[in]  coefs  is the 2d array of sph.-harm. coefficients:
        the first dimension of the array is the number of spherical harmonics (lmax+1)^2,
        and the second dimension is the number of radial grid points.
        If all coefficients of the l=0 harmonic are positive, a logarithmic scaling
        for this harmonic will be employed, and all l!=0 terms will be scaled relative to the l=0 term.
        After this optional scaling, all harmonic coefficients are spline-interpolated in log radius.
    */
    DensitySphericalHarmonic(const std::vector<double> &gridRadii,
        const std::vector< std::vector<double> > &coefs);

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "DensitySphericalHarmonic"; }

    /** return the radii of spline nodes */
    const std::vector<double>& getRadii() const { return gridRadii; }

    /** return the radii of spline nodes and the array of density expansion coefficients.
        \param[out]  radii  will contain the grid radii.
        \param[out]  coefsArray  will be filled with the array of coefficients at these radii.
    */
    void getCoefs(std::vector<double> &radii, std::vector< std::vector<double> > &coefsArray) const;

    /** fill the array of density expansion coefficients at a given radius.
        \param[out]  coefs  is a pre-allocated array of size ind.size(), which will be filled
        by this routine, but its elements corresponding to empty harmonics are left uninitialized.
    */
    void getCoefsAtRadius(double radius, double coefs[]) const;

    /** return the size of coefficients array, to be used for pre-allocating the output storage in
        getCoefsAtRadii() */
    inline unsigned int getCoefsSize() const { return ind.size(); }

private:
    /// radial grid
    const std::vector<double> gridRadii;

    /// indexing scheme for sph.-harm. coefficients
    const math::SphHarmIndices ind;

    /// radial dependence of each sph.-harm. expansion term
    std::vector<math::PtrFunction> spl;

    /// logarithmic density slope at small and large radii (rho ~ r^s)
    /// and the extrapolated value at origin
    double innerSlope, outerSlope, centralValue;

    /// whether the l=0 term is interpolated using log-scaling
    bool logScaling;

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    // the actual implementation
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;

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
    virtual std::string name() const { return "PowerLaw"; }
private:
    const math::SphHarmIndices ind; ///< indexing scheme for sph.-harm.coefficients
    double r0sq;                    ///< reference radius, squared
    bool inner;                     ///< whether this is an inward or outward extrapolation
    std::vector<double> S, U, W;    ///< sph.-harm.coefficients for extrapolation

    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;

    /// re-implement the density computation to avoid cancellation errors at large radii,
    /// by using only the U-terms which have non-zero Laplacian
    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;
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
        \param[in]  sym        is the required symmetry of the potential
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect;
        \param[in]  fixOrder   whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const Multipole> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** same as above, but takes a potential model as an input */
    static shared_ptr<const Multipole> create(
        const BasePotential& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0, double rmax = 0,
        bool fixOrder = false);

    /** create the potential from an N-body snapshot.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax       is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the size of logarithmic grid in R;
        \param[in]  rmin, rmax give the radial grid extent; 0 means auto-detect.
        \param[in]  smoothing  is the amount of smoothing applied during penalized spline fitting.
        \note OpenMP-parallelized loops over particles and over expansion coefficients.
    */
    static shared_ptr<const Multipole> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int gridSizeR, double rmin = 0., double rmax = 0., double smoothing = 1.);

    /** construct the potential from the set of spherical-harmonic coefficients.
        \param[in]  radii  is the grid in radius;
        \param[in]  Phi  is the matrix of harmonic coefficients for the potential;
                    its first dimension is the number of coefficients (lmax+1)^2,
                    and the second is the number of radial grid points;
        \param[in]  dPhi  is the matrix of radial derivatives of harmonic coefs
                    (same size as Phi, each element is  d Phi_{l,m}(r) / dr ).
    */
    Multipole(const std::vector<double> &radii,
        const std::vector<std::vector<double> > &Phi,
        const std::vector<std::vector<double> > &dPhi);

    /** return the radii of spline nodes */
    const std::vector<double>& getRadii() const { return gridRadii; }

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
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "Multipole"; }
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
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;

    virtual double densityCyl(const coord::PosCyl &pos, double /*time*/) const;
};


/// Basis-set expansion for potentials using the Zhao(1996) basis set
class BasisSet: public BasePotentialSph{
public:
    /** create the potential from the analytic density or potential model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential.
        \param[in]  src   is the input density or potential model;
        \param[in]  lmax  is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax  is the order of expansion in azimuth (phi);
        \param[in]  nmax  is the order of radial expansion (number of terms is nmax+1);
        \param[in]  eta   is the shape parameter of basis functions;
        \param[in]  r0    is the scale radius of basis functions (0 means auto-detect);
        \param[in]  fixOrder  whether to limit the order of the internal sph.-harm. expansion
        of the input density to the output order; if false (default), it may be higher
        (use a larger number of grid points in angles) to improve accuracy.
    */
    static shared_ptr<const BasisSet> create(
        const BaseDensity& src,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int nmax, double eta=1.0, double r0=0.0,
        bool fixOrder=false);

    /** create the potential from an N-body snapshot.
        \param[in]  particles  is the array of particles.
        \param[in]  sym  is the assumed symmetry of the input snapshot,
        which defines the list of spherical harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd l,m terms are zeros).
        \param[in]  lmax  is the order of sph.-harm. expansion in polar angle (theta);
        \param[in]  mmax  is the order of expansion in azimuth (phi);
        \param[in]  nmax  is the order of radial expansion (number of terms is nmax+1);
        \param[in]  eta   is the shape parameter of basis functions;
        \param[in]  r0    is the scale radius of basis functions (0 means auto-detect).
        \note OpenMP-parallelized loop over particles.
    */
    static shared_ptr<const BasisSet> create(
        const particles::ParticleArray<coord::PosCyl> &particles,
        coord::SymmetryType sym, int lmax, int mmax,
        unsigned int nmax, double eta=1.0, double r0=0.0);

    /** construct the potential from the set of basis-set expansion coefficients.
        \param[in]  eta  is the shape parameter of basis functions
        (0.5 for Clutton-Brock, 1 for Hernquist-Ostriker, values between 1 and 2 provide best results),
        the 0th order function has the 'Spheroid' (Zhao) double-power-law density profile with
        transition steepness alpha=1/eta, outer slope beta=3+1/eta, and inner slope gamma=2-1/eta;
        \param[in]  r0   is the scale radius of basis functions
        (typically should be comparable to half-mass radius);
        \param[in] coefs is the array of coefficients
        (first dimension is the number of spherical-harmonic coefficients (lmax+1)^2,
        second dimension is the number of radial basis functions nmax+1).
    */
    BasisSet(double eta, double r0, const std::vector<std::vector<double> > &coefs);

    /** return the array of basis-set expansion coefficients.
        \param[out] eta   will contain the shape parameter of basis functions;
        \param[out] r0    will contain the scale radius of basis functions;
        \param[out] coefs will contain the coefficients.
    */
    void getCoefs(double& eta, double& r0, std::vector<std::vector<double> > &coefs) const;

    virtual coord::SymmetryType symmetry() const { return ind.symmetry(); }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "BasisSet"; }

private:
    const math::SphHarmIndices ind;  ///< indexing scheme for sph.-harm. coefficients

    const double eta;  ///< shape parameter of basis functions

    const double r0;   ///< scale radius of basis functions

    const std::vector<std::vector<double> > coefs;  ///< arrays of expansion coefficients

    virtual void evalSph(const coord::PosSph &pos,
        double* potential, coord::GradSph* deriv, coord::HessSph* deriv2, double /*time*/) const;

    virtual double densitySph(const coord::PosSph &pos, double /*time*/) const;
};

}  // namespace
