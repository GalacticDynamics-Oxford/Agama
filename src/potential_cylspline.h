/** \file    potential_cylspline.h
    \brief   density and potential approximations based on 2d spline in cylindrical coordinates
    \author  Eugene Vasiliev
    \date    2014-2024

    The classes and routines in this file deal with a generic way of representing
    an arbitrary density or potential profile as a 1d expansion + 2d interpolation:
    the azimuthal dependence for a non-axisymmetric profile is represented by 
    a Fourier expansion in phi angle, with the coefficients of expansion being
    2d interpolated functions in meridional plane (R,z).

    The coefficients of expansion are stored as arrays of m-th harmonic terms
    (cos(m phi), sin(m phi), 0 <= m <= mmax), where each term is a 2d matrix
    of coefficients, with two additional arrays defining grids in R and z directions.
    The density interpolator uses one set of coefficients (the value of density),
    while the potential uses three - the value of potential and its R- and z-derivatives.

    Standalone routines compute the coefficients of expansion for density or potential,
    the latter case - in three variants: from the provided potential, from a density
    model via the solution of Poisson equation in cylindrical coordinates, or from
    an array of point masses (again solving the Poisson equation); these two last
    options are quite expensive computationally.

    Once computed, the coefficients may be used to construct instances of interpolator
    classes, and stored/loaded by `writePotential`/`readPotential` routines from
    potential_factory.h
*/
#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "math_linalg.h"
#include "smart.h"

namespace potential {

/** Density profile expressed as a Fourier expansion in azimuthal angle (phi)
    with coefficients interpolated on a 2d grid in meridional plane (R,z).
*/
class DensityAzimuthalHarmonic: public BaseDensity {
public:
    /** Construct the density interpolator from the input density profile.
        This is a static member function returning a pointer to a newly created object.
        The arguments have the same meaning as for `CylSpline::create`, but the grid extent
        (min/max) must be provided explicitly, i.e. is not determined automatically.
        The input density values are taken at the nodes of 2d grid in (R,z) and
        nphi distinct values of phi, where nphi=mmaxFourier+1 if the density is
        reflection-symmetric in y, or nphi=2*mmaxFourier+1 otherwise.
        The order of this internal Fourier expansion mmaxFourier will be fixed to mmax
        if fixOrder==true, otherwise will be higher than mmax to improve accuracy.
    */
    static shared_ptr<const DensityAzimuthalHarmonic> create(const BaseDensity& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false);

    /** construct the object from the array of coefficients */
    DensityAzimuthalHarmonic(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz,
        const std::vector< math::Matrix<double> > &coefs);
    virtual coord::SymmetryType symmetry() const { return sym; }
    virtual std::string name() const { return myName(); }
    static std::string myName() { return "DensityAzimuthalHarmonic"; }

    /** return the grid parameters */
    void getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const;

    /** retrieve the values of density expansion coefficients 
        and the nodes of 2d grid used for interpolation */
    void getCoefs(std::vector<double> &gridR, std::vector<double> &gridz, 
        std::vector< math::Matrix<double> > &coefs) const;

    /** return the value of m-th Fourier harmonic at the given point in R,z plane */
    double rho_m(int m, double R, double z) const;

private:
    std::vector<math::PtrInterpolator2d> spl;  ///< spline for rho_m(R,z)
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients
    double Rscale;            ///< radial scaling factor

    virtual double densityCar(const coord::PosCar &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    virtual double densitySph(const coord::PosSph &pos, double time) const {
        return densityCyl(toPosCyl(pos), time); }

    /** Return density interpolated on the 2d grid, or zero if the point lies outside the grid */
    virtual double densityCyl(const coord::PosCyl &pos, double time) const;
};

/** Generic potential approximation based on Fourier expansion in azimuthal angle (phi)
    with coefficients being 2d spline functions of R,z.
    It is suitable for strongly flattened, possibly non-axisymmetric models with
    non-singular density profiles at origin. The potential and its derivatives are accurately
    interpolated within the grid definition region (0<=R<=Rmax, -zmax<=z<=zmax),
    and extrapolated outside this region using asymptotic spherical-harmonic expansion,
    which approximates the potential and forces rather well at all radii, but corresponds
    to zero density outside the grid.
    The cost of evaluating the potential is roughly the same as for the spherical-harmonic
    potential approximation, although the cost of computing the coefficients (via non-member
    functions `computePotentialCoefsCyl` or static factory functions `create`) is higher.
*/
class CylSpline: public BasePotentialCyl
{
public:

    /** Create the potential from the provided density model.
        This is not a constructor but a static member function returning a shared pointer
        to the newly created potential: it creates the grids, computes the coefficients
        and calls the actual class constructor.
        It exists in several variants: the first one takes a density model as input
        and solves Poisson equation to find the potential azimuthal harmonic coefficients;
        the second one takes a potential model and computes these coefs directly.
        In both cases, if the input model is not axisymmetric, its angular Fourier expansion
        with order mmaxFourier will be used to compute the harmonics, where mmaxFourier is
        fixed to mmax if fixOrder==true, otherwise will be higher to improve accuracy.
        A third variant takes an N-body snapshot as input (discussed separately).
        If the grid extent (R/z min/max) is not specified (left as zeros), it is determined
        automatically from the requirement to enclose almost all of the model mass and have
        a sufficient resolution at origin.
        \param[in]  src        is the input density or potential model;
        \param[in]  sym        is the required symmetry of the potential
        (if set to ST_UNKNOWN, will be taken from the input density model);
        \param[in]  mmax       is the order of expansion in azimuth (phi);
        \param[in]  gridSizeR  is the number of grid nodes in cylindrical radius (semi-logarithmic);
        \param[in]  Rmin, Rmax give the radial grid extent (first non-zero node and
                    the outermost node); zero values mean auto-detect;
        \param[in]  gridSizez  is the number of grid nodes in vertical direction;
        \param[in]  zmin, zmax give the vertical grid extent (first non-zero positive node
                    and the outermost node; if the source model is not symmetric w.r.t.
                    z-reflection, a mirrored extension of the grid to negative z will be created);
                    zero values mean auto-detect;
        \param[in]  fixOrder   determines whether to limit the order of the internal Fourier
                    expansion of the input density to mmax; if false (default), use a larger
                    number of points for the integration in phi to improve the accuracy,
                    and then truncate the result back to mmax;
        \param[in]  useDerivs  specifies whether to compute potential derivatives from density.
        \note OpenMP-parallelized loop over nodes of a 2d grid in R,z when integrating the density
        (the latter is the input density when it is axisymmetric, otherwise an internally created
        instance of DensityAzimuthalHarmonic).
    */
    static shared_ptr<const CylSpline> create(const BaseDensity& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false, bool useDerivs=true);

    /** Same as above, but taking a potential model as an input. */
    static shared_ptr<const CylSpline> create(const BasePotential& src,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax,
        bool fixOrder=false);

    /** Create the potential from an N-body snapshot.
        \param[in] particles  is the array of particles.
        \param[in] sym  is the assumed symmetry of the input snapshot,
        which defines the list of angular harmonics to compute and to ignore
        (e.g. if it is set to coord::ST_TRIAXIAL, all negative or odd m terms are zeros).
        \param[in] mmax  is the order of angular expansion (if the symmetry includes
        coord::ST_ZROTATION flag, mmax will be set to zero).
        \param[in]  gridSizeR  is the number of grid nodes in cylindrical radius (semi-logarithmic);
        \param[in]  Rmin, Rmax give the radial grid extent (first non-zero node and
        the outermost node); zero values mean that they will be determined automatically from
        the requirement to enclose almost all particles and provide a good resolution.
        \param[in]  gridSizez  is the number of grid nodes in vertical direction;
        \param[in]  zmin, zmax give the vertical grid extent (first non-zero positive node
        and the outermost node); if the requested symmetry type does not include
        z-reflection, a mirrored extension of the grid to negative z will be created;
        zero values mean that they will be assigned automatically.
        \param[in]  useDerivs  specifies whether to compute potential derivatives
        and construct a quintic spline, or skip it and construct a cubic spline
        (due to noisy nature of N-body models, higher order does not necessarily imply more accuracy).
        \note OpenMP-parallelized loop over nodes of a 2d grid in R,z.
    */
    static shared_ptr<const CylSpline> create(
        const particles::ParticleArray<coord::PosCyl>& particles,
        coord::SymmetryType sym, int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax,
        unsigned int gridSizez, double zmin, double zmax, bool useDerivs=false);

    /** Construct the potential from previously computed coefficients.
        \param[in]  gridR  is the grid in cylindrical radius
        (nodes must start at 0 and be increasing with R);
        \param[in]  gridz  is the grid in the vertical direction:
        if it starts at 0 and covers the positive half-space, then the potential
        is assumed to be symmetric w.r.t. z-reflection, so that the internal grid
        will be extended to the negative half-space; in the opposite case it
        is assumed to be asymmetric and the grid must cover negative z too.
        \param[in]  Phi  is the 3d array of harmonic coefficients:
        the outermost dimension determines the order of expansion - the number of terms
        is 2*mmax+1, so that m runs from -mmax to mmax inclusive (i.e. the m=0 harmonic
        is contained in the element with index mmax).
        Each element of this array (apart from the one at mmax, i.e. for m=0) may be empty,
        in which case the corresponding harmonic is taken to be identically zero.
        Non-empty elements are matrices with dimension gridR.size() * gridz.size(),
        regardless of whether gridz covers only z>=0 or both positive and negative z.
        The indexing scheme is Phi[m+mmax](iR,iz) = Phi_m(gridR[iR], gridz[iz]).
        \param[in]  dPhidR  is the array of radial derivatives of the potential,
        with the same shape as Phi, and containing the same number of non-empty terms.
        \param[in]  dPhidz  is the array of vertical derivatives.
        If both dPhidR and dPhidz are empty arrays (not arrays with empty elements),
        as specified by default, then the potential is constructed using only the values
        of Phi at grid nodes, employing 2d cubic spline interpolation for each m term.
        If derivatives are provided, then the interpolation is based on quintic splines,
        improving the accuracy.
    */
    CylSpline(
        const std::vector<double> &gridR,
        const std::vector<double> &gridz,
        const std::vector< math::Matrix<double> > &Phi,
        const std::vector< math::Matrix<double> > &dPhidR = std::vector< math::Matrix<double> >(),
        const std::vector< math::Matrix<double> > &dPhidz = std::vector< math::Matrix<double> >() );

    virtual std::string name() const { return myName(); }
    static std::string myName() { return "CylSpline"; }
    virtual coord::SymmetryType symmetry() const { return sym; };
    virtual double enclosedMass(const double radius) const;

    /** return the grid parameters */
    void getGridExtent(double &Rmin, double &Rmax, double &zmin, double &zmax) const;

    /** retrieve coefficients of potential approximation.
        \param[out] gridR will be filled with the array of R-values of grid nodes;
        \param[out] gridz will be filled with the array of z-values of grid nodes:
        if the potential is symmetric w.r.t. z-reflection, then only the half-space with
        non-negative z is returned both in this array and in the coefficients.
        \param[out] Phi will contain array of sequentially stored 2d arrays
        (the size of the outer array equals the number of terms in azimuthal expansion (2*mmax+1),
        inner arrays contain gridR.size()*gridz.size() values).
        \param[out] dPhidR will contain the array of derivatives in the radial direction,
        with the same shape as Phi, if they were provided at construction (i.e., when using
        quintic interpolation internally), otherwise this will be an empty array.
        \param[out] dPhidz will contain the array of derivatives in the vertical direction,
        or an empty array if the derivatives were not provided at construction
    */
    void getCoefs(
        std::vector<double> &gridR,
        std::vector<double> &gridz,
        std::vector< math::Matrix<double> > &Phi,
        std::vector< math::Matrix<double> > &dPhidR,
        std::vector< math::Matrix<double> > &dPhidz) const;

private:
    /// array of 2d splines (for each m-component in the expansion in azimuthal angle)
    std::vector<math::PtrInterpolator2d> spl;
    coord::SymmetryType sym;  ///< type of symmetry deduced from coefficients
    double Rscale;            ///< radial scaling factor for coordinate transformation
    bool logScaling;          ///< flag for optional log-transformation of the m=0 term
    bool haveDerivs;          ///< whether the potential derivatives were provided at construction

    /// asymptotic behaviour at large radii described by `PowerLawMultipole`
    PtrPotential asymptOuter;

    /// compute potential and its derivatives
    virtual void evalCyl(const coord::PosCyl &pos,
        double* potential, coord::GradCyl* deriv, coord::HessCyl* deriv2, double /*time*/) const;
};

}  // namespace
