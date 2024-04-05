/** \file    galaxymodel_selfconsistent.h
    \brief   Self-consistent model of a multi-component galaxy specified by distribution functions
    \date    2015
    \author  Eugene Vasiliev

This module deals with models consisting of one or several mass components,
which together generate the (axisymmetric) total gravitational potential.
Each component contributes to it by either a density profile, a fixed potential, or both.
The density profile is computed from a distribution function (DF) by integrating
it over velocities; in doing so, the transformation between position/velocity and
action/angle variables (the latter are the arguments of DF) is provided by
the action finder associated with the total potential.
Not every component needs to be generated from a DF: it may present a static
density and/or potential profile.

In the present implementation, the overall potential is assembled from the contributions
of each component that provide a potential (not all of them need to do so),
plus two more potential expansions (Multipole for spheroidal components and CylSpline for
disk-like components) constructed from the sum of density profiles of all relevant components.
The rationale is that the multipole expansion is obviously not efficient for strongly
flattened density profiles, but on the other hand, CylSpline is less suitable for extended
envelopes and/or density cusps; so a combination of them suits the needs of both worlds.

For instance, a single static disk-like object (without a DF) could be represented by
two components: a DiskAnsatz potential and a composite density profile (sum of DiskDensity
plus DiskAnsatz with negative sign of surface density density, i.e. the residual), which together
generate the required density distribution of a separable disk model - same as in GalPot.
Alternatively, a disk specified by DF provides a density profile which will be used to
construct a CylSpline expansion for the potential, while a halo-like component with DF
provides density in the form of spherical-harmonic expansion to be used in Multipole potential.

The workflow of self-consistent modelling is the following:
1) assemble the list of components; 
2) generate the first guess for the overall potential and initialize the action finder
for the total potential;
3) recompute the density profiles of all components that have a specified DF (are not static);
4) recalculate the potential expansion(s) and put together all potential constituents to form
an updated potential and reinitialize the action finder;
5) repeat from step 2 until convergence.
6) if necessary, modify the list of components, e.g. by replacing a static profile with
a DF-based one: this is typically needed for disk-like components, whose DF depends implicitly
on the overall potential through tabulated epicyclic frequencies, so to construct a DF, one
needs to have a first guess for the total potential to start with.

The self-consistend model is described by a structure containing the parameters for
constructing both types of potential expansions, the overall potential and its associated
action finder, and the array of model components.
The workflow described above is split between classes as follows: 
(3) is performed by each component's `update` method, which receives the total potential
and the action finder as arguments.
(4) is performed by a non-member function `updateTotalPotential` that operates on
an instance of SelfConsistentModel structure.
Alternatively, steps 3 and 4 together (and optionally step 2 if it hasn't been done before)
are performed by another function `doIteration`.
There are presently no methods for testing the convergence (step 5), so the end-user
may simply repeat the loop a few times and hope that it converged.
Steps 1 and 6 are left at the discretion of the end-user.

A technical note on the potential expansions, in particular the Multipole.
Recall that a component specified by its DF provides its density profile
already in terms of a spherical-harmonic expansion. If this is the only density
component, its density is directly used in the Multipole potential, while if
there are several density constituents, another spherical-harmonic expansion of
the combined density is created. This double work has in fact a negligible overhead,
because most of the computational effort is spent on the first stage (computing
the density profile by integration over DF, and taking its spherical-harmonic expansion).
Moreover, a transformation between two spherical-harmonic expansions is exact
if the order of the second one (used in the Multipole potential) is the same or greater
than the first one (provided by components), and if their radial grids coincide;
if the latter is not true, it introduces a generally small additional interpolation error.

*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "smart.h"
#include <vector>

namespace galaxymodel{

/** Description of a single component of the total model.
    It may provide the density profile, to be used in the multipole or CylSpline
    expansion, or a potential to be added directly to the total potential, or both.
    In case that this component has a DF, its density and potential may be recomputed
    with the `update` method, using the given total potential and its action finder.
*/
class BaseComponent {
public:
    BaseComponent(bool _isDensityDisklike) : isDensityDisklike(_isDensityDisklike)  {};
    virtual ~BaseComponent() {};

    /** recalculate the density profile (and possibly the additional potential)
        by integrating the DF over velocities in the given total potential;
        in case that the component does not have a DF, this method does nothing.
    */
    virtual void update(const potential::BasePotential& totalPotential,
        const actions::BaseActionFinder& actFinder) = 0;

    /** return the pointer to the internally used density profile for the given component;
        if it returns NULL, then this component does not provide any density to contribute
        to the Multipole or CylSpline potential expansions of the SelfConsistentModel.
    */
    virtual potential::PtrDensity   getDensity()   const = 0;

    /** return the pointer to the additional potential component to be used as part of
        the total potential, or NULL if not applicable.
    */
    virtual potential::PtrPotential getPotential() const = 0;

    /** return the pointer to the DF of this component, or NULL if not applicable */
    virtual df::PtrDistributionFunction getDF() const = 0;

    /** in case the component has an associated density profile, it may be used
        in construction of either multipole (spherical-harmonic) potential expansion,
        or a 'CylSpline' expansion of potential in the meridional plane;
        the former is more suitable for spheroidal and the latter for disk-like components.
    */
    const bool isDensityDisklike;
private:
    // do not allow to copy or assign objects of this type
    BaseComponent(const BaseComponent& src);
    BaseComponent& operator= (const BaseComponent& src);
};


/** A specialization for the component that provides static (unchanging) density and/or
    potential profiles. For the density, it is necessary to specify whether it will be
    used in Multipole or CylSpline potential expansions. */
class ComponentStatic: public BaseComponent {
public:
    ComponentStatic(const potential::PtrDensity& dens, bool _isDensityDisklike) :
        BaseComponent(_isDensityDisklike), density(dens), potential() {}
    ComponentStatic(const potential::PtrPotential& pot) :
        BaseComponent(false), density(), potential(pot) {}
    ComponentStatic(const potential::PtrDensity& dens, bool _isDensityDisklike,
        const potential::PtrPotential& pot) :
        BaseComponent(_isDensityDisklike), density(dens), potential(pot) {}
    /** update does nothing */
    virtual void update(const potential::BasePotential&, const actions::BaseActionFinder&) {}
    virtual potential::PtrDensity   getDensity()   const { return density; }
    virtual potential::PtrPotential getPotential() const { return potential; }
    virtual df::PtrDistributionFunction getDF()    const { return df::PtrDistributionFunction(); }
private:
    potential::PtrDensity density;     ///< shared pointer to the input density, if provided
    potential::PtrPotential potential; ///< shared pointer to the input potential, if exists
};


/** A (partial) specialization for the component with the density profile computed from a DF,
    using either a spherical-harmonic expansion or a 2d interpolation in meridional plane
    (detailed in two derived classes).
    Since the density computation from DF is very expensive, the density object provided by
    this component does not directly represent this interface. 
    Instead, during the update procedure, the DF-integrated density is computed at a moderate
    number of points (<~ 10^3) and used in creating an intermediate representation, that in turn
    provides the density everywhere in space by suitably interpolating from the computed values.
    The two derived classes differ in the way this intermediate representation is constructed:
    either as a spherical-harmonic expansion, or as 2d interpolation in R-z plane.
*/
class BaseComponentWithDF: public BaseComponent {
public:
    /** create a component with the given distribution function
        (the DF remains constant during the iterative procedure) */
    BaseComponentWithDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity, bool _isDensityDisklike,
        double _relError, unsigned int _maxNumEval) :
    BaseComponent(_isDensityDisklike), distrFunc(df), density(initDensity),
    relError(_relError), maxNumEval(_maxNumEval) {}

    /** return the pointer to the internal density profile */
    virtual potential::PtrDensity   getDensity()   const { return density; }

    /** no additional potential component is provided, i.e., an empty pointer is returned */
    virtual potential::PtrPotential getPotential() const { return potential::PtrPotential(); }

    /* return the pointer to the DF */
    virtual df::PtrDistributionFunction getDF() const { return distrFunc; }

protected:
    /// shared pointer to the action-based distribution function (remains unchanged)
    const df::PtrDistributionFunction distrFunc;

    /// spherical-harmonic expansion of density profile of this component
    potential::PtrDensity density;

    /// Parameters controlling the accuracy of density computation:
    /// required relative error in density
    const double relError;

    /// maximum number of DF evaluations during density computation at a single point
    const unsigned int maxNumEval;
};


/** Specialization of a component with DF and spheroidal density profile,
    which will be represented by a spherical-harmonic expansion */
class ComponentWithSpheroidalDF: public BaseComponentWithDF {
public:
    /** construct a component with given DF and parameters of spherical-harmonic expansion
        for representing its density profile.
        \param[in]  df -- shared pointer to the distribution function of this component.
        \param[in]  initDensity -- the initial guess for the density profile of this component;
                    if hard to guess, one may start e.g. with a simple Plummer sphere with
                    correct total mass and a reasonable scale radius, but it doesn't matter much.
        \param[in]  lmax -- order of spherical-harmonic expansion in theta.
        \param[in]  mmax -- order of Fourier expansion in phi (0 for axisymmetric models).
        \param[in]  gridSizeR -- number of grid points in this radial grid.
        \param[in]  rmin,rmax -- determine the extent of (logarithmic) radial grid
                    used to compute the density profile of this component.
        \param[in]  relError -- relative accuracy of density computation.
        \param[in]  maxNumEval -- max # of DF evaluations per single density computation.
    */
    ComponentWithSpheroidalDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity,
        unsigned int lmax, unsigned int mmax, unsigned int gridSizeR, double rmin, double rmax,
        double relError=1e-3, unsigned int maxNumEval=1e5);

    /** reinitialize the density profile by recomputing the values of density at a set of 
        grid points in the meridional plane, and then constructing a spherical-harmonic
        density expansion from these values.
        \note OpenMP-parallelized loop over points in r,theta when computing density by integration.
    */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);

private:
    /// definition of spatial grid for computing the density profile:
    const unsigned int lmax, mmax; ///< order of angular-harmonic expansion
    const unsigned int gridSizeR;  ///< number of points in the radial grid
    const double rmin, rmax;       ///< min/max radii for the logarithmically spaced grid
};


/** Specialization of a component with DF and flattened density profile,
    which will be represented using 2d interpolation in the meridional plane */
class ComponentWithDisklikeDF: public BaseComponentWithDF {
public:
    /** create the component with the given DF and parameters of the grid for representing
        the density in the meridional plane.
        \param[in]  df -- shared pointer to the distribution function of this component.
        \param[in]  initDensity -- the initial guess for the density profile of this component.
        \param[in]  mmax -- order of Fourier expansion in phi (0 for axisymmetric models).
        \param[in]  gridSizeR -- size of grid in cylindrical radius,
        which should cover the range where the density is presumed to be non-negligible.
        \param[in]  Rmin, Rmax -- the first (positive) and the last radial grid point.
        \param[in]  gridSizez -- size of the vertical grid.
        \param[in]  zmin, zmax -- extent of the vertical grid.
        \param[in]  relError -- relative accuracy in density computation.
        \param[in]  maxNumEval -- maximum # of DF evaluations for a single density value.
    */
    ComponentWithDisklikeDF(const df::PtrDistributionFunction& df,
        const potential::PtrDensity& initDensity,
        unsigned int mmax,
        unsigned int gridSizeR, double Rmin, double Rmax, 
        unsigned int gridSizez, double zmin, double zmax,
        double relError=1e-3, unsigned int maxNumEval=1e5);

    /** reinitialize the density profile by recomputing the values of density at a set of 
        grid points in the meridional plane, and then constructing a density interpolator.
        \note OpenMP-parallelized loop over points in R,z when computing density by integration.
    */
    virtual void update(const potential::BasePotential& pot, const actions::BaseActionFinder& af);
private:
    const unsigned int mmax;       ///< order of Fourier expansion
    const unsigned int gridSizeR;  ///< size of the grid in cylindrical radius
    const double Rmin, Rmax;       ///< min/max grid nodes in radius
    const unsigned int gridSizez;  ///< size of the vertical grid
    const double zmin, zmax;       ///< min/max grid nodes in z
};


/// smart pointer to the model component
typedef shared_ptr<BaseComponent> PtrComponent;


/** The main object that puts together all ingredients of a self-consistent model:
    list of components, total potential and action finder, and the parameters of
    the two potential expansions that are constructed from the overall density.
    Note that this is not a class but a simple structure with no methods and no private members;
    all parameters of potential expansions must be explicitly initialized by the user
    before doing any modelling (the rationale is that there are too many of them to
    be meaningfully used in a constructor in the absense of named arguments in C++).
    Array of components should also be filled by the user before doing the modelling;
    however, potential and action finders will be initialized by the model iteration step
    automatically. Moreover, the list of components may be changed manually at any time
    between the iterations, and the overall potential and action finder may be retrieved
    by the user at any time.
*/
struct SelfConsistentModel {
public:
    /// array of model components
    std::vector<PtrComponent> components;

    /// total gravitational potential of all components (empty at the beginning)
    potential::PtrPotential totalPotential;

    /// action finder associated with the total potential (empty at the beginning)
    actions::PtrActionFinder actionFinder;

    /// whether to use the interpolated action finder (faster but less accurate)
    bool useActionInterpolation;

    /// whether to print out progress report messages
    bool verbose;

    /** parameters of grid for computing the multipole expansion of the combined
        density profile of spheroidal components;
        in general, these parameters should encompass the range of analogous parameters 
        of all components that have a spherical-harmonic density representation.
    */
    unsigned int lmaxAngularSph;  ///< order of angular-harmonic expansion in theta (l_max)
    unsigned int mmaxAngularSph;  ///< order of Fourier expansion in phi (m_max)
    unsigned int sizeRadialSph;   ///< number of grid points in radius
    double rminSph, rmaxSph;      ///< range of radii for the logarithmic grid

    /** parameters of grid for computing CylSpline expansion of the combined
        density profile of flattened (disk-like) components;
        the radial and vertical extent should be somewhat larger than the region where
        the overall density is non-negligible, and the resolution should match that
        of the density profiles of components.
    */
    unsigned int mmaxAngularCyl;  ///< order of Fourier expansion in phi (m_max)
    unsigned int sizeRadialCyl;   ///< number of grid nodes in cylindrical radius
    double RminCyl, RmaxCyl;      ///< innermost (non-zero) and outermost grid nodes in cylindrical radius
    unsigned int sizeVerticalCyl; ///< number of grid nodes in vertical (z) direction
    double zminCyl, zmaxCyl;      ///< innermost and outermost grid nodes in vertical direction

    /// assign default values
    SelfConsistentModel() :
        useActionInterpolation(true),
        verbose(true),
        lmaxAngularSph(0), mmaxAngularSph(0), sizeRadialSph(25), rminSph(0), rmaxSph(0),
        mmaxAngularCyl(0), sizeRadialCyl(20), RminCyl(0), RmaxCyl(0),
        sizeVerticalCyl(20), zminCyl(0), zmaxCyl(0)
    {}
};

/** recompute the total potential using the current density profiles for all components,
    and reinitialize the action finder;
    \throws a runtime_error exception if the total potential does not have any constituents.
*/
void updateTotalPotential(SelfConsistentModel& model);

/** Main iteration step: recompute the densities of all components, and then call 
    `updateTotalPotential`; if no potential is present at the beginning, it is initialized
    by a call to the same `updateTotalPotential` before recomputing the densities.
    \note OpenMP-parallelized loops in Component***::update().
*/
void doIteration(SelfConsistentModel& model);

}  // namespace
