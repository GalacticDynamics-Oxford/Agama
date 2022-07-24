/** \file    galaxymodel_densitygrid.h
    \brief   Spatial discretization schemes for Schwarzschild/FEM models
    \date    2009-2017
    \author  Eugene Vasiliev

*/
#pragma once
#include "potential_base.h"
#include "galaxymodel_target.h"
#include <vector>

namespace galaxymodel{

/** Base class for all density discretization schemes */
class BaseTargetDensity: public BaseTarget {
public:

    /// for density targets, only the three coordinates are used, not velocities
    virtual unsigned int numVars() const { return 3; }

    /// compute the projection of a DF-based model onto the basis elements of the spatial grid
    virtual void computeDFProjection(const GalaxyModel& model, StorageNumT* output) const;

    /// compute the projections of the input density onto all basis elements of the grid
    /// (by default uses a 3d numerical integration for all basis functions computed via `eval()`,
    /// but derived classes may provide optimized versions taking into account the structure
    /// of basis functions and their support)
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const;
};


/** Classical scheme for discretizing a spheroidal density.
    The space is divided into concentric shells in radius;
    the spherical surface of each shell (one octant with x,y,z>=0)
    is further divided into three equal panes by lines x=y, y=z, z=x
    joining at the central point x=y=z (marked by a dot on the diagram below).
    Each pane is divided into Nstrip * Nstrip individual cells.
    An example shows the scheme for Nstrip=2:
    \code
            /^\           z ^
           /   \            |
          / \ / \           |
         /   X   \        _-^-_
        / \ / \ / \  x <-¯     ¯-> y
       /   X   X   \
      /-_ / \./ \ _-\
     /   |-_ | _-|   \
    /   /   ¯|¯   \   \
    -------------------
    \endcode
    In the classical approach, the basis elements are non-overlapping top-hat blocks
    enclosed by these subdivision lines. This corresponds to B-splines of order N=0
    (piecewise-constant functions equal to 1 inside the corresponding block and 0 elsewhere);
    the number of such blocks in each shell is 3 * Nstrip^2.
    This scheme could be generalized to higher-order B-splines, in particular,
    N=1 corresponds to piecewise-linear basis elements, which equal to 1 at the corresponding
    grid nodes -- the intersections of lines -- and linearly drop to zero towards the adjacent
    nodes. The number of such nodes on the surface of each spherical shell is
    3 * Nstrip * (Nstrip+1) + 1. The basis functions also linearly interpolate in radius
    between adjacent shell surfaces; the innermost node is at origin and is not subdivided.
    Furthermore, the entire grid may be compressed in Y and Z directions by a constant factor,
    so that instead of spherical radius, we use the spheroidal radius r^2 = x^2 + (y/p)^2 + (z/q)^2.
    \tparam  N  is the degree of interpolating B-splines (0 or 1).
*/
template<int N>
class TargetDensityClassic: public BaseTargetDensity {
    const unsigned int stripsPerPane;     ///< number of strips in each direction in one pane
    const unsigned int valuesPerShell;    ///< number of basis functions in each spheroidal shell
    const std::vector<double> shellRadii; ///< spheroidal radii of the shells
    const double axisX, axisY, axisZ;     ///< flattening of the grid in each cartesian direction
public:
    /** construct the grid with given parameters.
        \param[in]  stripsPerPane  is the number of strips in each direction in one pane
        (so that the total number of grid cells is 3 * stripsPerPane^2), should be positive.
        \param[in]  shellRadii  is the array of grid nodes in spheroidal radius, should be increasing;
        if the first element is not at zero, then an additional node at zero is implied.
        \param[in]  axisYtoX, axisZtoX  are optional flattening parameters for the grid:
        the grid is geometrically scaled by these numbers in Y and Z directions,
        and at the same time all three directions are multiplied by a compensating factor
        (axisYtoX*axisZtoX)^{-1/3}  that brings the overall volume scaling to unity.
    */
    TargetDensityClassic(
        const unsigned int stripsPerPane,
        const std::vector<double>& shellRadii,
        const double axisYtoX=1., const double axisZtoX=1.);

    virtual const char* name() const;
    virtual std::string coefName(unsigned int index) const;

    /// total number of basis functions
    virtual unsigned int numValues() const { return valuesPerShell * shellRadii.size() + N; }

    /// adds the values of all nonzero basis functions at the input point, weighted by mult,
    /// to the output array; at most 1 (for N=0) or 8 (for N=1) values are non-zero at any point.
    virtual void addPoint(const double point[3], const double mult, double values[]) const;

    /// an optimized routine for computing the projection of the density profile
    /// onto the basis functions (in the case N=0 these are the masses contained in each cell)
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const;
};


/** Representation of a spheroidal density profile in terms of spherical harmonics.
    The radial basis functions are triangular-shaped blocks (i.e., B-splines of degree one),
    and the angular dependence is provided by a reduced set of spherical-harmonic functions
    (using only even l,m, 0 <= m <= mmax, m <= l <= lmax, and mmax <= lmax).
    There are `valuesPerShell` angular basis functions for each node in the radial grid,
    plus a single function for r=0 (only the 0th harmonic is used).
*/
class TargetDensitySphHarm: public BaseTargetDensity {
    const int lmax, mmax;             ///< order of angular expansion in theta and phi
    const unsigned int angularCoefs;  ///< number of angular coefs at each radius
    const std::vector<double> gridr;  ///< grid in spherical radius
public:
    /** construct the grid with given parameters.
        \param[in]  lmax  is the order of expansion in theta:
        lmax==0 means spherical symmetry, and for lmax>0 only even terms are used (corresponding to
        a triaxial symmetry), so the actual order is rounded down to the nearest even number.
        \param[in]  mmax  is the order of expansion in azimuthal angle (phi):
        mmax==0 means axisymmetry, and mmax may not exceed lmax; only terms with non-negative even m
        are used in the expansion, so the actual order is rounded down to the nearest even number.
        \param[in]  gridr  is the radial grid used in the expansion, should be in increasing order;
        if the first element is not at zero, then an additional node at zero is implied.
        \throw  std::invalid_argument  if the parameters are not valid.
    */
    TargetDensitySphHarm(const int lmax, const int mmax, const std::vector<double>& gridr);

    virtual const char* name() const;
    virtual std::string coefName(unsigned int index) const;

    /// total number of basis functions
    virtual unsigned int numValues() const { return angularCoefs * gridr.size() + 1; }

    /// compute the values of all basis functions at the point specified by its cartesian coordinates
    virtual void addPoint(const double point[3], const double mult, double values[]) const;

    /// an optimized routine for computing the projection of the density profile
    /// onto the basis functions
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const;
};


/** Representation of a density profile, possibly strongly flattened and
    moderately non-axisymmetric, in terms of a Fourier expansion in radius with
    coefficients specified on a rectangular grid in the meridional plane.
    The basis functions in the meridional plane are either non-overlapping top-hat blocks
    separated by the grid lines (if N=0), or triangular-shaped bilinear functions spanning
    two adjacent grid segments in each direction and centered on each grid node (if N=1).
    The azimuthal variation is expanded in even-order Fourier coefficients, up to m_max.
    \tparam  N  is the degree of interpolating B-splines (0 or 1).
*/
template<int N>
class TargetDensityCylindrical: public BaseTargetDensity {
    const int mmax;                    ///< order of angular expansion in azimuth (phi)
    const std::vector<double> gridR;   ///< grid in the cylindrical radius
    const std::vector<double> gridz;   ///< grid in the z direction
    const unsigned int totalNumValues; ///< total number of basis functions
public:
    /** construct the grid with given parameters.
        \param[in]  mmax  is the order of azimuthal Fourier expansion
        (0 means axisymmetry, and only even terms are used so that mmax is effectively
        rounded down to the nearest even number).
        \param[in]  gridR  is the grid in cylindrical radius (should be in increasing order,
        if the first element is >0, an extra node at R=0 is implied).
        \param[in]  gridz  is the grid in the vertical coordinate (covers only half-space,
        starts at zero or at a positive value, in the latter case an extra node at z=0 is implied).
        \throw  std::invalid_argument  if the parameters are not valid.
    */
    TargetDensityCylindrical(const int mmax,
        const std::vector<double>& gridR, const std::vector<double>& gridz);

    virtual const char* name() const;
    virtual std::string coefName(unsigned int index) const;

    /// total number of basis functions
    virtual unsigned int numValues() const { return totalNumValues; }

    /// compute the values of all basis functions at the point specified by its cartesian coordinates
    virtual void addPoint(const double point[3], const double mult, double values[]) const;

    /// compute the projections of the density profile onto the basis functions
    virtual std::vector<double> computeDensityProjection(const potential::BaseDensity& density) const;
};

}  // namespace
