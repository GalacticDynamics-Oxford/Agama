/** \file    actions_interfocal_distance_finder.h
    \brief   Estimation of interfocal distance for Staeckel fudge action/angle finder
    \author  Eugene Vasiliev
    \date    2015
 
    Routines in this file estimate the "interfocal distance" - the parameter of auxiliary
    prolate spheroidal coordinate system that is used in the Staeckel fudge approximation.
    The method employed here first estimates the orbit extent in the meridional plane
    (range of variation of R and z coordinates) from the given position/velocity point,
    then finds the best suitable interfocal distance from the potential derivatives
    (eq.9 in Sanders 2012), averaged over the area in meridional plane covered by the orbit.
    These two steps require 30-40 potential evaluations per given point, 
    so to speed up the computations, one may instead use an interpolator that returns 
    the value of interfocal distance from a precomputed grid in (E,L_z) space.
    This interpolator is created and used internally in actions::ActionFinderAxisymFudge class.
*/
#pragma once
#include "potential_base.h"
#include "math_spline.h"

namespace actions {
//  ------- Routines for estimating the interfocal distance for the Fudge approximation -------

/** Estimate the squared interfocal distance using the potential derivatives 
    (equation 9 in Sanders 2012), averaged over the given array of points in R,z plane.
    \param[in] potential  is the instance of potential, which must be axisymmetric;
    \param[in] traj  is the array of points (e.g., obtained by orbit integration);
    \tparam PointT  may be PosCyl or PosVelCyl
    \return  best-fit value of Delta^2, which may turn out to be negative (in that case, 
    the user should replace it with some small positive value to use the Staeckel Fudge).
*/
template<typename PointT>
double estimateSquaredInterfocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj);

template<typename PointT>
double estimateSquaredInterfocalDistanceThinOrbit(const std::vector<PointT>& traj);

void findClosedOrbitRZplane(const potential::BasePotential& poten, double E, double Lz, 
    double &Rthin, double& Jr, double& Jz, double &IFD);

/** Class that provides a faster evaluation of interfocal distance via smooth interpolation 
    over pre-computed grid in energy (E) and z-component of angular momentum (L_z) plane */
class InterfocalDistanceFinder {
public:
    explicit InterfocalDistanceFinder(
        const potential::BasePotential& potential, const unsigned int gridSize=50);

    /// Return an estimate of interfocal distance for the given point, based on the values of E and L_z
    double value(const coord::PosVelCyl& point) const;

    /** Return several key quantities as functions of energy and L_z:
        \param[out] maxJr: maximum value of Jr (for a planar orbit in x-y plane)
        \param[out] maxJz: maximum value of Jz (for a thin orbit in R-z plane)
        \param[out] Rthin: radius of this thin orbit
    */
    void params(double E, double Lz, double& maxJr, double& maxJz, double& Rthin) const;

private:
    const potential::BasePotential& potential;  ///< reference to the potential
    math::CubicSpline xLcirc;                   ///< interpolator for x(E) = Lcirc(E) / (Lcirc(E)+Lscale)
    double Lscale;                              ///< scaling factor for Lcirc = Lscale * x / (1-x)
    /// 2d interpolator for interfocal distance on the grid in E, Lz/Lcirc(E) plane
    math::LinearInterpolator2d interp;
    math::LinearInterpolator2d interpJr;
    math::LinearInterpolator2d interpJz;
    math::LinearInterpolator2d interpRt;
};

}  // namespace actions
