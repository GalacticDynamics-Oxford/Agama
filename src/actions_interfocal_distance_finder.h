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

/** Estimate the orbit extent in R and z directions; on success, return true 
    and fill the output arguments with the coordinates of closest and farthest corner point */
bool estimateOrbitExtent(
    const potential::BasePotential& potential, const coord::PosVelCyl& point,
    double& R1, double& R2, double& z1, double& z2);

/** Estimate the interfocal distance using the potential derivatives (equation 9 in Sanders 2012)
    averaged over the given box in the meridional plane  */
double estimateInterfocalDistanceBox(
    const potential::BasePotential& potential, 
    double R1, double R2, double z1, double z2);

/** Estimate the orbit extent and then estimate the best-fit interfocal distance over this region */
double estimateInterfocalDistance(
    const potential::BasePotential& potential, const coord::PosVelCyl& point);

/** Class that provides a faster evaluation of interfocal distance via smooth interpolation 
    over pre-computed grid in energy (E) and z-component of angular momentum (L_z) plane */
class InterfocalDistanceFinder {
public:
    explicit InterfocalDistanceFinder(
        const potential::BasePotential& potential, const unsigned int gridSize=50);
    /// Return an estimate of interfocal distance for the given point, based on the values of E and L_z
    double value(const coord::PosVelCyl& point) const;
private:
    const potential::BasePotential& potential;  ///< reference to the potential
    math::CubicSpline xLcirc;                   ///< interpolator for x(E) = Lcirc(E) / (Lcirc(E)+Lscale)
    double Lscale;                              ///< scaling factor for Lcirc = Lscale * x / (1-x)
    /// 2d interpolator for interfocal distance on the grid in E, Lz/Lcirc(E) plane
    math::LinearInterpolator2d interp;
};

}  // namespace actions
