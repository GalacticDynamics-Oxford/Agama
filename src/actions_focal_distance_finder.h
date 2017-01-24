/** \file    actions_focal_distance_finder.h
    \brief   Estimation of focal distance for Staeckel fudge action/angle finder
    \author  Eugene Vasiliev
    \date    2015

    Routines in this file estimate the "focal distance" - the parameter of auxiliary
    prolate spheroidal coordinate system that is used in the Staeckel fudge approximation.
    There are two methods:
    The first one finds the best suitable focal distance by fitting a regression 
    (eq.9 in Sanders 2012) for the potential derivatives computed at the given array 
    of points in R-z plane (which should normally belong to the same orbit).
    The second method computes the best-fit parameter of the coordinate system in which
    a shell orbit in R-z plane (the one with zero radial action) follows the coordinate line 
    lambda=const as closely as possible. This method is much more computationally expensive,
    so to speed up the computations, one should instead use an interpolator that returns 
    the value of focal distance from a precomputed grid in (E,L_z) space.
    This interpolator is created and used internally in actions::ActionFinderAxisymFudge class.
*/
#pragma once
#include "potential_base.h"
#include "potential_utils.h"
#include "math_spline.h"

namespace actions {

/** Estimate the focal distance using the potential derivatives 
    (equation 9 in Sanders 2012), averaged over the given array of points in R,z plane.
    \param[in] potential  is the instance of potential;
    \param[in] traj  is the array of points (e.g., obtained by orbit integration);
    \tparam PointT  may be Pos*** or PosVel*** in any coordinate system
    \return  best-fit value of focal distance 
    (if the estimated value is negative, it is replaced by zero).
*/
template<typename PointT>
double estimateFocalDistancePoints(
    const potential::BasePotential& potential, const std::vector<PointT>& traj);


/** Estimate the focal distance by locating a thin (shell) orbit in R-z plane 
    for the given values of energy and angular momentum, 
    and finding the parameters of prolate spheroidal coordinate system in which 
    the trajectory is as close to  lambda=const  as possible.
    Note that the procedure is rather computationally expensive (requires many more 
    potential evaluations than computing the actions and angles themselves);
    thus it is more efficient to create an instance of `ActionFinderAxisymFudge` class 
    that pre-computes this quantity on a grid in E,Lz space, and uses interpolation
    to find the suitable value for any position/velocity point.

    \param[in] poten  is the instance of axisymmetric potential;
    \param[in] E is the total energy of the orbit;
    \param[in] Lz is the angular momentum of the orbit;
    \param[out] R (optional) - if not NULL, will store the cylindrical radius at which 
    the shell orbit crosses the equatorial plane;
    \return  the best-fit value of focal distance for this shell orbit.
*/
double estimateFocalDistanceShellOrbit(
    const potential::BasePotential& poten, double E, double Lz, 
    double* R=0);

}  // namespace actions
