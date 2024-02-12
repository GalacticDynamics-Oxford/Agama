/** \file    actions_staeckel.h
    \brief   Action-angle finders using Staeckel potential approximation
    \author  Eugene Vasiliev
    \date    2015

Computation of actions and angles for axisymmetric systems, using two related methods:
1) exact expressions for Staeckel potentials  (e.g. for the Perfect Ellipsoid),
2) "Staeckel Fudge" approximation applicable for any axisymmetric potential.
To avoid confusion, the latter is called simply "Fudge" in what follows.

Most sub-steps are shared between the two methods; 
only the computation of integrals of motion, and the auxiliary function that enters 
the expression for canonical momentum, are specific to each method.

The implementation is inspired by the code written by Jason Sanders,
but virtually nothing of the original code remains.
*/
#pragma once
#include "actions_base.h"
#include "actions_focal_distance_finder.h"
#include "smart.h"

namespace actions {

/// \name  ------- Stand-alone driver routines that compute actions for a single point -------
///@{

/** Evaluate (exactly up to integration errors) any combination of actions, angles and frequencies
    in the Staeckel potential of an oblate Perfect Ellipsoid.
    \param[in]  potential is the input Staeckel potential.
    \param[in]  point is the position/velocity point.
    \param[out] act   if not NULL, will contain computed actions (Jr=Jz=NAN if E>=0).
    \param[out] ang   if not NULL, will contain corresponding angles (NAN if E>=0).
    \param[out] freq  if not NULL, will contain corresponding frequencies (NAN if E>=0).
*/
void evalAxisymStaeckel(
    const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point,
    Actions* act=NULL,
    Angles* ang=NULL,
    Frequencies* freq=NULL);

/** Evaluate approximately any combination of actions, angles and frequencies
    in an arbitrary axisymmetric potential, using the Staeckel fudge method.
    \param[in]  potential is an arbitrary axisymmetric potential.
    \param[in]  point     is the position/velocity point.
    \param[out] act   if not NULL, will contain computed actions (Jr=Jz=NAN if E>=0).
    \param[out] ang   if not NULL, will contain corresponding angles (NAN if E>=0).
    \param[out] freq  if not NULL, will contain corresponding frequencies (NAN if E>=0).
    \param[in]  focalDistance is the geometric parameter of best-fit coordinate system:
    the accuracy of the method depends on this parameter, which should be estimated by one of
    the methods from actions_focal_distance_finder.h.
    \throw      std::invalid_argument exception if the potential is not axisymmetric.
*/
void evalAxisymFudge(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point,
    Actions* act=NULL,
    Angles* ang=NULL,
    Frequencies* freq=NULL,
    double focalDistance=0);

///@}
/// \name  ------- Class interface to action/angle finders  -------
///@{

/** Action/angle finder for an Oblate Perfect Ellipsoid potential. */
class ActionFinderAxisymStaeckel: public BaseActionFinder {
public:
    explicit ActionFinderAxisymStaeckel(const potential::PtrOblatePerfectEllipsoid& potential) :
        pot(potential) {};

    virtual std::string name() const { return "AxisymmetricStaeckel"; }

    virtual void eval(const coord::PosVelCyl& point,
        Actions* act=NULL, Angles* ang=NULL, Frequencies* freq=NULL) const
    { evalAxisymStaeckel(*pot, point, act, ang, freq); }

private:
    const potential::PtrOblatePerfectEllipsoid pot;  ///< the potential in which actions are computed
};

/** Action/angle finder for a generic axisymmetric potential, based on Staeckel Fudge approximation.
    It is more suitable for massive computation in a fixed potential than just using 
    the standalone routines, because it estimates the focal distance using a pre-computed 
    interpolation grid, rather than doing it individually for each point. This results in 
    a considerable speedup in action computation, for a minor overhead during initialization.
    Additionally, it may set up an interpolation table for actions as functions of three integrals
    of motion (one of them being approximate), which speeds up the evaluation by another order of
    magnitude, for a moderate decrease in accuracy. Interpolated actions have small but non-negligible
    systematic bias, which depends on the potential as well as the phase-space location,
    hence they cannot be used for comparing the likelihood of a DF in different potentials.
*/
class ActionFinderAxisymFudge: public BaseActionFinder {
public:
    /** set up the action finder: an interpolator for the focal distance, and optionally
        interpolators for actions (if interpolate==true).
        \throw std::invalid_argument exception if the potential is not axisymmetric
        or std::runtime_error in case of other problems in initialization.
        \note OpenMP-parallelized loops over the grid in E,L for locating shell orbits,
        and optionally for constructing the action interpolator (if requested).
    */
    ActionFinderAxisymFudge(const potential::PtrPotential& potential, bool interpolate = false);

    virtual std::string name() const;

    virtual void eval(const coord::PosVelCyl& point,
        Actions* act=NULL, Angles* ang=NULL, Frequencies* freq=NULL) const;

    /** return the best-suitable focal distance for the given point, obtained by interpolation */
    double focalDistance(const coord::PosVelCyl& point) const;

private:
    const double invPhi0;                   ///< 1 / Phi(r=0)
    const potential::PtrPotential pot;      ///< the potential Phi in which actions are computed
    const potential::Interpolator interp;   ///< 1d interpolator for Lcirc(E)
    math::LinearInterpolator2d interpD;     ///< 2d interpolator for the focal distance Delta(E,Lz)
    math::CubicSpline2d interpR;            ///< 2d interpolator for Rshell(E,Lz) / Rcirc(E)
    math::CubicSpline3d intJr, intJz;       ///< 3d interpolators for Jr and Jz as functions of (E,Lz,I3)
};

///@}
}  // namespace actions
