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
#include "actions_interfocal_distance_finder.h"
#include "potential_perfect_ellipsoid.h"
#include "smart.h"

namespace actions {

/// \name  ------- Stand-alone driver routines that compute actions for a single point -------
///@{

/** Find exact actions in the Staeckel potential of oblate Perfect Ellipsoid.
    \param[in]  potential is the input Staeckel potential;
    \param[in]  point     is the position/velocity point;
    \return     actions for the given point, or Jr=Jz=NAN if the energy is positive;
    \throw      std::invalid_argument exception if some error occurs.
*/    
Actions actionsAxisymStaeckel(
    const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point);

/** Find exact actions and angles in the Staeckel potential of oblate Perfect Ellipsoid.
    \param[in]  potential is the input Staeckel potential;
    \param[in]  point     is the position/velocity point;
    \param[out] freq      if not NULL, store the frequencies of motion in this variable;
    \return     actions and angles for the given point, or Jr=Jz=NAN if the energy is positive;
    \throw      std::invalid_argument exception if some error occurs.
*/
ActionAngles actionAnglesAxisymStaeckel(
    const potential::OblatePerfectEllipsoid& potential, 
    const coord::PosVelCyl& point,
    Frequencies* freq=NULL);

/** Find approximate actions in a given axisymmetric potential, using the Staeckel Fudge method.
    \param[in]  potential is the arbitrary axisymmetric potential;
    \param[in]  point     is the position/velocity point;
    \param[in]  interfocalDistance is the geometric parameter of best-fit coordinate system:
    the accuracy of the method depends on this parameter, which should be estimated by one of 
    the methods from actions_interfocal_distance_finder.h;
    \return     actions for the given point, or Jr=Jz=NAN if the energy is positive;
    \throw      std::invalid_argument exception if the potential is not axisymmetric 
    or some other error occurs.
*/
Actions actionsAxisymFudge(
    const potential::BasePotential& potential, 
    const coord::PosVelCyl& point,
    double interfocalDistance);

/** Find approximate actions and angles in a given axisymmetric potential, 
    using the Staeckel Fudge method.
    \param[in]  potential is the arbitrary axisymmetric potential;
    \param[in]  point     is the position/velocity point;
    \param[in]  interfocalDistance is the geometric parameter of best-fit coordinate system;
    \param[out] freq      if not NULL, store the frequencies of motion in this variable;
    \return     actions and angles for the given point, or Jr=Jz=NAN if the energy is positive;
    \throw      std::invalid_argument exception if the potential is not axisymmetric
    or some other error occurs.
*/
ActionAngles actionAnglesAxisymFudge(
    const potential::BasePotential& potential, 
    const coord::PosVelCyl& point, 
    double interfocalDistance, 
    Frequencies* freq=NULL);

#if 0 /* temporarily disabled */
/** Compute the total energy and the third integral for an orbit in a Staeckel potential
    from the given values of actions.
    \param[in]  potential  is the arbitrary Staeckel potential, separable in spheroidal coordinates;
    \param[in]  acts       are the actions;
    \param[out] H          is the value of Hamiltonian (total energy);
    \param[out] I3         is the value of the third integral corresponding to the given actions;
    \throw      std::invalid_argument if Jr/Jz actions are negative, or some other error occurs.
*/
void computeIntegralsStaeckel(
    const potential::OblatePerfectEllipsoid& potential,
    const Actions& acts,
    math::PtrFunction &rad, math::PtrFunction &ver);
#endif

///@}
/// \name  ------- Class interface to action/angle finders  -------
///@{

/** Action/angle finder for an Oblate Perfect Ellipsoid potential. */
class ActionFinderAxisymStaeckel: public BaseActionFinder {
public:
    explicit ActionFinderAxisymStaeckel(const potential::PtrOblatePerfectEllipsoid& potential) :
        pot(potential) {};

    virtual Actions actions(const coord::PosVelCyl& point) const {
        return actionsAxisymStaeckel(*pot, point); }

    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=NULL) const {
        return actionAnglesAxisymStaeckel(*pot, point, freq); }

private:
    const potential::PtrOblatePerfectEllipsoid pot;  ///< the potential in which actions are computed
};

/** Action/angle finder for a generic axisymmetric potential, based on Staeckel Fudge approximation.
    It is more suitable for massive computation in a fixed potential than just using 
    the standalone routines, because it estimates the interfocal distance using a pre-computed 
    interpolation grid, rather than doing it individually for each point. This results in 
    a considerable speedup in action computation, for a negligible overhead during initialization.
*/
class ActionFinderAxisymFudge: public BaseActionFinder {
public:
    explicit ActionFinderAxisymFudge(const potential::PtrPotential& potential) :
        pot(potential), finder(*potential) {};

    virtual Actions actions(const coord::PosVelCyl& point) const {
        return actionsAxisymFudge(*pot, point, 
            finder.value(totalEnergy(*pot, point), point.R*point.vphi)); }

    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=NULL) const {
        return actionAnglesAxisymFudge(*pot, point,
            finder.value(totalEnergy(*pot, point), point.R*point.vphi), freq); }

private:
    const potential::PtrPotential pot;      ///< the potential in which actions are computed
    const InterfocalDistanceFinder finder;  ///< fast estimator of interfocal distance
};

///@}
}  // namespace actions
