/** \file    actions_spherical.h
    \brief   Action-angle finders for a generic spherical potential
    \author  Eugene Vasiliev
    \date    2015-2016
*/
#pragma once
#include "actions_base.h"
#include "potential_utils.h"
#include "math_spline.h"
#include "smart.h"

namespace actions {

/** Compute any combination of actions, angles and frequencies in a given spherical potential.
    \param[in]  potential is an arbitrary spherical potential.
    \param[in]  point  is the position/velocity point.
    \param[out] act    if not NULL, will contain computed actions (Jr=NAN if E>=0).
    \param[out] ang    if not NULL, will contain corresponding angles (NAN if E>=0).
    \param[out] freq   if not NULL, will contain corresponding frequencies (NAN if E>=0).
    \throw      std::invalid_argument exception if the potential is not spherical.
*/
void evalSpherical(
    const potential::BasePotential& potential,
    const coord::PosVelCyl& point,
    Actions* act=NULL,
    Angles* ang=NULL,
    Frequencies* freq=NULL);


/** Compute the total energy for an orbit in a spherical potential from the given values of actions.
    \param[in]  potential  is the arbitrary spherical potential.
    \param[in]  acts       are the actions.
    \return     the value of Hamiltonian (total energy) corresponding to the given actions,
                or NAN if input actions Jr or Jz are negative.
    \throw      std::invalid_argument exception if the potential is not spherical.
*/
double computeHamiltonianSpherical(const potential::BasePotential& potential, const Actions& acts);


/** Compute position/velocity from actions/angles in an arbitrary spherical potential.
    \param[in]  potential   is the instance of a spherical potential.
    \param[in]  actAng  is the combination of actions and angles.
    \param[out] freq    if not NULL, store the frequencies for these actions.
    \return     position and velocity point (NAN if input actions Jr or Jz are negative).
    \throw      std::invalid_argument exception if the potential is not spherical.
*/
coord::PosVelCyl mapSpherical(
    const potential::BasePotential &potential,
    const ActionAngles &actAng,
    Frequencies* freq=NULL);


/** Class for performing bidirectional transformations between action/angle and coordinate/momentum
    for an arbitrary spherical potential, using 2d interpolation tables */
class ActionFinderSpherical: public BaseActionFinder, public BaseActionMapper {
public:
    /// Initialize the internal interpolation tables; the potential itself is not used later on.
    /// \note OpenMP-parallelized loop over the radial grid in construction Interpolator2d
    /// from the input potential, and two other parallelized loops using this interpolator.
    explicit ActionFinderSpherical(const potential::BasePotential& potential);

    virtual std::string name() const { return myName; }

    virtual void eval(const coord::PosVelCyl& point,
        Actions* act=NULL, Angles* ang=NULL, Frequencies* freq=NULL) const;

    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const;

    /** return the interpolated value of radial action as a function of energy and angular momentum;
        also return the frequencies in Omegar and Omegaz if these arguments are not NULL */
    double Jr(double E, double L, double *Omegar=NULL, double *Omegaz=NULL) const;

    /** return the energy corresponding to the given actions */
    double E(const Actions& act) const;
private:
    const double invPhi0;                 ///< 1/(value of potential at r=0)
    const potential::Interpolator2d pot;  ///< interpolator for potential and peri/apocenter radii
    const math::QuinticSpline2d intJr;    ///< interpolator for Jr(E,L)
    const math::QuinticSpline2d intE;     ///< interpolator for E(Jr,L)
    const std::string myName;             ///< store the name of this action finder/mapper
};

// this class performs the conversion in both directions
typedef ActionFinderSpherical ActionMapperSpherical;

}  // namespace actions
