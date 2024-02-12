/** \file    actions_isochrone.h
    \brief   Action-angle finder and mapper for the Isochrone potential
    \author  Eugene Vasiliev
    \date    2016
*/
#pragma once
#include "actions_base.h"

namespace actions{

/** Compute any combination of actions, angles and frequencies
    in a spherical Isochrone potential specified by its total mass and scale radius.
    \param[in]  isochroneMass   is the total mass associated with the potential.
    \param[in]  isochroneRadius is the scale radius of the potential.
    \param[in]  point  is the position/velocity point.
    \param[out] act    if not NULL, will contain computed actions (Jr=NAN if E>=0).
    \param[out] ang    if not NULL, will contain corresponding angles (NAN if E>=0).
    \param[out] freq   if not NULL, will contain corresponding frequencies (NAN if E>=0).
*/
void evalIsochrone(
    const double isochroneMass, const double isochroneRadius,
    const coord::PosVelCyl& point,
    Actions* act=NULL,
    Angles* ang=NULL,
    Frequencies* freq=NULL);

/** Compute position/velocity from actions/angles in a spherical Isochrone potential.
    \param[in]  isochroneMass   is the total mass associated with the potential.
    \param[in]  isochroneRadius is the scale radius of the potential.
    \param[in]  actAng  is the action/angle point
    \param[out] freq    if not NULL, store the frequencies for these actions.
    \return     position and velocity point; NAN if Jr<0 or Jz<0.
*/
coord::PosVelCyl mapIsochrone(
    const double isochroneMass, const double isochroneRadius,
    const ActionAngles& actAng,
    Frequencies* freq=NULL);


/** Class for performing transformations between action/angle and coordinate/momentum for
    an isochrone potential (a trivial wrapper for the corresponding standalone functions) */
class ActionFinderIsochrone: public BaseActionFinder, public BaseActionMapper {
public:
    ActionFinderIsochrone(double _mass, double _radius): mass(_mass), radius(_radius) {}

    virtual std::string name() const;

    virtual void eval(const coord::PosVelCyl& point,
        Actions* act=NULL, Angles* ang=NULL, Frequencies* freq=NULL) const
    { evalIsochrone(mass, radius, point, act, ang, freq); }

    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const
    { return mapIsochrone (mass, radius, actAng, freq); }

private:
    const double mass, radius;  ///< parameters of the isochrone potential
};

// this class performs the conversion in both directions
typedef ActionFinderIsochrone ActionMapperIsochrone;

}  // namespace actions
