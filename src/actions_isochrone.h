/** \file    actions_isochrone.h
    \brief   Action-angle finder and mapper for the Isochrone potential
    \author  Eugene Vasiliev
    \date    2016
*/
#pragma once
#include "actions_base.h"

namespace actions{

Actions actionsIsochrone(
    const double isochroneMass, const double isochroneRadius,
    const coord::PosVelCyl& point);

/** Compute actions, angles and frequencies in a spherical Isochrone potential
    specified by total mass and scale radius.
    \param[in]  isochroneMass   is the total mass associated with the potential;
    \param[in]  isochroneRadius is the scale radius of the potential;
    \param[in]  point  is the position/velocity point;
    \param[out] freq   if not NULL, store the frequencies for these actions.
    \return     actions and angles, or Jr=NAN if the energy is positive.
*/
ActionAngles actionAnglesIsochrone(
    const double isochroneMass, const double isochroneRadius,
    const coord::PosVelCyl& point,
    Frequencies* freq=NULL);

/** Compute position/velocity from actions/angles in a spherical Isochrone potential.
    \param[in]  isochroneMass   is the total mass associated with the potential;
    \param[in]  isochroneRadius is the scale radius of the potential;
    \param[in]  actAng  is the action/angle point
    \param[out] freq    if not NULL, store the frequencies for these actions.
    \return     position and velocity point
*/
coord::PosVelCyl mapIsochrone(
    const double isochroneMass, const double isochroneRadius,
    const ActionAngles& actAng, Frequencies* freq=NULL);

/** Class for performing action/angle to coordinate/momentum transformation using 
    the Isochrone mapping expressed in modified spherical coordinates */
class ToyMapIsochrone: public BaseToyMap<coord::SphMod>{
public:
    const double M;  ///< total mass of isochrone potential
    const double b;  ///< scale radius of isochrone potential (may be zero in the Kepler case)
    ToyMapIsochrone(double isochroneMass, double isochroneRadius):
        M(isochroneMass), b(isochroneRadius) {};
    virtual coord::PosVelSphMod map(
        const ActionAngles& actAng,
        Frequencies* freq=NULL,
        DerivAct<coord::SphMod>* derivAct=NULL,
        DerivAng<coord::SphMod>* derivAng=NULL,
        coord::PosVelSphMod* derivParam=NULL) const;
};

}  // namespace actions