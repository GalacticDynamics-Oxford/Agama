/** \file    actions_base.h
    \brief   Base classes for actions, angles, and action/angle finders
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once
#include "coord.h"

/** Classes and routines for transformations between position/velocity and action/angle phase spaces */
namespace actions {

/** Actions in arbitrary potential */
struct Actions {
    double Jr;       ///< radial action or its analog, [0..infinity)
    double Jz;       ///< vertical action or its analog, [0..infinity)
    double Jphi;     ///< azimuthal action (equal to the z-component of angular momentum 
                     ///< in axisymmetric case, can have any value)
};

/** Angles in arbitrary potential */
struct Angles {
    double thetar;   ///< phase angle of radial motion
    double thetaz;   ///< phase angle of vertical motion
    double thetaphi; ///< phase angle of azimuthal motion
};

/** A combination of both actions and angles */
struct ActionAngles: Actions, Angles {
    ActionAngles() {};
    ActionAngles(const Actions& acts, const Angles& angs) : Actions(acts), Angles(angs) {};
};

/** Base class for action finders */
class BaseActionFinder{
public:
    BaseActionFinder() {};
    virtual ~BaseActionFinder() {};

    /** Evaluate actions for a given position/velocity point in cylindrical coordinates */
    virtual Actions actions(const coord::PosVelCyl& point) const = 0;

    /** Evaluate actions and angles for a given position/velocity point in cylindrical coordinates */
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point) const = 0;
};

}  // namespace action