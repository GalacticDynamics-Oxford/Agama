#pragma once
#include "coord.h"

namespace actions{

struct Actions{
    double Jr;   ///< radial action or its analog
    double Jz;   ///< vertical action or its analog
    double Jphi; ///< azimuthal action (z-component of angular momentum in axisymmetric case)
};

struct ActionAngles: public Actions{
    double thetar;
    double thetaz;
    double thetaphi;
};

/** Base class for action finders */
class BaseActionFinder{
public:
    BaseActionFinder() {};
    virtual ~BaseActionFinder() {};

    /** Evaluate actions for a given position/velocity point in cartesian coordinates */
    virtual Actions actions(const coord::PosVelCar& point) const = 0;
};
}  // namespace action