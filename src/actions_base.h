#pragma once
#include "coord.h"

namespace actions{

struct Actions {
    double Jr;       ///< radial action or its analog
    double Jz;       ///< vertical action or its analog
    double Jphi;     ///< azimuthal action (z-component of angular momentum in axisymmetric case)
//    Actions() {};
//    Actions(double _Jr, double _Jz, double _Jphi) : Jr(_Jr), Jz(_Jz), Jphi(_Jphi) {};
};

struct Angles {
    double thetar;   ///< phase angle of radial motion
    double thetaz;   ///< phase angle of vertical motion
    double thetaphi; ///< phase angle of azimuthal motion
//    Angles() {};
//    Angles(double _thetar, double _thetaz, double _thetaphi) :
//        thetar(angs.thetar), thetaz(angs.thetaz), thetaphi(angs.thetaphi) {};
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

    /** Evaluate actions for a given position/velocity point in cartesian coordinates */
    virtual Actions actions(const coord::PosVelCyl& point) const = 0;
};
}  // namespace action