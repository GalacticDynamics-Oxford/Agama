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
    Actions() {};
    Actions(double _Jr, double _Jz, double _Jphi) : Jr(_Jr), Jz(_Jz), Jphi(_Jphi) {};
};

/** Angles in arbitrary potential */
struct Angles {
    double thetar;   ///< phase angle of radial motion
    double thetaz;   ///< phase angle of vertical motion
    double thetaphi; ///< phase angle of azimuthal motion
    Angles() {};
    Angles(double tr, double tz, double tphi) : thetar(tr), thetaz(tz), thetaphi(tphi) {};
};

/** A combination of both actions and angles */
struct ActionAngles: Actions, Angles {
    ActionAngles() {};
    ActionAngles(const Actions& acts, const Angles& angs) : Actions(acts), Angles(angs) {};
};

/** Frequencies of motion (Omega = dH/dJ) */
struct Frequencies {
    double Omegar;    ///< frequency of radial motion, dH/dJr
    double Omegaz;    ///< frequency of vertical motion, dH/dJz
    double Omegaphi;  ///< frequency of azimuthal motion, dH/dJphi
    Frequencies() {};
    Frequencies(double omr, double omz, double omphi) : Omegar(omr), Omegaz(omz), Omegaphi(omphi) {};
};

/** Base class for action finders, which convert position/velocity pair to action/angle pair */
class BaseActionFinder{
public:
    BaseActionFinder() {};
    virtual ~BaseActionFinder() {};

    /** Evaluate actions for a given position/velocity point in cylindrical coordinates */
    virtual Actions actions(const coord::PosVelCyl& point) const = 0;

    /** Evaluate actions and angles for a given position/velocity point in cylindrical coordinates;
        if the output argument freq!=NULL, also store the frequencies */
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point, Frequencies* freq=0) const = 0;
};

/** Base class for action/angle mappers, which convert action/angle variables to position/velocity point */
class BaseActionMapper{
public:
    BaseActionMapper() {};
    virtual ~BaseActionMapper() {};

    /** Map a point in action/angle space to a position/velocity in physical space;
        if the output argument freq!=NULL, also store the frequencies */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=0) const = 0;
};

}  // namespace action