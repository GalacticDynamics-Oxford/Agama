/** \file    actions_base.h
    \brief   Base classes for actions, angles, and action/angle finders
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once
#include "coord.h"
#include <string>

/** Classes and routines for transformations between position/velocity and action/angle phase spaces */
namespace actions {

/** Actions in arbitrary potential */
struct Actions {
    double Jr;       ///< radial action or its analog, [0..infinity)
    double Jz;       ///< vertical action or its analog, [0..infinity)
    double Jphi;     ///< azimuthal action (equal to the z-component of angular momentum in
                     ///< axisymmetric case, can have any value)
    Actions() {}
    Actions(double _Jr, double _Jz, double _Jphi) : Jr(_Jr), Jz(_Jz), Jphi(_Jphi) {}
};

/** Angles in arbitrary potential */
struct Angles {
    double thetar;   ///< phase angle of radial motion
    double thetaz;   ///< phase angle of vertical motion
    double thetaphi; ///< phase angle of azimuthal motion
    Angles() {}
    Angles(double tr, double tz, double tphi) : thetar(tr), thetaz(tz), thetaphi(tphi) {}
};

/** A combination of both actions and angles */
struct ActionAngles: Actions, Angles {
    ActionAngles() {}
    ActionAngles(const Actions& acts, const Angles& angs) : Actions(acts), Angles(angs) {}
};

/** Frequencies of motion (Omega = dH/dJ) */
struct Frequencies {
    double Omegar;    ///< frequency of radial motion, dH/dJr
    double Omegaz;    ///< frequency of vertical motion, dH/dJz
    double Omegaphi;  ///< frequency of azimuthal motion, dH/dJphi
    Frequencies() {}
    Frequencies(double omr, double omz, double omphi) : Omegar(omr), Omegaz(omz), Omegaphi(omphi) {}
};


/** Base class for action finders, which convert position/velocity pair to an action/angle pair */
class BaseActionFinder{
public:
    BaseActionFinder() {}
    virtual ~BaseActionFinder() {}

    /** Return the name of the particular implementation of the action finder */
    virtual std::string name() const = 0;

    /** Evaluate any combination of actions, angles and frequencies for the given phase-space point.
        \param[in]  point  is a position/velocity point in cylindrical coordinates.
        \param[out] act  if not NULL, will compute and store actions in this variable.
        \param[out] ang  if not NULL, will compute and store angles in this variable.
        \param[out] freq if not NULL, will compute and store frequencies in this variable.
        In case of unbound orbit, at least radial action/angle/frequency and possibly other ones
        should contain NAN; the method does not throw exceptions.
    */
    virtual void eval(const coord::PosVelCyl& point,
        Actions* act=NULL, Angles* ang=NULL, Frequencies* freq=NULL) const = 0;

    /** Evaluate actions (a shortcut).
        \param[in]  point  is a position/velocity point in cylindrical coordinates.
        \return  a triplet of actions.
    */
    Actions actions(const coord::PosVelCyl& point) const {
        Actions act;
        eval(point, &act);
        return act;
    }

private:
    /// disable copy constructor and assignment operator
    BaseActionFinder(const BaseActionFinder&);
    BaseActionFinder& operator= (const BaseActionFinder&);
};

/** Base class for action/angle mappers, which convert action/angle variables
    to a position/velocity point */
class BaseActionMapper{
public:
    BaseActionMapper() {}
    virtual ~BaseActionMapper() {}

    /** return the name of the particular implementation of the action mapper */
    virtual std::string name() const = 0;

    /** Map a point in action/angle space to a position/velocity in physical space.
        \param[in]  actAng is a combination of actions and angles.
        \param[out] freq  if not NULL, store frequencies in this variable.
    */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const = 0;

private:
    /// disable copy constructor and assignment operator
    BaseActionMapper(const BaseActionMapper&);
    BaseActionMapper& operator= (const BaseActionMapper&);
};

}  // namespace action