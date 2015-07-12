/** \brief   Action-angle finders using Staeckel potential approximation
    \author  Eugene Vasiliev
    \date    2015
 
Computation of actions and angles for axisymmetric systems, using two related methods:
1) exact expressions for Staeckel potentials  (e.g. for the Perfect Ellipsoid),
2) "Staeckel Fudge" approximation applicable for any axisymmetric potential.
To avoid confusion, the latter is called simply "Fudge" in what follows.

Most sub-steps are shared between the two methods; 
only the computation of integrals of motion, and the auxiliary function that enters 
the expression for canonical momentum, are specific to each method.

The implementation is inspired by the code written by Jason Sanders, but virtually nothing 
of the original code remains.
*/
#pragma once
#include "actions_base.h"
#include "potential_staeckel.h"

namespace actions{

/// Action finder for the Axisymmetric Stackel Perfect Ellipsoid Potential
class ActionFinderAxisymmetricStaeckel : public BaseActionFinder{
private:
    const potential::StaeckelOblatePerfectEllipsoid& poten;
public:
    ActionFinderAxisymmetricStaeckel(const potential::StaeckelOblatePerfectEllipsoid& _poten) :
        BaseActionFinder(), poten(_poten) {};
    virtual Actions actions(const coord::PosVelCyl& point) const;
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point) const;
};

/// Action finder for arbitrary axisymmetric potential using Jason Sander's method
class ActionFinderAxisymmetricFudgeJS : public BaseActionFinder{
private:
    const potential::BasePotential& poten;
    double alpha, gamma;
public:
    ActionFinderAxisymmetricFudgeJS(const potential::BasePotential& _poten, double _alpha, double _gamma) :
        BaseActionFinder(), poten(_poten), alpha(_alpha), gamma(_gamma) {};
    virtual Actions actions(const coord::PosVelCyl& point) const;
    virtual ActionAngles actionAngles(const coord::PosVelCyl& point) const;
};

}  // namespace actions
