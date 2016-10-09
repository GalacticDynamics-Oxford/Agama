/** \file    actions_torus.h
    \brief   Wrapper for Torus code
    \author  Eugene Vasiliev
    \date    2015

*/
#pragma once
#include "actions_base.h"
#include "potential_base.h"
#include "smart.h"

namespace actions {

/** Wrapper for Paul McMillan's Torus code */
class ActionMapperTorus: public BaseActionMapper{
public:
    /** Construct a torus for the given axisymmetric potential and given values of actions;
        the potential is not subsequently used. */
    ActionMapperTorus(const potential::BasePotential& poten, const Actions& acts, double tol=0.003);

    /** Map a point in action/angle space to a position/velocity in physical space.
        Note that for this class, the values of actions are set at the constructor;
        an attempt to call this function with different set of actions will result in 
        a `std::invalid_argument` exception. */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const;
private:
    Torus::PtrTorus torus;  ///< hidden implementation details
};

}  // namespace actions
