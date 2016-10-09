/** \file    actions_newtorus.h
    \brief   New implementation of Torus mapping
    \author  Eugene Vasiliev
    \date    Feb 2016

*/
#pragma once
#include "actions_base.h"
#include "potential_base.h"
#include "smart.h"

namespace actions {

typedef shared_ptr<const BaseCanonicalMap> PtrCanonicalMap;
typedef shared_ptr<const BaseToyMap<coord::SphMod> > PtrToyMap;
typedef shared_ptr<const BasePointTransform<coord::SphMod> > PtrPointTransform;

class ActionMapperNewTorus: public BaseActionMapper{
public:
    /** Construct a torus for the given axisymmetric potential and given values of actions;
        the potential is not subsequently used. */
    ActionMapperNewTorus(const potential::BasePotential& pot, const Actions& acts, double tol=0.003);

    /** Map a point in action/angle space to a position/velocity in physical space.
        Note that for this class, the values of actions are set at the constructor;
        an attempt to call this function with different set of actions will result in 
        a `std::invalid_argument` exception. */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const;
private:
    const Actions acts;     ///< the values of actions for this torus
    Frequencies freqs;      ///< frequencies (dH/dJ evaluated at these actions)
    bool converged;         ///< flag indicating whether the torus construction was successful
    PtrCanonicalMap genFnc; ///< generating function that converts real to toy action/angles
    PtrToyMap toyMap;       ///< toy map that converts toy action/angles to coord/momenta
    PtrPointTransform pointTrans;  ///< point transformation from coord/momenta to cylindrical pos/vel
};
    
}  // namespace actions
