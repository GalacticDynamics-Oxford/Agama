/** \file    actions_torus.h
    \brief   Wrapper for Torus code
    \author  Eugene Vasiliev
    \date    2015-2024
*/
#pragma once
#include "actions_base.h"
#include "potential_base.h"
#include "smart.h"

namespace actions {

/** Wrapper for Paul McMillan's Torus code.
    This class internally creates and caches instances of Torus objects
    for each unique triplet of actions passed to its map() method.
*/
class ActionMapperTorus: public BaseActionMapper{
public:
    /** Initializes the placeholder for torus mapper(s) for the given axisymmetric potential.
        \param[in] potential is the potential used to create tori on-demand.
        \param[in] tol (optional) specifies the accuracy for torus mapper.
        \throw std::invalid_argument exception if the potential is not axisymmetric.
    */
    ActionMapperTorus(const potential::PtrPotential& potential, double tol=0.003);

    ~ActionMapperTorus();

    virtual std::string name() const;

    /** Map a point in action/angle space to a position/velocity point.
        If the triplet of actions has been provided in one of earlier calls,
        the corresponding Torus object will be retrieved from a cache,
        otherwise a new Torus will be created and stored in the cache table.
        \note this method is not thread-safe because of fatal flaws in the Torus implementation!
    */
    virtual coord::PosVelCyl map(const ActionAngles& actAng, Frequencies* freq=NULL) const;

private:
    class Impl;
    Impl* impl;
};

}  // namespace actions
