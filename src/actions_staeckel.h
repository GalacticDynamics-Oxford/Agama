#pragma once
#include "actions_base.h"
#include "potential_staeckel.h"

namespace actions{

    /// Action finder for the Axisymmetric Stackel Perfect Ellipsoid Potential
    class ActionFinderAxisymmetricStaeckel : public BaseActionFinder{
    private:
        const potential::StaeckelOblatePerfectEllipsoid& poten;
        //VecDoub find_limits(VecDoub_I x, VecDoub_I ints) const;
    public:
        ActionFinderAxisymmetricStaeckel(const potential::StaeckelOblatePerfectEllipsoid& _poten) :
            BaseActionFinder(), poten(_poten) {};
        virtual Actions actions(const coord::PosVelCar& point) const;
    };
}  // namespace actions
