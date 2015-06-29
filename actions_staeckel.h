#pragma once
#include "actions_base.h"
#include "potential_staeckel.h"

namespace actions{

    /// integrals of motion in an axisymmetric potential
    struct AxisymIntegrals {
        double H;  ///< total energy
        double Lz; ///< z-component of angular momentum
        double I3; ///< third integral
    };

    /// compute integrals of motion in the Staeckel potential of an oblate perfect ellipsoid
    AxisymIntegrals findIntegralsOfMotionOblatePerfectEllipsoid
        (const potential::StaeckelOblatePerfectEllipsoid& poten, const coord::PosVelCyl& point);

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
