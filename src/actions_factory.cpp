#include "actions_factory.h"
#include "actions_isochrone.h"
#include "actions_spherical.h"
#include "actions_staeckel.h"
#include "actions_torus.h"
#include "potential_analytic.h"
#include "potential_perfect_ellipsoid.h"

namespace actions {

void eval(const potential::BasePotential& pot, const coord::PosVelCyl& point,
    Actions* act, Angles* ang, Frequencies* freq, double focalDistance)
{
    const potential::Isochrone* potIso = dynamic_cast<const potential::Isochrone*>(&pot);
    if(potIso) {
        evalIsochrone(potIso->totalMass(), potIso->getRadius(), point, act, ang, freq);
        return;
    }

    if(isSpherical(pot)) {
        evalSpherical(pot, point, act, ang, freq);
        return;
    }

    const potential::OblatePerfectEllipsoid* potPE =
        dynamic_cast<const potential::OblatePerfectEllipsoid*>(&pot);
    if(potPE) {
        evalAxisymStaeckel(*potPE, point, act, ang, freq);
        return;
    }

    evalAxisymFudge(pot, point, act, ang, freq, focalDistance);
}

PtrActionFinder createActionFinder(const potential::PtrPotential& pot, bool interpolate)
{
    const potential::Isochrone* potIso = dynamic_cast<const potential::Isochrone*>(pot.get());
    if(potIso)
        return PtrActionFinder(new ActionFinderIsochrone(potIso->totalMass(), potIso->getRadius()));

    if(isSpherical(*pot))
       return PtrActionFinder(new actions::ActionFinderSpherical(*pot));

#if __cplusplus >= 201103L   // aliasing constructor for shared pointers only works in C++11
    const potential::OblatePerfectEllipsoid* potPE =
        dynamic_cast<const potential::OblatePerfectEllipsoid*>(pot.get());
    if(potPE)
        return PtrActionFinder(new actions::ActionFinderAxisymStaeckel(
            potential::PtrOblatePerfectEllipsoid(pot, potPE)));
#endif

    return PtrActionFinder(new actions::ActionFinderAxisymFudge(pot, interpolate));
}

PtrActionMapper createActionMapper(const potential::PtrPotential& pot, double tol)
{
    const potential::Isochrone* potIso = dynamic_cast<const potential::Isochrone*>(pot.get());
    if(potIso)
        return PtrActionMapper(new ActionMapperIsochrone(potIso->totalMass(), potIso->getRadius()));

    if(isSpherical(*pot))
        return PtrActionMapper(new actions::ActionMapperSpherical(*pot));

    if(tol==tol)  // non-default value for tol
        return PtrActionMapper(new actions::ActionMapperTorus(pot, tol));
    else
        return PtrActionMapper(new actions::ActionMapperTorus(pot /*, default_value_for_tol */));
}

}  // namespace actions
