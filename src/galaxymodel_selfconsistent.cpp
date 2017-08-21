#include "galaxymodel_selfconsistent.h"
#include "galaxymodel.h"
#include "actions_staeckel.h"
#include "actions_spherical.h"
#include "potential_composite.h"
#include "potential_multipole.h"
#include "potential_cylspline.h"
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <iostream>

namespace galaxymodel{

using potential::PtrDensity;
using potential::PtrPotential;

template<typename T>
const T& ensureNotNull(const T& x) {
    if(x) return x;
    throw std::invalid_argument("NULL pointer in assignment");
}

namespace{

/// Helper class for providing a BaseDensity interface to a density computed via integration over DF
class DensityFromDF: public potential::BaseDensity{
public:
    DensityFromDF(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df,
        double _relError, unsigned int _maxNumEval) :
    model(pot, af, df), relError(_relError), maxNumEval(_maxNumEval) {};

    virtual coord::SymmetryType symmetry() const { return coord::ST_AXISYMMETRIC; }
    virtual const char* name() const { return myName(); };
    static const char* myName() { return "DensityFromDF"; };
    virtual double enclosedMass(const double) const {  // should never be used -- too slow
        throw std::runtime_error("DensityFromDF: enclosedMass not implemented"); }
private:
    const GalaxyModel model;  ///< aggregate of potential, action finder and DF
    double       relError;    ///< requested relative error of density computation
    unsigned int maxNumEval;  ///< max # of DF evaluations per one density calculation

    virtual double densityCar(const coord::PosCar &pos) const {
        return densityCyl(toPosCyl(pos)); }

    virtual double densitySph(const coord::PosSph &pos) const {
        return densityCyl(toPosCyl(pos)); }

    /// compute the density as the integral of DF over velocity at a given position
    virtual double densityCyl(const coord::PosCyl &point) const {
        double result;
        computeMoments(model, point, &result, NULL, NULL, NULL, NULL, NULL, relError, maxNumEval);
        return result;
    }
};
} // anonymous namespace

//--------- Components with DF ---------//

ComponentWithSpheroidalDF::ComponentWithSpheroidalDF(
    const df::PtrDistributionFunction& df,
    const potential::PtrDensity& initDensity,
    unsigned int _lmax, unsigned int _mmax, unsigned int _gridSizeR, double _rmin, double _rmax,
    double _relError, unsigned int _maxNumEval)
:
    BaseComponentWithDF(ensureNotNull(df), initDensity, false, _relError, _maxNumEval),
    lmax(_lmax), mmax(_mmax), gridSizeR(_gridSizeR), rmin(_rmin), rmax(_rmax)
{}

void ComponentWithSpheroidalDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    density = potential::DensitySphericalHarmonic::create(
        DensityFromDF(totalPotential, actionFinder, *distrFunc, relError, maxNumEval),
        lmax, mmax, gridSizeR, rmin, rmax, false /*use exactly the requested order*/);
}

ComponentWithDisklikeDF::ComponentWithDisklikeDF(
    const df::PtrDistributionFunction& df,
    const potential::PtrDensity& initDensity,
    unsigned int _mmax,
    unsigned int _gridSizeR, double _Rmin, double _Rmax, 
    unsigned int _gridSizez, double _zmin, double _zmax,
    double _relError, unsigned int _maxNumEval)
:
    BaseComponentWithDF(ensureNotNull(df), initDensity, true, _relError, _maxNumEval),
    mmax(_mmax), gridSizeR(_gridSizeR), Rmin(_Rmin), Rmax(_Rmax),
    gridSizez(_gridSizez), zmin(_zmin), zmax(_zmax)
{}

void ComponentWithDisklikeDF::update(
    const potential::BasePotential& totalPotential,
    const actions::BaseActionFinder& actionFinder)
{
    density = potential::DensityAzimuthalHarmonic::create(
        DensityFromDF(totalPotential, actionFinder, *distrFunc, relError, maxNumEval),
        mmax, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax, false /*respect the expansion order*/);
}


//------------ Driver routines for self-consistent modelling ------------//

void doIteration(SelfConsistentModel& model)
{
    // need to initialize the potential before the first iteration
    if(!model.totalPotential)
        updateTotalPotential(model);

    for(unsigned int index=0; index<model.components.size(); index++) {
        // update the density of each component (this may be a no-op if the component is 'dead',
        // i.e. provides only a fixed density or potential, but does not possess a DF) -- 
        // the implementation is at the discretion of each component individually.
        std::cout << "Computing density for component "<<index<<"..."<<std::flush;
        model.components[index]->update(*model.totalPotential, *model.actionFinder);
        std::cout << "done"<<std::endl;
    }

    // now update the overall potential and reinit the action finder
    updateTotalPotential(model);
}

void updateTotalPotential(SelfConsistentModel& model)
{
    std::cout << "Updating potential..."<<std::flush;

    // temporary array of density and potential objects from components
    std::vector<PtrDensity> compDensSph;
    std::vector<PtrDensity> compDensDisk;
    std::vector<PtrPotential> compPot;

    // first retrieve non-zero density and potential objects from all components
    for(unsigned int i=0; i<model.components.size(); i++) {
        PtrDensity den = model.components[i]->getDensity();
        if(den) {
            if(model.components[i]->isDensityDisklike)
                compDensDisk.push_back(den);
            else
                compDensSph.push_back(den);
        }
        PtrPotential pot = model.components[i]->getPotential();
        if(pot)
            compPot.push_back(pot);
    }

    // the total density to be used in multipole expansion for spheroidal components
    PtrDensity totalDensitySph;
    // if more than one density component is present, create a temporary composite density object;
    if(compDensSph.size()>1)
        totalDensitySph.reset(new potential::CompositeDensity(compDensSph));
    else if(compDensSph.size()>0)
    // if only one component is present, simply copy it;
        totalDensitySph = compDensSph[0];
    // otherwise don't use multipole expansion at all

    // construct potential expansion from the total density
    // and add it as one of potential components (possibly the only one)
    if(totalDensitySph != NULL)
        compPot.push_back(potential::Multipole::create(*totalDensitySph,
            model.lmaxAngularSph, model.mmaxAngularSph,
            model.sizeRadialSph, model.rminSph, model.rmaxSph));

    // now the same for the total density to be used in CylSpline for the flattened components
    PtrDensity totalDensityDisk;
    if(compDensDisk.size()>1)
        totalDensityDisk.reset(new potential::CompositeDensity(compDensDisk));
    else if(compDensDisk.size()>0)
        totalDensityDisk = compDensDisk[0];

    if(totalDensityDisk != NULL)
        compPot.push_back(potential::CylSpline::create(*totalDensityDisk, model.mmaxAngularCyl,
            model.sizeRadialCyl,   model.RminCyl, model.RmaxCyl,
            model.sizeVerticalCyl, model.zminCyl, model.zmaxCyl, true /*use derivs*/));

    // now check if the total potential is elementary or composite
    if(compPot.size()==0)
        throw std::runtime_error("No potential is present in SelfConsistentModel");
    if(compPot.size()==1)
        model.totalPotential = compPot[0];
    else
        model.totalPotential.reset(new potential::CompositeCyl(compPot));

    // update the action finder
    std::cout << "done\nUpdating action finder..."<<std::flush;
    if(isSpherical(*model.totalPotential))
        model.actionFinder.reset(new actions::ActionFinderSpherical(*model.totalPotential));
    else
        model.actionFinder.reset(
            new actions::ActionFinderAxisymFudge(model.totalPotential, model.useActionInterpolation));
    std::cout << "done"<<std::endl;
}

}  // namespace
