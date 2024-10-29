#include "galaxymodel_base.h"
#include "galaxymodel_selfconsistent.h"
#include "actions_factory.h"
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
        DensityFromDF(GalaxyModel(totalPotential, actionFinder, *distrFunc), relError, maxNumEval),
        coord::ST_UNKNOWN, lmax, mmax, gridSizeR, rmin, rmax, /*fixOrder*/true);
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
        DensityFromDF(GalaxyModel(totalPotential, actionFinder, *distrFunc), relError, maxNumEval),
        coord::ST_UNKNOWN, mmax, gridSizeR, Rmin, Rmax, gridSizez, zmin, zmax, /*fixOrder*/true);
}


//------------ Driver routines for self-consistent modelling ------------//

static void updateActionFinder(SelfConsistentModel& model)
{
    // update the action finder after the potential has been reinitialized
    if(model.verbose)
        std::cout << "Updating action finder..."<<std::flush;
    model.actionFinder = actions::createActionFinder(
        model.totalPotential, model.useActionInterpolation);
    if(model.verbose)
        std::cout << "done" << std::endl;
}

void doIteration(SelfConsistentModel& model)
{
    // need to initialize the potential and the action finder before the first iteration
    if(!model.totalPotential)
        updateTotalPotential(model);
    else
        if(!model.actionFinder)
            updateActionFinder(model);

    for(unsigned int index=0; index<model.components.size(); index++) {
        // update the density of each component (this may be a no-op if the component is 'dead',
        // i.e. provides only a fixed density or potential, but does not possess a DF) -- 
        // the implementation is at the discretion of each component individually.
        if(model.verbose)
            std::cout << "Computing density for component "<<index<<"..."<<std::flush;
        model.components[index]->update(*model.totalPotential, *model.actionFinder);
        if(model.verbose)
            std::cout << "done"<<std::endl;
    }

    // now update the overall potential and reinit the action finder
    updateTotalPotential(model);
}

void updateTotalPotential(SelfConsistentModel& model)
{
    if(model.verbose)
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
            coord::ST_UNKNOWN, model.lmaxAngularSph, model.mmaxAngularSph,
            model.sizeRadialSph, model.rminSph, model.rmaxSph));

    // now the same for the total density to be used in CylSpline for the flattened components
    PtrDensity totalDensityDisk;
    if(compDensDisk.size()>1)
        totalDensityDisk.reset(new potential::CompositeDensity(compDensDisk));
    else if(compDensDisk.size()>0)
        totalDensityDisk = compDensDisk[0];

    if(totalDensityDisk != NULL)
        compPot.push_back(potential::CylSpline::create(*totalDensityDisk,
            coord::ST_UNKNOWN, model.mmaxAngularCyl,
            model.sizeRadialCyl,   model.RminCyl, model.RmaxCyl,
            model.sizeVerticalCyl, model.zminCyl, model.zmaxCyl));

    // now check if the total potential is elementary or composite
    if(compPot.size()==0)
        throw std::runtime_error("No potential is present in SelfConsistentModel");
    if(compPot.size()==1)
        model.totalPotential = compPot[0];
    else
        model.totalPotential.reset(new potential::Composite(compPot));

    if(model.verbose)
        std::cout << "done" << std::endl;
    // finally, create the action finder for the new potential
    updateActionFinder(model);
}

}  // namespace
