/** \file    galaxymodel.h
    \brief   A complete galaxy model
    \date    2015
    \author  Eugene Vasiliev, Payel Das
*/
#pragma once
#include "potential_base.h"
#include "actions_base.h"
#include "df_base.h"
#include "particles_base.h"

/// Complete specification of a galaxy model
namespace galaxymodel{

/** Data-only structure defining a galaxy model: 
    a combination of potential, action finder, and distribution function */
class GalaxyModel{
public:
    const potential::BasePotential &potential;     ///< gravitational potential
    const actions::BaseActionFinder &actFinder;    ///< action finder for the given potential
    const df::BaseDistributionFunction &distrFunc; ///< distribution function expressed in terms of actions

    /** Create an instance of the galaxy model from the three ingredients */
    GalaxyModel(
        const potential::BasePotential& pot,
        const actions::BaseActionFinder& af,
        const df::BaseDistributionFunction& df) :
    potential(pot), actFinder(af), distrFunc(df) {}
};

/** Computes density, first-order, and second-order moments of velocity 
    in polar cyclindrical coordinates; if some of them are not needed,
    pass NULL as the corresponding argument, and it will not be computed.
    \param[in]  model  is the galaxy model (potential + DF + action finder);
    \param[in]  required relative error in the integral;
    \param[in]  maximum number of evaluations in integral;
    \param[out] density;
    \param[out] 1st-order velocity moments;
    \param[out] 2nd-order velocity moments;
    \param[out] density error;
    \param[out] 1st-order velocity moment errors;
    \param[out] 2nd-order velocity moment errors;
*/
void computeMoments(const GalaxyModel& model,
    const coord::PosCyl& point, const double reqRelError, const int maxNumEval,
    double* density, coord::VelCyl* velocityFirstMoment, coord::Vel2Cyl* velocitySecondMoment,
    double* densityErr, coord::VelCyl* velocityFirstMomentErr, coord::Vel2Cyl* velocitySecondMomentErr);


/** Generate N-body samples of the distribution function 
    by sampling in action/angle space:
    sample actions directly from DF and angles uniformly from [0:2pi]^3,
    then use torus machinery to convert from action/angles to position/velocity.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[out] points will contain array of particles (position/velocity/mass)
    sampled from the distribution function;
    \param[out] actions (optional) will be filled with values of actions
    corresponding to each point; if not needed may pass NULL as this argument.
*/
void generateActionSamples(const GalaxyModel& model,
    const unsigned int numPoints, particles::PointMassArrayCar &points,
    std::vector<actions::Actions>* actions=0);


/** Generate N-body samples of the distribution function 
    by sampling in position/velocity space:
    use action finder to compute the actions corresponding to the given point,
    and evaluate the value of DF at the given actions.
    \param[in]  model  is the galaxy model;
    \param[in]  numPoints  is the required number of samples;
    \param[out] points will contain array of particles (position/velocity/mass)
    sampled from the distribution function;
*/
void generatePosVelSamples(const GalaxyModel& model,
    const unsigned int numPoints, particles::PointMassArrayCar &points);


/** Sample the density profile by discrete points.
    \param[in]  dens  is the density model;
    \param[in]  numPoints  is the required number of sampling points;
    \param[out] points is the array to be filled with the sampled coordinates.
*/
void generateDensitySamples(const potential::BaseDensity& dens,
    const unsigned int numPoints, particles::PointMassArray<coord::PosCyl>& points);

}  // namespace