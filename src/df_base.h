/** \file    df_base.h
    \brief   Base class for action-based distribution functions
    \date    201?-2015
    \authors Payel Das, Eugene Vasiliev
*/
#pragma once
#include "actions_base.h"
#include <vector>

/** Classes for dealing with action-base distribution functions */
namespace df{

/** Base class defining the action-based distribution function (DF) */
class BaseDistributionFunction{
public:
    BaseDistributionFunction() {};
    virtual ~BaseDistributionFunction() {};

    /** Compute the total mass, i.e., integral of DF over the entire phase space.
        The actions range between 0 and +inf for Jr and Jz, and between -inf and +inf for Jphi,
        and there is an additional factor (2*pi)^3 from the integration over angles
        (we assume that DF does not depend on angles, but we still need to integrate
        over the entire 6d phase space).
        Derived classes may return an analytically computed value if available,
        and the default implementation performs multidimension integration numerically.
        \param[in]  reqRelError - relative tolerance;
        \param[in]  maxNumEval - maximum number of evaluations of DF during integration;
        \param[out] error (optional) if not NULL, store the estimate of the error;
        \param[out] numEval (optional) if not NULL, store the actual number of DF evaluations.
        \returns    total mass
    */
    virtual double totalMass(const double reqRelError=1e-3, const int maxNumEval=100000,
        double* error=0, int* numEval=0) const;

    /** Value of distribution function for the given set of actions
        \param[in] J - actions
    */
    virtual double value(const actions::Actions &J) const=0;
};

/** Sample the distribution function in actions.
    In other words, draw N sampling points from the action space, so that the density of points 
    in the neighborhood of any point is proportional to the value of DF at this point 
    (point = triplet of actions).
    \param[in]  DF  is the distribution function;
    \param[in]  numSamples  is the required number of sampling points;
    \param[out] samples is the array to be filled with the sampled actions.
    \param[out] totalMass (optional) if not NULL, will store the Monte Carlo estimate 
    of the integral of the distribution function (i.e., the same quantity as computed by 
    BaseDistributionFunction::totalMass(), but calculated with a different method).
    \param[out] totalMassErr (optional) if not NULL, will store the error estimate of the integral.
 */
void sampleActions(const BaseDistributionFunction& DF, const int numSamples,
    std::vector<actions::Actions>& samples, double* totalMass=0, double* totalMassErr=0);

/// convert from scaled variables to the actual actions to be passed to DF
/// if jac!=NULL, store the value of jacobian of transformation in this variable
actions::Actions unscaleActions(const double vars[], double* jac=0);

}  // namespace df
