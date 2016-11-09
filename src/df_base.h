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
    virtual double totalMass(const double reqRelError=1e-5, const int maxNumEval=1e6,
        double* error=NULL, int* numEval=NULL) const;

    /** Value of distribution function for the given set of actions
        \param[in] J - actions
    */
    virtual double value(const actions::Actions &J) const=0;
};

/** Base class for multi-component distribution functions */
class BaseMulticomponentDF: public BaseDistributionFunction{
public:
    /// the number of components
    virtual unsigned int size() const = 0;

    /// value of the given DF component at the given actions
    virtual double valueOfComponent(const actions::Actions &J, unsigned int index) const = 0;

    /** values of all components at the given actions:
        \param[in]  J are the actions;
        \param[out] values will contain the values of all components,
        must point to an existing array of sufficient length.
    */
    virtual void valuesOfAllComponents(const actions::Actions &J, double values[]) const {
        for(unsigned int i=0; i<size(); i++)
            values[i] = valueOfComponent(J, i);
    }

    /// total value of multi-component DF is the sum of all components
    virtual double value(const actions::Actions &J) const {
        double sum=0;
        for(unsigned int i=0; i<size(); i++)
            sum += valueOfComponent(J, i);
        return sum;
    }
};

/** Helper class for scaling transformations in the action space,
    mapping the entire possible range of actions into a unit cube */
class BaseActionSpaceScaling {
public:
    virtual ~BaseActionSpaceScaling() {}

    /** Convert from scaled variables to the values of actions.
        \param[in]  vars  are the coordinates in the unit cube;
        \param[out] jac   if not NULL, will contain the value of jacobian of transformation;
        \return  the values of actions.
    */
    virtual actions::Actions toActions(const double vars[3], double *jac=NULL) const = 0;

    /** Convert from actions to scaled variables.
        \param[in]  acts  are the actions;
        \param[out] vars  will contain the scaled variables in the unit cube;
    */
    virtual void toScaled(const actions::Actions &acts, double vars[3]) const = 0;
};

class ActionSpaceScalingTriangLog: public BaseActionSpaceScaling {
public:
    virtual actions::Actions toActions(const double vars[3], double *jac=NULL) const;
    virtual void toScaled(const actions::Actions &acts, double vars[3]) const;
};

class ActionSpaceScalingRect: public BaseActionSpaceScaling {
    double scaleJm, scaleJphi;
public:
    ActionSpaceScalingRect(double scaleJm, double scaleJphi);
    virtual actions::Actions toActions(const double vars[3], double *jac=NULL) const;
    virtual void toScaled(const actions::Actions &acts, double vars[3]) const;
};


/** Compute the entropy  \f$  S = -\int d^3 J f(J) ln(f(J))  \f$.
    \param[in]  DF is the distribution function;
    \param[in]  reqRelError - relative tolerance;
    \param[in]  maxNumEval - maximum number of evaluations of DF during integration;
    \return  the value of entropy S.
*/
double totalEntropy(const BaseDistributionFunction& DF,
    const double reqRelError=1e-4, const int maxNumEval=1e6);


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
    std::vector<actions::Actions>& samples, double* totalMass=NULL, double* totalMassErr=NULL);

}  // namespace df
