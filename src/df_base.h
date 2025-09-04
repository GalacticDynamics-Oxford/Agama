/** \file    df_base.h
    \brief   Base class for action-based distribution functions
    \authors Eugene Vasiliev, Payel Das
    \date    2015-2017
*/
#pragma once
#include "actions_base.h"
#include <vector>

/** Classes for dealing with action-base distribution functions */
namespace df{

/** Derivatives of the DF with respect to actions */
struct DerivByActions {
    double dbyJr, dbyJz, dbyJphi;
};


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
        \param[in]  reqRelError - relative tolerance.
        \param[in]  maxNumEval - maximum number of evaluations of DF during integration.
        \returns    total mass
    */
    virtual double totalMass(const double reqRelError=1e-6, const int maxNumEval=1e6) const;

    /** Number of components in the case of a multi-component DF */
    virtual unsigned int numValues() const { return 1; }

    /** Evaluate the distribution function and optionally its derivatives w.r.t. actions,
        to be implemented in the derived classes.
        \param[in] J - a triplet of actions.
        \param[out] value - pointer to the variable that will store the DF value (non-NULL).
        \param[out] der (optional) - if not NULL, will store the DF derivatives w.r.t. actions.
    */
    virtual void evalDeriv(const actions::Actions &J,
        /*output*/ double* value, DerivByActions *der=NULL) const=0;

    /** Shortcut for getting the value of distribution function for the given set of actions J:
        in case than numValues>1, return a single value - the sum of all components */
    double value(const actions::Actions &J) const {
        double result;
        evalDeriv(J, &result);
        return result;
    }

    /** Vectorized evaluation of the DF for several input points at once, possibly reporting
        multiple values for a single point (if numValues>1 and separate output is requested).
        \param[in]  npoints - size of the input array of actions;
        \param[in]  J - array of actions of length npoints;
        \param[in]  separate - if numValues>1, this flag indicates whether to output a sum of
        all components (if separate is false) or each of them separately (if true);
        \param[out] values - output array that will be filled with DF values as follows:
        if separate is false, the length of output is npoints, and each element of the output
        array corresponds exactly to one input point;
        if separate is true, the length of output is npoints * numValues, and
        values[p * numValues + c] contains the value of c-th component at p-th input point.
        \param[out] derivs (optional) - if not NULL, should point to an array that will be 
        filled with DF derivatives in the same order as values (i.e., npoints if separate=false,
        or npoints * numValues if separate=true).
    */
    virtual void evalmany(const size_t npoints, const actions::Actions J[],
        bool /*separate*/, double values[], DerivByActions derivs[]=NULL) const
    {
        // default implementation for a single-component DF does not make a distinction between
        // separate or combined evaluation, and just loops over input points one by one
        for(size_t p=0; p<npoints; p++)
            evalDeriv(J[p], values+p, derivs? derivs+p : NULL);
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

}  // namespace df
