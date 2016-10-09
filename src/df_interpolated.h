/** \file    df_interpolated.h
    \brief   Distribution function specified in the interpolated form
    \date    2016
    \author  Eugene Vasiliev
*/
#pragma once
#include "df_base.h"
#include "math_spline.h"
#include "smart.h"

namespace df{

/** Parameters that describe an interpolated distribution function.
    It is represented on a 3d grid in scaled coordinates in the action space,
    where the first variable corresponds to the total action (sum of three actions)
    with a logarithmic transformation:  x = log(1 + (Jr+Jz+|Jphi|) / J0),
    and the other two variables range from 0 to 1 and determine the ratio between actions.
    The array of amplitudes defines the value of interpolated DF;
    in the case of linear interpolation, the amplitudes correspond to the values of DF at grid nodes,
    but in the case of cubic interpolation there is no straightforward correspondence between them;
    moreover, in the latter case the number of components (elements in the amplitudes array)
    is larger than the number of nodes in the 3d grid.
*/

template<int N>
class InterpolatedDF: public BaseMulticomponentDF{
public:
    /** Create an instance of interpolated distribution function with the given parameters.
        \param[in] scaling  is the instance of class that performs scaling transformation
        in action space, mapping the entire range of actions into a unit cube;
        \param[in] gridU, gridV, gridW  are the nodes of 1d grids in each of the scaled coordinate;
        \param[in] amplitudes  is the flattened array of amplitudes of basis functions;
        \throws std::invalid_argument exception if parameters are nonsense.
    */
    InterpolatedDF(const PtrActionSpaceScaling& scaling,
        const std::vector<double> &gridU, const std::vector<double> &gridV,
        const std::vector<double> &gridW, const std::vector<double> &amplitudes);

    /// the value of interpolated DF at the given actions
    virtual double value(const actions::Actions &J) const;

    /// the number of components in the interpolation array
    virtual unsigned int size() const { return amplitudes.size(); }

    /// the value of a single component at the given actions
    virtual double valueOfComponent(const actions::Actions &J, unsigned int indComp) const;

    /// values of all components at the given actions reported separately
    virtual void valuesOfAllComponents(const actions::Actions &J, double values[]) const;

    /** Compute the phase volume associated with the given component.
        The volume is given by the integral of interpolation kernel associated with this
        component over actions, multiplied by (2pi)^3 which is the integral over angles.
        The sum of products of component amplitudes times their phase volumes is equal
        to the integral of the DF over the entire action/angle space, i.e. the total mass.
        \param[in]  indComp  is the index of component, 0 <= indComp < size();
        \param[in]  reqRelError is the required accuracy (relative error);
        \return  the phase volume;
        \throw   std::range_error if the index is out of range.
    */
    double computePhaseVolume(const unsigned int indComp, const double reqRelError=1e-3) const;

private:
    /// converter between actions and coordinates on the interpolation grid
    PtrActionSpaceScaling scaling;

    /// the interpolator defined on the scaled grid in action space
    const math::BsplineInterpolator3d<N> interp;

    /// the amplitudes of 3d interpolation kernels
    const std::vector<double> amplitudes;
};

/** Initialize the parameters used to create an interpolated DF by collecting the values
    of the provided source DF at the nodes of a 3d grid in action space.
*/
template<int N>
std::vector<double> createInterpolatedDFAmplitudes(
    const BaseDistributionFunction& df, const BaseActionSpaceScaling& scaling,
    const std::vector<double> &gridU, const std::vector<double> &gridV,
    const std::vector<double> &gridW);

/** Initialize the parameters used to create an interpolated DF from the array of actions
    computed from an N-body snapshot, assuming that this array represents particles sampled
    from this DF.
*/
template<int N>
std::vector<double> createInterpolatedDFAmplitudesFromActionSamples(
    const std::vector<actions::Actions>& actions, const std::vector<double>& masses,
    const BaseActionSpaceScaling& scaling, const std::vector<double> &gridU,
    const std::vector<double> &gridV, const std::vector<double> &gridW);

}  // namespace df
