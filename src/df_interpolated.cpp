#include "df_interpolated.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace {
/// auxiliary class for computing the phase volume associated with a single component of interpolated DF
template<int N>
class InterpolatedDFintegrand: public math::IFunctionNdim{
    const math::BsplineInterpolator3d<N> &interp;
    const BaseActionSpaceScaling &scaling;
    const unsigned int indComp;
public:
    InterpolatedDFintegrand(const math::BsplineInterpolator3d<N> &_interp,
        const BaseActionSpaceScaling &_scaling, const unsigned int _indComp) :
        interp(_interp), scaling(_scaling), indComp(_indComp) {}
    virtual void eval(const double vars[], double values[]) const {
        double jac;
        scaling.toActions(vars, &jac);
        // (2pi)^3 comes from integration over angles
        values[0] = interp.valueOfComponent(vars, indComp) * TWO_PI_CUBE * jac;
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

/// auxiliary class for collecting the values of source DF at grid points in scaled action space
class DFscaled: public math::IFunctionNdim{
    const BaseDistributionFunction &df;
    const BaseActionSpaceScaling &scaling;
public:
    DFscaled(const BaseDistributionFunction &_df, const BaseActionSpaceScaling &_scaling) :
        df(_df), scaling(_scaling) {}
    virtual void eval(const double vars[], double values[]) const {
        values[0] = df.value(scaling.toActions(vars));
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};
}  // internal namespace

template<int N>
InterpolatedDF<N>::InterpolatedDF(const PtrActionSpaceScaling& _scaling,
    const std::vector<double> &gridU, const std::vector<double> &gridV,
    const std::vector<double> &gridW,  const std::vector<double> &_amplitudes)
:
    scaling(_scaling),
    interp(gridU, gridV, gridW),
    amplitudes(_amplitudes.empty() ? std::vector<double>(interp.numValues(), 1.) : _amplitudes)
{
    if(amplitudes.size() != interp.numValues())
        throw std::invalid_argument("InterpolatedDF: invalid array size");
    for(unsigned int i=0; i<amplitudes.size(); i++)
        if(amplitudes[i] < 0 || !isFinite(amplitudes[i]))
            throw std::invalid_argument("InterpolatedDF: amplitudes must be non-negative");
}

template<int N>
double InterpolatedDF<N>::value(const actions::Actions &J) const
{
    double vars[3];
    scaling->toScaled(J, vars);
    return interp.interpolate(vars, amplitudes);
}

template<int N>
void InterpolatedDF<N>::eval(const actions::Actions &J, double val[]) const
{
    double vars[3];
    scaling->toScaled(J, vars);
    interp.eval(vars, val);
    for(unsigned int i=0; i<amplitudes.size(); i++)
        val[i] *= amplitudes[i];
}

template<int N>
double InterpolatedDF<N>::computePhaseVolume(const unsigned int indComp, const double reqRelError) const
{
    if(indComp >= amplitudes.size())
        throw std::out_of_range("InterpolatedDF: component index out of range");
    double xlower[3], xupper[3];
    interp.nonzeroDomain(indComp, xlower, xupper);
    double result, error;
    const int maxNumEval = 10000;
    math::integrateNdim(InterpolatedDFintegrand<N>(interp, *scaling, indComp), xlower, xupper, 
        reqRelError, maxNumEval, &result, &error);
    return result;
}

/// routines for creating arrays of amplitudes
template<int N>
std::vector<double> createInterpolatedDFAmplitudes(
    const BaseDistributionFunction& df, const BaseActionSpaceScaling& scaling,
    const std::vector<double> &gridU, const std::vector<double> &gridV,
    const std::vector<double> &gridW)
{
    return math::createBsplineInterpolator3dArray<N>(DFscaled(df, scaling), gridU, gridV, gridW);
}

template<int N>
std::vector<double> createInterpolatedDFAmplitudesFromActionSamples(
const std::vector<actions::Actions>& actions, const std::vector<double>& masses,
const BaseActionSpaceScaling& scaling, const std::vector<double> &gridU,
const std::vector<double> &gridV, const std::vector<double> &gridW)
{
    if(actions.size() != masses.size())
        throw std::invalid_argument(
            "createInterpolatedDFParamsFromActionSamples: incorrect size of input arrays");
    math::Matrix<double> points(masses.size(), 3);
    std::vector<double> weights(masses.size());
    for(unsigned int i=0; i<masses.size(); i++) {
        scaling.toScaled(actions[i], &points(i, 0));
        double jac;
        scaling.toActions(&points(i, 0), &jac);
        weights[i] = 1 / jac;
    }
    return math::createBsplineInterpolator3dArrayFromSamples<N>(points, weights, gridU, gridV, gridW);
}

// force the compilation of template instantiations
template class InterpolatedDF<1>;
template class InterpolatedDF<3>;

template std::vector<double> createInterpolatedDFAmplitudes<1>(
    const BaseDistributionFunction&, const BaseActionSpaceScaling&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&);
template std::vector<double> createInterpolatedDFAmplitudes<3>(
    const BaseDistributionFunction&, const BaseActionSpaceScaling&,
    const std::vector<double>&, const std::vector<double>&, const std::vector<double>&);

template std::vector<double> createInterpolatedDFAmplitudesFromActionSamples<1>(
    const std::vector<actions::Actions>&, const std::vector<double>&,
    const BaseActionSpaceScaling&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);
template std::vector<double> createInterpolatedDFAmplitudesFromActionSamples<3>(
    const std::vector<actions::Actions>&, const std::vector<double>&,
    const BaseActionSpaceScaling&, const std::vector<double>&,
    const std::vector<double>&, const std::vector<double>&);

}  // namespace df
