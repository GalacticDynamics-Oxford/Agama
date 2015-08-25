#include "df_base.h"
#include "math_core.h"
#include "math_sample.h"
#include <cmath>

namespace df{

/// convert from scaled variables to the actual actions to be passed to DF
/// if jac!=NULL, store the value of jacobian of transformation in this variable
actions::Actions unscaleActions(const double vars[], double* jac)
{
    // scaled variables p, q and s lie in the range [0:1];
    // we define J0 = exp( 1/(1-s) - 1/s), q' = 2*q-1, and set
    // Jr = J0 p, Jphi = J0 (1-p) q', Jz = J0 (1-p) (1-|q'|), so that Jr+Jz+|Jphi| = J0.
    const double s  = vars[0], p = vars[1], q = vars[2];
    const double J0 = exp( 1/(1-s) - 1/s );
#if 1
    if(jac)
        *jac = math::withinReasonableRange(J0) ?   // if near J=0 or infinity, set jacobian to zero
            2*(1-p) * pow_3(J0) * (1/pow_2(1-s) + 1/pow_2(s)) : 0;
    actions::Actions acts;
    acts.Jr   = J0 * p;
    acts.Jphi = J0 * (1-p) * (2*q-1);
    acts.Jz   = J0 * (1-p) * (1-fabs(2*q-1));
#else
    if(jac)
        *jac = math::withinReasonableRange(J0) ?   // if near J=0 or infinity, set jacobian to zero
        2*(1-fabs(2*p-1)) * pow_3(J0) * (1/pow_2(1-s) + 1/pow_2(s)) : 0;
    actions::Actions acts;
    acts.Jphi = J0 * (2*p-1);
    acts.Jr   = J0 * (1-fabs(2*p-1)) * q;
    acts.Jz   = J0 * (1-fabs(2*p-1)) * (1-q);
#endif
    return acts;
}

/// helper class for integrating distribution function
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    DFIntegrandNdim(const BaseDistributionFunction& _df) :
        df(_df) {};

    /// compute the value of DF, taking into accound the scaling transformation for actions:
    /// input array of length 3 contains the three actions, scaled as described above;
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        double jac;  // will be initialized by the following call
        const actions::Actions act = unscaleActions(vars, &jac);
        double val = 0;
        if(jac!=0)
            val = df.value(act) * jac * TWO_PI_CUBE;   // integral over three angles
        else {
            // we're (almost) at zero or infinity in terms of magnitude of J
            // at infinity we expect that f(J) tends to zero,
            // while at J->0 the jacobian of transformation is exponentially small.
        }            
        values[0] = val;
    }

    /// number of variables (3 actions)
    virtual unsigned int numVars()   const { return 3; }
    /// number of values to compute (1 value of DF)
    virtual unsigned int numValues() const { return 1; }
private:
    const BaseDistributionFunction& df;  ///< the instance of DF
};

double BaseDistributionFunction::totalMass(const double reqRelError, const int maxNumEval,
    double* error, int* numEval) const
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral
    math::integrateNdim(DFIntegrandNdim(*this), xlower, xupper, 
        reqRelError, maxNumEval, &result, error, numEval);
    return result;
}

void sampleActions(const BaseDistributionFunction& DF, const int numSamples,
    std::vector<actions::Actions>& samples, double* totalMass, double* totalMassErr)
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    math::Matrix<double> result;   // the result array of actions
    DFIntegrandNdim fnc(DF);
    math::sampleNdim(fnc, xlower, xupper, numSamples, 0, result, 0, totalMass, totalMassErr);
    samples.resize(result.numRows());
    for(unsigned int i=0; i<result.numRows(); i++) {
        const double point[3] = {result(i,0), result(i,1), result(i,2)};
        samples[i] = unscaleActions(point);  // transform from scaled vars to actions
    }
}

}
