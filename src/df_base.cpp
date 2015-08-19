#include "df_base.h"
#include "math_core.h"
#include <cmath>

namespace df{

/// helper class for integrating distribution function
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    DFIntegrandNdim(const BaseDistributionFunction& _df, double _Jscale = 1) :
        df(_df), Jscale(_Jscale) {};

    /// compute the value of DF, taking into accound the scaling transformation for actions:
    /// input array of length 3 contains the three actions, scaled as described below;
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        if(vars[0]<0.01 || vars[0]>0.99) {  // we're (almost) at zero or infinity in terms of magnitude of J
            values[0] = 0;  // at infinity we expect that f(J) tends to zero,
            return;         // while at J->0 the jacobian of transformation is exponentially small.
        }
        double jac;  // will be initialized by the following call
        double val = df.value(getActions(vars, &jac));
        values[0]  = val * jac;
    }

    /// convert from scaled variables to the actual actions to be passed to DF
    /// if jac!=NULL, store the value of jacobian of transformation in this variable
    actions::Actions getActions(const double vars[], double* jac=0) const {
        // scaled variables p, q and s lie in the range [0:1];
        // we define J0 = exp( 1/(1-s) - 1/s), and set
        // J1 = J0 p, J2 = J0 (1-p) q, J3 = J0 (1-p-q+pq), so that Jr+Jz+|Jphi| = J0.
        const double s  = vars[0], p = vars[1], q = vars[2];
        const double J0 = exp( 1/(1-s) - 1/s );
        if(jac)
            *jac = (1-p) * pow_3(J0) * (1/pow_2(1-s) + 1/pow_2(s));
        actions::Actions acts;
        acts.Jr   = J0 * p;
        acts.Jz   = J0 * (1-p) * q;
        acts.Jphi = J0 * (1-p-q+p*q);
        return acts;
    }

    /// number of variables (3 actions)
    virtual unsigned int numVars()   const { return 3; }
    /// number of values to compute (1 value of DF)
    virtual unsigned int numValues() const { return 1; }
private:
    const BaseDistributionFunction& df;  ///< the instance of DF
    const double Jscale;                 ///< value for scaling transformation
};

double BaseDistributionFunction::totalMass(const double reqRelError, const int maxNumEval,
    double* error, int* numEval) const
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral
    math::integrateNdim(DFIntegrandNdim(*this), xlower, xupper, reqRelError, maxNumEval, &result, error, numEval);
    return result * TWO_PI_CUBE;   // integral over three angles
}

void sampleActions(const BaseDistributionFunction& DF, const int numSamples,
    std::vector<actions::Actions>& samples)
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    math::Matrix<double> result;   // the result array of actions
    DFIntegrandNdim fnc(DF);
    math::sampleNdim(fnc, xlower, xupper, numSamples, result);
    samples.resize(numSamples);
    for(int i=0; i<numSamples; i++) {
        const double point[3] = {result(i,0), result(i,1), result(i,2)};
        samples[i] = fnc.getActions(point);  // transform from scaled vars to actions
    }
}

}
