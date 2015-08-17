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
    /// input array of length 3 contains the three actions, scaled as  vars[i] = J_i/(J_i+Jscale)
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        actions::Actions acts;
        acts.Jr    = Jscale * vars[0] / (1-vars[0]);
        acts.Jz    = Jscale * vars[1] / (1-vars[1]);
        acts.Jphi  = Jscale * vars[2] / (1-vars[2]);
        double val = df.value(acts);
        acts.Jphi *= -1;  // Jphi may take both positive and negative values, sum the contribution of both
        val       += df.value(acts);
        double jac = pow_3(Jscale) / (pow_2(1-vars[0]) * pow_2(1-vars[1]) * pow_2(1-vars[2]) );
        values[0]  = val * jac * pow_3(2*M_PI);   // integral over three angles
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
    double Jscale = 1.; // value for scaling transformation
    // (note: the result shouldn't depend on it as long as the integral converges well)

    DFIntegrandNdim fnc(*this, Jscale);
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral

    math::integrateNdim(fnc, xlower, xupper, reqRelError, maxNumEval, &result, error, numEval);
    return result;
}

}
