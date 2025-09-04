#include "df_base.h"
#include "math_core.h"
#include "math_sample.h"
#include <cmath>
#include <stdexcept>

namespace df{

/// convert from scaled variables to the actual actions to be passed to DF
/// if jac!=NULL, store the value of jacobian of transformation in this variable
actions::Actions ActionSpaceScalingTriangLog::toActions(const double vars[3], double* jac) const
{
    const double u = vars[0], v = vars[1], w = vars[2];
    if(u<0 || u>1 || v<0 || v>1 || w<0 || w>1)
        throw std::range_error("ActionSpaceScaling: input variables outside unit cube");
    if(u>0.98 || u<0.02) {  // prevent Js from reaching infinity
        if(jac) *jac = 0;   // set the jacobian to zero, ignoring this part of the integration cube
        return u<0.5 ? actions::Actions(0, 0, 0) : actions::Actions(INFINITY, INFINITY, INFINITY);
    }
    double vv = M_PI * v*v * (3-2*v),  // cubic transformation to stretch the range near v=0,v=1
    sv,cv, Js = exp( 1/(1-u) - 1/u );  // hypot(Jr+Jz, Jphi)
    math::sincos(vv, sv, cv);
    double Jm = Js * sv;   // Jr+Jz
    if(jac) {
        *jac = M_PI * 6*v*(1-v) * Jm * Js * Js * (1/pow_2(1-u) + 1/pow_2(u));
    }
    return actions::Actions(Jm * w, Jm * (1-w), Js * cv);
}

void ActionSpaceScalingTriangLog::toScaled(const actions::Actions &acts, double vars[3]) const
{
    if(!(acts.Jr>=0 && acts.Jz>=0 && acts.Jphi==acts.Jphi))
        throw std::range_error("ActionSpaceScaling: input actions out of range");
    double Jm = fabs(acts.Jr + acts.Jz);
    double Js = sqrt(pow_2(Jm) + pow_2(acts.Jphi));
    double lJ = 0.5*log(Js);
    double xi = atan2(Jm, acts.Jphi) / M_PI;  // valid for all input arguments
    //double phi= (1./3) * acos(1 - 2 * xi);    // aux angle in the solution of a cubic eqn
    vars[0] = fabs(lJ) < 1 ?
        1 / (1 + sqrt(1 + pow_2(lJ)) - lJ) :
        0.5 * (sqrt(1 + pow_2(1/lJ)) * math::sign(lJ) + 1 - 1/lJ);
    //vars[1] = xi==0 || xi==0.5 || xi==1. ? xi :   // for some input values return the exact result
    //    0.5 * (1 - cos(phi) + M_SQRT3*sin(phi));  // otherwise the solution of a cubic eqn
    vars[1] = scale(math::ScalingCub(0,1), xi);
    vars[2] = acts.Jr==0 ? 0 : acts.Jr==INFINITY ? 1 : acts.Jr / Jm;
}

ActionSpaceScalingRect::ActionSpaceScalingRect(double _scaleJm, double _scaleJphi) :
    scaleJm(_scaleJm), scaleJphi(_scaleJphi)
{
    if(scaleJm<=0 || scaleJphi<=0 || !isFinite(scaleJm+scaleJphi))
        throw std::invalid_argument("ActionsSpaceScalingRect: invalid scaling factors");
}

actions::Actions ActionSpaceScalingRect::toActions(const double vars[3], double *jac) const
{
    const double u = vars[0], v = vars[1], w = vars[2], w1 = 1-w, Jm = v / (1-v) * scaleJm;
    if(u<0 || u>1 || v<0 || v>1 || w<0 || w1<0)
        throw std::range_error("ActionSpaceScaling: input variables outside unit cube");
    if(jac) {
        *jac = pow_2(scaleJm) * scaleJphi * v / pow_3(1-v) * (1/pow_2(1-u) + 1/pow_2(u));
        if(!(*jac > 1e-100 && *jac < 1e100))
            *jac = 0;
    }
    return actions::Actions(w==0 ? 0 : Jm * w, w==1 ? 0 : Jm * w1, scaleJphi * (1/(1-u) - 1/u));
}

void ActionSpaceScalingRect::toScaled(const actions::Actions &acts, double vars[3]) const
{
    if(!(acts.Jr>=0 && acts.Jz>=0 && acts.Jphi==acts.Jphi))
        throw std::range_error("ActionSpaceScaling: input actions out of range");
    double Jm = acts.Jr + acts.Jz;
    double Jp = acts.Jphi / scaleJphi;
    vars[0] = fabs(Jp) < 1 ?
        1 / (1 + sqrt(1 + pow_2(0.5*Jp)) - 0.5*Jp) :
        0.5 + sqrt(0.25 + pow_2(1/Jp)) * math::sign(Jp) - 1/Jp;
    vars[1] = 1 / (scaleJm/Jm + 1);
    vars[2] = Jm==0 ? 0 : Jm==INFINITY ? 0 : acts.Jr / Jm;
}

/// helper class for computing the integral of distribution function f
/// or -f * ln(f)  if LogTerm==true, in scaled coords in action space.
template <bool LogTerm>
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    const BaseDistributionFunction& df;        ///< the instance of DF
    const ActionSpaceScalingTriangLog scaling; ///< scaling transformation

    DFIntegrandNdim(const BaseDistributionFunction& _df) : df(_df), scaling() {};

    /// compute the value of DF, taking into accound the scaling transformation for actions:
    /// input array of length 3 contains the three actions, scaled as described above;
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        double jac;  // will be initialized by the following call
        const actions::Actions act = scaling.toActions(vars, &jac);
        if(jac!=0) {
            double val = df.value(act);
            if(!isFinite(val))
                val = 0;
            if(LogTerm && val>0)
                val *= -log(val);
            values[0] = val * jac * TWO_PI_CUBE;   // integral over three angles
        } else {
            // we're (almost) at zero or infinity in terms of magnitude of J
            // at infinity we expect that f(J) tends to zero,
            // while at J->0 the jacobian of transformation is exponentially small.
            values[0] = 0;
        }
    }

    /// number of variables (3 actions)
    virtual unsigned int numVars()   const { return 3; }
    /// number of values to compute (1 value of DF)
    virtual unsigned int numValues() const { return 1; }
};

double BaseDistributionFunction::totalMass(const double reqRelError, const int maxNumEval) const
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral
    math::integrateNdim(DFIntegrandNdim<false>(*this), xlower, xupper, reqRelError, maxNumEval, &result);
    return result;
}

double totalEntropy(const BaseDistributionFunction& DF, const double reqRelError, const int maxNumEval)
{
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result;
    math::integrateNdim(DFIntegrandNdim<true>(DF), xlower, xupper, reqRelError, maxNumEval, &result);
    return result;
}

}
