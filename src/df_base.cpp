#include "df_base.h"
#include "math_core.h"
#include "math_sample.h"
#include <cmath>
#include <stdexcept>

namespace df{

/// convert from scaled variables to the actual actions to be passed to DF
/// if jac!=NULL, store the value of jacobian of transformation in this variable
actions::Actions ActionSpaceScalingTriangLog::toActions(const double vars[], double* jac) const
{
    const double u = vars[0], v = vars[1], w = vars[2];
    if(u<0 || u>1 || v<0 || v>1 || w<0 || w>1)
        throw std::range_error("ActionSpaceScaling: input variables outside unit cube");
    const double
    Jsum = exp( 1/(1-u) - 1/u ),                         // Jr+Jz+|Jphi|
    Jm   = v==0 || v==1 ? 0 : Jsum * (1 - fabs(2*v-1));  // Jr+Jz
    if(jac)
        *jac  = (Jsum>1e-100 && Jsum<1e100) ?   // if near J=0 or infinity, set jacobian to zero
            (2 - fabs(4*v-2)) * pow_3(Jsum) * (1/pow_2(1-u) + 1/pow_2(u)) : 0;
    return actions::Actions(w==0 ? 0 : Jm * w, w==1 ? 0 : Jm * (1-w), v==0.5 ? 0 : Jsum * (2*v-1));
}

void ActionSpaceScalingTriangLog::toScaled(const actions::Actions &acts, double vars[3]) const
{
    if(!(acts.Jr>=0 && acts.Jz>=0 && acts.Jphi==acts.Jphi))
        throw std::range_error("ActionSpaceScaling: input actions out of range");
    double Jm = acts.Jr + acts.Jz;
    double Js = Jm + fabs(acts.Jphi);
    double lJ = 0.5*log(Js);
    vars[0] = fabs(lJ) < 1 ?
        1 / (1 + sqrt(1 + pow_2(lJ)) - lJ) :
        0.5 * (sqrt(1 + pow_2(1/lJ)) * math::sign(lJ) + 1 - 1/lJ);
    vars[1] = acts.Jphi==0 ? 0.5 : acts.Jphi==INFINITY ? 1 : acts.Jphi==-INFINITY ? 0 :
        0.5 + 0.5 * acts.Jphi / Js;
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
    const double u = vars[0], v = vars[1], w = vars[2], Jm = v / (1-v) * scaleJm;
    if(u<0 || u>1 || v<0 || v>1 || w<0 || w>1)
        throw std::range_error("ActionSpaceScaling: input variables outside unit cube");
    if(jac)
        *jac = pow_2(scaleJm) * scaleJphi * v / pow_3(1-v) * (1/pow_2(1-u) + 1/pow_2(u));
    return actions::Actions(w==0 ? 0 : Jm * w, w==1 ? 0 : Jm * (1-w), scaleJphi * (1/(1-u) - 1/u));
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
/// or f * ln(f)  if LogTerm==true, in scaled coords in action space.
template <bool LogTerm>
class DFIntegrandNdim: public math::IFunctionNdim {
public:
    DFIntegrandNdim(const BaseDistributionFunction& _df, const BaseActionSpaceScaling& _scaling) :
        df(_df), scaling(_scaling) {};

    /// compute the value of DF, taking into accound the scaling transformation for actions:
    /// input array of length 3 contains the three actions, scaled as described above;
    /// output a single value (DF multiplied by the jacobian of scaling transformation)
    virtual void eval(const double vars[], double values[]) const
    {
        double jac;  // will be initialized by the following call
        const actions::Actions act = scaling.toActions(vars, &jac);
        double val = 0;
        if(jac!=0) {
            double dfval = df.value(act);
            if(LogTerm && dfval>0)
                dfval *= log(dfval);
            val = dfval * jac * TWO_PI_CUBE;   // integral over three angles
        } else {
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
    const BaseDistributionFunction& df;    ///< the instance of DF
    const BaseActionSpaceScaling& scaling; ///< scaling transformation
};

double BaseDistributionFunction::totalMass(const double reqRelError, const int maxNumEval,
    double* error, int* numEval) const
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    double result;  // store the value of integral
    math::integrateNdim(DFIntegrandNdim<false>(*this, ActionSpaceScalingTriangLog()),
        xlower, xupper, reqRelError, maxNumEval, &result, error, numEval);
    return result;
}

double totalEntropy(const BaseDistributionFunction& DF, const double reqRelError, const int maxNumEval)
{
    double xlower[3] = {0, 0, 0};
    double xupper[3] = {1, 1, 1};
    double result;
    math::integrateNdim(DFIntegrandNdim<true>(DF, ActionSpaceScalingTriangLog()),
        xlower, xupper, reqRelError, maxNumEval, &result);
    return result;
}

void sampleActions(const BaseDistributionFunction& DF, const int numSamples,
    std::vector<actions::Actions>& samples, double* totalMass, double* totalMassErr)
{
    double xlower[3] = {0, 0, 0};  // boundaries of integration region in scaled coordinates
    double xupper[3] = {1, 1, 1};
    math::Matrix<double> result;   // the result array of actions
    ActionSpaceScalingTriangLog transf;
    DFIntegrandNdim<false> fnc(DF, transf);
    math::sampleNdim(fnc, xlower, xupper, numSamples, result, 0, totalMass, totalMassErr);
    samples.resize(result.rows());
    for(unsigned int i=0; i<result.rows(); i++) {
        const double point[3] = {result(i,0), result(i,1), result(i,2)};
        samples[i] = transf.toActions(point);  // transform from scaled vars to actions
    }
}

}
