#include "df_halo.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

DoublePowerLaw::DoublePowerLaw(const DoublePowerLawParam &inparams) :
    par(inparams)
{
    // sanity checks on parameters
    if(par.norm<=0)
        throw std::invalid_argument("DoublePowerLaw DF: normalization should be positive");
    if(par.J0<=0)
        throw std::invalid_argument("DoublePowerLaw DF: break action J0 must be positive");
    if(par.Jcutoff<0)
        throw std::invalid_argument("DoublePowerLaw DF: cutoff action Jcutoff must be non-negative");
    if(par.slopeOut<=3 && par.Jcutoff==0)
        throw std::invalid_argument(
            "DoublePowerLaw DF: mass diverges at large J (outer slope must be > 3)");
    if(par.slopeIn>=3)
        throw std::invalid_argument(
            "DoublePowerLaw DF: mass diverges at J->0 (inner slope must be < 3)");
    if(par.steepness<=0)
        throw std::invalid_argument("DoublePowerLaw DF: transition steepness parameter must be positive");
    if(par.cutoffStrength<=0)
        throw std::invalid_argument("DoublePowerLaw DF: cutoff strength parameter must be positive");
    if( par.coefJrIn <=0 || par.coefJzIn <=0 || par.coefJrIn + par.coefJzIn>=3 || 
        par.coefJrOut<=0 || par.coefJzOut<=0 || par.coefJrOut+par.coefJzOut>=3 )
        throw std::invalid_argument(
            "DoublePowerLaw DF: invalid weights in the linear combination of actions");
}

double DoublePowerLaw::value(const actions::Actions &J) const
{
    // linear combination of actions in the inner part of the model (for J<J0)
    double hJ  = par.coefJrIn * J.Jr + par.coefJzIn * J.Jz +
        (3-par.coefJrIn -par.coefJzIn) * fabs(J.Jphi);
    // linear combination of actions in the outer part of the model (for J>J0)
    double gJ  = par.coefJrOut* J.Jr + par.coefJzOut* J.Jz +
        (3-par.coefJrOut-par.coefJzOut)* fabs(J.Jphi);
    double val = par.norm / pow_3(2*M_PI * par.J0) *
        math::pow(1 + math::pow(par.J0 / hJ, par.steepness),  par.slopeIn  / par.steepness) *
        math::pow(1 + math::pow(gJ / par.J0, par.steepness), -par.slopeOut / par.steepness);
    if(par.Jcutoff>0)   // exponential cutoff at large J
        val *= exp(-math::pow(gJ / par.Jcutoff, par.cutoffStrength));
    return val;
}

}  // namespace df
