#include "df_halo.h"
#include <cmath>
#include <stdexcept>

namespace df{

DoublePowerLaw::DoublePowerLaw(const DoublePowerLawParam &inparams) :
    par(inparams)
{
    // sanity checks on parameters
    if( par.ar<=0 || par.az<=0 || par.aphi<=0 ||
        par.br<=0 || par.bz<=0 || par.bphi<=0 )
        throw std::invalid_argument(
            "DoublePowerLaw DF: coefficients in the linear combination of actions must be positive");
    if(par.j0<=0)
        throw std::invalid_argument("DoublePowerLaw DF: break action j0 must be positive");
    if(par.jcore<0)
        throw std::invalid_argument("DoublePowerLaw DF: core action jcore must be non-negative");
    if(par.alpha<0)
        throw std::invalid_argument("DoublePowerLaw DF: inner slope alpha must be non-negative");
    if(par.beta<=3)
        throw std::invalid_argument(
            "DoublePowerLaw DF: mass diverges at large J (outer slope beta must be > 3");
    if(par.jcore==0 && par.alpha>=3)
        throw std::invalid_argument("DoublePowerLaw DF: mass diverges at J->0");
}

double DoublePowerLaw::value(const actions::Actions &J) const {
    // linear combination of actions in the inner part of the model (for J<J0)
    double hJ   = par.ar*J.Jr + par.az*J.Jz + par.aphi*fabs(J.Jphi);
    // linear combination of actions in the outer part of the model (for J>J0)
    double gJ   = par.br*J.Jr + par.bz*J.Jz + par.bphi*fabs(J.Jphi);
    double prob = 1./pow_3(par.j0) *
        pow(1. + par.j0 / (hJ + par.jcore), par.alpha) *
        pow(1. + gJ / par.j0, -par.beta);
    return prob;
}

}  // namespace df
