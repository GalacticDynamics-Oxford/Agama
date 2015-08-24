#include "df_disk.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

PseudoIsothermal::PseudoIsothermal(
    const PseudoIsothermalParam &inparams, const potential::BasePotential& inpot) :
    par(inparams), potential(inpot)
{
    // sanity checks on parameters
    if(par.Rdisk<=0)
        throw std::invalid_argument("PseudoIsothermal DF: disk scale length must be positive");
    if(par.sigmar0<=0 || par.sigmaz0<=0)
        throw std::invalid_argument("PseudoIsothermal DF: velocity dispersion scale must be positive");
}

double PseudoIsothermal::value(const actions::Actions &J) const {
    double Rcirc     = R_from_Lz(potential, J.Jphi);
    double kappa, nu, Omega;   // epicyclic freqs
    epicycleFreqs(potential, Rcirc, kappa, nu, Omega);
    if(!math::isFinite(kappa+nu+Omega)) { //FIXME!!! workaround for r->0
        kappa=nu=Omega=1;
    }
    double exp_rad   = exp( -Rcirc / par.Rdisk );
    if(exp_rad==0)   // we're too far out
        return 0;
    double sigmarsq  = pow_2(par.sigmar0) * exp_rad;
    double sigmazsq  = pow_2(par.sigmaz0) * exp_rad;
    double Sigma     = par.Sigma0 * exp_rad;
    double exp_act   = exp( -kappa * J.Jr / sigmarsq - nu * J.Jz / sigmazsq );
    double exp_Jphi  = par.L0 == 0 ? 2. : par.L0 == INFINITY ? 1. : 1 + tanh(J.Jphi / par.L0);
    double numerator = par.norm * exp_act * exp_Jphi * Sigma * Omega * nu;
    if(numerator==0)
       return 0;
    else
       return numerator / (2*M_PI*M_PI * kappa * sigmarsq * sigmazsq);
}

}  // namespace df
