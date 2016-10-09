#include "df_disk.h"
#include "math_core.h"
#include <cmath>
#include <stdexcept>

namespace df{

PseudoIsothermal::PseudoIsothermal(
    const PseudoIsothermalParam &params, const potential::Interpolator& freqs) :
    par(params), freq(freqs)
{
    // sanity checks on parameters
    if(par.Rdisk<=0 || par.Rsigmar<=0 || par.Rsigmaz<=0)
        throw std::invalid_argument("PseudoIsothermal DF: disk scale length must be positive");
    if(par.sigmar0<=0 || par.sigmaz0<=0)
        throw std::invalid_argument("PseudoIsothermal DF: velocity dispersion scale must be positive");
    if(par.sigmabirth<=0 || par.sigmabirth>1)
        throw std::invalid_argument("PseudoIsothermal DF: invalid value for velocity dispersion at birth");
    math::prepareIntegrationTableGL(0, 1, NT, qx, qw);  // prepare table for integration over age
}

// part of DF that depends on radial and vertical action and velocity dispersions
static inline double df_JrJz(double sigmarsq, double sigmazsq, double sigmaminsq,
    double kappaJr, double nuJz)
{
    sigmarsq = fmax(sigmarsq, sigmaminsq);
    sigmazsq = fmax(sigmazsq, sigmaminsq);
    return exp( -kappaJr / sigmarsq - nuJz / sigmazsq ) / (sigmarsq * sigmazsq);
}
    
double PseudoIsothermal::value(const actions::Actions &J) const
{
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.epicycleFreqs(freq.R_from_Lz(fmax(par.Jphimin, fabs(J.Jphi))), kappa, nu, Omega);
    // obtain characteristic radius corresponding to the given z-component of angular momentum
    double Rcirc    = J.Jphi!=0 ? sqrt(fabs(J.Jphi) / Omega) : 0;
    // surface density follows an exponential profile in radius 
    double Sigma    = par.Sigma0 * exp( -Rcirc / par.Rdisk );
    if(Sigma < par.Sigma0 * 1e-100)   // we're too far out
        return 0;
    // squared radial velocity dispersion at the given radius
    double sigmarsq = pow_2(par.sigmar0 * exp ( -Rcirc / par.Rsigmar));
    // squared vertical velocity dispersion
    double sigmazsq = pow_2(par.sigmaz0 * exp ( -Rcirc / par.Rsigmaz));
    double exp_Jphi =                               // suppression factor for counterrotating orbits:
        par.Jphi0 == INFINITY || J.Jphi == 0 ? 1. : // do not distinguish the sign of Lz at all
        par.Jphi0 == 0 ? (J.Jphi>0 ? 2. : 0.) :     // strictly use only orbits with positive Lz
        1 + tanh(J.Jphi / par.Jphi0);               // intermediate regime, mildly cut off DF at Lz<0
    // if we have non-trivial age-velocity dispersion relation,
    // then we need to integrate over sub-populations convolved with star formation history
    double kappaJr  = kappa * J.Jr, nuJz = nu * J.Jz;
    double exp_JrJz = 0;
    if(par.beta == 0 || par.sigmabirth == 1)
        exp_JrJz = df_JrJz(sigmarsq, sigmazsq, pow_2(par.sigmamin), kappaJr, nuJz);
    else {  // integrate using the pre-initialized Gauss-Legendre table on the interval [0:1]
        double sumnorm = 0;
        double t1 = 1 / (pow(par.sigmabirth, -1/par.beta) - 1);
        for(int i=0; i<NT; i++) {
            // star formation rate exponentially increases with look-back time
            double weight = exp(qx[i] / par.Tsfr) * qw[i];
            // velocity dispersion scales as  [ (t+t1) / (1+t1) ]^beta
            double mult   = pow( (t1 + qx[i]) / (1 + qx[i]), par.beta);
            exp_JrJz += weight * df_JrJz(sigmarsq * mult, sigmazsq * mult,
                pow_2(par.sigmamin), kappaJr, nuJz);
            sumnorm  += weight;
        }
        exp_JrJz /= sumnorm;
    }
    double result = exp_JrJz * exp_Jphi * Sigma * Omega * nu / (4*M_PI*M_PI * kappa);
    return isFinite(result) ? result : 0;
}


}  // namespace df
