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
    if(par.Rdisk<=0 || par.Hdisk<=0 || par.Rsigmar<=0)
        throw std::invalid_argument("PseudoIsothermal DF: disk scale length and height must be positive");
    if(par.sigmar0<=0)
        throw std::invalid_argument("PseudoIsothermal DF: velocity dispersion scale must be positive");
    if(par.sigmabirth<=0 || par.sigmabirth>1)
        throw std::invalid_argument("PseudoIsothermal DF: invalid value for velocity dispersion at birth");
    math::prepareIntegrationTableGL(0, 1, NT, qx, qw);  // prepare table for integration over age
    double kappa, Omega;
    freq.epicycleFreqs(par.Hdisk, kappa, numin, Omega); // record the vertical epi.freq. at R=Hdisk
}

// part of DF that depends on radial and vertical action and velocity dispersions
inline double df_JrJz(double sigmarsq, double sigmazsq, double sigmaminsq,
    double kappaJr, double nuJz)
{
    sigmarsq = std::max(sigmarsq, sigmaminsq);
    sigmazsq = std::max(sigmazsq, sigmaminsq);
    return exp( -kappaJr / sigmarsq - nuJz / sigmazsq ) / (sigmarsq * sigmazsq);
}
    
double PseudoIsothermal::value(const actions::Actions &J) const
{
    double Rfreq = freq.R_from_Lz(std::max(par.Jphimin, fabs(J.Jphi)));
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.epicycleFreqs(Rfreq, kappa, nu, Omega);
    if(Rfreq < par.Hdisk) // for small radii, the vertical epicyclic frequency is computed roughly at
        nu = numin;       // disk scaleheight, not at z=0 (otherwise it could tend to infinity as r->0)
    // obtain characteristic radius corresponding to the given z-component of angular momentum
    double Rcirc    = J.Jphi!=0 ? sqrt(fabs(J.Jphi) / Omega) : 0;
    if(Rcirc > 100 * par.Rdisk)
        return 0;   // we're too far out
    // surface density follows an exponential profile in radius 
    double Sigma    = par.Sigma0 * exp( -Rcirc / par.Rdisk );
    // squared vertical velocity dispersion computed from the condition that the disk thickness is const
    double sigmazsq = 2 * pow_2(nu * par.Hdisk);
    // squared radial velocity dispersion at the given radius
    double sigmarsq = pow_2(par.sigmar0 * exp ( -Rcirc / par.Rsigmar));
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
        double t1 = 1 / (std::pow(par.sigmabirth, -1/par.beta) - 1);
        for(int i=0; i<NT; i++) {
            // t is the lookback time (stellar age) measured in units of galaxy time (ranges from 0 to 1)
            double t = qx[i];
            // star formation rate exponentially increases with look-back time
            double weight = exp(t / par.Tsfr) * qw[i];
            // velocity dispersion scales as  [ (t+t1) / (1+t1) ]^beta
            double mult   = std::pow( (t + t1) / (1 + t1), par.beta);
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
