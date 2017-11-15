#include "df_disk.h"
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
    // obtain characteristic radius of in-plane motion with the given Jr, Jphi
    const double kappaOverOmega = 1.0;
    double Rcirc = freq.R_from_Lz(fabs(J.Jphi) + kappaOverOmega * J.Jr);
    if(Rcirc > 20 * par.Rdisk)
        return 0;   // we're too far out, DF is negligibly small
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.epicycleFreqs(Rcirc, kappa, nu, Omega);
    // surface density follows an exponential profile in radius
    double Sigma = par.Sigma0 * exp( -Rcirc / par.Rdisk );
    // squared vertical velocity dispersion computed from the condition that the disk thickness is Hdisk
    double sigmazsq = 2 * pow_2(nu * par.Hdisk);
    // squared radial velocity dispersion at the given radius
    double sigmarsq = pow_2(par.sigmar0 * exp ( -Rcirc / par.Rsigmar ) );
    // suppression factor for counterrotating orbits
    double exp_Jphi = J.Jphi>0 ? 1. : exp( 2*Omega * J.Jphi / sigmarsq);
    // if we have a non-trivial age-velocity dispersion relation,
    // then we need to integrate over sub-populations convolved with star formation history
    double kappaJr  = kappa * J.Jr, nuJz = nu * J.Jz;
    double exp_JrJz = 0;
    if(par.beta == 0 || par.sigmabirth == 1)
        exp_JrJz = df_JrJz(sigmarsq, sigmazsq, pow_2(par.sigmamin), kappaJr, nuJz);
    else {  // integrate using the hard-coded Gauss-Legendre table on the interval [0:1]
        static const int NT = 5;      // number of points in quadrature rule for integration over age
        static const double qx[NT] =  // nodes of quadrature rule
        { 0.04691007703066802, 0.23076534494715845, 0.5, 0.76923465505284155, 0.95308992296933198 };
        static const double qw[NT] =  // weights of quadrature rule
        { 0.11846344252809454, 0.23931433524968324, 64./225, 0.23931433524968324, 0.11846344252809454 };
        double sumnorm = 0;
        double t1 = 1 / (std::pow(par.sigmabirth, -1/par.beta) - 1);
        for(int i=0; i<NT; i++) {
            // t is the lookback time (stellar age) measured in units of galaxy time (ranges from 0 to 1)
            double t = qx[i];
            // star formation rate exponentially increases with look-back time
            double weight = exp(t / par.Tsfr) * qw[i];
            // velocity dispersions {sigma_r, sigma_z} scale as  [ (t+t1) / (1+t1) ]^beta
            double multsq = std::pow( (t + t1) / (1 + t1), 2*par.beta);  // multiplied by sigma^2
            exp_JrJz += weight *
                df_JrJz(sigmarsq * multsq, sigmazsq * multsq, pow_2(par.sigmamin), kappaJr, nuJz);
            sumnorm  += weight;
        }
        exp_JrJz /= sumnorm;
    }
    double result = 1./(2*M_PI*M_PI) * Sigma * exp_JrJz * exp_Jphi * nu * Omega / kappa;
    return isFinite(result) ? result : 0;
}

}  // namespace df
