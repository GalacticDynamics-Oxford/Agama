#include "df_disk.h"
#include <cmath>
#include <stdexcept>

namespace df{

namespace{  // internal
    
/// compute the average of DF over stellar age:
/// ( \int_0^1 dt B^2(t) \exp[ t/t_0 - A*B(t) ] ) / ( \int_0^1 dt \exp[ t/t_0 ] ),
/// where B(t) = ( (t + t_1) / (1 + t_1) )^{-2\beta}
double averageOverAge(double A, const AgeVelocityDispersionParam& par)
{
    if(par.beta == 0 || par.sigmabirth == 1 || !isFinite(par.Tsfr))
        return exp(-A);
    // if we have a non-trivial age-velocity dispersion relation,
    // then we need to integrate over sub-populations convolved with star formation history
    static const int NT = 5;      // number of points in quadrature rule for integration over age
    static const double qx[NT] =  // nodes of quadrature rule
    { 0.04691007703066802, 0.23076534494715845, 0.5, 0.76923465505284155, 0.95308992296933198 };
    static const double qw[NT] =  // weights of quadrature rule
    { 0.11846344252809454, 0.23931433524968324, 64./225, 0.23931433524968324, 0.11846344252809454 };
    double s = std::pow(par.sigmabirth, 1./par.beta); 
    double integ=0, norm = 0;
    for(int i=0; i<NT; i++) {
        // t is the lookback time (stellar age) measured in units of galaxy time (ranges from 0 to 1)
        double t = qx[i];
        // star formation rate exponentially increases with look-back time
        double weight = exp(t / par.Tsfr) * qw[i];
        // velocity dispersions {sigma_r, sigma_z} scale as  [ t + s * (1-t) ]^beta
        double multsq = std::pow(t + (1-t) * s, -2*par.beta);  // multiplied by sigma^-2
        integ += weight * exp(-A * multsq) * pow_2(multsq);
        norm  += weight;
    }
    return integ / norm;
}
}

QuasiIsothermal::QuasiIsothermal(const QuasiIsothermalParam &params, const potential::Interpolator& freqs) :
    par(params), freq(freqs)
{
    // sanity checks on parameters
    if(par.Rdisk<=0)
        throw std::invalid_argument("QuasiIsothermal: disk scale radius Rdisk must be positive");
    if(par.sigmar0<=0)
        throw std::invalid_argument("QuasiIsothermal: velocity dispersion sigmar0 must be positive");
    if(par.Rsigmar<=0)
        throw std::invalid_argument("QuasiIsothermal: velocity scale radius Rsigmar must be positive");
    if(!( (par.sigmaz0==0 && par.Rsigmaz==0) ^ (par.Hdisk==0) ))
        throw std::invalid_argument("QuasiIsothermal: should have either "
            "Hdisk>0 to assign the vertical velocity dispersion from disk scaleheight, or "
            "Rsigmaz>0, sigmaz0>0 to make it exponential in radius");
    if(par.Hdisk<0 || par.sigmaz0<0 || par.Rsigmaz<0)  // these are optional but non-negative
        throw std::invalid_argument("QuasiIsothermal: parameters cannot be negative");
    if(par.sigmabirth<=0 || par.sigmabirth>1)
        throw std::invalid_argument("QuasiIsothermal: invalid value for velocity dispersion at birth");
}

double QuasiIsothermal::value(const actions::Actions &J) const
{
    // obtain the radius of in-plane motion with the given "characteristic" angular momentum
    double Rcirc = freq.R_from_Lz(sqrt(pow_2(par.Jmin) +
        pow_2(fabs(J.Jphi) + par.coefJr * J.Jr + par.coefJz * J.Jz)) );
    if(Rcirc > 20 * par.Rdisk)
        return 0;   // we're too far out, DF is negligibly small
    double kappa, nu, Omega;   // characteristic epicyclic freqs
    freq.epicycleFreqs(Rcirc, kappa, nu, Omega);
    // surface density follows an exponential profile in radius
    double Sigma = par.Sigma0 * exp( -Rcirc / par.Rdisk );
    // squared radial velocity dispersion is exponential in radius
    double sigmarsq = pow_2(par.sigmar0 * exp ( -Rcirc / par.Rsigmar ) ) + pow_2(par.sigmamin);
    // squared vertical velocity dispersion computed by either of the two methods: 
    double sigmazsq = pow_2(par.sigmamin) + (par.Hdisk>0 ?
        2 * pow_2(nu * par.Hdisk) :     // keep the disk thickness approximately equal to Hdisk, or
        pow_2(par.sigmaz0 * exp ( -Rcirc / par.Rsigmaz ) ) );  // make sigmaz exponential in radius
    // suppression factor for counterrotating orbits
    double negJphi = J.Jphi>0 ? 0. : 2*Omega * J.Jphi;
    double result = 1./(2*M_PI*M_PI) * Sigma * nu * Omega / (kappa * sigmarsq * sigmazsq) *
        averageOverAge( (kappa * J.Jr - negJphi) / sigmarsq + nu * J.Jz / sigmazsq, par);
    return isFinite(result) ? result : 0;
}


Exponential::Exponential(const ExponentialParam& params) :
    par(params)
{
    if(par.Jr0<=0 || par.Jz0<=0 || par.Jphi0<=0)
        throw std::invalid_argument("Exponential: scale actions must be positive");
    if(par.sigmabirth<=0 || par.sigmabirth>1)
        throw std::invalid_argument("Exponential: invalid value for velocity dispersion at birth");
}

double Exponential::value(const actions::Actions &J) const
{
    // weighted sum of actions
    double Jsum = fabs(J.Jphi) + par.coefJr * J.Jr + par.coefJz * J.Jz;
    double Jden = sqrt(pow_2(Jsum) + pow_2(par.addJden));
    double Jvel = sqrt(pow_2(Jsum) + pow_2(par.addJvel));
    // suppression factor for counterrotating orbits
    double negJphi = J.Jphi>0 ? 0. : J.Jphi;
    return 1. / TWO_PI_CUBE * par.mass / pow_2(par.Jr0 * par.Jz0 * par.Jphi0) *
        Jvel * Jvel * Jden * exp(-Jden / par.Jphi0) *
        averageOverAge(Jvel * ((J.Jr - negJphi) / pow_2(par.Jr0) + J.Jz / pow_2(par.Jz0)), par);
}

}  // namespace df
