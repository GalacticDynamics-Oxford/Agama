#include "df_disk.h"
#include "math_specfunc.h"
#include <cmath>
#include <stdexcept>

namespace df{

QuasiIsothermal::QuasiIsothermal(const QuasiIsothermalParam &params, const potential::Interpolator& freqs) :
    par(params), freq(freqs)
{
    // sanity checks on parameters
    if(!(par.Sigma0>0))
        throw std::invalid_argument("QuasiIsothermal: surface density Sigma0 must be positive");
    if(!(par.Rdisk>0))
        throw std::invalid_argument("QuasiIsothermal: disk scale radius Rdisk must be positive");
    if(!(par.sigmar0>0))
        throw std::invalid_argument("QuasiIsothermal: velocity dispersion sigmar0 must be positive");
    if(!(par.Rsigmar>0))
        throw std::invalid_argument("QuasiIsothermal: velocity scale radius Rsigmar must be positive");
    if(!( (par.Hdisk>0) ^ (par.sigmaz0>0 && par.Rsigmaz>0) ))
        throw std::invalid_argument("QuasiIsothermal: should have either "
            "Hdisk>0 to assign the vertical velocity dispersion from disk scaleheight, or "
            "Rsigmaz>0, sigmaz0>0 to make it exponential in radius");
    if(par.Hdisk<0 || par.sigmaz0<0 || par.Rsigmaz<0)  // these are optional but non-negative
        throw std::invalid_argument("QuasiIsothermal: parameters cannot be negative");
    if(!(par.coefJr>=0 && par.coefJz>=0))
        throw std::invalid_argument("QuasiIsothermal: mixing coefficients for Jr,Jz must be >=0");
    if(!(par.qJr>=0 && par.qJr<1 && par.qJz>=0 && par.qJz<1 && par.qJphi>=0 && par.qJphi<1))
        throw std::invalid_argument("QuasiIsothermal: q-coefficients must be >=0 and <1");
}

void QuasiIsothermal::evalDeriv(const actions::Actions &J,
    double *value, DerivByActions *deriv) const
{
    // weighted sum of actions
    double coefJphi = J.Jphi >= 0 ? 1 : -1,
    Jsum = coefJphi * J.Jphi + par.coefJr * J.Jr + par.coefJz * J.Jz,
    Jhat = sqrt(pow_2(Jsum) + pow_2(par.Jmin)),
    // radius of in-plane motion with the given "characteristic" angular momentum and its derivative
    dRcirc_dJhat,
    Rcirc = freq.R_from_Lz(Jhat, deriv ? &dRcirc_dJhat : NULL),
    kappa, nu, Omega, der_epifreq[3];   // characteristic epicyclic freqs and their radial derivatives
    freq.epicycleFreqs(Rcirc, kappa, nu, Omega, deriv ? der_epifreq : NULL);
    double
    // inverse squared radial velocity dispersion is exponential in radius
    invsigmarsq = 1 / (pow_2(par.sigmar0 * exp (-Rcirc / par.Rsigmar) ) + pow_2(par.sigmamin)),
    // inverse squared vertical velocity dispersion computed by either of the two methods:
    invsigmazsq = 1 / (pow_2(par.sigmamin) + (par.Hdisk>0 ?
        2 * pow_2(nu * par.Hdisk) :     // keep the disk thickness approximately equal to Hdisk, or
        pow_2(par.sigmaz0 * exp (-Rcirc / par.Rsigmaz) ) ) ),  // make sigmaz exponential in radius
    // suppression factor for counterrotating orbits
    negJphi = J.Jphi>0 ? 0. : 2*Omega * J.Jphi,
    // arguments of q-exponential functions for Jr, Jz and Rcirc(J)
    argJr   = (kappa * J.Jr - negJphi) * invsigmarsq,
    argJz   = nu * J.Jz * invsigmazsq,
    argRc   = Rcirc / par.Rdisk;
    *value  = 1./(2*M_PI*M_PI) * par.Sigma0 * (1 - par.qJphi) * (1 - par.qJr) * (1 - par.qJz) *
        nu * Omega / kappa * invsigmarsq * invsigmazsq *
        math::qexp(-argRc, -par.qJphi) *
        math::qexp(-argJr, -par.qJr) *
        math::qexp(-argJz, -par.qJz);

    if(deriv) {
        double
        dlnkappa_dRcirc = der_epifreq[0] / kappa,
        dlnnu_dRcirc    = der_epifreq[1] / nu,
        dlnOmega_dRcirc = der_epifreq[2] / Omega,
        dRcirc_dJ = dRcirc_dJhat * Jsum / Jhat,  // multiplied by coefJr,coefJz,coefJphi respectively
        dlnsigmarsq_dRcirc = -2 / par.Rsigmar * (1 - pow_2(par.sigmamin) * invsigmarsq),
        dlnsigmazsq_dRcirc = -(par.Hdisk>0 ?
            -pow_2(2 * nu * par.Hdisk) * invsigmazsq * dlnnu_dRcirc :
            2 / par.Rsigmaz * (1 - pow_2(par.sigmamin) * invsigmazsq) ),
        common = dRcirc_dJ * (
            dlnnu_dRcirc + dlnOmega_dRcirc - dlnkappa_dRcirc - dlnsigmarsq_dRcirc - dlnsigmazsq_dRcirc -
            1 / (par.Rdisk + par.qJphi * Rcirc) -
            kappa * invsigmarsq * (dlnkappa_dRcirc - dlnsigmarsq_dRcirc) * J.Jr / (1 + par.qJr * argJr) -
            nu    * invsigmazsq * (   dlnnu_dRcirc - dlnsigmazsq_dRcirc) * J.Jz / (1 + par.qJz * argJz) );
        if(negJphi)
            common += dRcirc_dJ * 2 * Omega * invsigmarsq *
                (dlnOmega_dRcirc - dlnsigmarsq_dRcirc) * J.Jphi / (1 + par.qJr * argJr);

        deriv->dbyJr   = *value * (common * par.coefJr - kappa * invsigmarsq / (1 + par.qJr * argJr));
        deriv->dbyJz   = *value * (common * par.coefJz - nu    * invsigmazsq / (1 + par.qJz * argJz));
        deriv->dbyJphi = *value * (common * coefJphi);
        if(negJphi)
            deriv->dbyJphi += *value * 2 * Omega * invsigmarsq / (1 + par.qJr * argJr);
    }
}


Exponential::Exponential(const ExponentialParam& params) :
    par(params)
{
    if(!(par.norm>0))
        throw std::invalid_argument("Exponential: overall normalization must be positive");
    if(!(par.Jr0>0 && par.Jz0>0 && par.Jphi0>0))
        throw std::invalid_argument("Exponential: scale actions must be positive");
    if(!(par.addJden>=0 && par.addJvel>=0))
        throw std::invalid_argument("Exponential: additive constants addJden, addJvel must be >=0");
    if(!(par.coefJr>=0 && par.coefJz>=0))
        throw std::invalid_argument("Exponential: mixing coefficients for Jr,Jz must be >=0");
    if(!(par.qJr>=0 && par.qJr<1 && par.qJz>=0 && par.qJz<1 && par.qJphi>=0 && par.qJphi<1))
        throw std::invalid_argument("Exponential: q-coefficients must be >=0 and <1");
}

void Exponential::evalDeriv(const actions::Actions &J,
    double *value, DerivByActions *deriv) const
{
    // weighted sum of actions
    double coefJphi = J.Jphi >= 0 ? 1 : -1,
    Jsum = coefJphi * J.Jphi + par.coefJr * J.Jr + par.coefJz * J.Jz,
    Jden = sqrt(pow_2(Jsum) + pow_2(par.addJden)),
    Jvel = sqrt(pow_2(Jsum) + pow_2(par.addJvel)),
    negJphi = J.Jphi>=0 ? 0. : J.Jphi,  // suppression factor for counterrotating orbits
    argJphi = Jden / par.Jphi0,
    argJr   = Jvel * (J.Jr - negJphi) / pow_2(par.Jr0),
    argJz   = Jvel * (J.Jz          ) / pow_2(par.Jz0);
    *value  = 1. / TWO_PI_CUBE * par.norm / pow_2(par.Jr0 * par.Jz0 * par.Jphi0) *
        Jvel * Jvel * Jden *
        math::qexp(-argJphi, -par.qJphi) *
        math::qexp(-argJr  , -par.qJr) *
        math::qexp(-argJz  , -par.qJz);

    if(deriv) {
        double common = Jsum * (
            (1 - argJphi/ (1 + par.qJphi*argJphi)) / pow_2(Jden) +
            (1 - argJr  / (1 + par.qJr * argJr  )  +
             1 - argJz  / (1 + par.qJz * argJz  )) / pow_2(Jvel) );
        deriv->dbyJr   = *value * (common * par.coefJr - Jvel / pow_2(par.Jr0) / (1 + par.qJr * argJr));
        deriv->dbyJz   = *value * (common * par.coefJz - Jvel / pow_2(par.Jz0) / (1 + par.qJz * argJz));
        deriv->dbyJphi = *value * (common * coefJphi);
        if(negJphi)
            deriv->dbyJphi += *value * Jvel / pow_2(par.Jr0) / (1 + par.qJr * argJr);
    }
}

}  // namespace df
