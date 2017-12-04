/** \file    df_disk.h
    \brief   Distribution function for the disk component
    \author  Eugene Vasiliev
    \date    2015-2017
*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"

namespace df{

/// \name   Classes for action-based disk distribution functions (DF)
///@{

/// parameters controlling the age-velocity dispersion relation
struct AgeVelocityDispersionParam{
double
    beta,       ///< factor describing the growth of velocity dispersion with age
    Tsfr,       ///< timescale for exponential decline of star formation rate in units of galaxy age
    sigmabirth; ///< ratio of velocity dispersion (sigma) at birth to the one at maximum age
AgeVelocityDispersionParam() :  ///< set default values disabling any variation
    beta(0), Tsfr(INFINITY), sigmabirth(1) {}
};


/// Parameters that describe a quasi-isothermal distribution function.
struct QuasiIsothermalParam: public AgeVelocityDispersionParam{
double
    Sigma0,   ///< surface density normalization (value at R=0)
    Rdisk,    ///< scale radius of the (exponential) disk surface density
    Hdisk,    ///< scale height of the disk (determines the vertical velocity dispersion:
              ///< sigma_z(R) = sqrt(2) nu(R) Hdisk), is mutually exclusive with {sigmaz0,Rsigmaz}
    // parameters describing the dependence of DF on the azimuthal, radial and vertical actions
    sigmar0,  ///< normalization of radial velocity dispersion at R=0
    sigmaz0,  ///< normalization of vertical velocity dispersion at R=0, is mutually exclusive with Hdisk
    sigmamin, ///< lower limit on the radial and vertical velocity dispersions
    Rsigmar,  ///< scale radius of radial velocity dispersion: sigmar=sigmar0*exp(-R/Rsigmar)
    Rsigmaz,  ///< scale radius of vertical velocity dispersion, is mutually exclusive with Hdisk
    // parameters determining the mapping between actions and radius
    coefJr,   ///< weight of radial action in the linear combination of actions Jsum = Jphi + cr Jr + cz Jz
    coefJz,   ///< same for the vertical action
    Jmin;     ///< lower cutoff for evaluating epicyclic frequencies: tildeJ = sqrt(Jsum^2 + Jmin^2)
QuasiIsothermalParam() :  ///< set default values for all fields
    Sigma0(0), Rdisk(0), Hdisk(0), sigmar0(0), sigmaz0(0), sigmamin(0),
    Rsigmar(0), Rsigmaz(0), coefJr(1.), coefJz(0.25), Jmin(0) {}
};

/** Distribution function for quasi-isothermal disk, similar but somewhat different from the one
    used in Binney&McMillan 2011, Binney&Piffl 2015.
    In the case of a single population (beta=0 or sigmabirth=1), it is given by
    \f$  f(J) = F(R_c)  f_r(J_r, R_c)  f_z(J_z, R_c)  f_\phi(J_\phi, R_c)  \f$, where
    \f$  F    = \Sigma(R_c) \Omega(R_c) / (2\pi^2 \kappa^2(R_c) )  \f$,
    \f$  f_r  = \kappa(R_c) / \sigma_r^2(R_c)  \exp( - \kappa J_r / \sigma_r^2)  \f$,
    \f$  f_z  = \nu(R_c)    / \sigma_z^2(R_c)  \exp( - \nu    J_z / \sigma_z^2)  \f$,
    \f$  f_\phi = max[1, \exp( 2\Omega J_\phi / \sigma_r^2 ) ]  \f$,
    \f$  \Sigma(R_c)   = \Sigma_0 \exp( -R_c / R_{disk} )  \f$,
    \f$  \sigma_r(R_c) = \sigma_{r,0} \exp( -R_c / R_{sigma,r} )  \f$,
    and two options for the vertical velocity dispersion: either
    \f$  \sigma_z(R_c) = \sqrt{2} H_{disk} \nu(R_c)  \f$  if Hdisk!=0, or
    \f$  \sigma_z(R_c) = \sigma_{z,0} \exp( -R_c / R_{sigma,z} )  \f$  if sigmaz0!=0 and Rsigmaz!=0.

    In the case that  \f$  \beta \ne 0  \f$, the above expressions are modified to take account
    of velocity dispersion that varies with age, together with the assumed exponentially declining
    star formation rate. Specifically, \f$  f_r, f_z  \f$  are replaced by integrals over ages:
    \f$  f_r f_z = [ \int_0^1  dt  S(t)  f_r(\sigma_r(t))  f_z(\sigma_z(t)) ] / [ \int_0^1 S(t) dt ] \f$,
    where t is the look-back time (stellar age) measured in units of galaxy age (varies from 0 to 1),
    star formation rate was greater in the past -- \f$  S(t) = \exp(t/T_{SFR})  \f$, and velocity
    dispersions scale with age as  \f$  \sigma_{r,z}  \propto  [ t + (1-t) \xi^{1/\beta} ]^\beta  \f$,
    where \f$ \xi = \sigma(t=0) / \sigma(t=1) \f$ = sigmabirth <= 1.

    The DF is defined in terms of surface density \f$ \Sigma \f$, velocity dispersions
    \f$ \sigma_r, \sigma_z \f$, and characteristic epicyclic frequencies
    \f$ \kappa, \nu, \Omega \f$, all evaluated at a radius R_c(J) that corresponds to the radius of
    a circular orbit with angular momentum J = max(Jmin, |J_phi| + k_r J_r + k_z J_z),
    with dimensionless mixing coefficients k_r, k_z of order unity. Hence the DF is expressed in terms
    of actions, using auxiliary functions of radius which are ultimately also functions of actions.
    Note, however, that the while these frequencies are computed from a particular potential
    and passed as the interpolator object, the DF uses them simply as one-parameter functions
    without regard to whether they actually correspond to the epicyclic frequencies in the potential
    that this DF is used. In other words, action-based DF may only depend on actions and on some
    arbitrary function of them, but not explicitly on the potential.
*/
class QuasiIsothermal: public BaseDistributionFunction{
    const QuasiIsothermalParam par;      ///< parameters of the DF
    const potential::Interpolator freq;  ///< interface providing the epicyclic frequencies and Rcirc
public:
    /** Create an instance of quasi-isothermal distribution function with given parameters
        \param[in] params  are the parameters of DF;
        \param[in] freq   is the instance of object that computes epicyclic frequencies:
        a copy of this object is kept in the DF, so that one may pass a temporary variable
        as the argument, like:
        ~~~~
            df = new QuasiIsothermal(params, potential::Interpolator(potential));
        ~~~~
        Since the potential itself is not used in the Interpolator object,
        the DF is independent of the actual potential in which it is later used,
        which could be different from the one that was employed at construction.
        \throws std::invalid_argument exception if parameters are nonsense
    */
    QuasiIsothermal(const QuasiIsothermalParam& params, const potential::Interpolator& freq);

    /** return value of DF for the given set of actions
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;
};


/// Parameters that describe the exponential distribution function
struct ExponentialParam: public AgeVelocityDispersionParam{
double
    mass,       ///< overall normalization factor with the dimension of mass (NOT the actual mass)
    Jr0,        ///< scale action setting the radial velocity dispersion
    Jz0,        ///< scale action setting the disk thickness and the vertical velocity dispersion
    Jphi0,      ///< scale action setting the disk radius
    addJden,    ///< additional contribution to the sum of actions that affects the density profile
    addJvel,    ///< same for the part that affects the velocity dispersion profiles
    coefJr,     ///< weight of radial action in the sum of actions
    coefJz;     ///< same for the vertical action
ExponentialParam() :  ///< set default values for all fields
    mass(0), Jr0(0), Jz0(0), Jphi0(0), addJden(0), addJvel(0), coefJr(1.0), coefJz(0.25) {}
};

/** Another type of disk distribution function, which resembles QuasiIsothermal and produces
    disk-like profiles, but does not explicitly reference any potential.

    \f$  f(J)   = F(Jsum)  f_r(J_r, Jsum)  f_z(J_z, Jsum)  f_\phi(J_\phi, Jsum)  \f$, where
    \f$  Jsum   = |J_\phi| + k_r J_r + k_z J_z  \f$  is a linear combination of actions,
    \f$  F      = M / (2\pi)^3 Jden / J_{\phi,0}^2  \exp( - Jden / J_{\phi,0} )  \f$,
    \f$  f_r    = Jvel / J_{r,0}^2 \exp( - J_r Jvel / J_{r,0}^2 )    \f$,
    \f$  f_r    = Jvel / J_{z,0}^2 \exp( - J_z Jvel / J_{z,0}^2 )    \f$,
    \f$  f_\phi = max[1,  \exp( [2/k_r] J_\phi Jvel / J_{r,0}^2 ) ]  \f$,
    \f$  Jden   = \sqrt{ Jsum^2 + addJden^2 }  \f$,
    \f$  Jvel   = \sqrt{ Jsum^2 + addJvel^2 }  \f$,

    The parameters  \f$  J_{\phi,0}, J_{z,0}, J_{r,0}  \f$  determine the radial scale length,
    vertical scale height (and hence the vertical velocity dispersion),
    and radial velocity dispersion, respectively;
    addJden and addJvel allow to tweak the radial dependence of density profile and velocity
    dispersion profiles at small radii;
    and dimensionless mixing coefficients k_r, k_z are of order unity.

    Generalization to the case of age--velocity dispersion relation is similar to QuasiIsothermal.
*/
class Exponential: public df::BaseDistributionFunction{
    const ExponentialParam par;     ///< parameters of the DF
public:
    Exponential(const ExponentialParam& params);    
    virtual double value(const actions::Actions &J) const;
};

///@}
}  // namespace df
