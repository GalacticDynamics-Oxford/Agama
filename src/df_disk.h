/** \file    df_disk.h
    \brief   Distribution function for the disk component
    \author  Eugene Vasiliev
    \date    2015-2024
*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"

namespace df{

/// \name   Classes for action-based disk distribution functions (DF)
///@{

/// Parameters that describe a quasi-isothermal distribution function.
struct QuasiIsothermalParam{
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
    qJr,      ///< parameter of the q-exponential distribution in Jr
    qJz,      ///< parameter of the q-exponential distribution in Jz
    qJphi,    ///< parameter of the q-exponential distribution in Jphi
    Jmin;     ///< lower cutoff for evaluating epicyclic frequencies: tildeJ = sqrt(Jsum^2 + Jmin^2)
QuasiIsothermalParam() :  ///< set default values for all fields (NAN means that it must be set manually)
    Sigma0(NAN), Rdisk(NAN), Hdisk(NAN), sigmar0(NAN), sigmaz0(NAN), sigmamin(0),
    Rsigmar(NAN), Rsigmaz(NAN), coefJr(1.), coefJz(0.25), qJr(0), qJz(0), qJphi(0), Jmin(0) {}
};

/** Distribution function for quasi-isothermal disk, similar but somewhat different from the one
    used in Binney&McMillan 2011, Binney&Piffl 2015:
    \f$  f(J) = F(R_c)  f_r(J_r, R_c)  f_z(J_z, R_c)  f_\phi(J_\phi, R_c)  \f$, where
    \f$  F    = \Sigma(R_c) \Omega(R_c) / (2\pi^2 \kappa^2(R_c) (1-q_{J_\phi} )  \f$,
    \f$  f_r  = \kappa(R_c) / \sigma_r^2(R_c)  \exp( - \kappa J_r / \sigma_r^2)  \f$,
    \f$  f_z  = \nu(R_c)    / \sigma_z^2(R_c)  \exp( - \nu    J_z / \sigma_z^2)  \f$,
    \f$  f_\phi = max[1, \exp( 2\Omega J_\phi / \sigma_r^2 ) ]  \f$,
    \f$  \Sigma(R_c)   = \Sigma_0     \exp_q( -R_c / R_{disk}, -q_{J_\phi})  \f$,
    \f$  \sigma_r(R_c) = \sigma_{r,0} \exp_q( -R_c / R_{sigma,r}, -q_{J_r})  \f$,
    and two options for the vertical velocity dispersion: either
    \f$  \sigma_z(R_c) = \sqrt{2} H_{disk} \nu(R_c)  \f$  if Hdisk>0, or
    \f$  \sigma_z(R_c) = \sigma_{z,0} \exp_q( -R_c / R_{sigma,z}, -q_{J_z})  \f$ 
    if sigmaz0>0 and Rsigmaz>0.
    Here \f$ \exp_q(x, q) = \exp(x) \f$ when q=0, or \f$ (1 + q x)^{1/q} \f$ when 0 < q < 1.

    The DF is defined in terms of surface density \f$ \Sigma \f$, velocity dispersions
    \f$ \sigma_r, \sigma_z \f$, and characteristic epicyclic frequencies
    \f$ \kappa, \nu, \Omega \f$, all evaluated at a radius R_c(J) that corresponds to the radius of
    a circular orbit with angular momentum \f$ J = \sqrt{J_{min}^2 + (|J_phi| + k_r J_r + k_z J_z)^2},
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

    /** compute the value of DF for the given set of actions, and optionally its derivatives */
    virtual void evalDeriv(const actions::Actions &J,
        /*output*/ double *value, DerivByActions *deriv=NULL) const;
};


/// Parameters that describe the exponential distribution function
struct ExponentialParam{
double
    norm,     ///< overall normalization factor with the dimension of mass (NOT the actual mass)
    Jr0,      ///< scale action setting the radial velocity dispersion
    Jz0,      ///< scale action setting the disk thickness and the vertical velocity dispersion
    Jphi0,    ///< scale action setting the disk radius
    addJden,  ///< additional contribution to the sum of actions that affects the density profile
    addJvel,  ///< same for the part that affects the velocity dispersion profiles
    coefJr,   ///< weight of radial action in the sum of actions
    coefJz,   ///< same for the vertical action
    qJr,      ///< parameter of the q-exponential distribution in Jr
    qJz,      ///< parameter of the q-exponential distribution in Jz
    qJphi;    ///< parameter of the q-exponential distribution in Jphi
ExponentialParam() :  ///< set default values for all fields
    norm(NAN), Jr0(NAN), Jz0(NAN), Jphi0(NAN), addJden(0), addJvel(0), coefJr(1.0), coefJz(0.25),
    qJr(0), qJz(0), qJphi(0) {}
};

/** Another type of disk distribution function, which resembles QuasiIsothermal and produces
    disk-like profiles, but does not explicitly reference any potential.

    \f$  f(J)   = F(Jsum)  f_r(J_r, Jsum)  f_z(J_z, Jsum)  f_\phi(J_\phi, Jsum)  \f$, where
    \f$  Jsum   = |J_\phi| + k_r J_r + k_z J_z  \f$  is a linear combination of actions,
    \f$  F      = M / (2\pi)^3 Jden / J_{\phi,0}^2  \exp_q( -Jden / J_{\phi,0}, -q_{J_\phi} )  \f$,
    \f$  f_r    = Jvel / J_{r,0}^2 \exp_q( -J_r Jvel / J_{r,0}^2, -q_{J_r} )  \f$,
    \f$  f_z    = Jvel / J_{z,0}^2 \exp_q( -J_z Jvel / J_{z,0}^2, -q_{J_z} )  \f$,
    \f$  f_\phi = max[1,  \exp( [2/k_r] J_\phi Jvel / J_{r,0}^2 ) ]  \f$,
    \f$  Jden   = \sqrt{ Jsum^2 + addJden^2 }  \f$,
    \f$  Jvel   = \sqrt{ Jsum^2 + addJvel^2 }  \f$,

    The parameters  \f$  J_{\phi,0}, J_{z,0}, J_{r,0}  \f$  determine the radial scale length,
    vertical scale height (and hence the vertical velocity dispersion),
    and radial velocity dispersion, respectively;
    addJden and addJvel allow to tweak the radial dependence of density profile and velocity
    dispersion profiles at small radii;
    and dimensionless mixing coefficients k_r, k_z are of order unity.
*/
class Exponential: public df::BaseDistributionFunction{
    const ExponentialParam par;     ///< parameters of the DF
public:
    Exponential(const ExponentialParam& params);
    virtual void evalDeriv(const actions::Actions &J,
        /*output*/ double *value, DerivByActions *deriv=NULL) const;
};

///@}
}  // namespace df
