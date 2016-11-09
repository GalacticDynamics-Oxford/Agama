/** \file    df_disk.h
    \brief   Distribution function for the disk component
*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"

namespace df{

/// \name   Class for action-based pseudo-isothermal disk distribution function (DF)
///@{

/// Parameters that describe a pseudo-isothermal distribution function.
struct PseudoIsothermalParam{
double
    Sigma0,   ///< surface density normalization (value at R=0)
    Rdisk,    ///< scale radius of the (exponential) disk surface density
    Jphimin,  ///< lower cutoff for evaluating epicyclic frequencies: take max(Jphi,Jphimin)
    // parameters describing the dependence of DF on the azimuthal, radial and vertical actions
    Jphi0,    ///< scale angular momentum determining the suppression of retrograde orbits
    sigmar0,  ///< normalization of radial velocity dispersion at R=0
    sigmaz0,  ///< normalization of vertical velocity dispersion at R=0
    sigmamin, ///< lower limit on the radial velocity dispersion: take max(sigmar,sigmamin)
    Rsigmar,  ///< scale radius of radial velocity dispersion: sigmar=sigmar0*exp(-R/Rsigmar)
    Rsigmaz,  ///< scale radius of vertical velocity dispersion (default for both should be 2*Rdisk)
    // parameters controlling the age-velocity dispersion relation (set beta=0 to disable)
    beta,     ///< factor describing the growth of velocity dispersion with age
    Tsfr,     ///< timescale for exponential decline of star formation rate in units of galaxy age
    sigmabirth;///< ratio of velocity dispersion at birth to the one at maximum age
PseudoIsothermalParam() :  ///< set default values for all fields
    Sigma0(0), Rdisk(0), Jphimin(0), Jphi0(0), sigmar0(0), sigmaz0(0), sigmamin(0),
    Rsigmar(0), Rsigmaz(0), beta(0), Tsfr(INFINITY), sigmabirth(1) {}
};

/** Distribution function for quasi-isothermal disk, used in Binney&McMillan 2011, Binney&Piffl 2015.
    \f$  f(J) = f_r(J_r, J_\phi)  f_z(J_z, J_\phi)  f_\phi(J_\phi)  \f$, where
    \f$  f_r  = \Omega(R_c) \Sigma(R_c) / (\pi \kappa(R_c) \sigma_r^2(R_c) )  \f$,
    \f$  f_z  = \nu(R_c) / (2\pi \sigma_z^2(R_c) )  \f$,
    \f$  f_\phi = 1 + \tanh( J_\phi / J_{\phi,0} )  \f$,
    \f$  \Sigma(R_c)   = \Sigma_0 \exp( -R_c / R_{disk} )  \f$,
    \f$  \sigma_r(R_c) = \sigma_{r,0} \exp( -R_c / R_{sigma,r} )  \f$,
    \f$  \sigma_z(R_c) = \sigma_{z,0} \exp( -R_c / R_{sigma,z} )  \f$
    for the simple case of a single population.

    In the case that  \f$  beta \ne 0  \f$, the above expressions are modified to take account
    of velocity dispersion that varies with age, together with the assumed exponentially declining
    star formation rate. Specifically, \f$  f_r, f_z  \f$  are replaced by integrals over ages:
    \f$  f_r f_z = [ \int_0^1  dt  S(t)  f_r(\sigma_r(t))  f_z(\sigma_z(t)) ] / [ \int_0^1 S(t) dt ] \f$,
    where t is the look-back time (stellar age) measured in units of galaxy age (varies from 0 to 1),
    star formation rate was greater in the past -- \f$  S(t) = \exp(t/T_{SFR})  \f$, and velocity
    dispersions scale with age as  \f$  \sigma_{r,z}  \propto  [ (t+t_1) / (1+t_1) ]^\beta  \f$,
    where t_1 is defined so that the ratio \f$ \sigma(t=0) / \sigma(t=1) \f$ = sigmabirth <= 1.

    The DF is defined in terms of surface density \f$ \Sigma \f$, velocity dispersions
    \f$ \sigma_r, \sigma_z \f$, and characteristic epicyclic frequencies
    \f$ \kappa, \nu, \Omega \f$, all evaluated at a radius R_c that corresponds to the radius
    of a circular orbit with angular momentum L_z = J_phi (recall that the DF is defined 
    in terms of actions only, not coordinates).
    Note, however, that the while these frequencies are computed from a particular potential
    and passed as the interpolator object, the DF uses them simply as one-parameter functions
    of the azimuthal actions, without regard to whether they actually correspond to
    the epicyclic frequencies in the potential that this DF is used.
    In other words, action-based DF may only depend on actions and on some arbitrary function
    of them, but not explicitly on the potential.
*/
class PseudoIsothermal: public BaseDistributionFunction{
    const PseudoIsothermalParam par;     ///< parameters of DF
    const potential::Interpolator freq;  ///< interface providing the epicyclic frequencies
    static const int NT = 10; ///< number of points in quadrature rule for integration over age
    double qx[NT], qw[NT];    ///< nodes and weights for quadrature rule
public:
    /** Create an instance of pseudo-isothermal distribution function with given parameters
        \param[in] params  are the parameters of DF;
        \param[in] freq   is the instance of object that computes epicyclic frequencies:
        a copy of this object is kept in the DF, so that one may pass a temporary variable
        as the argument, like:
        ~~~~
            df = new PseudoIsothermal(params, potential::Interpolator(potential));
        ~~~~
        Since the potential itself is not used in the Interpolator object,
        the DF is independent of the actual potential in which it is later used,
        which could be different from the one that was employed at construction.
        \throws std::invalid_argument exception if parameters are nonsense
    */
    PseudoIsothermal(const PseudoIsothermalParam& params, const potential::Interpolator& freq);

    /** return value of DF for the given set of actions
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;
};

///@}
}  // namespace df
