/** \file    df_halo.h
    \brief   Distribution functions for the spheroidal component (halo)
    \author  Eugene Vasiliev, James Binney
    \date    2015-2024
*/
#pragma once
#include "df_base.h"

namespace df{

/// \name   Classes for action-based double power-law distribution function (DF)
///@{

/// Parameters that describe a double power law distribution function.
struct DoublePowerLawParam{
double
    norm,      ///< normalization factor with the dimension of mass
    J0,        ///< break action (defines the transition between inner and outer regions)
    Jcutoff,   ///< cutoff action (sets exponential suppression at J>Jcutoff, INFINITY disables it)
    slopeIn,   ///< power-law index for actions below the break action (Gamma)
    slopeOut,  ///< power-law index for actions above the break action (Beta)
    steepness, ///< steepness of the transition between two asymptotic regimes (eta)
    cutoffStrength, ///< steepness of exponential suppression at J>Jcutoff (zeta)
    coefJrIn,  ///< contribution of radial   action to h(J), controlling anisotropy below J_0 (h_r)
    coefJzIn,  ///< contribution of vertical action to h(J), controlling anisotropy below J_0 (h_z)
    coefJrOut, ///< contribution of radial   action to g(J), controlling anisotropy above J_0 (g_r)
    coefJzOut, ///< contribution of vertical action to g(J), controlling anisotropy above J_0 (g_z)
    rotFrac,   ///< relative amplitude of the odd-Jphi component (-1 to 1, 0 means no rotation)
    Jphi0,     ///< controls the steepness of rotation and the size of non-rotating core
    Jcore;     ///< central core size for a Cole&Binney-type modified double-power-law halo
DoublePowerLawParam() :  ///< set default values for all fields (NAN means that it must be set manually)
    norm(NAN), J0(NAN), Jcutoff(INFINITY), slopeIn(NAN), slopeOut(NAN), steepness(1), cutoffStrength(2),
    coefJrIn(1), coefJzIn(1), coefJrOut(1), coefJzOut(1), rotFrac(0), Jphi0(0), Jcore(0) {}
};

/** General double power-law model.
    The distribution function is given by
    \f$  f(J) = norm / (2\pi J_0)^3
         (1 + (J_0 /h(J))^\eta )^{\Gamma / \eta}
         (1 + (g(J)/ J_0)^\eta )^{-B / \eta }
         \exp[ - (g(J) / J_{cutoff})^\zeta ]
         ( [J_{core} / h(J)]^2 - \beta J_{core} / h(J) + 1)^{-\Gamma/2}
         [1 + \kappa \tanh(J_\phi / J_{\phi,0}) ]   \f$,  where
    \f$  g(J) = g_r J_r + g_z J_z + g_\phi |J_\phi|  \f$,
    \f$  h(J) = h_r J_r + h_z J_z + h_\phi |J_\phi|  \f$.
    Gamma is the power-law slope of DF at small J (slopeIn), and Beta -- at large J (slopeOut),
    the transition occurs around J=J0, and its steepness is adjusted by the parameter eta.
    h_r, h_z and h_phi control the anisotropy of the DF at small J (their sum is always taken
    to be equal to 3, so that there are two free parameters -- coefJrIn = h_r, coefJzIn = h_z),
    and g_r, g_z, g_phi do the same for large J (coefJrOut = g_r, coefJzOut = g_z).
    Jcutoff is the threshold of an optional exponential suppression, and zeta measures its strength.
    Jcore is the size of the central core (if nonzero, f(J) tends to a constant limit as J --> 0
    even when the power-law slope Gamma is positive), and the auxiliary coefficient beta
    is assigned automatically from the requirement that the introduction of the core (almost)
    doesn't change the overall normalization (eq.5 in Cole&Binney 2017).
*/
class DoublePowerLaw: public BaseDistributionFunction{
    const DoublePowerLawParam par;  ///< parameters of DF
    const double beta;              ///< auxiliary coefficient for the case of a central core
public:
    /** Create an instance of double-power-law distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    DoublePowerLaw(const DoublePowerLawParam &params);

    /** compute the value of DF for the given set of actions, and optionally its derivatives */
    virtual void evalDeriv(const actions::Actions &J,
        /*output*/ double *value, DerivByActions *deriv=NULL) const;
};

///@}
}  // namespace df
