/** \file    df_halo.h
    \author  Eugene Vasiliev
    \date    2015-2016
    \brief   Distribution functions for the spheroidal component (halo)

*/
#pragma once
#include "df_base.h"
#include "potential_utils.h"

namespace df{

/// \name   Classes for action-based double power-law distribution function (DF)
///@{

/// Parameters that describe a double power law distribution function.
struct DoublePowerLawParam{
double
    norm,     ///< normalization factor with the dimension of mass
    J0,       ///< break action (defines the transition between inner and outer regions)
    Jcutoff,  ///< cutoff action (sets exponential suppression at J>Jcutoff, 0 to disable)
    slopeIn,  ///< power-law index for actions below the break action (Gamma)
    slopeOut, ///< power-law index for actions above the break action (Beta)
    steepness,///< steepness of the transition between two asymptotic regimes (eta)
    coefJrIn, ///< contribution of radial   action to h(J), controlling anisotropy below J_0 (h_r)
    coefJzIn, ///< contribution of vertical action to h(J), controlling anisotropy below J_0 (h_z)
    coefJrOut,///< contribution of radial   action to g(J), controlling anisotropy above J_0 (g_r)
    coefJzOut;///< contribution of vertical action to g(J), controlling anisotropy above J_0 (g_z)
DoublePowerLawParam() :  ///< set default values for all fields
    norm(0), J0(0), Jcutoff(0), slopeIn(0), slopeOut(0), steepness(1),
    coefJrIn(1), coefJzIn(1), coefJrOut(1), coefJzOut(1) {}
};

/** General double power-law model.
    The distribution function is given by
    \f$  f(J) = norm / (2\pi J_0)^3  (h(J)/J_0)^{-\Gamma}
         (1 + (g(J)/J_0)^\eta )^{ (\Beta-\Gamma) / \eta }
         \exp[ - (g(J) / J_{cutoff})^2 ] \f$,  where
    \f$  g(J) = g_r J_r + g_z J_z + g_\phi |J_\phi|  \f$,
    \f$  h(J) = h_r J_r + h_z J_z + h_\phi |J_\phi|  \f$.
    Gamma is the power-law slope of DF at small J (slopeIn), and Beta -- at large J (slopeOut),
    the transition occurs around J=J0, and its steepness is adjusted by the parameter eta.
    h_r, h_z and h_\phi control the anisotropy of the DF at small J (their sum is always taken
    to be unity, so that there are two free parameters -- coefJrIn = h_r, coefJzIn = h_z),
    and g_r, g_z, g_\phi do the same for large J (coefJrOut = g_r, coefJzOut = g_z).
*/
class DoublePowerLaw: public BaseDistributionFunction{
    const DoublePowerLawParam par;  ///< parameters of DF
public:
    /** Create an instance of double-power-law distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    DoublePowerLaw(const DoublePowerLawParam &params);

    /** return value of DF for the given set of actions.
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;
};

///@}
}  // namespace df
