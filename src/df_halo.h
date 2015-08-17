/** \file    df_halo.h
    \brief   Distribution function for the spheroidal component (halo)
*/
#pragma once
#include "df_base.h"

namespace df{

/// \name   Class for action-based double power-law distribution function (DF)
///@{

/// Parameters that describe a double power law distribution function.
struct DoublePowerLawParam{
    double jcore; ///< core action in inner part
    double alpha; ///< power-law index for actions below the break action
    double beta;  ///< power-law index for actions above the break action
    double j0;    ///< break action
    double ar;    ///< weight on radial actions below the break action
    double az;    ///< weight on z actions below the break action
    double aphi;  ///< weight oh angular actions below the break action
    double br;    ///< weight on radial actions above the break action
    double bz;    ///< weight on z actions above the break action
    double bphi;  ///< weight on angular actions above the break action
};

/** Posti et al(2015) double power-law model.
    The distribution function is given by
    \f$  f(J) = ( 1 + J_0 / (h(J) + J_{core}) )^\alpha / ( 1 + g(J) / J_0 )^\beta  \f$,
    where  \f$  h(J) = a_r J_r + a_z J_z + a_\phi J_\phi  \f$
    is the linear combination of actions in the inner part of the model (below the break action J_0),
    and    \f$  g(J) = b_r J_r + b_z J_z + b_\phi J_\phi  \f$
    is another linear combination in the outer part of the model.
*/
class DoublePowerLaw: public BaseDistributionFunction{
private:
    DoublePowerLawParam par;
public:
    /** Create an instance of double-power-law distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    explicit DoublePowerLaw(const DoublePowerLawParam &params);

    /** return value of DF for the given set of actions (not scaled by total mass).
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;

    /** The maximum probability (not scaled by total mass). */
    virtual double maxValue() const;
};

///@}
}  // namespace df
