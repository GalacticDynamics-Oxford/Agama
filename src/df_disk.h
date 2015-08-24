/** \file    df_disk.h
    \brief   Distribution function for the disk component
*/
#pragma once
#include "df_base.h"
#include "potential_base.h"

namespace df{

/// \name   Class for action-based pseudo-isothermal disk distribution function (DF)
///@{

/// Parameters that describe a double power law distribution function.
struct PseudoIsothermalParam{
    double norm;    ///< overall normalization factor
    double Rdisk;   ///< scale radius of the (exponential) disk surface density
    double L0;      ///< scale angular momentum determining the suppression of retrograde orbits
    double Sigma0;  ///< surface density normalization (value at origin)
    double sigmar0; ///< normalization of radial velocity dispersion at Rdisk
    double sigmaz0; ///< normalization of vertical velocity dispersion at Rdisk
};

class PseudoIsothermal: public BaseDistributionFunction{
private:
    PseudoIsothermalParam par;
    const potential::BasePotential& potential;
public:
    /** Create an instance of pseudo-isothermal distribution function with given parameters
        \param[in] params  are the parameters of DF
        \throws std::invalid_argument exception if parameters are nonsense
    */
    PseudoIsothermal(const PseudoIsothermalParam& params, const potential::BasePotential& pot);

    /** return value of DF for the given set of actions
        \param[in] J are the actions  */
    virtual double value(const actions::Actions &J) const;
};

///@}
}  // namespace df
