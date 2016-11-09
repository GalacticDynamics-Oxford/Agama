/** \file    df_pseudoisotropic.h
    \brief   DistributionFunction interface to spherical isotropic models
    \author  Eugene Vasiliev
    \date    2016
*/
#pragma once
#include "df_base.h"
#include "galaxymodel_spherical.h"
#include "actions_spherical.h"

namespace df{

/** A wrapper class for representing an arbitrary spherical isotropic distribution function,
    originally expressed in terms of phase volume `h`, as a regular action-based DF.
    The phase volume `h` in a spherically-symmetric potential is uniquely related to energy,
    and plays the same role as actions in a more general geometry: it is an adiabatic invariant,
    f(h) dh is the mass in the given volume of phase space regardless of the potential,
    and f(h) has the same expression in any potential.
    In order to put these models on the common ground with other action-based DFs,
    we express them in terms of actions, by using two successive mappings:
    (Jr, Jz+|Jphi|) => E  is provided by a spherical action mapper;
    E => h(E)  is provided by the PhaseVolume class,
    and finally h is used as the argument of DF.
    Both these auxiliary transformations are constructed for a given spherically-symmetric
    potential, but then the resulting DF can be viewed as a function of actions only,
    and used in an arbitrary potential just as any other f(J); the intermediate mappings
    become part of its definition. Of course, it may no longer yield isotropic velocity
    distributions -- hence the name "pseudo".
*/
class PseudoIsotropic: public BaseDistributionFunction{
    const math::LogLogSpline df;    ///< one-dimensional function of spherical phase volume h
    const potential::PhaseVolume pv;          ///< correspondence between E and h
    const actions::ActionFinderSpherical af;  ///< correspondence between (Jr,L) and E
public:
    PseudoIsotropic(const math::LogLogSpline& _df, const potential::BasePotential& potential) :
        df(_df), pv(potential::PotentialWrapper(potential)), af(potential) {}

    virtual double value(const actions::Actions &J) const
    {
        return df(pv(af.E(J)));
    }
};

}