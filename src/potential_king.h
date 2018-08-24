/** \file    potential_king.h
    \brief   Generalized King models
    \author  Eugene Vasiliev
    \date    2018
**/
#pragma once
#include "smart.h"

namespace potential {

/** Create a generalized King model represented by its potential.
    This family of lowered isothermal models is defined by the distribution function,
    \f$  f(E) = A E_\gamma[g, -(E - \Phi(r_t)) / \sigma^2 ]  \f$,  with an auxiliary function
    \f$  E_\gamma(g, x) = \exp(x) \gamma(g, x) / \Gamma(g)  \f$,  where
    \f$  \gamma(g, x)  \f$  is the lower incomplete Gamma function,
    \f$  \Gamma(g) = \gamma(g, \infty)  \f$  is the complete Gamma function,
    and A is a normalization constant.
    This definition follows Gieles&Zocchi(2015), which introduced a Python code (LIMEPY)
    implementing a more general family of models (possibly multi-component and anisotropic).
    In our case, we consider only isotropic single-component models.
    This routine constructs the potential of these models by solving a second-order ODE
    (equations 5-8 in GZ15) and returns an instance of `Multipole` potential
    (with only one term, l=0).
    The density obtained by differentiating the potential is somewhat inaccurate close to
    origin or to the truncation radius, but is accurate over most of the radial range.

    \param[in]  mass  is the total mass of the model.
    \param[in]  scaleRadius  is the King radius (roughly the radius where the density drops
    by a factor of two compared to its central value; the radius of the outer boundary, r_t,
    is larger by a factor depending on the concentration).
    \param[in]  W0  is the dimensionless depth of the potential well (King parameter):
    \f$  W_0 = [ \Phi(r_t) - \Phi(0) ] / \sigma^2  \f$;
    models with higher W0 have larger outer radius r_t in comparison with scale radius.
    \param[in]  trunc  is the parameter controlling the strength of truncation
    (it is called g in LIMEPY);
    larger values result in a more gentle density fall-off near the outer radius.
    The classical King model corresponds to g=1; possible values range from 0 to 3.5
    \return  an instance of Multipole potential corresponding to the model.
    \throw  std::invalid_argument if the parameters are incorrect;
    or std::runtime_error if the model failed to converge (unlikely).
*/
PtrPotential createKingPotential(double mass, double scaleRadius, double W0, double trunc=1);

/** Create a generalized King model represented by its density.
    This routine is analogous to `createKingPotential`, but returns an instance of
    density represented by the `DensitySphericalHarmonic` potential (with only one term, l=0).
    It produces a more accurate interpolation of the density profile than the one resulting
    from `createKingPotential`.
    \param[in]  mass  is the total mass of the model.
    \param[in]  scaleRadius  is the King radius.
    \param[in]  W0  is the dimensionless depth of the potential well (King parameter).
    \param[in]  trunc  is the parameter controlling the strength of truncation.
    \return  an instance of DensitySphericalHarmonic interpolator.
    \throw  std::invalid_argument if the parameters are incorrect;
    or std::runtime_error if the model failed to converge (unlikely).
*/
PtrDensity createKingDensity(double mass, double scaleRadius, double W0, double trunc=1);

}  // namespace
