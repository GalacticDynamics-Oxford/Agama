/** \file    galaxymodel_jeans.h
    \brief   Spherical and axisymmetric Jeans models
    \date    2014-2017
    \author  Eugene Vasiliev

    This module presents methods for solving the Jeans equations to determine
    second moments of velocity as a function of positions. Two methods are available:
    spherical models with a constant velocity anisotropy coefficient,
    and axisymmetric models in which the velocity ellipsoid is aligned with cylindrical
    coordinates, and the ratio of radial to vertical velocity dispersions is also constant
    (so-called Jeans anisotropic models).
    The purpose of these models is mainly to serve as reasonable initial conditions
    for orbit integration in the Schwarzschild approach. This application is further
    detailed in a separate module `galaxymodel_velocitysampler.cpp`.
*/
#pragma once
#include "potential_base.h"
#include "math_spline.h"

namespace galaxymodel{

/** Solve the Jeans equation for a spherical model with the given density and potential,
    assuming a constant velocity anisotropy, and construct an interpolator for the radial velocity
    dispersion as a function of radius.
    \param[in]  density  is the 1d function returning the density as a function of radius;
    \param[in]  potential  is the 1d function representing the radial dependence of the potential;
    \param[in]  beta  is the velocity anisotropy coefficient (beta=0 means isotropic model,
    0<beta<=1 - radially anisotropic, beta<0 - tangentially anisotropic),
    assumed to be independent of radius. The routine does not check physical validity of this parameter!
    (i.e. if the both the density and the potential are cored, beta may not be positive).
    \return  an interpolator representing the 1d radial velocity dispersion (sigma_r, not squared)
    as a function of radius. The 1d velocity dispersion in both tangential directions is
    sigma_theta = sigma_phi = sigma_r * sqrt(1-beta).
    \throw  std::runtime_error in case of numerical problems in constructing the model.
*/
math::LogLogSpline createJeansSphModel(
    const math::IFunction &density, const math::IFunction &potential, double beta);


/** Axisymmetric Jeans anisotropic model constructed for the given density and potential
    and providing the velocity dispersion tensor at any point in the (R,z) plane.
    The model is equivalent to the JAM method (Cappellari 2008) and relies on two assumptions:
    (a) the velocity ellipsoid is aligned with cylindrical coordinates;
    (b) the meridional anisotropy beta_m = 1 - sigma_z^2 / sigma_R^2 is constant.
    The computed velocity dispersions may not be accurate if the density is noisy;
    they are always constrained to be non-negative and capped by the escape velocity from above.
*/
class JeansAxi {
public:
    /** Construct the model for the given density, potential, and meridional anisotropy coefficient.
        \param[in]  density    is the density model (must have a finite mass).
        \param[in]  potential  is the potential (needs not be related to density via Poisson equation).
        Both density and potential need not be axisymmetric; if they are not, all internal computations
        are performed for azimuthally-averaged quantities.
        \param[in]  beta_m  is the meridional anisotropy coefficient (beta_m==0 means isotropic model),
        it is related to the parameter `b` defined in Cappellari(2008) by  b = 1 / (1-beta_m),
        and should be less than one.
        \param[in]  kappa  is the rotation parameter ranging from 0 (no streaming motion) to +-1
        (models with maximum streaming motion, so that the azimuthal velocity dispersion is
        sigma_phi = 1/2 kappa / Omega * sigma_R, as in the epicyclic approximation).
        \throw  an exception in case of invalid parameters or problems in constructing the model.
        \note OpenMP-parallelized loop over the radial grid.
    */
    JeansAxi(const potential::BaseDensity &density, const potential::BasePotential &potential,
        double beta_m, double kappa);

    /** Compute the moments of velocity at the given point */
    void moments(const coord::PosCyl& point, coord::VelCyl &vel, coord::Vel2Cyl &vel2) const;

private:
    const double bcoef;          ///< meridional velocity anisotropy coefficient b = 1 / (1-beta_m)
    const double kappa;          ///< rotation parameter
    math::LinearInterpolator2d intvphi2, intvz2;  ///< pre-computed interpolators for v_phi^2, v_z^2
    math::CubicSpline epicycleRatio;
};

}  // namespace