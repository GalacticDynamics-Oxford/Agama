/** \file    galaxymodel_velocitysampler.h
    \brief   Velocity assignment using Jeans or Eddington models
    \date    2010-2017
    \author  Eugene Vasiliev

    This module presents several routines for assigning velocities to particles.
    The main application is to construct suitable initial conditions for the orbit library
    in the context of Schwarzschild models. To this end, first the positions are drawn from
    the target density model, e.g., using the `sampleDensity()` routine, and then
    a reasonable velocity is assigned to each point, based on either of the three methods:

    - Eddington formula for the isotropic distribution function in a spherical model;
    - spherical Jeans equation for the radial and tangential velocity dispersion;
    - axisymmetric Jeans equation for three components of <v_j^2>, j=R,z,phi.

    Three separate routines implement these tasks for each method; they all follow a similar
    convention: take the array of particle coordinates, the potential, and an appropriate
    velocity generating interface, and return an array of positions+velocities.
    Another routine `assignVelocity()` presents a higher-level interface that automatically
    chooses between the three methods based on the provided arguments, and constructs
    the respective velocity generators internally.
*/
#pragma once
#include "galaxymodel_jeans.h"
#include "galaxymodel_spherical.h"

namespace galaxymodel{

/** Assign particle velocities using a spherical isotropic distribution function constructed
    by the Eddington inversion formula.
    \param[in]  pointCoords  is the array of particle coordinates and masses
    (created, for instance, by the `sampleDensity()` routine);
    \param[in]  pot  is the total potential (should match the potential used in constructing
    the spherical DF model);
    \param[in]  sphModel  is the spherical model constructed for the same density and potential
    (or their sphericalized versions);
    \return  an array of particles with the same coordinates and masses, but having velocities
    sampled from the DF of the spherical model at the corresponding particle's radius.
    \note OpenMP-parallelized loop over particles.
*/
particles::ParticleArrayCar assignVelocityEdd(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const SphericalIsotropicModelLocal& sphModel);

/** Assign particle velocities using a spherical Jeans model.
    \param[in]  pointCoords  is the array of particle coordinates and masses
    (created, for instance, by the `sampleDensity()` routine);
    \param[in]  pot  is the total gravitational potential (the Jeans model should have been
    constructed using either the same potential or its sphericalized version);
    \param[in]  jeansSphModel  is the spline interpolator for sigma_r(r)
    returned by the `createJeansSphModel()` routine;
    \param[in]  beta  is the velocity anisotropy coefficient used in construction of the Jeans model;
    \return  an array of particles with the same coordinates and masses, but having velocities
    sampled from a Gaussian distribution with the dispersion provided by the Jeans model
    at the corresponding particle's radius.
    \note OpenMP-parallelized loop over particles.
*/
particles::ParticleArrayCar assignVelocityJeansSph(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const math::IFunction& jeansSphModel,
    const double beta);

/** Assign particle velocities using an axisymmetric Jeans model.
    \param[in]  pointCoords  is the array of particle coordinates and masses;
    \param[in]  pot  is the gravitational potential (does not need to be axisymmetric,
    but should be the same as provided to the constructor of the Jeans model);
    \param[in]  jeansAxiModel  is the instance of the Jeans model that provides
    the first and second moments of velocity as functions of position;
    \return  an array of particles with the same coordinates and masses, but having velocities
    sampled from a triaxial Gaussian distribution provided by the Jeans model
    at the corresponding particle's radius.
    \note OpenMP-parallelized loop over particles.
*/
particles::ParticleArrayCar assignVelocityJeansAxi(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BasePotential& pot,
    const JeansAxi& jeansAxiModel);

/** A driver routine that assigns the particle velocities using one of the three methods,
    depending on input arguments.
    If neither beta nor kappa are provided, this implies the Eddington method (isotropic velocity
    drawn from a distribution function that is constructed for sphericalized versions of the input
    density and potential models).
    If beta is provided but kappa is not, this implies the spherical Jeans method
    with constant velocity anisotropy given by beta.
    If both beta and kappa are given, this means the axisymmetric Jeans method,
    with beta specifying the meridional velocity anisotropy and kappa - the amount of rotation.
    \param[in]  pointCoords  is the array of particle positions and masses;
    usually they would be drawn from the same density model as provided in the `dens` argument,
    for instance, using the `sampleDensity()` routine, but this is not necessary.
    \param[in]  dens  is the input density model (if it is not spherical or axisymmetric,
    an appropriately symmetrized version will be constructed internally).
    \param[in]  pot  is the total gravitational potential (may have any geometry);
    assigned velocities will not exceed the local escape velocity at any point.
    \param[in]  beta  is the velocity anisotropy coefficient for Jeans models.
    Its meaning is different for spherical or axisymmetric models: in the first case,
    beta = 1 - sigma_t^2 / (2 sigma_r^2), and in the second case, beta = 1 - sigma_z^2 / sigma_R^2.
    \param[in]  kappa  is the parameter specifying the amount of rotation in axisymmetric Jeans
    models, i.e. the way the total second moment of azimuthal velocity <v_phi^2> is distributed
    between the mean streaming velocity <v_phi> and the velocity dispersion <sigma_phi>:
    kappa = <v_phi> / sqrt( <v_phi^2> - <v_R^2> ),
    kappa=0 means no net rotation, kappa=+-1 corresponds to sigma_phi=sigma_R;
    \return  the array of particle positions, velocities and masses.
    \throw  std::runtime_error or any other exception in case of problems.
    \note OpenMP-parallelized loop over particles in all three methods.
*/
particles::ParticleArrayCar assignVelocity(
    const particles::ParticleArray<coord::PosCyl>& pointCoords,
    const potential::BaseDensity& dens,
    const potential::BasePotential& pot,
    const double beta=NAN,
    const double kappa=NAN);

}
