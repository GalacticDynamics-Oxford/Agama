/** \file    interface_c.h
    \brief   Plain C interface for a subset of Agama functions
    \author  Eugene Vasiliev
    \date    2021-2024
*/
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/// exception handling: if any of the routines below returned an error
/// (e.g. a NULL instead of a newly created potential object),
/// this routine will return a text message corresponding to the exception.
const char* agama_getError();

/// \name  Density and Potential
///@{

/// opaque pointer to an Agama density object
typedef struct agama_Density agama_Density;

/// opaque pointer to an Agama potential object
/// (which is derived from agama_Density, but C pointers do not know about inheritance..)
typedef struct agama_Potential agama_Potential;

/** construct a density object from the provided parameters.
    \param[in]  params: string containing the density type and other parameters,
                e.g., "type=Plummer, mass=2, scaleRadius=4".
                In case of more complicated (e.g., multi-component) models,
                one can read the parameters from an INI file, using the following syntax:
                "file=density_params.ini"
    \return     an opaque pointer to a density object, or NULL when parameters are invalid;
                in the latter case, agama_getError() will return a description of the error.
*/
agama_Density* agama_createDensity(const char* params);

/** construct a density object from the provided string of parameters,
    \param[in]  params: string containing the potential type and other parameters,
                e.g., "type=Multipole, density=Spheroid, gamma=0, beta=5, alpha=2, axisRatioZ=0.5".
                In case of more complicated (e.g., multi-component) models,
                one can read the parameters from an INI file, using the following syntax:
                "file=potential_params.ini"
    \return     an opaque pointer to a potential object, or NULL when parameters are invalid;
                in the latter case, agama_getError() will return a description of the error.
*/
agama_Potential* agama_createPotential(const char* params);

/** delete a previously constructed density object */
void agama_deleteDensity(agama_Density* density);

/** delete a previously constructed potential object */
void agama_deletePotential(agama_Potential* potential);

/** evaluate the density in cartesian coordinates.
    \param[in]  density:  an opaque pointer to an Agama density or potential object
                returned by agama_createDensity(), agama_createPotential() or similar routines.
    \param[in]  pos:  an array of 3 cartesian coordinates (x,y,z).
    \param[in]  time:  time at which the potential is evaluated.
    \return     the value of the density at the given point.
*/
double agama_evalDensity(const agama_Density* density, const double pos[3], double time);

/** evaluate the potential and, if necessary, its derivatives in cartesian coordinates.
    \param[in]  potential:  an opaque pointer to an Agama potential object
                returned by agama_createPotential() or similar routines.
    \param[in]  pos:  an array of 3 cartesian coordinates (x,y,z).
    \param[in]  time:  time at which the potential is evaluated.
    \param[out] deriv:  if not NULL, it should point to an array of 3 doubles to store
                potential derivatives: dPhi/dx, dPhi/dy, dPhi/dz.
    \param[out] deriv2: if not NULL, it should point to an array of 6 doubles to store
                second derivatives of the potential in the following order:
                d2Phi/dx2, d2Phi/dy2, d2Phi/dz2, d2Phi/dxdy, d2Phi/dydz, d2Phi/dzdx.
    \return     the value of the potential.
*/
double agama_evalPotential(const agama_Potential* potential, const double pos[3], double time,
    /*output*/ double deriv[3], /*output*/ double deriv2[6]);

/** Compute the cylindrical radius of a circular orbit in the equatorial plane
    for a given value of energy; the potential is axisymmetrized if necessary.
*/
double agama_R_circ(const agama_Potential* potential, double E);

/** Compute cylindrical radius of an orbit in the equatorial plane for a given value of
    z-component of angular momentum; the potential is axisymmetrized if necessary.
*/
double agama_R_from_Lz(const agama_Potential* potential, double Lz);

/** Compute the radius of a radial orbit in the equatorial plane with the given energy,
    i.e. the root of Phi(R)=E; the potential is axisymmetrized if necessary.
*/
double agama_R_max(const agama_Potential* potential, double E);

/** find peri- and apocenter radii of an orbit in the equatorial plane of the given potential,
    which has the given energy and angular momentum.
    \param[in]  potential:  an opaque pointer to an Agama potential object.
    \param[in]  E:  energy; should be between Phi(0) and Phi(infinity).
    \param[in]  L:  angular momentum (more specifically, its z-component, since the orbit
                is constructed in the xy plane); should be between 0 and Lcirc(E).
    \param[out] Rperi:  pericenter radius, lower root of Phi(R,z=0) + 0.5*L^2/R^2 = E.
    \param[out] Rapo:  apocenter radius, upper root of the same equation.
                When E is outside the allowed range, both Rperi and Rapo contain NaN.
                If L > Lcirc(E), both roots contain Rcirc(E).
*/
void agama_findPlanarOrbitExtent(const agama_Potential* potential, double E, double L,
    /*output*/ double* Rperi, /*output*/ double* Rapo);

///@}
/// \name  Action finder
///@{

/// opaque pointer to an Agama action finder object
typedef struct agama_ActionFinder agama_ActionFinder;

/** construct a suitable action finder for the given potential.
    \param[in]  potential:  an opaque pointer to an Agama potential object.
    \return  an opaque pointer to an action finder object, or NULL when the potential
             is unsuitable (e.g., not axisymmetric);
             in the latter case, agama_getError() will return a description of the error.
*/
agama_ActionFinder* agama_createActionFinder(const agama_Potential* potential);

/** delete a previously constructed action finder */
void agama_deleteActionFinder(agama_ActionFinder* actionFinder);

/** compute any combination of actions, angles and/or frequencies corresponding to
    the given position/velocity point, using a previously constructed ActionFinder object.
    \param[in]  actionFinder:  an opaque pointer to an Agama action finder object
                returned by agama_createActionFinder().
    \param[in]  posvel:  an array of 3 positions and 3 velocities in cartesian coordinates.
    \param[out] actions:  if not NULL, should point to an array of length 3 which will 
                contain actions (Jr, Jz, Jphi).
    \param[out] angles:  if not NULL, should point to an array of length 3 which will
                contain angles (thetar, thetaz, thetaphi).
    \param[out] frequencies:  if not NULL, should point to an array of length 3 which will
                contain frequencies (Omegar, Omegaz, Omegaphi).
*/
void agama_evalActionsAnglesFrequencies(
    const agama_ActionFinder* actionFinder, const double posvel[6],
    /*output*/ double actions[3], /*output*/ double angles[3], /*output*/ double frequencies[3]);

/** compute any combination of actions, angles and/or frequencies corresponding to
    the given position/velocity point, using a potential object directly.
    \param[in]  actionFinder:  an opaque pointer to an Agama potential object
                returned by agama_createPotential().
    \param[in]  fd:  focal distance (needed only for non-spherical potentials, otherwise ignored).
    \param[in]  posvel:  an array of 3 positions and 3 velocities in cartesian coordinates.
    \param[out] actions:  if not NULL, should point to an array of length 3 which will
                contain actions (Jr, Jz, Jphi).
    \param[out] angles:  if not NULL, should point to an array of length 3 which will
                contain angles (thetar, thetaz, thetaphi).
    \param[out] frequencies:  if not NULL, should point to an array of length 3 which will
                contain frequencies (Omegar, Omegaz, Omegaphi).
*/
void agama_evalActionsAnglesFrequenciesStandalone(
    const agama_Potential* potential, double fd, const double posvel[6],
    /*output*/ double actions[3], /*output*/ double angles[3], /*output*/ double frequencies[3]);

#ifdef __cplusplus
}
#endif
