/** \file    potential_factory.h
    \brief   Creation and input/output of Potential instances
    \author  Eugene Vasiliev
    \date    2010-2024

    This file provides several utility function to manage instances of BaseDensity and BasePotential:
    creating a density or potential model from parameters provided in `utils::KeyValueMap` objects,
    creating a potential from a set of point masses or from an N-body snapshot file,
    loading and storing density/potential coefficients in text files.
*/

#pragma once
#include "particles_base.h"
#include "potential_base.h"
#include "math_spline.h"
#include "smart.h"
#include "units.h"
#include "utils_config.h"

namespace potential {

/** Create an instance of density according to the parameters contained in the key-value map.
    \param[in] params is the list of parameters (should contain either "file=", or
    "density=...", or "type=...");
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters (such as mass and radii) into internal units; can be a trivial converter.
    \return    a new instance of PtrDensity on success.
    \throw     std::invalid_argument or std::runtime_error or other density-specific
    exception on failure
*/
PtrDensity createDensity(
    const utils::KeyValueMap& params,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of density expansion (DensitySphericalHarmonic or DensityAzimuthalHarmonic)
    or a modified (shifted, rotated, etc.) density from the user-provided density model,
    with parameters contained in the key-value map.
    \param[in] params is the list of parameters:
    if the "type=..." parameter specifies one of the density expansion classes, then the input
    density will be used to construct the corresponding expansion type, with other parameters
    such as grid sizes and orders of expansion provided as well;
    otherwise "type" should be empty and the input density will be used directly.
    If the parameter list contains modifier params such as "center=...", etc.,
    the input density or the previously constructed expansion will be wrapped into
    one or more modifiers according to the provided parameters.
    \param[in] dens   is the input density model;
    \param[in] converter (optional) is the unit converter for transforming dimensional quantities
    in parameters (e.g. the grid node placement) into internal units;
    \return    a new instance of PtrDensity on success:
    \throw     std::invalid_argument if the requested density is not of an expansion type,
    or any density-specific exception on failure (if some parameters are missing or invalid).
*/
PtrDensity createDensity(
    const utils::KeyValueMap& params,
    const PtrDensity& dens,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential according to the parameters contained in the key-value map.
    This is a shortcut to the more general routine that accepts a list of KeyValueMap's,
    and depending on the potential type, may produce an elementary or a composite potential,
    or wrap it into one or more modifiers.
    \param[in] params is the list of parameters (should contain either "file=..." or "type=...");
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters (such as mass and radii) into internal units; can be a trivial converter;
    \return    a new instance of PtrPotential on success.
    \throw     std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing, or refer to non-existent files)
*/
PtrPotential createPotential(
    const utils::KeyValueMap& params,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of composite potential according to the parameters contained in the
    array of key-value maps for each component. Note that there is no 1:1 correspondence 
    between the parameters and the components of the resulting potential, since the parameters 
    may contain several density models that would be used for initializing a single potential 
    expansion object (like in the case of GalPot), and components sharing the same set of modifiers
    will be grouped into separate composite potentials wrapped into corresponding modifiers.
    \param[in] params is the array of parameter lists, one per component.
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters (such as mass and radii) into internal units; can be a trivial converter.
    \return    a new instance of PtrPotential on success.
    \throw     std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing, or refer to non-existent files)
*/
PtrPotential createPotential(
    const std::vector<utils::KeyValueMap>& params,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential expansion for the user-provided density model,
    with parameters contained in the key-value map.
    \param[in] params is the list of parameters
    ("type=..." should specify one of the potential expansions, and other parameters such as
    grid sizes, expansion orders, etc. may be provided as well);
    if the parameter list specifies any modifiers, the result will be wrapped into one or more
    corresponding modifier classes.
    \param[in] dens   is the density model which will serve as the source to the potential;
    \param[in] converter (optional) is the unit converter for transforming dimensional quantities
    in parameters (e.g. the grid node placement) into internal units;
    \return    a new instance of PtrPotential on success.
    \throw     std::invalid_argument if the requested potential is not of an expansion type,
    or any potential-specific exception on failure (if some parameters are missing or invalid).
*/
PtrPotential createPotential(
    const utils::KeyValueMap& params,
    const PtrDensity& dens,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential expansion (e.g. Multipole) or a modified (shifted, rotated, etc.)
    potential for the user-provided potential model, with parameters contained in the key-value map.
    \param[in] params is the list of parameters ("type=..." specifies the expansion type,
    type should be empty when creating a modifier on top of the provided potential);
    \param[in] pot    is the potential model which will be approximated with the potential expansion
    or wrapped into a modifier;
    \param[in] converter (optional) is the unit converter for transforming dimensional quantities
    in parameters (e.g. the grid nodes) into internal units;
    \return    a new instance of PtrPotential on success.
    \throw     std::invalid_argument if params.type does not specify an expansion or a modifier,
    or any potential-specific exception on failure (if some parameters are missing or invalid).
*/
PtrPotential createPotential(
    const utils::KeyValueMap& params,
    const PtrPotential& pot,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential expansion from the provided array of particles.
    \param[in] params  is the list of required parameters (e.g., the type of potential expansion,
    number of terms, prescribed symmetry, etc.).
    \param[in] particles  is the array of particle positions and masses.
    \param[in] converter  is the unit converter for transforming the dimensional parameters 
    (min/max radii of grid) into internal units; can be a trivial converter. 
    Coordinates and masses of particles are _not_ transformed: if they are loaded from an external 
    N-body snapshot file, the conversion is applied at that stage, and if they come from 
    other routines in the library, they are already in internal units.
    \return    a new instance of PtrPotential on success.
    \throw     std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing).
*/
PtrPotential createPotential(
    const utils::KeyValueMap& params, 
    const particles::ParticleArray<coord::PosCyl>& particles,
    const units::ExternalUnits& converter = units::ExternalUnits());


/** Construct an interpolated spherical density profile from two arrays -- radii and
    enclosed mass M(<r).
    First a suitably scaled interpolator is constructed for M(r);
    if it is found to have a finite limiting value at r --> infinity, the asymptotic power-law
    behaviour of density at large radii will be correctly represented.
    Then the density at each point of the radial grid is computed from the derivative of
    this interpolator. The returned array may be used to construct a LogLogSpline interpolator
    or a DensitySphericalHarmonic object (obviously, with only one harmonic).
    \param[in]  gridr  is the grid in radius (must have positive values sorted in order of increase);
    typically the radial grid should be exponentially spaced with r[i+1]/r[i] ~ 1.2 - 2.
    \param[in]  gridm  is the array of enclosed mass at each radius (must be positive and monotonic);
    \return  an array of density values at the given radii.
    \throw   std::invalid_argument if the input arrays were incorrect
    (incompatible sizes, non-monotinic or negative values), or
    std::runtime_error if the interpolator failed to produce a positive-definite density.
*/
std::vector<double> densityFromCumulativeMass(
    const std::vector<double>& gridr,
    const std::vector<double>& gridm);


/** Read a file with the cumulative mass profile and construct a density model from it.
    The text file should be a whitespace- or comma-separated table with at least two columns
    (the rest is ignored) -- radius and the enclosed mass within this radius,
    both must be in increasing order. Lines not starting with a number are ignored.
    The enclosed mass profile should not include the central black hole (if present),
    because it could not be represented in terms of a density profile anyway.
    \param[in]  fileName  is the input file name.
    \return  an interpolated density profile, represented by a LogLogSpline class.
    \throw  std::runtime_error if the file does not exist, or the mass profile is not monotonic.
*/
math::LogLogSpline readMassProfile(const std::string& fileName);


/** Utility function providing a legacy interface compatible with the original GalPot (deprecated).
    It reads the parameters from a text file and converts them into the internal unit system,
    using the conversion factors in the provided `units::ExternalUnits` object,
    then constructs the potential using `createGalaxyPotential()` routine.
    Standard GalPot units are Kpc and Msun, so to simplify matters, one may instead use 
    the overloaded function `readGalaxyPotential()` that takes the instance of internal units
    as the second argument, and creates a converter from standard GalPot to these internal units.
    \param[in]  filename  is the name of parameter file;
    \param[in]  converter provides the conversion from GalPot to internal units;
    \return     a new instance of PtrPotential;
    \throw      a std::runtime_error exception if file is not readable
    or does not contain valid parameters.
*/
PtrPotential readGalaxyPotential(
    const std::string& filename,
    const units::ExternalUnits& converter);

/** Utility function providing a legacy interface compatible with the original GalPot (deprecated).
    It reads the parameters from a text file and converts them into the internal unit system, 
    then constructs the potential using `createGalaxyPotential()` routine.
    This function creates the instance of unit converter from standard GalPot units (Kpc and Msun)
    into the provided internal units, and calls the overloaded function `readGalaxyPotential()` 
    with this converter object as the second argument. 
    \param[in]  filename is the name of parameter file;
    \param[in]  unit     is the specification of internal unit system;
    \return     a new instance of PtrPotential;
    \throw      a std::runtime_error exception if file is not readable
    or does not contain valid parameters.
*/
inline PtrPotential readGalaxyPotential(
    const std::string& filename,
    const units::InternalUnits& unit) 
{   // create a temporary converter; velocity unit is not used
    return readGalaxyPotential(filename,
        units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun));
}


/** Create an elementary or a composite density from parameters provided in the INI file.
    The file must have one or more sections with names starting from Density,
    and each section should contain the parameters for an analytic density profile or
    coefficients of DensitySphericalHarmonic or DensityAzimuthalHarmonic models 
    previously stored by writeDensity().
    \param[in] iniFileName specifies the file to read;
    \param[in] converter is the unit converter for transforming the density coefficients;
    from dimensional into internal units; can be a trivial converter;
    \return    a new instance of PtrDensity on success;
    \throw     std::invalid_argument or std::runtime_error or other density-specific exception
    on failure (e.g., if the file does not exist, or does not contain valid coefficients).
*/
PtrDensity readDensity(
    const std::string& iniFileName,
    const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an elementary or composite potential according to the parameters contained in an INI file.
    \param[in] iniFileName is the name of an INI file that contains one or more sections
    with potential parameters, named as [Potential], [Potential1], ...
    These sections may contain references to other files with potential parameters (file=...),
    or parameters of analytic potential models, or coefficients of BasisSet, Multipole or CylSpline
    potential expansions previously stored by writePotential().
    \param[in] converter is the unit converter for transforming the dimensional quantities
    in parameters (such as mass and radii) into internal units; can be a trivial converter.
    \return    a new instance of PtrPotential on success (if there are several components
    in the INI file, the returned potential is composite).
    \throw     std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing, or refer to non-existent files)
*/
PtrPotential readPotential(
    const std::string& iniFileName,
    const units::ExternalUnits& converter = units::ExternalUnits());


/** Write density or potential expansion coefficients to a text (INI) file.
    The potential must be one of the following expansion classes: 
    `BasisSet`, `Multipole`, `CylSpline`,
    or the density may be `DensitySphericalHarmonic` or `DensityAzimuthalHarmonic`.
    The coefficients stored in a file may be later loaded by `readPotential()` or
    `readDensity()` routines.
    For a composite potential or density, all components are stored in the same file
    one after another, under separate sections [Potential], [Potential1], etc.
    For potential/density types not in the above list, only the name is stored (no parameters).
    NOTE: this routine currently does NOT guarantee that the potential/density is stored correctly
    (in a way that can be loaded back to produce the original object) in all cases;
    in particular, it ignores the potential modifiers and is unsuitable for potential/density types
    that are not expansions, but gives no warning or error in these cases!
    \param[in] fileName is the output file;
    \param[in] density is the reference to density or potential object;
    \param[in] converter is the unit converter for transforming the density or potential
    coefficients from internal into dimensional units; can be a trivial converter;
    \return    success or failure (the latter indicates I/O error; however, an unsuitable potential
    or density type does not produce a failure, but is silently ignored).
*/
bool writeDensity(
    const std::string& fileName,
    const BaseDensity& density,
    const units::ExternalUnits& converter = units::ExternalUnits());

/// alias to writeDensity
inline bool writePotential(
    const std::string& fileName,
    const BasePotential& potential,
    const units::ExternalUnits& converter = units::ExternalUnits()) {
    return writeDensity(fileName, potential, converter); }


/** return the symmetry type encoded in the string.
    Spherical, Axisymmetric, Triaxial and None are recognized by the first letter,
    whereas other types must be given by their numerical code.
    If the string is empty, the default value ST_TRIAXIAL is returned.
*/
coord::SymmetryType getSymmetryTypeByName(const std::string& name);

/** return the name of symmetry encoded in SymmetryType.
    Spherical, Axisymmetric, Triaxial and None are returned as symbolic names,
    other types as their numerical code.
*/
std::string getSymmetryNameByType(coord::SymmetryType type);

}  // namespace potential
