/** \file    potential_factory.h
    \brief   Creation and input/output of Potential instances
    \author  EV
    \date    2010-2015

    This file provides several utility function to manage instances of BaseDensity and BasePotential: 
    creating a density or potential model from parameters provided in ConfigPotential, 
    creating a potential from a set of point masses or from an N-body snapshot file,
    loading potential coefficients from a text file, 
    writing expansion coefficients to a text file,
    converting between potential parameters in `potential::ConfigPotential` structure and 
    a text array of key=value pairs (`utils::KeyValueMap`).
*/

#pragma once
#include "potential_base.h"
#include "particles_base.h"
#include "units.h"
#include <string>

// forward declaration
namespace utils { class KeyValueMap; }

namespace potential {

/** Create an instance of potential according to the parameters contained in an INI file.
    \param[in] iniFileName is the name of an INI file that contains one or more sections 
    with potential parameters, named as [Potential], [Potential1], ...
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters (such as mass and radii) into internal units; can be a trivial converter.
    \return    a new instance of BasePotential* on success (if there are several components
    in the INI file, the returned potential is composite).
    \throws    std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing, or refer to a non-existent file).
*/
const BasePotential* createPotential(
    const std::string& iniFileName, const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential according to the parameters contained in the key-value map.
    \param[in] params is the list of parameters;
    \param[in] converter is the unit converter for transforming the dimensional quantities 
    in parameters (such as mass and radii) into internal units; can be a trivial converter.
    \return    a new instance of BasePotential* on success.
    \throws    std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing, or refer to a non-existent file).
*/
const BasePotential* createPotential(
    const utils::KeyValueMap& params, const units::ExternalUnits& converter = units::ExternalUnits());

/** Create an instance of potential expansion from the provided particle snapshot.
    \param[in] params is the list of required parameters (e.g., the type of potential expansion,
    number of terms, prescribed symmetry, etc.).
    \param[in] converter is the unit converter for transforming the dimensional parameters 
    (min/max radii of grid) into internal units; can be a trivial converter. 
    Coordinates and masses of particles are not transformed: if they are loaded from an external 
    N-body snapshot file, the conversion is applied at that stage, and if they come from 
    other routines in the library, they are already in internal units.
    \param[in] points is the array of particle positions and masses.
    \tparam    ParticleT may be PosT<CoordSys> or PosVel<CoordSys>, with CoordSys = Car, Cyl or Sph.
    \return    a new instance of BasePotential* on success.
    \throws    std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if some of the parameters are invalid or missing).
*/
template<typename ParticleT>
const BasePotential* createPotentialFromPoints(
    const utils::KeyValueMap& params, const units::ExternalUnits& converter, 
    const particles::PointMassArray<ParticleT>& points);

/** Utility function providing a legacy interface compatible with the original GalPot.
    It reads the parameters from a text file and converts them into the internal unit system,
    using the conversion factors in the provided `units::ExternalUnits` object,
    then constructs the potential using `createGalaxyPotential()` routine.
    Standard GalPot units are Kpc and Msun, so to simplify matters, one may instead use 
    the overloaded function `readGalaxyPotential()` that takes the instance of internal units
    as the second argument, and creates a converter from standard GalPot to these internal units.
    \param[in]  filename  is the name of parameter file;
    \param[in]  converter provides the conversion from GalPot to internal units;
    \returns    the new CompositeCyl potential;
    \throws     a std::runtime_error exception if file is not readable or does not contain valid parameters.
*/
const potential::BasePotential* readGalaxyPotential(
    const std::string& filename, const units::ExternalUnits& converter);

/** Utility function providing a legacy interface compatible with the original GalPot.
    It reads the parameters from a text file and converts them into the internal unit system, 
    then constructs the potential using `createGalaxyPotential()` routine.
    This function creates the instance of unit converter from standard GalPot units (Kpc and Msun)
    into the provided internal units, and calls the overloaded function `readGalaxyPotential()` 
    with this converter object as the second argument. 
    \param[in]  filename is the name of parameter file;
    \param[in]  unit     is the specification of internal unit system;
    \returns    the new CompositeCyl potential
    \throws     a std::runtime_error exception if file is not readable or does not contain valid parameters.
*/
inline const potential::BasePotential* readGalaxyPotential(
    const std::string& filename, const units::InternalUnits& unit) 
{   // create a temporary converter; velocity unit is not used
    return readGalaxyPotential(filename, units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun));
}

/** Create a potential expansion from coefficients stored in a text file.
    The file must contain coefficients for BasisSetExp, SplineExp, or CylSplineExp;
    the potential type is determined automatically from the first line of the file.
    \param[in] coefFileName specifies the file to read.
    \return    a new instance of BasePotential* on success.
    \throws    std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if the file does not exist, or does not contain valid coefficients)
*/
const BasePotential* readPotentialCoefs(const std::string& coefFileName);

/** Write potential expansion coefficients to a text file.

    The potential must be one of the following expansion classes: 
    `BasisSetExp`, `SplineExp`, `CylSplineExp`.
    The coefficients stored in a file may be later loaded by `readPotential()` function.
    \param[in] coefFileName is the output file
    \param[in] potential is the pointer to potential
    \throws    std::invalid_argument if the potential is of inappropriate type,
    or std::runtime error if the file is not writeable.
*/
void writePotentialCoefs(const std::string& coefFileName, const BasePotential& potential);

/// return file extension for writing the coefficients of a given potential type,
/// or empty string if it is not one of the expansion types
const char* getCoefFileExtension(const std::string& potName);

/// return file extension for writing the coefficients of a given potential object,
/// or empty string if the potential type is not one of the expansion types
inline const char* getCoefFileExtension(const BasePotential& p) {
    return getCoefFileExtension(p.name()); }

}; // namespace
