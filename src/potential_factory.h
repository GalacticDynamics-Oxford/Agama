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

/// \name Definitions of all known potential types and parameters
///@{

/** List of all known potential and density types 
    (borrowed from SMILE, not everything is implemented here).
    Note that this type is not a substitute for the real class hierarchy:
    it is intended only to be used in factory methods such as 
    creating an instance of potential from its name 
    (e.g., passed as a string, or loaded from an ini file).
*/
enum PotentialType {

    //  Generic values that don't correspond to a concrete class
    PT_UNKNOWN,      ///< undefined
    PT_COEFS,        ///< pre-computed coefficients of potential expansion loaded from a coefs file
    PT_NBODY,        ///< N-body snapshot that is used for initializing a potential expansion

    //  Density models without a corresponding potential
    PT_ELLIPSOIDAL,  ///< a generalization of spherical mass profile with arbitrary axis ratios:  CDensityEllipsoidal
    PT_MGE,          ///< Multi-Gaussian expansion:  CDensityMGE
//    PT_SERSIC,       ///< Sersic density profile:  CDensitySersic
//    PT_EXPDISK,      ///< exponential (in R) disk with a choice of vertical density profile:  CDensityExpDisk

    //  Generic potential expansions
    PT_BSE,          ///< basis-set expansion for infinite systems:  `BasisSetExp`
    PT_SPLINE,       ///< spline spherical-harmonic expansion:  `SplineExp`
    PT_CYLSPLINE,    ///< expansion in azimuthal angle with two-dimensional meridional-plane interpolating splines:  `CylSplineExp`
    PT_MULTIPOLE,    ///< axisymmetric multipole expansion from GalPot:  `Multipole`

    //  Potentials with possibly infinite mass that can't be used as source density for a potential expansion
    PT_GALPOT,       ///< Walter Dehnen's GalPot (exponential discs and spheroids)
    PT_COMPOSITE,    ///< a superposition of multiple potential instances:  `CompositeCyl`
    PT_LOG,          ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC,     ///< triaxial simple harmonic oscillator:  `Harmonic`
//    PT_SCALEFREE,    ///< triaxial single power-law density profile:  CPotentialScaleFree
//    PT_SCALEFREESH,  ///< spherical-harmonic approximation to a triaxial power-law density:  CPotentialScaleFreeSH
    PT_NFW,          ///< spherical Navarro-Frenk-White profile:  `NFW`

    //  Analytic finite-mass potential models that can also be used as source density for a potential expansion
//    PT_SPHERICAL,    ///< arbitrary spherical mass model:  CPotentialSpherical
    PT_MIYAMOTONAGAI,///< axisymmetric Miyamoto-Nagai(1975) model:  `MiyamotoNagai`
    PT_DEHNEN,       ///< spherical, axisymmetric or triaxial Dehnen(1993) density model:  `Dehnen`
    PT_FERRERS,      ///< triaxial Ferrers model with finite extent:  `Ferrers`
    PT_PLUMMER,      ///< spherical Plummer model:  `Plummer`
//    PT_ISOCHRONE,    ///< spherical isochrone model:  `Isochrone`
    PT_PERFECT_ELLIPSOID,  ///< oblate axisymmetric Perfect Ellipsoid of Kuzmin/de Zeeuw :  `OblatePerfectEllipsoid`
};

/// structure that contains parameters for all possible potentials
struct ConfigPotential
{
    PotentialType potentialType;   ///< type of the potential
    PotentialType densityType;     ///< specifies the density model used for initializing a potential expansion
    SymmetryType symmetryType;     ///< degree of symmetry (mainly used to explicitly disregard certain terms in a potential expansion)
    double mass;                   ///< total mass of the model (not applicable to all potential types)
    double scaleRadius;            ///< scale radius of the model (if applicable)
    double scaleRadius2;           ///< second scale radius of the model (if applicable)
    double q, p;                   ///< axis ratio of the model (if applicable)
    double gamma;                  ///< central cusp slope (for Dehnen and scale-free models)
    double sersicIndex;            ///< Sersic index (for Sersic density model)
    unsigned int numCoefsRadial;   ///< number of radial terms in BasisSetExp or grid points in spline potentials
    unsigned int numCoefsAngular;  ///< number of angular terms in spherical-harmonic expansion
    unsigned int numCoefsVertical; ///< number of coefficients in z-direction for CylSplineExp potential
    double alpha;                  ///< shape parameter for BasisSetExp potential
    double splineSmoothFactor;     ///< amount of smoothing in SplineExp initialized from an N-body snapshot
    double splineRMin, splineRMax; ///< if nonzero, specifies the inner- and outermost grid node radii for SplineExp and CylSplineExp
    double splineZMin, splineZMax; ///< if nonzero, gives the grid extent in z direction for CylSplineExp
    std::string fileName;          ///< name of file with coordinates of points, or coefficients of expansion, or any other external data array
    units::ExternalUnits units;    ///< specification of length, velocity and mass units for N-body snapshot, in internal code units
    /// default constructor initializes the fields to some reasonable values
    ConfigPotential() :
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(ST_DEFAULT),
        mass(1.), scaleRadius(1.), scaleRadius2(1.), q(1.), p(1.), gamma(1.), sersicIndex(4.),
        numCoefsRadial(20), numCoefsAngular(6), numCoefsVertical(20),
        alpha(0.), splineSmoothFactor(1.), splineRMin(0), splineRMax(0), splineZMin(0), splineZMax(0),
        units()  {};
};

/** Parse the potential parameters contained in a text array of "key=value" pairs.
    \param[in]     params  is the array of string pairs "key" and "value", for instance,
    created from command-line arguments, or read from an INI file.
    \param[in,out] config  is the structure containing all parameters of potential;
    only those members are updated which have been found in the `params` array,
    except for potential/density/symmetryType and fileName which are initialized by default values.
*/
void parseConfigPotential(const utils::KeyValueMap& params, ConfigPotential& config);

/** Store the potential parameters into a text array of "key=value" pairs.
    \param[in]     config  is the structure containing all parameters of potential;
    \param[in,out] params  is the textual representation of these parameters, in terms of 
    array of string pairs "key" and "value". 
    All parameters of `config` are added to this array, possibly replacing pre-existing 
    elements with the same name; existing elements with key names that do not correspond 
    to any of potential parameters are not modified.
*/
void storeConfigPotential(const ConfigPotential& config, utils::KeyValueMap& params);

///@}
/// \name Factory routines that create an instance of specific potential from a set of parameters
///@{

/** Create a density model according to the parameters. 
    This only deals with finite-mass models, including some of the Potential descendants.
    This function is rarely needed by itself, and is used within `createPotential()` to construct 
    temporary density models for initializing a potential expansion.
    \param[in] config  contains the parameters (density type, mass, shape, etc.)
    \return    the instance of a class derived from BaseDensity
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular density model
*/
const BaseDensity* createDensity(const ConfigPotential& config);


/** Create an instance of potential model according to the parameters passed. 
    This is the main 'factory' function that can construct a potential from a variety of sources.
    \param[in,out] config  specifies the potential parameters, which could be modified, 
                   e.g., if the potential coefficients are loaded from a file.
    \return        the instance of potential
    \throw         std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular potential model
*/
const BasePotential* createPotential(ConfigPotential& config);


/** Create a potential of a generic expansion kind from a set of point masses.
    \param[in] config  contains the parameters (potential type, number of terms in expansion, etc.)
    \param[in] points  is the array of particles that are used in computing the coefficients; 
    \return    a new instance of potential
    \throw     std::invalid_argument exception if the potential type is incorrect,
    or any other potential-specific exception that may occur in the constructor
*/
template<typename ParticleT>
const BasePotential* createPotentialFromPoints(const ConfigPotential& config, 
    const particles::PointMassArray<ParticleT>& points);


/** Load a potential from a text or snapshot file.

    The input file may contain one of the following kinds of data:
    - a Nbody snapshot in a text or binary format, handled by classes derived from particles::BaseIOSnapshot;
    - a potential coefficients file for BasisSetExp, SplineExp, or CylSplineExp;
    - a density model described by CDensityEllipsoidal or CDensityMGE.

    The data format is determined from the file: the last two cases have 
    their specific signatures in the first line of the file, and if none is found, 
    the file is assumed to be an Nbody snapshot.

    In the second case, the file format determines the potential expansion type,
    and all other parameters are read from the coefficients file, so the only required element 
    in `config` is `fileName`. Upon reading the coefficients file, the relevant potential 
    parameters in `config` (e.g., `potentialType`, `numCoefsRadial`, etc.) are updated.

    In the other two cases, `config.potentialType` must be specified a priori,
    and `config.densityType` will be set to `PT_NB` (in the former case) or 
    `PT_ELLIPSOIDAL`, `PT_MGE` (in the latter case).
    Parameters of potential expansion (e.g., number of terms) must also be specified in `config`.
    Upon creating an instance of potential expansion class, its coefficients are stored 
    in a file with the same name and an appropriate extension for the given potential type,
    using `writePotential()` function, so that later one may instead load this coefficients file
    to speed up initialization.

    \param[in,out] config  contains the potential parameters and may be updated 
                   upon reading the file. The only parameter required in all cases is
                   `config.fileName`, which specifies the file to read.
    \return        a new instance of BasePotential* on success.
    \throws        std::invalid_argument or std::runtime_error or other potential-specific exception
                   on failure (e.g., if potential type is inappropriate, or a file does not exist).
*/
const BasePotential* readPotential(ConfigPotential& config);
    

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
    const char* filename, const units::ExternalUnits& converter);

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
    const char* filename, const units::InternalUnits& unit) {  // create a temporary converter; velocity unit is not used
    return readGalaxyPotential(filename, units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun));
}


/** Write potential expansion coefficients to a text file.

    The potential must be one of the following expansion classes: 
    `BasisSetExp`, `SplineExp`, `CylSplineExp`.
    The coefficients stored in a file may be later loaded by `readPotential()` function.
    \param[in] fileName is the output file
    \param[in] potential is the pointer to potential
    \throws    std::invalid_argument if the potential is of inappropriate type,
    or std::runtime error if the file is not writeable.
*/
void writePotential(const std::string &fileName, const BasePotential& potential);

///@}
/// \name Correspondence between potential/density names and corresponding classes
///@{

/// return the name of the potential of a given type, or empty string if unavailable
const char* getPotentialNameByType(PotentialType type);

/// return the name of the density of a given type, or empty string if unavailable
const char* getDensityNameByType(PotentialType type);

/// return the name of the symmetry of a given type, or empty string if unavailable
const char* getSymmetryNameByType(SymmetryType type);

/// return the type of density or potential object
PotentialType getPotentialType(const BaseDensity& d);

/// return the type of the potential model by its name, or PT_UNKNOWN if unavailable
PotentialType getPotentialTypeByName(const std::string& PotentialName);

/// return the type of the density model by its name, or PT_UNKNOWN if unavailable
PotentialType getDensityTypeByName(const std::string& DensityName);

/// return the type of symmetry by its name, or ST_DEFAULT if unavailable
SymmetryType getSymmetryTypeByName(const std::string& SymmetryName);

/// return file extension for writing the coefficients of potential of the given type,
/// or empty string if the potential type is not one of the expansion types
const char* getCoefFileExtension(PotentialType potType);

/// return file extension for writing the coefficients of a given potential object (overload)
inline const char* getCoefFileExtension(const BasePotential& p) {
    return getCoefFileExtension(getPotentialType(p)); }

/// find potential type by file extension, return PT_UNKNOWN if the extension is not recognized
PotentialType getCoefFileType(const std::string& fileName);

///@}

}; // namespace
