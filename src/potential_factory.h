/** \file    potential_factory.h
    \brief   Creation and input/output of Potential instances
    \author  EV
    \date    2010-2015

    This file provides several utility function to manage instances of BaseDensity and BasePotential: 
    creating a density or potential model from parameters provided in ConfigPotential, 
    creating a potential from a set of point masses or from an N-body snapshot file,
    loading potential coefficients from a text file, 
    writing expansion coefficients to a text file,
    and creating a spherical mass model that approximates the given density model.
    Note that potential here is elementary (non-composite, no central black hole).
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

/// List of all known potential and density types (borrowed from SMILE, not everything is implemented here)
enum PotentialType {
    PT_UNKNOWN,     ///< undefined
    PT_COEFS,       ///< not an actual density model, but a way to load pre-computed coefficients of potential expansion
    PT_COMPOSITE,   ///< a superposition of multiple potential instances:  CPotentialComposite
    PT_NB,          ///< a set of frozen particles:  CPotentialNB
    PT_BSE,         ///< basis-set expansion for infinite systems:  CPotentialBSE
    PT_BSECOMPACT,  ///< basis-set expansion for systems with non-singular density and finite extent:  CPotentialBSECompact
    PT_SPLINE,      ///< spline spherical-harmonic expansion:  CPotentialSpline
    PT_CYLSPLINE,   ///< expansion in azimuthal angle with two-dimensional meridional-plane interpolating splines:  CPotentialCylSpline
    PT_LOG,         ///< logaritmic potential:  CPotentialLog
    PT_HARMONIC,    ///< simple harmonic oscillator:  CPotentialHarmonic
    PT_SCALEFREE,   ///< single power-law density profile:  CPotentialScaleFree
    PT_SCALEFREESH, ///< spherical-harmonic approximation to a power-law density:  CPotentialScaleFreeSH
    PT_SPHERICAL,   ///< arbitrary spherical mass model:  CPotentialSpherical
    PT_DEHNEN,      ///< Dehnen(1993) density model:  CPotentialDehnen
    PT_MIYAMOTONAGAI,///< Miyamoto-Nagai(1975) flattened model:  CPotentialMiyamotoNagai
    PT_FERRERS,     ///< Ferrers finite-extent profile:  CPotentialFerrers
    PT_PLUMMER,     ///< Plummer model:  CDensityPlummer
    PT_ISOCHRONE,   ///< isochrone model:  CDensityIsochrone
    PT_PERFECTELLIPSOID,  ///< Kuzmin/de Zeeuw integrable potential:  CDensityPerfectEllipsoid
    PT_NFW,         ///< Navarro-Frenk-White profile:  CDensityNFW
    PT_SERSIC,      ///< Sersic density profile:  CDensitySersic
    PT_EXPDISK,     ///< exponential (in R) disk with a choice of vertical density profile:  CDensityExpDisk
    PT_ELLIPSOIDAL, ///< a generalization of spherical mass profile with arbitrary axis ratios:  CDensityEllipsoidal
    PT_MGE,         ///< Multi-Gaussian expansion:  CDensityMGE
    PT_GALPOT,      ///< Walter Dehnen's GalPot (exponential discs and spheroids)
};

/// structure that contains parameters for all possible potentials
struct ConfigPotential
{
    double mass;                             ///< total mass of the model (not applicable to all potential types)
    double scalerad;                         ///< scale radius of the model (if applicable)
    double scalerad2;                        ///< second scale radius of the model (if applicable)
    double q, p;                             ///< axis ratio of the model (if applicable)
    double gamma;                            ///< central cusp slope (for Dehnen and scale-free models)
    double sersicIndex;                      ///< Sersic index (for Sersic density model)
    unsigned int numCoefsRadial;             ///< number of radial terms in BasisSetExp or grid points in spline potentials
    unsigned int numCoefsAngular;            ///< number of angular terms in spherical-harmonic expansion
    unsigned int numCoefsVertical;           ///< number of coefficients in z-direction for Cylindrical potential
    double alpha;                            ///< shape parameter for BSE potential
#if 0
    double rmax;                             ///< radius of finite density model for BSECompact potential
    double treecodeEps;                      ///< treecode smooothing length (negative means adaptive smoothing based on local density, absolute value is the proportionality coefficient between eps and mean interparticle distance)
    double treecodeTheta;                    ///< tree cell opening angle
#endif
    PotentialType potentialType;             ///< currently selected potential type
    PotentialType densityType;               ///< if pot.type == BSE or Spline, this gives the underlying density profile approximated by these expansions or flags that an Nbody file should be used
    SymmetryType symmetryType;               ///< if using Nbody file with the above two potential expansions, may assume certain symmetry on the coefficients (do not compute them but just assign to zero)
    double splineSmoothFactor;               ///< for smoothing Spline potential coefs initialized from discrete point mass set
    double splineRMin, splineRMax;           ///< if nonzero, specifies the inner- and outermost grid node radii
    double splineZMin, splineZMax;           ///< if nonzero, gives the grid extent in z direction for Cylindrical spline potential
    std::string fileName;                    ///< name of file with coordinates of points, or coefficients of expansion, or any other external data array
#if 0
    double mbh;                              ///< mass of central black hole (in the composite potential)
    double binary_q;                         ///< binary BH mass ratio (0<=q<=1)
    double binary_sma;                       ///< binary BH semimajor axis
    double binary_ecc;                       ///< binary BH eccentricity (0<=ecc<1)
    double binary_phase;                     ///< binary BH orbital phase (0<=phase<2*pi)
#endif
    units::ExternalUnits units;              ///< specification of length, velocity and mass units for N-body snapshot, in internal code units
    /// default constructor initializes the fields to some reasonable values
    ConfigPotential() :
        mass(1.), scalerad(1.), scalerad2(1.), q(1.), p(1.), gamma(1.), sersicIndex(4.),
        numCoefsRadial(20), numCoefsAngular(0), numCoefsVertical(20), alpha(0.),
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(ST_DEFAULT),
        splineSmoothFactor(1.), splineRMin(0), splineRMax(0), splineZMin(0), splineZMax(0),
        units()  {};
};

/// parse the potential parameters contained in a text array of "key=value" pairs
void parseConfigPotential(const utils::KeyValueMap& params, ConfigPotential& config);

/// store the potential parameters into a text array of "key=value" pairs
void storeConfigPotential(const ConfigPotential& config, utils::KeyValueMap& params);

///@}
/// \name Factory routines that create an instance of specific potential from a set of parameters
///@{

/** create a density model according to the parameters. 
    This only deals with finite-mass models, including some of the Potential descendants.
    \param[in] config  contains the parameters (density type, mass, shape, etc.)
    \return    the instance of a class derived from BaseDensity
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular density model
*/
const BaseDensity* createDensity(const ConfigPotential& config);

/** create an instance of potential model according to the parameters passed. 
    \param[in,out] config  specifies the potential parameters, which could be modified, 
                   e.g. if the potential coefficients are loaded from a file.
                   Massive black hole (config->Mbh) is not included in the potential 
                   (the returned potential is non-composite)
    \return        the instance of potential
    \throw         std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular potential model
*/
const BasePotential* createPotential(ConfigPotential& config);

/** create a potential of a generic expansion kind from a set of point masses.
    \param[in] config  contains the parameters (potential type, number of terms in expansion, etc.)
    \param[in] points  is the array of particles that are used in computing the coefficients; 
    \return    a new instance of potential
    \throw     std::invalid_argument exception if the potential type is incorrect,
    or any other potential-specific exception that may occur in the constructor
*/
template<typename ParticleT>
const BasePotential* createPotentialFromPoints(const ConfigPotential& config, 
    const particles::PointMassArray<ParticleT>& points);

/** load a potential from a text or snapshot file.

    The input file may contain one of the following kinds of data:
    - a Nbody snapshot in a text or binary format, handled by classes derived from particles::BasicIOSnapshot;
    - a potential coefficients file for BasisSetExp, SplineExp, or CylSplineExp;
    - a density model described by CDensityEllipsoidal or CDensityMGE.
    The data format is determined from the first line of the file, and 
    if it is allowed by the parameters passed in `config`, then 
    the file is read and the instance of a corresponding potential is created. 
    If the input data was not the potential coefficients and the new potential 
    is of BasisSetExp, SplineExp or CylSplineExp type, then a new file with 
    potential coefficients is created and written via `writePotential()`, 
    so that later one may load this coef file instead of the original one, 
    which speeds up initialization.
    \param[in,out] config  contains the potential parameters and may be updated 
                   upon reading the file (e.g. the number of expansion coefficients 
                   may change). If the file doesn't contain appropriate kind of potential
                   (i.e. if config.potentialType is PT_NB but the file contains 
                   BSE coefficients or a description of MGE model), an error is returned.
                   `config.fileName` contains the file name from which the data is loaded.
    \return        a new instance of BasePotential* on success
    \throws        std::invalid_argument or std::runtime_error or other potential-specific exception
    on failure (e.g., if potential type is inappropriate, or a file does not exist)
*/
const BasePotential* readPotential(ConfigPotential& config);
    
/** Utility function providing a legacy interface compatible with the original GalPot.
    It reads the parameters from a text file and converts them into the internal unit system, 
    then constructs the potential using `createGalaxyPotential` routine.
    \param[in]  filename is the name of parameter file;
    \param[in]  units is the specification of internal unit system;
    \returns    the new CompositeCyl potential
    \throws     a std::runtime_error exception if file is not readable or does not contain valid parameters.
*/
const potential::BasePotential* readGalaxyPotential(const char* filename, const units::InternalUnits& units);
    
/** write potential expansion coefficients to a text file.

    The potential must be one of the following expansion classes: 
    BasisSetExp, SplineExp, CylSplineExp.
    The coefficients stored in a file may be later loaded by readPotential() function.
    \param[in] fileName is the output file
    \param[in] potential is the pointer to potential
    \throws    std::invalid_argument if the potential is of inappropriate type,
    or std::runtime error if the file is not writeable.
*/
void writePotential(const std::string &fileName, const BasePotential& potential);

#if 0
/** create an equivalent spherical mass model for the given density profile. 
    \param[in] density is the non-spherical density model to be approximated
    \param[in] poten (optional) another spherical mass model that provides the potential 
               (if it is not given self-consistently by this density profile).
    \param[in] numNodes specifies the number of radial grid points in the M(r) profile
    \param[in] Rmin is the radius of the innermost non-zero grid node (0 means auto-select)
    \param[in] Rmax is the radius outermost grid node (0 means auto-select)
    \return    new instance of CMassModel on success, or NULL on fail (e.g. if the density model was infinite) 
*/
CMassModel* createMassModel(const CDensity* density, int numNodes=NUM_RADIAL_POINTS_SPHERICAL_MODEL, 
    double Rmin=0, double Rmax=0, const CMassModel* poten=NULL);
#endif

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

/// return file extension for writing the coefficients of potential of the given type
const char* getCoefFileExtension(PotentialType potType);

/// find potential type by file extension
PotentialType getCoefFileType(const std::string& fileName);

///@}

}; // namespace
