#include "potential_factory.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_disk.h"
#include "potential_ferrers.h"
#include "potential_king.h"
#include "potential_multipole.h"
#include "potential_perfect_ellipsoid.h"
#include "potential_spheroid.h"
#include "particles_io.h"
#include "math_core.h"
#include "utils.h"
#include "utils_config.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <map>
/// OS- and filesystem-specific definitions
#ifdef _MSC_VER
#include <direct.h>
#pragma warning(disable:4996)  // prevent deprecation error on chdir and getcwd
#define DIRECTORY_SEPARATOR '\\'
#else
#include <unistd.h>
#define DIRECTORY_SEPARATOR '/'
#endif

namespace potential {

namespace {  // internal definitions and routines

/// order of the Multipole expansion for the GalPot potential
static const int GALPOT_LMAX = 32;

/// [default] order of the azimuthal Fourier expansion in case of non-axisymmetric components
static const int GALPOT_MMAX = 6;

/// number of radial points in the Multipole potential automatically constructed for GalPot
static const int GALPOT_NRAD = 50;

/// \name Definitions of all known potential types and parameters
//        -------------------------------------------------------
///@{

/** List of all known potential and density types.
    Note that this type is not a substitute for the real class hierarchy:
    it is intended only to be used in factory methods such as 
    creating an instance of potential from its name 
    (e.g., passed as a string, or loaded from an ini file).
    Implemented as a bitmask to enable testing for multiple choices in parseParam.
*/
enum PotentialType {
    PT_UNKNOWN      = 0, ///< unspecified/not provided
    PT_INVALID      = 1, ///< provided but does not correspond to a known class

    // density interpolators
    PT_DENS_SPHHARM = 2, ///< `DensitySphericalHarmonic`
    PT_DENS_CYLGRID = 4, ///< `DensityAzimuthalHarmonic`

    // generic potential expansions
    PT_BASISSET    = 16, ///< radial basis-set expansion: `BasisSet`
    PT_MULTIPOLE   = 32, ///< spherical-harmonic expansion:  `Multipole`
    PT_CYLSPLINE   = 64, ///< expansion in azimuthal angle with 2d interpolating splines in (R,z):  `CylSpline`

    // components of GalPot
    PT_DISK      = 1024, ///< separable disk density model:  `Disk`
    PT_SPHEROID  = 2048, ///< double-power-law 3d density model:  `Spheroid`
    PT_NUKER     = 4096, ///< double-power-law surface density profile: `Nuker`
    PT_SERSIC    = 8192, ///< Sersic profile:  `Sersic`

    // analytic potentials that can't be used as source density for a potential expansion
    PT_LOG                 =  65536, ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC            = 131072, ///< triaxial simple harmonic oscillator:  `Harmonic`
    PT_KEPLERBINARY        = 262144, ///< two point masses on a Kepler orbit: `KeplerBinary`
    PT_UNIFORMACCELERATION = 524288, ///< a spatially uniform but time-dependent acceleration: `UniformAcceleration`
    PT_EVOLVING            =1048576, ///< a time-dependent series of potentials: `Evolving`

    // analytic potential models that can also be used as source density for a potential expansion
    PT_NFW              =   4194304, ///< spherical Navarro-Frenk-White profile:  `NFW`
    PT_MIYAMOTONAGAI    =   8388608, ///< axisymmetric Miyamoto-Nagai(1975) model:  `MiyamotoNagai`
    PT_LONGMURALI       =  16777216, ///< triaxial Long&Murali(1992) bar model: `LongMurali'
    PT_DEHNEN           =  33554432, ///< spherical, axisymmetric or triaxial Dehnen(1993) density model:  `Dehnen`
    PT_FERRERS          =  67108864, ///< triaxial Ferrers model with finite extent:  `Ferrers`
    PT_PLUMMER          = 134217728, ///< spherical Plummer model:  `Plummer`
    PT_ISOCHRONE        = 268435456, ///< spherical isochrone model:  `Isochrone`
    PT_PERFECTELLIPSOID = 536870912, ///< axisymmetric model of Kuzmin/de Zeeuw :  `PerfectEllipsoid`
    PT_KING             =1073741824, ///< generalized King (lowered isothermal) model, represented by Multipole
};

/// parameters of various density/potential modifiers (not parsed or unit-converted);
/// this class also contains a reference to the unit converter, which is passed around to various routines
/// that actually use it to parse the parameter values or the expansion coefficients stored in text files
struct ModifierParams {
    const units::ExternalUnits& converter; ///< an instance of unit converter
    std::string center;      ///< coordinates of the center offset or the name of a file with these offsets
    std::string orientation; ///< orientation of the principal axes specified by Euler angles
    std::string rotation;    ///< name of a file with time-dependent rotation about the z axis, or a single value
    std::string scale;       ///< name of a file with time-dependent amplitude and length scale modulation, or 2 values
    ModifierParams(const units::ExternalUnits& _converter) : converter(_converter) {}
    // helper functions for comparing the values of modifier params
    bool operator == (const ModifierParams& x) const
    { return center==x.center && orientation==x.orientation && rotation==x.rotation && scale==x.scale; }
    bool operator != (const ModifierParams& x) const
    { return !(*this==x); }
    bool empty() const
    { return center.empty() && orientation.empty() && rotation.empty() && scale.empty(); }
};

/// a compendium of parameters for all possible density or potential models
/// (not all of them make sense for any given model)
struct AllParam: public ModifierParams
{
    PotentialType potentialType;      ///< type of the potential
    PotentialType densityType;        ///< density model used for initializing a potential expansion
    coord::SymmetryType symmetryType; ///< degree of symmetry
    // various subsets of these parameters are used for different potentials
    double mass;                      ///< total mass
    double surfaceDensity;            ///< central surface density for Disk, Nuker or Sersic models
    double densityNorm;               ///< density normalization for double-power-law models
    double scaleRadius;               ///< scale radius
    double scaleHeight;               ///< scale height for Disk, MiyamotoNagai and LongMurali models
    double barLength;                 ///< bar length for the LongMurali model
    double innerCutoffRadius;         ///< central hole for the Disk model
    double outerCutoffRadius;         ///< truncation radius for double-power-law models
    double v0;                        ///< limiting circular velocity for Logarithmic potential
    double Omega;                     ///< frequency for Harmonic potential
    double axisRatioY, axisRatioZ;    ///< axis ratios (y/x, z/x) in non-spherical cases
    double alpha, beta, gamma;        ///< parameters of double-power-law models
    double modulationAmplitude;       ///< disk surface density wiggliness
    double cutoffStrength;            ///< steepness of the exponential cutoff for double-power-law
    double sersicIndex;               ///< sersic index for Disk or Sersic models
    double W0;                        ///< dimensionless potential depth for King models
    double trunc;                     ///< truncation strength for generalized King models
    double binary_q;                  ///< parameters for the KeplerBinary potential: mass ratio q
    double binary_sma;                ///< binary semimajor axis (orbit size)
    double binary_ecc;                ///< binary eccentricity
    double binary_phase;              ///< orbital phase of the binary
    // parameters of potential expansions
    unsigned int gridSizeR; ///< number of radial grid points in Multipole and CylSpline potentials
    unsigned int gridSizez; ///< number of grid points in z-direction for CylSpline potential
    unsigned int nmax;      ///< order of radial expansion for BasisSet (actual number of terms is nmax+1)
    unsigned int lmax;      ///< number of angular terms in spherical-harmonic expansion
    unsigned int mmax;      ///< number of angular terms in azimuthal-harmonic expansion
    double rmin, rmax;      ///< inner- and outermost grid node radii for Multipole and CylSpline
    double zmin, zmax;      ///< grid extent in z direction for CylSpline
    double smoothing;       ///< amount of smoothing in Multipole initialized from an N-body snapshot
    double eta;             ///< shape parameters of basis functions for BasisSet (0.5-CB, 1.0-HO, etc.)
    double r0;              ///< scale radius of the basis functions for BasisSet
    bool fixOrder;          ///< whether to limit the internal SH density expansion to the output order
    std::string file;       ///< name of a file with coordinates of points, or coefficients of expansion
    double lengthUnit;      ///< dimensional length unit for Logarithmic (taken from ExternalUnits)
    /// default constructor initializes the fields to some reasonable values
    AllParam(const units::ExternalUnits& converter) :
        ModifierParams(converter),
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(coord::ST_UNKNOWN),
        mass(1.), surfaceDensity(NAN), densityNorm(NAN), scaleRadius(1.), scaleHeight(1.),
        barLength(0), innerCutoffRadius(0.), outerCutoffRadius(INFINITY),
        v0(1.), Omega(1.),
        axisRatioY(1.), axisRatioZ(1.),
        alpha(1.), beta(4.), gamma(1.),
        modulationAmplitude(0.), cutoffStrength(2.), sersicIndex(NAN), W0(NAN), trunc(1.),
        binary_q(0), binary_sma(0), binary_ecc(0), binary_phase(0),
        gridSizeR(25), gridSizez(25), nmax(12), lmax(6), mmax(6), rmin(0), rmax(0), zmin(0), zmax(0),
        smoothing(1.), eta(1.0), r0(0), fixOrder(false), lengthUnit(1)
    {}
};

///@}
/// \name Correspondence between enum potential and symmetry types and string names
//        -------------------------------------------------------------------------
///@{

/// return the type of the potential or density model by its name,
/// PT_UNKNOWN if the name is not provided, or PT_INVALID if it is provided incorrectly
PotentialType getPotentialTypeByName(const std::string& name)
{
    if(name.empty()) return PT_UNKNOWN;
    if(utils::stringsEqual(name, Logarithmic  ::myName())) return PT_LOG;
    if(utils::stringsEqual(name, Harmonic     ::myName())) return PT_HARMONIC;
    if(utils::stringsEqual(name, KeplerBinary ::myName())) return PT_KEPLERBINARY;
    if(utils::stringsEqual(name, NFW          ::myName())) return PT_NFW;
    if(utils::stringsEqual(name, Plummer      ::myName())) return PT_PLUMMER;
    if(utils::stringsEqual(name, Dehnen       ::myName())) return PT_DEHNEN;
    if(utils::stringsEqual(name, Ferrers      ::myName())) return PT_FERRERS;
    if(utils::stringsEqual(name, Isochrone    ::myName())) return PT_ISOCHRONE;
    if(utils::stringsEqual(name, SpheroidParam::myName())) return PT_SPHEROID;
    if(utils::stringsEqual(name, NukerParam   ::myName())) return PT_NUKER;
    if(utils::stringsEqual(name, SersicParam  ::myName())) return PT_SERSIC;
    if(utils::stringsEqual(name, DiskDensity  ::myName())) return PT_DISK;
    if(utils::stringsEqual(name, BasisSet     ::myName())) return PT_BASISSET;
    if(utils::stringsEqual(name, Multipole    ::myName())) return PT_MULTIPOLE;
    if(utils::stringsEqual(name, CylSpline    ::myName())) return PT_CYLSPLINE;
    if(utils::stringsEqual(name, MiyamotoNagai::myName())) return PT_MIYAMOTONAGAI;
    if(utils::stringsEqual(name, LongMurali   ::myName())) return PT_LONGMURALI;
    if(utils::stringsEqual(name, "King"))                  return PT_KING;
    if(utils::stringsEqual(name, Evolving     ::myName())) return PT_EVOLVING;
    if(utils::stringsEqual(name, UniformAcceleration     ::myName())) return PT_UNIFORMACCELERATION;
    if(utils::stringsEqual(name, PerfectEllipsoid        ::myName())) return PT_PERFECTELLIPSOID;
    if(utils::stringsEqual(name, DensitySphericalHarmonic::myName())) return PT_DENS_SPHHARM;
    if(utils::stringsEqual(name, DensityAzimuthalHarmonic::myName())) return PT_DENS_CYLGRID;
    return PT_INVALID;
}

} // internal namespace

// return the type of symmetry by its name, or ST_UNKNOWN if unavailable
coord::SymmetryType getSymmetryTypeByName(const std::string& symmetryName)
{
    if(symmetryName.empty())  // empty value is valid and means "unknown"
        return coord::ST_UNKNOWN;
    // compare only the first letter, case-insensitive
    switch(tolower(symmetryName[0])) {
        case 's': return coord::ST_SPHERICAL;
        case 'a': return coord::ST_AXISYMMETRIC;
        case 't': return coord::ST_TRIAXIAL;
        case 'b': return coord::ST_BISYMMETRIC;
        case 'r': return coord::ST_REFLECTION;
        case 'n': return coord::ST_NONE;
    }
    // otherwise it could be an integer constant representing the numerical value of sym.type
    int sym = coord::ST_UNKNOWN;
    try{ sym = utils::toInt(symmetryName); }
    catch(std::exception&) { sym = coord::ST_UNKNOWN; }  // parse error - it wasn't a valid number
    // a non-empty value was provided, but is not a valid one - now that's an error
    if(isUnknown(coord::SymmetryType(sym)))
        throw std::runtime_error("Invalid symmetry type: " + symmetryName);
    return static_cast<coord::SymmetryType>(sym);
}

// inverse of the above: return a symbolic name or a numerical code of symmetry type
std::string getSymmetryNameByType(coord::SymmetryType type)
{
    switch(type) {
        case coord::ST_UNKNOWN:      return "Unknown";
        case coord::ST_NONE:         return "None";
        case coord::ST_REFLECTION:   return "Reflection";
        case coord::ST_BISYMMETRIC:  return "Bisymmetric";
        case coord::ST_TRIAXIAL:     return "Triaxial";
        case coord::ST_AXISYMMETRIC: return "Axisymmetric";
        case coord::ST_SPHERICAL:    return "Spherical";
        default:  return utils::toString((int)type);
    }
}

namespace{

///@}
/// \name Conversion between string key/value maps and structured potential parameters
//        ----------------------------------------------------------------------------
///@{

/** Extract a value from the array of key=value pairs that matches the given key1 or its synonym key2,
    and assign it to the corresponding parameter in the AllParam structure if this makes sense.
*/
std::string popString(const utils::KeyValueMap& kvmap,
    std::vector<std::string>& keys, const std::string& key1, const std::string& key2="",
    bool isAllowed=true, const char* errorMessage=NULL)
{
    std::vector<std::string>::iterator found1=keys.end(), found2=keys.end();
    for(std::vector<std::string>::iterator key=keys.begin(); key!=keys.end(); key++) {
        if(utils::stringsEqual(*key, key1))
            found1 = key;
        if(!key2.empty() && utils::stringsEqual(*key, key2))
            found2 = key;
    }
    if(found1 == keys.end() && found2 == keys.end())
        return "";
    if(found1 != keys.end() && found2 != keys.end())
        throw std::runtime_error(
            "Duplicate values for synonymous parameters " + *found1 + " and " + *found2);
    std::vector<std::string>::iterator key = found1 != keys.end() ? found1 : found2;
    const std::string value = kvmap.getString(*key);
    if(value.empty())
        throw std::runtime_error("Empty value for parameter " + *key);
    if(!isAllowed)
        throw std::runtime_error(errorMessage!=NULL ? errorMessage:
            ("Parameter '" + *key + "' is not allowed for the given model").c_str());
    // remove the parsed parameter from the list of keys
    keys.erase(key);
    return value;
}

inline void assignParam(double& param, const std::string& value, double unit=1.0)
{
    if(!value.empty())
        param = utils::toDouble(value);
    param *= unit;
}

inline void assignParam(bool& param, const std::string& value)
{
    if(!value.empty())
        param = utils::toBool(value);
}

inline void assignParam(unsigned int& param, const std::string& value)
{
    if(value.empty())
        return;
    int result = utils::toInt(value);
    if(result < 0)
        throw std::runtime_error("Parse error: negative value "+value+" is not allowed");
    param = result;
}

/** Parse the potential or density parameters contained in a text array of "key=value" pairs.
    \param[in] kvmap  is the array of string pairs "key" and "value", for instance,
    created from command-line arguments, or read from an INI file;
    \param[in] converter  is the instance of unit converter for translating the dimensional
    parameters (such as mass or scale radius) into internal units (may be a trivial converter);
    \return    the structure containing all possible potential/density parameters,
    including the reference to the unit converter, which may be used in subsequent routines
    to interpret some of the parameters not parsed by this function (e.g. contained in text files).
    \throw std::runtime_error if the array contains parameters that are not allowed for the given model,
    or if there are duplicate values for the same parameter.
*/
AllParam parseParam(const utils::KeyValueMap& kvmap, const units::ExternalUnits& conv)
{
    AllParam param(conv);
    std::vector<std::string> keys = kvmap.keys();
    // assign parameters, checking that they make sense for the given density/potential type
    param.potentialType = getPotentialTypeByName(popString(kvmap, keys, "type", "", true));
    param.densityType   = getPotentialTypeByName(popString(kvmap, keys, "density", "",
        (param.potentialType == PT_UNKNOWN) ||
        (param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID |
        PT_BASISSET | PT_MULTIPOLE | PT_CYLSPLINE))));
    PotentialType type  = param.densityType != PT_UNKNOWN ? param.densityType : param.potentialType;
    if(type == PT_INVALID)
        // don't perform any further checks; the calling code will raise an exception if the type
        // is invalid, and we don't want to override it with another message about unknown parameters
        return param;
    bool massProvided = kvmap.contains("mass");
    assignParam(param.mass,                popString(kvmap, keys, "mass", "",
        type & (PT_DISK | PT_SPHEROID | PT_NUKER | PT_SERSIC | PT_KEPLERBINARY |
        PT_NFW | PT_MIYAMOTONAGAI | PT_LONGMURALI | PT_DEHNEN | PT_FERRERS | PT_PLUMMER |
        PT_ISOCHRONE | PT_PERFECTELLIPSOID | PT_KING)),
        conv.massUnit);
    assignParam(param.surfaceDensity,      popString(kvmap, keys, "surfaceDensity", "Sigma0",
        (type & (PT_DISK | PT_NUKER | PT_SERSIC)) && !massProvided,
        massProvided ? "Parameters 'mass' and 'surfaceDensity' are mutually exclusive" : NULL),
        conv.massUnit / pow_2(conv.lengthUnit));
    assignParam(param.densityNorm,         popString(kvmap, keys, "densityNorm", "rho0",
        (type & PT_SPHEROID) && !massProvided,
        massProvided ? "Parameters 'mass' and 'densityNorm' are mutually exclusive" : NULL),
        conv.massUnit / pow_3(conv.lengthUnit));
    assignParam(param.scaleRadius,         popString(kvmap, keys, "scaleRadius", "rscale",
        type & (PT_DISK | PT_SPHEROID | PT_NUKER | PT_SERSIC | PT_LOG |
        PT_NFW | PT_MIYAMOTONAGAI | PT_LONGMURALI | PT_DEHNEN | PT_FERRERS | PT_PLUMMER |
        PT_ISOCHRONE | PT_PERFECTELLIPSOID | PT_KING)),
        conv.lengthUnit);
    assignParam(param.scaleHeight,         popString(kvmap, keys, "scaleHeight", "scaleRadius2",
        type & (PT_DISK | PT_MIYAMOTONAGAI | PT_LONGMURALI)),
        conv.lengthUnit);
    assignParam(param.barLength,           popString(kvmap, keys, "barLength", "",
        type & PT_LONGMURALI),
        conv.lengthUnit);
    assignParam(param.innerCutoffRadius,   popString(kvmap, keys, "innerCutoffRadius", "",
        type & PT_DISK),
        conv.lengthUnit);
    assignParam(param.outerCutoffRadius,   popString(kvmap, keys, "outerCutoffRadius", "",
        type & (PT_SPHEROID | PT_NUKER)),
        conv.lengthUnit);
    assignParam(param.v0,                  popString(kvmap, keys, "v0", "",
        type & PT_LOG),
        conv.velocityUnit);
    assignParam(param.Omega,               popString(kvmap, keys, "Omega", "",
        type & PT_HARMONIC),
        conv.velocityUnit / conv.lengthUnit);
    assignParam(param.axisRatioY,          popString(kvmap, keys, "axisRatioY", "p",
        type & (PT_SPHEROID | PT_NUKER | PT_SERSIC | PT_LOG | PT_HARMONIC | PT_DEHNEN | PT_FERRERS)));
    assignParam(param.axisRatioZ,          popString(kvmap, keys, "axisRatioZ", "q",
        type & (PT_SPHEROID | PT_NUKER | PT_SERSIC | PT_LOG | PT_HARMONIC | PT_DEHNEN | PT_FERRERS |
        PT_PERFECTELLIPSOID)));
    assignParam(param.alpha,               popString(kvmap, keys, "alpha", "",
        type & (PT_SPHEROID | PT_NUKER)));
    assignParam(param.beta,                popString(kvmap, keys, "beta", "",
        type & (PT_SPHEROID | PT_NUKER)));
    assignParam(param.gamma,               popString(kvmap, keys, "gamma", "",
        type & (PT_SPHEROID | PT_NUKER | PT_DEHNEN)));
    assignParam(param.modulationAmplitude, popString(kvmap, keys, "modulationAmplitude", "",
        type & PT_DISK));
    assignParam(param.cutoffStrength,      popString(kvmap, keys, "cutoffStrength", "xi",
        type & (PT_SPHEROID | PT_NUKER)));
    assignParam(param.sersicIndex,         popString(kvmap, keys, "sersicIndex", "",
        type & (PT_DISK | PT_SERSIC)));
    assignParam(param.W0,                  popString(kvmap, keys, "W0", "",
        type & PT_KING));
    assignParam(param.trunc,               popString(kvmap, keys, "trunc", "",
        type & PT_KING));
    assignParam(param.binary_q,            popString(kvmap, keys, "binary_q", "",
        type & PT_KEPLERBINARY));
    assignParam(param.binary_sma,          popString(kvmap, keys, "binary_sma", "",
        type & PT_KEPLERBINARY),
        conv.lengthUnit);
    assignParam(param.binary_ecc,          popString(kvmap, keys, "binary_ecc", "",
        type & PT_KEPLERBINARY));
    assignParam(param.binary_phase,        popString(kvmap, keys, "binary_phase", "",
        type & PT_KEPLERBINARY));
    // parameters for the potential expansions
    assignParam(param.gridSizeR,           popString(kvmap, keys, "gridSizeR", "",
        param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID | PT_MULTIPOLE | PT_CYLSPLINE)));
    assignParam(param.gridSizez,           popString(kvmap, keys, "gridSizez", "",
        param.potentialType & (PT_DENS_CYLGRID | PT_CYLSPLINE)));
    assignParam(param.nmax,                popString(kvmap, keys, "nmax", "",
        param.potentialType & PT_BASISSET));
    assignParam(param.lmax,                popString(kvmap, keys, "lmax", "",
        param.potentialType & (PT_DISK | PT_SPHEROID | PT_NUKER | PT_SERSIC |
        PT_DENS_SPHHARM | PT_BASISSET | PT_MULTIPOLE)));
    param.mmax = param.lmax;  // update the default value before atttempting to parse the user-provided one
    assignParam(param.mmax,                popString(kvmap, keys, "mmax", "",
        param.potentialType & (PT_DISK | PT_SPHEROID | PT_NUKER | PT_SERSIC |
        PT_DENS_SPHHARM | PT_DENS_CYLGRID | PT_BASISSET | PT_MULTIPOLE | PT_CYLSPLINE)));
    assignParam(param.smoothing,           popString(kvmap, keys, "smoothing", "",
        param.potentialType & PT_MULTIPOLE));
    assignParam(param.rmin,                popString(kvmap, keys, "rmin", "",
        param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID | PT_MULTIPOLE | PT_CYLSPLINE)),
        conv.lengthUnit);
    assignParam(param.rmax,                popString(kvmap, keys, "rmax", "",
        param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID | PT_MULTIPOLE | PT_CYLSPLINE)),
        conv.lengthUnit);
    assignParam(param.zmin,                popString(kvmap, keys, "zmin", "",
        param.potentialType & (PT_DENS_CYLGRID | PT_CYLSPLINE)),
        conv.lengthUnit);
    assignParam(param.zmax,                popString(kvmap, keys, "zmax", "",
        param.potentialType & (PT_DENS_CYLGRID | PT_CYLSPLINE)),
        conv.lengthUnit);
    assignParam(param.eta,                 popString(kvmap, keys, "eta", "",
        param.potentialType & PT_BASISSET));
    assignParam(param.r0,                  popString(kvmap, keys, "r0", "",
        param.potentialType & PT_BASISSET),
        conv.lengthUnit);
    assignParam(param.fixOrder,            popString(kvmap, keys, "fixOrder", "",
        param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID |
        PT_BASISSET | PT_MULTIPOLE | PT_CYLSPLINE)));
    param.symmetryType = getSymmetryTypeByName(popString(kvmap, keys, "symmetry", "",
        (param.potentialType & (PT_DENS_SPHHARM | PT_DENS_CYLGRID |
        PT_BASISSET | PT_MULTIPOLE | PT_CYLSPLINE)) &&
        (param.densityType == PT_UNKNOWN /* no density model is provided */),
        "Parameter 'symmetry' is only allowed for density or potential expansions constructed from "
        "an N-body snapshot or from a user-defined density or potential model"));

    // this parameter is allowed only for the Evolving potential, and will be parsed later
    popString(kvmap, keys, "interpLinear", "linearInterp", param.potentialType == PT_EVOLVING);

    // this parameter may or may not be used; whether it is allowed will be determined later
    param.file = popString(kvmap, keys, "file");

    // this parameter is used only for the Logarithmic potential, but is not user-assignable
    param.lengthUnit = conv.lengthUnit;

    // modifier params (not parsed or unit-converted at this stage; allowed for any model)
    param.center     = popString(kvmap, keys, "center");
    param.orientation= popString(kvmap, keys, "orientation");
    param.rotation   = popString(kvmap, keys, "rotation");
    param.scale      = popString(kvmap, keys, "scale");

    // ensure that no unused (i.e. unknown) parameters remain in the input list
    if(!keys.empty()) {
        std::string unknownKeys;
        for(unsigned int i=0; i<keys.size(); i++)
            unknownKeys += i>0 ? ", " + keys[i] : keys[i];
        throw std::runtime_error(
            "Unknown parameter" + std::string(keys.size()>1 ? "s " : " ") + unknownKeys);
    }

    return param;
}

/// pick up the parameters for DiskDensity or DiskAnsatz from the list of all parameters
DiskParam parseDiskParam(const AllParam& param)
{
    DiskParam dparam;   // copy the relevant parameters into a dedicated structure
    dparam.scaleRadius         = param.scaleRadius;
    dparam.scaleHeight         = param.scaleHeight;
    dparam.innerCutoffRadius   = param.innerCutoffRadius;
    dparam.modulationAmplitude = param.modulationAmplitude;
    if(isFinite(param.sersicIndex))
        dparam.sersicIndex     = param.sersicIndex;  // otherwise keep the default value
    if(isFinite(param.surfaceDensity))
        dparam.surfaceDensity  = param.surfaceDensity;
    else {  // alternative way: specifying the total model mass instead of surface density at R=0
        dparam.surfaceDensity  = 1;
        dparam.surfaceDensity  = param.mass / dparam.mass();
    }
    return dparam;
}

/// pick up the parameters for Spheroid density
SpheroidParam parseSpheroidParam(const AllParam& param)
{
    SpheroidParam sparam;
    sparam.axisRatioY        = param.axisRatioY;
    sparam.axisRatioZ        = param.axisRatioZ;
    sparam.alpha             = param.alpha;
    sparam.beta              = param.beta;
    sparam.gamma             = param.gamma;
    sparam.scaleRadius       = param.scaleRadius;
    sparam.outerCutoffRadius = param.outerCutoffRadius;
    sparam.cutoffStrength    = param.cutoffStrength;
    if(isFinite(param.densityNorm))
        sparam.densityNorm   = param.densityNorm;
    else {  // alternative specification of the total model mass instead of density normalization
        sparam.densityNorm = 1;
        double norm = sparam.mass();
        if(!isFinite(norm))
            throw std::runtime_error("Spheroid model has infinite mass (provide densityNorm instead)");
        sparam.densityNorm = param.mass / norm;
    }
    return sparam;
}

/// pick up the parameters for Nuker density
NukerParam parseNukerParam(const AllParam& param)
{
    NukerParam nparam;
    nparam.axisRatioY        = param.axisRatioY;
    nparam.axisRatioZ        = param.axisRatioZ;
    nparam.alpha             = param.alpha;
    nparam.beta              = param.beta;
    nparam.gamma             = param.gamma;
    nparam.scaleRadius       = param.scaleRadius;
    nparam.outerCutoffRadius = param.outerCutoffRadius;
    nparam.cutoffStrength    = param.cutoffStrength;
    if(isFinite(param.surfaceDensity))
        nparam.surfaceDensity = param.surfaceDensity;
    else {  // alternative: specify the total model mass instead of surface density normalization
        nparam.surfaceDensity = 1;
        nparam.surfaceDensity = param.mass / nparam.mass();
    }
    return nparam;
}

/// pick up the parameters for Sersic density
SersicParam parseSersicParam(const AllParam& param)
{
    SersicParam sparam;
    sparam.axisRatioY  = param.axisRatioY;
    sparam.axisRatioZ  = param.axisRatioZ;
    sparam.scaleRadius = param.scaleRadius;
    if(isFinite(param.sersicIndex))
        sparam.sersicIndex    = param.sersicIndex;  // otherwise keep the default value
    if(isFinite(param.surfaceDensity))
        sparam.surfaceDensity = param.surfaceDensity;
    else {  // alternative way: specifying the total model mass instead of surface density at R=0
        sparam.surfaceDensity = 1;
        sparam.surfaceDensity = param.mass / sparam.mass();
    }
    return sparam;
}

/// pick up the parameters for KeplerBinary potential
KeplerBinaryParams parseKeplerBinaryParam(const AllParam& param)
{
    return KeplerBinaryParams(param.mass, param.binary_q, param.binary_sma, param.binary_ecc,
        param.binary_phase);
}

///@}
/// \name Filesystem utility
//        ------------------
///@{


/** Utility class for temporarily changing the working directory to the one containing
    the given filename, and then returning back to the previous working directory
    when the instance of this class goes out of scope (either in the normal course of events
    or after an exception, either way the current working directory will be restored).
    This class is used when reading/writing INI files containing the potential/density params:
    if they contain references to other files, those files will be treated as residing alongside
    the INI file, even if the latter is specified by a filename with a non-trivial path.
*/
class DirectoryChanger {
    std::string oldcwd;
public:
    /// check if the input filename contains a nontrivial path;
    /// if yes, store the current working directory and move to the one contained in the file path;
    /// throw an exception if it does not exist or could not be moved into.
    DirectoryChanger(const std::string& filenameWithPath)
    {
        std::string::size_type index = filenameWithPath.rfind(DIRECTORY_SEPARATOR);
        if(index == std::string::npos)
            return;
        // otherwise separate the path from the filename
        const size_t bufsize=32768;
        oldcwd.resize(bufsize);
        if(!getcwd(&oldcwd[0], bufsize))
            throw std::runtime_error("Failed to get the current working directory");  // unlikely
        oldcwd.resize(oldcwd.find('\0'));
        std::string newcwd = filenameWithPath.substr(0, index+1);
        if(chdir(newcwd.c_str()) != 0)
            throw std::runtime_error("Failed to change the current working directory to "+newcwd);
        FILTERMSG(utils::VL_VERBOSE, "DirectoryChanger",
            "Temporarily changed the current working directory to "+newcwd);
    }
    ~DirectoryChanger()
    {
        if(!oldcwd.empty()) {
            if(chdir(oldcwd.c_str()) != 0)
                // we can't throw an exception from the destructor, so just print a warning
                FILTERMSG(utils::VL_WARNING, "DirectoryChanger",
                    "Failed to restore working directory to "+oldcwd);
            FILTERMSG(utils::VL_VERBOSE, "DirectoryChanger",
                "Restored the current working directory to "+oldcwd);
        }
    }
};

///@}
/// \name Factory routines for constructing various Density & Potential classes from arrays of strings
//        --------------------------------------------------------------------------------------------
///@{

/// parse the array of spherical-harmonic coefficients
/// for Multipole or BasisSet potentials or DensitySphericalHarmonic
bool parseSphericalHarmonics(std::vector<std::string>& lines, const AllParam& params, size_t numLines,
    /*output*/ std::vector<double>& gridr, std::vector< std::vector<double> > &coefs)
{
    if(lines.size() < numLines+1)
        return false;
    unsigned int numTerms = pow_2(params.lmax+1);
    coefs.assign(numTerms, std::vector<double>(numLines));
    gridr.resize(numLines);  // numLines is either nmax+1 (for BasisSet) or gridSizeR (for others)
    std::vector<std::string> fields;
    for(unsigned int n=0; n<numLines; n++) {
        fields = utils::splitString(lines[n+1], " \t");
        if(fields.size() != numTerms+1)
            return false;
        gridr[n] = utils::toDouble(fields[0]);
        for(unsigned int ind=0; ind<numTerms; ind++)
            coefs[ind][n] = utils::toDouble(fields[ind+1]);
    }
    // remove the parsed lines from the array
    lines.erase(lines.begin(), lines.begin()+numLines+1);
    return true;
}

/// parse azimuthal harmonics from the array of lines:
/// one or more blocks corresponding to each m, where the block is a 2d matrix of values,
/// together with the coordinates in the 1st line and 1st column
bool parseAzimuthalHarmonics(std::vector<std::string>& lines, const AllParam& params,
    /*output*/ std::vector<double>& gridR, std::vector<double>& gridz,
    std::vector< math::Matrix<double> > &coefs)
{
    // total # of harmonics possible, not all of them need to be present in the file
    coefs.resize(params.mmax*2+1);
    gridR.resize(params.gridSizeR);
    std::vector<std::string> fields;
    // parse the array of lines until it is exhausted or an empty line is encountered
    while(! (lines.empty() || lines[0].empty()) ) {
        if(lines.size() < params.gridSizeR+2)
            return false;   // not enough remaining lines
        // 0th line is the index m
        fields = utils::splitString(lines[0], " \t");
        int m = utils::toInt(fields[0]);  // m (azimuthal harmonic index)
        if(fields.size() != 2 || fields[1] != "#m" || m < -(int)params.mmax || m > (int)params.mmax)
            return false;
        // 1st line is the z-grid
        fields = utils::splitString(lines[1], " \t");  // z-values
        if(fields.size() != params.gridSizez+1 || fields[0][0] != '#')
            return false;   // 0th element is comment, remaining are z-values
        gridz.resize(params.gridSizez);
        for(unsigned int iz=0; iz<params.gridSizez; iz++) {
            gridz[iz] = utils::toDouble(fields[iz+1]);
            if(iz>0 && gridz[iz]<=gridz[iz-1])
                return false;  // the values of z must be in increasing order
        }
        coefs[m + params.mmax] = math::Matrix<double>(params.gridSizeR, params.gridSizez, 0);
        // remaining lines are coefs
        for(unsigned int iR=0; iR<params.gridSizeR; iR++) {
            fields = utils::splitString(lines[iR+2], " \t");
            if(fields.size() != params.gridSizez+1)
                return false;  // 0th element is R-value, remaining are coefficients
            gridR[iR] = utils::toDouble(fields[0]);
            if(iR>0 && gridR[iR]<=gridR[iR-1])
                return false;  // the values of R should be in increasing order
            for(unsigned int iz=0; iz<params.gridSizez; iz++)
                coefs[m + params.mmax](iR, iz) = utils::toDouble(fields[iz+1]);
        }
        // remove the parsed lines from the array and proceed to the next block for a different m
        lines.erase(lines.begin(), lines.begin()+params.gridSizeR+2);
    }
    return true;
}

/// parse the array of coefficients and create a BasisSet potential
PtrPotential createBasisSetFromCoefs(std::vector<std::string>& lines, const AllParam& params)
{
    std::vector< std::vector<double> > coefs;
    std::vector< double > indices;
    bool ok = lines.size() >= params.nmax+3 &&
        lines[0] == "#Phi" && lines[1].size()>1 && lines[1][0] == '#';
    if(ok) {
        lines.erase(lines.begin());
        ok &= parseSphericalHarmonics(lines, params, params.nmax+1, /*output*/ indices, coefs);
        for(size_t i=0; i<indices.size(); i++)
            ok &= indices[i] == i;  // should be a sequence of integers from 0 to nmax inclusive
    }
    if(!ok)
        throw std::runtime_error("Error loading BasisSet potential");
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(pow_2(params.converter.velocityUnit), coefs[i]);
    return PtrPotential(new BasisSet(params.eta, params.r0, coefs));
}

/// parse the array of coefficients and create a Multipole potential
PtrPotential createMultipoleFromCoefs(std::vector<std::string>& lines, const AllParam& params)
{
    std::vector< std::vector<double> > coefsPhi, coefsdPhi;
    std::vector< double > gridr;
    bool ok = lines.size() >= 2*params.gridSizeR+5 &&
        lines[0] == "#Phi" && lines[params.gridSizeR+3] == "#dPhi/dr" &&
        lines[1].size()>1 && lines[1][0] == '#' &&
        lines[params.gridSizeR+4].size()>1 && lines[params.gridSizeR+4][0] == '#';
    if(ok) {
        lines.erase(lines.begin());
        ok &= parseSphericalHarmonics(lines, params, params.gridSizeR, /*output*/ gridr, coefsPhi);
        lines.erase(lines.begin(), lines.begin()+2);
        ok &= parseSphericalHarmonics(lines, params, params.gridSizeR, /*output*/ gridr, coefsdPhi);
    }
    if(!ok || coefsPhi.size() != coefsdPhi.size())
        throw std::runtime_error("Error loading Multipole potential");
    math::blas_dmul(params.converter.lengthUnit, gridr);
    for(unsigned int i=0; i<coefsPhi.size(); i++) {
        math::blas_dmul(pow_2(params.converter.velocityUnit), coefsPhi[i]);
        math::blas_dmul(pow_2(params.converter.velocityUnit)/params.converter.lengthUnit, coefsdPhi[i]);
    }
    return PtrPotential(new Multipole(gridr, coefsPhi, coefsdPhi));
}

/// parse the array of coefficients and create a CylSpline potential
PtrPotential createCylSplineFromCoefs(std::vector<std::string>& lines, const AllParam& params)
{
    std::vector<double> gridR, gridz;
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    bool okpot = lines.size()>1 && lines[0] == "#Phi";
    if(okpot) {
        lines.erase(lines.begin());
        okpot &= parseAzimuthalHarmonics(lines, params, /*output*/ gridR, gridz, Phi);
    }
    bool okder = okpot && lines.size()>2 && lines[0].empty() && lines[1] == "#dPhi/dR";
    if(okder) {
        lines.erase(lines.begin(), lines.begin()+2);
        okder &= parseAzimuthalHarmonics(lines, params, /*output*/ gridR, gridz, dPhidR);
    }
    okder = okder && lines.size()>2 && lines[0].empty() && lines[1] == "#dPhi/dz";
    if(okder) {
        lines.erase(lines.begin(), lines.begin()+2);
        okder &= parseAzimuthalHarmonics(lines, params, /*output*/ gridR, gridz, dPhidz);
    }
    okder &= dPhidR.size() == Phi.size() && dPhidz.size() == Phi.size();
    if(!okder) {  // have to live without derivatives...
        dPhidR.clear();
        dPhidz.clear();
    }
    if(!okpot)
        throw std::runtime_error("Error loading CylSpline potential");
    // convert units
    math::blas_dmul(params.converter.lengthUnit, gridR);
    math::blas_dmul(params.converter.lengthUnit, gridz);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(pow_2(params.converter.velocityUnit), Phi[i]);
        if(!okder) continue;  // no derivs
        math::blas_dmul(pow_2(params.converter.velocityUnit)/params.converter.lengthUnit, dPhidR[i]);
        math::blas_dmul(pow_2(params.converter.velocityUnit)/params.converter.lengthUnit, dPhidz[i]);
    }
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

/// parse the array of coefficients and create a SphericalHarmonic density
PtrDensity createDensitySphericalHarmonicFromCoefs(std::vector<std::string>& lines, const AllParam& params)
{
    std::vector< std::vector<double> > coefs;
    std::vector< double > gridr;
    bool ok = lines.size() >= params.gridSizeR+2 &&
        lines[0] == "#rho" && lines[1].size()>1 && lines[1][0] == '#';
    if(ok) {
        lines.erase(lines.begin());
        ok &= parseSphericalHarmonics(lines, params, params.gridSizeR, /*output*/ gridr, coefs);
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading ") + DensitySphericalHarmonic::myName());
    // convert units
    math::blas_dmul(params.converter.lengthUnit, gridr);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(params.converter.massUnit/pow_3(params.converter.lengthUnit), coefs[i]);
    return PtrDensity(new DensitySphericalHarmonic(gridr, coefs));
}

/// parse the array of coefficients and create an AzimuthalHarmonic density
PtrDensity createDensityAzimuthalHarmonicFromCoefs(std::vector<std::string>& lines, const AllParam& params)
{
    std::vector< math::Matrix<double> > coefs;
    std::vector< double > gridR, gridz;
    bool ok = lines.size() >= params.gridSizeR+3 && lines[0] == "#rho";
    if(ok) {
        lines.erase(lines.begin());
        ok &= parseAzimuthalHarmonics(lines, params, /*output*/ gridR, gridz, coefs);
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading ") + DensityAzimuthalHarmonic::myName());
    // convert units
    math::blas_dmul(params.converter.lengthUnit, gridR);
    math::blas_dmul(params.converter.lengthUnit, gridz);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(params.converter.massUnit/pow_3(params.converter.lengthUnit), coefs[i]);
    return PtrDensity(new DensityAzimuthalHarmonic(gridR, gridz, coefs));
}

///@}
/// \name Routines for storing various Density and Potential classes into a stream
//        ------------------------------------------------------------------------
///@{

/// write a block of spherical-harmonic coefs for the Multipole potential or DensitySphericalHarmonic
void writeSphericalHarmonics(std::ostream& strm,
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &coefs)
{
    assert(coefs.size()>0);
    int lmax = static_cast<int>(sqrt(coefs.size() * 1.0)-1);
    strm << "#radius";
    for(int l=0; l<=lmax; l++)
        for(int m=-l; m<=l; m++)
            strm << "\tl="<<l<<",m="<<m;  // header line
    strm << '\n';
    for(unsigned int n=0; n<radii.size(); n++) {
        strm << utils::pp(radii[n], 15);
        for(unsigned int i=0; i<coefs.size(); i++)
            strm << '\t' + (n>=coefs[i].size() || coefs[i][n] == 0 ? "0" :
                utils::pp(coefs[i][n], i==0 ? /*higher precision for l=0 coef*/22 : 15));
        strm << '\n';
    }
}

/// write a block of azimuthal-harmonic coefs for the CylSpline potential or DensityAzimuthalHarmonic
void writeAzimuthalHarmonics(std::ostream& strm,
    const std::vector<double>& gridR,
    const std::vector<double>& gridz,
    const std::vector< math::Matrix<double> >& data)
{
    int mmax = (static_cast<int>(data.size())-1)/2;
    assert(mmax>=0);
    for(int mm=0; mm<static_cast<int>(data.size()); mm++)
        if(data[mm].rows()*data[mm].cols()>0) {
            strm << (-mmax+mm) << "\t#m\n#R(row)\\z(col)";
            for(unsigned int iz=0; iz<gridz.size(); iz++)
                strm << "\t" + utils::pp(gridz[iz], 15);
            strm << "\n";
            for(unsigned int iR=0; iR<gridR.size(); iR++) {
                strm << utils::pp(gridR[iR], 15);
                for(unsigned int iz=0; iz<gridz.size(); iz++)
                    strm << "\t"  + utils::pp(data[mm](iR, iz), 15);
                strm << "\n";
            }
        }
}

void writePotentialBasisSet(std::ostream& strm, const BasisSet& pot,
    const units::ExternalUnits& converter)
{
    double eta, r0;
    std::vector< std::vector<double> > coefs;
    pot.getCoefs(eta, r0, coefs);
    assert(coefs.size() > 0 && coefs[0].size() > 0);
    // convert units
    r0 /= converter.lengthUnit;
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(1/pow_2(converter.velocityUnit), coefs[i]);
    strm << "nmax=" << (coefs[0].size()-1) << "\n";
    strm << "lmax=" << static_cast<int>(sqrt(coefs.size()*1.0)-1) << "\n";
    strm << "eta="  << utils::toString(eta,15) << "\n";
    strm << "r0="   << utils::toString(r0, 15) << "\n";
    strm << "symmetry=" << getSymmetryNameByType(pot.symmetry()) << "\n";
    strm << "Coefficients\n#Phi\n";
    writeSphericalHarmonics(strm, math::createUniformGrid(coefs[0].size(), 0, coefs[0].size()-1), coefs);
}

void writePotentialMultipole(std::ostream& strm, const Multipole& pot,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridr;
    std::vector< std::vector<double> > Phi, dPhi;
    pot.getCoefs(gridr, Phi, dPhi);
    assert(Phi.size() > 0 && Phi[0].size() == gridr.size() && dPhi[0].size() == Phi[0].size());
    // convert units
    math::blas_dmul(1/converter.lengthUnit, gridr);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(1/pow_2(converter.velocityUnit), Phi[i]);
        math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhi[i]);
    }
    strm << "gridSizeR=" << gridr.size() << "\n";
    strm << "lmax=" << static_cast<int>(sqrt(Phi.size()*1.0)-1) << "\n";
    strm << "symmetry=" << getSymmetryNameByType(pot.symmetry()) << "\n";
    strm << "Coefficients\n#Phi\n";
    writeSphericalHarmonics(strm, gridr, Phi);
    strm << "\n#dPhi/dr\n";
    writeSphericalHarmonics(strm, gridr, dPhi);
}

void writePotentialCylSpline(std::ostream& strm, const CylSpline& pot,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridR, gridz;
    std::vector<math::Matrix<double> > Phi, dPhidR, dPhidz;
    pot.getCoefs(gridR, gridz, Phi, dPhidR, dPhidz);
    strm << "gridSizeR=" << gridR.size() << "\n";
    strm << "gridSizez=" << gridz.size() << "\n";
    strm << "mmax=" << (Phi.size()/2) << "\n";
    strm << "symmetry=" << getSymmetryNameByType(pot.symmetry()) << "\n";
    strm << "Coefficients\n#Phi\n";
    // convert units
    math::blas_dmul(1/converter.lengthUnit, gridR);
    math::blas_dmul(1/converter.lengthUnit, gridz);
    for(unsigned int i=0; i<Phi.size(); i++)
        math::blas_dmul(1/pow_2(converter.velocityUnit), Phi[i]);
    writeAzimuthalHarmonics(strm, gridR, gridz, Phi);
    // write arrays of derivatives if they were provided
    // (only when the potential uses quintic interpolaiton internally)
    if(dPhidR.size()>0 && dPhidz.size()>0) {
        assert(dPhidR.size() == Phi.size() && dPhidz.size() == Phi.size());
        for(unsigned int i=0; i<dPhidR.size(); i++) {
            math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhidR[i]);
            math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhidz[i]);
        }
        strm << "\n#dPhi/dR\n";
        writeAzimuthalHarmonics(strm, gridR, gridz, dPhidR);
        strm << "\n#dPhi/dz\n";
        writeAzimuthalHarmonics(strm, gridR, gridz, dPhidz);
    }
}

void writeDensitySphericalHarmonic(std::ostream& strm, const DensitySphericalHarmonic& density,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridr;
    std::vector<std::vector<double> > coefs;
    density.getCoefs(gridr, coefs);
    // convert units
    math::blas_dmul(1/converter.lengthUnit, gridr);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(1/converter.massUnit*pow_3(converter.lengthUnit), coefs[i]);
    strm << "gridSizeR=" << gridr.size() << "\n";
    strm << "lmax=" << static_cast<int>(sqrt(coefs.size()*1.0)-1) << "\n";
    strm << "symmetry=" << getSymmetryNameByType(density.symmetry()) << "\n";
    strm << "Coefficients\n#rho\n";
    writeSphericalHarmonics(strm, gridr, coefs);
}

void writeDensityAzimuthalHarmonic(std::ostream& strm, const DensityAzimuthalHarmonic& density,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridR, gridz;
    std::vector<math::Matrix<double> > coefs;
    density.getCoefs(gridR, gridz, coefs);
    // convert units
    math::blas_dmul(1/converter.lengthUnit, gridR);
    math::blas_dmul(1/converter.lengthUnit, gridz);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(1/converter.massUnit*pow_3(converter.lengthUnit), coefs[i]);
    strm << "gridSizeR=" << gridR.size() << "\n";
    strm << "gridSizez=" << gridz.size() << "\n";
    strm << "mmax=" << (coefs.size()/2) << "\n";
    strm << "symmetry=" << getSymmetryNameByType(density.symmetry()) << "\n";
    strm << "Coefficients\n#rho\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, coefs);
}

/// write data (expansion coefs or components) for a single or composite density or potential to a stream.
/// this implementation is fairly incomplete - it flattens out any hierarchy of composite objects,
/// ignores any modifiers, and is not able to save parameters of most elementary objects (this may be
/// implemented in the future); it is primarily intended to store density/potential expansion coefficients
void writeAnyDensityOrPotential(std::ostream& strm, const BaseDensity* dens,
    const units::ExternalUnits& converter, int& counter)
{
    // check if this is a multicomponent or modified density
    const BaseComposite<BaseDensity>* cd = dynamic_cast<const BaseComposite<BaseDensity>* >(dens);
    if(cd) {
        for(unsigned int i=0; i<cd->size(); i++) {
            writeAnyDensityOrPotential(strm, cd->component(i).get(), converter, counter);
            if(i+1<cd->size())
                strm << '\n';
        }
        return;
    }

    // check if this is a multicomponent or modified potential
    const BaseComposite<BasePotential>* cp = dynamic_cast<const BaseComposite<BasePotential>* >(dens);
    if(cp) {
        for(unsigned int i=0; i<cp->size(); i++) {
            writeAnyDensityOrPotential(strm, cp->component(i).get(), converter, counter);
            if(i+1<cp->size())
                strm << '\n';
        }
        return;
    }

    // otherwise this is an elementary potential or density, so write a section header
    if(dynamic_cast<const BasePotential*>(dens))
        strm << "[Potential";
    else
        strm << "[Density";
    if(counter>0)
        strm << counter;
    strm << "]\n";
    strm << "type=" << dens->name() << '\n';
    counter++;

    const BasisSet* bs = dynamic_cast<const BasisSet*>(dens);
    if(bs) {
        writePotentialBasisSet(strm, *bs, converter);
        return;
    }
    const Multipole* mu = dynamic_cast<const Multipole*>(dens);
    if(mu) {
        writePotentialMultipole(strm, *mu, converter);
        return;
    }
    const CylSpline* cy = dynamic_cast<const CylSpline*>(dens);
    if(cy) {
        writePotentialCylSpline(strm, *cy, converter);
        return;
    }
    const DensitySphericalHarmonic* sh = dynamic_cast<const DensitySphericalHarmonic*>(dens);
    if(sh) {
        writeDensitySphericalHarmonic(strm, *sh, converter);
        return;
    }
    const DensityAzimuthalHarmonic* ah = dynamic_cast<const DensityAzimuthalHarmonic*>(dens);
    if(ah) {
        writeDensityAzimuthalHarmonic(strm, *ah, converter);
        return;
    }

    // otherwise don't know how to store this potential
    strm << "#other parameters are not stored\n";
}

///@}
/// \name Routines for auxiliary density/potential classes
//        ------------------------------------------------
///@{

inline bool isPairOfBrackets(char c1, char c2) { return (c1=='[' && c2==']') || (c1=='(' && c2==')'); }

inline std::string errorTimeDependentArray(int K, const std::string& str)
{
    return "readTimeDependentArray<" + utils::toString(K) + ">: "
    "input string \"" + str + "\" must be one of the following: "
    "(a) a sequence of " + utils::toString(K) + " comma- or space-separated values, "
    "(b) a 2d array with " + utils::toString(K+1) + " or " + utils::toString(K*2+1) + " columns, "
    "(c) the name of a file with this number of columns "
    "(timestamps, values and optionally their time derivatives)";
}

/** parse a single string with K values or read a file with a time-dependent K-dimensional vector:
    each line contains the timestamp and K or 2*K values, interpreted as K components of the vector
    (applying the unit conversion) and optionally their time derivatives at the given moment of time.
    The array read from a file is returned as K cubic splines (natural or Hermite),
    whereas in case of a single input string, these splines are "constant interpolators".
    \tparam K  is the dimension of the vector quantity (1,2,3...)
*/
template<int K>
static void readTimeDependentArray(
    const std::string& str, /*units:*/ double timeUnit, double valueUnit,
    /*output*/ math::CubicSpline spl[K])
{
    std::vector<std::string> fields = utils::splitString(str, ",; \t");
    if(fields.empty())
        throw std::runtime_error("readTimeDependentArray<" + utils::toString(K) + ">: "
            "empty input string");

    std::vector<double> time, val[K], der[K];

    // try to interpret the string as an array of K numbers
    // separated by spaces or commas and optionally enclosed in round or square brackets
    if(fields.size() == K) {
        try {
            // input string might be enclosed in a single pair of round or square brackets - remove them
            if(isPairOfBrackets(fields.front()[0], fields.back()[fields.back().size()-1])) {
                fields.front() = fields.front().substr(1);
                fields.back () = fields.back ().substr(0, fields.back().size()-1);
            }
            for(int k=0; k<K; k++) {
                spl[k] = math::CubicSpline(std::vector<double>(1, 0.),
                    std::vector<double>(1, utils::toDouble(fields[k]) * valueUnit));
            }
            return;
        }
        // if the conversion failed, continue and try to interpret the input string as a file name
        catch(std::exception&) {}
    }

    // otherwise the string might be a serialized 2d array like [[1,2,3],[4,5,6]]
    // enclosed in round or square brackets
    if( (fields.size() % (K+1) == 0 || fields.size() % (2*K+1) == 0) &&
        isPairOfBrackets(fields.front()[0], fields.back()[fields.back().size()-1]) )
    {
        // remove the outermost pair of brackets
        fields.front() = fields.front().substr(1);
        fields.back () = fields.back ().substr(0, fields.back().size()-1);
        // check that each group of (K+1) or (K*2+1) items is enclosed in brackets
        bool Kplus1 = fields.size() % (K+1) == 0, K2plus1 = fields.size() % (2*K+1) == 0;
        for(size_t l=0; l<fields.size() / (K+1); l++)
            Kplus1  &= isPairOfBrackets(fields[l * (K+1)][0],
                fields[(l+1) * (K+1) - 1][fields[(l+1) * (K+1) - 1].size() - 1]);
        for(size_t l=0; l<fields.size() / (K*2+1); l++)
            K2plus1 &= isPairOfBrackets(fields[l * (K*2+1)][0],
                fields[(l+1) * (K*2+1) - 1][fields[(l+1) * (K*2+1) - 1].size() - 1]);
        if(Kplus1) {
            for(size_t l=0; l<fields.size() / (K+1); l++) {
                // strip the opening bracket in the first column (time)
                time.push_back(utils::toDouble(fields[l * (K+1)].substr(1)) * timeUnit);
                // don't bother stripping the closing bracket, as toDouble() will stop parsing
                // the string once it encounters a non-numerical character
                for(int k=0; k<K; k++)
                    val[k].push_back(utils::toDouble(fields[l * (K+1) + k+1]) * valueUnit);
            }
            // create natural cubic splines from just the values (and regularize)
            for(int k=0; k<K; k++)
                spl[k] = math::CubicSpline(time, val[k], true);
            return;
        }
        if(K2plus1) {
            for(size_t l=0; l<fields.size() / (K*2+1); l++) {
                time.push_back(utils::toDouble(fields[l * (K*2+1)].substr(1)) * timeUnit);
                for(int k=0; k<K; k++) {
                    val[k].push_back(utils::toDouble(fields[l * (K*2+1) + k+1]) * valueUnit);
                    der[k].push_back(utils::toDouble(fields[l * (K*2+1) + k+1 + K]) *
                        valueUnit / timeUnit);
                }
            }
            // create Hermite splines from values and derivatives at each moment of time
            for(int k=0; k<K; k++)
                spl[k] = math::CubicSpline(time, val[k], der[k]);
            return;
        }
        throw std::runtime_error(errorTimeDependentArray(K, str));
    }

    // otherwise the input string is interpreted as the name of the file with the said array
    std::ifstream strm(str.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error(errorTimeDependentArray(K, str));
    std::string buffer;
    while(std::getline(strm, buffer) && !strm.eof()) {
        if(!buffer.empty() && utils::isComment(buffer[0]))  // commented line
            continue;
        fields = utils::splitString(buffer, ";, \t");
        size_t numFields = fields.size();
        if(numFields < K+1 ||
            !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
            continue;
        time.push_back(utils::toDouble(fields[0]) * timeUnit);
        for(int k=0; k<K; k++) {
            val[k].push_back(utils::toDouble(fields[k+1]) * valueUnit);
            if(numFields >= 2*K+1)
                der[k].push_back(utils::toDouble(fields[k+K+1]) * valueUnit / timeUnit);
        }
    }
    if(val[0].size() == 0)
        throw std::runtime_error("readTimeDependentArray<" + utils::toString(K) + ">: "
            "no valid entries in file \"" + str + "\"");
    if(der[0].size() == val[0].size()) {
        // create Hermite splines from values and derivatives at each moment of time
        for(int k=0; k<K; k++)
            spl[k] = math::CubicSpline(time, val[k], der[k]);
    } else if(der[0].empty()) {
        // create natural cubic splines from just the values (and regularize)
        for(int k=0; k<K; k++)
            spl[k] = math::CubicSpline(time, val[k], true);
    } else
        throw std::runtime_error("readTimeDependentArray<" + utils::toString(K) + ">: "
            "file \"" + str + "\" should contain either " + utils::toString(K+1) + " or " +
            utils::toString(2*K+1) + " columns");
}

///@}
/// \name Factory routines for creating instances of Density and Potential classes
//        ------------------------------------------------------------------------
///@{

// parse modifier params (in a particular order), and for each non-trivial one,
// replace the original density/potential object with a corresponding modifier wrapping the object
template<typename BaseDensityOrPotential>
void applyModifiers(
    shared_ptr<const BaseDensityOrPotential>& obj, const ModifierParams& param)
{
    // when applying more than one modifier, order is important
    if(!param.scale.empty()) {
        math::CubicSpline scale[2];
        readTimeDependentArray<2>(param.scale,
            /*units*/ param.converter.timeUnit, 1 /*dimensionless*/, /*output*/ scale);
        obj.reset(new Scaled<BaseDensityOrPotential>(obj, scale[0], scale[1]));
    }
    if(!param.rotation.empty()) {
        math::CubicSpline rotation;
        readTimeDependentArray<1>(param.rotation,
            /*units*/ param.converter.timeUnit, 1 /*dimensionless*/, /*output*/ &rotation);
        obj.reset(new Rotating<BaseDensityOrPotential>(obj, rotation));
    }
    if(!param.orientation.empty()) {
        // string should contain three Euler angles
        // (space and/or comma-separated, possibly surrounded by square or round brackets)
        std::vector<std::string> fields = utils::splitString(
            isPairOfBrackets(param.orientation[0], param.orientation[param.orientation.size() - 1]) ?
            param.orientation.substr(1, param.orientation.size()-2) :  // strip brackets
            param.orientation, ",; \t");
        if(fields.size() != 3)
            throw std::invalid_argument("'orientation' must specify three Euler angles");
        obj.reset(new Tilted<BaseDensityOrPotential>(obj,
            utils::toDouble(fields[0]), utils::toDouble(fields[1]), utils::toDouble(fields[2])));
    }
    if(!param.center.empty()) {
        // string could contain either three components of the fixed offset vector,
        // or the name of a file with time-dependent trajectory
        math::CubicSpline center[3];
        readTimeDependentArray<3>(param.center,
            /*units*/ param.converter.timeUnit, param.converter.lengthUnit, /*output*/ center);
        obj.reset(new Shifted<BaseDensityOrPotential>(obj, center[0], center[1], center[2]));
    }
}

/// create potential expansion of a given type from a set of point masses
PtrPotential createPotentialExpansionFromParticles(const AllParam& param,
    const particles::ParticleArray<coord::PosCyl>& particles)
{
    switch(param.potentialType) {
    case PT_BASISSET:
        return BasisSet::create(particles, param.symmetryType, param.lmax, param.mmax,
            param.nmax, param.eta, param.r0);
    case PT_MULTIPOLE:
        return Multipole::create(particles, param.symmetryType, param.lmax, param.mmax,
            param.gridSizeR, param.rmin, param.rmax, param.smoothing);
    case PT_CYLSPLINE:
        return CylSpline::create(particles, param.symmetryType, param.mmax,
            param.gridSizeR, param.rmin, param.rmax,
            param.gridSizez, param.zmin, param.zmax);
    default:
        throw std::invalid_argument("Unknown potential expansion type");
    }
}

/** Create an instance of analytic potential model according to the parameters passed. 
    \param[in] param  specifies the potential parameters
    \return    the instance of potential
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular potential model
*/
PtrPotential createAnalyticPotential(const AllParam& param)
{
    switch(param.potentialType) {
    case PT_LOG:
        return PtrPotential(new Logarithmic(
            param.v0, param.scaleRadius, param.axisRatioY, param.axisRatioZ, param.lengthUnit));
    case PT_HARMONIC:
        return PtrPotential(new Harmonic(param.Omega, param.axisRatioY, param.axisRatioZ));
    case PT_KEPLERBINARY:
        return PtrPotential(new KeplerBinary(parseKeplerBinaryParam(param)));
    case PT_MIYAMOTONAGAI:
        return PtrPotential(new MiyamotoNagai(param.mass, param.scaleRadius, param.scaleHeight));
    case PT_DEHNEN:
        return PtrPotential(new Dehnen(
            param.mass, param.scaleRadius, param.gamma, param.axisRatioY, param.axisRatioZ));
    case PT_FERRERS:
        return PtrPotential(new Ferrers(
            param.mass, param.scaleRadius, param.axisRatioY, param.axisRatioZ)); 
    case PT_PLUMMER:
        if(param.axisRatioY==1 && param.axisRatioZ==1)
            return PtrPotential(new Plummer(param.mass, param.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Plummer is not supported");
    case PT_ISOCHRONE:
        if(param.axisRatioY==1 && param.axisRatioZ==1)
            return PtrPotential(new Isochrone(param.mass, param.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Isochrone is not supported");
    case PT_NFW:
        if(param.axisRatioY==1 && param.axisRatioZ==1)
            return PtrPotential(new NFW(param.mass, param.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Navarro-Frenk-White is not supported");
    case PT_PERFECTELLIPSOID:
        if(param.axisRatioY==1)
            return PtrPotential(new PerfectEllipsoid(
                param.mass, param.scaleRadius, param.scaleRadius*param.axisRatioZ)); 
        else
            throw std::invalid_argument("Non-axisymmetric Perfect Ellipsoid is not supported");
    case PT_KING:
        return createKingPotential(param.mass, param.scaleRadius, param.W0, param.trunc);
    case PT_LONGMURALI:
        return PtrPotential(new LongMurali(
            param.mass, param.scaleRadius, param.scaleHeight, param.barLength));
    case PT_UNKNOWN:
        throw std::invalid_argument("Potential type not specified");
    default:
        throw std::invalid_argument("Invalid potential type");
    }
}

/** Create an instance of analytic density model according to the parameters passed. 
    \param[in] param  specifies the density parameters (type=... determines the type of model)
    \return    the instance of density
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular density model
*/
PtrDensity createAnalyticDensity(const AllParam& param)
{
    switch(param.potentialType) {
    case PT_DISK:
        return PtrDensity(new DiskDensity(parseDiskParam(param)));
    case PT_SPHEROID:
        return PtrDensity(new SpheroidDensity(parseSpheroidParam(param)));
    case PT_NUKER:
        return PtrDensity(new SpheroidDensity(parseNukerParam(param)));
    case PT_SERSIC:
        return PtrDensity(new SpheroidDensity(parseSersicParam(param)));
    case PT_KING:
        return createKingDensity(param.mass, param.scaleRadius, param.W0, param.trunc); 
    default:
        return createAnalyticPotential(param);
    }
}

/** Create an instance of potential expansion class according to the parameters passed in param,
    for the provided source density or potential */
template<typename BaseDensityOrPotential>
PtrPotential createPotentialExpansionFromSource(
    const AllParam& param, const shared_ptr<const BaseDensityOrPotential>& source)
{
    switch(param.potentialType) {
    case PT_BASISSET:
        return BasisSet::create(*source, param.symmetryType, param.lmax, param.mmax,
            param.nmax, param.eta, param.r0, param.fixOrder);
    case PT_MULTIPOLE:
        return Multipole::create(*source, param.symmetryType, param.lmax, param.mmax,
            param.gridSizeR, param.rmin, param.rmax, param.fixOrder);
    case PT_CYLSPLINE:
        return CylSpline::create(*source, param.symmetryType, param.mmax,
            param.gridSizeR, param.rmin, param.rmax,
            param.gridSizez, param.zmin, param.zmax, param.fixOrder);
    default: throw std::invalid_argument("Unknown potential expansion type");
    }
}

/** General routine for creating a potential expansion from the provided INI parameters */
PtrPotential createPotentialExpansion(const AllParam& param, const utils::KeyValueMap& kvmap)
{
    assert(param.potentialType == PT_BASISSET  ||
           param.potentialType == PT_MULTIPOLE ||
           param.potentialType == PT_CYLSPLINE);

    // dump the content of the INI section into an array of strings, and search for Coefficients
    std::vector<std::string> lines = kvmap.dumpLines();
    ptrdiff_t startLine = -1;
    for(size_t i=0; i<lines.size(); i++) {
        if(lines[i] == "Coefficients") {
            startLine = i+1;
            break;
        }
    }

    // three mutually exclusive alternatives
    bool haveCoefs  = startLine>=0;
    bool haveFile   = !param.file.empty();
    bool haveSource = param.densityType != PT_UNKNOWN;

    // option 1: coefficients are provided in the INI file
    if(haveCoefs && !haveFile && !haveSource) {
        lines.erase(lines.begin(), lines.begin()+startLine);
        switch(param.potentialType) {
            case PT_BASISSET:  return createBasisSetFromCoefs (lines, param);
            case PT_MULTIPOLE: return createMultipoleFromCoefs(lines, param);
            default:           return createCylSplineFromCoefs(lines, param);
        }
    }

    // option 2: N-body snapshot
    if(haveFile && !haveCoefs && !haveSource) {
        if(!utils::fileExists(param.file))
            throw std::runtime_error("File " + param.file + " does not exist");
        const particles::ParticleArrayCar particles = particles::readSnapshot(param.file, param.converter);
        if(particles.size()==0)
            throw std::runtime_error("Error loading N-body snapshot from " + param.file);
        return createPotentialExpansionFromParticles(param, particles);
    }

    // option 3: analytic density or potential model
    if(haveSource && !haveFile && !haveCoefs) {
        // create a temporary density or potential model to serve as the source for potential expansion
        AllParam srcpar(param);
        srcpar.potentialType = param.densityType;
        if( param.densityType == PT_DEHNEN ||
            param.densityType == PT_FERRERS ||
            param.densityType == PT_MIYAMOTONAGAI )
        {   // use an analytic potential as the source
            return createPotentialExpansionFromSource(param, createAnalyticPotential(srcpar));
        }
        else
        {   // otherwise use analytic density as the source
            return createPotentialExpansionFromSource(param, createAnalyticDensity(srcpar));
        }
    }

    throw std::invalid_argument( (
        param.potentialType == PT_BASISSET  ? BasisSet::myName() :
        param.potentialType == PT_MULTIPOLE ? Multipole::myName() : CylSpline::myName()) +
        " can be constructed in one of three possible ways: "
        "by providing an N-body snapshot in file=..., or a source density/potential model "
        "in density=..., or a table of coefficients (when loading from a file)");
}

/// create a time-dependent list of potentials
static PtrPotential createEvolvingPotential(
    const utils::KeyValueMap& kvmap, const units::ExternalUnits& converter)
{
    // dump the content of the INI section into an array of strings, and search for Timestamps
    std::vector<std::string> lines = kvmap.dumpLines();
    ptrdiff_t startLine = -1;
    for(size_t i=0; i<lines.size(); i++) {
        if(lines[i] == "Timestamps") {
            startLine = i+1;
            break;
        }
    }
    if(startLine < 0)
        throw std::runtime_error(
            "Evolving potential needs a list of timestamps and filenames after the line 'Timestamps'");

    bool interpLinear = kvmap.getBoolAlt("interpLinear", "linearInterp", false);
    std::vector<std::string> fields;
    std::vector<double> times;
    std::vector<PtrPotential> potentials;
    // attempt to parse the remaining lines in this INI section
    // as time stamps and names of corresponding potential files
    for(size_t index=startLine; index<lines.size(); index++) {
        if(lines[index].empty() || utils::isComment(lines[index][0]))  // commented line
            continue;
        fields = utils::splitString(lines[index], " \t");
        size_t numFields = fields.size();
        if(numFields < 2 ||
            !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
            continue;
        times.push_back(utils::toDouble(fields[0]) * converter.timeUnit);
        try {
            potentials.push_back(potential::readPotential(fields[1], converter));
        }
        catch(std::exception& e) {
            throw std::runtime_error("Error reading the potential from "+fields[1]+": "+e.what());
        }
    }
    return PtrPotential(new Evolving(times, potentials, interpLinear));
}

/** create and instance of UniformAcceleration potential from the time-dependent values in a file */
PtrPotential readUniformAcceleration(const std::string& filename, const units::ExternalUnits& converter)
{
    if(filename.empty())
        throw std::invalid_argument("Need to provide a file name for UniformAcceleration");
    math::CubicSpline acc[3];
    readTimeDependentArray<3>(filename,
        /*units*/ converter.timeUnit, converter.velocityUnit / converter.timeUnit /*acceleration*/,
        /*output*/ acc);
    return PtrPotential(new UniformAcceleration(acc[0], acc[1], acc[2]));
}

// a collection of would-be potential components with a common set of modifiers
struct Bunch {
    // all potential components (DiskAnsatz or any other potential class)
    std::vector<PtrPotential> componentsPot;
    // all density components that will contribute to a single additional Multipole potential
    std::vector<PtrDensity> componentsDens;
    // parameters of modifiers
    const ModifierParams modifiers;
    // order of polar and azimuthal expansion for galpot components (if provided by user)
    int galpot_lmax, galpot_mmax;
    // constructor
    Bunch(const ModifierParams& _modifiers) : modifiers(_modifiers), galpot_lmax(-1), galpot_mmax(-1) {}
#if __cplusplus < 201103L
    // a default assignment operator is not created pre-C++11, so needs to be provided explicitly,
    // but it appears to be unused anyway
    Bunch& operator=(const Bunch& src) {
        assert(!"should not be called");
        return *this;
    }
#endif
};

}  // end internal namespace

//---- public routines ----//

//---- construct a density profile from a cumulative mass profile ----//

std::vector<double> densityFromCumulativeMass(
    const std::vector<double>& gridr, const std::vector<double>& gridm)
{
    unsigned int size = gridr.size();
    if(size<3 || gridm.size()!=size)
        throw std::invalid_argument("densityFromCumulativeMass: invalid array sizes");
    // check monotonicity and convert to log-scaled radial grid
    std::vector<double> gridlogr(size), gridlogm(size), gridrho(size);
    for(unsigned int i=0; i<size; i++) {
        if(!(gridr[i] > 0 && gridm[i] > 0))
            throw std::invalid_argument("densityFromCumulativeMass: negative input values");
        if(i>0 && (gridr[i] <= gridr[i-1] || gridm[i] <= gridm[i-1]))
            throw std::invalid_argument("densityFromCumulativeMass: arrays are not monotonic");
        gridlogr[i] = log(gridr[i]);
    }
    // determine if the cumulative mass approaches a finite limit at large radii,
    // that is, M = Minf - A * r^B  with A>0, B<0
    double A, B, Minf;
    math::findAsymptote(gridr[size-1], gridr[size-2], gridr[size-3],
        gridm[size-1], gridm[size-2], gridm[size-3], /*output*/ A, B, Minf);
    if(B<0 && A<0) {  // viable extrapolation
        FILTERMSG(utils::VL_DEBUG, "densityFromCumulativeMass",
            "Extrapolated total mass=" + utils::toString(Minf) +
            ", rho(r)~r^" + utils::toString(B-3) + " at large radii");
    } else
        Minf = INFINITY;  // no finite limit detected
    // scaled mass to interpolate:  log[ M / (1 - M/Minf) ] as a function of log(r),
    // which has a linear asymptotic behaviour with slope -B as log(r) --> infinity;
    // if Minf = infinity, this additional term has no effect
    for(unsigned int i=0; i<size; i++)
        gridlogm[i] = log(gridm[i] / (1 - gridm[i] / Minf));
    math::CubicSpline spl(gridlogr, gridlogm, true /*enforce monotonicity*/);
    if(spl.extrema().size() > 2)  // should be only two, at both endpoints
        throw std::runtime_error("densityFromCumulativeMass: interpolated mass is not monotonic");
    // compute the density at each point of the input radial grid
    for(unsigned int i=0; i<size; i++) {
        double val, der;
        spl.evalDeriv(gridlogr[i], &val, &der);
        val = exp(val);
        gridrho[i] = der * val / (4*M_PI * pow_3(gridr[i]) * pow_2(1 + val / Minf));
        if(gridrho[i] <= 0)   // shouldn't occur if the spline is (strictly) monotonic
            throw std::runtime_error("densityFromCumulativeMass: interpolated density is non-positive");
    }
    return gridrho;
}

//------ read a cumulative mass profile from a file ------//

math::LogLogSpline readMassProfile(const std::string& filename)
{
    std::ifstream strm(filename.c_str());
    if(!strm)
        throw std::runtime_error("readMassProfile: can't read input file " + filename);
    std::vector<double> radius, mass;
    const std::string validDigits = "0123456789.-+";
    while(strm) {
        std::string str;
        std::getline(strm, str);
        std::vector<std::string> elems = utils::splitString(str, " \t,;");
        if(elems.size() < 2 || validDigits.find(elems[0][0]) == std::string::npos)
            continue;
        double r = utils::toDouble(elems[0]),  m = utils::toDouble(elems[1]);
        if(r<0)
            throw std::runtime_error("readMassProfile: radii should be positive");
        if(r==0 && m!=0)
            throw std::runtime_error("readMassProfile: M(r=0) should be zero");
        if(r>0) {
            radius.push_back(r);
            mass.push_back(m);
        }
    }
    return math::LogLogSpline(radius, densityFromCumulativeMass(radius, mass));
}

// create elementary density (analytic or from expansion coefs), including any modifiers
PtrDensity createDensity(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    AllParam param = parseParam(kvmap, converter);
    // if 'type=...' is not provided but 'density=...' is given, use that value
    if(!kvmap.contains("type") && kvmap.contains("density"))
        param.potentialType = param.densityType;

    if(!param.file.empty()) {
        if(param.potentialType != PT_UNKNOWN)
            throw std::invalid_argument("createDensity: cannot provide both type and file");
        return readDensity(param.file, converter);
    }

    PtrDensity result;
    // check if this is one of the two density expansions with coefficients provided in the kvmap
    if(param.potentialType == PT_DENS_SPHHARM || param.potentialType == PT_DENS_CYLGRID) {
        // dump the content of the INI section into an array of strings, and search for Coefficients
        std::vector<std::string> lines = kvmap.dumpLines();
        ptrdiff_t startLine = -1;
        for(size_t i=0; i<lines.size(); i++) {
            if(lines[i] == "Coefficients") {
                startLine = i+1;
                break;
            }
        }
        if(startLine < 0)
            throw std::invalid_argument("create" + std::string(
                param.potentialType == PT_DENS_SPHHARM ? DensitySphericalHarmonic::myName() :
                DensityAzimuthalHarmonic::myName()) + ": no coefficients provided");
        lines.erase(lines.begin(), lines.begin()+startLine);
        result = param.potentialType == PT_DENS_SPHHARM ?
            createDensitySphericalHarmonicFromCoefs(lines, param) :
            createDensityAzimuthalHarmonicFromCoefs(lines, param);
    } else
        // otherwise it must be one of the analytic density profiles
        result = createAnalyticDensity(param);

    applyModifiers(result, param);
    return result;
}

// create a density expansion approximating the input density model,
// and/or add a modifier on top of the input density
PtrDensity createDensity(
    const utils::KeyValueMap& kvmap,
    const PtrDensity& dens,
    const units::ExternalUnits& converter)
{
    AllParam param = parseParam(kvmap, converter);
    PtrDensity result;
    if(param.potentialType == PT_DENS_SPHHARM) {
        result = DensitySphericalHarmonic::create(*dens,
            param.symmetryType, param.lmax, param.mmax,
            param.gridSizeR, param.rmin, param.rmax,
            param.fixOrder);
    } else if(param.potentialType == PT_DENS_CYLGRID) {
        result = DensityAzimuthalHarmonic::create(*dens,
            param.symmetryType, param.mmax,
            param.gridSizeR, param.rmin, param.rmax,
            param.gridSizez, param.zmin, param.zmax,
            param.fixOrder);
    } else if(param.potentialType == PT_UNKNOWN)
        result = dens;   // if not creating an expansion, use the input density and add modifiers
    else
        throw std::invalid_argument(
            "createDensity: type should be either empty or specify one of the density expansion classes");
    // add modifiers on top of either input density or its expansion
    applyModifiers(result, param);
    return result;
}

// universal routine for creating a potential from several components
PtrPotential createPotential(
    const std::vector<utils::KeyValueMap>& kvmap,
    const units::ExternalUnits& converter)
{
    if(kvmap.size() == 0)
        throw std::runtime_error("Empty list of potential components");

    // the procedure would have been straightforward (iterate over the elements of the input array
    // of parameters and create an instance of potential for each parameter group),
    // if not for two complicating factors:
    // 1) Elements of the GalPot scheme (disk/spheroid/nuker/sersic density profiles) are considered
    // together and create Ndisk+1 potential components:
    // each Disk group is represented by one potential component (DiskAnsatz) and two density
    // components ("residuals") that are added to the list of components of a CompositeDensity;
    // all Spheroid, Nuker and Sersic density profiles are also added to this CompositeDensity;
    // and in the end a single Multipole potential is constructed from this density collection.
    // 2) Any parameter group may have one or more modifiers (center, rotation, orientation, scale).
    // As a consequence of the two circumstances, the elements of GalPot scheme sharing a common
    // list of modifiers are grouped into a single bunch of DiskAnsatz+Multipole combinations,
    // but there may be more than one such bunch if the modifiers vary between parameter groups.

    std::vector<Bunch> bunches; // list of bunches sharing a common list of modifiers

    // first loop over all parameter groups
    for(unsigned int i=0; i<kvmap.size(); i++) {
        const AllParam param = parseParam(kvmap[i], converter);

        // find the "bunch" with the same modifier params, or create a new one
        unsigned int indexBunch = 0;
        while(indexBunch < bunches.size() && bunches[indexBunch].modifiers != param)
            indexBunch++;
        if(indexBunch == bunches.size())  // add another bunch
            bunches.push_back(Bunch(param));
        Bunch& bunch = bunches[indexBunch];

        // if lmax or mmax are provided manually, override the default values for galpot
        if(kvmap[i].contains("lmax"))
            bunch.galpot_lmax = std::max<int>(bunch.galpot_lmax, kvmap[i].getInt("lmax"));
        if(kvmap[i].contains("mmax"))
            bunch.galpot_mmax = std::max<int>(bunch.galpot_mmax, kvmap[i].getInt("mmax"));

        // Several alternatives are possible, depending on the "type=..." and "file=..." params:
        switch(param.potentialType) {
        // 1. if a file=... is provided without type=..., then it must refer to another INI file
        case PT_UNKNOWN: {
            if(param.file.empty())
                throw std::invalid_argument("If type=... is not provided, need a file=...");
            bunch.componentsPot.push_back(readPotential(param.file, converter));
            break;
        }
        // 2. if the parameters describe a GalPot component, add it to the list of density and
        // possibly potential components
        case PT_DISK: {
            DiskParam dparam = parseDiskParam(param);
            // the two parts of disk profile: DiskAnsatz goes to the list of potentials...
            bunch.componentsPot.push_back(PtrPotential(new DiskAnsatz(dparam)));
            // ...and gets subtracted from the entire DiskDensity for the list of density components
            bunch.componentsDens.push_back(PtrDensity(new DiskDensity(dparam)));
            dparam.surfaceDensity *= -1;  // subtract the density of DiskAnsatz
            bunch.componentsDens.push_back(PtrDensity(new DiskAnsatz(dparam)));
            break;
        }
        case PT_SPHEROID: {
            bunch.componentsDens.push_back(PtrDensity(new SpheroidDensity(parseSpheroidParam(param))));
            break;
        }
        case PT_NUKER: {
            bunch.componentsDens.push_back(PtrDensity(new SpheroidDensity(parseNukerParam(param))));
            break;
        }
        case PT_SERSIC: {
            bunch.componentsDens.push_back(PtrDensity(new SpheroidDensity(parseSersicParam(param))));
            break;
        }
        // 3. if the potential type is one of the expansions and a source density/potential
        // is provided, construct the expansion from a temporary analytic den/pot object
        case PT_BASISSET:
        case PT_MULTIPOLE:
        case PT_CYLSPLINE: {
            bunch.componentsPot.push_back(createPotentialExpansion(param, kvmap[i]));
            break;
        }
        // 4,5. specifies a spatially-uniform time-dependent acceleration, or an evolving potential
        case PT_UNIFORMACCELERATION: {
            bunch.componentsPot.push_back(readUniformAcceleration(param.file, converter));
            break;
        }
        case PT_EVOLVING: {
            bunch.componentsPot.push_back(createEvolvingPotential(kvmap[i], converter));
            break;
        }
        // 6. the remaining alternative is an elementary potential, or an error
        default:
            bunch.componentsPot.push_back(createAnalyticPotential(param));
        }
    }

    // now loop over the list of bunches and finalize their construction
    // (convert each bunch into a composite potential, possibly wrapped into some modifiers)
    std::vector<PtrPotential> bunchPotentials;
    for(std::vector<Bunch>::iterator bunch = bunches.begin(); bunch != bunches.end(); ++bunch) {
        // if the list of density components is not empty, create an additional Multipole potential
        if(!bunch->componentsDens.empty()) {
            PtrDensity totalDens;
            if(bunch->componentsDens.size() == 1)
                totalDens = bunch->componentsDens[0];
            else
                totalDens.reset(new CompositeDensity(bunch->componentsDens));
            bunch->componentsPot.push_back(Multipole::create(*totalDens, coord::ST_UNKNOWN,
                bunch->galpot_lmax>=0 ? bunch->galpot_lmax : (isSpherical   (*totalDens) ? 0 : GALPOT_LMAX),
                bunch->galpot_mmax>=0 ? bunch->galpot_mmax : (isAxisymmetric(*totalDens) ? 0 : GALPOT_MMAX),
                GALPOT_NRAD));
        }
        if(bunch->modifiers.empty()) {
            // add all components with no modifiers individually to the overall list
            for(std::vector<PtrPotential>::iterator comp = bunch->componentsPot.begin();
                comp != bunch->componentsPot.end(); ++comp)
                bunchPotentials.push_back(*comp);
        } else {
            // if the bunch has more than one component, first create a temporary composite potential
            PtrPotential bunchPotential = bunch->componentsPot.size()==1 ?
                bunch->componentsPot[0] :
                PtrPotential(new Composite(bunch->componentsPot));
            // then apply the modifiers to the whole bunch (or its only component)
            applyModifiers(bunchPotential, bunch->modifiers);
            // and finally, append this bunch potential to the overall list
            bunchPotentials.push_back(bunchPotential);
        }
    }

    // finally, return either the potential of a single bunch,
    // or create yet another Composite potential from all bunches
    if(bunchPotentials.size() == 1)
        return bunchPotentials[0];
    else
        return PtrPotential(new Composite(bunchPotentials));
}

// create a potential from a single set of parameters
// (which may still turn into a composite potential if it happened to be one of GalPot things)
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    return createPotential(std::vector<utils::KeyValueMap>(1, kvmap), converter);
}

// create a potential expansion from the user-provided source density
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const PtrDensity& dens,
    const units::ExternalUnits& converter)
{
    const AllParam param = parseParam(kvmap, converter);
    PtrPotential result = createPotentialExpansionFromSource(param, dens);
    applyModifiers(result, param);
    return result;

}

// create a potential expansion from the user-provided source potential
// and/or add a modifier on top of the input potential
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const PtrPotential& pot,
    const units::ExternalUnits& converter)
{
    const AllParam param = parseParam(kvmap, converter);
    PtrPotential result;
    switch(param.potentialType) {
    case PT_MULTIPOLE:
    case PT_CYLSPLINE:
    case PT_BASISSET : {
        result = createPotentialExpansionFromSource(param, pot);
        break;
    }
    case PT_UNKNOWN: {
        result = pot;  // if an expansion type is not provided, use the input potential
        break;
    }
    default:
        throw std::invalid_argument("createPotential: type should be either empty "
            "or specify one of the potential expansion classes");
    }
    // add modifiers on top of either input potential or its expansion
    applyModifiers(result, param);
    return result;
}

// create potential from particles
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const particles::ParticleArray<coord::PosCyl>& particles,
    const units::ExternalUnits& converter)
{
    const AllParam param = parseParam(kvmap, converter);
    PtrPotential result = createPotentialExpansionFromParticles(param, particles);
    applyModifiers(result, param);
    return result;
}

// create/read density from an INI file (which may also contain density expansion coefficients)
PtrDensity readDensity(const std::string& iniFileName, const units::ExternalUnits& converter)
{
    if(iniFileName.empty())
        throw std::runtime_error("Empty file name");
    const utils::ConfigFile ini(iniFileName);
    // temporarily change the current directory (if needed) to ensure that the filenames
    // in the INI file are processed correctly with paths relative to the INI file itself
    DirectoryChanger dirchanger(iniFileName);
    std::vector<std::string> sectionNames = ini.listSections();
    std::vector<PtrDensity> components;
    for(unsigned int i=0; i<sectionNames.size(); i++)
        if(utils::stringsEqual(sectionNames[i].substr(0,7), "Density"))
            components.push_back(createDensity(ini.findSection(sectionNames[i]), converter));
    if(components.size() == 0)
        throw std::runtime_error("INI file does not contain any [Density] section");
    if(components.size() == 1)
        return components[0];
    return PtrDensity(new CompositeDensity(components));
}

// create a potential from INI file (which may also contain coefficients of potential expansions)
PtrPotential readPotential(const std::string& iniFileName, const units::ExternalUnits& converter)
{
    if(iniFileName.empty())
        throw std::runtime_error("Empty file name");
    const utils::ConfigFile ini(iniFileName);
    // temporarily change the current directory (if needed) to ensure that the filenames
    // in the INI file are processed correctly with paths relative to the INI file itself
    DirectoryChanger dirchanger(iniFileName);
    std::vector<std::string> sectionNames = ini.listSections();
    std::vector<utils::KeyValueMap> components;
    for(unsigned int i=0; i<sectionNames.size(); i++)
        if(utils::stringsEqual(sectionNames[i].substr(0,9), "Potential"))
            components.push_back(ini.findSection(sectionNames[i]));
    if(components.size() == 0)
        throw std::runtime_error("INI file does not contain any [Potential] section");
    return createPotential(components, converter);
}

///@}
/// \name Legacy interface for loading GalPot parameters from a text file (deprecated)
//        ----------------------------------------------------------------------------
///@{

PtrPotential readGalaxyPotential(const std::string& filename, const units::ExternalUnits& conv) 
{
    std::ifstream strm(filename.c_str());
    if(!strm) 
        throw std::runtime_error("Cannot open file "+std::string(filename));
    std::vector<utils::KeyValueMap> kvmap;
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields  = utils::splitString(buffer, "# \t");
    int num = utils::toInt(fields[0]);
    for(int i=0; i<num && ok; i++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        if(fields.size() >= 5)
            kvmap.push_back(utils::KeyValueMap(
                std::string("type=") +  DiskDensity::myName() +
                " surfaceDensity=" +    fields[0]+
                " scaleRadius=" +       fields[1]+
                " scaleHeight=" +       fields[2]+
                " innerCutoffRadius=" + fields[3]+
                " modulationAmplitude="+fields[4]));
        else ok=false;
    }
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    num = utils::toInt(fields[0]);
    for(int i=0; i<num && ok; i++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        if(fields.size() >= 5)
            kvmap.push_back(utils::KeyValueMap(
                std::string("type=") + SpheroidDensity::myName() +
                " densityNorm="      + fields[0]+
                " axisRatioZ="       + fields[1]+
                " gamma="            + fields[2]+
                " beta="             + fields[3]+
                " scaleRadius="      + fields[4]+
                (utils::toDouble(fields[5])!=0 ? " outerCutoffRadius=" + fields[5] : "") ));
        else ok=false;
    }
    return createPotential(kvmap, conv);
}

///@}

bool writeDensity(const std::string& fileName, const BaseDensity& dens,
    const units::ExternalUnits& converter)
{
    if(fileName.empty())
        return false;
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm)
        return false;
    int counter = 0;
    if(strm.good())
        writeAnyDensityOrPotential(strm, &dens, converter, counter);
    return strm.good();
}

}  // namespace potential
