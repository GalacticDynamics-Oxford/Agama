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
#ifdef _WIN32
#include <direct.h>
#define DIRECTORY_SEPARATOR '\\'
#else
#include <unistd.h>
#define DIRECTORY_SEPARATOR '/'
#endif

namespace potential {

namespace {  // internal definitions and routines

/// order of the Multipole expansion for the GalPot potential
static const int GALPOT_LMAX = 32;

/// order of the azimuthal Fourier expansion in case of non-axisymmetric components
static const int GALPOT_MMAX = 6;

/// number of radial points in the Multipole potential automatically constructed for GalPot
static const int GALPOT_NRAD = 50;

/// \name Definitions of all known potential types and parameters
//        -------------------------------------------------------
///@{

/** List of all known potential and density types 
    (borrowed from SMILE, not everything is implemented here).
    Note that this type is not a substitute for the real class hierarchy:
    it is intended only to be used in factory methods such as 
    creating an instance of potential from its name 
    (e.g., passed as a string, or loaded from an ini file).
*/
enum PotentialType {

    PT_UNKNOWN,      ///< unspecified/not provided
    PT_INVALID,      ///< provided but does not correspond to a known class

    // density interpolators
    PT_DENS_SPHHARM, ///< `DensitySphericalHarmonic`
    PT_DENS_CYLGRID, ///< `DensityAzimuthalHarmonic`

    // generic potential expansions
    PT_BASISSET,     ///< radial basis-set expansion: `BasisSet`
    PT_MULTIPOLE,    ///< spherical-harmonic expansion:  `Multipole`
    PT_CYLSPLINE,    ///< expansion in azimuthal angle with 2d interpolating splines in (R,z):  `CylSpline`

    // components of GalPot
    PT_DISK,         ///< separable disk density model:  `Disk`
    PT_SPHEROID,     ///< double-power-law 3d density model:  `Spheroid`
    PT_NUKER,        ///< double-power-law surface density profile: `Nuker`
    PT_SERSIC,       ///< Sersic profile:  `Sersic`

    // analytic potentials that can't be used as source density for a potential expansion
    PT_LOG,          ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC,     ///< triaxial simple harmonic oscillator:  `Harmonic`
    PT_KEPLERBINARY, ///< two point masses on a Kepler orbit: `KeplerBinary`
    PT_UNIFORMACCELERATION,  ///< a spatially uniform but time-dependent acceleration: `UniformAcceleration`
    PT_EVOLVING,     ///< a time-dependent series of potentials: `Evolving`

    // analytic potential models that can also be used as source density for a potential expansion
    PT_NFW,          ///< spherical Navarro-Frenk-White profile:  `NFW`
    PT_MIYAMOTONAGAI,///< axisymmetric Miyamoto-Nagai(1975) model:  `MiyamotoNagai`
    PT_DEHNEN,       ///< spherical, axisymmetric or triaxial Dehnen(1993) density model:  `Dehnen`
    PT_FERRERS,      ///< triaxial Ferrers model with finite extent:  `Ferrers`
    PT_PLUMMER,      ///< spherical Plummer model:  `Plummer`
    PT_ISOCHRONE,    ///< spherical isochrone model:  `Isochrone`
    PT_PERFECTELLIPSOID,  ///< axisymmetric model of Kuzmin/de Zeeuw :  `PerfectEllipsoid`
    PT_KING,         ///< generalized King (lowered isothermal) model, represented by Multipole

    // composite density or potential
    PT_COMPOSITE_DENSITY,    ///< `CompositeDensity`
    PT_COMPOSITE_POTENTIAL,  ///< `CompositePotential`
};

/// structure that contains parameters for all possible density or potential models
/// (not all of them make sense for any given model)
struct AllParam
{
    PotentialType potentialType;      ///< type of the potential
    PotentialType densityType;        ///< density model used for initializing a potential expansion
    coord::SymmetryType symmetryType; ///< degree of symmetry
    double mass;                      ///< total mass
    double surfaceDensity;            ///< central surface density for Disk, Nuker or Sersic models
    double densityNorm;               ///< density normalization for double-power-law models
    double scaleRadius;               ///< scale radius
    double scaleHeight;               ///< scale height or second scale radius
    double innerCutoffRadius;         ///< central hole for disk profiles
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
    unsigned int gridSizeR;  ///< number of radial grid points in Multipole and CylSpline potentials
    unsigned int gridSizez;  ///< number of grid points in z-direction for CylSpline potential
    double rmin, rmax;       ///< inner- and outermost grid node radii for Multipole and CylSpline
    double zmin, zmax;       ///< grid extent in z direction for CylSpline
    unsigned int lmax;       ///< number of angular terms in spherical-harmonic expansion
    unsigned int mmax;       ///< number of angular terms in azimuthal-harmonic expansion
    double smoothing;        ///< amount of smoothing in Multipole initialized from an N-body snapshot
    unsigned int nmax;       ///< order of radial expansion for BasisSet (actual number of terms is nmax+1)
    double eta;              ///< shape parameters of basis functions for BasisSet (0.5-CB, 1.0-HO, etc.)
    double r0;               ///< scale radius of the basis functions for BasisSet
    std::string file;        ///< name of file with coordinates of points, or coefficients of expansion
    std::string center;      ///< coordinates of the center offset or the name of a file with these offsets
    /// default constructor initializes the fields to some reasonable values
    AllParam() :
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(coord::ST_DEFAULT),
        mass(1.), surfaceDensity(NAN), densityNorm(NAN),
        scaleRadius(1.), scaleHeight(1.), innerCutoffRadius(0.), outerCutoffRadius(INFINITY),
        v0(1.), Omega(1.),
        axisRatioY(1.), axisRatioZ(1.),
        alpha(1.), beta(4.), gamma(1.),
        modulationAmplitude(0.), cutoffStrength(2.), sersicIndex(NAN), W0(NAN), trunc(1.),
        binary_q(0), binary_sma(0), binary_ecc(0), binary_phase(0),
        gridSizeR(25), gridSizez(25), rmin(0), rmax(0), zmin(0), zmax(0),
        lmax(6), mmax(6), smoothing(1.), nmax(12), eta(1.0), r0(0)
    {};
    /// convert to KeplerBinaryParams
    operator KeplerBinaryParams() const
    {
        return KeplerBinaryParams(mass, binary_q, binary_sma, binary_ecc, binary_phase);
    }
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
    if(utils::stringsEqual(name, "King"))                  return PT_KING;
    if(utils::stringsEqual(name, UniformAcceleration     ::myName())) return PT_UNIFORMACCELERATION;
    if(utils::stringsEqual(name, OblatePerfectEllipsoid  ::myName())) return PT_PERFECTELLIPSOID;
    if(utils::stringsEqual(name, DensitySphericalHarmonic::myName())) return PT_DENS_SPHHARM;
    if(utils::stringsEqual(name, DensityAzimuthalHarmonic::myName())) return PT_DENS_CYLGRID;
    if(utils::stringsEqual(name, CompositeDensity::myName())) return PT_COMPOSITE_DENSITY;
    if(utils::stringsEqual(name, Composite       ::myName())) return PT_COMPOSITE_POTENTIAL;
    if(utils::stringsEqual(name, Evolving        ::myName())) return PT_EVOLVING;
    return PT_INVALID;
}

} // internal namespace

// return the type of symmetry by its name, or ST_DEFAULT if unavailable
coord::SymmetryType getSymmetryTypeByName(const std::string& symmetryName)
{
    if(symmetryName.empty())
        return coord::ST_DEFAULT;
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
    int sym = -1;
    try{ sym = utils::toInt(symmetryName); }
    catch(std::exception&) { sym = -1; }  // parse error - it wasn't a valid number either
    if(sym<0 || sym>static_cast<int>(coord::ST_SPHERICAL))
        throw std::runtime_error("Invalid symmetry type: " + symmetryName);
    return static_cast<coord::SymmetryType>(sym);
}

// inverse of the above: return a symbolic name or a numerical code of symmetry type
std::string getSymmetryNameByType(coord::SymmetryType type)
{
    switch(type) {
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

/** Parse the potential or density parameters contained in a text array of "key=value" pairs.
    \param[in] kvmap  is the array of string pairs "key" and "value", for instance,
    created from command-line arguments, or read from an INI file;
    \param[in] converter  is the instance of unit converter for translating the dimensional
    parameters (such as mass or scale radius) into internal units (may be a trivial converter);
    \return    the structure containing all possible potential/density parameters
*/
AllParam parseParam(const utils::KeyValueMap& kvmap, const units::ExternalUnits& conv)
{
    AllParam param;
    param.potentialType       = getPotentialTypeByName(kvmap.getString("type"));
    param.densityType         = getPotentialTypeByName(kvmap.getString("density"));
    param.symmetryType        = getSymmetryTypeByName (kvmap.getString("symmetry"));
    param.file                = kvmap.getString("file");
    param.center              = kvmap.getString("center");
    param.mass                = kvmap.getDouble("mass", param.mass)
                              * conv.massUnit;
    param.surfaceDensity      = kvmap.getDoubleAlt("surfaceDensity", "Sigma0", param.surfaceDensity)
                              * conv.massUnit / pow_2(conv.lengthUnit);
    param.densityNorm         = kvmap.getDoubleAlt("densityNorm", "rho0", param.densityNorm)
                              * conv.massUnit / pow_3(conv.lengthUnit);
    param.scaleRadius         = kvmap.getDoubleAlt("scaleRadius", "rscale", param.scaleRadius)
                              * conv.lengthUnit;
    param.scaleHeight         = kvmap.getDoubleAlt("scaleHeight", "scaleRadius2", param.scaleHeight)
                              * conv.lengthUnit;
    param.innerCutoffRadius   = kvmap.getDouble("innerCutoffRadius", param.innerCutoffRadius)
                              * conv.lengthUnit;
    param.outerCutoffRadius   = kvmap.getDouble("outerCutoffRadius", param.outerCutoffRadius)
                              * conv.lengthUnit;
    param.v0                  = kvmap.getDouble("v0", param.v0)
                              * conv.velocityUnit;
    param.Omega               = kvmap.getDouble("Omega", param.Omega)
                              * conv.velocityUnit / conv.lengthUnit;
    param.axisRatioY          = kvmap.getDoubleAlt("axisRatioY", "p", param.axisRatioY);
    param.axisRatioZ          = kvmap.getDoubleAlt("axisRatioZ", "q", param.axisRatioZ);
    param.alpha               = kvmap.getDouble("alpha", param.alpha);
    param.beta                = kvmap.getDouble("beta",  param.beta);
    param.gamma               = kvmap.getDouble("gamma", param.gamma);
    param.modulationAmplitude = kvmap.getDouble("modulationAmplitude", param.modulationAmplitude);
    param.cutoffStrength      = kvmap.getDoubleAlt("cutoffStrength", "xi", param.cutoffStrength);
    param.sersicIndex         = kvmap.getDouble("sersicIndex", param.sersicIndex);
    param.W0                  = kvmap.getDouble("W0", param.W0);
    param.trunc               = kvmap.getDouble("trunc", param.trunc);
    param.binary_q            = kvmap.getDouble("binary_q",     param.binary_q);
    param.binary_sma          = kvmap.getDouble("binary_sma",   param.binary_sma)
                              * conv.lengthUnit;
    param.binary_ecc          = kvmap.getDouble("binary_ecc",   param.binary_ecc);
    param.binary_phase        = kvmap.getDouble("binary_phase", param.binary_phase);
    param.gridSizeR           = kvmap.getInt(   "gridSizeR", param.gridSizeR);
    param.gridSizez           = kvmap.getInt(   "gridSizeZ", param.gridSizez);
    param.rmin                = kvmap.getDouble("rmin", param.rmin)
                              * conv.lengthUnit;
    param.rmax                = kvmap.getDouble("rmax", param.rmax)
                              * conv.lengthUnit;
    param.zmin                = kvmap.getDouble("zmin", param.zmin)
                              * conv.lengthUnit;
    param.zmax                = kvmap.getDouble("zmax", param.zmax)
                              * conv.lengthUnit;
    param.lmax                = kvmap.getInt(   "lmax", param.lmax);
    param.mmax                = kvmap.contains( "mmax") ? kvmap.getInt("mmax") : param.lmax;
    param.smoothing           = kvmap.getDouble("smoothing", param.smoothing);
    param.nmax                = kvmap.getInt(   "nmax", param.nmax);
    param.eta                 = kvmap.getDouble("eta",  param.eta);
    param.r0                  = kvmap.getDouble("r0",   param.r0)
                              * conv.lengthUnit;

    // tweak: if 'type' is Plummer or NFW, but axis ratio is not unity or a cutoff radius is provided,
    // replace it with an equivalent Spheroid model, because the dedicated potential models
    // can only be spherical and non-truncated
    PotentialType type = param.densityType != PT_UNKNOWN ? param.densityType : param.potentialType;
    if( (type == PT_PLUMMER || type == PT_NFW) &&
        (param.axisRatioY != 1 || param.axisRatioZ !=1 || param.outerCutoffRadius!=INFINITY) )
    {
        param.alpha = type == PT_PLUMMER ? 2 : 1;
        param.beta  = type == PT_PLUMMER ? 5 : 3;
        param.gamma = type == PT_PLUMMER ? 0 : 1;
        if(param.outerCutoffRadius==INFINITY) {
            param.densityNorm = (type == PT_PLUMMER ? 0.75 : 0.25) / M_PI * param.mass /
                (pow_3(param.scaleRadius) * param.axisRatioY * param.axisRatioZ);
        } else
            param.densityNorm = NAN;    // will determine automatically from the total mass
        if(param.densityType == type)
            param.densityType = PT_SPHEROID;
        else
            param.potentialType = PT_SPHEROID;
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
        std::string newcwd = filenameWithPath.substr(0, index+1);
        if(chdir(newcwd.c_str()) != 0)
            throw std::runtime_error("Failed to change the current working directory to "+newcwd);
        utils::msg(utils::VL_VERBOSE, "DirectoryChanger",
            "Temporarily changed the current working directory to "+newcwd);
    }
    ~DirectoryChanger()
    {
        if(!oldcwd.empty()) {
            if(chdir(oldcwd.c_str()) != 0)
                // we can't throw an exception from the destructor, so just print a warning
                utils::msg(utils::VL_WARNING, "DirectoryChanger", "Failed to restore working directory");
            utils::msg(utils::VL_VERBOSE, "DirectoryChanger",
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
PtrPotential createBasisSetFromCoefs(std::vector<std::string>& lines,
    const AllParam& params, const units::ExternalUnits& converter)
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
        math::blas_dmul(pow_2(converter.velocityUnit), coefs[i]);
    return PtrPotential(new BasisSet(params.eta, params.r0, coefs));
}

/// parse the array of coefficients and create a Multipole potential
PtrPotential createMultipoleFromCoefs(std::vector<std::string>& lines,
    const AllParam& params, const units::ExternalUnits& converter)
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
    math::blas_dmul(converter.lengthUnit, gridr);
    for(unsigned int i=0; i<coefsPhi.size(); i++) {
        math::blas_dmul(pow_2(converter.velocityUnit), coefsPhi[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, coefsdPhi[i]);
    }
    return PtrPotential(new Multipole(gridr, coefsPhi, coefsdPhi));
}

/// parse the array of coefficients and create a CylSpline potential
PtrPotential createCylSplineFromCoefs(std::vector<std::string>& lines,
    const AllParam& params, const units::ExternalUnits& converter)
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
    math::blas_dmul(converter.lengthUnit, gridR);
    math::blas_dmul(converter.lengthUnit, gridz);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(pow_2(converter.velocityUnit), Phi[i]);
        if(!okder) continue;  // no derivs
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidR[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidz[i]);
    }
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

/// parse the array of coefficients and create a SphericalHarmonic density
PtrDensity createDensitySphericalHarmonicFromCoefs(std::vector<std::string>& lines,
    const AllParam& params, const units::ExternalUnits& converter)
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
    math::blas_dmul(converter.lengthUnit, gridr);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(converter.massUnit/pow_3(converter.lengthUnit), coefs[i]);
    return PtrDensity(new DensitySphericalHarmonic(gridr, coefs));
}

/// parse the array of coefficients and create an AzimuthalHarmonic density
PtrDensity createDensityAzimuthalHarmonicFromCoefs(std::vector<std::string>& lines,
    const AllParam& params, const units::ExternalUnits& converter)
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
    math::blas_dmul(converter.lengthUnit, gridR);
    math::blas_dmul(converter.lengthUnit, gridz);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(converter.massUnit/pow_3(converter.lengthUnit), coefs[i]);
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
    strm << "Coefficients\n#rho\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, coefs);
}

/// write data (expansion coefs or components) for a single or composite density or potential to a stream
void writeAnyDensityOrPotential(std::ostream& strm, const BaseDensity* dens,
    const units::ExternalUnits& converter, int& counter)
{
    // first, check if this is a shifted density/potential
    const ShiftedDensity* sd = dynamic_cast<const ShiftedDensity*>(dens);
    if(sd)
        dens = sd->dens.get();
    const Shifted* sp = dynamic_cast<const Shifted*>(dens);
    if(sp)
        dens = sp->pot.get();
    // we have no way of storing the shift parameters, but at least inform about the fact
    bool shifted = sd!=NULL || sp!=NULL;

    // check if this is a CompositeDensity
    const CompositeDensity* cd = dynamic_cast<const CompositeDensity*>(dens);
    if(cd) {
        if(shifted)
            strm << "#center is not stored\n";
        for(unsigned int i=0; i<cd->size(); i++) {
            writeAnyDensityOrPotential(strm, cd->component(i).get(), converter, counter);
            if(i+1<cd->size())
                strm << '\n';
        }
        return;
    }

    // check if this is a Composite potential
    const Composite* cp = dynamic_cast<const Composite*>(dens);
    if(cp) {
        if(shifted)
            strm << "#center is not stored\n";
        for(unsigned int i=0; i<cp->size(); i++) {
            writeAnyDensityOrPotential(strm, cp->component(i).get(), converter, counter);
            if(i+1<cp->size())
                strm << '\n';
        }
        return;
    }

    // otherwise this is an elementary potential or density, so write a section header
    const BasePotential* pot = dynamic_cast<const BasePotential*>(dens);
    if(pot)
        strm << "[Potential";
    else
        strm << "[Density";
    if(counter>0)
        strm << counter;
    strm << "]\n";
    strm << "type=" << dens->name() << '\n';
    if(shifted)
        strm << "#center is not stored\n";
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

/// read a file with 3 components of some time-dependent vector quantity
static void readTimeDependentVector(
    const std::string& filename, /*units:*/ double timeUnit, double valueUnit,
    /*output*/ math::CubicSpline& splx, math::CubicSpline& sply, math::CubicSpline& splz)
{
    std::ifstream strm(filename.c_str(), std::ios::in);
    if(!strm)
        throw std::runtime_error("readTimeDependentVector: cannot read from file "+filename);
    std::string buffer;
    std::vector<std::string> fields;
    std::vector<double> time, posx, posy, posz, velx, vely, velz;
    while(std::getline(strm, buffer) && !strm.eof()) {
        if(!buffer.empty() && utils::isComment(buffer[0]))  // commented line
            continue;
        fields = utils::splitString(buffer, "#;, \t");
        size_t numFields = fields.size();
        if(numFields < 4 ||
           !((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
            continue;
        time.push_back(utils::toDouble(fields[0]) * timeUnit);
        posx.push_back(utils::toDouble(fields[1]) * valueUnit);
        posy.push_back(utils::toDouble(fields[2]) * valueUnit);
        posz.push_back(utils::toDouble(fields[3]) * valueUnit);
        if(numFields >= 7) {
            velx.push_back(utils::toDouble(fields[4]) * valueUnit / timeUnit);
            vely.push_back(utils::toDouble(fields[5]) * valueUnit / timeUnit);
            velz.push_back(utils::toDouble(fields[6]) * valueUnit / timeUnit);
        }
    }
    if(posx.size() == 0) {
        throw std::runtime_error("readTimeDependentVector: no valid entries in "+filename);
    }
    if(velx.size() == posx.size()) {
        // create Hermite splines from values and derivatives at each moment of time
        splx = math::CubicSpline(time, posx, velx);
        sply = math::CubicSpline(time, posy, vely);
        splz = math::CubicSpline(time, posz, velz);
    } else {
        // create natural cubic splines from just the values (and regularize)
        splx = math::CubicSpline(time, posx, true);
        sply = math::CubicSpline(time, posy, true);
        splz = math::CubicSpline(time, posz, true);
    }
}

/** helper function for finding the slope of asymptotic power-law behaviour of a certain function:
    if  f(x) ~ f0 + a * x^b  as  x --> 0  or  x --> infinity,  then the slope b is given by
    solving the equation  [x1^b - x2^b] / [x2^b - x3^b] = [f(x1) - f(x2)] / [f(x2) - f(x3)],
    where x1, x2 and x3 are three consecutive points near the end of the interval.
    The arrays of x and corresponding f(x) are passed as parameters to this function,
    and its value() method is used in the root-finding routine.
*/
class SlopeFinder: public math::IFunctionNoDeriv {
    const double r12, r32, ratio;
public:
    SlopeFinder(double logx1, double logx2, double logx3, double f1, double f2, double f3) :
    r12(logx1-logx2), r32(logx3-logx2), ratio( (f1-f2) / (f2-f3) ) {}

    virtual double value(const double b) const {
        if(b==0)
            return -r12 / r32 - ratio;
        return (exp(b * r12) - 1) / (1 - exp(b * r32)) - ratio;
    }
};

///@}
/// \name Factory routines for creating instances of Density and Potential classes
//        ------------------------------------------------------------------------
///@{

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
            param.v0, param.scaleRadius, param.axisRatioY, param.axisRatioZ));
    case PT_HARMONIC:
        return PtrPotential(new Harmonic(param.Omega, param.axisRatioY, param.axisRatioZ));
    case PT_KEPLERBINARY:
        return PtrPotential(new KeplerBinary(param));
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
        if(param.axisRatioY==1 && param.axisRatioZ<1)
            return PtrPotential(new OblatePerfectEllipsoid(
                param.mass, param.scaleRadius, param.scaleRadius*param.axisRatioZ)); 
        else
            throw std::invalid_argument("May only create oblate axisymmetric Perfect Ellipsoid model");
    case PT_KING:
        return createKingPotential(param.mass, param.scaleRadius, param.W0, param.trunc);
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
    for the provided source density or potential
    (template parameter SourceType==BaseDensity or BasePotential) */
template<typename SourceType>
PtrPotential createPotentialExpansionFromSource(const AllParam& param, const SourceType& source)
{
    switch(param.potentialType) {
    case PT_BASISSET:
        return BasisSet::create(source, param.lmax, param.mmax,
            param.nmax, param.eta, param.r0);
    case PT_MULTIPOLE:
        return Multipole::create(source, param.lmax, param.mmax,
            param.gridSizeR, param.rmin, param.rmax);
    case PT_CYLSPLINE:
        return CylSpline::create(source, param.mmax,
            param.gridSizeR, param.rmin, param.rmax,
            param.gridSizez, param.zmin, param.zmax);
    default: throw std::invalid_argument("Unknown potential expansion type");
    }
}

/** General routine for creating a potential expansion from the provided INI parameters */
PtrPotential createPotentialExpansion(
    const AllParam& param, const utils::KeyValueMap& kvmap, const units::ExternalUnits& converter)
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
            case PT_BASISSET:  return createBasisSetFromCoefs (lines, param, converter);
            case PT_MULTIPOLE: return createMultipoleFromCoefs(lines, param, converter);
            default:           return createCylSplineFromCoefs(lines, param, converter);
        }
    }

    // option 2: N-body snapshot
    if(haveFile && !haveCoefs && !haveSource) {
        if(!utils::fileExists(param.file))
            throw std::runtime_error("File " + param.file + " does not exist");
        const particles::ParticleArrayCar particles = particles::readSnapshot(param.file, converter);
        if(particles.size()==0)
            throw std::runtime_error("Error loading N-body snapshot from " + param.file);

        PtrPotential pot = createPotentialExpansionFromParticles(param, particles);

        // store coefficients in a text file,
        // later may load this file instead for faster initialization
        try{
            writePotential(param.file + ".pot", *pot, converter);
        }
        catch(std::exception& ex) {  // not a critical error, but worth mentioning
            utils::msg(utils::VL_MESSAGE, "createPotential", ex.what());
        }
        return pot;
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
            return createPotentialExpansionFromSource(param, *createAnalyticPotential(srcpar));
        }
        else
        {   // otherwise use analytic density as the source
            return createPotentialExpansionFromSource(param, *createAnalyticDensity(srcpar));
        }
    }

    throw std::invalid_argument(
        std::string(param.potentialType == PT_BASISSET  ? "BasisSet" :
                    param.potentialType == PT_MULTIPOLE ? "Multipole" : "CylSpline") +
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
    math::CubicSpline splx, sply, splz;
    readTimeDependentVector(filename,
        /*units*/ converter.timeUnit, converter.velocityUnit / converter.timeUnit /*acceleration*/,
        /*output*/ splx, sply, splz);
    return PtrPotential(new UniformAcceleration(splx, sply, splz));
}

/** create an instance of Shifted (potential) or ShiftedDensity from the given object
    (Ptr = PtrDensity or PtrPotential) and a string containing the offset values or a file name */
template<typename Shifted, typename Ptr>
Ptr createOffset(const std::string& center, const units::ExternalUnits& converter, const Ptr& obj)
{
    // string could contain either three components of the fixed offset vector,
    // or the name of a file with time-dependent trajectory
    if(utils::fileExists(center)) {
        math::CubicSpline splx, sply, splz;
        readTimeDependentVector(center,
            /*units*/ converter.timeUnit, converter.lengthUnit,
            /*output*/ splx, sply, splz);
        return Ptr(new Shifted(obj, splx, sply, splz));
    } else {
        std::vector<std::string> values = utils::splitString(center, ",; ");
        double x = NAN, y = NAN, z = NAN;
        if(values.size() == 3) {
            x = utils::toDouble(values[0]) * converter.lengthUnit;
            y = utils::toDouble(values[1]) * converter.lengthUnit;
            z = utils::toDouble(values[2]) * converter.lengthUnit;
        }
        if(!isFinite(x+y+z))
            throw std::runtime_error(
                "\"center\" must be either a filename or a triplet of numbers");
        if(x==0 && y==0 && z==0)
            return obj;   // don't create entities without necessity
        else
            return Ptr(new Shifted(obj, x, y, z));
    }
}

// a collection of would-be potential components with a common center (see createPotential)
struct Bunch {
    std::string center;
    // all potential components (DiskAnsatz or any other potential class)
    std::vector<PtrPotential> componentsPot;
    // all density components that will contribute to a single additional Multipole potential
    std::vector<PtrDensity> componentsDens;
    // constructor
    Bunch(const std::string& _center=""): center(_center) {}
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
    double B = math::findRoot(SlopeFinder(
        gridlogr[size-1], gridlogr[size-2], gridlogr[size-3],
        gridm   [size-1], gridm   [size-2], gridm   [size-3] ), -100, 0, /*tolerance*/1e-6);
    double invMinf = 0;  // 1/Minf, or remain 0 if no finite limit is detected
    if(B<0) {
        double A =  (gridm[size-1] - gridm[size-2]) /
        (exp(B * gridlogr[size-2]) - exp(B * gridlogr[size-1]));
        if(A>0) {  // viable extrapolation
            invMinf = 1 / (gridm[size-1] + A * exp(B * gridlogr[size-1]));
            utils::msg(utils::VL_DEBUG, "densityFromCumulativeMass",
                "Extrapolated total mass=" + utils::toString(1/invMinf) +
                ", rho(r)~r^" + utils::toString(B-3) + " at large radii" );
        }
    }
    // scaled mass to interpolate:  log[ M / (1 - M/Minf) ] as a function of log(r),
    // which has a linear asymptotic behaviour with slope -B as log(r) --> infinity;
    // if Minf = infinity, this additional term has no effect
    for(unsigned int i=0; i<size; i++)
        gridlogm[i] = log(gridm[i] / (1 - gridm[i]*invMinf));
    math::CubicSpline spl(gridlogr, gridlogm, true /*enforce monotonicity*/);
    if(!spl.isMonotonic())
        throw std::runtime_error("densityFromCumulativeMass: interpolated mass is not monotonic");
    // compute the density at each point of the input radial grid
    for(unsigned int i=0; i<size; i++) {
        double val, der;
        spl.evalDeriv(gridlogr[i], &val, &der);
        val = exp(val);
        gridrho[i] = der * val / (4*M_PI * pow_3(gridr[i]) * pow_2(1 + val * invMinf));
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

// create elementary density (analytic or from expansion coefs)
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
            createDensitySphericalHarmonicFromCoefs(lines, param, converter) :
            createDensityAzimuthalHarmonicFromCoefs(lines, param, converter);
    } else
        // otherwise it must be one of the analytic density profiles
        result = createAnalyticDensity(param);

    // check if it needs to be off-centered
    std::string center = kvmap.getString("center");
    if(!center.empty())
        result = createOffset<ShiftedDensity>(center, converter, result);
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
    // 1) Elements of the GalPot scheme (disk and spheroid density profiles) are considered
    // together and create Ndisk+1 potential components:
    // each Disk group is represented by one potential component (DiskAnsatz) and two density
    // components ("residuals") that are added to the list of components of a CompositeDensity;
    // all Spheroid, Nuker and Sersic density profiles are also added to this CompositeDensity;
    // and in the end a single Multipole potential is constructed from this density collection.
    // 2) Any parameter group may have a non-trivial offset from origin, and the corresponding
    // potential will be wrapped into a Shifted potential modifier.
    // As a consequence of the two circumstances, the elements of GalPot scheme sharing a common
    // center offset are grouped into a single bunch of DiskAnsatz+Multipole combinations, but
    // there may be more than one such bunch if the center offsets vary between parameter groups.

    std::vector<Bunch> bunches;   // list of bunches sharing a common center

    // first loop over all parameter groups
    for(unsigned int i=0; i<kvmap.size(); i++) {
        const AllParam param = parseParam(kvmap[i], converter);

        // find the "bunch" with the same center, or create a new one
        unsigned int indexBunch = 0;
        while(indexBunch < bunches.size() && param.center != bunches[indexBunch].center)
            indexBunch++;
        if(indexBunch == bunches.size())  // add another bunch
            bunches.push_back(Bunch(param.center));
        Bunch& bunch = bunches[indexBunch];

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
            bunch.componentsPot.push_back(createPotentialExpansion(param, kvmap[i], converter));
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
        // 4. the remaining alternative is an elementary potential, or an error
        default:
            bunch.componentsPot.push_back(createAnalyticPotential(param));
        }
    }

    // now loop over the list of bunches and finalize their construction
    // (convert each bunch into a composite potential, possibly wrapped into a Shifted modifier)
    std::vector<PtrPotential> bunchPotentials;
    for(std::vector<Bunch>::iterator bunch = bunches.begin(); bunch != bunches.end(); ++bunch) {
        // if the list of density components is not empty, create an additional Multipole potential
        if(!bunch->componentsDens.empty()) {
            PtrDensity totalDens;
            if(bunch->componentsDens.size() == 1)
                totalDens = bunch->componentsDens[0];
            else
                totalDens.reset(new CompositeDensity(bunch->componentsDens));
            bunch->componentsPot.push_back(Multipole::create(*totalDens,
                isSpherical   (*totalDens) ? 0 : GALPOT_LMAX,
                isAxisymmetric(*totalDens) ? 0 : GALPOT_MMAX, GALPOT_NRAD));
        }
        // each bunch either has just one potential component, or produces a composite potential
        PtrPotential bunchPotential = bunch->componentsPot.size()==1 ?
            bunch->componentsPot[0] :
            PtrPotential(new Composite(bunch->componentsPot));
        // finally, if there is a nontrivial center offset, wrap the bunch into a Shifted potential
        if(!bunch->center.empty())
            bunchPotential = createOffset<Shifted>(bunch->center, converter, bunchPotential);
        bunchPotentials.push_back(bunchPotential);
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
    const BaseDensity& dens,
    const units::ExternalUnits& converter)
{
    PtrPotential result = createPotentialExpansionFromSource(parseParam(kvmap, converter), dens);
    std::string center = kvmap.getString("center");
    if(!center.empty())
        result = createOffset<Shifted>(center, converter, result);
    return result;
}

// create a potential expansion from the user-provided source potential
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const BasePotential& pot,
    const units::ExternalUnits& converter)
{
    PtrPotential result = createPotentialExpansionFromSource(parseParam(kvmap, converter), pot);
    std::string center = kvmap.getString("center");
    if(!center.empty())
        result = createOffset<Shifted>(center, converter, result);
    return result;
}

// create potential from particles
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const particles::ParticleArray<coord::PosCyl>& particles,
    const units::ExternalUnits& converter)
{
    PtrPotential result = createPotentialExpansionFromParticles(parseParam(kvmap, converter), particles);
    std::string center = kvmap.getString("center");
    if(!center.empty())
        result = createOffset<Shifted>(center, converter, result);
    return result;
}

// create/read density from an INI file (which may also contain density expansion coefficients)
PtrDensity readDensity(const std::string& iniFileName, const units::ExternalUnits& converter)
{
    if(iniFileName.empty())
        throw std::runtime_error("Empty file name");
    utils::ConfigFile ini(iniFileName);
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
    utils::ConfigFile ini(iniFileName);
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
