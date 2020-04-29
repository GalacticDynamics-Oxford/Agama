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
#include <fstream>
#include <map>

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

    PT_UNKNOWN,      ///< unspecified

    // density interpolators
    PT_DENS_SPHHARM, ///< `DensitySphericalHarmonic`
    PT_DENS_CYLGRID, ///< `DensityAzimuthalHarmonic`

    // generic potential expansions
    PT_MULTIPOLE,    ///< spherical-harmonic expansion:  `Multipole`
    PT_CYLSPLINE,    ///< expansion in azimuthal angle with 2d interpolating splines in (R,z):  `CylSpline`

    // components of GalPot
    PT_DISK,         ///< separable disk density model:  `Disk`
    PT_SPHEROID,     ///< double-power-law 3d density model:  `Spheroid`
    PT_NUKER,        ///< double-power-law surface density profile: `Nuker`
    PT_SERSIC,       ///< Sersic profile:  `Sersic`

    // potentials with infinite extent that can't be used as source density for a potential expansion
    PT_LOG,          ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC,     ///< triaxial simple harmonic oscillator:  `Harmonic`

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
    // parameters of potential expansions
    unsigned int gridSizeR;  ///< number of radial grid points in Multipole and CylSpline potentials
    unsigned int gridSizez;  ///< number of grid points in z-direction for CylSpline potential
    double rmin, rmax;       ///< inner- and outermost grid node radii for Multipole and CylSpline
    double zmin, zmax;       ///< grid extent in z direction for CylSpline
    unsigned int lmax;       ///< number of angular terms in spherical-harmonic expansion
    unsigned int mmax;       ///< number of angular terms in azimuthal-harmonic expansion
    double smoothing;        ///< amount of smoothing in Multipole initialized from an N-body snapshot
    std::string file;        ///< name of file with coordinates of points, or coefficients of expansion
    /// default constructor initializes the fields to some reasonable values
    AllParam() :
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(coord::ST_DEFAULT),
        mass(1.), surfaceDensity(NAN), densityNorm(NAN),
        scaleRadius(1.), scaleHeight(1.), innerCutoffRadius(0.), outerCutoffRadius(INFINITY),
        v0(1.), Omega(1.),
        axisRatioY(1.), axisRatioZ(1.),
        alpha(1.), beta(4.), gamma(1.),
        modulationAmplitude(0.), cutoffStrength(2.), sersicIndex(NAN), W0(NAN), trunc(1.),
        gridSizeR(25), gridSizez(25), rmin(0), rmax(0), zmin(0), zmax(0),
        lmax(6), mmax(6), smoothing(1.)
    {};
};

///@}
/// \name Correspondence between enum potential and symmetry types and string names
//        -------------------------------------------------------------------------
///@{

/// return the type of the potential or density model by its name, or PT_UNKNOWN if unavailable
PotentialType getPotentialTypeByName(const std::string& name)
{
    if(name.empty()) return PT_UNKNOWN;
    if(utils::stringsEqual(name, Logarithmic  ::myName())) return PT_LOG;
    if(utils::stringsEqual(name, Harmonic     ::myName())) return PT_HARMONIC;
    if(utils::stringsEqual(name, NFW          ::myName())) return PT_NFW;
    if(utils::stringsEqual(name, Plummer      ::myName())) return PT_PLUMMER;
    if(utils::stringsEqual(name, Dehnen       ::myName())) return PT_DEHNEN;
    if(utils::stringsEqual(name, Ferrers      ::myName())) return PT_FERRERS;
    if(utils::stringsEqual(name, Isochrone    ::myName())) return PT_ISOCHRONE;
    if(utils::stringsEqual(name, SpheroidParam::myName())) return PT_SPHEROID;
    if(utils::stringsEqual(name, NukerParam   ::myName())) return PT_NUKER;
    if(utils::stringsEqual(name, SersicParam  ::myName())) return PT_SERSIC;
    if(utils::stringsEqual(name, DiskDensity  ::myName())) return PT_DISK;
    if(utils::stringsEqual(name, Multipole    ::myName())) return PT_MULTIPOLE;
    if(utils::stringsEqual(name, CylSpline    ::myName())) return PT_CYLSPLINE;
    if(utils::stringsEqual(name, MiyamotoNagai::myName())) return PT_MIYAMOTONAGAI;
    if(utils::stringsEqual(name, "King"))                  return PT_KING;
    if(utils::stringsEqual(name, OblatePerfectEllipsoid  ::myName())) return PT_PERFECTELLIPSOID;
    if(utils::stringsEqual(name, DensitySphericalHarmonic::myName())) return PT_DENS_SPHHARM;
    if(utils::stringsEqual(name, DensityAzimuthalHarmonic::myName())) return PT_DENS_CYLGRID;
    if(utils::stringsEqual(name, CompositeDensity::myName())) return PT_COMPOSITE_DENSITY;
    if(utils::stringsEqual(name, CompositeCyl    ::myName())) return PT_COMPOSITE_POTENTIAL;
    return PT_UNKNOWN;
}

/// return file extension for writing the coefficients of potential of the given type,
/// or empty string if the potential type is not one of the expansion types
const char* getCoefFileExtension(PotentialType type)
{
    switch(type) {
        case PT_CYLSPLINE:  return ".coef_cyl";
        case PT_MULTIPOLE:  return ".coef_mul";
        case PT_COMPOSITE_DENSITY:
        case PT_COMPOSITE_POTENTIAL: return ".composite";
        default: return "";
    }
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
    int sym = utils::toInt(symmetryName);
    if(sym==0 && symmetryName!="0") {  // it wasn't a valid number either
        utils::msg(utils::VL_WARNING, "getSymmetryTypeByName", "Invalid symmetry type: " + symmetryName);
        sym = coord::ST_DEFAULT;
    }
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

/// return file extension for writing the coefficients of expansion of the given potential
const char* getCoefFileExtension(const std::string& potName) {
    return getCoefFileExtension(getPotentialTypeByName(potName)); }

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
    param.potentialType       = getPotentialTypeByName(kvmap.getString("Type"));
    param.densityType         = getPotentialTypeByName(kvmap.getString("Density"));
    param.symmetryType        = getSymmetryTypeByName (kvmap.getString("Symmetry"));
    param.file                = kvmap.getString("File");
    param.mass                = kvmap.getDouble("Mass", param.mass)
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

    // tweak: if 'type' is Plummer or NFW, but axis ratio is not unity or a cutoff radius is provided,
    // replace it with an equivalent Spheroid model, because the dedicated potential models
    // can only be spherical and non-truncated
    PotentialType type = param.densityType != PT_UNKNOWN ? param.densityType : param.potentialType;
    if( (type == PT_PLUMMER || type == PT_NFW) &&
        (param.axisRatioY != 1 || param.axisRatioZ !=1 || param.outerCutoffRadius!=INFINITY) ) {
        param.alpha = type == PT_PLUMMER ? 2 : 1;
        param.beta  = type == PT_PLUMMER ? 5 : 3;
        param.gamma = type == PT_PLUMMER ? 0 : 1;
        if(param.outerCutoffRadius==0) {
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
    nparam.axisRatioY  = param.axisRatioY;
    nparam.axisRatioZ  = param.axisRatioZ;
    nparam.alpha       = param.alpha;
    nparam.beta        = param.beta;
    nparam.gamma       = param.gamma;
    nparam.scaleRadius = param.scaleRadius;
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
/// \name Factory routines for constructing various Potential classes from data stored in a stream
//        ----------------------------------------------------------------------------------------
///@{

/// attempt to load coefficients of Multipole stored in a text file
PtrPotential readPotentialMultipole(std::istream& strm, const units::ExternalUnits& converter)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int ncoefsRadial = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int ncoefsAngular = utils::toInt(fields[0]);
    unsigned int numTerms = pow_2(ncoefsAngular+1);
    std::vector< std::vector<double> >
        coefsPhi (numTerms, std::vector<double>(ncoefsRadial)),
        coefsdPhi(numTerms, std::vector<double>(ncoefsRadial));
    std::vector< double > radii(ncoefsRadial);
    ok &= std::getline(strm, buffer).good();  // ignored
    ok &= std::getline(strm, buffer) && buffer.find("Phi") != std::string::npos;
    std::getline(strm, buffer);  // header, ignored
    for(int n=0; ok && n<ncoefsRadial; n++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        radii[n] = utils::toDouble(fields[0]);
        for(unsigned int ind=1; ind < std::min<size_t>(fields.size(), 1+numTerms); ind++)
            coefsPhi[ind-1][n] = utils::toDouble(fields[ind]);
    }
    ok &= std::getline(strm, buffer).good();  // empty line
    ok &= std::getline(strm, buffer).good() && buffer.find("dPhi/dr") != std::string::npos;
    ok &= std::getline(strm, buffer).good();  // header, ignored
    for(int n=0; ok && n<ncoefsRadial; n++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        for(unsigned int ind=1; ind < std::min<size_t>(fields.size(), 1+numTerms); ind++)
            coefsdPhi[ind-1][n] = utils::toDouble(fields[ind]);
    }
    if(!ok)
        throw std::runtime_error("Error loading potential");
    math::blas_dmul(converter.lengthUnit, radii);
    for(unsigned int i=0; i<numTerms; i++) {
        math::blas_dmul(pow_2(converter.velocityUnit), coefsPhi[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, coefsdPhi[i]);
    }
    return PtrPotential(new Multipole(radii, coefsPhi, coefsdPhi)); 
}

/// read an array of azimuthal harmonics from text stream:
/// one or more blocks corresponding to each m, where the block is a 2d matrix of values,
/// together with the coordinates in the 1st line and 1st column
bool readAzimuthalHarmonics(std::istream& strm,
    int mmax, unsigned int size_R, unsigned int size_z,
    std::vector<double>& gridR, std::vector<double>& gridz,
    std::vector< math::Matrix<double> > &data)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok=true;
    // total # of harmonics possible, not all of them need to be present in the file
    data.resize(mmax*2+1);
    gridR.resize(size_R);
    while(ok && std::getline(strm, buffer).good() && !strm.eof()) {
        if(buffer.size()==0 || buffer[0] == '\n' || buffer[0] == '\r')
            return ok;  // end block with an empty line
        fields = utils::splitString(buffer, "# \t");
        int m = utils::toInt(fields[0]);  // m (azimuthal harmonic index)
        if(m < -mmax || m > mmax)
            return false;
        std::getline(strm, buffer);  // z-values
        fields = utils::splitString(buffer, "# \t");
        if(fields.size() != size_z+1)   // 0th element is comment
            return false;
        gridz.resize(size_z);
        for(unsigned int iz=0; iz<size_z; iz++) {
            gridz[iz] = utils::toDouble(fields[iz+1]);
            if(iz>0 && gridz[iz]<=gridz[iz-1])
                ok=false;  // the values of z must be in increasing order
        }
        gridR.resize(size_R);
        data[m+mmax]=math::Matrix<double>(size_R, size_z, 0);
        for(unsigned int iR=0; ok && iR<size_R; iR++) {
            ok &= std::getline(strm, buffer).good();
            fields = utils::splitString(buffer, "# \t");
            gridR[iR] = utils::toDouble(fields[0]);
            if(iR>0 && gridR[iR]<=gridR[iR-1])
                ok=false;  // the values of R should be in increasing order
            for(unsigned int iz=0; ok && iz<size_z; iz++) {
                if(iz+1<fields.size())
                    data[m+mmax](iR, iz) = utils::toDouble(fields[iz+1]);
                else
                    ok=false;
            }
        }
    }
    return ok;
}

/// attempt to load coefficients of CylSpline stored in a text file
PtrPotential readPotentialCylSpline(std::istream& strm, const units::ExternalUnits& converter)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_R = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_z = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int mmax = utils::toInt(fields[0]);
    ok &= size_R>0 && size_z>0 && mmax>=0;
    std::vector<double> gridR, gridz;
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    ok &= std::getline(strm, buffer).good() && buffer.find("Phi") != std::string::npos;
    if(ok)
        ok &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, Phi);
    bool ok1 = std::getline(strm, buffer).good();  // empty line
    ok1 &= buffer.find("dPhi/dR") != std::string::npos;
    if(ok1)
        ok1 &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, dPhidR);
    ok1 &= std::getline(strm, buffer).good();  // empty line
    ok1 &= buffer.find("dPhi/dz") != std::string::npos;
    if(ok1)
        ok1 &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, dPhidz);
    ok1 &= dPhidR.size() == Phi.size() && dPhidz.size() == Phi.size();
    if(!ok1) {  // have to live without derivatives...
        dPhidR.clear();
        dPhidz.clear();
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") + CylSpline::myName());
    // convert units
    math::blas_dmul(converter.lengthUnit, gridR);
    math::blas_dmul(converter.lengthUnit, gridz);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(pow_2(converter.velocityUnit), Phi[i]);
        if(!ok1) continue;  // no derivs
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidR[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidz[i]);
    }
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

PtrDensity readDensitySphericalHarmonic(std::istream& strm, const units::ExternalUnits& converter)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int ncoefsRadial = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int ncoefsAngular = utils::toInt(fields[0]);
    unsigned int numTerms = pow_2(ncoefsAngular+1);
    std::getline(strm, buffer);  // unused
    std::vector< std::vector<double> > coefs(numTerms);
    std::vector< double > radii;
    ok &= std::getline(strm, buffer) && buffer.find("rho") != std::string::npos;
    std::getline(strm, buffer);  // header, ignored
    for(int n=0; ok && n<ncoefsRadial; n++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        radii.push_back(utils::toDouble(fields[0]));
        for(unsigned int ind=0; ind<numTerms; ind++)
            coefs[ind].push_back( ind+1<fields.size() ? utils::toDouble(fields[ind+1]) : 0);
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading ") + DensitySphericalHarmonic::myName());
    // convert units
    math::blas_dmul(converter.lengthUnit, radii);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(converter.massUnit/pow_3(converter.lengthUnit), coefs[i]);
    return PtrDensity(new DensitySphericalHarmonic(radii, coefs)); 
}

PtrDensity readDensityAzimuthalHarmonic(std::istream& strm, const units::ExternalUnits& converter)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_R = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_z = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int mmax = utils::toInt(fields[0]);
    ok &= size_R>0 && size_z>0 && mmax>=0;
    std::vector<double> gridR, gridz;
    std::vector< math::Matrix<double> > rho;
    ok &= std::getline(strm, buffer).good();
    ok &= buffer.find("rho") != std::string::npos;
    if(ok)
        ok &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, rho);
    if(!ok)
        throw std::runtime_error(std::string("Error loading ") + DensityAzimuthalHarmonic::myName());
    // convert units
    math::blas_dmul(converter.lengthUnit, gridR);
    math::blas_dmul(converter.lengthUnit, gridz);
    for(unsigned int i=0; i<rho.size(); i++)
        math::blas_dmul(converter.massUnit/pow_3(converter.lengthUnit), rho[i]);
    return PtrDensity(new DensityAzimuthalHarmonic(gridR, gridz, rho));
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


//------ load density or potential expansion coefficients from a text file ------//

PtrDensity readDensity(const std::string& fileName, const units::ExternalUnits& converter)
{
    if(fileName.empty()) {
        throw std::runtime_error("readDensity: empty file name");
    }
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) {
        throw std::runtime_error("readDensity: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strm, buffer).good();
    if(ok && buffer.size()<256) {  // to avoid parsing a binary file as a text
        std::vector<std::string> fields;
        fields = utils::splitString(buffer, "# \t");
        if(fields[0] == DensitySphericalHarmonic::myName()) {
            return readDensitySphericalHarmonic(strm, converter);
        }
        if(fields[0] == DensityAzimuthalHarmonic::myName()) {
            return readDensityAzimuthalHarmonic(strm, converter);
        }
        if(fields[0] == CompositeDensity::myName()) {
            // each line is a name of a file with the given component
            std::vector<PtrDensity> components;
            while(std::getline(strm, buffer).good() && !strm.eof())
                components.push_back(readDensity(buffer, converter));
            return PtrDensity(new CompositeDensity(components));
        }
    }
    throw std::runtime_error("readDensity: cannot find valid density coefficients in file "+fileName);
}

PtrPotential readPotential(const std::string& fileName, const units::ExternalUnits& converter)
{
    if(fileName.empty()) {
        throw std::runtime_error("readPotential: empty file name");
    }
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) {
        throw std::runtime_error("readPotential: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strm, buffer).good();
    if(ok && buffer.size()<256) {  // to avoid parsing a binary file as a text
        std::vector<std::string> fields;
        fields = utils::splitString(buffer, "# \t");
        if(fields[0] == Multipole::myName()) {
            return readPotentialMultipole(strm, converter);
        }
        if(fields[0] == CylSpline::myName()) {
            return readPotentialCylSpline(strm, converter);
        }
        if(fields[0] == CompositeCyl::myName()) {
            // each line is a name of a file with the given component
            std::string::size_type idx = fileName.find_last_of('/');
            // extract the path from the filename, and append it to all dependent filenames
            std::string prefix = idx != std::string::npos ? fileName.substr(0, idx+1) : "";
            std::vector<PtrPotential> components;
            while(std::getline(strm, buffer).good() && !strm.eof())
                components.push_back(readPotential(prefix+buffer, converter));
            return PtrPotential(new CompositeCyl(components));
        }
    }
    throw std::runtime_error("readPotential: cannot find valid potential coefficients in file "+fileName);
}

///@}
/// \name Routines for storing various Density and Potential classes into a stream
//        ------------------------------------------------------------------------
///@{
namespace {

void writeSphericalHarmonics(std::ostream& strm,
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &coefs)
{
    assert(coefs.size()>0);
    int lmax = static_cast<int>(sqrt(coefs.size() * 1.0)-1);
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

void writePotentialMultipole(std::ostream& strm, const Multipole& potMul,
    const units::ExternalUnits& converter)
{
    std::vector<double> radii;
    std::vector< std::vector<double> > Phi, dPhi;
    potMul.getCoefs(radii, Phi, dPhi);
    assert(Phi.size() > 0 && Phi[0].size() == radii.size() && dPhi[0].size() == Phi[0].size());
    // convert units
    math::blas_dmul(1/converter.lengthUnit, radii);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(1/pow_2(converter.velocityUnit), Phi[i]);
        math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhi[i]);
    }
    int lmax = static_cast<int>(sqrt(Phi.size()*1.0)-1);
    strm << Multipole::myName() << "\n" << 
        radii.size() << "\t#n_radial\n" << 
        lmax << "\t#l_max\n0\t#unused\n#Phi\n#radius";
    writeSphericalHarmonics(strm, radii, Phi);
    strm << "\n#dPhi/dr\n#radius";
    writeSphericalHarmonics(strm, radii, dPhi);
}

void writePotentialCylSpline(std::ostream& strm, const CylSpline& potential,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridR, gridz;
    std::vector<math::Matrix<double> > Phi, dPhidR, dPhidz;
    potential.getCoefs(gridR, gridz, Phi, dPhidR, dPhidz);
    strm << CylSpline::myName() << "\n" <<
        gridR.size() << "\t#size_R\n" <<
        gridz.size() << "\t#size_z\n" <<
        Phi.size()/2 << "\t#m_max\n";
    strm << "#Phi\n";
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
    std::vector<double> radii;
    std::vector<std::vector<double> > coefs;
    density.getCoefs(radii, coefs);
    // convert units
    math::blas_dmul(1/converter.lengthUnit, radii);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(1/converter.massUnit*pow_3(converter.lengthUnit), coefs[i]);
    int lmax = static_cast<int>(sqrt(coefs.size()*1.0)-1);
    strm << DensitySphericalHarmonic::myName() << "\n" <<
        radii.size() << "\t#n_radial\n" << 
        lmax << "\t#l_max\n0\t#unused\n#rho\n#radius";
    writeSphericalHarmonics(strm, radii, coefs);
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
    int mmax = coefs.size()/2;
    strm << DensityAzimuthalHarmonic::myName() << "\n" <<
        gridR.size() << "\t#size_R\n" <<
        gridz.size() << "\t#size_z\n" <<
        mmax << "\t#m_max\n";
    strm << "#rho\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, coefs);
}

} // end internal namespace

bool writeDensity(const std::string& fileName, const BaseDensity& dens,
    const units::ExternalUnits& converter)
{
    if(fileName.empty())
        return false;
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm)
        return false;
    PotentialType type = getPotentialTypeByName(dens.name());
    switch(type) {
    case PT_MULTIPOLE:
        writePotentialMultipole(strm, dynamic_cast<const Multipole&>(dens), converter);
        break;
    case PT_CYLSPLINE:
        writePotentialCylSpline(strm, dynamic_cast<const CylSpline&>(dens), converter);
        break;
    case PT_DENS_CYLGRID:
        writeDensityAzimuthalHarmonic(strm, dynamic_cast<const DensityAzimuthalHarmonic&>(dens), converter);
        break;
    case PT_DENS_SPHHARM:
        writeDensitySphericalHarmonic(strm, dynamic_cast<const DensitySphericalHarmonic&>(dens), converter);
        break;
    case PT_COMPOSITE_DENSITY: {
        strm << dens.name() << "\n";
        const CompositeDensity& comp = dynamic_cast<const CompositeDensity&>(dens);
        std::string::size_type idx = fileName.find_last_of('/');
        for(unsigned int i=0; i<comp.size(); i++) {
            std::string fileNameComp = fileName+'_'+utils::toString(i);
            std::string fileNameShort= idx != std::string::npos ? fileNameComp.substr(idx+1) : fileNameComp;
            if(writeDensity(fileNameComp, *comp.component(i), converter))
                strm << fileNameShort << '\n';
        }
        break;
    }
    case PT_COMPOSITE_POTENTIAL: {
        strm << dens.name() << "\n";
        const CompositeCyl& comp = dynamic_cast<const CompositeCyl&>(dens);
        std::string::size_type idx = fileName.find_last_of('/');
        for(unsigned int i=0; i<comp.size(); i++) {
            std::string fileNameComp = fileName+'_'+utils::toString(i);
            std::string fileNameShort= idx != std::string::npos ? fileNameComp.substr(idx+1) : fileNameComp;
            if(writeDensity(fileNameComp, *comp.component(i), converter))
                strm << fileNameShort << '\n';
        }
        break;
    }
    default:
        strm << "Unsupported type: " << dens.name() << "\n";
        return false;
    }
    return strm.good();
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
/// \name Factory routines for creating instances of Density and Potential classes
//        ------------------------------------------------------------------------
///@{

namespace {

/// create potential expansion of a given type from a set of point masses
PtrPotential createPotentialFromParticles(const AllParam& param,
    const particles::ParticleArray<coord::PosCyl>& particles)
{
    switch(param.potentialType) {
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
    switch(param.potentialType)
    {
    case PT_LOG:
        return PtrPotential(new Logarithmic(
            param.v0, param.scaleRadius, param.axisRatioY, param.axisRatioZ));
    case PT_HARMONIC:
        return PtrPotential(new Harmonic(param.Omega, param.axisRatioY, param.axisRatioZ));
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
    default:
        throw std::invalid_argument("Unknown potential type");
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
    switch(param.potentialType)
    {
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
PtrPotential createPotentialExpansion(const AllParam& param, const SourceType& source)
{
    switch(param.potentialType) {
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

/** Read potential coefficients from a text file, or create a potential expansion
    from N-body snapshot contained in a file
*/
PtrPotential readPotentialExpansion(const AllParam& param, const units::ExternalUnits& converter)
{
    if(!utils::fileExists(param.file))
        throw std::runtime_error("File "+param.file+" does not exist");

    // file may contain either coefficients of potential expansion,
    // in which case the potential type is inferred from the first line of the file,
    // or an N-body snapshot, in which case the potential type must be specified.
    try {
        return readPotential(param.file, converter);
    }   // if it contained valid coefs, all is fine
    catch(std::runtime_error&) {}  // ignore error if the file didn't contain valid coefs

    // otherwise the file is assumed to contain an N-body snapshot
    if(param.potentialType != PT_MULTIPOLE && param.potentialType != PT_CYLSPLINE)
        throw std::runtime_error("Must specify the potential expansion type to load an N-body snapshot");

    const particles::ParticleArrayCar particles = particles::readSnapshot(param.file, converter);
    if(particles.size()==0)
        throw std::runtime_error("Error loading N-body snapshot from " + param.file);

    PtrPotential poten = createPotentialFromParticles(param, particles);

    // store coefficients in a text file,
    // later may load this file instead for faster initialization
    writePotential(param.file + getCoefFileExtension(param.potentialType), *poten, converter);
    return poten;
}

}  // end internal namespace

// create elementary density
PtrDensity createDensity(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    AllParam param = parseParam(kvmap, converter);
    if(!param.file.empty())
        return readDensity(param.file, converter);
    // if 'type=...' is not provided but 'density=...' is given, use that value
    if(!kvmap.contains("type") && kvmap.contains("density"))
        param.potentialType = param.densityType;
    return createAnalyticDensity(param);
}

// universal routine for creating a potential from several components
PtrPotential createPotential(
    const std::vector<utils::KeyValueMap>& kvmap,
    const units::ExternalUnits& converter)
{
    if(kvmap.size() == 0)
        throw std::runtime_error("Empty list of potential components");

    // all potential components
    std::vector<PtrPotential> componentsPot;
    // all density components that will contribute to the additional Multipole potential
    std::vector<PtrDensity> componentsDens;

    // isolate the density profiles that are part of GalPot scheme:
    // Disk profile will be represented by one potential component (DiskAnsatz)
    // and two density components that will eventually be supplied to the Multipole potential;
    // Spheroid, Nuker and Sersic profiles will also be added to the Multipole;
    // any other potential components are constructed directly
    for(unsigned int i=0; i<kvmap.size(); i++)
    {
        const AllParam param = parseParam(kvmap[i], converter);
        if(!param.file.empty())
        {
            componentsPot.push_back(readPotentialExpansion(param, converter));
        }
        else switch(param.potentialType) {
        case PT_DISK: {
            DiskParam dparam = parseDiskParam(param);
            // the two parts of disk profile: DiskAnsatz goes to the list of potentials...
            componentsPot.push_back(PtrPotential(new DiskAnsatz(dparam)));
            // ...and gets subtracted from the entire DiskDensity for the list of density components
            componentsDens.push_back(PtrDensity(new DiskDensity(dparam)));
            dparam.surfaceDensity *= -1;  // subtract the density of DiskAnsatz
            componentsDens.push_back(PtrDensity(new DiskAnsatz(dparam)));
            break;
        }
        case PT_SPHEROID: {
            componentsDens.push_back(PtrDensity(new SpheroidDensity(parseSpheroidParam(param))));
            break;
        }
        case PT_NUKER: {
            componentsDens.push_back(PtrDensity(new SpheroidDensity(parseNukerParam(param))));
            break;
        }
        case PT_SERSIC: {
            componentsDens.push_back(PtrDensity(new SpheroidDensity(parseSersicParam(param))));
            break;
        }
        case PT_MULTIPOLE:
        case PT_CYLSPLINE: {
            // create a temporary density or potential model to serve as the source for potential expansion
            AllParam srcpar(param);
            srcpar.potentialType = param.densityType;
            if( param.densityType == PT_UNKNOWN )
                throw std::invalid_argument("Multipole or CylSpline need either a density model or a file");
            if( param.densityType == PT_DEHNEN ||
                param.densityType == PT_FERRERS ||
                param.densityType == PT_MIYAMOTONAGAI )
            {   // use an analytic potential as the source
                componentsPot.push_back(createPotentialExpansion(param, *createAnalyticPotential(srcpar)));
            }
            else
            {   // otherwise use analytic density as the source
                componentsPot.push_back(createPotentialExpansion(param, *createAnalyticDensity(srcpar)));
            }
            break;
        }
        default:  // the remaining alternative is an elementary potential, or an error
            componentsPot.push_back(createAnalyticPotential(param));
        }
    }

    // create an additional Multipole potential if needed
    if(!componentsDens.empty()) {
        PtrDensity totalDens;
        if(componentsDens.size() == 1)
            totalDens = componentsDens[0];
        else
            totalDens.reset(new CompositeDensity(componentsDens));
        componentsPot.push_back(Multipole::create(*totalDens,
            isSpherical   (*totalDens) ? 0 : GALPOT_LMAX,
            isAxisymmetric(*totalDens) ? 0 : GALPOT_MMAX, GALPOT_NRAD));
    }

    assert(componentsPot.size()>0);
    if(componentsPot.size() == 1)
        return componentsPot[0];
    else
        return PtrPotential(new CompositeCyl(componentsPot));
}

// create a potential from a single set of parameters
// (which may still turn into a composite potential if it happened to be one of GalPot things)
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    return createPotential(std::vector<utils::KeyValueMap>(1, kvmap), converter);
}

// create a potential from INI file
PtrPotential createPotential(
    const std::string& iniFileName,
    const units::ExternalUnits& converter)
{
    utils::ConfigFile ini(iniFileName);
    std::vector<std::string> sectionNames = ini.listSections();
    std::vector<utils::KeyValueMap> components;
    for(unsigned int i=0; i<sectionNames.size(); i++)
        if(utils::stringsEqual(sectionNames[i].substr(0,9), "Potential"))
            components.push_back(ini.findSection(sectionNames[i]));
    if(components.size() == 0)
        throw std::runtime_error("INI file does not contain any [Potential] section");
    return createPotential(components, converter);
}

// create a potential expansion from the user-provided source density
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const BaseDensity& dens,
    const units::ExternalUnits& converter)
{
    return createPotentialExpansion(parseParam(kvmap, converter), dens);
}

// create a potential expansion from the user-provided source potential
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const BasePotential& pot,
    const units::ExternalUnits& converter)
{
    return createPotentialExpansion(parseParam(kvmap, converter), pot);
}

// create potential from particles
PtrPotential createPotential(
    const utils::KeyValueMap& kvmap,
    const particles::ParticleArray<coord::PosCyl>& particles,
    const units::ExternalUnits& converter)
{
    return createPotentialFromParticles(parseParam(kvmap, converter), particles);
}

///@}
}  // namespace potential
