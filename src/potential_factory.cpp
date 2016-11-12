#include "potential_factory.h"
#include "potential_analytic.h"
#include "potential_composite.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_ferrers.h"
#include "potential_galpot.h"
#include "potential_multipole.h"
#include "potential_perfect_ellipsoid.h"
#include "potential_sphharm.h"
#include "particles_io.h"
#include "math_core.h"
#include "math_linalg.h"
#include "utils.h"
#include "utils_config.h"
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <map>

namespace potential {

namespace {  // internal definitions and routines

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
    PT_COMPOSITE,    ///< a superposition of multiple potential instances:  `CompositeCyl`

    //  Density models without a corresponding potential
//    PT_ELLIPSOIDAL,  ///< a generalization of spherical mass profile with arbitrary axis ratios:  CDensityEllipsoidal
//    PT_MGE,          ///< Multi-Gaussian expansion:  CDensityMGE
//    PT_SERSIC,       ///< Sersic density profile:  CDensitySersic

    //  Generic potential expansions
    PT_BSE,          ///< basis-set expansion for infinite systems:  `BasisSetExp`
    PT_SPLINE,       ///< [old] spline spherical-harmonic expansion:  `SplineExp`
    PT_CYLSPLINE,    ///< expansion in azimuthal angle with 2d interpolating splines in (R,z):  `CylSpline`
    PT_MULTIPOLE,    ///< spherical-harmonic expansion:  `Multipole`

    //  Components of Walter Dehnen's GalPot
    PT_DISK,         ///< separable disk density model:  `DiskDensity`
    PT_SPHEROID,     ///< two-power-law spheroid density model:  `SpheroidDensity`

    //  Density interpolators
    PT_DENS_SPHHARM, ///< DensitySphericalHarmonic
    PT_DENS_CYLGRID, ///< DensityCylGrid

    //  Potentials with infinite extent that can't be used as source density for a potential expansion
    PT_LOG,          ///< triaxial logaritmic potential:  `Logarithmic`
    PT_HARMONIC,     ///< triaxial simple harmonic oscillator:  `Harmonic`

    //  Analytic potential models that can also be used as source density for a potential expansion
    PT_NFW,          ///< spherical Navarro-Frenk-White profile:  `NFW`
    PT_MIYAMOTONAGAI,///< axisymmetric Miyamoto-Nagai(1975) model:  `MiyamotoNagai`
    PT_DEHNEN,       ///< spherical, axisymmetric or triaxial Dehnen(1993) density model:  `Dehnen`
    PT_FERRERS,      ///< triaxial Ferrers model with finite extent:  `Ferrers`
    PT_PLUMMER,      ///< spherical Plummer model:  `Plummer`
    PT_ISOCHRONE,    ///< spherical isochrone model:  `Isochrone`
    PT_PERFECTELLIPSOID,  ///< oblate axisymmetric Perfect Ellipsoid of Kuzmin/de Zeeuw :  `OblatePerfectEllipsoid`
};

/// structure that contains parameters for all possible potentials
struct ConfigPotential
{
    PotentialType potentialType;      ///< type of the potential
    PotentialType densityType;        ///< density model used for initializing a potential expansion
    coord::SymmetryType symmetryType; ///< degree of symmetry
    double mass;             ///< total mass of the model (not applicable to all potential types)
    double scaleRadius;      ///< scale radius of the model (if applicable)
    double scaleRadius2;     ///< second scale radius of the model (if applicable)
    double axisRatioY, axisRatioZ;   ///< axis ratios of the model (if applicable)
    double gamma;            ///< central cusp slope (for the Dehnen model)
    double sersicIndex;      ///< Sersic index (for the Sersic density profile)
    unsigned int gridSizeR;  ///< number of radial grid points in Multipole and CylSpline potentials
    unsigned int gridSizez;  ///< number of grid points in z-direction for CylSpline potential
    double rmin, rmax;       ///< inner- and outermost grid node radii for Multipole and CylSpline
    double zmin, zmax;       ///< grid extent in z direction for CylSpline
    unsigned int lmax;       ///< number of angular terms in spherical-harmonic expansion
    unsigned int mmax;       ///< number of angular terms in azimuthal-harmonic expansion
    double smoothing;        ///< amount of smoothing in Multipole initialized from an N-body snapshot
    std::string file;        ///< name of file with coordinates of points, or coefficients of expansion
    /// default constructor initializes the fields to some reasonable values
    ConfigPotential() :
        potentialType(PT_UNKNOWN), densityType(PT_UNKNOWN), symmetryType(coord::ST_DEFAULT),
        mass(1.), scaleRadius(1.), scaleRadius2(1.), axisRatioY(1.), axisRatioZ(1.), gamma(1.), sersicIndex(4.),
        gridSizeR(25), gridSizez(25), rmin(0), rmax(0), zmin(0), zmax(0), lmax(6), mmax(6), smoothing(1.)
    {};
};

///@}
/// \name Correspondence between enum potential and symmetry types and string names
//        -------------------------------------------------------------------------
///@{

typedef std::map<PotentialType, const char*> MapNameType;

/// lists all 'true' potentials, i.e. those providing a complete density-potential(-force) pair
static MapNameType PotentialNames;

/// lists all analytic density profiles 
/// (including those that don't have corresponding potential, but excluding general-purpose expansions)
static MapNameType DensityNames;

static bool mapinitialized = false;

/// create a correspondence between names and enum identifiers for potential and density types
static void initPotentialNameMap()
{
    PotentialNames.clear();
    PotentialNames[PT_COMPOSITE] = CompositeCyl::myName();
    PotentialNames[PT_LOG]       = Logarithmic::myName();
    PotentialNames[PT_HARMONIC]  = Harmonic::myName();
    PotentialNames[PT_NFW]       = NFW::myName();
    PotentialNames[PT_PLUMMER]   = Plummer::myName();
    PotentialNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    PotentialNames[PT_DEHNEN]    = Dehnen::myName();
    PotentialNames[PT_FERRERS]   = Ferrers::myName();
    PotentialNames[PT_PERFECTELLIPSOID] = OblatePerfectEllipsoid::myName();
    PotentialNames[PT_BSE]       = BasisSetExp::myName();
    PotentialNames[PT_SPLINE]    = SplineExp::myName();
    PotentialNames[PT_CYLSPLINE] = CylSpline::myName();
    PotentialNames[PT_MULTIPOLE] = Multipole::myName();
    PotentialNames[PT_DISK]      = DiskDensity::myName();
//    PotentialNames[PT_SCALEFREE] = CPotentialScaleFree::myName();
//    PotentialNames[PT_SCALEFREESH] = CPotentialScaleFreeSH::myName();
//    PotentialNames[PT_SPHERICAL] = CPotentialSpherical::myName();

    // list of density models available for BSE and Spline approximation
    DensityNames.clear();
//    DensityNames[PT_ELLIPSOIDAL] = CDensityEllipsoidal::myName();
//    DensityNames[PT_MGE] = CDensityMGE::myName();
//    DensityNames[PT_SERSIC] = CDensitySersic::myName();
    DensityNames[PT_COMPOSITE] = CompositeDensity::myName();
    DensityNames[PT_PLUMMER] = Plummer::myName();
    DensityNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    DensityNames[PT_DEHNEN]  = Dehnen::myName();
    DensityNames[PT_FERRERS] = Ferrers::myName();
    DensityNames[PT_ISOCHRONE] = Isochrone::myName();
    DensityNames[PT_PERFECTELLIPSOID] = OblatePerfectEllipsoid::myName();
    DensityNames[PT_DENS_CYLGRID] = DensityAzimuthalHarmonic::myName();
    DensityNames[PT_DENS_SPHHARM] = DensitySphericalHarmonic::myName();

    mapinitialized=true;
}

/// return the name of the potential of a given type, or empty string if unavailable
static const char* getPotentialNameByType(PotentialType type)
{
    if(!mapinitialized)
        initPotentialNameMap();
    MapNameType::const_iterator iter=PotentialNames.find(type);
    if(iter!=PotentialNames.end()) 
        return iter->second;
    return "";
}

/// return the type of the potential model by its name, or PT_UNKNOWN if unavailable
static PotentialType getPotentialTypeByName(const std::string& PotentialName)
{
    if(!mapinitialized)
        initPotentialNameMap();
    for(MapNameType::const_iterator iter=PotentialNames.begin(); 
        iter!=PotentialNames.end(); 
        ++iter)
        if(utils::stringsEqual(PotentialName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

/// return the type of the density model by its name, or PT_UNKNOWN if unavailable
static PotentialType getDensityTypeByName(const std::string& DensityName)
{
    if(!mapinitialized)
        initPotentialNameMap();
    for(MapNameType::const_iterator iter=DensityNames.begin(); 
        iter!=DensityNames.end(); 
        ++iter)
        if(utils::stringsEqual(DensityName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

/// return file extension for writing the coefficients of potential of the given type,
/// or empty string if the potential type is not one of the expansion types
static const char* getCoefFileExtension(PotentialType pottype)
{
    switch(pottype) {
        case PT_BSE:        return ".coef_bse";
        case PT_SPLINE:     return ".coef_spl";
        case PT_CYLSPLINE:  return ".coef_cyl";
        case PT_MULTIPOLE:  return ".coef_mul";
        case PT_COMPOSITE:  return ".composite";
        default: return "";
    }
}
} // internal namespace

// return the type of symmetry by its name, or ST_DEFAULT if unavailable
coord::SymmetryType getSymmetryTypeByName(const std::string& SymmetryName)
{
    if(SymmetryName.empty()) 
        return coord::ST_DEFAULT;
    // compare only the first letter, case-insensitive
    switch(tolower(SymmetryName[0])) {
        case 's': return coord::ST_SPHERICAL;
        case 'a': return coord::ST_AXISYMMETRIC;
        case 't': return coord::ST_TRIAXIAL;
        case 'n': return coord::ST_NONE;
    }
    // otherwise it could be an integer constant representing the numerical value of sym.type
    int sym = utils::toInt(SymmetryName);
    if(sym==0 && SymmetryName!="0")  // it wasn't a valid number either
        sym = coord::ST_DEFAULT;
    return static_cast<coord::SymmetryType>(sym);
}

// inverse of the above: return a symbolic name or a numerical code of symmetry type
std::string getSymmetryNameByType(coord::SymmetryType type)
{
    switch(type) {
        case coord::ST_NONE:         return "None";
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

/** Parse the potential parameters contained in a text array of "key=value" pairs.
    \param[in] params  is the array of string pairs "key" and "value", for instance,
    created from command-line arguments, or read from an INI file;
    \param[in] converter  is the instance of unit converter for translating the dimensional
    parameters (such as mass or scale radius) into internal units (may be a trivial converter);
    \return    the structure containing all parameters of potential
*/
static ConfigPotential parseParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    ConfigPotential config;
    config.potentialType = getPotentialTypeByName(params.getString("Type"));
    config.densityType   = getDensityTypeByName  (params.getString("Density"));
    config.symmetryType  = getSymmetryTypeByName (params.getString("Symmetry"));
    config.file        = params.getString("File");
    config.mass        = params.getDouble("Mass", config.mass) * conv.massUnit;
    config.axisRatioY  = params.getDoubleAlt("axisRatioY", "p", config.axisRatioY);
    config.axisRatioZ  = params.getDoubleAlt("axisRatioZ", "q", config.axisRatioZ);
    config.scaleRadius = params.getDoubleAlt("scaleRadius", "rscale", config.scaleRadius) * conv.lengthUnit;
    config.scaleRadius2= params.getDoubleAlt("scaleRadius2","scaleHeight",config.scaleRadius2) * conv.lengthUnit;
    config.gamma       = params.getDouble   ("Gamma", config.gamma);
    config.sersicIndex = params.getDouble   ("SersicIndex", config.sersicIndex);
    config.gridSizeR   = params.getInt("gridSizeR", config.gridSizeR);
    config.gridSizez   = params.getInt("gridSizeZ", config.gridSizez);
    config.rmin        = params.getDouble("rmin", config.rmin) * conv.lengthUnit;
    config.rmax        = params.getDouble("rmax", config.rmax) * conv.lengthUnit;
    config.zmin        = params.getDouble("zmin", config.zmin) * conv.lengthUnit;
    config.zmax        = params.getDouble("zmax", config.zmax) * conv.lengthUnit;
    config.lmax        = params.getInt("lmax", config.lmax);
    config.mmax        = params.contains("mmax") ? params.getInt("mmax", config.mmax) : config.lmax;
    config.smoothing   = params.getDouble("smoothing", config.smoothing);
    return config;
}

static DiskParam parseDiskParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    DiskParam config;
    config.surfaceDensity      = params.getDouble("surfaceDensity", config.surfaceDensity)
                               * conv.massUnit / pow_2(conv.lengthUnit);
    config.scaleRadius         = params.getDouble("scaleRadius", config.scaleRadius) * conv.lengthUnit;
    config.scaleHeight         = params.getDouble("scaleHeight", config.scaleHeight) * conv.lengthUnit;
    config.innerCutoffRadius   = params.getDouble("innerCutoffRadius", config.innerCutoffRadius) * conv.lengthUnit;
    config.modulationAmplitude = params.getDouble("modulationAmplitude", config.modulationAmplitude);
    // alternative way: specifying the total model mass instead of surface density at R=0
    if(params.contains("mass") && !params.contains("surfaceDensity")) {
        config.surfaceDensity = 1;
        config.surfaceDensity = params.getDouble("mass") * conv.massUnit / config.mass();
    }
    return config;
};

static SphrParam parseSphrParams(const utils::KeyValueMap& params, const units::ExternalUnits& conv)
{
    SphrParam config;
    config.densityNorm        = params.getDouble("densityNorm", config.densityNorm)
                              * conv.massUnit / pow_3(conv.lengthUnit);
    config.axisRatioY         = params.getDouble("axisRatioY", config.axisRatioY);
    config.axisRatioZ         = params.getDouble("axisRatioZ", config.axisRatioZ);
    config.alpha              = params.getDouble("alpha", config.alpha);
    config.beta               = params.getDouble("beta",  config.beta);
    config.gamma              = params.getDouble("gamma", config.gamma);
    config.scaleRadius        = params.getDouble("scaleRadius", config.scaleRadius) * conv.lengthUnit;
    config.outerCutoffRadius  = params.getDouble("outerCutoffRadius", config.outerCutoffRadius) * conv.lengthUnit;
    // alternative way: specifying the total model mass instead of volume density at r=scaleRadius
    if(params.contains("mass") && !params.contains("densityNorm")) {
        config.densityNorm = 1;
        config.densityNorm = params.getDouble("mass") * conv.massUnit / config.mass();
    }
    return config;
}

///@}
/// \name Factory routines for constructing various Potential classes from data stored in a stream
//        ----------------------------------------------------------------------------------------
///@{

/// attempt to load coefficients of BasisSetExp, SplineExp or Multipole stored in a text file
static PtrPotential readPotentialSphHarmExp(
    std::istream& strm, const units::ExternalUnits& converter, const PotentialType potentialType)
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
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    double param = utils::toDouble(fields[0]);   // meaning of this parameter depends on potential type
    if( (potentialType == PT_BSE && param<0.5) || 
        ((potentialType == PT_SPLINE || potentialType == PT_MULTIPOLE) && ncoefsRadial<4) ) 
        ok = false;
    std::vector< std::vector<double> > coefs, coefsPhi(numTerms), coefsdPhi(numTerms);
    std::vector< double > radii;
    std::getline(strm, buffer);  // time, ignored
    std::getline(strm, buffer);  // comments, ignored
    for(int n=0; ok && n<ncoefsRadial; n++) {
        ok &= std::getline(strm, buffer).good();
        fields = utils::splitString(buffer, "# \t");
        radii.push_back(utils::toDouble(fields[0]));
        // for BSE this field is basis function index, for spline the radii should be in increasing order
        if( (potentialType == PT_BSE && radii.back()!=n) || 
            (potentialType == PT_SPLINE && n>0 && radii.back()<=radii[n-1]) ) 
            ok = false;
        if(potentialType != PT_MULTIPOLE) {
            coefs.push_back( std::vector<double>() );
            for(int l=0; l<=ncoefsAngular; l++)
                for(int m=-l; m<=l; m++) {
                    unsigned int fi=1+l*(l+1)+m;
                    coefs.back().push_back( fi<fields.size() ? utils::toDouble(fields[fi]) : 0);
                }
        } else {  // for Multipole the indexing scheme is reversed
            for(unsigned int ind=0; ind<numTerms; ind++)
                coefsPhi[ind].push_back( ind+1<fields.size() ? utils::toDouble(fields[ind+1]) : 0);
        }
    }
    if(potentialType == PT_MULTIPOLE) {
        ok &= std::getline(strm, buffer).good();  // "dPhi/dR", ignored
        ok &= std::getline(strm, buffer).good();  // header, ignored
        for(int n=0; ok && n<ncoefsRadial; n++) {
            ok &= std::getline(strm, buffer).good();
            fields = utils::splitString(buffer, "# \t");
            for(unsigned int ind=0; ind<numTerms; ind++)
                coefsdPhi[ind].push_back( ind+1<fields.size() ? utils::toDouble(fields[ind+1]) : 0);
        }
    }    
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") +
            getPotentialNameByType(potentialType));
    math::blas_dmul(converter.lengthUnit, radii);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(pow_2(converter.velocityUnit), coefs[i]);
    for(unsigned int i=0; i<coefsPhi.size(); i++) {
        math::blas_dmul(pow_2(converter.velocityUnit), coefsPhi[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, coefsdPhi[i]);
    }        
    switch(potentialType)
    {
    case PT_BSE: 
        return PtrPotential(new BasisSetExp(/*Alpha*/param, coefs)); 
    case PT_SPLINE:
        return PtrPotential(new SplineExp(radii, coefs)); 
    case PT_MULTIPOLE:
        return PtrPotential(new Multipole(radii, coefsPhi, coefsdPhi)); 
    default:
        throw std::invalid_argument(std::string("Unknown potential type to load: ") +
            getPotentialNameByType(potentialType));
    }
}

/// read an array of azimuthal harmonics from text stream:
/// one or more blocks corresponding to each m, where the block is a 2d matrix of values,
/// together with the coordinates in the 1st line and 1st column
static bool readAzimuthalHarmonics(std::istream& strm,
    int mmax, unsigned int size_R, unsigned int size_z,
    std::vector<double>& gridR, std::vector<double>& gridz,
    std::vector< math::Matrix<double> > &data)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok=true;
    // total # of harmonics possible, not all of them need to be present in the file
    data.resize(mmax*2+1);
    while(ok && std::getline(strm, buffer).good() && !strm.eof()) {
        if(buffer.size()==0 || buffer[0] == '\n' || buffer[0] == '\r')
            return ok;  // end block with an empty line
        fields = utils::splitString(buffer, "# \t");
        int m = utils::toInt(fields[0]);  // m (azimuthal harmonic index)
        if(m < -mmax || m > mmax)
            ok=false;
        std::getline(strm, buffer);  // radii
        if(gridR.size()==0) {  // read values of R only once
            fields = utils::splitString(buffer, "# \t");
            for(unsigned int i=1; i<fields.size(); i++) {
                gridR.push_back(utils::toDouble(fields[i]));
                if(i>1 && gridR[i-1]<=gridR[i-2])
                    ok=false;  // the values must be in increasing order
            }
        }
        ok &= gridR.size() == size_R;
        gridz.clear();
        data[m+mmax]=math::Matrix<double>(size_R, size_z, 0);
        for(unsigned int iz=0; ok && iz<size_z; iz++) {
            ok &= std::getline(strm, buffer).good();
            fields = utils::splitString(buffer, "# \t");
            gridz.push_back(utils::toDouble(fields[0]));
            if(iz>0 && gridz.back()<=gridz[iz-1]) 
                ok=false;  // the values of z should be in increasing order
            for(unsigned int iR=0; iR<size_R; iR++) {
                double val=0;
                if(iR+1<fields.size())
                    val = utils::toDouble(fields[iR+1]);
                else
                    ok=false;
                data[m+mmax](iR, iz) = val;
            }
        }
    }
    return ok;
}

/// attempt to load coefficients of CylSpline stored in a text file
static PtrPotential readPotentialCylSpline(std::istream& strm, const units::ExternalUnits& converter)
{
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_R = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    int mmax = utils::toInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    fields = utils::splitString(buffer, "# \t");
    unsigned int size_z = utils::toInt(fields[0]);
    ok &= size_R>0 && size_z>0 && mmax>=0;
    std::vector<double> gridR, gridz;
    std::vector< math::Matrix<double> > Phi, dPhidR, dPhidz;
    ok &= std::getline(strm, buffer).good();
    ok &= buffer.find("Phi") != std::string::npos;
    if(ok)
        ok &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, Phi);
    bool ok1 = std::getline(strm, buffer).good();
    ok1 &= buffer.find("dPhi/dR") != std::string::npos;
    if(ok1)
        ok1 &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, dPhidR);
    ok1 &= std::getline(strm, buffer).good();
    ok1 &= buffer.find("dPhi/dz") != std::string::npos;
    if(ok1)
        ok1 &= readAzimuthalHarmonics(strm, mmax, size_R, size_z, gridR, gridz, dPhidz);
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
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidR[i]);
        math::blas_dmul(pow_2(converter.velocityUnit)/converter.lengthUnit, dPhidz[i]);
    }
    return PtrPotential(new CylSpline(gridR, gridz, Phi, dPhidR, dPhidz));
}

}  // end internal namespace

// Main routine: load potential expansion coefficients from a text file
PtrPotential readPotential(const std::string& fileName, const units::ExternalUnits& converter)
{
    if(fileName.empty()) {
        throw std::runtime_error("readPotentialCoefs: empty file name");
    }
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) {
        throw std::runtime_error("readPotentialCoefs: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strm, buffer).good();
    if(ok && buffer.size()<256) {  // to avoid parsing a binary file as a text
        std::vector<std::string> fields;
        fields = utils::splitString(buffer, "# \t");
        if(fields[0] == BasisSetExp::myName()) {
            return readPotentialSphHarmExp(strm, converter, PT_BSE);
        }
        if(fields[0] == SplineExp::myName()) {
            return readPotentialSphHarmExp(strm, converter, PT_SPLINE);
        }
        if(fields[0] == Multipole::myName()) {
            return readPotentialSphHarmExp(strm, converter, PT_MULTIPOLE);
        }
        if(fields[0] == CylSpline::myName()) {
            return readPotentialCylSpline(strm, converter);
        }
        if(fields[0] == CompositeCyl::myName()) {
            // each line is a name of a file with the given component
            std::vector<PtrPotential> components;
            while(ok && std::getline(strm, buffer).good() && !strm.eof())
                components.push_back(readPotential(buffer, converter));
            return PtrPotential(new CompositeCyl(components));
        }
    }
    throw std::runtime_error("readPotentialCoefs: cannot find "
        "valid potential coefficients in file "+fileName);
}

///@}
/// \name Routines for storing various Density and Potential classes into a stream
//        ------------------------------------------------------------------------
///@{
namespace {

template<bool MULTIPOLE_INDEXING_ORDER>
static void writeSphericalHarmonics(std::ostream& strm,
    const std::vector<double> &radii,
    const std::vector< std::vector<double> > &coefs)
{
    assert(coefs.size()>0);
    int lmax = static_cast<int>(sqrt((MULTIPOLE_INDEXING_ORDER ? coefs.size() : coefs[0].size()) * 1.0)-1);
    for(int l=0; l<=lmax; l++)
        for(int m=-l; m<=l; m++)
            strm << "\tl="<<l<<",m="<<m;  // header line
    strm << '\n' << std::setprecision(16);
    for(unsigned int n=0; n<radii.size(); n++) {
        strm << radii[n];
        if(MULTIPOLE_INDEXING_ORDER) {
            for(unsigned int i=0; i<coefs.size(); i++)
                strm << '\t' << (n<coefs[i].size() ? coefs[i][n] : 0.);
        } else {
            for(unsigned int i=0; i<coefs[n].size(); i++)
                strm << '\t' << coefs[n][i];
        }
        strm << '\n';
    }
}

static void writeAzimuthalHarmonics(std::ostream& strm,
    const std::vector<double>& gridR,
    const std::vector<double>& gridz,
    const std::vector< math::Matrix<double> >& data)
{
    int mmax = (static_cast<int>(data.size())-1)/2;
    assert(mmax>=0);
    for(int mm=0; mm<static_cast<int>(data.size()); mm++) 
        if(data[mm].rows()*data[mm].cols()>0) {
            strm << (-mmax+mm) << "\t#m\n#z\\R";
            for(unsigned int iR=0; iR<gridR.size(); iR++)
                strm << "\t" << gridR[iR];
            strm << "\n";
            for(unsigned int iz=0; iz<gridz.size(); iz++) {
                strm << gridz[iz];
                for(unsigned int iR=0; iR<gridR.size(); iR++)
                    strm << "\t" << data[mm](iR, iz);
                strm << "\n";
            }
        }
}
    
static void writePotentialBSE(std::ostream& strm, const BasisSetExp& potBSE,
    const units::ExternalUnits& converter)
{
    std::vector<double> indices;
    std::vector< std::vector<double> > coefs;
    indices.resize(potBSE.getNumCoefsRadial()+1);
    for(size_t i=0; i<indices.size(); i++) indices[i]=i*1.0;
    potBSE.getCoefs(coefs);
    assert(coefs.size() == indices.size());
    // convert units
    for(unsigned int i=0; i<coefs.size(); i++) {
        math::blas_dmul(1/pow_2(converter.velocityUnit), coefs[i]);
    }
    int lmax = potBSE.getNumCoefsAngular();
    strm << BasisSetExp::myName() << "\t#header\n" << 
        (potBSE.getNumCoefsRadial()+1) << "\t#n_radial\n" << 
        lmax << "\t#l_max\n" << 
        potBSE.getAlpha() <<"\t#alpha\n0\t#time\n#index";
    writeSphericalHarmonics<false>(strm, indices, coefs);
}

static void writePotentialSpline(std::ostream& strm, const SplineExp& potSpline,
    const units::ExternalUnits& converter)
{
    std::vector<double> radii;
    std::vector< std::vector<double> > coefs;
    potSpline.getCoefs(radii, coefs);
    assert(radii[0] == 0 && coefs.size() == radii.size());
    coefs[0].resize(1);       // retain only l=0 term for r=0, the rest is supposed to be zero
    // convert units
    math::blas_dmul(1/converter.lengthUnit, radii);
    for(unsigned int i=0; i<coefs.size(); i++) {
        math::blas_dmul(1/pow_2(converter.velocityUnit), coefs[i]);
    }
    int lmax = potSpline.getNumCoefsAngular();
    strm << SplineExp::myName() << "\t#header\n" << 
        (potSpline.getNumCoefsRadial()+1) << "\t#n_radial\n" << 
        lmax << "\t#l_max\n" <<
        0 <<"\t#unused\n0\t#time\n#radius";
    writeSphericalHarmonics<false>(strm, radii, coefs);
}

static void writePotentialMultipole(std::ostream& strm, const Multipole& potMul,
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
    strm << Multipole::myName() << "\t#header\n" << 
        Phi[0].size() << "\t#n_radial\n" << 
        lmax << "\t#l_max\n" <<
        0 <<"\t#unused\nPhi\n#radius";
    writeSphericalHarmonics<true>(strm, radii, Phi);
    strm << "dPhi/dr\n#radius";
    writeSphericalHarmonics<true>(strm, radii, dPhi);
}

static void writePotentialCylSpline(std::ostream& strm, const CylSpline& potential,
    const units::ExternalUnits& converter)
{
    std::vector<double> gridR, gridz;
    std::vector<math::Matrix<double> > Phi, dPhidR, dPhidz;
    potential.getCoefs(gridR, gridz, Phi, dPhidR, dPhidz);
    // convert units
    math::blas_dmul(1/converter.lengthUnit, gridR);
    math::blas_dmul(1/converter.lengthUnit, gridz);
    for(unsigned int i=0; i<Phi.size(); i++) {
        math::blas_dmul(1/pow_2(converter.velocityUnit), Phi[i]);
        math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhidR[i]);
        math::blas_dmul(1/pow_2(converter.velocityUnit)*converter.lengthUnit, dPhidz[i]);
    }
    int mmax = Phi.size()/2;
    strm << CylSpline::myName() << "\t#header\n" << gridR.size() << "\t#size_R\n" << mmax << "\t#m_max\n" <<
        gridz.size() << "\t#size_z\n" << std::setprecision(12);
    strm << "#Phi\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, Phi);
    strm << "\n#dPhi/dR\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, dPhidR);
    strm << "\n#dPhi/dz\n";
    writeAzimuthalHarmonics(strm, gridR, gridz, dPhidz);
}

static void writeDensitySphericalHarmonic(std::ostream& strm, const DensitySphericalHarmonic& density,
    const units::ExternalUnits& converter)
{
    std::vector<double> radii;
    std::vector<std::vector<double> > coefs;
    double innerSlope, outerSlope;
    density.getCoefs(radii, coefs, innerSlope, outerSlope);
    // convert units
    math::blas_dmul(1/converter.lengthUnit, radii);
    for(unsigned int i=0; i<coefs.size(); i++)
        math::blas_dmul(1/converter.massUnit*pow_3(converter.lengthUnit), coefs[i]);
    strm << DensitySphericalHarmonic::myName() <<
    "\n#InnerSlope:\t" << innerSlope << "\tOuterSlope:\t" << outerSlope << "\n#radius";
    writeSphericalHarmonics<true>(strm, radii, coefs);
}

static void writeDensityCylGrid(std::ostream& strm, const DensityAzimuthalHarmonic& density,
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
    if(type == PT_UNKNOWN)
        type = getDensityTypeByName(dens.name());
    switch(type) {
    case PT_BSE:
        writePotentialBSE(strm, dynamic_cast<const BasisSetExp&>(dens), converter);
        break;
    case PT_SPLINE:
        writePotentialSpline(strm, dynamic_cast<const SplineExp&>(dens), converter);
        break;
    case PT_MULTIPOLE:
        writePotentialMultipole(strm, dynamic_cast<const Multipole&>(dens), converter);
        break;
    case PT_CYLSPLINE:
        writePotentialCylSpline(strm, dynamic_cast<const CylSpline&>(dens), converter);
        break;
    case PT_DENS_CYLGRID:
        writeDensityCylGrid(strm, dynamic_cast<const DensityAzimuthalHarmonic&>(dens), converter);
        break;
    case PT_DENS_SPHHARM:
        writeDensitySphericalHarmonic(strm, dynamic_cast<const DensitySphericalHarmonic&>(dens), converter);
        break;
    case PT_COMPOSITE: {  // could be either composite density or composite potential
        strm << dens.name() << "\n";
        const CompositeDensity* compDens = NULL;
        const CompositeCyl* compPot = NULL;
        unsigned int numComp = 0;
        if(dens.name() == CompositeDensity::myName()) {
            compDens = dynamic_cast<const CompositeDensity*>(&dens);
            numComp  = compDens->size();
        }
        if(dens.name() == CompositeCyl::myName()) {
            compPot = dynamic_cast<const CompositeCyl*>(&dens);
            numComp = compPot->size();
        }
        assert((compDens!=NULL) ^ (compPot!=NULL));
        for(unsigned int i=0; i<numComp; i++) {
            const BaseDensity* dens=NULL;
            if(compDens)
                dens = compDens->component(i).get();
            else
                dens = compPot->component(i).get();
            assert(dens);
            std::string fileNameComp = fileName+'_'+utils::toString(i);
            if(writeDensity(fileNameComp, *dens, converter))
                strm << fileNameComp << '\n';
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

static void swallowRestofLine(std::ifstream& from) {
    char c;
    do {
        from.get(c);
    } while( from.good() && c !='\n');
}

PtrPotential readGalaxyPotential(const std::string& filename, const units::ExternalUnits& conv) 
{
    std::ifstream strm(filename.c_str());
    if(!strm) 
        throw std::runtime_error("Cannot open file "+std::string(filename));
    std::vector<DiskParam> diskpars;
    std::vector<SphrParam> sphrpars;
    bool ok=true;
    int num;
    strm>>num;
    swallowRestofLine(strm);
    if(num<0 || num>10 || !strm) ok=false;
    for(int i=0; i<num && ok; i++) {
        DiskParam dp;
        strm >> dp.surfaceDensity >> dp.scaleRadius >> dp.scaleHeight >>
            dp.innerCutoffRadius >> dp.modulationAmplitude;
        swallowRestofLine(strm);
        dp.surfaceDensity *= conv.massUnit/pow_2(conv.lengthUnit);
        dp.scaleRadius *= conv.lengthUnit;
        dp.scaleHeight *= conv.lengthUnit;
        dp.innerCutoffRadius *= conv.lengthUnit;
        if(strm) diskpars.push_back(dp);
        else ok=false;
    }
    strm>>num;
    swallowRestofLine(strm);
    ok=ok && strm;
    for(int i=0; i<num && ok; i++) {
        SphrParam sp;
        strm>>sp.densityNorm >> sp.axisRatioZ >> sp.gamma >> sp.beta >>
            sp.scaleRadius >> sp.outerCutoffRadius;
        swallowRestofLine(strm);
        sp.densityNorm *= conv.massUnit/pow_3(conv.lengthUnit);
        sp.scaleRadius *= conv.lengthUnit;
        sp.outerCutoffRadius *= conv.lengthUnit;
        if(strm) sphrpars.push_back(sp);
        else ok=false;
    }
    return createGalaxyPotential(diskpars, sphrpars);
}

///@}
/// \name Factory routines for creating instances of Density and Potential classes
//        ------------------------------------------------------------------------
///@{

namespace {

/// create potential expansion of a given type from a set of point masses
static PtrPotential createPotentialFromParticles(const ConfigPotential& config,
    const particles::ParticleArray<coord::PosCyl>& particles)
{
    switch(config.potentialType) {
    case PT_SPLINE:
        return PtrPotential(new SplineExp(
            config.gridSizeR, config.lmax, 
            particles, config.symmetryType, config.smoothing, 
            config.rmin, config.rmax));
    case PT_BSE:
        return PtrPotential(new BasisSetExp(
            /*config.alpha*/0., config.gridSizeR, config.lmax,
            particles, config.symmetryType));
    case PT_CYLSPLINE:
        return CylSpline::create(particles, config.symmetryType, config.mmax,
            config.gridSizeR, config.rmin, config.rmax,
            config.gridSizez, config.zmin, config.zmax);
    case PT_MULTIPOLE:
        return Multipole::create(particles, config.symmetryType, config.lmax, config.mmax,
            config.gridSizeR, config.rmin, config.rmax,
            config.smoothing);
    default:
        throw std::invalid_argument("Unknown potential expansion type");
    }
}

/** Create a density model according to the parameters. 
    This only deals with finite-mass models, including some of the Potential descendants.
    This function is used within `createPotential()` to construct 
    temporary density models for initializing a potential expansion.
    \param[in] params  contains the parameters (density type, mass, shape, etc.)
    \return    the instance of a class derived from BaseDensity
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular density model
*/
static PtrDensity createAnalyticDensity(const ConfigPotential& params)
{
    switch(params.densityType) 
    {
    case PT_DEHNEN:
        return PtrDensity(new Dehnen(
            params.mass, params.scaleRadius, params.gamma, params.axisRatioY, params.axisRatioZ));
    case PT_PLUMMER:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrDensity(new Plummer(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Plummer is not supported");
    case PT_ISOCHRONE:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrDensity(new Isochrone(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Isochrone is not supported");
    case PT_NFW:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrDensity(new NFW(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Navarro-Frenk-White is not supported");
    case PT_PERFECTELLIPSOID:
        if(params.axisRatioY==1 && params.axisRatioZ<1)
            return PtrDensity(new OblatePerfectEllipsoid(
                params.mass, params.scaleRadius, params.scaleRadius*params.axisRatioZ));
        else
            throw std::invalid_argument("May only create oblate axisymmetric Perfect Ellipsoid model");
    case PT_FERRERS:
        return PtrDensity(new Ferrers(params.mass, params.scaleRadius, params.axisRatioY, params.axisRatioZ));
    case PT_MIYAMOTONAGAI:
        return PtrDensity(new MiyamotoNagai(params.mass, params.scaleRadius, params.scaleRadius2));
    default:
        throw std::invalid_argument("Unknown density type");
    }
}

/** Create an instance of analytic potential model according to the parameters passed. 
    \param[in] params  specifies the potential parameters
    \return    the instance of potential
    \throw     std::invalid_argument exception if the parameters don't make sense,
    or any other exception that may occur in the constructor of a particular potential model
*/
static PtrPotential createAnalyticPotential(const ConfigPotential& params)
{
    switch(params.potentialType)
    {
    case PT_LOG:  // NB: it's not really 'mass' here but 'sigma'
        return PtrPotential(new Logarithmic(params.mass, params.scaleRadius, params.axisRatioY, params.axisRatioZ));
    case PT_HARMONIC:  // NB: it's not really 'mass' here but 'Omega'
        return PtrPotential(new Harmonic(params.mass, params.axisRatioY, params.axisRatioZ));
    case PT_MIYAMOTONAGAI:
        return PtrPotential(new MiyamotoNagai(params.mass, params.scaleRadius, params.scaleRadius2));
    case PT_DEHNEN:
        return PtrPotential(new Dehnen(
            params.mass, params.scaleRadius, params.gamma, params.axisRatioY, params.axisRatioZ));
    case PT_FERRERS:
        return PtrPotential(new Ferrers(params.mass, params.scaleRadius, params.axisRatioY, params.axisRatioZ)); 
    case PT_PLUMMER:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrPotential(new Plummer(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Plummer is not supported");
    case PT_ISOCHRONE:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrPotential(new Isochrone(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Isochrone is not supported");
    case PT_NFW:
        if(params.axisRatioY==1 && params.axisRatioZ==1)
            return PtrPotential(new NFW(params.mass, params.scaleRadius));
        else
            throw std::invalid_argument("Non-spherical Navarro-Frenk-White is not supported");
    case PT_PERFECTELLIPSOID:
        if(params.axisRatioY==1 && params.axisRatioZ<1)
            return PtrPotential(new OblatePerfectEllipsoid(
                params.mass, params.scaleRadius, params.scaleRadius*params.axisRatioZ)); 
        else
            throw std::invalid_argument("May only create oblate axisymmetric Perfect Ellipsoid model");
    default:
        throw std::invalid_argument("Unknown potential type");
    }
}

/** Create an instance of potential expansion class according to the parameters passed in params,
    for the provided source density or potential
    (template parameter SourceType==BaseDensity or BasePotential) */
template<typename SourceType>
static PtrPotential createPotentialExpansion(const ConfigPotential& params, const SourceType& source)
{
    switch(params.potentialType) {
    case PT_BSE: {
        return PtrPotential(new BasisSetExp(
            /*params.alpha*/0., params.gridSizeR, params.lmax, source));
    }
    case PT_SPLINE: {
        return PtrPotential(new SplineExp(
            params.gridSizeR, params.lmax, source,
            params.rmin, params.rmax));
    }
    case PT_CYLSPLINE: {
        return CylSpline::create(source, params.mmax,
            params.gridSizeR, params.rmin, params.rmax,
            params.gridSizez, params.zmin, params.zmax);
    }
    case PT_MULTIPOLE: {
        return Multipole::create(source, params.lmax, params.mmax,
            params.gridSizeR, params.rmin, params.rmax);
    }
    default: throw std::invalid_argument("Unknown potential expansion type");
    }
}

/// determines whether the potential is of an expansion type
static bool isPotentialExpansion(PotentialType type)
{
    return type == PT_SPLINE || type == PT_BSE || type == PT_CYLSPLINE || type == PT_MULTIPOLE;
}

/** Universal routine for creating any elementary (non-composite) potential,
    either an analytic one or a potential expansion constructed from a density model
    or loaded from a text file.
*/
static PtrPotential createAnyPotential(const ConfigPotential& params,
    const units::ExternalUnits& converter)
{
    if(!params.file.empty()) {
        // file may contain either coefficients of potential expansion,
        // in which case the potential type is inferred from the first line of the file,
        // or an N-body snapshot, in which case the potential type must be specified.
        try {
            return readPotential(params.file, converter);
        }   // if it contained valid coefs, all is fine
        catch(std::runtime_error&) {
            // otherwise the file is assumed to contain an N-body snapshot
            if(!isPotentialExpansion(params.potentialType))
                throw std::runtime_error("Must specify the potential expansion type");
            const particles::ParticleArrayCar particles =
                particles::readSnapshot(params.file, converter);
            if(particles.size()==0)
                throw std::runtime_error("Error loading N-body snapshot from " + params.file);
            PtrPotential poten = createPotentialFromParticles(params, particles);
            // store coefficients in a text file, 
            // later may load this file instead for faster initialization
            writePotential( (params.file + 
                getCoefFileExtension(params.potentialType)), *poten, converter);
            return poten;
        }
    }
    // if params did not contain a file name, they must specify the potential type
    if(params.potentialType == PT_UNKNOWN)
        throw std::runtime_error("Must specify the potential type");
    if(isPotentialExpansion(params.potentialType)) {
        // create a temporary density or potential model
        // to serve as the source for potential expansion
        if( params.densityType == PT_DEHNEN || 
            params.densityType == PT_FERRERS ||
            params.densityType == PT_MIYAMOTONAGAI )
        {   // use an analytic potential as the source
            ConfigPotential potparams(params);
            potparams.potentialType  = params.densityType;
            return createPotentialExpansion(params, *createAnalyticPotential(potparams));
        }
        // otherwise use analytic density as the source
        return createPotentialExpansion(params, *createAnalyticDensity(params));
    } else  // elementary potential, or an error
        return createAnalyticPotential(params);
}

}  // end internal namespace

// create elementary density
PtrDensity createDensity(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& converter)
{
    std::string type = kvmap.getString("type");
    if(utils::stringsEqual(type, DiskDensity::myName()))
        return PtrDensity(new DiskDensity(parseDiskParams(kvmap, converter)));
    if(utils::stringsEqual(type, SpheroidDensity::myName()))
        return PtrDensity(new SpheroidDensity(parseSphrParams(kvmap, converter)));
    return createAnalyticDensity(parseParams(kvmap, converter));
}

// create a potential from several components
PtrPotential createPotential(
    const std::vector<utils::KeyValueMap>& kvmap,
    const units::ExternalUnits& converter)
{
    if(kvmap.size() == 0)
        throw std::runtime_error("Empty list of potential components");
    std::vector<PtrPotential> components;

    // first we isolate all components that are part of GalPot
    std::vector<DiskParam> diskParams;
    std::vector<SphrParam> sphrParams;
    std::vector<ConfigPotential> params;
    for(unsigned int i=0; i<kvmap.size(); i++) {
        std::string type = kvmap[i].getString("type");
        if(utils::stringsEqual(type, DiskDensity::myName())) {
            diskParams.push_back(parseDiskParams(kvmap[i], converter));
        } else if(utils::stringsEqual(type, SpheroidDensity::myName())) {
            sphrParams.push_back(parseSphrParams(kvmap[i], converter));
        } else
            params.push_back(parseParams(kvmap[i], converter));
    }
    // create an array of GalPot components if needed
    if(diskParams.size()>0 || sphrParams.size()>0)
        components = createGalaxyPotentialComponents(diskParams, sphrParams);
    // add other components if they exist
    for(unsigned int i=0; i<params.size(); i++) {
        components.push_back(createAnyPotential(params[i], converter));
    }

    assert(components.size()>0);
    if(components.size() == 1)
        return components[0];
    else
        return PtrPotential(new CompositeCyl(components));
}

// create a potential from one component (which may still turn into a composite potential
// if it happened to be one of GalPot things)
PtrPotential createPotential(
    const utils::KeyValueMap& params,
    const units::ExternalUnits& converter)
{
    return createPotential(std::vector<utils::KeyValueMap>(1, params), converter);
}

// create a potential expansion from the user-provided source density
PtrPotential createPotential(
    const utils::KeyValueMap& params, const BaseDensity& dens,
    const units::ExternalUnits& converter)
{
    return createPotentialExpansion(parseParams(params, converter), dens);
}

// create a potential expansion from the user-provided source potential
PtrPotential createPotential(
    const utils::KeyValueMap& params, const BasePotential& pot,
    const units::ExternalUnits& converter)
{
    return createPotentialExpansion(parseParams(params, converter), pot);
}

// create a potential from INI file
PtrPotential createPotential(
    const std::string& iniFileName, const units::ExternalUnits& converter)
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

// create potential from particles
PtrPotential createPotential(const utils::KeyValueMap& params,
    const particles::ParticleArray<coord::PosCyl>& particles, const units::ExternalUnits& converter)
{
    return createPotentialFromParticles(parseParams(params, converter), particles);
}

///@}
}; // namespace
