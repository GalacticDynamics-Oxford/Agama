#include "potential_factory.h"
#include "particles_io.h"
#include "utils.h"
#include "potential_analytic.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_ferrers.h"
#include "potential_galpot.h"
#include "potential_perfect_ellipsoid.h"
#include "potential_sphharm.h"
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <map>

namespace potential {

const BaseDensity* createDensity(const ConfigPotential& config)
{
    switch(config.densityType) 
    {
    case PT_DEHNEN: 
        return new Dehnen(config.mass, config.scalerad, config.q, config.p, config.gamma); 
#if 0
    case PT_PLUMMER:
        return new Plummer(config->mass, config->scalerad, config->q, config->p);
    case PT_ISOCHRONE:
        return new CDensityIsochrone(config->mass, config->scalerad, config->q, config->p);
    case PT_PERFECTELLIPSOID:
        return new CDensityPerfectEllipsoid(config->mass, config->scalerad, config->q, config->p); 
    case PT_NFW: {
        double concentration = std::max<double>(config->scalerad2/config->scalerad, 1.0);
        return new CDensityNFW(config->mass, config->scalerad, config->q, config->p, concentration);
    }
    case PT_SERSIC:
        return new CDensitySersic(config->mass, config->scalerad, config->q, config->p, config->sersicIndex);
    case PT_EXPDISK:
        return new CDensityExpDisk(config->mass, config->scalerad, fabs(config->scalerad2),
            config->scalerad2>0 ? CDensityExpDisk::ED_EXP : CDensityExpDisk::ED_SECH2);
#endif
    case PT_FERRERS:
        return new Ferrers(config.mass, config.scalerad, config.q, config.p);
    case PT_MIYAMOTONAGAI:
        return new MiyamotoNagai(config.mass, config.scalerad, config.scalerad2);
    default:
        throw std::invalid_argument("Unknown density type");
    }
}

const BasePotential* createPotential(ConfigPotential& config)
{
    const BasePotential* potential=NULL;
    switch(config.potentialType)
    {
    case PT_LOG:
        potential = new Logarithmic(config.mass, config.scalerad, config.q, config.p);
        break;  // NB: it's not really 'mass' here but 'sigma'
    case PT_HARMONIC:
        potential = new Harmonic(config.mass, config.q, config.p);
        break;  // NB: it's not really 'mass' here but 'Omega'
    case PT_DEHNEN:
        potential = new Dehnen(config.mass, config.scalerad, config.q, config.p, config.gamma);
        break;
    case PT_MIYAMOTONAGAI:
        potential = new MiyamotoNagai(config.mass, config.scalerad, config.scalerad2);
        break;
    case PT_FERRERS:
        potential = new Ferrers(config.mass, config.scalerad, config.q, config.p); 
        break;
#if 0
    case PT_SCALEFREE:
        potential=new CPotentialScaleFree(config->mass, config->scalerad, config->q, config->p, config->Gamma);
        break;
    case PT_SCALEFREESH:
        if(config->DensityType==PT_COEFS)
            potential=readPotential(config);   // load coefs from a text file
        else
            potential=new CPotentialScaleFreeSH(config->mass, config->scalerad, config->q, config->p, config->Gamma, config->numCoefsAngular);
        break;
    case PT_NB:
            potential=readPotential(config);
        break;
#endif
    case PT_SPLINE:
    case PT_BSE:
    case PT_CYLSPLINE: {
        // two possible variants: either init from analytic density model or from Nbody/coef/massmodel file
        if( config.densityType == PT_NB || 
            config.densityType == PT_ELLIPSOIDAL || 
            config.densityType == PT_MGE || 
            config.densityType == PT_COEFS )
        {
            potential = readPotential(config);
        } else {
            // create a temporary instance of density model, use it for computing expansion coefs
            const BaseDensity* densModel = createDensity(config);
            switch(config.potentialType) {
            case PT_BSE:
                potential = new BasisSetExp(
                    config.alpha, config.numCoefsRadial, config.numCoefsAngular, *densModel);
                break;
            case PT_SPLINE: {
                if(config.splineRMin>0 && config.splineRMax>0)
                {
                    std::vector<double> grid;
                    math::createNonuniformGrid(config.numCoefsRadial+1, 
                        config.splineRMin, config.splineRMax, true, grid);
                    potential = new SplineExp(
                        config.numCoefsRadial, config.numCoefsAngular, *densModel, &grid);
                } else  // no user-supplied grid extent -- will be assigned by default
                    potential = new SplineExp(
                        config.numCoefsRadial, config.numCoefsAngular, *densModel);
                break;
            }
            case PT_CYLSPLINE: {
                if( config.densityType == PT_DEHNEN || 
                    config.densityType == PT_FERRERS ||
                    config.densityType == PT_MIYAMOTONAGAI ) 
                {   // use potential for initialization, without intermediate DirectPotential step
                    potential = new CylSplineExp(
                        config.numCoefsRadial, config.numCoefsVertical, config.numCoefsAngular,
                        // explicitly type cast to BasePotential to call an appropriate constructor
                        dynamic_cast<const BasePotential&>(*densModel),
                        config.splineRMin, config.splineRMax, config.splineZMin, config.splineZMax);
                } else {
                    potential = new CylSplineExp(
                        config.numCoefsRadial, config.numCoefsVertical, config.numCoefsAngular, 
                        *densModel, 
                        config.splineRMin, config.splineRMax, config.splineZMin, config.splineZMax);
                }
                break;
            }
            default: ;
            }
            delete densModel;
        }
        break;
    }
    default: ;
    }
    if(!potential)
        throw std::invalid_argument("Unknown potential type");
    config.potentialType = getPotentialType(*potential);
    config.symmetryType  = potential->symmetry();
    return potential;
}

/* ------- auxiliary function to create potential of a given type from a set of point masses ------ */
template<typename ParticleT> 
const BasePotential* createPotentialFromPoints(const ConfigPotential& config, 
    const particles::PointMassArray<ParticleT>& points)
{
    switch(config.potentialType) {
    case PT_SPLINE: {
        if(config.splineRMin>0 && config.splineRMax>0) 
        {
            std::vector<double> radii;
            math::createNonuniformGrid(config.numCoefsRadial+1, config.splineRMin, 
                config.splineRMax, true, radii);
            return new SplineExp(config.numCoefsRadial, config.numCoefsAngular, 
                points, config.symmetryType, config.splineSmoothFactor, &radii);
        } else
            return new SplineExp(config.numCoefsRadial, config.numCoefsAngular, 
                points, config.symmetryType, config.splineSmoothFactor);
    }
    case PT_CYLSPLINE:
        return new CylSplineExp(config.numCoefsRadial, config.numCoefsVertical,
            config.numCoefsAngular, points, config.symmetryType, 
            config.splineRMin, config.splineRMax,
            config.splineZMin, config.splineZMax);
    case PT_BSE:
        return new BasisSetExp(config.alpha, config.numCoefsRadial, 
            config.numCoefsAngular, points, config.symmetryType);
/*    else if(config.potentialType==PT_BSECOMPACT)
        pot = new CPotentialBSECompact(config.Rmax, config.numCoefsRadial, 
            config.numCoefsAngular, *points, config.symmetry);
    else if(config.potentialType==PT_SPHERICAL) {
        if(config.splineRMin>0 && config.splineRMax>0) 
        {
            std::vector<double> radii;
            createNonuniformGrid(config.numCoefsRadial+1, config.splineRMin, 
                config.splineRMax, true, &radii);
            pot = new CPotentialSpherical(*points, config.splineSmoothFactor, &radii);
        } else
            pot = new CPotentialSpherical(*points, config.splineSmoothFactor);
    }*/
    default:
        throw std::invalid_argument(std::string("Unknown potential type in createPotentialFromPoints: ")
            + getPotentialNameByType(config.potentialType));
    }
}
// instantiations
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosCar>& points);
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosVelCar>& points);
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosCyl>& points);
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosVelCyl>& points);
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosSph>& points);
template const BasePotential* createPotentialFromPoints(
    const ConfigPotential& config, const particles::PointMassArray<coord::PosVelSph>& points);


// attempt to load coefficients stored in a text file
static const BasePotential* createPotentialExpFromCoefs(ConfigPotential& config)
{        
    std::ifstream strm(config.fileName.c_str(), std::ios::in);
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strm, buffer);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsRadial = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    double param = utils::convertToDouble(fields[0]);   // meaning of this parameter depends on potential type
    if(/*ncoefsRadial==0 || ncoefsAngular==0 || */  // zero values are possible, means just a single term in expansion
        (config.potentialType == PT_BSE && param<0.5) || 
        //(ptype == PT_BSECOMPACT && param<=0) || 
        //(ptype == PT_SCALEFREESH && (param<0 || param>2 || ncoefsRadial!=0)) ||
        (config.potentialType == PT_SPLINE && ncoefsRadial<4) ) 
        ok = false;
    std::vector< std::vector<double> > coefs;
    std::vector< double > radii;
    //if(ncoefsRadial>MAX_NCOEFS_RADIAL) ncoefsRadial=MAX_NCOEFS_RADIAL;
    //if(ncoefsAngular>MAX_NCOEFS_ANGULAR) ncoefsAngular=MAX_NCOEFS_ANGULAR;
    while(ok && std::getline(strm, buffer))  // time, ignored
    {
        std::getline(strm, buffer);  // comments, ignored
        radii.clear();
        coefs.clear();
        for(unsigned int n=0; ok && n<=ncoefsRadial; n++)
        {
            std::getline(strm, buffer);
            utils::splitString(buffer, "# \t", fields);
            radii.push_back(utils::convertToDouble(fields[0]));
            // for BSE this field is basis function index, for spline the radii should be in increasing order
            if( (config.potentialType == PT_BSE && radii.back()!=n) || 
                (config.potentialType == PT_SPLINE && n>0 && radii.back()<=radii[n-1]) ) 
                ok = false;
            coefs.push_back( std::vector<double>() );
            for(int l=0; l<=static_cast<int>(ncoefsAngular); l++)
                for(int m=-l; m<=l; m++)
                {
                    unsigned int fi=1+l*(l+1)+m;
                    coefs.back().push_back( fi<fields.size() ? utils::convertToDouble(fields[fi]) : 0);
                }
        }
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") +
            getPotentialNameByType(config.potentialType)+
            " coefs from file "+config.fileName);
    const BasePotential* pot = NULL;
    switch(config.potentialType)
    {
    case PT_BSE: 
        config.alpha=param;
        pot = new BasisSetExp(/*Alpha*/param, coefs); 
        break;
    case PT_SPLINE:
        pot = new SplineExp(radii, coefs); 
        break;
    default:
        throw std::invalid_argument(std::string("Unknown potential type to load: ") +
            getPotentialNameByType(config.potentialType));
    }
    config.numCoefsRadial  = ncoefsRadial; 
    config.numCoefsAngular = ncoefsAngular;
    config.symmetryType    = pot->symmetry();
    config.densityType     = PT_COEFS;
    return pot;
}

static const BasePotential* createPotentialCylExpFromCoefs(ConfigPotential& config)
{
    std::ifstream strm(config.fileName.c_str(), std::ios::in);
    std::string buffer;
    std::vector<std::string> fields;
    // read coefs for cylindrical spline potential
    bool ok = std::getline(strm, buffer);  // header line
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    size_t size_R = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    size_t ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    size_t size_z = utils::convertToInt(fields[0]);
    ok &= std::getline(strm, buffer).good();  // time, ignored
    ok &= size_R>0 && size_z>0;
    std::vector<double> gridR, gridz;
    std::vector<std::vector<double> > coefs(2*ncoefsAngular+1);
    while(ok && std::getline(strm, buffer) && !strm.eof()) {
        utils::splitString(buffer, "# \t", fields);
        int m = utils::convertToInt(fields[0]);  // m (azimuthal harmonic index)
        if(m < -static_cast<int>(ncoefsAngular) || m > static_cast<int>(ncoefsAngular))
            ok=false;
        std::getline(strm, buffer);  // radii
        if(gridR.size()==0) {  // read values of R only once
            utils::splitString(buffer, "# \t", fields);
            for(size_t i=1; i<fields.size(); i++)
                gridR.push_back(utils::convertToDouble(fields[i]));
            if(gridR.size() != size_R)
                ok=false;
        }
        gridz.clear();
        coefs[m+ncoefsAngular].assign(size_R*size_z,0);
        for(size_t iz=0; ok && iz<size_z; iz++) {
            std::getline(strm, buffer);
            utils::splitString(buffer, "# \t", fields);
            gridz.push_back(utils::convertToDouble(fields[0]));
            if(iz>0 && gridz.back()<=gridz[iz-1]) 
                ok=false;  // the values of z should be in increasing order
            for(size_t iR=0; iR<size_R; iR++) {
                double val=0;
                if(iR+1<fields.size())
                    val = utils::convertToDouble(fields[iR+1]);
                else
                    ok=false;
                coefs[m+ncoefsAngular][iz*size_R+iR]=val;
            }
        }
    }
    if(!ok)
        throw std::runtime_error(std::string("Error loading potential ") +
            getPotentialNameByType(config.potentialType)+
            " coefs from file "+config.fileName);
    const BasePotential* pot = new CylSplineExp(gridR, gridz, coefs);
    config.numCoefsRadial    = size_R; 
    config.numCoefsAngular   = ncoefsAngular;
    config.numCoefsVertical  = (size_z+1)/2;
    config.symmetryType      = pot->symmetry();
    config.densityType       = PT_COEFS;
    return pot;
}

static const BasePotential* createPotentialExpFromEllMGE(ConfigPotential& config)
{
#if 0
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if((config->PotentialType == CPotential::PT_SPLINE ||
        //config->PotentialType == CPotential::PT_CYLSPLINE || 
        config->PotentialType == CPotential::PT_BSE ||
        config->PotentialType == CPotential::PT_BSECOMPACT) &&
        (config->DensityType == CPotential::PT_ELLIPSOIDAL ||
         config->DensityType == CPotential::PT_MGE))
    {   // attempt to load ellipsoidal or MGE mass model from a text file
        const CDensity* dens=NULL;
        std::getline(strm, buffer);
        if(buffer.substr(0, 11)=="Ellipsoidal") {
            std::vector<double> radii, inmass, axesradii, axesq, axesp;  // for loading Ellipsoidal mass profile
            while(std::getline(strm, buffer) && !strm.eof())
            {
                utils::splitString(buffer, "# \t", fields);
                if(fields.size()>=2 && ((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
                {
                    radii.push_back(utils::convertToDouble(fields[0]));
                    inmass.push_back(utils::convertToDouble(fields[1]));
                    if(fields.size()>=4)  // also has anisotropy parameters
                    {
                        axesradii.push_back(radii.back());
                        axesq.push_back(utils::convertToDouble(fields[2]));
                        axesp.push_back(utils::convertToDouble(fields[3]));
                    }
                }
            }
            if(radii.size()<=2) {
                my_error(FUNCNAME, "Error loading ellipsoidal mass profile from file "+fileName);
                return NULL;
            }
            dens=new CDensityEllipsoidal(radii, inmass, axesradii, axesq, axesp);
        } else 
        if(buffer.substr(0, 3)=="MGE") {
            CDensityMGE::vectorg data;
            while(std::getline(strm, buffer) && !strm.eof())
            {
                utils::splitString(buffer, "# \t", &fields);
                if(fields.size()>=2 && ((fields[0][0]>='0' && fields[0][0]<='9') || fields[0][0]=='-' || fields[0][0]=='+'))
                {
                    data.push_back(CDensityMGE::CGaussian(
                        utils::convertToDouble(fields[0]), utils::convertToDouble(fields[1]), 
                        fields.size()>=3 ? utils::convertToDouble(fields[2]) : 1,
                        fields.size()>=4 ? utils::convertToDouble(fields[3]) : 1));
                }
            }
            if(data.size()==0) {
                my_error(FUNCNAME, "Error loading MGE mass model from file "+fileName);
                return NULL;
            }
            dens=new CDensityMGE(data);
        } else {
            my_error(FUNCNAME, "Unknown density model in file "+fileName);
            return NULL;
        }
        const CPotential* pot=NULL;
        if(config->PotentialType==PT_SPLINE) {
            if(config->splineRMin>0 && config->splineRMax>0) 
            {
                std::vector<double> grid;
                createNonuniformGrid(config->numCoefsRadial+1, config->splineRMin, 
                    config->splineRMax, true, &grid);
                pot = new CPotentialSpline(config->numCoefsRadial, config->numCoefsAngular, dens, &grid);
            } else
                pot = new CPotentialSpline(config->numCoefsRadial, config->numCoefsAngular, dens);
        }
        else if(config->PotentialType==PT_BSE)
            pot = new CPotentialBSE(config->Alpha, config->numCoefsRadial, config->numCoefsAngular, dens);
        else if(config->PotentialType==PT_BSECOMPACT)
            pot = new CPotentialBSE(config->Rmax, config->numCoefsRadial, config->numCoefsAngular, dens);
        delete dens;
        // store coefficients in a text file, later may load this file instead for faster initialization
        if(pot!=NULL)
            writePotential(fileName+coefFileExtension(pot->PotentialType()), pot);
        return pot;
    }   // PT_ELLIPSOIDAL or PT_MGE
#else
    throw std::runtime_error(std::string(getPotentialNameByType(config.potentialType))+" not implemented");
#endif
}

/* ----- reading potential from a text or snapshot file ---- */
const BasePotential* readPotential(ConfigPotential& config)
{
    const std::string& fileName = config.fileName;
    if(fileName.empty()) {
        throw std::runtime_error("readPotential: empty file name");
    }
    std::ifstream strm(fileName.c_str(), std::ios::in);
    if(!strm) {
        throw std::runtime_error("readPotential: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strm, buffer);
    strm.close();
    if(ok && buffer.size()<256) {  // to avoid parsing a binary file as a text
        std::vector<std::string> fields;
        utils::splitString(buffer, "# \t", fields);
        if(fields[0] == "BSEcoefs") {
            config.potentialType = PT_BSE;
            return createPotentialExpFromCoefs(config);
        }
        if(fields[0] == "SHEcoefs") {
            config.potentialType = PT_SPLINE;
            return createPotentialExpFromCoefs(config);
        }
        if(fields[0] == "CylSpline") {
            config.potentialType = PT_CYLSPLINE;
            return createPotentialCylExpFromCoefs(config);
        }
        if(fields[0] == "Ellipsoidal" || fields[0] == "MGE") {
            return createPotentialExpFromEllMGE(config);
        }
    }

    // if the above didn't work, try to load a point mass set from the file
    particles::PointMassArrayCar points;
    particles::readSnapshot(fileName, config.units, points);
    if(points.size()==0)
        throw std::runtime_error("readPotential: error loading N-body snapshot from "+fileName);
    const BasePotential* pot = createPotentialFromPoints(config, points);
    // store coefficients in a text file, later may load this file instead for faster initialization
    writePotential(fileName + getCoefFileExtension(config.potentialType), *pot);
    return pot;
}

/* ------ writing potential expansion coefficients to a text file -------- */

static void writePotentialExp(const std::string &fileName, const BasePotentialSphericalHarmonic& potential)
{
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm) 
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);  // file not writable
    std::vector<double> indices;
    std::vector< std::vector<double> > coefs;
    size_t ncoefsAngular=0;
    switch(getPotentialTypeByName(potential.name()))
    {
    case PT_BSE: {
        const BasisSetExp& potBSE = dynamic_cast<const BasisSetExp&>(potential);
        indices.resize(potBSE.getNumCoefsRadial()+1);
        for(size_t i=0; i<indices.size(); i++) indices[i]=i*1.0;
        potBSE.getCoefs(&coefs);
        assert(coefs.size() == indices.size());
        ncoefsAngular = potBSE.getNumCoefsAngular();
        strm << "BSEcoefs\t#header\n" << 
            potBSE.getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" << 
            potBSE.getAlpha() <<"\t#alpha\n0\t#time\n";
        strm << "#index";
        break; 
    }
    case PT_SPLINE: {
        const SplineExp& potSpline = dynamic_cast<const SplineExp&>(potential);
        potSpline.getCoefs(&indices, &coefs);
        assert(coefs.size() == indices.size());
        assert(indices[0] == 0);  // leftmost radius is 0
        coefs[0].resize(1);     // retain only l=0 term for r=0, the rest is supposed to be zero
        ncoefsAngular = potSpline.getNumCoefsAngular();
        strm << "SHEcoefs\t#header\n" << 
            potSpline.getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" <<
            0 <<"\t#unused\n0\t#time\n";
        strm << "#radius";
        break; 
    }
    default:
        throw std::invalid_argument("Unknown type of potential to write");
    }
    for(int l=0; l<=static_cast<int>(ncoefsAngular); l++)
        for(int m=-l; m<=l; m++)
            strm << "\tl="<<l<<",m="<<m;  // header line
    strm << "\n";
    for(size_t n=0; n<indices.size(); n++)
    {
        strm << indices[n];
        strm << "\t" << std::setprecision(14) << coefs[n][0] << std::setprecision(7);   // leading coeft should be high-accuracy at least for spline potential
        for(size_t i=1; i<coefs[n].size(); i++)
            strm << "\t" << coefs[n][i];
        strm << "\n";
    }
    if(!strm.good())
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);
}

static void writePotentialCylExp(const std::string &fileName, const CylSplineExp& potential)
{
    std::ofstream strm(fileName.c_str(), std::ios::out);
    if(!strm) 
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);  // file not writable
    std::vector<double> gridR, gridz;
    std::vector<std::vector<double> > coefs;
    potential.getCoefs(gridR, gridz, coefs);
    int mmax = coefs.size()/2;
    strm << "CylSpline\t#header\n" << gridR.size() << "\t#size_R\n" << mmax << "\t#m_max\n" <<
        gridz.size() << "\t#size_z\n0\t#time\n" << std::setprecision(16);
    for(int m=0; m<static_cast<int>(coefs.size()); m++) 
        if(coefs[m].size()>0) {
            strm << (m-mmax) << "\t#m\n#z\\R";
            for(size_t iR=0; iR<gridR.size(); iR++)
                strm << "\t" << gridR[iR];
            strm << "\n";
            for(size_t iz=0; iz<gridz.size(); iz++) {
                strm << gridz[iz];
                for(size_t iR=0; iR<gridR.size(); iR++)
                    strm << "\t" << coefs[m][iz*gridR.size()+iR];
                strm << "\n";
            }
        } 
    if(!strm.good())
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);
}

void writePotential(const std::string &fileName, const BasePotential& potential)
{
    switch(getPotentialTypeByName(potential.name())) {
    case PT_BSE:
    case PT_SPLINE:
        writePotentialExp(fileName, dynamic_cast<const BasePotentialSphericalHarmonic&>(potential));
        break;
    case PT_CYLSPLINE:
        writePotentialCylExp(fileName, dynamic_cast<const CylSplineExp&>(potential));
        break;
    default:
        throw std::invalid_argument("Unknown type of potential to write");
    }
}

// load GalPot parameter file

static void swallowRestofLine(std::ifstream& from) {
    char c;
    do {
        from.get(c);
    } while( from.good() && c !='\n');
}

const potential::BasePotential* readGalaxyPotential(const char* filename, const units::InternalUnits& units) {
    std::ifstream strm(filename);
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
        strm>>dp.surfaceDensity >> dp.scaleLength >> dp.scaleHeight >> dp.innerCutoffRadius >> dp.modulationAmplitude;
        swallowRestofLine(strm);
        dp.surfaceDensity *= units.from_Msun_per_Kpc2;
        dp.scaleLength *= units.from_Kpc;
        dp.scaleHeight *= units.from_Kpc;
        dp.innerCutoffRadius *= units.from_Kpc;
        if(strm) diskpars.push_back(dp);
        else ok=false;
    }
    strm>>num;
    swallowRestofLine(strm);
    ok=ok && strm;
    for(int i=0; i<num && ok; i++) {
        SphrParam sp;
        strm>>sp.densityNorm >> sp.axisRatio >> sp.gamma >> sp.beta >> sp.scaleRadius >> sp.outerCutoffRadius;
        swallowRestofLine(strm);
        sp.densityNorm *= units.from_Msun_per_Kpc3;
        sp.scaleRadius *= units.from_Kpc;
        sp.outerCutoffRadius *= units.from_Kpc;
        if(strm) sphrpars.push_back(sp);
        else ok=false;
    }
    return createGalaxyPotential(diskpars, sphrpars);
}
    
//----------------------------------------------------------------------------//
// 'class factory'  (makes correspondence between enum potential and symmetry types and string names)
/// lists all 'true' potentials, i.e. those providing a complete density-potential(-force) pair
typedef std::map<PotentialType, const char*> PotentialNameMapType;

/// lists all analytic density profiles 
/// (including those that don't have corresponding potential, but excluding general-purpose expansions)
typedef std::map<PotentialType, const char*> DensityNameMapType;

/// lists available symmetry types
typedef std::map<SymmetryType,  const char*> SymmetryNameMapType;

PotentialNameMapType PotentialNames;
DensityNameMapType DensityNames;
SymmetryNameMapType SymmetryNames;
bool mapinitialized = false;

/// create a correspondence between names and enum identifiers for potential, density and symmetry types
void initPotentialAndSymmetryNameMap()
{
    PotentialNames.clear();
    PotentialNames[PT_LOG] = Logarithmic::myName();
    PotentialNames[PT_HARMONIC] = Harmonic::myName();
    PotentialNames[PT_DEHNEN] = Dehnen::myName();
//    PotentialNames[PT_SCALEFREE] = CPotentialScaleFree::myName();
//    PotentialNames[PT_SCALEFREESH] = CPotentialScaleFreeSH::myName();
    PotentialNames[PT_BSE] = BasisSetExp::myName();
//    PotentialNames[PT_BSECOMPACT] = CPotentialBSECompact::myName();
    PotentialNames[PT_SPLINE] = SplineExp::myName();
    PotentialNames[PT_CYLSPLINE] = CylSplineExp::myName();
//    PotentialNames[PT_SPHERICAL] = CPotentialSpherical::myName();
    PotentialNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    PotentialNames[PT_FERRERS] = Ferrers::myName();
//    PotentialNames[PT_NB] = CPotentialNB::myName();
    PotentialNames[PT_GALPOT] = "GalPot";

    // list of density models available for BSE and Spline approximation
    DensityNames.clear();
//    DensityNames[PT_NB] = CPotentialNB::myName();   // denotes not the actual tree-code potential, but rather the fact that the density model comes from discrete points in Nbody file
//    DensityNames[PT_ELLIPSOIDAL] = CDensityEllipsoidal::myName();
//    DensityNames[PT_MGE] = CDensityMGE::myName();
    DensityNames[PT_COEFS] = "Coefs";  // denotes that potential expansion coefs are loaded from a text file rather than computed from a density model
    DensityNames[PT_DEHNEN] = Dehnen::myName();
    DensityNames[PT_MIYAMOTONAGAI] = MiyamotoNagai::myName();
    DensityNames[PT_PLUMMER] = Plummer::myName();
//    DensityNames[PT_PERFECTELLIPSOID] = CDensityPerfectEllipsoid::myName();
//    DensityNames[PT_ISOCHRONE] = CDensityIsochrone::myName();
//    DensityNames[PT_EXPDISK] = CDensityExpDisk::myName();
    DensityNames[PT_NFW] = NFW::myName();
//    DensityNames[PT_SERSIC] = CDensitySersic::myName();
    DensityNames[PT_FERRERS] = Ferrers::myName();

    SymmetryNames[ST_NONE]         = "None";
    SymmetryNames[ST_REFLECTION]   = "Reflection";
    SymmetryNames[ST_TRIAXIAL]     = "Triaxial";
    SymmetryNames[ST_AXISYMMETRIC] = "Axisymmetric";
    SymmetryNames[ST_SPHERICAL]    = "Spherical";

    mapinitialized=true;
}
   
const char* getPotentialNameByType(PotentialType type)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    PotentialNameMapType::const_iterator iter=PotentialNames.find(type);
    if(iter!=PotentialNames.end()) 
        return iter->second;
    return "";
}

const char* getDensityNameByType(PotentialType type)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    DensityNameMapType::const_iterator iter=DensityNames.find(type);
    if(iter!=DensityNames.end()) 
        return iter->second;
    return "";
}

const char* getSymmetryNameByType(SymmetryType type)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    SymmetryNameMapType::const_iterator iter=SymmetryNames.find(type);
    if(iter!=SymmetryNames.end()) 
        return iter->second;
    return "";
}

PotentialType getPotentialType(const BaseDensity& d)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    const char* name = d.name();
    for(PotentialNameMapType::const_iterator iter=PotentialNames.begin(); 
        iter!=PotentialNames.end(); 
        iter++)
        if(name == iter->second)   // note that here we compare names by address of the string literal, not by string comparison
            return iter->first;
    return PT_UNKNOWN;
}

PotentialType getPotentialTypeByName(const std::string& PotentialName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    for(PotentialNameMapType::const_iterator iter=PotentialNames.begin(); 
        iter!=PotentialNames.end(); 
        iter++)
        if(utils::stringsEqual(PotentialName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

PotentialType getDensityTypeByName(const std::string& DensityName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    for(DensityNameMapType::const_iterator iter=DensityNames.begin(); 
        iter!=DensityNames.end(); 
        iter++)
        if(utils::stringsEqual(DensityName, iter->second)) 
            return iter->first;
    return PT_UNKNOWN;
}

SymmetryType getSymmetryTypeByName(const std::string& SymmetryName)
{
    if(!mapinitialized) initPotentialAndSymmetryNameMap();
    if(SymmetryName.empty()) 
        return ST_DEFAULT;
    // compare only the first letter (should abandon this simplification 
    // if more than one symmetry types are defined that could start with the same letter)
    for(SymmetryNameMapType::const_iterator iter=SymmetryNames.begin(); 
        iter!=SymmetryNames.end(); 
        iter++)
        if(tolower(SymmetryName[0]) == tolower(iter->second[0])) 
            return iter->first;
    return ST_DEFAULT;
}

const char* getCoefFileExtension(PotentialType pottype)
{
    switch(pottype) {
        case PT_BSE:        return ".coef_bse";
        case PT_BSECOMPACT: return ".coef_bsec";
        case PT_SPLINE:     return ".coef_spl";
        case PT_CYLSPLINE:  return ".coef_cyl";
        case PT_SCALEFREESH:return ".coef_sf";
        case PT_SPHERICAL:  return ".mass";
        case PT_GALPOT:     return ".Tpot";
        default: return "";
    }
}

PotentialType getCoefFileType(const std::string& fileName)
{
    if(utils::endsWithStr(fileName, ".coef_bse"))
        return PT_BSE;
    else if(utils::endsWithStr(fileName, ".coef_bsec"))
        return PT_BSECOMPACT;
    else if(utils::endsWithStr(fileName, ".coef_spl"))
        return PT_SPLINE;
    else if(utils::endsWithStr(fileName, ".coef_cyl"))
        return PT_CYLSPLINE;
    else if(utils::endsWithStr(fileName, ".coef_sf"))
        return PT_SCALEFREESH;
    else if(utils::endsWithStr(fileName, ".mass") || utils::endsWithStr(fileName, ".tab"))
        return PT_SPHERICAL;
    else if(utils::endsWithStr(fileName, ".Tpot"))
        return PT_GALPOT;
    else
        return PT_UNKNOWN;
}

}; // namespace
