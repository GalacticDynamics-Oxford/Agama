#include "potential_factory.h"
#include "particles_io.h"
#include "utils.h"
#include "potential_analytic.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_galpot.h"
#include "potential_staeckel.h"
#include "potential_sphharm.h"
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <map>

namespace potential {
#if 0
const CDensity* createDensity(const CConfigPotential* config)
{
    switch(config->DensityType) 
    {
    case PT_DEHNEN: 
        return new CPotentialDehnen(config->mass, config->scalerad, config->q, config->p, config->Gamma); 
    case PT_PLUMMER:
        return new CDensityPlummer(config->mass, config->scalerad, config->q, config->p);
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
    case PT_FERRERS:
        return new CPotentialFerrers(config->mass, config->scalerad, config->q, config->p);
    case PT_MIYAMOTONAGAI:
        return new CPotentialMiyamotoNagai(config->mass, config->scalerad, config->scalerad2);
    case PT_EXPDISK:
        return new CDensityExpDisk(config->mass, config->scalerad, fabs(config->scalerad2),
            config->scalerad2>0 ? CDensityExpDisk::ED_EXP : CDensityExpDisk::ED_SECH2);
    default: return NULL;
    }
}

const CPotential* createPotential(CConfigPotential* config)
{
    const CPotential* potential=NULL;
    switch(config->PotentialType)
    {
    case PT_LOG:
        potential=new CPotentialLog(config->mass, config->q, config->p, config->scalerad);
        break;
    case PT_HARMONIC:
        potential=new CPotentialHarmonic(config->mass, config->q, config->p);
        break;
    case PT_DEHNEN:
        potential=new CPotentialDehnen(config->mass, config->scalerad, config->q, config->p, config->Gamma);
        break;
    case PT_MIYAMOTONAGAI:
        potential=new CPotentialMiyamotoNagai(config->mass, config->scalerad, config->scalerad2);
        break;
    case PT_FERRERS:
        potential = new CPotentialFerrers(config->mass, config->scalerad, config->q, config->p); 
        break;
    case PT_SCALEFREE:
        potential=new CPotentialScaleFree(config->mass, config->scalerad, config->q, config->p, config->Gamma);
        break;
    case PT_SCALEFREESH:
        if(config->DensityType==PT_COEFS)
            potential=readPotential(config);   // load coefs from a text file
        else
            potential=new CPotentialScaleFreeSH(config->mass, config->scalerad, config->q, config->p, config->Gamma, config->numCoefsAngular);
        break;
#ifdef HAVE_GALPOT
    case PT_GALPOT:
#endif
    case PT_NB:
            potential=readPotential(config);
        break;
    case PT_SPLINE:
    case PT_BSE:
    case PT_BSECOMPACT:
    case PT_CYLSPLINE:
    case PT_SPHERICAL:
        // two possible variants: either init from analytic density model or from Nbody/coef/massmodel file
        if( config->DensityType==PT_NB || 
            config->DensityType==PT_ELLIPSOIDAL || 
            config->DensityType==PT_MGE || 
            config->DensityType==PT_COEFS )
        {
            potential=readPotential(config);
        } else {
            // create a temporary instance of density model, use it for computing expansion coefs
            const CDensity* densModel = createDensity(config);
            if(densModel!=NULL) 
            {
                if(config->PotentialType==CPotential::PT_BSE)
                    potential = new CPotentialBSE(
                        config->Alpha, config->numCoefsRadial, config->numCoefsAngular, densModel);
                else if(config->PotentialType==CPotential::PT_BSECOMPACT)
                    potential = new CPotentialBSECompact(
                        config->Rmax, config->numCoefsRadial, config->numCoefsAngular, densModel);
                else if(config->PotentialType==CPotential::PT_SPLINE) {
                    if(config->splineRMin>0 && config->splineRMax>0)
                    {
                        std::vector<double> grid;
                        createNonuniformGrid(config->numCoefsRadial+1, 
                            config->splineRMin, config->splineRMax, true, &grid);
                        potential = new CPotentialSpline(
                            config->numCoefsRadial, config->numCoefsAngular, densModel, &grid);
                    } else  // no user-supplied grid extent -- will be assigned by default
                        potential = new CPotentialSpline(
                            config->numCoefsRadial, config->numCoefsAngular, densModel);
                }
#ifdef HAVE_INTERP2D
                else if(config->PotentialType==CPotential::PT_CYLSPLINE) {
                    if(config->DensityType==PT_DEHNEN || config->DensityType==PT_FERRERS ||
                        config->DensityType==PT_MIYAMOTONAGAI) { // use potential for initialization, without intermediate CPotentialDirect step
                        potential = new CPotentialCylSpline(
                            config->numCoefsRadial, config->numCoefsVertical, config->numCoefsAngular, 
                            static_cast<const CPotential*>(densModel), // explicitly type cast to CPotential to call an appropriate constructor
                            config->splineRMin, config->splineRMax, config->splineZMin, config->splineZMax);
                    } else {
                        potential = new CPotentialCylSpline(
                            config->numCoefsRadial, config->numCoefsVertical, config->numCoefsAngular, densModel, 
                            config->splineRMin, config->splineRMax, config->splineZMin, config->splineZMax);
                    }
                }
#endif
                else if(config->PotentialType==CPotential::PT_SPHERICAL) {
                    const CMassModel* massModel=createMassModel(densModel, config->numCoefsRadial, config->splineRMin, config->splineRMax);
                    if(massModel) { 
                        potential = new CPotentialSpherical(*massModel);
                        delete massModel;
                    }
                }
                else 
                    my_error(FUNCNAME, "Unknown type of potential expansion");
                delete densModel;
            }
        }
        break;
    default: ;
    }
    if(!potential)
        return NULL;  // signal of error
    config->PotentialType = potential->PotentialType();
    config->symmetry  = potential->symmetry();
#ifdef DEBUGPRINT
    my_message(FUNCNAME, 
        "Gamma="+utils::convertToString(potential->getGamma())+", total mass="+utils::convertToString(potential->totalMass()));
#endif
    return potential;
}
#endif

/* ------- auxiliary function to create potential of a given type from a set of point masses ------ */
template<typename CoordT> 
const BasePotential* createPotentialFromPoints(const ConfigPotential& config, 
                                               const particles::PointMassSet<CoordT>& points)
{
    const BasePotential* pot=NULL;
/*    if(config.PotentialType==PT_NB) 
    {
        pot = new CPotentialNB(config.treecodeEps, config.treecodeTheta, *points);
    } 
    else*/ if(config.potentialType == PT_SPLINE)
    {
        if(config.splineRMin>0 && config.splineRMax>0) 
        {
            std::vector<double> radii;
            math::createNonuniformGrid(config.numCoefsRadial+1, config.splineRMin, 
                config.splineRMax, true, radii);
            pot = new SplineExp(config.numCoefsRadial, config.numCoefsAngular, 
                points, config.symmetryType, config.splineSmoothFactor, &radii);
        } else
            pot = new SplineExp(config.numCoefsRadial, config.numCoefsAngular, 
                points, config.symmetryType, config.splineSmoothFactor);
    }
    else if(config.potentialType==PT_CYLSPLINE)
        pot = new CylSplineExp(config.numCoefsRadial, config.numCoefsVertical,
            config.numCoefsAngular, points, config.symmetryType, 
            config.splineRMin, config.splineRMax,
            config.splineZMin, config.splineZMax);
    else if(config.potentialType == PT_BSE)
        pot = new BasisSetExp(config.alpha, config.numCoefsRadial, 
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
    return pot;
}
// instantiations
template const BasePotential* createPotentialFromPoints(const ConfigPotential& config, const particles::PointMassSet<coord::Car>& points);


// attempt to load coefficients stored in a text file
static const BasePotential* createPotentialExpFromCoefs(ConfigPotential& config)
{        
    std::ifstream strmText(config.fileName.c_str(), std::ios::in);
    std::string buffer;
    std::vector<std::string> fields;
    bool ok = std::getline(strmText, buffer);
    ok &= std::getline(strmText, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsRadial = utils::convertToInt(fields[0]);
    ok &= std::getline(strmText, buffer).good();
    utils::splitString(buffer, "# \t", fields);
    unsigned int ncoefsAngular = utils::convertToInt(fields[0]);
    ok &= std::getline(strmText, buffer).good();
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
    while(ok && std::getline(strmText, buffer))  // time, ignored
    {
        std::getline(strmText, buffer);  // comments, ignored
        radii.clear();
        coefs.clear();
        for(unsigned int n=0; ok && n<=ncoefsRadial; n++)
        {
            std::getline(strmText, buffer);
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
        throw std::runtime_error(std::string("Unknown potential type to load: ") +
            getPotentialNameByType(config.potentialType));
    }
    config.numCoefsRadial  = ncoefsRadial; 
    config.numCoefsAngular = ncoefsAngular;
    config.symmetryType = pot->symmetry();
    return pot;
}

static const BasePotential* createPotentialCylExpFromCoefs(ConfigPotential& config)
{
#if 0
    // read coefs for cylindrical spline potential
    if(!std::getline(strmText, buffer)) valid=false;
    utils::splitString(buffer, "# \t", &fields);
    size_t size_R=utils::convertToInt(fields[0]);
    if(!std::getline(strmText, buffer)) valid=false;
    utils::splitString(buffer, "# \t", &fields);
    size_t ncoefsAngular=utils::convertToInt(fields[0]);
    if(!std::getline(strmText, buffer)) valid=false;
    utils::splitString(buffer, "# \t", &fields);
    size_t size_z=utils::convertToInt(fields[0]);
    std::getline(strmText, buffer);  // time, ignored
    if(size_R==0 || size_z==0) valid=false;
    std::vector<double> gridR, gridz;
    std::vector<std::vector<double>> coefs(2*ncoefsAngular+1);
    while(valid && std::getline(strmText, buffer) && !strmText.eof()) {
        utils::splitString(buffer, "# \t", &fields);
        int m=utils::convertToInt(fields[0]);  // m (azimuthal harmonic index)
        if(m<-static_cast<int>(ncoefsAngular) || m>static_cast<int>(ncoefsAngular)) valid=false;
        std::getline(strmText, buffer);  // radii
        if(gridR.size()==0) {  // read values of R only once
            utils::splitString(buffer, "# \t", &fields);
            for(size_t i=1; i<fields.size(); i++)
                gridR.push_back(utils::convertToDouble(fields[i]));
            if(gridR.size()!=size_R) valid=false;
        }
        gridz.clear();
        coefs[m+ncoefsAngular].assign(size_R*size_z,0);
        for(size_t iz=0; valid && iz<size_z; iz++)
        {
            std::getline(strmText, buffer);
            utils::splitString(buffer, "# \t", &fields);
            gridz.push_back(utils::convertToDouble(fields[0]));
            if(iz>0 && gridz.back()<=gridz[iz-1]) 
                valid=false;  // the values of z should be in increasing order
            for(size_t iR=0; iR<size_R; iR++) 
            {
                double val=iR+1<fields.size() ? utils::convertToDouble(fields[iR+1]) : (valid=false, 0);
                coefs[m+ncoefsAngular][iz*size_R+iR]=val;
            }
        }
    }
    if(!valid || (config->PotentialType != CPotential::PT_UNKNOWN && ptype!=config->PotentialType))
    {   // load coefs only if potential type is the same as requested, or if request is not specific
        my_error(FUNCNAME,
            "Error loading potential "+std::string(getPotentialNameByType(config->PotentialType))+
            " coefs from file "+fileName);
        return NULL;
    }
    const CPotential* pot=new CPotentialCylSpline(gridR, gridz, coefs);
    config->numCoefsRadial=size_R; 
    config->numCoefsAngular=ncoefsAngular;
    config->numCoefsVertical=(size_z+1)/2;
    config->symmetry=pot->symmetry();
    return pot;
#endif
    throw std::runtime_error("Not implemented");
}

static const BasePotential* createPotentialExpFromEllMGE(ConfigPotential& config)
{
#if 0
    std::ifstream strmText(fileName.c_str(), std::ios::in);
    if((config->PotentialType == CPotential::PT_SPLINE ||
        //config->PotentialType == CPotential::PT_CYLSPLINE || 
        config->PotentialType == CPotential::PT_BSE ||
        config->PotentialType == CPotential::PT_BSECOMPACT) &&
        (config->DensityType == CPotential::PT_ELLIPSOIDAL ||
         config->DensityType == CPotential::PT_MGE))
    {   // attempt to load ellipsoidal or MGE mass model from a text file
        const CDensity* dens=NULL;
        std::getline(strmText, buffer);
        if(buffer.substr(0, 11)=="Ellipsoidal") {
            std::vector<double> radii, inmass, axesradii, axesq, axesp;  // for loading Ellipsoidal mass profile
            while(std::getline(strmText, buffer) && !strmText.eof())
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
            while(std::getline(strmText, buffer) && !strmText.eof())
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
    throw std::runtime_error("Ellipsoidal/MGE not implemented");
#endif
}

/* ----- reading potential from a text or snapshot file ---- */
const BasePotential* readPotential(ConfigPotential& config)
{
    const std::string& fileName = config.fileName;
    if(fileName.empty()) {
        throw std::runtime_error("readPotential: empty file name");
    }
    std::ifstream strmText(fileName.c_str(), std::ios::in);
    if(!strmText) {
        throw std::runtime_error("readPotential: cannot read from file "+fileName);
    }
    // check header
    std::string buffer;
    bool ok = std::getline(strmText, buffer);
    strmText.close();
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
    particles::PointMassSet<coord::Car> points;
    particles::readSnapshot(fileName, points);
    if(points.size()==0)
        throw std::runtime_error("readPotential: error loading N-body snapshot from "+fileName);
    const BasePotential* pot = createPotentialFromPoints(config, points);
  //  if(config.potentialType != PT_NB)
    {   // store coefficients in a text file, later may load this file instead for faster initialization
        writePotential(fileName + getCoefFileExtension(config.potentialType), *pot);
    }
    return pot;
}

/* ------ writing potential expansion coefficients to a text file -------- */

static void writePotentialExp(const std::string &fileName, const BasePotential& potential)
{
    std::ofstream strmText(fileName.c_str(), std::ios::out);
    if(!strmText) 
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);  // file not writable

    std::vector<double> indices;
    std::vector< std::vector<double> > coefs;
    size_t ncoefsAngular=0;
    switch(getPotentialTypeByName(potential.name()))
    {
#if 0
    case PT_SCALEFREESH: {
        const CPotentialScaleFreeSH* potSH=static_cast<const CPotentialScaleFreeSH*>(potential);
        indices.assign(1, 0);
        coefs.resize(1);
        potSH->getCoefs(&(coefs[0]));
        ncoefsAngular=potSH->getnumCoefsAngular();
        strmText << "SFcoefs\t#header\n" << 0 << "\t#n_radial\n" << ncoefsAngular << "\t#n_angular\n" << 
            potential->getGamma() << "\t#gamma\n0\t#time\n";
        strmText << "#index";
        break; 
    }
#endif
    case PT_BSE: {
        const BasisSetExp* potBSE = dynamic_cast<const BasisSetExp*>(&potential);
        indices.resize(potBSE->getNumCoefsRadial()+1);
        for(size_t i=0; i<indices.size(); i++) indices[i]=i*1.0;
        potBSE->getCoefs(&coefs);
        assert(coefs.size()==indices.size());
        ncoefsAngular = potBSE->getNumCoefsAngular();
        strmText << "BSEcoefs\t#header\n" << 
            potBSE->getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" << 
            potBSE->getAlpha() <<"\t#alpha\n0\t#time\n";
        strmText << "#index";
        break; 
    }
    case PT_SPLINE: {
        const SplineExp* potSpline = dynamic_cast<const SplineExp*>(&potential);
        potSpline->getCoefs(&indices, &coefs);
        assert(coefs.size()==indices.size());
        assert(indices[0]==0);  // leftmost radius is 0
        coefs[0].resize(1);     // retain only l=0 term for r=0, the rest is supposed to be zero
        ncoefsAngular = potSpline->getNumCoefsAngular();
        strmText << "SHEcoefs\t#header\n" << 
            potSpline->getNumCoefsRadial() << "\t#n_radial\n" << 
            ncoefsAngular << "\t#n_angular\n" <<
            0 <<"\t#unused\n0\t#time\n";
        strmText << "#radius";
        break; 
    }
    default:
        throw std::runtime_error("Unknown type of potential to write");
    }
    for(int l=0; l<=static_cast<int>(ncoefsAngular); l++)
        for(int m=-l; m<=l; m++)
            strmText << "\tl="<<l<<",m="<<m;  // header line
    strmText << "\n";
    for(size_t n=0; n<indices.size(); n++)
    {
        strmText << indices[n];
        strmText << "\t" << std::setprecision(14) << coefs[n][0] << std::setprecision(7);   // leading coeft should be high-accuracy at least for spline potential
        for(size_t i=1; i<coefs[n].size(); i++)
            strmText << "\t" << coefs[n][i];
        strmText << "\n";
    }
    if(!strmText.good())
        throw std::runtime_error("Cannot write potential coefs to file "+fileName);
}

static void writePotentialCylExp(const std::string &fileName, const BasePotential& potential)
{
#ifdef HAVE_INTERP2D
    if(potential->PotentialType()==PT_CYLSPLINE) {
        std::vector<double> gridR, gridz;
        std::vector<std::vector<double>> coefs;
        static_cast<const CPotentialCylSpline*>(potential)->getCoefs(&gridR, &gridz, &coefs);
        int mmax=coefs.size()/2;
        strmText << "CylSpline\t#header\n" << gridR.size() << "\t#size_R\n" << mmax << "\t#m_max\n" <<
            gridz.size() << "\t#size_z\n0\t#time\n" << std::setprecision(16);
        for(int m=0; m<static_cast<int>(coefs.size()); m++) 
            if(coefs[m].size()>0) {
                strmText << (m-mmax) << "\t#m\n#z\\R";
                for(size_t iR=0; iR<gridR.size(); iR++)
                    strmText << "\t" << gridR[iR];
                strmText << "\n";
                for(size_t iz=0; iz<gridz.size(); iz++) {
                    strmText << gridz[iz];
                    for(size_t iR=0; iR<gridR.size(); iR++)
                        strmText << "\t" << coefs[m][iz*gridR.size()+iR];
                    strmText << "\n";
                }
            } 
        if(!strmText) return false;
        return true;
    }
#else
    throw std::runtime_error("Not implemented");
#endif
}

void writePotential(const std::string &fileName, const BasePotential& potential)
{
    switch(getPotentialTypeByName(potential.name())) {
    case PT_BSE:
    case PT_SPLINE:
        writePotentialExp(fileName, potential);
        break;
    case PT_CYLSPLINE:
        writePotentialCylExp(fileName, potential);
        break;
    default:
        throw std::runtime_error("Unknown type of potential to write");
    }
}

// load GalPot parameter file

static void SwallowRestofLine(std::ifstream& from) {
    char c;
    do from.get(c); while( from.good() && c !='\n');
}

const potential::BasePotential* readGalaxyPotential(const char* filename, const units::Units& units) {
    std::ifstream strm(filename);
    if(!strm) 
        throw std::invalid_argument("Cannot open file "+std::string(filename));
    std::vector<DiskParam> diskpars;
    std::vector<SphrParam> sphrpars;
    bool ok=true;
    int num;
    strm>>num;
    SwallowRestofLine(strm);
    if(num<0 || num>10 || !strm) ok=false;
    for(int i=0; i<num && ok; i++) {
        DiskParam dp;
        strm>>dp.surfaceDensity >> dp.scaleLength >> dp.scaleHeight >> dp.innerCutoffRadius >> dp.modulationAmplitude;
        SwallowRestofLine(strm);
        dp.surfaceDensity *= units.from_Msun_per_Kpc2;
        dp.scaleLength *= units.from_Kpc;
        dp.scaleHeight *= units.from_Kpc;
        dp.innerCutoffRadius *= units.from_Kpc;
        if(strm) diskpars.push_back(dp);
        else ok=false;
    }
    strm>>num;
    SwallowRestofLine(strm);
    ok=ok && strm;
    for(int i=0; i<num && ok; i++) {
        SphrParam sp;
        strm>>sp.densityNorm >> sp.axisRatio >> sp.gamma >> sp.beta >> sp.scaleRadius >> sp.outerCutoffRadius;
        SwallowRestofLine(strm);
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
//    PotentialNames[PT_FERRERS] = CPotentialFerrers::myName();
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
//    DensityNames[PT_FERRERS] = CPotentialFerrers::myName();

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
