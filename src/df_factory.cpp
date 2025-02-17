#include "df_factory.h"
#include "df_disk.h"
#include "df_halo.h"
#include "df_spherical.h"
#include "utils.h"
#include <cassert>
#include <stdexcept>
#include <algorithm>
#ifndef _MSC_VER
#include <alloca.h>
#else
#include <malloc.h>
#endif

namespace df {

CompositeDF::CompositeDF(const std::vector<PtrDistributionFunction> &_components) :
    components(_components)
{
    if(components.empty())
        throw std::invalid_argument("CompositeDF: List of DF components cannot be empty");
}

void CompositeDF::evalmany(
    const size_t npoints, const actions::Actions J[], bool separate,
    /*output*/ double values[], DerivByActions derivs[]) const
{
    // the "separate" flag indicates whether to store values for each component separately
    // or sum them up; each component produces a single value for each input point, even if
    // this component itself is a composite DF, so the evalmany() method of each component
    // is always invoked with "separate=false"
    unsigned int ncomp = components.size();
    if(ncomp == 1) {
        // fast track: for a single DF component, it does not matter whether separate is true or false,
        // and we simply output a single value per input point
        components[0]->evalmany(npoints, J, /*separate*/false, values, derivs);
        return;
    }

    // the idea is to loop over components and for each one to call the vectorized method (evalmany),
    // computing the values of the given component for all input points at once,
    // but then we need to sum up values of all components at a given point (if separate is false)
    // or to reorder them so that all components for a given input point are stored contiguously,
    // (if separate is true). In both cases we need a temporary storage, which is allocated
    // on the stack (hence no need to deallocate it explicitly)
    double* compval = static_cast<double*>(alloca(npoints * sizeof(double)));
    DerivByActions* compder = derivs?
        static_cast<DerivByActions*>(alloca(npoints * sizeof(DerivByActions))) :
        NULL;
    if(!separate) {  // fill the output array with zeros and then add one component at a time
        std::fill(values, values+npoints, 0);
        if(derivs) {
            DerivByActions zero;
            zero.dbyJr = zero.dbyJz = zero.dbyJphi = 0;
            std::fill(derivs, derivs+npoints, zero);
        }
    }
    for(unsigned int c=0; c<ncomp; c++) {
        components[c]->evalmany(npoints, J, /*separate*/ false, /*output*/ compval, compder);
        if(separate) {
            for(size_t p=0; p<npoints; p++)
                values[p*ncomp+c] = compval[p];
            if(derivs) {
                for(size_t p=0; p<npoints; p++)
                    derivs[p*ncomp+c] = compder[p];
            }
        } else {
            for(size_t p=0; p<npoints; p++)
                values[p] += compval[p];
            if(derivs) {
                for(size_t p=0; p<npoints; p++) {
                    derivs[p].dbyJr   += compder[p].dbyJr;
                    derivs[p].dbyJz   += compder[p].dbyJz;
                    derivs[p].dbyJphi += compder[p].dbyJphi;
                }
            }
        }
    }
}

namespace{

void assignParam(double& param,
    const utils::KeyValueMap& kvmap,
    std::vector<std::string>& keys,
    const std::string& key1,
    const std::string& key2="",
    double unit=1.0)
{
    bool found1 = false, found2 = false;
    for(unsigned int i=0; i<keys.size(); i++) {
        found1 |= utils::stringsEqual(keys[i], key1);
        found2 |= !key2.empty() && utils::stringsEqual(keys[i], key2);
    }
    if(!found1 && !found2) {
        param *= unit;
        return;
    }
    if(found1 && found2)
        throw std::runtime_error("Duplicate values for synonymous parameters " + key1 + " and " + key2);
    const std::string& key = found1 ? key1 : key2;
    const std::string value = kvmap.getString(key);
    if(value.empty())
        throw std::runtime_error("Empty value for parameter " + key);
    param = utils::toDouble(value) * unit;
    // remove the parsed parameter from the list of keys
    for(unsigned int i=0; i<keys.size(); i++) {
        if(utils::stringsEqual(keys[i], key)) {
            keys.erase(keys.begin() + i);
            break;  // the key is encountered exactly once, no need to check further
        }
    }
}

DoublePowerLawParam parseDoublePowerLawParam(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv,
    std::vector<std::string>& keys)
{
    DoublePowerLawParam par;
    assignParam(par.norm,      kvmap, keys, "norm",    "", conv.massUnit);
    assignParam(par.J0,        kvmap, keys, "J0",      "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.Jcutoff,   kvmap, keys, "Jcutoff", "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.Jphi0,     kvmap, keys, "Jphi0",   "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.Jcore,     kvmap, keys, "Jcore",   "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.slopeIn,   kvmap, keys, "slopeIn");
    assignParam(par.slopeOut,  kvmap, keys, "slopeOut");
    assignParam(par.steepness, kvmap, keys, "steepness");
    assignParam(par.coefJrIn,  kvmap, keys, "coefJrIn");
    assignParam(par.coefJzIn,  kvmap, keys, "coefJzIn");
    assignParam(par.coefJrOut, kvmap, keys, "coefJrOut");
    assignParam(par.coefJzOut, kvmap, keys, "coefJzOut");
    assignParam(par.rotFrac,   kvmap, keys, "rotFrac");
    assignParam(par.cutoffStrength, kvmap, keys, "cutoffStrength");
    return par;
}

QuasiIsothermalParam parseQuasiIsothermalParam(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv,
    std::vector<std::string>& keys)
{
    QuasiIsothermalParam par;
    assignParam(par.Sigma0,  kvmap, keys, "Sigma0",  "", conv.massUnit / pow_2(conv.lengthUnit));
    assignParam(par.Rdisk,   kvmap, keys, "Rdisk",   "", conv.lengthUnit);
    assignParam(par.Hdisk,   kvmap, keys, "Hdisk",   "", conv.lengthUnit);
    assignParam(par.sigmar0, kvmap, keys, "sigmar0", "", conv.velocityUnit);
    assignParam(par.sigmaz0, kvmap, keys, "sigmaz0", "", conv.velocityUnit);
    assignParam(par.sigmamin,kvmap, keys, "sigmamin","", conv.velocityUnit);
    assignParam(par.Rsigmar, kvmap, keys, "Rsigmar", "", conv.lengthUnit);
    assignParam(par.Rsigmaz, kvmap, keys, "Rsigmaz", "", conv.lengthUnit);
    assignParam(par.coefJr,  kvmap, keys, "coefJr");
    assignParam(par.coefJz,  kvmap, keys, "coefJz");
    assignParam(par.Jmin,    kvmap, keys, "Jmin",    "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.qJr,     kvmap, keys, "qJr");
    assignParam(par.qJz,     kvmap, keys, "qJz");
    assignParam(par.qJphi,   kvmap, keys, "qJphi");
    return par;
}

ExponentialParam parseExponentialParam(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv,
    std::vector<std::string>& keys)
{
    ExponentialParam par;
    assignParam(par.norm,    kvmap, keys, "norm",    "", conv.massUnit);
    assignParam(par.Jr0,     kvmap, keys, "Jr0",     "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.Jz0,     kvmap, keys, "Jz0",     "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.Jphi0,   kvmap, keys, "Jphi0",   "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.addJden, kvmap, keys, "addJden", "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.addJvel, kvmap, keys, "addJvel", "", conv.lengthUnit * conv.velocityUnit);
    assignParam(par.coefJr,  kvmap, keys, "coefJr");
    assignParam(par.coefJz,  kvmap, keys, "coefJz");
    assignParam(par.qJr,     kvmap, keys, "qJr");
    assignParam(par.qJz,     kvmap, keys, "qJz");
    assignParam(par.qJphi,   kvmap, keys, "qJphi");
    return par;
}

}  // namespace

PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& kvmap,
    const potential::BasePotential* potential,
    const potential::BaseDensity* density,
    const units::ExternalUnits& converter)
{
    std::vector<std::string> keys = kvmap.keys();
    std::string type = kvmap.getString("type");
    PtrDistributionFunction result;
    // for some DF types, there are two alternative ways of specifying the normalization:
    // either directly as norm, Sigma0, etc., or as the total mass, from which the norm is computed
    // by creating a temporary instance of a corresponding DF class, and computing its mass
    bool massProvided = kvmap.contains("mass");
    double mass = NAN;
    if(massProvided)
        assignParam(mass, kvmap, keys, "mass", "", converter.massUnit);
    if(utils::stringsEqual(type, "DoublePowerLaw")) {
        if(potential != NULL || density != NULL)
            throw std::invalid_argument(type+" DF does not need potential or density");
        DoublePowerLawParam par = parseDoublePowerLawParam(kvmap, converter, keys);
        if(massProvided) {
            if(kvmap.contains("norm"))
                throw std::runtime_error("Parameters 'mass' and 'norm' are mutually exclusive");
            par.norm = 1.0;
            par.norm = mass / DoublePowerLaw(par).totalMass();
        }
        result = PtrDistributionFunction(new DoublePowerLaw(par));
    }
    else if(utils::stringsEqual(type, "Exponential")) {
        if(potential != NULL || density != NULL)
            throw std::invalid_argument(type+" DF does not need potential or density");
        ExponentialParam par = parseExponentialParam(kvmap, converter, keys);
        if(massProvided) {
            if(kvmap.contains("norm"))
                throw std::runtime_error("Parameters 'mass' and 'norm' are mutually exclusive");
            par.norm = 1.0;
            par.norm = mass / Exponential(par).totalMass();
        }
        result = PtrDistributionFunction(new Exponential(par));
    }
    else if(utils::stringsEqual(type, "QuasiIsothermal")) {
        if(density != NULL)
            throw std::invalid_argument(type+" DF does not need density");
        if(potential == NULL)
            throw std::invalid_argument("Need an instance of potential to initialize "+type+" DF");
        potential::Interpolator pot_interp(*potential);
        QuasiIsothermalParam par = parseQuasiIsothermalParam(kvmap, converter, keys);
        if(mass>0) {
            if(kvmap.contains("Sigma0"))
                throw std::runtime_error("Parameters 'mass' and 'Sigma0' are mutually exclusive");
            par.Sigma0 = 1.0;
            par.Sigma0 = mass / QuasiIsothermal(par, pot_interp).totalMass();
        }
        result = PtrDistributionFunction(new QuasiIsothermal(par, pot_interp));
    }
    else if(utils::stringsEqual(type, "QuasiSpherical")) {
        if(potential == NULL)
            throw std::invalid_argument("Need an instance of potential and optionally density "
                "to initialize "+type+" DF");
        if(density == NULL)
            density = potential;
        double beta0 = 0, r_a = INFINITY, rotFrac = 0, Jphi0 = 0;
        assignParam(beta0,   kvmap, keys, "beta", "beta0");
        assignParam(r_a,     kvmap, keys, "anisotropyRadius", "r_a", converter.lengthUnit);
        assignParam(rotFrac, kvmap, keys, "rotFrac");
        assignParam(Jphi0,   kvmap, keys, "Jphi0", "", converter.lengthUnit * converter.velocityUnit);
        result = PtrDistributionFunction(new QuasiSphericalCOM(
            potential::Sphericalized<potential::BaseDensity>  (*density),
            potential::Sphericalized<potential::BasePotential>(*potential),
            beta0, r_a, rotFrac, Jphi0));
    }
    else if(type.empty())
        throw std::runtime_error("Need to provide the distribution function type");
    else
        throw std::invalid_argument("Unknown type of distribution function");

    // check that all provided parameters were used (i.e. no unknown parameters are given)
    for(unsigned int i=0; i<keys.size(); i++) {
        if(utils::stringsEqual(keys[i], "type")) {
            keys.erase(keys.begin() + i);
            break;
        }
    }
    if(!keys.empty()) {
        std::string unknownKeys;
        for(unsigned int i=0; i<keys.size(); i++)
            unknownKeys += i>0 ? ", " + keys[i] : keys[i];
        throw std::runtime_error(
            "Unknown parameter" + std::string(keys.size()>1 ? "s " : " ") + unknownKeys);
    }

    return result;
}

}  // namespace df
