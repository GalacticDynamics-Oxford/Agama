#include "df_factory.h"
#include "df_disk.h"
#include "df_halo.h"
#include "utils.h"
#include "utils_config.h"
#include <cassert>
#include <stdexcept>

namespace df {

static DoublePowerLawParam parseDoublePowerLawParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    DoublePowerLawParam par;
    par.norm      = kvmap.getDouble("norm")    * conv.massUnit;
    par.J0        = kvmap.getDouble("J0")      * conv.lengthUnit * conv.velocityUnit;
    par.Jcutoff   = kvmap.getDouble("Jcutoff") * conv.lengthUnit * conv.velocityUnit;
    par.slopeIn   = kvmap.getDouble("slopeIn",   par.slopeIn);
    par.slopeOut  = kvmap.getDouble("slopeOut",  par.slopeOut);
    par.steepness = kvmap.getDouble("steepness", par.steepness);
    par.coefJrIn  = kvmap.getDouble("coefJrIn",  par.coefJrIn);
    par.coefJzIn  = kvmap.getDouble("coefJzIn",  par.coefJzIn);
    par.coefJrOut = kvmap.getDouble("coefJrOut", par.coefJrOut);
    par.coefJzOut = kvmap.getDouble("coefJzOut", par.coefJzOut);
    return par;
}

static PseudoIsothermalParam parsePseudoIsothermalParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    PseudoIsothermalParam par;
    par.Sigma0  = kvmap.getDouble("Sigma0")  * conv.massUnit / pow_2(conv.lengthUnit);
    par.Rdisk   = kvmap.getDouble("Rdisk")   * conv.lengthUnit;
    par.Jphimin = kvmap.getDouble("Jphimin") * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0   = kvmap.getDouble("Jphi0")   * conv.lengthUnit * conv.velocityUnit;
    par.sigmar0 = kvmap.getDouble("sigmar0") * conv.velocityUnit;
    par.sigmaz0 = kvmap.getDouble("sigmaz0") * conv.velocityUnit;
    par.sigmamin= kvmap.getDouble("sigmamin")* conv.velocityUnit;
    par.Rsigmar = kvmap.getDouble("Rsigmar", 2*par.Rdisk) * conv.lengthUnit;
    par.Rsigmaz = kvmap.getDouble("Rsigmaz", 2*par.Rdisk) * conv.lengthUnit;
    par.beta    = kvmap.getDouble("beta", par.beta);
    par.Tsfr    = kvmap.getDouble("Tsfr", par.Tsfr);  // dimensionless! in units of Hubble time
    par.sigmabirth = kvmap.getDouble("sigmabirth", par.sigmabirth);  // dimensionless ratio
    return par;
}

static void checkNonzero(const potential::BasePotential* potential, const std::string& type)
{
    if(potential == NULL)
        throw std::invalid_argument("Need an instance of potential to initialize "+type+" DF");
}

PtrDistributionFunction createDistributionFunction(
    const utils::KeyValueMap& kvmap,
    const potential::BasePotential* potential,
    const units::ExternalUnits& converter)
{
    std::string type = kvmap.getString("type");
    if(utils::stringsEqual(type, "DoublePowerLaw")) {
        DoublePowerLawParam params = parseDoublePowerLawParams(kvmap, converter);
        return PtrDistributionFunction(new DoublePowerLaw(params));
    }
    else if(utils::stringsEqual(type, "PseudoIsothermal")) {
        checkNonzero(potential, type);
        PseudoIsothermalParam params = parsePseudoIsothermalParams(kvmap, converter);
        return PtrDistributionFunction(new PseudoIsothermal(
            params, potential::Interpolator(*potential)));
    }
    else
        throw std::invalid_argument("Unknown type of distribution function");
}

}; // namespace
