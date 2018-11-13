#include "df_factory.h"
#include "df_disk.h"
#include "df_halo.h"
#include "utils.h"
#include "utils_config.h"
#include <cassert>
#include <stdexcept>

namespace df {

DoublePowerLawParam parseDoublePowerLawParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    DoublePowerLawParam par;
    par.norm      = kvmap.getDouble("norm")    * conv.massUnit;
    par.J0        = kvmap.getDouble("J0")      * conv.lengthUnit * conv.velocityUnit;
    par.Jcutoff   = kvmap.getDouble("Jcutoff") * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0     = kvmap.getDouble("Jphi0")   * conv.lengthUnit * conv.velocityUnit;
    par.slopeIn   = kvmap.getDouble("slopeIn",   par.slopeIn);
    par.slopeOut  = kvmap.getDouble("slopeOut",  par.slopeOut);
    par.steepness = kvmap.getDouble("steepness", par.steepness);
    par.coefJrIn  = kvmap.getDouble("coefJrIn",  par.coefJrIn);
    par.coefJzIn  = kvmap.getDouble("coefJzIn",  par.coefJzIn);
    par.coefJrOut = kvmap.getDouble("coefJrOut", par.coefJrOut);
    par.coefJzOut = kvmap.getDouble("coefJzOut", par.coefJzOut);
    par.rotFrac   = kvmap.getDouble("rotFrac",   par.rotFrac);
    par.cutoffStrength = kvmap.getDouble("cutoffStrength", par.cutoffStrength);
    return par;
}

QuasiIsothermalParam parseQuasiIsothermalParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    QuasiIsothermalParam par;
    par.Sigma0  = kvmap.getDouble("Sigma0")  * conv.massUnit / pow_2(conv.lengthUnit);
    par.Rdisk   = kvmap.getDouble("Rdisk")   * conv.lengthUnit;
    par.Hdisk   = kvmap.getDouble("Hdisk")   * conv.lengthUnit;
    par.sigmar0 = kvmap.getDouble("sigmar0") * conv.velocityUnit;
    par.sigmaz0 = kvmap.getDouble("sigmaz0") * conv.velocityUnit;
    par.sigmamin= kvmap.getDouble("sigmamin")* conv.velocityUnit;
    par.Rsigmar = kvmap.getDouble("Rsigmar") * conv.lengthUnit;
    par.Rsigmaz = kvmap.getDouble("Rsigmaz") * conv.lengthUnit;
    par.coefJr  = kvmap.getDouble("coefJr", par.coefJr);
    par.coefJz  = kvmap.getDouble("coefJz", par.coefJz);
    par.Jmin    = kvmap.getDouble("Jmin") * conv.lengthUnit * conv.velocityUnit;
    par.beta    = kvmap.getDouble("beta", par.beta);
    par.Tsfr    = kvmap.getDouble("Tsfr", par.Tsfr);  // dimensionless! in units of Hubble time (galaxy age)
    par.sigmabirth = kvmap.getDouble("sigmabirth", par.sigmabirth);  // dimensionless ratio
    return par;
}

ExponentialParam parseExponentialParams(
    const utils::KeyValueMap& kvmap,
    const units::ExternalUnits& conv)
{
    ExponentialParam par;
    par.norm   = kvmap.getDouble("norm")   * conv.massUnit;
    par.Jr0    = kvmap.getDouble("Jr0")    * conv.lengthUnit * conv.velocityUnit;
    par.Jz0    = kvmap.getDouble("Jz0")    * conv.lengthUnit * conv.velocityUnit;
    par.Jphi0  = kvmap.getDouble("Jphi0")  * conv.lengthUnit * conv.velocityUnit;
    par.addJden= kvmap.getDouble("addJden")* conv.lengthUnit * conv.velocityUnit;
    par.addJvel= kvmap.getDouble("addJvel")* conv.lengthUnit * conv.velocityUnit;
    par.coefJr = kvmap.getDouble("coefJr", par.coefJr);
    par.coefJz = kvmap.getDouble("coefJz", par.coefJz);
    par.beta   = kvmap.getDouble("beta", par.beta);
    par.Tsfr   = kvmap.getDouble("Tsfr", par.Tsfr);  // dimensionless! in units of Hubble time (galaxy age)
    par.sigmabirth = kvmap.getDouble("sigmabirth", par.sigmabirth);  // dimensionless ratio
    return par;
}

inline void checkNonzero(const potential::BasePotential* potential, const std::string& type)
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
        return PtrDistributionFunction(new DoublePowerLaw(parseDoublePowerLawParams(kvmap, converter)));
    }
    if(utils::stringsEqual(type, "Exponential")) {
        return PtrDistributionFunction(new Exponential(parseExponentialParams(kvmap, converter)));
    }
    else if(utils::stringsEqual(type, "QuasiIsothermal")) {
        checkNonzero(potential, type);
        return PtrDistributionFunction(new QuasiIsothermal(parseQuasiIsothermalParams(kvmap, converter),
            potential::Interpolator(*potential)));
    }
    else
        throw std::invalid_argument("Unknown type of distribution function");
}

}  // namespace df
