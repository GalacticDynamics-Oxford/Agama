#include "potential_factory.h"
#include "potential_utils.h"
#include "actions_factory.h"

namespace{
    std::string error;
}

extern "C" {

typedef potential::PtrDensity agama_Density;
typedef potential::PtrPotential agama_Potential;
typedef actions::PtrActionFinder agama_ActionFinder;

const char* agama_getError()
{
    return error.c_str();
}

agama_Density* agama_createDensity(const char* params)
{
    try{
        return new agama_Density(potential::createDensity(utils::KeyValueMap(params)));
    }
    catch(std::exception& e) {
        error = e.what();
        return NULL;
    }
}

agama_Potential* agama_createPotential(const char* params)
{
    try{
        utils::KeyValueMap kvmap(params);
        potential::PtrPotential result;
        if(kvmap.contains("G")) {
            // undocumented hack! instead of providing the full triplet of external units,
            // one can specify the value of G in these units, which will be used to scale the mass.
            double G = kvmap.popDouble("G");
            const units::InternalUnits unit(units::Kpc, units::Kpc/units::kms);
            result = potential::createPotential(kvmap,
                units::ExternalUnits(unit, units::Kpc, units::kms, units::Msun * unit.to_Msun * G));
        } else
            result = potential::createPotential(kvmap);
        return new agama_Potential(result);
    }
    catch(std::exception& e) {
        error = e.what();
        return NULL;
    }
}

void agama_deleteDensity(agama_Density* density)
{
    delete density;
}

void agama_deletePotential(agama_Potential* potential)
{
    delete potential;
}

double agama_evalDensity(const agama_Density* density, const double pos[3], double time)
{
    return density->get()->density(coord::PosCar(pos[0], pos[1], pos[2]), time);
}

double agama_evalPotential(const agama_Potential* potential, const double pos[3], double time,
    double deriv[3], double deriv2[6])
{
    double Phi;
    coord::GradCar grad;
    coord::HessCar hess;
    potential->get()->eval(coord::PosCar(pos[0], pos[1], pos[2]), &Phi,
        deriv? &grad : NULL, deriv2? &hess : NULL, time);
    if(deriv) {
        deriv[0] = grad.dx;
        deriv[1] = grad.dy;
        deriv[2] = grad.dz;
    }
    if(deriv2) {
        deriv2[0] = hess.dx2;
        deriv2[1] = hess.dy2;
        deriv2[2] = hess.dz2;
        deriv2[3] = hess.dxdy;
        deriv2[4] = hess.dydz;
        deriv2[5] = hess.dxdz;
    }
    return Phi;
}

double agama_R_circ(const agama_Potential* potential, double E)
{
    return R_circ(*potential->get(), E);
}

double agama_R_from_Lz(const agama_Potential* potential, double Lz)
{
    return R_from_Lz(*potential->get(), Lz);
}

double agama_R_max(const agama_Potential* potential, double E)
{
    return R_max(*potential->get(), E);
}

void agama_findPlanarOrbitExtent(const agama_Potential* potential, double E, double L,
    double* R1, double* R2)
{
    findPlanarOrbitExtent(*potential->get(), E, L, *R1, *R2);
}

agama_ActionFinder* agama_createActionFinder(const agama_Potential* potential)
{
    try{
        return new agama_ActionFinder(actions::createActionFinder(*potential));
    }
    catch(std::exception& e) {
        error = e.what();
        return NULL;
    }
}

void agama_deleteActionFinder(agama_ActionFinder* actionFinder)
{
    delete actionFinder;
}

void agama_evalActionsAnglesFrequencies(
    const agama_ActionFinder* actionFinder, const double posvel[6],
    double actions[3], double angles[3], double frequencies[3])
{
    const coord::PosVelCyl point = coord::toPosVelCyl(
        coord::PosVelCar(posvel[0], posvel[1], posvel[2], posvel[3], posvel[4], posvel[5]));
    actions::Actions act;
    actions::Angles ang;
    actions::Frequencies freq;
    (*actionFinder)->eval(point,
        actions? &act : NULL, angles? &ang : NULL, frequencies? &freq : NULL);
    if(actions) {
        actions[0] = act.Jr;
        actions[1] = act.Jz;
        actions[2] = act.Jphi;
    }
    if(angles) {
        angles[0] = ang.thetar;
        angles[1] = ang.thetaz;
        angles[2] = ang.thetaphi;
    }
    if(frequencies) {
        frequencies[0] = freq.Omegar;
        frequencies[1] = freq.Omegaz;
        frequencies[2] = freq.Omegaphi;
    }
}

void agama_evalActionsAnglesFrequenciesStandalone(
    const agama_Potential* potential, double fd, const double posvel[6],
    double actions[3], double angles[3], double frequencies[3])
{
    const coord::PosVelCyl point = coord::toPosVelCyl(
        coord::PosVelCar(posvel[0], posvel[1], posvel[2], posvel[3], posvel[4], posvel[5]));
    const potential::BasePotential& pot = *potential->get();
    actions::Actions act;
    actions::Angles ang;
    actions::Frequencies freq;
    try{
        actions::eval(pot, point,
            actions? &act : NULL, angles? &ang : NULL, frequencies? &freq : NULL, fd);
    }
    catch(std::exception& e) {
        error = e.what();
        act.Jr = act.Jz = act.Jphi = NAN;
        ang.thetar = ang.thetaz = ang.thetaphi = NAN;
        freq.Omegar = freq.Omegaz = freq.Omegaphi = NAN;
    }
    if(actions) {
        actions[0] = act.Jr;
        actions[1] = act.Jz;
        actions[2] = act.Jphi;
    }
    if(angles) {
        angles[0] = ang.thetar;
        angles[1] = ang.thetaz;
        angles[2] = ang.thetaphi;
    }
    if(frequencies) {
        frequencies[0] = freq.Omegar;
        frequencies[1] = freq.Omegaz;
        frequencies[2] = freq.Omegaphi;
    }
}

}