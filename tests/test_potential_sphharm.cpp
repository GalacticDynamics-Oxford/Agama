#include "potential_analytic.h"
#include "potential_dehnen.h"
#include "potential_sphharm.h"
#include "potential_factory.h"
#include "coord_utils.h"

/// define test suite in terms of points for various coord systems
const int numtestpoints=6;
const double pos_sph[numtestpoints][3] = {   // order: R, theta, phi
    {1  , 2  , 3},   // ordinary point
    {2  , 1  , 0},   // point in x-z plane
    {1  , 0  , 0},   // point along z axis
    {111, 0.7, 2},   // point at a large radius
    {.01, 2.5,-1},   // point at a small radius origin
    {0  , 0  , 0} }; // point at origin

/// write potential coefs into file, load them back and create a new potential from these coefs
const potential::BasePotential* write_read(const potential::BasePotential& pot)
{
    std::string coef_file("test_potential_sphharm");
    coef_file += potential::getCoefFileExtension(getPotentialType(pot));
    potential::writePotential(coef_file, pot);
    potential::ConfigPotential config;
    config.fileName = coef_file;
    const potential::BasePotential* newpot = potential::readPotential(config);
    std::remove(coef_file.c_str());
    return newpot;
}

bool test_suite(const potential::BasePotential& p, const potential::BasePotential& orig, double eps_pot)
{
    bool ok=true;
    const potential::BasePotential* newpot = write_read(p);
    std::cout << "---- testing "<<p.name()<<" with "<<orig.name()<<" ----\n";
    for(int ic=0; ic<numtestpoints; ic++) {
        double pot, pot_orig;
        coord::GradSph der,  der_orig;
        coord::HessSph der2, der2_orig;
        coord::PosSph point(pos_sph[ic][0], pos_sph[ic][1], pos_sph[ic][2]);
        newpot->eval(coord::toPosSph(point), &pot, &der, &der2);
        orig.eval(coord::toPosSph(point), &pot_orig, &der_orig, &der2_orig);
        double eps_der = eps_pot*100/point.r;
        double eps_der2= eps_der*10;
        bool pot_ok = (pot==pot) && fabs(pot-pot_orig)<eps_pot;
        bool der_ok = point.r==0 || equalGrad(der, der_orig, eps_der);
        bool der2_ok= point.r==0 || equalHess(der2, der2_orig, eps_der2) || point.theta==0; ///!!! this should be fixed
        ok &= pot_ok && der_ok && der2_ok;
        std::cout << "Point:  " << point << "Phi: " << pot << " (orig:" << pot_orig << (pot_ok?"":" *") << ")\n"
            "Force sphharm: " << der  << "\nForce origin.: " << der_orig  << (der_ok ?"":" *") << "\n"
            "Deriv sphharm: " << der2 << "\nDeriv origin.: " << der2_orig << (der2_ok?"":" *") << "\n";
    }
    delete newpot;
    return ok;
}

int main() {
    bool ok=true;
    const potential::Plummer plum(10., 5.);
    ok &= test_suite(potential::BasisSetExp(0., 30, 2, plum), plum, 1e-5);
    ok &= test_suite(potential::SplineExp(20, 2, plum), plum, 1e-5);
    const potential::Dehnen deh(3., 1.2, 0.8, 0.6, 1.5);
    ok &= test_suite(potential::BasisSetExp(2., 20, 6, deh), deh, 2e-4);
    ok &= test_suite(potential::SplineExp(20, 6, deh), deh, 2e-4);
    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}