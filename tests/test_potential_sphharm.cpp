#include "potential_analytic.h"
#include "potential_dehnen.h"
#include "potential_cylspline.h"
#include "potential_sphharm.h"
#include "potential_factory.h"
#include "particles_io.h"
#include "coord_utils.h"
#include <cstdlib>
#include <cstdio>

/// define test suite in terms of points for various coord systems
const int numtestpoints=8;
const double pos_sph[numtestpoints][3] = {   // order: R, theta, phi
    {1  , 2  , 3},   // ordinary point
    {2  , 1  , 0},   // point in x-z plane
    {1  , 0  , 0},   // point along z axis
    {2  , 3.14159,-1},   // point along z axis, z<0
    {0.5, 1.5707963, 1.5707963},  // point along y axis
    {111, 0.7, 2},   // point at a large radius
    {.01, 2.5,-1},   // point at a small radius origin
    {0  , 0  , 0} }; // point at origin

/// write potential coefs into file, load them back and create a new potential from these coefs
const potential::BasePotential* write_read(const potential::BasePotential& pot)
{
    std::string coefFile("test_potential_sphharm");
    coefFile += potential::getCoefFileExtension(getPotentialType(pot));
    potential::writePotential(coefFile, pot);
    potential::ConfigPotential config;
    config.fileName = coefFile;
    const potential::BasePotential* newpot = potential::readPotential(config);
    std::remove(coefFile.c_str());
    return newpot;
}

/// create a triaxial Hernquist model
void make_hernquist(int nbody, double q, double p, particles::PointMassArrayCar& output)
{
    output.data.clear();
    for(int i=0; i<nbody; i++) {
        double m = rand()*1.0/RAND_MAX;
        double r = 1/(1/sqrt(m)-1);
        double costheta = rand()*2.0/RAND_MAX - 1;
        double sintheta = sqrt(1-pow_2(costheta));
        double phi = rand()*2*M_PI/RAND_MAX;
        output.add(coord::PosVelCar(r*sintheta*cos(phi), r*sintheta*sin(phi)*q, r*costheta*p, 0, 0, 0), 1./nbody);
    }
}

const potential::BasePotential* create_from_file(
    const particles::PointMassArrayCar& points, const potential::PotentialType type)
{
    const std::string fileName = "test.txt";
    particles::BaseIOSnapshot* snap = particles::createIOSnapshotWrite("Text", fileName);
    snap->writeSnapshot(points);
    delete snap;
    potential::ConfigPotential config;
    config.fileName = fileName;
    config.potentialType   = type;
    config.numCoefsRadial  = 20;
    config.numCoefsAngular = 4;
    config.numCoefsVertical= 20;
    config.alpha           = 1.0;
    config.symmetryType    = potential::ST_TRIAXIAL;
    const potential::BasePotential* newpot = potential::readPotential(config);
    std::remove(fileName.c_str());
    std::remove((fileName+potential::getCoefFileExtension(type)).c_str());
    return newpot;
}

/// compare potential and its derivatives between the original model and its spherical-harmonic approximation
bool test_suite(const potential::BasePotential& p, const potential::BasePotential& orig, double eps_pot)
{
    bool ok=true;
    const potential::BasePotential* newpot = write_read(p);
    std::cout << "\033[1;32m---- testing "<<p.name()<<
        " with "<<potential::getSymmetryNameByType(orig.symmetry())<<" "<<orig.name()<<
        " (gamma="<<potential::getInnerDensitySlope(orig)<<") ----\033[0m\n";
    const char* err = "\033[1;31m **\033[0m";
    for(int ic=0; ic<numtestpoints; ic++) {
        double pot, pot_orig;
        coord::GradCyl der,  der_orig;
        coord::HessCyl der2, der2_orig;
        coord::PosSph point(pos_sph[ic][0], pos_sph[ic][1], pos_sph[ic][2]);
        newpot->eval(coord::toPosCyl(point), &pot, &der, &der2);
        orig.eval(coord::toPosCyl(point), &pot_orig, &der_orig, &der2_orig);
        double eps_der = eps_pot*100/point.r;
        double eps_der2= eps_der*10;
        bool pot_ok = (pot==pot) && fabs(pot-pot_orig)<eps_pot;
        bool der_ok = point.r==0 || equalGrad(der, der_orig, eps_der);
        bool der2_ok= point.r==0 || equalHess(der2, der2_orig, eps_der2);
        ok &= pot_ok && der_ok && der2_ok;
        std::cout << "Point:  " << point << 
            "Phi: " << pot << " (orig:" << pot_orig << (pot_ok?"":err) << ")\n"
            "Force sphharm: " << der  << "\nForce origin.: " << der_orig  << (der_ok ?"":err) << "\n"
            "Deriv sphharm: " << der2 << "\nDeriv origin.: " << der2_orig << (der2_ok?"":err) << "\n";
    }
    delete newpot;
    return ok;
}

int main() {
    srand(42);
    bool ok=true;
    const potential::Plummer plum(10., 5.);
    ok &= test_suite(potential::BasisSetExp(0., 30, 2, plum), plum, 1e-5);
    ok &= test_suite(potential::SplineExp(20, 2, plum), plum, 1e-5);
    ok &= test_suite(potential::CylSplineExp(20, 20, 0, plum), plum, 1e-4);
    const potential::Dehnen deh15(3., 1.2, 0.8, 0.6, 1.5);
    ok &= test_suite(potential::BasisSetExp(2., 20, 6, deh15), deh15, 2e-4);
    ok &= test_suite(potential::SplineExp(20, 6, deh15), deh15, 2e-4);
    const potential::Dehnen deh0(1., 1., 0.8, 0.6, 0.);
    ok &= test_suite(potential::BasisSetExp(1., 20, 6, deh0), deh0, 5e-5);
    ok &= test_suite(potential::SplineExp(20, 6, deh0), deh0, 5e-5);
    ok &= test_suite(potential::CylSplineExp(20, 20, 6, 
        static_cast<const potential::BaseDensity&>(deh0)), deh0, 1e-4);  // this forces potential to be computed via integration of density over volume
    particles::PointMassArrayCar points;
    const potential::Dehnen hernq(1., 1., 0.8, 0.6, 1.0);
    make_hernquist(100000, 0.8, 0.6, points);
    const potential::BasePotential* p = create_from_file(points, potential::PT_BSE);
    ok &= test_suite(*p, hernq, 2e-2);
    delete p;
    p = new potential::SplineExp(20, 4, points, potential::ST_TRIAXIAL);  // create_from_file(points, potential::PT_SPLINE);
    ok &= test_suite(*p, hernq, 2e-2);
    delete p;
    p = create_from_file(points, potential::PT_CYLSPLINE);
    ok &= test_suite(*p, hernq, 2e-2);
    delete p;
    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}