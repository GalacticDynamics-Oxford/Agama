#include "galaxymodel_base.h"
#include "math_random.h"
#include "math_sphharm.h"
#include "math_spline.h"
#include "particles_io.h"
#include "potential_analytic.h"
#include "potential_cylspline.h"
#include "potential_dehnen.h"
#include "potential_disk.h"
#include "potential_factory.h"
#include "potential_multipole.h"
#include "utils.h"
#include "utils_config.h"
#include "debug_utils.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using potential::PtrPotential;
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

/// write potential coefs into file, load them back and create a new potential from these coefs
PtrPotential writeRead(const potential::BasePotential& pot)
{
    const char* coefFile = "test_potential_expansions.coef";
    writePotential(coefFile, pot);
    PtrPotential newpot = potential::readPotential(coefFile);
    std::remove(coefFile);
    return newpot;
}

/// create a triaxial Dehnen model (could use galaxymodel::sampleNbody in a general case)
particles::ParticleArray<coord::PosCar> makeDehnen(int nbody, double gamma, double p, double q)
{
    //math::randomize();
    particles::ParticleArray<coord::PosCar> pts;
    for(int i=0; i<nbody; i++) {
        double m = math::random();
        double r = 1 / (pow(m, 1/(gamma-3)) - 1);  // known inversion of M(r)
        double costheta = math::random()*2 - 1;
        double phi = math::random()*2*M_PI;
        double R = r * sqrt(1 - pow_2(costheta));
        pts.add(coord::PosCar(R * cos(phi), p * R * sin(phi), q * r * costheta), 1./nbody);
    }
    return pts;
}

PtrPotential createFromFile(
    const particles::ParticleArray<coord::PosCar>& points, const std::string& potType)
{
    const std::string fileName = "test_potential_expansions.txt";
    writeSnapshot(fileName, points, "Text");
    // a rather lengthy way of setting parameters, used only for illustration:
    // normally these would be read from an INI file or from command line
    utils::KeyValueMap params;
    params.set("file", fileName);
    params.set("type", potType);
    params.set("gridSizeR", 20);
    params.set("gridSizeZ", 20);
    params.set("lmax", 6);
    PtrPotential newpot = potential::createPotential(params);
    std::remove(fileName.c_str());
    std::remove((fileName+potential::getCoefFileExtension(potType)).c_str());
    return newpot;
}

// test the accuracy of potential, force and density approximation at different radii
bool testAverageError(const potential::BasePotential& p1, const potential::BasePotential& p2, double eps)
{
    double gamma = getInnerDensitySlope(p2);
    std::string fileName = std::string("test_potential_") + p1.name() + "_" + p2.name() + 
        "_gamma" + utils::toString(gamma);
    std::ofstream strm;
    if(output) {
        writePotential(std::string("test_potential_") + p1.name() + "_gamma" + utils::toString(gamma), p1);
        strm.open(fileName.c_str());
    }
    // total density-weighted rms errors in potential, force and density, and total weight
    double totWeightedDifP=0, totWeightedDifF=0, totWeightedDifD=0, totWeight=0;
    const double dlogR=0.1;
    const int nptbin=1000;
    for(double logR=-4; logR<4; logR+=dlogR) {
        double weightedDifP=0, weightedDifF=0, weightedDifD=0, weight=0;
        for(int n=0; n<nptbin; n++) {
            coord::PosSph point( pow(10., logR+dlogR*n/nptbin),
                acos(math::random()*2-1), math::random()*2*M_PI);
            coord::GradCar g1, g2;
            double v1, v2, d1, d2;
            p1.eval(coord::toPosCar(point), &v1, &g1);
            p2.eval(coord::toPosCar(point), &v2, &g2);
            d1 = fmax(0, p1.density(point));
            d2 = fmax(0, p2.density(point));
            weightedDifP += pow_2((v1-v2) / v2) * pow_2(point.r) * d2;
            weightedDifF += (pow_2(g1.dx-g2.dx)+pow_2(g1.dy-g2.dy)+pow_2(g1.dz-g2.dz)) /
                (pow_2(g2.dx)+pow_2(g2.dy)+pow_2(g2.dz)) * pow_2(point.r) * d2;
            weightedDifD += d1==0 && d2==0 ? 0 : pow_2((d1-d2) / fmax(d1, d2)) * pow_2(point.r) * d2;
            weight += pow_2(point.r) * d2;
        }
        totWeightedDifP += weightedDifP;
        totWeightedDifF += weightedDifF;
        totWeightedDifD += weightedDifD;
        totWeight += weight;
        strm << pow(10., logR+0.5*dlogR) << '\t' <<
            sqrt(weightedDifP/weight) << '\t' <<
            sqrt(weightedDifF/weight) << '\t' <<
            sqrt(weightedDifD/weight) << '\n';
    }
    totWeightedDifP = sqrt(totWeightedDifP / totWeight);
    totWeightedDifF = sqrt(totWeightedDifF / totWeight);
    totWeightedDifD = sqrt(totWeightedDifD / totWeight);
    bool ok = totWeightedDifD<eps && totWeightedDifF<eps*0.1 && totWeightedDifP<eps*0.01;
    std::cout << p1.name() << " vs. " << p2.name() << 
        ": rmserror in potential=" << totWeightedDifP << 
        ", force=" << totWeightedDifF <<
        ", density=" << totWeightedDifD <<
        (ok ? "\n" : "\033[1;31m **\033[0m\n");
    return ok;
}

// test the accuracy of density approximation at different radii
bool testAverageError(const potential::BaseDensity& p1, const potential::BaseDensity& p2, double eps)
{
    double gamma = getInnerDensitySlope(p2);
    std::string fileName = std::string("test_density_") + p1.name() + "_" + p2.name() + 
        "_gamma" + utils::toString(gamma);
    std::ofstream strm;
    if(output)
        strm.open(fileName.c_str());
    double totWeightedDif=0, totWeight=0;  // total density-weighted rms error and total weight
    const double dlogR=0.1;
    const int nptbin=5000;
    for(double logR=-4; logR<4; logR+=dlogR) {
        double weightedDif=0, weight=0;
        for(int n=0; n<nptbin; n++) {
            double r     = pow(10., logR+dlogR*n/nptbin);
            double costh = math::random()*2-1;
            coord::PosCyl point( r*sqrt(1-pow_2(costh)), r*costh, math::random()*2*M_PI);
            double d1 = fmax(0, p1.density(point));
            double d2 = fmax(0, p2.density(point));
            weightedDif += d1==0 && d2==0 ? 0 :
                pow_2((d1-d2) / fmax(d1, d2)) * pow_2(r) * d2;
            weight += pow_2(r) * d2;
        }
        totWeightedDif += weightedDif;
        totWeight += weight;
        strm << pow(10., logR+0.5*dlogR) << '\t' << sqrt(weightedDif/weight) << '\n';
    }
    totWeightedDif = sqrt(totWeightedDif / totWeight);
    std::cout << p1.name() << " vs. " << p2.name() << 
        ": rmserror=" << totWeightedDif << 
    (totWeightedDif<eps ? "\n" : "\033[1;31m **\033[0m\n");
    return totWeightedDif<eps;
}

// test the consistency of spherical-harmonic expansion
bool testDensSH()
{
    const potential::Dehnen dens(1., 1., 1.2, 0.8, 0.5);
    std::vector<double> radii1 = math::createExpGrid(51, 0.01, 100);
    // twice denser grid: every other node coincides with that of the first grid
    std::vector<double> radii2 = math::createExpGrid(101,0.01, 100);
    std::vector<std::vector<double> > coefs1, coefs2;
    computeDensityCoefsSph(dens, math::SphHarmIndices(8, 6, dens.symmetry()), radii1, coefs1);
    potential::DensitySphericalHarmonic dens1(radii1, coefs1);
    // creating a sph-harm expansion from another s-h expansion:
    // should produce identical results if the location of radial grid points 
    // (partly) coincides with the first one, and the order of expansions lmax,mmax
    // are at least as large as the original ones (the extra coefs will be zero).
    computeDensityCoefsSph(dens1, math::SphHarmIndices(10, 8, dens1.symmetry()), radii2, coefs2);
    potential::DensitySphericalHarmonic dens2(radii2, coefs2);
    bool ok = true;
    // check that the two sets of coefs are identical at equal radii
    for(unsigned int c=0; c<coefs2.size(); c++)
        for(unsigned int k=0; k<radii1.size(); k++) {
            if(c<coefs1.size() && fabs(coefs1[c][k] - coefs2[c][k*2]) > 1e-13) {
                std::cout << "r=" << radii1[k]      << ", C1["<<c<<"]=" <<
                utils::toString(coefs1[c][k  ], 15) << ", C2["<<c<<"]=" <<
                utils::toString(coefs2[c][k*2], 15) << "\033[1;31m **\033[0m\n";
                ok = false;
            }
            if(c>=coefs1.size() && coefs2[c][k*2] != 0) {
                std::cout << "r=" << radii1[k] << ", C1["<<c<<"] is 0, C2["<<c<<"]=" <<
                utils::toString(coefs2[c][k*2], 15) << "\033[1;31m **\033[0m\n";
                ok = false;
            }
        }

    // test the accuracy at grid radii
    for(unsigned int k=0; k<radii1.size(); k++)
        for(double theta=0; theta<M_PI/2; theta+=0.31) 
            for(double phi=0; phi<M_PI; phi+=0.31) {
                coord::PosSph point(radii1[k], theta, phi);
                double d1 = dens1.density(point);
                double d2 = dens2.density(point);
                if(fabs(d1-d2) / fabs(d1) > 1e-12) { // two SH expansions should give the same result
                    std::cout << point << "dens1=" << utils::toString(d1, 15) <<
                    " dens2=" << utils::toString(d2, 15) << "\033[1;31m **\033[0m\n";
                    ok = false;
                }
            }
    return ok;
}

// definition of a single spherical-harmonic term with indices (l,m)
template<int l, int m>
double myfnc(double theta, double phi);
// first few spherical harmonics (with arbitrary normalization)
template<> double myfnc<0, 0>(double      , double    ) { return 1; }
template<> double myfnc<1,-1>(double theta, double phi) { return sin(theta)*sin(phi); }
template<> double myfnc<1, 0>(double theta, double    ) { return cos(theta); }
template<> double myfnc<1, 1>(double theta, double phi) { return sin(theta)*cos(phi); }
template<> double myfnc<2,-2>(double theta, double phi) { return (1-cos(2*theta))*sin(2*phi); }
template<> double myfnc<2,-1>(double theta, double phi) { return sin(theta)*cos(theta)*sin(phi); }
template<> double myfnc<2, 0>(double theta, double    ) { return 3*cos(2*theta)+1; }
template<> double myfnc<2, 1>(double theta, double phi) { return sin(theta)*cos(theta)*cos(phi); }
template<> double myfnc<2, 2>(double theta, double phi) { return (1-cos(2*theta))*cos(2*phi); }

// test spherical-harmonic transformation with the given set of indices and a given SH term (l,m)
template<int l, int m>
bool checkSH(const math::SphHarmIndices& ind)
{
    math::SphHarmTransformForward tr(ind);
    // array of original function values
    std::vector<double> d(tr.size());
    for(unsigned int i=0; i<d.size(); i++)
        d[i] = myfnc<l,m>(acos(tr.costheta(i)), tr.phi(i));
    // array of SH coefficients
    std::vector<double> c(ind.size());
    tr.transform(&d.front(), &c.front());
    math::eliminateNearZeros(c);
    // check that only one of them is non-zero
    unsigned int t0 = ind.index(l, m);  // index of the only non-zero coef
    for(unsigned int t=0; t<c.size(); t++)
        if((t==t0) ^ (c[t]!=0))  // xor operation
            return false;
    // array of function values after inverse transform
    std::vector<double> b(tr.size());
    for(unsigned int i=0; i<d.size(); i++) {
        double tau = tr.costheta(i) / (sqrt(1-pow_2(tr.costheta(i))) + 1);
        b[i] = math::sphHarmTransformInverse(ind, &c.front(), tau, tr.phi(i));
        if(fabs(d[i]-b[i]) > 2e-15)
            return false;
    }
    return true;
}

// a simple density profile that is constant within a sphere of given radius,
// which may be offset from the origin
class DensityBlob: public potential::BaseDensity{
public:
    double r, x, y, z;
    DensityBlob(double _r, double _x, double _y, double _z) : r(_r), x(_x), y(_y), z(_z) {}
    virtual coord::SymmetryType symmetry() const {
        return static_cast<coord::SymmetryType>(coord::ST_SPHERICAL*(x==0&&y==0&&z==0) |
        coord::ST_XREFLECTION*(x==0) | coord::ST_YREFLECTION*(y==0) | coord::ST_ZREFLECTION*(z==0));
    }
    virtual const char* name() const { return "Blob"; };
    virtual double densityCar(const coord::PosCar &pos) const {
        return (pow_2(pos.x-x) + pow_2(pos.y-y) + pow_2(pos.z-z) < pow_2(r)) ? 3/(4*M_PI*pow_3(r)) : 0;
    }
    virtual double densityCyl(const coord::PosCyl &pos) const { return densityCar(toPosCar(pos)); }
    virtual double densitySph(const coord::PosSph &pos) const { return densityCar(toPosCar(pos)); }
};

// test that the approximation of a "density blob" indeed closely resembles a sphere
bool testBlob(const potential::BaseDensity& approx, const DensityBlob& blob)
{
    const int npt=100000;
    particles::ParticleArray<coord::PosCar> points(galaxymodel::sampleDensity(approx, npt));
    double avgr = 0, avgr2 = 0;
    int npt_in_sphere = 0;
    for(int i=0; i<npt; i++) {
        double r = sqrt(
            pow_2(points[i].first.x - blob.x) +
            pow_2(points[i].first.y - blob.y) +
            pow_2(points[i].first.z - blob.z));
        if(r <= blob.r) {
            avgr += r;
            avgr2+= r*r;
            npt_in_sphere++;
        }
    }
    avgr /= npt_in_sphere;
    avgr2/= npt_in_sphere;
    bool ok = npt_in_sphere > npt * 0.85  &&
        fabs(avgr  / (0.75 * blob.r) - 1) < 0.05  &&
        fabs(avgr2 / (0.6 * pow_2(blob.r)) - 1) < 0.05;
    std::cout << "Blob-test for " << approx.name() <<
        " at (" << blob.x << ',' << blob.y << ',' << blob.z << "): " <<
        (npt_in_sphere*100./npt) << "% of mass is inside the sphere, "
        "<r>=" << avgr << ", <r^2>=" << avgr2 <<
        (ok ? "\n" : "\033[1;31m **\033[0m\n");
    return ok;
}

int main() {
    bool ok=true;

    // 1. check the correctness and reversibility of math::SphHarmTransform

    // perform several tests, some of them are expected to fail - because... (see comments below)
    ok &= checkSH<0, 0>(math::SphHarmIndices(4, 2, coord::ST_TRIAXIAL));
    ok &= checkSH<1,-1>(math::SphHarmIndices(4, 4, coord::ST_XREFLECTION));
        // axisymmetric implies z-reflection symmetry, but we have l=1 term
    ok &=!checkSH<1, 0>(math::SphHarmIndices(2, 0, coord::ST_AXISYMMETRIC));
        // while in this case it's only z-rotation symmetric, so a correct variant is below
    ok &= checkSH<1, 0>(math::SphHarmIndices(2, 0, coord::ST_ZROTATION));
    ok &= checkSH<1, 1>(math::SphHarmIndices(6, 3, coord::ST_YREFLECTION));
    ok &= checkSH<1, 1>(math::SphHarmIndices(3, 2, coord::ST_ZREFLECTION));
        // odd-m cosine term is not possible with x-reflection symmetry
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 1, coord::ST_XREFLECTION));
        // odd-l term is not possible with mirror symmetry
    ok &=!checkSH<1, 1>(math::SphHarmIndices(6, 2, coord::ST_REFLECTION));
        // y-reflection implies no sine terms (mmin==0), but we have them
    ok &=!checkSH<2,-2>(math::SphHarmIndices(6, 4, coord::ST_YREFLECTION));
        // x-reflection excludes even-m sine terms
    ok &=!checkSH<2,-2>(math::SphHarmIndices(5, 2, coord::ST_XREFLECTION));
    ok &= checkSH<2,-2>(math::SphHarmIndices(6, 4, coord::ST_ZREFLECTION));
        // terms with odd l+m are excluded under z-reflection 
    ok &=!checkSH<2,-1>(math::SphHarmIndices(2, 2, coord::ST_ZREFLECTION));
    ok &= checkSH<2,-1>(math::SphHarmIndices(2, 1,
        static_cast<coord::SymmetryType>(coord::ST_REFLECTION | coord::ST_XREFLECTION)));
    ok &= checkSH<2, 0>(math::SphHarmIndices(2, 2, coord::ST_AXISYMMETRIC));
    ok &= checkSH<2, 1>(math::SphHarmIndices(3, 1,
        static_cast<coord::SymmetryType>(coord::ST_REFLECTION | coord::ST_YREFLECTION)));
    ok &= checkSH<2, 2>(math::SphHarmIndices(5, 3, coord::ST_TRIAXIAL));
    if(!ok)
        std::cout << "Spherical-harmonic transform failed \033[1;31m**\033[0m\n";

    // 2. check the reversibility of SH expansion of density
    ok &= testDensSH();

    // 3. accuracy tests for density and potential approximations
    // 3a. original test profiles
    const potential::NFW test1_NFWSph(1., 1.);                           // spherical cuspy Navarro-Frenk-White
    const potential::Dehnen test1_Dehnen0Sph (1., 10., 0.0);             // spherical cored Dehnen
    const potential::Dehnen test2_Dehnen0Tri (1., 1.0, 0.0, 0.8, 0.5);   // triaxial cored Dehnen
    const potential::Dehnen test3_Dehnen15Tri(3., 5.0, 1.5, 0.8, 0.5);   // triaxial cuspy Dehnen
    const potential::MiyamotoNagai test4_MNAxi(1., 3.0, 0.5);            // axisymmetric Miyamoto-Nagai
    const potential::DiskParam test5_ExpdiskParam(1., 5., 0.5, 0, 0);    // double-exponential disk params
    const potential::DiskDensity test5_ExpdiskAxi(test5_ExpdiskParam);   // density profile of d-exp disk
    // potential of d-exp disk, computed using the GalPot approach
    PtrPotential test5_Galpot = potential::createPotential(utils::KeyValueMap(
        "type=Disk surfaceDensity=1 scaleRadius=5 scaleHeight=0.5"));
    const potential::Dehnen test6_Dehnen05Tri(1., 1., 0.5, 0.8, 0.5);    // triaxial weakly cuspy
    // N-body representation of the same profile
    particles::ParticleArray<coord::PosCar> test6_points = makeDehnen(100000, 0.5, 0.8, 0.5);
    const DensityBlob test7x(1., 0.6, 0.0, 0.0);   // a constant-density sphere shifted along x-axis
    const DensityBlob test7y(1., 0.0, 0.7, 0.0);   // same for y-axis
    const DensityBlob test7z(1., 0.0, 0.0, 0.8);   // same for z-axis
    const DensityBlob test7d(1., 0.3, 0.4, 0.5);   // shifted diagonally

    // 3b. test the approximating density profiles
    std::cout << "--- Testing accuracy of density profile interpolators: "
        "print density-weighted rms error in density ---\n";
    ok &= testAverageError(
        *potential::DensitySphericalHarmonic::create(test2_Dehnen0Tri, 10, 10, 30, 1e-2, 1e3),
        test2_Dehnen0Tri, 0.005);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test2_Dehnen0Tri, 10, 30, 1e-2, 1e3, 30, 1e-2, 1e3),
        test2_Dehnen0Tri, 0.005);
    ok &= testAverageError(
        *potential::DensitySphericalHarmonic::create(test3_Dehnen15Tri, 10, 10, 30, 1e-3, 1e3),
        test3_Dehnen15Tri, 0.005);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test3_Dehnen15Tri, 10, 40, 1e-3, 1e3, 40, 1e-3, 1e3),
        test3_Dehnen15Tri, 0.5);
    ok &= testAverageError(
        *potential::DensitySphericalHarmonic::create(test4_MNAxi, 40, 0, 40, 1e-2, 1e2),
        test4_MNAxi, 0.5);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test4_MNAxi, 0, 20, 1e-2, 1e2, 20, 1e-2, 1e2),
        test4_MNAxi, 0.05);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test4_MNAxi, 0, 40, 1e-2, 2e3, 40, 1e-2, 2e2),
        test4_MNAxi, 0.005);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test5_ExpdiskAxi, 0, 20, 1e-2, 50, 20, 1e-2, 20),
        test5_ExpdiskAxi, 0.05);
    ok &= testAverageError(
        *potential::DensityAzimuthalHarmonic::create(test5_ExpdiskAxi, 0, 30, 1e-2, 100, 30, 1e-2, 100),
        test5_ExpdiskAxi, 0.005);
    ok &= testAverageError(*test5_Galpot,
        test5_ExpdiskAxi, 0.05);
    ok &= testAverageError(
        *potential::CylSpline::create(test5_ExpdiskAxi, 0, 30, 1e-2, 100, 30, 1e-2, 50),
        test5_ExpdiskAxi, 0.05);

    // 3c. test the approximating potential profiles
    std::cout << "--- Testing potential approximations: "
    "print density-averaged rms errors in potential, force and density ---\n";

    // spherical, cuspy and infinite
    std::cout << "--- Spherical NFW ---\n";
    PtrPotential test1n = potential::Multipole::create(test1_NFWSph, 0, 0, 20, 1e-3, 1e3);
    ok &= testAverageError(*test1n, test1_NFWSph, 0.001);

    // spherical, cored
    std::cout << "--- Spherical Dehnen gamma=0 ---\n";
    PtrPotential test1c = potential::CylSpline::create(
        // this forces potential to be computed via integration of density over volume
        static_cast<const potential::BaseDensity&>(test1_Dehnen0Sph), 0, 20, 0., 0., 20, 0., 0.);
    PtrPotential test1m = potential::Multipole::create(
        static_cast<const potential::BaseDensity&>(test1_Dehnen0Sph), 0, 0, 20);
    ok &= testAverageError(*test1m, test1_Dehnen0Sph, 0.002);
    ok &= testAverageError(*test1c, test1_Dehnen0Sph, 0.02);

    // mildly triaxial, cored
    std::cout << "--- Triaxial Dehnen gamma=0 ---\n";
    PtrPotential test2m = potential::Multipole::create(
        static_cast<const potential::BaseDensity&>(test2_Dehnen0Tri), 8, 6, 20);
    clock_t clock = std::clock();
    PtrPotential test2c = potential::CylSpline::create(  // from density via integration
        static_cast<const potential::BaseDensity&>(test2_Dehnen0Tri), 6, 20, 0., 0., 20, 0., 0.);
    std::cout << (std::clock()-clock)*1.0/CLOCKS_PER_SEC << " seconds to create CylSpline\n";
    PtrPotential test2d = potential::CylSpline::create(  // directly from potential
        test2_Dehnen0Tri, 6, 20, 0., 0., 20, 0., 0.);
    PtrPotential test2c_clone = writeRead(*test2c);
    ok &= testAverageError(*test2m, test2_Dehnen0Tri, 0.01);
    ok &= testAverageError(*test2d, test2_Dehnen0Tri, 0.02);
    ok &= testAverageError(*test2c, test2_Dehnen0Tri, 0.02);
    ok &= testAverageError(*test2c, *test2c_clone, 3e-4);

    // mildly triaxial, cuspy
    std::cout << "--- Triaxial Dehnen gamma=1.5 ---\n";
    PtrPotential test3m = potential::Multipole::create(
        static_cast<const potential::BaseDensity&>(test3_Dehnen15Tri), 6, 6, 20);
    PtrPotential test3m_clone = writeRead(*test3m);
    ok &= testAverageError(*test3m, test3_Dehnen15Tri, 0.02);
    ok &= testAverageError(*test3m, *test3m_clone, 1e-9);

    // strongly flattened exp.disk; the 'true' potential is not available,
    // so we compare two approximations: GalPot and CylSpline
    std::cout << "--- Axisymmetric ExpDisk ---\n";
    PtrPotential test5c = potential::CylSpline::create(
        test5_ExpdiskAxi, 0, 20, 5e-2, 50., 20, 1e-2, 10.);
    ok &= testAverageError(*test5c, *test5_Galpot, 0.05);

    // mildly triaxial, created from N-body samples
    std::cout << "--- Triaxial Dehnen gamma=0.5 from N-body samples ---\n";
    PtrPotential test6c = potential::CylSpline::create(test6_points,
        coord::ST_TRIAXIAL, 6, 20, 0., 0., 20, 0., 0.);
    PtrPotential test6m = potential::Multipole::create(test6_points, coord::ST_TRIAXIAL, 6, 6, 20);
    ok &= testAverageError(*test6m, test6_Dehnen05Tri, 1.0);
    ok &= testAverageError(*test6c, test6_Dehnen05Tri, 1.5);

    std::cout << "--- Testing the accuracy of representation of an off-centered constant-density sphere ---"
        "\n--- Ideally all mass should be contained within the sphere radius, <r>=3/4, <r^2>=3/5 ---\n";
    ok &= testBlob(*potential::Multipole::create(test7x, 8, 8, 40, 0.05, 2.0), test7x);
    ok &= testBlob(*potential::Multipole::create(test7y, 8, 8, 40, 0.05, 2.0), test7y);
    ok &= testBlob(*potential::Multipole::create(test7z, 8, 8, 40, 0.05, 2.0), test7z);
    ok &= testBlob(*potential::Multipole::create(test7d, 8, 8, 40, 0.05, 2.0), test7d);
    ok &= testBlob(*potential::CylSpline::create(test7x, 6, 20, 0.1, 2.0, 10, 0.1, 1.0), test7x);
    ok &= testBlob(*potential::CylSpline::create(test7y, 6, 20, 0.1, 2.0, 10, 0.1, 1.0), test7y);
    ok &= testBlob(*potential::CylSpline::create(test7z, 6, 10, 0.1, 1.0, 20, 0.1, 2.0), test7z);
    ok &= testBlob(*potential::CylSpline::create(test7d, 6, 15, 0.1, 1.5, 15, 0.1, 1.5), test7d);

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}