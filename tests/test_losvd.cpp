/** \name   test_losvd.cpp
    \author Eugene Vasiliev
    \date   2017-2026

    This program tests the computation of PSF-convolved aperture masses
    (integrals of LOSVD over the velocity dimension).
    We put a dozen points on the XY plane and a similar number of rectangular 'apertures'
    rotated at arbitrary angles.
    Each point is then convolved with a gaussian PSF, and its contribution to each of these apertures
    is computed both analytically and numerically, using the B-spline representation of the LOSVD
    datacube on a regular grid in the XY plane, convolved and rebinned onto the apertures.
    The results should agree to high accuracy.
    This test involves the math_geometry module (intersection of arbitrarily shaped polygons/apertures
    with the regular pixels of the LOSVD datacube), and the galaxymodel_losvd module (computation
    of LOSVDs and PSF convolution).

    In addition (unrelated to the above tests, but still relevant for the LOSVD representation),
    it tests the accuracy of computation of Gauss-Hermite moments from B-spline functions.
*/
#include "galaxymodel_losvd.h"
#include "math_core.h"
#include "math_gausshermite.h"
#include "math_random.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

// whether to produce an output file and a plotting script for gnuplot
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

// error indicator printed in case of test failures
const char err[] = " \033[1;31m**\033[0m";

// order of the Gauss-Hermite expansions for the test
const int GHORDER = 10;

// to check the scale-invariance of GH moment computation, all test functions
// are scaled in amplitude and spatial size by the large coefficients below
const double AMPL = 1e30, SIZE = 1e-10;

// basis and Gauss-Hermite coefficients for the rectangle-shaped function on [-0.5 .. 0.5]
const double GHBASIS0[3] = {1.061561973177746, 0, 0.3571466130807738};
const double GHCOEFS0[GHORDER+1] = {1, 0, 0,
    0, -0.18777738377182350, 0,  0.1577856826917629,
    0, -0.07224243252790478, 0, -0.006536202603263886};

// same for a sawtooth function that rises from 0 at x=-1 to 1 at x=0.5 and falls back to 0 at x=1
const double GHBASIS1[3] = {1.011701990345273, 0.2429530678299940, 0.4502615552291587};
const double GHCOEFS1[GHORDER+1] = {1, 0, 0,
    -0.1688712732469682,  -0.03065785371567972,  0.08028935406658369, 0.02845749201370504,
    -0.00933975268261174, -0.03200862579566654, -0.01064457811888961, 0.02569901243544993};

// same for the bell-shaped curve defined below, which has infinite extent
const double GHBASIS2[3] = {1.452339214350519, 1.3, 0.6100157521613355};
const double GHCOEFS2[GHORDER+1] = {1, 0, 0,
    0, 0.07646932144461256, 0, 0.008956722097210347,
    0, 0.02001224050019141, 0, 0.006057579882561532};

/// A simple bell-shaped curve for testing the Gauss-Hermite expansion
class BellCurve: public math::IFunctionNoDeriv {
public:
    virtual double value(double x) const {
        return AMPL / SIZE / pow_2(pow_2(x / SIZE - GHBASIS2[1]) + 1);
    }
};

std::string str(double x, int prec=8)
{
    std::string result = utils::toString(x, fabs(x) <= 1e-10 ? 3 : prec);
    if(result.size() < 8)
        result.resize(8, ' ');
    return result;
}


bool testGH()
{
    std::cout << "param\ttrue_value\tfree_basis\tdifference\tfixed_basis\tdifference\n";

    std::cout << "Test 0: rectangle block\n";
    std::vector<double> grid0(2);
    grid0[0] = -0.5 * SIZE;
    grid0[1] =  0.5 * SIZE;
    std::vector<double> ampl0(1, 1.0 * AMPL / SIZE);
    // first approach: construct the matrix for converting the amplitudes of B-spline expansion
    // into GH coefficients, using a fixed GH basis (true parameters of the best-fit Gaussian)
    math::Matrix<double> ghmat0 = math::computeGaussHermiteMatrix(0, grid0, GHORDER,
        GHBASIS0[0] * AMPL, GHBASIS0[1] * SIZE, GHBASIS0[2] * SIZE);
    // second approach: find the parameters of the best-fit Gaussian and then compute the GH coefs
    // using a different function; this tests both the Gaussian fitting and the computation of
    // coefficients for an interpolator function rather than separately for B-spline basis functions
    math::GaussHermiteExpansion ghexp0(math::BsplineWrapper<0>(grid0, ampl0), GHORDER);
    double dif0fix = 0, dif0free = std::max(
        fabs(ghexp0.ampl()  / AMPL - GHBASIS0[0]), std::max(
        fabs(ghexp0.center()/ SIZE - GHBASIS0[1]),
        fabs(ghexp0.width() / SIZE - GHBASIS0[2])));
    std::cout << "ampl\t"      + str(GHBASIS0[0]) + '\t' +
        str(ghexp0.ampl()   / AMPL              ) + '\t' +
        str(ghexp0.ampl()   / AMPL - GHBASIS0[0]) + '\n';
    std::cout << "center\t"    + str(GHBASIS0[1]) + '\t' +
        str(ghexp0.center() / SIZE              ) + '\t' +
        str(ghexp0.center() / SIZE - GHBASIS0[1]) + '\n';
    std::cout << "width\t"     + str(GHBASIS0[2]) + '\t' +
        str(ghexp0.width()  / SIZE              ) + '\t' +
        str(ghexp0.width()  / SIZE - GHBASIS0[2]) + '\n';
    for(int i=0; i<=GHORDER; i++) {
        double hfix = ghmat0(i, 0) * ampl0[0], hfree = ghexp0.coefs()[i];
        std::cout << 'h' + utils::toString(i) + '\t' + str(GHCOEFS0[i]) + '\t' +
            str(hfree) + '\t' + str(hfree - GHCOEFS0[i]) + '\t' +
            str(hfix)  + '\t' + str(hfix  - GHCOEFS0[i]) + '\n';
        dif0free = std::max(dif0free, fabs(hfree - GHCOEFS0[i]));
        dif0fix  = std::max(dif0fix,  fabs(hfix  - GHCOEFS0[i]));
    }
    bool ok0 = dif0fix < 1e-12 && dif0free < 1e-12;
    std::cout << "Max error with freely-fitted basis:     " + str(dif0free) +
        ",\tfixed basis:    " + str(dif0fix) + (ok0 ? "" : err) + '\n';

    std::cout << "Test 1: skewed sawtooth function\n";
    std::vector<double> grid1(3);
    grid1[0] = -1.0 * SIZE;
    grid1[1] =  0.5 * SIZE;
    grid1[2] =  1.0 * SIZE;
    std::vector<double> ampl1(3);
    ampl1[1] = 1.0 * AMPL / SIZE;
    math::Matrix<double> ghmat1 = math::computeGaussHermiteMatrix(1, grid1, GHORDER,
        GHBASIS1[0] * AMPL, GHBASIS1[1] * SIZE, GHBASIS1[2] * SIZE);
    math::GaussHermiteExpansion ghexp1(math::BsplineWrapper<1>(grid1, ampl1), GHORDER);
    double dif1fix = 0, dif1free = std::max(
        fabs(ghexp1.ampl()  / AMPL - GHBASIS1[0]), std::max(
        fabs(ghexp1.center()/ SIZE - GHBASIS1[1]),
        fabs(ghexp1.width() / SIZE - GHBASIS1[2])));
    std::cout << "ampl\t"      + str(GHBASIS1[0]) + '\t' +
        str(ghexp1.ampl()   / AMPL              ) + '\t' +
        str(ghexp1.ampl()   / AMPL - GHBASIS1[0]) + '\n';
    std::cout << "center\t"    + str(GHBASIS1[1]) + '\t' +
        str(ghexp1.center() / SIZE              ) + '\t' +
        str(ghexp1.center() / SIZE - GHBASIS1[1]) + '\n';
    std::cout << "width\t"     + str(GHBASIS1[2]) + '\t' +
        str(ghexp1.width()  / SIZE              ) + '\t' +
        str(ghexp1.width()  / SIZE - GHBASIS1[2]) + '\n';
    for(int i=0; i<=GHORDER; i++) {
        double hfix = ghmat1(i, 1) * ampl1[1], hfree = ghexp1.coefs()[i];
        std::cout << 'h' + utils::toString(i) + '\t' + str(GHCOEFS1[i]) + '\t' +
            str(hfree) + '\t' + str(hfree - GHCOEFS1[i]) + '\t' +
            str(hfix)  + '\t' + str(hfix  - GHCOEFS1[i]) + '\n';
        dif1free = std::max(dif1free, fabs(hfree - GHCOEFS1[i]));
        dif1fix  = std::max(dif1fix,  fabs(hfix  - GHCOEFS1[i]));
    }
    bool ok1 = dif1fix < 1e-12 && dif1free < 1e-12;
    std::cout << "Max error with freely-fitted basis:     " + str(dif1free) +
        ",\tfixed basis:    " + str(dif1fix) + (ok1 ? "" : err) + '\n';

    std::cout << "Test 2: bell-shaped function\n";
    // first approach: fix the parameters of the base Gaussian to the true values
    math::GaussHermiteExpansion ghexp2fix (BellCurve(), GHORDER,
        GHBASIS2[0] * AMPL, GHBASIS2[1] * SIZE, GHBASIS2[2] * SIZE);
    // second approach: find the parameters of the best-fit Gaussian
    math::GaussHermiteExpansion ghexp2free(BellCurve(), GHORDER);
    double dif2fix = 0, dif2free = std::max(
        fabs(ghexp2free.ampl()  / AMPL - GHBASIS2[0]), std::max(
        fabs(ghexp2free.center()/ SIZE - GHBASIS2[1]),
        fabs(ghexp2free.width() / SIZE - GHBASIS2[2])));
    std::cout << "ampl\t"          + str(GHBASIS2[0]) + '\t' +
        str(ghexp2free.ampl()   / AMPL              ) + '\t' +
        str(ghexp2free.ampl()   / AMPL - GHBASIS2[0]) + '\n';
    std::cout << "center\t"        + str(GHBASIS2[1]) + '\t' +
        str(ghexp2free.center() / SIZE              ) + '\t' +
        str(ghexp2free.center() / SIZE - GHBASIS2[1]) + '\n';
    std::cout << "width\t"         + str(GHBASIS2[2]) + '\t' +
        str(ghexp2free.width()  / SIZE              ) + '\t' +
        str(ghexp2free.width()  / SIZE - GHBASIS2[2]) + '\n';
    for(int i=0; i<=GHORDER; i++) {
        double hfix = ghexp2fix.coefs()[i], hfree = ghexp2free.coefs()[i];
        std::cout << 'h' + utils::toString(i) + '\t' + str(GHCOEFS2[i]) + '\t' +
            str(hfree) + '\t' + str(hfree - GHCOEFS2[i]) + '\t' +
            str(hfix)  + '\t' + str(hfix  - GHCOEFS2[i]) + '\n';
        dif2free = std::max(dif2free, fabs(hfree - GHCOEFS2[i]));
        dif2fix  = std::max(dif2fix,  fabs(hfix  - GHCOEFS2[i]));
    }
    bool ok2 = dif2fix < 1e-6 && dif2free < 1e-4;
    std::cout << "Max error with freely-fitted basis:     " + str(dif2free) +
        ",\tfixed basis:    " + str(dif2fix) + (ok2 ? "" : err) + '\n';

    return ok0 && ok1 && ok2;
}

int main()
{
    bool ok = true;
    if(!testGH()) {
        std::cout << "Gauss-Hermite test failed" << err;
        ok = false;
    }
    const int DEGREE = 3;
    const double pixelSize = 0.125;
    const double velbin    = 7.;
    const int gridSizeX = 8. /pixelSize;
    const int gridSizeY = 10./pixelSize;
    const int gridSize  = std::min(gridSizeX, gridSizeY);
    const int gridSizeV = 9;

    // generate random points
    const size_t numPoints = 24;
    std::vector<math::Point2d> points;
    for(size_t p=0; p<numPoints; ++++p) {
        double x = (math::random()-0.5) * gridSizeX * pixelSize * 0.8;
        double y = (math::random()-0.5) * gridSizeY * pixelSize * 0.8;
        // add the point and its reflected copy
        points.push_back(math::Point2d( x,  y));
        points.push_back(math::Point2d(-x, -y));
    }

    // construct gaussian PSFs
    const size_t numPsf = 2;
    std::vector<galaxymodel::GaussianPSF> psf(numPsf);
    psf[0].width = 0.25;  // narrow component
    psf[0].ampl  = 0.7;   // 70% of total weight
    psf[1].width = 1.2;   // wide component
    psf[1].ampl  = 0.3;   // remaining 30% of total weight

    // construct arbitrarily rotated rectangular apertures
    const size_t numApertures = 10;
    std::vector<double> expected(numApertures);  // analytically computed values for each aperture
    std::vector<math::Polygon> apertures;
    for(size_t a=0; a<numApertures; a++) {
        double angle = math::random() * 2*M_PI;
        double size1 = math::random() * gridSize * pixelSize * 0.2;
        double size2 = math::random() * gridSize * pixelSize * 0.2;
        double posx  = (math::random()-0.5) * gridSizeX * pixelSize * 0.55;
        double posy  = (math::random()-0.5) * gridSizeY * pixelSize * 0.55;
        math::Polygon ap(4);
        ap[0].x = posx - size1 * cos(angle) + size2 * sin(angle);
        ap[0].y = posy - size1 * sin(angle) - size2 * cos(angle);
        ap[1].x = posx + size1 * cos(angle) + size2 * sin(angle);
        ap[1].y = posy + size1 * sin(angle) - size2 * cos(angle);
        ap[2].x = posx + size1 * cos(angle) - size2 * sin(angle);
        ap[2].y = posy + size1 * sin(angle) + size2 * cos(angle);
        ap[3].x = posx - size1 * cos(angle) - size2 * sin(angle);
        ap[3].y = posy - size1 * sin(angle) + size2 * cos(angle);
        apertures.push_back(ap);
        // compute analytic integrals over psfs for all points
        for(size_t p=0; p<numPoints; p++) {
            // coordinates of the point in the rotated and shifted aperture coord.frame
            double x = (points[p].x-posx) * cos(angle) + (points[p].y-posy) * sin(angle);
            double y = (points[p].y-posy) * cos(angle) - (points[p].x-posx) * sin(angle);
            for(size_t g=0; g<numPsf; g++) {
                expected[a] += 0.25 * psf[g].ampl *
                    (math::erf((x+size1) / M_SQRT2 / psf[g].width) - math::erf((x-size1) / M_SQRT2 / psf[g].width)) *
                    (math::erf((y+size2) / M_SQRT2 / psf[g].width) - math::erf((y-size2) / M_SQRT2 / psf[g].width));
            }
        }
    }

    // construct the LOSVD recorder
    galaxymodel::LOSVDParams params;
    params.alpha = 0.;     // viewing angles for transforming the intrinsic to projected coords
    params.beta  = 0.;
    params.gamma = 0.;
    params.gridx = math::createUniformGrid(gridSizeX+1, -0.5*gridSizeX*pixelSize, 0.5*gridSizeX*pixelSize);
    params.gridy = params.gridx;
    params.gridv = math::createUniformGrid(gridSizeV+1, -0.5*gridSizeV*velbin, 0.5*gridSizeV*velbin);
    params.spatialPSF = psf;       // spatial psf
    params.velocityPSF = 0.;       // velocity psf
    params.apertures = apertures;  // polygons defining the apertures
    galaxymodel::TargetLOSVD<DEGREE> lgrid(params);

    // compute integrals over basis functions of the velocity grid
    math::FiniteElement1d<DEGREE> velfem(math::BsplineInterpolator1d<DEGREE>(params.gridv));
    std::vector<double> velint = velfem.computeProjVector(
        std::vector<double>(velfem.integrPoints().size(), 1.));

    // add points to the datacube
    math::Matrix<double> datacube = lgrid.newDatacube();
    double pp[6] = {0.};
    for(size_t p=0; p<numPoints; p++) {
        pp[0] = points[p].x;
        pp[1] = points[p].y;
        pp[5] = (math::random()-0.5) * gridSizeV * velbin;
        lgrid.addPoint(pp, 1., datacube.data());
    }

    // obtain amplitudes of b-spline decomposition
    math::Matrix<galaxymodel::StorageNumT> aper(numApertures, velfem.interp.numValues());
    lgrid.finalizeDatacube(datacube, aper.data());

    // compare the computed aperture masses (LOSVD collapsed along the v_z dimension) from
    // the B-spline representation with the analytical expectations
    std::cout << "\naperture  true_value\tresult  \tdifference\n";
    for(size_t a=0; a<numApertures; a++) {
        double result = 0;
        for(size_t v=0; v<aper.cols(); v++)
            result += aper(a,v) * velint[v];
        std::cout << utils::toString(a) + '\t' + str(expected[a]) + '\t' +
            str(result) + '\t' + str(result - expected[a]);
        if(fabs(result - expected[a]) > 1e-4) {
            ok = false;
            std::cout << err;
        }
        std::cout << "\n";
    }

    if(output) {
        std::ofstream strm("test_losvd.dat");
        strm << "#Points(x,y):\n";
        for(size_t p=0; p<numPoints; p++) {
            strm << points[p].x << ' ' << points[p].y << '\n';
        }
        strm << "\n#Rectangular apertures:\n";
        for(size_t a=0; a<numApertures; a++) {
            strm <<
            apertures[a][0].x << ' ' << apertures[a][0].y << '\n' <<
            apertures[a][1].x << ' ' << apertures[a][1].y << '\n' <<
            apertures[a][2].x << ' ' << apertures[a][2].y << '\n' <<
            apertures[a][3].x << ' ' << apertures[a][3].y << '\n' <<
            apertures[a][0].x << ' ' << apertures[a][0].y << '\n' << '\n';
        }
        strm.close();
        strm.open("test_losvd.plt");  // gnuplot script
        strm << "set term pdf size 10cm,10cm\nset output 'test_losvd.pdf'\nset key off\n"
            "plot 'test_losvd.dat' every :::::0 u 1:2 pt 7, '' every :::1 u 1:2 w l, \\\n"
            "    '' every :::1 u 1:2:(sprintf('%d',column(-1)-1)) w labels, \\\n"
            "    " << params.gridy[0];
        for(size_t i=0; i<numPsf; i++)
            strm << " + " << psf[i].ampl << "/" << psf[i].width <<
                "/sqrt(2*pi) * exp(-0.5*(x/" << psf[i].width << ")**2)";
        strm << "\n";
    }

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
