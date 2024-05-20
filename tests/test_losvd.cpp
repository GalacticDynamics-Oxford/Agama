/** \name   test_losvd.cpp
    \author Eugene Vasiliev
    \date   2017-2019

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
*/
#include "galaxymodel_losvd.h"
#include "math_core.h"
#include "math_random.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>

// whether to produce an output file and a plotting script for gnuplot
const bool output = utils::verbosityLevel >= utils::VL_VERBOSE;

int main()
{
    bool ok = true;
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
    for(size_t a=0; a<numApertures; a++) {
        double sum = 0;
        for(size_t v=0; v<aper.cols(); v++)
            sum += aper(a,v) * velint[v];
        std::cout << "Aperture " << a << " value = " << sum << ", expected = " << expected[a];
        if(fabs(sum - expected[a]) > 1e-4) {
            ok = false;
            std::cout << " \033[1;31m**\033[0m";
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
