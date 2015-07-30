#include "math_spline.h"
#include "math_core.h"
#include <iostream>
#include <fstream>
#include <cmath>

const int NNODES  = 20;
const int NPOINTS = 10000;
const double XMIN = 0.2;
const double XMAX = 12.;
const double DISP = 0.5;  // y-dispersion
const bool OUTPUT = false;
int main()
{
    bool ok=true;
    std::vector<double> xnodes(NNODES);
    math::createNonuniformGrid(NNODES, XMIN, XMAX, true, xnodes);
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = rand()*XMAX/RAND_MAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5)*4;
    }
    math::SplineApprox appr(xvalues, xnodes);
    if(appr.isSingular())
        std::cout << "Warning, matrix is singular\n";

    std::vector<double> ynodes1, ynodes2;
    double deriv_left, deriv_right, rms, edf, lambda;

    appr.fitDataOptimal(yvalues1, ynodes1, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case A: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    math::CubicSpline fit1(xnodes, ynodes1, deriv_left, deriv_right);
    ok &= rms<0.1 && edf>=2 && edf<NNODES+2 && lambda>0;

    appr.fitDataOversmooth(yvalues2, .5, ynodes2, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case B: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    math::CubicSpline fit2(xnodes, ynodes2, deriv_left, deriv_right);
    ok &= rms<1.0 && edf>=2 && edf<NNODES+2 && lambda>0;

    if(OUTPUT) {
        std::ofstream strm("test_math_spline.dat");
        for(size_t i=0; i<xnodes.size(); i++)
            strm << xnodes[i] << "\t" << ynodes1[i] << "\t" << ynodes2[i] << "\n";
        strm << "\n";
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << yvalues1[i] << "\t" << yvalues2[i] << "\t" <<
                fit1(xvalues[i]) << "\t" << fit2(xvalues[i]) << "\n";
    }

    // test 2d spline
    const int NNODESX=8;
    const int NNODESY=4;
    const int NN=99;    // number of intermediate points for checking the values
    std::vector<double> xval(NNODESX,0);
    std::vector<double> yval(NNODESY,0);
    std::vector< std::vector<double> > zval(NNODESX);
    for(int i=1; i<NNODESX; i++)
        xval[i] = xval[i-1] + rand()*1.0/RAND_MAX + 0.5;
    for(int j=1; j<NNODESY; j++)
        yval[j] = yval[j-1] + rand()*1.0/RAND_MAX + 0.5;
    for(int i=0; i<NNODESX; i++) {
        zval[i].resize(NNODESY);
        for(int j=0; j<NNODESY; j++)
            zval[i][j] = rand()*1.0/RAND_MAX;
    }
    math::CubicSpline2d spl2d(xval, yval, zval, 0., NAN, 1., -1.);
    // compare values and derivatives at grid nodes
    for(int i=0; i<NNODESX; i++) {
        double z, dy;
        spl2d.evalDeriv(xval[i], yval.front(), &z, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0 && math::fcmp(z, zval[i].front(), 1e-13)==0;
        spl2d.evalDeriv(xval[i], yval.back(), &z, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0 && math::fcmp(z, zval[i].back(), 1e-13)==0;
    }
    for(int j=0; j<NNODESY; j++) {
        double z, dx;
        spl2d.evalDeriv(xval.front(), yval[j], &z, &dx);
        ok &= math::fcmp(dx, 0.)==0 && math::fcmp(z, zval.front()[j], 1e-13)==0;
        spl2d.evalDeriv(xval.back(), yval[j], &z, &dx);
        ok &= fabs(dx)<10 && math::fcmp(z, zval.back()[j], 1e-13)==0;
    }
    // compare derivatives on the entire edge
    for(int i=0; i<=NN; i++) {
        double x = i*xval.back()/NN;
        double dy;
        spl2d.evalDeriv(x, yval.front(), NULL, NULL, &dy);
        ok &= math::fcmp(dy, 1., 1e-13)==0;
        spl2d.evalDeriv(xval[i], yval.back(), NULL, NULL, &dy);
        ok &= math::fcmp(dy, -1., 1e-13)==0;
        double y = i*yval.back()/NN;
        double dx;
        spl2d.evalDeriv(xval.front(), y, NULL, &dx);
        ok &= math::fcmp(dx, 0.)==0;
    }

    if(OUTPUT) {
        std::ofstream strm("test_math_spline2d.dat");
        for(int i=0; i<=NN; i++) {  // output for Gnuplot splot routine
            double x = i*xval.back()/NN;
            for(int j=0; j<=NN; j++) {
                double y = j*yval.back()/NN;
                double z, dx, dy, dxy;
                spl2d.evalDeriv(x, y, &z, &dx, &dy, NULL, &dxy, NULL);
                ok &= z>=-1 && z<=2;
                strm << x << " " << y << " " << z << " " << dx << " " << dy << " " << dxy << "\n";
            }
            strm << "\n";
        }
    }

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
