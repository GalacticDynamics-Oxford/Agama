#include "math_spline.h"
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
    mathutils::createNonuniformGrid(NNODES, XMIN, XMAX, true, xnodes);
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = rand()*XMAX/RAND_MAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(rand()*1./RAND_MAX-0.5)*4;
    }
    mathutils::SplineApprox appr(xvalues, xnodes);
    if(appr.isSingular())
        std::cout << "Warning, matrix is singular\n";

    std::vector<double> ynodes1, ynodes2;
    double deriv_left, deriv_right, rms, edf, lambda;

    appr.fitDataOptimal(yvalues1, ynodes1, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case A: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    mathutils::CubicSpline fit1(xnodes, ynodes1, deriv_left, deriv_right);
    ok &= rms<0.1 && edf>=2 && edf<NNODES+2 && lambda>0;

    appr.fitDataOversmooth(yvalues2, .5, ynodes2, deriv_left, deriv_right, &rms, &edf, &lambda);
    std::cout << "case B: RMS="<<rms<<", EDF="<<edf<<", lambda="<<lambda<<"\n";
    mathutils::CubicSpline fit2(xnodes, ynodes2, deriv_left, deriv_right);
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

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
