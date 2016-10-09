//#define COMPARE_WD_PSPLINE
#define STRESS_TEST

#include "math_spline.h"
#include "math_core.h"
#include "math_sphharm.h"
#include "math_sample.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>˚
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>
#ifdef COMPARE_WD_PSPLINE
#include "torus/WD_Pspline.h"
#endif

const bool OUTPUT = utils::verbosityLevel >= utils::VL_VERBOSE;

// provides the integral of sin(x)*x^n
class testfnc: public math::IFunctionIntegral {
    virtual double integrate(double x1, double x2, int n=0) const {
        return antideriv(x2,n)-antideriv(x1,n);
    }
    double antideriv(double x, int n) const {
        switch(n) {
            case 0: return -cos(x);
            case 1: return -cos(x)*x+sin(x);
            case 2: return  cos(x)*(2-x*x) + 2*x*sin(x);
            case 3: return  cos(x)*x*(6-x*x) + 3*sin(x)*(x*x-2);
            default: return NAN;
        }
    }
};

// provides the integrand for numerical integration of sin(x)*f(x)
class testfncint: public math::IFunctionNoDeriv {
public:
    testfncint(const math::IFunction& _f): f(_f) {};
    virtual double value(const double x) const {
        return sin(x) * f(x);
    }
private:
    const math::IFunction& f;
};

// provides the integrand for numerical integration of f(x)^2
class squaredfnc: public math::IFunctionNoDeriv {
public:
    squaredfnc(const math::IFunction& _f): f(_f) {};
    virtual double value(const double x) const {
        return pow_2(f(x));
    }
private:
    const math::IFunction& f;
};

// test the integration of a spline function
bool test_integral(const math::CubicSpline& f, double x1, double x2)
{
    double result_int = f.integrate(x1, x2);
    double result_ext = math::integrateAdaptive(f, x1, x2, 1e-10);
    std::cout << "Ordinary intergral on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    result_int = f.integrate(x1, x2, testfnc());
    result_ext = math::integrateAdaptive(testfncint(f), x1, x2, 1e-10);
    std::cout << "Weighted intergral on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    result_int = f.integrate(x1, x2, f);
    result_ext = math::integrateAdaptive(squaredfnc(f), x1, x2, 1e-10);
    std::cout << "Integral of f(x)^2 on ["<<x1<<":"<<x2<<
        "]: internal routine = "<<result_int<<", adaptive integration = "<<result_ext<<"\n";
    if(fabs(result_int-result_ext)>1e-10) return false;
    return true;
}

// provides a function of 1 variable to interpolate
class testfnc1d: public math::IFunction {
public:
    void evalDeriv(const double x, double* val, double* der, double* der2, double* der3) const
    {
        double sqx = sqrt(x);
        double y0 = sin(4*sqx);
        double y0p  = cos(4*sqx) * 2 / sqx;
        if(val)
            *val = y0;
        if(der)
            *der = y0p;
        if(der2)
            *der2 = -(4*y0 + 0.5*y0p) / x;
        if(der3)
            *der3 = (6*y0 + (0.75-4*x)*y0p) / pow_2(x);
    }
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        evalDeriv(x, val, der, der2, NULL);
    }
    virtual unsigned int numDerivs() const { return 3; }
};

// provides a function of 2 variables to interpolate
class testfnc2d: public math::IFunctionNdimDeriv {
public:
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        if(values)
            *values = sin(pow_2(vars[0])) * sin(vars[1]);
        if(derivs) {
            derivs[0] = 2*vars[0] * cos(pow_2(vars[0])) * sin(vars[1]);
            derivs[1] = sin(pow_2(vars[0])) * cos(vars[1]);
        }
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 1; }
};

// provides a function of 3 variables to interpolate
class testfnc3d: public math::IFunctionNdim {
public:
    virtual void eval(const double vars[], double values[]) const {
        values[0] = (sin(vars[0]+0.5*vars[1]+0.25*vars[2]) * cos(-0.2*vars[0]*vars[1]+vars[2]) + 1) /
            sqrt(1 + 0.1*(pow_2(vars[0]) + pow_2(vars[1]) + pow_2(vars[2])));
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

//----------- test penalized smoothing spline fit to noisy data -------------//
bool testPenalizedSplineFit()
{
    bool ok=true;
    const int NNODES  = 20;
    const int NPOINTS = 10000;
    const double XMIN = 0.2;
    const double XMAX = 12.;
    const double DISP = 0.5;  // y-dispersion
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    xnodes.pop_back();
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);

    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = math::random()*XMAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5)*4;
    }
    math::SplineApprox appr(xnodes, xvalues);
    double rms, edf;

    math::CubicSpline fit1(xnodes, appr.fitOptimal(yvalues1, &rms, &edf));
    std::cout << "case A: RMS="<<rms<<", EDF="<<edf<<"\n";
    ok &= rms<0.2 && edf>=2 && edf<NNODES+2;

    math::CubicSpline fit2(xnodes, appr.fitOversmooth(yvalues2, .5, &rms, &edf));
    std::cout << "case B: RMS="<<rms<<", EDF="<<edf<<"\n";
    ok &= rms<1.0 && edf>=2 && edf<NNODES+2;

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_fit.dat");
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << yvalues1[i] << "\t" << yvalues2[i] << "\t" <<
                fit1(xvalues[i]) << "\t" << fit2(xvalues[i]) << "\n";
    }
    return ok;
}

//-------- test penalized spline log-density estimation ---------//

// density distribution described by a sum of two Gaussians
class Density1: public math::IFunctionNoDeriv {
public:
    double mean1, disp1, mean2, disp2;  // parameters of the density
    double norm;  // overall normalization
    int d;        // multiply P(x) by ln(P(x)^d
    Density1(double _mean1, double _disp1, double _mean2, double _disp2, double _norm) :
        mean1(_mean1), disp1(_disp1), mean2(_mean2), disp2(_disp2), norm(_norm), d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = norm * (
            0.5 * exp( -0.5 * pow_2((x-mean1)/disp1) ) / sqrt(2*M_PI) / disp1 +
            0.5 * exp( -0.5 * pow_2((x-mean2)/disp2) ) / sqrt(2*M_PI) / disp2 );
        return d==0 ? P : P * math::powInt(log(P), d);
    }
    // sample a point from this density
    double sample() const {
        double x1, x2;
        math::getNormalRandomNumbers(x1, x2);
        if(x2>=0)  // attribute the point to either of the two Gaussians with 50% probability
            return mean1 + disp1 * x1;
        else
            return mean2 + disp2 * x1;
    }
};

// density described by the beta distribution
class Density2: public math::IFunctionNoDeriv {
public:
    double a, b;  // parameters of the density (must be >=0)
    double norm;  // normalization factor
    double cap;   // upper bound on the value of density (needed for rejection sampling)
    int d;        // multiply P(x) by ln(P(x)^d
    Density2(double _a, double _b, double _norm) :
        a(_a), b(_b),
        norm(_norm * math::gamma(a+b+2) / math::gamma(a+1) / math::gamma(b+1)),
        cap(norm * pow(a, a) * pow(b, b) / pow(a+b, a+b)),
        d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = x>=0 && x<=1 ? norm * pow(x, a) * pow(1-x, b) : 0;
        return d==0 ? P : P * math::powInt(log(P), d);
    }
    // sample a point from this density using the rejection algorithm
    double sample() const {
        while(1) {
            double x = math::random(), p = value(x), y = math::random() * cap;
            assert(p<=cap);
            if(y<p)
                return x;
        }
    }
};

// Gaussian kernel density estimate for an array of points
double kernelDensity(double x, const std::vector<double>& xvalues,
    const std::vector<double>& weights)
{
    const double width = 0.05;
    double sum=0;
    for(unsigned int i=0; i<xvalues.size(); i++)
        sum += weights[i] * exp( -0.5*pow_2((x-xvalues[i])/width) );
    return sum / sqrt(2*M_PI) / width;
}

bool testPenalizedSplineDensity()
{
    bool ok=true;
#if 1
    const double MEAN1= 2.0, DISP1=0.1, MEAN2=3.0, DISP2=1.0;// parameters of density function
    const double NORM = 1.;   // normalization (sum of weights of all samples)
    const double XMIN = 0;//fmin(MEAN1-3*DISP1, MEAN2-3*DISP2);  // limits of the interval for
    const double XMAX = 6;//fmax(MEAN1+3*DISP1, MEAN2+3*DISP2);  // constructing the estimators
    // whether to assume that the domain is infinite
    const math::FitOptions OPTIONS = math::FitOptions(math::FO_INFINITE_LEFT | math::FO_INFINITE_RIGHT);
    // density function from which the samples are drawn
    Density1 dens(MEAN1, DISP1, MEAN2, DISP2, NORM);
#else
    const double A    = 0., B = 0.5;  // parameters of density fnc
    const double NORM = 10.;
    const double XMIN = 0., XMAX = 1.;
    const math::FitOptions OPTIONS = math::FitOptions();
    Density2 dens(A, B, NORM);
#endif
    const int NPOINTS = 1000; // # of points to sample
    const int NNODES  = 49;    // nodes in the estimated density function
    const int NCHECK  = 321;   // points to measure the estimated density
    const double SMOOTHING=.5; // amount of smoothing applied to penalized spline estimate
    const int NTRIALS = 120;   // number of different realizations of samples
    const double XCUT = 3.;    // unequal-mass sampling: for x>XCUT, retain only a subset of
    const int MASSMULT= 1;     // samples with proportionally higher weight each
    std::vector<double> xvalues, weights;  // array of sample points

    // first perform Monte Carlo experiment to estimate the average log-likelihood of
    // a finite array of samples drawn from the density function, and its dispersion.
    math::Averager avgL;
    for(int t=0; t<NTRIALS; t++) {
        double logL = 0;
        xvalues.clear();
        weights.clear();
        for(int i=0; i<NPOINTS; i++) {
            double x = dens.sample();
            if(x<XCUT || math::random()<1./MASSMULT) {
                xvalues.push_back(x);
                weights.push_back(NORM/NPOINTS * (x<XCUT ? 1 : MASSMULT) /* (x>XMIN&&x<XMAX)*/);
                logL += log(dens(x)) * weights.back();
            }
        }
        avgL.add(logL);
    }
    std::cout << "Finite-sample log L = " << avgL.mean() << " +- " << sqrt(avgL.disp());
    // compare with theoretical expectation
    dens.d=1;   // integrate P(x) times ln(P(x)^d
    double E = math::integrateAdaptive(dens, XMIN-5, XMAX+5, 1e-6);
    dens.d=2;
    double Q = math::integrateAdaptive(dens, XMIN-5, XMAX+5, 1e-6)*NORM;
    double D = sqrt((Q-E*E)/xvalues.size());  // estimated rms scatter in log-likelihood
    dens.d=0;   // restore the original function
    std::cout << "  Expected log L = " << E << " +- " << D << "\n";

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_logdens_points.dat");
        for(size_t i=0; i<xvalues.size(); i++)
            strm << xvalues[i] << "\t" << weights[i] << "\n";
    }

    // grid defining the logdensity functions
    std::vector<double> grid = math::createUniformGrid(NNODES, XMIN, XMAX);

    // grid of points to check the results
    std::vector<double> testgrid = math::createUniformGrid(NCHECK, grid.front()-1, grid.back()+1);
    std::vector<double> truedens(NCHECK);
    for(int j=0; j<NCHECK; j++)
        truedens[j] = log(dens(testgrid[j]));
    // spline approximation for the true density - to test how well it is described
    // by a cubic spline defined by a small number of nodes
    math::CubicSpline spltrue(grid, math::SplineApprox(grid, testgrid).fit(truedens));
    // estimators of various degree constructed from a finite array of samples
    math::LinearInterpolator spl1(grid,
        math::splineLogDensity<1>(grid, xvalues, weights, OPTIONS));  // linear fit
    math::CubicSpline spl3o(grid,
        math::splineLogDensity<3>(grid, xvalues, weights, OPTIONS));  // optimally smoothed cubic
    math::CubicSpline spl3p(grid,
        math::splineLogDensity<3>(grid, xvalues, weights, OPTIONS, SMOOTHING));  // penalized cubic
    double logLtrue=0, logL1=0, logL3o=0, logL3p=0, logL3s=0;
    for(unsigned int i=0; i<xvalues.size(); i++) {
        // evaluate the likelihood of the sampled points against the true underlying density
        // and against all approximations
        logLtrue += weights[i] * log(dens(xvalues[i]));
        logL1    += weights[i] * spl1(xvalues[i]);
        logL3o   += weights[i] * spl3o(xvalues[i]);
        logL3p   += weights[i] * spl3p(xvalues[i]);
        logL3s   += weights[i] * spltrue(xvalues[i]);
    }
    ok &= fabs(logLtrue-logL1) < 3*D && fabs(logLtrue-logL3o) < 3*D &&
         fabs(logLtrue-logL3p) < 3*D && fabs(logL3o-logL3p-SMOOTHING*D) < 0.5*D;
    std::cout << "Log-likelihood: true density = " << logLtrue <<
        ", its cubic spline approximation = " << logL3s <<
        ", linear B-spline estimate = " << logL1 <<
        ", optimally smoothed cubic B-spline estimate = " << logL3o <<
        ", more heavily smoothed cubic = " << logL3p << '\n';
    if(OUTPUT) {
        std::ofstream strm("test_math_spline_logdens.dat");
        for(int j=0; j<NCHECK; j++) {
            double x = testgrid[j];
            double kernval = log(kernelDensity(x, xvalues, weights));
            strm << x << '\t' << spl1(x) << '\t' << spl3o(x) << '\t' << spl3p(x) << '\t' <<
                kernval << '\t' << truedens[j] << '\t' << spltrue(x) << '\n';
        }
    }
    return ok;
}

//-------- test cubic and quintic splines ---------//
bool test1dSpline()
{
    // accuracy of approximation of an oscillating fnc //
    bool ok=true;
    const int NNODES  = 20;
    const int NSUBINT = 16;
    const double XMIN = 0.2;
    const double XMAX = 12.0123456;
    testfnc1d fnc;
    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    xnodes[1]=(xnodes[1]+xnodes[2])/2;  // slightly squeeze grid spacing to allow
    xnodes[0]*=2;                       // a better interpolation of a strongly varying function
    for(int i=0; i<NNODES; i++) {
        fnc.evalDeriv(xnodes[i], &yvalues[i], &yderivs[i]);
    }
    math::CubicSpline   fcubna(xnodes, yvalues);  // cubic spline with natural boundary conditions
    math::CubicSpline   fcubcl(xnodes, yvalues,
        yderivs.front(), yderivs.back());         // cubic, clamped -- specify derivs at the boundaries
    math::HermiteSpline fhermi(xnodes, yvalues, yderivs);  // hermite cubic spline -- specify derivs at all nodes
    math::QuinticSpline fquint(xnodes, yvalues, yderivs);  // quintic spline -- specify derivs at all nodes
    // collect the derivatives from cubic spline and construct an equivalent Hermite spline
    std::vector<double> yderivscubcl(NNODES);
    for(int i=0; i<NNODES; i++) {
        fcubcl.evalDeriv(xnodes[i], NULL, &yderivscubcl[i]);
    }
    math::HermiteSpline fhercu(xnodes, yvalues, yderivscubcl);  // should be the same as fcubcl +- eps
    std::ofstream strm;
    if(OUTPUT)
        strm.open("test_math_spline1d.dat");
    double sumerrv3n = 0, sumerrv3 = 0, sumerrv5 = 0, sumerrvh = 0,
        sumerrd3 = 0, sumerrd5 = 0, sumerrdh = 0, sumerrs3 = 0, sumerrs5 = 0, sumerrsh = 0,
        maxdifhercu = 0;
    for(int i=0; i<=(NNODES-1)*NSUBINT; i++) {
        double xa = xnodes[i/NSUBINT];
        double xb = i<(NNODES-1)*NSUBINT ? xnodes[i/NSUBINT+1] : xa;
        double x  = xa*(1 - (i%NSUBINT)*1.0/NSUBINT) + xb*(i%NSUBINT)/NSUBINT;
        double y0, y0p, y0pp, y0ppp;
        fnc.evalDeriv(x, &y0, &y0p, &y0pp, &y0ppp);
        double y3n = fcubna(x);
        double y3, y3p, y3pp;
        fcubcl.evalDeriv(x, &y3, &y3p, &y3pp);
        double yh, yhp, yhpp;
        fhermi.evalDeriv(x, &yh, &yhp, &yhpp);
        double yc, ycp, ycpp;
        fhercu.evalDeriv(x, &yc, &ycp, &ycpp);
        double y5, y5p, y5pp, y5ppp=fquint.deriv3(x);
        fquint.evalDeriv(x, &y5, &y5p, &y5pp);
        sumerrv3n += pow_2(y0-y3n);
        sumerrv3  += pow_2(y0-y3);
        sumerrvh  += pow_2(y0-yh);
        sumerrv5  += pow_2(y0-y5);
        sumerrd3  += pow_2(y0p-y3p);
        sumerrdh  += pow_2(y0p-yhp);
        sumerrd5  += pow_2(y0p-y5p);
        sumerrs3  += pow_2(y0pp-y3pp);
        sumerrsh  += pow_2(y0pp-yhpp);
        sumerrs5  += pow_2(y0pp-y5pp);
        if(i%NSUBINT == 0) {  // we are exactly at one of the nodes of input grid
            int k = i/NSUBINT;
            ok &= y3n == yvalues[k];  // this should be exact to machine precision by construction
            ok &= y3  == yvalues[k];
            ok &= yh  == yvalues[k] && yhp == yderivs[k];
            ok &= yc  == yvalues[k];
            ok &= y5  == yvalues[k] && y5p == yderivs[k];
        }
        maxdifhercu = fmax(maxdifhercu, fabs(yc-y3)+fabs(ycp-y3p)+fabs(ycpp-y3pp));
        if(OUTPUT)
            strm << x << '\t' << y0 << ' ' << 
            y3n << ' ' << y3  << ' ' << yh  << ' ' << y5  << '\t' <<
            y0p << ' ' << y3p << ' ' << yhp << ' ' << y5p << '\t' <<
            y0pp<< ' ' << y3pp<< ' ' << yhpp<< ' ' << y5pp<< '\t' <<
            y0ppp << ' ' << y5ppp << "\n";
    }
    sumerrv3n = (sqrt(sumerrv3n / ((NNODES-1)*NSUBINT)));
    sumerrv3  = (sqrt(sumerrv3  / ((NNODES-1)*NSUBINT)));
    sumerrvh  = (sqrt(sumerrvh  / ((NNODES-1)*NSUBINT)));
    sumerrv5  = (sqrt(sumerrv5  / ((NNODES-1)*NSUBINT)));
    sumerrd3  = (sqrt(sumerrd3  / ((NNODES-1)*NSUBINT)));
    sumerrdh  = (sqrt(sumerrdh  / ((NNODES-1)*NSUBINT)));
    sumerrd5  = (sqrt(sumerrd5  / ((NNODES-1)*NSUBINT)));
    sumerrs3  = (sqrt(sumerrs3  / ((NNODES-1)*NSUBINT)));
    sumerrsh  = (sqrt(sumerrsh  / ((NNODES-1)*NSUBINT)));
    sumerrs5  = (sqrt(sumerrs5  / ((NNODES-1)*NSUBINT)));
    std::cout << "RMS error in cubic spline: " << sumerrv3n <<
        ", in clamped cubic spline: " << sumerrv3 <<
        ", in hermite cubic spline: " << sumerrvh <<
        ", in quintic spline: " << sumerrv5 << 
        ", max|cubic-hermite|=" << maxdifhercu << "\n";
    ok &= maxdifhercu < 1e-13 && sumerrv3n<5e-3 &&
        sumerrv3<2.7e-4 && sumerrvh<2.3e-4 && sumerrv5<3.4e-5 &&
        sumerrd3<1.8e-3 && sumerrdh<1.3e-3 && sumerrd5<9e-4   &&
        sumerrs3<0.04   && sumerrsh<0.04   && sumerrs5<0.05;
    if(OUTPUT)
        strm.close();

    // test the integration function //
    ok &= test_integral(fcubcl, (xnodes[0]+xnodes[1])/2, xnodes.back());
    ok &= test_integral(fcubna, -1.234567, xnodes.back()+1.);

    return ok;
}

//----------- test 2d cubic and quintic spline ------------//
bool test2dSpline()
{
    bool ok=true;
    const int NNODESX=7;
    const int NNODESY=8;
    const int NN=99;    // number of intermediate points for checking the values
    const double XMAX = 1.81, YMAX = 2*M_PI;
    std::vector<double> xval = math::createNonuniformGrid(NNODESX, XMAX*0.20, XMAX, true);
    std::vector<double> yval = math::createNonuniformGrid(NNODESY, YMAX*0.15, YMAX, true);
    math::Matrix<double> fval(NNODESX, NNODESY), fderx(NNODESX, NNODESY), fdery(NNODESX, NNODESY);
    testfnc2d fnc;
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double der[2];
            fnc.evalDeriv(xy, &fval(i, j), der);
            fderx(i, j) = der[0];
            fdery(i, j) = der[1];
        }
    // create a 2d cubic spline with natural boundary conditions at three edges
    // and a prescribed derivative at the fourth edge
    math::CubicSpline2d spl2dc(xval, yval, fval, 0, NAN, NAN, NAN);
    // create a 2d quintic spline with prescribed derivatives at all nodes
    math::QuinticSpline2d spl2dq(xval, yval, fval, fderx, fdery);

#ifdef COMPARE_WD_PSPLINE
    double *WD_X[2], **WD_Y[3], **WD_Z[4];
    WD_X[0] = new double[NNODESX];
    WD_X[1] = new double[NNODESY];
    int WD_K[2] = {NNODESX, NNODESY};
    WD::Alloc2D(WD_Y[0],WD_K);
    WD::Alloc2D(WD_Y[1],WD_K);
    WD::Alloc2D(WD_Y[2],WD_K);
    WD::Alloc2D(WD_Z[0],WD_K);
    WD::Alloc2D(WD_Z[1],WD_K);
    WD::Alloc2D(WD_Z[2],WD_K);
    WD::Alloc2D(WD_Z[3],WD_K);
    for(int i=0; i<NNODESX; i++)
        WD_X[0][i] = xval[i];
    for(int j=0; j<NNODESY; j++)
        WD_X[1][j] = yval[j];
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            WD_Y[0][i][j] = fval(i, j);
            WD_Y[1][i][j] = fderx(i, j);
            WD_Y[2][i][j] = fdery(i, j);
        }
    WD::Pspline2D(WD_X, WD_Y, WD_K, WD_Z);
#endif

    // check the values (for both cubic and quintic splines) and derivatives (for quintic spline only)
    // at all nodes of 2d grid -- should exactly equal the input values (with machine precision)
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double f, fder[2], c, q, qder[2];
            fnc.evalDeriv(xy, &f, fder);
            c = spl2dc.value(xy[0], xy[1]);
            spl2dq.evalDeriv(xy[0], xy[1], &q, &qder[0], &qder[1]);
            if(c != f || q != f || fder[0] != qder[0] || fder[1] != qder[1])
                ok = false;
        }

    std::ofstream strm;
    if(OUTPUT)  // output for Gnuplot splot routine
        strm.open("test_math_spline2d.dat");
    double maxerrc = 0, maxerrq = 0, maxerrcder = 0, maxerrqder = 0;
    for(int i=0; i<=NN; i++) {
        double x = XMAX*(i*1./NN);
        for(int j=0; j<=NN; j++) {
            double y = YMAX*(j*1./NN);

            double xy[2] = {x, y}, f, fder[2];
            fnc.evalDeriv(xy, &f, fder);
            double c, cx, cy, cxx, cxy, cyy, q, qx, qy, qxx, qxy, qyy;
            spl2dc.evalDeriv(x, y, &c, &cx, &cy, &cxx, &cxy, &cyy);
            spl2dq.evalDeriv(x, y, &q, &qx, &qy, &qxx, &qxy, &qyy);
            maxerrc = fmax(maxerrc, fabs(c-f));
            maxerrq = fmax(maxerrq, fabs(q-f));
            maxerrcder = fmax(maxerrcder, fmax(fabs(cx-fder[0]), fabs(cy-fder[1])));
            maxerrqder = fmax(maxerrqder, fmax(fabs(qx-fder[0]), fabs(qy-fder[1])));
#ifdef COMPARE_WD_PSPLINE
            const double EPS=1e-13;
            double wder[2];
            double wder2x[2], wder2y[2]; 
            double* wder2[] = {wder2x, wder2y};
            double wval = WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, xy, wder, wder2);
            ok &= fabs(q-wval   )  <EPS &&
                fabs(qx -wder[0])  <EPS &&
                fabs(qy -wder[1])  <EPS &&
                fabs(qxx-wder2x[0])<EPS &&
                fabs(qxy-wder2x[1])<EPS &&
                fabs(qyy-wder2y[1])<EPS;
#endif
            if(OUTPUT)
                strm << x << ' ' << y << '\t' << 
                    f << ' ' << fder[0] << ' ' << fder[1] << '\t' <<
                    c << ' ' << cx << ' ' << cy << ' ' << cxx << ' ' << cxy << ' ' << cyy << '\t' <<
                    q << ' ' << qx << ' ' << qy << ' ' << qxx << ' ' << qxy << ' ' << qyy << '\t' <<
#ifdef COMPARE_WD_PSPLINE
                    wval <<' '<< wder[0] <<' '<< wder[1] <<' '<< wder2x[0] <<' '<< wder2x[1] <<' '<< wder2y[1] <<
#endif
                    '\n';
        }
        if(OUTPUT)
            strm << "\n";
    }
    if(OUTPUT)
        strm.close();

#ifdef STRESS_TEST
    //----------- test the performance of 2d spline calculation -------------//
    double z, dx, dy, dxx, dxy, dyy;
    int NUM_ITER = 1000;
    double RATE = 1.0 * NUM_ITER * NN * NN * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dc.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            }
        }
    }
    std::cout << "Cubic spline with 2nd deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dc.evalDeriv(x, y, &z, &dx, &dy);
            }
        }
    }
    std::cout << ", 1st deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dc.evalDeriv(x, y, &z);
            }
        }
    }
    std::cout << ", no deriv: " << utils::pp(RATE/(std::clock()-clk),7) << " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dq.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            }
        }
    }
    std::cout << "Quintic spline with 2nd deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dq.evalDeriv(x, y, &z, &dx, &dy);
            }
        }
    }
    std::cout << ", 1st deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dq.evalDeriv(x, y, &z);
            }
        }
    }
    std::cout << ", no deriv: " << utils::pp(RATE/(std::clock()-clk),7) << " eval/s\n";
#ifdef COMPARE_WD_PSPLINE
    double wder[2];
    double wder2x[2], wder2y[2]; 
    double* wder2[] = {wder2x, wder2y};
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx, wder, wder2);
            }
        }
    }
    std::cout << "WD's Pspline with 2nd deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx, wder);
            }
        }
    }
    std::cout << ", 1st deriv: " << utils::pp(RATE/(std::clock()-clk),7);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                double wx[2] = {x, y};
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx);
            }
        }
    }
    std::cout << ", no deriv: " << utils::pp(RATE/(std::clock()-clk),7) << " eval/s\n";
#endif
#endif
    std::cout << "Max error in cubic 2d spline value: " << maxerrc << ", deriv: " << maxerrcder <<
        ", quintic 2d spline value: " << maxerrq << ", deriv: " << maxerrqder << "\n";
    ok &= maxerrc < 0.006 && maxerrcder < 0.09 && maxerrq < 0.002 && maxerrqder < 0.03;

    return ok;
}

//----------- test 3d interpolation ------------//
bool test3dSpline()
{
    bool ok=true;
    testfnc3d fnc3d;
    const int NNODESX=8, NNODESY=4, NNODESZ=6;
    std::vector<double>
    xval=math::createUniformGrid(NNODESX, 0, 6),
    yval=math::createUniformGrid(NNODESY, 0, 3),
    zval=math::createUniformGrid(NNODESZ, 0, 5);
    math::Matrix<double> samples;
    std::vector<double> lval3d(math::createInterpolator3dArray<1>(fnc3d, xval, yval, zval));
    std::vector<double> cval3d(math::createInterpolator3dArray<3>(fnc3d, xval, yval, zval));
    /*
    double integr_quad, interr_quad, integr_samp, interr_samp;
    const double xlower[3] = {xval.front(), yval.front(), zval.front()};
    const double xupper[3] = {xval.back(),  yval.back(),  zval.back() };
    math::integrateNdim(fnc3d, xlower, xupper, 1e-3, 1e5, &integr_quad, &interr_quad);
    math::sampleNdim(fnc3d, xlower, xupper, 1e5, samples, NULL, &integr_samp, &interr_samp);
    std::cout << "3d function: integral over domain by quadrature=" << integr_quad << " +- " <<
    interr_quad << ", by sampling=" << integr_samp << " +- " << interr_samp << "\n";

    std::vector<double> lsam3d(math::createInterpolator3dArrayFromSamples<1>(
        samples, std::vector<double>(samples.rows(), integr_samp/samples.rows()), xval, yval, zval));
    std::vector<double> csam3d(math::createInterpolator3dArrayFromSamples<3>(
        samples, std::vector<double>(samples.rows(), integr_samp/samples.rows()), xval, yval, zval));
    */
    math::LinearInterpolator3d lin3d(xval, yval, zval);
    math::CubicInterpolator3d  cub3d(xval, yval, zval);

    double point[3];
    // test the values of interpolated function at grid nodes
    for(int i=0; i<NNODESX; i++) {
        point[0] = xval[i];
        for(int j=0; j<NNODESY; j++) {
            point[1] = yval[j];
            for(int k=0; k<NNODESZ; k++) {
                point[2] = zval[k];
                double v;
                fnc3d.eval(point, &v);
                double l = lin3d.interpolate(point, lval3d);
                double c = cub3d.interpolate(point, cval3d);
                ok &= math::fcmp(v, c, 1e-13)==0;
                ok &= math::fcmp(v, l, 1e-15)==0;
            }
        }
    }
    // test accuracy of approximation
    double sumsqerr_l=0, sumsqerr_c=0;
    const int NNN=24;    // number of intermediate points for checking the values
    std::ofstream strm;
    if(OUTPUT)
        strm.open("test_math_spline3d.dat");
    for(int i=0; i<=NNN; i++) {
        point[0] = i*xval.back()/NNN;
        for(int j=0; j<=NNN; j++) {
            point[1] = j*yval.back()/NNN;
            for(int k=0; k<=NNN; k++) {
                point[2] = k*zval.back()/NNN;
                double v;
                fnc3d.eval(point, &v);
                double l = lin3d.interpolate(point, lval3d);
                double c = cub3d.interpolate(point, cval3d);
                sumsqerr_l += pow_2(l-v);
                sumsqerr_c += pow_2(c-v);
                if(OUTPUT)
                    strm << point[0] << ' ' << point[1] << ' ' << point[2] << '\t' <<
                    v << ' ' << l << ' ' << c << "\n";
            }
            if(OUTPUT)
                strm << "\n";
        }
    }
    if(OUTPUT)
        strm.close();
    sumsqerr_l = sqrt(sumsqerr_l / pow_3(NNN+1));
    sumsqerr_c = sqrt(sumsqerr_c / pow_3(NNN+1));
    std::cout << "RMS error in linear 3d interpolator: " << sumsqerr_l << 
        ", cubic 3d interpolator:" << sumsqerr_c << "\n";
    ok &= sumsqerr_l<0.1 && sumsqerr_c<0.05;

    return ok;
}

bool printFail(const char* msg)
{
    std::cout << "\033[1;31m " << msg << " failed\033[0m\n";
    return (msg==0);  // false
}

int main()
{
    std::cout << std::setprecision(12);
    bool ok=true;
    ok &= testPenalizedSplineFit() || printFail("Penalized spline fit");
    ok &= testPenalizedSplineDensity() || printFail("Penalized spline density estimator");
    ok &= test1dSpline() || printFail("1d spline");
    ok &= test2dSpline() || printFail("2d spline");
    ok &= test3dSpline() || printFail("3d spline");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
