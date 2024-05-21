/** \name   test_math_spline.cpp
    \author Eugene Vasiliev
    \date   2015-2024

    Test the accuracy of various spline-related operations (interpolation in 1d, 2d, 3d; fitting, etc..)
*/
#include "math_spline.h"
#include "math_core.h"
#include "math_random.h"
#include "math_sphharm.h"
#include "math_sample.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>

const bool OUTPUT = utils::verbosityLevel >= utils::VL_VERBOSE;
const unsigned int MIN_CLOCKS = CLOCKS_PER_SEC/2;  // 0.5 seconds per performance test

bool testCond(bool condition, const char* errorMessage)
{
    if(!condition)
        std::cout << errorMessage << '\n';
    return condition;
}

std::string testLess(double val, double limit, bool &condition)
{
    if(!(val<limit)) {
        condition = false;
        return "\033[1;31m"+utils::toString(val)+"\033[0m";
    }
    return utils::toString(val);
}

// provides the integral of sin(x)*x^n
class testfncsin: public math::IFunctionIntegral {
    virtual double integrate(double x1, double x2, int n=0) const {
        return antideriv(x2,n)-antideriv(x1,n);
    }
    double antideriv(double x, int n) const {
        switch(n) {
            case 0: return -cos(x);
            case 1: return -cos(x)*x+sin(x);
            case 2: return  cos(x)*(2-x*x) + 2*x*sin(x);
            case 3: return  cos(x)*x*(6-x*x) + 3*sin(x)*(x*x-2);
            case 4: return -cos(x)*(24-x*x*(12-x*x)) + 4*x*sin(x)*(x*x-6);
            case 5: return -cos(x)*x*(120-x*x*(20-x*x)) + 5*sin(x)*(24-x*x*(12-x*x));
            default: return NAN;
        }
    }
};

// provides the integrand for numerical integration of sin(x)*f(x)
class testfncintsin: public math::IFunctionNoDeriv {
public:
    testfncintsin(const math::IFunction& _f): f(_f) {};
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

//#define TESTFNC1D_SMOOTH

// provides a function of 1 variable to interpolate
class testfnc1d: public math::IFunction {
public:
    void evalDeriv(const double x, double* val, double* der, double* der2, double* der3) const
    {
#ifdef TESTFNC1D_SMOOTH
        // a very special case of a function with 3 zero derivatives at x=0 and x=pi
        double y = x / M_PI;
        double q = M_PI*y*y*(3-2*y);
        double s = sin(q);
        double c = cos(q);
        double t = 6*y*(1-y);
        if(val)
            *val = s*s;
        if(der)
            *der = 2*t*s*c;
        if(der2)
            *der2 = 2*t*t*(c*c-s*s) + 12/M_PI*(1-2*y)*s*c;
        if(der3)
            *der3 = -8*(t*t*t+3/M_PI/M_PI)*c*s + 216/M_PI*y*(1-y)*(1-2*y)*(c*c-s*s);
#else
        // a more general case of an oscillating function
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
#endif
    }
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        evalDeriv(x, val, der, der2, NULL);
    }
    virtual unsigned int numDerivs() const { return 3; }
};

// provides a function to interpolate via log-spline
class testfnclog: public math::IFunction {
public:
    virtual void evalDeriv(const double x, double* val=NULL, double* der=NULL, double* der2=NULL) const {
        if(val)
            *val  = x<1? 0 : (x-1) * pow_2(x-10) / pow(x, 6);
        if(der)
            *der  = x<1? 0 : (30-3*x) * (x*x-18*x+20) / pow(x, 7);
        if(der2)
            *der2 = x<1? 0 : (12*x*x*x - 420*x*x + 3600*x - 4200) / pow(x, 8);
    }
    virtual unsigned int numDerivs() const { return 2; }
};

// provides a function of 2 variables to interpolate
class testfnc2d: public math::IFunctionNdimDeriv {
public:
    virtual void evalDeriv(const double vars[], double values[], double *derivs=NULL) const
    {
        double val = exp(-pow_2(vars[0])-pow_2(vars[1]));
        if(values)
            *values = val;
        if(derivs) {
            derivs[0] = -2*vars[0]*val;
            derivs[1] = -2*vars[1]*val;
            // output second and higher derivs as well
            derivs[2] = (4*pow_2(vars[0])-2)*val; // f_xx
            derivs[3] =  4* vars[0]*vars[1] *val; // f_xy
            derivs[4] = (4*pow_2(vars[1])-2)*val; // f_yy
            derivs[5] = 4*vars[1]*(1-2*pow_2(vars[0]))*val; // f_xxy
            derivs[6] = 4*vars[0]*(1-2*pow_2(vars[1]))*val; // f_xyy
            derivs[7] = 4*(1-pow_2(vars[0]))*(1-2*pow_2(vars[1]))*val; // f_xxyy
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

// a function with sharp gradients for testing monotone interpolation
template<int N>
class testfncNd: public math::IFunctionNdim, public math::IFunctionNoDeriv {
    static const int POW = 6;
public:
    virtual void eval(const double vars[], double values[]) const {
        double sum = 0;
        for(int i=0; i<N; i++)
            sum += pow(vars[i], POW);
        values[0] = exp(-sum);
    }
    virtual double value(const double x) const {
        return exp(-pow(x, POW));
    }
    virtual unsigned int numVars()   const { return N; }
    virtual unsigned int numValues() const { return 1; }
};

// original function to interpolate and convolve
//#define testfncorig sin
double testfncorig(double x) { return tanh((2-fabs(x-4))/0.5)*0.5+0.5; }

/** A normalized asymmetric exponential convolution kernel centered at origin with a scale sigma:
    \f$   f(x) = \exp( x/sigma ) / \sigma  \f$  if x<=0,  otherwise 0
*/
class ExpKernel: public math::IFunction, public math::IFunctionIntegral {
    const double sigma;
public:
    ExpKernel(double _sigma) : sigma(_sigma) {}

    /* Compute the value and up to two derivatives of the function */
    virtual void evalDeriv(double x, /*output*/ double *val, double *der, double *der2) const
    {
        double v = x <= 0 ? exp(x/sigma) / sigma : 0;
        if(val)
            *val = v;
        if(der)
            *der = v/sigma;
        if(der2)
            *der2 = v/pow_2(sigma);
    }
    virtual unsigned int numDerivs() const { return 2; }

    /** Compute the value of integral of the function times x^n on the interval [x1..x2] */
    virtual double integrate(double x1, double x2, int n=0) const
    {
        double v1 = x1 <=0 ? math::gammainc(n+1, -x1/sigma) : math::gamma(n+1);
        double v2 = x2 <=0 ? math::gammainc(n+1, -x2/sigma) : math::gamma(n+1);
        return (v2-v1) * math::pow(-sigma, n);
    }
};

// convolution of two functions
class Convolved: public math::IFunctionNoDeriv {
    const double x;
    const math::IFunction &F, &G;
public:
    Convolved(double _x, const math::IFunction& _F, const math::IFunction& _G) :
        x(_x), F(_F), G(_G) {}
    virtual double value(const double y) const {
        return F(y) * nan2num(G(x-y));
    }
};

//----------- test penalized smoothing spline fit to noisy data -------------//
bool testPenalizedSplineFit()
{
    std::cout << "\033[1;33m1d penalized fitting\033[0m\n";
    bool ok=true;
    const int NNODES  = 20;
    const int NPOINTS = 10000;
    const double XMIN = 0.2;
    const double XMAX = 12.;
    const double DISP = 0.5;  // y-dispersion
    // construct the grid of knots for the spline approximation
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    // eliminate the last knot -- the spline will be linearly extrapolated
    // beyond the new last knot
    xnodes.pop_back();

    // arrays of x-points and corresponding function values
    // (two sets of values with different noise levels)
    std::vector<double> xvalues(NPOINTS), yvalues1(NPOINTS), yvalues2(NPOINTS);
    for(int i=0; i<NPOINTS; i++) {
        xvalues [i] = math::random()*XMAX;
        yvalues1[i] = sin(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5);
        yvalues2[i] = cos(4*sqrt(xvalues[i])) + DISP*(math::random()-0.5)*4;
    }

    // construct the spline approximation object, common for both sets of y-values
    math::SplineApprox appr(xnodes, xvalues);
    double rms1, edf1, rms2, edf2, rmsw, edfw;

    math::CubicSpline fit1(xnodes, appr.fitOptimal(yvalues1, &rms1, &edf1));
    std::cout << "case A: RMS=" << rms1 << ", EDF=" << edf1 << "\n";
    ok &= rms1<0.2 && edf1>=2 && edf1<NNODES+2;

    math::CubicSpline fit2(xnodes, appr.fitOversmooth(yvalues2, .5, &rms2, &edf2));
    std::cout << "case B: RMS=" << rms2 << ", EDF=" << edf2 << "\n";
    ok &= rms2<1.0 && edf2>=2 && edf2<NNODES+2;

    // test the weighted regression:
    // split some of the input points into four identical ones,
    // and assign them a four times smaller weight
    std::vector<double> weights(NPOINTS, 1.0);
    for(int i=0; i<NPOINTS; i++) {
        weights[i] *= 1e5;  // test scale-invariance w.r.t. rescaling the weights
        // split some points that lie above the original trend line (i.e. make a biased distribution)
        if(yvalues1[i] >= sin(4*sqrt(xvalues[i])) && math::random() <= 0.5) {
            weights[i] *= 0.25;
            for(int k=1; k<4; k++) {  // add duplicate points
                xvalues. push_back(xvalues [i]);
                yvalues1.push_back(yvalues1[i]);
                yvalues2.push_back(yvalues2[i]);
                weights. push_back(weights [i]);
            }
        }
    }
    // create another spline approximation object for this expanded set of points
    math::SplineApprox apprw(xnodes, xvalues, weights);
    // fitting with the same EDF as the original dataset should recover the same result
    apprw.fit(yvalues1, edf1, &rmsw);
    std::cout << "case A': RMS=" << rmsw <<", EDF=" << edf1 << "\n";
    ok &= math::fcmp(rmsw, rms1, 1e-8) == 0;
    apprw.fit(yvalues2, edf2, &rmsw);
    std::cout << "case B': RMS=" << rmsw <<", EDF=" << edf2 << "\n";
    ok &= math::fcmp(rmsw, rms2, 1e-8) == 0;
    // fitting with "optimal" or "oversmoothed" option will result in a somewhat less smoothed curve
    // than the original one, because the number of data points has increased
    math::CubicSpline fitw1(xnodes, apprw.fitOptimal(yvalues1, &rmsw, &edfw));
    std::cout << "case A\": RMS=" << rmsw <<", EDF=" << edfw << "\n";
    ok &= math::fcmp(rmsw, rms1, 1e-2) == 0 && math::fcmp(edfw, edf1, 1e-2) == 0;
    math::CubicSpline fitw2(xnodes, apprw.fitOversmooth(yvalues2, .5, &rmsw, &edfw));
    std::cout << "case B\": RMS=" << rmsw <<", EDF=" << edfw << "\n";
    ok &= math::fcmp(rmsw, rms2, 1e-2) == 0 && math::fcmp(edfw, edf2, 1e-2) == 0;

    if(OUTPUT) {
        std::ofstream strm("test_math_spline_fit.dat");
        for(int i=0; i<NPOINTS; i++) {
            strm << xvalues[i] << "\t" << yvalues1[i] << "\t" << yvalues2[i] << "\t" <<
                fit1(xvalues[i]) << "\t" << fitw1(xvalues[i]) << "\t" <<
                fit2(xvalues[i]) << "\t" << fitw2(xvalues[i]) << "\n";
        }
    }
    return ok;
}

//-------- test penalized spline log-density estimation ---------//

// density distribution described by a sum of two Gaussians
class Density1: public math::IFunctionNoDeriv {
public:
    double mean1, disp1, mean2, disp2;  // parameters of the density
    double norm;  // overall normalization
    int d;        // multiply P(x) by ln(P(x))^d
    Density1(double _mean1, double _disp1, double _mean2, double _disp2, double _norm) :
        mean1(_mean1), disp1(_disp1), mean2(_mean2), disp2(_disp2), norm(_norm), d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = norm * (
            0.5 * exp( -0.5 * pow_2((x-mean1)/disp1) ) / sqrt(2*M_PI) / disp1 +
            0.5 * exp( -0.5 * pow_2((x-mean2)/disp2) ) / sqrt(2*M_PI) / disp2 );
        return d==0 ? P : P * math::pow(log(P), d);
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
    int d;        // multiply P(x) by ln(P(x))^d
    Density2(double _a, double _b, double _norm) :
        a(_a), b(_b),
        norm(_norm * math::gamma(a+b+2) / math::gamma(a+1) / math::gamma(b+1)),
        cap(norm * pow(a, a) * pow(b, b) / pow(a+b, a+b)),
        d(0) {}
    // evaluate the density at the given point
    virtual double value(double x) const {
        double P = x>=0 && x<=1 ? norm * pow(x, a) * pow(1-x, b) : 0;
        return d==0 ? P : P * math::pow(log(P), d);
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
    std::cout << "\033[1;33m1d density estimate\033[0m\n";
    bool ok=true;
#if 1
    const double MEAN1= 2.0, DISP1=0.1, MEAN2=3.0, DISP2=1.0; // parameters of density function
    const double NORM = 1.;   // normalization (sum of weights of all samples)
    const double XMIN = 0;    //fmin(MEAN1-3*DISP1, MEAN2-3*DISP2);  // limits of the interval for
    const double XMAX = 6;    //fmax(MEAN1+3*DISP1, MEAN2+3*DISP2);  // constructing the estimators
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
    const int NPOINTS = 2000;  // # of points to sample
    const int NNODES  = 49;    // nodes in the estimated density function
    const int NCHECK  = 321;   // points to measure the estimated density
    const double SMOOTHING=.5; // amount of smoothing applied to penalized spline estimate
    const int NTRIALS = 300;   // number of different realizations of samples
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
    math::CubicSpline spl3t(grid,
        math::splineLogDensity<3>(grid, xvalues, weights,
        math::FitOptions( OPTIONS | math::FO_PENALTY_3RD_DERIV)));  // with penalty for 3rd deriv
    double logLtrue=0, logL1=0, logL3o=0, logL3p=0, logL3t=0, logL3s=0;
    for(unsigned int i=0; i<xvalues.size(); i++) {
        // evaluate the likelihood of the sampled points against the true underlying density
        // and against all approximations
        logLtrue += weights[i] * log(dens(xvalues[i]));
        logL1    += weights[i] * spl1(xvalues[i]);
        logL3o   += weights[i] * spl3o(xvalues[i]);
        logL3p   += weights[i] * spl3p(xvalues[i]);
        logL3t   += weights[i] * spl3t(xvalues[i]);
        logL3s   += weights[i] * spltrue(xvalues[i]);
    }
    ok &= fabs(logLtrue-logL1) < 3*D && fabs(logLtrue-logL3o) < 3*D &&
         fabs(logLtrue-logL3p) < 3*D && fabs(logLtrue-logL3t) < 3*D &&
         fabs(logL3o-logL3p-SMOOTHING*D) < 0.5*D;
    std::cout << "Log-likelihood: true density = " << logLtrue <<
        ", its cubic spline approximation = " << logL3s <<
        ", linear B-spline estimate = " << logL1 <<
        ", optimally smoothed cubic B-spline estimate = " << logL3o <<
        ", more heavily smoothed cubic = " << logL3p <<
        ", 3rd-derivative smoothed cubic = " << logL3t << '\n';
    if(OUTPUT) {
        std::ofstream strm("test_math_spline_logdens.dat");
        for(int j=0; j<NCHECK; j++) {
            double x = testgrid[j];
            double kernval = log(kernelDensity(x, xvalues, weights));
            strm << x << '\t' << spl1(x) << '\t' << spl3o(x) << '\t' << spl3p(x) << '\t' <<
                spl3t(x) << '\t' << kernval << '\t' << truedens[j] << '\t' << spltrue(x) << '\n';
        }
    }
    return ok;
}

//-------- test cubic and quintic splines ---------//

// test the integration of a spline function
bool testIntegral(const math::BaseInterpolator1d& f, double x1, double x2)
{
    // in case that the input interval is larger than the range of xvalues for the interpolator,
    // set the limits of the "external" integration to the interpolator boundaries
    // (otherwise it would include the region where the interpolator is linearly extrapolated,
    // but its integral is still evaluated as if it were zero).
    double ex1 = x1<x2 ? fmax(x1, f.xmin()) : fmin(x1, f.xmax());
    double ex2 = x1<x2 ? fmin(x2, f.xmax()) : fmax(x2, f.xmin());
    double result_int = f.integrate(x1, x2);
    double result_ext = math::integrateAdaptive(f, ex1, ex2, 1e-13);
    double error1 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Ordinary intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error1,8) + '\n';
    result_int = f.integrate(x1, x2, testfncsin());
    result_ext = math::integrateAdaptive(testfncintsin(f), ex1, ex2, 1e-13);
    double error2 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Weighted intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error2,8) + '\n';
    result_int = f.integrate(x1, x2, f);
    result_ext = math::integrateAdaptive(squaredfnc(f), ex1, ex2, 1e-13);
    double error3 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Integral of f(x)^2 on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error3,8) + '\n';
    return error1 < 1e-13 && error2 < 1e-13 && error3 < 1e-13;
}

bool testExtremaAndRoots()
{
    bool ok = true;
    std::vector<double> x = math::createUniformGrid(5, -1, 3), y(5), d(5), v;

    // pure quartic function in the inner two segments
    y[0] = y[4] = 47./4;
    y[1] = y[3] = 3./4;
    d[4] = 183./8;
    d[0] = -d[4];
    d[3] = 3.0;
    d[1] = -d[3];
    math::QuinticSpline sq(x, y, d);
    v = sq.extrema();
    ok &= testCond(v.size() >= 3 && v[0] == x[0] && v.back() == x[4], "extrema Quintic 1");
    for(size_t i=1; i+1 < v.size(); i++)  // may have additional spurious extrema near the true minimum
        ok &= testCond(fabs(v[i] - x[2]) < 1e-7, "extrema Quintic 1");
    v = sq.roots();
    ok &= testCond(v.size() >= 1, "roots Quintic 1a");
    for(size_t i=0; i < v.size(); i++)  // may have additional spurious roots around the true one
        ok &= testCond(fabs(v[i] - x[2]) < 1e-7, "roots Quintic 1a");
    v = sq.roots(3./64);
    ok &= testCond(v.size() == 2 && fabs(v[0]-0.5) <= 2*DBL_EPSILON && fabs(v[1]-1.5) <= 2*DBL_EPSILON,
        "roots Quintic 1b");

    // pure quintic function in the inner two segments
    y[4] = 43./2;
    y[0] = -y[4];
    y[1] = -y[3];
    d[0] = d[4] = 195./4;
    d[1] = d[3] = 15./4;
    sq = math::QuinticSpline(x, y, d);
    v = sq.extrema();
    ok &= testCond(v.size() >= 2 && v[0] == x[0] && v.back() == x[4], "extrema Quintic 2");
    for(size_t i=1; i+1 < v.size(); i++)  // may have spurious extrema around the inflection point
        ok &= testCond(fabs(v[i] - x[2]) < 1e-5, "extrema Quintic 2");
    v = sq.roots();
    ok &= testCond(v.size() >= 1, "roots Quintic 2");
    for(size_t i=0; i < v.size(); i++)  // may have additional spurious roots around the true one
        ok &= testCond(fabs(v[i] - x[2]) < 1e-5, "roots Quintic 2");

    // linear functions: extrema can be located only at grid nodes or endpoints of the interval
    math::LinearInterpolator sl(x, y);
    v = sl.extrema(2.5, 0.5);
    ok &= testCond(v.size() == 2 && v[0] == 0.5 && v[1] == 2.5, "extrema Linear");
    v = sl.roots(0.375);
    ok &= testCond(v.size() == 1 && v[0] == 1.5, "roots Linear");
    math::BsplineWrapper<1> bs1(x, y);
    v = bs1.extrema(0.5, 2.5);
    ok &= testCond(v.size() == 2 && v[0] == 0.5 && v[1] == 2.5, "roots B-spline-1");
    v = bs1.roots(0.375);
    ok &= testCond(v.size() == 1 && v[0] == 1.5, "roots B-spline-1");

    // oscillating function resembling two periods of a sine, with local maxima
    // at -0.516... and 1.5, and minima at 0.5 and 2.516...
    y[0] = y[1] = y[2] = y[3] = y[4] = 0;
    d[0] = d[4] = 1.25;
    d[1] = d[3] = -1.0;
    d[2] = 1.0;
    sq = math::QuinticSpline(x, y, d);
    v = sq.extrema(-0.5, 4);
    ok &= testCond(v.size() == 5 && v[0] == -0.5 &&
        fabs(v[1]-0.5) <= 2*DBL_EPSILON &&
        fabs(v[2]-1.5) <= 2*DBL_EPSILON &&
        v[3] > 2.5 && v[4] == 3, "extrema Quintic 3");
    v = sq.roots(0.25, 0, 2);
    ok &= testCond(v.size() == 2 &&
        fabs(v[0] - (2-1/M_SQRT2)) < 2*DBL_EPSILON &&
        fabs(v[1] - (1+1/M_SQRT2)) < 2*DBL_EPSILON, "roots Quintic 3");

    // cubic spline function with a flat bottom
    y[0] = 1.0; y[1] = 0.375; y[2] = y[3] = 0.25; y[4] = 0.5;
    math::CubicSpline sc(x, y, false, -0.75, 0.75);
    v = sc.extrema();
    ok &= testCond(v.size() == 3 && v[0] == x[0] && v[1] == x[2] && v[2] == x[4], "extrema Cubic");
    v = sc.roots(0.25);
    ok &= testCond(v.size() >= 1, "roots Cubic");
    for(size_t i=0; i<v.size(); i++)
        ok &= testCond(fabs(v[i] - x[2]) < 1e-5 || fabs(v[i] - x[3]) < 1e-5, "roots Cubic");

    // same, but using B-splines
    y.resize(7);
    y[0] = 1.0; y[1] = 1.0; y[2] = 0.25; y[3] = 0.25; y[4] = 0.25; y[5] = 0.25; y[6] = 0.5;
    math::BsplineWrapper<3> bs3(x, y);
    v = bs3.extrema();
    ok &= testCond(v.size() == 3 && v[0] == x[0] && v[1] == x[2] && v[2] == x[4], "extrema B-spline-3");
    v = bs3.roots(0.25);
    ok &= testCond(v.size() >= 1, "roots B-spline-3");
    for(size_t i=0; i<v.size(); i++)
        ok &= testCond(fabs(v[i] - x[2]) < 1e-5 || fabs(v[i] - x[3]) < 1e-5, "roots B-spline-3");

    // for a degree-0 B-spline, roots can be found only at grid nodes, and extrema - at grid nodes
    // or endpoints of the interval
    y.resize(4);
    math::BsplineWrapper<0> bs0(x, y);
    v = bs0.extrema(2.5, -0.5);
    ok &= testCond(v.size() == 2 && v[0] == -0.5 && v[1] == x[2], "extrema B-spline-0");
    v = bs0.roots(0.5);
    ok &= testCond(v.size() == 1 && v[0] == x[2], "roots B-spline-0");

    return ok;
}

bool testEmptySplines()
{
    std::vector<double> empty;
    math::LinearInterpolator eLinear(empty, empty);
    math::CubicSpline eCubic(empty, empty);
    math::CubicSpline eHermite(empty, empty, empty);
    math::LogLogSpline eLogLog(empty, empty);
    math::QuinticSpline eQuintic(empty, empty, empty);
    double tmpval=0, tmpder=0, tmpder2=0;
    bool ok = true;
    eLinear.evalDeriv(0, &tmpval, &tmpder, &tmpder2);
    ok &= !(tmpval==tmpval || tmpder==tmpder || tmpder2==tmpder2);   // expected to be NaN
    eCubic.evalDeriv(0, &tmpval, &tmpder, &tmpder2);
    ok &= !(tmpval==tmpval || tmpder==tmpder || tmpder2==tmpder2);
    eHermite.evalDeriv(0, &tmpval, &tmpder, &tmpder2);
    ok &= !(tmpval==tmpval || tmpder==tmpder || tmpder2==tmpder2);
    eLogLog.evalDeriv(0, &tmpval, &tmpder, &tmpder2);
    ok &= !(tmpval==tmpval || tmpder==tmpder || tmpder2==tmpder2);
    eQuintic.evalDeriv(0, &tmpval, &tmpder, &tmpder2);
    ok &= !(tmpval==tmpval || tmpder==tmpder || tmpder2==tmpder2);
    return ok;
}

bool testLength1Splines()
{
    double delta = 0.25;  // all floats in this test are exactly representable
    std::vector<double> x(1, 0.625), val(1, 0.5), der(1, -1.5);
    math::LinearInterpolator oLinear(x, val);
    math::CubicSpline oCubic(x, val, true, der[0], 999 /*derivRight - should be ignored*/);
    math::CubicSpline oHermite(x, val, der);
    math::LogLogSpline oLogLog(x, val);
    math::QuinticSpline oQuintic(x, val, der);
    double tmpval=0, tmpder=0, tmpder2=0;
    bool ok = true;

    // linear and log-log should produce a constant, since they have no information about the derivative
    oLinear.evalDeriv(x[0] + delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0]) && (tmpder == 0) && (tmpder2 == 0);
    oLinear.evalDeriv(x[0] - delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0]) && (tmpder == 0) && (tmpder2 == 0);

    oLogLog.evalDeriv(x[0] + delta, &tmpval, &tmpder, &tmpder2);
    ok &= math::fcmp(tmpval, val[0]) == 0 && (tmpder == 0) && (tmpder2 == 0);
    oLogLog.evalDeriv(x[0] - delta, &tmpval, &tmpder, &tmpder2);
    ok &= math::fcmp(tmpval, val[0]) == 0 && (tmpder == 0) && (tmpder2 == 0);

    // other ones should extrapolate linearly given the derivative at a single point
    oCubic.evalDeriv(x[0] + delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] + delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);
    oCubic.evalDeriv(x[0] - delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] - delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);

    oHermite.evalDeriv(x[0] + delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] + delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);
    oHermite.evalDeriv(x[0] - delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] - delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);

    oQuintic.evalDeriv(x[0] + delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] + delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);
    oQuintic.evalDeriv(x[0] - delta, &tmpval, &tmpder, &tmpder2);
    ok &= (tmpval == val[0] - delta * der[0]) && (tmpder == der[0]) && (tmpder2 == 0);

    return ok;
}

bool testLogScaledSplines()
{
    const int NNODES=41;
    const double XMIN=1., XMAX=100.;
    testfnclog fnc;
    std::vector<double> xnodes = math::createExpGrid(NNODES, XMIN, XMAX);
    xnodes[20] = 10.;  // exact value where the function goes to zero
    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    for(int i=0; i<NNODES; i++)
        fnc.evalDeriv(xnodes[i], &yvalues[i], &yderivs[i]);
    math::CubicSpline  fSpl(xnodes, yvalues, true);
    math::LogLogSpline fLog(xnodes, yvalues);
    math::LogLogSpline fDer(xnodes, yvalues, yderivs);
    double errValS = 0, errValL = 0, errValD = 0, errDerS = 0, errDerL = 0, errDerD = 0;
    std::ofstream strm;
    if(OUTPUT)
        strm.open("test_math_spline_log.dat");
    xnodes = math::createExpGrid(1001, XMIN/2, XMAX*2);
    xnodes[500] = 10.;
    for(unsigned int i=0; i<xnodes.size(); i++) {
        double fval, fder, fder2, sval, sder, sder2, lval, lder, lder2, dval, dder, dder2;
        fnc. evalDeriv(xnodes[i], &fval, &fder, &fder2);
        fSpl.evalDeriv(xnodes[i], &sval, &sder, &sder2);
        fLog.evalDeriv(xnodes[i], &lval, &lder, &lder2);
        fDer.evalDeriv(xnodes[i], &dval, &dder, &dder2);
        double norm = xnodes[i] * fmax(xnodes[i]-1, 0.);
        errValS += pow_2(sval - fval)*norm;
        errValL += pow_2(lval - fval)*norm;
        errValD += pow_2(dval - fval)*norm;
        errDerS += pow_2(sder - fder)*norm;
        errDerL += pow_2(lder - fder)*norm;
        errDerD += pow_2(dder - fder)*norm;
        if(OUTPUT) strm <<
            xnodes[i] << "\t" << fval << " " << sval << " " << lval << " " << dval << "\t" <<
            fder << " " << sder << " " << lder << " " << dder << "\n";
    }
    return
        errValS < 0.2 && errValL < 0.2 && errValD < 0.02 &&
        errDerS < 150 && errDerL < 150 && errDerD < 1.;
}

bool testGaussianIntegral(double x1, double x2, double sigma, int n)
{
    math::Gaussian gauss(sigma);
    // numerical integration
    double intnum = math::integrate(math::FncProduct(math::Monomial(n), gauss), x1, x2, 1e-12);
    // analytic integration provided by the IFunctionIntegral interface
    double intan  = gauss.integrate(x1, x2, n);
    return fabs(intnum-intan) <= fmax(1.5e-15, fabs(intan) * 1e-12);
}

template<class Kernel>
bool testSplineConvolution(const math::BaseInterpolator1d& fnc, const double x, const Kernel& kernel)
{
    // numerical integration
    double intnum = math::integrateAdaptive(Convolved(x, fnc, kernel), fnc.xmin(), fnc.xmax(), 1e-13);
    // analytic integration
    double intan  = fnc.convolve(x, kernel);
    double reldif = intan/intnum - 1;
    return fabs(reldif) < 1e-13;
}

template<int N>
double getAmplFiniteElement(const math::FiniteElement1d<N>& fe, const math::IFunctionIntegral& kernel,
    /*output*/ std::vector<double>& ampl, std::vector<double>& convampl, std::vector<double>& conv2ampl)
{
    const unsigned int gridSize = fe.integrPoints().size();
    // 0. collect the function values at the nodes of integration grid
    std::vector<double> fncValues(gridSize);
    for(unsigned int p=0; p<gridSize; p++)
        fncValues[p] = testfncorig(fe.integrPoints()[p]);
    // 1. compute the projection integrals, using the collected function values, and solve the linear
    // equation to find the amplitudes of the B-spline approximation of the original function
    std::vector<double> pv = fe.computeProjVector(fncValues);
    math::BandMatrix<double> pm = fe.computeProjMatrix();
    ampl = solveBand(pm, pv);
    // 2. compute the amplitudes of the B-spline expansion of the original function convolved
    // with the give kernel, using directly the values of the original function collected at step 0
    pv = fe.computeConvVector(fncValues, kernel);
    convampl = solveBand(pm, pv);
    // 3. compute the amplitudes of the B-spline expansion of the convolution of the _approximate_
    // (interpolated) function and the same kernel: they are obtained by applying the B-spline
    // convolution matrix to the amplitudes of interpolated function produced at step 1
    math::Matrix<double> cm = fe.computeConvMatrix(kernel);
    math::blas_dgemv(math::CblasNoTrans, 1., cm, ampl, 0., /*output*/pv);
    conv2ampl = solveBand(pm, pv);
    // 4. consistency check: the B-spline convolution matrix with a zero-width kernel should be
    // identical to the Gram (projection) matrix "pm" found in step 1
    cm = fe.computeConvMatrix(math::Gaussian(0.));
    math::blas_daxpy(-1, math::Matrix<double>(pm), cm);
    return sqrt(math::blas_dnrm2(cm) / cm.size());  // should be close to machine epsilon
}

bool testFiniteElement()
{
    // test the analytic integration provided by the IFunctionIntegral interface of math::Gaussian
    if( !testGaussianIntegral( 1.0, 2.0, 0.5, 0) ||
        !testGaussianIntegral( 3.5, 4.0, 0.5, 0) ||
        !testGaussianIntegral( 0.2, 2.0, 0.6, 1) ||
        !testGaussianIntegral(-3.0, 2.0, 0.7, 2) ||
        !testGaussianIntegral(-6.0,-5.0, 0.7, 2) ||
        !testGaussianIntegral(-1.0,-3.0, 0.8, 3) ||
        !testGaussianIntegral( 1.0,-2.0, 0.9, 4) ||
        !testGaussianIntegral(-1.0, 2.0, 0.3, 5) ||
        !testGaussianIntegral( 0.0, 2.0, 0.2, 6) ||
        !testGaussianIntegral(-1.0, 1.0, 0.3, 5) ||
        !testGaussianIntegral(-1.0, 1.0, 0.2, 6) ||
        !testGaussianIntegral( 0.0, 2.0, 0.5, 7) ||
        !testGaussianIntegral(-2.0, 0.0, 0.5, 7) ||
        !testGaussianIntegral(-2.0, 2.0, 0.5, 7) ||
        !testGaussianIntegral( 0.0, 2.0, 0.5, 8) ||
        !testGaussianIntegral(-2.0, 0.0, 0.5, 8) ||
        !testGaussianIntegral(-2.0, 2.0, 0.5, 8) ||
        !testGaussianIntegral( 2.0,-2.0, 0.5, 8) )
    {
        std::cout << "Analytic integration of a Gaussian times monomial function "
            "\033[1;31m failed\033[0m\n";
        return false;
    }

    // the function to approximate is  a somewhat smoothed top-hat function;
    // the convolution kernel is a one-sided exponential profile with scale SIGMA
    const int NNODES  = 21;
    const int NTEST   = std::max<int>(10*(NNODES-1)+1, 101);
    const double XMIN = 0.;
    const double XMAX = 10.;
    const double SIGMA= 0.5;
    std::vector<double> xnodes = math::createUniformGrid(NNODES, XMIN, XMAX);
    ExpKernel kernel(SIGMA);
    math::FiniteElement1d<0> fe0((math::BsplineInterpolator1d<0>(xnodes)));
    math::FiniteElement1d<1> fe1((math::BsplineInterpolator1d<1>(xnodes)));
    math::FiniteElement1d<2> fe2((math::BsplineInterpolator1d<2>(xnodes)));
    math::FiniteElement1d<3> fe3((math::BsplineInterpolator1d<3>(xnodes)));
    std::vector<double> am0, cam0, tam0, am1, cam1, tam1, am2, cam2, tam2, am3, cam3, tam3;
    double
    zam0 = getAmplFiniteElement(fe0, kernel, am0, cam0, tam0),
    zam1 = getAmplFiniteElement(fe1, kernel, am1, cam1, tam1),
    zam2 = getAmplFiniteElement(fe2, kernel, am2, cam2, tam2),
    zam3 = getAmplFiniteElement(fe3, kernel, am3, cam3, tam3);
    std::ofstream strm;
    if(OUTPUT) {
        strm.open("test_math_spline_femconv.dat");
        strm << "x         orig_fnc  conv_fnc  fem0      femconv0  fem2conv0 "
        "fem1      femconv1  fem2conv1 fem2      femconv2  fem2conv2 fem3      femconv3  fem2conv3\n";
    }
    double err0=0, erc0=0, ert0=0, err1=0, erc1=0, ert1=0, err2=0, erc2=0, ert2=0, err3=0, erc3=0, ert3=0;
    for(int i=0; i<NTEST; i++) {
        double x = XMIN + (XMAX-XMIN) / (NTEST-1) * i,
        origfnc  = testfncorig(x),
        convfnc  = math::integrateAdaptive(
            Convolved(x, ExpKernel(SIGMA), math::FncWrapper(testfncorig)), x-XMAX, x-XMIN, 1e-6),
        fem0     = fe0.interp.interpolate(x, am0),
        femconv0 = fe0.interp.interpolate(x, cam0),
        fem2conv0= fe0.interp.interpolate(x, tam0),
        fem1     = fe1.interp.interpolate(x, am1),
        femconv1 = fe1.interp.interpolate(x, cam1),
        fem2conv1= fe1.interp.interpolate(x, tam1),
        fem2     = fe2.interp.interpolate(x, am2),
        femconv2 = fe2.interp.interpolate(x, cam2),
        fem2conv2= fe2.interp.interpolate(x, tam2),
        fem3     = fe3.interp.interpolate(x, am3),
        femconv3 = fe3.interp.interpolate(x, cam3),
        fem2conv3= fe3.interp.interpolate(x, tam3);

        err0 += pow_2(fem0-origfnc);
        err1 += pow_2(fem1-origfnc);
        err2 += pow_2(fem2-origfnc);
        err3 += pow_2(fem3-origfnc);
        erc0 += pow_2(femconv0-convfnc);
        erc1 += pow_2(femconv1-convfnc);
        erc2 += pow_2(femconv2-convfnc);
        erc3 += pow_2(femconv3-convfnc);
        ert0 += pow_2(femconv0-fem2conv0);
        ert1 += pow_2(femconv1-fem2conv1);
        ert2 += pow_2(femconv2-fem2conv2);
        ert3 += pow_2(femconv3-fem2conv3);
        if(OUTPUT)
            strm << utils::pp(x, 9) +' ' +
            utils::pp(origfnc,   9) +' ' +
            utils::pp(convfnc,   9) +' ' +
            utils::pp(fem0,      9) +' ' +
            utils::pp(femconv0,  9) +' ' +
            utils::pp(fem2conv0, 9) +' ' +
            utils::pp(fem1,      9) +' ' +
            utils::pp(femconv1,  9) +' ' +
            utils::pp(fem2conv1, 9) +' ' +
            utils::pp(fem2,      9) +' ' +
            utils::pp(femconv2,  9) +' ' +
            utils::pp(fem2conv2, 9) +' ' +
            utils::pp(fem3,      9) +' ' +
            utils::pp(femconv3,  9) +' ' +
            utils::pp(fem2conv3, 9) +'\n';
    }
    err0 = sqrt(err0 / NTEST);
    erc0 = sqrt(erc0 / NTEST);
    ert0 = sqrt(ert0 / NTEST);
    err1 = sqrt(err1 / NTEST);
    erc1 = sqrt(erc1 / NTEST);
    ert1 = sqrt(ert1 / NTEST);
    err2 = sqrt(err2 / NTEST);
    erc2 = sqrt(erc2 / NTEST);
    ert2 = sqrt(ert2 / NTEST);
    err3 = sqrt(err3 / NTEST);
    erc3 = sqrt(erc3 / NTEST);
    ert3 = sqrt(ert3 / NTEST);
    std::cout << "Finite-element approximation and convolution: RMS error "
    "in FEM0=" + utils::pp(err0, 8) + ", conv0=" + utils::pp(erc0, 8) +
    ",  FEM1=" + utils::pp(err1, 8) + ", conv1=" + utils::pp(erc1, 8) +
    ",  FEM2=" + utils::pp(err2, 8) + ", conv2=" + utils::pp(erc2, 8) +
    ",  FEM3=" + utils::pp(err3, 8) + ", conv3=" + utils::pp(erc3, 8) + "\n";
    return err0 < 0.06  && erc0 < 0.05  && ert0 < 1e-15 && zam0 < 1e-15
        && err1 < 0.01  && erc1 < 0.005 && ert1 < 2e-3  && zam1 < 1e-15
        && err2 < 0.003 && erc2 < 0.002 && ert2 < 3e-4  && zam2 < 1e-15
        && err3 < 0.005 && erc3 < 0.001 && ert3 < 9e-4  && zam3 < 1e-15;
}

#ifdef  TESTFNC1D_SMOOTH
const double XMIN1D = 0;
const double XMAX1D = M_PI;
#else
const double XMIN1D = 0.4;
const double XMAX1D = 6.2832;
#endif

// performance test - repeat spline evaluation many times
template<int numDeriv>
std::string evalSpline(const math::IFunction& fnc)
{
    const int NPOINTS = 100;
    clock_t clk = std::clock();
    int iter=0;
    // run for a fixed wall-clock time (don't check it too often, though), but variable # of iterations
    for(; iter%1000!=999 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NPOINTS; i++) {
            double x = i * (XMAX1D-XMIN1D) / NPOINTS + XMIN1D, v, d, s;
            fnc.evalDeriv(x, &v, numDeriv>=1 ? &d : NULL, numDeriv>=2 ? &s : NULL);
        }
    }
    return utils::pp(1.0 * iter * NPOINTS * CLOCKS_PER_SEC / (std::clock()-clk), 6);
}

std::string evalConv(const math::BaseInterpolator1d& fnc, const math::IFunctionIntegral& kernel)
{
    const int NPOINTS = 100;
    clock_t clk = std::clock();
    int iter=0;
    // run for a fixed wall-clock time (don't check it too often, though), but variable # of iterations
    for(; iter%100!=99 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NPOINTS; i++) {
            double x = i * (XMAX1D-XMIN1D) / NPOINTS + XMIN1D;
            fnc.convolve(x, kernel);
        }
    }
    double result = 1.0 * iter * NPOINTS * CLOCKS_PER_SEC / (std::clock()-clk);
    return result<1e6 ? utils::toString(int(result)) : utils::pp(result, 6);
}

inline bool testnan(const math::IFunction& fnc) {
    double result = fnc(NAN);
    return result != result;
}


bool test1dSpline()
{
    std::cout << "\033[1;33m1d interpolation\033[0m\n";

    const int NNODES  = 20;
    const int NSUBINT = 50;
    std::vector<double> yvalues(NNODES), yderivs(NNODES), yderivs2(NNODES);
    std::vector<double> xnodes = math::createUniformGrid(NNODES, XMIN1D, XMAX1D);
    testfnc1d fnc;   // the original function that we are approximating
    for(int i=0; i<NNODES; i++)
        fnc.evalDeriv(xnodes[i], &yvalues[i], &yderivs[i], &yderivs2[i]);

    // a collection of approximating splines of various types:
    // 1. linear interpolator
    math::LinearInterpolator fLinear(xnodes, yvalues);

    // 2. cubic spline with natural boundary conditions
    math::CubicSpline   fNatural(xnodes, yvalues);

    // 3. cubic, clamped -- specify derivs at the boundaries
    math::CubicSpline   fClamped(xnodes, yvalues, /*regularize*/false, yderivs.front(), yderivs.back());

    // 4. hermite cubic interpolator -- specify derivs at all nodes
    math::CubicSpline   fHermite(xnodes, yvalues, yderivs);

    // 5. natural quintic spline constructed from function values only
    math::QuinticSpline fQuiNatu(xnodes, yvalues);

    // 6. clamped quintic spline constructed from function values plus first derivatives at endpoints
    math::QuinticSpline fQuiClam(xnodes, yvalues, yderivs.front(), yderivs.back());

    // 7. "standard" quintic spline constructed from values and first derivatives at all nodes
    math::QuinticSpline fQuintic(xnodes, yvalues, yderivs);

    // 8. hermite quintic interpolator constructed from values, first and second derivs at all nodes
    math::QuinticSpline fQuiHerm(xnodes, yvalues, yderivs, yderivs2);

    // 9-10: 1d cubic B-spline represented as a sum over basis functions;
    // this class only stores the x-grid and presents a method for computing the interpolant
    // for any array of amplitudes passed as a parameter -- we will use two different ones:
    math::FiniteElement1d<3> fBspline((math::BsplineInterpolator1d<3>(xnodes)));

    // 9. compute the amplitudes from another cubic spline, which results in an equivalent
    // representation in terms of B-splines
    std::vector<double> amplBsplineEquivClamped = fBspline.computeAmplitudes(fClamped);

    // 10. compute the amplitudes from the original function in a different way:
    // instead of using just from the function values and possibly the derivatives at the grid
    // nodes, as in all previous cases, it collects the function values at several points
    // inside each grid segment, thus achieving a somewhat better overall approximation
    std::vector<double> amplBsplineOrig = fBspline.computeAmplitudes(fnc);

    // 11. 1d clamped cubic spline equivalent to a B-spline (this demonstrates the equivalence
    // in both directions, as the amplitudes of this B-spline were computed from another cubic spline).
    // The array of amplitudes can be simply passed to the constructor of CubicSpline instead of
    // an array of function values -- the mode of operation is determined from the size of this array;
    // if it is equal to the number of x-nodes, it refers to the values of the function at these nodes;
    // if it is longer by two elements, it refers to the B-spline amplitudes.
    math::CubicSpline fClampedEquivBspline(xnodes, amplBsplineEquivClamped);

    // 12. 1d quadratic B-spline representing the derivative of the cubic B-spline
    math::BsplineInterpolator1d<2> fBsplineDeriv(xnodes);
    std::vector<double> amplBsplineDeriv = fBspline.interp.deriv(amplBsplineOrig);

    // 13. reverse procedure: cubic B-spline which is the antiderivative of the quadratic B-spline
    std::vector<double> amplBsplineAnti = fBsplineDeriv.antideriv(amplBsplineDeriv);

    // 14. cubic spline initialized from amplitudes of a 2nd degree B-spline created at step 12
    math::CubicSpline fCubicDeriv(xnodes, amplBsplineDeriv);

    // 15. a small degree-2 B-spline approximating a standard Gaussian
    std::vector<double> xnodes2 = math::createUniformGrid(4, -3, 3), ampl2(5); ampl2[2]=0.5;
    math::BsplineWrapper<2> bsplGaussian(xnodes2, ampl2);

    // output file
    std::ofstream strm;

    // accumulators for computing rms errors in values, derivatives and second derivatives
    // of these approximations
    double errLinearVal=0,
        errBsplineVal=0, errBsplineDer=0, errBsplineDer2=0,
        errNaturalVal=0, errNaturalDer=0, errNaturalDer2=0,
        errClampedVal=0, errClampedDer=0, errClampedDer2=0,
        errHermiteVal=0, errHermiteDer=0, errHermiteDer2=0,
        errQuiNatuVal=0, errQuiNatuDer=0, errQuiNatuDer2=0,
        errQuiClamVal=0, errQuiClamDer=0, errQuiClamDer2=0,
        errQuinticVal=0, errQuinticDer=0, errQuinticDer2=0,
        errQuiHermVal=0, errQuiHermDer=0, errQuiHermDer2=0,
        errBsplineEquivClamped=0, errClampedEquivBspline=0,
        errBsplineQuad=0, errCubicEquivBsplineQuad = 0;
    bool oknat = true, okcla = true, okher = true, okqui = true;

    // loop through the range of x covering the input grid,
    // using points both at grid nodes and inside grid segments
    const int NPOINTS = (NNODES-1)*NSUBINT;
    for(int i=0; i<=NPOINTS; i++) {
        double xa = xnodes[i/NSUBINT];
        double xb = i<NPOINTS ? xnodes[i/NSUBINT+1] : xa;
        double x  = xa*(1 - (i%NSUBINT)*1.0/NSUBINT) + xb*(i%NSUBINT)/NSUBINT;

        // 0. the original function value and derivs
        double origVal, origDer, origDer2, origDer3;
        fnc.evalDeriv(x, &origVal, &origDer, &origDer2, &origDer3);

        // 1. linear interpolator
        double fLinearVal = fLinear.value(x);

        // 2. natural cubic spline
        double fNaturalVal, fNaturalDer, fNaturalDer2;
        fNatural.evalDeriv(x, &fNaturalVal, &fNaturalDer, &fNaturalDer2);

        // 3. clamped cubic spline
        double fClampedVal, fClampedDer, fClampedDer2;
        fClamped.evalDeriv(x, &fClampedVal, &fClampedDer, &fClampedDer2);

        // 4. hermite interpolator constructed from the original function
        double fHermiteVal, fHermiteDer, fHermiteDer2;
        fHermite.evalDeriv(x, &fHermiteVal, &fHermiteDer, &fHermiteDer2);

        // 5. natural quintic spline
        double fQuiNatuVal, fQuiNatuDer, fQuiNatuDer2;
        fQuiNatu.evalDeriv(x, &fQuiNatuVal, &fQuiNatuDer, &fQuiNatuDer2);

        // 6. quintic spline from approximate derivatives
        double fQuiClamVal, fQuiClamDer, fQuiClamDer2;
        fQuiClam.evalDeriv(x, &fQuiClamVal, &fQuiClamDer, &fQuiClamDer2);

        // 7. quintic spline from exact first derivatives
        double fQuinticVal, fQuinticDer, fQuinticDer2;
        fQuintic.evalDeriv(x, &fQuinticVal, &fQuinticDer, &fQuinticDer2);

        // 8. quintic hermite interpolator from the original function and its two derivatives
        double fQuiHermVal, fQuiHermDer, fQuiHermDer2;
        fQuiHerm.evalDeriv(x, &fQuiHermVal, &fQuiHermDer, &fQuiHermDer2);

        // 9. b-spline equivalent to the clamped cubic spline
        double fBsplineEquivClampedVal = fBspline.interp.interpolate(x, amplBsplineEquivClamped);

        // 10. b-spline constructed from the original function
        double fBsplineVal = fBspline.interp.interpolate(x, amplBsplineOrig);
        double fBsplineDer = fBspline.interp.interpolate(x, amplBsplineOrig, 1);
        double fBsplineDer2= fBspline.interp.interpolate(x, amplBsplineOrig, 2);

        // 11. clamped cubic spline equivalent to the b-spline
        double fClampedEquivBsplineVal = fClampedEquivBspline(x);

        // 12. quadratic b-spline representing the derivative of the original cubic b-spline
        double fBsplineQuad = fBsplineDeriv.interpolate(x, amplBsplineDeriv);

        // 13. ordinary cubic spline equivalent to the above 2nd degree B-spline
        double fCubicEquivBsplineQuad = fCubicDeriv(x);

        // accumulate errors of various approximations
        errLinearVal  += pow_2(origVal - fLinearVal );
        errBsplineVal += pow_2(origVal - fBsplineVal);
        errNaturalVal += pow_2(origVal - fNaturalVal);
        errClampedVal += pow_2(origVal - fClampedVal);
        errHermiteVal += pow_2(origVal - fHermiteVal);
        errQuiNatuVal += pow_2(origVal - fQuiNatuVal);
        errQuiClamVal += pow_2(origVal - fQuiClamVal);
        errQuinticVal += pow_2(origVal - fQuinticVal);
        errQuiHermVal += pow_2(origVal - fQuiHermVal);
        errBsplineDer += pow_2(origDer - fBsplineDer);
        errNaturalDer += pow_2(origDer - fNaturalDer);
        errClampedDer += pow_2(origDer - fClampedDer);
        errHermiteDer += pow_2(origDer - fHermiteDer);
        errQuiNatuDer += pow_2(origDer - fQuiNatuDer);
        errQuiClamDer += pow_2(origDer - fQuiClamDer);
        errQuinticDer += pow_2(origDer - fQuinticDer);
        errQuiHermDer += pow_2(origDer - fQuiHermDer);
        errBsplineDer2+= pow_2(origDer2- fBsplineDer2);
        errNaturalDer2+= pow_2(origDer2- fNaturalDer2);
        errClampedDer2+= pow_2(origDer2- fClampedDer2);
        errHermiteDer2+= pow_2(origDer2- fHermiteDer2);
        errQuiNatuDer2+= pow_2(origDer2- fQuiNatuDer2);
        errQuiClamDer2+= pow_2(origDer2- fQuiClamDer2);
        errQuinticDer2+= pow_2(origDer2- fQuinticDer2);
        errQuiHermDer2+= pow_2(origDer2- fQuiHermDer2);

        // keep track of error in equivalent representations of the same interpolant
        errBsplineEquivClamped = fmax(errBsplineEquivClamped,
            fabs(fBsplineEquivClampedVal - fClampedVal));
        errClampedEquivBspline = fmax(errClampedEquivBspline,
            fabs(fBsplineEquivClampedVal - fClampedEquivBsplineVal));
        errBsplineQuad = fmax(errBsplineQuad, fabs(fBsplineQuad - fBsplineDer));
        errCubicEquivBsplineQuad = fmax(errCubicEquivBsplineQuad,
            fabs(fBsplineQuad - fCubicEquivBsplineQuad));

        // if x coincides with a grid node, the values of interpolants should be
        // exact to machine precision by construction
        if(i%NSUBINT == 0) {
            int k = i/NSUBINT;
            oknat &= fNaturalVal == yvalues[k];
            okcla &= fClampedVal == yvalues[k];
            okher &= fHermiteVal == yvalues[k] && fHermiteDer == yderivs[k];
            okqui &= fQuinticVal == yvalues[k] && fQuinticDer == yderivs[k] &&
                     fQuiNatuVal == yvalues[k] && fQuiHermVal == yvalues[k] &&
                     fQuiHermDer == yderivs[k] && fQuiHermDer2== yderivs2[k];
            if(k==0 || k==NNODES-1)   // for a clamped spline, also check endpoint derivs
                okcla &= fClampedDer == yderivs[k];
        }

        // dump the values to a file if requested
        if(OUTPUT) {
            if(i==0) {
                strm.open("test_math_spline1d.dat");
                strm << "x\torig:f f' f''\t"
                     "CubBspline:f f' f''\tCubNatural:f f' f''\t"
                     "CubClamped:f f' f''\tCubHermite:f f' f''\t"
                     "QuiNatural:f f' f''\tQuiClamped:f f' f''\t"
                     "QuinticStd:f f' f''\tQuiHermite:f f' f''\n";
            }
            strm << utils::pp(x,   9) +'\t'+
            utils::pp(origVal,    10) +' ' +
            utils::pp(origDer,     9) +' ' +
            utils::pp(origDer2,    9) +'\t'+
            utils::pp(fBsplineVal,10) +' ' +
            utils::pp(fBsplineDer, 9) +' ' +
            utils::pp(fBsplineDer2,9) +'\t'+
            utils::pp(fNaturalVal,10) +' ' +
            utils::pp(fNaturalDer, 9) +' ' +
            utils::pp(fNaturalDer2,9) +'\t'+
            utils::pp(fClampedVal,10) +' ' +
            utils::pp(fClampedDer, 9) +' ' +
            utils::pp(fClampedDer2,9) +'\t'+
            utils::pp(fHermiteVal,10) +' ' +
            utils::pp(fHermiteDer, 9) +' ' +
            utils::pp(fHermiteDer2,9) +'\t'+
            utils::pp(fQuiNatuVal,10) +' ' +
            utils::pp(fQuiNatuDer, 9) +' ' +
            utils::pp(fQuiNatuDer2,9) +'\t'+
            utils::pp(fQuiClamVal,10) +' ' +
            utils::pp(fQuiClamDer, 9) +' ' +
            utils::pp(fQuiClamDer2,9) +'\t'+
            utils::pp(fQuinticVal,10) +' ' +
            utils::pp(fQuinticDer, 9) +' ' +
            utils::pp(fQuinticDer2,9) +'\t'+
            utils::pp(fQuiHermVal,10) +' ' +
            utils::pp(fQuiHermDer, 9) +' ' +
            utils::pp(fQuiHermDer2,9) +'\n';
            if(i==NPOINTS)
                strm.close();
        }
    }
    errLinearVal   = sqrt(errLinearVal  / NPOINTS);
    errBsplineVal  = sqrt(errBsplineVal / NPOINTS);
    errNaturalVal  = sqrt(errNaturalVal / NPOINTS);
    errClampedVal  = sqrt(errClampedVal / NPOINTS);
    errHermiteVal  = sqrt(errHermiteVal / NPOINTS);
    errQuiNatuVal  = sqrt(errQuiNatuVal / NPOINTS);
    errQuiClamVal  = sqrt(errQuiClamVal / NPOINTS);
    errQuinticVal  = sqrt(errQuinticVal / NPOINTS);
    errQuiHermVal  = sqrt(errQuiHermVal / NPOINTS);
    errBsplineDer  = sqrt(errBsplineDer / NPOINTS);
    errNaturalDer  = sqrt(errNaturalDer / NPOINTS);
    errClampedDer  = sqrt(errClampedDer / NPOINTS);
    errHermiteDer  = sqrt(errHermiteDer / NPOINTS);
    errQuiNatuDer  = sqrt(errQuiNatuDer / NPOINTS);
    errQuiClamDer  = sqrt(errQuiClamDer / NPOINTS);
    errQuinticDer  = sqrt(errQuinticDer / NPOINTS);
    errQuiHermDer  = sqrt(errQuiHermDer / NPOINTS);
    errBsplineDer2 = sqrt(errBsplineDer2/ NPOINTS);
    errNaturalDer2 = sqrt(errNaturalDer2/ NPOINTS);
    errClampedDer2 = sqrt(errClampedDer2/ NPOINTS);
    errHermiteDer2 = sqrt(errHermiteDer2/ NPOINTS);
    errQuiNatuDer2 = sqrt(errQuiNatuDer2/ NPOINTS);
    errQuiClamDer2 = sqrt(errQuiClamDer2/ NPOINTS);
    errQuinticDer2 = sqrt(errQuinticDer2/ NPOINTS);
    errQuiHermDer2 = sqrt(errQuiHermDer2/ NPOINTS);

    std::cout <<
        "RMS error in linear interpolator: "   + utils::pp(errLinearVal,  8) +
        "\nin cubic B-spline (f,f',f''):     " + utils::pp(errBsplineVal, 8) +
        "," + utils::pp(errBsplineDer, 8) + ","+ utils::pp(errBsplineDer2,8) +
        "\nin natural cubic spline :         " + utils::pp(errNaturalVal, 8) +
        "," + utils::pp(errNaturalDer, 8) + ","+ utils::pp(errNaturalDer2,8) +
        "\nin clamped cubic spline:          " + utils::pp(errClampedVal, 8) +
        "," + utils::pp(errClampedDer, 8) + ","+ utils::pp(errClampedDer2,8) +
        "\nin hermite cubic interpolator:    " + utils::pp(errHermiteVal, 8) +
        "," + utils::pp(errHermiteDer, 8) + ","+ utils::pp(errHermiteDer2,8) +
        "\nin natural quintic spline:        " + utils::pp(errQuiNatuVal, 8) +
        "," + utils::pp(errQuiNatuDer, 8) + ","+ utils::pp(errQuiNatuDer2,8) +
        "\nin clamped quintic spline:        " + utils::pp(errQuiClamVal, 8) +
        "," + utils::pp(errQuiClamDer, 8) + ","+ utils::pp(errQuiClamDer2,8) +
        "\nin standard quintic spline:       " + utils::pp(errQuinticVal, 8) +
        "," + utils::pp(errQuinticDer, 8) + ","+ utils::pp(errQuinticDer2,8) +
        "\nin hermite quintic interpolator:  " + utils::pp(errQuiHermVal, 8) +
        "," + utils::pp(errQuiHermDer, 8) + ","+ utils::pp(errQuiHermDer2,8) +
        "\nmax|cubic-bspline|: " + utils::pp(errBsplineEquivClamped, 8) +
        " and " + utils::pp(errClampedEquivBspline, 8) + "\n";

    // test the integration functions
    const double X1 = (xnodes[0]+xnodes[1])/2, X2 = xnodes[xnodes.size()-2]-0.1;
    bool okintcla = testIntegral(fClamped, X1, X2);
    bool okintnat = testIntegral(fNatural, -1.234567, xnodes.back()+1);
    bool okintqui = testIntegral(fQuintic, xnodes.back()+1, -1.234567);
    double intClamped = fClamped.integrate(X1, X2);
    double intBspline = fBspline.interp.integrate(X1, X2, amplBsplineEquivClamped);
    double intNumeric = math::integrateAdaptive(fClamped, X1, X2, 1e-14);
    double intBsplGaussian = bsplGaussian.integrate(xnodes2.front(), xnodes2.back());
    bool okintnum = fabs(intClamped-intNumeric) < 1e-13 && fabs(intClamped-intBspline) < 1e-13 &&
        fabs(intBsplGaussian-1) < 1e-14;
    // check that the amplitudes of the cubic B-spline that represents the antiderivative
    // of the quadratic B-spline representing the derivative of the original spline
    // are the same as the amplitudes of the original spline (up to a constant term - the 0th amplitude)
    double errBsplineAnti = 0;
    for(unsigned int i=0; i<amplBsplineAnti.size(); i++)
        errBsplineAnti = fmax(errBsplineAnti,
            fabs(amplBsplineAnti[i] - amplBsplineOrig[i] + amplBsplineOrig[0]));

    // test log-scaled splines
    bool oklogspl = testLogScaledSplines();

    // test monotonicity filter
    bool okmon;
    {
        std::vector<double> xx(5), yy(5);  // a semi-monotonic sequence of values with a sharp jump
        xx[0] = 1.2;  xx[1] = 1.6;  xx[2] = 1.8;  xx[3] = 2.2;  xx[4] = 2.5;
        yy[0] = 0.4;  yy[1] = 0.5;  yy[2] = 1.0;  yy[3] = 1.1;  yy[4] = 1.1;
        math::CubicSpline splmon(xx, yy, true);
        math::CubicSpline splnon(xx, yy, false);
        okmon = splmon.extrema().size()==2 && splnon.extrema().size()>2;
    }

    // test finite-element approximation and convolution
    bool okfem = testFiniteElement();

    // test analytic convolution
    bool okcon = true;
    okcon &= testSplineConvolution(fLinear , 4.5, ExpKernel(1.5));
    okcon &= testSplineConvolution(fClamped, 4.5, ExpKernel(1.5));
    okcon &= testSplineConvolution(fQuintic, 4.5, ExpKernel(1.5));
    okcon &= testSplineConvolution(math::BsplineWrapper<2>(
        xnodes, amplBsplineDeriv), 4.5, ExpKernel(1.5));
    okcon &= testSplineConvolution(fQuintic, 2.5, bsplGaussian);
    okcon &= testSplineConvolution(fClamped, 2.5, math::Gaussian(1.0));
    okcon &= testSplineConvolution(fQuintic, 2.5, math::Gaussian(1.0));
    okcon &= testSplineConvolution(fClamped, 2.5, math::Gaussian(0.1));
    okcon &= testSplineConvolution(fQuintic, 2.5, math::Gaussian(0.1));
    // convolution with a Gaussian of zero width (delta function) should return the value of f(x)
    okcon &= fabs(fLinear (2.34) - fLinear .convolve(2.34, math::Gaussian(0))) < 1e-15;
    okcon &= fabs(fQuintic(3.45) - fQuintic.convolve(3.45, math::Gaussian(0))) < 1e-15;
    okcon &= fabs(bsplGaussian(0.4) - bsplGaussian.convolve(0.4, math::Gaussian(0))) < 1e-15;

    // check that splines return NaN when the input is NaN
    bool oknan = testnan(fLinear) && testnan(fNatural) && testnan(fQuintic) && testnan(bsplGaussian);

    bool okroots = testExtremaAndRoots();

    bool ok =
    testCond(testEmptySplines(), "empty splines failed") &&
    testCond(testLength1Splines(), "length-1 splines failed") &&
    testCond(oknat, "natural cubic spline values at grid nodes are inexact") &&
    testCond(okcla, "clamped cubic spline values at grid nodes are inexact") &&
    testCond(okher, "hermite cubic spline values or derivatives at grid nodes are inexact") &&
    testCond(okqui, "quintic spline values or derivatives at grid nodes are inexact") &&
    testCond(errBsplineEquivClamped   < 1e-14, "bspline<>clamped") &&
    testCond(errClampedEquivBspline   < 1e-14, "clamped<>bspline") &&
    testCond(errCubicEquivBsplineQuad < 1e-14, "bspline2<>cubic") &&
    testCond(errBsplineQuad < 1e-14, "cubic bpline deriv<>quadratic bspline") &&
    testCond(errBsplineAnti < 1e-14, "quadratic bpline antideriv<>cubic bspline") &&
    testCond(errBsplineVal < 1.0e-4, "error in cubic b-spline is too large") &&
    testCond(errNaturalVal < 1.0e-3, "error in natural cubic spline is too large") &&
    testCond(errClampedVal < 4.0e-4, "error in clamped cubic spline is too large") &&
    testCond(errHermiteVal < 3.1e-4, "error in hermite cubic interpolator is too large") &&
    testCond(errQuiNatuVal < 1.0e-3, "error in natural quintic spline is too large") &&
    testCond(errQuiClamVal < 4.0e-4, "error in clamped quintic spline is too large") &&
    testCond(errQuinticVal < 1.2e-4, "error in standard quintic spline is too large") &&
    testCond(errQuiHermVal < 0.8e-5, "error in hermite quintic interpolator is too large") &&
    testCond(errBsplineDer < 3.7e-3, "error in cubic b-spline derivative is too large") &&
    testCond(errNaturalDer < 1.2e-2, "error in natural cubic spline derivative is too large") &&
    testCond(errClampedDer < 4.5e-3, "error in clamped cubic spline derivative is too large") &&
    testCond(errHermiteDer < 3.5e-3, "error in hermite cubic spline derivative is too large") &&
    testCond(errQuiNatuDer < 1.3e-2, "error in natural quintic spline derivative is too large") &&
    testCond(errQuiClamDer < 4.0e-3, "error in clamped quintic spline derivative is too large") &&
    testCond(errQuinticDer < 1.4e-3, "error in standard quintic spline derivative is too large") &&
    testCond(errQuiHermDer < 1.0e-4, "error in hermite quintic interpolator derivative is too large") &&
    testCond(errBsplineDer2< 0.12,   "error in cubic b-spline 2nd derivative is too large") &&
    testCond(errNaturalDer2< 0.25,   "error in natural cubic spline 2nd derivative is too large") &&
    testCond(errClampedDer2< 0.085,  "error in clamped cubic spline 2nd derivative is too large") &&
    testCond(errHermiteDer2< 0.075,  "error in hermite cubic spline 2nd derivative is too large") &&
    testCond(errQuiNatuDer2< 0.25,   "error in natural quintic spline 2nd derivative is too large") &&
    testCond(errQuiClamDer2< 0.065,  "error in clamped quintic spline 2nd derivative is too large") &&
    testCond(errQuinticDer2< 0.035,  "error in standard quintic spline 2nd derivative is too large") &&
    testCond(errQuiHermDer2< 0.003,  "error in hermite quintic interpolator 2nd derivative is too large") &&
    testCond(okintcla, "integral of clamped spline is incorrect") &&
    testCond(okintnat, "integral of natural spline is incorrect") &&
    testCond(okintqui, "integral of quintic spline is incorrect") &&
    testCond(okintnum, "integral of B-spline is incorrect") &&
    testCond(oklogspl, "log-scaled splines failed") &&
    testCond(okmon, "monotonicity analysis failed") &&
    testCond(okfem, "finite-element failed") &&
    testCond(okcon, "analytic convolution of splines failed") &&
    testCond(okroots, "finding extrema and roots failed") &&
    testCond(oknan, "splines do not return NAN when they should");

    //----------- test the performance of 1d spline calculation -------------//
    std::cout << "Cubic   spline w/o deriv: " + evalSpline<0>(fNatural) + ", 1st deriv: " +
    evalSpline<1>(fClamped) + ", 2nd deriv: " + evalSpline<2>(fNatural) + " eval/s\n";
    std::cout << "Quintic spline w/o deriv: " + evalSpline<0>(fQuintic) + ", 1st deriv: " +
    evalSpline<1>(fQuintic) + ", 2nd deriv: " + evalSpline<2>(fQuintic) + " eval/s\n";
    std::cout << "B-spline of degree N=3:   " +
    evalSpline<0>(math::BsplineWrapper<3>(xnodes, amplBsplineOrig)) + " eval/s\n";
    math::Gaussian unitGaussian(1.0), tinyGaussian(0.001);
    std::cout << "Linear interp. w/convol.: " + evalConv(fLinear , bsplGaussian) + ", " +
    evalConv(fLinear , unitGaussian) +  ", "  + evalConv(fLinear , tinyGaussian) + " eval/s\n";
    std::cout << "Cubic   spline w/convol.: " + evalConv(fNatural, bsplGaussian) + ", " +
    evalConv(fNatural, unitGaussian) +  ", "  + evalConv(fNatural, tinyGaussian) + " eval/s\n";
    std::cout << "Quintic spline w/convol.: " + evalConv(fQuintic, bsplGaussian) + ", " +
    evalConv(fQuintic, unitGaussian) +  ", "  + evalConv(fQuintic, tinyGaussian) + " eval/s\n";
    return ok;
}

//----------- test 2d cubic and quintic spline ------------//

// performance test - repeat spline evaluation many times
template<int numDeriv>
std::string evalSpline2d(const math::BaseInterpolator2d& fnc)
{
    const int NPOINTS = 100;
    const double XMIN = -1.9, XMAX = 2.2, YMIN = -2.1, YMAX = 1.7;
    clock_t clk = std::clock();
    int iter=0;
    for(; iter%100!=99 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NPOINTS; i++) {
            double x = (XMAX-XMIN)*(i*1./NPOINTS)+XMIN;
            for(int j=0; j<=NPOINTS; j++) {
                double y = (YMAX-YMIN)*(j*1./NPOINTS)+YMIN;
                double z, dx, dy, dxx, dxy, dyy;
                fnc.evalDeriv(x, y, &z, numDeriv>=1 ? &dx : NULL, numDeriv>=1 ? &dy : NULL,
                    numDeriv>=2 ? &dxx : NULL, numDeriv>=2 ? &dxy : NULL, numDeriv>=2 ? &dyy : NULL);
            }
        }
    }
    return utils::pp(1.0 * iter * pow_2(NPOINTS+1) * CLOCKS_PER_SEC / (std::clock()-clk), 6);
}

bool test2dSpline()
{
    std::cout << "\033[1;33m2d interpolation\033[0m\n";
    bool ok=true;
    const int NNODESX=8;
    const int NNODESY=7;
    const int NN=100;    // number of intermediate points for checking the values
    const double XMIN = -1.9, XMAX = 2.2, YMIN = -2.1, YMAX = 1.7;
    std::vector<double> xval = math::createUniformGrid(NNODESX, XMIN, XMAX);
    std::vector<double> yval = math::createUniformGrid(NNODESY, YMIN, YMAX);
    math::Matrix<double>
        fval (NNODESX, NNODESY), fderx (NNODESX, NNODESY),
        fdery(NNODESX, NNODESY), fderxy(NNODESX, NNODESY);
    testfnc2d fnc;
    for(int i=0; i<NNODESX; i++)
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double der[8];
            fnc.evalDeriv(xy, &fval(i, j), der);
            fderx (i, j) = der[0];
            fdery (i, j) = der[1];
            fderxy(i, j) = der[3];
        }
    // 2d bilinear interpolator
    math::LinearInterpolator2d lin2d(xval, yval, fval);
    // 2d cubic spline
    math::CubicSpline2d cub2d(xval, yval, fval);
    // 2d quintic spline with prescribed derivatives at all nodes
    math::QuinticSpline2d qui2d(xval, yval, fval, fderx, fdery);
    // 2d quintic spline with prescribed 1st derivatives and mixed 2nd derivative at all nodes
    math::QuinticSpline2d mix2d(xval, yval, fval, fderx, fdery, fderxy);

    // check the values (for both cubic and quintic splines) and derivatives (for quintic spline only)
    // at all nodes of 2d grid -- should exactly equal the input values (with machine precision)
    for(int i=0; i<NNODESX; i++) {
        for(int j=0; j<NNODESY; j++) {
            double xy[2] = {xval[i], yval[j]};
            double f, fder[8], l, c, q, qder[8], cxy, qxy, qxx, qyy;
            fnc.evalDeriv(xy, &f, fder);
            l = lin2d .value(xy[0], xy[1]);
            cub2d.evalDeriv(xy[0], xy[1], &c, NULL, NULL, NULL, &cxy);
            qui2d.evalDeriv(xy[0], xy[1], &q, &qder[0], &qder[1], &qxx, &qxy, &qyy);
            if(l != f || c != f || q != f || fder[0] != qder[0] || fder[1] != qder[1])
                ok = false;
        }
    }
    if(!ok) std::cout << "Values or derivs at grid nodes are inconsistent\n";

    double
        nanl = lin2d.value(NAN, yval[0]),
        nanc = cub2d.value(NAN, yval[0]),
        nanq = qui2d.value(NAN, yval[0]);
    if(nanl==nanl || nanc==nanc || nanq==nanq) {
        std::cout << "2d splines do not return NAN when they should\n";
        ok = false;
    }

    std::ofstream strm;
    if(OUTPUT) { // output for Gnuplot splot routine
        strm.open("test_math_spline2d.dat");
        strm << "x y\tfunc fx fy fxx fxy fyy\tcubic cx cy cxx cxy cyy\tquintic qx qy qxx qxy qyy\n";
    }
    double sumerrl = 0, sumerrc = 0, sumerrq = 0, sumerrm = 0,
        sumerrcder = 0, sumerrqder = 0, sumerrmder = 0, sumerrcder2 = 0, sumerrqder2 = 0, sumerrmder2 = 0;
    for(int i=0; i<=NN; i++) {
        double x = (XMAX-XMIN)*(i*1./NN)+XMIN;
        for(int j=0; j<=NN; j++) {
            double y = (YMAX-YMIN)*(j*1./NN)+YMIN;
            double xy[2] = {x, y}, f, d[8];
            fnc.evalDeriv(xy, &f, d);
            double l, c, cx, cy, cxx, cxy, cyy, q, qx, qy, qxx, qxy, qyy, m, mx, my, mxx, mxy, myy;
            l =  lin2d.value(x, y);
            cub2d.evalDeriv(x, y, &c, &cx, &cy, &cxx, &cxy, &cyy);
            qui2d.evalDeriv(x, y, &q, &qx, &qy, &qxx, &qxy, &qyy);
            mix2d.evalDeriv(x, y, &m, &mx, &my, &mxx, &mxy, &myy);

            sumerrl     += pow_2(l-f);
            sumerrc     += pow_2(c-f);
            sumerrq     += pow_2(q-f);
            sumerrm     += pow_2(m-f);
            sumerrcder  += pow_2(cx-d[0]) + pow_2(cy-d[1]);
            sumerrqder  += pow_2(qx-d[0]) + pow_2(qy-d[1]);
            sumerrmder  += pow_2(mx-d[0]) + pow_2(my-d[1]);
            sumerrcder2 += pow_2(cxx-d[2]) + pow_2(cxy-d[3]) + pow_2(cyy-d[4]);
            sumerrqder2 += pow_2(qxx-d[2]) + pow_2(qxy-d[3]) + pow_2(qyy-d[4]);
            sumerrmder2 += pow_2(mxx-d[2]) + pow_2(mxy-d[3]) + pow_2(myy-d[4]);

            if(OUTPUT)
                strm << utils::pp(x, 7) + ' ' + utils::pp(y, 7) + '\t' +
                utils::pp(f, 10) + ' ' + utils::pp(d[0],8)+ ' ' + utils::pp(d[1],8)+ ' ' +
                utils::pp(d[2],7)+ ' ' + utils::pp(d[3],7)+ ' ' + utils::pp(d[4],7)+ '\t'+
                utils::pp(c, 10) + ' ' + utils::pp(cx, 8) + ' ' + utils::pp(cy, 8) + ' ' +
                utils::pp(cxx,7) + ' ' + utils::pp(cxy,7) + ' ' + utils::pp(cyy,7) + '\t'+
                utils::pp(q, 10) + ' ' + utils::pp(qx, 8) + ' ' + utils::pp(qy, 8) + ' ' +
                utils::pp(qxx,7) + ' ' + utils::pp(qxy,7) + ' ' + utils::pp(qyy,7) + '\t'+
                utils::pp(m, 10) + ' ' + utils::pp(mx, 8) + ' ' + utils::pp(my, 8) + ' ' +
                utils::pp(mxx,7) + ' ' + utils::pp(mxy,7) + ' ' + utils::pp(myy,7) + '\n';
        }
        if(OUTPUT)
            strm << "\n";
    }
    if(OUTPUT)
        strm.close();
    sumerrl     = sqrt(sumerrl)       / (NN+1);
    sumerrc     = sqrt(sumerrc)       / (NN+1);
    sumerrq     = sqrt(sumerrq)       / (NN+1);
    sumerrm     = sqrt(sumerrm)       / (NN+1);
    sumerrcder  = sqrt(sumerrcder /2) / (NN+1);
    sumerrqder  = sqrt(sumerrqder /2) / (NN+1);
    sumerrmder  = sqrt(sumerrmder /2) / (NN+1);
    sumerrcder2 = sqrt(sumerrcder2/3) / (NN+1);
    sumerrqder2 = sqrt(sumerrqder2/3) / (NN+1);
    sumerrmder2 = sqrt(sumerrmder2/3) / (NN+1);
    std::cout << "RMS error in linear 2d interpolator: " + utils::pp(sumerrl, 8) +
        "\ncubic   2d spline value:  " + utils::pp(sumerrc, 8) +
        ", deriv: " + utils::pp(sumerrcder, 8) + ", 2nd deriv: " + utils::pp(sumerrcder2, 8) +
        "\nquintic 2d spline value:  " + utils::pp(sumerrq, 8) +
        ", deriv: " + utils::pp(sumerrqder, 8) + ", 2nd deriv: " + utils::pp(sumerrqder2, 8) +
        "\nq-mixed 2d spline value:  " + utils::pp(sumerrm, 8) +
        ", deriv: " + utils::pp(sumerrmder, 8) + ", 2nd deriv: " + utils::pp(sumerrmder2, 8) +"\n";
    ok &= sumerrl < 0.03 &&
        sumerrc < 0.003 && sumerrcder < 0.012 && sumerrcder2 < 0.075 &&
        sumerrq < 1.e-4 && sumerrqder < 7.e-4 && sumerrqder2 < 0.006 &&
        sumerrm < 6.e-5 && sumerrmder < 3.e-4 && sumerrmder2 < 0.003;

    //----------- test the performance of 2d spline calculation -------------//
    std::cout <<"Linear interpolator:      " + evalSpline2d<0>(lin2d) + " eval/s\n";
    std::cout <<"Cubic   spline w/o deriv: " + evalSpline2d<0>(cub2d) + ", 1st deriv: " +
    evalSpline2d<1>(cub2d) + ", 2nd deriv: " + evalSpline2d<2>(cub2d) + " eval/s\n";
    std::cout <<"Quintic spline w/o deriv: " + evalSpline2d<0>(qui2d) + ", 1st deriv: " +
    evalSpline2d<1>(qui2d) + ", 2nd deriv: " + evalSpline2d<2>(qui2d) + " eval/s\n";
    return ok;
}

//----------- test 3d interpolation ------------//
bool test3dSpline()
{
    std::cout << "\033[1;33m3d interpolation\033[0m\n";
    bool ok=true;
    testfnc3d fnc3d;
    const int NNODESX=10, NNODESY=8, NNODESZ=7;
    std::vector<double>
    xval=math::createUniformGrid(NNODESX, 0, 6.283185),
    yval=math::createUniformGrid(NNODESY, 0, 2.718282),
    zval=math::createUniformGrid(NNODESZ, 0, 5.012345);
    // B-spline interpolators of degree 1 and 3 are constructed by computing the amplitudes;
    // for the N=1 B-spline these are just the function values at grid points,
    // but for the N=3 one we must solve a (sparse) linear system
    std::vector<double> lval3d(math::createBsplineInterpolator3dArray<1>(fnc3d, xval, yval, zval));
    std::vector<double> cval3d(math::createBsplineInterpolator3dArray<3>(fnc3d, xval, yval, zval));
    math::BsplineInterpolator3d<1> blin3d(xval, yval, zval);
    math::BsplineInterpolator3d<3> bcub3d(xval, yval, zval);
    // Linear interpolator constructed from the values of function at grid points
    math::LinearInterpolator3d lin3d(xval, yval, zval, lval3d);
    // Cubic spline in 3d constructed from the values of function at grid points
    math::CubicSpline3d        spl3d(xval, yval, zval, lval3d);
    // 3d cubic spline constructed from the amplitudes of a 3d B-spline of degree N=3
    math::CubicSpline3d        spc3d(xval, yval, zval, cval3d);
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
                double ll = lin3d.value(point[0], point[1], point[2]);
                double sl = spl3d.value(point[0], point[1], point[2]);
                double sc = spc3d.value(point[0], point[1], point[2]);
                double bl = blin3d.interpolate(point, lval3d);
                double bc = bcub3d.interpolate(point, cval3d);
                ok &= v == ll && v == sl && v == bl;  // this should be exact
                ok &= math::fcmp(v, sc, 1e-13)==0;
                ok &= math::fcmp(v, bc, 1e-13)==0;
            }
        }
    }
    if(!ok) std::cout << "Values or derivs at grid nodes are inconsistent\n";

    point[0] = NAN;
    double
        nanl = lin3d.value(NAN, yval[0], zval[0]),
        nanc = spl3d.value(NAN, yval[0], zval[0]),
        nanb = bcub3d.interpolate(point, cval3d);
    if(nanl==nanl || nanc==nanc || nanb==nanb) {
        std::cout << "3d splines do not return NAN when they should\n";
        ok = false;
    }

    // test accuracy of approximation
    double sumsqerr_l=0, sumsqerr_c=0, sumsqerr_s=0;
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
                double ll = lin3d.value(point[0], point[1], point[2]);
                double sl = spl3d.value(point[0], point[1], point[2]);
                double sc = spc3d.value(point[0], point[1], point[2]);
                double bl = blin3d.interpolate(point, lval3d);
                double bc = bcub3d.interpolate(point, cval3d);
                sumsqerr_l += pow_2(v-bl);
                sumsqerr_c += pow_2(v-bc);
                sumsqerr_s += pow_2(sl-bc) + pow_2(sc-bc) + pow_2(ll-bl);
                if(OUTPUT)
                    strm << point[0] << ' ' << point[1] << ' ' << point[2] << '\t' <<
                    v << ' ' << bl << ' ' << bc << "\n";
            }
            if(OUTPUT)
                strm << "\n";
        }
        if(OUTPUT)
            strm << "\n";
    }
    if(OUTPUT)
        strm.close();
    sumsqerr_l = sqrt(sumsqerr_l / pow_3(NNN+1));
    sumsqerr_c = sqrt(sumsqerr_c / pow_3(NNN+1));
    sumsqerr_s = sqrt(sumsqerr_s / pow_3(NNN+1));
    std::cout << "RMS error in linear 3d interpolator: " << utils::pp(sumsqerr_l, 8) << 
        ", cubic 3d interpolator: " << utils::pp(sumsqerr_c, 8) <<
        ", difference between cubic spline and B-spline: " << utils::pp(sumsqerr_s, 8) << "\n";
    ok &= sumsqerr_l<0.04 && sumsqerr_c<0.01 && sumsqerr_s<1e-15;

    // test performance of various interpolators
    double RATE = 1.0 * pow_3(NNN+1) * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    int iter=0;
    for(; iter%10!=9 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    double s;
                    lin3d.eval(point, &s);
                }
            }
        }
    }
    std::cout << "Linear interpolator:    " + utils::pp(RATE*iter/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    iter=0;
    for(; iter%10!=9 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    double s;
                    spl3d.eval(point, &s);
                }
            }
        }
    }
    std::cout << "Cubic spline:           " + utils::pp(RATE*iter/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    iter=0;
    for(; iter%10!=9 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    blin3d.interpolate(point, lval3d);
                }
            }
        }
    }
    std::cout << "B-spline of degree N=1: " + utils::pp(RATE*iter/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    iter=0;
    for(; iter%10!=9 || std::clock()-clk < MIN_CLOCKS; iter++) {
        for(int i=0; i<=NNN; i++) {
            point[0] = i*xval.back()/NNN;
            for(int j=0; j<=NNN; j++) {
                point[1] = j*yval.back()/NNN;
                for(int k=0; k<=NNN; k++) {
                    point[2] = k*zval.back()/NNN;
                    bcub3d.interpolate(point, cval3d);
                }
            }
        }
    }
    std::cout << "B-spline of degree N=3: " + utils::pp(RATE*iter/(std::clock()-clk), 6) + " eval/s\n";

    return ok;
}

//----------- test 1d/2d/3d monotonic cubic splines ------------//
bool testMonoSpline()
{
    std::cout << "\033[1;33mMonotonic interpolation\033[0m\n";
    bool okpos1=true, okpos2=true, okpos3=true, okerr1=true, okerr2=true, okerr3=true;
    testfncNd<1> f1;
    testfncNd<2> f2;
    testfncNd<3> f3;
    const double XMIN=-1.8, XMAX=2.2, YMIN=-1.6, YMAX=2.4, ZMIN=-1.5, ZMAX=2.0;
    const int Mmin=6, Mmax=10;  // number of points in interpolation grids
    const int L=40;             // number of test points
    math::Matrix<double> result1(L,     (Mmax-Mmin+1)*2+2);  // results for the 1d case
    math::Matrix<double> result2(L*L,   (Mmax-Mmin+1)*2+3);  // -"- for the 2d case
    math::Matrix<double> result3(L*L*L, (Mmax-Mmin+1)*2+4);  // -"- 3d
    for(int M=Mmin; M<=Mmax; M++) {
        // define the grids in 1,2 or 3 dimensions
        std::vector<double>
        xval=math::createUniformGrid(M, XMIN, XMAX),
        yval=math::createUniformGrid(M, YMIN, YMAX),
        zval=math::createUniformGrid(M, ZMIN, ZMAX),
        fval1d(M),
        fval2d(M*M),
        fval3d(M*M*M);
        math::MatrixView<double> fmat2d(M, M, &fval2d.front());
        // fill the input arrays for interpolation with values at grid nodes
        for(int i=0; i<M; i++) {
            fval1d[i] = f1(xval[i]);
            for(int j=0; j<M; j++) {
                double xyz[3] = {xval[i], yval[j], 0};
                f2.eval(xyz, &fval2d[ i * M + j ]);
                for(int k=0; k<M; k++) {
                    xyz[2] = zval[k];
                    f3.eval(xyz, &fval3d[ (i * M + j) * M + k ]);
                }
            }
        }
        // construct regularized and non-regularized splines
        math::CubicSpline c1(xval, fval1d, false/*no regularizing filter*/);
        math::CubicSpline r1(xval, fval1d, true/*use regularizing filter*/);
        math::CubicSpline2d c2(xval, yval, fmat2d, false);
        math::CubicSpline2d r2(xval, yval, fmat2d, true);
        math::CubicSpline3d c3(xval, yval, zval, fval3d, false);
        math::CubicSpline3d r3(xval, yval, zval, fval3d, true);
        // test the approximation accuracy by comparing original and interpolated values on a finer grid
        double ec1=0, er1=0, ec2=0, er2=0, ec3=0, er3=0;
        for(int i=0; i<L; i++) {
            double xyz[3] = { (XMIN*(L-0.5-i) + XMAX*(0.5+i)) / L, 0, 0};
            double f1o, f1c, f1r;
            f1o = f1(xyz[0]);
            f1c = c1(xyz[0]);
            f1r = r1(xyz[0]);
            ec1 += pow_2(f1c-f1o);
            er1 += pow_2(f1r-f1o);
            okpos1 &= f1r>=0;
            result1(i, 0) = xyz[0];
            result1(i, 1) = f1o;
            result1(i, (M-Mmin)*2+2) = f1c;
            result1(i, (M-Mmin)*2+3) = f1r;
            for(int j=0; j<L; j++) {
                xyz[1] = (YMIN*(L-0.5-j) + YMAX*(0.5+j)) / L;
                double f2o, f2c, f2r;
                f2.eval(xyz, &f2o);
                c2.eval(xyz, &f2c);
                r2.eval(xyz, &f2r);
                ec2 += pow_2(f2c-f2o);
                er2 += pow_2(f2r-f2o);
                okpos2 &= f2r>=0;
                result2(i*L+j, 0) = xyz[0];
                result2(i*L+j, 1) = xyz[1];
                result2(i*L+j, 2) = f2o;
                result2(i*L+j, (M-Mmin)*2+3) = f2c;
                result2(i*L+j, (M-Mmin)*2+4) = f2r;
                for(int k=0; k<L; k++) {
                    xyz[2] = (ZMIN*(L-0.5-k) + ZMAX*(0.5+k)) / L;
                    double f3o, f3c, f3r;
                    f3.eval(xyz, &f3o);
                    c3.eval(xyz, &f3c);
                    r3.eval(xyz, &f3r);
                    ec3 += pow_2(f3c-f3o);
                    er3 += pow_2(f3r-f3o);
                    okpos3 &= f3r>=0;
                    result3((i*L+j)*L+k, 0) = xyz[0];
                    result3((i*L+j)*L+k, 1) = xyz[1];
                    result3((i*L+j)*L+k, 2) = xyz[2];
                    result3((i*L+j)*L+k, 3) = f3o;
                    result3((i*L+j)*L+k, (M-Mmin)*2+4) = f3c;
                    result3((i*L+j)*L+k, (M-Mmin)*2+5) = f3r;
                }
            }
        }
        double norm = 1./M / (1 + pow(0.1*M, 4));  // characteristic magnitude of the rms error
        std::cout << "rms error for "   << M << " grid points:" <<
        "  1d cubic: " << testLess(sqrt(ec1/L)  , norm*1.2, okerr1) <<
        ", 1d reg: "   << testLess(sqrt(er1/L)  , norm*0.9, okerr1) <<
        ", 2d cubic: " << testLess(sqrt(ec2)/L  , norm*1.1, okerr2) <<
        ", 2d reg: "   << testLess(sqrt(er2)/L  , norm*0.8, okerr1) <<
        ", 3d cubic: " << testLess(sqrt(ec3/L)/L, norm*0.8, okerr2) <<
        ", 3d reg: "   << testLess(sqrt(er3/L)/L, norm*0.6, okerr3) << "\n";
    }
    if(OUTPUT) {
        std::ofstream strm("test_math_spline1d_mono.dat");
        for(int i=0; i<L; i++) {
            for(int c=0, C=result1.cols(); c<C; c++)
                strm << result1(i,c) << (c<C-1?'\t':'\n');
        }
        strm.close();
        strm.open("test_math_spline2d_mono.dat");
        for(int i=0; i<L; i++) {
            for(int j=0; j<L; j++) {
                for(int c=0, C=result2.cols(); c<C; c++)
                    strm << result2(i*L+j,c) << (c<C-1?'\t':'\n');
            }
            strm << '\n';
        }
        strm.close();
        strm.open("test_math_spline3d_mono.dat");
        for(int i=0; i<L; i++) {
            for(int j=0; j<L; j++) {
                for(int k=0; k<L; k++) {
                    for(int c=0, C=result3.cols(); c<C; c++)
                        strm << result3((i*L+j)*L+k,c) << (c<C-1?'\t':'\n');
                }
                strm << '\n';
            }
            strm << '\n';
        }
    }
    bool ok =
    testCond(okpos1, "1d monotonic spline is negative") &&
    testCond(okpos2, "2d monotonic spline is negative") &&
    testCond(okpos3, "3d monotonic spline is negative") &&
    testCond(okerr1, "1d spline interpolation error is too large") &&
    testCond(okerr2, "2d spline interpolation error is too large") &&
    testCond(okerr3, "3d spline interpolation error is too large");
    return ok;
}

bool printFail(const char* msg)
{
    std::cout << "\033[1;31m " << msg << " failed\033[0m\n";
    return (msg==0);  // false
}

int main()
{
    bool ok=true;
    ok &= testPenalizedSplineFit() || printFail("Penalized spline fit");
    ok &= testPenalizedSplineDensity() || printFail("Penalized spline density estimator");
    ok &= testMonoSpline() || printFail("Monotonic spline interpolation");
    ok &= test1dSpline() || printFail("1d spline");
    ok &= test2dSpline() || printFail("2d spline");
    ok &= test3dSpline() || printFail("3d spline");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
