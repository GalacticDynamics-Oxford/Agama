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
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cassert>
#ifdef COMPARE_WD_PSPLINE
#include "torus/WD_Pspline.h"
#endif

const bool OUTPUT = utils::verbosityLevel >= utils::VL_VERBOSE;

bool testCond(bool condition, const char* errorMessage)
{
    if(!condition)
        std::cout << errorMessage << '\n';
    return condition;
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
    std::cout << "case A: RMS=" << rms<<", EDF="<<edf<<"\n";
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

// test the integration of a spline function
bool test_integral(const math::CubicSpline& f, double x1, double x2)
{
    double result_int = f.integrate(x1, x2);
    double result_ext = math::integrateAdaptive(f, x1, x2, 1e-13);
    double error1 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Ordinary intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error1,8) + '\n';
    result_int = f.integrate(x1, x2, testfncsin());
    result_ext = math::integrateAdaptive(testfncintsin(f), x1, x2, 1e-13);
    double error2 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Weighted intergral on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error2,8) + '\n';
    result_int = f.integrate(x1, x2, f);
    result_ext = math::integrateAdaptive(squaredfnc(f), x1, x2, 1e-13);
    double error3 = fabs((result_int-result_ext) / result_ext);
    std::cout << "Integral of f(x)^2 on [" + utils::pp(x1,10) +':'+ utils::pp(x2,10) +
        "]: result=" + utils::pp(result_int,8) + ", error=" + utils::pp(error3,8) + '\n';
    return error1 < 1e-13 && error2 < 1e-13 && error3 < 3e-12;
    // the large error in the last case is apparently due to roundoff errors
}

bool testLogScaledSplines()
{
    // function to approximate: f(x) = cos(x-1) + x, x=1..Pi+1
    const int NNODES=8;
    std::vector<double> xnodes = math::createUniformGrid(NNODES, 1, 1+M_PI);
    std::vector<double> yvalues(NNODES);
    for(int i=0; i<NNODES; i++)
        yvalues[i] = cos(xnodes[i]-1) + xnodes[i];
    double derLeft = 1., derRight = 1.;
    math::CubicSpline  fSpl(xnodes, yvalues, derLeft, derRight);
    math::LogSpline    fLog(xnodes, yvalues, derLeft, derRight);
    math::LogLogSpline fDbl(xnodes, yvalues, derLeft, derRight);
    double errValS = 0, errValL = 0, errValD = 0, errDerS = 0, errDerL = 0, errDerD = 0,
        errDer2S = 0, errDer2L = 0, errDer2D = 0;
    for(double x=1.; x<1.+M_PI; x+=0.01) {
        double fval = cos(x-1) + x, fder = 1 - sin(x-1), fder2 = -cos(x-1);
        double sval, sder, sder2, lval, lder, lder2, dval, dder, dder2;
        fSpl.evalDeriv(x, &sval, &sder, &sder2);
        fLog.evalDeriv(x, &lval, &lder, &lder2);
        fDbl.evalDeriv(x, &dval, &dder, &dder2);
        errValS = fmax(errValS, fabs(sval - fval));
        errValL = fmax(errValL, fabs(lval - fval));
        errValD = fmax(errValD, fabs(dval - fval));
        errDerS = fmax(errDerS, fabs(sder - fder));
        errDerL = fmax(errDerL, fabs(lder - fder));
        errDerD = fmax(errDerD, fabs(dder - fder));
        errDer2S= fmax(errDer2S,fabs(sder2- fder2));
        errDer2L= fmax(errDer2L,fabs(lder2- fder2));
        errDer2D= fmax(errDer2D,fabs(dder2- fder2));
    }
    return
        errValS < 2e-4 && errValL < 4e-4 && errValD < 3e-4 &&
        errDerS < 1e-3 && errDerL < 3e-3 && errDerD < 3e-3 &&
        errDer2S< 2e-2 && errDer2L< 5e-2 && errDer2D< 4e-2;
}

bool test1dSpline()
{
    const int NNODES  = 20;
    const int NSUBINT = 16;
    const double XMIN = 0.2;
    const double XMAX = 12.0123456;
    testfnc1d fnc;   // the original function that we are approximating
    std::vector<double> yvalues(NNODES), yderivs(NNODES);
    std::vector<double> xnodes = math::createNonuniformGrid(NNODES, XMIN, XMAX, false);
    xnodes[1]=(xnodes[1]+xnodes[2])/2;  // slightly squeeze grid spacing to allow
    xnodes[0]*=2;                       // a better interpolation of a strongly varying function
    for(int i=0; i<NNODES; i++) {
        fnc.evalDeriv(xnodes[i], &yvalues[i], &yderivs[i]);
    }

    // a collection of approximating splines of various types:

    // 1. cubic spline with natural boundary conditions
    math::CubicSpline   fNatural(xnodes, yvalues);

    // 2. cubic, clamped -- specify derivs at the boundaries
    math::CubicSpline   fClamped(xnodes, yvalues, yderivs.front(), yderivs.back());

    // 3. hermite cubic spline -- specify derivs at all nodes
    math::HermiteSpline fHermite(xnodes, yvalues, yderivs);

    // 4. a hermite spline constructed from an ordinary cubic spline,
    // by collecting the first derivatives at all grid nodes - should be equivalent
    std::vector<double> yderivsClamped(NNODES);
    for(int i=0; i<NNODES; i++)
        fClamped.evalDeriv(xnodes[i], NULL, &yderivsClamped[i]);
    math::HermiteSpline fHermiteEquivClamped(xnodes, yvalues, yderivsClamped);

    // 5. quintic spline -- specify derivs at all nodes
    math::QuinticSpline fQuintic(xnodes, yvalues, yderivs);

    // 6-7: 1d cubic B-spline represented as a sum over basis functions;
    // this class only stores the x-grid and presents a method for computing the interpolant
    // for any array of amplitudes passed as a parameter -- we will use two different ones:
    math::BsplineInterpolator1d<3> fBspline(xnodes);

    // 6. compute the amplitudes from another cubic spline, which results in an equivalent
    // representation in terms of B-splines
    std::vector<double> amplBsplineEquivClamped =
        math::createBsplineInterpolator1dArray<3>(fClamped, xnodes);

    // 7. compute the amplitudes from the original function in a different way:
    // instead of using just from the function values and possibly the derivatives at the grid
    // nodes, as in all previous cases, it collects the function values at several points
    // inside each grid segment, thus achieving a somewhat better overall approximation
    std::vector<double> amplBsplineOrig = math::createBsplineInterpolator1dArray<3>(fnc, xnodes);

    // 8. 1d clamped cubic spline equivalent to a B-spline (this demonstrates the equivalence
    // in both directions, as the amplitudes of this B-spline were computed from another cubic spline).
    // The array of amplitudes can be simply passed to the constructor of CubicSpline instead of
    // an array of function values -- the mode of operation is determined from the size of this array;
    // if it is equal to the number of x-nodes, it refers to the values of the function at these nodes;
    // if it is longer by two elements, it refers to the B-spline amplitudes.
    math::CubicSpline fClampedEquivBspline(xnodes, amplBsplineEquivClamped);

    std::ofstream strm;

    // accumulators for computing rms errors in values, derivatives and second derivatives
    // of these approximations
    double errClampedVal=0,  errHermiteVal=0,  errQuinticVal=0, errNaturalVal=0, errBsplineVal=0,
           errClampedDer=0,  errHermiteDer=0,  errQuinticDer=0,
           errClampedDer2=0, errHermiteDer2=0, errQuinticDer2=0,
           errBsplineEquivClamped=0, errClampedEquivBspline=0, errHermiteEquivClamped=0;
    bool oknat = true, okcla = true, okher = true, okqui = true, okhcl = true;

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

        // 1. natural cubic spline (only collect the value)
        double fNaturalVal = fNatural(x);

        // 2. clamped cubic spline
        double fClampedVal, fClampedDer, fClampedDer2;
        fClamped.evalDeriv(x, &fClampedVal, &fClampedDer, &fClampedDer2);

        // 3. hermite spline constructed from the original function
        double fHermiteVal, fHermiteDer, fHermiteDer2;
        fHermite.evalDeriv(x, &fHermiteVal, &fHermiteDer, &fHermiteDer2);

        // 4. hermite spline equivalent to the clamped cubic spline
        double fHermiteEquivClampedVal, fHermiteEquivClampedDer, fHermiteEquivClampedDer2;
        fHermiteEquivClamped.evalDeriv(x, &fHermiteEquivClampedVal,
            &fHermiteEquivClampedDer, &fHermiteEquivClampedDer2);

        // 5. quintic spline
        double fQuinticVal, fQuinticDer, fQuinticDer2, fQuinticDer3=fQuintic.deriv3(x);
        fQuintic.evalDeriv(x, &fQuinticVal, &fQuinticDer, &fQuinticDer2);

        // 6. b-spline equivalent to the clamped cubic spline
        double fBsplineEquivClampedVal = fBspline.interpolate(x, amplBsplineEquivClamped);

        // 7. b-spline constructed from the original function
        double fBsplineVal = fBspline.interpolate(x, amplBsplineOrig);

        // 8. clamped cubic spline equivalent to the b-spline
        double fClampedEquivBsplineVal = fClampedEquivBspline(x);

        // accumulate errors of various approximations
        errNaturalVal += pow_2(origVal - fNaturalVal);
        errClampedVal += pow_2(origVal - fClampedVal);
        errHermiteVal += pow_2(origVal - fHermiteVal);
        errBsplineVal += pow_2(origVal - fBsplineVal);
        errQuinticVal += pow_2(origVal - fQuinticVal);
        errClampedDer += pow_2(origDer - fClampedDer);
        errHermiteDer += pow_2(origDer - fHermiteDer);
        errQuinticDer += pow_2(origDer - fQuinticDer);
        errClampedDer2+= pow_2(origDer2- fClampedDer2);
        errHermiteDer2+= pow_2(origDer2- fHermiteDer2);
        errQuinticDer2+= pow_2(origDer2- fQuinticDer2);

        // keep track of error in equivalent representations of the same interpolant
        errBsplineEquivClamped = fmax(errBsplineEquivClamped,
            fabs(fBsplineEquivClampedVal - fClampedVal));
        errClampedEquivBspline = fmax(errClampedEquivBspline,
            fabs(fBsplineEquivClampedVal - fClampedEquivBsplineVal));
        errHermiteEquivClamped = fmax(errHermiteEquivClamped,
            fabs(fHermiteEquivClampedVal - fClampedVal) +
            fabs(fHermiteEquivClampedDer - fClampedDer) +
            fabs(fHermiteEquivClampedDer2- fClampedDer2));

        // if x coincides with a grid node, the values of interpolants should be
        // exact to machine precision by construction
        if(i%NSUBINT == 0) {
            int k = i/NSUBINT;
            oknat &= fNaturalVal == yvalues[k];
            okcla &= fClampedVal == yvalues[k];
            okher &= fHermiteVal == yvalues[k] && fHermiteDer == yderivs[k];
            okqui &= fQuinticVal == yvalues[k] && fQuinticDer == yderivs[k];
            okhcl &= fHermiteEquivClampedVal == fClampedVal;
        }

        // dump the values to a file if requested
        if(OUTPUT) {
            if(i==0) {
                strm.open("test_math_spline1d.dat");
                strm << "x\torig:f f' f'' f'''\tbspline:f\tnatural:f\t"
                    "clamped:f f' f''\thermite:f f' f''\tquintic:f f' f'' f'''\n";
            }
            strm << utils::pp(x,   7) +'\t'+
            utils::pp(origVal,     7) +' ' +
            utils::pp(origDer,     7) +' ' +
            utils::pp(origDer2,    7) +' ' +
            utils::pp(origDer3,    7) +'\t'+
            utils::pp(fNaturalVal, 7) +'\t'+
            utils::pp(fBsplineVal, 7) +'\t'+
            utils::pp(fClampedVal, 7) +' ' +
            utils::pp(fClampedDer, 7) +' ' +
            utils::pp(fClampedDer2,7) +'\t'+
            utils::pp(fHermiteVal, 7) +' ' +
            utils::pp(fHermiteDer, 7) +' ' +
            utils::pp(fHermiteDer2,7) +'\t'+
            utils::pp(fQuinticVal, 7) +' ' +
            utils::pp(fQuinticDer, 7) +' ' +
            utils::pp(fQuinticDer2,7) +'\t'+
            utils::pp(fQuinticDer3,7) +'\n';
            if(i==NPOINTS)
                strm.close();
        }
    }
    errNaturalVal  = sqrt(errNaturalVal / NPOINTS);
    errBsplineVal  = sqrt(errBsplineVal / NPOINTS);
    errClampedVal  = sqrt(errClampedVal / NPOINTS);
    errHermiteVal  = sqrt(errHermiteVal / NPOINTS);
    errQuinticVal  = sqrt(errQuinticVal / NPOINTS);
    errClampedDer  = sqrt(errClampedDer / NPOINTS);
    errHermiteDer  = sqrt(errHermiteDer / NPOINTS);
    errQuinticDer  = sqrt(errQuinticDer / NPOINTS);
    errClampedDer2 = sqrt(errClampedDer2/ NPOINTS);
    errHermiteDer2 = sqrt(errHermiteDer2/ NPOINTS);
    errQuinticDer2 = sqrt(errQuinticDer2/ NPOINTS);

    std::cout << "RMS error in ordinary cubic spline: " << utils::pp(errNaturalVal, 8) <<
        ", in clamped cubic spline: " << utils::pp(errClampedVal, 8) <<
        ", in hermite cubic spline: " << utils::pp(errHermiteVal, 8) <<
        ", in cubic B-spline: " << utils::pp(errBsplineVal, 8) <<
        ", in quintic spline: " << utils::pp(errQuinticVal, 8) << 
        ", max|cubic-hermite|=" << utils::pp(errHermiteEquivClamped, 8) << 
        ", max|cubic-bspline|=" << utils::pp(errBsplineEquivClamped, 8) <<
        " and " << utils::pp(errClampedEquivBspline, 8) << "\n";

    // test the integration functions //
    const double X1 = (xnodes[0]+xnodes[1])/2, X2 = xnodes[xnodes.size()-2]-0.1;
    bool okintcla = test_integral(fClamped, X1, X2);
    bool okintnat = test_integral(fNatural, -1.234567, xnodes.back()+1.);
    double intClamped = fClamped.integrate(X1, X2);
    double intBspline = fBspline.integrate(X1, X2, amplBsplineEquivClamped);
    double intNumeric = math::integrateAdaptive(fClamped, X1, X2, 1e-14);
    bool okintnum = fabs(intClamped-intNumeric) < 1e-13 && fabs(intClamped-intBspline) < 1e-13;

    // test log-scaled splines
    bool oklogspl = testLogScaledSplines();

    return
    testCond(oknat, "natural cubic spline values at grid nodes are inexact") &&
    testCond(okcla, "clamped cubic spline values at grid nodes are inexact") &&
    testCond(oknat, "hermite cubic spline values or derivatives at grid nodes are inexact") &&
    testCond(oknat, "quintic spline values or derivatives at grid nodes are inexact") &&
    testCond(oknat, "hermite and clamped cubic spline values are not equal") &&
    testCond(errHermiteEquivClamped < 1e-13, "hermite<>clamped") &&
    testCond(errBsplineEquivClamped < 1e-15, "bspline<>clamped") &&
    testCond(errClampedEquivBspline < 1e-15, "clamped<>bspline") &&
    testCond(errNaturalVal < 5.0e-3, "error in natural cubic spline is too large") &&
    testCond(errBsplineVal < 1.4e-4, "error in cubic b-spline is too large") &&
    testCond(errClampedVal < 2.7e-4, "error in clamped cubic spline is too large") &&
    testCond(errHermiteVal < 2.3e-4, "error in hermite cubic spline is too large") &&
    testCond(errQuinticVal < 3.4e-5, "error in quintic spline is too large") &&
    testCond(errClampedDer < 1.8e-3, "error in clamped cubic spline derivative is too large") &&
    testCond(errHermiteDer < 1.3e-3, "error in hermite cubic spline derivative is too large") &&
    testCond(errQuinticDer < 9.0e-4, "error in quintic spline derivative is too large") &&
    testCond(errClampedDer2< 0.04, "error in clamped cubic spline 2nd derivative is too large")   &&
    testCond(errHermiteDer2< 0.04, "error in hermite cubic spline 2nd derivative is too large")   &&
    testCond(errQuinticDer2< 0.05, "error in quintic spline 2nd derivative is too large") &&
    testCond(okintcla, "integral of clamped spline is incorrect") &&
    testCond(okintnat, "integral of natural spline is incorrect") &&
    testCond(okintnum, "integral of B-spline is incorrect") &&
    testCond(oklogspl, "log-scaled splines failed");
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
    // 2d bilinear interpolator
    math::LinearInterpolator2d lin2d(xval, yval, fval);
    // 2d cubic spline with natural boundary conditions at three edges
    // and a prescribed derivative at the fourth edge
    math::CubicSpline2d spl2dc(xval, yval, fval, 0, NAN, NAN, NAN);
    // 2d quintic spline with prescribed derivatives at all nodes
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
            double f, fder[2], l, c, q, qder[2];
            fnc.evalDeriv(xy, &f, fder);
            l = lin2d .value(xy[0], xy[1]);
            c = spl2dc.value(xy[0], xy[1]);
            spl2dq.evalDeriv(xy[0], xy[1], &q, &qder[0], &qder[1]);
            if(l != f || c != f || q != f || fder[0] != qder[0] || fder[1] != qder[1])
                ok = false;
        }

    std::ofstream strm;
    if(OUTPUT)  // output for Gnuplot splot routine
        strm.open("test_math_spline2d.dat");
    double maxerrl = 0, maxerrc = 0, maxerrq = 0, maxerrcder = 0, maxerrqder = 0;
    for(int i=0; i<=NN; i++) {
        double x = XMAX*(i*1./NN);
        for(int j=0; j<=NN; j++) {
            double y = YMAX*(j*1./NN);

            double xy[2] = {x, y}, f, fder[2];
            fnc.evalDeriv(xy, &f, fder);
            double l, c, cx, cy, cxx, cxy, cyy, q, qx, qy, qxx, qxy, qyy;
            l =  lin2d.value(x, y);
            spl2dc.evalDeriv(x, y, &c, &cx, &cy, &cxx, &cxy, &cyy);
            spl2dq.evalDeriv(x, y, &q, &qx, &qy, &qxx, &qxy, &qyy);
            maxerrl = fmax(maxerrl, fabs(l-f));
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
                    f << ' ' << fder[0] << ' ' << fder[1] << '\t' << l << '\t' <<
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

    std::cout << "Max error in linear 2d interpolator: " + utils::pp(maxerrl, 8) +
        ", cubic 2d spline value: " + utils::pp(maxerrc, 8) +
        ", deriv: " + utils::pp(maxerrcder, 8) +
        ", quintic 2d spline value: " + utils::pp(maxerrq, 8) +
        ", deriv: " + utils::pp(maxerrqder, 8) + "\n";
    ok &= maxerrc < 0.006 && maxerrcder < 0.09 && maxerrq < 0.002 && maxerrqder < 0.03;

#ifdef STRESS_TEST
    //----------- test the performance of 2d spline calculation -------------//
    double z, dx, dy, dxx, dxy, dyy;
    int NUM_ITER = 1000;
    double RATE = 1.0 * NUM_ITER * pow_2(NN+1) * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                lin2d.evalDeriv(x, y, &z);
            }
        }
    }
    std::cout << "Linear interpolator:          " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";
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
    std::cout << "Cubic spline with no deriv:   " + utils::pp(RATE/(std::clock()-clk), 6);
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
    std::cout << ", 1st deriv: " + utils::pp(RATE/(std::clock()-clk), 6);
    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
        for(int i=0; i<=NN; i++) {
            double x = XMAX*(i*1./NN);
            for(int j=0; j<=NN; j++) {
                double y = YMAX*(j*1./NN);
                spl2dc.evalDeriv(x, y, &z, &dx, &dy, &dxx, &dxy, &dyy);
            }
        }
    }
    std::cout << ", 2nd deriv: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

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
    std::cout << "Quintic spline without deriv: " + utils::pp(RATE/(std::clock()-clk), 6);
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
    std::cout << ", 1st deriv: " + utils::pp(RATE/(std::clock()-clk), 6);
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
    std::cout << ", 2nd deriv: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";
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
                WD::Psplev2D(WD_X, WD_Y, WD_Z, WD_K, wx);
            }
        }
    }
    std::cout << "WD's Pspline with no deriv: " << utils::pp(RATE/(std::clock()-clk), 6);
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
    std::cout << ", 1st deriv: " << utils::pp(RATE/(std::clock()-clk), 6);
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
    std::cout << ", 2nd deriv: " << utils::pp(RATE/(std::clock()-clk), 6) << " eval/s\n";
#endif
#endif
    return ok;
}

//----------- test 3d interpolation ------------//
bool test3dSpline()
{
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
    }
    if(OUTPUT)
        strm.close();
    sumsqerr_l = sqrt(sumsqerr_l / pow_3(NNN+1));
    sumsqerr_c = sqrt(sumsqerr_c / pow_3(NNN+1));
    sumsqerr_s = sqrt(sumsqerr_s / pow_3(NNN+1));
    std::cout << "RMS error in linear 3d interpolator: " << utils::pp(sumsqerr_l, 8) << 
        ", cubic 3d interpolator: " << utils::pp(sumsqerr_c, 8) <<
        ", difference between cubic spline and B-spline: " << utils::pp(sumsqerr_s, 8) << "\n";
    ok &= sumsqerr_l<0.1 && sumsqerr_c<0.05 && sumsqerr_s<1e-15;

#ifdef STRESS_TEST
    // test performance of various interpolators
    int NUM_ITER = 100;
    double RATE = 1.0 * NUM_ITER * pow_3(NNN+1) * CLOCKS_PER_SEC;
    clock_t clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
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
    std::cout << "Linear interpolator:    " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
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
    std::cout << "Cubic spline:           " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
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
    std::cout << "B-spline of degree N=1: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";

    clk = std::clock();
    for(int t=0; t<NUM_ITER; t++) {
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
    std::cout << "B-spline of degree N=3: " + utils::pp(RATE/(std::clock()-clk), 6) + " eval/s\n";
#endif
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
    ok &= test1dSpline() || printFail("1d spline");
    ok &= test2dSpline() || printFail("2d spline");
    ok &= test3dSpline() || printFail("3d spline");
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
