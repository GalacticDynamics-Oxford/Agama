#include "math_core.h"
#include "math_fit.h"
#include "math_random.h"
#include "math_sample.h"
#include "math_specfunc.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
int numEval=0;

class test1: public math::IFunctionNoDeriv, public math::IFunctionNdim{
    virtual double value(double x) const{
        return 1/sqrt(1-x*x);
    }
    virtual void eval(const double x[], double val[]) const{
        val[0] = value(x[0]);
    }
    virtual unsigned int numVars() const { return 1; }
    virtual unsigned int numValues() const { return 1; }
};

class test2: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        return pow(1-x*x*x*x,-2./3);
    }
};

class test3: public math::IFunction{
public:
    int nd;
    test3(int nder) : nd(nder) {};
    virtual void evalDeriv(double x, double* val, double* der=0, double* =0) const{
        numEval++;
        *val = math::sign(x-0.3)*pow(fabs(x-0.3), 1./5);
        if(der) *der = *val/5/(x-0.3);
    }
    virtual unsigned int numDerivs() const { return nd; }
};

class test4: public math::IFunction, public math::IFunctionNdim{
public:
    int nd;
    test4(int nder) : nd(nder) {};
    virtual void evalDeriv(double x, double* val, double* der=0, double* =0) const{
        numEval++;
        *val = x-1+1e-3/sqrt(x);
        if(der) *der = 1-0.5e-3/pow(x,1.5);
    }
    virtual void eval(const double x[], double val[]) const{
        evalDeriv(x[0], val);
    }
    virtual unsigned int numDerivs() const { return nd; }
    virtual unsigned int numVars() const { return 1; }
    virtual unsigned int numValues() const { return 1; }
};

class test5: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        numEval++;
        return exp(1.0001-x)*(x<INFINITY ? (x-1)-1e5*(x-1.0001)*(x-1.0001)-1e-4 : 1) - 1e-12*(1+1/x);
    }
};

class test6: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        numEval++;;
        return sin(1e4*x);
    }
};

// test function for multidimensional minimization
static const double  // rotation
    A00 = 0.8786288646, A01 = -0.439043856, A02 = 0.1877546558,
    A10 = 0.4474142786, A11 = 0.8943234085, A12 = -0.002470791,
    A20 = -0.166828598, A21 = 0.0861750222, A22 = 0.9822128505,
    c0 = -0.5, c1 = -1., c2 = 2,    // center
    s0 = 2.0,  s1 = 0.5, s2 = 0.1;  // scale
class test7Ndim: public math::IFunctionNdimDeriv{
public:
    // 3-dimensional paraboloid centered at c[], scaled with s[] and rotated with orthogonal matrix A[][]
    virtual void evalDeriv(const double x[], double val[], double der[]) const{
        double x0 = (x[0]-c0)*s0, x1 = (x[1]-c1)*s1, x2 = (x[2]-c2)*s2;
        double v0 = x0*A00 + x1*A01 + x2*A02;
        double v1 = x0*A10 + x1*A11 + x2*A12;
        double v2 = x0*A20 + x1*A21 + x2*A22;
        double v  = x0*v0  + x1*v1  + x2*v2;
        if(val)
            val[0] = 1. - 1. / (1 + v*v);
        if(der) {
            double m = 2 * v / pow_2(1 + v*v);
            der[0] = m * (v0 + A00*x0 + A10*x1 + A20*x2) * s0;
            der[1] = m * (v1 + A01*x0 + A11*x1 + A21*x2) * s1;
            der[2] = m * (v2 + A02*x0 + A12*x1 + A22*x2) * s2;
        }
        numEval++;
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
};

// test function for sampleNdim
#if 1
// a 3d function that is positive inside a toroidal region in space 
static const double Rout = 3, Rin = 1;  // outer and inner radii of the torus
class test8Ndim: public math::IFunctionNdim{
public:
    test8Ndim()
    {
        exact = 2*pow_2(M_PI*Rin)*Rout;    // volume of a torus
        ymin[0]=-4; ymin[1]=-4; ymin[2]=-2;
        ymax[0]=+4; ymax[1]=+4; ymax[2]=+2;
    }
    // 3-dimensional torus rotated with orthogonal matrix A[][]
    virtual void eval(const double x[], double val[]) const{
        double x0 = x[0]*A00+x[1]*A01+x[2]*A02;
        double x1 = x[0]*A10+x[1]*A11+x[2]*A12;
        double x2 = x[0]*A20+x[1]*A21+x[2]*A22;
        val[0] = pow_2(sqrt(x0*x0+x1*x1)-Rout)+x2*x2 <= Rin*Rin ? 1.0+x0*0.2 : 0.0;
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++numEval;
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
    double ymin[3], ymax[3];  // boundaries of the integration region
    double exact;             // exact analytic value of the integral
};
#else
// a 3d function with an integrable singularity at the corner, taken from from GSL
class test8Ndim: public math::IFunctionNdim{
public:
    test8Ndim()
    {
        exact = 1.393203929685677;
        ymin[0]=ymin[1]=ymin[2]=0;
        ymax[0]=ymax[1]=ymax[2]=M_PI;
    }
    virtual void eval(const double x[], double val[]) const{
        val[0] = 1./pow_3(M_PI) / (1 - cos(x[0]) * cos(x[1]) * cos(x[2]));
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++numEval;
    }
    virtual unsigned int numVars()   const { return 3; }
    virtual unsigned int numValues() const { return 1; }
    double ymin[3], ymax[3];  // boundaries of the integration region
    double exact;             // exact analytic value of the integral
};
#endif

// test functions for estimating the accuracy of Gauss-Legendre integration
class test_GL_powerlaw: public math::IFunctionNoDeriv{
public:
    test_GL_powerlaw(double _p): p(_p) {};
    virtual double value(double x) const{
        return pow(x, p);
    }
    double exactValue(double xmin, double xmax) const{
        return p==-1 ? log(xmax/xmin) : (pow(xmax, p+1) - pow(xmin, p+1)) / (p+1);
    }
    double p;
};

class test_GL_angular: public math::IFunctionNoDeriv{
    double q2, gamma2;
public:
    test_GL_angular(double q, double gamma): q2(q*q), gamma2(gamma/2) {}
    virtual double value(double costheta) const{
        return pow(1-pow_2(costheta)*(1-1/q2), -gamma2);
    }
};


// test of least-square fitting
class test9LM: public math::IFunctionNdimDeriv {
public:
    test9LM() {  // init data points
        for(int i=0; i<numDataPoints; i++) {
            double x = i*1.0/numDataPoints;
            dataX[i] = x;
            dataY[i] = cos(1.5*x*M_PI) + 0.2*sin(x*5*M_PI) + // some deterministic noise
            0.1*cos(x*21.4231) + 0.07*sin(x*67.56473) + 0.03*sqrt(fabs(tan(x*38.8322)));
        }
    }
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {   // fit function a*sin(b*x+c) to data points
        double a = vars[0], b = vars[1], c = vars[2];
        for(int i=0; i<numDataPoints; i++) {
            double sinx = sin(b * dataX[i] + c);
            if(values)
                values[i] = a * sinx - dataY[i];
            if(derivs) {
                double cosx = cos(b * dataX[i] + c);
                derivs[i*3  ] = sinx;
                derivs[i*3+1] = a * cosx * dataX[i];
                derivs[i*3+2] = a * cosx;
            }
        }
        numEval++;
    }
    void dump(const double vars[], const char* fileName) const
    {
        std::ofstream strm(fileName);
        double a = vars[0], b = vars[1], c = vars[2];
        for(int i=0; i<numDataPoints; i++)
            strm << dataX[i] << '\t' << dataY[i] << '\t' << a*sin(b * dataX[i] + c) << '\n';
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return numDataPoints; }
private:
    static const int numDataPoints = 100;
    double dataX[numDataPoints], dataY[numDataPoints];
};

// represent the least-square fitting problem as a general minimization problem
class test9min: public math::IFunctionNdimDeriv {
public:
    test9min(const math::IFunctionNdimDeriv& _F) : F(_F) {}
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        std::vector<double> val(F.numValues());
        std::vector<double> jac(F.numValues()*F.numVars());
        F.evalDeriv(vars, &val[0], &jac[0]);
        if(values)
            *values = 0;
        if(derivs)
            for(unsigned int i=0; i<F.numVars(); i++)
                derivs[i] = 0;
        for(unsigned int k=0; k<F.numValues(); k++) {
            if(values)
                *values += 0.5 * pow_2(val[k]);
            if(derivs)
                for(unsigned int i=0; i<F.numVars(); i++)
                    derivs[i] += val[k] * jac[k*F.numVars()+i];
        }
    }
    virtual unsigned int numVars() const { return 3; }
    virtual unsigned int numValues() const { return 1; }
private:
    const math::IFunctionNdimDeriv& F;
};

// test of multidimensional root-finding using Rosenbrock's function
class test10Ndim: public math::IFunctionNdimDeriv {
public:
    test10Ndim(double _a, double _b) : a(_a), b(_b) {}
    virtual void evalDeriv(const double vars[], double values[], double *derivs=0) const
    {
        numEval++;;
        if(values) {
            values[0] = a * (1 - vars[0]);
            values[1] = b * (vars[1] - pow_2(vars[0]));
        }
        if(derivs) {
            derivs[0] = -a;
            derivs[1] = 0;
            derivs[2] = -2*b * vars[0];
            derivs[3] = b;
        }
    }
    virtual unsigned int numVars() const { return 2; }
    virtual unsigned int numValues() const { return 2; }
private:
    double a, b;
};

bool err() {
    std::cout << "\033[1;31m **\033[0m\n";
    return false;
}

template<typename Scaling> bool testScaling(const Scaling& scaling)
{
    double maxerr = 0, maxder = 0;
    for(int i=0; i<1000; i++) {
        double s = math::random() * 0.7, duds;
        double u = math::unscale(scaling, s, &duds);
        double duds_fd = (
            math::unscale(scaling, s - 2*SQRT_DBL_EPSILON) -
            math::unscale(scaling, s - 1*SQRT_DBL_EPSILON) * 8 +
            math::unscale(scaling, s + 1*SQRT_DBL_EPSILON) * 8 -
            math::unscale(scaling, s + 2*SQRT_DBL_EPSILON) ) / (12 * SQRT_DBL_EPSILON);
        double S = math::scale(scaling, u);
        if(S!=0 && u!=0)
            // skip values which are rounded to the boundary because of the loss of precision
            maxerr = fmax(maxerr, fabs(S-s));
        if(isFinite(duds_fd + duds))
            maxder = fmax(maxder, fabs(duds_fd - duds) / fmax(fabs(duds), 1));
    }
    if(maxerr > 1e-15 || maxder > 1e-7)
        return err();
    return true;
}

int main()
{
    std::cout << std::setprecision(10);
    bool ok=true;

    // scaling transformations
    std::cout << "Scaling: InfLeft";
    ok &= testScaling(math::ScalingSemiInf(-12.34));
    std::cout << " InfRight";
    ok &= testScaling(math::ScalingSemiInf(0.123));
    std::cout << " Inf0";
    ok &= testScaling(math::ScalingSemiInf());
    std::cout << " Inf";
    ok &= testScaling(math::ScalingDoubleInf());
    std::cout << " DInf0";
    ok &= testScaling(math::ScalingDoubleInf(1e-10));
    std::cout << " DInf-";
    ok &= testScaling(math::ScalingDoubleInf(1e+10));
    std::cout << " DInf+";
    ok &= testScaling(math::ScalingInf());
    std::cout << " Lin";
    ok &= testScaling(math::ScalingLin(-10.98, -2.345));
    std::cout << " Cub";
    ok &= testScaling(math::ScalingCub(0,1));
    std::cout << " Qui";
    ok &= testScaling(math::ScalingQui(0,1));
    std::cout << "\n";

    // special functions
    double maxerrk=0, maxerrs=0, maxerrc=0;
    for(double ecc=0.; ecc<0.999; ecc = 1-(1-ecc)*0.9) {
        for(double phase=-0.05; phase<1e10; phase<6.4? phase+=0.01 : phase*=1.5) {
            double eta = math::solveKepler(ecc, phase);
            double sineta, coseta;
            math::sincos(eta, sineta, coseta);
            double phasek = eta - ecc * sineta;
            double phasew = math::wrapAngle(phase);
            if(fabs(phasek - phasew)>0.1)
                std::cout << utils::toString(phase, 17) <<
                     ", " << utils::toString(phasek,17) <<
                     ", " << utils::toString(phasew,17) << "\n";
            maxerrk = fmax(maxerrk, fabs(phasek - phasew));
            maxerrs = fmax(maxerrs, fabs(sin(eta) - sineta));
            maxerrc = fmax(maxerrc, fabs(cos(eta) - coseta));
        }
    }
    std::cout << "Specfunc: E(sin)=" << utils::toString(maxerrs,4);
    ok &= maxerrs < 5e-16 || err();
    std::cout << ", E(cos)=" << utils::toString(maxerrc,4);
    ok &= maxerrc < 5e-16 || err();
    std::cout << ", E(kepler)=" << utils::toString(maxerrk,4);
    ok &= maxerrk < 2e-15 || err();
    std::cout << "\n";

    // integration routines
    const double toler = 1e-6;
    double exact = (M_PI*2/3), error=0, result, result1;
    test1 t1;
    result = math::integrateGL(t1, -1, 1./2, math::MAX_GL_ORDER);
    math::integrateGL(t1, -1., 1./2, math::MAX_GL_ORDER, &result1);  // same but using IFunctionNdim interface
    std::cout << "Int1: fixed GL="<<result<<" (delta="<<(result-exact)<<", neval="<<math::MAX_GL_ORDER;
    ok &= (fabs(1-result/exact)<0.03 && fabs(result-result1)<1e-15) || err();
    result = math::integrate(t1, -1, 1./2, toler, &error, &numEval);
    std::cout << "), naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<2e-3 && fabs(result-exact)<error) || err();
    result = math::integrateAdaptive(t1, -1, 1./2, toler, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<toler && fabs(result-exact)<error) || err();
    math::ScaledIntegrand<math::ScalingCub> t1s(math::ScalingCub(-1, 1), t1);
    result = math::integrate(t1s, math::scale(t1s.scaling, -1), math::scale(t1s.scaling, 0.5),
        toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<1e-12 && fabs(result-exact)<fmax(error,1e-12)) || err();
    result = math::integrateGL(t1s, math::scale(t1s.scaling, -1), math::scale(t1s.scaling, 0.5), 13);
    std::cout<<"), scaled GL="<<result<<" (delta="<<(result-exact)<<", neval=13)\n";
    ok &= (fabs(1-result/exact)<1e-12) || err();

    exact = 2.274454287;
    test2 t2;
    result = math::integrate(t2, -1, 2./3, toler, &error, &numEval);
    std::cout << "Int2: naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<2e-2 && fabs(result-exact)<error) || err();
    result = math::integrateAdaptive(t2, -1, 2./3, toler*15, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<1e-5 && fabs(result-exact)<error) || err();
    math::ScaledIntegrand<math::ScalingCub> t2s(math::ScalingCub(-1, 1), t2);
    result = math::integrate(t2s, math::scale(t2s.scaling, -1), math::scale(t2s.scaling, 2./3),
        toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= (fabs(1-result/exact)<2e-4 && fabs(result-exact)<error) || err();
    result = math::integrateGL(t2s, math::scale(t2s.scaling, -1), math::scale(t2s.scaling, 2./3), 20);
    std::cout<<"), scaled GL="<<result<<" (delta="<<(result-exact)<<", neval=20)\n";
    ok &= (fabs(1-result/exact)<1e-2) || err();

    // root-finding
    exact=0.3;
    numEval=0;
    result = math::findRoot(test3(0), 0, 0.8, toler);
    std::cout << "Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(1-result/exact)<toler) || err();
    numEval=0;
    result = math::findRoot(test3(1), 0, 0.8, toler);
    std::cout << "with derivative: Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(1-result/exact)<toler) || err();

    exact=1.000002e-6;
    numEval=0;
    result = math::findRoot(test4(0), 1e-15, 0.8, 1e-8);
    std::cout << "Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(result-exact)<1e-8*0.8) || err();
    numEval=0;
    result = math::findRoot(test4(1), 1e-15, 0.8, 1e-8);
    std::cout << "with derivative: Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(result-exact)<1e-8*0.8) || err();

    double x0 = exact*1.5;
    double x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToPositive();
    result = test4(0)(x1);
    std::cout << "positive value at x="<<x1<<", value="<<result<<"\n";
    ok &= (isFinite(x1+result) && x1>0 && x1<exact && result>0) || err();
    x0 = exact*0.9;
    x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToNegative();
    result = test4(0)(x1);
    std::cout << "negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= (isFinite(x1+result) && result<0) || err();
    x1 = x0 + math::PointNeighborhood(test4(1), x0).dxToNegative();
    result = test4(0)(x1);
    std::cout << "(with deriv) negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= (isFinite(x1+result) && result<0) || err();

    x0 = 1.00009;
    exact = 1.000100000002;
    x1 = x0 + math::PointNeighborhood(test5(), x0).dxToPositive();
    result = test5()(x1);
    std::cout << "f5: positive value at x="<<exact<<"+"<<(x1-exact)<<", value="<<result<<"\n";
    ok &= (isFinite(x1+result) && result>0) || err();
    numEval=0;
    result = math::findRoot(test5(), 1, x1, toler);
    std::cout << "Root5="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(result-exact)<toler*exact) || err();

    exact=1.000109999998;
    numEval=0;
    result = math::findRoot(test5(), math::ScalingSemiInf(x1), toler);
    std::cout << "Another root="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(result-exact)<toler*exact) || err();

    // minimization
    numEval=0;
    exact=0.006299605249;
    result = math::findMin(test4(0), 1e-15, 1, NAN, toler);
    std::cout << "Minimum of f4(x) at x="<<result<<" is "<<test4(0)(result)<<
        " (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= (fabs(result-exact)<toler*exact) || err();
    numEval=0;
    double xinit[] = {0.5};
    double xstep[] = {0.1};
    double xresult[1];
    int numIter = findMinNdim(test4(0), xinit, xstep, toler, 100, xresult);
    std::cout << "N-dimensional minimization (N=1) of the same function: minimum at x="<<xresult[0]<<
        " is "<<test4(0)(xresult[0])<<" (delta="<<(xresult[0]-exact)<<
        "; neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= (fabs(result-exact)<toler) || err();

    numEval=0;
    double yinit[] = {5.0,-4.,2.5};
    double ystep[] = {0.1,0.1,0.1};
    double yresult[3];
    numIter = findMinNdim(test7Ndim(), yinit, ystep, 1e-10, 1000, yresult);
    test7Ndim().eval(yresult, &result);
    std::cout << "N-dimensional minimization (N=3): minimum at x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        " is "<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= (fabs(yresult[0]-c0) * fabs(yresult[1]-c1) * fabs(yresult[2]-c2) < 1e-10) || err();

    numEval=0;
    numIter = findMinNdimDeriv(test7Ndim(), yinit, ystep[0], 1e-10, 1000, yresult);
    test7Ndim().eval(yresult, &result);
    std::cout << "Min. same func. with derivatives: minimum at x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        " is "<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    // the test function is quartic, not quadratic, near minimum,
    // which is tough for derivative-based minimizers - hence a looser tolerance
    ok &= (fabs(yresult[0]-c0) * fabs(yresult[1]-c1) * fabs(yresult[2]-c2) < 1e-8) || err();

    // N-dimensional root finding
    numEval=0;
    yinit[0] = -10;
    yinit[1] = -5;
    numIter = math::findRootNdimDeriv(test10Ndim(1, 10), yinit, 1e-10, 100, yresult);
    std::cout << "RootNdim(N=2): "<<yresult[0]<<","<<yresult[1]<<
        " (delta="<<(yresult[0]-1)<<","<<(yresult[1]-1)<<"; neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= (fabs(yresult[0]-1)<toler && fabs(yresult[1]-1)<toler) || err();

    // N-dimensional integration
    numEval=0;
    test8Ndim fnc8;
    integrateNdim(fnc8, fnc8.ymin, fnc8.ymax, toler, 1000000, &result, &error);
    std::cout << "Volume of a 3d torus = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc8.exact)<<"; neval="<<numEval<<")\n";
    ok &= (error < 2.0 && fabs(result-fnc8.exact) < error*2) || err();

    // N-dimensional sampling
    numEval=0;
    math::Matrix<double> points;
    sampleNdim(fnc8, fnc8.ymin, fnc8.ymax, 100000, points, NULL, &result, &error);
    std::cout << "Monte Carlo Volume of a 3d torus = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc8.exact)<<"; neval="<<numEval<<")\n";
    ok &= (error < 1.0 && fabs(result-fnc8.exact) < error*2) || err();
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream fout("sampleNdim.dat");
        for(unsigned int i=0; i<points.rows(); i++)
            fout << points(i,0) << "\t" << points(i,1) << "\t" << points(i,2) << "\n";
    }

#if 0
    // test the accuracy of fixed-order (n) Gauss-Legendre quadrature in integrating a power-law function in radius
    for(double p=-40; p<=40; p+=1.77) {
        test_GL_powerlaw tpl(p);
        for(int n=8; n<=32; n*=2) {
            double xmin=1., xmax=1.5;
            result = math::integrateGL(tpl, xmin, xmax, n);
            exact  = tpl.exactValue(xmin, xmax);
            std::cout << "p="<<p<<", N="<<n<<": error("<<xmax<<")="<<(result-exact)/exact;
            xmax=2.0;
            result = math::integrateGL(tpl, xmin, xmax, n);
            exact  = tpl.exactValue(xmin, xmax);
            std::cout << ", error("<<xmax<<")="<<(result-exact)/exact<<"\n";
        }
    }
    // test accuracy of fixed-order GL quadrature for computing spherical-harmonic coefficients
    // of a function mimicking a power-law density profile with flattening ( f ~ (R+z/q)^-gamma )
    test_GL_angular testfnc1(0.25, 1.);
    test_GL_angular testfnc2(0.5, 2.);
    test_GL_angular testfnc3(0.5, 4.);
    double exact1 = math::integrateAdaptive(testfnc1, -1, 1, 1e-14);
    double exact2 = math::integrateAdaptive(testfnc2, -1, 1, 1e-14);
    double exact3 = math::integrateAdaptive(testfnc3, -1, 1, 1e-14);
    for(int l=0; l<32; l+=1) {
        std::cout << l << "\t" << (math::integrateGL(testfnc1, -1, 1, l+1)-exact1) << '\t' <<
        (math::integrateGL(testfnc2, -1, 1, l+1)-exact2) << '\t' << (math::integrateGL(testfnc3, -1, 1, l+1)-exact3) << '\n';
    }
#endif

    // nonlinear least-square fitting using Levenberg-Marquardt
    numEval=0;
    test9LM fncLM;
    test9min fncMin(fncLM);
    yinit[0] = yinit[1] = yinit[2] = 0.5;
    numIter = nonlinearMultiFit(fncLM, yinit, 1e-4, 100, yresult);
    fncMin.eval(yresult, &result);
    std::cout << "Nonlinear least-square fit: parameters x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        ", sum square dif="<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    //fncLM.dump(yresult, "fit.log");
    ok &= fabs(result) < 1.5 || err();  // well it's not a particularly good fit by design

    // same problem using a generic minimizer with derivatives
    numEval = 0;
    numIter = findMinNdimDeriv(fncMin, yinit, ystep[0], 1e-4, 100, yresult);
    fncMin.eval(yresult, &result);
    std::cout << "Same with deriv. minimizer: parameters x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        ", sum square dif="<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= fabs(result) < 1.5 || err();

    // same problem using a generic minimizer without derivatives
    numEval = 0;
    numIter = findMinNdim(fncMin, yinit, ystep, 1e-4, 100, yresult);
    fncMin.eval(yresult, &result);
    std::cout << "Same minimizer w/o derivs : parameters x=("<<
        yresult[0]<<","<<yresult[1]<<","<<yresult[2]<<")"
        ", sum square dif="<<result<<" (neval="<<numEval<<", nIter="<<numIter<<")\n";
    ok &= fabs(result) < 1.5 || err();

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
