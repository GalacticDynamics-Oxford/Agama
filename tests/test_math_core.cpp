/** \name   test_math_core.cpp
    \author Eugene Vasiliev
    \date   2015-2024

    Test the accuracy of various mathematical algorithms (root-finding, integration, etc..)
*/
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

class Integrand: public math::IFunctionNoDeriv, public math::IFunctionNdim{
    virtual void eval(const double x[], double val[]) const{
        val[0] = value(x[0]);
    }
    virtual unsigned int numVars() const { return 1; }
    virtual unsigned int numValues() const { return 1; }
};

class TestInt1: public Integrand{
    virtual double value(double x) const{
        return (3./2/M_PI) / sqrt(1-x*x);
    }
};

class TestInt2: public Integrand{
    virtual double value(double x) const{
        // normalization is some combination of gamma and hypergeometric functions
        return pow(1-x*x*x*x,-2./3) / 2.27445428374349;
    }
};

class TestInt3: public Integrand{
    virtual double value(double x) const{
        return 33*M_PI * x * sin(33*M_PI * x*x);
    }
};

class TestInt4: public Integrand{
    virtual double value(double x) const{
        return 1/(1.001-x) / log(1001);
    }
};

class TestInt5: public Integrand{
    virtual double value(double x) const{
        const double a=0.4, b=0.01;
        return 1 / (pow_2(x-a) + pow_2(b)) * b / (atan(a/b) + atan((1-a)/b));
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

// another test case for integrateNdim - a function with an integrable (in D>=2) singularity
template<int D>
class test9Ndim: public math::IFunctionNdim{
public:
    test9Ndim() : exact(
        D==2 ? 4*log(1+M_SQRT2) + 4*atanh(1/M_SQRT2) :
        D==3 ? 24*asinh(1/M_SQRT2) - 2*M_PI : NAN /*not known*/)
    {}
    virtual void eval(const double x[], double result[]) const {
        result[0] = 0;
        for(int d=0; d<D; d++)
            result[0] += pow_2(x[d]);
        if(result[0]>0)  // else leave at 0 instead of infinity
            result[0] = 1 / sqrt(result[0]);
#ifdef _OPENMP
#pragma omp atomic
#endif
        ++numEval;
    }
    virtual unsigned int numVars()   const { return D; }
    virtual unsigned int numValues() const { return 1; }
    const double exact;   // exact analytic value of the integral on [-1..1]^D
};

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
    std::cout << "\033[1;31m **\033[0m";
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

bool testIntegration(const Integrand& fnc, double a, double b,
    double tolerNaiveGL, double tolerNaiveGK, double tolerAdapt,
    double tolerScaledGL, double tolerScaledGK)
{
    const int GLORDER_NAIVE = math::MAX_GL_ORDER, GLORDER_SCALED = 15;
    const double exact = 1.0;
    bool ok = true;
    double result, resultN, error, errorN;
    int numEval;
    math::ScaledIntegrand<math::ScalingCub> sfnc(math::ScalingCub(-1, 1), fnc);

    result = math::integrateGL(fnc, a, b, GLORDER_NAIVE);
    // same but using IFunctionNdim interface
    math::integrateGL(fnc, a, b, GLORDER_NAIVE, &resultN);
    std::cout << "fixed GL=" << result << " (delta=" << (result-exact) <<
        ", dif_1d_vs_Nd=" << (result-resultN) << ", neval=" << GLORDER_NAIVE;
    ok &= (fabs(1-result/exact) < tolerNaiveGL && fabs(result-resultN) < 4e-15) || err();

    result = math::integrate(fnc, a, b, tolerNaiveGK, &error, &numEval);
    std::cout << "), naive=" << result << " +- " << error <<
        " (delta=" << (result-exact) << ", neval=" << numEval;
    ok &= (fabs(1-result/exact) < tolerNaiveGK && fabs(result-exact) < error) || err();

    result = math::integrateAdaptive(fnc, a, b, tolerAdapt, &error, &numEval);
    std::cout << "), adaptive=" << result << " +- " << error <<
        " (delta=" << (result-exact) << ", neval=" << numEval;
    ok &= (fabs(1-result/exact) < tolerAdapt && fabs(result-exact) < error) || err();

    math::integrateNdim(fnc, &a, &b, tolerAdapt, 100000, &resultN, &errorN, &numEval);
    std::cout << "), adaptiveNdim=" << resultN << " +- " << errorN <<
        " (delta=" << (resultN-exact) << ", neval=" << numEval;
    ok &= (fabs(1-resultN/exact) < tolerAdapt && fabs(resultN-exact) < error) || err();

    result = math::integrateGL(sfnc, math::scale(sfnc.scaling, a), math::scale(sfnc.scaling, b),
        GLORDER_SCALED);
    std::cout << "), scaledGL=" << result <<
        " (delta=" << (result-exact) << ", neval=" << GLORDER_SCALED;
    ok &= (fabs(1-result/exact) < tolerScaledGL) || err();

    result = math::integrate(sfnc, math::scale(sfnc.scaling, a), math::scale(sfnc.scaling, b),
        tolerScaledGK, &error, &numEval);
    std::cout << "), scaled=" << result << " +- " << error <<
        " (delta=" << (result-exact) << ", neval=" << numEval;
    ok &= (fabs(1-result/exact) < tolerScaledGK && fabs(result-exact) < error) || err();

    std::cout << ")\n";
    return ok;
}

inline bool equal(double a, double b)
{
    if(!(a==b)) return false;
    if(a==0) return std::signbit(a) == std::signbit(b);  // +0. and -0. are distinct
    return true;
}

inline bool swapIfNeeded(double& x1, double& x2)
{
    if(x1 > x2) {
        std::swap(x1, x2);
        return true;
    }
    return false;
}

inline double evalQuartic(double a4, double a3, double a2, double a1, double a0, double x,
    double* error=NULL)
{
    // estimate the rounding error on the function value coming from the cancellation of terms
    double
    val3 = (fabs(a3) + fabs(x * a4)),
    val2 = (fabs(a2) + fabs(x * val3)),
    val1 = (fabs(a1) + fabs(x * val2)),
    val0 = (fabs(a0) + fabs(x * val1)),
    result = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4))),
    // estimate the rounding error on the function value from the inexactness of its argument
    // multiplied by the derivative
    deriv  = a1 + x * (a2 * 2 + x * (a3 * 3 + x * a4 * 4));
    if(error)
        *error = DBL_EPSILON * fmax(fabs(x * deriv), val0);
    return result;
}

bool testSolvePoly()
{
    double toler = 5 * DBL_EPSILON;
    bool ok = true;
    double roots[4];
    int nroots;

    // quadratic
    nroots = math::solveQuadratic(1, 1, 1, roots);
    ok &= nroots==0;
    nroots = math::solveQuadratic(4, 4, 1, roots);
    ok &= nroots==1 && fabs(roots[0] + 0.5) < toler;
    nroots = math::solveQuadratic(3, -4, 1, roots);
    ok &= nroots==2 &&
        fabs(roots[0] - 1./3) < toler &&
        fabs(roots[1] - 1) < toler;
    nroots = math::solveQuadratic(1, -1, -1, roots);
    ok &= nroots==2 &&
        fabs(roots[0] - (0.5-sqrt(1.25))) < toler &&
        fabs(roots[1] - (0.5+sqrt(1.25))) < toler;

    // cubic
    nroots = math::solveCubic(2, 3, -3, -9, roots);
    ok &= nroots==1 && fabs(roots[0] - 1.5) < toler;
    nroots = math::solveCubic(2, 0, -6, 4, roots);
    ok &= nroots==2 &&
        fabs(roots[0] + 2) < toler &&
        fabs(roots[1] - 1) < toler;
    nroots = math::solveCubic(3, 5, -9, -4, roots);
    ok &= nroots==3 &&
        fabs(roots[0] - (-1.5-sqrt(1.25))) < toler &&
        fabs(roots[1] - (-1.5+sqrt(1.25))) < toler &&
        fabs(roots[2] - 4./3) < toler;

    // quartic
    nroots = math::solveQuartic(1, -2, -2, 5, -2, roots);
    ok &= nroots==4 &&
        fabs(roots[0] - (-0.5-sqrt(1.25))) < toler &&
        fabs(roots[1] - (-0.5+sqrt(1.25))) < toler &&
        fabs(roots[2] - 1) < toler &&
        fabs(roots[3] - 2) < toler;
    nroots = math::solveQuartic(2, 0, -5, -1, 1, roots);
    ok &= nroots==4 &&
        fabs(roots[0] - (-0.5-sqrt(0.75))) < toler &&
        fabs(roots[1] - ( 0.5-sqrt(1.25))) < toler &&
        fabs(roots[2] - (-0.5+sqrt(0.75))) < toler &&
        fabs(roots[3] - ( 0.5+sqrt(1.25))) < toler;
    nroots = math::solveQuartic(2, -4, -9, 27, -135./8, roots);  // -5/2, 3/2 @3
    ok &= nroots==2 &&
        fabs(roots[0] + 2.5) < toler &&
        fabs(roots[1] - 1.5) < toler;
    /*nroots = math::solveQuartic(3, 2, -1, -4./9, 4./27, roots);  // -2/3 @2, 1/3 @2
    ok &= nroots==2 &&  /// this fails because of roundoff errors, need exactly representable numbers..
        fabs(roots[0] + 2./3) < toler &&
        fabs(roots[1] - 1./3) < toler;*/
    nroots = math::solveQuartic(8, -8, -1, 1.5, 9./32, roots);  // -1/4 @2, 3/4 @2
    ok &= nroots==2 &&
        fabs(roots[0] + 1./4) < toler &&
        fabs(roots[1] - 3./4) < toler;
    nroots = math::solveQuartic(4, 0, -3, -7, -3, roots);  // -1/2, 3/2
    ok &= nroots==2 &&
        fabs(roots[0] + 0.5) < toler &&
        fabs(roots[1] - 1.5) < toler;
    nroots = math::solveQuartic(4, 4, -7, -10, -3, roots);  // -1 @2, -1/2, 3/2
    ok &= nroots==3 &&
        fabs(roots[0] + 1.0) < toler &&
        fabs(roots[1] + 0.5) < toler &&
        fabs(roots[2] - 1.5) < toler;
    nroots = math::solveQuartic(1, -4, 6, -4, 1, roots);  // extreme case of multiplicity 4;
    ok &= nroots==1 && fabs(roots[0] - 1) < toler;  // in general it won't give accurate results
    if(!ok) {
        std::cout << "Polynomial root-finding failed";
        err();
    }

    double max_rel_error = 0;
    int nroots_extra = 0, nroots_missed = 0;
    for(int i=0; i<10000; i++) {
        double x1, c2, d2, a3;
        math::getNormalRandomNumbers(x1, c2);
        math::getNormalRandomNumbers(d2, a3);
        x1  = sinh(x1);          // spread out the location of the first root
        c2  = pow_3(c2) + x1;    // make the peak of the parabola sometimes close to x1
        d2 *= pow_2(pow_2(d2));  // make the vertical offset of the peak sometimes very small
        if(i%100 == 0) d2 = 0;
        // generate equations of the form a3 * (x-x1) * ((x-c2)^2 - d2)
        double a2 = -a3 * (x1 + 2 * c2);
        double a1 =  a3 * ((2 * x1 + c2) * c2 - d2);
        double a0 = -a3 * x1 * (c2 * c2 - d2);
        int nroots_expected = 1;
        double roots_expected[3] = {x1, NAN, NAN};
        if(d2 >= 0)
            roots_expected[nroots_expected++] = c2 - sqrt(d2);
        if(d2 > 0)
            roots_expected[nroots_expected++] = c2 + sqrt(d2);
        switch(nroots_expected) {
            case 2:
                swapIfNeeded(roots_expected[0], roots_expected[1]);
                break;
            case 3:
                swapIfNeeded(roots_expected[0], roots_expected[1]);
                swapIfNeeded(roots_expected[0], roots_expected[2]);
                swapIfNeeded(roots_expected[1], roots_expected[2]);
                break;
            default: ;
        }

        nroots = math::solveCubic(a3, a2, a1, a0, roots);
        // we do not directly check the roots, but rather the values of the polynomial at these points
        if(nroots != nroots_expected) {
            nroots_extra += nroots > nroots_expected;
            nroots_missed+= nroots < nroots_expected;
            // don't check the function values in this case,
            // as we haven't determined which roots are genuine
        } else {
            for(int p=0; p<nroots; p++) {
                double error, f = evalQuartic(0, a3, a2, a1, a0, roots[p], &error),
                f_expected  = evalQuartic(0, a3, a2, a1, a0, roots_expected[p]);
                double rel_error = fabs(f) / fmax(error, fabs(f_expected));
                max_rel_error = fmax(max_rel_error, rel_error);
                ok &= p==0 || roots[p] > roots[p-1];
            }
        }
    }
    // forgive some cases of missed or extra roots arising due to roundoff errors;
    // need a more robust treatment of special cases (multiple roots)..
    std::cout << "Analytic solution to cubic equation: error = " << max_rel_error << " epsilons;";
    ok &= (max_rel_error <= 2 && nroots_extra + nroots_missed < 150) || err();

    max_rel_error = 0;
    nroots_extra  = 0;
    nroots_missed = 0;
    for(int i=0; i<10000; i++) {
        double c1, d1, c2, d2;
        math::getNormalRandomNumbers(c1, c2);
        math::getNormalRandomNumbers(d1, d2);
        c1  = sinh(c1);         // spread out the location of the peak of the first parabola
        c2  = pow_3(c2) + c1;   // make the peak of the second parabola sometimes close to the first
        d1 *= pow_2(pow_2(d1)); // make the vertical offset often very small (either sign)
        d2 *= pow_2(d2);        // same for the second parabola, but less extreme
        double h1 = c1 * c1 - d1, h2 = c2 * c2 - d2;
        if(i%100 == 0) d2 = 0;
        // generate equations of the form ((x-c1)^2 - d1) ((x-c2)^2 - d2)
        double a3 =-2 * c1 - 2 * c2;
        double a2 = 4 * c1 * c2 + h1 + h2;
        double a1 =-2 * c1 * h2 - 2 * c2 * h1;
        double a0 = h1 * h2;
        int nroots_expected = 0;
        double roots_expected[4] = {NAN, NAN, NAN, NAN};
        if(d1 >= 0)
            roots_expected[nroots_expected++] = c1 - sqrt(d1);
        if(d1 > 0)
            roots_expected[nroots_expected++] = c1 + sqrt(d1);
        if(d2 >= 0)
            roots_expected[nroots_expected++] = c2 - sqrt(d2);
        if(d2 > 0)
            roots_expected[nroots_expected++] = c2 + sqrt(d2);
        switch(nroots_expected) {
            case 2:
                swapIfNeeded(roots_expected[0], roots_expected[1]);
                break;
            case 3:
                swapIfNeeded(roots_expected[0], roots_expected[1]);
                swapIfNeeded(roots_expected[0], roots_expected[2]);
                swapIfNeeded(roots_expected[1], roots_expected[2]);
                break;
            case 4:
                swapIfNeeded(roots_expected[0], roots_expected[2]);
                swapIfNeeded(roots_expected[1], roots_expected[3]);
                swapIfNeeded(roots_expected[1], roots_expected[2]);
                break;
            default: ;
        }

        nroots = math::solveQuartic(1, a3, a2, a1, a0, roots);
        if(nroots != nroots_expected) {
            nroots_extra  += nroots > nroots_expected;
            nroots_missed += nroots < nroots_expected;
            // don't check the function values in this case,
        } else {
            for(int p=0; p<nroots; p++) {
                double error, f = evalQuartic(1, a3, a2, a1, a0, roots[p], &error),
                    f_expected  = evalQuartic(1, a3, a2, a1, a0, roots_expected[p]);
                double rel_error = fabs(f) / fmax(error, fabs(f_expected));
                max_rel_error = fmax(max_rel_error, rel_error);
                ok &= p==0 || roots[p] > roots[p-1];
            }
        }
    }
    std::cout << " quartic equation: error = " << max_rel_error << " epsilons";
    ok &= (max_rel_error <= 2 && nroots_extra + nroots_missed < 150) || err();

    std::cout << '\n';
    return ok;
}

template<typename QRNG>
bool testQRNG(const char* name, int ndim=6, int npoints=1048576)
{
    utils::Timer timer;
    std::vector<QRNG> gens;
    const int ncreate = 10000;
    for(int a=0; a<ncreate; a++) {
        gens.clear();
        for(int d=0; d<ndim; d++)
            gens.push_back(QRNG(d));
    }
    double t1 = timer.deltaSeconds();
    std::vector<double> sum(ndim);
    for(int k=0; k<npoints; k++) {
        for(int d=0; d<ndim; d++) {
            double val = gens[d](k);
            sum[d] += val - 0.5;
        }
    }
    double t2 = timer.deltaSeconds();
    std::vector<double> dif(sum);
    for(int k=npoints-1; k>=0; k--) {
        for(int d=0; d<ndim; d++)
            dif[d] -= gens[d](k) - 0.5;
    }
    double t3 = timer.deltaSeconds();

    std::cout << name << ": create in " << utils::toString(t1/ndim/ncreate * 1e9, 4) << " ns; "
        "sequential eval: " << utils::toString((t2-t1)/ndim/npoints * 1e9, 3) << " ns/point; "
        "reverse order: " << utils::toString((t3-t2)/ndim/npoints * 1e9, 3) << " ns/point; drift: ";
    bool ok = true;
    double totalsum = 0;
    for(int d=0; d<ndim; d++) {
        totalsum += fabs(sum[d]);
        ok &= (fabs(sum[d]) < 2) && (fabs(dif[d]) < fabs(sum[d])*1e-9);
    }
    std::cout << utils::toString(totalsum/ndim, 3) << " ";
    if(!ok)
        err();
    std::cout << "\n";
    return ok;
}

bool testPRNG(int npoints=4194304)
{
    utils::Timer timer;
    double sum1 = 0, sum2 = 0;
    for(int k=0; k<npoints; k++)
        sum1 += math::random() - 0.5;
    double t1 = timer.deltaSeconds();
    math::PRNGState state = 1;
    for(int k=0; k<npoints; k++)
        state = math::hash(&state, 1);
    double t2 = timer.deltaSeconds();
    for(int k=0; k<npoints; k++)
        sum2 += math::random(&state) - 0.5;
    double t3 = timer.deltaSeconds();
    std::cout << "PRNG1: " << utils::toString(t1/npoints * 1e9, 3) << " ns/point; "
        "PRNG2: " << utils::toString((t3-t2)/npoints * 1e9, 3) << " ns/point; "
        "hash: "  << utils::toString((t2-t1)/npoints * 1e9, 3) << " ns/point";
    bool ok = fmax(sum1, sum2) < sqrt(npoints)*3;
    if(!ok)
        err();
    std::cout << "\n";
    return ok;
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
    double maxerrk=0, maxerrs=0, maxerrc=0, maxerra=0;
    for(double ecc=0.; ecc<0.999; ecc = 1-(1-ecc)*0.9) {
        for(double phase=-0.05; phase<1e10; phase<6.4? phase+=0.01 : phase*=1.5) {
            double eta = math::solveKepler(ecc, phase);
            double sineta, coseta;
            math::sincos(eta, sineta, coseta);
            double aeta = math::atan2(sineta, coseta);
            double phasek = eta - ecc * sineta;
            double phasew = math::wrapAngle(phase);
            if(fabs(phasek - phasew)>0.1)
                std::cout << utils::toString(phase, 17) <<
                     ", " << utils::toString(phasek,17) <<
                     ", " << utils::toString(phasew,17) << "\n";
            maxerrk = fmax(maxerrk, fabs(phasek - phasew));
            maxerrs = fmax(maxerrs, fabs(sin(eta) - sineta));
            maxerrc = fmax(maxerrc, fabs(cos(eta) - coseta));
            maxerra = fmax(maxerra, fabs(aeta / (eta > M_PI ? eta - 2*M_PI : eta) - 1));
        }
    }
    std::cout << "Specfunc: E(sin)=" << utils::toString(maxerrs,4);
    ok &= maxerrs < 5e-16 || err();
    std::cout << ", E(cos)=" << utils::toString(maxerrc,4);
    ok &= maxerrc < 5e-16 || err();
    std::cout << ", E(atan)=" << utils::toString(maxerra,4);
    // check special cases
    ok &= math::atan(-INFINITY) == -M_PI/2 && math::atan(+INFINITY) == +M_PI/2 &&
        equal(math::atan2(-1.,-0.), atan2(-1.,-0.)) && equal(math::atan2(-1.,+0.), atan2(-1.,+0 )) &&
        equal(math::atan2(-0.,-1.), atan2(-0.,-1.)) && equal(math::atan2(-0.,+1.), atan2(-0.,+1.)) &&
        equal(math::atan2(-0.,-0.), atan2(-0.,-0.)) && equal(math::atan2(-0.,+0.), atan2(-0.,+0.)) &&
        equal(math::atan2(+0.,-0.), atan2(+0.,-0.)) && equal(math::atan2(+0.,+0.), atan2(+0.,+0.)) &&
        equal(math::atan2(+0.,-1.), atan2(+0.,-1.)) && equal(math::atan2(+0.,+1.), atan2(+0.,+1.)) &&
        equal(math::atan2(+1.,-0.), atan2(+1.,-0.)) && equal(math::atan2(+1.,+0.), atan2(+1.,+0.));
    ok &= maxerra < 5e-16 || err();
    std::cout << ", E(kepler)=" << utils::toString(maxerrk,4);
    ok &= maxerrk < 2e-15 || err();
    std::cout << "\n";

    // test pseudo- and quasi-random number generators
    ok &= testPRNG();
    ok &= testQRNG<math::QuasiRandomHalton>("QRNG-Halton");
    ok &= testQRNG<math::QuasiRandomSobol> ("QRNG-Sobol ");

    // integration routines
    std::cout << "Integration in several variants\n";
    ok &= testIntegration(TestInt1(), -1, 0.5, 0.02, 0.002,1e-4, 1e-9, 1e-9);
    ok &= testIntegration(TestInt2(), -1,2./3, 0.04, 0.02, 1e-4, 0.01, 2e-3);
    ok &= testIntegration(TestInt3(),  0, 1.0, 20.0, 0.01, 1e-4, 0.50, 1e-3);
    ok &= testIntegration(TestInt4(),  0, 1.0, 0.02, 0.001,1e-4, 2e-3, 1e-3);
    ok &= testIntegration(TestInt5(),  0, 1.0, 0.50, 0.05, 1e-4, 0.80, 0.20);

    // low-degree polynomial root-finding
    ok &= testSolvePoly();

    // general root-finding
    const double toler = 1e-6;
    double exact=0.3, error=0, result;
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

    // find coefficients a,b,c of a function f(x) = a*x^b+c from its values at three points
    double A=-1.5, B=-2.0, C=1.0, X1=0.64e4, X2=0.8e4, X3=1e4,
        F1=A*pow(X1,B)+C, F2=A*pow(X2,B)+C, F3=A*pow(X3,B)+C, a, b, c;
    math::findAsymptote(X1, X2, X3, F1, F2, F3, a, b, c);
    std::cout << a << " " << b << " " << c << "\n";
    std::cout << "findAsymptote of f(x)=a*x^b+c: error in a="<<(a-A)<<", b="<<(b-B)<<", c="<<(c-C)<<"\n";
    ok &= fabs(a-A) + fabs(b-B) + fabs(c-C) < toler || err();

    // N-dimensional integration
    numEval=0;
    test8Ndim fnc8;
    integrateNdim(fnc8, fnc8.ymin, fnc8.ymax, toler, 1000000, &result, &error);
    std::cout << "Volume of a 3d torus = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc8.exact)<<"; neval="<<numEval<<")\n";
    ok &= (error < 2.0 && fabs(result-fnc8.exact) < fmin(0.01*result, error)) || err();

    // N-dimensional sampling
    numEval=0;
    math::Matrix<double> points;
    std::vector<double> weights;
    sampleNdim(fnc8, fnc8.ymin, fnc8.ymax, 100000, math::SM_DEFAULT, points, weights, &error);
    result = 0;
    for(size_t i=0; i<weights.size(); i++)
        result += weights[i];
    std::cout << "Monte Carlo Volume of a 3d torus = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc8.exact)<<"; neval="<<numEval<<")\n";
    ok &= (error < 1.0 && fabs(result-fnc8.exact) < fmin(0.01*result, error)) || err();
    if(utils::verbosityLevel >= utils::VL_VERBOSE) {
        std::ofstream fout("sampleNdim.dat");
        for(unsigned int i=0; i<points.rows(); i++)
            fout << points(i,0) << "\t" << points(i,1) << "\t" << points(i,2) << "\n";
    }

#if 0   // these tests fail at the moment
    numEval=0;
    test9Ndim<2> fnc9a;
    double xlow[3] = {-1,-1,-1}, xupp[3] = {1,1,1};
    integrateNdim(fnc9a, xlow, xupp, toler, 10000, &result, &error);
    std::cout << "Integrable singularity in 2d: integral = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc9a.exact)<<"; neval="<<numEval<<")\n";
    ok &= (result > 0 && error < 0.01 && fabs(result-fnc9a.exact) < error) || err();
    numEval=0;
    test9Ndim<3> fnc9b;
    integrateNdim(fnc9b, xlow, xupp, toler, 10000, &result, &error);
    std::cout << "Integrable singularity in 3d: integral = "<<result<<" +- "<<error<<
        " (delta="<<(result-fnc9b.exact)<<"; neval="<<numEval<<")\n";
    ok &= (result > 0 && error < 0.01 && fabs(result-fnc9b.exact) < error) || err();
#endif

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
