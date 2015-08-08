#include "math_core.h"
#include <iostream>
#include <iomanip>
#include <cmath>
int numEval=0;

class test1: public math::IFunctionNoDeriv{
    virtual double value(double x) const{
        return 1/sqrt(1-x*x);
    }
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

class test4: public math::IFunction{
public:
    int nd;
    test4(int nder) : nd(nder) {};
    virtual void evalDeriv(double x, double* val, double* der=0, double* =0) const{
        numEval++;
        *val = x-1+1e-3/sqrt(x);
        if(der) *der = 1-0.5e-3/pow(x,1.5);
    }
    virtual unsigned int numDerivs() const { return nd; }
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

int main()
{
    std::cout << std::setprecision(10);
    bool ok=true;

    // integration routines
    const double toler = 1e-6;
    double exact = (M_PI*2/3), error=0, result;
    result = math::integrate(test1(), -1, 1./2, toler, &error, &numEval);
    std::cout << "Int1: naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<2e-3 && fabs(result-exact)<error;
    result = math::integrateAdaptive(test1(), -1, 1./2, toler, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<toler && fabs(result-exact)<error;
    test1 t1;
    math::ScaledIntegrandEndpointSing test1s(t1, -1, 1);
    result = math::integrate(test1s, test1s.y_from_x(-1), test1s.y_from_x(1./2), toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<1e-8 && fabs(result-exact)<error;

    exact = 2.274454287;
    result = math::integrate(test2(), -1, 2./3, toler, &error, &numEval);
    std::cout << "Int1: naive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<2e-2 && fabs(result-exact)<error;
    result = math::integrateAdaptive(test2(), -1, 2./3, toler*15, &error, &numEval);
    std::cout << "), adaptive="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval;
    ok &= fabs(1-result/exact)<toler*15 && fabs(result-exact)<error;
    test2 t2;
    math::ScaledIntegrandEndpointSing test2s(t2, -1, 1);
    result = math::integrate(test2s, test2s.y_from_x(-1), test2s.y_from_x(2./3), toler, &error, &numEval);
    std::cout<<"), scaled="<<result<<" +- "<<error<<" (delta="<<(result-exact)<<", neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<2e-4 && fabs(result-exact)<error;

    // root-finding
    exact=0.3;
    numEval=0;
    result = math::findRoot(test3(0), 0, 0.8, toler);
    std::cout << "Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<toler;
    numEval=0;
    result = math::findRoot(test3(1), 0, 0.8, toler);
    std::cout << "with derivative: Root3="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(1-result/exact)<toler;

    exact=1.000002e-6;
    numEval=0;
    result = math::findRoot(test4(0), 1e-15, 0.8, 1e-8);
    std::cout << "Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<1e-8*0.8;
    numEval=0;
    result = math::findRoot(test4(1), 1e-15, 0.8, 1e-8);
    std::cout << "with derivative: Root4="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<1e-8*0.8;

    double x0 = exact*2;
    double x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToPositive();
    result = test4(0)(x1);
    std::cout << "positive value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && x1>0 && x1<exact && result>0;
    x0 = exact*0.9;
    x1 = x0 + math::PointNeighborhood(test4(0), x0).dxToNegative();
    result = test4(0)(x1);
    std::cout << "negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result<0;
    x1 = x0 + math::PointNeighborhood(test4(1), x0).dxToNegative();
    result = test4(1)(x1);
    std::cout << "(with deriv) negative value at x="<<x1<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result<0;
    
    x0 = 1.00009;
    exact = 1.000100000002;
    x1 = x0 + math::PointNeighborhood(test5(), x0).dxToPositive();
    result = test5()(x1);
    std::cout << "f5: positive value at x="<<exact<<"+"<<(x1-exact)<<", value="<<result<<"\n";
    ok &= math::isFinite(x1+result) && result>0;
    numEval=0;
    result = math::findRoot(test5(), 1, x1, toler);
    std::cout << "Root5="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<toler*exact;
    
    exact=1.000109999998;
    numEval=0;
    result = math::findRoot(test5(), x1, INFINITY, toler);
    std::cout << "Another root="<<result<<" (delta="<<(result-exact)<<"; neval="<<numEval<<")\n";
    ok &= fabs(result-exact)<toler*exact;

    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}
