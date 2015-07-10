#include "mathutils.h"
#include <iostream>
#include <iomanip>
#include <cmath>

double test1(double x, void* param){
    *(static_cast<int*>(param))+=1;
    return 1/sqrt(1-x*x);
}

double test2(double x, void* param){
    *(static_cast<int*>(param))+=1;
    return pow(1-x*x*x*x,-2./3);
}

int main()
{
    std::cout << std::setprecision(10);
    bool ok=true;

    double exact = (M_PI*2/3);
    int nev=0, nev_sc=0;
    double integ = mathutils::integrate(test1, &nev, -1, 1./2);
    double integ_sc = mathutils::integrate_scaled(test1, &nev_sc, -1, 1./2, -1, 1);
    std::cout << "Int1: naive="<<integ<<", scaled="<<integ_sc<<", exact="<<exact<<"; neval="<<nev<<",neval_scaled="<<nev_sc<<"\n";
    ok &= fabs(integ-exact)<5e-3 && fabs(integ_sc-exact)<1e-8;

    exact = 2.274454287;
    nev=0; nev_sc=0;
    integ = mathutils::integrate(test2, &nev, -1, 2./3);
    integ_sc = mathutils::integrate_scaled(test2, &nev_sc, -1, 2./3, -1, 1);
    std::cout << "Int2: naive="<<integ<<", scaled="<<integ_sc<<", exact="<<exact<<"; neval="<<nev<<",neval_scaled="<<nev_sc<<"\n";
    ok &= fabs(integ-exact)<5e-2 && fabs(integ_sc-exact)<5e-4;
    if(ok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}