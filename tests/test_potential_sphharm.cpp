#include "potential_analytic.h"
#include "potential_sphharm.h"
#include <iostream>
#include <fstream>
#include <cmath>

/// define test suite in terms of points for various coord systems
const int numtestpoints=5;
const double posvel_car[numtestpoints][6] = {
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {0,-1, 2, 0,   0.2,-0.5},   // point in y-z plane 
    {2, 0,-1, 0,  -0.3, 0.4},   // point in x-z plane
    {0, 0, 1, 0,   0,   0.5},   // point along z axis
    {0, 0, 0,-0.4,-0.2,-0.1}};  // point at origin with nonzero velocity
const double posvel_cyl[numtestpoints][6] = {   // order: R, z, phi
    {1, 2, 3, 0.4, 0.2, 0.3},   // ordinary point
    {2,-1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {0, 2, 0, 0,  -0.5, 0  },   // point along z axis, vphi must be zero
    {0,-1, 2, 0.5, 0.3, 0  },   // point along z axis, vphi must be zero, but vR is non-zero
    {0, 0, 0, 0.3,-0.5, 0  }};  // point at origin with nonzero velocity in R and z
const double posvel_sph[numtestpoints][6] = {   // order: R, theta, phi
    {1, 2, 3, 0.4, 0.3, 0.2},   // ordinary point
    {2, 1, 0,-0.3, 0.4, 0  },   // point in x-z plane
    {1, 0, 0,-0.5, 0,   0  },   // point along z axis, vphi must be zero
    {1,3.14159, 2, 0.5, 0.3, 1e-4},   // point almost along z axis, vphi must be small, but vtheta is non-zero
    {0, 2,-1, 0.5, 0,   0  }};  // point at origin with nonzero velocity in R

int main() {
    bool allok=true;
    const potential::Plummer plum(10., 5.);
    potential::BasisSetExp bse(0., 20, 2, plum);
    potential::SplineExp spl(20, 2, plum);
    for(int ic=0; ic<numtestpoints; ic++) {
        double pot;
        coord::GradSph der;
        coord::HessSph der2;
        bse.eval(coord::PosVelSph(posvel_sph[ic]), &pot, &der, &der2);
        allok &= (pot==pot);
        std::cout << pot << " " << der.dr << "\t";
        spl.eval(coord::PosVelSph(posvel_sph[ic]), &pot, &der, &der2);
        allok &= (pot==pot);
        std::cout << pot << " " << der.dr << "\n";
    }
    if(allok)
        std::cout << "ALL TESTS PASSED\n";
    return 0;
}