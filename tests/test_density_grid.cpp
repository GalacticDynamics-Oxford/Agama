/** \name   test_density_grid.cpp
    \author Eugene Vasiliev
    \date   2017

    Test the spatial density discretization scheme for Schwarzschild modelling
*/
#include "galaxymodel_densitygrid.h"
#include "potential_ferrers.h"
#include "utils.h"
#include <cmath>
#include <numeric>
#include <iostream>

// mass, radius and axis ratios of triaxial Ferrers model
const double
    mass     = 2.345,
    radius   = 0.8765,
    axisYtoX = 0.8,
    axisZtoX = 0.6,
    rmult    = radius * cbrt(axisYtoX * axisZtoX);
// number of shells
const int NR = 10;
// radii of Ferrers model enclosing mass equal to 0.1, 0.2, ..., 1
const double radii[NR] = {
    rmult * 0.2939973372408013,
    rmult * 0.3795586772117003,
    rmult * 0.4447632539265056,
    rmult * 0.5013744085912341,
    rmult * 0.5540282084616279,
    rmult * 0.6055625378937006,
    rmult * 0.6585714934216137,
    rmult * 0.7167206042341939,
    rmult * 0.7885680919310256,
    rmult * 1.0 };

bool check(const galaxymodel::TargetDensityClassic<1>& grid, double x, double y, double z,
    int c1, int c2, int c3, int c4)
{
    const int N=grid.numValues(), P=(N-1) / NR;
    double p[3] = {x, y*axisYtoX, z*axisZtoX}, r=sqrt(x*x+y*y+z*z);
    std::vector<double> v(N);
    grid.eval(p, &v[0]);
    double sum = 0.;
    int s=0;
    while(s<NR-1 && radii[s]<r)  ++s;
    int i1 = c1 + s*P+1, i2 = c2 + s*P+1, i3 = c3 + s*P+1, i4 = c4 + s*P+1,
        i5 = std::max(0, i1-P), i6 = std::max(0, i2-P), i7 = std::max(0, i3-P), i8 = std::max(0, i4-P);
    for(int i=0; i<N; i++) {
        if((i==i1||i==i2||i==i3||i==i4||i==i5||i==i6||i==i7||i==i8) ^ (v[i]!=0))
            return false;
        sum += v[i];
    }
    return fabs(sum-1.)<1e-14;
}

bool test(const galaxymodel::BaseTargetDensity& grid)
{
    const int N=grid.numValues();
    std::vector<double> v(N);
    bool ok=true;
    for(int i=0; i<N; i++) {
        std::string str = grid.coefName(i);  // has the form "x=... y=... z=..."
        // parse the coordinates, and also slightly reduce the numbers
        // to avoid falling out of the grid for the outermost shell
        double p[3] = {
            utils::toDouble(str.substr(str.find("x=")+2)) * 0.999999,
            utils::toDouble(str.substr(str.find("y=")+2)) * 0.999999,
            utils::toDouble(str.substr(str.find("z=")+2)) * 0.999999 };
        grid.eval(p, &v[0]);
        // check that only the indicated basis function is (close to) unity at this point,
        // and all others are nearly zero (up to float-to-string conversion roundoff)
        for(int j=0; j<N; j++)
            if(fabs(v[j]-(i==j)) > 1e-4) {
                ok=false;
                std::cout << str << " : " << i << ", " << j << " => " << v[j] << '\n';
            }
    }
    return ok;
}

int main()
{
    potential::Ferrers dens(mass, radius, axisYtoX, axisZtoX);
    std::vector<double> rad(radii, radii + NR);
    galaxymodel::TargetDensityClassic<0> grid0(4, rad, axisYtoX, axisZtoX);
    galaxymodel::TargetDensityClassic<1> grid1(4, rad, axisYtoX, axisZtoX);
    bool ok = true;
    ok &= check(grid1, 0.51, 0.01, 0.50, 15, 16, 44, 49);
    ok &= check(grid1, 0.01, 0.51, 0.50, 23, 24, 28, 29);
    ok &= check(grid1, 0.01, 0.50, 0.51, 55, 56, 24, 29);
    ok &= check(grid1, 0.50, 0.51, 0.01, 35, 36,  4,  9);
    ok &= check(grid1, 0.50, 0.50, 0.51, 58, 59, 39, 60);
    ok &= check(grid1, 0.51, 0.50, 0.50, 18, 19, 59, 60);
    ok &= check(grid1, 0.50, 0.51, 0.50, 38, 39, 19, 60);
    ok &= check(grid1, 0.10, 0.10, 0.11, 58, 59, 39, 60);
    ok &= test (grid0);
    ok &= test (grid1);
    std::vector<double> masses = grid0.computeDensityProjection(dens);
    double sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum - mass) < 1e-10;
    masses = grid1.computeDensityProjection(dens);
    sum = std::accumulate(masses.begin(), masses.end(), 0.);
    ok &= fabs(sum - mass) < 1e-10;
    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}
