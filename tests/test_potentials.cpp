#include "orbit.h"
#include "potential_analytic.h"
#include "potential_galpot.h"
#include "units.h"
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

const double integr_eps=1e-8;  // integration accuracy parameter
const double eps=1e-6;  // accuracy of comparison

template<typename coordSysT>
bool test_orbit_int(const potential::BasePotential& potential,
    const coord::PosVelT<coordSysT>& initial_conditions,
    const double total_time, const double timestep, const bool output)
{
    std::vector<coord::PosVelT<coordSysT> > traj;
    orbit::integrate(potential, initial_conditions, total_time, timestep, traj, integr_eps);
    double avgH=0, avgLz=0, dispH=0, dispLz=0;
    for(size_t i=0; i<traj.size(); i++) {
        double H =potential::totalEnergy(potential, traj[i]);
        double Lz=coord::Lz(traj[i]);
        avgH +=H;  dispH +=pow_2(H);
        avgLz+=Lz; dispLz+=pow_2(Lz);
        if(output) {
            double xv[6];
            coord::toPosVelCar(traj[i]).unpack_to(xv);
            std::cout << i*timestep<<"   " <<xv[0]<<" "<<xv[1]<<" "<<xv[2]<<"  "<<
                xv[3]<<" "<<xv[4]<<" "<<xv[5]<<"   "<<H<<" "<<Lz<<" "<<potential.density(traj[i])<<"\n";
        }
    }
    avgH/=traj.size();
    avgLz/=traj.size();
    dispH/=traj.size();
    dispLz/=traj.size();
    dispH= sqrt(std::max<double>(0, dispH -pow_2(avgH)));
    dispLz=sqrt(std::max<double>(0, dispLz-pow_2(avgLz)));
    std::cout << coordSysT::name() << 
    ":  E=" <<avgH <<" +- "<<dispH<<
    ",  Lz="<<avgLz<<" +- "<<dispLz<<"\n";
    return dispH<eps && (dispLz<eps || ((potential.symmetry() & potential::BasePotential::ST_AXISYMMETRIC) == 0));
}

bool test_potential(const potential::BasePotential& potential,
    const coord::PosVelCar& initcond,
    const double total_time, const double timestep, const bool output)
{
    bool ok=true;
    ok&= test_orbit_int(potential, coord::toPosVelCar(initcond), total_time, timestep, output);
    ok&= test_orbit_int(potential, coord::toPosVelCyl(initcond), total_time, timestep, output);
    ok&= test_orbit_int(potential, coord::toPosVelSph(initcond), total_time, timestep, output);
    return ok;
}

bool test_galpot(const char* params,
    const coord::PosVelCar& initcond,
    const double total_time, const double timestep, const bool output)
{
    const char* params_file="test_galpot_params.pot";
    std::ofstream out(params_file);
    out<<params;
    out.close();
    const potential::BasePotential* gp = potential::readGalaxyPotential(params_file, units::galactic_Myr);
    std::remove(params_file);
    if(gp==NULL) {
        std::cout<<"Potential not created\n";
        return 0;
    }
    bool ok = test_potential(*gp, initcond, total_time, timestep, output);
    delete gp;
    return ok;
}

const char* test_galpot_params[] = {
"2\n"  // stanard galaxy model
"7.52975e+08 3 0.3 0 0\n"
"1.81982e+08 3.5 0.9 0 0\n"
"2\n"
"9.41496e+10 0.5 0 1.8 0.075 2.1\n"
"1.25339e+07 1 1 3 17 0\n", 
"3\n"  // test various types of discs
"1e10 1 -0.1 0 0\n"   // vertically isothermal
"2e9  3 0    0 0\n"   // vertically thin
"5e9  2 0.2 0.4 0.3\n"// with inner hole and wiggles
"1\n"
"1e12 0.8 1 2 0.04 10\n"  };// log density profile with cutoff

const int numtestpoints=3;
const double init_cond[numtestpoints][6] = {
  {1, 0.5, 0.2, 0.1, 0.2, 0.3},
  {2, 0, 0, 0, 0, 0},
  {0, 0, 1, 0, 0, 0} };

int main() {
  std::cout<<std::setprecision(10);
  const double total_time=1000.;
  const double timestep=1.;
  bool output=false;
  bool allok=true;
  for(int ip=0; ip<numtestpoints; ip++) {
    const coord::PosVelCar initcond(init_cond[ip]);
    std::cout << "   ::::initial position::::";
    for(int k=0; k<6; k++) std::cout<<" "<<init_cond[ip][k];
    std::cout <<"\n";
    std::cout << "   Plummer\n";
    allok &= test_potential(potential::Plummer(10.,5.), initcond, total_time, timestep, output);
    std::cout << "   NFW\n";
    allok &= test_potential(potential::NFW(10.,10.), initcond, total_time, timestep, output);
    std::cout << "   MiyamotoNagai\n";
    allok &= test_potential(potential::MiyamotoNagai(5.,2.,0.2), initcond, total_time, timestep, output);
    std::cout << "   Logarithmic\n";
    allok &= test_potential(potential::Logarithmic(1.,0.01,.8,.5), initcond, total_time, timestep, output);
    std::cout << "   Harmonic\n";
    allok &= test_potential(potential::Logarithmic(1.,.7,.5), initcond, total_time, timestep, output);
    std::cout << "   GalPot\n";
    allok &= test_galpot(test_galpot_params[0], initcond, total_time, timestep, output);
    std::cout << "   GalPot/2\n";
    if(ip!=1)  // test in x-y plane takes forever
      allok &= test_galpot(test_galpot_params[1], initcond, total_time, timestep, output);
  }
  if(allok)
    std::cout << "ALL TESTS PASSED\n";
  return 0;
}