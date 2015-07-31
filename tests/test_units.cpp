#include "units.h"
#include "potential_analytic.h"
#include <iostream>
#include <cmath>

bool test_units(const units::Units& unit) {
  double mass=5e11*unit.from_Msun;
  double scale_radius=10.*unit.from_Kpc;
  double scale_height=2. *unit.from_Kpc;
  std::cout << "Units: "
    "1 mass u.="<<unit.to_Msun<<" Msun, "
    "1 len.u.=" <<unit.to_Kpc <<" Kpc, "
    "1 time u.="<<unit.to_Myr <<" Myr, "
    "1 vel.u.=" <<unit.to_kms <<" km/s\n";
  const potential::MiyamotoNagai pot(mass, scale_radius, scale_height);
  double solar_radius=8.0*unit.from_Kpc;
  const coord::PosCar point(solar_radius, 0., 0.);
  double poten_value = potential::value(pot, point);
  double v_escape = sqrt(-2*poten_value);
  double v_circ   = potential::v_circ(pot, solar_radius);
  double dens     = pot.density(coord::toPosCyl(point));
  double galactic_year = 2*M_PI*solar_radius / v_circ;
  double v_escape_in_kms = v_escape*unit.to_kms;
  double galactic_year_in_Myr = galactic_year*unit.to_Myr;
  double angmom_in_Kpc_kms = solar_radius*v_circ*unit.to_Kpc_kms;
  double density_in_Msun_per_pc3 = dens*unit.to_Msun_per_pc3;
  std::cout <<
    "V_esc="<<v_escape_in_kms<<" km/s, "
    "galactic year="<<galactic_year_in_Myr<<" Myr, "
    "angular momentum="<<angmom_in_Kpc_kms<<" Kpc*km/s, "
    " or "<<solar_radius*v_circ*unit.to_Kpc*unit.to_Kpc/unit.to_Myr<<" Kpc^2/Myr, "
    "density="<<density_in_Msun_per_pc3<<" Msun/pc3\n";
  return 
    fabs(v_escape_in_kms-546.1)<0.1 &&
    fabs(galactic_year_in_Myr-229.5)<0.1 &&
    fabs(angmom_in_Kpc_kms-1713.5)<0.5 &&
    fabs(density_in_Msun_per_pc3-0.0939)<0.0001;
}

int main()
{
  if(test_units(units::galactic_Myr) &&
     test_units(units::galactic_kms) &&
     test_units(units::Units(units::pc, units::yr*40000)))
      std::cout << "ALL TESTS PASSED\n";
  return 0;
}