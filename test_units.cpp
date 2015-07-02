#include "units.h"
#include "potential_analytic.h"
#include <iostream>
#include <cmath>

int main() {

  //const units::Units unit(units::galactic_Myr);
  //const units::Units unit(units::galactic_kms);
  const units::Units unit(units::pc, units::yr*40000);
  double mass=1e12*unit.from_Msun;
  double scale_radius=20.*unit.from_Kpc;
  std::cout << "Units: "
    "1 mass u.="<<unit.to_Msun<<" Msun, "
    "1 len.u.=" <<unit.to_Kpc <<" Kpc, "
    "1 time u.="<<unit.to_Myr <<" Myr, "
    "1 vel.u.=" <<unit.to_kms <<" km/s\n";
  const potential::PlummerPotential pot(mass, scale_radius);
  double solar_radius=8.0*unit.from_Kpc;
  const coord::PosSph point(solar_radius, 1., 2.);
  double poten_value = potential::Phi(pot, point);
  double v_escape = sqrt(-2*poten_value);
  double v_circ   = potential::v_circ(pot, solar_radius);
  double dens     = pot.density(coord::toPosCyl(point));
  double galactic_year = 2*M_PI*solar_radius / v_circ;
  std::cout <<
    "V_esc="<<v_escape*unit.to_kms<<" km/s, "
    "galactic year="<<galactic_year*unit.to_Myr<<" Myr, "
    "angular momentum="<<solar_radius*v_circ*unit.to_Kpc_kms<<" Kpc*km/s, "
    " or "<<solar_radius*v_circ*unit.to_Kpc*unit.to_Kpc/unit.to_Myr<<" Kpc^2/Myr, "
    "density="<<dens*unit.to_Msun_per_pc3<<" Msun/pc3\n";

  return 0;
}