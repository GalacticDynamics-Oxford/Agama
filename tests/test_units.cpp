/** \file    test_units.cpp
    \date    June 2015
    \author  Eugene Vasiliev

    This example demonstrates the unit conversion procedures used throughout the library.
    Three different instances of internal units are created,
    some calculation involving gravitational potentials are performed,
    and the output is given back in physical units.
*/
#include "units.h"
#include "potential_analytic.h"
#include "potential_utils.h"
#include <iostream>
#include <cmath>

/// this shows the use of internal units with explicitly applied conversion at each input/output
bool testUnits(const units::InternalUnits& unit) {
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
  double poten_value = pot.value(point);
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

/// this shows the use of ExternalUnits to perform automatic conversions, in the same way
/// as done by various routines in the library (i.e. the user doesn't have to deal with this)
bool testExtUnits(const units::ExternalUnits& converter,
    double galaxyMassInUserUnits,   // these are dimensional quantities in user-provided units,
    double galaxyRadiusInUserUnits, // normally they would be read from a data file
    double galaxyHeightInUserUnits,
    double solarRadiusInUserUnits,
    const char* velocityUnitName) // the name is only for printing purpose
{
    const potential::MiyamotoNagai pot(galaxyMassInUserUnits*converter.massUnit,
        galaxyRadiusInUserUnits*converter.lengthUnit, galaxyHeightInUserUnits*converter.lengthUnit);
    double velEscapeInInternalUnits =
        sqrt(-2*pot.value(coord::PosCar(solarRadiusInUserUnits*converter.lengthUnit, 0., 0.)));
    // the unit of velocity is actually not known to this routine,
    // its meaning is provided by the user that created the instance of ExternalUnits;
    // here we simply print out the user-supplied string
    double velEscapeInUserUnits = velEscapeInInternalUnits/converter.velocityUnit;
    std::cout << "V_esc="<<velEscapeInUserUnits<<' '<<velocityUnitName<<'\n';
    return fabs(velEscapeInUserUnits-546.1)<0.1;
}

int main()
{
    // demonstrate that the results do not depend on the choice of internal unit system
    bool ok =
        testUnits(units::galactic_Myr) &&
        testUnits(units::galactic_kms) &&
        testUnits(units::InternalUnits(units::pc, units::yr*40000));

    const double galaxyMass   = 50;   // in fixed 'external' units - in this case, 1e10 Msun, specified by the user
    const double galaxyRadius = 1e4;  // in parsec (specified by the user)
    const double galaxyHeight = 2e3;  // also in parsec
    const double solarRadius  = 8e3;  // radius of solar orbit in the Galaxy, in parsec
    // here we create an arbitrary instance of internal unit system (again, doesn't matter which particular one),
    // and specify the physical units that our dimensional quantities are in
    ok &= testExtUnits(
        units::ExternalUnits(units::weird_units, 1*units::pc, 1*units::kms, 1e10*units::Msun),
        galaxyMass, galaxyRadius, galaxyHeight, solarRadius, "km/s");

    if(ok)
        std::cout << "\033[1;32mALL TESTS PASSED\033[0m\n";
    else
        std::cout << "\033[1;31mSOME TESTS FAILED\033[0m\n";
    return 0;
}