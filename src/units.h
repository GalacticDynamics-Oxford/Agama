#pragma once

namespace units {

  // base astronomical units expressed in CGS units
  const double 
    pc           = 3.08568e18,       ///< parsec in cm
    Msun         = 1.98855e33,       ///< solar mass in gram
    yr           = 60*60*24*365.25,  ///< Julian year in seconds
    kms          = 1.e5,             ///< velocity in km/s
    Kpc          = 1.e3*pc,          ///< kiloparsec
    Mpc          = 1.e6*pc,          ///< megaparsec
    c_light      = 2.99792458e10,    ///< speed of light
    ly           = c_light*yr,       ///< light-year in cm
    Myr          = 1.e6*yr,          ///< megayear
    Gyr          = 1.e9*yr,          ///< megayear
    Kpc_kms      = Kpc*kms,          ///< angular momentum
    Msun_per_pc2 = Msun/(pc*pc),     ///< surface density
    Msun_per_pc3 = Msun/(pc*pc*pc),  ///< volume density
    Msun_per_Kpc2= Msun/(Kpc*Kpc),   ///< surface density
    Msun_per_Kpc3= Msun/(Kpc*Kpc*Kpc),///< volume density
    Gev_per_cm3  = 1.782662e-24,     ///< volume density in g/cm^3
    Grav  = 6.67384e-8;              ///< gravitational constant in CGS

  class Units {
  private:
    const double 
      length_unit, mass_unit, time_unit;
  public:
    const double
      from_Msun,
      from_pc,
      from_Kpc,
      from_Mpc,
      from_ly,
      from_yr,
      from_Myr,
      from_Gyr,
      from_kms,
      from_Kpc_kms,
      from_Msun_per_pc2,
      from_Msun_per_pc3,
      from_Msun_per_Kpc2,
      from_Msun_per_Kpc3,
      from_Gev_per_cm3,
      to_Msun,
      to_pc,
      to_Kpc,
      to_Mpc,
      to_ly,
      to_yr,
      to_Myr,
      to_Gyr,
      to_kms,
      to_Kpc_kms,
      to_Msun_per_pc2,
      to_Msun_per_pc3,
      to_Msun_per_Kpc2,
      to_Msun_per_Kpc3,
      to_Gev_per_cm3;
    Units(double length_unit_in_cm, double time_unit_in_s) :
      length_unit(length_unit_in_cm),
      mass_unit(length_unit_in_cm*length_unit_in_cm*length_unit_in_cm/time_unit_in_s/time_unit_in_s/units::Grav),
      time_unit(time_unit_in_s),
      from_Msun(units::Msun/mass_unit),
      from_pc  (units::pc  /length_unit),
      from_Kpc (units::Kpc /length_unit),
      from_Mpc (units::Mpc /length_unit),
      from_ly  (units::ly  /length_unit),
      from_yr  (units::yr  /time_unit),
      from_Myr (units::Myr /time_unit),
      from_Gyr (units::Gyr /time_unit),
      from_kms (units::kms /(length_unit/time_unit)),
      from_Kpc_kms (units::Kpc_kms/(length_unit*length_unit/time_unit)),
      from_Msun_per_pc2 (units::Msun_per_pc2 /(mass_unit/length_unit/length_unit)),
      from_Msun_per_pc3 (units::Msun_per_pc3 /(mass_unit/length_unit/length_unit/length_unit)),
      from_Msun_per_Kpc2(units::Msun_per_Kpc2/(mass_unit/length_unit/length_unit)),
      from_Msun_per_Kpc3(units::Msun_per_Kpc3/(mass_unit/length_unit/length_unit/length_unit)),
      from_Gev_per_cm3  (units::Gev_per_cm3  /(mass_unit/length_unit/length_unit/length_unit)),
      to_Msun(1./from_Msun),
      to_pc  (1./from_pc),
      to_Kpc (1./from_Kpc),
      to_Mpc (1./from_Mpc),
      to_ly  (1./from_ly),
      to_yr  (1./from_yr),
      to_Myr (1./from_Myr),
      to_Gyr (1./from_Gyr),
      to_kms (1./from_kms),
      to_Kpc_kms(1./from_Kpc_kms),
      to_Msun_per_pc2(1./from_Msun_per_pc2),
      to_Msun_per_pc3(1./from_Msun_per_pc3),
      to_Msun_per_Kpc2(1./from_Msun_per_Kpc2),
      to_Msun_per_Kpc3(1./from_Msun_per_Kpc3),
      to_Gev_per_cm3(1./from_Gev_per_cm3)
    {};  // empty constructor, apart from the initialization list
  };

  static const Units galactic_Myr(units::Kpc, units::Myr);
  static const Units galactic_kms(units::Kpc, units::Kpc/units::kms);
  static const Units weird_units(2.71828*units::pc, 42*units::ly/units::kms);

}  // namespace units
