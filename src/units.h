/** \file    units.h
    \brief   Unit systems
    \author  Eugene Vasiliev
    \date    2015
*/
#pragma once

/** Unit systems */
namespace units {

/// \name   base astronomical units expressed in CGS units
///@{
const double 
    pc           = 3.08568e18,        ///< parsec in cm
    Msun         = 1.98855e33,        ///< solar mass in gram
    yr           = 60*60*24*365.25,   ///< Julian year in seconds
    kms          = 1.e5,              ///< velocity in km/s
    Kpc          = 1.e3*pc,           ///< kiloparsec
    Mpc          = 1.e6*pc,           ///< megaparsec
    c_light      = 2.99792458e10,     ///< speed of light
    ly           = c_light*yr,        ///< light-year in cm
    Myr          = 1.e6*yr,           ///< megayear
    Gyr          = 1.e9*yr,           ///< megayear
    Kpc_kms      = Kpc*kms,           ///< angular momentum
    Msun_per_pc2 = Msun/(pc*pc),      ///< surface density
    Msun_per_pc3 = Msun/(pc*pc*pc),   ///< volume density
    Msun_per_Kpc2= Msun/(Kpc*Kpc),    ///< surface density
    Msun_per_Kpc3= Msun/(Kpc*Kpc*Kpc),///< volume density
    Gev_per_cm3  = 1.782662e-24,      ///< volume density in Gev/cm^3
    Grav         = 6.67384e-8;        ///< gravitational constant in CGS
///@}

/** Unit system and conversion class.
    A global instance of this class should be used throughout the code for conversion
    between internal units (in which there are two independent dimensional scales,
    namely length and time, and the gravitational constant equals unity)
    and physical units.
    The choice of two dimensional scales should not matter for computing any quantity
    that is correctly converted to physical units; this freedom of choice enables 
    to check the invariance of results w.r.t the internal unit system.
    Once this choice is made at the beginning of the program, by creating an instance 
    of `units::InternalUnits` class (let it be named `modelUnit`), the following rules apply:
    to convert from physical units to internal units, one should multiply the dimensional 
    quantity (e.g. velocity of a star, expressed in km/s) by `modelUnit.from_***` 
    (in this case, `from_kms`).
    To convert back from model units to physical units, one multiplies by `modelUnit.to_***`.
    These multiplications can be chained to represent a dimensional unit that is not 
    listed as a member of this class, for instance, one may multiply by 
    `modelUnit.to_Kpc/modelUnit.to_Myr` to obtain the velocity expressed in Kpc/Myr.
    Finally, one may simply use e.g. `1.0*modelUnit.to_Msun` to obtain the value of 
    scaling parameters in requested physical units.
*/
struct InternalUnits {
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
    /// Create an internal unit system from two scaling parameters -- length and time scales
    InternalUnits(double length_unit_in_cm, double time_unit_in_s) :
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

/// standard galactic units with length scale of 1 kpc and time scale of 1 Myr
static const InternalUnits galactic_Myr(units::Kpc, units::Myr);

/// standard galactic units with length scale of 1 kpc and velocity scale of 1 km/s
static const InternalUnits galactic_kms(units::Kpc, units::Kpc/units::kms);

/// manifestly non-standard units for testing the invariance of results w.r.t the choice of unit system
static const InternalUnits weird_units(2.71828*units::pc, 42.*units::ly/units::kms);


/** Specification of external unit system for converting external data to internal units.
    The input data that arrives from various sources can have different conventions regarding 
    the dimensional units. In particular, it does not need to comply to the 'pure dynamical'
    convention adopted throughout the code, namely that G=1, thus it has in general three 
    free parameters, which could be taken to be e.g. length, velocity and mass scales.
    This class is designed to convert between positions, velocities and masses of an N-body 
    snapshot and the internal unit system, according to the following procedure.
    To load a snapshot in which these quantities are expressed e.g. 
    in kiloparsecs, km/s and solar masses, and convert the data into the internal unit system 
    specified by a global instance  of `units::InternalUnits` class (let's name it `modelUnit`), 
    one has to create an instance of the conversion class 
    `units::ExternalUnits extUnit (modelUnit, 1.0*units::Kpc, 1.0*units.kms, 1.0*units.Msun);`.
    This conversion class would need to be passed to routines for reading/writing N-body snapshots
    (in particles_io.h) and for constructing a potential approximation from N-body snapshots 
    (in potential_factory.h, in this case, as a member of potential::ConfigPotential class).
*/
struct ExternalUnits {
    const double lengthUnit;   ///< length unit of the external dataset, expressed in internal units
    const double velocityUnit; ///< velocity unit of the external dataset, expressed in internal units
    const double massUnit;     ///< mass unit of the external dataset, expressed in internal units
    const double timeUnit;     ///< time unit of the external dataset, expressed in internal units
    /// construct a trivial converter, for the case that no conversion is actually needed
    ExternalUnits() :
        lengthUnit(1.), velocityUnit(1.), massUnit(1.), timeUnit(1.) {};
    /** construct a converter for the given internal unit system and specified external units,
        the latter expressed in CGS unit system using the constants defined in this header file */
    ExternalUnits(const InternalUnits& unit, double _lengthUnit, double _velocityUnit, double _massUnit) :
        lengthUnit(unit.from_pc*_lengthUnit/pc), 
        velocityUnit(unit.from_kms*_velocityUnit/kms), 
        massUnit(unit.from_Msun*_massUnit/Msun),
        timeUnit(lengthUnit/velocityUnit) {};
};
}  // namespace units
