/** \file    units.h
    \brief   Unit systems
    \author  Eugene Vasiliev
    \date    2015
*/

#pragma once

/** Unit systems

    The approach for handling dimensional quantities in the library is based on
    several principles.

    1. All physical quantities inside the code are represented in some internal units.
    A somewhat unusual convention is that we do not use any particular convention for
    these units, apart from the requirement that the gravitational constant is unity:
    this still leaves free  the choice of any two out of three basic units -
    length, time and mass.
    The motivation for this freedom is to ensure that all routines in the library
    are (nearly) scale-invariant, because the underlying physics is.
    Thus no hard-coded cutoff values or absolute tolerances are allowed.
    This makes the code harder to write, but easier to (re)use in different contexts,
    and also more robust.
    These internal units are set up to some values at the beginning of the program,
    by creating an instance of `InternalUnits` class, but are only required for
    data exchange with the external world, not between the routines in the library.
    For instance, to print out the value of gravitational potential at the location
    of the Sun, we may use something like
    ~~~~
    std::cout << "Potential at solar radius: " << 
        potential.value (coords::PosCyl (8*modelUnit.from_Kpc, 0, 0) )
        * pow_2(modelUnit.to_kms) << " (km/s)^2" << std::endl;
    ~~~~
    where we converted 8 Kpc _to_ internal length units, and the value of potential
    _from_ the internal units to (km/s)^2.
    Here `modelUnit` is the global instance of InternalUnits class.

    2. The data in the outside world can be expressed in various units, and the
    conversion to internal units is performed "at the state boundary" of the library.
    All routines that have to deal with this conversion are specially marked,
    and receive an instance of `ExternalUnits` class.
    Notably, this applies to the conversion of any model parameters read from
    configuration files (such as scale radius and mass of the density model,
    or normalization of the distribution function).
    The motivation to introduce another unit class in addition to `InternalUnits`
    is twofold.
    On the one hand, one cannot use the approach of the above code snippet for
    the operations that are performed within the library itself, such as reading
    the data from an N-body snapshot; thus the conversion done by these routines
    is essentially the same as in that example, but uses the conversion coefficients
    provided by the user via the `ExternalUnits` instance.
    On the other hand, one may choose not to specify any conversion at all
    (i.e., a trivial conversion - all factors are 1). This regime is more suitable
    for 'theoretical usage', like creating a Plummer potential with mass and scale
    radius equal to 1, importing an N-body snapshot in standard N-body units, and
    examining all internal quantities without the need to apply any unit conversion
    (which goes somewhat at odds with the motivation from the previous paragraph,
    but is quite handy as long as no observational data is concerned).
*/
namespace units {

/// \name   base astronomical units expressed in CGS units
///@{
const double
    pc           = 3.0856775815e18,   ///< parsec in cm
    Msun         = 1.988409871e33,    ///< solar mass in gram
    yr           = 60*60*24*365.25,   ///< Julian year in seconds
    kms          = 1.e5,              ///< velocity in km/s
    Kpc          = 1.e3*pc,           ///< kiloparsec
    Mpc          = 1.e6*pc,           ///< megaparsec
    c_light      = 2.99792458e10,     ///< speed of light
    arcsec       = 3.141593/180/60/60,///< arcsecond in radians (dimensionless)
    mas_per_yr   = 1e-3*arcsec/yr,    ///< milliarcsecond per year (unit of proper motion)
    ly           = c_light*yr,        ///< light-year in cm
    Myr          = 1.e6*yr,           ///< megayear
    Gyr          = 1.e9*yr,           ///< megayear
    Kpc_kms      = Kpc*kms,           ///< angular momentum
    Msun_per_pc2 = Msun/(pc*pc),      ///< surface density
    Msun_per_pc3 = Msun/(pc*pc*pc),   ///< volume density
    Msun_per_Kpc2= Msun/(Kpc*Kpc),    ///< surface density
    Msun_per_Kpc3= Msun/(Kpc*Kpc*Kpc),///< volume density
    Gev_per_cm3  = 1.782662e-24,      ///< volume density in Gev/cm^3
    Grav         = 6.6743e-8;         ///< gravitational constant in CGS
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
      from_mas_per_yr,
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
      to_mas_per_yr,
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
      mass_unit(length_unit_in_cm*length_unit_in_cm*length_unit_in_cm/time_unit_in_s/time_unit_in_s/Grav),
      time_unit(time_unit_in_s),
      from_Msun(Msun/mass_unit),
      from_pc  (pc  /length_unit),
      from_Kpc (Kpc /length_unit),
      from_Mpc (Mpc /length_unit),
      from_ly  (ly  /length_unit),
      from_yr  (yr  /time_unit),
      from_Myr (Myr /time_unit),
      from_Gyr (Gyr /time_unit),
      from_mas_per_yr(mas_per_yr * time_unit),
      from_kms (kms /(length_unit/time_unit)),
      from_Kpc_kms (Kpc_kms/(length_unit*length_unit/time_unit)),
      from_Msun_per_pc2 (Msun_per_pc2 /(mass_unit/length_unit/length_unit)),
      from_Msun_per_pc3 (Msun_per_pc3 /(mass_unit/length_unit/length_unit/length_unit)),
      from_Msun_per_Kpc2(Msun_per_Kpc2/(mass_unit/length_unit/length_unit)),
      from_Msun_per_Kpc3(Msun_per_Kpc3/(mass_unit/length_unit/length_unit/length_unit)),
      from_Gev_per_cm3  (Gev_per_cm3  /(mass_unit/length_unit/length_unit/length_unit)),
      to_Msun(1./from_Msun),
      to_pc  (1./from_pc),
      to_Kpc (1./from_Kpc),
      to_Mpc (1./from_Mpc),
      to_ly  (1./from_ly),
      to_yr  (1./from_yr),
      to_Myr (1./from_Myr),
      to_Gyr (1./from_Gyr),
      to_mas_per_yr(1./from_mas_per_yr),
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
    This class is designed to convert between positions, velocities and masses of an
    external dataset (e.g., an N-body snapshot) and the internal unit system,
    according to the following procedure.
    To load a snapshot in which these quantities are expressed e.g.
    in kiloparsecs, km/s and solar masses, and convert the data into the internal unit system
    specified by a global instance  of `units::InternalUnits` class (let's name it `modelUnit`),
    one has to create an instance of the conversion class
    `units::ExternalUnits extUnit (modelUnit, 1.0*units::Kpc, 1.0*units::kms, 1.0*units::Msun);`.
    This conversion class would need to be passed to routines for reading/writing N-body snapshots
    (in particles_io.h) and to routines for creating a potential instance from the parameters
    provided in the configuration file (in potential_factory.h)
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
        lengthUnit  (unit.from_pc   * _lengthUnit / pc),
        velocityUnit(unit.from_kms  * _velocityUnit / kms),
        massUnit    (unit.from_Msun * _massUnit / Msun),
        timeUnit    (lengthUnit / velocityUnit) {};
};
}  // namespace units
