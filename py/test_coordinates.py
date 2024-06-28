#!/usr/bin/python

'''
This example compares the celestial coordinate transformations provided by Agama with those from Astropy.
Agama provides only a few basic transformations: between two celestial reference frames (i.e. rotation),
between celestial and Cartesian coordinates centered on the observer, and between standard Galactocentric
Cartesian coordinates and Galactic celestial coordinates (the latter includes the spatial and velocity
shifts).
But it may be much faster than Astropy for small datasets, because it avoids overheads of the large
framework (in particular, Agama does not keep track of units: all angles must be in radians,
and length, velocity and time scales should be in agreement - e.g., if distances are in kpc,
and velocities are in km/s, then the proper motions are in units of km/s/kpc = 0.211 mas/yr).
'''
try:
    import numpy, astropy.coordinates as coord, astropy.units as unit
except ImportError:
    import sys
    sys.exit("\033[1;33mSKIPPED DUE TO ASTROPY IMPORT ERROR\033[0m")

# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

numpy.random.seed(42)         # make experiments repeatable
unit_angle= unit.radian       # units for celestial coordinates (latitude/longitude) are non-negociable
unit_dist = unit.kpc          # distance and
unit_vlos = unit.km / unit.s  # velocity units may be selected arbitrarily, while
unit_pm   = unit_angle * unit_vlos / unit_dist  # unit of PM follows from the previous two choices

def test(ra, dec, dist, pmra, pmdec, vlos):
    kwargs = dict(ra=ra*unit_angle, dec=dec*unit_angle, distance=dist*unit_dist,
        pm_ra_cosdec=pmra*unit_pm, pm_dec=pmdec*unit_pm, radial_velocity=vlos*unit_vlos)
    try:
        gsky = coord.SkyCoord(frame='icrs', **kwargs).transform_to(coord.Galactic)
    except ValueError:  # old version of astropy
        gsky = coord.ICRS(**kwargs).transform_to(coord.Galactic)
    l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic, ra, dec, pmra, pmdec)
    # check the (near-)equivalence of Astropy and our celestial coordinate transformations
    diff_astropy_sky = numpy.amax([
    abs(gsky.l.to_value(unit_angle) - l % (2*numpy.pi)),
    abs(gsky.b.to_value(unit_angle) - b),
    abs(gsky.pm_l_cosb.to_value(unit_pm) - pml),
    abs(gsky.pm_b.to_value(unit_pm) - pmb) ])
    # check the invertibility of celestial coordinate transformations
    a, d, ma, md = agama.transformCelestialCoords(agama.fromGalactictoICRS, l, b, pml, pmb)
    diff_forward_inverse = numpy.amax([ abs(ra % (2*numpy.pi) - a % (2*numpy.pi)), abs(dec-d), abs(pmra-ma), abs(pmdec-md) ])
    # check the invertibility of transforming to/from Cartesian coords centered on the observer
    x, y, z, vx, vy, vz = agama.getCartesianCoords(l, b, dist, pml, pmb, vlos)
    L, B, D, ML, MB, VL = agama.getCelestialCoords(x, y, z, vx, vy, vz)
    diff_cartesian = numpy.amax([ abs(l % (2*numpy.pi) - L % (2*numpy.pi)), abs(b-B), abs(dist-D), abs(pml-ML), abs(pmb-MB), abs(vlos-VL) ])
    # check the more complicated transformation to/from Galactocentric Cartesian coords
    X, Y, Z, VX, VY, VZ = agama.getGalactocentricFromGalactic(l, b, dist, pml, pmb, vlos)
    L, B, D, ML, MB, VL = agama.getGalacticFromGalactocentric(X, Y, Z, VX, VY, VZ)
    diff_galactocentric = numpy.amax([ abs(l % (2*numpy.pi) - L % (2*numpy.pi)), abs(b-B), abs(dist-D), abs(pml-ML), abs(pmb-MB), abs(vlos-VL) ])
    # compare this conversion with the Astropy transformation from Galactocentric Cartesian coordinates
    # to Galactic sky coordinates; by default, these two frames in Astropy are not quite consistent with
    # each other in a natural way, and some manual adjustments are needed to ensure that this conversion
    # follows the expectations (e.g., that x=y=z=0 corresponds to l=0,b=0).
    kwargs = dict(galcen_distance=8.122*unit_dist, z_sun=0.0208*unit_dist, roll=-3.01077232808e-5*unit.degree,
        galcen_v_sun=coord.CartesianDifferential([12.9,245.6,7.78]*unit_vlos),
        galcen_coord=coord.ICRS(ra=266.4049882865447*unit.degree, dec=-28.93617776179147*unit.degree),
        x=X*unit_dist, y=Y*unit_dist, z=Z*unit_dist,
        v_x=VX*unit_vlos, v_y=VY*unit_vlos, v_z=VZ*unit_vlos)
    try:
        gsky = coord.SkyCoord(frame='galactocentric', **kwargs).transform_to(coord.Galactic)
    except ValueError:  # old version of astropy
        gsky = coord.Galactocentric(**kwargs).transform_to(coord.Galactic)
    diff_astropy_galactocentric = numpy.amax([
    abs(gsky.l.to_value(unit_angle) - l % (2*numpy.pi)),
    abs(gsky.b.to_value(unit_angle) - b),
    abs(gsky.pm_l_cosb.to_value(unit_pm) - pml),
    abs(gsky.pm_b.to_value(unit_pm) - pmb) ])
    print('1 km/s = %.12f mas/yr' % (1*unit_pm).to_value(unit.mas/unit.yr))
    print(('Errors in conversion between ICRS<->Galactic: %.4g, Heliocentric Cartesian<->Sky: %.4g, '+
        'Galactocentric Cartesian<->Galactic: %.4g; Astropy ICRS<->Galactic: %.4g, '
        'Astropy Galactocentric<->Galactic: %.4g') %
        (diff_forward_inverse, diff_cartesian, diff_galactocentric,
        diff_astropy_sky, diff_astropy_galactocentric))
    return (diff_forward_inverse < 1e-11) and (diff_cartesian < 1e-12) and (diff_galactocentric < 1e-12) \
        and(diff_astropy_sky < 1e-12) and (diff_astropy_galactocentric < 1e-11)

N = 1000
if test(
    ra   = numpy.random.uniform(0, 2*numpy.pi, size=N),
    dec  = numpy.random.uniform(-numpy.pi/2, numpy.pi/2, size=N),
    dist = numpy.random.uniform(1,10,size=N),
    pmra = numpy.random.uniform(-100,100,size=N),
    pmdec= numpy.random.uniform(-100,100,size=N),
    vlos = numpy.random.uniform(-100,100,size=N) ):
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
