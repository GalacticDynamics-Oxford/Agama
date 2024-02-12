#!/usr/bin/python
# illustrate the use of different action finders and mappers
import numpy, time
# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

agama.setUnits(length=1, velocity=2, mass=1e6)
numpy.set_printoptions(linewidth=999)
numpy.random.seed(42)

Nact = 100   # unique values of actions
Nang = 1000  # number of angles for each set of actions (i.e. the total number of points is Nact*Nang)
print('%d actions times %d angles' % (Nact, Nang))
act = numpy.repeat(numpy.random.random(size=(Nact, 3)) - numpy.array([0, 0, 0.5]), Nang, axis=0)
ang = numpy.random.random(size=(Nact * Nang, 3)) * 2*numpy.pi
aa = numpy.hstack((act, ang))

# set up three versions of the same potential, which trigger different implementations of action finders/mappers
pot_iso = agama.Potential(type='Isochrone')
pot_sph = agama.Potential(type='Isochrone', rotation=0)   # force the potential to be "generic spherical"
pot_axi = agama.Potential(pot_iso, agama.Potential(type='MiyamotoNagai', mass=0))   # force it to be axisymmetric

af_iso = agama.ActionFinder(pot_iso)  # specialized action finder for the Isochrone potential
af_sph = agama.ActionFinder(pot_sph)  # specialized action finder for a generic spherical potential (interpolated)
af_axi = agama.ActionFinder(pot_axi)  # specialized action finder for an axisymmetric potential (Staeckel fudge)

am_iso = agama.ActionMapper(pot_iso)  # specialized action mapper for the Isochrone potential
am_sph = agama.ActionMapper(pot_sph)  # specialized action mapper for a generic spherical potential
am_axi = agama.ActionMapper(pot_axi)  # specialized action mapper for an axisymmetric potential (Torus)
print(am_axi)  # initially the torus mapping is "empty", i.e. has no tori - they will be created on demand

# measure performance of various action mappers
# (requesting frequencies does not add extra cost, since they are computed anyway)
t0 = time.time()
xv_iso, omm_iso = am_iso(aa, frequencies=True)
t1 = time.time()
xv_sph, omm_sph = am_sph(aa, frequencies=True)
t2 = time.time()
xv_axi, omm_axi = am_axi(aa, frequencies=True)
t3 = time.time()
# repeated call should be much cheaper, since it uses cached tori
xv_axi, omm_axi = am_axi(aa, frequencies=True)
t4 = time.time()
print(am_axi)  # now the torus mapping has cached all 'Nact' tori

# cost of action finders depends on which quantities are needed (actions, angles and/or frequencies).
# isochrone
act_iso = af_iso(xv_iso)  # only actions
t5a = time.time()
omf_iso = af_iso(xv_iso, actions=False, frequencies=True)  # only frequencies
t5b = time.time()
act_iso, omf_iso = af_iso(xv_iso, frequencies=True)  # actions and frequencies
t5c = time.time()
act_iso, ang_iso, omf_iso = af_iso(xv_iso, angles=True, frequencies=True)  # actions, angles and frequencies
t5d = time.time()
# spherical, using interpolated action finder
act_spf = af_sph(xv_iso)
t6a = time.time()
omf_spf = af_sph(xv_iso, actions=False, frequencies=True)
t6b = time.time()
act_spf, omf_spf = af_sph(xv_iso, frequencies=True)
t6c = time.time()
act_spf, ang_spf, omf_spf = af_sph(xv_iso, angles=True, frequencies=True)
t6d = time.time()
# spherical, using standalone action routine
act_sps = agama.actions(pot_sph, xv_iso)
t7a = time.time()
omf_sps = agama.actions(pot_sph, xv_iso, actions=False, frequencies=True)
t7b = time.time()
act_sps, omf_sps = agama.actions(pot_sph, xv_iso, frequencies=True)
t7c = time.time()
act_sps, ang_sps, omf_sps = agama.actions(pot_sph, xv_iso, angles=True, frequencies=True)
t7d = time.time()
# axisymmetric Staeckel fudge
act_axi = af_axi(xv_iso)
t8a = time.time()
omf_axi = af_axi(xv_iso, actions=False, frequencies=True)
t8b = time.time()
act_axi, omf_axi = af_axi(xv_iso, frequencies=True)
t8c = time.time()
act_axi, ang_axi, omf_axi = af_axi(xv_iso, angles=True, frequencies=True)
t8d = time.time()

allok = True
def checkLess(x, limit):
    global allok
    std = numpy.mean(x**2)**0.5
    result = '%.3g' % std
    if std > limit:
        result += ' \033[1;31m**\033[0m'
        allok = False
    return result

print('Action mapping with isochrone: %.3g s' % (t1-t0))
print('Action mapping with spherical: %.3g s, rms error in posvel: %s, frequencies: %s' %
    (t2-t1, checkLess(xv_sph - xv_iso, 1e-5), checkLess(omm_sph - omm_iso, 1e-7)))
print('Action mapping with torus, initial cost: %.3g s, repeated: %.3g s, rms error in posvel: %s, frequencies: %s' %
    (t3-t2, t4-t3, checkLess(xv_axi - xv_iso, 1e-3), checkLess(omm_axi - omm_iso, 1e-4)))
print('Action finding with isochrone: %.3g s, frequencies: %.3g s, actions+freq: %.3g s, act+ang+freq: %.3g s' %
    (t5a-t4,  t5b-t5a, t5c-t5b, t5d-t5c))
print('Isochrone (roundtrip), rms error in actions: %s, angles: %s, frequencies: %s' %
    (checkLess(act_iso - act, 1e-13), checkLess(ang_iso - ang, 1e-13), checkLess(omf_iso - omm_iso, 1e-13)))
print('Action finding with spherical (interpolated): %.3g s, frequencies: %.3g s, actions+freq: %.3g s, act+ang+freq: %.3g s' %
    (t6a-t5d, t6b-t6a, t6c-t6b, t6d-t6c))
print('Spherical (interpol.), rms error in actions: %s, angles: %s, frequencies: %s' %
    (checkLess(act_spf - act, 1e-7), checkLess(ang_spf - ang, 2e-6), checkLess(omf_spf - omm_iso, 1e-6)))
print('Action finding with spherical (standalone):   %.3g s, frequencies: %.3g s, actions+freq: %.3g s, act+ang+freq: %.3g s' %
    (t7a-t6d, t7b-t7a, t7c-t7b, t7d-t7c))
print('Spherical (st/alone),  rms error in actions: %s, angles: %s, frequencies: %s' %
    (checkLess(act_sps - act, 1e-10), checkLess(ang_sps - ang, 1e-11), checkLess(omf_sps - omm_iso, 1e-11)))
print('Action finding with axisym. Staeckel fudge:  %.3g s, frequencies: %.3g s, actions+freq: %.3g s, act+ang+freq: %.3g s' %
    (t8a-t7d, t8b-t8a, t8c-t8b, t8d-t8c))
print('Axisym Staeckel fudge, rms error in actions: %s, angles: %s, frequencies: %s' %
    (checkLess(act_axi - act, 1e-3), checkLess(ang_axi - ang, 5e-2), checkLess(omf_axi - omm_iso, 1e-3)))

if allok:
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
