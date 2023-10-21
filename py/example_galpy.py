#!/usr/bin/python
"""
This script illustrates the use of the class agama.GalpyPotential, which is a subclass of
both galpy.potential.Potential and agama.Potential, and provides both interfaces
suitable for orbit integration and action finder routines.
Internally, this potential can represent either a native galpy potential or a native agama potential,
depending on how it is constructed, but the public API is the same in both cases.
Of course, routines from agama will be much more efficient when this object represents an agama-native
potential internally; the reverse is not necessarily true, since galpy does not know whether this
object internally represents a galpy-native or agama-native potential, and hence C acceleration
will not be used even if the galpy-native potential supports it.
"""

import agama
import galpy.potential, galpy.actionAngle, galpy.orbit
import numpy, time, matplotlib.pyplot as plt

### galpy uses these units, so instruct agama to do the same
agama.setUnits( mass=1., length=8., velocity=220.)  # Msun, kpc, km/s

### set up galpy potential
g_pot_native = galpy.potential.MWPotential2014

### set up a chimera potential that is initialized from the existing galpy potential,
### and provides both galpy and agama interfaces - when used with galpy routines,
### its behaviour (but not performance!) should be identical to g_pot_native
g_pot_hybrid = agama.GalpyPotential(g_pot_native)

### set up equivalent potential from the agama library (up to a constant offset)
a_pot_native = agama.Potential("../data/MWPotential2014galpy.ini")

### set up another chimera that is now initialized from the existing agama potential,
### and again provides both interfaces -- when used with agama classes and routines,
### its behaviour and performance are identical to a_pot_native
a_pot_hybrid = agama.GalpyPotential(a_pot_native)

### note: the same effect is achieved by initializing the hybrid potential directly from the file
a_pot_hybrid = agama.GalpyPotential("../data/MWPotential2014galpy.ini")

### since a galpy-native potential has poor performance in computationally heavy tasks within agama,
### we construct an approximation for it represented by a native CylSpline potential expansion
dt = time.time()
### note: the CylSpline expansion would be better suited for the disky potential, but unfortunately,
### it cannot be used because some of galpy potentials do not return a valid value at r=0;
### instead we use a Multipole expansion (which never needs the value of the original potential at 0);
### it is less accurate in this case, but still acceptable
#g_pot_approx = agama.Potential(type='CylSpline', potential=g_pot_hybrid, symmetry='axi', rmin=0.01, rmax=10, zmin=0.01, zmax=10)
g_pot_approx = agama.Potential(type='Multipole', potential=g_pot_hybrid, symmetry='axi', rmin=0.01, rmax=10, lmax=20)
#g_pot_approx.export('example_galpy.ini')  # may save the potential coefs for later use
print('Time to set up a CylSpline approximation to galpy potential: %.4g s' % (time.time()-dt))

### initialization of the action finder needs to be done once for the given potential
dt = time.time()
a_actfinder = agama.ActionFinder(a_pot_hybrid, interp=False)
print('Time to set up agama action finder: %.4g s' % (time.time()-dt))
### we have a faster but less accurate "interpolated action finder", which takes a bit longer to initialize
dt = time.time()
i_actfinder = agama.ActionFinder(a_pot_hybrid, interp=True)
print('Time to set up agama interpolated action finder: %.4g s' % (time.time()-dt))

### conversion from prolate spheroidal to cylindrical coords
def ProlSphToCyl(la, nu, fd):
    return ( ((la - fd*fd) * (1 - abs(nu)/fd**2))**0.5, (la*abs(nu))**0.5 / fd * numpy.sign(nu) )

### show coordinate grid in prolate spheroidal coords
def plotCoords(fd, maxR):
    la = numpy.linspace(0, maxR, 32)**2 + fd**2
    ls = numpy.linspace(0, 1, 21)
    nu = ls*ls*(3-2*ls)*fd**2
    for i in range(len(la)):
        lineR, linez = ProlSphToCyl(la[i], nu, fd)
        plt.plot(lineR, linez, 'grey', lw=0.5)
        plt.plot(lineR,-linez, 'grey', lw=0.5)
    for i in range(len(nu)):
        lineR, linez = ProlSphToCyl(la, nu[i], fd)
        plt.plot(lineR, linez, 'grey', lw=0.5)
        plt.plot(lineR,-linez, 'grey', lw=0.5)

### convert position/velocity from cylindrical coordinates (in the galpy convention) to cartesian
def toCyl(car):
    x, y, z, vx, vy, vz = numpy.array(car).T
    R = (x**2 + y**2)**0.5
    phi = numpy.arctan2(y, x)
    cosphi, sinphi = numpy.cos(phi), numpy.sin(phi)
    vR   = vx*cosphi + vy*sinphi
    vphi = vy*cosphi - vx*sinphi
    return numpy.array([R, vR, vphi, z, vz, phi]).T

### convert position/velocity from cartesian to cylindrical system in the galpy convention
def toCar(cyl):
    R, vR, vphi, z, vz, phi = numpy.array(cyl).T
    cosphi, sinphi = numpy.cos(phi), numpy.sin(phi)
    return numpy.array([R*cosphi, R*sinphi, z, vR*cosphi-vphi*sinphi, vR*sinphi+vphi*cosphi, vz]).T

### run tests for the given input point and orbit size;
### ic is the array of initial conditions in the galpy convention: R, vR, vphi, z, vz, phi
def compare(ic, inttime, numsteps):
    times = numpy.linspace(0, inttime, numsteps)
    times1 = times[:numsteps//10]   # compute only 1/10th of the orbit to save time in galpy

    ### integrate the orbit in galpy using the native MWPotential2014 from galpy
    g_orb_obj = galpy.orbit.Orbit(ic)
    dt = time.time()
    g_orb_obj.integrate(times, g_pot_native)
    g_orb_g_native = g_orb_obj.getOrbit()
    print('Time to integrate the orbit in galpy, using galpy-native potential [1]: %.4g s' % (time.time()-dt))

    ### integrate the orbit again with the galpy routine, but now using the chimera potential
    ### representing a galpy-native underlying potential.
    ### since galpy has no idea that this potential has a C implementation, it switches to a slower
    ### pure-Python integration algorithm, so to save time, we compute only 1/10th of the orbit
    dt = time.time()
    g_orb_obj.integrate(times1, g_pot_hybrid)
    g_orb_g_hybrid = g_orb_obj.getOrbit()
    print('Time to integrate 1/10th of the orbit in galpy, using galpy-hybrid potential [2]: %.4g s' %
        (time.time()-dt))

    ### integrate the orbit with the galpy routine, but using the chimera potential
    ### representing an agama-native underlying potential instead
    ### (also much slower because of repeated transfer of control between C++ and Python)
    dt = time.time()
    g_orb_obj.integrate(times1, a_pot_hybrid)
    g_orb_a_hybrid = g_orb_obj.getOrbit()
    print('Time to integrate 1/10th of the orbit in galpy, using agama-hybrid potential [3]: %.4g s' %
        (time.time()-dt))

    ### integrate the orbit with the agama routine, using the galpy-native potential through the chimera
    ### (note different calling conventions and the use of cartesian coordinates)
    dt = time.time()
    a_orb_g_hybrid = agama.orbit(potential=g_pot_hybrid, time=inttime, trajsize=numsteps, ic=toCar(ic))[1]
    print('Time to integrate the orbit in agama, using galpy-hybrid potential [4]: %.4g s' % (time.time()-dt))

    ### integrate the orbit in a native agama potential approximation constructed from the galpy potential
    dt = time.time()
    a_orb_g_approx = agama.orbit(potential=g_pot_approx, time=inttime, trajsize=numsteps, ic=toCar(ic))[1]
    print('Time to integrate the orbit in agama, using galpy-approx potential [5]: %.4g s' % (time.time()-dt))

    ### using both the orbit integration routine and the native potential from agama - much faster
    dt = time.time()
    a_orb_a_native = agama.orbit(potential=a_pot_native, time=inttime, trajsize=numsteps, ic=toCar(ic))[1]
    print('Time to integrate the orbit in agama, using agama-native potential [6]: %.4g s' % (time.time()-dt))

    ### the same but with the chimera potential representing an agama-native potential
    ### (should be identical to the above, since in both cases the same underlying C++ object is used)
    dt = time.time()
    a_orb_a_hybrid = agama.orbit(potential=a_pot_hybrid, time=inttime, trajsize=numsteps, ic=toCar(ic))[1]
    print('Time to integrate the orbit in agama, using agama-hybrid potential [7]: %.4g s' % (time.time()-dt))

    ### compare the differences between the orbits computed in different ways
    ### (both the potentials and integration routines are not exactly equivalent);
    ### use only common initial segment (1/10th of the orbit) for comparison
    print('Differences between orbits: ' +
        '[1]-[2]=%g, ' % numpy.max(numpy.abs(toCar(g_orb_g_native[:len(times1)])-toCar(g_orb_g_hybrid))) +
        '[1]-[3]=%g, ' % numpy.max(numpy.abs(toCar(g_orb_g_native[:len(times1)])-toCar(g_orb_a_hybrid))) +
        '[1]-[4]=%g, ' % numpy.max(numpy.abs(toCar(g_orb_g_native)-a_orb_g_hybrid)[:len(times1)]) +
        '[1]-[5]=%g, ' % numpy.max(numpy.abs(toCar(g_orb_g_native)-a_orb_g_approx)[:len(times1)]) +
        '[1]-[6]=%g, ' % numpy.max(numpy.abs(toCar(g_orb_g_native)-a_orb_a_native)[:len(times1)]) +
        '[6]-[4]=%g, ' % numpy.max(numpy.abs(a_orb_a_native-a_orb_g_hybrid)[:len(times1)]) +
        '[6]-[7]=%g'   % numpy.max(numpy.abs(a_orb_a_native-a_orb_a_hybrid)[:len(times1)]) )  # should be zero

    ### convert the orbits to the same galpy coordinate convention
    gg_orb = g_orb_g_native  # it's already in this convention
    ga_orb = g_orb_a_hybrid
    ag_orb = toCyl(a_orb_g_hybrid)
    aa_orb = toCyl(a_orb_a_native)
    ### in galpy, this is the only tool that can estimate focal distance,
    ### but it requires the orbit to be computed first
    delta = float(galpy.actionAngle.estimateDeltaStaeckel(g_pot_hybrid, gg_orb[:,0], gg_orb[:,3]))
    print('Focal distance estimated from the entire trajectory: Delta=%.4g' % delta)

    ### plot the orbit(s) in R,z plane, along with the prolate spheroidal coordinate grid
    plt.figure(figsize=(12,8))
    plt.axes([0.06, 0.56, 0.43, 0.43])
    plotCoords(delta, 1.5)
    plt.plot(gg_orb[:,0],gg_orb[:,3], 'b', label='galpy native')  # R,z
    plt.plot(aa_orb[:,0],aa_orb[:,3], 'g', label='agama native', dashes=[4,2])
    plt.plot(ag_orb[:,0],ag_orb[:,3], 'y', label='agama using galpy potential', dashes=[1,1])
    plt.plot(ga_orb[:,0],ga_orb[:,3], 'r', label='galpy using agama potential')
    plt.xlabel("R/8kpc")
    plt.ylabel("z/8kpc")
    plt.xlim(0, 1.2)
    plt.ylim(-1,1)
    plt.legend(loc='lower left', ncol=2)

    ### create galpy action/angle finder for the given value of Delta
    ### note: using c=False in the routine below is much slower but apparently more accurate,
    ### comparable to the agama for the same value of delta
    g_actfinder = galpy.actionAngle.actionAngleStaeckel(pot=g_pot_native, delta=delta, c=True)

    ### find the actions for each point of the orbit using galpy action finder
    dt = time.time()
    g_act = g_actfinder(gg_orb[:,0],gg_orb[:,1],gg_orb[:,2],gg_orb[:,3],gg_orb[:,4],fixed_quad=True)
    print('Time to compute actions in galpy: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' %
        (numpy.mean(g_act[0]), numpy.std(g_act[0]), numpy.mean(g_act[2]), numpy.std(g_act[2])))

    ### use the agama action routine for the same value of Delta as in galpy (explicity specify focal distance):
    ### the result is almost identical but computed much faster
    dt = time.time()
    c_act = agama.actions(point=a_orb_a_hybrid, potential=a_pot_hybrid, fd=delta)
    print('Time to compute actions in agama using galpy-estimated focal distance: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' %
        (numpy.mean(c_act[:,0]), numpy.std(c_act[:,0]), numpy.mean(c_act[:,1]), numpy.std(c_act[:,1])))

    ### use the agama action finder (initialized at the beginning) that automatically determines
    ### the best value of Delta (same computational cost as the previous one)
    dt = time.time()
    a_act = a_actfinder(a_orb_a_hybrid)   # use the focal distance estimated by action finder
    print('Time to compute actions in agama using pre-initialized focal distance: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' %
        (numpy.mean(a_act[:,0]), numpy.std(a_act[:,0]), numpy.mean(a_act[:,1]), numpy.std(a_act[:,1])))

    ### use the interpolated agama action finder (initialized at the beginning) - less accurate but faster
    dt = time.time()
    i_act = i_actfinder(a_orb_a_hybrid)
    print('Time to compute actions in agama with interpolated action finder: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' %
        (numpy.mean(i_act[:,0]), numpy.std(i_act[:,0]), numpy.mean(i_act[:,1]), numpy.std(i_act[:,1])))

    ### plot Jr vs Jz
    plt.axes([0.55, 0.56, 0.43, 0.43])
    plt.plot(g_act[0],  g_act[2],   c='b', label='galpy')
    plt.plot(c_act[:,0],c_act[:,1], c='g', label=r'agama, $\Delta=%.4f$'%delta, dashes=[4,2])
    plt.plot(a_act[:,0],a_act[:,1], c='r', label=r'agama, $\Delta=$auto')
    plt.plot(i_act[:,0],i_act[:,1], c='c', label=r'agama, interpolated')
    plt.xlabel("$J_r$")
    plt.ylabel("$J_z$")
    plt.legend(loc='lower left', frameon=False)

    ### plot Jr(t) and Jz(t)
    plt.axes([0.06, 0.06, 0.92, 0.43])
    plt.plot(times, g_act[0],   c='b', label='galpy')
    plt.plot(times, g_act[2],   c='b')
    plt.plot(times, c_act[:,0], c='g', label=r'agama, $\Delta=%.4f$'%delta, dashes=[4,2])
    plt.plot(times, c_act[:,1], c='g', dashes=[4,2])
    plt.plot(times, a_act[:,0], c='r', label=r'agama, $\Delta=$auto')
    plt.plot(times, a_act[:,1], c='r')
    plt.plot(times, i_act[:,0], c='c', label=r'agama, interpolated')
    plt.plot(times, i_act[:,1], c='c')
    plt.text(0, c_act[0,0], '$J_r$', fontsize=16)
    plt.text(0, c_act[0,1], '$J_z$', fontsize=16)
    plt.xlabel("t")
    plt.ylabel("$J_r, J_z$")
    plt.legend(loc='center right', ncol=2, frameon=False)
    #plt.ylim(0.14,0.25)
    plt.xlim(0,50)
    plt.show()

compare([0.5, 0.82, 0.28, 0, 1.0, 0], 100., 1000)
