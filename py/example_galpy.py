#!/usr/bin/python
"""
This script illustrates the use of the class agama.GalpyPotential, which is a subclass of
both galpy.potential.Potential and agama.Potential, and provides both interfaces
suitable for orbit integration and action finder routines.
"""

import agama
import galpy.potential, galpy.actionAngle, galpy.orbit
import numpy, time, matplotlib.pyplot as plt

### set up galpy potential
g_pot = galpy.potential.MWPotential2014

### set up equivalent potential from the Agama library, which also provides potential interface for galpy
agama.setUnits( mass=1., length=8., velocity=220)
a_pot = agama.GalpyPotential("../data/MWPotential2014galpy.ini")

### initialization of the action finder needs to be done once for the given potential
dt = time.time()
a_actfinder = agama.ActionFinder(a_pot, interp=False)
print('Time to set up action finder: %.4g s' % (time.time()-dt))
### we have a faster but less accurate "interpolated action finder", which takes a bit longer to initialize
dt = time.time()
i_actfinder = agama.ActionFinder(a_pot, interp=True)
print('Time to set up interpolated action finder: %.4g s' % (time.time()-dt))

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

### ic is the array of initial conditions: R, z, phi, vR, vz, vphi
def compare(ic, inttime, numsteps):
    times = numpy.linspace(0, inttime, numsteps)

    ### integrate the orbit in galpy using MWPotential2014 from galpy
    g_orb_obj = galpy.orbit.Orbit([ic[0],ic[3],ic[5],ic[1],ic[4],ic[2]])
    dt = time.time()
    g_orb_obj.integrate(times, g_pot)
    g_orb = g_orb_obj.getOrbit()
    print('Time to integrate orbit in galpy: %.4g s' % (time.time()-dt))

    ### integrate the orbit with the galpy routine, but using Agama potential instead
    ### (much slower because of repeated transfer of control between C++ and Python
    dt = time.time()
    g_orb_obj.integrate(times[:numsteps//10], a_pot)
    a_orb = g_orb_obj.getOrbit()
    print('Time to integrate 1/10th of the orbit in galpy using Agama potential: %.4g s' % (time.time()-dt))

    ### integrate the same orbit (note different calling conventions - cartesian coordinates as input)
    ### using both the orbit integration routine and the potential from Agama - much faster
    dt = time.time()
    times_c, c_orb_car = agama.orbit(ic=[ic[0],0,ic[1],ic[3],ic[5],ic[4]], \
        potential=a_pot, time=inttime, trajsize=numsteps)
    print('Time to integrate orbit in Agama: %.4g s' % (time.time()-dt))

    ### make it compatible with galpy's convention (native output is in cartesian coordinates)
    c_orb = c_orb_car*1.0
    c_orb[:,0] = (c_orb_car[:,0]**2+c_orb_car[:,1]**2)**0.5
    c_orb[:,3] =  c_orb_car[:,2]

    ### in galpy, this is the only tool that can estimate focal distance,
    ### but it requires the orbit to be computed first
    delta = float(galpy.actionAngle.estimateDeltaStaeckel(g_pot, g_orb[:,0], g_orb[:,3]))
    print('Focal distance estimated from the entire trajectory: Delta=%.4g' % delta)

    ### plot the orbit(s) in R,z plane, along with the prolate spheroidal coordinate grid
    plt.figure(figsize=(12,8))
    plt.axes([0.04, 0.54, 0.45, 0.45])
    plotCoords(delta, 1.5)
    plt.plot(g_orb[:,0],g_orb[:,3], 'b', label='galpy')  # R,z
    plt.plot(c_orb[:,0],c_orb[:,3], 'g', label='Agama')
    plt.plot(a_orb[:,0],a_orb[:,3], 'r', label='galpy using Agama potential')
    plt.xlabel("R/8kpc")
    plt.ylabel("z/8kpc")
    plt.xlim(0, 1.2)
    plt.ylim(-1,1)
    plt.legend(loc='lower left', frameon=False)

    ### create galpy action/angle finder for the given value of Delta
    ### note: using c=False in the routine below is much slower but apparently more accurate,
    ### comparable to the Agama for the same value of delta
    g_actfinder = galpy.actionAngle.actionAngleStaeckel(pot=g_pot, delta=delta, c=True)

    ### find the actions for each point of the orbit using galpy action finder
    dt = time.time()
    g_act = g_actfinder(g_orb[:,0],g_orb[:,1],g_orb[:,2],g_orb[:,3],g_orb[:,4],fixed_quad=True)
    print('Time to compute actions in galpy: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
        (numpy.mean(g_act[0]), numpy.std(g_act[0]), numpy.mean(g_act[2]), numpy.std(g_act[2])))

    ### use the Agama action routine for the same value of Delta as in galpy -
    ### the result is almost identical but computed much faster
    dt = time.time()
    c_act = agama.actions(point=c_orb_car, potential=a_pot, fd=delta)   # explicitly specify focal distance
    print('Time to compute actions in Agama using Galpy-estimated focal distance: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
        (numpy.mean(c_act[:,0]), numpy.std(c_act[:,0]), numpy.mean(c_act[:,1]), numpy.std(c_act[:,1])))

    ### use the Agama action finder (initialized at the beginning) that automatically determines
    ### the best value of Delta (same computational cost as the previous one)
    dt = time.time()
    a_act = a_actfinder(c_orb_car)   # use the focal distance estimated by action finder
    print('Time to compute actions in Agama using pre-initialized focal distance: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
        (numpy.mean(a_act[:,0]), numpy.std(a_act[:,0]), numpy.mean(a_act[:,1]), numpy.std(a_act[:,1])))

    ### use the interpolated Agama action finder (initialized at the beginning) - less accurate but faster
    dt = time.time()
    i_act = i_actfinder(c_orb_car)
    print('Time to compute actions in Agama with interpolated action finder: %.4g s' % (time.time()-dt))
    print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
        (numpy.mean(i_act[:,0]), numpy.std(i_act[:,0]), numpy.mean(i_act[:,1]), numpy.std(i_act[:,1])))

    ### plot Jr vs Jz
    plt.axes([0.54, 0.54, 0.45, 0.45])
    plt.plot(g_act[0],  g_act[2],   c='b', label='galpy')
    plt.plot(c_act[:,0],c_act[:,1], c='g', label=r'Agama,$\Delta=%.4f$'%delta)
    plt.plot(a_act[:,0],a_act[:,1], c='r', label=r'Agama,$\Delta=$auto')
    plt.plot(i_act[:,0],i_act[:,1], c='c', label=r'Agama,interpolated')
    plt.xlabel("$J_r$")
    plt.ylabel("$J_z$")
    plt.legend(loc='lower left', frameon=False)

    ### plot Jr(t) and Jz(t)
    plt.axes([0.04, 0.04, 0.95, 0.45])
    plt.plot(times,   g_act[0],   c='b', label='galpy')
    plt.plot(times,   g_act[2],   c='b')
    plt.plot(times_c, c_act[:,0], c='g', label='Agama,$\Delta=%.4f$'%delta)
    plt.plot(times_c, c_act[:,1], c='g')
    plt.plot(times_c, a_act[:,0], c='r', label='Agama,$\Delta=$auto')
    plt.plot(times_c, a_act[:,1], c='r')
    plt.plot(times_c, i_act[:,0], c='c', label='Agama,interpolated')
    plt.plot(times_c, i_act[:,1], c='c')
    plt.text(0, c_act[0,0], '$J_r$', fontsize=16)
    plt.text(0, c_act[0,1], '$J_z$', fontsize=16)
    plt.xlabel("t")
    plt.ylabel("$J_r, J_z$")
    plt.legend(loc='center right', ncol=2, frameon=False)
    #plt.ylim(0.14,0.25)
    plt.xlim(0,50)
    plt.show()

compare([0.5, 0, 0, 0.82, 1.0, 0.28], 100., 1000)
#compare([0.8, 0, 0, 0.3, 0.223, 0.75], 100., 1000)

