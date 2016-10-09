#!/usr/bin/python

# this demo script shows how to use Agama library in galpy:
# creating a galpy-compatible potential, integrating an orbit and using the action finder

import galpy
from galpy.potential import *
from galpy.actionAngle import *
from galpy.orbit import *
import agama        # this allows access to potential, action finder, orbit integration, etc. as standalone classes and routines
import galpy_agama  # this enables galpy-compatible interface to potentials
import numpy, matplotlib, matplotlib.pyplot as plt, time
matplotlib.rcParams['legend.frameon']=False

###1. set up galpy potential
g_bulge = PowerSphericalPotentialwCutoff(alpha=1.8, rc=1.9/8, amp=0.03)
g_disk  = MiyamotoNagaiPotential(a=3./8, b=0.28/8, amp=0.755)
g_halo  = NFWPotential(a=2., amp=4.85)
g_pot   = [g_bulge,g_disk,g_halo]   # same as MWPotential2014

###2. set up equivalent potential from the Agama library
agama.setUnits( mass=1., length=8., velocity=345.67)
p_bulge = {"type":"SpheroidDensity", "densityNorm":6.669e9,
    "gamma":1.8, "beta":1.8, "scaleRadius":1, "outerCutoffRadius":1.9/8};
p_disk  = {"type":"MiyamotoNagai", "mass":1.678e11, "scaleradius":3./8, "scaleheight":0.28/8};
p_halo  = {"type":"SpheroidDensity", "densityNorm":1.072e10,
    "gamma":1.0, "beta":3.0, "scaleRadius":2.};
### one can create the genuine instance of Agama potential as follows:
#c_pot   = agama.Potential("../data/MWPotential2014.ini")   # read parameters from ini file
#c_pot   = agama.Potential(p_bulge, p_disk, p_halo) # or create potential from a list of parameters
### or one can instead create a galpy-compatible potential as follows:
w_pot   = galpy_agama.CPotential(p_bulge, p_disk, p_halo)  # same as above, two variants
### ...and then use _pot member variable to access the instance of raw Agama potential
dt = time.time()
c_actfinder = agama.ActionFinder(w_pot._pot)
print 'Time to set up action finder: %s s' % (time.time()-dt)
### this needs to be done once for the given potential,
### and initializes the interfocal distance estimator for all values of E and L

### conversion from prolate spheroidal to cylindrical coords
def ProlSphToCyl(la, nu, ifd):
    return ( ((la - ifd*ifd) * (1 - abs(nu)/ifd**2))**0.5, (la*abs(nu))**0.5 / ifd * numpy.sign(nu) )

### show coordinate grid in prolate spheroidal coords
def plotCoords(ifd, maxR):
    la = numpy.linspace(0, maxR, 32)**2 + ifd**2
    ls = numpy.linspace(0, 1, 21)
    nu = ls*ls*(3-2*ls)*ifd**2
    for i in range(len(la)):
        lineR, linez = ProlSphToCyl(la[i], nu, ifd)
        plt.plot(lineR, linez, 'r', lw=0.5)
        plt.plot(lineR,-linez, 'r', lw=0.5)
    for i in range(len(nu)):
        lineR, linez = ProlSphToCyl(la, nu[i], ifd)
        plt.plot(lineR, linez, 'r', lw=0.5)
        plt.plot(lineR,-linez, 'r', lw=0.5)

### ic is the array of initial conditions: R, z, phi, vR, vz, vphi
def compare(ic, inttime, numsteps):
    g_orb_obj = galpy.orbit.Orbit([ic[0],ic[3],ic[5],ic[1],ic[4],ic[2]])
    times = numpy.linspace(0, inttime, numsteps)
    dt = time.time()
    g_orb_obj.integrate(times, g_pot)
    g_orb = g_orb_obj.getOrbit()
    print 'Time to integrate orbit in galpy: %s s' % (time.time()-dt)

    dt = time.time()
    c_orb_car = agama.orbit(ic=[ic[0],0,ic[1],ic[3],ic[5],ic[4]], pot=w_pot._pot, time=inttime, step=inttime/numsteps)
    print 'Time to integrate orbit in Agama: %s s' % (time.time()-dt)
    times_c = numpy.linspace(0,inttime,len(c_orb_car[:,0]))
    ### make it compatible with galpy's convention (native output is in cartesian coordinates)
    c_orb = c_orb_car*1.0
    c_orb[:,0] = (c_orb_car[:,0]**2+c_orb_car[:,1]**2)**0.5
    c_orb[:,3] = c_orb_car[:,2]

    ### in galpy, this is the only tool that can estimate interfocal distance,
    ### but it requires the orbit to be computed first
    delta = estimateDeltaStaeckel(g_orb[:,0], g_orb[:,3], pot=g_pot)
    print "interfocal distance Delta=",delta

    ### plot the orbit(s) in R,z plane, along with the prolate spheroidal coordinate grid
    plt.axes([0.05, 0.55, 0.45, 0.45])
    plotCoords(delta, 1.5)
    plt.plot(g_orb[:,0],g_orb[:,3], 'b', label='galpy')  # R,z
    plt.plot(c_orb[:,0],c_orb[:,3], 'g', label='Agama')  # R,z
    plt.xlabel("R/8kpc")
    plt.ylabel("z/8kpc")
    plt.xlim(0, 1.2)
    plt.ylim(-1,1)
    plt.legend()

    ### plot R(t), z(t)
    plt.axes([0.55, 0.55, 0.45, 0.45])
    plt.plot(times, g_orb[:,0], label='R')
    plt.plot(times, g_orb[:,3], label='z')
    plt.xlabel("t")
    plt.ylabel("R,z")
    plt.legend()
    plt.xlim(0,50)

    ### create galpy action/angle finder for the given value of Delta
    ### note: using c=False in the routine below is much slower but apparently more accurate,
    ### comparable to the Agama for the same value of delta
    g_actfinder = galpy.actionAngle.actionAngleStaeckel(pot=g_pot, delta=delta, c=True)

    ### find the actions for each point of the orbit
    dt = time.time()
    g_act = g_actfinder(g_orb[:,0],g_orb[:,1],g_orb[:,2],g_orb[:,3],g_orb[:,4],fixed_quad=True)
    print 'Time to compute actions in galpy: %s s' % (time.time()-dt)

    ### use the Agama action routine for the same value of Delta as in galpy
    dt = time.time()
    c_act = agama.actions(point=c_orb_car, pot=w_pot._pot, ifd=delta)   # explicitly specify interfocal distance
    print 'Time to compute actions in Agama: %s s' % (time.time()-dt)

    ### use the Agama action finder (initialized at the beginning) that automatically determines the best value of Delta
    dt = time.time()
    a_act = c_actfinder(c_orb_car)   # use the interfocal distance estimated by action finder
    print 'Time to compute actions in Agama: %s s' % (time.time()-dt)

    ### plot Jr vs Jz
    plt.axes([0.05, 0.05, 0.45, 0.45])
    plt.plot(g_act[0],g_act[2], label='galpy')
    plt.plot(c_act[:,0],c_act[:,1], label=r'Agama,$\Delta='+str(delta)+'$')
    plt.plot(a_act[:,0],a_act[:,1], label=r'Agama,$\Delta=$auto')
    plt.xlabel("$J_r$")
    plt.ylabel("$J_z$")
    plt.legend()

    ### plot Jr(t) and Jz(t)
    plt.axes([0.55, 0.05, 0.45, 0.45])
    plt.plot(times, g_act[0], label='galpy', c='b')
    plt.plot(times, g_act[2], c='b')
    plt.plot(times_c, c_act[:,0], label='Agama,$\Delta='+str(delta)+'$', c='g')
    plt.plot(times_c, c_act[:,1], c='g')
    plt.plot(times_c, a_act[:,0], label='Agama,$\Delta=$auto', c='r')
    plt.plot(times_c, a_act[:,1], c='r')
    plt.text(0, c_act[0,0], '$J_r$', fontsize=16)
    plt.text(0, c_act[0,1], '$J_z$', fontsize=16)
    plt.xlabel("t")
    plt.ylabel("$J_r, J_z$")
    plt.legend(loc='center right')
    plt.xlim(0,50)
    plt.show()

compare([0.5, 0, 0, 0.82, 1.0, 0.28], 100., 1000)
