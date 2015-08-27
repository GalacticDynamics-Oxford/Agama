#!/usr/bin/python

import py_wrapper
import numpy
import time
import galpy
from galpy.potential import *
from galpy.actionAngle import *
from galpy.orbit import *
import matplotlib.pyplot as plt

#1. set up galpy potential
g_bulge = PowerSphericalPotentialwCutoff(alpha=1.8, rc=1.9/8, amp=0.03)
g_disk  = MiyamotoNagaiPotential(a=3./8, b=0.28/8, amp=0.755)
g_halo  = NFWPotential(a=2., amp=4.85)
g_pot   = [g_bulge,g_disk,g_halo]   # same as MWPotential2014

#2. set up equivalent potential
py_wrapper.set_units( mass=1., length=8., velocity=345.67)
p_bulge = {"type":"TwoPowerLawSpheroid", "densityNorm":6.669e9,
    "gamma":1.8, "beta":1.8, "scaleRadius":1, "outerCutoffRadius":1.9/8};
p_disk  = {"type":"MiyamotoNagai", "mass":1.678e11, "scaleradius":3./8, "scaleheight":0.28/8};
p_halo  = {"type":"TwoPowerLawSpheroid", "densityNorm":1.072e10,
    "gamma":1.0, "beta":3.0, "scaleRadius":2.};
#c_pot   = py_wrapper.Potential("MWPotential2014.ini")
c_pot   = py_wrapper.Potential(p_bulge, p_disk, p_halo)
c_actfinder = py_wrapper.ActionFinder(c_pot)

# ic is the array of initial conditions: R, z, phi, vR, vz, vphi
def compare(ic, time, numsteps):
    g_orb_obj = galpy.orbit.Orbit([ic[0],ic[3],ic[5],ic[1],ic[4],ic[2]])
    times = numpy.linspace(0, time, numsteps)
    g_orb_obj.integrate(times, g_pot)
    g_orb = g_orb_obj.getOrbit()
    delta = estimateDeltaStaeckel(g_orb[:,0], g_orb[:,3], pot=g_pot)
    print "Delta=",delta
    g_actfinder = galpy.actionAngle.actionAngleStaeckel(pot=g_pot, delta=delta, c=False)

    c_orb_car = py_wrapper.orbit(ic=[ic[0],0,ic[1],ic[3],ic[5],ic[4]], pot=c_pot, time=time, step=time/numsteps)
    # make it compatible with galpy's convention
    c_orb = c_orb_car*1.0
    c_orb[:,0] = (c_orb_car[:,0]**2+c_orb_car[:,1]**2)**0.5
    c_orb[:,3] = c_orb_car[:,2]
    ##c_orb[:,1] = (c_orb_car[:,0]*c_orb_car[:,3]+c_orb_car[:,1]*c_orb_car[:,4])/c_orb[:,0]
    #plt.plot(g_orb[:,0],g_orb[:,3])  # R,z
    #plt.plot(c_orb[:,0],c_orb[:,3])  # R,z
    #plt.show()

    g_act = g_actfinder(g_orb[:,0],g_orb[:,1],g_orb[:,2],g_orb[:,3],g_orb[:,4],fixed_quad=True)
    c_act = py_wrapper.actions(point=c_orb_car, pot=c_pot, ifd=delta)
    #c_act = c_actfinder(c_orb_car)
    plt.plot(g_act[0],g_act[2])
    plt.plot(c_act[:,0],c_act[:,1])
    plt.show()

#compare([1,0,0,0.3,1.1,0.25,1.1],100.,1000)
compare([0.5015, 0, 0, 1.164*0.5**.5, 1.164*0.75**.5, 0.2855],100.,1000)