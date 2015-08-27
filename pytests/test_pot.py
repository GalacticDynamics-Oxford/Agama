#!/usr/bin/python

import py_wrapper
import py_unsio
import numpy

#1. get in some N-body snapshot
a = py_unsio.CunsIn("../temp/disk.gadget","all","all")
a.nextFrame("")
ok,pos = a.getArrayF("all","pos")
pos = pos.reshape(-1,3)
ok,mass = a.getArrayF("all","mass")
print "Loaded",len(mass),"points"

#2. create a potential from this snapshot
p = py_wrapper.Potential(type="SplineExp", points=(pos,mass))
print "Created a",p.name(),"potential"

#3. compute something interesting
pot = p.potential(pos)
indx = numpy.argmin(pot)
print "Lowest value of potential is",pot[indx],"for point",pos[indx]
dens = p.density(pos)
indx = numpy.argmax(dens)
print "Highest value of density is",dens[indx],"for point",pos[indx]

#4. test potential derivatives
_,derivs = p.force_deriv(pos)
print "RMS error in density vs. laplacian =", \
    ( ( (4*3.141592654*dens + (derivs[:,0]+derivs[:,1]+derivs[:,2]) )**2 ).mean() )**0.5

#5. test spline smoothing
rad = (pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)**0.5
knots = [0,0.5,1,1.5,2,3,4,5,6,7,8,10,15,20,30,50,100]
spl = py_wrapper.SplineApprox(rad,pot,knots)
print "RMS error in Phi(r) approximating spline =", ( ((pot-spl(rad))**2).mean() )**0.5

print "ALL TESTS PASSED"