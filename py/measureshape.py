#!/usr/bin/env python
"""
A simple tool for measuring the axis ratios of an N-body snapshot, using the moment of inertia
"""
import agama, numpy, sys
if len(sys.argv)<=1: exit("Provide file name or 'test' to use an internally created fiducial triaxial snapshot")
if sys.argv[1] == 'test':
    # create a triaxial and rotated Hernquist model
    nbody    = 100000
    radius   = 1 / (numpy.random.random(size=nbody)**-0.5 - 1)
    costheta = numpy.random.uniform(-1, 1, size=nbody)
    sintheta = (1-costheta**2)**0.5
    phi      = numpy.random.uniform(0, 2*numpy.pi, size=nbody)
    pos      = numpy.column_stack((
      radius * sintheta * numpy.cos(phi),
      radius * sintheta * numpy.sin(phi),
      radius * costheta ))
    pos     *= numpy.array([1.0, 0.7, 0.4])  # make it triaxial
    pos      = pos.dot(agama.makeRotationMatrix(*list(numpy.random.random(size=3))))  # rotate by some random angles
    mass     = numpy.ones(nbody) / nbody
else:
    pos, mass = agama.readSnapshot(sys.argv[1])

def getaxes(pos, mass, radius):
    evec = numpy.eye(3)   # initial guess for axes orientation
    axes = numpy.ones(3)  # and axis ratios; these are updated at each iteration
    while True:
        # use particles within the elliptical radius less than the provided value
        ellpos  = pos[:,0:3].dot(evec) / axes
        filter  = numpy.sum(ellpos**2, axis=1) < radius**2
        inertia = pos[filter,0:3].T.dot(pos[filter,0:3] * mass[filter,None])
        val,vec = numpy.linalg.eigh(inertia)
        order   = numpy.argsort(-val)  # sort axes in decreasing order
        evec    = vec[:,order]         # updated axis directions
        axesnew = (val[order] / numpy.prod(val)**(1./3))**0.5  # updated axis ratios, normalized so that ax*ay*az=1
        #print evec,axesnew,sum(filter)
        if sum(abs(axesnew-axes))<0.01: break
        axes    = axesnew
    return axes, numpy.sum(mass[filter])

# compute axis ratio in several roughly equal-mass bins (each next one includes the previous ones).
# bin radii are estimated from the sphericalized enclosed mass profile, so the actual mass
# in the rectified bins may differ from the prescribed enclosed mass fraction.
# "radius" refers to the sphericalized radius of each bin:
# rsph = (ax * ay * az)**(1./3),  where ax,ay,az are the lengths of the principal axes;
# axis ratios are reported as 1 >= ay/ax >= az/ax, so the longest axis (ax) is longer than rsph.
sphrad = numpy.sum(pos**2, axis=1)**0.5
order  = numpy.argsort(sphrad)
cummass= numpy.cumsum(mass[order])
nbins  = 20
indbin = numpy.searchsorted(cummass, numpy.linspace(0.04, 0.99, 20) * cummass[-1])
binrad = sphrad[order][indbin]
print("#radius\tmass   \ty/x    \tz/x")
for i in range(nbins):
    axes, binmass = getaxes(pos, mass, binrad[i])
    print("%.3g\t%.3g\t%.3f\t%.3f" % (binrad[i], binmass, axes[1]/axes[0], axes[2]/axes[0]))
