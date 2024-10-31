#!/usr/bin/python

"""
Test the correctness of evaluating composite densities and potentials with
various offsets from coordinate origin
"""
import numpy
# if the module has been installed to the globally known directory, just import it
try: import agama
except ImportError:  # otherwise load the shared library from the parent folder
    import sys
    sys.path += ['../']
    try: import agama
    except ImportError as ex: sys.exit("\033[1;31mFAILED TO IMPORT AGAMA: %s\033[0m" % ex)

# three groups of three components in each, with different centers for each group.
# note that the composite density contains these individual components as given,
# but the composite potential arranges them into three groups with different centers,
# and each group has a possibly different complement of ingredients, following
# the GalPot scheme for splitting/grouping Disk and Spheroid/Sersic/Nuker models
# into DiskAnsatz and Multipole potentials
centers=[
"0,0,0",
(0,1,2),
numpy.array([1,2,3])
]
params = [
dict(type='disk',         center=centers[0], scaleheight=0.2),
dict(type='spheroid',     center=centers[0], gamma=1, beta=4, alpha=1),
dict(type='ferrers',      center=centers[0], p=0.8, q=0.6),
dict(type='plummer',      center=centers[1]),
dict(type='sersic',       center=centers[1]),
dict(type='nuker',        center=centers[1]),
dict(type='disk',         center=centers[2], scaleheight=-0.2),
dict(type='dehnen',       center=centers[2]),
dict(type='miyamotonagai',center=centers[2]),
]

d=agama.Density  (*params)
p=agama.Potential(*params)
ok = len(d) == len(params) and len(p) == len(centers)
print("Density: %i items. %s" % (len(d), d))
print("Potential: %i items. %s" % (len(p), p))
for i in range(3):
    print("Potential %i: %i items. %s" % (i, len(p[i]), p[i]))

# despite the complicated grouping scheme, we may reasonably expect that the density
# returned by the two objects is the same (up to numerical errors in potential
# approximation), and equals the trace of the Hessian of the potential (up to 4pi)
points = [[0,0,0.1], [1,1,0.5], (0.5,0.5,1)]
D = d.density(points)
P = p.density(points)
H = numpy.sum(-p.eval(points, der=True)[:,0:3], axis=1) / (4*numpy.pi)
ok &= numpy.allclose(P, D, rtol=0.01) and numpy.allclose(P, H, rtol=1e-15)
print("rho from Density:   %s\nrho from Potential: %s\nrho from hessian:   %s" % (D, P, H))
# check the agreement in each of the three sub-groups
Ds = D*0; Ps = P*0
for i in range(3):
    d=agama.Density(*params[i*3:i*3+3])
    Di = d.   density(points)
    Pi = p[i].density(points)
    Hi = numpy.sum(-p[i].eval(points, der=True)[:,0:3], axis=1) / (4*numpy.pi)
    ok &= numpy.allclose(Pi, Di, rtol=0.01) and numpy.allclose(Pi, Hi, rtol=1e-15)
    Ds += Di
    Ps += Pi
    print("rho from Density:   %s\nrho from Potential: %s\nrho from hessian:   %s" % (D, P, H))

ok &= numpy.allclose(Ds, D, rtol=1e-15)
ok &= numpy.allclose(Ps, P, rtol=1e-15)

if ok:
    print("\033[1;32mALL TESTS PASSED\033[0m")
else:
    print("\033[1;31mSOME TESTS FAILED\033[0m")
