#!/usr/bin/python

"""
This example illustrates various usage scenarios for Target objects, used in Schwarzschild models.
"""
import agama,numpy

numpy.set_printoptions(linewidth=100, formatter={'all':lambda x: '%11.6f'%x})

# set up some random physical units, to test the correction of unit conversion
agama.setUnits(length=3.4567, mass=12345, velocity=6.7)

# physical constituents of a model: potential and a corresponding isotropic DF
pot   = agama.Potential(type='Plummer', scaleRadius=0.1, mass=100)
dens  = pot
df    = agama.DistributionFunction(type='QuasiSpherical', potential=pot)
gm    = agama.GalaxyModel(pot, df)
vesc  = (-2*pot.potential(0,0,0))**0.5  # escape velocity from r=0
#print("escape velocity: %g" % vesc)

# define a grid in radius and a series of concentric rings for recording LOSVD
gridr = agama.nonuniformGrid(10, 0.025, 1)  # grid in radius (denser at small radii)
gridx = agama.symmetricGrid (19, 0.025, 1)  # grids in x,y - symmetrically mirror about origin
gridv = numpy.linspace(-vesc, vesc, 25)     # grid in velocity space
ang   = numpy.linspace(0, 2*numpy.pi, 73)   # approximate a circle with a polygon with this many vertices
sa    = numpy.sin(ang); sa[36]=sa[72]=0
ca    = numpy.cos(ang); ca[18]=ca[54]=0
circ  = [numpy.column_stack((rr*ca, rr*sa)) for rr in gridr[1:]]
poly  = [circ[0]] + [ numpy.vstack((p1, p2[::-1])) for p1,p2 in zip(circ[1:],circ[:-1]) ]
numaper = len(poly)   # total number of apertures (all concentric rings except the central one, which is a circle)

# define different types of target objects
tar_dens  = agama.Target(type='DensitySphHarm', gridr=gridr)
tar_shell = agama.Target(type='KinemShell', gridr=gridr, degree=1)
degree    = 2   # B-spline degree for LOSVD representation (2 or 3 are strongly recommended)
tar_losvd = agama.Target(type='LOSVD', degree=degree, apertures=poly, gridx=gridx, gridv=gridv, symmetry='s')

# generate N-body samples from the model,
# and integrate orbits with (some of) these initial conditions
xv, mass  = gm.sample(1000000)
ic = xv[:10000]  # use only a subset of ICs, to save time
mat_dens, mat_shell, mat_losvd = agama.orbit(potential=pot, ic=ic,
    time=100*pot.Tcirc(ic), targets=[tar_dens, tar_shell, tar_losvd])

# as the IC were drawn from the actual DF of the model,
# we expect that all orbits have the same weight in the model
orbitweight = pot.totalMass() / len(ic)

# now apply the targets to several kinds of objects:
print("Applying Targets to various objects (takes time)...")
# 1) collection of orbits, summing up equally-weighted contributions of all orbits
dens_orb   = numpy.sum(mat_dens,  axis=0) * orbitweight
shell_orb  = numpy.sum(mat_shell, axis=0) * orbitweight
losvd_orb  = numpy.sum(mat_losvd, axis=0) * orbitweight
# 2) the entire model (df + potential)
dens_df    = tar_dens (gm)
shell_df   = tar_shell(gm)
losvd_df   = tar_losvd(gm)   # this **is** expensive...
# 3) N-body realization of the model
dens_nb    = tar_dens ((xv,mass))
shell_nb   = tar_shell((xv,mass))
losvd_nb   = tar_losvd((xv,mass))
# 4) the 3d density profile of the model
dens_dens  = tar_dens (dens)
# all these operations should produce roughly equal results, so illustrate this

# Target DensitySphHarm produces an array of masses, which should sum up to nearly the total mass
print("3d density discretized on a radial grid (masses associated with grid nodes) computed from...")
print("[radius]   orbit library   smooth DF   N-body model   density profile")
print(numpy.column_stack((gridr, dens_orb, dens_df, dens_nb, dens_dens)))
print("Total: %6.1f %11.6f %11.6f %11.6f %11.6f" %
    (pot.totalMass(), sum(dens_orb), sum(dens_df), sum(dens_nb), sum(dens_dens)))

# Target KinemShell produces an array of length 2*len(gridr), with the elements being
# density-weighted radial (first half) and tangential (second half) squared velocity dispersions.
# To get the actual velocity dispersion, we divide these numbers by the array produced by
# Target DensitySphHarm, which is the mass associated with each grid node computed in the same way.
print("3d velocity dispersion profiles on a radial grid (first radial, then tangential)")
print("[radius]   orbit library   smooth DF   N-body model   DF moments")
mom = gm.moments(numpy.column_stack((gridr, gridr*0, gridr*0)), dens=False, vel2=True)
print(numpy.column_stack((gridr,
    (shell_orb[:len(gridr)] / dens_orb)**0.5,
    (shell_df [:len(gridr)] / dens_df )**0.5,
    (shell_nb [:len(gridr)] / dens_nb )**0.5,
    mom[:,0]**0.5 )))
print(numpy.column_stack((gridr,
    (shell_orb[len(gridr):] * 0.5 / dens_orb)**0.5,  # tangenial contains a sum of two components
    (shell_df [len(gridr):] * 0.5 / dens_df )**0.5,  # (theta and phi), so divide by two to get
    (shell_nb [len(gridr):] * 0.5 / dens_nb )**0.5,  # a one-dimensional velocity dispersion
    ( (mom[:,1] + mom[:,2]) * 0.5)**0.5 )))          # similarly, average between theta and z

# Target LOSVD produces the line-of-sight velocity distributions in the given apertures
# in the image plane, when applied to objects with velocity information
# (GalaxyModel, N-body snapshot, or orbit library).
# Integrating these LOSVDs along the velocity axis, one obtains the projected mass
# in each aperture (integral of surface density over the area of the aperture),
# and this is also the result of applying Target LOSVD to a Density object.
print("2d surface density integrated over apertures (projected masses in each aperture)")
print("[radius]   orbit library   smooth DF   N-body model   density   GH moments")
bsi = agama.bsplineIntegrals(degree, gridv)
ghm = agama.ghMoments(degree=degree, gridv=gridv, matrix=losvd_df, ghorder=6).reshape(numaper,-1)
surf_dens  = tar_losvd(dens)
surf_orb   = losvd_orb.reshape(numaper,-1).dot(bsi)
surf_df    = losvd_df .reshape(numaper,-1).dot(bsi)
surf_nb    = losvd_nb .reshape(numaper,-1).dot(bsi)
print(numpy.column_stack((gridr[1:], surf_orb, surf_df, surf_nb, surf_dens, ghm[:,0])))
print("Total: %6.1f %11.6f %11.6f %11.6f %11.6f %11.6f" %
    (pot.totalMass(), sum(surf_orb), sum(surf_df), sum(surf_nb), sum(surf_dens), sum(ghm[:,0])))

# Of course, the main result of Target LOSVD are the velocity profiles, not just their integrals;
# we compare the profiles produced from different objects in each aperture
import matplotlib.pyplot as plt
ax=plt.subplots(3, 3, figsize=(10,10))[1].reshape(-1)
gridvplot = numpy.linspace(-vesc, vesc, 201)
for i in range(numaper):
    ax[i].plot(gridvplot, agama.bsplineInterp(degree, gridv, losvd_orb.reshape(numaper,-1)[i], gridvplot), label='orbits')
    ax[i].plot(gridvplot, agama.bsplineInterp(degree, gridv, losvd_df .reshape(numaper,-1)[i], gridvplot), label='smooth DF')
    ax[i].plot(gridvplot, agama.bsplineInterp(degree, gridv, losvd_nb .reshape(numaper,-1)[i], gridvplot), label='N-body')
    ax[i].plot(gridvplot, agama.ghInterp(ghm[i,0], ghm[i,1], ghm[i,2], ghm[i,3:], gridvplot), label='GaussHermite')
    if i==0: ax[i].legend(loc='upper left', fontsize=8, frameon=False)
    ax[i].text(0, 0, 'r<%.3g' % gridr[i+1], ha='center', va='bottom')
plt.tight_layout()
plt.show()
