#!/usr/bin/python
"""
    This example demonstrates the use of action finder (Staeckel approximation)
    to compute actions for particles from an N-body simulation.
    The N-body system consists of a disk and a halo,
    the two components being stored in separate text files.
    They are not provided in the distribution, but could be created by running
    example_self_consistent_model.py
    The potential is computed from the snapshot itself, by creating a suitable
    potential expansion for each component: Multipole for the halo and CylSpline
    for the disk. This actually takes most of the time. We save the constructed
    potentials to text files, and on the subsequent launches of this program
    they are loaded from these files, speeding up the initialization.
    Then we compute actions for all particles from the disk component,
    and store them in a text file.

    An equivalent example in C++ is located in tests folder.
"""
import agama, numpy
try: from time import process_time as clock  # Python 3.3+
except ImportError: from time import clock   # older Python

#1. set units (in Msun, Kpc, km/s)
agama.setUnits(mass=1, length=1, velocity=1)

#2. get in N-body snapshots: columns 0 to 2 are position, 3 to 5 are velocity, 6 is mass
tbegin = clock()
try:
    diskParticles = numpy.loadtxt("model_stars_final")
    haloParticles = numpy.loadtxt("model_dm_final")
except:
    exit("Input snapshot files are not available; " \
        "you may create them by running example_self_consistent_model.py")

print("%g s to load %d disk particles (total mass=%g Msun) " \
    "and %d halo particles (total mass=%g Msun)" % \
    ( clock()-tbegin, \
    diskParticles.shape[0], numpy.sum(diskParticles[:,6]), \
    haloParticles.shape[0], numpy.sum(haloParticles[:,6]) ) )


#3. create an axisymmetric potential from these snapshots

try:
    #3a. try to load potentials from previously stored text files instead of computing them
    diskPot = agama.Potential("model_stars_final.ini")
    haloPot = agama.Potential("model_dm_final.ini")

except:
    # 3b: these files don't exist on the first run, so we have to create the potentials
    tbegin  = clock()
    haloPot = agama.Potential( \
        type="Multipole", particles=(haloParticles[:,0:3], haloParticles[:,6]), \
        symmetry='a', gridsizeR=20, lmax=2)
    print("%f s to init %s potential for the halo; value at origin=%f (km/s)^2" % \
        ((clock()-tbegin), haloPot, haloPot.potential(0,0,0)))
    tbegin  = clock()
    # manually specify the spatial grid for the disk potential,
    # although one may rely on the automatic choice of these parameters (as we did for the halo)
    diskPot = agama.Potential( \
        type="CylSpline", particles=(diskParticles[:,0:3], diskParticles[:,6]), \
        gridsizer=20, gridsizez=20, symmetry='a', Rmin=0.2, Rmax=100, Zmin=0.05, Zmax=50)
    print("%f s to init %s potential for the disk; value at origin=%f (km/s)^2" % \
        ((clock()-tbegin), diskPot, diskPot.potential(0,0,0)))

    # save the potentials into text files; on the next call may load them instead of re-computing
    diskPot.export("model_stars_final.ini")
    haloPot.export("model_dm_final.ini")

#3c. combine the two potentials into a single composite one
totalPot  = agama.Potential(diskPot, haloPot)

#4. compute actions for disk particles
tbegin    = clock()
actFinder = agama.ActionFinder(totalPot)
print("%f s to init action finder" % (clock()-tbegin))

tbegin    = clock()
actions   = actFinder(diskParticles[:,0:6])
print("%f s to compute actions for %i particles" % (clock()-tbegin,  diskParticles.shape[0]))

#5. write out data
Rz        = numpy.vstack(
    ( numpy.sqrt(diskParticles[:,0]**2 + diskParticles[:,1]**2), diskParticles[:,2] ) ).T
energy    = (totalPot.potential(diskParticles[:,0:3]) + \
    0.5 * numpy.sum(diskParticles[:,3:6]**2, axis=1) ).reshape(-1,1)
numpy.savetxt( "disk_actions.txt", numpy.hstack((Rz, actions, energy)), \
    header="R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]", \
    fmt="%.6g", delimiter="\t" )
