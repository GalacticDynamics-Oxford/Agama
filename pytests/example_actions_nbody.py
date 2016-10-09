#!/usr/bin/python
"""
    This example demonstrates the use of action finder (Staeckel approximation)
    to compute actions for particles from an N-body simulation.
    The N-body system consists of a disk and a halo (10^5 particles each),
    the two components being stored in separate text files.
    The potential is computed from the snapshot itself, by creating a suitable
    potential expansion for each component: Multipole for the halo and CylSpline
    for the disk. This actually takes most of the time. We save the constructed
    potentials to text files, and on the subsequent launches of this program
    they are loaded from these files, speeding up the initialization.
    Then we compute actions for all particles from the disk component,
    and store them in a text file.

    An equivalent example in C++ is located in tests folder.
"""
import agama, numpy, time

#1. set units (in Msun, Kpc, km/s)
agama.setUnits(mass=1e10, length=1, velocity=1)

#2. get in N-body snapshots: columns 0 to 2 are position, 3 to 5 are velocity, 6 is mass
tbegin = time.clock()
diskParticles = numpy.loadtxt("../data/disk.dat")
haloParticles = numpy.loadtxt("../data/halo.dat")
print "%g s to load %d disk particles (total mass=%g Msun) " \
    "and %d halo particles (total mass=%g Msun)" % \
    ( time.clock()-tbegin, \
    diskParticles.shape[0], numpy.sum(diskParticles[:,6] * 1e10), \
    haloParticles.shape[0], numpy.sum(haloParticles[:,6] * 1e10) )


#3. create an axisymmetric potential from these snapshots

try:
    #3a. try to load potentials from previously stored text files instead of computing them
    haloPot = agama.Potential(file="halo.coef_mul")
    diskPot = agama.Potential(file="disk.coef_cyl")

except:
    # 3b: these files don't exist on the first run, so we have to create the potentials
    tbegin  = time.clock()
    haloPot = agama.Potential( \
        type="Multipole", particles=(haloParticles[:,0:3], haloParticles[:,6]), \
        symmetry='a', gridsizeR=20, lmax=2)
    print (time.clock()-tbegin), "s to init", haloPot.name(), "potential for the halo; ", \
        "value at origin=", haloPot.potential(0,0,0), "(km/s)^2"

    tbegin  = time.clock()
    diskPot = agama.Potential( \
        type="CylSpline", particles=(diskParticles[:,0:3], diskParticles[:,6]), \
        gridsizer=20, gridsizez=20, mmax=0, Rmin=0.2, Rmax=50, Zmin=0.02, Zmax=10)
    print (time.clock()-tbegin), "s to init", diskPot.name(), "potential for the disk; ", \
        "value at origin=", diskPot.potential(0,0,0), "(km/s)^2"

    # save the potentials into text files; on the next call may load them instead of re-computing
    diskPot.export("disk.coef_cyl")
    haloPot.export("halo.coef_mul")

#3c. combine the two potentials into a single composite one
totalPot  = agama.Potential(diskPot, haloPot)

#4. compute actions for disk particles
tbegin    = time.clock()
actFinder = agama.ActionFinder(totalPot)
print (time.clock()-tbegin),"s to init action finder"

tbegin    = time.clock()
actions   = actFinder(diskParticles[:,0:6])
print (time.clock()-tbegin), "s to compute actions for", diskParticles.shape[0], "particles"

#5. write out data
Rz        = numpy.vstack(
    ( numpy.sqrt(diskParticles[:,0]**2 + diskParticles[:,1]**2), diskParticles[:,2] ) ).T
energy    = (totalPot.potential(diskParticles[:,0:3]) + \
    0.5 * numpy.sum(diskParticles[:,3:6]**2, axis=1) ).reshape(-1,1)
numpy.savetxt( "disk_actions.txt", numpy.hstack((Rz, actions, energy)), \
    header="R[Kpc]\tz[Kpc]\tJ_r[Kpc*km/s]\tJ_z[Kpc*km/s]\tJ_phi[Kpc*km/s]\tE[(km/s)^2]", \
    fmt="%.6g", delimiter="\t" )
