#!/usr/bin/python
"""
This example illustrates the use of Schwarzschild method to construct a triaxial Dehnen model.
"""
import agama, numpy

# density profile
den = agama.Density(type='Dehnen', axisRatioY=0.8, axisRatioZ=0.6, scaleRadius=1, mass=1, gamma=1)
# potential corresponding to this density
pot = agama.Potential(type='Multipole', lmax=8, mmax=6, density=den, gridsizer=30)
# target1 is discretized density profile
target1 = agama.Target(type='DensityClassicLinear', density=den, gridsizer=15, \
    axisratioy=0.8, axisratioz=0.6, stripsPerPane=2, innerShellMass=0.04, outerShellMass=0.96)
# target2 is discretized kinematic constraint (we will enforce velocity isotropy)
target2 = agama.Target(type='KinemJeans', density=den, potential=pot, gridSizeR=10, degree=1)
# construct initial conditions for the orbit library
initcond,weightprior = den.sample(2000, potential=pot)
# integration time is 50 orbital periods
inttimes = 50*pot.Tcirc(initcond)

# integrate all orbits, storing the recorded data corresponding to each target
# in the data1 and data2 arrays, and the trajectories - in trajs
data1, data2, trajs = agama.orbit(potential=pot, ic=initcond, time=inttimes, \
    trajsize=101, targets=[target1,target2])

# data2 is kinematic data: for each orbit (row) it contains
# Ngrid values of rho sigma_r^2, then the same number of values of rho sigma_t^2;
# we combine them into a single array of Ngrid values that enforce isotropy (sigma_t^2 - 2 sigma_r^2 = 0)
Ngrid = len(target2)
datak = 2*data2[:,:Ngrid] - data2[:,Ngrid:]
weights = agama.optsolve(matrix=(data1.T, datak.T), rhs=[target1.values(),numpy.zeros(Ngrid)], \
    rpenl=[numpy.ones_like(target1.values()), numpy.ones(Ngrid)], \
    xpenq=numpy.ones(len(initcond))*0.1 )

# check if all constraints were satisfied
delta1 = data1.T.dot(weights) - target1.values()
for i,d in enumerate(delta1):
    if abs(d)>1e-8:  print "DensityConstraint",i,"not satisfied:", target1[i], d
delta2 = datak.T.dot(weights)
for i,d in enumerate(delta2):
    if abs(d)>1e-8:  print "KinemConstraint",i,"not satisfied:", d

# export an N-body model
nbody=50000
status,result = agama.sampleOrbitLibrary(nbody, trajs, weights)
if not status:
    # this may occur if there was not enough recorded trajectory points for some high-weight orbits:
    # in this case their indices and the required numbers of points are returned in the result tuple
    indices,trajsizes = result
    print "reintegrating",len(indices),"orbits; max # of sampling points is", max(trajsizes)
    trajs[indices] = agama.orbit(potential=pot, ic=initcond[indices], time=inttimes[indices], \
        trajsize=trajsizes)
    status,result = agama.sampleOrbitLibrary(nbody,trajs[:,1],weights)
    if not status: print "Failed to produce output N-body model"
posvel,mass = result
numpy.savetxt("model_nbody.txt", numpy.hstack((posvel, mass.reshape(-1,1))), fmt="%6g")

# also store the entire Schwarzschild model in a numpy binary archive
numpy.savez_compressed("model_data", ic=initcond, inttime=inttimes, weight=weights, \
    data1=data1, data2=data2, cons1=target1.values(), cons2=target2.values())
# store trajectories in a numpy binary file
numpy.save("model_traj", trajs)
# store the orbit initial conditions and weights in a text file
numpy.savetxt("model_orbits", numpy.hstack((initcond, weights.reshape(-1,1), weightprior.reshape(-1,1), \
    inttimes.reshape(-1,1))), header='x y z vx vy vz weight prior inttime')
