#!/usr/bin/python
"""
This example illustrates the use of Schwarzschild method to construct a triaxial Dehnen model.
"""
import agama, numpy

# Hernquist density profile
den = agama.Density(type='Spheroid', axisRatioY=0.8, axisRatioZ=0.6, scaleRadius=1, mass=1, gamma=1, beta=4)

# potential corresponding to this density
pot = agama.Potential(type='Multipole', lmax=8, mmax=6, density=den)

# target1 is discretized density profile
target1 = agama.Target(type='DensityClassicLinear', gridr=agama.nonuniformGrid(25, 0.1, 20.0), \
    axisratioy=0.8, axisratioz=0.6, stripsPerPane=2)

# target2 is discretized kinematic constraint (we will enforce velocity isotropy)
target2 = agama.Target(type='KinemShell', gridR=agama.nonuniformGrid(15, 0.2, 10.0), degree=1)

# construct initial conditions for the orbit library
initcond,weightprior = den.sample(5000, potential=pot)

# integration time is 50 orbital periods
inttimes = 50*pot.Tcirc(initcond)
# integrate all orbits, storing the recorded data corresponding to each target
# in the data1 and data2 arrays, and the trajectories - in trajs
# (by specifying dtype=object, orbits are represented by instances of agama.Orbit
# providing interpolated trajectories recorded at each timestep of the ODE integrator)
data1, data2, trajs = agama.orbit(potential=pot, ic=initcond, time=inttimes, \
    dtype=object, targets=[target1,target2])

# assemble the matrix equation which contains three blocks:
# total mass, discretized density, and velocity anisotropy

# a single value for the total mass constraint, each orbit contributes 1 to it
mass = den.totalMass()
data0= numpy.ones((len(initcond), 1))

# discretized density profile to be used as the density constraint
rhs1 = target1(den)

# data2 is kinematic data: for each orbit (row) it contains
# Ngrid values of rho sigma_r^2, then the same number of values of rho sigma_t^2;
# we combine them into a single array of Ngrid values that enforce isotropy (sigma_t^2 - 2 sigma_r^2 = 0)
Ngrid = len(target2) // 2
datak = 2*data2[:,:Ngrid] - data2[:,Ngrid:]
rhs2  = numpy.zeros(Ngrid)

# solve the matrix equation for orbit weights
weights = agama.solveOpt(matrix=(data0.T, data1.T, datak.T), rhs=([mass], rhs1, rhs2), \
    rpenl=([numpy.inf], numpy.ones_like(rhs1), numpy.ones_like(rhs2)), \
    xpenq=numpy.ones(len(initcond))*0.1 )

# check if all constraints were satisfied
delta1 = data1.T.dot(weights) - rhs1
for i,d in enumerate(delta1):
    if abs(d)>1e-8:
        print("DensityConstraint %i not satisfied: %s, val=%.4g, rel.err=%.4g" % \
        (i, target1[i], rhs1[i], d / rhs1[i]))
delta2 = datak.T.dot(weights)
for i,d in enumerate(delta2):
    if abs(d)>1e-8:  print("KinemConstraint %i not satisfied: %.4g" % (i, d))

# export an N-body model
nbody=100000
status,result = agama.sampleOrbitLibrary(nbody, trajs, weights)
if not status:
    # this may occur if there was not enough recorded trajectory points for some high-weight orbits:
    # in this case their indices and the required numbers of points are returned in the result tuple.
    # This cannot happen if orbits are represented by interpolator objects rather than pre-recorded arrays.
    indices,trajsizes = result
    print("reintegrating %i orbits; max # of sampling points is %i" % (len(indices), max(trajsizes)))
    trajs[indices] = agama.orbit(potential=pot, ic=initcond[indices], time=inttimes[indices], \
        trajsize=trajsizes)
    status,result = agama.sampleOrbitLibrary(nbody, trajs, weights)
    if not status: print("Failed to produce output N-body model")
agama.writeSnapshot("schwarzschild_model_nbody.txt", result, 'text')   # one could also use numpy.savetxt

# also store the entire Schwarzschild model in a numpy binary archive
numpy.savez_compressed("schwarzschild_model_data", ic=initcond, inttime=inttimes, weight=weights, \
    data1=data1, data2=data2, cons1=rhs1)

# store the orbit initial conditions and weights in a text file
numpy.savetxt("schwarzschild_model_orbits", \
    numpy.column_stack((initcond, weights, weightprior, inttimes)), \
    header='x y z vx vy vz weight prior inttime')
