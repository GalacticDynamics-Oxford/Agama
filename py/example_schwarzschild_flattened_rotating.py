#!/usr/bin/python
"""
This example illustrates the use of Schwarzschild method to construct a flattened rotating system.
The density profile follows the Sersic law with axis ratio z/x=0.6,
and there is a central black hole of mass 0.1 (the total stellar mass is 1).
The script constructs the orbit library, assigns the weights to the orbits,
then generates an N-body model representing the system (except the central black hole),
and plots various diagnostic information:
1) distribution of orbit weights as a function of mean radius and eccentricity of orbits --
one can check that there are no orbits with extremely high weight (this usually indicates
that the size of the orbit library needs to be increased),
2) kinematic profiles - mean rotation velocity and three components of velocity dispersion tensor
as functions of radius. Currently there is no way to constrain kinematics explicitly at the stage
of the optimization problem (solution for the orbit weights), but one can adjust it implicitly
by modifying the parameters of the initial condition generator - here we use the axisymmetric
anisotropic Jeans equations, and the two parameters (anisotropy and rotation fraction) can be tuned.
"""
import agama, numpy

# stellar potential
potstars = agama.Potential(type='Sersic', sersicIndex=4, axisRatioZ=0.6, scaleRadius=1, mass=1)

# central black hole (slightly softened)
potbh = agama.Potential(type='Plummer', scaleRadius=1e-4, mass=0.1)

# total potential
pot = agama.Potential(potstars, potbh)

# discretized density profile is recorded on a grid of radial points and spherical harmonics up to lmax
gridr = agama.nonuniformGrid(50, 0.02, 20.0)   # !! make sure that the grid covers the range of interest !!
target = agama.Target(type='DensitySphHarm', gridr=gridr, lmax=8, mmax=0)

# discretized density profile to be used as the density constraint
rhs = target(potstars)
#print("Density constraint values")
#for i in range(len(gridr)): print("%s: mass= %g" % (target[i], rhs[i]))

# construct initial conditions for the orbit library:
# use the anisotropic Jeans eqs to set the (approximate) shape of the velocity ellipsoid and the mean v_phi;
beta  = 0.0   # velocity anisotropy in R/z plane: >0 - sigma_R>sigma_z, <0 - reverse.
kappa = 0.8   # sets the amount of rotation v_phi vs. dispersion sigma_phi; 1 is rather fast rotation, 0 - no rot.
initcond,_ = potstars.sample(10000, potential=pot, beta=beta, kappa=kappa)

# integration time is 100 orbital periods
inttimes = 100*pot.Tcirc(initcond)

# integrate all orbits, storing the recorded density data and trajectories represented by interpolator objects
data, trajs = agama.orbit(potential=pot, ic=initcond, time=inttimes, dtype=object, targets=target,
    method='dprkn8', accuracy=1e-6)  # use a more efficient orbit integrator and a lower (but still sufficient) accuracy

# assemble the matrix equation which contains two blocks:
# total mass, discretized density

# a single value for the total mass constraint, each orbit contributes 1 to it
mass = potstars.totalMass()
data0= numpy.ones((len(initcond), 1))

# solve the matrix equation for orbit weights
weights = agama.solveOpt(matrix=(data0.T, data.T), rhs=([mass], rhs),
    rpenl=([numpy.inf], numpy.ones(len(rhs))*numpy.inf),
    xpenq=numpy.ones(len(initcond)) )

#numpy.savetxt('orbits', numpy.column_stack((initcond, weights, inttimes))[numpy.argsort(inttimes)], '%g')

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
    status,result = agama.sampleOrbitLibrary(nbody,trajs[:,1],weights)
    if not status: print("Failed to produce output N-body model")
agama.writeSnapshot("flattened_rotating_model.nemo", result,'n')


# various diagnostic plots
import matplotlib.pyplot as plt
ax=plt.subplots(1,2,figsize=(12,6))[1]

# plot orbit weights as a function of mean radius and eccentricity (shown in color)
rperi, rapo = pot.Rperiapo(initcond).T
ravg = (rapo+rperi)/2
ecc  = (rapo-rperi)/(rapo+rperi)
meanw= numpy.median(weights)
plt.colorbar(
    ax[0].scatter(ravg, weights, c=ecc, s=20*numpy.log(1+weights/meanw),
    marker='.', linewidths=0, cmap='mist', vmin=0, vmax=1),
    ax=ax[0], orientation='vertical', label='eccentricity')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim(meanw*0.1, numpy.amax(weights)*1.2)
ax[0].set_xlim(min(ravg)*0.8, max(ravg)*1.2)
ax[0].set_xlabel('mean radius')
ax[0].set_ylabel('orbit weight')

# plot particle kinematics
xv = result[0]
R  = (xv[:,0]**2+xv[:,1]**2)**0.5
vR = (xv[:,0]*xv[:,3]+xv[:,1]*xv[:,4]) / R
vt = (xv[:,0]*xv[:,4]-xv[:,1]*xv[:,3]) / R
vz =  xv[:,5]
Rmin, Rmax = numpy.percentile(R, [1, 99])
gridR  = numpy.logspace(numpy.log10(Rmin), numpy.log10(Rmax), 25)
centrR = (gridR[1:]*gridR[:-1])**0.5
norm   = numpy.histogram(R, bins=gridR)[0]
meanvt = numpy.histogram(R, bins=gridR, weights=vt)[0] / norm
sigmaR =(numpy.histogram(R, bins=gridR, weights=vR**2)[0] / norm)**0.5
sigmat =(numpy.histogram(R, bins=gridR, weights=vt**2)[0] / norm - meanvt**2)**0.5
sigmaz =(numpy.histogram(R, bins=gridR, weights=vz**2)[0] / norm)**0.5
ax[1].plot(centrR, sigmaR, label='sigma_R')
ax[1].plot(centrR, sigmat, label='sigma_phi')
ax[1].plot(centrR, sigmaz, label='sigma_z')
ax[1].plot(centrR, meanvt, label='v_phi', dashes=[3,2])
ax[1].set_xscale('log')
ax[1].set_xlim(min(gridR), max(gridR))
ax[1].set_xlabel('radius')
ax[1].set_ylabel('velocity mean/dispersion')
ax[1].legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.show()
