#!/usr/bin/python
"""
Create a simple single-component flattened self-consistent model
determined by its distribution function in terms of actions.
We use the DoublePowerLaw DF approximately corresponding to the Sersic
model, with parameters found by the program example_doublepowerlaw.exe
(fitting the DF to a spherical Sersic profile with n=2) and then
adjusted manually to create some flattening and rotation.
"""
import agama, numpy, matplotlib.pyplot as plt

# the distribution function defining the model
params = dict(
    type     = 'DoublePowerLaw',
    J0       = 1e3,
    slopeIn  = 1.5,
    slopeOut = 1.5,
    steepness= 1.0,
    coefJrIn = 0.8,
    coefJzIn = 1.7,
    coefJrOut= 0.8,
    coefJzOut= 1.7,
    Jcutoff  = 0.56,
    cutoffstrength=1.5,
    rotFrac  = 1.0,  # make a rotating model (just for fun)
    Jphi0    = 0.,   # size of non-(or weakly-)rotating core
    norm     = 1.0)

# compute the mass and rescale norm to get the total mass = 1
params['norm'] /= agama.DistributionFunction(**params).totalMass()
df = agama.DistributionFunction(**params)

# initial guess for the density profile
dens = agama.Potential(type='sersic', mass=1, scaleRadius=0.65, sersicindex=2)

# define the self-consistent model consisting of a single component
params = dict(rminSph=0.01, rmaxSph=10., sizeRadialSph=25, lmaxAngularSph=8)
comp = agama.Component(df=df, density=dens, disklike=False, **params)
scm = agama.SelfConsistentModel(**params)
scm.components=[comp]

# prepare visualization
r=numpy.logspace(-2.,2.)
xyz=numpy.vstack((r,r*0,r*0)).T
plt.plot(r, dens.density(xyz), label='Init density', color='k')

# perform several iterations of self-consistent modelling procedure
for i in range(6):
    scm.iterate()
    print('Iteration %i, Phi(0)=%g, Mass=%g' % \
        (i, scm.potential.potential(0,0,0), scm.potential.totalMass()))
    plt.plot(r, scm.potential.density(xyz), label='Iteration #'+str(i))

# save the final density/potential profile and create an N-body snapshot
comp.getDensity().export('flattened_sersic_density.ini')
scm.potential.export('flattened_sersic_potential.ini')
agama.writeSnapshot('flattened_sersic_nbody.nemo',
    agama.GalaxyModel(scm.potential, df).sample(1000000), 'nemo')

# show the results
plt.legend(loc='lower left')
plt.xlabel("r")
plt.ylabel(r'$\rho$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 2e2)
plt.xlim(0.01, 10)
plt.show()
