#!/usr/bin/python

"""
Create a simple single-component spherical self-consistent model
determined by its distribution function in terms of actions.
We use the true DF of the Plummer model, expressed in terms of actions,
and start iterations from a deliberately wrong initial guess;
nevertheless, after 10 iterations we converge to the true solution within 1%;
each iteration approximately halves the error.
"""
import agama, numpy, matplotlib.pyplot as plt

# the distribution function defining the model
truepot = agama.Potential(type='Plummer')
df = agama.DistributionFunction(type='QuasiSpherical', potential=truepot, density=truepot)

# initial guess for the density profile - deliberately a wrong one
dens = agama.Density(type='Dehnen', mass=0.1, scaleRadius=0.5)

# define the self-consistent model consisting of a single component
params = dict(rminSph=0.001, rmaxSph=1000., sizeRadialSph=40, lmaxAngularSph=0)
comp = agama.Component(df=df, density=dens, disklike=False, **params)
scm = agama.SelfConsistentModel(**params)
scm.components=[comp]

# prepare visualization
r=numpy.logspace(-2.,2.)
xyz=numpy.vstack((r,r*0,r*0)).T
plt.plot(r, dens.density(xyz), label='Init density')
plt.plot(r, truepot.density(xyz), label='True density', c='k')[0].set_dashes([4,4])

# perform several iterations of self-consistent modelling procedure
for i in range(10):
    scm.iterate()
    print('Iteration %i, Phi(0)=%g, Mass=%g' % \
        (i, scm.potential.potential(0,0,0), scm.potential.totalMass()))
    plt.plot(r, scm.potential.density(xyz), label='Iteration #'+str(i))

# save the final density/potential profile
scm.potential.export("simple_scm.ini")

# show the results
plt.legend(loc='lower left')
plt.xlabel("r")
plt.ylabel(r'$\rho$')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-5, 1)
plt.xlim(0.02, 20)
plt.show()
