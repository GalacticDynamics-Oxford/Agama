#!/usr/bin/python
"""
Create a simple single-component flattened self-consistent model
determined by its distribution function in terms of actions.
We use the DoublePowerLaw DF approximately corresponding to
a flattened Sersic profile with index n=2 and axis ratio q=0.5,
using the parameters found by example_doublepowerlaw.py and
slightly adjusting them to add some rotation.
"""
import agama, numpy, matplotlib.pyplot as plt

# the distribution function defining the model
dfparams = dict(
    type     = 'DoublePowerLaw',
    J0       = 1.0,
    slopeIn  = 1.6,
    slopeOut = 1.6,
    steepness= 1.0,
    coefJrIn = 0.9,
    coefJzIn = 1.6,
    coefJrOut= 0.8,
    coefJzOut= 1.7,
    Jcutoff  = 0.7,
    cutoffstrength = 2.0,
    rotFrac  = 1.0,  # make a rotating model (just for fun)
    Jphi0    = 0.05, # size of non-(or weakly-)rotating core
    mass     = 1.0)

df = agama.DistributionFunction(**dfparams)

# initial guess for the density profile
dens = agama.Density(type='sersic', mass=1, scaleRadius=0.6, sersicindex=2)

# define the self-consistent model consisting of a single component
gridparams = dict(rminSph=0.01, rmaxSph=10., sizeRadialSph=25, lmaxAngularSph=8)
comp = agama.Component(df=df, density=dens, disklike=False, **gridparams)
scm = agama.SelfConsistentModel(**gridparams)
scm.components=[comp]

# prepare visualization
r = numpy.logspace(-2.,1.)
xyz = numpy.vstack((r,r*0,r*0)).T
ax = plt.subplots(1, 3, figsize=(15,5))[1]
ax[0].plot(r, dens.density(xyz), label='Init density', color='lightgray', lw=3, dashes=[2,2])

# perform several iterations of self-consistent modelling procedure
for i in range(5):
    scm.iterate()
    print('Iteration %i, Phi(0)=%g, Mass=%g' %
        (i, scm.potential.potential(0,0,0), scm.potential.totalMass()))
    # for a fair comparison, show the spherically-averaged density profile
    sphDensity = agama.Density(type='DensitySphericalHarmonic', density=comp.density, lmax=0)
    ax[0].plot(r, sphDensity.density(xyz), label='Iteration #'+str(i))

ax[0].legend(loc='lower left', frameon=False)
ax[0].set_xlabel('r')
ax[0].set_ylabel(r'$\rho$')
ax[0].set_xscale('log')
ax[0].set_yscale('log')
ax[0].set_ylim(1e-5, 2e2)
ax[0].set_xlim(min(r), max(r))

# save the final density/potential profile and create an N-body snapshot
print('Creating N-body model')
comp.density.export('example_self_consistent_model_flattened_density.ini')
scm.potential.export('example_self_consistent_model_flattened_potential.ini')
gm = agama.GalaxyModel(scm.potential, df)
xv, m = gm.sample(1000000)
agama.writeSnapshot('example_self_consistent_model_flattened.snap', (xv, m), 'text')

# show the axis ratio
print('Determining shape')
r = numpy.logspace(-2, 1, 16)
xyz = numpy.column_stack((r, r*0, r*0))
axes = comp.density.principalAxes(r)[0]
ax[1].plot(r, axes[:,2]/axes[:,0])
ax[1].set_xlabel('r')
ax[1].set_ylabel(r'$z/R$')
ax[1].set_xscale('log')
ax[1].set_ylim(0, 1)
ax[1].set_xlim(min(r), max(r))

print('Computing kinematic profiles')
rho, meanv, vel2 = gm.moments(xyz, dens=True, vel=True, vel2=True)
vcirc = (-r * scm.potential.force(xyz)[:,0])**0.5
ax[2].plot(r, vcirc, label=r'$v_{\sf circ}$')
ax[2].plot(r, meanv[:,1], label=r'$\overline{v_\phi}$')
ax[2].plot(r,  vel2[:,0]**0.5, label=r'$\sigma_R$')
ax[2].plot(r, (vel2[:,1]-meanv[:,1]**2)**0.5, label=r'$\sigma_\phi$')
ax[2].plot(r,  vel2[:,2]**0.5, label=r'$\sigma_z$')
ax[2].set_xlabel('r')
ax[2].set_ylabel('velocity')
ax[2].set_xscale('log')
ax[2].set_xlim(min(r), max(r))
ax[2].legend(loc='upper left', frameon=False)

plt.tight_layout()
plt.show()
