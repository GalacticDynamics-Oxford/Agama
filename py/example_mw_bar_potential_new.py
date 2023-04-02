#!/usr/bin/python
'''
This script uses an analytic approximation for the bar model from Portail et al.(2017),
in combination with several other mass components (central black hole, nuclear star cluster,
nuclear stellar disk, thin and thick stellar disks, gas disks, and dark halo)
to create a realistic Milky Way potential satisfying present-day observational constraints.
The bar model is taken from `example_mw_bar_potential.py', but other components are different;
this variant rectifies some of the weaknesses of the original bar model that was not fitted
to the region outside the central 5 kpc.
The left panel shows the circular-velocity curve (in the axisymmetrized potential),
and the right panel shows examples of a few orbits in this potential.

Reference: Sormani et al. 2022 (MNRAS Letters/514/L5); Hunter et al. (in prep.)

Authors: Mattia Sormani, Glen Hunter, Eugene Vasiliev
'''
import agama, numpy, matplotlib.pyplot as plt
from example_mw_bar_potential import makeBarDensity

def makePotentialModel(axisymmetric=False):
    """
    Create the potential of the entire Milky Way model:
    3-component bar density as defined above,
    central structures (SgrA*, NSC and NSD),
    two stellar disks (thin and thick), two gas disks, and dark halo.
    If axisymmetric=True, create the axisymmetrized version of the potential,
    otherwise it has a triaxial symmetry (but is not rotating by itself).
    """

    # Analytic Plummer potential for SagA*
    # (NB: scale radius is arbitrarily set to 1 pc, much larger than the Schwarzschild radius!)
    params_BH = dict(type="Plummer", mass=4.1e6, scaleRadius=1e-3)

    # Axisymmetric flattened spheroidal potential for the nuclear star cluster (Chatzopoulos+ 2015)
    params_NSC = dict(type="Spheroid",
        mass=6.1e7, gamma=0.71, axisRatioZ=0.73,
        scaleRadius=0.0059, alpha=1, beta=4, outerCutoffRadius=0.1)

    # two-component axisymmetric flatten spheroidal potential for the nuclear stellar disk (Sormani+ 2020)
    params_NSD = [
        dict(type="Spheroid", densityNorm=2.00583e12, gamma=0, beta=0,
            axisRatioZ=0.37, outerCutoffRadius=0.00506, cutoffStrength=0.72),
        dict(type="Spheroid",densityNorm=1.53e12, gamma=0,
            beta=0, axisRatioZ=0.37, outerCutoffRadius=0.0246, cutoffStrength=0.79) ]

    # analytic bar density (Sormani+ 2022)
    dens_bar = makeBarDensity()

    # Stellar disk (thin/thick) with a central hole where the bar replaces it)
    params_disk = [
        dict(type="Disk", surfaceDensity=1.332e9, scaleRadius=2.0, scaleHeight=0.3,
            innerCutoffRadius=2.7, sersicIndex=1),  # thin stellar disk
        dict(type="Disk", surfaceDensity=8.97e8, scaleRadius=2.8, scaleHeight=0.9,
            innerCutoffRadius=2.7, sersicIndex=1) ] # thick stellar disk

    params_gas = [
        dict(type="Disk", surfaceDensity=5.81e7, scaleRadius=7, scaleHeight=-0.085,
            innerCutoffRadius=4, sersicIndex=1),   # HI gas disk
        dict(type="Disk", surfaceDensity=2.68e9, scaleRadius=1.5, scaleHeight=-0.045,
            innerCutoffRadius=12, sersicIndex=1) ] # molecular gas disk

    # Einasto Dark matter potential with rho_{-2} = 2.216e7 Msun/kpc^3 and r_{-2} = 16.42 kpc
    params_dark = dict(type="Spheroid", densitynorm=2.774e11, beta=0, gamma=0,
        outerCutoffRadius=8.682e-6, cutoffStrength = 0.1704)

    # total potential consists of only two components to maximize the efficiency
    return agama.Potential(
        agama.Potential(type="CylSpline", density=[dens_bar] + params_disk + params_gas,
            gridSizeR=30, gridSizez=25, Rmin=0.1, Rmax=50, zmin=0.05, zmax=20,
            mmax=8 if not axisymmetric else 0),
        agama.Potential(type="Multipole",
            density=[params_dark, params_BH, params_NSC] + params_NSD,
            lmax=12, gridSizeR=41, rmin=1e-5, rmax=1000),
        )


agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s
pot = makePotentialModel()
# create an axisymmetrized version of the potential for plotting the true circular-velocity curve
pot_axi = makePotentialModel(axisymmetric=True)

r=agama.nonuniformGrid(100, 0.001, 500)
xyz=numpy.column_stack((r,r*0,r*0))
ax=plt.subplots(1, 2, figsize=(16,8))[1]
ax[0].plot(r, (-r*pot_axi[0].force(xyz)[:,0])**0.5, 'c', label='disky components (bar, stellar and gas disks)')
ax[0].plot(r, (-r*pot_axi[1].force(xyz)[:,0])**0.5, 'y', label='spheroidal components (BH, NSC, NSD and halo)')
ax[0].plot(r, (-r*pot_axi   .force(xyz)[:,0])**0.5, 'r', label='total')
ax[0].legend(loc='lower right', frameon=False)
ax[0].set_xlabel('radius [kpc]')
ax[0].set_ylabel('circular velocity [km/s]')
ax[0].set_xlim(0,10)

# integrate and show a few orbits
numorbits=10
numpy.random.seed(42)
ic=numpy.random.normal(size=(numorbits,6)) * numpy.array([2.0, 0.0, 0.4, 50., 40., 30.])
ic[:,0] += -6.
ic[:,4] += 220
bar_angle = -25.0 * numpy.pi/180  # orientation of the bar w.r.t. the Sun
Omega = -39.0  # km/s/kpc - the value is negative since the potential rotates clockwise
orbits = agama.orbit(potential=pot, ic=ic, time=10., trajsize=1000, Omega=Omega)[:,1]
sina, cosa = numpy.sin(bar_angle), numpy.cos(bar_angle)
rmax = 10.0   # plotting range
cmap = plt.get_cmap('mist')
for i,o in enumerate(orbits):
    ax[1].plot(o[:,0]*cosa-o[:,1]*sina, o[:,0]*sina+o[:,1]*cosa, color=cmap(i*1.0/numorbits), lw=0.5)
ax[1].plot(-8.2,0.0, 'ko', ms=5)  # Solar position
ax[1].text(-8.0,0.0, 'Sun')
ax[1].set_xlabel('x [kpc]')
ax[1].set_ylabel('y [kpc]')
ax[1].set_xlim(-rmax, rmax)
ax[1].set_ylim(-rmax, rmax)

# overplot the surface density contours
print('Computing surface density of bar+disk')
gridr  = numpy.linspace(-rmax, rmax, 101)  # 1d grid
gridxy = numpy.column_stack((numpy.repeat(gridr, len(gridr)), numpy.tile(gridr, len(gridr))))  # 2d grid
Sigma  = pot[0].projectedDensity(gridxy, gamma=-bar_angle)  # surface density for a stellar component rotated by bar_angle
logSigma = 2.5 * numpy.log10(Sigma / numpy.max(Sigma))      # log-scaled to magnitudes per unit square
# thick lines spaced by one magnitude, thin lines lie halfway between thick ones
ax[1].contour(gridr, gridr, logSigma.reshape(len(gridr), len(gridr)).T,
    levels=numpy.linspace(-8,0,17), colors='k', zorder=5, linewidths=[2,1], linestyles='solid')

plt.tight_layout()
plt.show()
