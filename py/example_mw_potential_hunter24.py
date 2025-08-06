#!/usr/bin/python
"""
This script uses the analytic approximation for the bar model from Portail et al.(2017),
in combination with several other mass components (central black hole, nuclear star cluster,
nuclear stellar disk, thin and thick stellar disks, gas disks, dark halo, and spiral arms)
to create a realistic Milky Way potential satisfying present-day observational constraints.
The bar model is taken from `example_mw_bar_potential.py', but other components are different;
this variant rectifies some of the weaknesses of the original bar model that was not fitted
to the region outside the central 5 kpc.
The left panel shows the circular-velocity curve (in the axisymmetrized potential),
and the right panel shows examples of a few orbits in this potential.
After running this script, several potential variants are stored in .ini files,
and can be reused later for doing orbit integrations and other purposes
(see the docstring of makePotentialModel() for details).

References: Sormani et al. 2022 (MNRAS Letters/514/L5); Hunter et al. 2024 (A&A/642/216)

Authors: Mattia Sormani, Glen Hunter, Eugene Vasiliev
"""
import agama, numpy, matplotlib.pyplot as plt, scipy.special
from example_mw_bar_potential import makeBarDensity

agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s
filename_full     = 'MWPotentialHunter24_full.ini'
filename_axi      = 'MWPotentialHunter24_axi.ini'
filename_spiral   = 'MWPotentialHunter24_spiral.ini'
filename_rotating = 'MWPotentialHunter24_rotating.ini'
filename_rotspiral= 'MWPotentialHunter24_rotspiral.ini'
Omega_bar         = -37.5  # bar pattern speed in units of km/s/kpc, negative since the bar rotates CW
Omega_spiral      = -22.5  # spiral pattern speed
angle_bar         = -0.44  # in radians; corresponds to approximately -25 degrees

def makePotentialModel():
    """
    Create the potential of the entire Milky Way model:
    3-component bar density as defined in a separate file (example_mw_bar_potential.py),
    central structures (SgrA*, NSC and NSD), two stellar disks (thin and thick),
    two gas disks (same as in McMillan 2017), dark halo, and optionally four spiral arms.
    Even though the model has many ingredients, the total potential consists of
    only two components (three if spiral arms are used): all spherical or axisymmetric
    structures (SgrA*, NSC, NSD, and dark halo) are bunched together into a single Multipole
    potential, and all remaining baryonic components (bar, thin/thick stellar disks, and
    two gas disks) are represented by a single triaxial CylSpline potential.
    The spiral arms are represented by a separate bisymmetric CylSpline potential,
    since (1) they are optional, (2) their pattern speed differs from that of the bar.
    The resulting potentials are stored in several files:
    MWPotentialHunter2024_full.ini - the combination of all components except spiral arms,
        i.e., axisymmetric Multipole and triaxial CylSpline, with the bar oriented along
        the x axis, non-rotating.
    MWPotentialHunter2024_axi.ini - same components, but using axisymmetrized version of
        the CylSpline, suitable for plotting the circular-velocity curve.
    MWPotentialHunter2024_spiral.ini - a separate CylSpline potential for spiral arms,
        oriented to match their location relative to the bar, non-rotating.
    MWPotentialHunter2024_rotating.ini - all components except spirals,
        with added clockwise rotation with the angular speed Omega_bar = -37.5 km/s/kpc
        and oriented such that at t=0 the bar major axis is at -25 degrees from the x axis.
    MWPotentialHunter2024_rotspiral.ini - same but with added spirals
        rotating clockwise with the angular speed Omega_spiral = -22.5 km/s/kpc.
    """

    # Analytic Plummer potential for SgrA*
    # (NB: scale radius is arbitrarily set to 1 pc, much larger than the Schwarzschild radius!)
    params_BH = dict(type='Plummer', mass=4.1e6, scaleRadius=1e-3)

    # Axisymmetric flattened spheroidal potential for the nuclear star cluster (Chatzopoulos+ 2015)
    params_NSC = dict(type='Spheroid', mass=6.1e7, gamma=0.71, beta=4, alpha=1,
        axisRatioZ=0.73, scaleRadius=0.0059, outerCutoffRadius=0.1)

    # two-component axisymmetric flatten spheroidal potential for the nuclear stellar disk (Sormani+ 2020)
    params_NSD = [
        dict(type='Spheroid', densityNorm=2.00583e12, gamma=0, beta=0, alpha=1,
            axisRatioZ=0.37, outerCutoffRadius=0.00506, cutoffStrength=0.72),
        dict(type='Spheroid',densityNorm=1.53e12, gamma=0, beta=0, alpha=1,
            axisRatioZ=0.37, outerCutoffRadius=0.0246, cutoffStrength=0.79) ]

    # analytic bar density (Sormani+ 2022)
    dens_bar = makeBarDensity()

    # Stellar disk (thin/thick) with a central hole where the bar replaces it)
    params_disk = [
        dict(type='Disk', surfaceDensity=1.332e9, scaleRadius=2.0, scaleHeight=0.3,
            innerCutoffRadius=2.7, sersicIndex=1),  # thin stellar disk
        dict(type='Disk', surfaceDensity=8.97e8, scaleRadius=2.8, scaleHeight=0.9,
            innerCutoffRadius=2.7, sersicIndex=1) ] # thick stellar disk

    params_gas = [
        dict(type='Disk', surfaceDensity=5.81e7, scaleRadius=7, scaleHeight=-0.085,
            innerCutoffRadius=4, sersicIndex=1),   # HI gas disk
        dict(type='Disk', surfaceDensity=2.68e9, scaleRadius=1.5, scaleHeight=-0.045,
            innerCutoffRadius=12, sersicIndex=1) ] # molecular gas disk

    # Einasto Dark matter potential with rho_{-2} = 2.216e7 Msun/kpc^3 and r_{-2} = 16.42 kpc
    params_dark = dict(type='Spheroid', densitynorm=2.774e11, gamma=0, beta=0, alpha=1,
        outerCutoffRadius=8.682e-6, cutoffStrength = 0.1704)

    # all spheroidal components put into a single axisymmetric Multipole potential
    pot_mul = agama.Potential(type='Multipole',
        density=agama.Density(params_dark, params_BH, params_NSC, *params_NSD),
        lmax=12, gridSizeR=36, rmin=1e-4, rmax=1000)

    # all remaining components (except spirals) are represented by a CylSpline potential
    # in two variants: axisymmetric (suitable for plotting the circular velocity profile)
    # and triaxial (for all other purposes)
    params_cylspline = dict(type='CylSpline',
        density=agama.Density(dens_bar, *(params_disk + params_gas)),
        gridSizeR=30, gridSizez=32, Rmin=0.1, Rmax=200, zmin=0.05, zmax=200)
    pot_cyl_axi  = agama.Potential(mmax=0, **params_cylspline)
    pot_cyl_full = agama.Potential(mmax=8, **params_cylspline)

    # spiral arms go into a separate CylSpline potential, which has zero amplitude in the m=0 term,
    # so does not contribute to the axisymmetrized version of the total potential
    density_disk = agama.Density(*params_disk)   # density of the stellar disk
    m   = 2                     # number of spiral arms
    i   = 12.5 * numpy.pi/180   # pitch angle of the logarithmic spiral
    Ra  = 9.64                  # fiducial radius [kpc]
    sig = 5.0                   # width of the spiral [kpc]
    # two sets of m=2 spirals with different phase angles (not equivalent to a single m=4 spiral)
    phi1= 139.5 * numpy.pi/180  # phase angle of the first set of spiral arms
    phi2= 69.75 * numpy.pi/180  # phase angle of the second set of spiral arms
    def densfnc(pos):
        R   = (pos[:,0]**2 + pos[:,1]**2)**0.5
        phi = numpy.arctan2(pos[:,1], pos[:,0])
        F   = (m/numpy.tan(i)) * numpy.log(numpy.maximum(R, 1e-12) / Ra)
        S   = (-2 * scipy.special.i0e(-(R/sig)**2) +
            numpy.exp(-(R/sig)**2 * (1 - numpy.cos(m * (phi+phi1) - F))) +
            numpy.exp(-(R/sig)**2 * (1 - numpy.cos(m * (phi+phi2) - F))) )
        # the minimum of S for the entire range of phi and R is around -0.33 for each of the two spiral components;
        # make sure that the total amplitude of the spiral is no larger than -1
        amp = numpy.minimum(0.36 * (R/8.179)**2, 1.5)
        return amp * S * density_disk.density(pos)
    pot_spiral = agama.Potential(type='CylSpline', density=densfnc,
        Rmin=0.2, Rmax=40, gridSizeR=80, zmin=0.1, zmax=10, gridSizeZ=20, mmax=8, symmetry='b')

    # now export these potentials into several files
    agama.Potential(pot_mul, pot_cyl_full).export(filename_full)
    agama.Potential(pot_mul, pot_cyl_axi ).export(filename_axi)
    pot_spiral.export(filename_spiral)

    # the remaining two files with rotation are created "manually", since the Potential.export() method
    # currently cannot store the information about potential modifiers (including rotation)
    rotation_bar = '''file=%s
# The line below specifies both the orientation of the bar and the its pattern speed as follows:
# the first pair of numbers [0,phi0] sets the angle of CCW rotation of the bar (in radians)
# at present time (t=0);
# the second pair [1,phi0+Omega*1] sets its angle at time t=1.
# Given that there are only two numbers, the angle is linearly interpolated between them,
# and always extrapolated linearly beyond the endpoints of the interval, corresponding to
# a constant pattern speed Omega (here it is negative, i.e. CW rotation).
# Units are: length=1 kpc, velocity=1 km/s (hence pattern speed is km/s/kpc);
# there are no potential parameters with the dimension of mass, so the mass unit is unspecified.
rotation=[[0,%.2f],[1,%.2f]]
''' % (filename_full, angle_bar, angle_bar + Omega_bar)
    with open(filename_rotating, 'w') as f:
        f.write('''# Milky Way potential from Hunter+ 2024 with added rotation
[Potential]
''' + rotation_bar)
    with open(filename_rotspiral, 'w') as f:
        f.write('''# Milky Way potential from Hunter+ 2024 with added rotation and spiral arms
[Potential main]
%s
[Potential spiral]
file=%s
# same principle as above, but for a different pattern speed of the spiral arms
rotation=[[0,%.2f],[1,%.2f]]
''' % (rotation_bar, filename_spiral, angle_bar, angle_bar + Omega_spiral))


##### MAIN PROGRAM #####

# load three versions of the potential
try:
    pot_axi = agama.Potential(filename_axi)
except RuntimeError:  # file does not exist - create on the first run
    makePotentialModel()
    pot_axi = agama.Potential(filename_axi)         # axisymmetrized version - for plotting the circular-velocity curve
pot_full = agama.Potential(filename_full)           # full potential with a bar but no spirals - for orbit integration
pot_rotating = agama.Potential(filename_rotating)   # same but already rotating - for another variant of orbit integration
# for plotting the surface density, use only the cylspline component (i.e. stars) of the full potential
pot_plot = pot_full[1]
assert 'CylSpline' in str(pot_plot)  # make sure we didn't mix up indexing of components

r=agama.nonuniformGrid(100, 0.001, 500)
xyz=numpy.column_stack((r,r*0,r*0))
ax=plt.subplots(1, 2, figsize=(12,6), dpi=100)[1]
ax[0].plot(r, (-r*pot_axi[0].force(xyz)[:,0])**0.5, 'y', label='spheroidal components (BH, NSC, NSD and halo)')
ax[0].plot(r, (-r*pot_axi[1].force(xyz)[:,0])**0.5, 'c', label='disky components (bar, stellar and gas disks)')
ax[0].plot(r, (-r*pot_axi   .force(xyz)[:,0])**0.5, 'r', label='total')
ax[0].legend(loc='lower right', frameon=False, fontsize=12)
ax[0].set_xlabel('radius [kpc]')
ax[0].set_ylabel('circular velocity [km/s]')
ax[0].set_xlim(0,10)

# integrate and show a few orbits, using several approaches;
# ICs are given in the Galactocentric frame, in which the Sun is located approximately at x=-8 kpc
time = 10.0
trajsize = 1001
numorbits = 8
numpy.random.seed(42)
ic=numpy.random.normal(size=(numorbits,6)) * numpy.array([2.0, 0.0, 0.4, 50., 40., 30.])
ic[:,0] += -6.
ic[:,4] += 220
colors = plt.get_cmap('mist')(numpy.linspace(0, 1, numorbits+1)[:-1])
rmax = 10.0  # plotting range

# option 1: perform orbit integration in the rotating frame, in which the bar orientation is kept fixed,
# using pot_full (a version of the potential with the bar oriented along the x axis).
# Since in the real Milky Way the bar is actually rotated by some angle_bar w.r.t. x axis,
# we have two further options: either rotate our ICs or change the orientation of the potential.

# option 1a: rotate the ICs into the coordinate frame where the bar is oriented along the x axis (i.e. pot_full),
# integrate the orbits in the rotating frame, then rotate the result back into the Galactocentric frame.
sina, cosa = numpy.sin(angle_bar), numpy.cos(angle_bar)
ic_rot = numpy.column_stack([
    ic[:,0] * cosa + ic[:,1] * sina,  # coordinates in the frame rotated CCW by angle_bar
    ic[:,1] * cosa - ic[:,0] * sina,  # (note that angle_bar is negative in our case)
    ic[:,2],
    ic[:,3] * cosa + ic[:,4] * sina,
    ic[:,4] * cosa - ic[:,3] * sina,
    ic[:,5],
])
times, orbits = agama.orbit(potential=pot_full, ic=ic_rot, time=time, trajsize=trajsize, Omega=Omega_bar).T
for i in range(numorbits):
    ax[1].plot(
        orbits[i][:,0] * cosa - orbits[i][:,1] * sina,  # rotate back by the same angle
        orbits[i][:,1] * cosa + orbits[i][:,0] * sina,
        color=colors[i], lw=0.5, label=str(i))

# option 1b: stick to the Galactocentric coordinate system for ICs and output trajectories,
# but change the orientation of the potential (keeping it fixed at angle_bar, *not* rotating with time);
# orbit integration is still carried out in the frame corotating with the bar.
pot_rotated = agama.Potential(potential=pot_full, rotation=angle_bar)
times, orbits = agama.orbit(potential=pot_rotated, ic=ic, time=time, trajsize=trajsize, Omega=Omega_bar).T
for i in range(numorbits):
    ax[1].plot(orbits[i][:,0], orbits[i][:,1], color=colors[i], lw=0.5, dashes=[4,2])

# option 2: use the rotating potential and perform integration in the inertial frame,
# then transform the resulting orbit into the frame corotating with the bar.
# The difference from the previous one is that the potential is now rotating in time, not just rotated once,
# and its orientation at t=0 is already set to angle_bar.
# This option is more general than the first two, since it can be used with a more complicated potential
# pot_spiral, which has two components rotating with different pattern speeds.
times, orbits = agama.orbit(potential=pot_rotating, ic=ic, time=time, trajsize=trajsize).T
for i in range(numorbits):
    sina, cosa = numpy.sin(times[i] * Omega_bar), numpy.cos(times[i] * Omega_bar)
    ax[1].plot(
        orbits[i][:,0] * cosa + orbits[i][:,1] * sina,
        orbits[i][:,1] * cosa - orbits[i][:,0] * sina,
        color=colors[i], lw=1, dashes=[2,4])

# all three sets of curves should be nearly identical, at least near the beginning of the orbit;
# due to accumulation of roundoff errors, trajectories integrated in rotating and non-rotating frames
# eventually diverge, linearly for regular orbits or exponentially for chaotic ones.
ax[1].legend(frameon=True, fontsize=10)
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
Sigma  = pot_plot.projectedDensity(gridxy, gamma=-angle_bar)  # surface density for a stellar component rotated by angle_bar
logSigma = 2.5 * numpy.log10(Sigma / numpy.max(Sigma))        # log-scaled to magnitudes per unit square
# thick lines spaced by one magnitude, thin lines lie halfway between thick ones
ax[1].contour(gridr, gridr, logSigma.reshape(len(gridr), len(gridr)).T,
    levels=numpy.linspace(-8,0,17), colors='k', zorder=5, linewidths=[2,1], linestyles='solid')

plt.tight_layout()
plt.show()
