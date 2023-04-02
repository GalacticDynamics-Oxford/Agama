#!/usr/bin/python
"""
Context: The Large Magellanic Cloud (LMC) is the most massive satellite of the Milky Way,
and it perturbs the motion of stars in the outer regions of the Galaxy due to two effects:
(1) stars that pass in its vicinity are deflected and form an overdensity behind the LMC
on its trajectory, which in turn is responsible for dynamical friction (local pertubations);
(2) the Milky Way itself is not fixed in space, but moves in response to the gravitational
pull of the LMC. Moreover, stars at large distances from the MW center experience different
amounts of perturbation from the LMC, and therefore are systematically shifted and
accelerated w.r.t. the Galactic center (global perturbations).

This script illustrates both effects in a simplified setup, where the relative motion
of the Milky Way and the LMC is approximated as if both galaxies had rigid (moving but
non-deforming) potentials, and then the orbits of test particles in the Milky Way halo
are computed in the resulting time-dependent potential of the MW-LMC system.
"""
import sys, agama, numpy, scipy.integrate, scipy.ndimage, scipy.special, matplotlib, matplotlib.pyplot as plt

agama.setUnits(length=1, velocity=1, mass=1)  # work in units of 1 kpc, 1 km/s, 1 Msun)

Trewind = -4.0  # initial time [Gyr] - the LMC orbit is computed back to that time
Tcurr   =  0.0  # current time
# heliocentric ICRS celestial coordinates and velocity of the LMC
# (PM from Luri+ 2021, distance from Pietrzynski+ 2019, center and velocity from van der Marel+ 2002)
ra, dec, dist, pmra, pmdec, vlos = 81.28, -69.78, 49.6, 1.858, 0.385, 262.2
# transform to Galactocentric cartesian position/velocity, using built-in routines from Agama
# (hence the manual conversion factors from degrees to radians and from mas/yr to km/s/kpc)
l, b, pml, pmb = agama.transformCelestialCoords(agama.fromICRStoGalactic,
    ra * numpy.pi/180, dec * numpy.pi/180, pmra, pmdec)
posvelLMC = agama.getGalactocentricFromGalactic(l, b, dist, pml*4.74, pmb*4.74, vlos)

# Create a simple but realistic model of the Milky Way with a bulge, a single disk,
# and a spherical dark halo
paramBulge = dict(
    type              = 'Spheroid',
    mass              = 1.2e10,
    scaleRadius       = 0.2,
    outerCutoffRadius = 1.8,
    gamma             = 0.0,
    beta              = 1.8)
paramDisk  = dict(
    type='Disk',
    mass              = 5.0e10,
    scaleRadius       = 3.0,
    scaleHeight       = -0.4)
paramHalo  = dict(
    type              = 'Spheroid',
    densityNorm       = 1.35e7,
    scaleRadius       = 14,
    outerCutoffRadius = 300,
    cutoffStrength    = 4,
    gamma             = 1,
    beta              = 3)
densMWhalo = agama.Density(paramHalo)
potMW      = agama.Potential(paramBulge, paramDisk, paramHalo)

# create a sphericalized MW potential and a corresponding isotropic halo distribution function
potMWsph   = agama.Potential(type='Multipole', potential=potMW, lmax=0, rmin=0.01, rmax=1000)
gmHalo     = agama.GalaxyModel(potMWsph,
    agama.DistributionFunction(type='quasispherical', density=densMWhalo, potential=potMWsph))

# compute the velocity dispersion in the MW halo needed for the dynamical friction
rgrid      = numpy.logspace(1, 3, 16)
xyzgrid    = numpy.column_stack([rgrid, rgrid*0, rgrid*0])
sigmafnc   = agama.Spline(rgrid, gmHalo.moments(xyzgrid, dens=False, vel=False, vel2=True)[:,0]**0.5)

# Create the LMC potential - a spherical truncated NFW profile with mass and radius
# related by the equation below, which produces approximately the same enclosed mass
# profile in the inner region, satisfying the observational constraints, as shown
# in Fig.3 of Vasiliev,Belokurov&Erkal 2021.
massLMC    = 1.5e11
radiusLMC  = (massLMC/1e11)**0.6 * 8.5
bminCouLog = radiusLMC * 2.0   # minimum impact parameter in the Coulomb logarithm
potLMC     = agama.Potential(
    type              = 'spheroid',
    mass              = massLMC,
    scaleradius       = radiusLMC,
    outercutoffradius = radiusLMC*10,
    gamma             = 1,
    beta              = 3)

######## PART ONE ########
# Simulate (approximately!) the past trajectory of the MW+LMC system under mutual gravity.
# Here, we integrate in time a 12-dimensional ODE system for positions & velocities of
# both galaxies in the external inertial reference frame. The acceleration of each galaxy
# is computed by taking the gradient of the rigid (non-deforming) potential of the other
# galaxy at the location of the first galaxy's center, and then assuming that the entire
# first galaxy experiences the same acceleration and continues to move as a rigid body.
# The same procedure then is applied in reverse. Moreover, we add a dynamical friction
# acceleration to the LMC, but not to the Milky Way; it is computed using the standard
# Chandrasekhar's formula, but with a spatially-varying value of Coulomb logarithm,
# which has been calibrated against full N-body simulations.
# This simplified model is certainly not physically correct, e.g. manifestly violates
# Newton's third law, but still captures the main features of the actual interaction.
print("Computing the past orbits of the Milky Way and the LMC")
def difeq(vars, t):
    x0    = vars[0:3]          # MW position
    v0    = vars[3:6]          # MW velocity
    x1    = vars[6:9]          # LMC position
    v1    = vars[9:12]         # LMC velocity
    dx    = x1-x0              # relative offset
    dv    = v1-v0              # relative velocity
    dist  = sum(dx**2)**0.5    # distance between the galaxies
    vmag  = sum(dv**2)**0.5    # magnitude of relative velocity
    f0    = potLMC.force(-dx)  # force from LMC acting on the MW center
    f1    = potMW .force( dx)  # force from MW acting on the LMC
    rho   = potMW.density(dx)  # actual MW density at this point
    sigma = sigmafnc(dist)     # approximate MW velocity dispersion at this point
    # distance-dependent Coulomb logarithm
    # (an approximation that best matches the results of N-body simulations)
    couLog= max(0, numpy.log(dist / bminCouLog)**0.5)
    X     = vmag / (sigma * 2**.5)
    drag  = -(4*numpy.pi * rho * dv / vmag *
        (scipy.special.erf(X) - 2/numpy.pi**.5 * X * numpy.exp(-X*X)) *
        massLMC * agama.G**2 / vmag**2 * couLog)   # dynamical friction force
    return numpy.hstack((v0, f0, v1, f1 + drag))

Tstep   = 1./64
tgrid   = numpy.linspace(Trewind, Tcurr, round((Tcurr-Trewind)/Tstep)+1)
ic      = numpy.hstack((numpy.zeros(6), posvelLMC))
sol     = scipy.integrate.odeint(difeq, ic, tgrid[::-1])[::-1]

# After obtaining the solution for trajectories of both galaxies,
# we transform it into a more convenient form, namely, into the non-inertial
# reference frame centered at the Milky Way center at all times.
# In this frame, the total time-dependent gravitational potential consists of
# three terms. First is the rigid potential of the Milky Way itself.
# Because the latter moves on a curvilinear trajectory, we need to add
# a corresponding spatially uniform acceleration field. Finally, the potential
# of the LMC is also rigid but moves in space.

# LMC trajectory in the MW-centric (non-inertial) reference frame
# (7 columns: time, 3 position and 3 velocity components)
trajLMC = numpy.column_stack([tgrid, sol[:,6:12] - sol[:,0:6]])
# MW trajectory in the inertial frame
trajMWx = agama.Spline(tgrid, sol[:,0], der=sol[:,3])
trajMWy = agama.Spline(tgrid, sol[:,1], der=sol[:,4])
trajMWz = agama.Spline(tgrid, sol[:,2], der=sol[:,5])
# MW centre acceleration is minus the second derivative of its trajectory in the inertial frame
accMW   = numpy.column_stack([tgrid, -trajMWx(tgrid, 2), -trajMWy(tgrid, 2), -trajMWz(tgrid, 2)])
potacc  = agama.Potential(type='UniformAcceleration', file=accMW)
potLMCm = agama.Potential(potential=potLMC, center=trajLMC)  # potential of the moving LMC

# finally, the total time-dependent potential in the non-inertial MW-centric reference frame
potTotal= agama.Potential(potMW, potLMCm, potacc)

######## PART TWO ########
# Compute the perturbations caused by the moving LMC and the accelerating
# Milky Way on the orbits of test particles in the Milky Way halo.
# We create a large sample of stars initially in equilibrium with
# the isolated Milky Way potential, then integrate their orbits
# in the time-dependent potential of two interacting galaxies,
# and plot the changes in density and mean velocity in several radial bins.

if len(sys.argv)>1:
    Nstars = int(float(sys.argv[1]))
else:
    Nstars = 100000

# create the initial conditions for the halo objects (for simplicity,
# using the density profile of the DM halo rather than a separate stellar halo)
ic = gmHalo.sample(Nstars)[0]

# integrate the orbits of these objects in the time-dependent total potential
# and record the "final conditions" (present-day position&velocity)
print("Integrating the trajectories of %i stars in the Milky Way halo "
    "in the time-dependent potential of the LMC + MW" % len(ic))
if Nstars < 1e6:
    print("To increase the resolution and reduce the Poisson noise in maps, run the script "
        "with a larger number of orbits, e.g. 1e6 (provide the number in the command line)")

fc = numpy.vstack(agama.orbit(potential=potTotal, ic=ic, time=Tcurr-Trewind,
    timestart=Trewind, trajsize=1)[:,1])

# present-day positions and Solar reflex-corrected velocities of these stars
# in the heliocentric system (for an observer at rest w.r.t. the Galactic center)
l,b,d,ml,mb,vl = agama.getGalacticFromGalactocentric(*fc.T, galcen_v_sun=[0,0,0])
ml *= d; mb *= d

# the background density of stars in the unperturbed halo could be taken from "ic",
# but to reduce the Poisson noise, we create a higher-resolution sample of stars
oversampling_factor = 10
ic_hr = densMWhalo.sample(Nstars * oversampling_factor)[0]
# convert the positions to the heliocentric system
l0,b0,d0 = agama.getGalacticFromGalactocentric(*ic_hr.T)

# LMC trajectory in the heliocentric system
lLMC, bLMC, dLMC = agama.getGalacticFromGalactocentric(*trajLMC[:,1:4].T)

def projectMollweide(lon, lat):
    """
    Convert the longitude/latitude expressed in radians into coordinates in the ellipse
    with semimajor axes 2x1, typical for showing all-sky maps of some quantity
    """
    ang = numpy.array(lat)
    bla = numpy.pi/2 * numpy.sin(lat)
    # solve a nonlinear equation for ang by Newton's method, carefully designing the first approximation
    w = (1 - abs(ang) * 2/numpy.pi)**(2./3)
    ang = (1 - w * (1 + (1-w) * (-0.09 - 0.086*w) )) * numpy.sign(ang) * numpy.pi/2
    for it in range(3):
        ang -= 0.5 * (ang + 0.5*numpy.sin(2*ang) - bla) / numpy.cos(ang)**2
    X = lon * 2/numpy.pi * numpy.cos(ang)
    Y = numpy.sin(ang)
    return X, Y

X,Y   = projectMollweide(l, b)
X0,Y0 = projectMollweide(l0,b0)
Xl,Yl = projectMollweide(lLMC,bLMC)
# show perturbation maps as 2d histograms in X,Y
gridX, gridY = numpy.linspace(-2,2,81), numpy.linspace(-1,1,41)
centX, centY = (gridX[1:]+gridX[:-1])/2, (gridY[1:]+gridY[:-1])/2
cntrX = numpy.repeat(centX, len(centY)).reshape(len(centX), len(centY))
cntrY = numpy.tile  (centY, len(centX)).reshape(len(centX), len(centY))
def smooth(hist):
    # Gaussian smoothing of a region inside the ellipse
    smoothing = (Nstars/1e7)**-0.25
    hist[hist==0] = numpy.mean(hist)  # fill the outside region with the mean value
    hist = scipy.ndimage.gaussian_filter(hist, smoothing)
    hist[ (cntrX/2.0)**2 + cntrY**2 > 1.0 ] = numpy.nan
    return hist

def showmap(ax, qty):
    ax.set_axis_off()
    ax.imshow(qty.T, extent=[-2,2,-1.01,0.99], aspect='auto', interpolation='nearest', origin='lower',
        cmap='bluered', vmin=-1, vmax=1)
    ax.add_artist(matplotlib.patches.Ellipse((0,0), 4, 2, fill=False, color='k', lw=0.5, clip_on=False))
    ax.set_xlim(2, -2)
    ax.set_ylim(-1, 1)
    ax.plot(Xl, Yl, color='g', dashes=[2,1.5], lw=0.25)
    ax.plot(Xl[dlf], Yl[dlf], color='g', lw=0.75)
    ax.plot(Xl[-1 ], Yl[-1 ], 'o', mew=0, ms=1.5, color='g')

plt.rc('axes', linewidth=0.5)
plt.rc('font', size=6)
plt.rc('ytick.major', size=1)
plt.figure(figsize=(6.4,3.2), dpi=200)
distbins = [0, 30, 60, 100, 150]
for i in range(4):
    dmin = distbins[i]
    dmax = distbins[i+1]
    filt0= (d0  >=dmin) * (d0  <=dmax)
    filt = (d   >=dmin) * (d   <=dmax)
    dlf  = (dLMC>=dmin) * (dLMC<=dmax)
    print('%g < D [kpc] < %g: %i stars' % (dmin, dmax, numpy.sum(filt)))
    his0 = smooth(numpy.histogram2d(X0[filt0], Y0[filt0], bins=(gridX,gridY))[0]) / oversampling_factor
    his  = smooth(numpy.histogram2d(X[filt], Y[filt], bins=(gridX,gridY))[0])
    hml  = smooth(numpy.histogram2d(X[filt], Y[filt], bins=(gridX,gridY), weights=ml[filt])[0]) / his
    hmb  = smooth(numpy.histogram2d(X[filt], Y[filt], bins=(gridX,gridY), weights=mb[filt])[0]) / his
    hvl  = smooth(numpy.histogram2d(X[filt], Y[filt], bins=(gridX,gridY), weights=vl[filt])[0]) / his
    showmap(plt.axes([0.005 + i*0.22, 0.73, 0.20, 0.20]), his/his0-1)  # over/underdensity map
    showmap(plt.axes([0.005 + i*0.22, 0.49, 0.20, 0.20]), hml/80.0)    # mean velocity in the "l" direction
    showmap(plt.axes([0.005 + i*0.22, 0.25, 0.20, 0.20]), hmb/80.0)    # mean velocity in the "b" direction
    showmap(plt.axes([0.005 + i*0.22, 0.01, 0.20, 0.20]), hvl/80.0)    # mean line-of-sight velocity
    plt.text(0.115 + i*0.22, 0.99, '%g < D [kpc] < %g' % (dmin, dmax), ha='center', va='top',
        transform=plt.gcf().transFigure, fontsize=7)

def showcolorbar(ax, label, vmin, vmax):
    ax.imshow(numpy.linspace(0,1,256).reshape(-1,1), extent=[0,1,vmin,vmax], aspect='auto', interpolation='nearest',
        origin='lower', cmap='bluered', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_ylabel(label, labelpad=-40, fontsize=8)

showcolorbar(plt.axes([0.89, 0.73, 0.02, 0.20]), r'$\rho / \rho_0 - 1$', -1, 1)
showcolorbar(plt.axes([0.89, 0.49, 0.02, 0.20]), r'$D\times \mu_l \;\sf[km/s]$', -80, 80)
showcolorbar(plt.axes([0.89, 0.25, 0.02, 0.20]), r'$D\times \mu_b \;\sf[km/s]$', -80, 80)
showcolorbar(plt.axes([0.89, 0.01, 0.02, 0.20]), r'$v_\mathsf{los} \;\sf[km/s]$', -80, 80)
#plt.savefig('example_lmc_mw_interaction.pdf')
plt.show()
