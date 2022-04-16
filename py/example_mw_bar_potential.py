#!/usr/bin/python
'''
This script defines an analytic approximation for the barred Milky Way model from Portail et al.(2017)
and constructs a corresponding CylSpline potential, which can be used to integrate orbits, etc.
The density is represented by four components: an X-shaped inner bar, two instances of long bars,
and an axisymmetric disk. In addition, there is a 'central mass concentration' (a triaxial disk)
and a flattened axisymmetric dark halo, which is represented by a separate Multipole potential.
This potential model is a good fit for the central region of the Galaxy (within ~5kpc),
but is not very realistic further out.
The left panel shows the circular-velocity curve (in the axisymmetrized potential),
and the right panel shows examples of a few orbits in this potential.

Reference: Sormani et al.(submitted)

Authors: Mattia Sormani, Eugene Vasiliev
'''
import agama, numpy, matplotlib.pyplot as plt

# Nearly identical to the built-in Disk density profile, but with a slightly different
# vertical profile containing an additional parameter 'verticalSersicIndex'
def makeDisk(**params):
    surfaceDensity      = params['surfaceDensity']
    scaleRadius         = params['scaleRadius']
    scaleHeight         = params['scaleHeight']
    innerCutoffRadius   = params['innerCutoffRadius']
    sersicIndex         = params['sersicIndex']
    verticalSersicIndex = params['verticalSersicIndex']
    def density(xyz):
        R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
        return (surfaceDensity / (4*scaleHeight) *
            numpy.exp( - (R/scaleRadius)**sersicIndex - innerCutoffRadius/(R+1e-100)) /
            numpy.cosh( (abs(xyz[:,2]) / scaleHeight)**verticalSersicIndex ) )
    return agama.Density(density)

# Modification of equation 9 of Coleman et al. 2020 (https://arxiv.org/abs/1911.04714)
def makeXBar(**params):
    densityNorm = params['densityNorm']
    x0   = params['x0']
    y0   = params['y0']
    z0   = params['z0']
    xc   = params['xc']
    yc   = params['yc']
    c    = params['c']
    alpha= params['alpha']
    cpar = params['cpar']
    cperp= params['cperp']
    m    = params['m']
    n    = params['n']
    outerCutoffRadius = params['outerCutoffRadius']
    def density(xyz):
        r  = numpy.sum(xyz**2, axis=1)**0.5
        a  = ( ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(cpar/cperp) +
            (abs(xyz[:,2]) / z0)**cpar )**(1/cpar)
        ap = ( ((xyz[:,0] + c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
        am = ( ((xyz[:,0] - c * xyz[:,2]) / xc)**2 + (xyz[:,1] / yc)**2 )**(0.5)
        return (densityNorm / numpy.cosh(a**m) * numpy.exp( -(r/outerCutoffRadius)**2) *
            (1 + alpha * (numpy.exp(-ap**n) + numpy.exp(-am**n) ) ) )
    return agama.Density(density)

# Modification of equation 9 of Wegg et al. 2015 (https://arxiv.org/pdf/1504.01401.pdf)
def makeLongBar(**params):
    densityNorm = params['densityNorm']
    x0   = params['x0']
    y0   = params['y0']
    cpar = params['cpar']
    cperp= params['cperp']
    scaleHeight = params['scaleHeight']
    innerCutoffRadius   = params['innerCutoffRadius']
    outerCutoffRadius   = params['outerCutoffRadius']
    innerCutoffStrength = params['innerCutoffStrength']
    outerCutoffStrength = params['outerCutoffStrength']
    def density(xyz):
        R = (xyz[:,0]**2 + xyz[:,1]**2)**0.5
        a = ( (abs(xyz[:,0]) / x0)**cperp + (abs(xyz[:,1]) / y0)**cperp )**(1/cperp)
        return densityNorm / numpy.cosh(xyz[:,2] / scaleHeight)**2 * numpy.exp(-a**cpar
            -(R/outerCutoffRadius)**outerCutoffStrength - (innerCutoffRadius/R)**innerCutoffStrength)
    return agama.Density(density)

# additional central mass concentration as described in sec.7.3 of Portail et al.(2017)
def makeCMC(mass, scaleRadius, scaleHeight, axisRatioY):
    norm = mass / (4 * numpy.pi * scaleRadius**2 * scaleHeight * axisRatioY)
    return agama.Density(lambda xyz:
        norm * numpy.exp(-(xyz[:,0]**2 + (xyz[:,1]/axisRatioY)**2)**0.5 / scaleRadius
            - abs(xyz[:,2]) / scaleHeight) )

# create the total density profile with 4 component from the provided array of unnamed parameters
def makeDensityModel(params):
    ind=0
    densityDisk = makeDisk(
        surfaceDensity=params[ind+0],
        scaleRadius=params[ind+1],
        innerCutoffRadius=params[ind+2],
        scaleHeight=params[ind+3],
        sersicIndex=params[ind+4],
        verticalSersicIndex=params[ind+5])
    ind+=6
    densityXBar = makeXBar(
         densityNorm=params[ind+0],
         x0=params[ind+1],
         y0=params[ind+2],
         z0=params[ind+3],
         cpar=params[ind+4],
         cperp=params[ind+5],
         m=params[ind+6],
         outerCutoffRadius=params[ind+7],
         alpha=params[ind+8],
         c=params[ind+9],
         n=params[ind+10],
         xc=params[ind+11],
         yc=params[ind+12])
    ind+=13
    densityLongBar1 = makeLongBar(
        densityNorm=params[ind+0],
        x0=params[ind+1],
        y0=params[ind+2],
        scaleHeight=params[ind+3],
        cperp=params[ind+4],
        cpar=params[ind+5],
        outerCutoffRadius=params[ind+6],
        innerCutoffRadius=params[ind+7],
        outerCutoffStrength=params[ind+8],
        innerCutoffStrength=params[ind+9] )
    ind+=10
    densityLongBar2 = makeLongBar(
        densityNorm=params[ind+0],
        x0=params[ind+1],
        y0=params[ind+2],
        scaleHeight=params[ind+3],
        cperp=params[ind+4],
        cpar=params[ind+5],
        outerCutoffRadius=params[ind+6],
        innerCutoffRadius=params[ind+7],
        outerCutoffStrength=params[ind+8],
        innerCutoffStrength=params[ind+9] )
    ind+=10
    assert len(params)==ind, 'invalid number of parameters'
    return agama.Density(densityDisk, densityXBar, densityLongBar1, densityLongBar2)


# create the potential of the entire model:
# 4-component stellar density as defined above, plus central mass concentration, plus dark halo
def makePotentialModel(params):
    # combined 4 components and the CMC represented by a single triaxial CylSpline potential
    mmax = 12  # order of azimuthal Fourier expansion (higher order means better accuracy,
    # but values greater than 12 *significantly* slow down the computation!)
    pot_bary = agama.Potential(type='CylSpline',
        density=agama.Density(makeDensityModel(params), makeCMC(0.2e10, 0.25, 0.05, 0.5)),
        symmetry='t', mmax=mmax, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20)
    # flattened axisymmetric dark halo with the Einasto profile
    pot_dark = agama.Potential(type='Multipole',
        density='Spheroid', axisratioz=0.8, gamma=0, beta=0,
        outerCutoffRadius=1.84, cutoffStrength=0.74, densityNorm=0.0263e10,
        gridsizer=26, rmin=0.01, rmax=1000, lmax=8)
    return agama.Potential(pot_bary, pot_dark)

params = numpy.array(
  # disk
  [ 1.03063359e+09, 4.75409497e+00, 4.68804907e+00, 1.51100601e-01,
    1.53608780e+00, 7.15915848e-01 ] +
  # short/thick bar
  [ 3.16273226e+09, 4.90209137e-01, 3.92017253e-01, 2.29482096e-01,
    1.99110223e+00, 2.23179266e+00, 8.73227940e-01, 4.36983774e+00,
    6.25670015e-01, 1.34152138e+00, 1.94025114e+00, 7.50504078e-01,
    4.68875471e-01] +
  # long bar 1
  [ 4.95381575e+08, 5.36363324e+00, 9.58522229e-01, 6.10542494e-01,
    9.69645220e-01, 3.05125124e+00, 3.19043585e+00, 5.58255674e-01,
    1.67310332e+01, 3.19575493e+00] +
  # long bar 2
  [ 1.74304936e+13, 4.77961423e-01, 2.66853061e-01, 2.51516920e-01,
    1.87882599e+00, 9.80136710e-01, 2.20415408e+00, 7.60708626e+00,
   -2.72907665e+01, 1.62966434e+00]
)

agama.setUnits(length=1, mass=1, velocity=1)  # 1 kpc, 1 Msun, 1 km/s
den = makeDensityModel(params)
pot = makePotentialModel(params)
pot.export('Portail17.ini')
print('Created MW potential: total mass in stars=%.3g Msun, halo=%.3g Msun' %
    (pot[0].totalMass(), pot[1].totalMass()))
# create an axisymmetrized version of the potential for plotting the true circular-velocity curve
pot_axi = agama.Potential(
    agama.Potential(type='CylSpline', potential=pot[0],
        mmax=0, gridsizeR=25, gridsizez=25, Rmin=0.1, Rmax=40, zmin=0.05, zmax=20),
    pot[1])

r=numpy.linspace(0,10,101)
xyz=numpy.column_stack((r,r*0,r*0))
ax=plt.subplots(1, 2, figsize=(16,8))[1]
ax[0].plot(r, (-r*pot_axi[0].force(xyz)[:,0])**0.5, 'c', label='stars')
ax[0].plot(r, (-r*pot_axi[1].force(xyz)[:,0])**0.5, 'y', label='halo')
ax[0].plot(r, (-r*pot_axi   .force(xyz)[:,0])**0.5, 'r', label='total')
ax[0].legend(loc='lower right', frameon=False)
ax[0].set_xlabel('radius [kpc]')
ax[0].set_ylabel('circular velocity [km/s]')

# integrate and show a few orbits
numorbits=10
numpy.random.seed(42)
ic=numpy.random.normal(size=(numorbits,6)) * numpy.array([2.0, 0.0, 0.4, 50., 40., 30.])
ic[:,0] += -6.
ic[:,4] += 220
orbits = agama.orbit(potential=pot, ic=ic, time=10., trajsize=1000, Omega=-39.0)[:,1]
bar_angle = -25.0 * numpy.pi/180  # orientation of the bar w.r.t. the Sun
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
print('Computing surface density')
gridr  = numpy.linspace(-rmax, rmax, 101)  # 1d grid
gridxy = numpy.column_stack((numpy.repeat(gridr, len(gridr)), numpy.tile(gridr, len(gridr))))  # 2d grid
Sigma  = pot[0].projectedDensity(gridxy, gamma=-bar_angle)  # surface density for a stellar component rotated by bar_angle
logSigma = 2.5 * numpy.log10(Sigma / numpy.max(Sigma))      # log-scaled to magnitudes per unit square
# thick lines spaced by one magnitude, thin lines lie halfway between thick ones
ax[1].contour(gridr, gridr, logSigma.reshape(len(gridr), len(gridr)).T,
    levels=numpy.linspace(-8,0,17), colors='k', zorder=5, linewidths=[2,1], linestyles='solid')

plt.tight_layout()
plt.show()
