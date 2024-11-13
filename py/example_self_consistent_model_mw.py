#!/usr/bin/python
"""
This example demonstrates the machinery for constructing multicomponent self-consistent models
specified by distribution functions (DFs) in terms of actions.
We create a Milky Way model with four disk, bulge, stellar and dark halo components
defined by their DFs, and a static density profile of gas disk.
The thin disk is split into 3 age groups, and there is a separate thick disk.
Then we perform several iterations of recomputing the density profiles of components from
their DFs and recomputing the total potential.
Finally, we create N-body representations of all mass components:
dark matter halo, stars (bulge, several disks and stellar halo combined), and gas disk,
and compute various diagnostic quantities written into text files.
The DFs for the disky and spheroidal components used here differ from the built-in DF types, and
are defined in the first part of the file; their parameters are contained in a separate INI file.
The DF parameters are optimized to fit Gaia DR2 data, as described in Binney&Vasiliev 2023.
These new DFs are implemented as Python functions, and for this reason the script is less
computationally efficient than the pure C++ equivalent program example_self_consistent_model_mw.cpp;
there are several reasons for this:
(a) Overheads from transferring the control between the C++ computational core and the Python
callback functions. This is partly mitigated by the fact that these functions are called
in a vectorized way, with several input points at once (~few dozen for the main part of the script,
where the integrals of DFs over velocity space are carried out, and much more for the last part,
where the DFs are sampled into an N-body snapshot).
(b) The mathematical computations are slower in Python, although again this is mostly mitigated
by performing them in a vectorized way on NumPy arrays with many elements at once.
(c) The computation of density from DF is OpenMP-parallelized in the C++ core, but because of GIL,
the user-defined functions can be entered only from one thread at a time. Most of the computational
time is spent on calculating the actions, which happens inside the C++ core and is fully parallelized,
and the evaluation of DF is usually sub-dominant, but may become a bottleneck when too many threads
are used; for this reason, we limit their number to 4. A special free-threading version of Python
(starting from 3.13) does not have GIL and benefits from OpenMP parallelization in principle, 
but not necessarily in practice.
In addition, the results (potential, density, velocity profiles) of the C++ and Python programs
slightly differ due to unavoidable differences in floating-point operations implemented in two
languages, which are greatly amplified in the course of iterations (but stay at the level 10^-3).
"""
import agama, numpy, sys, os, time, warnings
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

# user-defined modifications of spheroidal (DoublePowerLaw) and disky (Exponential) DFs
def createNewDoublePowerLawDF(**params):
    def df(J):
        modJphi = abs(J[:,2])
        L = J[:,1] + modJphi
        c = L / (L + J[:,0])
        jt = (1.5 * J[:,0] + L) / L0
        jta = jt**alpha
        xi = jta / (1+jta)
        rat = (1-xi) * Fin + xi * Fout
        a = 0.5 * (rat+1)
        b = 0.5 * (rat-1)
        cL = numpy.where(L>0, J[:,1] * (a + b * modJphi / L) + modJphi, 0)
        fac = numpy.exp(beta * numpy.sin(numpy.pi/2 * c))
        hJ = J[:,0] / fac + 0.5 * (1 + c * xi) * fac * cL
        gJ = hJ
        result = norm / (2*numpy.pi * J0)**3 * (1 + J0/hJ)**slopeIn * (1 + gJ/J0)**(-slopeOut)
        if Jcutoff > 0:
            result *= numpy.exp(-(gJ / Jcutoff)**cutoffStrength)
        if Jcore > 0:
            result *= (1 + Jcore/hJ * (Jcore/hJ - zeta))**(-0.5*slopeIn)
        if rotFrac != 0:
            result *= 1 + rotFrac * numpy.tanh(J[:,2] / Jphi0)
        return result
    J0, L0, slopeIn, slopeOut, rotFrac, Jphi0, alpha, beta, Fin, Fout, Jcutoff, cutoffStrength, Jcore = (
        float(params[name]) for name in 
        ('J0', 'L0', 'slopeIn', 'slopeOut', 'rotFrac', 'Jphi0', 'alpha', 'beta',
        'Fin', 'Fout', 'Jcutoff', 'cutoffStrength', 'Jcore'))
    if Jcore > 0:
        import scipy.optimize, scipy.integrate
        def rootFnc(zeta):
            def integrand(t):
                hJ = Jcore * t*t*(3-2*t) / (1-t)**2 / (1+2*t)
                dhJdt = Jcore * 6*t / (1-t)**3 / (1+2*t)**2
                return (hJ**2 * dhJdt * (1 + J0/hJ)**slopeIn * (1+hJ/J0)**(-slopeOut) *
                    ((1 + Jcore/hJ * (Jcore/hJ-zeta))**(-0.5*slopeIn) - 1))
            return scipy.integrate.fixed_quad(integrand, 0.0, 1.0, n=20)[0]
        zeta = scipy.optimize.brentq(rootFnc, 0.0, 2.0)
    norm = 1.0
    norm = float(params['mass']) / agama.DistributionFunction(df).totalMass()
    print("Created a %s DF with mass %s" % (params['type'], params['mass']))
    return df

def createNewExponentialDF(**params):
    def df(J):
        Jp = numpy.maximum(0, J[:,2])
        Jvel = Jp + addJvel
        Jden = Jp + addJden
        xr = (Jvel / Jphi0)**pr / Jr0
        xz = (Jvel / Jphi0)**pz / Jz0
        fr = xr * numpy.exp(-xr * J[:,0])
        fz = xz * numpy.exp(-xz * J[:,1])
        fp = norm / Jphi0**2 * abs(J[:,2]) * numpy.exp(-Jden / Jphi0)
        return numpy.where(J[:,2] > 0, fr * fz * fp, 0)
    Jr0, Jz0, Jphi0, pr, pz, addJden, addJvel = (float(params[name]) for name in
    ('Jr0', 'Jz0', 'Jphi0', 'pr', 'pz', 'addJden', 'addJvel'))
    norm = 1.0
    norm = float(params['mass']) / agama.DistributionFunction(df).totalMass()
    print("Created a %s DF with mass %s" % (params['type'], params['mass']))
    return df

# common header for output files, listing the DF components in order of adding them to the total DF
header= "bulge\tthin,young\tthin,middle\tthin,old\tthick\tstellarhalo"

# print velocity dispersion, in-plane and surface density profiles of each stellar component to a file
def writeRadialProfile(model):
    print("Writing radial density and velocity profiles")
    radii = numpy.hstack(([1./8, 1./4], numpy.linspace(0.5, 16, 32), numpy.linspace(18, 30, 7)))
    xy    = numpy.column_stack((radii, radii*0))
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    # in-plane density and velocity moments, separately for each DF component
    z0dens, meanv, meanv2 = model.moments(xyz, dens=True, vel=True, vel2=True, separate=True)
    # projected density for each DF component
    Sigma = model.moments(xy, dens=True, vel=False, vel2=False, separate=True)
    numpy.savetxt("mwmodel_surface_density.txt", numpy.column_stack((radii, Sigma * 1e-6)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[Msun/pc^2]")
    numpy.savetxt("mwmodel_volume_density.txt", numpy.column_stack((radii, z0dens * 1e-9)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[Msun/pc^3]")
    numpy.savetxt("mwmodel_sigmaR.txt", numpy.column_stack((radii, meanv2[:,:,0]**0.5)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[km/s]")
    numpy.savetxt("mwmodel_sigmaz.txt", numpy.column_stack((radii, meanv2[:,:,2]**0.5)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[km/s]")
    numpy.savetxt("mwmodel_sigmaphi.txt", numpy.column_stack((radii, (meanv2[:,:,1]-meanv[:,:,1]**2)**0.5)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[km/s]")
    numpy.savetxt("mwmodel_meanvphi.txt", numpy.column_stack((radii, meanv[:,:,1])),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[km/s]")

# print vertical density profile for several sub-components of the stellar DF
def writeVerticalDensityProfile(model):
    print("Writing vertical density profile")
    height = numpy.hstack((numpy.linspace(0, 0.5, 11), numpy.linspace(0.75, 5.0, 18)))
    xyz   = numpy.column_stack((height*0 + solarRadius, height*0, height))
    dens  = model.moments(xyz, dens=True, vel=False, vel2=False, separate=True) * 1e-9
    numpy.savetxt("mwmodel_vertical_density.txt", numpy.column_stack((height, dens)),
        fmt="%.6g", delimiter="\t", header="radius\t"+header+"[Msun/pc^3]")

# print velocity distributions at the given point to a file
def writeVelocityDistributions(model):
    points = [[solarRadius-2, 0, 0], [solarRadius, 0, 0], [solarRadius+2, 0, 0], [solarRadius, 0, 2]]
    print("Writing velocity distributions")
    # create grids in velocity space for computing the spline representation of VDF
    v_max = 400.0    # km/s
    gridv = agama.symmetricGrid(75, 6.0, v_max)
    # compute the distributions (represented as cubic splines)
    result = model.vdf(points, gridv=gridv, separate=True, dens=True)  # three sets of VDFs and density
    rho = result[-1]
    # output f(v) at a different grid of velocity values
    gridv = numpy.linspace(-v_max, v_max, int(v_max+1))
    for ip,point in enumerate(points):
        for iv in range(3):
            numpy.savetxt("mwmodel_vdf_R%g_z%g_v%s.txt" % (point[0], point[2], ["R","phi","z"][iv]),
                numpy.column_stack([gridv] +
                [rho[ip][ic] * result[iv][ip][ic](gridv) * 1e-9 for ic in range(len(result[iv][ip]))]),
                fmt="%.6g", delimiter="\t", header="v\t"+header+"[Msun/pc^3/(km/s)]")

# display some information after each iteration
def printoutInfo(model):
    compStars = model.components[0].density
    compDark  = model.components[1].density
    pt0 = (solarRadius, 0, 0)
    pt1 = (solarRadius, 0, 1)
    print("Disk total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (compStars.totalMass(), compStars.density(pt0)*1e-9, compStars.density(pt1)*1e-9))
    print("Halo total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (compDark .totalMass(), compDark .density(pt0)*1e-9, compDark .density(pt1)*1e-9))
    print("Potential at origin=-(%g km/s)^2, total mass=%g Msun" % \
        ((-model.potential.potential(0,0,0))**0.5, model.potential.totalMass()))
    compStars.export("mwmodel_density_stars.ini")
    compDark .export("mwmodel_density_dark.ini")
    model.potential.export("mwmodel_potential.ini")
    # write out the rotation curve (separately for each component, and the total one)
    radii = numpy.logspace(-2., 2., 81)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcomp = numpy.column_stack([(-pot.force(xyz)[:,0] * radii)**0.5 for pot in model.potential])
    vtot  = numpy.sum(vcomp**2, axis=1)**0.5
    numpy.savetxt("mwmodel_rotcurve.txt",
        numpy.column_stack((radii, vtot, vcomp)), fmt="%.6g", delimiter="\t",
        header="radius[Kpc]\tv_circ,total[km/s]\tdarkmatter\tstars+gas")


if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = os.path.dirname(os.path.realpath(sys.argv[0])) + "/../data/SCM_MW.ini"
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenGasDisk  = dict(ini.items("Potential gas disk"))
    iniDFyoungDisk   = dict(ini.items("DF young disk"))
    iniDFmiddleDisk  = dict(ini.items("DF middle disk"))
    iniDFoldDisk     = dict(ini.items("DF old disk"))
    iniDFhighADisk   = dict(ini.items("DF highA disk"))
    iniDFStellarHalo = dict(ini.items("DF stellar halo"))
    iniDFDarkHalo    = dict(ini.items("DF dark halo"))
    iniDFBulge       = dict(ini.items("DF bulge"))
    iniSCMDisk       = dict(ini.items("SelfConsistentModel disk"))
    iniSCMHalo       = dict(ini.items("SelfConsistentModel halo"))
    iniSCM           = dict(ini.items("SelfConsistentModel"))
    solarRadius      = ini.getfloat("Data", "SolarRadius")

    # define external unit system describing the data (including the parameters in INI file)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create the initial potential from all sections of the INI file starting with "[Potential..."
    model.potential = agama.Potential(iniFileName)

    print("\033[1;37mInitializing\033[0m")
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress division by zero warning
    time0 = time.time()
    # the initialization is costly because of computing the DF normalization (integral over all J);
    # the equivalent operation with C++-native DF types is almost instantaneous
    dfDarkHalo    = createNewDoublePowerLawDF(**iniDFDarkHalo)
    dfBulge       = createNewDoublePowerLawDF(**iniDFBulge)
    dfYoungDisk   = createNewExponentialDF(**iniDFyoungDisk)
    dfMiddleDisk  = createNewExponentialDF(**iniDFmiddleDisk)
    dfOldDisk     = createNewExponentialDF(**iniDFoldDisk)
    dfHighADisk   = createNewExponentialDF(**iniDFhighADisk)
    dfStellarHalo = createNewDoublePowerLawDF(**iniDFStellarHalo)
    # composite DF of all stellar components
    dfStellar     = agama.DistributionFunction(
        dfBulge, dfYoungDisk, dfMiddleDisk, dfOldDisk, dfHighADisk, dfStellarHalo)
    time1 = time.time()
    print("%g seconds to initialize DFs" % (time1-time0))

    # replace the disk, halo and bulge SCM components with the DF-based ones
    model.components = [
        agama.Component(df=dfStellar,  disklike=True,  **iniSCMDisk),
        agama.Component(df=dfDarkHalo, disklike=False, **iniSCMHalo),
        agama.Component(density=agama.Density(**iniPotenGasDisk), disklike=True)
    ]

    # Limit the number of OpenMP threads to at most 4, because the user-defined DF functions prevent
    # effective parallelization due to Python GIL (although the action computation is still parallelized,
    # so the overall procedure benefits from having a few, but not too many, threads).
    # This is not necessary for free-threading Python 3.13+, which has no GIL.
    numThreads = (0  if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()
        else min(agama.setNumThreads(0).prevNumThreads, 4))
    with agama.setNumThreads(numThreads):
        # do a few iterations to obtain the self-consistent density profile for the entire system
        for iteration in range(1,5):
            print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
            model.iterate()
            printoutInfo(model)
    time2 = time.time()
    print("%g seconds to build the model" % (time2-time1))

    # output various profiles (only for stellar components)
    print("\033[1;37mComputing diagnostics\033[0m")
    modelStars = agama.GalaxyModel(model.potential, dfStellar, model.af)
    # again limit the number of OpenMP threads to avoid congestion
    with agama.setNumThreads(numThreads):
        writeRadialProfile(modelStars)
        writeVerticalDensityProfile(modelStars)
        writeVelocityDistributions(modelStars)
    time3 = time.time()
    print("%g seconds to compute diagnostics" % (time3-time2))

    # export model to an N-body snapshot;
    # here we do not need to limit the number of threads, since the sampling procedure
    # is much more vectorized than integration (which is at the heart of all previous steps),
    # and the overhead of calling the user-defined Python function is relatively lower.
    print("\033[1;37mCreating an N-body representation of the model\033[0m")
    format = "nemo"  # could use 'text', 'nemo' or 'gadget' here
    agama.writeSnapshot("mwmodel_dm_final.nbody",
        agama.GalaxyModel(potential=model.potential, df=dfDarkHalo, af=model.af).sample(750000),
        format)
    agama.writeSnapshot("model_stars_final.nbody",
        modelStars.sample(200000),
        format)
    # we didn't use an action-based DF for the gas disk, leaving it as a static component;
    # to create an N-body representation, we sample the density profile and assign velocities
    # from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    agama.writeSnapshot("model_gas_final.nbody",
        model.components[2].density.sample(50000, potential=model.potential, beta=0, kappa=1),
        format)
    print("%g seconds to create an N-body snapshot" % (time.time()-time3))
