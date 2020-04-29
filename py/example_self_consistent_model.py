#!/usr/bin/python
"""
This example demonstrates the machinery for constructing multicomponent self-consistent models
specified by distribution functions (DFs) in terms of actions.
We create a four-component galaxy with disk, bulge and halo components defined by their DFs,
and a static density profile of gas disk.
Then we perform several iterations of recomputing the density profiles of components from their DFs
and recomputing the total potential.
Finally, we create N-body representations of all mass components: dark matter halo,
stars (bulge, thin and thick disks and stellar halo combined), and gas disk.
A modification of this script that creates a self-consistent three-component model
(disk, bulge and halo) is given in example_self_consistent_model3.py
This example is the Python counterpart of tests/example_self_consistent_model.cpp
"""
import agama, numpy, sys, os
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

# write out the rotation curve (separately for each component, and the total one)
def writeRotationCurve(filename, potentials):
    radii = numpy.logspace(-2., 2., 81)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcomp = numpy.column_stack([(-potential.force(xyz)[:,0] * radii)**0.5 for potential in potentials])
    vtot  = numpy.sum(vcomp**2, axis=1)**0.5
    numpy.savetxt(filename, numpy.column_stack((radii, vtot, vcomp)), fmt="%.6g", delimiter="\t", \
        header="radius[Kpc]\tv_circ,total[km/s]\tdisk\tbulge\thalo")

# print surface density profiles to a file
def writeSurfaceDensityProfile(filename, model):
    print("Writing surface density profile")
    radii = numpy.hstack(([1./8, 1./4], numpy.linspace(0.5, 16, 32), numpy.linspace(18, 30, 7)))
    Sigma = model.projectedMoments(radii, separate=True)[0] * 1e-6  # convert from Msun/Kpc^2 to Msun/pc^2
    numpy.savetxt(filename, numpy.column_stack((radii, Sigma)), fmt="%.6g", delimiter="\t", \
        header="Radius[Kpc]\tsurfaceDensity[Msun/pc^2]")

# print vertical density profile for several sub-components of the stellar DF
def writeVerticalDensityProfile(filename, model):
    print("Writing vertical density profile")
    height = numpy.hstack((numpy.linspace(0, 1.5, 13), numpy.linspace(2, 8, 13)))
    xyz   = numpy.column_stack((height*0 + solarRadius, height*0, height))
    dens  = model.moments(xyz, dens=True, vel=False, vel2=False, separate=True) * 1e-9  # convert from Msun/Kpc^3 to Msun/pc^3
    numpy.savetxt(filename, numpy.column_stack((height, dens)), fmt="%.6g", delimiter="\t", \
        header="z[Kpc]\tThinDisk\tThickDisk\tStellarHalo[Msun/pc^3]")

# print velocity distributions at the given point to a file
def writeVelocityDistributions(filename, model):
    point = (solarRadius, 0, 0.1)
    print("Writing velocity distributions at (R=%g, z=%g)" % (point[0], point[2]))
    # create grids in velocity space for computing the spline representation of VDF
    v_max = 360.0    # km/s
    gridv = numpy.linspace(-v_max, v_max, 75) # use the same grid for all dimensions
    # compute the distributions (represented as cubic splines)
    splvR, splvz, splvphi = model.vdf(point, gridv)
    # output f(v) at a different grid of velocity values
    gridv = numpy.linspace(-v_max, v_max, 201)
    numpy.savetxt(filename, numpy.column_stack((gridv, splvR(gridv), splvz(gridv), splvphi(gridv))),
        fmt="%.6g", delimiter="\t", header="V\tf(V_R)\tf(V_z)\tf(V_phi) [1/(km/s)]")

# display some information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densHalo = model.components[2].getDensity()
    pt0 = (solarRadius, 0, 0)
    pt1 = (solarRadius, 0, 1)
    print("Disk total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densDisk.totalMass(), densDisk.density(pt0)*1e-9, densDisk.density(pt1)*1e-9))  # per pc^3, not kpc^3
    print("Halo total mass=%g Msun, rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densHalo.totalMass(), densHalo.density(pt0)*1e-9, densHalo.density(pt1)*1e-9))
    print("Potential at origin=-(%g km/s)^2, total mass=%g Msun" % \
        ((-model.potential.potential(0,0,0))**0.5, model.potential.totalMass()))
    densDisk.export ("dens_disk_" +iteration);
    densBulge.export("dens_bulge_"+iteration);
    densHalo.export ("dens_halo_" +iteration);
    model.potential.export("potential_"+iteration);
    writeRotationCurve("rotcurve_"+iteration, (model.potential[1],  # disk potential (CylSpline)
        agama.Potential(type='Multipole', lmax=6, density=densBulge),        # -"- bulge
        agama.Potential(type='Multipole', lmax=6, density=densHalo) ) )      # -"- halo


if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = os.path.dirname(os.path.realpath(sys.argv[0])) + "/../data/SCM.ini"
    ini = RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenThinDisk = dict(ini.items("Potential thin disk"))
    iniPotenThickDisk= dict(ini.items("Potential thick disk"))
    iniPotenGasDisk  = dict(ini.items("Potential gas disk"))
    iniPotenBulge    = dict(ini.items("Potential bulge"))
    iniPotenDarkHalo = dict(ini.items("Potential dark halo"))
    iniDFThinDisk    = dict(ini.items("DF thin disk"))
    iniDFThickDisk   = dict(ini.items("DF thick disk"))
    iniDFStellarHalo = dict(ini.items("DF stellar halo"))
    iniDFDarkHalo    = dict(ini.items("DF dark halo"))
    iniDFBulge       = dict(ini.items("DF bulge"))
    iniSCMHalo       = dict(ini.items("SelfConsistentModel halo"))
    iniSCMBulge      = dict(ini.items("SelfConsistentModel bulge"))
    iniSCMDisk       = dict(ini.items("SelfConsistentModel disk"))
    iniSCM           = dict(ini.items("SelfConsistentModel"))
    solarRadius      = ini.getfloat("Data", "SolarRadius")

    # define external unit system describing the data (including the parameters in INI file)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityBulge       = agama.Density(**iniPotenBulge)
    densityDarkHalo    = agama.Density(**iniPotenDarkHalo)
    densityThinDisk    = agama.Density(**iniPotenThinDisk)
    densityThickDisk   = agama.Density(**iniPotenThickDisk)
    densityGasDisk     = agama.Density(**iniPotenGasDisk)
    densityStellarDisk = agama.Density(densityThinDisk, densityThickDisk)  # composite

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityStellarDisk, disklike=True))
    model.components.append(agama.Component(density=densityBulge,       disklike=False))
    model.components.append(agama.Component(density=densityDarkHalo,    disklike=False))
    model.components.append(agama.Component(density=densityGasDisk,     disklike=True))

    # compute the initial potential
    model.iterate()
    printoutInfo(model, "init")

    print("\033[1;33m**** STARTING MODELLING ****\033[0m\nInitial masses of density components: " \
        "Mdisk=%g Msun, Mbulge=%g Msun, Mhalo=%g Msun, Mgas=%g Msun" % \
        (densityStellarDisk.totalMass(), densityBulge.totalMass(), \
        densityDarkHalo.totalMass(), densityGasDisk.totalMass()))

    # create the dark halo DF
    dfHalo  = agama.DistributionFunction(potential=model.potential, **iniDFDarkHalo)
    # same for the bulge
    dfBulge = agama.DistributionFunction(potential=model.potential, **iniDFBulge)
    # same for the stellar components (thin/thick disks and stellar halo)
    dfThinDisk    = agama.DistributionFunction(potential=model.potential, **iniDFThinDisk)
    dfThickDisk   = agama.DistributionFunction(potential=model.potential, **iniDFThickDisk)
    dfStellarHalo = agama.DistributionFunction(potential=model.potential, **iniDFStellarHalo)
    # composite DF of all stellar components except the bulge
    dfStellar     = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo)
    # composite DF of all stellar components including the bulge
    dfStellarAll  = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo, dfBulge)

    # replace the disk, halo and bulge SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfStellar, disklike=True, **iniSCMDisk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
    model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

    # we can compute the masses even though we don't know the density profile yet
    print("Masses of DF components: " \
        "Mdisk=%g Msun (Mthin=%g, Mthick=%g, Mstel.halo=%g); Mbulge=%g Msun; Mdarkhalo=%g Msun" % \
        (dfStellar.totalMass(), dfThinDisk.totalMass(), dfThickDisk.totalMass(), \
        dfStellarHalo.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

    # do a few more iterations to obtain the self-consistent density profile for the entire system
    for iteration in range(1,6):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, "iter"+str(iteration))

    # output various profiles (only for stellar components)
    print("\033[1;33mComputing density profiles and velocity distribution\033[0m")
    modelStars = agama.GalaxyModel(model.potential, dfStellar, model.af)
    writeSurfaceDensityProfile ("model_stars_final.surfdens", modelStars)
    writeVerticalDensityProfile("model_stars_final.vertical", modelStars)
    writeVelocityDistributions ("model_stars_final.veldist",  modelStars)

    # export model to an N-body snapshot
    print("\033[1;33mCreating an N-body representation of the model\033[0m")
    format = 'text'  # could use 'text', 'nemo' or 'gadget' here

    # first create a representation of density profiles without velocities
    # (just for demonstration), by drawing samples from the density distribution
    print("Writing N-body sampled density profile for the dark matter halo")
    agama.writeSnapshot("dens_dm_final", model.components[2].getDensity().sample(800000), format)
    print("Writing N-body sampled density profile for the stellar bulge, disk and halo")
    # recall that component[0] contains stellar disks and stellar halo, and component[1] - bulge
    densStars = agama.Density(model.components[0].getDensity(), model.components[1].getDensity())
    agama.writeSnapshot("dens_stars_final", densStars.sample(200000), format)

    # now create genuinely self-consistent models of all components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print("Writing a complete DF-based N-body model for the dark matter halo")
    agama.writeSnapshot("model_dm_final", \
        agama.GalaxyModel(potential=model.potential, df=dfHalo, af=model.af).sample(800000), format)
    print("Writing a complete DF-based N-body model for the stellar bulge, disk and halo")
    agama.writeSnapshot("model_stars_final", \
        agama.GalaxyModel(potential=model.potential, df=dfStellarAll, af=model.af).sample(200000), format)
    # we didn't use an action-based DF for the gas disk, leaving it as a static component;
    # to create an N-body representation, we sample the density profile and assign velocities
    # from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    print("Writing an N-body model for the gas disk")
    agama.writeSnapshot("model_gas_final", \
        model.components[3].getDensity().sample(24000, potential=model.potential, beta=0, kappa=1), format)
