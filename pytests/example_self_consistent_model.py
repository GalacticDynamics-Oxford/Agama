#!/usr/bin/python
"""
This example demonstrates the machinery for constructing multicomponent self-consistent models
specified by distribution functions in terms of actions.
We create a two-component galaxy with disk, bulge and halo components, using a two-stage approach:
first, we take a static potential/density profile for the disk, and find a self-consistent
density profile of the bulge and the halo components in the presence of the disk potential;
second, we replace the static disk with a DF-based component and find the overall self-consistent
model for all components. The rationale is that a reasonable guess for the total potential
is already needed before constructing the DF for the disk component, since the latter relies
upon plausible radially-varying epicyclic frequencies.
Both stages require a few iterations to converge.
Finally, we create N-body representations of all mass components: dark matter halo,
stars (bulge, thin and thick disks and stellar halo combined), and gas disk.
A modification of this script that creates a self-consistent three-component model
(bulge, disk and halo) is given in example_self_consistent_model3.py
This example is the Python counterpart of tests/example_self_consistent_model.cpp
"""
import agama, numpy, ConfigParser, sys, os

# write out the rotation curve (separately for the disky and spheroidal components:
# the first one contains stellar disk, halo and the gas disk, the second - stellar bulge and dark halo
def writeRotationCurve(filename, potential):
    potential.export(filename)
    numComp = len(potential)    # number of components in the composite potential (2)
    radii = numpy.logspace(-1.5, 2, 71)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcirc = numpy.zeros((len(radii), numComp+2))
    vcirc[:,0] = radii
    # first 2 columns are circular velocities of individual components, the last one is the total
    for c in range(numComp+1):
        pot = potential[c] if c<numComp else potential
        vcirc[:,c+1] = (-pot.force(xyz)[:,0] * radii)**0.5
    numpy.savetxt(filename, vcirc, fmt="%.6g", delimiter="\t", \
        header="radius[Kpc]\t" + "\t".join([str(pot) for pot in potential]) + "\tv_circ,total[km/s]")

# print surface density profiles to a file
def writeSurfaceDensityProfile(filename, model, df):
    print "Writing surface density profile"
    radii = numpy.hstack(([1./8, 1./4], numpy.linspace(0.5, 16, 32), numpy.linspace(18, 30, 7)))
    Sigma = agama.GalaxyModel(potential=model.potential, df=df, af=model.af). \
        projectedMoments(radii)[0] * 1e-6  # convert from Msun/Kpc^2 to Msun/pc^2
    numpy.savetxt(filename, numpy.column_stack((radii, Sigma)), fmt="%.6g", delimiter="\t", \
        header="Radius[Kpc]\tsurfaceDensity[Msun/pc^2]")

# print vertical density profile for several sub-components of the stellar DF
def writeVerticalDensityProfile(filename, model, df):
    print "Writing vertical density profile"
    numComp = len(df)   # number of DF components
    height = numpy.hstack((numpy.linspace(0, 1.5, 13), numpy.linspace(2, 8, 13)))
    xyz   = numpy.column_stack((height*0 + solarRadius, height*0, height))
    dens  = numpy.zeros((len(height), numComp+1))
    dens[:,0] = height
    for c in range(numComp):
        dens[:,c+1] = agama.GalaxyModel(potential=model.potential, df=df[c], af=model.af). \
            moments(xyz, dens=True, vel=False, vel2=False) * 1e-9  # convert from Msun/Kpc^3 to Msun/pc^3
    numpy.savetxt(filename, dens, fmt="%.6g", delimiter="\t", \
        header="z[Kpc]\tThinDisk\tThickDisk\tStellarHalo[Msun/pc^3]")

# print velocity distributions at the given point to a file
def writeVelocityDistributions(filename, model, df):
    point = (solarRadius, 0, 0.1)
    print "Writing velocity distributions at (R=%g, z=%g)" % (point[0], point[2])
    # choose the range of velocity to build the distribution
    v_escape = (-2 * model.potential.potential(point))**0.5;
    v_circ   = (-model.potential.force(point)[0] * point[0])**0.5
    v_max    = min(0.8*v_escape, 2*v_circ)
    # create grids in velocity space for computing the spline representation of VDF
    gridvR   = numpy.linspace(-v_max, v_max, 75)
    gridvz   = gridvR   # for simplicity, use the same grid for all dimensions
    gridvphi = gridvR
    # compute the distributions (represented as cubic splines)
    splvR, splvz, splvphi = agama.GalaxyModel(potential=model.potential, df=df, af=model.af). \
        vdf(point, gridvR, gridvz, gridvphi)  # the last three arguments are optional
    # output f(v) at a different grid of velocity values
    gridv = numpy.linspace(-v_max, v_max, 201)
    numpy.savetxt(filename, numpy.column_stack((gridv, splvR(gridv), splvz(gridv), splvphi(gridv))),
        fmt="%.6g", delimiter="\t", header="V\tf(V_R)\tf(V_z)\tf(V_phi) [1/(km/s)]")

# display some information after each iteration
def printoutInfo(model, iteration):
    densHalo  = model.components[0].getDensity()
    densBulge = model.components[1].getDensity()
    densDisk  = model.components[2].getDensity()
    pt0 = (solarRadius, 0, 0)
    pt1 = (solarRadius, 0, 1)
    print \
        "Disk total mass=%g Msun," % densDisk.totalMass(), \
        "rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densDisk.density(pt0)*1e-9, densDisk.density(pt1)*1e-9)  # per pc^3, not kpc^3
    print \
        "Bulge total mass=%g Msun," % densBulge.totalMass(), \
        "rho(1 kpc)=%g Msun/pc^3" % (densBulge.density(1, 0, 0)*1e-9)
    print \
        "Halo total mass=%g Msun," % densHalo.totalMass(), \
        "rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densHalo.density(pt0)*1e-9, densHalo.density(pt1)*1e-9)
    print "Potential at origin=-(%g km/s)^2," % (-model.potential.potential(0,0,0))**0.5, \
        "total mass=%g Msun" % model.potential.totalMass()
    densDisk.export("dens_disk_iter"+str(iteration));
    densHalo.export("dens_halo_iter"+str(iteration));
    writeRotationCurve("rotcurve_iter"+str(iteration), model.potential)

if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = os.path.dirname(os.path.realpath(sys.argv[0])) + "/../data/SCM.ini"
    ini = ConfigParser.RawConfigParser()
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
    model.components.append(agama.Component(density=densityDarkHalo,    disklike=False))
    model.components.append(agama.Component(density=densityBulge,       disklike=False))
    model.components.append(agama.Component(density=densityStellarDisk, disklike=True))
    model.components.append(agama.Component(density=densityGasDisk,     disklike=True))

    # compute the initial potential
    model.iterate()
    writeRotationCurve("rotcurve_init", model.potential)

    print "\033[1;33m**** STARTING ONE-COMPONENT MODELLING ****\033[0m\nMasses are: " \
        "Mbulge=%g Msun,"% densityBulge.totalMass(), \
        "Mgas=%g Msun,"  % densityGasDisk.totalMass(), \
        "Mdisk=%g Msun," % densityStellarDisk.totalMass(), \
        "Mhalo=%g Msun"  % densityDarkHalo.totalMass()

    # create the dark halo DF
    dfHalo  = agama.DistributionFunction(potential=model.potential, **iniDFDarkHalo)
    # same for the bulge
    dfBulge = agama.DistributionFunction(potential=model.potential, **iniDFBulge)

    # replace the halo and bulge SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)

    # do a few iterations to determine the self-consistent density profile of the halo and the bulge
    for iteration in range(1,5):
        print "\033[1;37mStarting iteration #%d\033[0m" % iteration
        model.iterate()
        printoutInfo(model, iteration)

    # now that we have a reasonable guess for the total potential,
    # we may initialize the DF of the stellar components (thin/thick disks and stellar halo)
    dfThinDisk    = agama.DistributionFunction(potential=model.potential, **iniDFThinDisk)
    dfThickDisk   = agama.DistributionFunction(potential=model.potential, **iniDFThickDisk)
    dfStellarHalo = agama.DistributionFunction(potential=model.potential, **iniDFStellarHalo)
    # composite DF of all stellar components except the bulge
    dfStellar     = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo)
    # composite DF of all stellar components including the bulge
    dfStellarAll  = agama.DistributionFunction(dfThinDisk, dfThickDisk, dfStellarHalo, dfBulge)

    # we can compute the masses even though we don't know the density profile yet
    print "\033[1;33m**** STARTING TWO-COMPONENT MODELLING ****\033[0m\nMasses are:", \
        "Mdisk=%g Msun" % dfStellar.totalMass(), \
        "(Mthin=%g, Mthick=%g, Mstel.halo=%g);" % \
        (dfThinDisk.totalMass(), dfThickDisk.totalMass(), dfStellarHalo.totalMass()), \
        "Mbulge=%g Msun;" % dfBulge.totalMass(), \
        "Mdarkhalo=%g Msun" % dfHalo.totalMass()

    # replace the static disk + stellar halo component with a DF-based one
    model.components[2] = agama.Component(df=dfStellar, disklike=True, **iniSCMDisk)

    # do a few more iterations to obtain the self-consistent density profile for the entire system
    for iteration in range(5,11):
        print "\033[1;37mStarting iteration #%d\033[0m" % iteration
        model.iterate()
        printoutInfo(model, iteration)

    # output various profiles
    print "\033[1;33mComputing density profiles and velocity distribution\033[0m"
    writeSurfaceDensityProfile ("model_stars_final.surfdens", model, dfStellar)
    writeVerticalDensityProfile("model_stars_final.vertical", model, dfStellar)
    writeVelocityDistributions ("model_stars_final.veldist",  model, dfStellar)

    # export model to an N-body snapshot
    print "\033[1;33mCreating an N-body representation of the model\033[0m"
    format = 'text'  # could use 'text', 'nemo' or 'gadget' here

    # first create a representation of density profiles without velocities
    # (just for demonstration), by drawing samples from the density distribution
    print "Writing N-body sampled density profile for the dark matter halo"
    agama.writeSnapshot("dens_dm_final", model.components[0].getDensity().sample(800000), format)
    print "Writing N-body sampled density profile for the stellar bulge, disk and halo"
    # recall that component[1] contains stellar disks and stellar halo, and component[2] - bulge
    densStars = agama.Density(model.components[1].getDensity(), model.components[2].getDensity())
    agama.writeSnapshot("dens_stars_final", densStars.sample(200000), format)

    # now create genuinely self-consistent models of all components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print "Writing a complete DF-based N-body model for the dark matter halo"
    agama.writeSnapshot("model_dm_final", \
        agama.GalaxyModel(potential=model.potential, df=dfHalo, af=model.af).sample(800000), format)
    print "Writing a complete DF-based N-body model for the stellar bulge, disk and halo"
    agama.writeSnapshot("model_stars_final", \
        agama.GalaxyModel(potential=model.potential, df=dfStellarAll, af=model.af).sample(200000), format)
    # we didn't use an action-based DF for the gas disk, leaving it as a static component;
    # to create an N-body representation, we sample the density profile and assign velocities
    # from the axisymmetric Jeans equation with equal velocity dispersions in R,z,phi
    print "Writing an N-body model for the gas disk"
    agama.writeSnapshot("model_gas_final", \
        model.components[3].getDensity().sample(24000, potential=model.potential, beta=0, kappa=1), format)
