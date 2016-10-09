#!/usr/bin/python
"""
This example demonstrates the machinery for constructing multicomponent self-consistent models
specified by distribution functions in terms of actions.
We create a two-component galaxy with disk and halo components, using a two-stage approach:
first, we take a static potential/density profile for the disk, and find a self-consistent
density profile of the halo component in the presence of the disk potential;
second, we replace the static disk with a DF-based component and find the overall self-consistent
model for both components. The rationale is that a reasonable guess for the total potential
is already needed before constructing the DF for the disk component, since the latter relies
upon plausible radially-varying epicyclic frequencies.
Both stages require a few iterations to converge.
Finally, we create N-body representations of both components.
This example is the Python counterpart of tests/example_self_consistent_model.cpp
"""

import agama, numpy, ConfigParser

# py_unsio is optional
have_py_unsio = True
try: import py_unsio
except ImportError: have_py_unsio = False

def writeNbodySnapshot(filename, snapshot):
    # snapshot is a tuple of two arrays: Nx6 position/velocity, N masses
    posvel, mass = snapshot
    if have_py_unsio:
        out = py_unsio.CunsOut(filename+".nemo", "nemo")
        out.setArrayF("pos", posvel[:,:3].reshape(-1).astype(numpy.float32))
        if posvel.shape[1]>3:
            out.setArrayF("vel", posvel[:,3:].reshape(-1).astype(numpy.float32))
        out.setArrayF("mass", mass.astype(numpy.float32))
        out.save()
    else:
        numpy.savetxt(filename, numpy.hstack((posvel, mass.reshape(-1,1))), fmt="%.6g")

def writeRotationCurve(filename, potential):
    potential.export(filename)
    radii = numpy.logspace(-1.5, 2, 71)
    vcirc = (-potential.force( numpy.vstack((radii, radii*0, radii*0)).T)[:,0] * radii)**0.5
    numpy.savetxt(filename, numpy.vstack((radii, vcirc)).T, fmt="%.6g", header="radius\tv_circ")

def printoutInfo(model, iteration):
    densHalo = model.components[0].getDensity()
    densDisc = model.components[1].getDensity()
    pt0 = (8.3, 0, 0)
    pt1 = (8.3, 0, 1)
    print \
        "Disc total mass=%g Msun," % densDisc.totalMass(), \
        "rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densDisc.density(pt0)*1e-9, densDisc.density(pt1)*1e-9)  # per pc^3, not kpc^3
    print \
        "Halo total mass=%g Msun," % densHalo.totalMass(), \
        "rho(Rsolar,z=0)=%g, rho(Rsolar,z=1kpc)=%g Msun/pc^3" % \
        (densHalo.density(pt0)*1e-9, densHalo.density(pt1)*1e-9)
    print "Potential at origin=-(%g km/s)^2," % (-model.pot.potential(0,0,0))**0.5, \
        "total mass=%g Msun" % model.pot.totalMass()
    densDisc.export("dens_disc_iter"+str(iteration));
    densHalo.export("dens_halo_iter"+str(iteration));
    writeRotationCurve("rotcurve_iter"+str(iteration), model.pot)

if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = "../data/SCM.ini"
    ini = ConfigParser.RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenThinDisc = dict(ini.items("Potential thin disc"))
    iniPotenThickDisc= dict(ini.items("Potential thick disc"))
    iniPotenGasDisc  = dict(ini.items("Potential gas disc"))
    iniPotenBulge    = dict(ini.items("Potential bulge"))
    iniPotenDarkHalo = dict(ini.items("Potential dark halo"))
    iniDFThinDisc    = dict(ini.items("DF thin disc"))
    iniDFThickDisc   = dict(ini.items("DF thick disc"))
    iniDFStellarHalo = dict(ini.items("DF stellar halo"))
    iniDFDarkHalo    = dict(ini.items("DF dark halo"))
    iniSCMHalo       = dict(ini.items("SelfConsistentModel halo"))
    iniSCMDisc       = dict(ini.items("SelfConsistentModel disc"))
    iniSCM           = dict(ini.items("SelfConsistentModel"))

    # define external unit system describing the data (including the parameters in INI file)
    agama.setUnits(length=1, velocity=1, mass=1)   # in Kpc, km/s, Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityBulge       = agama.Density(**iniPotenBulge)
    densityDarkHalo    = agama.Density(**iniPotenDarkHalo)
    densityThinDisc    = agama.Density(**iniPotenThinDisc)
    densityThickDisc   = agama.Density(**iniPotenThickDisc)
    densityGasDisc     = agama.Density(**iniPotenGasDisc)
    densityStellarDisc = agama.Density(densityThinDisc, densityThickDisc)  # composite

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(dens=densityDarkHalo,    disklike=False))
    model.components.append(agama.Component(dens=densityStellarDisc, disklike=True))
    model.components.append(agama.Component(dens=densityBulge,       disklike=False))
    model.components.append(agama.Component(dens=densityGasDisc,     disklike=True))

    # compute the initial potential
    model.iterate()
    writeRotationCurve("rotcurve_init", model.pot)

    print "**** STARTING ONE-COMPONENT MODELLING ****\nMasses are:  " \
        "Mbulge=%g Msun,"% densityBulge.totalMass(), \
        "Mgas=%g Msun,"  % densityGasDisc.totalMass(), \
        "Mdisc=%g Msun," % densityStellarDisc.totalMass(), \
        "Mhalo=%g Msun"  % densityDarkHalo.totalMass()

    # create the dark halo DF from the parameters in INI file;
    # here the initial potential is only used to create epicyclic frequency interpolation table
    dfHalo = agama.DistributionFunction(pot=model.pot, **iniDFDarkHalo)

    # replace the halo SCM component with the DF-based one
    model.components[0] = agama.Component(df=dfHalo, disklike=False, **iniSCMHalo)

    # do a few iterations to determine the self-consistent density profile of the halo
    for iteration in range(1,6):
        print "Starting iteration #%d" % iteration
        model.iterate()
        printoutInfo(model, iteration)

    # now that we have a reasonable guess for the total potential,
    # we may initialize the DF of the stellar components
    dfThinDisc    = agama.DistributionFunction(pot=model.pot, **iniDFThinDisc)
    dfThickDisc   = agama.DistributionFunction(pot=model.pot, **iniDFThickDisc)
    dfStellarHalo = agama.DistributionFunction(pot=model.pot, **iniDFStellarHalo)
    # composite DF of all stellar components except the bulge
    dfStellar = agama.DistributionFunction(dfThinDisc, dfThickDisc, dfStellarHalo)

    # we can compute the masses even though we don't know the density profile yet
    print "**** STARTING TWO-COMPONENT MODELLING ****\nMasses are: ", \
        "Mdisc=%g Msun" % dfStellar.totalMass(), \
        "(Mthin=%g, Mthick=%g, Mstel.halo=%g); " % \
        (dfThinDisc.totalMass(), dfThickDisc.totalMass(), dfStellarHalo.totalMass()), \
        "Mdarkhalo=%g Msun" % dfHalo.totalMass()

    # and replace the static disc component them with a DF-based disc one
    model.components[1] = agama.Component(df=dfStellar, disklike=True, \
        gridR=agama.nonuniformGrid(int(iniSCMDisc['sizeRadialCyl']), \
            float(iniSCMDisc['RminCyl']), float(iniSCMDisc['RmaxCyl'])), \
        gridz=agama.nonuniformGrid(int(iniSCMDisc['sizeVerticalCyl']), \
            float(iniSCMDisc['zminCyl']), float(iniSCMDisc['zmaxCyl'])) )

    # do a few more iterations to obtain the self-consistent density profile for both discs
    for iteration in range(6,11):
        print "Starting iteration #%d" % iteration
        model.iterate()
        printoutInfo(model, iteration)

    # export model to an N-body snapshot
    print "Creating an N-body representation of the model"

    # first create a representation of density profiles without velocities
    # (just for demonstration), by drawing samples from the density distribution
    print "Sampling halo density"
    writeNbodySnapshot("dens_halo_iter10", model.components[0].getDensity().sample(100000))
    print "Sampling disc density"
    writeNbodySnapshot("dens_disc_iter10", model.components[1].getDensity().sample(100000))

    # now create genuinely self-consistent models of both components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print "Sampling halo DF"
    writeNbodySnapshot("model_halo_iter10", \
        agama.GalaxyModel(pot=model.pot, df=dfHalo, af=model.af).sample(100000))
    print "Sampling disc DF"
    writeNbodySnapshot("model_disc_iter10", \
        agama.GalaxyModel(pot=model.pot, df=dfStellar, af=model.af).sample(100000))
