#!/usr/bin/python

# Example of construction of a three-component disk-bulge-halo equilibrium model of a galaxy

import agama, numpy, ConfigParser

# py_unsio is optional
have_py_unsio = False
#try: import py_unsio
#except ImportError: have_py_unsio = False

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
        numpy.savetxt(filename+".txt", numpy.hstack((posvel, mass.reshape(-1,1))), fmt="%.6g")

def writeRotationCurve(filename, potential):
    potential.export(filename)
    radii = numpy.logspace(-2, 1.5, 71)
    vcirc = (-potential.force( numpy.vstack((radii, radii*0, radii*0)).T)[:,0] * radii)**0.5
    numpy.savetxt(filename, numpy.vstack((radii, vcirc)).T, fmt="%.6g", header="radius\tv_circ")

def printoutInfo(model, iteration):
    densHalo = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densDisk = model.components[2].getDensity()
    pt0 = (2.0, 0, 0)
    pt1 = (2.0, 0, 0.5)
    print \
        "Disk  total mass=%g," % densDisk.totalMass(), \
        "rho(R=2,z=0)=%g, rho(R=2,z=0.5)=%g" % \
        (densDisk.density(pt0), densDisk.density(pt1))
    print \
        "Bulge total mass=%g," % densBulge.totalMass(), \
        "rho(R=0.5,z=0)=%g" % \
        (densBulge.density(0.4, 0, 0))
    print \
        "Halo  total mass=%g," % densHalo.totalMass(), \
        "rho(R=2,z=0)=%g, rho(R=2,z=0.5)=%g" % \
        (densHalo.density(pt0), densHalo.density(pt1))
    print "Potential at origin=-(%g km/s)^2," % (-model.pot.potential(0,0,0))**0.5, \
        "total mass=%g" % model.pot.totalMass()
    densDisk. export("dens_disk_iter" +str(iteration));
    densBulge.export("dens_bulge_iter"+str(iteration));
    densHalo. export("dens_halo_iter" +str(iteration));
    writeRotationCurve("rotcurve_iter"+str(iteration), model.pot)

if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = "../data/SCM3.ini"
    ini = ConfigParser.RawConfigParser()
    ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenHalo  = dict(ini.items("Potential halo"))
    iniPotenBulge = dict(ini.items("Potential bulge"))
    iniPotenDisk  = dict(ini.items("Potential disk"))
    iniDFDisk     = dict(ini.items("DF disk"))
    iniSCMHalo    = dict(ini.items("SelfConsistentModel halo"))
    iniSCMBulge   = dict(ini.items("SelfConsistentModel bulge"))
    iniSCMDisk    = dict(ini.items("SelfConsistentModel disk"))
    iniSCM        = dict(ini.items("SelfConsistentModel"))

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial ('guessed') density profiles of all components
    densityBulge = agama.Density(**iniPotenBulge)
    densityHalo  = agama.Density(**iniPotenHalo)
    densityDisk  = agama.Density(**iniPotenDisk)

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(dens=densityHalo,  disklike=False))
    model.components.append(agama.Component(dens=densityBulge, disklike=False))
    model.components.append(agama.Component(dens=densityDisk,  disklike=True))

    # compute the initial potential
    model.iterate()
    writeRotationCurve("rotcurve_init", model.pot)

    # initialize the DFs of spheroidal components using the Eddington inversion formula
    # for their respective density profiles in the spherically-symmetric initial guess for the potential
    pot_sph = agama.Potential(type='Multipole', density=model.pot, lmax=0, gridsizer=100, rmin=1e-3, rmax=1e3)
    dfHalo  = agama.DistributionFunction(type='PseudoIsotropic', pot=pot_sph, dens=densityHalo)
    dfBulge = agama.DistributionFunction(type='PseudoIsotropic', pot=pot_sph, dens=densityBulge)
    printoutInfo(model, 0)

    print "**** STARTING ONE-COMPONENT MODELLING ****\nMasses are:  " \
        "Mhalo=%g,"  % densityHalo.totalMass(), \
        "Mbulge=%g," % densityBulge.totalMass(), \
        "Mdisk=%g"   % densityDisk.totalMass()

    # replace the halo SCM component with the DF-based one
    model.components[0] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)

    # do a couple of iterations to determine the self-consistent density profile of the halo
    print "Starting iteration #1"
    model.iterate()
    printoutInfo(model, 1)

    # now that we have a reasonable guess for the total potential,
    # we may initialize the DF of the stellar disk
    dfDisk = agama.DistributionFunction(pot=model.pot, **iniDFDisk)

    # we can compute the masses even though we don't know the density profile yet
    print "**** STARTING TWO-COMPONENT MODELLING ****\nMasses are: ", \
        "Mhalo=%g,"  % dfHalo.totalMass(), \
        "Mbulge=%g," % dfBulge.totalMass(), \
        "Mdisk=%g"   % dfDisk.totalMass()

    # and replace the static disk component them with a DF-based disk one
    model.components[2] = agama.Component(df=dfDisk, disklike=True, \
        gridR=agama.nonuniformGrid(int(iniSCMDisk['sizeRadialCyl']), \
            float(iniSCMDisk['RminCyl']), float(iniSCMDisk['RmaxCyl'])), \
        gridz=agama.nonuniformGrid(int(iniSCMDisk['sizeVerticalCyl']), \
            float(iniSCMDisk['zminCyl']), float(iniSCMDisk['zmaxCyl'])) )

    # do a few more iterations to obtain the self-consistent density profile for both disks
    for iteration in range(6,9):
        print "Starting iteration #%d" % iteration
        model.iterate()
        printoutInfo(model, iteration)

    print "Computing disk density and velocity profiles"
    R=numpy.linspace(0.2,10,50)
    xyz=numpy.vstack((R,R*0,R*0)).T
    Sigma,_   = agama.GalaxyModel(pot=model.pot, df=dfDisk, af=model.af).projectedMoments(R)
    rho,sigma = agama.GalaxyModel(pot=model.pot, df=dfDisk, af=model.af).moments(xyz)
    force, deriv = model.pot.forceDeriv(xyz)
    kappa = numpy.sqrt(-deriv[:,0]-3*force[:,0]/R)
    ToomreQ = sigma[:,0]**0.5*kappa/3.36/Sigma
    numpy.savetxt("disk_plane",
        numpy.vstack((R, Sigma, rho, sigma[:,0]**0.5, sigma[:,1]**0.5, ToomreQ)).T, fmt="%.6g")

    # export model to an N-body snapshot
    print "Creating an N-body representation of the model"

    # first create a representation of density profiles without velocities
    # (just for demonstration), by drawing samples from the density distribution
    print "Sampling halo density"
    writeNbodySnapshot("dens_halo_iter10",  model.components[0].getDensity().sample(800000))
    print "Sampling bulge density"
    writeNbodySnapshot("dens_bulge_iter10", model.components[1].getDensity().sample(40000))
    print "Sampling disk density"
    writeNbodySnapshot("dens_disk_iter10",  model.components[2].getDensity().sample(160000))

    # now create genuinely self-consistent models of both components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print "Sampling halo DF"
    writeNbodySnapshot("model_halo_iter10", \
        agama.GalaxyModel(pot=model.pot, df=dfHalo, af=model.af).sample(800000))
    print "Sampling bulge DF"
    writeNbodySnapshot("model_bulge_iter10", \
        agama.GalaxyModel(pot=model.pot, df=dfBulge, af=model.af).sample(40000))
    print "Sampling disk DF"
    writeNbodySnapshot("model_disk_iter10", \
        agama.GalaxyModel(pot=model.pot, df=dfDisk, af=model.af).sample(160000))
