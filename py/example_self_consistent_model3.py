#!/usr/bin/python
"""
Example of construction of a three-component disk-bulge-halo equilibrium model of a galaxy.
The approach is explained in example_self_consistent_model.py;
this example differs in that it has a somewhat simpler structure (only a single stellar disk
component, no stellar halo or gas disk) and adds a central supermassive black hole.
Another modification is that the halo and the bulge are represented by 'quasi-isotropic' DF:
it is a spherical isotropic DF that is constructed using the Eddington inversion formula
for the given density profile in the spherically-symmetric approximation of the total potential.
This DF is then expressed in terms of actions and embedded into the 'real', non-spherical
potential, giving rise to a somewhat different density profile; however, it is close enough
to the input one. Then a few more iterations are needed to converge towards a self-consistent
model.
"""

import agama, numpy, os, sys, matplotlib.pyplot as plt
try:
    from ConfigParser import RawConfigParser  # python 2
except ImportError:
    from configparser import RawConfigParser  # python 3

# write out the circular velocity curve for the entire model and per component
def writeRotationCurve(filename, potentials, names):
    radii = numpy.logspace(-3.0, 2.0, 101)
    xyz   = numpy.column_stack((radii, radii*0, radii*0))
    vcomp2= numpy.column_stack([-potential.force(xyz)[:,0] * radii for potential in potentials])
    vtot2 = numpy.sum(vcomp2, axis=1)
    numpy.savetxt(filename, numpy.column_stack((radii, vtot2**0.5, vcomp2**0.5)), fmt="%.6g", header="radius\tVcTotal\t"+"\t".join(names))

# print some diagnostic information after each iteration
def printoutInfo(model, iteration):
    densDisk = model.components[0].getDensity()
    densBulge= model.components[1].getDensity()
    densHalo = model.components[2].getDensity()
    pt0 = (2.0, 0, 0)
    pt1 = (2.0, 0, 0.25)
    pt2 = (0.0, 0, 2.0)
    print("Disk  total mass=%g, rho(R=2,z=0)=%g, rho(R=2,z=0.25)=%g" % \
        (densDisk.totalMass(), densDisk.density(pt0), densDisk.density(pt1)))
    print("Bulge total mass=%g, rho(R=0.5,z=0)=%g" % \
        (densBulge.totalMass(), densBulge.density(0.4, 0, 0)))
    print("Halo  total mass=%g, rho(R=2,z=0)=%g, rho(R=0,z=2)=%g" % \
        (densHalo.totalMass(), densHalo.density(pt0), densHalo.density(pt2)))
    # report only the potential of stars+halo, excluding the potential of the central BH (0th component)
    pot0 = model.potential.potential(0,0,0) - model.potential[0].potential(0,0,0)
    print("Potential at origin=-(%g)^2, total mass=%g" % ((-pot0)**0.5, model.potential.totalMass()))
    densDisk. export("dens_disk_" +iteration)
    densBulge.export("dens_bulge_"+iteration)
    densHalo. export("dens_halo_" +iteration)
    model.potential.export("potential_"+iteration)
    # separate the contributions of bulge and halo, which are normally combined
    # into the Multipole potential of all spheroidal components
    writeRotationCurve("rotcurve_"+iteration, (
        model.potential[0], # potential of the BH
        model.potential[2], # potential of the disk
        agama.Potential(type='Multipole', lmax=6, density=densBulge),  # -"- bulge
        agama.Potential(type='Multipole', lmax=6, density=densHalo) ), # -"- halo
        ('BH', 'Disk', 'Bulge', 'Halo') )

if __name__ == "__main__":
    # read parameters from the INI file
    iniFileName = os.path.dirname(os.path.realpath(sys.argv[0])) + "/../data/SCM3.ini"
    ini = RawConfigParser()
    #ini.optionxform=str  # do not convert key to lowercase
    ini.read(iniFileName)
    iniPotenHalo  = dict(ini.items("Potential halo"))
    iniPotenBulge = dict(ini.items("Potential bulge"))
    iniPotenDisk  = dict(ini.items("Potential disk"))
    iniPotenBH    = dict(ini.items("Potential BH"))
    iniDFDisk     = dict(ini.items("DF disk"))
    iniSCMHalo    = dict(ini.items("SelfConsistentModel halo"))
    iniSCMBulge   = dict(ini.items("SelfConsistentModel bulge"))
    iniSCMDisk    = dict(ini.items("SelfConsistentModel disk"))
    iniSCM        = dict(ini.items("SelfConsistentModel"))

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(**iniSCM)

    # create initial density profiles of all components
    densityDisk  = agama.Density(**iniPotenDisk)
    densityBulge = agama.Density(**iniPotenBulge)
    densityHalo  = agama.Density(**iniPotenHalo)
    potentialBH  = agama.Potential(**iniPotenBH)

    # add components to SCM - at first, all of them are static density profiles
    model.components.append(agama.Component(density=densityDisk,  disklike=True))
    model.components.append(agama.Component(density=densityBulge, disklike=False))
    model.components.append(agama.Component(density=densityHalo,  disklike=False))
    model.components.append(agama.Component(potential=potentialBH))

    # compute the initial potential
    model.iterate()
    printoutInfo(model,'init')

    # construct the DF of the disk component, using the initial (non-spherical) potential
    dfDisk  = agama.DistributionFunction(potential=model.potential, **iniDFDisk)
    # initialize the DFs of spheroidal components using the Eddington inversion formula
    # for their respective density profiles in the initial potential
    dfBulge = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityBulge)
    dfHalo  = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential, density=densityHalo)

    print("\033[1;33m**** STARTING ITERATIVE MODELLING ****\033[0m\nMasses (computed from DF): " \
        "Mdisk=%g, Mbulge=%g, Mhalo=%g" % (dfDisk.totalMass(), dfBulge.totalMass(), dfHalo.totalMass()))

    # replace the initially static SCM components with the DF-based ones
    model.components[0] = agama.Component(df=dfDisk,  disklike=True,  **iniSCMDisk)
    model.components[1] = agama.Component(df=dfBulge, disklike=False, **iniSCMBulge)
    model.components[2] = agama.Component(df=dfHalo,  disklike=False, **iniSCMHalo)

    # do a few more iterations to obtain the self-consistent density profile for both disks
    for iteration in range(1,5):
        print("\033[1;37mStarting iteration #%d\033[0m" % iteration)
        model.iterate()
        printoutInfo(model, 'iter%d'%iteration)

    # export model to an N-body snapshot
    print("\033[1;33mCreating an N-body representation of the model\033[0m")
    format = 'text'  # one could also use 'nemo' or 'gadget' here

    # first create a representation of density profiles without velocities
    # (just for demonstration), by drawing samples from the density distribution
    print("Sampling disk density")
    agama.writeSnapshot("dens_disk_final",  model.components[0].getDensity().sample(160000), format)
    print("Sampling bulge density")
    agama.writeSnapshot("dens_bulge_final", model.components[1].getDensity().sample(40000), format)
    print("Sampling halo density")
    agama.writeSnapshot("dens_halo_final",  model.components[2].getDensity().sample(800000), format)

    # now create genuinely self-consistent models of both components,
    # by drawing positions and velocities from the DF in the given (self-consistent) potential
    print("Sampling disk DF")
    agama.writeSnapshot("model_disk_final", \
        agama.GalaxyModel(potential=model.potential, df=dfDisk,  af=model.af).sample(1600000), format)
    print("Sampling bulge DF")
    agama.writeSnapshot("model_bulge_final", \
        agama.GalaxyModel(potential=model.potential, df=dfBulge, af=model.af).sample(400000), format)
    print("Sampling halo DF")
    # note: use a 10x larger particle mass for halo than for bulge/disk
    agama.writeSnapshot("model_halo_final", \
        agama.GalaxyModel(potential=model.potential, df=dfHalo,  af=model.af).sample(3000000), format)

    # the remaining part computes and plots various diagnostics
    print("\033[1;33mComputing disk density and velocity profiles\033[0m")
    ax=plt.subplots(2, 3, figsize=(16,10))[1].reshape(-1)
    # take only the disk component
    modelDisk = agama.GalaxyModel(potential=model.potential, df=dfDisk, af=model.af)
    # radial grid for computing various quantities in the disk plane
    Sigma0 = float(iniPotenDisk["surfacedensity"])
    Rdisk  = float(iniPotenDisk["scaleradius"])
    Hdisk  =-float(iniPotenDisk["scaleheight"])
    sigmar0= float(iniDFDisk["sigmar0"])
    rsigmar= float(iniDFDisk["rsigmar"])
    R   = agama.nonuniformGrid(60, 0.01*Rdisk, 10.0*Rdisk)
    xyz = numpy.column_stack((R, R*0, R*0))
    print("Computing surface density")
    Sigma,rmsh,rmsv   = modelDisk.projectedMoments(R)
    print("Computing 3d density and velocity dispersion")
    rho,vel,sigma = modelDisk.moments(xyz, dens=True, vel=True, vel2=True)
    force, deriv = model.potential.forceDeriv(xyz)
    kappa = numpy.sqrt(-deriv[:,0] - 3*force[:,0]/R)
    ToomreQ = sigma[:,0]**0.5 * kappa / 3.36 / Sigma
    numpy.savetxt("disk_plane",
        numpy.column_stack((R, Sigma, rho, rmsh, sigma[:,0]**0.5, (sigma[:,2]-vel**2)**0.5, \
        sigma[:,1]**0.5, vel, ToomreQ)),
        header="R Sigma rho(R,z=0) height sigma_R sigma_phi sigma_z v_phi ToomreQ", fmt="%.6g")
    ax[0].plot(R, Sigma / (Sigma0 * numpy.exp(-R/Rdisk)), 'r-', label=r'$\Sigma$')
    ax[0].plot(R, rho   / (Sigma0 * numpy.exp(-R/Rdisk) * 0.25/Hdisk), 'g-', label=r'$\rho_{z=0}$')
    ax[0].plot(R, sigma[:,0]**0.5 / (sigmar0 * numpy.exp(-R/rsigmar)), 'b-', label=r'$\sigma_r$')
    ax[0].plot(R, rmsh / (1.8*Hdisk), 'm-', label=r'$h$')
    ax[0].set_xscale("log")
    ax[0].set_xlabel("R")
    ax[0].set_ylim(0,2)
    ax[0].legend(loc='lower right', frameon=False)
    # vertical grid for density
    print("Computing vertical density profiles")
    z = numpy.linspace(0, 6*Hdisk, 31)
    R = numpy.array([0.05, 0.5, 1.0, 3.0]) * Rdisk
    xyz = numpy.column_stack((numpy.tile(R, len(z)), numpy.zeros(len(R)*len(z)), numpy.repeat(z, len(R))))
    rho = modelDisk.moments(xyz, vel2=False).reshape(len(z), len(R))
    numpy.savetxt("vertical_density", numpy.column_stack((z, rho)), fmt="%.6g",
        header="z\\R:\t" + "\t".join(["%.4g" % r for r in R]))
    colors=['r','g','b','m']
    for i,r in enumerate(R):
        ax[1].plot(z, rho[:,i], '-', c=colors[i], label='R='+str(r))
        rho_init = Sigma0 * numpy.exp(-r/Rdisk) / Hdisk / (numpy.exp(-z/2/Hdisk) + numpy.exp(z/2/Hdisk))**2
        ax[1].plot(z, rho_init, ':', c=colors[i])
    ax[1].set_xlabel('z')
    ax[1].set_ylabel(r'$\rho$')
    ax[1].set_yscale('log')
    ax[1].legend(loc='lower left', frameon=False)
    # grid for computing velocity distributions
    print("Computing velocity distributions")
    z = numpy.array([0, 2*Hdisk])
    xyz = numpy.column_stack((numpy.tile(R, len(z)), numpy.zeros(len(R)*len(z)), numpy.repeat(z, len(R))))
    # create grids in velocity space for computing the spline representation of VDF (optional)
    # range: 0.75 v_escape from the galactic center (excluding the central BH potential)
    v_max = 0.75 * (-2 * (model.potential.potential(0,0,0)-model.potential[0].potential(0,0,0)))**0.5
    gridv = numpy.linspace(-v_max, v_max, 80)  # use the same grid for all dimensions
    # compute the distributions (represented as cubic splines)
    splvR, splvz, splvphi = modelDisk.vdf(xyz, gridv)
    # output f(v) at a different grid of velocity values
    gridv = numpy.linspace(-v_max, v_max, 251)
    for i,p in enumerate(xyz):
        numpy.savetxt("veldist_R="+str(p[0])+"_z="+str(p[2]),
            numpy.column_stack((gridv, splvR[i](gridv), splvz[i](gridv), splvphi[i](gridv))),
            fmt="%.6g", delimiter="\t", header="V\tf(V_R)\tf(V_z)\tf(V_phi) [1/(km/s)]")
        if i<len(ax)-2:
            ax[i+2].plot(gridv, splvR  [i](gridv), 'r-', label='$f(v_R)$')
            ax[i+2].plot(gridv, splvz  [i](gridv), 'g-', label='$f(v_z)$')
            ax[i+2].plot(gridv, splvphi[i](gridv), 'b-', label='$f(v_\phi)$')
            ax[i+2].set_xlabel('v')
            ax[i+2].set_yscale('log')
            ax[i+2].legend(loc='upper left', frameon=False)
            ax[i+2].set_xlim(-v_max, v_max)
            ax[i+2].set_ylim(1e-7,10)
            ax[i+2].text(0, 5e-7, "R="+str(p[0])+"_z="+str(p[2]), ha='center')
    plt.tight_layout()
    plt.show()
