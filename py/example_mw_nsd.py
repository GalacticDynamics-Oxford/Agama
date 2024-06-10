#!/usr/bin/python
'''
Construct a self-consistent DF-based model of the Milky Way nuclear stellar disk (NSD)
with an additional non-self-consistent component representing the nuclear star cluster (NSC).
This model is fitted to the current observational constraints, as detailed in
Sormani et al. 2022 (MNRAS/512/1857).
Authors: Mattia Sormani, Jason Sanders, Eugene Vasiliev
Date: Feb 2022
'''

import numpy, agama, matplotlib.pyplot as plt

def plotVcirc(model, iteration):
    rhos   = (model.components[0].density, model.components[1].density)
    pots   = (  # recompute potentials of both components separately, using a multipole Poisson solver
        agama.Potential(type='Multipole', lmax=6,  density=rhos[0], rmin=1e-4, rmax=0.1),
        agama.Potential(type='Multipole', lmax=12, density=rhos[1], rmin=1e-3, rmax=1.0))
    r      = numpy.linspace(0.0, 1.0, 1001)
    xyz    = numpy.column_stack((r, r*0, r*0))
    vcomp2 = numpy.column_stack([-pot.force(xyz)[:,0]*r for pot in pots])
    vtot2  = numpy.sum(vcomp2, axis=1)
    #print('NSC total mass=%g'%rhos[0].totalMass())
    print('NSD total mass=%g'%rhos[1].totalMass())
    # plot the circular-velocity curves for both NSC and NSD separately, and their sum
    plt.figure(figsize=(6,4))
    ax = plt.axes([0.15, 0.15, 0.8, 0.8])
    ax.plot(r, vtot2**0.5, color='k', label='total')
    ax.plot(r, vcomp2[:,0]**0.5, '--', color='k', label='NSC')
    ax.plot(r, vcomp2[:,1]**0.5, ':',  color='k', label='NSD')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 140)
    ax.set_xlabel(r'$R\, {\rm [kpc]}$', fontsize=16)
    ax.set_ylabel(r'$v_{\rm circ}\, {\rm [km/s]}$', fontsize=16)
    ax.grid()
    ax.legend()
    plt.savefig('nsd_vcirc_iter%i.pdf' % iteration)
    plt.close()

def plotModel(model, df):
    print('Creating density and kinematic plots...')
    gridx  = numpy.linspace(0, 0.70, 141)
    gridz  = numpy.linspace(0, 0.25, 51)
    gridxz = numpy.column_stack((numpy.tile(gridx, len(gridz)), numpy.repeat(gridz, len(gridx))))
    gridxyz= numpy.column_stack((numpy.tile(gridx, len(gridz)), numpy.zeros(len(gridx)*len(gridz)), numpy.repeat(gridz, len(gridx))))
    nsc = model.components[0].density
    nsd = model.components[1].density
    rho_nsd = nsd.density(gridxyz).reshape(len(gridz), len(gridx))
    Sig_nsd = nsd.projectedDensity(gridxz, beta=numpy.pi/2).reshape(len(gridz), len(gridx))
    plt.figure(figsize=(15,10), dpi=75)
    ax = numpy.array(
        [plt.axes([0.035, 0.81-0.195*i, 0.38, 0.18]) for i in range(5)] +
        [plt.axes([0.425, 0.81-0.195*i, 0.38, 0.18]) for i in range(5)]).reshape(2,5).T
    ax[0,0].contour(gridx, gridz, numpy.log10(rho_nsd), levels=numpy.log10(numpy.max(rho_nsd))+numpy.linspace(-6,-0,16), cmap='Blues')
    ax[0,1].contour(gridx, gridz, numpy.log10(Sig_nsd), levels=numpy.log10(numpy.max(Sig_nsd))+numpy.linspace(-6,-0,16), cmap='Blues')
    # compute moments on a coarser grid
    gridx = agama.nonuniformGrid(20, 0.02, 0.70)
    gridz = agama.nonuniformGrid(10, 0.02, 0.25)
    gridxz = numpy.column_stack((numpy.tile(gridx, len(gridz)), numpy.repeat(gridz, len(gridx))))
    gridxyz= numpy.column_stack((numpy.tile(gridx, len(gridz)), numpy.zeros(len(gridx)*len(gridz)), numpy.repeat(gridz, len(gridx))))
    gm = agama.GalaxyModel(model.potential, df)
    gridv = numpy.linspace(0, 100, 11)
    # intrinsic moments
    vel, vel2 = gm.moments(gridxyz, dens=False, vel=True, vel2=True, beta=numpy.pi/2)
    plt.clabel(ax[1,0].contour(gridx, gridz, -vel[:,2].reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[2,0].contour(gridx, gridz, numpy.sqrt(vel2[:,2]-vel[:,2]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[3,0].contour(gridx, gridz, numpy.sqrt(vel2[:,0]-vel[:,0]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[4,0].contour(gridx, gridz, numpy.sqrt(vel2[:,1]-vel[:,1]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    # projected moments
    vel, vel2 = gm.moments(gridxz, dens=False, vel=True, vel2=True, beta=numpy.pi/2)
    plt.clabel(ax[1,1].contour(gridx, gridz, -vel[:,2].reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[2,1].contour(gridx, gridz, numpy.sqrt(vel2[:,2]-vel[:,2]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[3,1].contour(gridx, gridz, numpy.sqrt(vel2[:,0]-vel[:,0]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    plt.clabel(ax[4,1].contour(gridx, gridz, numpy.sqrt(vel2[:,1]-vel[:,1]**2).reshape(len(gridz), len(gridx)), levels=gridv, cmap='Blues'), fmt='%.0f')
    labels = ['density', 'v_los', 'sigma_los', 'sigma_R', 'sigma_z']
    colors = ['c', 'm', 'b', 'r', 'g']
    # fiducial fields for computing projected velocity distributions
    points = [ [0.05, 0.01], [0.15, 0.01], [0.15, 0.10], [0.30, 0.01] ]
    for i in range(5):
        ax[i,0].set_ylabel('z [kpc]', labelpad=4)
        ax[i,1].set_yticklabels([])
        ax[i,0].text(0.5, 0.98, 'intrinsic '+labels[i], ha='center', va='top', transform=ax[i,0].transAxes, color=colors[i])
        ax[i,1].text(0.5, 0.98, 'projected '+labels[i], ha='center', va='top', transform=ax[i,1].transAxes, color=colors[i])
        if i<4:
            ax[i,0].set_xticklabels([])
            ax[i,1].set_xticklabels([])
        else:
            ax[i,0].set_xlabel('R [kpc]', labelpad=2)
            ax[i,1].set_xlabel('R [kpc]', labelpad=2)
        ax[i,0].set_xlim(min(gridx), max(gridx))
        ax[i,1].set_xlim(min(gridx), max(gridx))
        ax[i,0].set_ylim(min(gridz), max(gridz))
        ax[i,1].set_ylim(min(gridz), max(gridz))
        for k,point in enumerate(points):
            ax[i,1].text(point[0], point[1], chr(k+65), ha='center', va='center', color='olive')
    gridv = numpy.linspace(-250, 250, 26)   # coarse grid for computing the velocity distribution
    gridV = numpy.linspace(-250, 250, 101)  # fine grid for plotting a smoother spline-interpolated VDF
    vdfx, vdfz, vdfd = gm.vdf(points, gridv=gridv, beta=numpy.pi/2)
    for i in range(4):
        ax=plt.axes([0.845, 0.75-0.24*i, 0.15, 0.225])
        ax.plot(gridV, vdfd[i](-gridV), 'b', label='f(v_los)')
        ax.plot(gridV, vdfx[i](-gridV), 'r', label='f(v_R)')
        ax.plot(gridV, vdfz[i](-gridV), 'g', label='f(v_z)')
        ax.set_yscale('log')
        ax.text(0.02, 0.98, 'field %s' % chr(i+65), ha='left', va='top', transform=ax.transAxes, color='olive')
        ax.set_xlim(min(gridv), max(gridv))
        ax.set_ylim(1e-5, 3e-2)
        if i<3:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('V [km/s]')
        if i==0:
            ax.legend(loc='lower center', frameon=False)
            ax.text(1.0, 1.05, 'projected velocity distributions in fields', ha='right', va='center', transform=ax.transAxes)
        ax.set_ylabel('f(V)', labelpad=3)
    plt.savefig('nsd_model.pdf')
    plt.show()


if __name__ == '__main__':

    agama.setUnits(length=1, velocity=1, mass=1e10)   # 1 kpc, 1 km/s, 1e10 Msun

    # initialize the SelfConsistentModel object (only the potential expansion parameters)
    model = agama.SelfConsistentModel(
        RminCyl        = 0.005,
        RmaxCyl        = 1.0,
        sizeRadialCyl  = 25,
        zminCyl        = 0.005,
        zmaxCyl        = 1.0,
        sizeVerticalCyl= 25,
        RminSph        = 0.0001,
        RmaxSph        = 0.1,
        sizeRadialSph  = 25,
        lmaxAngularSph = 8
    )

    # construct a two component model: NSD + NSC
    # NSD -> generated self-consistently
    # NSC -> kept fixed as an external potential

    # NSC best-fitting model from Chatzopoulos et al. 2015 (see Equation 28 here: https://arxiv.org/pdf/2007.06577.pdf)
    density_NSC_init = agama.Density(type='Dehnen',mass=6.1e-3,gamma=0.71,scaleRadius=5.9e-3,axisRatioZ=0.73)

    # NSD model 3 from Sormani et al. 2020 (see Equation 24 here: https://arxiv.org/pdf/2007.06577.pdf)
    d1 = agama.Density(type='Spheroid',DensityNorm=0.9*222.885,gamma=0,beta=0,axisRatioZ=0.37,outerCutoffRadius=0.0050617,cutoffStrength=0.7194)
    d2 = agama.Density(type='Spheroid',DensityNorm=0.9*169.975,gamma=0,beta=0,axisRatioZ=0.37,outerCutoffRadius=0.0246,cutoffStrength=0.7933)
    density_NSD_init = agama.Density(d1,d2)

    # add both NSC and NSD components as static density profiles for the moment:
    # assign NSC and SMBH to the Multipole potential, and NSD to the CylSpline
    model.components.append(agama.Component(density=density_NSC_init, disklike=False))
    model.components.append(agama.Component(density=density_NSD_init, disklike=True))

    # compute the initial guess for the potential
    model.iterate()
    plotVcirc(model, 0)

    # introduce DF for the NSD component
    mass     = 0.097
    Rdisk    = 0.075
    Hdisk    = 0.025
    sigmar0  = 75.0
    Rsigmar  = 1.0
    sigmamin = 2.0
    Jmin     = 10.0
    dfNSD = agama.DistributionFunction(potential=model.potential, type='QuasiIsothermal',
        mass=mass, Rdisk=Rdisk, Hdisk=Hdisk, sigmar0=sigmar0, Rsigmar=Rsigmar, sigmamin=sigmamin, Jmin=Jmin)

    # replace the static density of the NSD by a DF-based component
    model.components[1] = agama.Component(df=dfNSD, disklike=True,
        RminCyl=0.005, RmaxCyl=0.75, sizeRadialCyl=20, zminCyl=0.005, zmaxCyl=0.25, sizeVerticalCyl=15)

    # iterate to make NSD DF & potential self-consistent
    for iteration in range(1,5):
        print('Starting iteration #%d' % iteration)
        model.iterate()
        plotVcirc(model, iteration)

    # plot density and kinematic structure of the final model
    plotModel(model, dfNSD)
