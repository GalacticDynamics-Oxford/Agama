#!/usr/bin/python
'''
This file is part of the Gaia Challenge, and contains routines for plotting the results.
See gc_runfit.py for the overall description.

This program creates plots with the density and velocity dispersion profiles
for three cases referring to the same mock data sample,
but fitted with different assumptions about the data related to the tracer population:
full 6d phase-space coodinates with no errors, 5d (except for z) with a 2km/s error
on velocity, and 3d (only x,y and vz, the latter with a 2km/s error).
Each case should be previously processed with the gc_runfit.py program,
and the inferred best-fit parameters stored in its own folder with the same name
as the data file, prepended by '3', '5' or '6'.
After all results have been obtained, this program creates summary plots for all 3 cases,
showing (1) the density profile and its log-derivative of the dark matter
(which is responsible for the overall potential), and
(2) the density and velocity dispersion profiles of the tracer population.
'''
import sys, os, numpy, matplotlib, agama
matplotlib.use('Agg')
from matplotlib import pyplot
from gc_modelparams import ModelParams
import triangle as corner
matplotlib.rcParams['legend.frameon']=False

num_chain_samples = 3200  # plot this many samples from the tail of the chain


def plot_profiles(filename, indx, label):
    points   = numpy.loadtxt(filename)
    ptradii  = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2)**0.5
    ptpercnt = numpy.percentile(ptradii, [1, 99])
    chain    = numpy.loadtxt(filename+".chain")
    chain    = chain[-num_chain_samples:, :]
    dens     = numpy.zeros((chain.shape[0], len(radii)))
    gamma    = numpy.zeros((chain.shape[0], len(midradii)))
    for i in range(chain.shape[0]):
        dens [i,:] = model.createPotential(chain[i,:]).density(xyz)
        gamma[i,:] = numpy.log(dens[i,1:]/dens[i,:-1]) / numpy.log(radii[1:]/radii[:-1])
    true_dens  = model.truePotential.density(xyz)
    true_gamma = numpy.log(true_dens[1:]/true_dens[:-1]) / numpy.log(radii[1:]/radii[:-1])
    print "done with",filename

    cntr = numpy.percentile(gamma, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
    axes[0,indx].fill_between(midradii, cntr[0,:], cntr[4,:], color='lightgray')  # 2 sigma
    axes[0,indx].fill_between(midradii, cntr[1,:], cntr[3,:], color='gray')       # 1 sigma
    axes[0,indx].plot(midradii, cntr[2,:], color='k')  # median
    axes[0,indx].plot(midradii, true_gamma, color='r', lw=3, linestyle='--', label='log slope')
    axes[0,indx].plot([ptpercnt[0],ptpercnt[0]], [-4,-3], ':g')
    axes[0,indx].plot([ptpercnt[1],ptpercnt[1]], [-4,-3], ':g')
    axes[0,indx].text((ptpercnt[0]*ptpercnt[1])**0.5, -3.5, '98% data points', ha='center', color='green', fontsize=8)
    axes[0,indx].set_xscale('log')
    axes[0,indx].set_xlim(rmin, rmax)
    axes[0,indx].set_ylim(-4, 0)
    axes[0,indx].set_xlabel('$r$')
    axes[0,indx].set_ylabel(r'$\gamma$')

    cntr = numpy.percentile(dens, [2.3, 15.9, 50, 84.1, 97.7], axis=0)
    axes[1,indx].fill_between(radii, cntr[0,:], cntr[4,:], color='lightgray')  # 2 sigma
    axes[1,indx].fill_between(radii, cntr[1,:], cntr[3,:], color='gray')       # 1 sigma
    axes[1,indx].plot(radii, cntr[2,:], color='k')  # median
    axes[1,indx].plot(radii, true_dens, color='r', lw=3, linestyle='--', label='DM density')
    axes[1,indx].set_xscale('log')
    axes[1,indx].set_yscale('log')
    axes[1,indx].set_xlim(rmin, rmax)
    axes[1,indx].set_ylim(true_dens[-1]*0.5, true_dens[0]*5)
    axes[1,indx].set_xlabel('$r$')
    axes[1,indx].set_ylabel(r'$\rho$')
    axes[1,indx].text( (rmin*rmax)**0.5, true_dens[-1], label, ha='center')


def compute_orig_moments(points):
    '''
    Compute the moments (density and velocity dispersions in radial and tangential direction)
    of the original data points (tracer particles), binned in radius.
    Return: tuple of four arrays: radii, density, radial velocity dispersion and tangential v.d.,
    where the array of radii is one element longer than the other three arrays, and denotes the bin boundaries,
    and the other arrays contain the values in each bin.
    '''
    radii = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2) ** 0.5
    velsq = (points[:,3]**2 + points[:,4]**2 + points[:,5]**2)
    velradsq = ((points[:,0]*points[:,3] + points[:,1]*points[:,4] + points[:,2]*points[:,5]) / radii) ** 2
    veltansq = (velsq - velradsq) * 0.5  # per each of the two tangential directions
    sorted_radii = numpy.sort(radii)
    indices = numpy.linspace(0, len(radii)-1, 30).astype(int)
    hist_boundaries = sorted_radii[indices]
    sumnum,_ = numpy.histogram(radii, bins=hist_boundaries, weights=numpy.ones_like(radii))
    sumvelradsq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=velradsq)
    sumveltansq,_ = numpy.histogram(radii, bins=hist_boundaries, weights=veltansq)
    binvol = 4*3.1416/3 * (hist_boundaries[1:]**3 - hist_boundaries[:-1]**3)
    density= sumnum/len(points[:,0]) / binvol
    sigmar = (sumvelradsq/sumnum)**0.5
    sigmat = (sumveltansq/sumnum)**0.5
    return hist_boundaries, density, sigmar, sigmat


def plot_tracers(filename, indx, label):
    '''
    plot radial profiles of velocity dispersion of tracer particles and their density
    '''
    points   = numpy.loadtxt(filename)
    ptradii  = (points[:,0]**2 + points[:,1]**2 + points[:,2]**2)**0.5
    ptpercnt = numpy.percentile(ptradii, [1, 99])
    params   = numpy.loadtxt(filename+".best")
    for i in range(params.shape[0]):
        try:
            pot = model.createPotential(params[i])
            df  = model.createDF(params[i])
        except ValueError as err:
            print err, params[i]
        dens, mom = agama.GalaxyModel(pot, df).moments(xyz, dens=True, vel=False, vel2=True)
        axes[0,indx].plot(radii, mom[:,0]**0.5, color='r', alpha=0.5)  # velocity dispersions
        axes[0,indx].plot(radii, mom[:,1]**0.5, color='b', alpha=0.5)
        axes[1,indx].plot(radii, dens, color='k', alpha=0.5)    # tracer density
    print "done with",filename

    # binned vel.disp. and density profile from input particles; emulating steps plot
    bins, dens, sigmar, sigmat = compute_orig_moments(points)
    axes[0,indx].plot(numpy.hstack(zip(bins[:-1], bins[1:])), numpy.hstack(zip(sigmar, sigmar)), \
        color='r', lw=1, label=r'$\sigma_r$')
    axes[0,indx].plot(numpy.hstack(zip(bins[:-1], bins[1:])), numpy.hstack(zip(sigmat, sigmat)), \
        color='b', lw=1, label=r'$\sigma_t$')
    axes[1,indx].plot(numpy.hstack(zip(bins[:-1], bins[1:])), numpy.hstack(zip(dens, dens)), \
        color='g', lw=1, label=r'Tracer density, binned')

    # analytic density profile
    axes[1,indx].plot(radii, model.tracerDensity.density(xyz), \
        color='g', lw=3, linestyle='--', label=r'Tracer density, analytic')
    densmin = model.tracerDensity.density(bins[-1],0,0)
    densmax = model.tracerDensity.density(bins[0] ,0,0)*5

    axes[0,indx].plot([ptpercnt[0],ptpercnt[0]], [velmin,velmin+3], ':g')
    axes[0,indx].plot([ptpercnt[1],ptpercnt[1]], [velmin,velmin+3], ':g')
    axes[0,indx].text((ptpercnt[0]*ptpercnt[1])**0.5, velmin+2, '98% data points', ha='center', color='green', fontsize=8)
    axes[0,indx].set_xscale('log')
    axes[0,indx].set_yscale('linear')
    axes[0,indx].legend(loc='upper right')
    axes[0,indx].set_xlim(rmin, rmax)
    axes[0,indx].set_ylim(velmin, velmax)
    axes[0,indx].set_xlabel('$r$')
    axes[0,indx].set_ylabel(r'$\sigma_{r,t}$')
    axes[1,indx].set_xscale('log')
    axes[1,indx].set_yscale('log')
    #axes[1,indx].legend(loc='lower left')
    axes[1,indx].set_xlim(rmin, rmax)
    axes[1,indx].set_ylim(densmin, densmax)
    axes[1,indx].set_xlabel('$r$')
    axes[1,indx].set_ylabel(r'$\rho$')
    axes[1,indx].text( (rmin*rmax)**0.5, densmin*2, label, ha='center')


################  MAIN PROGRAM  ##################
#base     = "gs010_bs050_rcrs100_rarcinf_core_0400mpc3_df"
if len(sys.argv)<=1:
    print "Provide the data file name as the command-line argument"
    exit()
agama.setUnits(mass=1, length=1, velocity=1)
base     = sys.argv[1]
model    = ModelParams(base)
rmin     = 0.01
rmax     = 100.
velmin   = 0.
velmax   = 40.
radii    = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), 25)
midradii = (radii[1:] * radii[:-1])**0.5
xyz      = numpy.vstack((radii, numpy.zeros_like(radii), numpy.zeros_like(radii))).T

# plot the inferred density of dark matter and its log-slope as functions of radius
fig,axes = pyplot.subplots(2, 3, figsize=(12,8))
plot_profiles("6"+base+"/"+base+"_1000_0.dat",     0,  '6d, no errors')
plot_profiles("5"+base+"/"+base+"_1000_0_err.dat", 1, r'5d, $\delta v$=2 km/s')
plot_profiles("3"+base+"/"+base+"_1000_0_err.dat", 2, r'3d, $\delta v$=2 km/s')
fig.tight_layout()
pyplot.savefig(base+"_darkmatter.png")

# plot the density and velocity dispersions of tracer particles as functions of radius
fig,axes = pyplot.subplots(2, 3, figsize=(12,8))
plot_tracers("6"+base+"/"+base+"_1000_0.dat",     0,  '6d, no errors')
plot_tracers("5"+base+"/"+base+"_1000_0_err.dat", 1, r'5d, $\delta v$=2 km/s')
plot_tracers("3"+base+"/"+base+"_1000_0_err.dat", 2, r'3d, $\delta v$=2 km/s')
fig.tight_layout()
pyplot.savefig(base+"_tracers.png")
