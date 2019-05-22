#!/usr/bin/python
'''
This example illustrates the use of Agama to construct a smooth potential approximation
for a snapshot from the FIRE simulation (stored in the GIZMO format, similar to GADGET).
It relied on external python packages - gizmo_analysis and utilities (sic!), both hosted at
https://bitbucket.org/awetzel/

Author:  Robyn Sanderson, with contributions from Andrew Wetzel, Eugene Vasiliev
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import agama
import numpy as np
import gizmo_analysis as ga
import utilities as ut
import sys

# base directory for all the simulations - change to the correct path, include trailing slash
# note that these files, of course, are not provided in the Agama distribution.
# the folder should contain a subfolder with the name of the simulation (e.g., 'm12i'),
# and in that subfolder there should be a file 'snapshot_times.txt', and yet another subfolder
# 'output', which contains files 'snapshot_***.*.hdf5'
sims_dir = '../../FIRE/'

# list of labels for symmetry spec
symmlabel={'a':'axi','s':'sph','t':'triax','n':'none'}

# define the physical units used in the code: the choice below corresponds to
# length scale = 1 kpc, velocity = 1 km/s, mass = 1 Msun
agama.setUnits(mass=1,length=1,velocity=1)

# tunable parameters for the potentials are:
# GridSizeR, rmin, rmax - specify the (logarithmic) grid in spherical radius
# (for Multipole) or cylindrical radius (for CylSpline); for the latter, there is a second grid in z,
# defined by GridSizeZ, zmin, zmax (by default the same as the radial grid).
# The minimum radius should be comparable to the smallest resolvable feature in the snapshot
# (e.g., the extent of the central cusp or core, or the disk scale height), but not smaller.
# The maximum radius defines the region where the potential and density approximated in full detail;
# note that for Multipole, the density is extrapolated as a power law outside this radius,
# and the potential takes into account the contribution of input particles at all radii,
# while for CylSpline the particles outside the grid are ignored, and the density is zero there.
# lmax - the order of multipole expansion;
# mmax - the order of azimuthal Fourier expansion for CylSpline (if it is axisymmetric, mmax is ignored)
# Note that the default parameters are quite sensible, but not necessarily optimal for your case.

def fitPotential(sim_name,
                 nsnap=600,
                 symmetry='a',
                 subsample_factor=1,
                 rmax_sel=600,
                 rmax_ctr=10,
                 rmax_exp=500,
                 save_coords=True):

        '''
        constructs a hybrid two-component basis expansion model of the potential for one Gizmo snapshot.
        dark matter and hot gas are represented by an expansion in spherical harmonics.
        remaining baryons (stars and cold gas) are represented by an azimuthal harmonic expansion in phi and a quintic spline in (R,z).
        (see Agama docs, sections 2.2.2 and 2.2.3 for more details).

        Arguments:
        sim_name [str]: name of simulation folder within main dir
        nsnap [int]: snapshot number
        symmetry [char]: 's' for spherical, 'a' for axisymmetric, 't' for triaxial, 'n' for none (see table 2 of Agama docs)
        subsample_factor [int]: factor by which to subsample (for speedup/testing)
        rmax_sel [float]: max radius in kpc to select particles for fitting the model
        rmax_ctr [float]: radius (kpc) defining the subset of stars which are used to define the reference frame (centering and rotation)
        rmax_exp [float]: max radial extent in kpc of the hybrid potential expansion (both components)
        save_coords [bool]: save center position, velocity, mean acceleration, and rotation matrix for principal-axis frame, to a file
        '''
        print('reading snapshot')

        #read in the snapshot
        part = ga.io.Read.read_snapshots(species=['gas', 'star', 'dark'],
                                         snapshot_values=nsnap,
                                         simulation_directory=sims_dir+sim_name,
                                         # snapshot_directory='output_accel',
                                         particle_subsample_factor=subsample_factor,
                                         assign_host_coordinates=True,
                                         assign_host_principal_axes=True)

        # start with default centering and rotation to define aperture

        dist=ut.particle.get_distances_wrt_center(part, species=['gas','star','dark'],
                rotation=part.host_rotation_tensors[0], total_distance = True)
        dist_vectors = ut.particle.get_distances_wrt_center(part, species=['gas','star','dark'],
                rotation=part.host_rotation_tensors[0])


        # compute new centering and rotation using a fixed aperture in stars

        sp = 'star'
        ctr_indices = np.where(dist[sp]<rmax_ctr)[0]

        m = part[sp]['mass'][ctr_indices]
        pos = part[sp]['position'][ctr_indices]
        vel = part[sp]['velocity'][ctr_indices]
        new_ctr = np.multiply(m,pos.T).sum(axis=1)/np.sum(m)
        new_vctr = np.multiply(m,vel.T).sum(axis=1)/np.sum(m)
        new_rot = ut.particle.get_principal_axes(part,'star',part_indices=ctr_indices,
                                                 print_results=False)

        #optionally compute acceleration of center of mass frame if it was recorded 

        save_accel = ('acceleration' in part[sp].keys())
        if save_accel:
                print('saving acceleration of COM frame')
                accel = part[sp]['acceleration'][ctr_indices]
                new_actr = np.multiply(m,accel.T).sum(axis=1)/np.sum(m)

        # recompute distances in the new frame

        dist=ut.particle.get_distances_wrt_center(part,
                                                  species=['star','gas','dark'],
                                                  center_position=new_ctr,
                                                  rotation=new_rot['rotation.tensor'],
                                                  total_distance = True)
        dist_vectors = ut.particle.get_distances_wrt_center(part,
                                                            species=['star','gas','dark'],
                                                            center_position=new_ctr,
                                                            rotation=new_rot['rotation.tensor'])

        #pick out gas and stars within the region that we want to supply to the model

        m_gas_tot = part['gas']['mass'].sum()*subsample_factor

        pos_pa_gas = dist_vectors['gas'][dist['gas']<rmax_sel]
        m_gas = part['gas']['mass'][dist['gas']<rmax_sel]*subsample_factor
        print('{0:.3g} of {1:.3g} solar masses in gas selected'.format(m_gas.sum(),m_gas_tot))


        m_star_tot = part['star']['mass'].sum()*subsample_factor

        pos_pa_star = dist_vectors['star'][dist['star']<rmax_sel]
        m_star = part['star']['mass'][dist['star']<rmax_sel]*subsample_factor
        print('{0:.3g} of {1:.3g} solar masses in stars selected'.format(m_star.sum(),m_star_tot))

        #separate cold gas in disk (modeled with cylspline) from hot gas in halo (modeled with multipole)

        tsel = (np.log10(part['gas']['temperature'])<4.5)
        rsel = (dist['gas']<rmax_sel)

        pos_pa_gas_cold = dist_vectors['gas'][tsel&rsel]
        m_gas_cold = part['gas']['mass'][tsel&rsel]*subsample_factor
        print('{0:.3g} of {1:.3g} solar masses are cold gas to be modeled with cylspline'.format(m_gas_cold.sum(),m_gas.sum()))

        pos_pa_gas_hot = dist_vectors['gas'][(~tsel)&rsel]
        m_gas_hot = part['gas']['mass'][(~tsel)&rsel]*subsample_factor
        print('{0:.3g} of {1:.3g} solar masses are hot gas to be modeled with multipole'.format(m_gas_hot.sum(),m_gas.sum()))


        #combine components that will be fed to the cylspline part
        pos_pa_bar = np.vstack((pos_pa_star,pos_pa_gas_cold))
        m_bar = np.hstack((m_star,m_gas_cold))


        #pick out the dark matter
        m_dark_tot = part['dark']['mass'].sum()*subsample_factor

        rsel = dist['dark']<rmax_sel
        pos_pa_dark=dist_vectors['dark'][rsel]
        m_dark = part['dark']['mass'][rsel]*subsample_factor
        print('{0:.3g} of {1:.3g} solar masses in dark matter selected'.format(m_dark.sum(),m_dark_tot))

        #stack with hot gas for multipole density
        pos_pa_dark = np.vstack((pos_pa_dark,pos_pa_gas_hot))
        m_dark = np.hstack((m_dark,m_gas_hot))

        if save_coords:
                #save the Hubble parameter to transform to comoving units
                hubble = part.info['hubble']
                scalefactor = part.info['scalefactor']

        del(part)

        #right now, configured to save to a new directory in the simulation directory.
        #Recommended, since it's generally useful to have around

        output_stem = sims_dir+sim_name+'/potential/{0:.0f}kpc/{1}_{2:d}'.format(rmax_ctr,sim_name,nsnap)
        try:    # create the directory if it didn't exist
                os.makedirs(os.path.dirname(output_stem))
        except OSError as e:
                if e.errno != errno.EEXIST:
                        raise

        if save_coords:
                cname = '{0}_coords.txt'.format(output_stem)
                print('Saving coordinate transformation to {0}'.format(cname))
                with open(cname,'w') as f:
                        f.write('# Hubble parameter and scale factor (to convert physical <-> comoving) \n')
                        f.write('{0:.18g} {1:.18g}\n'.format(hubble, scalefactor))
                        f.write('# center position (kpc comoving)\n')
                        np.savetxt(f,new_ctr)
                        f.write('# center velocity (km/s physical)\n')
                        np.savetxt(f,new_vctr)
                        if save_accel:
                                f.write('# center acceleration (km/s^2 physical)\n')
                                np.savetxt(f,new_actr)
                        f.write('# rotation to principal-axis frame\n')
                        np.savetxt(f,new_rot['rotation.tensor'])


        print('Computing multipole expansion coefficients for dark matter/hot gas component')

        p_dark=agama.Potential(type='multipole',
                               particles=(pos_pa_dark, m_dark),
                               lmax=4, symmetry=symmetry,
                               rmin=0.1, rmax=rmax_exp)


        p_dark.export('{0}.dark.{1}.coef_mul'.format(output_stem,symmlabel[symmetry]))
        print('Computing cylindrical spline coefficients for stellar/cold gas component')

        p_bar = agama.Potential(type='cylspline',
                                particles=(pos_pa_bar, m_bar),
                                mmax=4, symmetry=symmetry,
                                #gridsizer=40, gridsizez=40,
                                rmin=0.1, rmax=rmax_exp)

        p_bar.export('{0}.bar.{1}.coef_cylsp'.format(output_stem,symmlabel[symmetry]))
        print('done, enjoy your potentials!')


if __name__ == "__main__":
        import os, errno
        import argparse

        parser = argparse.ArgumentParser(description="Constructs and saves a hybrid "
                                         "two-component basis expansion model of the potential for one Gizmo snapshot. "
                                         "The dark matter and hot gas are represented by an expansion in spherical harmonics. "
                                         "The cold gas and stars are represented by an azimuthal harmonic "
                                         "expansion in phi and a quintic spline in (R,z). See Agama docs, sections "
                                         "2.2.2 and 2.2.3 for more details.")

        parser.add_argument('--simname',help='name of simulation folder within main dir (required)', default=None)
        parser.add_argument('--nsnap', type=int, help='snapshot number', default=600)
        parser.add_argument('--symmetry', help="s' for spherical, 'a' for axisymmetric, 't' for triaxial, 'n' for none (see table 2 of Agama docs)", default='a')
        parser.add_argument('--subfactor', type=int, help='factor by which to subsample snapshot particles (for speedup/testing)', default=1)
        parser.add_argument('--rsel', type=float, help='max radius in kpc to select particles for the model (should be larger than rmaxc, rmaxm)', default=600)
        parser.add_argument('--rmaxc', type=float, help='max radius in kpc to select particles for centering the model', default=10)
        parser.add_argument('--rmaxe', type=float, help='max radial extent in kpc of the hybrid potential expansion', default=400)
        parser.add_argument('--savec', type=bool, help='save coordinate transformation to a file', default=True)


        args=parser.parse_args()

        #check that at least simname is specified
        if args.simname is None:
                print('Error: no simulation specified')
                parser.print_help()
                parser.exit()

        #check that symmetry is a valid letter
        if args.symmetry not in 'satn':
                print('Error: invalid symmetry specification')
                parser.print_help()
                parser.exit()

        #otherwise start the program
        fitPotential(args.simname,
                     nsnap=args.nsnap,
                     symmetry=args.symmetry,
                     subsample_factor=args.subfactor,
                     rmax_sel=args.rsel,
                     rmax_ctr=args.rmaxc,
                     rmax_exp=args.rmaxe,
                     save_coords=args.savec)
