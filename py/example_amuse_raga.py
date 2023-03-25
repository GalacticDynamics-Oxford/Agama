"""
Illustration of the use of Raga stellar-dynamical code from the AMUSE framework.

We create a Plummer model with a spectrum of masses, and follow the coupled
dynamical and stellar evolution represented by Raga and SeBa, respectively.
"""

useRaga = True or False   # choose whether to use Raga or a conventional N-body code (much slower!)

import numpy, matplotlib.pyplot as plt
from amuse.lab import *
from amuse.community.agama.interface import Agama

if __name__ in ('__main__', '__plot__'):

    # set up parameters:
    numpy.random.seed(42)  # make experiments repeatable
    N         = 16384
    Rcluster  = 1. | units.parsec
    #masses   = new_kroupa_mass_distribution(N)
    masses    = new_powerlaw_mass_distribution(N, mass_min=0.4|units.MSun, mass_max=20|units.MSun, alpha=-2.5)
    Mcluster  = masses.sum()
    converter = nbody_system.nbody_to_si(Mcluster, Rcluster)
    particles = new_plummer_model(N, converter)
    particles.mass = masses
    stellarevol = SeBa(redirection='null')
    stellarevol.particles.add_particles(particles)
    particles.radius = stellarevol.particles.radius
    if useRaga:
        cluster = Agama(converter, redirection='none', number_of_workers=8,
        updatepotential=True, coulombLog=5.0, symmetry='s',
        filelog='raga_multimass_plummer.log', fileoutputpotential='raga_multimass_plummer.pot',
        fileoutputrelaxation='raga_multimass_plummer.rel', fileoutput='raga_multimass_plummer.out', fileoutputformat='nemo',
        outputinterval=1. | units.Myr,
        numSamplesPerEpisode=10 )#, particles=particles)
    else:
        #cluster = ph4(converter, channel_type='sockets', number_of_workers=8, parameters=[("epsilon_squared", (0.01 | units.parsec)**2)])
        cluster = ph4(converter, channel_type='sockets', mode='gpu', parameters=[("epsilon_squared", (0.01 | units.parsec)**2)])
        #cluster = Gadget2(converter, channel_type='sockets', number_of_workers=8, parameters=[("epsilon_squared", (0.01 | units.parsec)**2)], redirection='null')
        #cluster = Bonsai(converter, channel_type='sockets', parameters=[("epsilon_squared", (0.01 | units.parsec)**2)], redirection='null')
    cluster.particles.add_particles(particles)

    channel_from_stellar_to_gravity   = stellarevol.particles.new_channel_to(cluster.particles, attributes=['mass'])
    channel_from_gravity_to_framework = cluster.particles.new_channel_to(particles)

    # evolve and make plots
    times = numpy.linspace(0,20,21) | units.Myr

    print("Start evolution")
    for i,t in enumerate(times):
        print("Evolve gravity")
        cluster.evolve_model(t)
        print("Evolve stars")
        stellarevol.evolve_model(t)
        channel_from_stellar_to_gravity.copy()
        channel_from_gravity_to_framework.copy()

        print("Evolved the system to time %.1f Myr" % t.value_in(units.Myr) +
            ", total mass=%.1f Msun"                % particles.mass.sum().value_in(units.MSun) +
            ", Ekin=%.4g Msun*(km/s)^2"             % cluster.kinetic_energy.value_in(units.MSun*units.kms**2) + 
            ", Etot=%.4g Msun*(km/s)^2"             % (cluster.potential_energy + cluster.kinetic_energy).value_in(units.MSun*units.kms**2))
        x=particles.x.value_in(units.parsec)
        y=particles.y.value_in(units.parsec)
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, marker='.', s=5*particles.mass.value_in(units.MSun)**0.5, edgecolors=None, c='k', alpha=0.2)
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.title("t=%.1f Myr" % t.value_in(units.Myr))
        basename = 'example_amuse_raga' if useRaga else 'example_amuse_nbody'
        plt.savefig('%s_%i.png' % (basename, i))
        plt.close()

    cluster.stop()
    stellarevol.stop()

