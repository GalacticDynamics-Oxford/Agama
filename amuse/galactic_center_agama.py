"""
Evolves a cluster in the potention of the galactic center
  
Uses the bridge integrator to couple different codes.
In this example a cluster is evolved  circling the galactic center, represented by a static potential.
"""

import numpy

from amuse.units import units
from amuse.units import constants
from amuse.units import nbody_system
from amuse.ext.bridge import bridge
from amuse.community.phiGRAPE.interface import PhiGRAPE
from amuse.community.agama.interface import Agama
from matplotlib import pyplot
from amuse.ic.kingmodel import new_king_model

if __name__ in ('__main__', '__plot__'):

    # set up parameters:
    N = 100
    W0 = 3
    Rinit = 50. | units.parsec
    timestep = 0.01 | units.Myr
    Mcluster = 4.e4 | units.MSun
    Rcluster = 0.7 | units.parsec
    converter = nbody_system.nbody_to_si(Mcluster,Rcluster)

    # create a globular cluster model
    particles = new_king_model(N, W0, convert_nbody=converter)
    particles.radius = 0.0| units.parsec
    cluster = PhiGRAPE(converter, parameters=[("epsilon_squared", (0.01 | units.parsec)**2)])

    # create the external potential of the Galaxy
    galaxy = Agama(converter, type="Dehnen", gamma=1.8, \
        rscale=1000.| units.parsec, mass=1.6e10 | units.MSun, channel_type='sockets')

    # shift the cluster to an orbit around Galactic center
    acc,_,_ = galaxy.get_gravity_at_point(0|units.kpc, Rinit, 0|units.kpc, 0|units.kpc)
    vcirc = (-acc * Rinit)**0.5
    print "Vcirc=", vcirc.value_in(units.kms), "km/s"
    particles.x  += Rinit
    particles.vy += vcirc
    cluster.particles.add_particles(particles)

    # set up bridge; cluster is evolved under influence of the galaxy
    sys = bridge(verbose=False)
    sys.add_system(cluster, (galaxy,), False)

    # evolve and make plots
    times = units.Myr([0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4])
    f = pyplot.figure(figsize=(8,16))

    for i,t in enumerate(times):
        sys.evolve_model(t, timestep=timestep)
        print "Evolved the system to time",t

        x=sys.particles.x.value_in(units.parsec)
        y=sys.particles.y.value_in(units.parsec)

        subplot=f.add_subplot(4,2,i+1)
        subplot.plot(x,y,'r .')
        subplot.plot([0.],[0.],'b +')
        subplot.set_xlim(-60,60)
        subplot.set_ylim(-60,60)
        subplot.set_title(t)
        if i==7:
            subplot.set_xlabel('parsec')

    cluster.stop()
    galaxy.stop()
    pyplot.show()
#    pyplot.savefig('test.eps')

