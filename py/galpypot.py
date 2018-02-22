#!/usr/bin/python
"""
This module provides the class AgamaPotential that inherits galpy.potential.Potential,
and can be used as a regular galpy potential class, although for the orbit integration
or action computation, the native Agama counterparts are preferred.

If this script is run as the main program, it illustrates the use of both Galpy and Agama
potentials, orbit integration and action finder routines for the same example orbit.
"""

from galpy.potential import Potential as GPotential
import numpy as _numpy
class AgamaPotential(GPotential):
    """Class that implements a Galpy interface to Agama potentials"""
    def __init__(self,*args,**kwargs):
        """
Initialize a potential from parameters provided in an INI file
or as named arguments to the constructor.
Arguments are the same as for regular agama.Potential (see below);
an extra keyword "normalize=..." has the same meaning as in Galpy:
if True, normalize such that vc(1.,0.)=1., or,
if given as a number, such that the force is this fraction of the force
necessary to make vc(1.,0.)=1.
"""
        from galpy.potential import Potential as GPotential
        from agama import Potential as APotential
        GPotential.__init__(self,amp=1.)
        normalize=False
        for key, value in kwargs.items():
            if key=="normalize":
                normalize=value
                del kwargs[key]
        self._pot = APotential(*args,**kwargs)  # regular Agama potential
        if normalize or (isinstance(normalize,(int,float)) and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv=False

    # extend the docstring
    from agama import Potential as APotential
    __init__.__doc__ += "\n" + APotential.__doc__
    del APotential

    def _coord(self,R,z,phi):
        """convert input cylindrical coordinates to a Nx3 array in cartesian coords"""
        if phi is None: phi=0.
        return _numpy.array((R*_numpy.cos(phi), R*_numpy.sin(phi), z)).T

    def _evaluate(self,R,z,phi=0.,t=0.):
        """evaluate the potential at cylindrical coordinates R,z,phi"""
        return self._pot.potential(self._coord(R,z,phi))

    def _Rforce(self,R,z,phi=0.,t=0.):
        """evaluate the radial force for this potential: -dPhi/dR"""
        coord=self._coord(R,z,phi)
        force=_numpy.array(self._pot.force(coord))
        return (force.T[0]*coord.T[0] + force.T[1]*coord.T[1]) / R

    def _zforce(self,R,z,phi=0.,t=0.):
        """evaluate the vertical force for this potential: -dPhi/dz"""
        return _numpy.array(self._pot.force(self._coord(R,z,phi))).T[2]

    def _phiforce(self,R,z,phi=0.,t=0.):
        """evaluate the azimuthal force for this potential: -dPhi/dphi"""
        coord=self._coord(R,z,phi)
        force=_numpy.array(self._pot.force(coord))
        return force.T[1]*coord.T[0] - force.T[0]*coord.T[1]

    def _dens(self,R,z,phi=0.,t=0.):
        """evaluate the density for this potential"""
        return self._pot.density(self._coord(R,z,phi))

    def _2deriv(self,R,z,phi):
        """evaluate the potential derivatives in cartesian coordinates"""
        coord=self._coord(R,z,phi)
        force,deriv=self._pot.forceDeriv(coord)
        return coord.T, _numpy.array(force).T, _numpy.array(deriv).T

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """evaluate the second radial derivative for this potential: d2Phi / dR^2"""
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[0]*coord[0]**2 + deriv[1]*coord[1]**2 +
               2*deriv[3]*coord[0]*coord[1]) / R**2

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """evaluate the second vertical derivative for this potential: d2Phi / dz^2"""
        return -_numpy.array(self._pot.forceDeriv(self._coord(R,z,phi))[1]).T[2]

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """evaluate the second azimuthal derivative for this potential: d2Phi / dphi^2"""
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[0]*coord[1]**2 + deriv[1]*coord[0]**2 -
               2*deriv[3]*coord[0]*coord[1] - force[0]*coord[0] - force[1]*coord[1])

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """evaluate the mixed R,z derivative for this potential: d2Phi / dR dz"""
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[5]*coord[0] + deriv[4]*coord[1]) / R

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """evaluate the mixed R,phi derivative for this potential: d2Phi / dR dphi"""
        coord,force,deriv=self._2deriv(R,z,phi)
        return -((deriv[1]-deriv[0])*coord[1]*coord[0] + deriv[3]*(coord[0]**2-coord[1]**2)
            - force[0]*coord[1] + force[1]*coord[0]) / R

    def _zphideriv(self,R,z,phi=0.,t=0.):
        """evaluate the mixed z,phi derivative for this potential: d2Phi / dz dphi"""
        coord,force,deriv=self._2deriv(R,z,phi)
        return -(deriv[4]*coord[0] - deriv[5]*coord[1])


if __name__ == '__main__':
    # this demo script shows how to use Agama library in galpy:
    # creating a galpy-compatible potential, integrating an orbit and using the action finder

    import agama
    import galpy.potential, galpy.actionAngle, galpy.orbit
    import matplotlib, matplotlib.pyplot as plt, time
    matplotlib.rcParams['legend.frameon']=False

    ### set up galpy potential
    g_pot = galpy.potential.MWPotential2014

    ### set up equivalent potential from the Agama library
    agama.setUnits( mass=1., length=8., velocity=220)
    a_pot = AgamaPotential("../data/MWPotential2014galpy.ini")
    c_pot = a_pot._pot   # the instance of raw Agama potential

    ### initialization of the action finder needs to be done once for the given potential
    dt = time.time()
    c_actfinder = agama.ActionFinder(c_pot, interp=False)
    print('Time to set up action finder: %.4g s' % (time.time()-dt))
    ### we have a faster but less accurate "interpolated action finder", which takes a bit longer to initialize
    dt = time.time()
    i_actfinder = agama.ActionFinder(c_pot, interp=True)
    print('Time to set up interpolated action finder: %.4g s' % (time.time()-dt))

    ### conversion from prolate spheroidal to cylindrical coords
    def ProlSphToCyl(la, nu, fd):
        return ( ((la - fd*fd) * (1 - abs(nu)/fd**2))**0.5, (la*abs(nu))**0.5 / fd * _numpy.sign(nu) )

    ### show coordinate grid in prolate spheroidal coords
    def plotCoords(fd, maxR):
        la = _numpy.linspace(0, maxR, 32)**2 + fd**2
        ls = _numpy.linspace(0, 1, 21)
        nu = ls*ls*(3-2*ls)*fd**2
        for i in range(len(la)):
            lineR, linez = ProlSphToCyl(la[i], nu, fd)
            plt.plot(lineR, linez, 'grey', lw=0.5)
            plt.plot(lineR,-linez, 'grey', lw=0.5)
        for i in range(len(nu)):
            lineR, linez = ProlSphToCyl(la, nu[i], fd)
            plt.plot(lineR, linez, 'grey', lw=0.5)
            plt.plot(lineR,-linez, 'grey', lw=0.5)

    ### ic is the array of initial conditions: R, z, phi, vR, vz, vphi
    def compare(ic, inttime, numsteps):
        times = _numpy.linspace(0, inttime, numsteps)

        ### integrate the orbit in galpy using MWPotential2014 from galpy
        g_orb_obj = galpy.orbit.Orbit([ic[0],ic[3],ic[5],ic[1],ic[4],ic[2]])
        dt = time.time()
        g_orb_obj.integrate(times, g_pot)
        g_orb = g_orb_obj.getOrbit()
        print('Time to integrate orbit in galpy: %.4g s' % (time.time()-dt))

        ### integrate the orbit with the galpy routine, but using Agama potential instead
        ### (much slower because of repeated transfer of control between C++ and Python
        dt = time.time()
        g_orb_obj.integrate(times[:numsteps//10], a_pot)
        a_orb = g_orb_obj.getOrbit()
        print('Time to integrate 1/10th of the orbit in galpy using Agama potential: %.4g s' % (time.time()-dt))

        ### integrate the same orbit (note different calling conventions - cartesian coordinates as input)
        ### using both the orbit integration routine and the potential from Agama - much faster
        dt = time.time()
        times_c, c_orb_car = agama.orbit(ic=[ic[0],0,ic[1],ic[3],ic[5],ic[4]], \
            potential=c_pot, time=inttime, trajsize=numsteps)
        print('Time to integrate orbit in Agama: %.4g s' % (time.time()-dt))

        ### make it compatible with galpy's convention (native output is in cartesian coordinates)
        c_orb = c_orb_car*1.0
        c_orb[:,0] = (c_orb_car[:,0]**2+c_orb_car[:,1]**2)**0.5
        c_orb[:,3] =  c_orb_car[:,2]

        ### in galpy, this is the only tool that can estimate focal distance,
        ### but it requires the orbit to be computed first
        delta = galpy.actionAngle.estimateDeltaStaeckel(g_pot, g_orb[:,0], g_orb[:,3])
        print('Focal distance estimated from the entire trajectory: Delta=%.4g' % delta)

        ### plot the orbit(s) in R,z plane, along with the prolate spheroidal coordinate grid
        plt.figure(figsize=(12,8))
        plt.axes([0.04, 0.54, 0.45, 0.45])
        plotCoords(delta, 1.5)
        plt.plot(g_orb[:,0],g_orb[:,3], 'b', label='galpy')  # R,z
        plt.plot(c_orb[:,0],c_orb[:,3], 'g', label='Agama')
        plt.plot(a_orb[:,0],a_orb[:,3], 'r', label='galpy using Agama potential')
        plt.xlabel("R/8kpc")
        plt.ylabel("z/8kpc")
        plt.xlim(0, 1.2)
        plt.ylim(-1,1)
        plt.legend(loc='lower left')

        ### create galpy action/angle finder for the given value of Delta
        ### note: using c=False in the routine below is much slower but apparently more accurate,
        ### comparable to the Agama for the same value of delta
        g_actfinder = galpy.actionAngle.actionAngleStaeckel(pot=g_pot, delta=delta, c=True)

        ### find the actions for each point of the orbit using galpy action finder
        dt = time.time()
        g_act = g_actfinder(g_orb[:,0],g_orb[:,1],g_orb[:,2],g_orb[:,3],g_orb[:,4],fixed_quad=True)
        print('Time to compute actions in galpy: %.4g s' % (time.time()-dt))
        print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
            (_numpy.mean(g_act[0]), _numpy.std(g_act[0]), _numpy.mean(g_act[2]), _numpy.std(g_act[2])))

        ### use the Agama action routine for the same value of Delta as in galpy -
        ### the result is almost identical but computed much faster
        dt = time.time()
        c_act = agama.actions(point=c_orb_car, potential=c_pot, fd=delta)   # explicitly specify focal distance
        print('Time to compute actions in Agama using Galpy-estimated focal distance: %.4g s' % (time.time()-dt))
        print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
            (_numpy.mean(c_act[:,0]), _numpy.std(c_act[:,0]), _numpy.mean(c_act[:,1]), _numpy.std(c_act[:,1])))

        ### use the Agama action finder (initialized at the beginning) that automatically determines
        ### the best value of Delta (same computational cost as the previous one)
        dt = time.time()
        a_act = c_actfinder(c_orb_car)   # use the focal distance estimated by action finder
        print('Time to compute actions in Agama using pre-initialized focal distance: %.4g s' % (time.time()-dt))
        print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
            (_numpy.mean(a_act[:,0]), _numpy.std(a_act[:,0]), _numpy.mean(a_act[:,1]), _numpy.std(a_act[:,1])))

        ### use the interpolated Agama action finder (initialized at the beginning) - less accurate but faster
        dt = time.time()
        i_act = i_actfinder(c_orb_car)
        print('Time to compute actions in Agama with interpolated action finder: %.4g s' % (time.time()-dt))
        print('Jr = %.6g +- %.4g, Jz = %.6g +- %.4g' % \
            (_numpy.mean(i_act[:,0]), _numpy.std(i_act[:,0]), _numpy.mean(i_act[:,1]), _numpy.std(i_act[:,1])))

        ### plot Jr vs Jz
        plt.axes([0.54, 0.54, 0.45, 0.45])
        plt.plot(g_act[0],  g_act[2],   c='b', label='galpy')
        plt.plot(c_act[:,0],c_act[:,1], c='g', label=r'Agama,$\Delta=%.4f$'%delta)
        plt.plot(a_act[:,0],a_act[:,1], c='r', label=r'Agama,$\Delta=$auto')
        plt.plot(i_act[:,0],i_act[:,1], c='c', label=r'Agama,interpolated')
        plt.xlabel("$J_r$")
        plt.ylabel("$J_z$")
        plt.legend(loc='lower left')

        ### plot Jr(t) and Jz(t)
        plt.axes([0.04, 0.04, 0.95, 0.45])
        plt.plot(times,   g_act[0],   c='b', label='galpy')
        plt.plot(times,   g_act[2],   c='b')
        plt.plot(times_c, c_act[:,0], c='g', label='Agama,$\Delta=%.4f$'%delta)
        plt.plot(times_c, c_act[:,1], c='g')
        plt.plot(times_c, a_act[:,0], c='r', label='Agama,$\Delta=$auto')
        plt.plot(times_c, a_act[:,1], c='r')
        plt.plot(times_c, i_act[:,0], c='c', label='Agama,interpolated')
        plt.plot(times_c, i_act[:,1], c='c')
        plt.text(0, c_act[0,0], '$J_r$', fontsize=16)
        plt.text(0, c_act[0,1], '$J_z$', fontsize=16)
        plt.xlabel("t")
        plt.ylabel("$J_r, J_z$")
        plt.legend(loc='center right',ncol=2)
        #plt.ylim(0.14,0.25)
        plt.xlim(0,50)
        plt.show()

    compare([0.5, 0, 0, 0.82, 1.0, 0.28], 100., 1000)
    #compare([0.8, 0, 0, 0.3, 0.223, 0.75], 100., 1000)

# clean up the namespace from imported elements
del GPotential
