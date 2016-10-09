#!/usr/bin/python
### This module allows to use potentials from the AGAMA C++ library as regular galpy potentials

import math
import agama, galpy
from galpy.potential import Potential
class CPotential(galpy.potential.Potential):
    """Class that implements an interface to C++ potentials
    """
    def __init__(self,*args,**kwargs):
        """
        NAME:
           __init__
        PURPOSE:
           initialize a potential from parameters provided in an INI file
           or as named arguments to the constructor (see below).
        INPUT:
           normalize - if True, normalize such that vc(1.,0.)=1., or,
           if given as a number, such that the force is this fraction of the force
           necessary to make vc(1.,0.)=1.
        HISTORY:
           2014-12-05 EV
        """
        Potential.__init__(self,amp=1.)
        normalize=False
        for key, value in kwargs.items():
            if key=="normalize":
                normalize=value
                del kwargs[key]
        self._pot = agama.Potential(*args,**kwargs)
        if normalize or \
                (isinstance(normalize,(int,float)) \
                    and not isinstance(normalize,bool)):
            self.normalize(normalize)
        self.hasC= False
        self.hasC_dxdv=False
    __init__.__doc__ += agama.Potential.__doc__

    def _evaluate(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z,phi)
        HISTORY:
           2014-12-05 EV
        """
        return self._pot.potential(R*math.cos(phi), R*math.sin(phi), z)

    def _Rforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force=self._pot.force(coord[0], coord[1], z)
        return (force[0]*coord[0]+force[1]*coord[1])/R

    def _zforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        HISTORY:
           2014-12-05 EV
        """
        return self._pot.force(R*math.cos(phi), R*math.sin(phi), z)[2]

    def _phiforce(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phiforce
        PURPOSE:
           evaluate the azimuthal force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the azimuthal force
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force=self._pot.force(coord[0], coord[1], z)
        return force[1]*coord[0]-force[0]*coord[1]

    def _dens(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _dens
        PURPOSE:
           evaluate the density for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the density
        HISTORY:
           2014-12-05 EV
        """
        return self._pot.density(R*math.cos(phi), R*math.sin(phi), z)

    def _R2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _R2deriv
        PURPOSE:
           evaluate the second radial derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second radial derivative
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force,deriv=self._pot.forceDeriv(coord[0], coord[1], z)
        return -(deriv[0]*coord[0]**2+deriv[1]*coord[1]**2+2*deriv[3]*coord[0]*coord[1])/R**2

    def _z2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _z2deriv
        PURPOSE:
           evaluate the second vertical derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second vertical derivative
        HISTORY:
           2014-12-05 EV
        """
        return -self._pot.forceDeriv(R*math.cos(phi), R*math.sin(phi), z)[1][2]

    def _phi2deriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _phi2deriv
        PURPOSE:
           evaluate the second azimuthal derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the second azimuthal derivative
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force,deriv=self._pot.forceDeriv(coord[0], coord[1], z)
        return -(deriv[0]*coord[1]**2 + deriv[1]*coord[0]**2 - 2*deriv[3]*coord[0]*coord[1]
                 - force[0]*coord[0] - force[1]*coord[1])

    def _Rzderiv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rzderiv
        PURPOSE:
           evaluate the mixed R,z derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dz
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force,deriv=self._pot.forceDeriv(coord[0], coord[1], z)
        return -(deriv[5]*coord[0]+deriv[4]*coord[1])/R

    def _Rphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _Rphideriv
        PURPOSE:
           evaluate the mixed R,phi derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dR/dphi
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force,deriv=self._pot.forceDeriv(coord[0], coord[1], z)
        return -((deriv[1]-deriv[0])*coord[1]*coord[0] + deriv[3]*(coord[0]**2-coord[1]**2)
                 - force[0]*coord[1] + force[1]*coord[0])/R

    def _zphideriv(self,R,z,phi=0.,t=0.):
        """
        NAME:
           _zphideriv
        PURPOSE:
           evaluate the mixed z,phi derivative for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           d2phi/dz/dphi
        HISTORY:
           2014-12-05 EV
        """
        coord=R*math.cos(phi), R*math.sin(phi)
        force,deriv=self._pot.forceDeriv(coord[0], coord[1], z)
        return -(deriv[4]*coord[0]-deriv[5]*coord[1])/R
