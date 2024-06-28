#!/usr/bin/python
# illustrate the use of Torus Machine for transformation from action/angle to position/velocity
import agama, numpy, matplotlib.pyplot as plt
ax = plt.subplots(2,2, figsize=(15,10))[1]
pot = agama.Potential(type='Spheroid', gamma=0, beta=5, alpha=2, mass=10, axisratioz=.6)  # flattened Plummer
act = numpy.array([1.0, 2.0, 3.0])  # Jr, Jz, Jphi - some fiducial values
am = agama.ActionMapper(pot)  # J,theta -> x,v (Torus)
af = agama.ActionFinder(pot)  # x,v -> J,theta (Staeckel fudge)
t = numpy.linspace(0, 100, 1001)
# construct the orbit using Torus
Omega_torus = am(numpy.hstack((act, [0,0,0])), frequencies=True)[1]
xv_torus = am(numpy.column_stack((act + t[:,None]*0, Omega_torus*t[:,None])))
ax[0,0].plot((xv_torus[:,0]**2+xv_torus[:,1]**2)**0.5, xv_torus[:,2], label='torus', dashes=[4,2])
# construct the orbit by numerically integrating the equations of motion
_,xv_orbit = agama.orbit(ic=xv_torus[0], potential=pot, time=t[-1], trajsize=len(t))
ax[0,0].plot((xv_orbit[:,0]**2+xv_orbit[:,1]**2)**0.5, xv_orbit[:,2], label='orbit')
# compute actions for the torus orbit using Staeckel approximation
J,theta,Omega = af(xv_torus, angles=True)
ax[0,1].plot(t, J, label='J,torus', dashes=[4,2])
ax[1,1].plot(t, Omega, label='Omega,torus', dashes=[4,2])
ax[1,0].plot(t, theta, label='theta,torus', dashes=[4,2])
# same for the actual orbit
J,theta,Omega = af(xv_orbit, angles=True)
ax[0,1].plot(t, J, label='J,orbit')
ax[1,1].plot(t, Omega, label='Omega,orbit')
ax[1,0].plot(t, theta, label='theta,orbit')
ax[0,0].legend()
ax[0,1].legend()
ax[1,0].legend()
ax[1,1].legend()
plt.tight_layout()
plt.show()
