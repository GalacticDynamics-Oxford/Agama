#!/usr/bin/python
'''
This interactive example shows the meridional plane {x,z} (left panel) and the Poincare
surface of section {x,v_x} for an axisymmetric potential, where x is either cylindrical
radius R when L_z>0 or the x coordinate otherwise, and points on the SoS are placed
when passing through the z=0 plane with v_z>0.
Upon right-clicking at any point inside the zero-velocity curve on the surface of section,
a new orbit starting from these initial conditions is added to the plot.
The parameters of the potential, energy and L_z are specified at the beginning of the script.
'''
import agama, numpy, scipy.optimize, matplotlib, matplotlib.pyplot as plt
plt.rc('axes', linewidth=0.5)
plt.rc('font', size=8)
# consider motion in the x-z plane of an axisymmetric potential
# (with Lz=0 for motion in a flattened 2d potential, or Lz>0 for the motion in the meridional plane)
#pot  = agama.Potential(type='spheroid', gamma=1.5, q=0.5)
pot  = agama.Potential(type='disk', scaleheight=0.1)
rmax = 2.0
E    = pot.potential(rmax,0,0)
Lzmax= 2*numpy.pi * pot.Rcirc(E=E)**2 / pot.Tcirc(E)
Lz   = 0.1 * Lzmax

def init_axes(arg=None):
    axorb.cla()
    axpss.cla()
    axorb.set_xlim(0 if Lz>0 else -rmax, rmax)
    axorb.set_aspect('equal')
    axpss.set_xlim(axorb.get_xlim())
    axorb.set_xlabel('$x$', fontsize=12)
    axorb.set_ylabel('$z$', fontsize=12)
    axpss.set_xlabel('$x$', fontsize=12)
    axpss.set_ylabel('$p_x$', fontsize=12)
    # plot boundaries of orbit plane and surface of section
    Rp,Ra= pot.Rperiapo(E, Lz)
    xmin = -Ra if Lz==0 else Rp
    xmax = Ra
    grid = numpy.linspace(0, 1, 100)
    grid = grid * grid * (3-2*grid) * (xmax-xmin) + xmin
    vval = numpy.maximum(0, 2*E - 2*pot.potential(numpy.column_stack((grid,grid*0,grid*0))) - (Lz/grid)**2)**0.5
    zval = numpy.hstack([0,
        numpy.array([ scipy.optimize.brentq(lambda z: pot.potential(xx,0,z) - E + 0.5*(Lz/xx)**2, 0, xmax) for xx in grid[1:-1]]),
        0])
    axorb.plot(numpy.hstack((grid[:-1], grid[::-1])), numpy.hstack((zval[:-1], -zval[::-1])), color='k', lw=0.5)
    axpss.plot(numpy.hstack((grid[:-1], grid[::-1])), numpy.hstack((vval[:-1], -vval[::-1])), color='k', lw=0.5)
    axorb.text(0.5, 1.01, 'orbit plane',        ha='center', va='bottom', transform=axorb.transAxes, fontsize=10)
    axpss.text(0.5, 1.01, 'surface of section', ha='center', va='bottom', transform=axpss.transAxes, fontsize=10)
    plt.draw()

def run_orbit(ic):
    color = numpy.random.random(size=3)*0.8
    # create an orbit represented by a spline interpolator
    orbit = agama.orbit(ic=ic, potential=pot, time=100*pot.Tcirc(ic), dtype=object)
    # get all crossing points with z=0
    timecross = orbit.z.roots()
    # select those at which vz>=0
    timecross = timecross[orbit.z(timecross, der=1) >= 0]
    # get recorded trajectory sampled at every timestep...
    traj = orbit(orbit)
    # ...and at all crossing times
    trajcross = orbit(timecross)
    if Lz==0:
        axorb.plot(traj[:,0], traj[:,2], color=color, lw=0.5, alpha=0.5)
        axpss.plot(trajcross[:,0], trajcross[:,3], 'o', color=color, mew=0, ms=1.5)
    else:
        # orbit in the R,z plane, and SoS in the R, v_R plane
        axorb.plot((traj[:,0]**2 + traj[:,1]**2)**0.5, traj[:,2], color=color, lw=0.5, alpha=0.5)
        R = (trajcross[:,0]**2 + trajcross[:,1]**2)**0.5
        vR= (trajcross[:,0]*trajcross[:,3] + trajcross[:,1]*trajcross[:,4]) / R
        axpss.plot(R, vR, 'o', color=color, mew=0, ms=1.5)

def add_point(event):
    if event.inaxes is not axpss or event.button != 3: return
    x, vx = event.xdata, event.ydata
    vz2 = 2 * (E - pot.potential(x,0,0)) - (Lz/x)**2 - vx**2
    if vz2>0:
        run_orbit([x, 0, 0, vx, Lz/x, vz2**0.5])
        plt.draw()

fig   = plt.figure(figsize=(6,3), dpi=200)
axorb = plt.axes([0.08,0.14,0.4,0.8])
axpss = plt.axes([0.58,0.14,0.4,0.8])
button_clear = matplotlib.widgets.Button(plt.axes([0.90,0.88,0.08,0.06]), 'clear')
fig.canvas.mpl_connect('button_press_event', add_point)
button_clear.on_clicked(init_axes)
init_axes()
print('Right-click on the Surface of Section to start an orbit')
plt.show()
