import numpy, agama, gala.potential, time
numpy.set_printoptions(linewidth=100, precision=8)
numpy.random.seed(42)

gpot1=gala.potential.PlummerPotential(1,1) # native Gala potential
Gpot1=agama.GalaPotential(gpot1)           # chimera object providing both Gala and Agama interfaces
apot1=agama.Potential(type='plummer')      # native Agama potential
Apot1=agama.GalaPotential(type='plummer')  # exactly the same functionality for Agama, but adds a Gala interface

gpot2=gala.potential.MiyamotoNagaiPotential(m=1, a=1, b=0.1)
Gpot2=agama.GalaPotential(gpot2)
Apot2=agama.GalaPotential(type='miyamotonagai', scaleradius=1, scaleheight=0.1)

gpot3=gala.potential.HernquistPotential(m=2.3, c=4.5)
Gpot3=agama.GalaPotential(gpot3)
Apot3=agama.GalaPotential(type='dehnen', gamma=1, mass=2.3, scaleradius=4.5)

gpot4=gala.potential.NFWPotential(m=1.2, r_s=3.4)
Gpot4=agama.GalaPotential(gpot4)
Apot4=agama.GalaPotential(type='nfw', mass=1.2, scaleradius=3.4)

# some random test points
points=numpy.random.normal(size=(5,3))  # positions
pointv=numpy.random.normal(size=(20,6))  # positions and velocities

def test(gpot, Gpot, Apot):
    '''
    gpot is the native gala potential;
    Gpot is the same one accessed through wrapper;
    Apot is the equivalent native agama potential
    '''
    print("\nTesting G=%s and A=%s" % (Gpot, Apot))
    #0. retrieve dimensional factors for converting the output of agama-related methods
    # to the potential's unit system (assuming it is identical for both potentials)
    lu = Gpot.units['length']
    pu = Gpot.units['energy'] / Gpot.units['mass']
    vu = Gpot.units['length'] / Gpot.units['time']
    fu = Gpot.units['length'] / Gpot.units['time']**2
    du = Gpot.units['mass'] / Gpot.units['length']**3
    #1. test that the values of potential coincide, using both interfaces
    print("G energy vs potential: %.3g" % numpy.max(abs(Gpot.energy(points.T) / Gpot.potential(points)/pu - 1)))
    print("A energy vs potential: %.3g" % numpy.max(abs(Apot.energy(points.T) / Apot.potential(points)/pu - 1)))
    print("G energy vs A energy:  %.3g" % numpy.max(abs(Gpot.energy(points.T) / Apot.energy(points.T) - 1)))
    #2. test the accelerations - these don't exactly match,
    # because agama evaluates them by finite-differencing if initialized from a "foreign" gala potential
    print("G acceleration vs force:  %.3g" % numpy.max(abs(Gpot.acceleration(points.T).T / Gpot.force(points)/fu - 1)))
    print("A acceleration vs force:  %.3g" % numpy.max(abs(Apot.acceleration(points.T).T / Apot.force(points)/fu - 1)))
    print("G gradient vs A gradient: %.3g" % numpy.max(abs(Gpot.gradient(points.T) / Apot.gradient(points.T) - 1)))
    #3. test the hessian
    print("G acc.deriv vs A acc.deriv:  %.3g" % numpy.max(abs(Gpot.eval(points,der=1) / Apot.eval(points,der=1) - 1)))
    print("G hessian vs A hessian:      %.3g" % numpy.max(abs(Gpot.hessian(points.T) / Apot.hessian(points.T) - 1)))
    #4. test density - again no exact match since it is computed by finite differences in Gpot
    print("G density [gala] vs [agama]: %.3g" % numpy.max(abs(Gpot.density(points.T) / Gpot.agamadensity(points)/du - 1)))
    print("A density [gala] vs [agama]: %.3g" % numpy.max(abs(Apot.density(points.T) / Apot.agamadensity(points)/du - 1)))
    print("G density vs A density:      %.3g" % numpy.max(abs(Gpot.density(points.T) / Apot.density(points.T) - 1)))
    #5. test orbit integration using both gala and agama routines with either potential
    # create some initial conditions for orbits:
    # take the positions and assign velocity to be comparable to circular velocity at each point
    ic = numpy.hstack((points, numpy.random.normal(size=points.shape) * Apot.circular_velocity(points.T)[:,None] / vu))
    t0 = time.time()
    g_orb_g = gpot.integrate_orbit(ic.T, dt=10, n_steps=100, Integrator=gala.integrate.DOPRI853Integrator, Integrator_kwargs=dict(rtol=1e-8,atol=0))
    t1 = time.time()
    g_orb_G = Gpot.integrate_orbit(ic.T, dt=10, n_steps=100, Integrator=gala.integrate.DOPRI853Integrator, Integrator_kwargs=dict(rtol=1e-8,atol=0))
    t2 = time.time()
    g_orb_A = Apot.integrate_orbit(ic.T, dt=10, n_steps=100, Integrator=gala.integrate.DOPRI853Integrator, Integrator_kwargs=dict(rtol=1e-8,atol=0))
    t3 = time.time()
    a_orb_G = numpy.dstack(agama.orbit(potential=Gpot, ic=ic, time=1000, trajsize=101, dtype=float)[:,1])
    t4 = time.time()
    a_orb_A = numpy.dstack(agama.orbit(potential=Apot, ic=ic, time=1000, trajsize=101, dtype=float)[:,1])
    t5 = time.time()
    print("gala  orbit integration for g (native): %.4g s" % (t1-t0) +
         ", G (wrapper): %.4g s" % (t2-t1) + ", A: %.4g s" % (t3-t2))
    print("agama orbit integration for G: %.4g s" % (t4-t3) + ", A: %.4g s" % (t5-t4))
    deltaEg = numpy.max(abs(g_orb_G.energy()[-1] / g_orb_G.energy()[0] - 1))
    Eainit  = Apot.potential(a_orb_A[0, 0:3].T) + 0.5*numpy.sum(a_orb_A[0, 3:6]**2, axis=0)
    Eafinal = Apot.potential(a_orb_A[-1,0:3].T) + 0.5*numpy.sum(a_orb_A[-1,3:6]**2, axis=0)
    deltaEa = numpy.max(abs(Eafinal / Eainit - 1))
    # shape of the output is different: for gala, it is 3 x nsteps x norbits; for agama (dstack'ed) - nsteps x 6 x norbits
    g_orb_G = g_orb_G.xyz.reshape(3, len(g_orb_G.t), len(ic))
    g_orb_A = g_orb_A.xyz.reshape(3, len(g_orb_A.t), len(ic))
    a_orb_G = numpy.swapaxes(a_orb_G*lu, 0, 1)[0:3]  # now the shape is 3 x nsteps x norbits
    a_orb_A = numpy.swapaxes(a_orb_A*lu, 0, 1)[0:3]
    maxrad  = numpy.max(numpy.sum(a_orb_A**2, axis=0)**0.5, axis=0)  # normalization factor for relative deviations in position
    print("gala  orbits G vs A: %.3g" % numpy.max(numpy.sum(abs(g_orb_G - g_orb_A), axis=0) / maxrad))
    print("agama orbits G vs A: %.3g" % numpy.max(numpy.sum(abs(a_orb_G - a_orb_A), axis=0) / maxrad))
    print("gala vs agama: %.3g"       % numpy.max(numpy.sum(abs(g_orb_G - a_orb_G), axis=0) / maxrad))
    print("energy error in gala: %.3g, agama: %.3g" % (deltaEg, deltaEa))
    # check the equivalence of hessian evaluated in a vectorized call vs. in a loop one point at a time
    hess_G_vec  = Gpot.hessian(points.T)
    hess_G_loop = numpy.dstack([Gpot.hessian(point) for point in points])
    hess_A_vec  = Apot.hessian(points.T)
    hess_A_loop = numpy.dstack([Apot.hessian(point) for point in points])
    if numpy.any(hess_G_vec != hess_G_loop) or numpy.any(hess_A_vec != hess_A_loop):
        print("Hessian vectorization test failed")

test(gpot1, Gpot1, Apot1)
test(gpot2, Gpot2, Apot2)
test(gpot3, Gpot3, Apot3)
test(gpot4, Gpot4, Apot4)

# now test potentials defined with units
agama.setUnits(length=1, mass=1, velocity=977.7922217536624)  # velocity unit = 1 kpc / 1 Myr
gpot6=gala.potential.MilkyWayPotential()
Gpot6=agama.GalaPotential(gpot6)
Apot6=agama.GalaPotential(
    dict(type='miyamotonagai', mass=6.8e10, scaleradius=3.0, scaleheight=0.28),  # disk
    dict(type='dehnen', mass=5.00e9, scaleradius=1.0),   # bulge
    dict(type='dehnen', mass=1.71e9, scaleradius=0.07),  # nucleus
    dict(type='nfw',    mass=5.4e11, scaleradius=15.62), # halo
    units=Gpot6.units)

test(gpot6, Gpot6, Apot6)
