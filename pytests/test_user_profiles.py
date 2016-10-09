#!/usr/bin/python

"""
Demonstrates the use of user-defined Python routines for density and distribution function.
Note that whenever such objects are used, this turns off OpenMP parallelization in the C++ library,
because the Python callback functions cannot be used in a multi-threaded context;
however, these callback functions are designed to handle possibly many points simultaneously
using numpy vectorized math operations, which improves efficiency.
"""
import agama, numpy

# user-defined density profile should be a function with a single argument --
# a 2d array Mx3, evaluating the density simultaneously at M points,
# i.e., it should operate with columns of the input array x[:,0], x[:,1], etc.
# To create a user function with arbitrary internal constants
# (such as mass, scale radius, etc.) which are not fixed a priori,
# we use a 'factory routine' that takes these internal parameters as arguments,
# and creates an anonymous (lambda) function with these parameters built-in.
def makeUserDensity(mass, radius):
    # we use column-wise sum (along axis 1) to obtain r^2 = x[0]^2+x[1]^2+x[2]^2
    return lambda x: \
        3 / (4*numpy.pi) * mass * radius**2 * \
        (numpy.sum(x**2, axis=1) + radius**2) ** -2.5

# define the parameters of the density profile
mass = 3.0
radius = 1.5

# original density/potential model using a C++ object
pot_orig = agama.Potential(type="Plummer", mass=mass, scaleradius=radius)

# potential approximation computed from the user-supplied density profile
MyPlummer = makeUserDensity(mass, radius)
pot_appr = agama.Potential(type="Multipole", density=MyPlummer)

pot0_orig = pot_orig.potential(0,0,0)
pot0_appr = pot_appr.potential(0,0,0)
print "Phi_appr(0)=%.8g  (true value=%.8g)" % (pot0_appr, pot0_orig)
print "rho_appr(1)=%.8g  (true value=%.8g,  user value=%.8g)" % \
    ( pot_appr.density(1,0,0), pot_orig.density(1,0,0), MyPlummer(numpy.array([[1,0,0]])) )

# user-defined distribution function, again it must be a function of a single argument --
# a 2d array Mx3, where the columns are Jr,Jz,Jphi, and rows are M independent points
# in action space where the function should be evaluated simultaneously.
# Here, instead of creating a lambda function, we fix the internal constants at the outset.
J0   = 2.0    # constants -- parameters of the DF
Beta = 5.0
def MyDF(J):
    return (J[:,0] + J[:,1] + abs(J[:,2]) + J0) ** -Beta

# original DF using a C++ object
df_orig = agama.DistributionFunction( \
    type="DoublePowerLaw", J0=J0, slopeIn=0, slopeOut=Beta, norm=2*numpy.pi**3)

# to compute the total mass, we create an proxy instance of DistributionFunction class
# with a composite DF consisting of a single component (the user-supplied function);
# this proxy class provides the totalMass() method
# that the user-defined Python function itself does not have
mass_orig = df_orig.totalMass()
mass_my = agama.DistributionFunction(MyDF).totalMass()
print "DF mass=%.8g  (orig value=%.8g)" % (mass_my, mass_orig)

# GalaxyModel objects constructed from the C++ DF and from the Python function
# (without the need of a proxy object; in fact, GalaxyModel.df is itself a proxy object)
gm_orig = agama.GalaxyModel(df=df_orig, pot=pot_orig)
gm_my   = agama.GalaxyModel(df=MyDF, pot=pot_appr)
# note that different potentials were used for gm_orig and gm_my, so the results may slightly disagree

# DF moments (density and velocity dispersion) computed from the C++ DF object
dens_orig, veldisp_orig = gm_orig.moments(point=(1,0,0))
print "original DF at r=1: density=%.8g, sigma_r=%.8g, sigma_t=%.8g" % \
    ( dens_orig, veldisp_orig[0], veldisp_orig[1] )

# DF moments computed from the Python distribution function
dens_my, veldisp_my = gm_my.moments(point=(1,0,0))
print "user-def DF at r=1: density=%.8g, sigma_r=%.8g, sigma_t=%.8g" % \
    ( dens_my, veldisp_my[0], veldisp_my[1] )
# gm_my.df.totalMass()  will give the same result as mass_my

if abs(pot0_orig-pot0_appr)<1e-6 and abs(mass_orig-mass_my)<1e-6 and abs(dens_orig-dens_my)<1e-6:
    print "\033[1;32mALL TESTS PASSED\033[0m"
else:
    print "\033[1;31mSOME TESTS FAILED\033[0m"
