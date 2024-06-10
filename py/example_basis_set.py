#!/usr/bin/python

'''
Example of basis-set potential expansions in agama, galpy and gala.
The latter two libraries implement the Hernquist-Ostriker basis set,
while agama provides a more general Zhao basis set (of which HO is a special case).
We set up a squashed, rotated and shifted Plummer model (so that it has no symmetry at all),
and also sample this analytic density profile with particles.
Then we create an agama BasisSet expansion from both analytic model and particle snapshot;
they do not exactly match because of discreteness noise in the latter case
(it can be reduced by increasing the particle number, but then it would take ages to i
Then we compare the results with the SCF potentials from galpy and gala;
galpy can create the expansion from an analytic density model,
while gala offers both analytic and n-body initialization (but the former one is too slow).
The expansion coefficients computed by either code can be converted into agama-compatible
INI file, from which a native agama BasisSet potential is constructed, and its value and
derivatives are compared to those of the two libraries.
If using the same set of coefficients (i.e. agama potential constructed from foreign coefs),
the results agree to machine precision; the computation of coefficients from an n-body snapshot
also produces identical results in agama and gala, while the initialization from an analytic
density profile in galpy gives slightly different coefs due to integration inaccuracies.
Both construction and evaluation of basis-set potentials is significantly faster in agama;
however, for most practical purposes, the Multipole potential from agama is preferrable
for better performance and accuracy than BasisSet.
'''
import numpy, agama, time
numpy.random.seed(42)

def createPlummer(nbody=100000, rscale=1.2, p=0.8, q=0.6,
    alpha=1.0, beta=2.0, gamma=3.0, offset=numpy.array([0.1,0.2,0.3])):
    # create a triaxial Plummer model with axis ratios y/x=p, z/x=q,
    # rotated w.r.t. principal axes by Euler angles alpha,beta,gamma and shifted from origin by offset
    radius = rscale * (numpy.random.random(size=nbody)**(-2./3) - 1)**-0.5
    costheta = numpy.random.random(size=nbody)*2-1
    sintheta = (1-costheta**2)**0.5
    phi = numpy.random.random(size=nbody) * 2*numpy.pi
    xyz = numpy.column_stack((
        radius*sintheta*numpy.cos(phi),
        radius*sintheta*numpy.sin(phi) * p,
        radius*costheta * q))
    rotmat = agama.makeRotationMatrix(alpha, beta, gamma)
    xyzrot = xyz.dot(rotmat) + offset
    norm = rscale**2 * 3./(4*numpy.pi*p*q)
    def densfnc(xyzrot):
        xyz = (xyzrot-offset).dot(rotmat.T)
        return norm * (rscale**2 + xyz[:,0]**2 + (xyz[:,1]/p)**2 + (xyz[:,2]/q)**2)**-2.5
    return xyzrot, numpy.ones(nbody)/nbody, densfnc

particles, masses, densfnc = createPlummer()
# a small fraction of particle positions are used to compare the potentials
points = particles[:5000]
den_true = densfnc(points)
nmax = 8   # order of radial expansion
lmax = 6   # order of angular expansion
r0   = 1.5 # scale radius of basis functions


print('creating and evaluating agama BasisSet potential')
t0 = time.time()
pot_agama_nbody = agama.Potential(type='BasisSet', nmax=nmax, lmax=lmax, eta=1.0, r0=r0,
    symmetry='none', particles=(particles,masses))
pot_agama_nbody.export('example_basis_set_nbody.ini')
pot_agama_densfnc = agama.Potential(type='BasisSet', nmax=nmax, lmax=lmax, eta=1.0, r0=r0,
    symmetry='none', density=densfnc)
pot_agama_densfnc.export('example_basis_set_densfnc.ini')

Phi_agama_nbody   = pot_agama_nbody.  potential(points)
Phi_agama_densfnc = pot_agama_densfnc.potential(points)
acc_agama_nbody   = pot_agama_nbody  .force(points)
acc_agama_densfnc = pot_agama_densfnc.force(points)
den_agama_nbody   = pot_agama_nbody  .density(points)
den_agama_densfnc = pot_agama_densfnc.density(points)
print('time for constructing and evaluating agama BasisSet potentials: %.3g s' % (time.time()-t0))
print('potential difference between BasisSet potentials initialized from nbody and analytic density: %g' %
    numpy.mean( (Phi_agama_nbody / Phi_agama_densfnc - 1)**2 )**0.5 )
print('acceleration difference between BasisSet potentials initialized from nbody and analytic density: %g' %
    numpy.mean( numpy.sum((acc_agama_nbody-acc_agama_densfnc)**2, axis=1)**0.5 /
        numpy.sum(acc_agama_densfnc**2, axis=1)**0.5 ) )
print('density difference between the true density and BasisSet initialized from analytic density: %g' %
    numpy.mean( (den_agama_densfnc / den_true - 1)**2 )**0.5 )
print('density difference between the true density and BasisSet initialized from nbody snapshot: %g' %
    numpy.mean( (den_agama_nbody / den_true - 1)**2 )**0.5 )

### in addition to BasisSet potentials, we construct and evaluate Multipole potentials
### from both analytic density profile and n-body snapshot
print('creating and evaluating agama Multipole potential')
t0 = time.time()
pot_multipole_nbody = agama.Potential(type='Multipole', lmax=lmax,
    symmetry='none', particles=(particles,masses))
pot_multipole_densfnc = agama.Potential(type='Multipole', lmax=lmax, rmin=0.01,rmax=100,
    symmetry='none', density=densfnc)
print('time for constructing and evaluating agama Multipole potentials: %.3g s' % (time.time()-t0))
print('density difference between the true density and Multipole initialized from analytic density: %g' %
    numpy.mean( (pot_multipole_densfnc.density(points) / den_true - 1)**2 )**0.5 )
print('density difference between the true density and Multipole initialized from nbody snapshot: %g' %
    numpy.mean( (pot_multipole_nbody.density(points) / den_true - 1)**2 )**0.5 )


def convertCoefsToAgamaPotential(r0, Acos, Asin=None, filename='tmppotential.ini'):
    '''
    convert the arrays of cosine and sine coefs from galpy or gala into the agama input format,
    and create the equivalent agama BasisSet potential
    '''
    nmax = Acos.shape[0]-1
    lmax = Acos.shape[1]-1
    assert Acos.shape[2]==lmax+1
    if Asin is None: Asin=numpy.zeros(Acos.shape)
    with open(filename, 'w') as inifile:
        inifile.write('[Potential]\ntype=BasisSet\nnmax=%i\nlmax=%i\nr0=%g\nCoefficients\n#Phi\n#(array)\n' %
            (nmax, lmax, r0))
        for n in range(nmax+1):
            inifile.write(str(n))
            for l in range(lmax+1):
                # first come sine terms in reverse order: m=l, l-1, ..., 1
                for m in range(l):
                    inifile.write('\t%.15g' % (Asin[n,l,l-m] * 0.5**0.5))
                # then come cosine terms in normal order: m=0, 1, ..., l
                for m in range(l+1):
                    inifile.write('\t%.15g' % (Acos[n,l,m] * (0.5**0.5 if m>0 else 1)))
            inifile.write('\n')
    return agama.Potential(filename)

def testGalpy():
    try: import galpy.potential
    except ImportError:
        print('galpy not available, skipping test')
        return
    print('creating galpy scf potential')
    t0=time.time()
    Acos,Asin = galpy.potential.scf_compute_coeffs(
        lambda R,z,phi: densfnc(numpy.array([[R*numpy.cos(phi),R*numpy.sin(phi),z]]))[0],
        N=nmax+1, L=lmax+1, a=r0, phi_order=40)
    pot_galpy_native= galpy.potential.SCFPotential(Acos=Acos, Asin=Asin, a=r0)
    if hasattr(pot_galpy_native, 'phitorque'):  # renamed from phiforce in newer versions
        pot_galpy_native.phiforce = pot_galpy_native.phitorque
    t1=time.time()
    print('evaluating agama scf potential initialized from galpy coefficients')
    Acos[:,:,0] *= 0.5; Asin[:,:,0] *= 0.5  # by some strange convention, m=0 terms are doubled in galpy
    pot_agama_galpy = convertCoefsToAgamaPotential(r0, Acos, Asin, 'example_basis_set_galpy.ini')
    Phi_agama_galpy = pot_agama_galpy.potential(points)
    acc_agama_galpy = pot_agama_galpy.force(points)
    print('evaluating galpy scf potential')
    t2=time.time()
    # need to convert from cylindrical to cartesian coords
    pointscyl = [(points[:,0]**2+points[:,1]**2)**0.5, points[:,2], numpy.arctan2(points[:,1],points[:,0])]
    Phi_galpy_native= pot_galpy_native(*pointscyl)
    acc_galpy_R     = pot_galpy_native.Rforce(*pointscyl)
    acc_galpy_z     = pot_galpy_native.zforce(*pointscyl)
    acc_galpy_phi   = pot_galpy_native.phiforce(*pointscyl)
    acc_galpy_native= numpy.column_stack((
        acc_galpy_R*numpy.cos(pointscyl[2]) - acc_galpy_phi*numpy.sin(pointscyl[2])/pointscyl[0],
        acc_galpy_R*numpy.sin(pointscyl[2]) + acc_galpy_phi*numpy.cos(pointscyl[2])/pointscyl[0],
        acc_galpy_z))
    print('time for constructing galpy scf potential: %.3g s, evaluating it: %.3g s' %
        (t1-t0, time.time()-t2))
    print('potential difference between agama(native) and agama(from galpy): %g' %
        numpy.mean( (Phi_agama_galpy/Phi_agama_densfnc - 1)**2 )**0.5 )
    print('potential difference between galpy(native) and agama(from galpy): %g' %
        numpy.mean( (Phi_agama_galpy/Phi_galpy_native  - 1)**2 )**0.5 )
    print('acceleration difference between agama(native) and agama(from galpy): %g' %
        numpy.mean( numpy.sum((acc_agama_galpy-acc_agama_densfnc)**2, axis=1)**0.5 /
            numpy.sum(acc_agama_nbody**2, axis=1)**0.5 ) )
    print('acceleration difference between galpy(native) and agama(from galpy): %g' %
        numpy.mean( numpy.sum((acc_agama_galpy-acc_galpy_native )**2, axis=1)**0.5 /
            numpy.sum( acc_galpy_native**2, axis=1)**0.5 ) )

def testGala():
    try: import gala.potential
    except ImportError:
        print('gala not available, skipping test')
        return
    print('creating gala scf potential')
    t0=time.time()
    Acos,Asin = gala.potential.scf.compute_coeffs_discrete(
        particles, mass=masses, nmax=nmax, lmax=lmax, r_s=r0)
    # the initialization of gala scf from analytic density is too slow
    #(Acos,_),(Asin,_) = gala.potential.scf.compute_coeffs(
    #    lambda x,y,z: densfnc(numpy.array([[x,y,z]]),
    #    nmax=nmax, lmax=lmax, M=1, r_s=r0, args=(), skip_m=False)
    pot_gala_native = gala.potential.scf.SCFPotential(Snlm=Acos, Tnlm=Asin, m=1, r_s=r0)
    t1=time.time()
    print('evaluating agama scf potential initialized from gala coefficients')
    pot_agama_gala  = convertCoefsToAgamaPotential(r0, Acos, Asin, 'example_basis_set_gala.ini')
    Phi_agama_gala  = pot_agama_gala.potential(points)
    acc_agama_gala  = pot_agama_gala.force(points)
    print('evaluating gala scf potential')
    t2=time.time()
    Phi_gala_native = pot_gala_native.energy(points.T)
    acc_gala_native = pot_gala_native.acceleration(points.T).T
    print('time for constructing gala scf potential: %.3g s, evaluating it: %.3g s' %
        (t1-t0, time.time()-t2))
    print('potential difference between agama(native) and agama(from gala): %g' %
        numpy.mean( (Phi_agama_gala/Phi_agama_nbody - 1)**2 )**0.5 )
    print('potential difference between  gala(native) and agama(from gala): %g' %
        numpy.mean( (Phi_agama_gala/ Phi_gala_native - 1)**2 )**0.5 )
    print('acceleration difference between agama(native) and agama(from gala): %g' %
        numpy.mean( numpy.sum((acc_agama_gala-acc_agama_nbody)**2, axis=1)**0.5 /
            numpy.sum(acc_agama_nbody**2, axis=1)**0.5 ) )
    print('acceleration difference between  gala(native) and agama(from gala): %g' %
        numpy.mean( numpy.sum((acc_agama_gala- acc_gala_native)**2, axis=1)**0.5 /
            numpy.sum( acc_gala_native**2, axis=1)**0.5 ) )

testGalpy()
testGala()
