C  Demonstrates how to create and use potentials from AGAMA library in FORTRAN.
C  There are several possible ways:
C  (1)  loading the parameters from an INI file (one or several components);
C  (2)  passing the parameters for one component as a single string;
C  (3)  providing a FORTRAN routine that returns a density at a given point,
C       and using it to create a potential approximation with the parameters
C       provided in a text string;
C  (4)  providing a FORTRAN routine that returns potential and force at a given point,
C       and creating a potential approximation for it in the same way as above.
C  Due to the absense of a native pointer type in FORTRAN, the pointer to the C++ object
C  should be stored in a placeholder variable of type CHAR*8, which is passed
C  as the first argument to all functions in this module.

      program example
      implicit none
C  This is not an actual string, but a placeholder to keep the pointer to the C++ object
      character*8 c_obj1, c_obj2, c_obj3, c_obj4, c_obj5
C  Functions provided by the AGAMA library
      double precision agama_potential, agama_potforce,
     &    agama_potforcederiv, agama_density
C  User-defined density and potential functions
      double precision user_density, user_potential
C  It is necessary to declare them as functions, not variables
      external user_density, user_potential
C  Local variables
      double precision xyz(3), pot0, pot1, pot2, den0, den1, den2,
     &    force0(3), force1(3), force2(3), deriv(6)
      logical success
      success = .true.

C  Example 1:  constructing a potential from parameters provided in a single string;
C  the first parameter is always the placeholder for the pointer to C++ object,
C  the second one is a string encoding all parameters.
      call agama_initfromparam(c_obj1,
     &    'type=Spheroid axisRatioZ=0.5 gamma=0.5 beta=5.0 '//
     &    'scaleRadius=2.5 densityNorm=1')

C  Example 2:  constructing a potential approximation from the user-provided
C  density profile (which is equivalent to the one specified above, but uses
C  somewhat different parameters for the potential expansion)
      print*, 'Create a potential from the user-provided density'
      call agama_initfromdens(c_obj2,
c     &    'type=Multipole symmetry=Axisymmetric',
     &    'type=CylSpline symmetry=Axisym',
     &    user_density)

C  Compute potential and density at some location
      xyz(1)= 0.9d0
      xyz(2)= 0.6d0
      xyz(3)= 0.3d0
      pot1  = agama_potential(c_obj1, xyz)
      den1  = agama_density  (c_obj1, xyz)
      pot2  = agama_potential(c_obj2, xyz)
      den2  = agama_density  (c_obj2, xyz)
      den0  = user_density(xyz)
      print*, 'Position(x,y,z)=', xyz
      print*, 'Multipole potential=', pot1
      print*, 'Multipole   density=', den1
      print*, 'CylSpline potential=', pot2
      print*, 'CylSpline   density=', den2
      print*, 'original    density=', den0

C  Check the accuracy
      if(abs(den1-den0) + abs(den2-den0) > den0 * 1.d-3) then
          print*, '**FAILED**'
          success = .false.
      endif

C  Compute force and force derivatives (potential is also returned)
      pot1 = agama_potforce(c_obj1, xyz, force1)
      print*, 'Force=', force1
      pot2 = agama_potforcederiv(c_obj2, xyz, force2, deriv)
      print*, 'Force derivs: dFx/dx=', deriv(1), 'dFy/dy=', deriv(2),
     &    'dFz/dz=', deriv(3), 'dFx/dy=', deriv(4),
     &    'dFx/dz=', deriv(5), 'dFy/dz=', deriv(6)

C  Example 3:  construct two different potential approximations from
C  the user-provided function that computes potential and force
C  (useful if this is an expensive operation)
      print*, 'Create approximations for the user-provided potential'
      call agama_initfrompot(c_obj3,
     &    'type=Multipole symmetry=Triaxial lmax=8 rmin=0.01 rmax=100',
     &     user_potential)
      call agama_initfrompot(c_obj4,
     &    'type=CylSpline, symmetry=Triaxial, rmin=0.01, rmax=100',
     &     user_potential)
      pot1 = agama_potforce(c_obj3, xyz, force1)
      print*, 'Multipole potential=', pot1, 'force=', force1
      pot2 = agama_potforce(c_obj4, xyz, force2)
      print*, 'CylSpline potential=', pot2, 'force=', force2
      pot0 = user_potential(xyz, force0)
      print*, 'Original  potential=', pot0, 'force=', force0

C  check the accuracy
      if(abs(pot1-pot0) + abs(pot2-pot0) > abs(pot0)      * 1.d-4
     &    .or.  abs(force1(1)-force0(1)) > abs(force0(1)) * 1.d-3
     &    .or.  abs(force2(1)-force0(1)) > abs(force0(1)) * 1.d-3
     &    .or.  abs(force1(2)-force0(2)) > abs(force0(2)) * 1.d-3
     &    .or.  abs(force2(2)-force0(2)) > abs(force0(2)) * 1.d-3
     &    .or.  abs(force1(3)-force0(3)) > abs(force0(3)) * 1.d-3
     &    .or.  abs(force2(3)-force0(3)) > abs(force0(3)) * 1.d-3) then
          print*, '**FAILED**'
          success = .false.
      endif

C  Example 4:  constructing a potential from parameters stored in an INI file
      call agama_initfromfile(c_obj5, '../data/BT08.ini')
      print*, 'Potential=', agama_potential(c_obj5, xyz)
      print*, 'Density=', agama_density(c_obj5, xyz)

      if(success) then
          print*, 'ALL TESTS PASSED'
      else
          print*, 'SOME TESTS FAILED'
      endif
      end program example


C  This is the function that provides the user-defined density profile;
C  in this example it is an axisymmetric double-power-law:
C  rho = r**-alpha * (1+r)**(alpha-beta), where r**2 = x**2+y**2+(z/q)**2
      function user_density(x)
      implicit none
      double precision user_density, x(3), rs
c     parameters of density profile
      double precision inner_exp, outer_exp, scale_radius, q
      data inner_exp/0.5/, outer_exp/5.0/, scale_radius/2.5/, q/0.5/

      rs = dsqrt(x(1)**2 + x(2)**2 + (x(3)/q)**2) / scale_radius
      user_density = rs**(-inner_exp) * (1+rs)**(inner_exp-outer_exp)
      end


C  This is the function that provides the user-defined potential profile;
C  in this example it is a triaxial plummer-like potential
C  with constant flattening of equipotential surface
      function user_potential(x, force)
      implicit none
      double precision user_potential, x(3), force(3), rs, q, p
      data q/0.8/, p/0.6/

      rs = dsqrt(1 + x(1)**2 + (x(2)/q)**2 + (x(3)/p)**2)
      user_potential = -1. / rs
      force(1) = -x(1) / rs**3
      force(2) = -x(2) / rs**3 / q**2
      force(3) = -x(3) / rs**3 / p**2
      end
