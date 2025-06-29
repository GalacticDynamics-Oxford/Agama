project('agama', 'cpp', 'c',
  version: '1.0.152',
  default_options: [
    'cpp_std=c++11',
    'optimization=2',
    'buildtype=release',
    'cpp_eh=sc',
  ]
)

# Import required modules
fs = import('fs')

# Find Python and NumPy
python = import('python').find_installation(pure: false)

# Get NumPy include directory
numpy_result = run_command(python, '-c', 'import numpy; print(numpy.get_include())', check: true)
numpy_inc = numpy_result.stdout().strip()
numpy_dep = declare_dependency(include_directories: include_directories(numpy_inc))

# Get build options
enable_gsl = get_option('gsl')
enable_eigen = get_option('eigen')
enable_openmp = get_option('openmp')
enable_cvxopt = get_option('cvxopt')

# Find dependencies - fail if enabled but not found
gsl_dep = dependency('gsl', required: enable_gsl)
openmp_dep = dependency('openmp', required: enable_openmp)
eigen_dep = dependency('eigen3', required: enable_eigen)

# Include directories
include_dirs = ['src', 'src/torus']

if enable_cvxopt
  include_dirs += ['extras/cvxopt/include']
endif

inc_dirs = include_directories(include_dirs)

# Compiler setup
cpp = meson.get_compiler('cpp')

# Define source files
agama_sources = files([
  'src/actions_factory.cpp',
  'src/actions_focal_distance_finder.cpp',
  'src/actions_isochrone.cpp',
  'src/actions_spherical.cpp',
  'src/actions_staeckel.cpp',
  'src/actions_torus.cpp',
  'src/coord.cpp',
  'src/cubature.cpp',
  'src/df_base.cpp',
  'src/df_disk.cpp',
  'src/df_factory.cpp',
  'src/df_halo.cpp',
  'src/df_spherical.cpp',
  'src/galaxymodel_base.cpp',
  'src/galaxymodel_densitygrid.cpp',
  'src/galaxymodel_fokkerplanck.cpp',
  'src/galaxymodel_jeans.cpp',
  'src/galaxymodel_losvd.cpp',
  'src/galaxymodel_selfconsistent.cpp',
  'src/galaxymodel_spherical.cpp',
  'src/galaxymodel_velocitysampler.cpp',
  'src/interface_amuse.cpp',
  'src/interface_c.cpp',
  'src/interface_fortran.cpp', 
  'src/interface_nemo.cpp',
  'src/interface_python.cpp',
  'src/math_core.cpp',
  'src/math_fit.cpp',
  'src/math_gausshermite.cpp',
  'src/math_geometry.cpp',
  'src/math_linalg.cpp',
  'src/math_ode.cpp',
  'src/math_optimization.cpp',
  'src/math_random.cpp',
  'src/math_sample.cpp',
  'src/math_specfunc.cpp',
  'src/math_sphharm.cpp',
  'src/math_spline.cpp',
  'src/orbit.cpp',
  'src/orbit_variational.cpp',
  'src/particles_io.cpp',
  'src/potential_analytic.cpp',
  'src/potential_base.cpp',
  'src/potential_composite.cpp',
  'src/potential_cylspline.cpp',
  'src/potential_dehnen.cpp',
  'src/potential_disk.cpp',
  'src/potential_factory.cpp',
  'src/potential_ferrers.cpp',
  'src/potential_king.cpp',
  'src/potential_multipole.cpp',
  'src/potential_perfect_ellipsoid.cpp',
  'src/potential_spheroid.cpp',
  'src/potential_utils.cpp',
  'src/raga_binary.cpp',
  'src/raga_core.cpp',
  'src/raga_losscone.cpp',
  'src/raga_potential.cpp',
  'src/raga_relaxation.cpp',
  'src/raga_trajectory.cpp',
  'src/utils.cpp',
  'src/utils_config.cpp'
])

# Torus library sources
torus_sources = files([
  'src/torus/CHB.cc',
  'src/torus/Fit.cc',
  'src/torus/Fit2.cc',
  'src/torus/GeneratingFunction.cc',
  'src/torus/Orb.cc',
  'src/torus/PJMNum.cc',
  'src/torus/Point_ClosedOrbitCheby.cc',
  'src/torus/Point_None.cc',
  'src/torus/Torus.cc',
  'src/torus/Toy_Isochrone.cc',
  'src/torus/WD_Numerics.cc'
])

# Collect all dependencies
all_deps = [python.dependency(), numpy_dep]

# Preprocessor definitions
cpp_args = [
  '-DHAVE_PYTHON',
  '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
  '-DVERSION_INFO="' + meson.project_version() + '"'
]

# Add optional dependencies
if gsl_dep.found()
  all_deps += gsl_dep
  cpp_args += '-DHAVE_GSL'
endif

if eigen_dep.found()
  all_deps += eigen_dep
  cpp_args += '-DHAVE_EIGEN'
endif

if openmp_dep.found()
  all_deps += openmp_dep
endif

if enable_cvxopt
  cpp_args += '-DHAVE_CVXOPT'
endif

# Create the Python extension module
agama_module = python.extension_module(
  '_agama',
  sources: [agama_sources, torus_sources],
  include_directories: inc_dirs,
  dependencies: all_deps,
  cpp_args: cpp_args,
  install: true,
  subdir: 'agama'
)

# Install Python files
python.install_sources(
  files('agama/__init__.py', 'agama/pygama.py', 'agama/schwarzlib.py', 'agama/agamacolormaps.py', 'agama/nemofile.py', 'agama/interface_amuse.py'),
  subdir: 'agama'
)

# Install examples (optional, can be disabled with -Dexamples=false)
examples_option = get_option('examples')
if examples_option
  install_subdir('examples', install_dir: python.get_install_dir() / 'agama')
endif
