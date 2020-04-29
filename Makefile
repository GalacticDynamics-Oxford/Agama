# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

# some OS-dependent wizardry needed to ensure that the executables are linked to
# the shared library using a relative path, so that it will be looked for in the same
# folder as the executables (a symlink is created as exe/agama.so -> agama.so).
# This allows the executables to be copied/moved to and run from any other folder
# without the need to put agama.so into a system-wide folder such as /usr/local/lib,
# or add it to LD_LIBRARY_PATH, provided that a copy of the shared library or a symlink
# to it resides in the same folder as the executable program.
ifeq ($(shell uname -s),Darwin)
# on MacOS we need to modify the header of the shared library, which is then
# automatically used to embed the correct relative path into each executable
SO_FLAGS += -Wl,-install_name,@executable_path/agama.so
else
# on Linux we need to pass this flag to every compiled executable in the exe/ subfolder
EXE_FLAGS += -Wl,-rpath,'$$ORIGIN'
endif

# set up folder names
SRCDIR    = src
OBJDIR    = obj
EXEDIR    = exe
TESTSDIR  = tests
TORUSDIR  = src/torus

# sources of the main library
SOURCES   = \
            py_wrapper.cpp \
            math_core.cpp \
            math_fit.cpp \
            math_gausshermite.cpp \
            math_geometry.cpp \
            math_linalg.cpp \
            math_ode.cpp \
            math_optimization.cpp \
            math_random.cpp \
            math_sample.cpp \
            math_specfunc.cpp \
            math_sphharm.cpp \
            math_spline.cpp \
            particles_io.cpp \
            actions_focal_distance_finder.cpp \
            actions_isochrone.cpp \
            actions_spherical.cpp \
            actions_staeckel.cpp \
            actions_torus.cpp \
            coord.cpp \
            cubature.cpp \
            df_base.cpp \
            df_disk.cpp \
            df_factory.cpp \
            df_halo.cpp \
            df_spherical.cpp \
            galaxymodel_base.cpp \
            galaxymodel_densitygrid.cpp \
            galaxymodel_fokkerplanck.cpp \
            galaxymodel_jeans.cpp \
            galaxymodel_losvd.cpp \
            galaxymodel_selfconsistent.cpp \
            galaxymodel_spherical.cpp \
            galaxymodel_velocitysampler.cpp \
            orbit.cpp \
            orbit_lyapunov.cpp \
            potential_analytic.cpp \
            potential_base.cpp \
            potential_composite.cpp \
            potential_cylspline.cpp \
            potential_dehnen.cpp \
            potential_disk.cpp \
            potential_factory.cpp \
            potential_ferrers.cpp \
            potential_king.cpp \
            potential_multipole.cpp \
            potential_perfect_ellipsoid.cpp \
            potential_spheroid.cpp \
            potential_utils.cpp \
            raga_base.cpp  \
            raga_core.cpp   \
            raga_binary.cpp  \
            raga_losscone.cpp \
            raga_potential.cpp \
            raga_relaxation.cpp \
            raga_trajectory.cpp  \
            utils.cpp \
            utils_config.cpp \
            fortran_wrapper.cpp \
            nemo_wrapper.cpp \

# ancient Torus code
TORUSSRC  = CHB.cc \
            Fit.cc \
            Fit2.cc \
            GeneratingFunction.cc \
            Orb.cc \
            PJMCoords.cc \
            PJMNum.cc \
            Point_ClosedOrbitCheby.cc \
            Point_None.cc \
            Torus.cc \
            Toy_Isochrone.cc \
            WD_Numerics.cc \

# test and example programs
TESTSRCS  = test_math_core.cpp \
            test_math_linalg.cpp \
            test_math_spline.cpp \
            test_coord.cpp \
            test_units.cpp \
            test_utils.cpp \
            test_orbit_integr.cpp \
            test_orbit_lyapunov.cpp \
            test_potentials.cpp \
            test_potential_expansions.cpp \
            test_actions_isochrone.cpp \
            test_actions_spherical.cpp \
            test_actions_staeckel.cpp \
            test_actions_torus.cpp \
            test_action_finder.cpp \
            test_df_halo.cpp \
            test_df_spherical.cpp \
            test_density_grid.cpp \
            test_losvd.cpp \
            example_actions_nbody.cpp \
            example_df_fit.cpp \
            example_doublepowerlaw.cpp \
            example_self_consistent_model.cpp \
            mkspherical.cpp \
            phaseflow.cpp \
            raga.cpp \

TESTFORTRAN = test_fortran.f

LIBNAME  = agama.so
OBJECTS  = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
TESTEXE  = $(patsubst %.cpp,$(EXEDIR)/%.exe,$(TESTSRCS))
TORUSOBJ = $(patsubst %.cc,$(OBJDIR)/%.o,$(TORUSSRC))
TESTEXEFORTRAN = $(patsubst %.f,$(EXEDIR)/%.exe,$(TESTFORTRAN))
CXXFLAGS += -I$(SRCDIR)

# this is the default target (build all), if make is launched without parameters
all:  $(LIBNAME) $(TESTEXE) $(TESTEXEFORTRAN) nemo amuse

# one may recompile just the shared library by running 'make lib'
lib:  $(LIBNAME)

$(LIBNAME):  $(OBJECTS) $(TORUSOBJ) Makefile Makefile.local
	$(LINK) -shared -o $(LIBNAME) $(OBJECTS) $(TORUSOBJ) $(LINK_FLAGS) $(CXXFLAGS) $(SO_FLAGS)

# for each executable file, first make sure that the exe/ folder exists,
# and create a symlink named agama.so pointing to ../agama.so in that folder if needed
$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	@mkdir -p $(EXEDIR)
	@[ -L $(EXEDIR)/$(LIBNAME) ] || ln -s ../$(LIBNAME) $(EXEDIR)/$(LIBNAME)
	$(LINK) -o "$@" "$<" $(CXXFLAGS) $(LIBNAME) $(EXE_FLAGS)

$(TESTEXEFORTRAN):  $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME)
	-$(FC) -o "$@" $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME) $(CXXFLAGS) $(EXE_FLAGS)

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp Makefile.local
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) $(COMPILE_FLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc Makefile.local
	$(CXX) -c $(CXXFLAGS) $(COMPILE_FLAGS) -o "$@" "$<"

clean:
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d $(EXEDIR)/*.exe $(LIBNAME)

# run tests (both C++ and python)
test:
	(cd py; python alltest.py)

# create Doxygen documentation from in-code comments
doxy:
	doxygen Doxyfile

# if NEMO is present, the library may be used as a plugin providing external potentials within NEMO
ifdef NEMOOBJ
NEMO_PLUGIN = $(NEMOOBJ)/acc/agama.so

nemo: $(NEMO_PLUGIN)

$(NEMO_PLUGIN): $(LIBNAME)
	-cp $(LIBNAME) $(NEMO_PLUGIN)
else
nemo:
endif

# if AMUSE is installed, compile the plugin and put it into the AMUSE community code folder
ifdef AMUSE_DIR

# standard amuse configuration include (do we need it?):
#-include $(AMUSE_DIR)/config.mk
AMUSE_WORKER_DIR = $(AMUSE_DIR)/src/amuse/community/agama
AMUSE_WORKER     = $(AMUSE_WORKER_DIR)/agama_worker
AMUSE_INTERFACE  = $(AMUSE_WORKER_DIR)/interface.py
AMUSE_WORKER_INIT= $(AMUSE_WORKER_DIR)/__init__.py
MPICXX   ?= mpicxx

amuse: $(AMUSE_WORKER)

$(AMUSE_WORKER_INIT):
	@mkdir -p $(AMUSE_WORKER_DIR)
	echo>>$(AMUSE_WORKER_DIR)/__init__.py

$(AMUSE_WORKER): py/amuse_interface.py $(SRCDIR)/amuse_wrapper.cpp $(OBJECTS) $(TORUSOBJ) $(AMUSE_WORKER_INIT)
	cp py/amuse_interface.py $(AMUSE_INTERFACE)
	cp py/example_amuse.py   $(AMUSE_WORKER_DIR)
	cp py/test_amuse.py      $(AMUSE_WORKER_DIR)
	$(AMUSE_DIR)/build.py --type=H $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.h"
	$(AMUSE_DIR)/build.py --type=c $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.cpp"
	$(MPICXX) -o "$@" "$(AMUSE_WORKER_DIR)/worker_code.cpp" $(SRCDIR)/amuse_wrapper.cpp $(OBJECTS) $(TORUSOBJ) $(LINK_FLAGS) $(CXXFLAGS) $(MUSE_LD_FLAGS)

else
amuse:
endif

# auto-dependency tracker (works with GCC-compatible compilers?)
DEPENDS = $(patsubst %.cpp,$(OBJDIR)/%.d,$(SOURCES))
COMPILE_FLAGS += -MMD -MP
-include $(DEPENDS)

.PHONY: clean test lib doxy nemo amuse
