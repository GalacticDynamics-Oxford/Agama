# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

# some OS-dependent wizardry needed to ensure that the executables are linked
# with properly embedded releative path to the shared library,
# so that they could be moved to and run from any other folder
#UNAME = $(shell uname -s)
ifeq ($(shell uname -s),Darwin)
# on MacOS we need to modify the header of the shared library, which is then automatically used
# to embed the correct relative path into each executable in the exe/ subfolder
SO_FLAGS += -Wl,-install_name,@executable_path/../agama.so
else
# on Linux we need to pass this flag to every compiled executable
# (which are put into exe/ subfolder, hence the relative path to the shared library is ../agama.so)
EXE_FLAGS += -Wl,-rpath,'$$ORIGIN/..'
endif

# set up folder names
SRCDIR    = src
OBJDIR    = obj
EXEDIR    = exe
TESTSDIR  = tests
TORUSDIR  = src/torus

# sources of the main library
SOURCES   = \
            math_core.cpp \
            math_fit.cpp \
            math_geometry.cpp \
            math_linalg.cpp \
            math_ode.cpp \
            math_optimization.cpp \
            math_sample.cpp \
            math_specfunc.cpp \
            math_sphharm.cpp \
            math_spline.cpp \
            particles_io.cpp \
            py_wrapper.cpp \
            actions_genfnc.cpp \
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
            df_interpolated.cpp \
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
            potential_factory.cpp \
            potential_ferrers.cpp \
            potential_galpot.cpp \
            potential_multipole.cpp \
            potential_perfect_ellipsoid.cpp \
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
            amuse_wrapper.cpp \

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
            test_isochrone.cpp \
            test_staeckel.cpp \
            test_action_finder.cpp \
            test_torus.cpp \
            test_df_halo.cpp \
            test_df_spherical.cpp \
            test_density_grid.cpp \
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

$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	@mkdir -p $(EXEDIR)
	$(LINK) -o "$@" "$<" $(CXXFLAGS) $(LIBNAME) $(EXE_FLAGS)

$(TESTEXEFORTRAN):  $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME)
	-$(FC) -o "$@" $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME) $(CXXFLAGS) $(EXE_FLAGS)

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp Makefile.local
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) $(COMPILE_FLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc Makefile.local
	$(CXX) -c $(CXXFLAGS) $(COMPILE_FLAGS) -o "$@" "$<"

clean:
	rm -f $(OBJDIR)/*.o $(EXEDIR)/*.exe $(LIBNAME)

# run tests (both C++ and python)
test:
	(cd py; python alltest.py)

# create Doxygen documentation from in-code comments
doxy:
	doxygen Doxyfile

# if NEMO is present, the library may be used as a plugin providing external potentials within NEMO
nemo:  $(LIBNAME)
ifdef NEMOOBJ
	-cp $(LIBNAME) $(NEMOOBJ)/acc/agama.so
endif

# if AMUSE is installed, compile the plugin and put it into the AMUSE community code folder
ifdef AMUSE_DIR

# standard amuse configuration include (do we need it?):
#-include $(AMUSE_DIR)/config.mk
AMUSE_WORKER_DIR = $(AMUSE_DIR)/src/amuse/community/agama
AMUSE_WORKER     = $(AMUSE_WORKER_DIR)/agama_worker
AMUSE_INTERFACE  = $(AMUSE_WORKER_DIR)/interface.py
MPICXX   ?= mpicxx

amuse: $(AMUSE_INTERFACE) $(AMUSE_WORKER)

$(AMUSE_INTERFACE): py/amuse_interface.py
	@mkdir -p $(AMUSE_WORKER_DIR)
	echo>>$(AMUSE_WORKER_DIR)/__init__.py
	cp py/amuse_interface.py $(AMUSE_INTERFACE)
	cp py/example_amuse.py   $(AMUSE_WORKER_DIR)
	cp py/test_amuse.py      $(AMUSE_WORKER_DIR)

$(AMUSE_WORKER): $(OBJECTS) $(TORUSOBJ) $(AMUSE_INTERFACE)
	$(AMUSE_DIR)/build.py --type=H $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.h"
	$(AMUSE_DIR)/build.py --type=c $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.cpp"
	$(MPICXX) -o "$@" "$(AMUSE_WORKER_DIR)/worker_code.cpp" $(OBJECTS) $(TORUSOBJ) $(LINK_FLAGS) $(CXXFLAGS) $(MUSE_LD_FLAGS)

else
amuse:
endif


.PHONY: clean test lib doxy nemo amuse
