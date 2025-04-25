# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

# this file is shared between Makefile (Linux/MacOS) and Makefile.msvc (Windows)
# and contains the list of source files and folders
include Makefile.list

LIBNAME_SHARED = agama.so
LIBNAME_STATIC = agama.a
OBJECTS  = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
TESTEXE  = $(patsubst %.cpp,$(EXEDIR)/%.exe,$(TESTSRCS))
TORUSOBJ = $(patsubst %.cc,$(OBJDIR)/%.o,$(TORUSSRC))
TESTEXEFORTRAN = $(patsubst %.f,$(EXEDIR)/%.exe,$(TESTFORTRAN))
COMPILE_FLAGS_ALL += -I$(SRCDIR)

# this is the default target (build all), if make is launched without parameters
all:  lib $(TESTEXE) $(TESTEXEFORTRAN) nemo amuse

# one may recompile just the shared and static versions of the library by running 'make lib'
lib:  $(LIBNAME_STATIC) $(LIBNAME_SHARED)

$(LIBNAME_STATIC):  $(OBJECTS) $(TORUSOBJ) Makefile Makefile.local Makefile.list
	$(AR) ru $(LIBNAME_STATIC) $(OBJECTS) $(TORUSOBJ)

$(LIBNAME_SHARED):  $(OBJECTS) $(TORUSOBJ) Makefile Makefile.local Makefile.list
	$(LINK) -shared -o $(LIBNAME_SHARED) $(OBJECTS) $(TORUSOBJ) $(LINK_FLAGS_ALL) $(LINK_FLAGS_LIB) $(LINK_FLAGS_LIB_AND_EXE_STATIC)

# two possible choices for linking the executable programs:
# 1. shared (default) uses the shared library agama.so, which makes the overall code size smaller,
# but requires that the library is present in the same folder as the executable files.
# 2. static (turned on by declaring an environment variable AGAMA_STATIC) uses the static library libagama.a
# together with all other third-party libraries (libgsl.a, libgslcblas.a, possibly nemo, unsio, glpk, etc.)
# but notably _excluding_ python, since it is used only in two places:
# (a) the python extension module (which is the shared library itself), and
# (b) in math::quadraticOptimizationSolve when compiled with HAVE_CVXOPT,
# but the latter function is not used by any other part of the library or test/example programs,
# except the "solveOpt" function provided by the Python interface (again).
# So it is safe to ignore python in static linking of C++/C/Fortran programs.

ifndef AGAMA_STATIC
# for each executable file, first make sure that the exe/ folder exists,
# and create a symlink named agama.so pointing to ../agama.so in that folder if needed
# (if this was an actual file and not a symlink, then delete it first and then create a symlink)
$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME_SHARED)
	@mkdir -p $(EXEDIR)
	@[ -f $(EXEDIR)/$(LIBNAME_SHARED) -a ! -L $(EXEDIR)/$(LIBNAME_SHARED) ] && rm $(EXEDIR)/$(LIBNAME_SHARED) || true
	@[ -L $(EXEDIR)/$(LIBNAME_SHARED) ] || ln -s ../$(LIBNAME_SHARED) $(EXEDIR)/$(LIBNAME_SHARED)
	$(LINK) -o "$@" "$<" $(COMPILE_FLAGS_ALL) $(LIBNAME_SHARED) $(LINK_FLAGS_ALL) $(LINK_FLAGS_EXE_SHARED)
$(TESTEXEFORTRAN):  $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME_SHARED)
	-$(FC) -o "$@" $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME_SHARED) $(LINK_FLAGS_ALL) $(LINK_FLAGS_EXE_SHARED)
else
$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME_STATIC)
	@mkdir -p $(EXEDIR)
	$(LINK) -o "$@" "$<" $(COMPILE_FLAGS_ALL) $(LIBNAME_STATIC) $(LINK_FLAGS_ALL) $(LINK_FLAGS_LIB_AND_EXE_STATIC)
$(TESTEXEFORTRAN):  $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME_STATIC)
	-$(FC) -o "$@" $(TESTSDIR)/$(TESTFORTRAN) $(LIBNAME_STATIC) $(LINK_FLAGS_ALL) $(LINK_FLAGS_LIB_AND_EXE_STATIC) $(LINK_FLAGS_EXE_STATIC_NONCPP)
endif


$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp Makefile.local
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(COMPILE_FLAGS_ALL) $(COMPILE_FLAGS_LIB) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc Makefile.local
	$(CXX) -c $(COMPILE_FLAGS_ALL) $(COMPILE_FLAGS_LIB) -o "$@" "$<"

clean:
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d $(EXEDIR)/*.exe $(LIBNAME_SHARED) $(EXEDIR)/$(LIBNAME_SHARED) $(LIBNAME_STATIC)

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

$(NEMO_PLUGIN): $(LIBNAME_SHARED)
	-cp $(LIBNAME_SHARED) $(NEMO_PLUGIN)
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

$(AMUSE_WORKER): py/interface_amuse.py $(SRCDIR)/interface_amuse.cpp $(OBJECTS) $(TORUSOBJ) $(AMUSE_WORKER_INIT)
	cp py/interface_amuse.py $(AMUSE_INTERFACE)
	cp py/example_amuse.py   $(AMUSE_WORKER_DIR)
	cp py/test_amuse.py      $(AMUSE_WORKER_DIR)
	-$(AMUSE_DIR)/bin/amusifier --type=H $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.h"
	-$(AMUSE_DIR)/bin/amusifier --type=c $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.cpp"
	-$(MPICXX) -o "$@" "$(AMUSE_WORKER_DIR)/worker_code.cpp" $(SRCDIR)/interface_amuse.cpp $(COMPILE_FLAGS_ALL) $(LIBNAME_STATIC) $(LINK_FLAGS_ALL) $(LINK_FLAGS_LIB_AND_EXE_STATIC) $(MUSE_LD_FLAGS)

else
amuse:
endif

# auto-dependency tracker (works with GCC-compatible compilers?)
DEPENDS = $(patsubst %.cpp,$(OBJDIR)/%.d,$(SOURCES))
COMPILE_FLAGS_LIB += -MMD -MP
-include $(DEPENDS)

.PHONY: clean test lib doxy nemo amuse
