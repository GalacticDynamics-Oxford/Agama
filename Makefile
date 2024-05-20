# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

# this file is shared between Makefile (Linux/MacOS) and Makefile.msvc (Windows)
# and contains the list of source files and folders
include Makefile.list

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
# (if this was an actual file and not a symlink, then delete it first and then create a symlink)
$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	@mkdir -p $(EXEDIR)
	@[ -f $(EXEDIR)/$(LIBNAME) -a ! -L $(EXEDIR)/$(LIBNAME) ] && rm $(EXEDIR)/$(LIBNAME) || true
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
	rm -f $(OBJDIR)/*.o $(OBJDIR)/*.d $(EXEDIR)/*.exe $(LIBNAME) $(EXEDIR)/$(LIBNAME)

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

$(AMUSE_WORKER): py/interface_amuse.py $(SRCDIR)/interface_amuse.cpp $(OBJECTS) $(TORUSOBJ) $(AMUSE_WORKER_INIT)
	cp py/interface_amuse.py $(AMUSE_INTERFACE)
	cp py/example_amuse.py   $(AMUSE_WORKER_DIR)
	cp py/test_amuse.py      $(AMUSE_WORKER_DIR)
	-$(AMUSE_DIR)/build.py --type=H $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.h"
	-$(AMUSE_DIR)/build.py --type=c $(AMUSE_INTERFACE) AgamaInterface -o "$(AMUSE_WORKER_DIR)/worker_code.cpp"
	-$(MPICXX) -o "$@" "$(AMUSE_WORKER_DIR)/worker_code.cpp" $(SRCDIR)/interface_amuse.cpp $(OBJECTS) $(TORUSOBJ) $(LINK_FLAGS) $(CXXFLAGS) $(MUSE_LD_FLAGS)

else
amuse:
endif

# auto-dependency tracker (works with GCC-compatible compilers?)
DEPENDS = $(patsubst %.cpp,$(OBJDIR)/%.d,$(SOURCES))
COMPILE_FLAGS += -MMD -MP
-include $(DEPENDS)

.PHONY: clean test lib doxy nemo amuse
