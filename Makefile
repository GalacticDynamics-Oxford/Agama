# this file is for general settings such as file list, etc.
# machine-specific settings such as include paths and #defines are in Makefile.local
include Makefile.local

SRCDIR    = src
OBJDIR    = obj
LIBDIR    = lib
EXEDIR    = exe
TESTSDIR  = tests
LEGACYDIR = src/legacy
TORUSDIR  = src/torus

# sources of the main library
SOURCES   = coord.cpp \
            potential_base.cpp \
            potential_factory.cpp \
            potential_analytic.cpp \
            potential_composite.cpp \
            potential_cylspline.cpp \
            potential_dehnen.cpp \
            potential_ferrers.cpp \
            potential_galpot.cpp \
            potential_perfect_ellipsoid.cpp \
            potential_sphharm.cpp \
            particles_io.cpp \
            actions_interfocal_distance_finder.cpp \
            actions_staeckel.cpp \
            actions_torus.cpp \
            math_core.cpp \
            math_ode.cpp \
            math_spline.cpp \
            math_specfunc.cpp \
            cubature.cpp \
            orbit.cpp \
            utils.cpp \
            utils_config.cpp \
            WD_Numerics.cpp

# ancient code
#LEGACYSRC = Stackel_JS.cpp \
#            coordsys.cpp

# ancient Torus code
TORUSSRC  = CHB.cc \
            Fit.cc \
            GeneratingFunction.cc \
            Orb.cc \
            PJMCoords.cc \
            PJMNum.cc \
            Point_ClosedOrbitCheby.cc \
            Point_None.cc \
            Torus.cc \
            Toy_Isochrone.cc

# test programs
TESTSRCS  = test_math_core.cpp \
            test_math_spline.cpp \
            test_coord.cpp \
            test_units.cpp \
            test_orbit_integr.cpp \
            test_potentials.cpp \
            test_potential_sphharm.cpp \
            test_staeckel.cpp \
            test_actionfinder.cpp \
            test_actions_nbody.cpp \
            test_torus.cpp \

LIBNAME   = $(LIBDIR)/libfJ.a
PY_WRAPPER= $(LIBDIR)/py_wrapper.so

HEADERS   = $(SOURCES:.cpp=.h)
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES)) 
TESTEXE   = $(patsubst %.cpp,$(EXEDIR)/%.exe,$(TESTSRCS))
LEGACYOBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LEGACYSRC)) 
TORUSOBJ  = $(patsubst %.cc, $(OBJDIR)/%.o,$(TORUSSRC)) 

all:      $(LIBNAME) $(TESTEXE) $(PY_WRAPPER)

$(LIBNAME):  $(OBJECTS) $(LEGACYOBJ) $(TORUSOBJ) Makefile Makefile.local
	@mkdir -p $(LIBDIR)
	ar rv $(LIBNAME) $(OBJECTS) $(LEGACYOBJ) $(TORUSOBJ)

$(EXEDIR)/%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	@mkdir -p $(EXEDIR)
	$(CXX) -o "$@" "$<" $(CXXFLAGS) $(LIBNAME) $(LFLAGS)

$(PY_WRAPPER): $(SRCDIR)/py_wrapper.cpp $(LIBNAME)
	$(CXX) -c $(CXXFLAGS) $(PYFLAGS) $(SRCDIR)/py_wrapper.cpp -o $(OBJDIR)/py_wrapper.o
	$(CXX) -shared -o $(PY_WRAPPER) $(OBJDIR)/py_wrapper.o $(LIBNAME) $(LFLAGS) $(PYFLAGS)

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(LEGACYDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc
	$(CXX) -c $(CXXFLAGS) -Wno-unused-variable -o "$@" "$<"

clean:
	rm -f $(OBJDIR)/*.o $(EXEDIR)/*.exe $(LIBNAME) $(PY_WRAPPER)

test:
	cp $(TESTSDIR)/test_all.pl $(EXEDIR)/
	(cd $(EXEDIR); ./test_all.pl)

.PHONY: clean test
