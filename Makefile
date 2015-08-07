CXX       = g++
CXXFLAGS += -Wall -Wno-overflow -O3 -I$(SRCDIR) $(DEFINES) -fPIC -fdata-sections -ffunction-sections
LFLAGS   += -fPIC -lgsl -lgslcblas
ARC       = ar

# this flag apparently is only relevant for MacOS and reduces the size of executable files by removing unused code
#LFLAGS  += -Wl,-dead_strip

# uncomment (and possibly modify) the three lines below  to use UNSIO library for input/output of N-body snapshots
DEFINES  += -DHAVE_UNSIO
CXXFLAGS += -I/Users/user/Documents/nemo/inc -I/Users/user/Documents/nemo/inc/uns
LFLAGS   += -L/Users/user/Documents/nemo/lib -lunsio -lnemo

# uncomment the three lines below and adjust the paths  to use Cuba library for multidimensional integration 
# (otherwise use Cubature library bundled with the code)
DEFINES  += -DHAVE_CUBA
CXXFLAGS += -I/Users/user/Documents/soft/cuba-4.2
LFLAGS   +=   /Users/user/Documents/soft/cuba-4.2/libcuba.a

SRCDIR    = src
OBJDIR    = obj
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
            test_potentials.cpp \
            test_potential_sphharm.cpp \
            test_staeckel.cpp \
            test_actionfinder.cpp \
            test_actions_nbody.cpp \
            test_torus.cpp \

LIBNAME   = libfJ.a

HEADERS   = $(SOURCES:.cpp=.h)
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES)) 
TESTEXE   = $(patsubst %.cpp,%.exe,$(TESTSRCS))
LEGACYOBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LEGACYSRC)) 
TORUSOBJ  = $(patsubst %.cc, $(OBJDIR)/%.o,$(TORUSSRC)) 

all:      $(LIBNAME) $(TESTEXE)

$(LIBNAME):  $(OBJECTS) $(LEGACYOBJ) $(TORUSOBJ)
	ar rv $(LIBNAME) $(OBJECTS) $(LEGACYOBJ) $(TORUSOBJ)

%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	$(CXX) -o "$@" "$<" $(CXXFLAGS) $(LIBNAME) $(LFLAGS)

clean:
	rm -f $(OBJECTS) $(LEGACYOBJ) $(TORUSOBJ) *.exe

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(SRCDIR)/%.h Makefile
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(LEGACYDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(TORUSDIR)/%.cc
	$(CXX) -c $(CXXFLAGS) -Wno-unused-variable -o "$@" "$<"

test:
	./test_all.pl

.PHONY: clean test
