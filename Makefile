CXX       = g++
CXXFLAGS += -Wall -O3 -I$(SRCDIR) $(DEFINES) -fPIC -fdata-sections -ffunction-sections
LFLAGS   += -fPIC -Wl,-dead_strip -lgsl -lgslcblas
ARC       = ar

# uncomment (and possibly modify) the three lines below to use UNSIO library for input/output of N-body snapshots
DEFINES  += -DHAVE_UNSIO
CXXFLAGS += -I/Users/user/Documents/nemo/inc -I/Users/user/Documents/nemo/inc/uns
LFLAGS   += -L/Users/user/Documents/nemo/lib -lunsio -lnemo

#DEFINES  += -DHAVE_CUBATURE
#CXXFLAGS += -I/Users/user/Documents/soft/cubature-1.0
#LFLAGS   +=   /Users/user/Documents/soft/cubature-1.0/hcubature.o

SRCDIR    = src
OBJDIR    = obj
TESTSDIR  = tests
LEGACYDIR = src/legacy

SOURCES   = coord.cpp \
            potential_base.cpp \
            potential_factory.cpp \
            potential_analytic.cpp \
            potential_composite.cpp \
            potential_cylspline.cpp \
            potential_dehnen.cpp \
            potential_ferrers.cpp \
            potential_galpot.cpp \
            potential_sphharm.cpp \
            potential_staeckel.cpp \
            particles_io.cpp \
            Numerics.cpp \
            actions_staeckel.cpp \
            actions_interfocal_distance_finder.cpp \
            math_core.cpp \
            math_spline.cpp \
            math_specfunc.cpp \
            orbit.cpp \
            utils.cpp \

#LEGACYSRC = Stackel_JS.cpp \
#            coordsys.cpp 

TESTSRCS  = test_math_core.cpp \
            test_math_spline.cpp \
            test_coord.cpp \
            test_units.cpp \
            test_potentials.cpp \
            test_potential_sphharm.cpp \
            test_staeckel.cpp \
            test_actionfinder.cpp \
            test_actions_nbody.cpp \

LIBNAME   = libfJ.a

HEADERS   = $(SOURCES:.cpp=.h)
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES)) 
TESTEXE   = $(patsubst %.cpp,%.exe,$(TESTSRCS))
LEGACYOBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LEGACYSRC)) 

all:      $(LIBNAME) $(TESTEXE)

$(LIBNAME):  $(OBJECTS) $(LEGACYOBJ)
	ar rv $(LIBNAME) $(OBJECTS) $(LEGACYOBJ)

%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	$(CXX) -o "$@" "$<" $(CXXFLAGS) $(LIBNAME) $(LFLAGS)

clean:
	rm -f $(OBJECTS) $(LEGACYOBJ) *.exe

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

$(OBJDIR)/%.o:  $(LEGACYDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

test:
	./test_all.pl

.PHONY: clean test
