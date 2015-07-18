CXX       = g++
CXXFLAGS += -Wall -O3 $(DEFINES) -fPIC -fdata-sections -ffunction-sections
LFLAGS   += -fPIC -lgsl -lgslcblas
ARC       = ar

SRCDIR    = src
OBJDIR    = obj
TESTSDIR  = tests
LEGACYDIR = src/legacy

SOURCES   = coord.cpp \
            potential_base.cpp \
            potential_analytic.cpp \
            potential_staeckel.cpp \
            potential_composite.cpp \
            potential_galpot.cpp \
            Numerics.cpp \
            actions_staeckel.cpp \
            mathutils.cpp \
            orbit.cpp 

#LEGACYSRC = Stackel_JS.cpp \
#            coordsys.cpp 

TESTSRCS  = test_mathutils.cpp \
            test_coord.cpp \
            test_units.cpp \
            test_potentials.cpp \
            test_staeckel.cpp \
            test_actionfinder.cpp

LIBNAME   = libfJ.a

HEADERS   = $(SOURCES:.cpp=.h)
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES)) 
TESTEXE   = $(patsubst %.cpp,%.exe,$(TESTSRCS))
LEGACYOBJ = $(patsubst %.cpp,$(OBJDIR)/%.o,$(LEGACYSRC)) 

all:      $(LIBNAME) $(TESTEXE)

$(LIBNAME):  $(OBJECTS) $(LEGACYOBJ)
	ar rv $(LIBNAME) $(OBJECTS) $(LEGACYOBJ)

%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	$(CXX) -o "$@" "$<" $(CXXFLAGS) -I$(SRCDIR) $(LIBNAME) $(LFLAGS)

clean:
	rm -f $(OBJECTS) $(LEGACYOBJ) *.exe

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	@mkdir -p $(OBJDIR)
	$(CXX) -c $(CXXFLAGS) -I$(SRCDIR) -o "$@" "$<"

$(OBJDIR)/%.o:  $(LEGACYDIR)/%.cpp
	$(CXX) -c $(CXXFLAGS) -I$(SRCDIR) -o "$@" "$<"

.PHONY: clean
