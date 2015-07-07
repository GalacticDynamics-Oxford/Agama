CXX       = g++
CXXFLAGS += -Wall -O3 $(DEFINES) -fPIC -fdata-sections -ffunction-sections
LFLAGS   += -fPIC -lgsl -lgslcblas
ARC       = ar

SRCDIR    = src
OBJDIR    = obj
TESTSDIR  = tests
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
#            WDMath.cpp 
TESTSRCS  = test_coord.cpp \
            test_units.cpp \
            test_potentials.cpp \
            test_staeckel.cpp

LIBNAME   = libfJ.a

HEADERS   = $(SOURCES:.cpp=.h)
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES)) 
TESTEXE   = $(patsubst %.cpp,%.exe,$(TESTSRCS))

all:      $(LIBNAME) $(TESTEXE)

$(LIBNAME):  $(OBJECTS)
	ar rv $(LIBNAME) $(OBJECTS)

%.exe:  $(TESTSDIR)/%.cpp $(LIBNAME)
	$(CXX) $(CXXFLAGS) -I$(SRCDIR) $(LFLAGS) $(LIBNAME) -o "$@" "$<"

clean:
	rm -f $(OBJECTS) *.exe

$(OBJDIR)/%.o:  $(SRCDIR)/%.cpp $(SRCDIR)/%.h
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

.PHONY: clean
