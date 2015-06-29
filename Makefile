CXX       = g++
CXXFLAGS += -Wall -O3 $(DEFINES) -fPIC -fdata-sections -ffunction-sections
LINK      = g++
LFLAGS   += -fPIC -lgsl -lgslcblas
SOURCES   = coord.cpp potential_base.cpp potential_spherical.cpp potential_staeckel.cpp \
            actions_staeckel.cpp orbit.cpp
            
OBJECTS   = $(SOURCES:.cpp=.o)

all:    test_units test_staeckel

test_units:    $(OBJECTS) test_units.o
	$(LINK) $(LFLAGS) -o test_units.exe $(OBJECTS) test_units.o

test_staeckel:    $(OBJECTS) test_staeckel.o
	$(LINK) $(LFLAGS) -o test_staeckel.exe $(OBJECTS) test_staeckel.o

clean:
	rm -f *.o *.exe

%.o:    %.cpp %.h
	$(CXX) -c $(CXXFLAGS) -o "$@" "$<"

.PHONY: clean
