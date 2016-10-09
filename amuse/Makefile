# standard amuse configuration include
# config.mk will be made after ./configure has run
AMUSE_DIR?=..
-include $(AMUSE_DIR)/config.mk

-include ../Makefile.local

MPICXX   ?= mpicxx

CFLAGS   += -I../src
CXXFLAGS += $(CFLAGS)
LDFLAGS  += $(LFLAGS) $(MUSE_LD_FLAGS)

OBJS = interface.o

CODELIB = ../lib/libagama.a

CODE_GENERATOR = $(AMUSE_DIR)/build.py

all: agama_worker

clean:
	$(RM) -f *.so *.o *.pyc worker_code.cc worker_code.h 
	$(RM) *~ agama_worker worker_code.cc
	make -C clean

$(CODELIB):
	make -C all

worker_code.cc: interface.py
	$(CODE_GENERATOR) --type=c interface.py AgamaInterface -o $@

worker_code.h: interface.py
	$(CODE_GENERATOR) --type=H interface.py AgamaInterface -o $@

agama_worker: worker_code.cc worker_code.h $(CODELIB) $(OBJS)
	$(MPICXX) $(CXXFLAGS) $< $(OBJS) $(CODELIB) -o $@ $(LDFLAGS)

.cc.o: $<
	$(CXX) $(CXXFLAGS) -c -o $@ $< 
