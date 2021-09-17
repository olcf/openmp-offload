CC       =  xlc_r
CXX      =  xlC_r
FC       =  xlf2008_r

CFLAGS   = -O2 -qsmp=omp -qoffload
CXXFLAGS = -O2 -qsmp=omp -qoffload
FFLAGS   = -O2 -qsmp=omp -qoffload

LIBS     = -Wl,-rpath=$(OLCF_XL_ROOT)/lib  -lm

COMPILER = xl
