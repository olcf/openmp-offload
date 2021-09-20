CC       =  xlc_r
CXX      =  xlC_r
FC       =  xlf2008_r

CFLAGS   = -Ofast -qsmp=omp -qoffload
CXXFLAGS = -Ofast -qsmp=omp -qoffload
FFLAGS   = -Ofast -qsmp=omp -qoffload

LIBS     = -Wl,-rpath=$(OLCF_XL_ROOT)/lib  -lm

COMPILER = xl
