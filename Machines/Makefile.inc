ifdef OLCF_XL_ROOT
  include Machines/Makefile.Summit.xl.inc
endif

ifdef OLCF_GCC_ROOT
  include Machines/Makefile.Summit.gcc.inc
endif
