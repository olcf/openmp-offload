ifdef OLCF_XL_ROOT
  include Machines/Makefile.Summit.xl.inc
endif

ifdef OLCF_GCC_ROOT
  include Machines/Makefile.Summit.gcc.inc
endif

ifdef OLCF_NVHPC_ROOT
  include Machines/Makefile.Summit.nvhpc.inc
endif

ifdef OLCF_LLVM_ROOT
  include Machines/Makefile.Summit.llvm.inc
endif
