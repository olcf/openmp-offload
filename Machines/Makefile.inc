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

ifdef NERSC_HOST 
   ifdef NVHPC
      include Machines/Makefile.Cori.nvhpc.inc
   endif
   ifdef GCC_ROOT
      include Machines/Makefile.Cori.gcc.inc
   endif
   ifdef LLVM_ROOT
      include Machines/Makefile.Cori.llvm.inc
   endif
   ifdef CRAY_CCE_SHARE
      include Machines/Makefile.Cori.cce.inc
   endif
endif
