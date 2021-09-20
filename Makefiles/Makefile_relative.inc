ifdef OLCF_XL_ROOT
  include ../../Makefiles/Makefile.Summit.xl.inc
endif

ifdef OLCF_GCC_ROOT
  include ../../Makefiles/Makefile.Summit.gcc.inc
endif

ifdef OLCF_NVHPC_ROOT
  include ../../Makefiles/Makefile.Summit.nvhpc.inc
endif

ifdef OLCF_LLVM_ROOT
  include ../../Makefiles/Makefile.Summit.llvm.inc
endif

ifdef NERSC_HOST 
   ifdef NVHPC
      include ../../Makefiles/Makefile.Cori.nvhpc.inc
   endif
   ifdef GCC_ROOT
      include ../../Makefiles/Makefile.Cori.gcc.inc
   endif
   ifdef LLVM_ROOT
      include ../../Makefiles/Makefile.Cori.llvm.inc
   endif
   ifdef CRAY_CCE_SHARE
      include ../../Makefiles/Makefile.Cori.cce.inc
   endif
endif
