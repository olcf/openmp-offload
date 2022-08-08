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
   ifeq "$(PE_ENV)" "NVIDIA"
      include ../../Makefiles/Makefile.pm.nvhpc.inc
   else   ifeq "$(PE_ENV)" "GNU"
      include ../../Makefiles/Makefile.pm.gcc.inc
   else ifeq "$(PE_ENV)" "LLVM"
      include ../../Makefiles/Makefile.pm.llvm.inc
   else ifeq "$(PE_ENV)" "CRAY"
      include ../../Makefiles/Makefile.pm.cce.inc
   endif
endif
