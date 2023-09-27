ifeq ($(LMOD_SYSTEM_NAME),frontier)
   ifndef COMPILER
     $(error COMPILER not defined. use make COMPILER=value. Possible values cce, rocm, gcc)
   else ifeq ($(COMPILER),cce)
     include ../../Makefiles/Makefile.Frontier.cce.inc
   else ifeq ($(COMPILER),rocm)
     include ../../Makefiles/Makefile.Frontier.rocm.inc
   else ifeq ($(COMPILER),gcc)
     include ../../Makefiles/Makefile.Frontier.gcc.inc
   else
     $(error Unsupported COMPILER value of $(COMPILER). Supported values:  cce | rocm | gcc) 
   endif
endif

ifdef NERSC_HOST 
   ifeq "$(PE_ENV)" "NVIDIA"
      include ../../Makefiles/Makefile.pm.nvhpc.inc
   else   ifeq "$(PE_ENV)" "GNU"
      include ../../Makefiles/Makefile.pm.gcc.inc
   else ifeq "$(PE_ENV)" "llvm"
      include ../../Makefiles/Makefile.pm.llvm.inc
   else ifeq "$(PE_ENV)" "CRAY"
      include ../../Makefiles/Makefile.pm.cce.inc
   endif
endif
