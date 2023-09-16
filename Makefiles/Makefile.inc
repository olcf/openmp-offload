ifeq ($(LMOD_SYSTEM_NAME),frontier)
   ifndef COMPILER
     $(error COMPILER not defined. use make COMPILER=value. Possible values cce, rocm, gcc)
   else ifeq ($(COMPILER),cce)
     $(info Make sure the following modules are loaded: PrgEnv-cray cce craype-accel-amd-gfx90a )
     include ../../Makefiles/Makefile.Frontier.cce.inc
   else ifeq ($(COMPILER),rocm)
     $(info Make sure the following modules are loaded: PrgEnv-amd  amd )
     include ../../Makefiles/Makefile.Frontier.rocm.inc
   else ifeq ($(COMPILER),gcc)
     $(info Make sure the following cammends were executed first: PrgEnv-cray cce craype-accel-amd-gfx90a )
     $(info   module use /sw/crusher/ums/compilers/modulefiles)
     $(info   module load  module load gcc/13.2.1-dev-latest )
     include ../../Makefiles/Makefile.Frontier.gcc.inc
   else
     $(error Unsupported COMPILER value of $(COMPILER)) 
   endif
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
