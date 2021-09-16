#module load xl

include Machines/Makefile.inc

vpath %.c C
vpath %.f90 Fortran

all: Jacobi.Fortran Jacobi.C

%.o : %.f90
	$(FC) -c $(FFLAGS) $(LIBS) $< -o $@
	
Jacobi.Fortran : Jacobi.f90
	$(FC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe
	rm -f *.mod

Jacobi.C : Jacobi.c
	$(CC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
