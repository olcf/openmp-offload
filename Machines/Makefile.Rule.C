include ../../Machines/Makefile.inc

jacobi.Fortran : Jacobi.f90
	$(FC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe
	rm -f *.mod

jacobi.C : jacobi.c
	$(CC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
