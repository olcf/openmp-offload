#module load xl

include ../../Machines/Makefile_relative.inc

all: jacobi.C

%.o : %.f90
	$(FC) -c $(FFLAGS) $(LIBS) $< -o $@
	
jacobi.Fortran : jacobi.f90
	$(FC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe
	rm -f *.mod

jacobi.C : jacobi.c
	$(CC) $(CFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

jacobi.CPP : jacobi.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
