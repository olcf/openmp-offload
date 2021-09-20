include ../../Machines/Makefile.inc

jacobi.C : jacobi.c
	$(CC) $(FFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
