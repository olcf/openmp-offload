include ../../Makefiles/Makefile.inc

jacobi.C : jacobi.c
	$(CC) $(CFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
