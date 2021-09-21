include ../../Makefiles/Makefile.inc

jacobi.CPP : jacobi.cpp
	$(CXX) $(CXXFLAGS) $(LIBS) $< -o ${@}.${COMPILER}.exe

clean:
	rm -f *.o *.mod *.exe
