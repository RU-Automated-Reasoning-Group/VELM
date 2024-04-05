CXX = g++
HOME= /usr/local/include
POLAR_HOME = ../POLAR_Tool/POLAR
FLOWSTAR_HOME = ../POLAR_Tool/flowstar/flowstar-toolbox
LIBS = -lpolar -lflowstar -lmpfr -lgmp -lgsl -lgslcblas -lm -lglpk
CFLAGS = -I . -I $(HOME) -g -O3 -std=c++11
LINK_FLAGS = -g  -L$(POLAR_HOME) -L$(FLOWSTAR_HOME) -L/usr/local/lib


all: acc acc_test acc_one_iter

acc: acc.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

acc_test: acc_test.o
	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)

acc_one_iter: acc_one_iter.o

	g++ -O3 -w $(LINK_FLAGS) -o $@ $^ $(LIBS)
%.o: %.cc
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.cpp
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<
%.o: %.c
	$(CXX) -O3 -c $(CFLAGS) -o $@ $<


clean:
	rm -f *.o acc

