CPP = g++
CPPFLAGS = -g -std=c++11
DEPS = gaussian_cl.hpp
OBJS = gaussian_cl.cpp main.cpp
EXEC = gaussian_cl

UNAME := $(shell uname)

ifeq($(UNAME), Linux)
LIBS = -lOpenCL
LIBS_DIR = -L/usr/local/cuda-9.0/lib64
endif

ifeq($(UNAME), Darwin)
LIBS = -framework OpenCL
endif

.PHONY: all me clean

all: $(OBJS)
	$(CPP) -o $(EXEC) $^ $(CPPFLAGS) $(LIBS)

me: $(OBJS)
	$(CPP) -o $(EXEC) $^ $(CPPFLAGS) $(LIBS_DIR) $(LIBS)

%.o: %.cpp $(DEPS)
	$(CPP) -c -o $@ $< $(CPPFLAGS)

clean: 
	rm -rf *.o
	rm -rf 
