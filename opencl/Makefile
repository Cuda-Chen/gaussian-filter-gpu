CPP = g++
CPPFLAGS = -g -std=c++11
DEPS = gaussian_cl.hpp
OBJS = gaussian_cl.o main.o
EXEC = gaussian_cl

UNAME := $(shell uname)
OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

ifeq ($(UNAME), Linux)
LIBS += -lOpenCL
LIBS_DIR = -L/usr/local/cuda-9.0/lib64
endif

ifeq ($(UNAME), Darwin)
LIBS += -framework OpenCL
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
	rm -rf $(EXEC) 
