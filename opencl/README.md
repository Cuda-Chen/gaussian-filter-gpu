# Gaussian Filter OpenCL Version
Naive implementation of Gaussian filter with OpenCL (mainly GPU).

# Dependencies
* OpenCV (for reading, writing and thresholding image only)
* CMake (for building, optimal)
* OpenCL (of course)

# How to Compile and Run
If you're using UNIX-like system type these commands
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ ./gaussian_cl
```
or type the following commands if you prefer to use makefile
```
$ make
$ ./gaussian_cl
```

# Note
I do not use any padding techniques in this project.
