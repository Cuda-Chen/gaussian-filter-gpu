# Gaussian Filter CUDA Version
Naive implementation of Gaussian filter with CUDA.

# Dependencies
* OpenCV (for reading, writing and thresholding image only)
* CMake (for building)
* CUDA (of course)

# How to Compile and Run
If you're using UNIX-like system type these commands
```
$ mkdir build
$ cd build
$ cmake ..
$ make
$ cd ..
$ ./gaussian_cuda
```

# Note
I do not use any padding techniques in this project.
