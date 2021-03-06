#include <cmath>
#include <iostream>
#include <cstdio>

#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/opencl.h>
#endif

#include "gaussian_cl.hpp"

#define MAX_SOURCE_SIZE (0x100000)

using std::cerr;
using std::endl;

const double PI = 3.14159;

void generateKernel(int width, int height, double sigma, double *kernel)
{
	double sum = 0.0;
	int strideWidth = width / 2;
	int strideHeight = height / 2;
	
	// generate kernel
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] = exp(-(pow(i - strideHeight, 2) + pow(j - strideWidth, 2)) / (2 * sigma * sigma))
				/ (2 * PI * sigma * sigma);
			sum += kernel[(i * width) + j];
		}
	}

	// then normalize each element
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			kernel[(i * width) + j] /= sum;
		}
	}
}

const char *getErrorString(cl_int error)
{
    switch(error)
    {
    // run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

    // compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

    // extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}

void gaussianFilter(unsigned char *src, unsigned char *dst,
	int width, int height, int kernelWidth, int kernelHeight, double sigma)
{
	double *gaussianKernel = new double[kernelWidth * kernelHeight];

	generateKernel(kernelWidth, kernelHeight, sigma, gaussianKernel);

	// OpenCL initialization
	cl_int err;
	cl_platform_id cpPlatform; // OpenCL platform
        cl_device_id device_id; // device ID
        cl_context context; // context
        cl_command_queue queue; // command queue
        cl_program program; // program
        cl_kernel kernel; // kernel

	// get all platforms
	err = clGetPlatformIDs(1, &cpPlatform, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}

	// get default device of the default platform (here I use GPU)
        err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if(err != CL_SUCCESS)
        {   
                cerr << getErrorString(err) << endl;
                exit(1);
        }

	// create a context
        context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        if(err != CL_SUCCESS)
        {   
                cerr << getErrorString(err) << endl;
                exit(1);
        }

	// create a command queue
        queue = clCreateCommandQueue(context, device_id, 0, &err);
        if(err != CL_SUCCESS)
        {   
                cerr << getErrorString(err) << endl;
                exit(1);
        }

	// create the compute program from source buffer
        FILE *fp;
        char *source_str;
        size_t source_size;
	const char *kernel_code = "gaussian.cl";
        fp = fopen(kernel_code, "r");
        if(!fp)
        {
                cerr << "error reading " << kernel_code << endl;
                exit(1);
        }
        source_str = (char *)malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);
        program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
                        (const size_t *)&source_size, &err);
        if(err != CL_SUCCESS)
        {
                cerr << getErrorString(err) << endl;
                exit(1);
        }

	// build program
        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if(err != CL_SUCCESS)
        {
                cerr << "error building program" << endl;
                cerr << "Error type: " << getErrorString(err) << endl;
                char build_log[10000];
                cl_int error2;
                error2 =  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 10000, build_log, NULL);
                cerr << "log:" << endl << build_log << endl;
                exit(1);
        }

	// create kernel
        kernel = clCreateKernel(program, "gaussian", &err);
        if(!kernel || err != CL_SUCCESS)
        {
                cerr << getErrorString(err) << endl;
                exit(1);
        }

	// stuffs will be loaded to device
	cl_mem d_src = clCreateBuffer(context, CL_MEM_READ_ONLY, height * width * sizeof(unsigned char),
		NULL, NULL);
	cl_mem d_dst = clCreateBuffer(context, CL_MEM_READ_WRITE, height * width * sizeof(unsigned char),
		NULL, NULL);
	cl_mem d_gaussianKernel = clCreateBuffer(context, CL_MEM_READ_ONLY, kernelHeight * kernelWidth * sizeof(double),
		NULL, NULL);

	// write input/output data to device
	err = clEnqueueWriteBuffer(queue, d_src, CL_TRUE, 0, height * width * sizeof(unsigned char),
		src, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_dst, CL_TRUE, 0, height * width * sizeof(unsigned char),
		dst, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_gaussianKernel, CL_TRUE, 0, kernelHeight * kernelWidth * sizeof(double),
		gaussianKernel, 0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}

	// set the arguments to our compute kernel
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_src);
	err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dst);
	err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_gaussianKernel);
	err |= clSetKernelArg(kernel, 3, sizeof(int), &width);
	err |= clSetKernelArg(kernel, 4, sizeof(int), &height);
	err |= clSetKernelArg(kernel, 5, sizeof(int), &kernelWidth);
	err |= clSetKernelArg(kernel, 6, sizeof(int), &kernelHeight);
	err |= clSetKernelArg(kernel, 7, sizeof(double), &sigma);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err);
		exit(1);
	}

	// execute our compute kernel
	size_t global_item_size[2] = {(size_t)height, (size_t)width};
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_item_size, NULL,
		0, NULL, NULL);
	if(err != CL_SUCCESS)
	{
		cerr << getErrorString(err) << endl;
		exit(1);
	}

	err = clEnqueueReadBuffer(queue, d_dst, CL_TRUE, 0, height * width * sizeof(unsigned char),
		dst, 0, NULL, NULL);

	// wait till our compute kernel is complete
	clFinish(queue);

	// release OpenCL objects
	clReleaseMemObject(d_src);
	clReleaseMemObject(d_dst);
	clReleaseMemObject(d_gaussianKernel);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);	

	// release host objects
	free(source_str);
	delete [] gaussianKernel;
}
