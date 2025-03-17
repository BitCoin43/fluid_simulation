#pragma once
#ifndef __CUDACC__
//#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <memory>

class Surface {
public:
	Surface();
	Surface(uint2 dim);
	~Surface();
	void init(uint2 dim);
private:
	cudaArray_t data;
	size_t pitch;
	cudaSurfaceObject_t surf;
	uint2 dim;
	friend class Graphics;
};

