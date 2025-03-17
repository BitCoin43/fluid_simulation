#pragma once
#include "Surface.h"



class Mask {
public:
	Mask();
	Mask(uint2 dim);
	~Mask();
	void init(uint2 dim);
	void copyFromHost(float* data);
private:
	uint2 dim;
	cudaArray_t data;
	size_t pitch;
	cudaSurfaceObject_t surf;
	friend class Graphics;
};
