#pragma once
#include "Surface.h"

class TextureMask {
public:
	TextureMask();
	TextureMask(uint2 dim, float* data);
private:
	cudaArray_t data;
	size_t pitch;
	cudaTextureObject_t texture;
	uint2 dim;
	friend class Graphics;
};