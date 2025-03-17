#pragma once
#include "../../math/Color.cuh"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace re {


	__global__ void fill(Color* colors, Color c, int n);

	__global__ void renderRect(Color* colors, Color c, int2 start, int2 end, int2 dim);

	__global__ void renderElips(Color* colors, Color c, int2 x, int r, int2 dim);

	__global__ void fill(
		cudaSurfaceObject_t surf, 
		uint2 dim, float4 color
	);

	__global__ void surfaceToLinear2D(
		cudaSurfaceObject_t surf,
		Color* colors, int2 start,
		uint2 surf_dim, uint2 colors_dim
	);

	__global__ void MaskToSurface(
		cudaSurfaceObject_t mask,
		cudaSurfaceObject_t surf,
		uint2 mask_dim, uint2 surf_dim,
		int2 start, Fcolor color
	);


}