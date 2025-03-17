#include "Primitive.h"

#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_launch_parameters.h>
#include <surface_functions.h>
#include <surface_indirect_functions.h>



namespace re {

	__device__ bool pointOutDim(int2 point, uint2 dim) {
		bool result = false;
		result |= point.x < 0;
		result |= point.y < 0;
		result |= point.x >= dim.x;
		result |= point.y >= dim.y;

		return result;
	}

	__global__ void fill(Color* colors, Color c, int n)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid < n) colors[tid] = c;
	}

	__global__ void renderRect(Color* colors, Color c, int2 start, int2 end, int2 dim)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x + start.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y + start.y;

		if (x < 0 || x >= dim.x || x >= end.x) return;
		if (y < 0 || y >= dim.y || y >= end.y) return;

		colors[y * dim.x + x] = c;
	}

	__global__ void renderElips(Color* colors, Color c, int2 center, int r, int2 dim)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x - r;
		int y = threadIdx.y + blockIdx.y * blockDim.y - r;

		if (r * r >= x * x + y * y) {
			colors[(y + center.y) * dim.x + x + center.x] = c;
		}
	}

	__global__ void fill(cudaSurfaceObject_t surf, uint2 dim, float4 color)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < dim.x && y < dim.y) {
			surf2Dwrite(color, surf, x * 16, y);
		}
	}

	__global__ void surfaceToLinear2D(
		cudaSurfaceObject_t surf,
		Color* colors, int2 start,
		uint2 surf_dim, uint2 colors_dim
	)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
		if (x < surf_dim.x && y < surf_dim.y) {
			float4 col;
			surf2Dread(&col, surf, x * 16, y);

			Color color(
				col.x * 255,
				col.y * 255,
				col.z * 255,
				col.w * 255
			);

			start.x += x;
			start.y += y;

			if (start.x < colors_dim.x && start.y < colors_dim.y && start.x > -1 && start.y > -1) {
				colors[start.y * colors_dim.x + start.x] = color;
			}
		}
	}

	__global__ void MaskToSurface(
		cudaSurfaceObject_t mask,
		cudaSurfaceObject_t surf,
		uint2 mask_dim, uint2 surf_dim,
		int2 start, Fcolor color
	)
	{
		unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

		start.x += x;
		start.y += y;

		//if (x >= mask_dim.x || y >= mask_dim.y) return;
		//if (start.x >= surf_dim.x || start.y >= surf_dim.y) return;
		//
		//if (x < 0 || y < 0) return;
		//if (start.x < 0 || start.y < 0) return;

		if (pointOutDim(make_int2(x, y), mask_dim)) return;
		if (pointOutDim(make_int2(start.x, start.y), surf_dim)) return;

		float alpha;
		float4 bg;
		Fcolor back;

		surf2Dread(&alpha, mask, x * 4, y);
		surf2Dread(reinterpret_cast<float4*>(&back), surf, start.x * 16, start.y);

		Fcolor up = color.blend(back, alpha);

		surf2Dwrite(*reinterpret_cast<float4*>(&up), surf, start.x * 16, start.y);


	}



}