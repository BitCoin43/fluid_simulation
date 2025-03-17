#include "Surface.h"

Surface::Surface():
	data(0), pitch(0),
	surf(0)
{
	dim = make_uint2(0, 0);
}

Surface::Surface(uint2 dim)
{
	this->dim = dim;

	cudaChannelFormatDesc chanel_descriptor = cudaCreateChannelDesc(
		32, 32, 32, 32,
		cudaChannelFormatKindFloat
	);

	cudaMallocArray(
		&data, &chanel_descriptor,
		dim.x, dim.y,
		cudaArraySurfaceLoadStore
	);

	pitch = dim.x * sizeof(float) * 4;

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data;

	surf = 0;
	cudaCreateSurfaceObject(&surf, &res_desc);
}

Surface::~Surface()
{
	if (surf != 0) cudaDestroySurfaceObject(surf);
	if (data != 0) cudaFreeArray(data);
}

void Surface::init(uint2 dim)
{
	if (surf != 0) return;

	this->dim = dim;

	cudaChannelFormatDesc chanel_descriptor = cudaCreateChannelDesc(
		32, 32, 32, 32,
		cudaChannelFormatKindFloat
	);

	cudaMallocArray(
		&data, &chanel_descriptor,
		dim.x, dim.y,
		cudaArraySurfaceLoadStore
	);

	pitch = dim.x * sizeof(float) * 4;

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data;

	surf = 0;
	cudaCreateSurfaceObject(&surf, &res_desc);
}
