#include "Mask.h"
#include <string>

Mask::Mask():
	data(0), pitch(0),
	surf(0)
{
	dim = make_uint2(0, 0);
}

Mask::Mask(uint2 dim)
{
	this->dim = dim;
	auto err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string dd = cudaGetErrorString(err);

		int a = 0;
	}
	cudaChannelFormatDesc chanel_descriptor = cudaCreateChannelDesc(
		32, 0, 0, 0,
		cudaChannelFormatKindFloat
	);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string dd = cudaGetErrorString(err);

		int a = 0;
	}
	cudaMallocArray(
		&data, &chanel_descriptor,
		dim.x, dim.y,
		cudaArraySurfaceLoadStore
	);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string dd = cudaGetErrorString(err);

		int a = 0;
	}
	pitch = dim.x * sizeof(float);

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data;
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string dd = cudaGetErrorString(err);

		int a = 0;
	}
	surf = 0;
	cudaCreateSurfaceObject(&surf, &res_desc);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::string dd = cudaGetErrorString(err);

		int a = 0;
	}
}

Mask::~Mask()
{
	if (surf != 0) cudaDestroySurfaceObject(surf);
	if (data != 0) cudaFreeArray(data);
}

void Mask::init(uint2 dim)
{
	if (surf != 0) return;

	this->dim = dim;

	cudaChannelFormatDesc chanel_descriptor = cudaCreateChannelDesc(
		32, 0, 0, 0,
		cudaChannelFormatKindFloat
	);

	cudaMallocArray(
		&data, &chanel_descriptor,
		dim.x, dim.y,
		cudaArraySurfaceLoadStore
	);

	pitch = dim.x * sizeof(float);

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = data;

	surf = 0;
	cudaCreateSurfaceObject(&surf, &res_desc);
}

void Mask::copyFromHost(float* data)
{
	cudaMemcpy2DToArray(this->data, 0, 0, data, pitch, dim.x * sizeof(float), dim.y, cudaMemcpyHostToDevice);
}
