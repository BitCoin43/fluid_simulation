#include "TextureMask.h"

TextureMask::TextureMask():
	texture(0), pitch(0)
{
}

TextureMask::TextureMask(uint2 dim, float* data)
{
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaMallocArray(&this->data, &channel_desc, dim.x, dim.y);
	pitch = dim.x * sizeof(float);

	cudaMemcpy2DToArray(this->data, 0, 0, data, pitch, dim.x * sizeof(float), dim.y, cudaMemcpyHostToDevice);

	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = this->data;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeWrap;
	tex_desc.addressMode[1] = cudaAddressModeWrap;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 1;

	cudaCreateTextureObject(&texture, &res_desc, &tex_desc, NULL);

}
