#include "Graphics.h"
#include "./2d/Primitive.h"


#ifndef __CUDACC__
#define __call_kernel(func, blocks, threads, ...) func(__VA_ARGS__)
#else
#define __call_kernel(func, blocks, threads, ...) func<<<blocks,threads>>>(__VA_ARGS__)
#endif


Graphics::Graphics(int width, int height)
{
	cudaMalloc((void**)&frame, sizeof(Color) * height * width);
	this->width = width;
	this->height = height;

	text.init(30);
}

Graphics::~Graphics()
{
}

void Graphics::fillFrame(Color c)
{
	int size = width * height;
	int threads = 1024;
	__call_kernel(
		re::fill,
		ceil(size / (float)threads), threads,
		frame, c, size
	);
}

void Graphics::renderLine(Color c, vec2 start, vec2 end)
{

}

void Graphics::renderRect(Color c, vec2 start, vec2 end)
{
	int2 a = normalizeVec2(start);
	int2 b = normalizeVec2(end);

	dim3 treads(16, 16);
	dim3 blocks(ceil((b.x - a.x) / 16.0f), ceil((b.y - a.y) / 16.0f));
	__call_kernel(
		re::renderRect,
		blocks, treads,
		frame, c, a, b, make_int2(width, height)
	);
}

void Graphics::renderElips(Color c, vec2 x, float r)
{
	int2 a = normalizeVec2(x);
	int R = r * height;

	dim3 treads(16, 16);
	dim3 blocks(ceil((R * 2) / 16.0f), ceil((R * 2) / 16.0f));
	__call_kernel(
		re::renderElips,
		blocks, treads,
		frame, c, a, R, make_int2(width, height)
	);
}

void Graphics::fill(Surface* surf, Fcolor color)
{
	dim3 threads(16, 16);
	dim3 blocks(
		(surf->dim.x + threads.x - 1) / threads.x,
		(surf->dim.y + threads.y - 1) / threads.y
	);
	__call_kernel(
		re::fill,
		blocks, threads,
		surf->surf, surf->dim, color.make()
	);
}


void Graphics::surfaceToFrame(Surface* surf, vec2 start)
{
	dim3 threads(16, 16);
	dim3 blocks(
		(surf->dim.x + threads.x - 1) / threads.x,
		(surf->dim.y + threads.y - 1) / threads.y
	);
	int2 st = normalizeVec2(start);

	__call_kernel(
		re::surfaceToLinear2D,
		blocks, threads,
		surf->surf, frame, normalizeVec2(start), surf->dim, make_uint2(width, height)
	);
}

void Graphics::surfaceToFrame(Surface* surf, int2 start)
{
	dim3 threads(16, 16);
	dim3 blocks(
		(surf->dim.x + threads.x - 1) / threads.x,
		(surf->dim.y + threads.y - 1) / threads.y
	);

	__call_kernel(
		re::surfaceToLinear2D,
		blocks, threads,
		surf->surf, frame, start, surf->dim, make_uint2(width, height)
	);
}

void Graphics::renderLine(Surface* surf, Fcolor color, int2 start, int2 end)
{
}

void Graphics::renderRect(Surface* surf, Fcolor color, int2 start, int2 end)
{
}

void Graphics::renderElips(Surface* surf, Fcolor c, int2 x, int r)
{
}

void Graphics::print(Surface* surf, std::string text, int2 start, Fcolor color)
{
	this->text.print(text, start, color, surf);
}

uchar4* Graphics::getFrame()
{
	return reinterpret_cast<uchar4*>(frame);
}

uint2 Graphics::getDimFromVec2(vec2 v)
{
	return make_uint2(
		v.x * height,
		v.y * height
	);
}

int2 Graphics::normalizeVec2(vec2 v)
{
	v += vec2(1, 1);
	v /= 2.0f;

	return make_int2(v.x * height + (width - height) / 2, v.y * height);
}

uint2 Graphics::forceNormalizeVec2(vec2 v)
{
	int2 a = normalizeVec2(v);

	if (a.x < 0) a.x = 0;
	if (a.x >= width) a.x = width - 1;

	if (a.y < 0) a.y = 0;
	if (a.y >= height) a.x = height - 1;

	return make_uint2(a.x, a.y);
}

Graphics::Text::Text() :
	font_size(-1), textures(nullptr)
{

}

void Graphics::Text::init(int fontSize)
{
	font_size = fontSize;

	textures = new Glyph * [256];
	for (int i = 0; i < 256; i++) {
		textures[i] = nullptr;
	}

	FT_Init_FreeType(&library);
	FT_New_Face(library, "C:\\fonts\\CascadiaMonoPL-Regular.otf", 0, &face);

	FT_Set_Pixel_Sizes(face, 0, fontSize);
}

Graphics::Text::~Text()
{
	FT_Done_Face(face);
	FT_Done_FreeType(library);
	for (int i = 0; i < 256; i++) {
		if (textures[i] != nullptr) {
			delete textures[i];
		}
	}
	delete[] textures;
}

void Graphics::Text::print(std::string text, int2 start, Fcolor color, Surface* dst)
{

	for (int i = 0; i < text.size(); i++) {
		Glyph* glyph = textures[text[i]];
		if (glyph == nullptr) {
			FT_Load_Char(face, text[i], FT_LOAD_RENDER);
			FT_GlyphSlot slot = face->glyph;
			FT_Outline* outline = &slot->outline;
			FT_BBox box;
			FT_Outline_Get_CBox(outline, &box);
			float* data = new float[slot->bitmap.rows * slot->bitmap.width];
			for (unsigned int row = 0; row < slot->bitmap.rows; row++) {
				for (unsigned int col = 0; col < slot->bitmap.width; col++) {
					int alpha = slot->bitmap.buffer[row * slot->bitmap.width + col];

					float _alpha = static_cast<float>(alpha) / 255.0f;

					data[col + slot->bitmap.width * row] = _alpha;
				}
			}

			auto err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::string dd = cudaGetErrorString(err);

				int a = 0;
			}

			textures[text[i]] = new Glyph();

			glyph = textures[text[i]];

			glyph->advance = slot->advance.x >> 6;
			glyph->horiBearingX = slot->metrics.horiBearingX >> 6;
			glyph->horiBearingY = slot->metrics.horiBearingY >> 6;
			glyph->dim = make_uint2(slot->bitmap.width, slot->bitmap.rows);
			if (glyph->dim.x == 0 || glyph->dim.y == 0) {
				glyph->have_texture = false;
			}
			else {
				glyph->have_texture = true;
				textures[text[i]]->texture.init(make_uint2(slot->bitmap.width, slot->bitmap.rows));
				glyph->texture.copyFromHost(data);
				delete[] data;

			}


			err = cudaGetLastError();
			if (err != cudaSuccess) {
				std::string dd = cudaGetErrorString(err);

				int a = 0;
			}


		}

		auto err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::string dd = cudaGetErrorString(err);

			int a = 0;
		}

		dim3 threads(16, 16);
		dim3 blocks((glyph->dim.x + threads.x - 1) / threads.x, (glyph->dim.y + threads.y - 1) / threads.y);

		int2 _start = make_int2(
			start.x + glyph->horiBearingX,
			start.y - glyph->horiBearingY
		);
		if (glyph->have_texture) {

			__call_kernel(
				re::MaskToSurface,
				blocks, threads,
				glyph->texture.surf, dst->surf, glyph->texture.dim, dst->dim, _start, color
			);
		}
		cudaDeviceSynchronize();

		err = cudaGetLastError();
		if (err != cudaSuccess) {
			std::string dd = cudaGetErrorString(err);

			int a = 0;
		}

		start.x += glyph->advance;
	}
}
