#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#ifndef __CUDACC__
//#define __CUDACC__
#endif

#define __HD__ __host__ __device__

typedef unsigned char uchar;
class Color {
public:
	__HD__  Color() :
		r(0), g(0), b(0), a(255) {}
	__HD__  Color(uchar r, uchar g, uchar b) :
		r(r), g(g), b(b), a(255) {}
	__HD__  Color(uchar r, uchar g, uchar b, uchar a) :
		r(r), g(g), b(b), a(a) {}
public:
	uchar r, g, b, a;
};

struct Fcolor {
public:
	__HD__ Fcolor()
	{}
	__HD__ Fcolor(float r, float g, float b):
		r(r), g(g), b(b), a(1.0f) {}
	__HD__ Fcolor(float r, float g, float b, float a) :
		r(r), g(g), b(b), a(a) {}
	__HD__ float4 make() {
		return make_float4(r, g, b, a);
	}
	__HD__ Fcolor blend(Fcolor background, float alpha) {
		return Fcolor(
			r * alpha + background.r * (1.0f - alpha),
			g * alpha + background.g * (1.0f - alpha),
			b * alpha + background.b * (1.0f - alpha),
			background.a
		);
	}
public:
	float r, g, b, a;
};

const Color purple(102, 0, 204);
const Color black(0, 0, 0);
const Color red(255, 0, 0);
const Color blue(0, 0, 255);
const Color grey(90, 90, 90);
const Color white(255, 255, 255);

//#undef __HD__