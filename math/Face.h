#pragma once
#include "VectorMath.h"

class Face {
public:
	Face()
	{
	}
	Face(vec3 a, vec3 b, vec3 c) :
		a(a), b(b), c(c)
	{
	}
public:
	vec3 getNormal()
	{
		return vec3();
	}
public:
	vec3 a, b, c;
};

#undef _HD_
