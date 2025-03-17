#pragma once
#include "VectorMath.h"

class matrix {
public:
	matrix() = default;
	matrix(vec3 a, vec3 b, vec3 c);
	float x[9];
};

vec3 operator*(vec3& v, matrix& m);