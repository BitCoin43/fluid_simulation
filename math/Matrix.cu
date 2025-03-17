#include "Matrix.h"

matrix::matrix(vec3 a, vec3 b, vec3 c)
{
	x[0] = a.x;
	x[1] = a.y;
	x[2] = a.z;

	x[3] = b.x;
	x[4] = b.y;
	x[5] = b.z;

	x[6] = c.x;
	x[7] = c.y;
	x[8] = c.z;
}

vec3 operator*(vec3& v, matrix& m)
{
	return vec3(
		m.x[0] * v.x + m.x[3] * v.y + m.x[6] * v.z,
		m.x[1] * v.x + m.x[4] * v.y + m.x[7] * v.z,
		m.x[2] * v.x + m.x[5] * v.y + m.x[8] * v.z
	);
}
