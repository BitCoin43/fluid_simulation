#pragma once
#include "Vector.cuh"

inline __HD__ vec3 operator+(const vec3& a, const vec3& b)
{
	return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __HD__ vec3 operator-(const vec3& a, const vec3& b)
{
	return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __HD__ vec3 operator*(const vec3& a, const float& b)
{
	return vec3(a.x * b, a.y * b, a.z * b);
}
inline __HD__ vec3 operator/(const vec3& a, const float& b)
{
	return vec3(a.x / b, a.y / b, a.z / b);
}


inline __HD__ vec2 operator+(const vec2& a, const vec2& b)
{
	return vec2(a.x + b.x, a.y + b.y);
}
inline __HD__ vec2 operator-(const vec2& a, const vec2& b)
{
	return vec2(a.x - b.x, a.y - b.y);
}
inline __HD__ vec2 operator*(const vec2& a, const float& b)
{
	return vec2(a.x * b, a.y * b);
}
inline __HD__ vec2 operator/(const vec2& a, const float& b)
{
	return vec2(a.x / b, a.y / b);
}


inline __HD__ vec3 cross(vec3 a, vec3 b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

inline __HD__ float dot(vec3 a, vec3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __HD__ float getLength(vec3 v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline __HD__ vec3 normalize(vec3 v)
{
	return v / getLength(v);
}
