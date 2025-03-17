#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>

#define __HD__ __host__ __device__

class vec3 {
public:
	__HD__ vec3():x(0.0f), y(0.0f), z(0.0f){};
	__HD__ vec3(float x, float y, float z) :
		x(x), y(y), z(z)
	{
	}
public:
	__HD__ vec3& operator+=(const vec3& v)
	{
		this->x += v.x;
		this->y += v.y;
		this->z += v.z;
		return *this;
	}
	__HD__ vec3& operator-=(const vec3& v)
	{
		this->x -= v.x;
		this->y -= v.y;
		this->z -= v.z;
		return *this;
	}
	__HD__ vec3& operator*=(const float& v)
	{
		this->x *= v;
		this->y *= v;
		this->z *= v;
		return *this;
	}
	__HD__ vec3& operator/=(const float& v)
	{
		this->x /= v;
		this->y /= v;
		this->z /= v;
		return *this;
	}
public:
	float x, y, z;
};

class vec2 {
public:
	__HD__ vec2() :
		x(0.0f), y(0.0f)
	{
	}
	__HD__ vec2(float x, float y) :
		x(x), y(y)
	{
	}
public:
	__HD__ vec2& operator+=(const vec2& v)
	{
		this->x += v.x;
		this->y += v.y;
		return *this;
	}
	__HD__ vec2& operator-=(const vec2& v)
	{
		this->x -= v.x;
		this->y -= v.y;
		return *this;
	}
	__HD__ vec2& operator*=(const float& v)
	{
		this->x *= v;
		this->y *= v;
		return *this;
	}
	__HD__ vec2& operator/=(const float& v)
	{
		this->x /= v;
		this->y /= v;
		return *this;
	}
public:
	float x, y;
};

