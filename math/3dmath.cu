#include "3dmath.h"

__host__ __device__ float RayTriangleIntersection(
	const vec3& origin,
	const vec3& direction,
	Face& face
)
{
	vec3 edge1 = face.b - face.a;
	vec3 edge2 = face.c - face.a;
	
	vec3 h = cross(direction, edge2);
	float  a = dot(edge1, h);
	
	if (a > -1e-5 && a < 1e-5)
		return FLT_MAX;
	
	float  f = 1.0f / a;
	
	vec3 s = origin - face.a;
	float  u = f * dot(s, h);
	
	if (u < 0.0f || u > 1.0f)
		return FLT_MAX;
	
	vec3 q = cross(s, edge1);
	float  v = f * dot(direction, q);
	
	if (v < 0.0f || (u + v) > 1.0f)
		return FLT_MAX;
	
	float d = dot(edge2, q);
	float  t = f * d;
	
	return t;
}

__host__ __device__ bool intersect_ray_sphere(
	const vec3& ray_origin,
	const vec3& ray_direction,
	const vec3& sphere_center,
	const float& sphere_radius
)
{
	vec3 dist = sphere_center - ray_origin;
	float B = dot(dist, ray_direction);
	float C = dot(dist, dist) - sphere_radius * sphere_radius;
	float D = B * B - C;
	return D > 0.0;
	//return false;
}

__host__ __device__ vec3 barycentricCoordinates(
	vec3& p, vec3& v1,
	vec3& v2, vec3& v3
)
{
	vec3 vv0 = v2 - v1, vv1 = v3 - v1, vv2 = p - v1;
	float d00 = dot(vv0, vv0);
	float d01 = dot(vv1, vv0);
	float d11 = dot(vv1, vv1);
	float d20 = dot(vv0, vv2);
	float d21 = dot(vv1, vv2);
	float denom = d00 * d11 - d01 * d01;
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0 - v - w;
	return vec3(u, v, w);
	//return v3;
}

