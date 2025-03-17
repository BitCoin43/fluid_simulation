#pragma once
#include "VectorMath.h"
#include <cfloat>
#include "Matrix.h"
#include "Face.h"

__host__ __device__ float RayTriangleIntersection(
	const vec3& origin, 
	const vec3& direction,
	Face& face
);

__host__ __device__ bool intersect_ray_sphere(
	const vec3& ray_origin, 
	const vec3& ray_direction, 
	const vec3& sphere_center, 
	const float& sphere_radius
);

__host__ __device__ vec3 barycentricCoordinates(
	vec3& p, vec3& v1,
	vec3& v2, vec3& v3
);

