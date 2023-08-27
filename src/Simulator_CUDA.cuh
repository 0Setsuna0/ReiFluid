#pragma once
#include "cuda_Helper.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.h>


//cubic spline kernel,comes from [Monaghan1992]
static __device__ float cubicSplineKernel_device(const float distance, const float kernelRadius)
{
	const auto q = 2.0f * fabs(distance) / kernelRadius;
	if (q > 2.0f || q < 1e-6f) return 0.0f;
	else {
		const auto a = 0.25f / (PI * kernelRadius * kernelRadius * kernelRadius);
		return a * ((q > 1.0f) ? (2.0f - q) * (2.0f - q) * (2.0f - q) : ((3.0f * q - 6.0f) * q * q + 4.0f));
	}
}

static __device__ float3 cubicSplineKernelGradient_device(const float3 dir, const float kernelRadius)
{
	auto q = 2.0f * length(dir) / kernelRadius;
	if (q > 2.0f) return make_float3(0.0f);
	else
	{
		auto a = dir / (PI * (q + 1e-6f) * kernelRadius * kernelRadius * kernelRadius * kernelRadius * kernelRadius);
		if (q > 1.0f)
		{
			return a * ((12.0f - 3.0f * q) * q - 12.0f);
		}
		else
		{
			return a * ((9.0f * q - 12.0f) * q);
		}
	}
}

//viscosity force attrib, based on second derivative of kernel
static __device__ float viscosityLaplacian_device(const float distance, const float kernelRadius)
{
	if (distance > kernelRadius)
	{
		return 0.0f;
	}
	else
	{
		return 45.0f * (kernelRadius - distance) / (PI * powf(kernelRadius, 6));
	}
}

static __device__ int particleIndex2GridCell_device(const int3 pos, const int3 gridResolution)
{
	if (pos.x >= 0 && pos.x < gridResolution.x && pos.y >= 0 && pos.y < gridResolution.y && pos.z >= 0 && pos.z < gridResolution.z)
	{
		//return __mul24((__mul24(pos.x, gridResolution.y) + pos.y), gridResolution.z) + pos.z;
		return (pos.x * gridResolution.y + pos.y) * gridResolution.z + pos.z;
	}	
	else
	{
		return gridResolution.x * gridResolution.y * gridResolution.z;
	}
}

static __global__ void putParticles2GridCell_global(int* p2g, float3* pos, const float gridLength, const int3 gridResolution, const int N)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x;
	if (i >= N) return;
	p2g[i] = particleIndex2GridCell_device(make_int3(pos[i] / gridLength), gridResolution);
}


static __device__ float3 surfaceTensionGradient_device(float3 r, const float radius)
{
	const auto x = length(r);
	if (x > radius || x < 1e6f) return make_float3(0.0f);
	else {
		auto cube = [](float x) {return x * x * x; };
		const float3 a = 136.0241f * -r / (PI * cube(radius) * cube(radius) * cube(radius) * x);
		return a * ((2.0f * x <= radius) ?
			(2.0f * cube(radius - x) * cube(x) - 0.0156f * cube(radius) * cube(radius)) :
			(cube(radius - x) * cube(x)));
	}
}
