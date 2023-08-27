#pragma once
#include "cuda_Helper.h"
#include <helper_gl.h>
#include <vector>
class Particles 
{
public:
	explicit Particles(const std::vector<float3>& p)
		:particles_num(p.size())
	{
		//allocate memory in gpu device space
		CUDA_CHECK(cudaMalloc((void**)&pos, sizeof(float3) * p.size()));
		CUDA_CHECK(cudaMalloc((void**)&vel, sizeof(float3) * p.size()));
		CUDA_CHECK(cudaMalloc((void**)&pressure, sizeof(float) * p.size()));
		CUDA_CHECK(cudaMalloc((void**)&density, sizeof(float) * p.size()));
		CUDA_CHECK(cudaMalloc((void**)&mass, sizeof(float) * p.size()));
		CUDA_CHECK(cudaMalloc((void**)&p2g, sizeof(int) * p.size()));

		//copy cpu data, place into gpu buffer
		CUDA_CHECK(cudaMemcpy(pos, &p[0], sizeof(float3) * p.size(), cudaMemcpyHostToDevice));
	}
	~Particles()
	{
		cudaFree(pos);
		cudaFree(vel);
		cudaFree(pressure);
		cudaFree(density);
		cudaFree(mass);
		cudaFree(p2g);
	}

	inline unsigned int size() const { return particles_num; }

	void advect(float dt);

	void copyPosition(float3* rPos);

	void logPosition();

	void debugAdvect();
public:
	//gpu data
	float3* pos;
	float3* vel;
	float* pressure;
	float* density;
	float* mass;
	int* p2g;

	int particles_num;
	
};
