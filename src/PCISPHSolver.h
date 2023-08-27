#pragma once
#include "BaseSolver.h"
class PCISolver : BaseSolver
{
public:
	void step(Particles* fluids, Particles* boundary,
		int* gridStartFluid, int* gridStartBoundary,
		float3 gridSpacing,
		int3 gridResolution,
		float gridLength,
		float radius,
		float dt,
		float rho0,
		float rhoB,
		float stiffness,
		float viscosity,
		float3 G,
		float tension,
		float airPressure
	) override;

	void advect(Particles* fluids, float dt, float3 gridSpacing) override;

	void applyForce(Particles* fluids, const float dt, const float3 G) override;

	void applyViscosity(Particles* fluids,
		int* gridStartFluid,
		int3 gridResolution,
		float gridLength,
		float rho0,
		float radius,
		float viscosityCoef,
		float dt);

	void correctionIterate(Particles* fluids, int* gridStartF, Particles* boundary, int* gridStartB, int3 gridResolution,
		float gridLength, float rho0, float radius, float dt, float3 gridSpacing);

	void initBuffer(Particles* fluids);

	void computeVel(Particles* fluids, float dt);

	void computeDelta(float gridLength, float rho0, float radius, float dt, float3 gridSpacing, float* delta);

	PCISolver(Particles* particles)
	{
		CUDA_CHECK(cudaMalloc((void**)&F, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&Fp, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&pPos, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&pVel, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&pDensity, sizeof(float) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&densityError, sizeof(float) * particles->particles_num));
	}

private:
	int maxIterationNum = 5;
	float maxDensityErrorRatio = 0.01f;
	float densityErrorRatio;
	float3* F;
	float3* Fp;

	float* pDensity;
	float* densityError;
	float3* pPos;
	float3* pVel;
};