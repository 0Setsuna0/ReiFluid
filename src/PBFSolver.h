#pragma once
#include "BaseSolver.h"
class PBFSolver:BaseSolver
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

	void predictPosition(Particles* fluids, float dt, float3 gridSpacing);

	void copyPosition(Particles* fluid);

	void applyViscosity(Particles* fluids, int* gridStartFluid, int3 gridResolution,
		float gridLength, float rho0, float radius, float viscosityCoef);

	void neighborSearch(Particles* particles);

	void applyConstraintLimitation(Particles* fluids, Particles* boundary, int* gridStartFluid, int* gridStartBoundary, int3 gridResolution,
		float gridLength, float3 gridSpacing, float3* predictedPos, float rho0, int maxIterationNum, float radius);

	void initPredictPos(float3* pos, int N)
	{
		cudaMemcpy(predictPos, pos, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
		initOnce = false;
	}

	PBFSolver(Particles* particles) {
		CUDA_CHECK(cudaMalloc((void**)&tempBuffer, sizeof(int) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&predictPos, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&dpos, sizeof(float3) * particles->particles_num));
		CUDA_CHECK(cudaMalloc((void**)&lambda, sizeof(float) * particles->particles_num));
		cudaMemcpy(predictPos, particles->pos, sizeof(float3) * particles->particles_num, cudaMemcpyDeviceToDevice);
	}


	~PBFSolver() {}

public:
	int maxIterationNum = 20;
	int* tempBuffer;
	float3* predictPos;
	float3* dpos;
	float* lambda;
	bool initOnce = true;
};