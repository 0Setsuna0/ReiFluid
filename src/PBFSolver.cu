#define __CUDACC__
#include "PBFSolver.h"
#include "cuda_Helper.h"
#include "Simulator_CUDA.cuh"
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

__global__ void handleBoundaryConditionPBF_global(float3* pos, float3* vel, const int N, const float3 gridSpacing)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N) return;
	if (pos[p_i].x <= 0.0f)
	{
		pos[p_i].x = 0.05f;
		vel[p_i].x = 0.0f;
	}
	if (pos[p_i].x >= gridSpacing.x)
	{
		pos[p_i].x = gridSpacing.x * 0.9f;
		vel[p_i].x = 0.0f;
	}
	if (pos[p_i].y <= 0.0f)
	{
		pos[p_i].y = 0.05f;
		vel[p_i].y = 0.0f;
	}
	if (pos[p_i].y >= gridSpacing.y)
	{
		pos[p_i].y = gridSpacing.y * 0.9f;
		vel[p_i].y = 0.0f;
	}
	if (pos[p_i].z <= 0.0f)
	{
		pos[p_i].z = 0.05f;
		vel[p_i].z = 0.0f;
	}
	if (pos[p_i].z >= gridSpacing.z)
	{
		pos[p_i].z = gridSpacing.z * 0.9f;
		vel[p_i].z = 0.0f;
	}
}

void PBFSolver::advect(Particles* fluids, float dt, float3 gridSpacing)
{
	fluids->advect(dt);
	int N = fluids->particles_num;
	handleBoundaryConditionPBF_global << <(N - 1) / 256 + 1, 256 >> > (fluids->pos, 
		fluids->vel, fluids->particles_num, gridSpacing);
}

__global__ void applyGravity_global(float3* vel, int N, float3 G, float dt)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N) return;
	vel[p_i] += dt * G;
}

void PBFSolver::applyForce(Particles* fluids, const float dt, const float3 G)
{
	int N = fluids->particles_num;
	applyGravity_global << <(N - 1) / blockSize + 1, blockSize >> > (fluids->vel, N, G, dt);
}

__global__ void predict_pos_global(float3* ppos, float3* vel, float3* pos, float dt, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N) return;
	ppos[p_i] = pos[p_i] + vel[p_i] * dt;
}

void PBFSolver::predictPosition(Particles* fluids, float dt, float3 gridSpacing)
{
	int N = fluids->particles_num;
	//cudaMemcpy(predictPos, fluids->pos, sizeof(float3) * N, cudaMemcpyDeviceToDevice);
	//advect(fluids, dt, gridSpacing);

	predict_pos_global << <(N - 1) / blockSize + 1, blockSize >> > (predictPos, fluids->vel, fluids->pos, dt, N);
}

void PBFSolver::neighborSearch(Particles* particles)
{
	int N = particles->particles_num;
	CUDA_CHECK(cudaMemcpy(tempBuffer, particles->p2g, sizeof(int) * N, cudaMemcpyDeviceToDevice));
	thrust::sort_by_key(thrust::device, tempBuffer, tempBuffer + N, predictPos);
}

__device__ void accumulateLambda_device(float* density, float3* gradientSum, float* sampleLambda, int gridStart, int gridEnd,
	float* mass, float3 pos_i, float3* pos, float rho0, float radius)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*density += mass[p_j] * cubicSplineKernel_device(length(pos_i - pos[p_j]), radius); 
		float3 gradient = -mass[p_j] * cubicSplineKernelGradient_device(pos_i - pos[p_j], radius) / rho0;
		*gradientSum += gradient;
		*sampleLambda += dot(gradient, gradient);
	}
}

__global__ void computeLambda_global(float* density, float* lambda, float3* posF, float* massF, float3* posB, float* massB,
	int* gridStartF, int* gridStartB, int3 gridResolution, float gridLength, float radius, float rho0, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N) return;
	float3 gradientSum = make_float3(0.0f, 0.0f, 0.0f);
	float sampleLambda = 0.0f;
	density[p_i] = 0.0f;
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	for (int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(make_int3(posF[p_i] / gridLength) + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)continue;
				accumulateLambda_device(&density[p_i], &gradientSum, &sampleLambda, gridStartF[gridIdx], gridStartF[gridIdx + 1],
					massF, posF[p_i], posF, rho0, radius);
				accumulateLambda_device(&density[p_i], &gradientSum, &sampleLambda, gridStartB[gridIdx], gridStartB[gridIdx + 1],
					massB, posF[p_i], posB, rho0, radius);
			}
	if (density[p_i] > rho0)
	{
		lambda[p_i] = -(density[p_i] / rho0 - 1.0f) / (dot(gradientSum, gradientSum) + sampleLambda + 1e-6);
	}
	else
	{
		lambda[p_i] = 0.0f;
	}

}

__device__ void accumulateDposF_device(float3* dpos, int p_i, float3* pos, float* lambda, float* mass, int gridStart,
	int gridEnd, float radius, float rho0)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*dpos += mass[p_j] * ((lambda[p_i] + lambda[p_j]) / rho0) * cubicSplineKernelGradient_device(pos[p_i] - pos[p_j], radius) ;
	}
}

__device__ void accumulateDposB_device(float3* dpos, float3 pos_i, float3* pos, float lambda_i, float* mass,
	int gridStart, int gridEnd, float radius, float rho0)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*dpos += mass[p_j] * (lambda_i / rho0) * cubicSplineKernelGradient_device(pos_i - pos[p_j], radius);
	}
}

__global__ void computeDpos_global(float3* dPos, float3* posF, float3* posB, float* lambda, float* massF,
	float* massB, int* gridStartF, int* gridStartB, int3 gridResolution, float gridLength, float radius, float rho0, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N) return;
	float3 dpos = make_float3(0.0f);
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	for (int a = -1; a < 2; a++)
		for (int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(make_int3(posF[p_i] / gridLength) + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)continue;
				accumulateDposF_device(&dpos, p_i, posF, lambda, massF, gridStartF[gridIdx], gridStartF[gridIdx + 1], radius, rho0);
				accumulateDposB_device(&dpos, posF[p_i], posB, lambda[p_i], massB, gridStartB[gridIdx], gridStartB[gridIdx + 1], radius, rho0);
			}
	dPos[p_i] = dpos;
}

__global__ void addDpos_global(float3* dpos, float3* pos, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	pos[p_i] += dpos[p_i];
}

void PBFSolver::applyConstraintLimitation(Particles* fluids, Particles* boundary, int* gridStartFluid, int* gridStartBoundary, int3 gridResolution,
	float gridLength, float3 gridSpacing, float3* predictedPos, float rho0, int maxIterationNum, float radius)
{
	int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	for (int iter = 0; iter < maxIterationNum; iter++)
	{
		computeLambda_global << < blockNum, blockSize>> > (fluids->density, lambda, predictedPos, fluids->mass, boundary->pos, boundary->mass,
			gridStartFluid, gridStartBoundary, gridResolution, gridLength, radius, rho0, N);

		computeDpos_global << < blockNum, blockSize >> > (dpos, predictedPos, boundary->pos, lambda, fluids->mass, boundary->mass, gridStartFluid,
			gridStartBoundary, gridResolution, gridLength, radius, rho0, N);

		addDpos_global << < blockNum, blockSize >> > (dpos, predictedPos, N);

		handleBoundaryConditionPBF_global << <blockNum, blockSize >> > (predictedPos,
			fluids->vel, fluids->particles_num, gridSpacing);
	}
}

__device__ void computeViscosity_device(float3* force, int p_i, float3* pos, float3* vel, int gridStart,
	int gridEnd, float* mass, float radius, float rho0)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*force += mass[p_j] * (vel[p_j] - vel[p_i]) / rho0 * cubicSplineKernel_device(length(pos[p_i] - pos[p_j]), radius);
	}
}

__global__ void computeViscosityForce_global(float3* vel, float3* pos, int* gridStart, int3 gridResolution,
	float gridLength, float radius, float* mass, float viscosityCoef, float rho0, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N)return;
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	int blockNum = (N - 1) / 256 + 1;
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	for (int a = -1; a < 2; a++)
		for (int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(make_int3(pos[p_i] / gridLength) + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)
					continue;
				computeViscosity_device(&force, p_i, pos, vel, gridStart[gridIdx], gridStart[gridIdx + 1], mass, radius, rho0);
			}
	vel[p_i] += viscosityCoef * force;
}

void PBFSolver::applyViscosity(Particles* fluids, int* gridStartFluid, int3 gridResolution,
	float gridLength, float rho0, float radius, float viscosityCoef)
{
	int N = fluids->particles_num;
	computeViscosityForce_global << <(N - 1) / blockSize + 1, blockSize >> > (fluids->vel, predictPos, gridStartFluid, gridResolution,
		gridLength, radius, fluids->mass, 0.1f, rho0, N);
}

__global__ void updateVelocity_global(float3* ppos, float3* pos, float3* vel, float dt, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N)return;
	vel[p_i] = (ppos[p_i] - pos[p_i]) / dt;
}

__global__ void copy_pos_global(float3* ppos, float3* pos, int N)
{
	int p_i = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	if (p_i >= N)return;
	pos[p_i] = ppos[p_i];
}

void PBFSolver::copyPosition(Particles* fluids)
{
	int N = fluids->particles_num;
	copy_pos_global << <(N - 1) / blockSize + 1, blockSize >> > (predictPos, fluids->pos, N);
}

void PBFSolver::step(Particles* fluids, Particles* boundary,
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
)
{
	int N = fluids->particles_num;


	neighborSearch(fluids);

	applyForce(fluids, dt, G);

	predictPosition(fluids, dt, gridSpacing);

	applyConstraintLimitation(fluids, boundary, gridStartFluid, gridStartBoundary, gridResolution,
		gridLength, gridSpacing, predictPos, rho0, maxIterationNum, radius);

	updateVelocity_global << <(N - 1) / blockSize, blockSize >> > (predictPos, fluids->pos, fluids->vel, dt, fluids->particles_num);

	applyViscosity(fluids, gridStartFluid, gridResolution,
		gridLength, rho0, radius, 0.1f);

	copyPosition(fluids);

}

