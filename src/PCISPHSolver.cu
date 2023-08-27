#define __CUDACC__
#include "PCISPHSolver.h"
#include "cuda_Helper.h"
#include "Simulator_CUDA.cuh"
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define LOG_FLOAT3(fx) printf("(%f, %f, %f)\n", fx.x, fx.y, fx.z)

__global__ void handleBoundaryConditionPCI_global(float3* pos, float3* vel, const int N, const float3 gridSpacing)
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

__global__ void applyGravity_global(float3* f, float3 G, int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	f[p_i] = G;
}

void PCISolver::applyForce(Particles* fluids, const float dt, const float3 G)
{
	applyGravity_global<<<(fluids->particles_num - 1) / blockSize, blockSize>>>(F, G, fluids->particles_num);
}

__device__ void accumulateViscosityPCI_device(float3* force, float3* pos, float3* vel, float* mass, const int p_i, const int gridStart, const int gridEnd,
	const float rho0, const float radius)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*force += mass[p_i] * (vel[p_j] - vel[p_i]) / rho0 * viscosityLaplacian_device(length(pos[p_i] - pos[p_j]), radius);
	}
}

__global__ void computeViscosityPCI_global(float3* pos, float3* vel,
	float* mass, int* gridStart, const int3 gridResolution, const float gridLength,
	const float rho0, const float radius, const float viscosityCoef, const float dt, const int N, float3* f)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int outFlag = __mul24(__mul24(gridResolution.x, gridResolution.y), gridResolution.z);
	int3 gridPos = make_int3(pos[p_i] / gridLength);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	if (p_i >= N) return;
	for (int a = -1; a < 2; a++)
		for (int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)continue;
				accumulateViscosityPCI_device(&force, pos, vel, mass, p_i, gridStart[gridIdx], gridStart[gridIdx + 1], rho0, radius);
			}
	f[p_i] += viscosityCoef * force;
}

void PCISolver::applyViscosity(Particles* fluids,
	int* gridStartFluid, int3 gridResolution, float gridLength, float rho0,
	float radius, float viscosityCoef, float dt)
{
	int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	computeViscosityPCI_global << <blockNum, blockSize >> > (fluids->pos, fluids->vel, fluids->mass,
		gridStartFluid, gridResolution, gridLength, rho0, radius, viscosityCoef, dt, N, F);
}

__global__ void debugFunc(float* p, float3* fp, int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N) return;
	if (p_i % 12000 == 0)
	{
		printf("pressure[%d] is %f \n", p_i, p[p_i]);
		printf("force[%d] (%f, %f, %f)\n", p_i, fp[p_i].x, fp[p_i].y, fp[p_i].z);
	}

}
void PCISolver::initBuffer(Particles* fluids)
{
	int N = fluids->particles_num;
	THRUST_CHECK(thrust::fill(thrust::device, fluids->pressure, fluids->pressure + N, 0.0f);)
	THRUST_CHECK(thrust::fill(thrust::device, Fp, Fp + fluids->particles_num, make_float3(0.0f, 0.0f, 0.0f));)
	//debugFunc<<<(N -1) / blockSize, blockSize>>>(fluids->pressure, Fp, N);
}



__global__ void predict_global(float3* pVel, float3* pPos, float3* vel, float3* pos,
	float3* f, float3* fp, float dt, int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	pVel[p_i] = vel[p_i] + (f[p_i] + fp[p_i]) * dt;
	pPos[p_i] = pos[p_i] + pVel[p_i] * dt;
}

__device__ void accumulateWeight_device(float3* pPos, float* weightSum, float* mass, float radius, int gridStart, int gridEnd, int p_i)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*weightSum += mass[p_j] * cubicSplineKernel_device(length(pPos[p_j] - pPos[p_i]), radius);
	}
}

__device__ void accumulateWeightB_device(float3* bPos, float3 pPos_i, float* mass, float* weightSum, float radius, int gridStart, int gridEnd)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*weightSum += mass[p_j] * cubicSplineKernel_device(length(bPos[p_j] - pPos_i), radius);
	}
}

__global__ void computePressurePCI_global(float3* pPos, float* mass, int * gridStart, float3* bPos, float* massB, int* gridStartB, int3 gridResolution, float gridLength,
	float rho0, float radius, int N, float delta, float* densityError, float* pressure, float* density)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	float densityt = 0.0f;
	int outFlag = __mul24(__mul24(gridResolution.x, gridResolution.y), gridResolution.z);
	for(int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(make_int3(pPos[p_i] / gridLength) + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)
					continue;
				accumulateWeight_device(pPos, &densityt, mass, radius, gridStart[gridIdx], gridStart[gridIdx + 1], p_i);
				accumulateWeightB_device(bPos, pPos[p_i], massB, &densityt, radius, gridStartB[gridIdx], gridStartB[gridIdx + 1]);
			}
	float densityErrort = densityt - rho0;
	float pressuret = delta * densityErrort;

	if (pressuret < 0.0)
	{
		pressuret *= -0.0001f;
		densityErrort *= -0.0001f;
	}

	pressure[p_i] += pressuret;
	density[p_i] = densityt;
	densityError[p_i] = densityErrort;
}

__device__ void accumulatePressureFluidPCI_device(float3* force, const int p_i, float3* pos,
	float* mass, float* density, float* pressure, const int gridStart, const int gridEnd, const float radius)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		if (p_i == p_j)
			continue;
		//clamp, avoid dividing by zero
		*force += -mass[p_j] * (pressure[p_i] / fmaxf(1e-6f, density[p_i] * density[p_i]) + pressure[p_j] / fmaxf(1e-6f, density[p_j] * density[p_j]))
			* cubicSplineKernelGradient_device(pos[p_i] - pos[p_j], radius);
	}
}

__device__ void accumulatePressureBoundaryPCI_device(float3* force, const float3 pos_i, float3* pos, float* mass,
	const float density, const float pressure, const int gridStart, const int gridEnd, const float radius)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*force += -mass[p_j] * (pressure / fmaxf(1e-6f, density * density)) * cubicSplineKernelGradient_device(pos_i - pos[p_j], radius);
	}
}

__global__ void computePressureForcePCI_global(float3* fp, float3* posF, float* massF, float* density,
	float* pressure, int* gridStartF, float3* posB, float* massB, int* gridStartB,
	const int3 gridResolution, const float gridLength, const float radius, const int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	int3 gridPos = make_int3(posF[p_i] / gridLength);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	for (int a = -1; a < 2; a++)
		for (int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag) continue;
				accumulatePressureFluidPCI_device(&force, p_i, posF, massF, density, pressure, gridStartF[gridIdx], gridStartF[gridIdx + 1], radius);
				accumulatePressureBoundaryPCI_device(&force, posF[p_i], posB, massB, density[p_i], pressure[p_i], gridStartB[gridIdx], gridStartB[gridIdx + 1], radius);
			}

	// copy
	if (length(force) > 1000.0f)
		force = normalize(force) * 1000.0f;

	fp[p_i] = force;
}

void PCISolver::correctionIterate(Particles* fluids, int* gridStartF, Particles* boundary, int* gridStartB, int3 gridResolution,
	float gridLength, float rho0, float radius, float dt, float3 gridSpacing)
{
	int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	float delta;
	computeDelta(gridLength, rho0, radius, dt, gridSpacing, &delta);
	printf("delta is %f\n", delta);
	delta = 30;
	for (int i = 0; i < maxIterationNum; i++)
	{
		//predict velosity and position
		predict_global << <blockNum, blockSize >> > (pVel, pPos, fluids->vel, fluids->pos, F, Fp, dt, N);

		//enforce boundary condition
		handleBoundaryConditionPCI_global<<<blockNum, blockSize>>>(pPos, pVel, N, gridSpacing);
		
		//compute pressure from density error
		computePressurePCI_global<<<blockNum, blockSize>>>(pPos, fluids->mass, gridStartF, boundary->pos, boundary->mass, gridStartB, gridResolution, gridLength, rho0, 
			radius, N, delta, densityError, fluids->pressure, pDensity);

		//compute pressure force
		computePressureForcePCI_global<<<blockNum, blockSize>>>(Fp, fluids->pos, fluids->mass, pDensity, fluids->pressure, gridStartF, boundary->pos,
			boundary->mass, gridStartB, gridResolution, gridLength, radius, N);

		//compute max density error
		float* iter = thrust::max_element(thrust::device, densityError, densityError + N);
		float maxDensityError;
		cudaMemcpy(&maxDensityError, iter, sizeof(float), cudaMemcpyDeviceToHost);
		densityErrorRatio = maxDensityError / rho0;

		if (std::fabs(densityErrorRatio) < maxDensityErrorRatio)
			break;
	}
}

__global__ void updateVel_global(float3* f, float3* fp, float3* vel, float dt, int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	vel[p_i] += (f[p_i] + fp[p_i]) * dt;
}

void PCISolver::computeVel(Particles* fluids, float dt)
{
	updateVel_global<<<(fluids->particles_num - 1) / blockSize + 1, blockSize>>>(F, Fp, fluids->vel, dt, 
		fluids->particles_num);
}

void PCISolver::advect(Particles* fluids, float dt, float3 gridSpacing)
{
	fluids->advect(dt);
	handleBoundaryConditionPCI_global << <(fluids->particles_num - 1) / blockSize + 1, blockSize >> > (fluids->pos, fluids->vel,
		fluids->particles_num, gridSpacing);
}

void PCISolver::step(Particles* fluids, Particles* boundary,
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
	//1: compute force which includes gravity and viscosity
	applyForce(fluids, dt, G);

	applyViscosity(fluids, gridStartFluid, gridResolution, gridLength, rho0, radius, viscosity, dt);

	//2: initialize pressure and pressure force
	initBuffer(fluids);

	//3: iteration, compute pressure force
	correctionIterate(fluids, gridStartFluid, boundary, gridStartBoundary, gridResolution, gridLength, rho0, radius,
		dt, gridSpacing);

	//4: finally update actual velocity and position of particle
	computeVel(fluids, dt);

	advect(fluids, dt, gridSpacing);
}

void PCISolver::computeDelta(float gridLength, float rho0, float radius, float dt, float3 gridSpacing, float* delta)
{
	float denom = 0;
	float3 denom1;
	float denom2 = 0;

	for(int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				float3 samplePoint = make_float3(a, b, c) * gridLength;
				float distance = length(samplePoint);
				if (distance > radius) continue;
				float3 direction = (distance > 0.0) ? samplePoint / distance : make_float3(0.0f, 0.0f, 0.0f);

				float3 gradWij = cubicSplineKernelGradient_device(direction, radius);
				denom1 += gradWij;
				denom2 += dot(gradWij, gradWij);
			}
	denom += -dot(denom1, denom1) - denom2;
	*delta =  - (0.5 * rho0 * rho0) / (dt * dt * 76.59e-6f * 76.59e-6f) * (1.0f / denom);
}