//gridStart and gridEnd are used for looking through neighboring particles

#include "cuda_Helper.h"
#include <helper_functions.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include "SPHSolver.h"

//use neighboring particles to compute particle i's density
 __device__ void computeFluidDensity_device(float* density, const int p_i,
	float3* pos, float* mass, const int gridStart, const int gridEnd, const float kernelRadius)
{
	//input atrrib, p_i stands for particle i, gridStart stands for particle i's gridCell
	auto p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
			*density += mass[p_j] * cubicSplineKernel_device(length(pos[p_i] - pos[p_j]), kernelRadius);
	}
	
}

__device__ void computeBoundaryDensity_device(float* density, const float3 pos_i, float3* pos, 
	float* mass, const int gridStart, const int gridEnd, const float radius)
{
	unsigned int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*density += mass[p_j] * cubicSplineKernel_device(length(pos_i - pos[p_j]), radius);
	}
}

__global__ void computeDensity_global(float* density, const int N,
	float3* posF, float* massF, int* gridStartF,
	float3* posB, float* massB, int* gridStartB,
	const int3 gridResolution, const float gridLength, const float kernelRadius)
{
	const unsigned int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N) return;
	auto outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	int3 gridPos = make_int3(posF[p_i] / gridLength);
	//find related grids'particles
	for (int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				auto gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)
				{
					continue;
				}
				computeFluidDensity_device(&density[p_i], p_i, posF, massF, gridStartF[gridIdx], gridStartF[gridIdx + 1], kernelRadius);
				computeBoundaryDensity_device(&density[p_i], posF[p_i], posB, massB, gridStartB[gridIdx], gridStartB[gridIdx + 1], kernelRadius);

			}
	cudaDeviceSynchronize();

}

void SPHSolver::updateDensity(Particles* fluids, Particles* boundaries,
	int* gridStartFluid,
	int* gridStartBoundary,
	int3 gridResolution,
	float gridLength,
	float radius) const
{
	//N stands for the total particles'number
	int N = fluids->particles_num;
	//firstly, set all density to be zero
	thrust::fill(thrust::device, fluids->density, fluids->density + N, 0);
	computeDensity_global <<<(N - 1) / blockSize + 1, blockSize>>>(fluids->density, N,
		fluids->pos, fluids->mass, gridStartFluid, boundaries->pos, boundaries->mass, gridStartBoundary,
		gridResolution, gridLength, radius);
}

__global__ void handleBoundaryCondition_global(float3* pos, float3* vel, const int N, const float3 gridSpacing)
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

void SPHSolver::advect(Particles* fluids, float dt, float3 gridSpacing)
{
	fluids->advect(dt);
	handleBoundaryCondition_global << <(fluids->particles_num - 1) / blockSize + 1, blockSize >> > (fluids->pos, fluids->vel, fluids->particles_num, gridSpacing);
}

__global__ void computePressure_global(float* pressure, float* density, const int N,const int rho0, const float stiff)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	//use low stiffness equation of state to compute pressure,rho0 stands for the desired standard density
	pressure[p_i] = stiff * (powf((density[p_i] / rho0), 7) - 1.0f);
	if (pressure[p_i] < 0.0f)
		pressure[p_i] = 0.0f;
}

__device__ void accumulatePressureFluid_device(float3* force, const int p_i, float3* pos,
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

__device__ void accumulatePressureBoundary_device(float3* force, const float3 pos_i, float3* pos, float* mass,
	const float density, const float pressure, const int gridStart, const int gridEnd, const float radius)
{
	int p_j = gridStart;
	for ( p_j; p_j < gridEnd; p_j++)
	{
		*force += -mass[p_j] * (pressure / fmaxf( 1e-6f, density * density)) * cubicSplineKernelGradient_device(pos_i - pos[p_j], radius);
	}
}

__global__ void computePressureForce_global(float3* velF, float3* posF, float* massF, float* density,
	float* pressure, int* gridStartF, float3* posB, float* massB, int* gridStartB,
	const int3 gridResolution, const float gridLength, const float radius, const float dt, const int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N)return;
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	int3 gridPos = make_int3(posF[p_i] / gridLength);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	for (int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag) continue;
				accumulatePressureFluid_device(&force, p_i, posF, massF, density, pressure, gridStartF[gridIdx], gridStartF[gridIdx + 1], radius);
				accumulatePressureBoundary_device(&force, posF[p_i], posB, massB, density[p_i], pressure[p_i], gridStartB[gridIdx], gridStartB[gridIdx + 1], radius);
			}

	// copy
	if (length(force) > 1000.0f)
		force = normalize(force) * 1000.0f;

	velF[p_i] += force * dt;
}

__global__ void densityReset_global(float* p, int N)
{
	int p_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_i >= N)return;
	p[p_i] = 0.0f;
}
//apply pressure force
void SPHSolver::applyPressure(Particles* fluids, Particles* boundaries,
	int* gridStartFluid, int* gridStartBoundary, float rho0,
	float stiff, int3 gridResolution, float gridLength,
	float radius, float dt) 
{
	 int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	
	//update density
	thrust::fill(thrust::device, fluids->density, fluids->density + fluids->particles_num, 0.0f);
	computeDensity_global << <blockNum, blockSize >> > (fluids->density, N,
		fluids->pos, fluids->mass, gridStartFluid, boundaries->pos, boundaries->mass, gridStartBoundary,
		gridResolution, gridLength, radius);
	
	//compute pressure
	computePressure_global << <blockNum, blockSize >> > (fluids->pressure, fluids->density, N, rho0, stiff);

	//compute pressure force and apply it
	computePressureForce_global << <blockNum, blockSize >> > (fluids->vel, fluids->pos, fluids->mass, fluids->density, fluids->pressure, gridStartFluid, boundaries->pos, boundaries->mass, gridStartBoundary,
		gridResolution, gridLength, radius, dt, N);
}

__device__ void accumulateViscosity_device(float3* force, float3* pos, float3* vel, float* mass, const int p_i, const int gridStart, const int gridEnd,
	const float rho0, const float radius)
{
	int p_j = gridStart;
	for (p_j; p_j < gridEnd; p_j++)
	{
		*force += mass[p_i] * (vel[p_j] - vel[p_i]) / rho0 * viscosityLaplacian_device(length(pos[p_i] - pos[p_j]), radius);
	}
}

__global__ void computeViscosity_global(float3* pos, float3* vel,
	float* mass, int* gridStart, const int3 gridResolution, const float gridLength,
	const float rho0, const float radius, const float viscosityCoef,const float dt, const int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	int outFlag = __mul24(__mul24(gridResolution.x, gridResolution.y), gridResolution.z);
	int3 gridPos = make_int3(pos[p_i] / gridLength);
	float3 force = make_float3(0.0f, 0.0f, 0.0f);
	if (p_i >= N) return;
	for(int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)continue;
				accumulateViscosity_device(&force, pos, vel, mass, p_i, gridStart[gridIdx], gridStart[gridIdx + 1], rho0, radius);
			}
	vel[p_i] += viscosityCoef * force * dt;
}

void SPHSolver::applyViscosity(Particles* fluids,
	int* gridStartFluid, int3 gridResolution, float gridLength, float rho0,
	float radius, float viscosityCoef, float dt)
{
	int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	computeViscosity_global << <blockNum, blockSize >> > (fluids->pos, fluids->vel, fluids->mass,
		gridStartFluid, gridResolution, gridLength, rho0, radius, viscosityCoef, dt, N);
}

__global__ void computeGravity_global(float3* vel, const float3 G,const float dt,const int N)
{
	int p_i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (p_i >= N) return;
	vel[p_i] += dt * G;
}

void SPHSolver::applyForce(Particles* fluids, float dt, float3 G)
{
	int N = fluids->particles_num;
	int blockNum = (N - 1) / blockSize + 1;
	computeGravity_global << <blockNum, blockSize >> > (fluids->vel, G, dt, N);
}

void SPHSolver::step(Particles* fluids, Particles* boundary,
	int* gridStartFluid, int* gridStartBoundary, float3 gridSpacing, int3 gridResolution,
	float gridLength, float radius, float dt, float rho0, float rhoB, float stiffness,
	float viscosity, float3 G, float tension, float airPressure)
{
	applyForce(fluids, dt, G);

	applyPressure(fluids, boundary, gridStartFluid,
		gridStartBoundary, rho0, stiffness,
		gridResolution, gridLength, radius, dt);

	applyViscosity(fluids, gridStartFluid, gridResolution, 
		gridLength, rho0, radius,
		viscosity, dt);

	advect(fluids, dt, gridSpacing);
}