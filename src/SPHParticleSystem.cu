#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "device_atomic_functions.h"
#include "SPHParticleSystem.h"

SPHParticleSystem::SPHParticleSystem(Particles* mfluids,
	Particles* mboundaries,
	SPHSolver* solver,
	PBFSolver* pbfSolver,
	PCISolver* pciSolver,
	float3 gridSpacing,
	int3 gridResolution,
	float gridLength,
	float kernelRadius,
	float stiffness,
	float viscosityCoef,
	float rho0,
	float rhoB,
	float3 G,
	float dt)
	:fluids(mfluids), boundaries(mboundaries), sphSolver(solver), pbfSolver(pbfSolver), pciSolver(pciSolver),
	gridSpacing(gridSpacing), gridResolution(gridResolution), gridLength(gridLength),
	kernelRadius(kernelRadius), stiffness(stiffness), viscosityCoef(viscosityCoef), 
	rho0(rho0), rhoB(rhoB), G(G), dt(dt)
{
	CUDA_CHECK(cudaMalloc((void**)&gridStartFluid, sizeof(int) * (gridResolution.x * gridResolution.y * gridResolution.z + 1)));
	CUDA_CHECK(cudaMalloc((void**)&gridStartBoundary, sizeof(int) * (gridResolution.x * gridResolution.y * gridResolution.z + 1)));
	CUDA_CHECK(cudaMalloc((void**)&tempBuffer, sizeof(int) * max(fluids->particles_num + boundaries->particles_num, gridResolution.x * gridResolution.y * gridResolution.z + 1)));
	
	neighborSearching(boundaries, gridStartBoundary);
	initBoundary();
	
	const float m = 76.596750762082e-6f;
	thrust::fill(thrust::device, fluids->mass, fluids->mass + fluids->particles_num, m);

}

__global__ void gridCellAdd_global(int* gridStart, int* p2g,const int N)
{
	int i = blockIdx.x* blockDim.x + threadIdx.x; //one ddimension thread arrangement
	if (i >= N) return;
	//given a particle's index i, use p2g to tansfer i into grid index, note that particle i is in grid[gridIndex]
	atomicAdd(&gridStart[p2g[i]], 1);
}

//copy
void SPHParticleSystem::neighborSearching(Particles* particles, int* gridStart)
{
	int N = particles->particles_num;
	int blockNum = (N - 1) / sphSolver->blockSize + 1;
	int blockSize = sphSolver->blockSize;
	
	//step1:put the particles into gridcell(init p2g), p2g[p_i]=gidx stands for particle p_i is in grid[gidx]
	putParticles2GridCell_global<<<blockNum, blockSize>>>(particles->p2g, particles->pos, gridLength, gridResolution, N);
	//step2:sort temp p2g, also sort pos according to temp p2g
	cudaMemcpy(tempBuffer, particles->p2g, sizeof(int) * N, cudaMemcpyDeviceToDevice);
	THRUST_CHECK(thrust::sort_by_key(thrust::device, tempBuffer, tempBuffer + N, particles->pos);)
	//step3:sort temp p2g, also sort vel according to temp p2g
	cudaMemcpy(tempBuffer, particles->p2g, sizeof(int) * N, cudaMemcpyDeviceToDevice);
	THRUST_CHECK(thrust::sort_by_key(thrust::device, tempBuffer, tempBuffer + N, particles->vel);)
	
	//reset gridStart, note that gridStart is the actual grid storage structure, it storage the particle idx of each grid
	THRUST_CHECK(thrust::fill(thrust::device, gridStart, gridStart + gridResolution.x * gridResolution.y * gridResolution.z + 1, 0);)

	//update gridStart
	gridCellAdd_global<<<blockNum, blockSize>>>(gridStart, particles->p2g, N);
	//prefix sum
	thrust::exclusive_scan(thrust::device, gridStart, gridStart + gridResolution.x * gridResolution.y * gridResolution.z + 1, gridStart);
}

__device__ void accumulateBoundary_device(float* sum_kernel, const int& p_i, const int g_i, float3* pos, 
	int* gridStart, const int3& gridResolution, const float& kernelRadius)
{
	//out of boundary
	if (g_i == (gridResolution.x * gridResolution.y * gridResolution.z)) return;
	//look through all particles in grid g_i
	int p_j = gridStart[g_i];
	for (p_j; p_j < gridStart[g_i + 1]; p_j++)
	{
		*sum_kernel += cubicSplineKernel_device(length(pos[p_i] - pos[p_j]), kernelRadius);
	}
}

__global__ void computeBoundaryMass_global(float* mass, float3* pos, int* gridStart, const int3 gridResolution, 
	const float gridLength, const float rhoB, const float kernelRadius, const int N)
{
	int p_i = blockIdx.x* blockDim.x + threadIdx.x;
	if (p_i >= N) return;
	int3 gridPos = make_int3(pos[p_i] / gridLength);
	int outFlag = gridResolution.x * gridResolution.y * gridResolution.z;
	for(int a = -1; a < 2; a++)
		for(int b = -1; b < 2; b++)
			for (int c = -1; c < 2; c++)
			{
				int gridIdx = particleIndex2GridCell_device(gridPos + make_int3(a, b, c), gridResolution);
				if (gridIdx == outFlag)continue;
				accumulateBoundary_device(&mass[p_i], p_i, gridIdx, pos, gridStart, gridResolution, kernelRadius);
			}
	mass[p_i] = rhoB / fmaxf( 1e-6 ,mass[p_i]);

}

void SPHParticleSystem::initBoundary()
{
	computeBoundaryMass_global<<<(boundaries->particles_num - 1) / sphSolver->blockSize + 1, sphSolver->blockSize>>>(boundaries->mass, 
		boundaries->pos, gridStartBoundary, gridResolution, gridLength, rhoB, kernelRadius, boundaries->particles_num);
}

float SPHParticleSystem::step(SimParams* sim)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	G = sim->G;
	neighborSearching(fluids, gridStartFluid);
	switch (solverType)
	{
	case SPH:
		sphSolver->step(fluids, boundaries, gridStartFluid, gridStartBoundary, gridSpacing,
			gridResolution, gridLength, kernelRadius, dt, rho0, rhoB, stiffness, viscosityCoef, G, tension, airPressure);
		break;
	case PBF:
		pbfSolver->step(fluids, boundaries, gridStartFluid, gridStartBoundary, gridSpacing,
			gridResolution, gridLength, kernelRadius, dt * 5, rho0, rhoB, stiffness, viscosityCoef, G, tension, airPressure);
		break;
	case PCISPH:
		pciSolver->step(fluids, boundaries, gridStartFluid, gridStartBoundary, gridSpacing,
			gridResolution, gridLength, kernelRadius, dt , rho0, rhoB, stiffness, viscosityCoef, G, tension, airPressure);
		break;
	default:
		break;
	}

	cudaDeviceSynchronize();


	float milliseconds;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return milliseconds;
}

