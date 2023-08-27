#include <vector>
#include "cuda_Helper.h"
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "Particles.h"
#include "device_launch_parameters.h"

__global__ void advect_global(float3* pos, float3* vel, const float dt, const int N)
{
	int p_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_i >= N)return;

	vel[p_i] *= 0.99997f;
	pos[p_i] += vel[p_i] * dt;
}

void Particles::advect(float dt)
{
	int N = particles_num;
	advect_global<<<(N - 1) / 256 + 1, 256>>>(pos, vel, dt, N);

}

__global__ void copyPos_global(float3* rPos, float3* pos, const int N)
{
	int p_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_i >= N)return;
	rPos[p_i] = pos[p_i];

}

void Particles::copyPosition(float3* rPos)
{
	int N = particles_num;
	copyPos_global<<<(N - 1) / 256 + 1, 256 >>>(rPos, pos, N);
	cudaDeviceSynchronize();
}

__global__ void log_global(float3* pos, int N)
{

	int p_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_i >= N)return;
	printf(" Particle[%d]'s position: (%f, %f, %f)\n", p_i, pos[p_i].x, pos[p_i].y, pos[p_i].z);
}

void Particles::logPosition()
{
	int N = particles_num;
	log_global << <(N - 1) / 256 + 1, 256 >> > (pos, N);
}

__global__ void debugPos_gloabal(float3* pos, int N)
{
	int p_i = blockDim.x * blockIdx.x + threadIdx.x;
	if (p_i >= N)return;
	pos[p_i] += make_float3(0, -0.001f, 0);
}

void Particles::debugAdvect()
{
	int N = particles_num;
	debugPos_gloabal << <(N - 1) / 256 + 1, 256 >> > (pos, N);
}