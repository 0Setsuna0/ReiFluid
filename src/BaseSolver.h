#pragma once
#include "cuda_Helper.h"
#include "Particles.h"
#include <memory>
class BaseSolver
{
public:
	

	virtual ~BaseSolver(){}

	virtual void step(Particles* fluids, Particles* boundary,
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
	) = 0;

	virtual void advect(Particles* fluids,float dt, float3 gridSpacing) = 0;
	virtual void applyForce(Particles* fluids,const float dt,const float3 G) = 0;

public:
	int blockSize = 256;
};