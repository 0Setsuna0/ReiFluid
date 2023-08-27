#pragma once
#include "BaseSolver.h"
#include "Particles.h"
#include "Simulator_CUDA.cuh"
class SPHSolver:public BaseSolver
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

	SPHSolver() {};
	virtual ~SPHSolver() {};

	void advect(Particles* fluids,float dt, float3 gridSpacing) override;
	
	void applyForce(Particles* fluids, const float dt, const float3 G) override;
	
	void updateDensity(Particles* fluids, Particles* boundaries,
		int* gridStartFluid,
		int* gridEndFluid,
		int3 gridResolution,
		float gridLength,
		float radius
	) const;
	
	void applyPressure(Particles* fluids, Particles* boundaries,
		int* gridStartFluid,
		int* gridStartBoundary,
		float rho0,
		float stiff,
		int3 gridResolution,
		float gridLength,
		float radius,
		float dt
	);

	void applyViscosity(Particles* fluids, 
		int* gridStartFluid,
		int3 gridResolution,
		float gridLength,
		float rho0,
		float radius,
		float viscosityCoef,
		float dt
	);

};

