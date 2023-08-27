#pragma once
#include "SPHSolver.h"
#include "SimParams.cuh"
#include "PBFSolver.h"
#include "PCISPHSolver.h"
class SPHParticleSystem 
{
public:
	SPHParticleSystem(Particles* fluids,
		Particles* boundaries,
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
		float dt);

	float step(SimParams* sim);

	void initBoundary();

	void neighborSearching(Particles* particles, int* gridStart);
public:
	//partilces
	Particles* fluids;
	Particles* boundaries;

	//solver
	SPHSolver* sphSolver;
	PBFSolver* pbfSolver;
	PCISolver* pciSolver;

	//neighboring particle storage
	int* gridStartFluid;
	int* gridStartBoundary;
	int* tempBuffer;
	
	//grid cell attrib
	float3 gridSpacing;
	int3 gridResolution;
	float gridLength;

	//sph physical simulation coef
	float kernelRadius;
	float stiffness;
	float viscosityCoef;
	float rho0;
	float rhoB;
	float tension;
	float airPressure;

	//else
	float3 G;
	float dt;
	enum SolverType
	{
		SPH, PBF, PCISPH
	};
	SolverType solverType = PBF;
};