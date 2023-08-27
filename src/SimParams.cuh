#include <vector_types.h>
class SimParams
{
public:
	float3 gridSpacing;
	float sphSpacing;
	float kernelRadius;
	float gridLength;
	float dt;
	float rho0;
	float rhoB;
	float stiffness;
	float3 G;
	float viscosity;
	float tension;
	float airPressure;
	int3 gridResolution;
	
	SimParams(float3 gridSpacing,
		float sphSpacing,
		float kernelRadius,
		float gridLength,
		float dt,
		float rho0,
		float rhoB,
		float stiffness,
		float3 G,
		float viscosity,
		float tension,
		float airPressure, int3 gridResolution)
		:gridSpacing(gridSpacing), sphSpacing(sphSpacing),
		kernelRadius(kernelRadius), gridLength(gridLength),
		dt(dt), rho0(rho0), rhoB(rhoB),stiffness(stiffness),
		G(G),viscosity(viscosity),tension(tension),airPressure(airPressure),
		gridResolution(gridResolution)
	{

	}
};