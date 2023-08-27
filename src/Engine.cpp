#include "Engine.h"
#include <cuda_gl_interop.h>
Engine::Engine()
{
	ui = UI();
	//camera = Camera();
	renderer = Renderer();
	//parameter reference from: https://github.com/zhai-xiao/CPP-Fluid-Particles
	const float3 spaceSize = make_float3(1.0f);
	const float sphSpacing = 0.02f;
	const float sphSmoothingRadius = 2.0f * sphSpacing;
	const float sphCellLength = 1.01f * sphSmoothingRadius;
	const float dt = 0.002f;
	const float sphRho0 = 1.0f;
	const float sphRhoBoundary = 1.4f * sphRho0;
	const float sphM0 = 76.596750762082e-6f;
	const float sphStiff = 10.0f;
	const float3 sphG = make_float3(0.0f, -9.8f, 0.0f);
	const float sphVisc = 5e-4f;
	const float sphSurfaceTensionIntensity = 0.0001f;
	const float sphAirPressure = 0.0001f;
	const int3 cellSize = make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));
	sim = new SimParams(spaceSize, sphSpacing, sphSmoothingRadius, sphCellLength, dt, sphRho0, sphRhoBoundary,
		sphStiff, sphG, sphVisc, sphSurfaceTensionIntensity, sphAirPressure, cellSize);
	InitParticleSystem();

}

void Engine::InitParticleSystem()
{
	//init particles
	std::vector<float3> pos;
	for (auto i = 0; i < 36; ++i) {
		for (auto j = 0; j < 24; ++j) {
			for (auto k = 0; k < 24; ++k) {
				auto x = make_float3(0.27f + 0.02f * j,
					0.10f + 0.02f * i,
					0.27f + 0.02f * k);
				pos.push_back(x);
			}
		}
	}
	fluids = new Particles(pos);
	
	
	float3 spaceSize = sim->gridSpacing;
	float sphCellLength = sim->gridLength;
	printf("grid space: (%f,%f,%f), length:%f\n", spaceSize.x, spaceSize.y, spaceSize.z, sphCellLength);
	pos.clear();
	const auto compactSize = 2 * make_int3(ceil(spaceSize.x / sphCellLength), ceil(spaceSize.y / sphCellLength), ceil(spaceSize.z / sphCellLength));
	// front and back
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.y; ++j) {
			auto x = make_float3(i, j, 0) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, j, compactSize.z - 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// top and bottom
	for (auto i = 0; i < compactSize.x; ++i) {
		for (auto j = 0; j < compactSize.z - 2; ++j) {
			auto x = make_float3(i, 0, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(i, compactSize.y - 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	// left and right
	for (auto i = 0; i < compactSize.y - 2; ++i) {
		for (auto j = 0; j < compactSize.z - 2; ++j) {
			auto x = make_float3(0, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
			x = make_float3(compactSize.x - 1, i + 1, j + 1) / make_float3(compactSize - make_int3(1)) * spaceSize;
			pos.push_back(0.99f * x + 0.005f * spaceSize);
		}
	}
	boundaries = new Particles(pos);

	//init solver
	sphSolver = new SPHSolver();
	pbfSolver = new PBFSolver(fluids);
	pciSolver = new PCISolver(fluids);

	//init particle system
	ps = new SPHParticleSystem(fluids, boundaries, sphSolver, pbfSolver, pciSolver, sim->gridSpacing, sim->gridResolution,
		sim->gridLength, sim->kernelRadius, sim->stiffness, sim->viscosity, sim->rho0,
		sim->rhoB, sim->G, sim->dt);
}

void Engine::Init()
{
	ui.init();
	renderer.RenderingInit(sizeof(float3) * ps->fluids->particles_num);
}

void Engine::Simulate()
{
	++frameId;


	auto milliseconds = ps->step(sim);
	
	totalTime += milliseconds;

	printf("Frame %d - %2.2f ms, avg time - %2.2f ms/frame (%3.2f FPS)\r",
		frameId % 10000, milliseconds, totalTime / float(frameId), float(frameId) * 1000.0f / totalTime);
}

void Engine::Rendering(glm::mat4& viewm) 
{
	float3* posPtr;
	
	cudaGLMapBufferObject((void**)&posPtr, renderer.particlesVBO);
	fluids->copyPosition(posPtr);
	cudaGLUnmapBufferObject(renderer.particlesVBO);

	renderer.RenderingDraw(renderer.particlesVBO, renderer.particlesVAO, ps->fluids->particles_num, viewm);
}

void Engine::Update(Locked_Center_Camera* camera)
{
	while (!ui.windowClosed())
	{
		ui.loop();
		ui.NewImguiFrame();
		
		sim->G = make_float3(camera->GetG().x, camera->GetG().y, camera->GetG().z) * 9.8f;
		Simulate();
		Rendering(camera->GetViewMat());
		

		ui.RenderImgui();
	}
}

void Engine::Close()
{
	ui.close();
}
