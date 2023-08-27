#pragma once
#include "UI.h"
#include "Camera.h"
#include "Renderer.h"
#include "SPHParticleSystem.h"
class Engine
{
	
public:
	SimParams* sim;
	Particles* fluids;
	Particles* boundaries;
	SPHSolver* sphSolver;
	PBFSolver* pbfSolver;
	PCISolver* pciSolver;
	SPHParticleSystem* ps;
	UI ui;
	Renderer renderer;

public:
	Engine();
	void Init();
	void Update(Locked_Center_Camera* camera);
	void Close();

	void InitParticleSystem();
	void Simulate();
	void Rendering(glm::mat4& viewm);
private:
	int frameId = 0;
	float totalTime = 0.0f;
};

