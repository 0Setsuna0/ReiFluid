#pragma once
#include "GLEW/glew.h"
#include "Shader.h"
#include "Camera.h"
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class Renderer
{
public:
	unsigned int testvbo, tesetvao;
	unsigned int particlesVBO, particlesVAO;
	Shader *testShader;
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 projection;
	Locked_Center_Camera* camera;
public:
	Renderer();
	~Renderer();
	void RenderingDraw(const GLuint& posVBO, const GLuint& posVAO, int num, glm::mat4 viewm);
	void RenderingDrawBoundary(const GLuint& posVBO, const GLuint& posVAO);
	void RenderingInit(const int length);
private:

};


