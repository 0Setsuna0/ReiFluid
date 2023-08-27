#include "Renderer.h"
#include "cuda_Helper.h"
#include <cuda_gl_interop.h>
Renderer::Renderer()
{
	model = glm::mat4(1);
	view = glm::lookAt(glm::vec3(2.2f, 2.2f, 2.2f),
		glm::vec3(0.5f, 0.5f, 0.5f),
		glm::vec3(0.0f, 1.0f, 0.0f));
	projection = glm::perspective(glm::radians(45.0f), (float)800 / (float)600, 0.1f, 100.0f);
	
}
Renderer::~Renderer()
{
	
}

void Renderer::RenderingInit(const int length)
{
	glEnable(GL_DEPTH_TEST);
	float points[] = {
	0.0f, 0.0f, 0.0f,
	1.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 1.0f,
	1.0f, 1.0f, 0.0f,
	1.0f, 0.0f, 1.0f,
	0.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f
	};
	glGenBuffers(1, &testvbo);
	glBindBuffer(GL_ARRAY_BUFFER, testvbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

	glGenVertexArrays(1, &testvbo);
	glBindVertexArray(tesetvao);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//create vbo and vao
	glGenBuffers(1, &particlesVBO);
	glBindBuffer(GL_ARRAY_BUFFER, particlesVBO);
	glBufferData(GL_ARRAY_BUFFER, length, nullptr, GL_DYNAMIC_DRAW);

	glGenVertexArrays(1, &particlesVAO);
	glBindVertexArray(particlesVAO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//register vbo with cuda
	CUDA_CHECK(cudaGLRegisterBufferObject(particlesVBO));

	testShader = new  Shader("./src/shader/vertex.glsl", "./src/shader/fragment.glsl");
}

void Renderer::RenderingDraw(const GLuint& posVBO, const GLuint& posVAO, int num, glm::mat4 viewm)
{
	glClearColor(0.75, 0.75, 0.75, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_POINT_SPRITE);

	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBindVertexArray(posVAO);
	testShader->use();
	testShader->setMat4("model", model);
	testShader->setMat4("view", viewm);
	testShader->setMat4("projection", projection);
	glDrawArrays(GL_POINTS, 0, num);
}
