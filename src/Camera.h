#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
class Locked_Center_Camera
{
public:
	
	void mouseCallBack(double xpos, double ypos);
	void mouseButtionCallBack(int button, int action, int mods);
	void scrollCallBack();

	Locked_Center_Camera(float w, float h, glm::vec3 center, glm::vec3 pos, glm::vec3 up);
	
	glm::mat4& GetViewMat();

	void Reset(glm::vec3 center, glm::vec3 pos, glm::vec3 up);
	
	inline glm::vec3& GetG() { return G; }

private:
	glm::vec3 centerPos;
	glm::vec3 cameraPos;
	glm::vec3 cameraUp;
	glm::vec3 cameraFront;
	glm::vec3 cameraRight;
	
	glm::vec3 worldUp;

	glm::vec3 G = glm::vec3(0, -1, 0);
	glm::vec3 GRight;

	glm::mat4 viewMat;

	float windowWidth, windowHeight;
	float lastX, lastY;
	bool mouseButtonPressed = false;
	bool firstProcecure = true;
	bool firstMouse = true;

	float yaw, pitch;
	float cameraRadius;
};
