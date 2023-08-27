#include "Camera.h"

glm::mat4& Locked_Center_Camera::GetViewMat()
{
	viewMat = glm::lookAt(cameraPos, centerPos, cameraUp);
	return viewMat;
}

Locked_Center_Camera::Locked_Center_Camera(float w, float h, glm::vec3 center, glm::vec3 pos, glm::vec3 up)
	:windowWidth(w), windowHeight(h), centerPos(center),cameraPos(pos) ,worldUp(up)
{
	cameraRadius = glm::distance(center, pos);
	cameraFront = glm::normalize(center - pos);
	cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
	GRight = glm::normalize(glm::cross(cameraFront, G));
	cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));
}

void Locked_Center_Camera::Reset(glm::vec3 center, glm::vec3 pos, glm::vec3 up)
{
	centerPos = center;
	cameraPos = pos;
	//cameraUp = up;
}

void Locked_Center_Camera::mouseCallBack(double xpos, double ypos)
{
	if (!mouseButtonPressed)
		return;
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	if (firstProcecure)
	{
		lastX = xpos;
		lastY = ypos;
		firstProcecure = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;
	
	lastX = xpos;
	lastY = ypos;
	
	float sensity = 0.2f;
	xoffset *= sensity;
	yoffset *= sensity;

	yaw += xoffset;
	pitch += yoffset;

	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
	if (yaw > 60.0f)
		yaw = 60;
	if (yaw < -60)
		yaw = -60;

	glm::vec3 front;
	front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	front.y = -sin(glm::radians(pitch));
	front.z = sin(glm::radians(yaw)) * cos(glm::radians(yaw));

	cameraFront = glm::normalize(-front);
	cameraRight = glm::normalize(glm::cross(cameraFront, worldUp));
	cameraUp = glm::normalize(glm::cross(cameraRight, cameraFront));
	cameraPos = centerPos -   cameraFront * cameraRadius;

	GRight = glm::normalize(glm::cross(cameraFront, -worldUp));
	G = glm::normalize(glm::cross(GRight, cameraFront));
}

void Locked_Center_Camera::mouseButtionCallBack(int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		mouseButtonPressed = true;
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE)
	{
		mouseButtonPressed = false;
		firstProcecure = true;
	}
}

void Locked_Center_Camera::scrollCallBack()
{

}