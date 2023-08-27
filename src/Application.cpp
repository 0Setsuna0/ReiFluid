#include "Engine.h"

Locked_Center_Camera* camera;


void mouse_pos_call_back(GLFWwindow* window, double xpos, double ypos);
void mouse_button_call_back(GLFWwindow* window, int button, int action, int mods);

int main()
{
	camera = new Locked_Center_Camera(800, 600, glm::vec3(0.5f, 0.5f, 0.5f), 
		glm::vec3(2.2f, 2.2f, 2.2f), glm::vec3(0.0f, 1.0f, 0.0f));

	Engine engine;
	engine.Init();
	
	glfwSetCursorPosCallback(engine.ui.getWindow(), mouse_pos_call_back);
	glfwSetMouseButtonCallback(engine.ui.getWindow(), mouse_button_call_back);

	engine.Update(camera);
	engine.Close();
}

void mouse_pos_call_back(GLFWwindow* window, double xpos, double ypos)
{
	camera->mouseCallBack(xpos, ypos);
}

void mouse_button_call_back(GLFWwindow* window, int button, int action, int mods)
{
	camera->mouseButtionCallBack(button, action, mods);
}
