#pragma once
#include <GLEW/glew.h>
#include <GLFW/glfw3.h>
#include "Camera.h"



class UI
{
public:
	UI();
	bool init();
	void loop();
	void close();

	inline unsigned int getScreenHeight() { return window_height; }
	inline unsigned int getScreenWidth() { return window_width; }

	inline bool windowClosed() { return glfwWindowShouldClose(window); }

	inline GLFWwindow* getWindow() { return window; }
	void NewImguiFrame();
	void RenderImgui();

private:
	unsigned int window_width, window_height;
	GLFWwindow* window;


};

