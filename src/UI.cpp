#include "UI.h"
#include <iostream>
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"
UI::UI()
{

}

bool UI::init()
{
	if (!glfwInit())
	{
		std::cout << "GLFW init fail" << std::endl;
		return -1;
	}

	window_width = 1280;
	window_height = 720;
	window = glfwCreateWindow(window_width, window_height, "Rei_Fluid", NULL, NULL);
	if (!window)
	{
		std::cout << "GLFWWindow init fail" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (glewInit() != GLEW_OK)
	{
		std::cout << "GLEW init fail" << std::endl;
	}
	std::cout << glGetString(GL_VERSION) << std::endl;

	// imgui
	const char* glsl_version = "#version 130";

	// Setup Dear ImGui context
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Dear ImGui style
	ImGui::StyleColorsLight();

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(getWindow(), true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	return true;
}

void UI::NewImguiFrame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void UI::RenderImgui()
{
	ImGui::Begin("UI");
	ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}



void UI::loop()
{
	glfwSwapBuffers(window);
	glfwPollEvents();
}

void UI::close()
{
	// Cleanup
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();


	glfwTerminate();
}