#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/wglew.h>


#include <cuda.h>

#include "Mouse.h"

#include <surface_functions.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

#include <chrono>
#include "Keyboard.h"

#include <ShellScalingApi.h>
#pragma comment(lib, "Shcore.lib")


inline int2 operator+(const int2 a, const int2 b);

class Window {
public:
	static Window* global;
public:
	Window(int width, int height, HINSTANCE hInst, int dev_width, int dev_height);
	~Window();
	bool Run();
	void UpdateWindowBuffer(uchar4* buffer);
	void UpdateInput();
	HWND getWindow();
	void moveWindow(int2 delta);
private:
	BOOL SetDpiAwareness();
	void EnableOpenGL(HWND hWnd, HDC* hDC, HGLRC* hRC);
	void GLEW_init();
	void DisableOpenGL(HWND hWnd, HDC hDC, HGLRC hRC);
private:
	static LRESULT StartMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
	static LRESULT MessageSetup(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
	LRESULT HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
private:
	void make_buffers(GLuint& VBO, GLuint& VAO);
	GLuint make_texture(int width, int height);
	void compile_shaders(GLuint& vertexShader, GLuint& fragmentShader);
	GLuint link_shaders(GLuint vertexShader, GLuint fragmentShader);
private:
	int win_width, win_height, dev_width, dev_height;
	int2 win_pos;
private:
	HWND hWnd;
	HDC hDC;
	HGLRC hRC;
	MSG msg;
	bool active = true;
private:
	GLuint VBO, VAO;
	GLuint texture;
	cudaGraphicsResource_t cudaResource;
	cudaArray_t texturePtr;
public:
	Keyboard kbd;
	Mouse mouse;
};
