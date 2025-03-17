#include "Window.h"

inline int2 operator+(const int2 a, const int2 b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

Window::Window(int width, int height, HINSTANCE hInst, int dev_width, int dev_height) :
	win_width(width),
	win_height(height),
	dev_width(dev_width),
	dev_height(dev_height),
	kbd(256)
{
	win_pos.x = 0;
	win_pos.y = 0;
	Window::global = this;
	WNDCLASS wc;
	wc.style = CS_OWNDC;
	wc.lpfnWndProc = StartMessage;
	wc.cbClsExtra = 0;
	wc.cbWndExtra = 0;
	wc.hInstance = hInst;
	wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	wc.hCursor = LoadCursor(NULL, IDC_ARROW);
	wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
	wc.lpszMenuName = NULL;
	wc.lpszClassName = "GL";
	RegisterClass(&wc);

	SetDpiAwareness();

	hWnd = CreateWindow(
		"GL", "Window",
		WS_POPUP,
		0, 0, width, height,
		NULL, NULL, hInst, NULL);


	EnableOpenGL(hWnd, &hDC, &hRC);
	GLEW_init();

	GLuint vertexShader, fragmentShader;
	compile_shaders(vertexShader, fragmentShader);
	GLuint shaderProgram = link_shaders(vertexShader, fragmentShader);
	make_buffers(VBO, VAO);

	texture = make_texture(dev_width, dev_height);

	cudaGraphicsGLRegisterImage(&cudaResource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaResource, 0);
	cudaGraphicsSubResourceGetMappedArray(&texturePtr, cudaResource, 0, 0);

	glBindVertexArray(VAO);
	glBindTexture(GL_TEXTURE_2D, texture);
	glUseProgram(shaderProgram);

	ShowWindow(hWnd, SW_SHOWDEFAULT);//SW_SHOWMAXIMIZED);//SW_SHOWDEFAULT
	SetWindowPos(hWnd, NULL, 100, 0, width, height, SWP_NOMOVE | SWP_NOZORDER);
	wglSwapIntervalEXT(0);
}

Window::~Window() {
	DisableOpenGL(hWnd, hDC, hRC);
	DestroyWindow(hWnd);
}

bool Window::Run() {
	return active;
}

void Window::UpdateWindowBuffer(uchar4* buffer)
{
	cudaMemcpyToArray(texturePtr, 0, 0, buffer, sizeof(uchar4) * dev_width * dev_height, cudaMemcpyDeviceToDevice);

	cudaDeviceSynchronize();

	glDrawArrays(GL_TRIANGLES, 0, 6);

	glFlush();
}

void Window::UpdateInput()
{
	if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
	{
		if (msg.message == WM_QUIT)
		{
			active = false;
		}
		else
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}
}

HWND Window::getWindow()
{
	return hWnd;
}

void Window::moveWindow(int2 delta)
{
	win_pos = win_pos + delta;
	MoveWindow(hWnd, win_pos.x, win_pos.y, win_width, win_height, true);
}

BOOL Window::SetDpiAwareness()
{
	HRESULT result = SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
	return SUCCEEDED(result);
}

void Window::EnableOpenGL(HWND hWnd, HDC* hDC, HGLRC* hRC)
{
	PIXELFORMATDESCRIPTOR pfd;
	int iFormat;

	*hDC = GetDC(hWnd);

	ZeroMemory(&pfd, sizeof(pfd));
	pfd.nSize = sizeof(pfd);
	pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER_DONTCARE;
	pfd.iPixelType = PFD_TYPE_RGBA;
	pfd.cColorBits = 24;
	pfd.cDepthBits = 16;
	pfd.iLayerType = PFD_MAIN_PLANE;

	iFormat = ChoosePixelFormat(*hDC, &pfd);
	SetPixelFormat(*hDC, iFormat, &pfd);

	*hRC = wglCreateContext(*hDC);
	wglMakeCurrent(*hDC, *hRC);
}

void Window::GLEW_init()
{
	glewExperimental = GL_TRUE;
	glewInit();
}

void Window::DisableOpenGL(HWND hWnd, HDC hDC, HGLRC hRC)
{
	wglMakeCurrent(NULL, NULL);
	wglDeleteContext(hRC);
	ReleaseDC(hWnd, hDC);
}

LRESULT Window::StartMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (msg == WM_NCCREATE)
	{
		const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
		Window* const pWnd = reinterpret_cast<Window*>(pCreate->lpCreateParams);
		SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pWnd));
		SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&Window::MessageSetup));
		return Window::global->HandleMessage(hwnd, msg, wParam, lParam);
	}
	return DefWindowProcA(hwnd, msg, wParam, lParam);
}

LRESULT Window::MessageSetup(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	// retrieve ptr to window class
	Window* const pWnd = reinterpret_cast<Window*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
	// forward message to window class handler
	return Window::global->HandleMessage(hwnd, msg, wParam, lParam);
}

LRESULT Window::HandleMessage(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_DESTROY:
	{
		PostQuitMessage(0);
	}break;
	case WM_QUIT:
	{
		PostQuitMessage(0);
	}break;
	case WM_SIZE:
	{

	}break;
	// ************ KEYBOARD MESSAGES ************ //
	case WM_KEYDOWN:
	{
		switch (wParam) {
		case VK_ESCAPE: {
			PostQuitMessage(0);
		}break;
		default: {
			unsigned char key = static_cast<unsigned char>(wParam);
			kbd.OnKeyPressed(key);
		}
		}
	}break;
	case WM_KEYUP: {
		kbd.OnKeyReleased(static_cast<char>(wParam));
		break;
	}
	case WM_LBUTTONDOWN:
	{
		mouse.setLeftKey(true);
		SetForegroundWindow(hwnd);
		break;
	}
	case WM_RBUTTONDOWN:
	{
		mouse.setRightKey(true);
		break;
	}
	case WM_LBUTTONUP:
	{
		mouse.setLeftKey(false);
		break;
	}
	case WM_RBUTTONUP:
	{
		mouse.setRightKey(false);
		break;
	}
	case WM_MOUSEWHEEL:
	{
		int scroll = GET_WHEEL_DELTA_WPARAM(wParam);
		mouse.setScroll(scroll);
	}
	}
	return DefWindowProcA(hwnd, msg, wParam, lParam);
}

void Window::make_buffers(GLuint& VBO, GLuint& VAO)
{
	GLfloat vertices[] = {
			 1.0f,  1.0f, 0.0f,   0.0f, 0.0f, 1.0f,  1.0f, -1.0f,
			 1.0f, -1.0f, 0.0f,   0.1f, 0.0f, 0.0f,  1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,   0.0f, 1.0f, 0.0f,  0.0f, -1.0f,
			 1.0f, -1.0f, 0.0f,   0.1f, 0.1f, 0.0f,  1.0f, 0.0f,
			-1.0f, -1.0f, 0.0f,   0.1f, 0.0f, 1.0f,  0.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,   0.0f, 1.0f, 0.0f,  0.0f, -1.0f
	};
	glGenBuffers(1, &VBO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);
}

GLuint Window::make_texture(int width, int height)
{
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	return texture;
}

void Window::compile_shaders(GLuint& vertexShader, GLuint& fragmentShader)
{
	const GLchar* vertexShaderSource = "#version 330 core\n"
		"layout (location = 0) in vec3 position;\n"
		"layout (location = 1) in vec3 color;\n"
		"layout (location = 2) in vec2 texCoord;\n"
		"out vec3 ourColor;\n"
		"out vec2 TexCoord;\n"
		"void main()\n"
		"{\n"
		"gl_Position = vec4(position.x, position.y, 0.0f, 1.0);\n"
		"ourColor = color;\n"
		"TexCoord = texCoord;\n"
		"}\0";
	const GLchar* fragmentShaderSource =
		"#version 330 core\n"
		"in vec3 ourColor;\n"
		"in vec2 TexCoord;\n"
		"out vec4 color;\n"
		"uniform sampler2D ourTexture;\n"
		"void main()\n"
		"{\n"
		"    color = texture(ourTexture, TexCoord);\n"
		"}\0";
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);
}

inline GLuint Window::link_shaders(GLuint vertexShader, GLuint fragmentShader) {
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	// Check for linking errors
	GLint success;
	GLchar infoLog[512];
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
	}
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

Window* Window::global = nullptr;