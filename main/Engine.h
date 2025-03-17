#pragma once
#include "../system/Window.h"
#include "../render/Graphics.h"
#include "../math/Rect.h"
#include "../fluid2d/fluid2d.h"
#include "../system/Timer.h"
#include "../system/Terminal.h"

class Engine {
public:
	Engine(HINSTANCE hinst);
	void start();
private:
	void render();

private:
	void update();
private:
	Window* wnd;
	Keyboard* kbd;
	Mouse* mouse;
	Graphics* gfx;
	Timer time;
	Fluid2d fluid;
	Terminal cmd;
private:
	bool run;
};