#include "fluid2d.h"
#include <thread>

Fluid2d::Fluid2d():
	time(128),
	gfx(nullptr),
	active(false),
	tps(0.0f)
{
	bariers = new Rect[3];
	bariers_count = 3;
	float width = 0.02f;

	float size = 0.9;
	bariers[0].start = vec2(-1, -1) * size;
	bariers[0].end = (vec2(-1, 1) + vec2(width, 0)) * size;

	bariers[1].start = (vec2(1, -1) - vec2(width, 0)) * size;
	bariers[1].end = vec2(1, 1) * size;

	bariers[2].start = (vec2(-1, 1) - vec2(0, width)) * size;
	bariers[2].end = vec2(1, 1) * size;

	drops_count = 1;
	drops = new Drop[drops_count];
	drod_radius = 0.01;
	
}

Fluid2d::~Fluid2d()
{
	active = false;
	while (!end);
	delete[] bariers, drops;
}

void Fluid2d::init(Graphics* gfx)
{
	this->gfx = gfx;
}

void Fluid2d::start()
{
	active = true;
	g = vec2(0, 9.81) / 1000;
	std::thread th(&Fluid2d::simulate, this);
	th.detach();
}

void Fluid2d::render()
{
	for (int i = 0; i < bariers_count; i++) {
		gfx->renderRect(grey, bariers[i].start, bariers[i].end);
	}

	for (int i = 0; i < drops_count; i++) {
		gfx->renderElips(blue, drops[i].pos, drod_radius);
	}
}

void Fluid2d::stop()
{
	active = false;
}

float* Fluid2d::getTps()
{
	return &tps;
}

void Fluid2d::reset()
{
	drops[0].pos = vec2(0, 0);
	drops[0].veclosity = vec2(0, 0);
}

int Fluid2d::dropBarierCollision(int id)
{
	for (int i = 0; i < bariers_count; i++) {
		if (drops[id].pos.x >= bariers[i].start.x && drops[id].pos.x <= bariers[i].end.x) {
			if (drops[id].pos.y >= bariers[i].start.y && drops[id].pos.y <= bariers[i].end.y) {
				return i;
			}
		}
	}
	return -1;
}

void Fluid2d::simulate()
{
	time.Start();
	time.End();
	while (active) {
		time.Start();

		for (int i = 0; i < drops_count; i++) {
			drops[i].veclosity += g * time.GetDelta();
		}

		if (dropBarierCollision(0) == -1) {
			float del = time.GetDelta();
			drops[0].pos += drops[0].veclosity;
		}

		time.End();
		tps = time.GetFPS();
	}
	end = true;
}
