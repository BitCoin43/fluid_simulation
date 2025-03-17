#pragma once
#include "../math/Rect.h"
#include "../system/Timer.h"
#include "../render/Graphics.h"

class Fluid2d {
public:
	Fluid2d();
	~Fluid2d();
	void init(Graphics* gfx);
	void start();
	void render();
	void stop();
	float* getTps();
	void reset();
public:
	class Drop {
	public:
		vec2 pos;
		vec2 veclosity;
	};
private:
	int dropBarierCollision(int id);
	void simulate();
private:
	Rect* bariers;
	int bariers_count;
	Graphics* gfx;
	Drop* drops;
	int drops_count;
	float drod_radius;
	Timer time;
	bool active;
	bool end = false;

	float tps;
private:
	vec2 g;
};