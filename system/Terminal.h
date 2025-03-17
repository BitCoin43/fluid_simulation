#pragma once
#include "../render/Graphics.h"
#include <string>
#include <vector>

class Terminal {
public:
	Terminal();
	~Terminal();
	void clear();
	void init(Graphics* gfx, vec2 dim);
	void newDinamicField(std::string name, float* value);
	void render();
	void swapActive();
private:
	class DinamicField {
	public:
		std::string name;
		float* value;
	};
private:
	Surface surf;
	Fcolor background;
	Graphics* gfx;
	uint2 dim;
	std::vector<DinamicField> dinamic;
	bool active;
private:
	int2 padding;
	Fcolor text_color;
};