#include "Terminal.h"

Terminal::Terminal() :
	gfx(nullptr),
	background(0.1f, 0.1f, 0.1f),
	text_color(1.0f, 1.0f, 1.0f),
	active(false)
{
	padding = make_int2(20, 50);
}

Terminal::~Terminal()
{
	dinamic.clear();
}

void Terminal::clear()
{
	dinamic.clear();
}

void Terminal::init(Graphics* gfx, vec2 dim)
{
	this->gfx = gfx;
	this->dim = gfx->getDimFromVec2(dim);

	surf.init(this->dim);
	gfx->fill(&surf, background);

	//gfx->print(&surf, "Hello Terminal!", padding, Fcolor(0, 1, 0));
}

void Terminal::newDinamicField(std::string name, float* value)
{
	DinamicField field;
	field.name = name;
	field.value = value;

	dinamic.push_back(std::move(field));
}

void Terminal::render()
{
	if (active) {
		//gfx->print(&surf, "A", make_int2(50, 50), Fcolor(0, 1, 0));
		gfx->fill(&surf, background);
		for (int i = 0; i < dinamic.size(); i++) {
			std::string line = dinamic[i].name + std::to_string(*dinamic[i].value);
			gfx->print(&surf, line, make_int2(padding.x, padding.y + 30 * i), Fcolor(0, 1, 0));
		}
		gfx->surfaceToFrame(&surf, make_int2(0, 0));
	}
}

void Terminal::swapActive()
{
	active = !active;
}
