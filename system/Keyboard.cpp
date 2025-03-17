#include "Keyboard.h"

Keyboard::Keyboard(int size)
{
	keystates = new bool[size];
	for (int i = 0; i < size; i++) {
		keystates[i] = false;
	}
}

Keyboard::~Keyboard()
{
	delete[] keystates;
}

bool Keyboard::KeyIsPressed(unsigned char keycode) const
{
	if (keycode < nKeys)
		return keystates[keycode];
	else {
		return false;
	}
}


void Keyboard::OnKeyPressed(unsigned char keycode)
{
	if (this == nullptr) {
		int a = 0;
	}
	if (keycode < nKeys) {
		keystates[keycode] = true;
	}
}

void Keyboard::OnKeyReleased(unsigned char keycode)
{
	if (keycode < nKeys)
		keystates[keycode] = false;
}