#include "Mouse.h"

Mouse::Mouse() :
	left_key(false), right_key(false),
	scroll(0)
{
}

int2 Mouse::getCursor(int2 wnd_pos)
{
	int2 res = getGlobalCursor();
	res.x -= wnd_pos.x;
	res.y -= wnd_pos.y;
	return res;
}

int2 Mouse::getGlobalCursor()
{
	POINT cursorPos;
	GetCursorPos(&cursorPos);
	return make_int2(cursorPos.x, cursorPos.y);
}

void Mouse::setScroll(int scr)
{
	scroll += scr;
}

int Mouse::getScroll()
{
	return scroll;
}

void Mouse::zeroScroll()
{
	scroll = 0;
}

void Mouse::setLeftKey(bool value)
{
	left_key = value;
}

void Mouse::setRightKey(bool value)
{
	right_key = value;
}

bool Mouse::getLeftKey(bool value)
{
	return left_key;
}

bool Mouse::getRightKey(bool value)
{
	return right_key;
}
