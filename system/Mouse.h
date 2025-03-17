#pragma once
#include <cuda_runtime.h>
#include <Windows.h>

class Mouse {
public:
	Mouse();
	int2 getCursor(int2 wnd_pos);
	int2 getGlobalCursor();
	void setScroll(int scr);
	int getScroll();
	void zeroScroll();
	void setLeftKey(bool value);
	void setRightKey(bool value);
	bool getLeftKey(bool value);
	bool getRightKey(bool value);
private: 
	bool left_key;
	bool right_key;
	int scroll;
};
