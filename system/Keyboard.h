#pragma once

//#include <bitset>


class Keyboard
{
	//friend class Window;
public:
	Keyboard(int size);
	Keyboard(const Keyboard&) = delete;
	~Keyboard();
	Keyboard& operator=(const Keyboard&) = delete;
	bool KeyIsPressed(unsigned char keycode) const;
	void OnKeyPressed(unsigned char keycode);
	void OnKeyReleased(unsigned char keycode);
private:
private:
	static constexpr unsigned int nKeys = 256;
	//std::bitset<256> keystates;
	int ggg;
	bool* keystates = nullptr;
	bool f = false;
};