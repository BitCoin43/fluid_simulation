#pragma once
#include <chrono>
#include <thread>

class Timer {
public:
	Timer(int max_fps);
	float GetFPS();
	float GetDelta();
	void Start();
	void End();
private:
	std::chrono::time_point<std::chrono::high_resolution_clock>  start;
	std::chrono::time_point<std::chrono::high_resolution_clock>  end;
	double time_to_frame;
	float fps;
};