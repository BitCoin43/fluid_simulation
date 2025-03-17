#include "Timer.h"

Timer::Timer(int max_fps)
{
	if (max_fps != 0) {
		time_to_frame = 1000.0 / max_fps;
	}
	else {
		time_to_frame = 0.0;
	}
	fps = 0.0f;
}

float Timer::GetFPS()
{
	return fps;
}

float Timer::GetDelta()
{
	return 1.0 / fps;
}

void Timer::Start()
{
	start = std::chrono::high_resolution_clock::now();
}

void Timer::End()
{
	end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::milli> elapsed =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	double _elapsed = elapsed.count();

	std::chrono::duration<double, std::milli> time_to_sleep(time_to_frame - _elapsed);

	if (time_to_sleep.count() > 0.0) {
		std::this_thread::sleep_for(time_to_sleep);
		fps = 1000.0 / (_elapsed + time_to_sleep.count());
	}
	else {
		fps = 1000.0 / _elapsed;
	}

	//fps = 1000.0 / elapsed.count();
}
