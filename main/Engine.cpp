#include "Engine.h"

Engine::Engine(HINSTANCE hinst) :
	time(144)
{
	int gpu_width = 1920 * 2, gpu_height = 1080 * 2;
	wnd = new Window(1920 * 2, 1080 * 2, hinst, gpu_width, gpu_height);
	kbd = &wnd->kbd;
	mouse = &wnd->mouse;
	gfx = new Graphics(gpu_width, gpu_height);
	run = true;
}

void Engine::start()
{
	cmd.init(gfx, vec2(0.4f, 0.8f));
	float fps = 0;
	cmd.newDinamicField("fps: ", &fps);

	fluid.init(gfx);
	cmd.newDinamicField("tps: ", fluid.getTps());
	fluid.start();


	while (run) {
		time.Start();

		wnd->UpdateInput();

		update();
		render();

		fps = time.GetFPS();

		wnd->UpdateWindowBuffer(gfx->getFrame());

		time.End();

	}

	fluid.stop();
	cmd.clear();
}

void Engine::render()
{
	gfx->fillFrame(black);

	fluid.render();

	cmd.render();
}

void Engine::update()
{
	run &= wnd->Run();
	if (wnd->kbd.KeyIsPressed(192) ) {
		cmd.swapActive();
		wnd->kbd.OnKeyReleased(192);
	}
	if (wnd->kbd.KeyIsPressed('R')) {
		fluid.reset();
		wnd->kbd.OnKeyReleased('R');
	}
	
}
