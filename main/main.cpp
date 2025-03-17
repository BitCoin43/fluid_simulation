#include "Engine.h"

int WINAPI WinMain(
	HINSTANCE hInst, 
	HINSTANCE hPrevInstance, 
	LPSTR lpCmdLine, 
	int nShowCmd
) {
	Engine engine(hInst);
	engine.start();


	return 0;
}