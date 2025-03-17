#pragma once
#include "../math/3dmath.h"
#include "../math/Color.cuh"
#include "./2d/Mask.h"
#include <ft2build.h>
#include FT_FREETYPE_H
#include FT_OUTLINE_H 
#include <string>

class Graphics {
public:
	Graphics(int width, int height);
	~Graphics();
public:
	void fillFrame(Color c);
	void renderLine(Color c, vec2 start, vec2 end);
	void renderRect(Color c, vec2 start, vec2 end);
	void renderElips(Color c, vec2 x, float r);
public:
	void fill(Surface* surf, Fcolor color);
	void surfaceToFrame(Surface* surf, vec2 start);
	void surfaceToFrame(Surface* surf, int2 start);
	void renderLine(Surface* surf, Fcolor color, int2 start, int2 end);
	void renderRect(Surface* surf, Fcolor color, int2 start, int2 end);
	void renderElips(Surface* surf, Fcolor c, int2 x, int r);
	void print(Surface* surf, std::string text, int2 start, Fcolor color);
public:
	uchar4* getFrame();
	uint2 getDimFromVec2(vec2 v);
private:
	int2 normalizeVec2(vec2 v);
	uint2 forceNormalizeVec2(vec2 v);
private:
	class Text {
	public:
		Text();
		void init(int fontSize);
		~Text();
		void print(std::string text, int2 start, Fcolor color, Surface* dst);
		//void print(uchar4* mem, int2 dim, std::string text, int x, int y, uchar4 color, int size);
		//void set_mem(uchar4* dev_mem);
		//int getAdvance(int size);
	private:
		FT_Library library;
		FT_Face face;
	private:
		class Glyph {
		public:
			Mask texture;
			uint2 dim;
			int advance;
			int horiBearingX;
			int horiBearingY;
			bool have_texture;
		};
	public:
		int font_size;
		Glyph** textures;
	};
private:
	int width, height;
	Color* frame;
	Text text;
};