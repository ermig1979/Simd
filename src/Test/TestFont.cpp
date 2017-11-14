/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Test/TestUtils.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"

//#define TEST_FONT_GENERATION

#if defined(WIN32) && defined(TEST_FONT_GENERATION)
using namespace std;
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#pragma comment (lib, "GDIPlus.lib")

#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"

namespace Test
{
    struct GdiPlus
    {
        GdiPlus()
        {
            Gdiplus::GdiplusStartupInput gdiplusStartupInput;
            Gdiplus::GdiplusStartup(&_gdiPlusToken, &gdiplusStartupInput, NULL);
        }

        ~GdiPlus()
        {
            Gdiplus::GdiplusShutdown(_gdiPlusToken);
        }

    private:
        ULONG_PTR _gdiPlusToken;
    };

    bool GenerateFont()
    {
        const int TEST_CHAR_MIN = 32, TEST_CHAR_MAX = 127;
        wchar_t text[TEST_CHAR_MAX - TEST_CHAR_MIN + 1] = { 0 };
        for (int i = TEST_CHAR_MIN; i < TEST_CHAR_MAX; ++i)
            text[i - TEST_CHAR_MIN] = i;

        GdiPlus gdiPlus;

        View bgra(16000, 256, View::Bgra32);
        Simd::Fill(bgra, 127);
        Gdiplus::Bitmap bitmap((int)bgra.width, (int)bgra.height, (int)bgra.stride, PixelFormat32bppARGB, bgra.data);
        Gdiplus::Font font(Gdiplus::FontFamily::GenericMonospace(), 152.0);
        Gdiplus::SolidBrush brush(Gdiplus::Color::White);
        Gdiplus::Graphics graphics(&bitmap);

        graphics.DrawString(text, -1, &font, Gdiplus::PointF(0, 0), &brush);

        Gdiplus::RectF rectf;
        text[65] = 0;
        graphics.MeasureString(L"_", -1, &font, Gdiplus::PointF(0, 0), &rectf);
        //Simd::DrawRectangle(bgra, Test::Rect(0, 0, (ptrdiff_t)rect.Width(), (int)rect.Height()), uint32_t(-1));
        //for (int i = 32; i < NUMBER; ++i)
        //{
        //    text[0] = i; 
        //    graphics.DrawString(text, -1, &font, Gdiplus::PointF((i - 32) * 14, 00), &brush);
        //    graphics.DrawString(text, -1, &font, Gdiplus::PointF((i - 32) * 14, 20), &brush);
        //}

        bgra.Save("font.ppm");

        return true;
    }

    bool generateFont = GenerateFont();
}
#endif
