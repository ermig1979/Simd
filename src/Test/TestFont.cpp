/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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

#include "Simd/SimdFont.hpp"

//#define TEST_GENERATE_FONT

#if defined(WIN32) && defined(TEST_GENERATE_FONT)
using namespace std;
#define NOMINMAX
#include <windows.h>
#include <gdiplus.h>
#pragma comment (lib, "GDIPlus.lib")

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

    struct FontData
    {
        FontData()
        {
            _ofs.open("font_data.txt");
            _size = 0;
        }

        bool Good() const
        {
            return _ofs.is_open();
        }

        ~FontData()
        {
            if(_ofs.is_open())
                _ofs.close();
        }

        bool Write(ptrdiff_t value)
        {
            std::stringstream ss;
            ss << value << ", ";
            _ofs << ss.str();
            _size += ss.str().size();
            if (_size > 100)
            {
                _ofs << std::endl;
                _size = 0;
            }
            _ofs.flush();
            return true;
        }

    private:
        std::ofstream _ofs;
        size_t _size;
    };

    bool GenerateFont()
    {
        const int TEST_CHAR_MIN = 32;
        const int TEST_CHAR_MAX = 127;
        const int TEST_SIZE_MAX = 256;
        const float TEST_FONT_SIZE = 152.0f;
        const uint8_t TEST_THRESHOLD = 127;

        GdiPlus gdiPlus;

        View bgra(TEST_SIZE_MAX, TEST_SIZE_MAX, View::Bgra32);
        View gray(TEST_SIZE_MAX, TEST_SIZE_MAX, View::Gray8);
        Simd::Fill(bgra, 0);
        Gdiplus::Bitmap bitmap((int)bgra.width, (int)bgra.height, (int)bgra.stride, PixelFormat32bppARGB, bgra.data);
        Gdiplus::Font font(Gdiplus::FontFamily::GenericMonospace(), TEST_FONT_SIZE, Gdiplus::FontStyleBold);
        Gdiplus::SolidBrush brush(Gdiplus::Color::White);
        Gdiplus::Graphics graphics(&bitmap);

        Gdiplus::RectF r12, r11, r21;
        graphics.MeasureString(L"12", -1, &font, Gdiplus::PointF(0, 0), &r12);
        graphics.MeasureString(L"1", -1, &font, Gdiplus::PointF(0, 0), &r11);
        graphics.MeasureString(L"1\n2", -1, &font, Gdiplus::PointF(0, 0), &r21);
        ptrdiff_t indentX = (ptrdiff_t)::floor(r11.Width - r12.Width*0.5f);
        ptrdiff_t indentY = (ptrdiff_t)::floor(r11.Height - r21.Height*0.5f);
        Size sizeMax((ptrdiff_t)::ceil(r11.Width) - 2*indentX, (ptrdiff_t)::ceil(r11.Height) - 2*indentY);

        FontData data;
        if (!data.Good())
            return false;
        data.Write(TEST_CHAR_MIN);
        data.Write(TEST_CHAR_MAX);
        data.Write(sizeMax.x);
        data.Write(sizeMax.y);
        data.Write(indentX);
        data.Write(indentY);
        for (int s = TEST_CHAR_MIN; s < TEST_CHAR_MAX; ++s)
        {
            Simd::Fill(bgra, 0);
            wchar_t text[2] = { wchar_t(s) };
            graphics.DrawString(text, -1, &font, Gdiplus::PointF(0, 0), &brush);
            Simd::Convert(bgra, gray);

            typedef std::vector<ptrdiff_t> Row;
            typedef std::vector<Row> Rows;
            Rows rows(sizeMax.y);
            for (ptrdiff_t y = 0; y < sizeMax.y; ++y)
            {
                const uint8_t * row = gray.Row<uint8_t>(y);
                bool above = false;
                for (ptrdiff_t x = 0; x < sizeMax.y; ++x)
                {
                    if (!above && row[x] > TEST_THRESHOLD)
                    {
                        above = true;
                        rows[y].push_back(x);
                    }
                    if (above && row[x] <= TEST_THRESHOLD)
                    {
                        above = false;
                        rows[y].push_back(x);
                    }
                }
                assert(!above);
            }

            ptrdiff_t top = indentY, bottom = indentY;
            for (ptrdiff_t y = 0; y < sizeMax.y; ++y)
            {
                if (rows[y].size())
                {
                    top = y;
                    break;
                }
            }
            for (ptrdiff_t y = sizeMax.y - 1; y >= 0; --y)
            {
                if (rows[y].size())
                {
                    bottom = y + 1;
                    break;
                }
            }

            data.Write(s);
            data.Write(top - indentY);
            data.Write(bottom - indentY);
            for (ptrdiff_t y = top; y < bottom; ++y)
            {
                data.Write(rows[y].size()/2);
                for (size_t i = 0; i < rows[y].size(); ++i)
                    data.Write(std::max<ptrdiff_t>(0, rows[y][i] - indentX));
            }
        }

        return true;
    }

    bool generateFont = GenerateFont();
}
#endif //TEST_GENERATE_FONT

//-----------------------------------------------------------------------------

namespace Test
{
    bool FontDrawSpecialTest()
    {
        typedef Simd::Pixel::Bgra32 Color;

        View image(W, H, View::Bgra32);
        for (size_t y = 0; y < image.height; ++y)
        {
            Color * row = image.Row<Color>(y);
            for (size_t x = 0; x < image.width; ++x)
                row[x] = Color(255, (uint8_t)x, (uint8_t)y);
        }

        String text = "First_string,\nSecond-line.";
        Simd::Font font(16);
        for (size_t i = 0; i < 10; ++i)
        {
            font.Resize(Random(H / 2) + 16);
            font.Draw(image, text, Point(Random(W) - W/3, Random(H) - H/2),
                Color(Random(255), Random(255), Random(255)));
        }

        font.Resize(32);
        font.Draw(image, text, View::BottomRight, Color(0, 0, 0));
        font.Draw(image, text, View::TopLeft, Color(0, 0, 0));

        image.Save("texts.ppm");

        return true;
    }
}
