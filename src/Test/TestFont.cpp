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
        Gdiplus::Font font(Gdiplus::FontFamily::GenericMonospace(), TEST_FONT_SIZE);
        Gdiplus::SolidBrush brush(Gdiplus::Color::White);
        Gdiplus::Graphics graphics(&bitmap);

        Size sizeMax;
        for (int s = TEST_CHAR_MIN; s < TEST_CHAR_MAX; ++s)
        {
            Gdiplus::RectF rectf;
            wchar_t text[2] = { wchar_t(s) };
            graphics.MeasureString(text, -1, &font, Gdiplus::PointF(0, 0), &rectf);
            sizeMax.x = Simd::Max(sizeMax.x, (ptrdiff_t)::ceil(rectf.Width));
            sizeMax.y = Simd::Max(sizeMax.y, (ptrdiff_t)::ceil(rectf.Height));
        }

        FontData data;
        if (!data.Good())
            return false;
        data.Write(TEST_CHAR_MIN);
        data.Write(TEST_CHAR_MAX);
        data.Write(sizeMax.x);
        data.Write(sizeMax.y);
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

            ptrdiff_t top = 0, bottom = 0;
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
            data.Write(top);
            data.Write(bottom);
            for (ptrdiff_t y = top; y < bottom; ++y)
            {
                data.Write(rows[y].size()/2);
                for (size_t i = 0; i < rows[y].size(); ++i)
                    data.Write(rows[y][i]);
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
        View image(W, H, View::Bgra32);

        Simd::FillBgra(image, 0, 64, 128, 255);

        Simd::Font font(16);

        font.Draw(image, "Test string", Point(100, 100), 0xFFFF7777);

        image.Save("texts.ppm");

        return true;
    }
}
