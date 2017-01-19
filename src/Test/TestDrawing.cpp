/*
* Tests for Simd Library (http://simd.sourceforge.net).
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

namespace Test
{
	namespace
	{
		struct FuncAB
		{
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
                const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride);
			FuncPtr func;
			String description;

			FuncAB(const FuncPtr & f, const String & d) : func(f), description(d) {}

			void Call(const View & src, const View & alpha, const View & dstSrc, View & dstDst) const
			{
                Simd::Copy(dstSrc, dstDst);
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, src.ChannelCount(), alpha.data, alpha.stride, dstDst.data, dstDst.stride);
			}
		};	
	}

#define FUNC_AB(func) FuncAB(func, #func)

    bool AlphaBlendingAutoTest(View::Format format, int width, int height, const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);
        View b(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(b);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, a, b, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, a, b, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool AlphaBlendingAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            FuncAB f1c = FuncAB(f1.func, f1.description + ColorDescription(format));
            FuncAB f2c = FuncAB(f2.func, f2.description + ColorDescription(format));
            
            result = result && AlphaBlendingAutoTest(format, W, H, f1c, f2c);
            result = result && AlphaBlendingAutoTest(format, W + O, H - O, f1c, f2c);
            result = result && AlphaBlendingAutoTest(format, W - O, H + O, f1c, f2c);
        }

        return result;
    }

    bool AlphaBlendingAutoTest()
    {
        bool result = true;

        result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Base::AlphaBlending), FUNC_AB(SimdAlphaBlending));

#ifdef SIMD_SSE2_ENABLE
        if(Simd::Sse2::Enable)
            result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Sse2::AlphaBlending), FUNC_AB(SimdAlphaBlending));
#endif 

#ifdef SIMD_SSSE3_ENABLE
        if(Simd::Ssse3::Enable)
            result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Ssse3::AlphaBlending), FUNC_AB(SimdAlphaBlending));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Avx2::AlphaBlending), FUNC_AB(SimdAlphaBlending));
#endif 

#ifdef SIMD_VMX_ENABLE
        if(Simd::Vmx::Enable)
            result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Vmx::AlphaBlending), FUNC_AB(SimdAlphaBlending));
#endif

#ifdef SIMD_NEON_ENABLE
		if (Simd::Neon::Enable)
			result = result && AlphaBlendingAutoTest(FUNC_AB(Simd::Neon::AlphaBlending), FUNC_AB(SimdAlphaBlending));
#endif

        return result;    
    }

    //-----------------------------------------------------------------------

    bool AlphaBlendingDataTest(bool create, View::Format format, int width, int height, const FuncAB & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b(width, height, format, NULL, TEST_ALIGN(width));

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        if(create)
        {
            FillRandom(s);
            FillRandom(a);
            FillRandom(b);

            TEST_SAVE(s);
            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(s, a, b, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(d1);

            f.Call(s, a, b, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool AlphaBlendingDataTest(bool create)
    {
        bool result = true;

        FuncAB f = FUNC_AB(SimdAlphaBlending);

        for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && AlphaBlendingDataTest(create, format, DW, DH, FuncAB(f.func, f.description + Data::Description(format)));
        }

        return result;
    }
}

//-----------------------------------------------------------------------------

#include "Simd/SimdDrawing.hpp"

namespace Test
{
    bool DrawLineSpecialTest()
    {
        View image(W, H, View::Gray8);

        Simd::Fill(image, 0);

        const size_t o = 55, n = 256, m = 20, w = 3;

        for (size_t i = o; i < n; ++i)
        {
            ptrdiff_t x1 = Random(W * 2) - W / 2;
            ptrdiff_t y1 = Random(H * 2) - H / 2;
            ptrdiff_t x2 = i%m == 0 ? x1 : Random(W * 2) - W / 2;
            ptrdiff_t y2 = i%m == 1 ? y1 : Random(H * 2) - H / 2;
            Simd::DrawLine(image, x1, y1, x2, y2, uint8_t(i), Random(w) + 1);
        }

        image.Save("lines.pgm");

        return true;
    }

    bool DrawRectangleSpecialTest()
    {
        View image(W, H, View::Gray8);

        Simd::Fill(image, 0);

        const size_t o = 55, n = 256, w = 3;

        for (size_t i = o; i < n; i += 5)
        {
            ptrdiff_t x1 = Random(W * 5 / 4) - W / 8;
            ptrdiff_t y1 = Random(H * 5 / 4) - H / 8;
            ptrdiff_t x2 = Random(W * 5 / 4) - W / 8;
            ptrdiff_t y2 = Random(H * 5 / 4) - H / 8;

            Simd::DrawRectangle(image, Rect(std::min(x1, x2), std::min(y1, y2), std::max(x1, x2), std::max(y1, y2)), uint8_t(i), Random(w) + 1);
        }

        image.Save("rectangles.pgm");

        return true;
    }
}