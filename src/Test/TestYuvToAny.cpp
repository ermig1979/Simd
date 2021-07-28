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

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                size_t width, size_t height, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & y, const View & u, const View & v, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, dst.data, dst.stride);
            }
        };
    }

#define FUNC(function) Func(function, #function)

    bool YuvToAnyAutoTest(int width, int height, int dx, int dy, View::Format dstType, const Func & f1, const Func & f2, int maxDifference)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        const int uvWidth = width / dx;
        const int uvHeight = height / dy;

        View y(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(y);
        View u(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
        FillRandom(u);
        View v(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
        FillRandom(v);

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, dst2));

        result = result && Compare(dst1, dst2, maxDifference, true, 64, 255);

        return result;
    }

    bool YuvToAnyAutoTest(int dx, int dy, View::Format dstType, const Func & f1, const Func & f2, int maxDifference = 0)
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(W, H, dx, dy, dstType, f1, f2, maxDifference);
        result = result && YuvToAnyAutoTest(W + O*dx, H - O*dy, dx, dy, dstType, f1, f2, maxDifference);
        result = result && YuvToAnyAutoTest(W - O*dx, H + O*dy, dx, dy, dstType, f1, f2, maxDifference);

        return result;
    }

    bool Yuv444pToBgrAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Base::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Sse41::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Avx2::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Avx512bw::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Vmx::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Bgr24, FUNC(Simd::Neon::Yuv444pToBgr), FUNC(SimdYuv444pToBgr));
#endif

        return result;
    }

    bool Yuv422pToBgrAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Base::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Sse41::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Avx2::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Avx512bw::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Vmx::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Bgr24, FUNC(Simd::Neon::Yuv422pToBgr), FUNC(SimdYuv422pToBgr));
#endif

        return result;
    }

    bool Yuv420pToBgrAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Base::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Sse41::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Avx2::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Avx512bw::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Vmx::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Bgr24, FUNC(Simd::Neon::Yuv420pToBgr), FUNC(SimdYuv420pToBgr));
#endif

        return result;
    }

    bool Yuv444pToHslAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(1, 1, View::Hsl24, FUNC(Simd::Base::Yuv444pToHsl), FUNC(SimdYuv444pToHsl));

        return result;
    }

    bool Yuv444pToHsvAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(1, 1, View::Hsv24, FUNC(Simd::Base::Yuv444pToHsv), FUNC(SimdYuv444pToHsv));

        return result;
    }

#if defined(SIMD_NEON_ENABLE) && (SIMD_NEON_RCP_ITER > -1)
    const int MAX_DIFFERECE = 1;
#else
    const int MAX_DIFFERECE = 0;
#endif

    bool Yuv444pToHueAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Base::Yuv444pToHue), FUNC(SimdYuv444pToHue), MAX_DIFFERECE);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Sse2::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Avx2::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Avx512bw::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Vsx::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Neon::Yuv444pToHue), FUNC(SimdYuv444pToHue), MAX_DIFFERECE);
#endif

        return result;
    }

    bool Yuv420pToHueAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Base::Yuv420pToHue), FUNC(SimdYuv420pToHue), MAX_DIFFERECE);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Sse2::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Avx2::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Avx512bw::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Vsx::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Neon::Yuv420pToHue), FUNC(SimdYuv420pToHue), MAX_DIFFERECE);
#endif

        return result;
    }

    bool Yuv444pToRgbAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(1, 1, View::Rgb24, FUNC(Simd::Base::Yuv444pToRgb), FUNC(SimdYuv444pToRgb));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Rgb24, FUNC(Simd::Sse41::Yuv444pToRgb), FUNC(SimdYuv444pToRgb));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Rgb24, FUNC(Simd::Avx2::Yuv444pToRgb), FUNC(SimdYuv444pToRgb));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Rgb24, FUNC(Simd::Avx512bw::Yuv444pToRgb), FUNC(SimdYuv444pToRgb));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(1, 1, View::Rgb24, FUNC(Simd::Neon::Yuv444pToRgb), FUNC(SimdYuv444pToRgb));
#endif

        return result;
    }

    bool Yuv422pToRgbAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(2, 1, View::Rgb24, FUNC(Simd::Base::Yuv422pToRgb), FUNC(SimdYuv422pToRgb));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Rgb24, FUNC(Simd::Sse41::Yuv422pToRgb), FUNC(SimdYuv422pToRgb));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Rgb24, FUNC(Simd::Avx2::Yuv422pToRgb), FUNC(SimdYuv422pToRgb));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Rgb24, FUNC(Simd::Avx512bw::Yuv422pToRgb), FUNC(SimdYuv422pToRgb));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(2, 1, View::Rgb24, FUNC(Simd::Neon::Yuv422pToRgb), FUNC(SimdYuv422pToRgb));
#endif

        return result;
    }

    bool Yuv420pToRgbAutoTest()
    {
        bool result = true;

        result = result && YuvToAnyAutoTest(2, 2, View::Rgb24, FUNC(Simd::Base::Yuv420pToRgb), FUNC(SimdYuv420pToRgb));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Rgb24, FUNC(Simd::Sse41::Yuv420pToRgb), FUNC(SimdYuv420pToRgb));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Rgb24, FUNC(Simd::Avx2::Yuv420pToRgb), FUNC(SimdYuv420pToRgb));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Rgb24, FUNC(Simd::Avx512bw::Yuv420pToRgb), FUNC(SimdYuv420pToRgb));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && YuvToAnyAutoTest(2, 2, View::Rgb24, FUNC(Simd::Neon::Yuv420pToRgb), FUNC(SimdYuv420pToRgb));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool YuvToAnyDataTest(bool create, int width, int height, int dx, int dy, View::Format dstType, const Func & f, int maxDifference = 0)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        const int uvWidth = width / dx;
        const int uvHeight = height / dy;

        View y(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));
        View v(uvWidth, uvHeight, View::Gray8, NULL, TEST_ALIGN(uvWidth));

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(y);
            FillRandom(u);
            FillRandom(v);

            TEST_SAVE(y);
            TEST_SAVE(u);
            TEST_SAVE(v);

            f.Call(y, u, v, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(y);
            TEST_LOAD(u);
            TEST_LOAD(v);

            TEST_LOAD(dst1);

            f.Call(y, u, v, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, maxDifference, true, 64, 255);
        }

        return result;
    }

    bool Yuv420pToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 2, 2, View::Bgr24, FUNC(SimdYuv420pToBgr));

        return result;
    }

    bool Yuv422pToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 2, 1, View::Bgr24, FUNC(SimdYuv422pToBgr));

        return result;
    }

    bool Yuv444pToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 1, 1, View::Bgr24, FUNC(SimdYuv444pToBgr));

        return result;
    }

    bool Yuv444pToHslDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 1, 1, View::Hsl24, FUNC(SimdYuv444pToHsl));

        return result;
    }

    bool Yuv444pToHsvDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 1, 1, View::Hsv24, FUNC(SimdYuv444pToHsv));

        return result;
    }

    bool Yuv420pToHueDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 2, 2, View::Gray8, FUNC(SimdYuv420pToHue), MAX_DIFFERECE);

        return result;
    }

    bool Yuv444pToHueDataTest(bool create)
    {
        bool result = true;

        result = result && YuvToAnyDataTest(create, DW, DH, 1, 1, View::Gray8, FUNC(SimdYuv444pToHue), MAX_DIFFERECE);

        return result;
    }
}
