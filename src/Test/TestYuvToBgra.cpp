/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"

namespace Test
{
    namespace
    {
        struct FuncYuva2
        {
            typedef void(*FuncPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
                const uint8_t* a, size_t aStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, SimdYuvType yuvType);

            FuncPtr func;
            String description;

            FuncYuva2(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Call(const View& y, const View& u, const View& v, const View& a, View& bgra, SimdYuvType yuvType) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(y.data, y.stride, u.data, u.stride, v.data, v.stride, a.data, a.stride, y.width, y.height, bgra.data, bgra.stride, yuvType);
            }
        };
    }

#define FUNC_YUVA2(function) FuncYuva2(function, #function)

    bool YuvaToBgra2AutoTest(int width, int height, const FuncYuva2& f1, const FuncYuva2& f2, int dx, int dy, SimdYuvType yuvType)
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
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, a, bgra1, yuvType));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, a, bgra2, yuvType));

        result = result && Compare(bgra1, bgra2, 0, true, 64);

        return result;
    }

    bool YuvaToBgra2AutoTest(const FuncYuva2& f1, const FuncYuva2& f2, int dx, int dy)
    {
        bool result = true;

        result = result && YuvaToBgra2AutoTest(W, H, f1, f2, dx, dy, SimdYuvBt601);
        result = result && YuvaToBgra2AutoTest(W + O * dx, H - O * dy, f1, f2, dx, dy, SimdYuvBt709);
        result = result && YuvaToBgra2AutoTest(W - O * dx, H + O * dy, f1, f2, dx, dy, SimdYuvBt2020);

        return result;
    }

    bool Yuva420pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Base::Yuva420pToBgraV2), FUNC_YUVA2(SimdYuva420pToBgraV2), 2, 2);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Sse41::Yuva420pToBgraV2), FUNC_YUVA2(SimdYuva420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx2::Yuva420pToBgraV2), FUNC_YUVA2(SimdYuva420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx512bw::Yuva420pToBgraV2), FUNC_YUVA2(SimdYuva420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Neon::Yuva420pToBgraV2), FUNC_YUVA2(SimdYuva420pToBgraV2), 2, 2);
#endif 

        return result;
    }

    bool Yuva422pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Base::Yuva422pToBgraV2), FUNC_YUVA2(SimdYuva422pToBgraV2), 2, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Sse41::Yuva422pToBgraV2), FUNC_YUVA2(SimdYuva422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx2::Yuva422pToBgraV2), FUNC_YUVA2(SimdYuva422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx512bw::Yuva422pToBgraV2), FUNC_YUVA2(SimdYuva422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Neon::Yuva422pToBgraV2), FUNC_YUVA2(SimdYuva422pToBgraV2), 2, 1);
#endif 

        return result;
    }

    bool Yuva444pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Base::Yuva444pToBgraV2), FUNC_YUVA2(SimdYuva444pToBgraV2), 1, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Sse41::Yuva444pToBgraV2), FUNC_YUVA2(SimdYuva444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx2::Yuva444pToBgraV2), FUNC_YUVA2(SimdYuva444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Avx512bw::Yuva444pToBgraV2), FUNC_YUVA2(SimdYuva444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvaToBgra2AutoTest(FUNC_YUVA2(Simd::Neon::Yuva444pToBgraV2), FUNC_YUVA2(SimdYuva444pToBgraV2), 1, 1);
#endif 

        return result;
    }

    //---------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncYuv2
        {
            typedef void(*FuncPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
                size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType yuvType);

            FuncPtr func;
            String description;

            FuncYuv2(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Call(const View& y, const View& u, const View& v, View& bgra, uint8_t alpha, SimdYuvType yuvType) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgra.data, bgra.stride, alpha, yuvType);
            }
        };
    }

#define FUNC_YUV2(function) FuncYuv2(function, #function)

    bool YuvToBgra2AutoTest(int width, int height, const FuncYuv2& f1, const FuncYuv2& f2, int dx, int dy, SimdYuvType yuvType)
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

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        const uint8_t alpha = 0xFE;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, bgra1, alpha, yuvType));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, bgra2, alpha, yuvType));

        result = result && Compare(bgra1, bgra2, 0, true, 64);

        return result;
    }

    bool YuvToBgra2AutoTest(const FuncYuv2& f1, const FuncYuv2& f2, int dx, int dy)
    {
        bool result = true;

        result = result && YuvToBgra2AutoTest(W, H, f1, f2, dx, dy, SimdYuvBt601);
        result = result && YuvToBgra2AutoTest(W + O * dx, H - O * dy, f1, f2, dx, dy, SimdYuvBt709);
        result = result && YuvToBgra2AutoTest(W - O * dx, H + O * dy, f1, f2, dx, dy, SimdYuvBt2020);

        return result;
    }

    bool Yuv420pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Base::Yuv420pToBgraV2), FUNC_YUV2(SimdYuv420pToBgraV2), 2, 2);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv420pToBgraV2), FUNC_YUV2(SimdYuv420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv420pToBgraV2), FUNC_YUV2(SimdYuv420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv420pToBgraV2), FUNC_YUV2(SimdYuv420pToBgraV2), 2, 2);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Neon::Yuv420pToBgraV2), FUNC_YUV2(SimdYuv420pToBgraV2), 2, 2);
#endif 

        return result;
    }

    bool Yuv422pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Base::Yuv422pToBgraV2), FUNC_YUV2(SimdYuv422pToBgraV2), 2, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv422pToBgraV2), FUNC_YUV2(SimdYuv422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv422pToBgraV2), FUNC_YUV2(SimdYuv422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv422pToBgraV2), FUNC_YUV2(SimdYuv422pToBgraV2), 2, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Neon::Yuv422pToBgraV2), FUNC_YUV2(SimdYuv422pToBgraV2), 2, 1);
#endif 

        return result;
    }

    bool Yuv444pToBgraV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Base::Yuv444pToBgraV2), FUNC_YUV2(SimdYuv444pToBgraV2), 1, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv444pToBgraV2), FUNC_YUV2(SimdYuv444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv444pToBgraV2), FUNC_YUV2(SimdYuv444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv444pToBgraV2), FUNC_YUV2(SimdYuv444pToBgraV2), 1, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Neon::Yuv444pToBgraV2), FUNC_YUV2(SimdYuv444pToBgraV2), 1, 1);
#endif 

        return result;
    }

    bool Yuv444pToRgbaV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Base::Yuv444pToRgbaV2), FUNC_YUV2(SimdYuv444pToRgbaV2), 1, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv444pToRgbaV2), FUNC_YUV2(SimdYuv444pToRgbaV2), 1, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv444pToRgbaV2), FUNC_YUV2(SimdYuv444pToRgbaV2), 1, 1);
#endif 

//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && YuvToBgra2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv444pToRgbaV2), FUNC_YUV2(SimdYuv444pToRgbaV2), 1, 1);
//#endif 
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable && TestNeon())
//            result = result && YuvToRgba2AutoTest(FUNC_YUV2(Simd::Neon::Yuv444pToRgbaV2), FUNC_YUV2(SimdYuv444pToRgbaV2), 1, 1);
//#endif 

        return result;
    }
}
