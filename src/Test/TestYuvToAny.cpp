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
#include "Test/TestOptions.h"

#include "Simd/SimdFrame.hpp"

#include <unordered_set>

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

    bool Yuv444pToHslAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToAnyAutoTest(1, 1, View::Hsl24, FUNC(Simd::Base::Yuv444pToHsl), FUNC(SimdYuv444pToHsl));

        return result;
    }

    bool Yuv444pToHsvAutoTest()
    {
        bool result = true;

        if (TestBase())
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

        if (TestBase())
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Base::Yuv444pToHue), FUNC(SimdYuv444pToHue), MAX_DIFFERECE);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Sse41::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Avx2::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Avx512bw::Yuv444pToHue), FUNC(SimdYuv444pToHue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToAnyAutoTest(1, 1, View::Gray8, FUNC(Simd::Neon::Yuv444pToHue), FUNC(SimdYuv444pToHue), MAX_DIFFERECE);
#endif

        return result;
    }

    bool Yuv420pToHueAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Base::Yuv420pToHue), FUNC(SimdYuv420pToHue), MAX_DIFFERECE);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Sse41::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Avx2::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw() && W > Simd::Avx512bw::DA)
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Avx512bw::Yuv420pToHue), FUNC(SimdYuv420pToHue));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToAnyAutoTest(2, 2, View::Gray8, FUNC(Simd::Neon::Yuv420pToHue), FUNC(SimdYuv420pToHue), MAX_DIFFERECE);
#endif

        return result;
    }

    bool Yuv420pToUyvy422AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToAnyAutoTest(2, 2, View::Uyvy16, FUNC(Simd::Base::Yuv420pToUyvy422), FUNC(SimdYuv420pToUyvy422));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToAnyAutoTest(2, 2, View::Uyvy16, FUNC(Simd::Sse41::Yuv420pToUyvy422), FUNC(SimdYuv420pToUyvy422));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToAnyAutoTest(2, 2, View::Uyvy16, FUNC(Simd::Avx2::Yuv420pToUyvy422), FUNC(SimdYuv420pToUyvy422));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw() && W > Simd::Avx512bw::DA)
            result = result && YuvToAnyAutoTest(2, 2, View::Uyvy16, FUNC(Simd::Avx512bw::Yuv420pToUyvy422), FUNC(SimdYuv420pToUyvy422));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToAnyAutoTest(2, 2, View::Uyvy16, FUNC(Simd::Neon::Yuv420pToUyvy422), FUNC(SimdYuv420pToUyvy422));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncYuv2
        {
            typedef void(*FuncPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
                size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType);

            FuncPtr func;
            String description;

            FuncYuv2(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Call(const View& y, const View& u, const View& v, View& bgr, SimdYuvType yuvType) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, bgr.data, bgr.stride, yuvType);
            }
        };
    }

#define FUNC_YUV2(function) FuncYuv2(function, #function)

    bool YuvToBgr2AutoTest(int width, int height, const FuncYuv2& f1, const FuncYuv2& f2, int dx, int dy, SimdYuvType yuvType)
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

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(y, u, v, bgr1, yuvType));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(y, u, v, bgr2, yuvType));

        result = result && Compare(bgr1, bgr2, 0, true, 64);

        return result;
    }

    bool YuvToBgr2AutoTest(const FuncYuv2& f1, const FuncYuv2& f2, int dx, int dy)
    {
        bool result = true;

        result = result && YuvToBgr2AutoTest(W, H, f1, f2, dx, dy, SimdYuvBt601);
        result = result && YuvToBgr2AutoTest(W + O * dx, H - O * dy, f1, f2, dx, dy, SimdYuvBt709);
        result = result && YuvToBgr2AutoTest(W - O * dx, H + O * dy, f1, f2, dx, dy, SimdYuvBt2020);

        return result;
    }

    bool Yuv420pToBgrV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv420pToBgrV2), FUNC_YUV2(SimdYuv420pToBgrV2), 2, 2);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv420pToBgrV2), FUNC_YUV2(SimdYuv420pToBgrV2), 2, 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv420pToBgrV2), FUNC_YUV2(SimdYuv420pToBgrV2), 2, 2);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv420pToBgrV2), FUNC_YUV2(SimdYuv420pToBgrV2), 2, 2);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv420pToBgrV2), FUNC_YUV2(SimdYuv420pToBgrV2), 2, 2);
#endif 

        return result;
    }

    bool Yuv422pToBgrV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv422pToBgrV2), FUNC_YUV2(SimdYuv422pToBgrV2), 2, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv422pToBgrV2), FUNC_YUV2(SimdYuv422pToBgrV2), 2, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv422pToBgrV2), FUNC_YUV2(SimdYuv422pToBgrV2), 2, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv422pToBgrV2), FUNC_YUV2(SimdYuv422pToBgrV2), 2, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv422pToBgrV2), FUNC_YUV2(SimdYuv422pToBgrV2), 2, 1);
#endif 

        return result;
    }

    bool Yuv444pToBgrV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv444pToBgrV2), FUNC_YUV2(SimdYuv444pToBgrV2), 1, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv444pToBgrV2), FUNC_YUV2(SimdYuv444pToBgrV2), 1, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv444pToBgrV2), FUNC_YUV2(SimdYuv444pToBgrV2), 1, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv444pToBgrV2), FUNC_YUV2(SimdYuv444pToBgrV2), 1, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv444pToBgrV2), FUNC_YUV2(SimdYuv444pToBgrV2), 1, 1);
#endif 

        return result;
    }

    bool Yuv420pToRgbV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv420pToRgbV2), FUNC_YUV2(SimdYuv420pToRgbV2), 2, 2);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv420pToRgbV2), FUNC_YUV2(SimdYuv420pToRgbV2), 2, 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv420pToRgbV2), FUNC_YUV2(SimdYuv420pToRgbV2), 2, 2);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv420pToRgbV2), FUNC_YUV2(SimdYuv420pToRgbV2), 2, 2);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv420pToRgbV2), FUNC_YUV2(SimdYuv420pToRgbV2), 2, 2);
#endif 

        return result;
    }

    bool Yuv422pToRgbV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv422pToRgbV2), FUNC_YUV2(SimdYuv422pToRgbV2), 2, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv422pToRgbV2), FUNC_YUV2(SimdYuv422pToRgbV2), 2, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv422pToRgbV2), FUNC_YUV2(SimdYuv422pToRgbV2), 2, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv422pToRgbV2), FUNC_YUV2(SimdYuv422pToRgbV2), 2, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv422pToRgbV2), FUNC_YUV2(SimdYuv422pToRgbV2), 2, 1);
#endif 

        return result;
    }

    bool Yuv444pToRgbV2AutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Base::Yuv444pToRgbV2), FUNC_YUV2(SimdYuv444pToRgbV2), 1, 1);

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Sse41::Yuv444pToRgbV2), FUNC_YUV2(SimdYuv444pToRgbV2), 1, 1);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx2::Yuv444pToRgbV2), FUNC_YUV2(SimdYuv444pToRgbV2), 1, 1);
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Avx512bw::Yuv444pToRgbV2), FUNC_YUV2(SimdYuv444pToRgbV2), 1, 1);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && YuvToBgr2AutoTest(FUNC_YUV2(Simd::Neon::Yuv444pToRgbV2), FUNC_YUV2(SimdYuv444pToRgbV2), 1, 1);
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    bool Yuv420pToRgbV2SpecialTest(const Options & options)
    {
        bool result = true;

        String path = ROOT_PATH + "/data/image/city.jpg";
        View orig;
        if (!orig.Load(path, View::Bgr24))
            return false;

        typedef Simd::Frame<Simd::Allocator> Frame;

        Frame bgr(orig.Size(), Frame::Bgr24);
        Frame yuv(orig.Size(), Frame::Yuv420p);
        Frame rgb(orig.Size(), Frame::Rgb24);

        Simd::Copy(orig, bgr.planes[0]);
         
        Simd::Convert(bgr, yuv);
        Simd::Convert(yuv, rgb);

        bgr.planes[0].Save("bgr.jpg");
        rgb.planes[0].Save("rgb.jpg");

        return result;
    }
}
