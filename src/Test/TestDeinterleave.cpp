/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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

namespace Test
{
    namespace
    {
        struct Func2
        {
            typedef void(*FuncPtr)(
                const uint8_t * uv, size_t uvStride, size_t width, size_t height,
                uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

            FuncPtr func;
            String desc;

            Func2(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & uv, View * u, View * v) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(uv.data, uv.stride, uv.width, uv.height, u ? u->data : 0, u ? u->stride : 0, v ? v->data : 0, v ? v->stride : 0);
            }

            void Update(int hasU, int hasV)
            {
                desc = desc + "-" + std::to_string(hasU) + std::to_string(hasV);
            }
        };
    }

#define FUNC2(function) Func2(function, #function)

    bool DeinterleaveUvAutoTest(int width, int height, int hasU, int hasV, Func2 f1, Func2 f2)
    {
        bool result = true;

        f1.Update(hasU, hasV);
        f2.Update(hasU, hasV);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));
        FillRandom(uv);

        View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(uv, hasU ? &u1 : 0, hasV ? &v1 : 0));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(uv, hasU ? &u2 : 0, hasV ? &v2 : 0));

        if (hasU) result = result && Compare(u1, u2, 0, true, 64, 0, "u");
        if (hasV) result = result && Compare(v1, v2, 0, true, 64, 0, "v");

        return result;
    }

    bool DeinterleaveUvAutoTest(int hasU, int hasV, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        result = result && DeinterleaveUvAutoTest(W, H, hasU, hasV, f1, f2);
        result = result && DeinterleaveUvAutoTest(W + O, H - O, hasU, hasV, f1, f2);

        return result;
    }

    bool DeinterleaveUvAutoTest(const Func2& f1, const Func2& f2)
    {
        bool result = true;

        result = result && DeinterleaveUvAutoTest(1, 1, f1, f2);
        result = result && DeinterleaveUvAutoTest(1, 0, f1, f2);
        result = result && DeinterleaveUvAutoTest(0, 1, f1, f2);

        return result;
    }

    bool DeinterleaveUvAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Base::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options) && W >= Simd::Sse41::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Sse41::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options) && W >= Simd::Avx2::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Avx2::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Avx512bw::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif

#ifdef SIMD_AMXBF16_ENABLE
        if (Simd::AmxBf16::Enable && TestAmxBf16(options))
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::AmxBf16::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options) && W >= Simd::Neon::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Neon::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

        return result;
    }

    //------------------------------------------------------------------------------------------------

    namespace
    {
        struct Func3
        {
            typedef void(*FuncPtr)(
                const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
                uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride);

            FuncPtr func;
            String desc;

            Func3(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & bgr, View * b, View * g, View * r) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(bgr.data, bgr.stride, bgr.width, bgr.height, b ? b->data : 0, b ? b->stride : 0,
                    g ? g->data : 0, g ? g->stride : 0, r ? r->data : 0, r ? r->stride : 0);
            }

            void Update(int hasB, int hasG, int hasR)
            {
                desc = desc + "-" + std::to_string(hasB) + std::to_string(hasG) + std::to_string(hasR);
            }
        };
    }

#define FUNC3(function) Func3(function, #function)

    bool DeinterleaveBgrAutoTest(int width, int height, int hasB, int hasG, int hasR, Func3 f1, Func3 f2)
    {
        bool result = true;

        f1.Update(hasB, hasG, hasR);
        f2.Update(hasB, hasG, hasR);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View bgr(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        FillRandom(bgr);

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bgr, hasB ? &b1 : 0, hasG ? &g1 : 0, hasR ? &r1 : 0));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bgr, hasB ? &b2 : 0, hasG ? &g2 : 0, hasR ? &r2 : 0));

        if (hasB) result = result && Compare(b1, b2, 0, true, 64, 0, "b");
        if (hasG) result = result && Compare(g1, g2, 0, true, 64, 0, "g");
        if (hasR) result = result && Compare(r1, r2, 0, true, 64, 0, "r");

        return result;
    }

    bool DeinterleaveBgrAutoTest(int hasB, int hasG, int hasR, const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        result = result && DeinterleaveBgrAutoTest(W, H, hasB, hasG, hasR, f1, f2);
        result = result && DeinterleaveBgrAutoTest(W + O, H - O, hasB, hasG, hasR, f1, f2);

        return result;
    }

    bool DeinterleaveBgrAutoTest(const Func3& f1, const Func3& f2)
    {
        bool result = true;

        result = result && DeinterleaveBgrAutoTest(1, 1, 1, f1, f2);
        result = result && DeinterleaveBgrAutoTest(1, 1, 0, f1, f2);
        result = result && DeinterleaveBgrAutoTest(1, 0, 1, f1, f2);
        result = result && DeinterleaveBgrAutoTest(0, 1, 1, f1, f2);
        result = result && DeinterleaveBgrAutoTest(1, 0, 0, f1, f2);
        result = result && DeinterleaveBgrAutoTest(0, 1, 0, f1, f2);
        result = result && DeinterleaveBgrAutoTest(0, 0, 1, f1, f2);

        return result;
    }

    bool DeinterleaveBgrAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Base::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options) && W >= Simd::Sse41::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Sse41::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options) && W >= Simd::Avx2::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Avx2::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Avx512bw::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_AMXBF16_ENABLE
        if (Simd::AmxBf16::Enable && TestAmxBf16(options))
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::AmxBf16::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options) && W >= Simd::Neon::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Neon::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif

        return result;
    }

    //------------------------------------------------------------------------------------------------

    namespace
    {
        struct Func4
        {
            typedef void(*FuncPtr)(
                const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
                uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride);

            FuncPtr func;
            String desc;

            Func4(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & bgra, View * b, View * g, View * r, View * a) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(bgra.data, bgra.stride, bgra.width, bgra.height, b ? b->data : 0, b ? b->stride : 0,
                    g ? g->data : 0, g ? g->stride : 0, r ? r->data : 0, r ? r->stride : 0, a ? a->data : 0, a ? a->stride : 0);
            }

            void Update(int hasB, int hasG, int hasR, int hasA)
            {
                desc = desc + "-" + std::to_string(hasB) + std::to_string(hasG) + std::to_string(hasR) + std::to_string(hasA);
            }
        };
    }

#define FUNC4(function) Func4(function, #function)

    bool DeinterleaveBgraAutoTest(int width, int height, int hasB, int hasG, int hasR, int hasA, Func4 f1, Func4 f2)
    {
        bool result = true;

        f1.Update(hasB, hasG, hasR, hasA);
        f2.Update(hasB, hasG, hasR, hasA);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View bgra(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        FillRandom(bgra);

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bgra, hasB ? &b1 : 0, hasG ? &g1 : 0, hasR ? &r1 : 0, hasA ? &a1 : 0));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bgra, hasB ? &b2 : 0, hasG ? &g2 : 0, hasR ? &r2 : 0, hasA ? &a2 : 0));

        if (hasB) result = result && Compare(b1, b2, 0, true, 64, 0, "b");
        if (hasG) result = result && Compare(g1, g2, 0, true, 64, 0, "g");
        if (hasR) result = result && Compare(r1, r2, 0, true, 64, 0, "r");
        if (hasA) result = result && Compare(a1, a2, 0, true, 64, 0, "a");

        return result;
    }

    bool DeinterleaveBgraAutoTest(int hasB, int hasG, int hasR, int hasA, const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        result = result && DeinterleaveBgraAutoTest(W, H, hasB, hasG, hasR, hasA, f1, f2);
        result = result && DeinterleaveBgraAutoTest(W + O, H - O, hasB, hasG, hasR, hasA, f1, f2);

        return result;
    }

    bool DeinterleaveBgraAutoTest(const Func4& f1, const Func4& f2)
    {
        bool result = true;

        result = result && DeinterleaveBgraAutoTest(1, 1, 1, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 1, 1, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 1, 0, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 1, 0, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 0, 1, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 0, 1, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 0, 0, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(1, 0, 0, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 1, 1, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 1, 1, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 1, 0, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 1, 0, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 0, 1, 1, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 0, 1, 0, f1, f2);
        result = result && DeinterleaveBgraAutoTest(0, 0, 0, 1, f1, f2);

        return result;
    }

    bool DeinterleaveBgraAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Base::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options) && W >= Simd::Sse41::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Sse41::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options) && W >= Simd::Avx2::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Avx2::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Avx512bw::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_AMXBF16_ENABLE
        if (Simd::AmxBf16::Enable && TestAmxBf16(options))
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::AmxBf16::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options) && W >= Simd::Neon::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Neon::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

        return result;
    }
}
