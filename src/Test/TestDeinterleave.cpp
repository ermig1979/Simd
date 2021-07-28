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
        struct Func2
        {
            typedef void(*FuncPtr)(
                const uint8_t * uv, size_t uvStride, size_t width, size_t height,
                uint8_t * u, size_t uStride, uint8_t * v, size_t vStride);

            FuncPtr func;
            String description;

            Func2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & uv, View & u, View & v) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(uv.data, uv.stride, uv.width, uv.height, u.data, u.stride, v.data, v.stride);
            }
        };
    }

#define FUNC2(function) Func2(function, #function)

    bool DeinterleaveUvAutoTest(int width, int height, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));
        FillRandom(uv);

        View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(uv, u1, v1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(uv, u2, v2));

        result = result && Compare(u1, u2, 0, true, 32, 0, "u");
        result = result && Compare(v1, v2, 0, true, 32, 0, "v");

        return result;
    }

    bool DeinterleaveUvAutoTest(const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        result = result && DeinterleaveUvAutoTest(W, H, f1, f2);
        result = result && DeinterleaveUvAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool DeinterleaveUvAutoTest()
    {
        bool result = true;

        result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Base::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Sse2::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Avx2::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Avx512bw::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Vmx::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && DeinterleaveUvAutoTest(FUNC2(Simd::Neon::DeinterleaveUv), FUNC2(SimdDeinterleaveUv));
#endif 

        return result;
    }

    namespace
    {
        struct Func3
        {
            typedef void(*FuncPtr)(
                const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
                uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride);

            FuncPtr func;
            String description;

            Func3(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & bgr, View & b, View & g, View & r) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(bgr.data, bgr.stride, bgr.width, bgr.height, b.data, b.stride, g.data, g.stride, r.data, r.stride);
            }
        };
    }

#define FUNC3(function) Func3(function, #function)

    bool DeinterleaveBgrAutoTest(int width, int height, const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View bgr(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        FillRandom(bgr);

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bgr, b1, g1, r1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bgr, b2, g2, r2));

        result = result && Compare(b1, b2, 0, true, 32, 0, "b");
        result = result && Compare(g1, g2, 0, true, 32, 0, "g");
        result = result && Compare(r1, r2, 0, true, 32, 0, "r");

        return result;
    }

    bool DeinterleaveBgrAutoTest(const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        result = result && DeinterleaveBgrAutoTest(W, H, f1, f2);
        result = result && DeinterleaveBgrAutoTest(W + O, H - O, f1, f2);

        return result;
    }

    bool DeinterleaveBgrAutoTest()
    {
        bool result = true;

        result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Base::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Sse41::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Avx2::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Avx512bw::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && DeinterleaveBgrAutoTest(FUNC3(Simd::Neon::DeinterleaveBgr), FUNC3(SimdDeinterleaveBgr));
#endif

        return result;
    }

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

            void Update(bool alpha)
            {
                desc = desc + (alpha ? "[1]" : "[0]");
            }

            void Call(const View & bgra, View & b, View & g, View & r, View & a) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(bgra.data, bgra.stride, bgra.width, bgra.height, b.data, b.stride, g.data, g.stride, r.data, r.stride, a.data, a.stride);
            }
        };
    }

#define FUNC4(function) Func4(function, #function)

    bool DeinterleaveBgraAutoTest(bool alpha, int width, int height, Func4 f1, Func4 f2)
    {
        bool result = true;

        f1.Update(alpha);
        f2.Update(alpha);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View bgra(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        FillRandom(bgra);

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a1, a2;
        if (alpha)
        {
            a1.Recreate(width, height, View::Gray8, NULL, TEST_ALIGN(width));
            a2.Recreate(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bgra, b1, g1, r1, a1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bgra, b2, g2, r2, a2));

        result = result && Compare(b1, b2, 0, true, 32, 0, "b");
        result = result && Compare(g1, g2, 0, true, 32, 0, "g");
        result = result && Compare(r1, r2, 0, true, 32, 0, "r");
        if(alpha)
            result = result && Compare(a1, a2, 0, true, 32, 0, "a");

        return result;
    }

    bool DeinterleaveBgraAutoTest(const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        result = result && DeinterleaveBgraAutoTest(true, W, H, f1, f2);
        result = result && DeinterleaveBgraAutoTest(true, W + O, H - O, f1, f2);
        result = result && DeinterleaveBgraAutoTest(false, W, H, f1, f2);
        result = result && DeinterleaveBgraAutoTest(false, W + O, H - O, f1, f2);

        return result;
    }

    bool DeinterleaveBgraAutoTest()
    {
        bool result = true;

        result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Base::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Sse41::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Avx2::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Avx512bw::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && DeinterleaveBgraAutoTest(FUNC4(Simd::Neon::DeinterleaveBgra), FUNC4(SimdDeinterleaveBgra));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool DeinterleaveUvDataTest(bool create, int width, int height, const Func2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View uv(width, height, View::Uv16, NULL, TEST_ALIGN(width));

        View u1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View u2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(uv);

            TEST_SAVE(uv);

            f.Call(uv, u1, v1);

            TEST_SAVE(u1);
            TEST_SAVE(v1);
        }
        else
        {
            TEST_LOAD(uv);

            TEST_LOAD(u1);
            TEST_LOAD(v1);

            f.Call(uv, u2, v2);

            TEST_SAVE(u2);
            TEST_SAVE(v2);

            result = result && Compare(u1, u2, 0, true, 32, 0, "u");
            result = result && Compare(v1, v2, 0, true, 32, 0, "v");
        }

        return result;
    }

    bool DeinterleaveUvDataTest(bool create)
    {
        bool result = true;

        result = result && DeinterleaveUvDataTest(create, DW, DH, FUNC2(SimdDeinterleaveUv));

        return result;
    }

    bool DeinterleaveBgrDataTest(bool create, int width, int height, const Func3 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View bgr(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(bgr);

            TEST_SAVE(bgr);

            f.Call(bgr, b1, g1, r1);

            TEST_SAVE(b1);
            TEST_SAVE(g1);
            TEST_SAVE(r1);
        }
        else
        {
            TEST_LOAD(bgr);

            TEST_LOAD(b1);
            TEST_LOAD(g1);
            TEST_LOAD(r1);

            f.Call(bgr, b2, g2, r2);

            TEST_SAVE(b2);
            TEST_SAVE(g2);
            TEST_SAVE(r2);

            result = result && Compare(b1, b2, 0, true, 32, 0, "b");
            result = result && Compare(g1, g2, 0, true, 32, 0, "g");
            result = result && Compare(r1, r2, 0, true, 32, 0, "r");
        }

        return result;
    }

    bool DeinterleaveBgrDataTest(bool create)
    {
        bool result = true;

        result = result && DeinterleaveBgrDataTest(create, DW, DH, FUNC3(SimdDeinterleaveBgr));

        return result;
    }

    bool DeinterleaveBgraDataTest(bool create, int width, int height, const Func4 & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << width << ", " << height << "].");

        View bgra(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        View b1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View b2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r2(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(bgra);

            TEST_SAVE(bgra);

            f.Call(bgra, b1, g1, r1, a1);

            TEST_SAVE(b1);
            TEST_SAVE(g1);
            TEST_SAVE(r1);
            TEST_SAVE(a1);
        }
        else
        {
            TEST_LOAD(bgra);

            TEST_LOAD(b1);
            TEST_LOAD(g1);
            TEST_LOAD(r1);
            TEST_LOAD(a1);

            f.Call(bgra, b2, g2, r2, a2);

            TEST_SAVE(b2);
            TEST_SAVE(g2);
            TEST_SAVE(r2);
            TEST_SAVE(a2);

            result = result && Compare(b1, b2, 0, true, 32, 0, "b");
            result = result && Compare(g1, g2, 0, true, 32, 0, "g");
            result = result && Compare(r1, r2, 0, true, 32, 0, "r");
            result = result && Compare(a1, a2, 0, true, 32, 0, "a");
        }

        return result;
    }

    bool DeinterleaveBgraDataTest(bool create)
    {
        bool result = true;

        result = result && DeinterleaveBgraDataTest(create, DW, DH, FUNC4(SimdDeinterleaveBgra));

        return result;
    }
}
