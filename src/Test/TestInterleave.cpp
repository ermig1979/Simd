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
            typedef void(*FuncPtr)(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
                size_t width, size_t height, uint8_t * uv, size_t uvStride);

            FuncPtr func;
            String description;

            Func2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & u, const View & v, View & uv) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(u.data, u.stride, v.data, v.stride, u.width, u.height, uv.data, uv.stride);
            }
        };
    }
#define FUNC2(function) Func2(function, #function)

    bool InterleaveUvAutoTest(int width, int height, const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View u(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(u);
        View v(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(v);

        View uv1(width, height, View::Uv16, NULL, TEST_ALIGN(width));
        View uv2(width, height, View::Uv16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(u, v, uv1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(u, v, uv2));

        result = result && Compare(uv1, uv2, 0, true, 32);

        return result;
    }

    bool InterleaveUvAutoTest(const Func2 & f1, const Func2 & f2)
    {
        bool result = true;

        result = result && InterleaveUvAutoTest(W, H, f1, f2);
        result = result && InterleaveUvAutoTest(W + O, H - O, f1, f2);
        result = result && InterleaveUvAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool InterleaveUvAutoTest()
    {
        bool result = true;

        result = result && InterleaveUvAutoTest(FUNC2(Simd::Base::InterleaveUv), FUNC2(SimdInterleaveUv));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && InterleaveUvAutoTest(FUNC2(Simd::Sse2::InterleaveUv), FUNC2(SimdInterleaveUv));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterleaveUvAutoTest(FUNC2(Simd::Avx2::InterleaveUv), FUNC2(SimdInterleaveUv));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterleaveUvAutoTest(FUNC2(Simd::Avx512bw::InterleaveUv), FUNC2(SimdInterleaveUv));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && InterleaveUvAutoTest(FUNC2(Simd::Vmx::InterleaveUv), FUNC2(SimdInterleaveUv));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterleaveUvAutoTest(FUNC2(Simd::Neon::InterleaveUv), FUNC2(SimdInterleaveUv));
#endif

        return result;
    }

    namespace
    {
        struct Func3
        {
            typedef void(*FuncPtr)(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride,
                size_t width, size_t height, uint8_t * bgr, size_t bgrStride);

            FuncPtr func;
            String description;

            Func3(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & b, const View & g, const View & r, View & bgr) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(b.data, b.stride, g.data, g.stride, r.data, r.stride, bgr.width, bgr.height, bgr.data, bgr.stride);
            }
        };
    }
#define FUNC3(function) Func3(function, #function)

    bool InterleaveBgrAutoTest(int width, int height, const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(b);
        View g(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(g);
        View r(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(r);

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(b, g, r, bgr1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(b, g, r, bgr2));

        result = result && Compare(bgr1, bgr2, 0, true, 64);

        return result;
    }

    bool InterleaveBgrAutoTest(const Func3 & f1, const Func3 & f2)
    {
        bool result = true;

        result = result && InterleaveBgrAutoTest(W, H, f1, f2);
        result = result && InterleaveBgrAutoTest(W + O, H - O, f1, f2);
        result = result && InterleaveBgrAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool InterleaveBgrAutoTest()
    {
        bool result = true;

        result = result && InterleaveBgrAutoTest(FUNC3(Simd::Base::InterleaveBgr), FUNC3(SimdInterleaveBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && InterleaveBgrAutoTest(FUNC3(Simd::Sse41::InterleaveBgr), FUNC3(SimdInterleaveBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterleaveBgrAutoTest(FUNC3(Simd::Avx2::InterleaveBgr), FUNC3(SimdInterleaveBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterleaveBgrAutoTest(FUNC3(Simd::Avx512bw::InterleaveBgr), FUNC3(SimdInterleaveBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterleaveBgrAutoTest(FUNC3(Simd::Neon::InterleaveBgr), FUNC3(SimdInterleaveBgr));
#endif

        return result;
    }

    namespace
    {
        struct Func4
        {
            typedef void(*FuncPtr)(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride,
                size_t width, size_t height, uint8_t * bgra, size_t bgraStride);

            FuncPtr func;
            String description;

            Func4(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & b, const View & g, const View & r, const View & a, View & bgra) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(b.data, b.stride, g.data, g.stride, r.data, r.stride, a.data, a.stride, bgra.width, bgra.height, bgra.data, bgra.stride);
            }
        };
    }
#define FUNC4(function) Func4(function, #function)

    bool InterleaveBgraAutoTest(int width, int height, const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(b);
        View g(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(g);
        View r(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(r);
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(b, g, r, a, bgra1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(b, g, r, a, bgra2));

        result = result && Compare(bgra1, bgra2, 0, true, 64);

        return result;
    }

    bool InterleaveBgraAutoTest(const Func4 & f1, const Func4 & f2)
    {
        bool result = true;

        result = result && InterleaveBgraAutoTest(W, H, f1, f2);
        result = result && InterleaveBgraAutoTest(W + O, H - O, f1, f2);
        result = result && InterleaveBgraAutoTest(W - O, H + O, f1, f2);

        return result;
    }

    bool InterleaveBgraAutoTest()
    {
        bool result = true;

        result = result && InterleaveBgraAutoTest(FUNC4(Simd::Base::InterleaveBgra), FUNC4(SimdInterleaveBgra));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && InterleaveBgraAutoTest(FUNC4(Simd::Sse41::InterleaveBgra), FUNC4(SimdInterleaveBgra));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && InterleaveBgraAutoTest(FUNC4(Simd::Avx2::InterleaveBgra), FUNC4(SimdInterleaveBgra));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && InterleaveBgraAutoTest(FUNC4(Simd::Avx512bw::InterleaveBgra), FUNC4(SimdInterleaveBgra));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && InterleaveBgraAutoTest(FUNC4(Simd::Neon::InterleaveBgra), FUNC4(SimdInterleaveBgra));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool InterleaveUvDataTest(bool create, int width, int height, const Func2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View u(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View v(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View uv1(width, height, View::Uv16, NULL, TEST_ALIGN(width));
        View uv2(width, height, View::Uv16, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(u);
            FillRandom(v);

            TEST_SAVE(u);
            TEST_SAVE(v);

            f.Call(u, v, uv1);

            TEST_SAVE(uv1);
        }
        else
        {
            TEST_LOAD(u);
            TEST_LOAD(v);

            TEST_LOAD(uv1);

            f.Call(u, v, uv2);

            TEST_SAVE(uv2);

            result = result && Compare(uv1, uv2, 0, true, 32);
        }

        return result;
    }

    bool InterleaveUvDataTest(bool create)
    {
        bool result = true;

        result = result && InterleaveUvDataTest(create, DW, DH, FUNC2(SimdInterleaveUv));

        return result;
    }

    bool InterleaveBgrDataTest(bool create, int width, int height, const Func3 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View bgr1(width, height, View::Bgr24, NULL, TEST_ALIGN(width));
        View bgr2(width, height, View::Bgr24, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(b);
            FillRandom(g);
            FillRandom(r);

            TEST_SAVE(b);
            TEST_SAVE(g);
            TEST_SAVE(r);

            f.Call(b, g, r, bgr1);

            TEST_SAVE(bgr1);
        }
        else
        {
            TEST_LOAD(b);
            TEST_LOAD(g);
            TEST_LOAD(r);

            TEST_LOAD(bgr1);

            f.Call(b, g, r, bgr2);

            TEST_SAVE(bgr2);

            result = result && Compare(bgr1, bgr2, 0, true, 32);
        }

        return result;
    }

    bool InterleaveBgrDataTest(bool create)
    {
        bool result = true;

        result = result && InterleaveBgrDataTest(create, DW, DH, FUNC3(SimdInterleaveBgr));

        return result;
    }

    bool InterleaveBgraDataTest(bool create, int width, int height, const Func4 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View b(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View g(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View r(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View a(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View bgra1(width, height, View::Bgra32, NULL, TEST_ALIGN(width));
        View bgra2(width, height, View::Bgra32, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(b);
            FillRandom(g);
            FillRandom(r);
            FillRandom(a);

            TEST_SAVE(b);
            TEST_SAVE(g);
            TEST_SAVE(r);
            TEST_SAVE(a);

            f.Call(b, g, r, a, bgra1);

            TEST_SAVE(bgra1);
        }
        else
        {
            TEST_LOAD(b);
            TEST_LOAD(g);
            TEST_LOAD(r);
            TEST_LOAD(a);

            TEST_LOAD(bgra1);

            f.Call(b, g, r, a, bgra2);

            TEST_SAVE(bgra2);

            result = result && Compare(bgra1, bgra2, 0, true, 32);
        }

        return result;
    }

    bool InterleaveBgraDataTest(bool create)
    {
        bool result = true;

        result = result && InterleaveBgraDataTest(create, DW, DH, FUNC4(SimdInterleaveBgra));

        return result;
    }
}
