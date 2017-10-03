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

namespace Test
{
    namespace
    {
        struct FuncFB
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst);

            FuncPtr func;
            String description;

            FuncFB(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float lower, float upper, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const float*)src.data, src.width, &lower, &upper, dst.data);
            }
        };
    }

#define FUNC_FB(function) FuncFB(function, #function)

    bool Float32ToUint8AutoTest(size_t size, const FuncFB & f1, const FuncFB & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -0.10, upper = 0.10;
        FillRandom32f(src, -0.11, 0.11);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, lower, upper, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, lower, upper, dst2));

        result = result && Compare(dst1, dst2, 1, true, 32);

        return result;
    }

    bool Float32ToUint8AutoTest(const FuncFB & f1, const FuncFB & f2)
    {
        bool result = true;

        result = result && Float32ToUint8AutoTest(W*H, f1, f2);
        result = result && Float32ToUint8AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float32ToUint8AutoTest()
    {
        bool result = true;

        result = result && Float32ToUint8AutoTest(FUNC_FB(Simd::Base::Float32ToUint8), FUNC_FB(SimdFloat32ToUint8));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && Float32ToUint8AutoTest(FUNC_FB(Simd::Sse2::Float32ToUint8), FUNC_FB(SimdFloat32ToUint8));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Float32ToUint8AutoTest(FUNC_FB(Simd::Avx2::Float32ToUint8), FUNC_FB(SimdFloat32ToUint8));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Float32ToUint8AutoTest(FUNC_FB(Simd::Avx512bw::Float32ToUint8), FUNC_FB(SimdFloat32ToUint8));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Float32ToUint8AutoTest(FUNC_FB(Simd::Neon::Float32ToUint8), FUNC_FB(SimdFloat32ToUint8));
#endif 

        return result;
    }

    namespace
    {
        struct FuncBF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst);

            FuncPtr func;
            String description;

            FuncBF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float lower, float upper, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, &lower, &upper, (float*)dst.data);
            }
        };
    }

#define FUNC_BF(function) FuncBF(function, #function)

    bool Uint8ToFloat32AutoTest(size_t size, const FuncBF & f1, const FuncBF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -0.10, upper = 0.10;
        FillRandom(src);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, lower, upper, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, lower, upper, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32);

        return result;
    }

    bool Uint8ToFloat32AutoTest(const FuncBF & f1, const FuncBF & f2)
    {
        bool result = true;

        result = result && Uint8ToFloat32AutoTest(W*H, f1, f2);
        result = result && Uint8ToFloat32AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Uint8ToFloat32AutoTest()
    {
        bool result = true;

        result = result && Uint8ToFloat32AutoTest(FUNC_BF(Simd::Base::Uint8ToFloat32), FUNC_BF(SimdUint8ToFloat32));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && Uint8ToFloat32AutoTest(FUNC_BF(Simd::Sse2::Uint8ToFloat32), FUNC_BF(SimdUint8ToFloat32));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Uint8ToFloat32AutoTest(FUNC_BF(Simd::Avx2::Uint8ToFloat32), FUNC_BF(SimdUint8ToFloat32));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Uint8ToFloat32AutoTest(FUNC_BF(Simd::Avx512bw::Uint8ToFloat32), FUNC_BF(SimdUint8ToFloat32));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Uint8ToFloat32AutoTest(FUNC_BF(Simd::Neon::Uint8ToFloat32), FUNC_BF(SimdUint8ToFloat32));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool Float32ToUint8DataTest(bool create, size_t size, const FuncFB & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -0.10, upper = 0.10;

        if (create)
        {
            FillRandom32f(src, -0.11, 0.11);

            TEST_SAVE(src);

            f.Call(src, lower, upper, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, lower, upper, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 1, true, 32);
        }

        return result;
    }

    bool Float32ToUint8DataTest(bool create)
    {
        bool result = true;

        result = result && Float32ToUint8DataTest(create, DH, FUNC_FB(SimdFloat32ToUint8));

        return result;
    }

    bool Uint8ToFloat32DataTest(bool create, size_t size, const FuncBF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Gray8, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -0.10, upper = 0.10;

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, lower, upper, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, lower, upper, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32);
        }

        return result;
    }

    bool Uint8ToFloat32DataTest(bool create)
    {
        bool result = true;

        result = result && Uint8ToFloat32DataTest(create, DH, FUNC_BF(SimdUint8ToFloat32));

        return result;
    }
}
