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
        struct FuncSH
        {
            typedef void (*FuncPtr)(const float * src, size_t size, uint16_t * dst);

            FuncPtr func;
            String description;

            FuncSH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const float*)src.data, src.width, (uint16_t*)dst.data);
            }
        };       
    }

#define FUNC_SH(function) FuncSH(function, #function)

    bool Float32ToFloat16AutoTest(size_t size, const FuncSH & f1, const FuncSH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, 1, true, 32);

        return result;
    }

    bool Float32ToFloat16AutoTest(const FuncSH & f1, const FuncSH & f2)
    {
        bool result = true;

        result = result && Float32ToFloat16AutoTest(W*H, f1, f2);
        result = result && Float32ToFloat16AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float32ToFloat16AutoTest()
    {
        bool result = true;

        result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Base::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));

#ifdef SIMD_AVX2_ENABLE
        if(Simd::Avx2::Enable)
            result = result && Float32ToFloat16AutoTest(FUNC_SH(Simd::Avx2::Float32ToFloat16), FUNC_SH(SimdFloat32ToFloat16));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHS
        {
            typedef void(*FuncPtr)(const uint16_t * src, size_t size, float * dst);

            FuncPtr func;
            String description;

            FuncHS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const uint16_t*)src.data, src.width, (float*)dst.data);
            }
        };
    }

#define FUNC_HS(function) FuncHS(function, #function)

    bool Float16ToFloat32AutoTest(size_t size, const FuncHS & f1, const FuncHS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View origin(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(origin, -10.0, 10.0);
        ::SimdFloat32ToFloat16((const float*)origin.data, size, (uint16_t*)src.data);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32);

        return result;
    }

    bool Float16ToFloat32AutoTest(const FuncHS & f1, const FuncHS & f2)
    {
        bool result = true;

        result = result && Float16ToFloat32AutoTest(W*H, f1, f2);
        result = result && Float16ToFloat32AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float16ToFloat32AutoTest()
    {
        bool result = true;

        result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Base::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Float16ToFloat32AutoTest(FUNC_HS(Simd::Avx2::Float16ToFloat32), FUNC_HS(SimdFloat16ToFloat32));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool Float32ToFloat16DataTest(bool create, size_t size, const FuncSH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));

        if(create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 1, true, 32);
        }

        return result;
    }

    bool Float32ToFloat16DataTest(bool create)
    {
        bool result = true;

        result = result && Float32ToFloat16DataTest(create, DW*DH, FUNC_SH(SimdFloat32ToFloat16));

        return result;
    }

    bool Float16ToFloat32DataTest(bool create, size_t size, const FuncHS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            View origin(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(origin, -10.0, 10.0);
            ::SimdFloat32ToFloat16((const float*)origin.data, size, (uint16_t*)src.data);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32);
        }

        return result;
    }

    bool Float16ToFloat32DataTest(bool create)
    {
        bool result = true;

        result = result && Float16ToFloat32DataTest(create, DW*DH, FUNC_HS(SimdFloat16ToFloat32));

        return result;
    }
}
