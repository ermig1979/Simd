/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Test/TestTensor.h"
#include "Test/TestString.h"

namespace Test
{
    namespace
    {
        struct FuncSB
        {
            typedef void(*FuncPtr)(const float * src, size_t size, uint16_t * dst);

            FuncPtr func;
            String description;

            FuncSB(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const float*)src.data, src.width, (uint16_t*)dst.data);
            }
        };
    }

#define FUNC_SB(function) FuncSB(function, #function)

    bool Float32ToBFloat16AutoTest(size_t size, const FuncSB & f1, const FuncSB & f2)
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

    bool Float32ToBFloat16AutoTest(const FuncSB & f1, const FuncSB & f2)
    {
        bool result = true;

        result = result && Float32ToBFloat16AutoTest(W*H, f1, f2);
        result = result && Float32ToBFloat16AutoTest(W*H - 1, f1, f2);

        return result;
    }

    bool Float32ToBFloat16AutoTest()
    {
        bool result = true;

        result = result && Float32ToBFloat16AutoTest(FUNC_SB(Simd::Base::Float32ToBFloat16), FUNC_SB(SimdFloat32ToBFloat16));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Float32ToBFloat16AutoTest(FUNC_SB(Simd::Sse41::Float32ToBFloat16), FUNC_SB(SimdFloat32ToBFloat16));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Float32ToBFloat16AutoTest(FUNC_SB(Simd::Avx2::Float32ToBFloat16), FUNC_SB(SimdFloat32ToBFloat16));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Float32ToBFloat16AutoTest(FUNC_SB(Simd::Avx512bw::Float32ToBFloat16), FUNC_SB(SimdFloat32ToBFloat16));
#endif 

#ifdef SIMD_AVX512BF16_ENABLE
        if (Simd::Avx512bf16::Enable)
            result = result && Float32ToBFloat16AutoTest(FUNC_SB(Simd::Avx512bf16::Float32ToBFloat16), FUNC_SB(SimdFloat32ToBFloat16));
#endif 

        return result;
    }

    //---------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncBS
        {
            typedef void(*FuncPtr)(const uint16_t* src, size_t size, float* dst);

            FuncPtr func;
            String description;

            FuncBS(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Call(const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((const uint16_t*)src.data, src.width, (float*)dst.data);
            }
        };
    }

#define FUNC_BS(function) FuncBS(function, #function)

    bool BFloat16ToFloat32AutoTest(size_t size, const FuncBS& f1, const FuncBS& f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View origin(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View src(size, 1, View::Int16, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(origin, -10.0, 10.0);
        ::SimdFloat32ToBFloat16((const float*)origin.data, size, (uint16_t*)src.data);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32);

        return result;
    }

    bool BFloat16ToFloat32AutoTest(const FuncBS& f1, const FuncBS& f2)
    {
        bool result = true;

        result = result && BFloat16ToFloat32AutoTest(W * H, f1, f2);
        result = result && BFloat16ToFloat32AutoTest(W * H - 1, f1, f2);

        return result;
    }

    bool BFloat16ToFloat32AutoTest()
    {
        bool result = true;

        result = result && BFloat16ToFloat32AutoTest(FUNC_BS(Simd::Base::BFloat16ToFloat32), FUNC_BS(SimdBFloat16ToFloat32));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && BFloat16ToFloat32AutoTest(FUNC_BS(Simd::Sse41::BFloat16ToFloat32), FUNC_BS(SimdBFloat16ToFloat32));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && BFloat16ToFloat32AutoTest(FUNC_BS(Simd::Avx2::BFloat16ToFloat32), FUNC_BS(SimdBFloat16ToFloat32));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && BFloat16ToFloat32AutoTest(FUNC_BS(Simd::Avx512bw::BFloat16ToFloat32), FUNC_BS(SimdBFloat16ToFloat32));
#endif 

        return result;
    }
}
