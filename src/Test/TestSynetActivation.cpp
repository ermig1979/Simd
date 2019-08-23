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
#include "Test/TestTensor.h"

namespace Test
{
    namespace
    {
        struct FuncElu32f
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * alpha, float * dst);

            FuncPtr func;
            String desc;

            FuncElu32f(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const Tensor32f & src, const float & alpha, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Size(), &alpha, dst.Data());
            }
        };
    }

#define FUNC_ELU32F(func) FuncElu32f(func, #func)

    bool SynetElu32fAutoTest(size_t size, const FuncElu32f & f1, const FuncElu32f & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << size << "].");

        Tensor32f src( Shape({ size }));
        Tensor32f dst1(Shape({ size }));
        Tensor32f dst2(Shape({ size }));

        FillRandom(src, -10.0, 10.0);
        float alpha = 1.1;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, alpha, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, alpha, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetElu32fAutoTest(const FuncElu32f & f1, const FuncElu32f & f2)
    {
        bool result = true;

        result = result && SynetElu32fAutoTest(W * H, f1, f2);
        result = result && SynetElu32fAutoTest(W * H - O , f1, f2);

        return result;
    }

    bool SynetElu32fAutoTest()
    {
        bool result = true;

        result = result && SynetElu32fAutoTest(FUNC_ELU32F(Simd::Base::SynetElu32f), FUNC_ELU32F(SimdSynetElu32f));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetElu32fAutoTest(FUNC_ELU32F(Simd::Sse2::SynetElu32f), FUNC_ELU32F(SimdSynetElu32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetElu32fAutoTest(FUNC_ELU32F(Simd::Avx2::SynetElu32f), FUNC_ELU32F(SimdSynetElu32f));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetElu32fAutoTest(FUNC_ELU32F(Simd::Avx512f::SynetElu32f), FUNC_ELU32F(SimdSynetElu32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetElu32fAutoTest(FUNC_ELU32F(Simd::Neon::SynetElu32f), FUNC_ELU32F(SimdSynetElu32f));
#endif 

        return result;
    }
}
