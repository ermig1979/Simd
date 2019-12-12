/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
        float alpha = 1.1f;

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

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncHswish32f
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * shift, const float * scale, float * dst);

            FuncPtr func;
            String desc;

            FuncHswish32f(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const Tensor32f & src, const float & shift, const float & scale, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Size(), &shift, &scale, dst.Data());
            }
        };
    }

#define FUNC_HSWISH32F(func) FuncHswish32f(func, #func)

    bool SynetHswish32fAutoTest(size_t size, const FuncHswish32f & f1, const FuncHswish32f & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << size << "].");

        Tensor32f src(Shape({ size }));
        Tensor32f dst1(Shape({ size }));
        Tensor32f dst2(Shape({ size }));

        FillRandom(src, -10.0, 10.0);
        float shift = 3.0f;
        float scale = 1.0f / 6.0f;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, shift, scale, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, shift, scale, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetHswish32fAutoTest(const FuncHswish32f & f1, const FuncHswish32f & f2)
    {
        bool result = true;

        result = result && SynetHswish32fAutoTest(W * H, f1, f2);
        result = result && SynetHswish32fAutoTest(W * H - O, f1, f2);

        return result;
    }

    bool SynetHswish32fAutoTest()
    {
        bool result = true;

        result = result && SynetHswish32fAutoTest(FUNC_HSWISH32F(Simd::Base::SynetHswish32f), FUNC_HSWISH32F(SimdSynetHswish32f));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetHswish32fAutoTest(FUNC_HSWISH32F(Simd::Sse::SynetHswish32f), FUNC_HSWISH32F(SimdSynetHswish32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetHswish32fAutoTest(FUNC_HSWISH32F(Simd::Avx::SynetHswish32f), FUNC_HSWISH32F(SimdSynetHswish32f));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetHswish32fAutoTest(FUNC_HSWISH32F(Simd::Avx512f::SynetHswish32f), FUNC_HSWISH32F(SimdSynetHswish32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetHswish32fAutoTest(FUNC_HSWISH32F(Simd::Neon::SynetHswish32f), FUNC_HSWISH32F(SimdSynetHswish32f));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncRR
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * lower, const float * upper, float * dst);

            FuncPtr func;
            String desc;

            FuncRR(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, float lower, float upper, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, src.width, &lower, &upper, (float*)dst.data);
            }
        };
    }

#define FUNC_RR(function) FuncRR(function, #function)

    bool SynetRestrictRange32fAutoTest(size_t size, const FuncRR & f1, const FuncRR & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -1.0f, upper = 1.0f;
        FillRandom32f(src, 2.0f*lower, 2.0f*upper);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, lower, upper, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, lower, upper, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetRestrictRange32fAutoTest(const FuncRR & f1, const FuncRR & f2)
    {
        bool result = true;

        result = result && SynetRestrictRange32fAutoTest(H*W, f1, f2);
        result = result && SynetRestrictRange32fAutoTest(H*W + O, f1, f2);

        return result;
    }

    bool SynetRestrictRange32fAutoTest()
    {
        bool result = true;

        result = result && SynetRestrictRange32fAutoTest(FUNC_RR(Simd::Base::SynetRestrictRange32f), FUNC_RR(SimdSynetRestrictRange32f));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetRestrictRange32fAutoTest(FUNC_RR(Simd::Sse::SynetRestrictRange32f), FUNC_RR(SimdSynetRestrictRange32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetRestrictRange32fAutoTest(FUNC_RR(Simd::Avx::SynetRestrictRange32f), FUNC_RR(SimdSynetRestrictRange32f));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetRestrictRange32fAutoTest(FUNC_RR(Simd::Avx512f::SynetRestrictRange32f), FUNC_RR(SimdSynetRestrictRange32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetRestrictRange32fAutoTest(FUNC_RR(Simd::Neon::SynetRestrictRange32f), FUNC_RR(SimdSynetRestrictRange32f));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncSP
        {
            typedef void(*FuncPtr)(const float* src, size_t size, const float* beta, const float* threshold, float* dst);

            FuncPtr func;
            String desc;

            FuncSP(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Call(const View& src, float beta, float threshold, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, src.width, &beta, &threshold, (float*)dst.data);
            }
        };
    }

#define FUNC_SP(function) FuncSP(function, #function)

    bool SynetSoftplus32fAutoTest(size_t size, const FuncSP& f1, const FuncSP& f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float beta = 3.0f, threshold = 20.0f;
        FillRandom32f(src, -10.0f, 10.0f);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, beta, threshold, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, beta, threshold, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetSoftplus32fAutoTest(const FuncSP& f1, const FuncSP& f2)
    {
        bool result = true;

        result = result && SynetSoftplus32fAutoTest(H * W, f1, f2);
        result = result && SynetSoftplus32fAutoTest(H * W + O, f1, f2);

        return result;
    }

    bool SynetSoftplus32fAutoTest()
    {
        bool result = true;

        result = result && SynetSoftplus32fAutoTest(FUNC_SP(Simd::Base::SynetSoftplus32f), FUNC_SP(SimdSynetSoftplus32f));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetSoftplus32fAutoTest(FUNC_SP(Simd::Sse2::SynetSoftplus32f), FUNC_SP(SimdSynetSoftplus32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetSoftplus32fAutoTest(FUNC_SP(Simd::Avx2::SynetSoftplus32f), FUNC_SP(SimdSynetSoftplus32f));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetSoftplus32fAutoTest(FUNC_SP(Simd::Avx512f::SynetSoftplus32f), FUNC_SP(SimdSynetSoftplus32f));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetSoftplus32fAutoTest(FUNC_SP(Simd::Neon::SynetSoftplus32f), FUNC_SP(SimdSynetSoftplus32f));
#endif 

        return result;
    }
}
