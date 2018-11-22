/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
        struct FuncAB
        {
            typedef void(*FuncPtr)(const float * bias, size_t count, size_t size, float * dst, SimdBool trans);

            FuncPtr func;
            String desc;

            FuncAB(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdBool trans)
            {
                desc = desc + (trans ? "[1]" : "[0]" );
            }

            void Call(const View & bias, size_t count, size_t size, SimdBool trans, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(desc);
                func((float*)bias.data, count, size, (float*)dstDst.data, trans);
            }
        };
    }

#define FUNC_AB(function) FuncAB(function, #function)

    bool SynetAddBiasAutoTest(size_t count, size_t size, SimdBool trans, FuncAB f1, FuncAB f2)
    {
        bool result = true;

        f1.Update(trans);
        f2.Update(trans);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(bias, -10.0, 10.0);
        FillRandom32f(dstSrc, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bias, count, size, trans, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bias, count, size, trans, dstSrc, dstDst2));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);

        return result;
    }

    bool SynetAddBiasAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(H, W, SimdFalse, f1, f2);
        result = result && SynetAddBiasAutoTest(H - O, W + O, SimdFalse, f1, f2);
        result = result && SynetAddBiasAutoTest(H, W, SimdTrue, f1, f2);
        result = result && SynetAddBiasAutoTest(H - O, W + O, SimdTrue, f1, f2);

        return result;
    }

    bool SynetAddBiasAutoTest()
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sse::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx512f::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

        return result;
    }

    SIMD_INLINE String ToString(SimdSynetEltwiseOperationType type)
    {
        switch (type)
        {
        case SimdSynetEltwiseOperationProduct:
            return "[Pro]";
        case SimdSynetEltwiseOperationSum:
            return "[Sum]";
        case SimdSynetEltwiseOperationMax:
            return "[Max]";
        case SimdSynetEltwiseOperationMin:
            return "[Min]";
        }
        assert(0);
        return "[U]";
    }

    namespace
    {
        struct FuncELF
        {
            typedef void(*FuncPtr)(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst);

            FuncPtr func;
            String desc;

            FuncELF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncELF(const FuncELF & f, SimdSynetEltwiseOperationType type, size_t count) : func(f.func), desc(f.desc + ToString(type) + "[" + ToString(count) + "]") {}

            void Call(FloatPtrs src, const View & weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.data(), (float*)weight.data, count, size, type, (float*)dst.data);
            }
        };
    }

#define FUNC_ELF(function) FuncELF(function, #function)
#define ARGS_ELF(count, type, f1, f2) count, type, FuncELF(f1, type, count), FuncELF(f2, type, count)

    bool SynetEltwiseLayerForwardAutoTest(size_t size, size_t count, SimdSynetEltwiseOperationType type, const FuncELF & f1, const FuncELF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(size, count, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(src, -1.0, 1.0);
        FloatPtrs psrc(count);
        for (size_t i = 0; i < count; ++i)
            psrc[i] = src.Row<float>(i);
        View weight(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(weight, -1.0, 1.0);
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(psrc, weight, count, size, type, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(psrc, weight, count, size, type, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetEltwiseLayerForwardAutoTest(const FuncELF & f1, const FuncELF & f2)
    {
        bool result = true;

        for (SimdSynetEltwiseOperationType type = SimdSynetEltwiseOperationProduct; type <= SimdSynetEltwiseOperationMin; type = (SimdSynetEltwiseOperationType)((size_t)type + 1))
        {
            for (size_t count = 2; count <= 3; ++count)
            {
                result = result && SynetEltwiseLayerForwardAutoTest(H*W, ARGS_ELF(count, type, f1, f2));
                result = result && SynetEltwiseLayerForwardAutoTest(H*W + O, ARGS_ELF(count, type, f1, f2));
            }
        }

        return result;
    }

    bool SynetEltwiseLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Base::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Sse::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx2::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx512f::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

        return result;
    }

    namespace
    {
        struct FuncFLF0
        {
            typedef void(*FuncPtr)(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncFLF0(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, const View & bias, const View & scale, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)bias.data, (float*)scale.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_FLF0(function) FuncFLF0(function, #function)

    bool SynetFusedLayerForward0AutoTest(size_t count, size_t size, const FuncFLF0 & f1, const FuncFLF0 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(scale, -10.0, 10.0);
        FillRandom32f(bias, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetFusedLayerForward0AutoTest(const FuncFLF0 & f1, const FuncFLF0 & f2)
    {
        bool result = true;

        result = result && SynetFusedLayerForward0AutoTest(H, W, f1, f2);
        result = result && SynetFusedLayerForward0AutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetFusedLayerForward0AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward0AutoTest(FUNC_FLF0(Simd::Base::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward0AutoTest(FUNC_FLF0(Simd::Sse::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward0AutoTest(FUNC_FLF0(Simd::Avx::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward0AutoTest(FUNC_FLF0(Simd::Avx512f::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF1
        {
            typedef void(*FuncPtr)(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncFLF1(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, const View & bias0, const View & scale1, const View & bias1, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)bias0.data, (float*)scale1.data, (float*)bias1.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_FLF1(function) FuncFLF1(function, #function)

    bool SynetFusedLayerForward1AutoTest(size_t count, size_t size, const FuncFLF1 & f1, const FuncFLF1 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias0(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale1(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias1(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(bias0, -10.0, 10.0);
        FillRandom32f(scale1, -10.0, 10.0);
        FillRandom32f(bias1, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias0, scale1, bias1, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias0, scale1, bias1, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetFusedLayerForward1AutoTest(const FuncFLF1 & f1, const FuncFLF1 & f2)
    {
        bool result = true;

        result = result && SynetFusedLayerForward1AutoTest(H, W, f1, f2);
        result = result && SynetFusedLayerForward1AutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetFusedLayerForward1AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward1AutoTest(FUNC_FLF1(Simd::Base::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward1AutoTest(FUNC_FLF1(Simd::Sse::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward1AutoTest(FUNC_FLF1(Simd::Avx::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward1AutoTest(FUNC_FLF1(Simd::Avx512f::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF2
        {
            typedef void(*FuncPtr)(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float *slope, float * dst);

            FuncPtr func;
            String desc;

            FuncFLF2(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, const View & scale, const View & bias, size_t count, size_t size, float slope, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)scale.data, (float*)bias.data, count, size, &slope, (float*)dst.data);
            }
        };
    }

#define FUNC_FLF2(function) FuncFLF2(function, #function)

    bool SynetFusedLayerForward2AutoTest(size_t count, size_t size, const FuncFLF2 & f1, const FuncFLF2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(scale, -10.0, 10.0);
        FillRandom32f(bias, -10.0, 10.0);
        const float slope = 0.1;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, count, size, slope, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, count, size, slope, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetFusedLayerForward2AutoTest(const FuncFLF2 & f1, const FuncFLF2 & f2)
    {
        bool result = true;

        result = result && SynetFusedLayerForward2AutoTest(H, W, f1, f2);
        result = result && SynetFusedLayerForward2AutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetFusedLayerForward2AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward2AutoTest(FUNC_FLF2(Simd::Base::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward2AutoTest(FUNC_FLF2(Simd::Sse::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward2AutoTest(FUNC_FLF2(Simd::Avx::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward2AutoTest(FUNC_FLF2(Simd::Avx512f::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

        return result;
    }

    namespace
    {
        struct FuncIPLF
        {
            typedef void(*FuncPtr)(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncIPLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncIPLF(const FuncIPLF & f, bool bias) : func(f.func), desc(f.desc + (bias ? "[1]" : "[0]")) {}

            void Call(const View & src, const View & weight, const View & bias, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)weight.data, (float*)bias.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_IPLF(function) FuncIPLF(function, #function)
#define ARGS_IPLF(bias, f1, f2) bias, FuncIPLF(f1, bias), FuncIPLF(f2, bias)

    bool SynetInnerProductLayerForwardAutoTest(size_t count, size_t size, bool hasBias, const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View weight(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias;
        View dst1(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(weight, -10.0, 10.0);
        if (hasBias)
        {
            bias.Recreate(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(bias, -10.0, 10.0);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, weight, bias, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, weight, bias, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest(const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(false, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(false, f1, f2));

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Base::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Sse::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx2::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx512f::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

        return result;
    }

    namespace
    {
        struct FuncLLCC
        {
            typedef void(*FuncPtr)(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst);

            FuncPtr func;
            String desc;

            FuncLLCC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, size_t half, size_t count, size_t size, const float * k, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, half, count, size, k, (float*)dst.data);
            }
        };
    }

#define FUNC_LLCC(function) FuncLLCC(function, #function)

    bool SynetLrnLayerCrossChannelsAutoTest(size_t half, size_t count, size_t size, const FuncLLCC & f1, const FuncLLCC & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        float k[3] = { 1.00, 0.10, -0.75 };

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, half, count, size, k, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, half, count, size, k, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetLrnLayerCrossChannelsAutoTest(const FuncLLCC & f1, const FuncLLCC & f2)
    {
        bool result = true;

        result = result && SynetLrnLayerCrossChannelsAutoTest(2, H, W, f1, f2);
        result = result && SynetLrnLayerCrossChannelsAutoTest(2, H - O, W + O, f1, f2);

        return result;
    }

    bool SynetLrnLayerCrossChannelsAutoTest()
    {
        bool result = true;

        result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Base::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Sse2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx512f::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

        return result;
    }

    namespace
    {
        struct FuncPLF
        {
            typedef void(*FuncPtr)(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans);

            FuncPtr func;
            String desc;

            FuncPLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdBool trans)
            {
                desc = desc + "[" + ToString<int>(trans) + "]";
            }

            void Call(const View & src, const View & slope, size_t count, size_t size, SimdBool trans, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)slope.data, count, size, (float*)dst.data, trans);
            }
        };
    }

#define FUNC_PLF(function) FuncPLF(function, #function)

    bool SynetPreluLayerForwardAutoTest(size_t count, size_t size, SimdBool trans, FuncPLF f1, FuncPLF f2)
    {
        bool result = true;

        f1.Update(trans);
        f2.Update(trans);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View slope(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(slope, -1.0, 1.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, count, size, trans, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, count, size, trans, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetPreluLayerForwardAutoTest(const FuncPLF & f1, const FuncPLF & f2)
    {
        bool result = true;

        result = result && SynetPreluLayerForwardAutoTest(H, W, SimdFalse, f1, f2);
        result = result && SynetPreluLayerForwardAutoTest(H - O, W + O, SimdFalse, f1, f2);
        result = result && SynetPreluLayerForwardAutoTest(H, W, SimdTrue, f1, f2);
        result = result && SynetPreluLayerForwardAutoTest(H - O, W + O, SimdTrue, f1, f2);

        return result;
    }

    bool SynetPreluLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetPreluLayerForwardAutoTest(FUNC_PLF(Simd::Base::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetPreluLayerForwardAutoTest(FUNC_PLF(Simd::Sse::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPreluLayerForwardAutoTest(FUNC_PLF(Simd::Avx::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetPreluLayerForwardAutoTest(FUNC_PLF(Simd::Avx512f::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif

        return result;
    }

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

    bool SynetRestrictRangeAutoTest(size_t size, const FuncRR & f1, const FuncRR & f2)
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

    bool SynetRestrictRangeAutoTest(const FuncRR & f1, const FuncRR & f2)
    {
        bool result = true;

        result = result && SynetRestrictRangeAutoTest(H*W, f1, f2);
        result = result && SynetRestrictRangeAutoTest(H*W + O, f1, f2);

        return result;
    }

    bool SynetRestrictRangeAutoTest()
    {
        bool result = true;

        result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Base::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Sse::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Avx::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Avx512f::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

        return result;
    }

    namespace
    {
        struct FuncSLF
        {
            typedef void(*FuncPtr)(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans);

            FuncPtr func;
            String desc;

            FuncSLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdBool trans, bool bias)
            {
                desc = desc + "[" + ToString<int>(trans) + "-" + ToString<int>(bias) + "]";
            }

            void Call(const View & src, const View & scale, const View & bias, size_t count, size_t size, SimdBool trans, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)scale.data, (float*)bias.data, count, size, (float*)dst.data, trans);
            }
        };
    }

#define FUNC_SLF(function) FuncSLF(function, #function)

    bool SynetScaleLayerForwardAutoTest(size_t count, size_t size, SimdBool trans, bool hasBias, FuncSLF f1, FuncSLF f2)
    {
        bool result = true;

        f1.Update(trans, hasBias);
        f2.Update(trans, hasBias);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias;
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(scale, -10.0, 10.0);
        if (hasBias)
        {
            bias.Recreate(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(bias, -10.0, 10.0);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, count, size, trans, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, count, size, trans, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest(const FuncSLF & f1, const FuncSLF & f2)
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(H, W, SimdFalse, false, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, SimdFalse, false, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H, W, SimdFalse, true, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, SimdFalse, true, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H, W, SimdTrue, false, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, SimdTrue, false, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H, W, SimdTrue, true, f1, f2);
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, SimdTrue, true, f1, f2);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Base::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Sse::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx2::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx512f::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool SynetAddBiasDataTest(bool create, size_t count, size_t size, SimdBool trans, const FuncAB & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(bias, -10.0, 10.0);
            FillRandom32f(dstSrc, -10.0, 10.0);

            TEST_SAVE(bias);
            TEST_SAVE(dstSrc);

            f.Call(bias, count, size, trans, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(bias);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(bias, count, size, trans, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetAddBiasDataTest(bool create)
    {
        return SynetAddBiasDataTest(create, DH, DW, SimdFalse, FUNC_AB(SimdSynetAddBias));
    }

    bool SynetEltwiseLayerForwardDataTest(bool create, size_t size, size_t count, SimdSynetEltwiseOperationType type, const FuncELF & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << size << "].");
        View src(size, count, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FloatPtrs psrc(count);
        for (size_t i = 0; i < count; ++i)
            psrc[i] = src.Row<float>(i);
        View weight(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -1.0, 1.0);
            FillRandom32f(weight, -1.0, 1.0);

            TEST_SAVE(src);
            TEST_SAVE(weight);

            f.Call(psrc, weight, count, size, type, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weight);

            TEST_LOAD(dst1);

            f.Call(psrc, weight, count, size, type, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetEltwiseLayerForwardDataTest(bool create)
    {
        bool result = true; 

        for (SimdSynetEltwiseOperationType type = SimdSynetEltwiseOperationProduct; type <= SimdSynetEltwiseOperationMin; type = (SimdSynetEltwiseOperationType)((size_t)type + 1))
            for (size_t count = 2; count <= 2; ++count)
                result = result && SynetEltwiseLayerForwardDataTest(create, DH*DW, count, type, FuncELF(FUNC_ELF(SimdSynetEltwiseLayerForward), type, count));
       
        return result;
    }

    bool SynetLrnLayerCrossChannelsDataTest(bool create, size_t half, size_t count, size_t size, const FuncLLCC & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        float k[3] = { 1.00, 0.10, -0.75 };

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, half, count, size, k, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, half, count, size, k, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetLrnLayerCrossChannelsDataTest(bool create)
    {
        return SynetLrnLayerCrossChannelsDataTest(create, 2, DH, DW, FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
    }

    bool SynetScaleLayerForwardDataTest(bool create, size_t count, size_t size, const FuncSLF & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);
            FillRandom32f(scale, -10.0, 10.0);
            FillRandom32f(bias, -10.0, 10.0);

            TEST_SAVE(src);
            TEST_SAVE(scale);
            TEST_SAVE(bias);

            f.Call(src, scale, bias, count, size, SimdFalse, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(scale);
            TEST_LOAD(bias);

            TEST_LOAD(dst1);

            f.Call(src, scale, bias, count, size, SimdFalse, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetScaleLayerForwardDataTest(bool create)
    {
        return SynetScaleLayerForwardDataTest(create, DH, DW, FUNC_SLF(SimdSynetScaleLayerForward));
    }
}
