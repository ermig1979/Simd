/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#include "Simd/SimdSynetDynamicQuantizedInnerProduct16b.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    typedef Simd::DynamicQuantizedInnerProduct16bParam Param;

    namespace
    {
        struct FuncDQIP16b
        {
            typedef void* (*FuncPtr)(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC,
                SimdBool constB, SimdBool bias, SimdConvolutionActivationType activation, SimdBool avoidOverflow);

            FuncPtr func;
            String desc;

            FuncDQIP16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Param& p)
            {
                desc = desc + "[" + p.Info() + "]";
            }

            void Call(void* context, const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetDynamicQuantizedInnerProduct16bForward(context, A, B, buf, C);
            }
        };
    }

#define FUNC_DQIP16B(function) \
    FuncDQIP16b(function, std::string(#function))

    static bool QuantizeB(const Tensor32f& Bf, SimdBool overflow, Tensor8i& Bi, Tensor32f& scale)
    {
        size_t N = Bf.Axis(1), K = Bf.Axis(0);
        const float* src = Bf.Data();
        int8_t* dst = Bi.Data();
        int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
        float q = overflow ? 63.0f : 127.0f;
        for (size_t j = 0; j < N; ++j)
        {
            float max = 0;
            for (size_t k = 0; k < K; ++k)
            {
                size_t offset = k * N + j;
                max = std::max(max, std::abs(src[offset]));
            }
            float range = std::max(0.000001f, max);
            float invScale = q / range;
            scale.Data()[j] = range / q;
            for (size_t k = 0; k < K; ++k)
            {
                size_t offset = k * N + j;
                dst[offset] = (int8_t)Simd::RestrictRange((int)std::nearbyint(src[offset] * invScale), lo, hi);
            }
        }
        return true;
    }

    static float GetRange(const float* src, size_t size)
    {
        float min = FLT_MAX;
        float max = -FLT_MAX;
        for (size_t i = 0; i < size; ++i)
        {
            float val = src[i];
            min = Simd::Min(val, min);
            max = Simd::Max(val, max);
        }
        return max - min;
    }

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(float eps, const Param& p, FuncDQIP16b f1, FuncDQIP16b f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        Srand(0);

        Shape sA = Shp(p.M, p.K), sB = Shp(p.K, p.N), sC = Shp(p.M, p.N), 
            sN = Shp(p.N), sP = Shp(std::max<size_t>(p.N, 2));
        Tensor32f Af(sA), Bf(sB), C1f(sC), C2f(sC), C3f(sC), scale(sN), bias(sN), params(sP);
        Tensor16u Ab(sA), Bb(sB), C1b(sC), C2b(sC);
        Tensor8i Bi(sB);

        FillRandom(Af.Data(), Af.Size(), -1.0f, 1.0f);
        FillRandom(Bf.Data(), Bf.Size(), -1.0f, 1.0f);
        FillRandom(bias.Data(), bias.Size(), -1.0f, 1.0f);
        FillRandom(params.Data(), params.Size(), 0.0f, 1.0f);

        if (p.activation == ::SimdConvolutionActivationHswish)
        {
            params.Data()[0] = 3.0f;
            params.Data()[1] = 1.0f / 6.0f;
        }
        else if (p.activation == ::SimdConvolutionActivationMish)
            params.Data()[0] = 20.0f;
        else
        {
            params.Data()[0] = 0.1f;
            params.Data()[1] = 1.1f;
        }

        Fill(C1f, 1.0f);
        Fill(C2f, 2.0f);
        Fill(C3f, 3.0f);

        if(p.typeA == SimdTensorData16b)
            SimdFloat32ToBFloat16(Af.Data(), Af.Size(), Ab.Data());
        if (p.typeB == SimdTensorData16b)
            SimdFloat32ToBFloat16(Bf.Data(), Bf.Size(), Bb.Data());
        else if (p.typeB == SimdTensorData8i)
            QuantizeB(Bf, p.overflow, Bi, scale);

        void* context1 = f1.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.constB, p.bias, p.activation, p.overflow);
        void* context2 = f2.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.constB, p.bias, p.activation, p.overflow);
        if (context1 == NULL || context2 == NULL)
            return false;

        const uint8_t* pB = NULL;
        const float* pScale = NULL;
        if (p.constB)
        {
            if (p.typeB == SimdTensorData8i)
            {
                pB = (const uint8_t*)Bi.Data();
                pScale = scale.Data();
            }
            else
                pB = (const uint8_t*)Bf.Data();
        }
        const float* pBias = p.bias ? bias.Data() : NULL;
        const float* pParams = (p.activation != SimdConvolutionActivationIdentity) ? params.Data() : NULL;

        ::SimdSynetDynamicQuantizedInnerProduct16bSetParams(context1, pB, pScale, pBias, pParams);
        ::SimdSynetDynamicQuantizedInnerProduct16bSetParams(context2, pB, pScale, pBias, pParams);

        Tensor8u buf;
        buf.Extend(Shp(::SimdSynetDynamicQuantizedInnerProduct16bExternalBufferSize(context1)));
        buf.Extend(Shp(::SimdSynetDynamicQuantizedInnerProduct16bExternalBufferSize(context2)));

        const uint8_t* pA = p.typeA == SimdTensorData32f ? (const uint8_t*)Af.Data() : (const uint8_t*)Ab.Data();
        if (p.constB)
            pB = NULL;
        else
            pB = p.typeB == SimdTensorData32f ? (const uint8_t*)Bf.Data() : (const uint8_t*)Bb.Data();
        uint8_t* pC1 = p.typeC == SimdTensorData32f ? (uint8_t*)C1f.Data() : (uint8_t*)C1b.Data();
        uint8_t* pC2 = p.typeC == SimdTensorData32f ? (uint8_t*)C2f.Data() : (uint8_t*)C2b.Data();

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, pA, pB, buf.Data(), pC1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, pA, pB, buf.Data(), pC2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (p.typeC == SimdTensorData16b)
        {
            eps = eps * 7.6f;
            SimdBFloat16ToFloat32(C1b.Data(), C1b.Size(), C1f.Data());
            SimdBFloat16ToFloat32(C2b.Data(), C2b.Size(), C2f.Data());
        }
        result = result && Compare(C1f, C2f, eps, true, 64, DifferenceBoth);

        if (1)
        {
            void* context3 = SimdSynetInnerProduct32fInit(p.M, p.N, p.K, SimdFalse, SimdTrue, p.bias, p.activation);
            ::SimdSynetInnerProduct32fSetParams(context3, Bf.Data(), NULL, pBias, params.Data());
            ::SimdSynetInnerProduct32fForward(context3, Af.Data(), NULL, NULL, C3f.Data());
            ::SimdRelease(context3);

            float e = EPS * GetRange(C3f.Data(), C3f.Size()) * 5.0f;
            result = result && Compare(C1f, C3f, e, true, 64, DifferenceBoth, " Compare to SynetInnerProduct32f.");
        }

        return result;
    }

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(float eps, SimdBool o, const FuncDQIP16b& f1, const FuncDQIP16b& f2)
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b, i8 = SimdTensorData8i;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

#ifdef NDEBUG
        //result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(3333, 4444, 555, f32, i8, f32, t, t, aId, o), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, f32, i8, f32, t, t, aId, o), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, f32, i8, b16, t, t, aId, o), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, b16, i8, f32, t, t, aId, o), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, b16, i8, b16, t, t, aId, o), f1, f2);
        //result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, f32, f32, f32, t, t, aId, o), f1, f2);
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, f32, f32, f32, f, t, aId, o), f1, f2);
        //result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(333, 444, 555, b16, b16, b16, f, t, aId, o), f1, f2);
#else
        result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(eps, Param(8, 8, 8, f32, f32, f32, f, t, aId, o), f1, f2);
#endif

        return result;
    }

    bool SynetDynamicQuantizedInnerProduct16bForwardAutoTest(const Options& options)
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;

        if (TestBase(options))
            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, t, FUNC_DQIP16B(Simd::Base::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, t, FUNC_DQIP16B(Simd::Sse41::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
#endif 

//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, t, FUNC_DQIP16B(Simd::Avx2::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, t, FUNC_DQIP16B(Simd::Avx512bw::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif
//
//#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
//        if (Simd::Avx512vnni::Enable && TestAvx512vnni(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, f, FUNC_DQIP16B(Simd::Avx512vnni::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif
//
//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, f, FUNC_DQIP16B(Simd::AmxBf16::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable && TestNeon(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, t, FUNC_DQIP16B(Simd::Neon::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif
//
//#ifdef SIMD_SVE2_ENABLE
//        if (Simd::Sve2::Enable && TestSve2(options))
//            result = result && SynetDynamicQuantizedInnerProduct16bForwardAutoTest(EPS, f, FUNC_DQIP16B(Simd::Sve2::SynetDynamicQuantizedInnerProduct16bInit), FUNC_DQIP16B(SimdSynetDynamicQuantizedInnerProduct16bInit));
//#endif

        return result;
    }
#endif
}
