/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Test/TestTensor.h"
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetInnerProduct16b.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncIP16b
        {
            typedef void* (*FuncPtr)(size_t M, size_t N, size_t K, SimdTensorDataType typeA, SimdTensorDataType typeB, SimdTensorDataType typeC, SimdBool transB, SimdBool constB, SimdBool bias);

            FuncPtr func;
            String desc;

            FuncIP16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Simd::InnerProductParam16b& p)
            {
                desc = desc + "[" + p.Info() + "]";
            }

            void Call(void* context, const uint8_t* A, const uint8_t* B, uint8_t* buf, uint8_t* C) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetInnerProduct16bForward(context, A, B, buf, C);
            }
        };
    }

#define FUNC_IP16B(function) \
    FuncIP16b(function, std::string(#function))

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

    bool SynetInnerProduct16bForwardAutoTest(float eps, Simd::InnerProductParam16b p, FuncIP16b f1, FuncIP16b f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Shape sA = Shp(p.M, p.K), sB = p.transB ? Shp(p.N, p.K) : Shp(p.K, p.N), sC = Shp(p.M, p.N);
        Tensor32f Af(sA), Bf(sB), C1f(sC), C2f(sC), C3f(sC), bias(Shp(p.N));
        Tensor16u Ab(sA), Bb(sB), C1b(sC), C2b(sC);

        FillRandom(Af.Data(), Af.Size(), -1.0, 1.0f);
        FillRandom(Bf.Data(), Bf.Size(), -1.0, 1.0f);
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        SimdFloat32ToBFloat16(Af.Data(), Af.Size(), Ab.Data());
        SimdFloat32ToBFloat16(Bf.Data(), Bf.Size(), Bb.Data());

        Fill(C1f, 1.0f);
        Fill(C1f, 2.0f);

        const uint8_t* A = p.typeA == SimdTensorData32f ? (uint8_t*)Af.Data() : (uint8_t*)Ab.Data();
        const uint8_t* B = p.typeB == SimdTensorData32f ? (uint8_t*)Bf.Data() : (uint8_t*)Bb.Data();
        uint8_t* C1 = p.typeC == SimdTensorData32f ? (uint8_t*)C1f.Data() : (uint8_t*)C1b.Data();
        uint8_t* C2 = p.typeC == SimdTensorData32f ? (uint8_t*)C2f.Data() : (uint8_t*)C2b.Data();

        void* context1 = f1.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.transB, p.constB, p.bias);
        void* context2 = f2.func(p.M, p.N, p.K, p.typeA, p.typeB, p.typeC, p.transB, p.constB, p.bias);

        if (context1 == NULL)
            return true;

        ::SimdSynetInnerProduct16bSetParams(context1, Bf.Data(), bias.Data());
        ::SimdSynetInnerProduct16bSetParams(context2, Bf.Data(), bias.Data());

        Tensor8u buf;
        buf.Extend( Shp(SimdSynetInnerProduct16bExternalBufferSize(context1)) );
        buf.Extend( Shp(SimdSynetInnerProduct16bExternalBufferSize(context2)) );

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, A, B, buf.Data(), C1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, A, B, buf.Data(), C2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (p.typeC == SimdTensorData16b)
        {
            eps = eps * 7.1f;
            SimdBFloat16ToFloat32(C1b.Data(), C1b.Size(), C1f.Data());
            SimdBFloat16ToFloat32(C2b.Data(), C2b.Size(), C2f.Data());
        }
        result = result && Compare(C1f, C2f, eps, true, 64, DifferenceBoth);

        if(1)
        {
            void* context3 = SimdSynetInnerProduct32fInit(p.M, p.K, p.N, p.transB, SimdConvolutionActivationIdentity);
            ::SimdSynetInnerProduct32fSetParams(context3, Bf.Data(), NULL, p.bias ? bias.Data() : NULL, NULL);
            ::SimdSynetInnerProduct32fForward(context3, Af.Data(), C3f.Data());
            ::SimdRelease(context3);

            float e = EPS * GetRange(C3f.Data(), C3f.Size()) * 3.0f;
            result = result && Compare(C1f, C3f, e, true, 64, DifferenceAbsolute, " Compare to SynetInnerProduct32f.");
        }

        return result;
    }

    bool SynetInnerProduct16bForwardAutoTest(float eps, const FuncIP16b& f1, const FuncIP16b& f2)
    {
        bool result = true;

        SimdBool t = SimdTrue, f = SimdFalse;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
        using Param = Simd::InnerProductParam16b;

#if defined(NDEBUG)
#if 0
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, f32, f32, b16, f, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, b16, b16, f32, f, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, f32, f32, b16, t, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, b16, b16, f32, t, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, b16, b16, b16, f, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(128, 128, 128, b16, b16, b16, t, f, t), f1, f2);
#endif
#if 0
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, f32, f32, f32, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, f32, f32, b16, f, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, b16, b16, f32, f, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, f32, f32, f32, t, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, f32, f32, b16, t, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(127, 129, 131, b16, b16, f32, t, t, f), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 512, 8192, b16, b16, b16, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 512, 8192, b16, b16, b16, f, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 512, 8192, b16, b16, b16, t, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 512, 512, b16, b16, b16, f, t, t), f1, f2);

#endif
#if 0
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(9, 128, 9, f32, f32, f32, f, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(9, 128, 9, f32, f32, f32, f, t, t), f1, f2);
#endif
#if 1
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(10, 64, 256, b16, f32, f32, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(10, 60, 256, b16, f32, f32, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(10, 64, 240, f32, f32, f32, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(10, 64, 240, b16, f32, f32, f, t, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(10, 60, 240, b16, f32, f32, f, t, t), f1, f2);
#endif
#else
        //result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 512, 4096, b16, b16, b16, t, f, t), f1, f2);
        result = result && SynetInnerProduct16bForwardAutoTest(eps, Param(1, 1000, 1280, f32, f32, f32, f, t, t), f1, f2);
#endif

        return result;
    }

    bool SynetInnerProduct16bForwardAutoTest()
    {
        const float EPS = 0.001f;
        bool result = true;

        if (TestBase())
            result = result && SynetInnerProduct16bForwardAutoTest(EPS, FUNC_IP16B(Simd::Base::SynetInnerProduct16bInit), FUNC_IP16B(SimdSynetInnerProduct16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetInnerProduct16bForwardAutoTest(EPS, FUNC_IP16B(Simd::Sse41::SynetInnerProduct16bInit), FUNC_IP16B(SimdSynetInnerProduct16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetInnerProduct16bForwardAutoTest(EPS, FUNC_IP16B(Simd::Avx2::SynetInnerProduct16bInit), FUNC_IP16B(SimdSynetInnerProduct16bInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetInnerProduct16bForwardAutoTest(EPS, FUNC_IP16B(Simd::Avx512bw::SynetInnerProduct16bInit), FUNC_IP16B(SimdSynetInnerProduct16bInit));
#endif

#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))   
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && SynetInnerProduct16bForwardAutoTest(EPS, FUNC_IP16B(Simd::AmxBf16::SynetInnerProduct16bInit), FUNC_IP16B(SimdSynetInnerProduct16bInit));
#endif

        return result;
    }
#endif
}
