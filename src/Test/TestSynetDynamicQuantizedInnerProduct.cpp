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

#include "Simd/SimdSynetQuantizedInnerProduct.h"

#include "Simd/SimdMath.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncDQIP
        {
            typedef void*(*FuncPtr)(size_t M, size_t N, size_t K, SimdBool bias, SimdConvolutionActivationType activation);

            FuncPtr func;
            String desc;

            FuncDQIP(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t M, size_t N, size_t K, SimdBool b, SimdConvolutionActivationType a)
            {
                std::stringstream ss;
                ss << M << "x" << K << "-" << N << " ";
                ss << (b ? "b" : "o");
                desc = desc + "[" + ss.str() + "]";
            }

            void Call(void * context, const float *A, uint8_t * buf, float * C) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetDynamicQuantizedInnerProductForward(context, A, buf, C);
            }
        };
    }

#define FUNC_QIP(function) \
    FuncQIP(function, std::string(#function))

    struct DqipParams
    {
        Tensor32f a, b, scale, bias, params, c, c1, c2;
        Tensor8i b8i;

        bool Init(size_t M, size_t N, size_t K, SimdBool bs, SimdConvolutionActivationType act, SimdBool overflow)
        {
            Shape sA = Shp(M, K), sB = Shp(K, N), sC = Shp(M, N);

            a.Reshape(sA);
            FillRandom(a, -0.9, 1.1f);

            b.Reshape(sB);
            FillRandom(b, -1.1, 1.0f);

            if (!QuantizeB(b, overflow, b8i, scale))
                return false;

            bias.Reshape(Shp(N));
            FillRandom(bias, -1.1, 1.2f);

            params.Reshape(Shp(N));
            FillRandom(params, 0, 1.0f);
            if (act == ::SimdConvolutionActivationHswish)
            {
                params.Data()[0] = 3.0f;
                params.Data()[1] = 1.0f / 6.0f;
            }
            else if (act == ::SimdConvolutionActivationMish)
                params.Data()[0] = 20.0f;
            else
            {
                params.Data()[0] = 0.1f;
                params.Data()[1] = 1.1f;
            }

            c.Reshape(sC);

            void* context = ::SimdSynetInnerProduct32fInit(M, N, K, SimdFalse, SimdTrue, bs, act);
            if (context == NULL)
                return false;

            ::SimdSynetInnerProduct32fSetParams(context, b.Data(), NULL, bias.Data(), params.Data());

            ::SimdSynetInnerProduct32fForward(context, a.Data(), NULL, NULL, c.Data());

            ::SimdRelease(context);

            c1.Reshape(sC);
            c2.Reshape(sC);

            return true;
        }

    private:
        static bool QuantizeB(const Tensor32f& src, SimdBool overflow, Tensor8i& dst, Tensor32f& scale)
        {
            size_t size = src.Size(), N = src.Axis(1), K = size / N;
            dst.Reshape(src.Shape());
            scale.Reshape(Shp(N));
            const float* psrc = src.Data();
            int8_t* pdst = dst.Data();
            int lo = overflow ? -64 : -128, hi = overflow ? 63 : 127;
            for (size_t j = 0; j < N; ++j)
            {
                float max = 0;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    max = std::max(max, std::abs(psrc[offset]));
                }
                float range = std::max(0.000001f, max);
                float _scale = range / (overflow ? 63.0f : 127.0f), invScale = (overflow ? 63.0f : 127.0f) / range;
                scale.Data()[j] = _scale;
                for (size_t k = 0; k < K; ++k)
                {
                    size_t offset = k * N + j;
                    pdst[offset] = Simd::RestrictRange((int)std::nearbyint(psrc[offset] * invScale), lo, hi);
                }
            }
            return true;
        }
    };

    bool SynetDynamicQuantizedInnerProductForwardAutoTest(float eps, size_t M, size_t N, size_t K, SimdBool b, SimdConvolutionActivationType a, SimdBool o, FuncDQIP f1, FuncDQIP f2)
    {
        bool result = true;

        f1.Update(M, N, K, b, a);
        f2.Update(M, N, K, b, a);

        if (M == 1)
            o = SimdTrue;

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        Srand(0);

        DqipParams dp;
        if (!dp.Init(M, N, K, b, a, o))
            return false;

        void * context1 = f1.func(M, N, K, b, a);
        void * context2 = f2.func(M, N, K, b, a);
        if (context1 == NULL)
            return true;

        Tensor8u buf8u;
        buf8u.Extend({ ::SimdSynetDynamicQuantizedInnerProductExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetDynamicQuantizedInnerProductExternalBufferSize(context2) });

        ::SimdSynetDynamicQuantizedInnerProductSetParams(context1, dp.b8i.Data(), dp.scale.Data(), dp.bias.Data(), dp.params.Data());
        ::SimdSynetDynamicQuantizedInnerProductSetParams(context2, dp.b8i.Data(), dp.scale.Data(), dp.bias.Data(), dp.params.Data());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, dp.a.Data(), buf8u.Data(), dp.c1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, dp.a.Data(), buf8u.Data(), dp.c2.Data()));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dp.c1, dp.c2, eps, true, 64, DifferenceBoth);

        int controlDiffMax = o ? 2 : 3;
        result = result && Compare(dp.c1, dp.c, controlDiffMax, true, 64, "control");

        return result;
    }

    bool SynetDynamicQuantizedInnerProductForwardAutoTest(SimdBool o, const FuncDQIP& f1, const FuncDQIP& f2)
    {
        bool result = true;

        const float e = EPS;
        const SimdBool f = SimdFalse, t = SimdTrue;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

#ifdef NDEBUG
#if 1
        result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(e, 768, 384, 96, t, aGe, o, f1, f2);
#endif
#else
        result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(e, 768, 384, 96, t, aGe, o, f1, f2);
#endif

        return result;
    }

    bool SynetDynamicQuantizedInnerProductForwardAutoTest(const Options & options)
    {
        bool result = true;

        const SimdBool f = SimdFalse, t = SimdTrue;

        //if (TestBase(options))
        //    result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(t, FUNC_DQIP(Simd::Base::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(t, FUNC_DQIP(Simd::Sse41::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(t, FUNC_DQIP(Simd::Avx2::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(t, FUNC_DQIP(Simd::Avx512bw::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif
//
//#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
//        if (Simd::Avx512vnni::Enable && TestAvx512vnni(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(f, FUNC_DQIP(Simd::Avx512vnni::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif
//
//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(f, FUNC_DQIP(Simd::AmxBf16::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable && TestNeon(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(t, FUNC_DQIP(Simd::Neon::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif
//
//#ifdef SIMD_SVE2_ENABLE
//        if (Simd::Sve2::Enable && TestSve2(options))
//            result = result && SynetDynamicQuantizedInnerProductForwardAutoTest(f, FUNC_DQIP(Simd::Sve2::SynetDynamicQuantizedInnerProductInit), FUNC_DQIP(SimdSynetDynamicQuantizedInnerProductInit));
//#endif

        return result;
    }
#endif
}
