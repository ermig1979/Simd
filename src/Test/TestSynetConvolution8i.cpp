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
#include "Test/TestSynetConvolutionParam.h"

#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<false> Param;

        struct FuncC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p, SimdSynetCompatibilityType c)
            {
                const char* afs[] = { "-id", "-re", "-lr", "-rr", "-pr", "-el", "-hs", "-mi", "-hi", "-sw" };
                desc = desc + p.Decription(String(afs[p.conv.activation]) + (Simd::Base::Overflow(c) ? "-o" : Simd::Base::Narrowed(c) ? "-n" : "-p"));
            }

            void Call(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetConvolution8iForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    void FillDstStat(Param p, int neg, SimdSynetCompatibilityType comp, const Tensor32f& weight, const Tensor32f & bias, const Tensor32f& params,
        const Tensor32f & src, Tensor32f& buf, Tensor32f & dst, float* dstMin, float* dstMax, float* dstScale, float* dstShift)
    {
        p.conv.srcT = SimdTensorData32f;
        p.conv.dstT = SimdTensorData32f;
        void * context = SimdSynetConvolution32fInit(p.batch, &p.conv, NULL);
        buf.Extend({ SimdSynetConvolution32fExternalBufferSize(context) });
        SimdSynetConvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());
        SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
        SimdRelease(context);
        SetDstStat(p.conv.dstC, neg, comp, dst, dstMin, dstMax, dstScale, dstShift);
    }

    bool SynetConvolution8iForwardAutoTest(float eps, Param p, int neg, SimdSynetCompatibilityType comp, FuncC f1, FuncC f2)
    {
        bool result = true;

        f1.Update(p, comp);
        f2.Update(p, comp);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters & c = p.conv;

        Tensor32f weight(p.WeightShape());
        FillRandom(weight.Data(), weight.Size(), -1.0, 1.0f);

        Tensor32f bias({ c.dstC });
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        Tensor32f params({ c.dstC });
        FillRandom(params.Data(), params.Size(), -3.0f, 3.0f);
        if (p.conv.activation == ::SimdConvolutionActivationHswish)
        {
            params.Data()[0] = 3.0f;
            params.Data()[1] = 1.0f / 6.0f;
        }
        else if (p.conv.activation == ::SimdConvolutionActivationMish)
            params.Data()[0] = 20.0f;
        else if (p.conv.activation == ::SimdConvolutionActivationHardSigmoid)
        {
            params.Data()[0] = 1.0f / 6.0f;
            params.Data()[1] = 0.5f;
        }
        else
        {
            params.Data()[0] = 0.1f;
            params.Data()[1] = 1.1f;
        }

        Tensor32f srcMin({ c.srcC }), srcMax({ c.srcC }), dstMin({ c.dstC }), dstMax({ c.dstC });
        Tensor32f src32f(p.SrcShape(), p.conv.srcF), dst32f1(p.DstShape(), p.conv.dstF), dst32f2(p.DstShape(), p.conv.dstF), buf32f;
        Tensor8u src8u(p.SrcShape(), p.conv.srcF), dst8u1(p.DstShape(), p.conv.dstF), dst8u2(p.DstShape(), p.conv.dstF), buf8u;
        //dst8u2.Reshape({ 1000000 }); dst8u2.Extend(p.DstShape());

        FillRandom(src32f, srcMin.Data(), srcMax.Data(), p.conv.srcC, neg);
        SetSrc32fTo8u(src32f, srcMin.Data(), srcMax.Data(), c.srcC, neg, comp, NULL, NULL, src8u);
        FillDstStat(p, neg, comp, weight, bias, params, src32f, buf32f, dst32f1, dstMin.Data(), dstMax.Data(), NULL, NULL);

        const float* stats[4] = { srcMin.Data(), srcMax.Data(), dstMin.Data(), dstMax.Data() };
        const uint8_t * src = p.conv.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : src8u.Data();
        uint8_t* dst1 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : dst8u1.Data();
        uint8_t* dst2 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : dst8u2.Data();

        Fill(dst32f1, 0.1f);
        Fill(dst32f2, 1.1f);

        Fill(dst8u1, uint8_t(1));
        Fill(dst8u2, uint8_t(2));

        void * context1 = f1.func(p.batch, &p.conv, comp);
        void * context2 = f2.func(p.batch, &p.conv, comp);

        buf8u.Extend({ ::SimdSynetConvolution8iExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetConvolution8iExternalBufferSize(context2) });

        ::SimdSynetConvolution8iSetParams(context1, weight.Data(), bias.Data(), params.Data(), stats);
        ::SimdSynetConvolution8iSetParams(context2, weight.Data(), bias.Data(), params.Data(), stats);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (Simd::Base::FmaAvoid(comp) ? 0 : 1);
#else
        int differenceMax = 1;
#endif

        if(p.conv.dstT == SimdTensorData32f)
            result = result && Compare(dst32f1, dst32f2, eps*eps, true, 64, DifferenceBoth);
        else
            result = result && Compare(dst8u1, dst8u2, differenceMax, true, 64);

        return result;
    }

    bool SynetConvolution8iForwardAutoTest(const FuncC& f1, const FuncC& f2, SimdSynetCompatibilityType c)
    {
        bool result = true;

        const Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);
        const float e = EPS;
        const SimdBool t0 = SimdFalse, t1 = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu, 
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu, 
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish;
        //SimdSynetCompatibilityType c = (SimdSynetCompatibilityType)((SimdCpuInfo(SimdCpuInfoAvx512vnni) ? SimdSynetCompatibilityFmaUse : SimdSynetCompatibility8iOverflow)  | SimdSynetCompatibilityFmaAvoid);

#ifdef NDEBUG
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _7, _1, _2, _3, _3, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _5, _2, _3, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 32, 150, 150, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, f32), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 32, 150, 150, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _3, _1, _1, _1, _1, 1, aId, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 2000, 30, 30, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 5000, 30, 30, 400, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 145, 60, 75, 97, _3, _1, _1, _1, _1, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 48, 48, 96, _3, _1, _1, _1, _1, 1, aRe, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 384, 19, 19, 36, _3, _1, _1, _1, _1, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        //result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 40, 128, 128, 128, _5, _1, _1, _0, _0, 1, aRe, t1, u8, u8), 0, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(10, 3, 300, 300, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(2, 64, 80, 80, 81, _1, _1, _1, _0, _0, 1, aId, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(3, 3, 300, 300, 32, _5, _2, _3, _0, _0, 1, aRe, t1, u8, f32), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 192, 15, 15, 256, _3, _1, _1, _1, _1, 1, aLr, t1, u8, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 30, 30, 128, _5, _1, _2, _0, _0, 1, aRr, t1, f32, f32), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 32, 160, 160, 32, _3, _1, _2, _1, _1, 1, aPr, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 80, 100, 100, 80, _1, _1, _1, _0, _0, 1, aEl, t1, u8, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 384, 8, 12, 256, _3, _1, _1, _1, _1, 1, aHs, t1, u8, u8), 1, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 40, 60, 128, _3, _1, _1, _1, _1, 128, aRe, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 99, 90, 130, 99, _3, _1, _2, _1, _1, 99, aPr, t1, f32, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 64, 70, 102, 64, _5, _1, _2, _2, _2, 64, aId, t1, u8, f32), 0, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 20, 12, 20, _1, _1, _1, _0, _0, 1, aMi, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 20, 12, 128, _1, _1, _1, _0, _0, 1, aMi, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 20, 12, 128, _3, _1, _1, _1, _1, 128, aMi, t1, u8, u8), 1, c, f1, f2);
#endif
#if 1
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 224, 12, 12, 224, _3, _1, _1, _1, _1, 1, aPr, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 224, 12, 14, 224, _3, _1, _1, Size(0, 1), Size(0, 1), 1, aPr, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 112, 24, 24, 112, _3, _1, _1, _1, _1, 1, aPr, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 112, 24, 26, 112, _3, _1, _1, Size(0, 1), Size(0, 1), 1, aPr, t1, u8, u8), 1, c, f1, f2);
#endif
#if 1
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 80, 100, 100, 80, _1, _1, _1, _0, _0, 1, aSw, t1, u8, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 384, 8, 12, 256, _3, _1, _1, _1, _1, 1, aSw, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 64, 70, 102, 64, _5, _1, _2, _2, _2, 64, aSw, t1, u8, f32), 0, c, f1, f2);
#endif
#else
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 2000, 30, 30, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
#endif

        return result;
    }

    bool SynetConvolution8iForwardAutoTest(const FuncC& f1, const FuncC& f2)
    {
        bool result = true;

        SimdSynetCompatibilityType fma = SimdSynetCompatibilityFmaAvoid;
        SimdSynetCompatibilityType p = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | fma);
        SimdSynetCompatibilityType o = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iOverflow | fma);
        SimdSynetCompatibilityType n = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | fma);

        //result = result && SynetConvolution8iForwardAutoTest(f1, f2, p);
        //result = result && SynetConvolution8iForwardAutoTest(f1, f2, o);
        result = result && SynetConvolution8iForwardAutoTest(f1, f2, n);

        return result;
    }

    bool SynetConvolution8iForwardAutoTest()
    {
        bool result = true;

        result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Base::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Sse41::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx2::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx512bw::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_AVX512VNNI_ENABLE
        if (Simd::Avx512vnni::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx512vnni::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Neon::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif 

        return result;
    }
#endif
}
