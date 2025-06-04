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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestSynetConvolutionParam.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<false> Param;

        struct FuncQC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv);

            FuncPtr func;
            String desc;

            FuncQC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                const char* afs[] = { "-id", "-re", "-lr", "-rr", "-pr", "-el", "-hs", "-mi", "-hi", "-sw", "-ge" };
                std::stringstream extra;
                extra << (p.conv.srcT == SimdTensorData32f ? "-f" : "-u");
                extra << (p.conv.dstT == SimdTensorData32f ? "f" : "u");
                extra << afs[p.conv.activation];
                desc = desc + p.Decription(extra.str());
            }

            void Call(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetQuantizedConvolutionForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_QC(function) \
    FuncQC(function, std::string(#function))

    bool SynetQuantizedConvolutionInit32f(Param p, Tensor32f& src, Tensor32f & weight, Tensor32f& bias, Tensor32f& params, Tensor32f& dst)
    {
        p.conv.srcT = SimdTensorData32f;
        p.conv.dstT = SimdTensorData32f;

        src.Reshape(p.SrcShape());
        FillRandom(src, -1.0, 1.0f);

        weight.Reshape(p.WeightShape());
        FillRandom(weight, -1.0, 1.0f);

        bias.Reshape(Shp(p.conv.dstC));
        FillRandom(bias, -1.0, 1.0f);

        params.Reshape(Shp(p.conv.dstC));
        FillRandom(params, 0.0f, 2.0f);
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

        dst.Reshape(p.DstShape());

        void* context = ::SimdSynetConvolution32fInit(p.batch, &p.conv);
        if (context == NULL)
            return false;

        Tensor32f buf;
        buf.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context) });

        ::SimdSynetConvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());

        ::SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());

        ::SimdRelease(context);

        return true;
    }

    bool SynetQuantizedConvolutionForwardAutoTest(float eps, Param p, FuncQC f1, FuncQC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters& c = p.conv;

        Tensor32f src32f, weight32f, bias32f, params32f, dst32f;
        SynetQuantizedConvolutionInit32f(p, src32f, weight32f, bias32f, params32f, dst32f);
     
        //
        //Tensor8i weight8i(p.WeightShape());
        //FillRandom(weight8i, -128, 127);

        //Tensor32i bias32i(Shp(c.dstC));
        //FillRandom(bias32i, -32*1024, 32*1024 - 1);



        //Tensor32f srcMin({ c.srcC }), srcMax({ c.srcC }), dstMin({ c.dstC }), dstMax({ c.dstC });
        //Tensor32f src32f(p.SrcShape(), p.conv.srcF), dst32f1(p.DstShape(), p.conv.dstF), dst32f2(p.DstShape(), p.conv.dstF), buf32f;
        //Tensor8u src8u(p.SrcShape(), p.conv.srcF), dst8u1(p.DstShape(), p.conv.dstF), dst8u2(p.DstShape(), p.conv.dstF), buf8u;
        ////dst8u2.Reshape({ 1000000 }); dst8u2.Extend(p.DstShape());

        //FillRandom(src32f, srcMin.Data(), srcMax.Data(), p.conv.srcC, neg);
        //SetSrc32fTo8u(src32f, srcMin.Data(), srcMax.Data(), c.srcC, neg, comp, NULL, NULL, src8u);
        //FillDstStat(p, neg, comp, weight, bias, params, src32f, buf32f, dst32f1, dstMin.Data(), dstMax.Data(), NULL, NULL);

        //const float* stats[4] = { srcMin.Data(), srcMax.Data(), dstMin.Data(), dstMax.Data() };
        //const uint8_t * src = p.conv.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : src8u.Data();
        //uint8_t* dst1 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : dst8u1.Data();
        //uint8_t* dst2 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : dst8u2.Data();

        //Fill(dst32f1, 0.1f);
        //Fill(dst32f2, 1.1f);

        //Fill(dst8u1, uint8_t(1));
        //Fill(dst8u2, uint8_t(2));

        //void * context1 = f1.func(p.batch, &p.conv);
        //void * context2 = f2.func(p.batch, &p.conv);

        //buf8u.Extend({ ::SimdSynetQuantizedConvolutionExternalBufferSize(context1) });
        //buf8u.Extend({ ::SimdSynetQuantizedConvolutionExternalBufferSize(context2) });

        //::SimdSynetConvolution8iSetParams(context1, weight.Data(), bias.Data(), params.Data(), stats);
        //::SimdSynetConvolution8iSetParams(context2, weight.Data(), bias.Data(), params.Data(), stats);

        //TEST_ALIGN(SIMD_ALIGN);

        //TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        //TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        //::SimdRelease(context1);
        //::SimdRelease(context2);

        //int differenceMax = 0;

        //if(p.conv.dstT == SimdTensorData32f)
        //    result = result && Compare(dst32f1, dst32f2, eps, true, 64, DifferenceBoth);
        //else
        //    result = result && Compare(dst8u1, dst8u2, differenceMax, true, 64);

        return result;
    }

    bool SynetQuantizedConvolutionForwardAutoTest(const FuncQC& f1, const FuncQC& f2)
    {
        bool result = true;

        const Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);
        const float e = EPS;
        const SimdBool t0 = SimdFalse, t1 = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu, 
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu, 
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

#ifdef NDEBUG
#if 1
        result = result && SynetQuantizedConvolutionForwardAutoTest(e, Param(1, 512, 32, 32, 256, _1, _1, _1, _0, _0, 1, aRe, t1, u8, f32), f1, f2);
#endif
#else
        result = result && SynetQuantizedConvolutionForwardAutoTest(e, Param(1, 128, 30, 40, 76, _3, _1, _1, _1, _1, 1, aPr, t1, u8, u8), f1, f2);
#endif

        return result;
    }

    bool SynetQuantizedConvolutionForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Base::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Sse41::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx2::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx512bw::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#if defined(SIMD_AVX512VNNI_ENABLE) && !defined(SIMD_AMX_EMULATE)
//        if (Simd::Avx512vnni::Enable && TestAvx512vnni())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Avx512vnni::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::AmxBf16::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable && TestNeon())
//            result = result && SynetQuantizedConvolutionForwardAutoTest(FUNC_QC(Simd::Neon::SynetQuantizedConvolutionInit), FUNC_QC(SimdSynetQuantizedConvolutionInit));
//#endif 

        return result;
    }
#endif
}
