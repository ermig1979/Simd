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
#include "Test/TestSynetConvolutionParam.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynetDeconvolution16b.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<true> Param;

        struct FuncD
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncD(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param& p, SimdSynetCompatibilityType c)
            {
                const char* afs[] = { "-id", "-re", "-lr", "-rr", "-pr", "-el", "-hs", "-mi", "-hi", "-sw", "-ge" };
                std::stringstream extra;
                extra << (p.conv.srcT == SimdTensorData32f ? "-f" : "-b");
                extra << (p.conv.dstT == SimdTensorData32f ? "f" : "b");
                extra << afs[p.conv.activation];
                desc = desc + p.Decription(extra.str());
            }

            void Call(void* context, const uint8_t* src, uint8_t* buf, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetDeconvolution16bForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_D(function) \
    FuncD(function, std::string(#function))

    bool SynetDeconvolution16bForwardAutoTest(float eps, const Param & p, SimdSynetCompatibilityType comp, FuncD f1, FuncD f2)
    {
        bool result = true;

        f1.Update(p, comp);
        f2.Update(p, comp);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters& c = p.conv;
        srand(0);
        Tensor32f weight(p.WeightShape());
        FillRandom(weight.Data(), weight.Size(), -1.0, 1.0f);

        Tensor32f bias({ c.dstC });
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        Tensor32f params({ c.dstC });
        FillRandom(params.Data(), params.Size(), 0.0f, 2.0f);

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

        Tensor32f src32f(p.SrcShape(), p.conv.srcF), dst32f1(p.DstShape(), p.conv.dstF), dst32f2(p.DstShape(), p.conv.dstF), buf32f;
        Tensor16u src16u(p.SrcShape(), p.conv.srcF), dst16u1(p.DstShape(), p.conv.dstF), dst16u2(p.DstShape(), p.conv.dstF), buf16u;
        FillRandom(src32f.Data(), src32f.Size(), -1.0, 1.0f);

        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16u.Data());

        const uint8_t* src = p.conv.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : (uint8_t*)src16u.Data();
        uint8_t* dst1 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : (uint8_t*)dst16u1.Data();
        uint8_t* dst2 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : (uint8_t*)dst16u2.Data();

        Fill(dst32f1, 0.1f);
        Fill(dst32f2, 1.1f);

        SimdFloat32ToBFloat16(dst32f1.Data(), dst32f1.Size(), dst16u1.Data());
        SimdFloat32ToBFloat16(dst32f2.Data(), dst32f2.Size(), dst16u2.Data());

        void* context1 = f1.func(p.batch, &p.conv, comp);
        void* context2 = f2.func(p.batch, &p.conv, comp);

        Tensor8u buf8u1, buf8u2;
        buf8u1.Extend({ ::SimdSynetDeconvolution16bExternalBufferSize(context1) });
        buf8u2.Extend({ ::SimdSynetDeconvolution16bExternalBufferSize(context2) });
        Fill(buf8u1, uint8_t(1));
        Fill(buf8u2, uint8_t(2));

        ::SimdSynetDeconvolution16bSetParams(context1, weight.Data(), bias.Data(), params.Data());
        ::SimdSynetDeconvolution16bSetParams(context2, weight.Data(), bias.Data(), params.Data());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u1.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u2.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (p.conv.dstT == SimdTensorData16b)
            eps = eps * 8.0f;

        if (p.conv.dstT == SimdTensorData16b)
        {
            SimdBFloat16ToFloat32(dst16u1.Data(), dst16u1.Size(), dst32f1.Data());
            SimdBFloat16ToFloat32(dst16u2.Data(), dst16u2.Size(), dst32f2.Data());
        }
        result = result && Compare(dst32f1, dst32f2, eps, true, 64, DifferenceBoth);

        if (1)
        {
            SimdConvolutionParameters c = p.conv;
            c.srcT = SimdTensorData32f;
            c.dstT = SimdTensorData32f;

            void* context3 = SimdSynetDeconvolution32fInit(p.batch, &c, SimdSynetCompatibilityDefault);

            Tensor32f dst32f3(p.DstShape(), p.conv.dstF), buf32f(Shp(::SimdSynetDeconvolution32fExternalBufferSize(context3)));

            ::SimdSynetDeconvolution32fSetParams(context3, weight.Data(), NULL, bias.Data(), params.Data());

            ::SimdSynetDeconvolution32fForward(context3, src32f.Data(), buf32f.Data(), dst32f3.Data());

            ::SimdRelease(context3);

            result = result && Compare(dst32f1, dst32f3, 0.03, true, 64, DifferenceBoth, " Compare to SynetDeconvolution32f.");
        }

        return result;
    }

    bool SynetDeconvolution16bForwardAutoTest(float eps, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _6(6, 6), _7(7, 7);
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
        const SimdBool tF = SimdFalse, tT = SimdTrue;
        SimdSynetCompatibilityType c = (SimdSynetCompatibilityType)(SimdSynetCompatibilityFmaUse | SimdSynetCompatibility16bfSoft);
#ifdef NDEBUG
#if 1
        //result = result && SynetDeconvolution16bForwardAutoTest(eps, Param(1, 720, 192, 256, 64, _4, _1, _2, _1, _1, 1, aId, tF, f32, f32), c, f1, f2);
        result = result && SynetDeconvolution16bForwardAutoTest(eps, Param(1, 24, 12, 16, 32, _2, _1, _1, _1, _1, 1, aId, tF, f32, f32), c, f1, f2);
#endif
#else
#if 1
        //result = result && SynetDeconvolution16bForwardAutoTest(eps, Param(1, 720, 192, 256, 64, _4, _1, _2, _1, _1, 1, aId, tF, f32, f32), c, f1, f2);
        result = result && SynetDeconvolution16bForwardAutoTest(eps, Param(1, 24, 12, 16, 32, _2, _1, _1, _1, _1, 1, aId, tF, f32, f32), c, f1, f2);
#endif
#endif

        return result;
    }

    bool SynetDeconvolution16bForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetDeconvolution16bForwardAutoTest(EPS, FUNC_D(Simd::Base::SynetDeconvolution16bInit), FUNC_D(SimdSynetDeconvolution16bInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41())
//            result = result && SynetDeconvolution16bForwardAutoTest(EPS, FUNC_D(Simd::Sse41::SynetDeconvolution16bInit), FUNC_D(SimdSynetDeconvolution16bInit));
//#endif 
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2())
//            result = result && SynetDeconvolution16bForwardAutoTest(EPS, FUNC_D(Simd::Avx2::SynetDeconvolution16bInit), FUNC_D(SimdSynetDeconvolution16bInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && SynetDeconvolution16bForwardAutoTest(EPS, FUNC_D(Simd::Avx512bw::SynetDeconvolution16bInit), FUNC_D(SimdSynetDeconvolution16bInit));
//#endif
//
//#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
//        if (Simd::AmxBf16::Enable && TestAmxBf16())
//            result = result && SynetDeconvolution16bForwardAutoTest(EPS, FUNC_D(Simd::AmxBf16::SynetDeconvolution16bInit), FUNC_D(SimdSynetDeconvolution16bInit));
//#endif

        return result;
    }
#endif
}
