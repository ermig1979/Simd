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

#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution32f.h"
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
                ::SimdSynetConvolution16bForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    bool SynetConvolution16bForwardAutoTest(float eps, const Param& p, SimdSynetCompatibilityType comp, FuncC f1, FuncC f2)
    {
        bool result = true;

        f1.Update(p, comp);
        f2.Update(p, comp);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters& c = p.conv;
        srand(0);
        Tensor32f weight(p.WeightShape());
        FillRandom(weight.Data(), weight.Size(), -10.0, 10.0f);

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
        if (context1 == NULL)
        {
            TEST_LOG_SS(Info, f1.desc << " can't create context!.");
            return false;
        }
        void* context2 = f2.func(p.batch, &p.conv, comp);
        if (context2 == NULL)
        {
            TEST_LOG_SS(Info, f2.desc << " can't create context!.");
            return false;
        }

        Tensor8u buf8u1, buf8u2;
        buf8u1.Extend({ ::SimdSynetConvolution16bExternalBufferSize(context1) });
        buf8u2.Extend({ ::SimdSynetConvolution16bExternalBufferSize(context2) });
        Fill(buf8u1, uint8_t(1));
        Fill(buf8u2, uint8_t(-1));

        ::SimdSynetConvolution16bSetParams(context1, weight.Data(), bias.Data(), params.Data());
        ::SimdSynetConvolution16bSetParams(context2, weight.Data(), bias.Data(), params.Data());

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

        if(0)
        {
            SimdConvolutionParameters c = p.conv;
            c.srcT = SimdTensorData32f;
            c.dstT = SimdTensorData32f;

            void* context3 = SimdSynetConvolution32fInit(p.batch, &c);

            Tensor32f dst32f3(p.DstShape(), p.conv.dstF), buf32f(Shp( ::SimdSynetConvolution32fExternalBufferSize(context3)));

            ::SimdSynetConvolution32fSetParams(context3, weight.Data(), NULL, bias.Data(), params.Data());

            ::SimdSynetConvolution32fForward(context3, src32f.Data(), buf32f.Data(), dst32f3.Data());

            ::SimdRelease(context3);

            result = result && Compare(dst32f1, dst32f3, 0.03, true, 64, DifferenceBoth, " Compare to SynetConvolution32f.");//0.129
        }

        return result;
    }

    bool SynetConvolution16bForwardAutoTest(float eps, const FuncC& f1, const FuncC& f2)
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
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 128, 128, 127, _1, _1, _1, _0, _0, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 128, 128, 127, _1, _1, _1, _0, _0, 1, aId, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 99, 101, 101, 128, _1, _1, _1, _0, _0, 1, aId, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 512, 32, 32, 256, _1, _1, _1, _0, _0, 1, aId, tT, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 32, 32, 256, _1, _1, _1, _0, _0, 1, aLr, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 99, 101, 101, 149, _1, _1, _1, _0, _0, 1, aId, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1255, 32, 32, 128, _1, _1, _1, _0, _0, 1, aId, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 128, 64, 64, 128, _3, _1, _2, _1, _1, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 128, 32, 32, 128, _3, _1, _1, _1, _1, 1, aGe, tT, b16, b16), c, f1, f2); 
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 32, 16, 16, 256, _5, _1, _1, _2, _2, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 32, 32, 256, _3, _1, _1, _1, _1, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 16, 16, 256, _1, _1, _1, _0, _0, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64, 16, 16, 256, Size(3, 4), _1, _1, _1, Size(1, 2), 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 48, 24, 24, 64, _3, _1, _1, _1, _1, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 3072, 24, 24, 256, _1, _1, _1, _0, _0, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 600, 24, 24, 64, _1, _1, _1, _0, _0, 1, aRe, tT, b16, b16), c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(12, 768 + 768, 5, 5, 255, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768 + 768, 127, 127, 255, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 77, 77, 255, _3, _1, _1, _1, _1, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 32, 16, 16, 256, _5, _1, _1, _2, _2, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 32, 32, 256, _3, _1, _1, _1, _1, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 16, 16, 256, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64, 16, 16, 256, Size(3, 4), _1, _1, _1, Size(1, 2), 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 48, 24, 24, 64, _3, _1, _1, _1, _1, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 3072, 24, 24, 256, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 600, 24, 24, 64, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 3072, 64, 64, 768, _1, _1, _1, _0, _0, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 64, 64, 768, _1, _1, _1, _0, _0, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 64, 64, 768, _1, _1, _1, _0, _0, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _1, _1, _1, _0, _0, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _1, _1, _1, _0, _0, 1, aId, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _1, _1, _1, _0, _0, 1, aId, tT, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _1, _1, _1, _0, _0, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _3, _1, _1, _1, _1, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _3, _1, _1, _1, _1, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _5, _1, _1, _2, _2, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _5, _1, _1, _2, _2, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _7, _1, _1, _3, _3, 1, aId, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _7, _1, _1, _3, _3, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64, 128, 128, 256, _3, _1, _1, _1, _1, 1, aId, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64, 128, 128, 256, _3, _1, _1, _1, _1, 1, aId, tT, b16, b16), c, f1, f2);

        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 32, 32, 256, Size(3, 4), _1, _1, _1, Size(1, 2), 1, aId, tT, b16, b16), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64, 32, 32, 32, _3, _1, _1, _1, _1, 1, aId, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 64 * 9, 32, 32, 32, _1, _1, _1, _0, _0, 1, aId, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 768, 16, 16, 256, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 32, 32, 1024, _3, _1, _1, _1, _1, 1, aId, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 128, 64, 64, 128, _3, _1, _1, _1, _1, 1, aId, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 128, 64, 64, 128, _3, _1, _1, _1, _1, 1, aId, tT, b16, b16), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 255, 31, 31, 1023, _3, _1, _1, _1, _1, 1, aGe, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 127, 64, 64, 128, _3, _1, _1, _1, _1, 1, aPr, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 36, 36, 256, _5, _1, _1, _0, _0, 1, aId, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(24, 2048, 6, 6, 256, _3, _1, _1, _0, _0, 1, aId, tT, f32, f32), c, f1, f2);
#endif
#if 0
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 44, 44, 256, _1, _1, _1, _0, _0, 1, aSw, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 32, 321, 321, 16, _2, _1, _1, _0, _0, 1, aRe, tT, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 16, 320, 320, 32, _2, _1, _1, _0, _1, 1, aRe, tT, b16, f32), c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 48, 48, 256, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 48, 48, 256, _1, _1, _1, _0, _0, 1, aPr, tF, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 48, 48, 256, _1, _1, _1, _0, _0, 1, aPr, tF, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 48, 48, 256, _1, _1, _1, _0, _0, 1, aPr, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 48, 48, 256, _1, _1, _1, _0, _0, 1, aPr, tT, f32, f32), c, f1, f2);
        //result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 256, 16, 16, 2560, _1, _1, _1, _0, _0, 1, aPr, tT, f32, b16), c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 2255, 55, 55, 155, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 55, 16, 16, 55, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 55, 15, 16, 55, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 56, 15, 15, 56, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 55, 15, 15, 56, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 55, 15, 15, 55, _1, _1, _1, _0, _0, 1, aPr, tF, b16, b16), c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 76, 64, 64, 76, _7, _1, _1, _3, _3, 76, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 76, 64, 64, 76, _7, _1, _1, _3, _3, 76, aPr, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 76, 64, 64, 76, _3, _1, _1, _1, _1, 76, aRe, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 152, 32, 32, 152, _7, _1, _1, _3, _3, 152, aPr, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 304, 16, 16, 304, _7, _1, _1, _3, _3, 304, aPr, tT, f32, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 304, 16, 16, 304, _3, _1, _1, _1, _1, 304, aRe, tT, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 304, 17, 15, 304, _7, _1, _1, _3, _3, 304, aRe, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 608, 8, 8, 608, _7, _1, _1, _3, _3, 608, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 152, 32, 32, 152, _5, _1, _2, _2, _2, 152, aPr, tT, b16, b16), c, f1, f2);
#endif
#if 1
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 7, 7, 1024, _1, _1, _1, _0, _0, 1, aRe, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 8, 8, 1536, _1, _1, _1, _0, _0, 1, aRe, tT, b16, f32), c, f1, f2);

        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 7, 7, 1536, _1, _1, _1, _0, _0, 1, aRe, tT, b16, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 7, 7, 1536, _1, _1, _1, _0, _0, 1, aRe, tT, f32, b16), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 7, 7, 1536, _1, _1, _1, _0, _0, 1, aRe, tT, b16, f32), c, f1, f2);
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 1024, 7, 7, 1536, _1, _1, _1, _0, _0, 1, aRe, tT, f32, f32), c, f1, f2);
#endif

#else
        result = result && SynetConvolution16bForwardAutoTest(eps, Param(1, 2048, 7, 7, 2048, _1, _1, _1, _0, _0, 1, aRe, tT, b16, f32), c, f1, f2);
#endif

        return result;
    }

    bool SynetConvolution16bForwardAutoTest()
    {
        const float EPS = 0.001f;
        bool result = true;

        if(TestBase())
            result = result && SynetConvolution16bForwardAutoTest(EPS, FUNC_C(Simd::Base::SynetConvolution16bInit), FUNC_C(SimdSynetConvolution16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetConvolution16bForwardAutoTest(EPS, FUNC_C(Simd::Sse41::SynetConvolution16bInit), FUNC_C(SimdSynetConvolution16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetConvolution16bForwardAutoTest(EPS, FUNC_C(Simd::Avx2::SynetConvolution16bInit), FUNC_C(SimdSynetConvolution16bInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetConvolution16bForwardAutoTest(EPS, FUNC_C(Simd::Avx512bw::SynetConvolution16bInit), FUNC_C(SimdSynetConvolution16bInit));
#endif

#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
        if (Simd::AmxBf16::Enable && TestAmxBf16())
            result = result && SynetConvolution16bForwardAutoTest(EPS, FUNC_C(Simd::AmxBf16::SynetConvolution16bInit), FUNC_C(SimdSynetConvolution16bInit));
#endif

        return result;
    }
#endif
}
