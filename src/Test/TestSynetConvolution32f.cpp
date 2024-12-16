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
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv);

            FuncPtr func;
            String desc;

            FuncC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                desc = desc + p.Decription();
            }

            void Call(void * context, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
            }
        };
    }

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    bool SynetConvolution32fForwardAutoTest(float eps, const Param & p, FuncC f1, FuncC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters & c = p.conv;
        Tensor32f src({p.batch, p.trans ? c.srcH : c.srcC, p.trans ? c.srcW : c.srcH, p.trans ? c.srcC : c.srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight({ p.trans ? c.kernelY : c.dstC, p.trans ? c.kernelX : c.srcC / c.group,
            p.trans ? c.srcC / c.group : c.kernelY, p.trans ? c.dstC : c.kernelX });
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
        else if(p.conv.activation == ::SimdConvolutionActivationMish)
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

        Tensor32f buf1, buf2;

        Tensor32f dst1({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });
        Tensor32f dst2({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });

        ::SimdFill32f(dst1.Data(), dst1.Size(), params.Data() + 0);
        ::SimdFill32f(dst2.Data(), dst2.Size(), params.Data() + 1);

        void * context1 = f1.func(p.batch, &p.conv);
        void * context2 = f2.func(p.batch, &p.conv);

        buf1.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context1) });
        buf2.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context2) });

        FillRandom(buf1.Data(), buf1.Size(), 1.0f, 1.0f);
        FillRandom(buf2.Data(), buf2.Size(), 2.0f, 2.0f);

        ::SimdSynetConvolution32fSetParams(context1, weight.Data(), NULL, bias.Data(), params.Data());
        ::SimdSynetConvolution32fSetParams(context2, weight.Data(), NULL, bias.Data(), params.Data());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf1, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf2, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetConvolution32fForwardAutoTest(float eps, SimdConvolutionActivationType a, SimdBool t, const FuncC& f1, const FuncC& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _6(6, 6), _7(7, 7);
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
        const SimdBool tF = SimdFalse, tT = SimdTrue;

#ifdef NDEBUG
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 112, 96, 32, _3, _1, _3, Size(1, 0), Size(1, 0), 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 114, 96, 32, _3, _1, _3, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 19, 16, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 7, 7, 128, _7, _1, _1, _0, _0, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 56, 56, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 112, 112, 16, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 7, 6, 128, _3, _1, _2, Size(0, 1), Size(1, 1), 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 4, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 256, 256, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 128, 128, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 64, 64, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 32, 32, 384, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 768, 16, 16, 768, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1536, 8, 8, 1536, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3072, 4, 4, 3072, _1, _1, _1, _0, _0, 1, a, t), f1, f2);//slow

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 256, 256, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);//slow
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 128, 128, 32, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 64, 64, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 32, 32, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 16, 16, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 8, 8, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1024, 4, 4, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 10, 256, 256, 10, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 20, 128, 128, 20, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 40, 64, 64, 40, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 80, 32, 32, 80, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 160, 16, 16, 160, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 320, 8, 8, 320, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 640, 4, 4, 640, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 7, 7, 768, _3, _1, _1, _1, _1, 384, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 14, 14, 384, _7, _1, _2, _3, _3, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 28, 28, 192, _7, _1, _2, _3, _3, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 56, 56, 96, _7, _1, _2, _3, _3, 48, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 7, 1024, _3, _1, _1, _1, _1, 512, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 14, 512, _7, _1, _2, _3, _3, 256, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 28, 28, 256, _7, _1, _2, _3, _3, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 56, 56, 128, _7, _1, _2, _3, _3, 64, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 49, 29, 29, 98, _7, _1, _2, _3, _3, 49, a, t), f1, f2);
#endif
#if 0 
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 24, 24, 96, _1, _1, _2, _0, _0, 1, aGe, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 14, 14, 128, _5, _1, _1, _2, _2, 1, aEl, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 160, 28, 28, 160, _3, _1, _2, _1, _1, 10, aRe, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 7, 7, 768, _3, _1, _1, _1, _1, 384, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 28, 28, 256, _3, _1, _1, _1, _1, 1, aLr, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 24, 24, 512, _1, _1, _1, _0, _0, 1, aHs, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 24, 24, 256, _5, _1, _1, _2, _2, 256, aPr, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 5, 5, 128, _1, _1, _1, _0, _0, 1, aMi, tF), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 24, 24, 64, Size(24, 24), _1, _1, _0, _0, 64, aHi, tF), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 24, 24, 256, _3, _1, _1, _1, _1, 1, aMi, tF), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 64, 64, 16, _3, _1, _1, _1, _1, 1, aPr, tF), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 24, 24, 256, _1, _1, _1, _0, _0, 1, aHi, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 24, 24, 128, _3, _1, _1, _2, _2, 1, aGe, tT), f1, f2);
#endif
#if 0 
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 32, 32, 128, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 32, 32, 128, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 32, 32, 128, _3, _1, _1, _1, _1, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 32, 32, 128, _3, _1, _1, _1, _1, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 91, 33, 33, 65, _3, _1, _1, _1, _1, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(25, 67, 15, 15, 35, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3000, 15, 15, 35, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3001, 5, 5, 1201, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 93, 115, 115, 31, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3000, 15, 15, 35, _1, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 2, 33, 32, _2, _1, _1, _0, _0, 1, aId, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 32, 32, 256, _3, _1, _1, _1, _1, 1, aRe, tT), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 32, 32, 256, _1, _1, _1, _0, _0, 1, aRe, tT), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3072, 64, 64, 768, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 768, 64, 64, 768, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _7, _1, _1, _3, _3, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 128, 128, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 16, 16, 256, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 768, 16, 16, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 16, 16, 256, Size(3, 4), _1, _1, _1, Size(1, 2), 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 24, 24, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3072, 24, 24, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 600, 24, 24, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(12, 2000, 5, 5, 255, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(24, 2048, 6, 6, 255, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 1
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 816, 14, 14, 816, _5, _1, _1, _2, _2, 816, aRe, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 304, 16, 16, 304, _3, _1, _1, _1, _1, 304, aRe, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 304, 16, 16, 304, _7, _1, _1, _3, _3, 304, aPr, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 608, 8, 8, 608, _7, _1, _1, _3, _3, 608, aRe, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 608, 7, 7, 608, _7, _1, _1, _3, _3, 608, aRe, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 152, 32, 32, 152, _7, _1, _1, _3, _3, 152, aRe, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 76, 64, 64, 76, _7, _1, _1, _3, _3, 76, aRe, t), f1, f2);
#endif
#else
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 44, 44, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
        return result;
    }

    bool SynetConvolution32fForwardAutoTest(float eps, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
        const SimdBool tF = SimdFalse, tT = SimdTrue;

#ifdef NDEBUG
        //result = result && SynetConvolution32fForwardAutoTest(eps, SimdConvolutionActivationGelu, tF, f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, SimdConvolutionActivationRelu, tT, f1, f2);
#else
        //result = result && SynetConvolution32fForwardAutoTest(eps, SimdConvolutionActivationGelu, tF, f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, SimdConvolutionActivationRelu, tT, f1, f2);
#endif

        return result;
    }

    bool SynetConvolution32fForwardAutoTest()
    {
        const float EPS = 0.001f;
        bool result = true;

        if(TestBase())
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Base::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetConvolution32fForwardAutoTest(4 * EPS, FUNC_C(Simd::Sse41::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Avx2::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Avx512bw::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Neon::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

        return result;
    }
#endif
}
