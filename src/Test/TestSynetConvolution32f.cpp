/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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

#include "Simd/SimdSynetConvolution32f.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<false> Param;

        struct FuncC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

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

        Tensor32f buf;

        Tensor32f dst1({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });
        Tensor32f dst2({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });

        ::SimdFill32f(dst1.Data(), dst1.Size(), params.Data() + 0);
        ::SimdFill32f(dst2.Data(), dst2.Size(), params.Data() + 1);

        void * context1 = f1.func(p.batch, &p.conv, NULL);
        void * context2 = f2.func(p.batch, &p.conv, NULL);

        buf.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context1) });
        buf.Extend({ ::SimdSynetConvolution32fExternalBufferSize(context2) });

        ::SimdSynetConvolution32fSetParams(context1, weight.Data(), NULL, bias.Data(), params.Data());
        ::SimdSynetConvolution32fSetParams(context2, weight.Data(), NULL, bias.Data(), params.Data());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetConvolution32fForwardAutoTest(float eps, ::SimdConvolutionActivationType a, ::SimdBool t, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);

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
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 112, 96, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 112, 96, 32, _3, _1, _3, Size(1, 0), Size(1, 0), 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 19, 16, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 7, 6, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 7, 6, 128, _3, _1, _2, Size(0, 1), Size(1, 1), 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 4, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1024, 13, 13, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 10, 10, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 10, 10, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 20, 20, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 17, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 150, 150, 96, _2, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 17, 150, 150, 96, _2, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 150, 150, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 17, 150, 150, 96, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _1, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _1, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _2, _1, _1, _1, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _2, _1, _2, _1, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _4, _1, _1, _2, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _4, _1, _2, _2, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _7, _1, _1, _3, _3, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _7, _1, _2, _3, _3, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 150, 150, 16, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 75, 75, 144, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 38, 38, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 38, 38, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 38, 38, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 19, 19, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 19, 19, 384, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 19, 19, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 19, 19, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 19, 19, 576, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 10, 10, 160, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 160, 10, 10, 960, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 960, 10, 10, 160, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 960, 10, 10, 320, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 320, 10, 10, 1280, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1280, 10, 10, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 5, 5, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 5, 5, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 3, 3, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 3, 3, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 2, 2, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 2, 2, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 2, 2, 64, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 1, 1, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 12, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1280, 10, 10, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 5, 5, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 3, 3, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 2, 2, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 1, 1, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 75, 75, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 38, 38, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 19, 19, 384, _3, _1, _1, _1, _1, 384, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 576, _3, _1, _1, _1, _1, 576, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 960, 10, 10, 960, _3, _1, _1, _1, _1, 960, a, t), f1, f2);

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 75, 75, 144, _3, _1, _2, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 38, 38, 192, _3, _1, _2, _0, _1, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 576, _3, _1, _2, _1, _1, 576, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 10, 10, 256, _3, _1, _2, _0, _1, 256, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 5, 5, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 3, 3, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 2, 2, 64, _3, _1, _2, _0, _1, 64, a, t), f1, f2);
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
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 300, 300, 16, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 112, 112, 16, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 180, 320, 10, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 10, 89, 159, 16, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 87, 157, 32, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 224, 224, 16, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 75, 75, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 38, 38, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 19, 19, 384, _3, _1, _1, _1, _1, 384, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 576, _3, _1, _1, _1, _1, 576, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 960, 10, 10, 960, _3, _1, _1, _1, _1, 960, a, t), f1, f2);

        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 75, 75, 144, _3, _1, _2, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 38, 38, 192, _3, _1, _2, _0, _1, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 576, _3, _1, _2, _1, _1, 576, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 10, 10, 256, _3, _1, _2, _0, _1, 256, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 5, 5, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 3, 3, 128, _3, _1, _2, _1, _1, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 2, 2, 64, _3, _1, _2, _0, _1, 64, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 38, 32, 32, _3, _1, _2, _0, _1, 32, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 19, 16, 64, _3, _1, _3, _1, _1, 64, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 728, 14, 14, 728, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 56, 48, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 28, 24, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 300, 300, 32, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 150, 150, 32, _3, _1, _1, _1, _1, 32, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 150, 150, 16, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 150, 150, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 150, 150, 96, _3, _1, _2, _0, _1, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 75, 75, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 112, 96, 64, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 56, 48, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 56, 48, 128, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 28, 24, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 28, 24, 256, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 12, 512, _3, _1, _2, _0, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 1024, 1024, 24, _7, _1, _4, _3, _3, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 128, 128, 64, _5, _1, _2, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 116, 8, 8, 116, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 160, 160, 16, _3, _1, _1, _1, _1, 16, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 160, 160, 48, _3, _1, _2, _1, _0, 48, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 80, 80, 48, _3, _1, _2, _1, _0, 48, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 80, 80, 48, _3, _1, _1, _1, _1, 48, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 20, 20, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 20, 20, 192, _3, _1, _1, _1, _1, 192, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 40, 40, 96, _3, _1, _1, _1, _1, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 40, 40, 96, _3, _1, _2, _1, _0, 96, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 160, 160, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 8, 160, 160, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 16, 160, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 8, 16, 160, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 8, 80, 80, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 80, 80, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 8, 80, 80, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 115, 63, 4, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 115, 63, 2, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 7, 7, 128, _7, _1, _1, _0, _0, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 5, 5, 128, _5, _1, _1, _0, _0, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 3, 3, 128, _3, _1, _1, _0, _0, 128, a, t), f1, f2);
#endif
#if 0        
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 14, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 14, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 7, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 3, 3, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 23, 23, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 256, 14, 14, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 256, 14, 14, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 256, 14, 12, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 512, 7, 7, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 512, 7, 6, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 512, 3, 3, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 32, 23, 23, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 128, 24, 24, 8, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 128, 7, 7, 128, _7, _1, _1, _0, _0, 128, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 256, 2, 2, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 128, 64, 64, 128, _3, _1, _1, _1, _1, 2, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 128, 64, 64, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 128, 64, 64, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(2, 48, 160, 160, 48, _3, _1, _2, _1, _0, 48, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 8, 8, 48, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 19, 19, 12, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 21, 192, 192, 21, _3, _1, _1, _1, _1, 21, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 90, 192, 192, 90, _3, _1, _2, _0, _1, 90, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1152, 12, 12, 12, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1156, 12, 12, 12, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 192, 192, 96, _3, _1, _2, _1, _1, 96, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 96, 96, 144, _3, _1, _1, _1, _1, 144, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 96, 96, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 144, 96, 96, 24, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 144, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 17, 17, 128, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 17, 17, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 17, 17, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 20, 20, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 24, 42, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1280, 7, 12, 36, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 4, 6, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);

        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 24, 42, 16, _3, _2, _1, _2, _2, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 24, 42, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 25, 43, 16, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 24, 42, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 20, 20, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 10, 10, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 5, 5, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 4, 6, 18, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 4, 6, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 320, 7, 12, 1280, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1280, 7, 12, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 576, 7, 12, 160, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 14, 24, 576, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 240, 135, 10, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 10, 119, 67, 16, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 117, 65, 32, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 115, 63, 4, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 115, 63, 2, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 1024, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 256, 128, 128, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps*10, Param(1, 1024, 128, 128, 1024, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 120, 12, 12, 120, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 240, 135, 10, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 768, 10, 4, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 240, 135, 27, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 34, 32, 32, 34, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 34, 32, 32, 34, _3, _1, _1, _1, _1, 34, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 34, 32, 32, 34, _5, _1, _1, _2, _2, 34, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 34, 32, 32, 34, _7, _1, _1, _3, _3, 34, a, t), f1, f2);
#endif
#if 0        
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 48, 48, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 48, 24, 24, 48, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 12, 12, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 24, 96, 96, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 24, 48, 48, 24, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 48, 24, 24, 48, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 96, 12, 12, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0        
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 13, 13, 1152, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1152, 13, 13, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1152, 13, 13, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 448, 6, 6, 2048, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 2048, 6, 6, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 64, 128, 1, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 8, 60, 256, _2, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0        
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 96, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 80, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 64, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 48, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 32, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 16, 160, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 89, 159, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 150, 150, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 150, 150, 128, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 150, 150, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 150, 150, 32, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 1024, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 256, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 512, 64, 64, 512, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 75, 75, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 75, 75, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 17, 17, 128, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 17, 17, 64, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 17, 17, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 240, 135, 27, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 37, 47, 48, _3, _1, _2, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 16, 26, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 96, 96, 48, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 12, 1, 1, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 24, 1, 1, 384, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 3, 1, 1, 48, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 6, 1, 1, 96, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 40, 23, 16, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 40, 23, 16, _3, _2, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 40, 23, 16, _3, _3, _1, _3, _3, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 16, 40, 23, 16, _3, _5, _1, _5, _5, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 7, 59, 256, Size(9, 3), _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 6, 6, 2048, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 2048, 6, 6, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 20, 12, 20, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 20, 12, 128, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 20, 12, 128, _3, _1, _1, _1, _1, 128, a, t), f1, f2);
#endif
#if 0
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 448, 6, 6, 2048, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 13, 13, 1152, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 16, 16, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 16, 16, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 96, 12, 12, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 12, 12, 96, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 96, 14, 14, 96, _3, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 2048, 6, 6, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 10, 1, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 1, 1, 192, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 5, 2, 192, _1, _1, _1, _0, _0, 1, a, SimdFalse), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 192, 1, 1, 192, _1, _1, _1, _0, _0, 1, a, SimdFalse), f1, f2);
#endif
#if 0
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 728, 14, 14, 728, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 728, 14, 14, 728, _1, _1, _1, _0, _0, 728, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 27, 27, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(10, 384, 27, 27, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 384, 270, 27, 32, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 64, 109, 109, 128, _1, _1, _2, _0, _0, 1, a, t), f1, f2);
#endif
#if 0
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 128, 14, 14, 128, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 14, 14, 96, _5, _1, _1, _2, _2, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 1280, 14, 14, 1536, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 14, 14, 1536, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 512, 14, 14, 512, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#if 1
        //result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 1024, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 75, 75, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 32, 104, 104, 16, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
#else
        //result = result && SynetConvolution32fForwardAutoTest(eps * 10, Param(1, 1024, 64, 64, 256, _3, _1, _1, _1, _1, 1, a, t), f1, f2); 
        result = result && SynetConvolution32fForwardAutoTest(eps, Param(1, 256, 75, 75, 256, _1, _1, _1, _0, _0, 1, a, t), f1, f2);
#endif
        return result;
    }

    bool SynetConvolution32fForwardAutoTest(float eps, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationIdentity, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationRelu, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationRelu, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationLeakyRelu, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationRestrictRange, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationPrelu, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationElu, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationHswish, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationMish, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationHardSigmoid, ::SimdFalse, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationHardSigmoid, ::SimdTrue, f1, f2);
        //result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationSwish, ::SimdFalse, f1, f2);
        result = result && SynetConvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationSwish, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetConvolution32fForwardAutoTest()
    {
        const float EPS = 0.001f;
        bool result = true;

        result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Base::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Sse2::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetConvolution32fForwardAutoTest(4 * EPS, FUNC_C(Simd::Sse41::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Avx::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Avx2::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Avx512f::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvolution32fForwardAutoTest(2 * EPS, FUNC_C(Simd::Neon::SynetConvolution32fInit), FUNC_C(SimdSynetConvolution32fInit));
#endif

        return result;
    }
#endif
}
