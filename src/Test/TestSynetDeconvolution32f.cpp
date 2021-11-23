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

#include "Simd/SimdSynetDeconvolution32f.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        typedef Test::SynetConvolutionParam<true> Param;

        struct FuncD
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * conv, SimdGemm32fNNPtr gemm);

            FuncPtr func;
            String description;

            FuncD(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(const Param & p)
            {
                description = description + p.Decription();
            }

            void Call(const Param & p, const Tensor32f & weight, const Tensor32f & bias, const Tensor32f & params, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst) const
            {
                void * context = func(p.batch, &p.conv, NULL);
                buf.Extend({ ::SimdSynetDeconvolution32fExternalBufferSize(context) });
                ::SimdSynetDeconvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());
                {
                    TEST_PERFORMANCE_TEST(description);
                    ::SimdSynetDeconvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
                }
                ::SimdRelease(context);
            }
        };
    }

#define FUNC_D(function) \
    FuncD(function, std::string(#function))

    bool SynetDeconvolution32fForwardAutoTest(float eps, const Param & p, FuncD f1, FuncD f2)
    {
        bool result = true;


        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.description << " & " << f2.description << "].");

        const SimdConvolutionParameters & c = p.conv;
        Tensor32f src({p.batch, p.trans ? c.srcH : c.srcC, p.trans ? c.srcW : c.srcH, p.trans ? c.srcC : c.srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight({ c.srcC, p.trans ? c.kernelY : c.dstC / c.group, p.trans ? c.kernelX : c.kernelY, p.trans ? c.dstC / c.group : c.kernelX });
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

        Tensor32f buf;

        Tensor32f dst1({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });
        Tensor32f dst2({ p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW });

        ::SimdFill32f(dst1.Data(), dst1.Size(), params.Data() + 0);
        ::SimdFill32f(dst2.Data(), dst2.Size(), params.Data() + 1);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, weight, bias, params, src, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, weight, bias, params, src, buf, dst2));

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetDeconvolution32fForwardAutoTest(float eps, ::SimdConvolutionActivationType a, ::SimdBool t, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);

#ifdef NDEBUG
#if 0
        result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 24, 11, 20, 24, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
        result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 24, 22, 40, 24, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
        result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 24, 44, 80, 24, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 32, 44, 80, 30, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
#endif
#if 1
        result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 512, 44, 80, 512, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
#endif
#else
        result = result && SynetDeconvolution32fForwardAutoTest(eps, Param(1, 512, 44, 80, 512, _2, _1, _2, _0, _0, 1, a, t), f1, f2);
#endif
        return result;
    }

    bool SynetDeconvolution32fForwardAutoTest(float eps, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationIdentity, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationRelu, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationLeakyRelu, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationRestrictRange, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationPrelu, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationElu, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationHswish, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationMish, ::SimdTrue, f1, f2);
        //result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationHardSigmoid, ::SimdTrue, f1, f2);
        result = result && SynetDeconvolution32fForwardAutoTest(eps, ::SimdConvolutionActivationSwish, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetDeconvolution32fForwardAutoTest()
    {
        bool result = true;

        result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Base::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Sse2::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Avx::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Avx2::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Avx512f::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetDeconvolution32fForwardAutoTest(EPS, FUNC_D(Simd::Neon::SynetDeconvolution32fInit), FUNC_D(SimdSynetDeconvolution32fInit));
#endif

        return result;
    }
#endif
}
