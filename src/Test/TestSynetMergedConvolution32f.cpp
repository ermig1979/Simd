/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

#include "Simd/SimdSynetMergedConvolution32f.h"

namespace Test
{
    namespace
    {
        struct Param
        {
            SimdBool trans, add;
            size_t batch;
            SimdConvolutionParameters conv[3];
            mutable float *weight[3], *bias[3], *params[3];

            Param(size_t n, size_t c0, size_t h0, size_t w0, size_t k0, size_t s0, ::SimdConvolutionActivationType a0, 
                size_t c1, size_t k1, size_t s1, ::SimdConvolutionActivationType a1, size_t c2, ::SimdConvolutionActivationType a2, SimdBool a) 
            {
                trans = ::SimdTrue;
                batch = n;
                this->add = a;

                conv[0].srcC = c0;
                conv[0].srcH = h0;
                conv[0].srcW = w0;
                conv[0].dstC = c1;
                conv[0].kernelY = k0;
                conv[0].kernelX = k0;
                conv[0].dilationY = 1;
                conv[0].dilationX = 1;
                conv[0].strideY = s0;
                conv[0].strideX = s0;
                conv[0].padY = s0 == 1 || (conv[0].srcH & 1) ? (k0 - 1) / 2 : 0;
                conv[0].padX = s0 == 1 || (conv[0].srcW & 1) ? (k0 - 1) / 2 : 0;
                conv[0].padH = (k0 - 1) / 2;
                conv[0].padW = (k0 - 1) / 2;
                conv[0].group = 1;
                conv[0].activation = a0;
                conv[0].dstH = (conv[0].srcH + conv[0].padY + conv[0].padH - conv[0].kernelY) / conv[0].strideY + 1;
                conv[0].dstW = (conv[0].srcW + conv[0].padX + conv[0].padW - conv[0].kernelX) / conv[0].strideX + 1;

                conv[1].srcC = c1;
                conv[1].srcH = conv[0].dstH;
                conv[1].srcW = conv[0].dstW;
                conv[1].dstC = c1;
                conv[1].kernelY = k1;
                conv[1].kernelX = k1;
                conv[1].dilationY = 1;
                conv[1].dilationX = 1;
                conv[1].strideY = s1;
                conv[1].strideX = s1;
                conv[1].padY = s1 == 1 || (conv[1].srcH & 1) ? (k1 - 1) / 2 : 0;
                conv[1].padX = s1 == 1 || (conv[1].srcW & 1) ? (k1 - 1) / 2 : 0;
                conv[1].padH = (k1 - 1) / 2;
                conv[1].padW = (k1 - 1) / 2;
                conv[1].group = c1;
                conv[1].activation = a1;
                conv[1].dstH = (conv[1].srcH + conv[1].padY + conv[1].padH - conv[1].kernelY) / conv[1].strideY + 1;
                conv[1].dstW = (conv[1].srcW + conv[1].padX + conv[1].padW - conv[1].kernelX) / conv[1].strideX + 1;

                conv[2].srcC = c1;
                conv[2].srcH = conv[1].dstH;
                conv[2].srcW = conv[1].dstW;
                conv[2].dstC = c2;
                conv[2].kernelY = 1;
                conv[2].kernelX = 1;
                conv[2].dilationY = 1;
                conv[2].dilationX = 1;
                conv[2].strideY = 1;
                conv[2].strideX = 1;
                conv[2].padY = 0;
                conv[2].padX = 0;
                conv[2].padH = 0;
                conv[2].padW = 0;
                conv[2].group = 1;
                conv[2].activation = a2;
                conv[2].dstH = (conv[2].srcH + conv[2].padY + conv[2].padH - conv[2].kernelY) / conv[2].strideY + 1;
                conv[2].dstW = (conv[2].srcW + conv[2].padX + conv[2].padW - conv[2].kernelX) / conv[2].strideX + 1;

                for (size_t i = 0; i < 3; ++i)
                {
                    conv[i].srcT = SimdTensorData32f;
                    conv[i].srcF = SimdTensorFormatNhwc;
                    conv[i].dstT = SimdTensorData32f;
                    conv[i].dstF = SimdTensorFormatNhwc;
                }
            }
        };

        struct FuncMC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * params, size_t count, SimdBool add);

            FuncPtr func;
            String description;

            FuncMC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << p.batch << "x" << p.conv[0].srcC << "x" << p.conv[0].srcH << "x" << p.conv[0].srcW;
                ss << "-" << p.conv[0].dstC << "x" << p.conv[0].kernelY << "x" << p.conv[0].strideY;
                ss << "-" << p.conv[1].kernelY << "x" << p.conv[1].strideY << "-" << p.conv[2].dstC;
                ss << "]";
                description = ss.str();
            }

            void Call(const Param & p, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst) const
            {
                void * context = func(p.batch, p.conv, 3, p.add);
                buf.Extend({ ::SimdSynetMergedConvolution32fExternalBufferSize(context) });
                ::SimdSynetMergedConvolution32fSetParams(context, p.weight, NULL, p.bias, p.params);
                if (p.add)
                {
                    float value = 1.1f;
                    SimdFill32f(dst.Data(), dst.Size(), &value);
                }
                {
                    TEST_PERFORMANCE_TEST(description);
                    ::SimdSynetMergedConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
                }
                ::SimdRelease(context);
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    bool SynetMergedConvolution32fForwardAutoTest(float eps, const Param & p, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << "].");

        Tensor32f src({ p.batch, p.conv[0].srcH, p.conv[0].srcW, p.conv[0].srcC });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight[3], bias[3], params[3];
        for (size_t i = 0; i < 3; ++i)
        {
            weight[i].Reshape({ p.conv[i].kernelY, p.conv[i].kernelX, p.conv[i].srcC / p.conv[i].group, p.conv[i].dstC });
            FillRandom(weight[i].Data(), weight[i].Size(), -1.0, 1.0f);
            p.weight[i] = weight[i].Data();

            bias[i].Reshape({ p.conv[i].dstC });
            FillRandom(bias[i].Data(), bias[i].Size(), -1.0, 1.0f);
            p.bias[i] = bias[i].Data();

            params[i].Reshape({ Simd::Max<size_t>(2, p.conv[i].dstC) });
            FillRandom(params[i].Data(), params[i].Size(), -1.0, 1.0f);
            params[i].Data()[0] = 0.0f + 0.1f * float(i);
            params[i].Data()[1] = 1.0f + 0.1f * float(i);
            p.params[i] = params[i].Data();
        }

        Tensor32f buf;

        Tensor32f dst1({ p.batch, p.conv[2].dstH, p.conv[2].dstW, p.conv[2].dstC}, SimdTensorFormatNhwc, 0.01f);
        Tensor32f dst2({ p.batch, p.conv[2].dstH, p.conv[2].dstW, p.conv[2].dstC}, SimdTensorFormatNhwc, 0.02f);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, buf, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, buf, dst2));

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetMergedConvolution32fForwardAutoTest(float eps, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;
        const SimdBool t = SimdTrue, f = SimdFalse;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationRestrictRange, a1 = ::SimdConvolutionActivationRestrictRange, a2 = ::SimdConvolutionActivationIdentity;
        const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationPrelu, a1 = ::SimdConvolutionActivationElu, a2 = ::SimdConvolutionActivationPrelu;
#ifdef NDEBUG
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 3, 384, 384, 3, 2, a0, 32, 3, 1, a1, 16, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 16, 192, 192, 1, 1, a0, 96, 3, 2, a1, 24, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 1, 1, a0, 144, 3, 1, a1, 24, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 1, 1, a0, 144, 3, 2, a1, 32, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 32, 48, 48, 1, 1, a0, 192, 3, 1, a1, 32, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 32, 48, 48, 1, 1, a0, 192, 3, 2, a1, 64, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 64, 24, 24, 1, 1, a0, 384, 3, 1, a1, 64, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 64, 24, 24, 1, 1, a0, 384, 3, 1, a1, 96, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 96, 24, 24, 1, 1, a0, 576, 3, 1, a1, 96, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 160, 12, 12, 1, 1, a0, 960, 3, 1, a1, 320, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 160, 12, 12, 1, 1, a0, 960, 3, 1, a1, 160, a2, f), f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 3, 384, 384, 3, 2, a0, 35, 3, 1, a1, 17, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 17, 192, 192, 1, 1, a0, 99, 3, 2, a1, 27, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 27, 96, 96, 1, 1, a0, 147, 3, 1, a1, 27, a2, f), f1, f2);
#endif
#if 1
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 160, 10, 10, 1, 1, a0, 960, 3, 1, a1, 160, a2, f), f1, f2);
#endif
#else
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 160, 10, 10, 1, 1, a0, 960, 3, 1, a1, 160, a2, f), f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 24, 96, 96, 1, 1, a0, 144, 3, 1, a1, 24, a2, f), f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(1, 32, 22, 22, 1, 1, a0, 175, 3, 2, a1, 64, a2, f), f1, f2);
#endif
        return result;
    }

    bool SynetMergedConvolution32fForwardAutoTest()
    {
        bool result = true;

        result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Base::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Sse2::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx2::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx512f::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Neon::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

        return result;
    }
}
