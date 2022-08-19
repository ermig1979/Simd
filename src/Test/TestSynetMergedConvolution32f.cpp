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
#include "Test/TestString.h"

#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct Cnv
        {
            SimdConvolutionActivationType a;
            size_t k, s, d;
            Cnv(SimdConvolutionActivationType a_, size_t k_, size_t s_, size_t d_ = - 1) : a(a_), k(k_), s(s_), d(d_) {}
        };

        struct Param
        {
            SimdBool trans, add;
            size_t batch, count;
            SimdConvolutionParameters conv[3];
            mutable float *weight[3], *bias[3], *params[3];

            Param(const Shape & in, const Cnv & c0, const Cnv& c1, const Cnv& c2, SimdBool a)
            {
                count = 3;
                trans = ::SimdTrue;
                batch = in[0];
                add = a;
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
                SetConv(conv + 2, c2);
            }

            Param(const Shape& in, const Cnv& c0, const Cnv& c1)
            {
                count = 2;
                trans = SimdTrue;
                batch = in[0];
                add = SimdFalse;
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
            }

        private:
            static void SetConv(SimdConvolutionParameters* conv, const Cnv & c, const Shape & s = Shape())
            {
                conv[0].srcC = s.empty() ? conv[-1].dstC : s[1];
                conv[0].srcH = s.empty() ? conv[-1].dstH : s[2];
                conv[0].srcW = s.empty() ? conv[-1].dstW : s[3];
                conv[0].dstC = c.d == -1 ? conv[0].srcC : c.d;
                conv[0].kernelY = c.k;
                conv[0].kernelX = c.k;
                conv[0].dilationY = 1;
                conv[0].dilationX = 1;
                conv[0].strideY = c.s;
                conv[0].strideX = c.s;
                conv[0].padY = c.s == 1 || (conv[0].srcH & 1) ? (c.k - 1) / 2 : (c.k - 1) / 2 - 1;
                conv[0].padX = c.s == 1 || (conv[0].srcW & 1) ? (c.k - 1) / 2 : (c.k - 1) / 2 - 1;
                conv[0].padH = (c.k - 1) / 2;
                conv[0].padW = (c.k - 1) / 2;
                conv[0].group = c.d == -1 ? conv[0].srcC : 1;
                conv[0].activation = c.a;
                conv[0].dstH = (conv[0].srcH + conv[0].padY + conv[0].padH - conv[0].kernelY) / conv[0].strideY + 1;
                conv[0].dstW = (conv[0].srcW + conv[0].padX + conv[0].padW - conv[0].kernelX) / conv[0].strideX + 1;
                conv[0].srcT = SimdTensorData32f;
                conv[0].srcF = SimdTensorFormatNhwc;
                conv[0].dstT = SimdTensorData32f;
                conv[0].dstF = SimdTensorFormatNhwc;
            }
        };

        struct FuncMC
        {
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * params, size_t count, SimdBool add, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String description;

            FuncMC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(const Param & p, SimdSynetCompatibilityType c)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << p.count << ":" << p.batch << "x" << p.conv[0].srcC << "x" << p.conv[0].srcH << "x" << p.conv[0].srcW;
                for (size_t i = 0; i < p.count; ++i)
                    ss << "-" << (p.conv[i].group != 1 ? String("") : ToString(p.conv[i].dstC) + "x") << p.conv[i].kernelY << "x" << p.conv[i].strideY;
                ss << (Simd::Base::Bf16Soft(c) ? "-bf16" : "-fp32") << "]";
                description = ss.str();
            }

            void Call(void* context, const Tensor32f & src, Tensor32f & buf, Tensor32f & dst, int add) const
            {
                if (add)
                {
                    float value = 1.1f;
                    SimdFill32f(dst.Data(), dst.Size(), &value);
                }
                TEST_PERFORMANCE_TEST(description);
                ::SimdSynetMergedConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    bool SynetMergedConvolution32fForwardAutoTest(float eps, const Param & p, SimdSynetCompatibilityType c, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p, c);
        f2.Update(p, c);

        TEST_LOG_SS(Info, "Test [" << f1.description << " & " << f2.description << "].");

        Tensor32f src(Shp(p.batch, p.conv[0].srcH, p.conv[0].srcW, p.conv[0].srcC));
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight[3], bias[3], params[3];
        for (size_t i = 0; i < p.count; ++i)
        {
            weight[i].Reshape(Shp(p.conv[i].kernelY, p.conv[i].kernelX, p.conv[i].srcC / p.conv[i].group, p.conv[i].dstC));
            FillRandom(weight[i].Data(), weight[i].Size(), -1.0, 1.0f);
            p.weight[i] = weight[i].Data();

            bias[i].Reshape(Shp(p.conv[i].dstC));
            FillRandom(bias[i].Data(), bias[i].Size(), -1.0, 1.0f);
            p.bias[i] = bias[i].Data();

            params[i].Reshape(Shp(Simd::Max<size_t>(2, p.conv[i].dstC)));
            FillRandom(params[i].Data(), params[i].Size(), -1.0, 1.0f);
            if (p.conv[i].activation == ::SimdConvolutionActivationHswish)
            {
                params[i].Data()[0] = 3.0f;
                params[i].Data()[1] = 1.0f / 6.0f;
            }
            else if (p.conv[i].activation == ::SimdConvolutionActivationMish)
                params[i].Data()[0] = 20.0f;
            else if (p.conv[i].activation == ::SimdConvolutionActivationHardSigmoid)
            {
                params[i].Data()[0] = 1.0f / 6.0f;
                params[i].Data()[1] = 0.5f;
            }
            else
            {
                params[i].Data()[0] = 0.0f + 0.1f * float(i);
                params[i].Data()[1] = 1.0f + 0.1f * float(i);
            }
            p.params[i] = params[i].Data();
        }

        Tensor32f buf;

        const SimdConvolutionParameters & end = p.conv[p.count - 1];
        Tensor32f dst1(Shp(p.batch, end.dstH, end.dstW, end.dstC), SimdTensorFormatNhwc, 0.01f);
        Tensor32f dst2(Shp(p.batch, end.dstH, end.dstW, end.dstC), SimdTensorFormatNhwc, 0.02f);

        TEST_ALIGN(SIMD_ALIGN);

        void* context1 = f1.func(p.batch, p.conv, p.count, p.add, c);
        void* context2 = f2.func(p.batch, p.conv, p.count, p.add, c);

        buf.Extend({ ::SimdSynetMergedConvolution32fExternalBufferSize(context1) });
        buf.Extend({ ::SimdSynetMergedConvolution32fExternalBufferSize(context2) });

        ::SimdSynetMergedConvolution32fSetParams(context1, p.weight, NULL, p.bias, p.params);
        ::SimdSynetMergedConvolution32fSetParams(context2, p.weight, NULL, p.bias, p.params);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf, dst1, p.add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf, dst2, p.add));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        float epsilon = (Simd::Base::FmaAvoid(c) ? eps * eps : 0.07f);
#else
        float epsilon = eps;
#endif
        result = result && Compare(dst1, dst2, epsilon, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetMergedConvolution32fForwardAutoTest(float eps, SimdSynetCompatibilityType c, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;
        const SimdBool t = SimdTrue, f = SimdFalse;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationHswish, a1 = ::SimdConvolutionActivationIdentity, a2 = ::SimdConvolutionActivationPrelu;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationPrelu, a1 = ::SimdConvolutionActivationRestrictRange, a2 = ::SimdConvolutionActivationHswish;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationMish, a1 = ::SimdConvolutionActivationMish, a2 = ::SimdConvolutionActivationMish;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationHardSigmoid, a1 = ::SimdConvolutionActivationHardSigmoid, a2 = ::SimdConvolutionActivationHardSigmoid;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationSwish, a1 = ::SimdConvolutionActivationSwish, a2 = ::SimdConvolutionActivationSwish;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationIdentity, a1 = ::SimdConvolutionActivationIdentity, a2 = ::SimdConvolutionActivationIdentity;
        //const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationPrelu, a1 = ::SimdConvolutionActivationPrelu, a2 = ::SimdConvolutionActivationPrelu;
        const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationRestrictRange, a1 = ::SimdConvolutionActivationRestrictRange, a2 = ::SimdConvolutionActivationRestrictRange;
#if defined(NDEBUG)
#if 1
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 384, 389), Cnv(a0, 3, 2, 32), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 16), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 16, 192, 199), Cnv(a0, 1, 1, 96), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 24), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 24, 96, 99), Cnv(a0, 1, 1, 144), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 24), t), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 24, 96, 98), Cnv(a0, 1, 1, 144), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 32), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 48, 49), Cnv(a0, 1, 1, 192), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 32), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 48, 48), Cnv(a0, 1, 1, 192), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 64), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 24, 26), Cnv(a0, 1, 1, 384), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 64), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 24, 25), Cnv(a0, 1, 1, 384), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 96), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 96, 24, 24), Cnv(a0, 1, 1, 576), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 96), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps * 2.0f, Param(Shp(1, 160, 12, 16), Cnv(a0, 1, 1, 960), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 320), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps * 2.0f, Param(Shp(1, 160, 12, 15), Cnv(a0, 1, 1, 960), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 160), t), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 384, 384), Cnv(a0, 3, 2, 35), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 17), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 17, 192, 192), Cnv(a0, 1, 1, 99), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 27), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 27, 96, 96), Cnv(a0, 1, 1, 147), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 27), f), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 34, 32, 32), Cnv(a0, 1, 1, 34), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 34), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 34, 32, 32), Cnv(a0, 1, 1, 34), Cnv(a1, 5, 1), Cnv(a2, 1, 1, 34), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 34, 32, 32), Cnv(a0, 1, 1, 34), Cnv(a1, 7, 1), Cnv(a2, 1, 1, 34), f), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 320, 320), Cnv(a0, 3, 2, 16), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 8), f), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 60, 80), Cnv(a0, 1, 1, 48), Cnv(a1, 3, 1)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 48, 70, 81), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 64)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 320, 320), Cnv(a0, 3, 2, 16), Cnv(a1, 3, 1)), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 320, 180), Cnv(a0, 3, 2, 16), Cnv(a1, 3, 1)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 16, 160, 90), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 2)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 80, 45), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 1)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 80, 45), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 2)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 40, 23), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 1)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 1)), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 128)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 30)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 12)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 6)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 128)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 256)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 20)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 8)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 4)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 256)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 20)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 8)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 4)), c, f1, f2);
#endif
#if 1
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 19, 63, 81), Cnv(a0, 1, 1, 51), Cnv(a1, 3, 2)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 1280, 12, 21), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 2)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 48, 70, 81), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 64)), c, f1, f2);
#endif
#if 0
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 64, 10, 6), Cnv(a0, 1, 1, 256), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 64), t), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), f), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 20)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 128)), c, f1, f2);
#endif
#if 1
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 728, 28, 28), Cnv(a0, 1, 1, 728), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 728), f), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 728, 14, 14), Cnv(a0, 1, 1, 728), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 728), f), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 728, 28, 28), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 728)), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 728, 14, 14), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 728)), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 24, 96, 99), Cnv(a0, 1, 1, 144), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 24), t), c, f1, f2);
#endif
#else
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 16, 192, 199), Cnv(a0, 1, 1, 96), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 24), f), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 16, 24, 48), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 32), f), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 32, 24, 32), Cnv(a0, 1, 1, 144), Cnv(a1, 3, 1)), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 24, 96, 99), Cnv(a0, 1, 1, 144), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 24), t), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 19, 63, 81), Cnv(a0, 1, 1, 51), Cnv(a1, 3, 2)), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 35, 63, 81), Cnv(a0, 1, 1, 51), Cnv(a1, 3, 2)), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps * 2.0f, Param(Shp(1, 160, 12, 15), Cnv(a0, 1, 1, 960), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 160), t), c, f1, f2);
        //result = result && SynetMergedConvolution32fForwardAutoTest(eps * 2.0f, Param(Shp(1, 64, 14, 14), Cnv(a0, 1, 1, 384), Cnv(a1, 3, 1), Cnv(a2, 1, 1, 64), t), c, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, Param(Shp(1, 3, 480, 640), Cnv(a0, 3, 2, 16), Cnv(a1, 3, 1)), c, f1, f2);
#endif
        return result;
    }

    bool SynetMergedConvolution32fForwardAutoTest(float eps, const FuncMC& f1, const FuncMC& f2)
    {
        bool result = true;

        SimdSynetCompatibilityType fp32 = SimdSynetCompatibilityDefault;
        SimdSynetCompatibilityType bf16 = (SimdSynetCompatibilityType)(SimdSynetCompatibility16bfSoft | SimdSynetCompatibilityFmaAvoid);

#if defined(NDEBUG)
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, fp32, f1, f2);
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, bf16, f1, f2);
#else
        result = result && SynetMergedConvolution32fForwardAutoTest(eps, bf16, f1, f2);
#endif

        return result;
    }

    bool SynetMergedConvolution32fForwardAutoTest()
    {
        bool result = true;

        result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Base::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Sse41::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx2::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx512bw::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#if defined(SIMD_AVX512BF16_ENABLE) && !defined(SIMD_AMX_EMULATE)
        if (Simd::Avx512bf16::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Avx512bf16::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#if defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
        if (Simd::Amx::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Amx::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetMergedConvolution32fForwardAutoTest(EPS, FUNC_MC(Simd::Neon::SynetMergedConvolution32fInit), FUNC_MC(SimdSynetMergedConvolution32fInit));
#endif 

        return result;
    }
#endif
}
