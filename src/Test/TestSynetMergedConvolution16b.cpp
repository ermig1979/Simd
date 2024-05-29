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
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynetMergedConvolution16b.h"
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
            size_t batch, count;
            SimdSynetCompatibilityType comp;
            SimdConvolutionParameters conv[3];
            mutable float *weight[3], *bias[3], *params[3];

            Param(const Shape & in, const Cnv & c0, const Cnv& c1, const Cnv& c2, SimdTensorDataType s, SimdTensorDataType d, SimdSynetCompatibilityType c)
            {
                count = 3;
                batch = in[0];
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
                SetConv(conv + 2, c2);
                conv[0].srcT = s;
                conv[2].dstT = d;
                comp = c;
            }

            Param(const Shape& in, const Cnv& c0, const Cnv& c1, SimdTensorDataType s, SimdTensorDataType d, SimdSynetCompatibilityType c)
            {
                count = 2;
                batch = in[0];
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
                conv[0].srcT = s;
                conv[1].dstT = d;
                comp = c;
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
            typedef void*(*FuncPtr)(size_t batch, const SimdConvolutionParameters * params, size_t count, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncMC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Param & p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.count << ":" << p.batch << "x" << p.conv[0].srcC << "x" << p.conv[0].srcH << "x" << p.conv[0].srcW;
                for (size_t i = 0; i < p.count; ++i)
                    ss << "-" << (p.conv[i].group != 1 ? String("") : ToString(p.conv[i].dstC) + "x") << p.conv[i].kernelY << "x" << p.conv[i].strideY;
                ss << "-" << (p.conv[0].srcT == SimdTensorData32f ? "f" : "b") << (p.conv[p.count - 1].dstT == SimdTensorData32f ? "f" : "b") << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t* src, uint8_t * buf, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetMergedConvolution16bForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    bool SynetMergedConvolution16bForwardAutoTest(float eps, Param p, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test [" << f1.desc << " & " << f2.desc << "].");

        SimdConvolutionParameters & beg = p.conv[0];
        SimdConvolutionParameters & end = p.conv[p.count - 1];

        Shape srcS = Shp(p.batch, beg.srcH, beg.srcW, beg.srcC), dstS = Shp(p.batch, end.dstH, end.dstW, end.dstC);

        Tensor32f weight[3], bias[3], params[3], buf32f;
        Tensor32f src32f(srcS, beg.srcF), dst32f1(dstS, end.dstF), dst32f2(dstS, end.dstF), dst32f3(dstS, end.dstF);
        Tensor16u src16u(srcS, beg.srcF), dst16u1(dstS, end.dstF), dst16u2(dstS, end.dstF);
        Tensor8u buf8u;

        srand(0);
        FillRandom(src32f.Data(), src32f.Size(), -1.0, 1.0f);
        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16u.Data());

        for (size_t i = 0; i < p.count; ++i)
        {
            size_t dc = p.conv[i].dstC;
            weight[i].Reshape(Shp(p.conv[i].kernelY, p.conv[i].kernelX, p.conv[i].srcC / p.conv[i].group, dc));
            FillRandom(weight[i].Data(), weight[i].Size(), -1.0, 1.0f);
            p.weight[i] = weight[i].Data();

            bias[i].Reshape(Shp(dc));
            FillRandom(bias[i].Data(), bias[i].Size(), -1.0, 1.0f);
            p.bias[i] = bias[i].Data();

            params[i].Reshape(Shp(Simd::Max<size_t>(2, dc)));
            FillRandom(params[i].Data(), params[i].Size(), 0.0, 2.0f);
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

        Fill(dst32f1, 1.1f);
        Fill(dst32f2, 2.2f);
        SimdFloat32ToBFloat16(dst32f1.Data(), dst32f1.Size(), dst16u1.Data());
        SimdFloat32ToBFloat16(dst32f2.Data(), dst32f2.Size(), dst16u2.Data());

        const uint8_t* src = beg.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : (uint8_t*)src16u.Data();
        uint8_t* dst1 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : (uint8_t*)dst16u1.Data();
        uint8_t* dst2 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : (uint8_t*)dst16u2.Data();

        void* context1 = f1.func(p.batch, p.conv, p.count, p.comp);
        void* context2 = f2.func(p.batch, p.conv, p.count, p.comp);

        if (context1 == NULL || context2 == NULL)
            return result;

        buf8u.Extend({ ::SimdSynetMergedConvolution16bExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetMergedConvolution16bExternalBufferSize(context2) });

        ::SimdSynetMergedConvolution16bSetParams(context1, p.weight, NULL, p.bias, p.params);
        ::SimdSynetMergedConvolution16bSetParams(context2, p.weight, NULL, p.bias, p.params);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        if (end.dstT == SimdTensorData16b)
            eps = eps * 8.0f;
        else
            eps = eps * 2.0f;

        if (end.dstT == SimdTensorData16b)
        {
            SimdBFloat16ToFloat32(dst16u1.Data(), dst16u1.Size(), dst32f1.Data());
            SimdBFloat16ToFloat32(dst16u2.Data(), dst16u2.Size(), dst32f2.Data());
        }
        result = result && Compare(dst32f1, dst32f2, eps, true, 64, DifferenceBoth);

        if (1)
        {
            beg.srcT = SimdTensorData32f;
            end.dstT = SimdTensorData32f;

            void* context3 = SimdSynetMergedConvolution32fInit(p.batch, p.conv, p.count, SimdFalse, SimdSynetCompatibilityDefault);

            Tensor32f dst32f3(dstS, end.dstF);
            Fill(dst32f3, 3.3f);
            Tensor32f buf32f(Shp(::SimdSynetMergedConvolution32fExternalBufferSize(context3)));

            ::SimdSynetMergedConvolution32fSetParams(context3, p.weight, NULL, p.bias, p.params);

            ::SimdSynetMergedConvolution32fForward(context3, src32f.Data(), buf32f.Data(), dst32f3.Data());

            ::SimdRelease(context3);

            result = result && Compare(dst32f1, dst32f3, 0.07, true, 64, DifferenceBoth, " Compare to SynetMergedConvolution32f.");//0.129
        }

        return result;
    }

    bool SynetMergedConvolution16bForwardAutoTest(float eps, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
        SimdSynetCompatibilityType c = (SimdSynetCompatibilityType)(SimdSynetCompatibility16bfSoft | SimdSynetCompatibilityFmaUse);
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
        const SimdConvolutionActivationType a0 = aSw, a1 = aSw, a2 = aSw;
#if defined(NDEBUG)
#if 1
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 555, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 1555), f32, f32, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 8, 6), Cnv(a0, 1, 1, 1548), Cnv(a1, 3, 1), f32, f32, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), f32, f32, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 128), b16, b16, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 8, 6), Cnv(a0, 1, 1, 1548), Cnv(a1, 3, 1), b16, b16, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), b16, b16, c), f1, f2);
#endif
#else
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 1024, 8, 6), Cnv(a0, 1, 1, 1548), Cnv(a1, 3, 1), b16, b16, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 64, 40, 23), Cnv(a0, 3, 2), Cnv(a1, 1, 1, 128), b16, b16, c), f1, f2);
        result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 256, 10, 6), Cnv(a0, 1, 1, 64), Cnv(a1, 3, 2), Cnv(a2, 1, 1, 256), b16, b16, c), f1, f2);
        //result = result && SynetMergedConvolution16bForwardAutoTest(eps, Param(Shp(1, 128, 20, 12), Cnv(a0, 3, 1), Cnv(a1, 1, 1, 128), b16, b16, c), f1, f2);
#endif
        return result;
    }

    bool SynetMergedConvolution16bForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Base::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Sse41::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Avx2::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::Avx512bw::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
#endif 

//#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))
//        if (Simd::AmxBf16::Enable && TestAmxBf16())
//            result = result && SynetMergedConvolution16bForwardAutoTest(EPS, FUNC_MC(Simd::AmxBf16::SynetMergedConvolution16bInit), FUNC_MC(SimdSynetMergedConvolution16bInit));
//#endif

        return result;
    }
#endif
}
