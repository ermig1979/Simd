/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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

#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynet.h"

namespace Test
{
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
            int neg;
            SimdSynetCompatibilityType comp;
            SimdConvolutionParameters conv[3];
            mutable float *weight[3], *bias[3], *params[3], *stats[6];

            Param(const Shape & in, const Cnv & c0, const Cnv& c1, const Cnv& c2, SimdTensorDataType s, SimdTensorDataType d, int n, SimdSynetCompatibilityType c)
            {
                count = 3;
                batch = in[0];
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
                SetConv(conv + 2, c2);
                conv[0].srcT = s;
                conv[2].dstT = d;
                neg = n;
                comp = c;
            }

            Param(const Shape& in, const Cnv& c0, const Cnv& c1, SimdTensorDataType s, SimdTensorDataType d, int n, SimdSynetCompatibilityType c)
            {
                count = 2;
                batch = in[0];
                SetConv(conv + 0, c0, in);
                SetConv(conv + 1, c1);
                conv[0].srcT = s;
                conv[2].dstT = d;
                neg = n;
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
                ss << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t* src, uint8_t * buf, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetConvolution8iForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_MC(function) \
    FuncMC(function, std::string(#function))

    void FillDstStat(const Param& p, size_t i, const Tensor32f& weight, const Tensor32f& bias, const Tensor32f& params, 
        const Tensor32f& src, Tensor32f& buf, Tensor32f& dst, float* min, float* max)
    {
        SimdConvolutionParameters conv = p.conv[i];
        conv.srcT = SimdTensorData32f;
        conv.dstT = SimdTensorData32f;
        void* context = SimdSynetConvolution32fInit(p.batch, &conv, NULL);
        buf.Extend({ SimdSynetConvolution32fExternalBufferSize(context) });
        dst.Reshape(Shp(p.batch, conv.dstH, conv.dstW, conv.dstC));
        SimdSynetConvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());
        SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
        SimdRelease(context);
        SetDstStat(conv.dstC, p.neg, p.comp, dst, min, max, NULL, NULL);
    }

    bool SynetMergedConvolution8iForwardAutoTest(float eps, const Param & p, FuncMC f1, FuncMC f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters & beg = p.conv[0];
        const SimdConvolutionParameters & end = p.conv[p.count - 1];

        Tensor32f weight[3], bias[3], params[3], min[4], max[4], tmp[3];
        Tensor32f src32f(Shp(p.batch, beg.srcH, beg.srcW, beg.srcC), beg.srcF), buf32f;
        Tensor32f dst32f1(Shp(p.batch, end.dstH, end.dstW, end.dstC), end.dstF), dst32f2(dst32f1.Shape(), dst32f1.Format());
        Tensor8u src8u(src32f.Shape(), src32f.Format()), dst8u1(dst32f1.Shape(), dst32f1.Format()), dst8u2(dst32f1.Shape(), dst32f1.Format()), buf8u;

        min[0].Reshape(Shp(beg.srcC)), p.stats[0] = min[0].Data();
        max[0].Reshape(Shp(beg.srcC)), p.stats[1] = max[0].Data();
        FillRandom(src32f, min[0].Data(), max[0].Data(), beg.srcC, p.neg);
        SetSrc32fTo8u(src32f, min[0].Data(), max[0].Data(), beg.srcC, p.neg, p.comp, NULL, NULL, src8u);
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
            FillRandom(params[i].Data(), params[i].Size(), -3.0, 3.0f);
            if (p.conv[i].activation == ::SimdConvolutionActivationHswish)
            {
                params[i].Data()[0] = 3.0f;
                params[i].Data()[1] = 1.0f / 6.0f;
            }
            else
            {
                params[i].Data()[0] = 0.0f + 0.1f * float(i);
                params[i].Data()[1] = 1.0f + 0.1f * float(i);
            }
            p.params[i] = params[i].Data();

            min[i + 1].Reshape(Shp(dc)), p.stats[i*2 + 2] = min[i + 1].Data();
            max[i + 1].Reshape(Shp(dc)), p.stats[i*2 + 3] = max[i + 1].Data();
            FillDstStat(p, i, weight[i], bias[i], params[i], i ? tmp[i - 1] : src32f, buf32f, tmp[i], min[i + 1].Data(), max[i + 1].Data());
        }

        Fill(dst32f1, 1.0f);
        Fill(dst32f2, 2.0f);

        Fill(dst8u1, uint8_t(1));
        Fill(dst8u2, uint8_t(2));

        const uint8_t* src = beg.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : src8u.Data();
        uint8_t* dst1 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : dst8u1.Data();
        uint8_t* dst2 = end.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : dst8u2.Data();

        void* context1 = f1.func(p.batch, p.conv, p.count, p.comp);
        void* context2 = f2.func(p.batch, p.conv, p.count, p.comp);

        buf8u.Extend({ ::SimdSynetMergedConvolution8iExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetMergedConvolution8iExternalBufferSize(context2) });

        ::SimdSynetMergedConvolution8iSetParams(context1, p.weight, NULL, p.bias, p.params, p.stats);
        ::SimdSynetMergedConvolution8iSetParams(context2, p.weight, NULL, p.bias, p.params, p.stats);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (Simd::Base::FmaAvoid(p.comp) ? 0 : 1);
#else
        int differenceMax = 1;
#endif
        if (end.dstT == SimdTensorData32f)
            result = result && Compare(dst32f1, dst32f2, eps * eps, true, 64, DifferenceBoth);
        else
            result = result && Compare(dst8u1, dst8u2, differenceMax, true, 64);

        return result;
    }

    bool SynetMergedConvolution8iForwardAutoTest(float eps, const FuncMC & f1, const FuncMC & f2)
    {
        bool result = true;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        SimdSynetCompatibilityType p = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType o = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iOverflow | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType n = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaAvoid);
        const ::SimdConvolutionActivationType a0 = ::SimdConvolutionActivationPrelu, a1 = ::SimdConvolutionActivationHswish, a2 = ::SimdConvolutionActivationIdentity;
#ifdef NDEBUG
#if 1
        result = result && SynetMergedConvolution8iForwardAutoTest(eps, Param(Shp(1, 32, 80, 45), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 2), u8, u8, 1, n), f1, f2);
#endif
#else
        result = result && SynetMergedConvolution8iForwardAutoTest(eps, Param(Shp(1, 32, 80, 45), Cnv(a0, 1, 1, 32), Cnv(a1, 3, 2), u8, u8, 1, n), f1, f2);
#endif
        return result;
    }

    bool SynetMergedConvolution8iForwardAutoTest()
    {
        bool result = true;

        result = result && SynetMergedConvolution8iForwardAutoTest(EPS, FUNC_MC(Simd::Base::SynetMergedConvolution8iInit), FUNC_MC(SimdSynetMergedConvolution8iInit));

        return result;
    }
}
