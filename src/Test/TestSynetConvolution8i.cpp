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
#include "Test/TestSynetConvolutionParam.h"

#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynet.h"

namespace Test
{
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
                desc = desc + p.Decription(Simd::Base::Overflow(c) ? "-o" : Simd::Base::Narrowed(c) ? "-n" : "-p");
            }

            void Call(void * context, const uint8_t * src, uint8_t * buf, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetConvolution8iForward(context, src, buf, dst);
            }
        };
    }

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    void FillSrc8u(const Param& p, int neg, SimdSynetCompatibilityType comp, const Tensor32f& src, const float* min, const float* max, Tensor8u & dst)
    {
        int uMin = Simd::Base::Narrowed(comp) ? Simd::Base::U8_NARROWED_MIN : Simd::Base::U8_PRECISE_MIN;
        int uMax = Simd::Base::Narrowed(comp) ? Simd::Base::U8_NARROWED_MAX : Simd::Base::U8_PRECISE_MAX;
        int iMin = Simd::Base::Narrowed(comp) ? Simd::Base::I8_NARROWED_MIN : Simd::Base::I8_PRECISE_MIN;
        int iMax = Simd::Base::Narrowed(comp) ? Simd::Base::I8_NARROWED_MAX : Simd::Base::I8_PRECISE_MAX;
        Buffer32f scale(p.conv.srcC), shift(p.conv.srcC);
        for (size_t i = 0; i < p.conv.srcC; ++i)
        {
            float abs = std::max(Simd::Abs(min[i]), Simd::Abs(max[i]));
            scale[i] = (neg ? iMax : uMax) / abs;
            shift[i] = float(neg ? -iMin : uMin);// -min[i] * scale[i];
        }
        for (size_t b = 0; b < p.batch; ++b)
        {
            if (p.trans)
            {
                for (size_t y = 0; y < p.conv.srcH; ++y)
                    for (size_t x = 0; x < p.conv.srcW; ++x)
                        for (size_t c = 0; c < p.conv.srcC; ++c)
                            dst.Data({ b, y, x, c })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, y, x, c })[0], scale[c], shift[c], uMin, uMax);

            }
            else
            {
                for (size_t c = 0; c < p.conv.srcC; ++c)
                    for (size_t y = 0; y < p.conv.srcH; ++y)
                        for (size_t x = 0; x < p.conv.srcW; ++x)
                            dst.Data({ b, c, y, x })[0] = Simd::Base::SynetConvert32fTo8u(src.Data({ b, c, y, x })[0], scale[c], shift[c], uMin, uMax);
            }
        }
    }

    void FillDstStat(Param p, const Tensor32f& weight, const Tensor32f & bias, const Tensor32f& params, 
        const Tensor32f & src, Tensor32f& buf, Tensor32f & dst, float* min, float* max)
    {
        p.conv.srcT = SimdTensorData32f;
        p.conv.dstT = SimdTensorData32f;
        void * context = SimdSynetConvolution32fInit(p.batch, &p.conv, NULL);
        buf.Extend({ SimdSynetConvolution32fExternalBufferSize(context) });
        SimdSynetConvolution32fSetParams(context, weight.Data(), NULL, bias.Data(), params.Data());
        SimdSynetConvolution32fForward(context, src.Data(), buf.Data(), dst.Data());
        SimdRelease(context);
        Fill(min, p.conv.dstC, FLT_MAX);
        Fill(max, p.conv.dstC, -FLT_MAX);
        for (size_t b = 0; b < p.batch; ++b)
        {
            if (p.trans)
            {
                for (size_t y = 0; y < p.conv.dstH; ++y)
                    for (size_t x = 0; x < p.conv.dstW; ++x)
                        for (size_t c = 0; c < p.conv.dstC; ++c)
                        {
                            min[c] = std::min(min[c], dst.Data({ b, y, x, c })[0]);
                            max[c] = std::max(max[c], dst.Data({ b, y, x, c })[0]);
                        }
            }
            else
            {
                for (size_t c = 0; c < p.conv.dstC; ++c)
                    for (size_t y = 0; y < p.conv.dstH; ++y)
                        for (size_t x = 0; x < p.conv.dstW; ++x)
                        {
                            min[c] = std::min(min[c], dst.Data({ b, c, y, x })[0]);
                            max[c] = std::max(max[c], dst.Data({ b, c, y, x })[0]);
                        }
            }
        }
    }

    bool SynetConvolution8iForwardAutoTest(float eps, Param p, int neg, SimdSynetCompatibilityType comp, FuncC f1, FuncC f2)
    {
        bool result = true;

        f1.Update(p, comp);
        f2.Update(p, comp);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        const SimdConvolutionParameters & c = p.conv;

        Tensor32f weight(p.WeightShape());
        FillRandom(weight.Data(), weight.Size(), -1.0, 1.0f);

        Tensor32f bias({ c.dstC });
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        Tensor32f params({ c.dstC });
        FillRandom(params.Data(), params.Size(), -3.0f, 3.0f);
        if (p.conv.activation == ::SimdConvolutionActivationHswish)
        {
            params.Data()[0] = 3.0f;
            params.Data()[1] = 1.0f / 6.0f;
        }
        else
        {
            params.Data()[0] = 0.1f;
            params.Data()[1] = 1.1f;
        }

        Tensor32f srcMin({ c.srcC }), srcMax({ c.srcC }), dstMin({ c.dstC }), dstMax({ c.dstC });
        Tensor32f src32f(p.SrcShape(), p.conv.srcF), dst32f1(p.DstShape(), p.conv.dstF), dst32f2(p.DstShape(), p.conv.dstF), buf32f;
        Tensor8u src8u(p.SrcShape(), p.conv.srcF), dst8u1(p.DstShape(), p.conv.dstF), dst8u2(p.DstShape(), p.conv.dstF), buf8u;
        //dst8u2.Reshape({ 1000000 }); dst8u2.Extend(p.DstShape());

        FillRandom(src32f, srcMin.Data(), srcMax.Data(), p.conv.srcC, neg);
        FillSrc8u(p, neg, comp, src32f, srcMin.Data(), srcMax.Data(), src8u);
        FillDstStat(p, weight, bias, params, src32f, buf32f, dst32f1, dstMin.Data(), dstMax.Data());

        const float* stats[4] = { srcMin.Data(), srcMax.Data(), dstMin.Data(), dstMax.Data() };
        const uint8_t * src = p.conv.srcT == SimdTensorData32f ? (uint8_t*)src32f.Data() : src8u.Data();
        uint8_t* dst1 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : dst8u1.Data();
        uint8_t* dst2 = p.conv.dstT == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : dst8u2.Data();

        Fill(dst32f1, 0.1f);
        Fill(dst32f2, 1.1f);

        Fill(dst8u1, uint8_t(1));
        Fill(dst8u2, uint8_t(2));

        void * context1 = f1.func(p.batch, &p.conv, comp);
        void * context2 = f2.func(p.batch, &p.conv, comp);

        buf8u.Extend({ ::SimdSynetConvolution8iExternalBufferSize(context1) });
        buf8u.Extend({ ::SimdSynetConvolution8iExternalBufferSize(context2) });

        ::SimdSynetConvolution8iSetParams(context1, weight.Data(), bias.Data(), params.Data(), stats);
        ::SimdSynetConvolution8iSetParams(context2, weight.Data(), bias.Data(), params.Data(), stats);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, buf8u.Data(), dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, buf8u.Data(), dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (Simd::Base::FmaAvoid(comp) ? 0 : 1);
#else
        int differenceMax = 1;
#endif

        if(p.conv.dstT == SimdTensorData32f)
            result = result && Compare(dst32f1, dst32f2, eps*eps, true, 64, DifferenceBoth);
        else
            result = result && Compare(dst8u1, dst8u2, differenceMax, true, 64);

        return result;
    }

    bool SynetConvolution8iForwardAutoTest(const FuncC& f1, const FuncC& f2, SimdSynetCompatibilityType c)
    {
        bool result = true;

        const Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _7(7, 7);
        const float e = EPS;
        const SimdBool t0 = SimdFalse, t1 = SimdTrue;
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu, aPr = SimdConvolutionActivationPrelu;
        //SimdSynetCompatibilityType c = (SimdSynetCompatibilityType)((SimdCpuInfo(SimdCpuInfoAvx512vnni) ? SimdSynetCompatibilityFmaUse : SimdSynetCompatibility8iOverflow)  | SimdSynetCompatibilityFmaAvoid);

#ifdef NDEBUG
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _7, _1, _2, _3, _3, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _5, _2, _3, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 32, 150, 150, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, f32), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 32, 150, 150, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _3, _1, _1, _1, _1, 1, aId, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 2000, 30, 30, 64, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 5000, 30, 30, 400, _1, _1, _1, _0, _0, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 145, 60, 75, 97, _3, _1, _1, _1, _1, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 128, 48, 48, 96, _3, _1, _1, _1, _1, 1, aRe, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 384, 19, 19, 36, _3, _1, _1, _1, _1, 1, aRe, t1, f32, u8), 0, c, f1, f2);
        //result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 40, 128, 128, 128, _5, _1, _1, _0, _0, 1, aRe, t1, u8, u8), 0, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 2000, 2000, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
        //result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 3000, 3000, 32, _3, _1, _1, _1, _1, 1, aId, t1, f32, u8), 0, c, f1, f2);
#endif
#if 0
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 300, 300, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(10, 3, 300, 300, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
#endif
#if 1
        result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 64, 60, 60, 64, _1, _1, _1, _0, _0, 1, aId, t1, f32, u8), 0, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(2, 32, 150, 150, 32, _3, _1, _1, _1, _1, 1, aPr, t1, u8, u8), 1, c, f1, f2);
        result = result && SynetConvolution8iForwardAutoTest(e, Param(3, 3, 300, 300, 32, _5, _2, _3, _0, _0, 1, aRe, t1, u8, f32), 1, c, f1, f2);
#endif
#else
        result = result && SynetConvolution8iForwardAutoTest(e, Param(2, 32, 150, 150, 32, _3, _1, _1, _1, _1, 1, aPr, t1, u8, u8), 1, c, f1, f2);
        //result = result && SynetConvolution8iForwardAutoTest(e, Param(1, 3, 1000, 1000, 32, _1, _1, _1, _0, _0, 1, aId, t1, u8, u8), 0, c, f1, f2);
#endif

        return result;
    }

    bool SynetConvolution8iForwardAutoTest(const FuncC& f1, const FuncC& f2)
    {
        bool result = true;

        SimdSynetCompatibilityType p = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType o = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iOverflow | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType n = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaAvoid);

        result = result && SynetConvolution8iForwardAutoTest(f1, f2, p);
        result = result && SynetConvolution8iForwardAutoTest(f1, f2, o);
        result = result && SynetConvolution8iForwardAutoTest(f1, f2, n);

        return result;
    }

    bool SynetConvolution8iForwardAutoTest()
    {
        bool result = true;

        result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Base::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Sse41::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx2::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx512bw::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_AVX512VNNI_ENABLE
        if (Simd::Avx512vnni::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Avx512vnni::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvolution8iForwardAutoTest(FUNC_C(Simd::Neon::SynetConvolution8iInit), FUNC_C(SimdSynetConvolution8iInit));
#endif 

        return result;
    }
}
