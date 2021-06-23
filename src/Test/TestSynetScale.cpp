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
#include "Test/TestString.h"

#include "Simd/SimdSynetScale8i.h"
#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncScLF
        {
            typedef void(*FuncPtr)(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncScLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format, int bias, int comp)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(bias) + "-" + ToString(comp) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & scale, const Tensor32f & bias, size_t channels, size_t height, size_t width, SimdTensorFormatType format, int comp, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), scale.Data(), bias.Data(), channels, height, width, dst.Data(), format, (SimdSynetCompatibilityType)comp);
            }
        };
    }

#define FUNC_SCLF(function) FuncScLF(function, #function)

    bool SynetScaleLayerForwardAutoTest(size_t channels, size_t height, size_t width, SimdTensorFormatType format, int hasBias, int comp, FuncScLF f1, FuncScLF f2)
    {
        bool result = true;

        f1.Update(format, hasBias, comp);
        f2.Update(format, hasBias, comp);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << height << ", " << width << "].");

        Tensor32f src(ToShape(channels, height, width, format));
        Tensor32f scale(ToShape(channels, format));
        Tensor32f bias;
        Tensor32f dst1(ToShape(channels, height, width, format));
        Tensor32f dst2(ToShape(channels, height, width, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        if (hasBias)
        {
            bias.Reshape(ToShape(channels, format));
            FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        }
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, channels, height, width, format, comp, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, channels, height, width, format, comp, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest(int mask, const FuncScLF & f1, const FuncScLF & f2)
    {
        bool result = true;

        //result = result && SynetScaleLayerForwardAutoTest(16, 24, 24, SimdTensorFormatNhwc, 1, 0, f1, f2);

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                for (int hasBias = 0; hasBias <= 1; ++hasBias)
                {
                    for (int comp = 0; comp <= 2; ++comp)
                    {
                        result = result && SynetScaleLayerForwardAutoTest(C, (int)sqrt(H), (int)sqrt(W), format, hasBias, comp, f1, f2);
                        result = result && SynetScaleLayerForwardAutoTest(C - O, (int)sqrt(H) + O/2, (int)sqrt(W) + O/2, format, hasBias, comp, f1, f2);
                    }
                }
            }
        }
        //result = result && SynetScaleLayerForwardAutoTest(3, W*W, SimdTensorFormatNhwc, true, f1, f2);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(TFM_ANY, FUNC_SCLF(Simd::Base::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_128, FUNC_SCLF(Simd::Sse2::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_256, FUNC_SCLF(Simd::Avx::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_256, FUNC_SCLF(Simd::Avx2::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_512, FUNC_SCLF(Simd::Avx512f::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_128, FUNC_SCLF(Simd::Neon::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    struct Scale8iParam : public Simd::Base::Scale8iParam
    {
        int neg, bias;

        Scale8iParam(size_t ba, size_t ch, size_t sp, SimdTensorDataType st, SimdTensorDataType dt, SimdTensorFormatType f, SimdSynetCompatibilityType co, int n, int bi)
            : Simd::Base::Scale8iParam(ba, ch, sp, st, dt, f, co), neg(n), bias(bi) { }

        Shape SrcShape() const
        {
            return format == SimdTensorFormatNhwc ? Shape({ batch, spatial, channels }) : Shape({ batch, channels, spatial });
        }
    };

    namespace
    {
        struct FuncS8i
        {
            typedef void* (*FuncPtr)(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, 
                SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncS8i(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Scale8iParam & p)
            {
                desc = desc + "[" + ToString(p.batch) + "x" + ToString(p.channels) + "x" + ToString(p.spatial) + 
                    (p.srcType == SimdTensorData8u ? "-u" : "-f") + (p.dstType == SimdTensorData8u ? "-u" : "-f") +
                    (p.format == SimdTensorFormatNhwc ? "-1" : "-0") + (Simd::Base::Narrowed(p.compatibility) ? "-n" : "-p") + "]";
            }

            void Call(void* context, const uint8_t* src, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetScale8iForward(context, src, dst);
            }
        };
    }

#define FUNC_S8I(function) \
    FuncS8i(function, std::string(#function))

    void FillDstStat(const Scale8iParam& p, const Tensor32f& scale, const Tensor32f& bias, const Tensor32f& src, Tensor32f& dst, float* dstMin, float* dstMax, float* dstScale, float* dstShift)
    {
        SimdSynetScaleLayerForward(src.Data(), scale.Data(), bias.Data(), p.channels, 1, p.spatial, dst.Data(), p.format, p.compatibility);
        SetDstStat(p.channels, p.neg, p.compatibility, dst, dstMin, dstMax, dstScale, dstShift);
    }

    bool SynetScale8iForwardAutoTest(float eps, const Scale8iParam& p, FuncS8i f1, FuncS8i f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor32f scale({ p.channels });
        FillRandom(scale.Data(), scale.Size(), -1.0, 1.0f);

        Tensor32f bias;
        if (p.bias)
        {
            bias.Reshape({ p.channels });
            FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);
        }

        Tensor32f srcMin({ p.channels }), srcMax({ p.channels }), dstMin({ p.channels }), dstMax({ p.channels });
        Tensor32f srcScale({ p.channels }), srcShift({ p.channels }), dstScale({ p.channels }), dstShift({ p.channels });
        Tensor32f src32f(p.SrcShape(), p.format), dst32f1(p.SrcShape(), p.format), dst32f2(p.SrcShape(), p.format);
        Tensor8u src8u(p.SrcShape(), p.format), dst8u1(p.SrcShape(), p.format), dst8u2(p.SrcShape(), p.format);

        FillRandom(src32f, srcMin.Data(), srcMax.Data(), p.channels, p.neg);
        SetSrc32fTo8u(src32f, srcMin.Data(), srcMax.Data(), p.channels, p.neg, p.compatibility, srcScale.Data(), srcShift.Data(), src8u);
        FillDstStat(p, scale, bias, src32f, dst32f1, dstMin.Data(), dstMax.Data(), dstScale.Data(), dstShift.Data());

        const float* stats[4] = { srcMin.Data(), srcMax.Data(), dstMin.Data(), dstMax.Data() };
        const uint8_t* src = p.srcType == SimdTensorData32f ? (uint8_t*)src32f.Data() : src8u.Data();
        uint8_t* dst1 = p.dstType == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : dst8u1.Data();
        uint8_t* dst2 = p.dstType == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : dst8u2.Data();

        Fill(dst32f1, 0.1f);
        Fill(dst32f2, 1.1f);

        Fill(dst8u1, uint8_t(1));
        Fill(dst8u2, uint8_t(2));

        void* context1 = f1.func(p.batch, p.channels, p.spatial, p.srcType, p.dstType, p.format, p.compatibility);
        void* context2 = f2.func(p.batch, p.channels, p.spatial, p.srcType, p.dstType, p.format, p.compatibility);

        ::SimdSynetScale8iSetParams(context1, scale.Data(), bias.Data(), stats);
        ::SimdSynetScale8iSetParams(context2, scale.Data(), bias.Data(), stats);

        ::SimdSynetScale8iSetParams(context1, scale.Data(), bias.Data(), NULL);
        ::SimdSynetScale8iSetParams(context2, scale.Data(), bias.Data(), NULL);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (Simd::Base::FmaAvoid(p.compatibility) ? 0 : 1);
#else
        int differenceMax = 1;
#endif

        if (p.dstType == SimdTensorData32f)
            result = result && Compare(dst32f1, dst32f2, eps * eps, true, 64, DifferenceBoth);
        else
            result = result && Compare(dst8u1, dst8u2, differenceMax, true, 64);

        return result;
    }

    bool SynetScale8iForwardAutoTest(const FuncS8i& f1, const FuncS8i& f2, SimdTensorDataType s, SimdTensorDataType d, SimdTensorFormatType f)
    {
        bool result = true;

        const float e = EPS;
        SimdSynetCompatibilityType cP = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType cN = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaAvoid);

        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(2, 3, 90007, s, d, f, cN, 1, 1), f1, f2);
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(1, 255, 10005, s, d, f, cP, 1, 0), f1, f2);
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(1, 65, 1603, s, d, f, cN, 0, 1), f1, f2);

        return result;
    }

    bool SynetScale8iForwardAutoTest(const FuncS8i& f1, const FuncS8i& f2)
    {
        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdTensorFormatType nchw = SimdTensorFormatNchw, nhwc = SimdTensorFormatNhwc;

        bool result = true;

        result = result && SynetScale8iForwardAutoTest(f1, f2, u8, u8, nchw);
        result = result && SynetScale8iForwardAutoTest(f1, f2, u8, f32, nchw);
        result = result && SynetScale8iForwardAutoTest(f1, f2, f32, u8, nchw);
        result = result && SynetScale8iForwardAutoTest(f1, f2, f32, f32, nchw);

        result = result && SynetScale8iForwardAutoTest(f1, f2, u8, u8, nhwc);
        result = result && SynetScale8iForwardAutoTest(f1, f2, u8, f32, nhwc);
        result = result && SynetScale8iForwardAutoTest(f1, f2, f32, u8, nhwc);
        result = result && SynetScale8iForwardAutoTest(f1, f2, f32, f32, nhwc);

        return result;
    }

    bool SynetScale8iForwardAutoTest()
    {
        bool result = true;

        result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Base::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Sse41::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Avx2::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Avx512bw::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif

        return result;
    }
#endif
}
