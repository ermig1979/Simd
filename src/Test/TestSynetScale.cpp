/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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

#include "Simd/SimdSynetScale8i.h"
#include "Simd/SimdSynetScale16b.h"
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

    bool SynetScaleLayerForwardAutoTest(const FuncScLF & f1, const FuncScLF & f2)
    {
        bool result = true;

        //result = result && SynetScaleLayerForwardAutoTest(16, 24, 24, SimdTensorFormatNhwc, 1, 0, f1, f2);

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
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

        //result = result && SynetScaleLayerForwardAutoTest(3, W*W, SimdTensorFormatNhwc, true, f1, f2);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SCLF(Simd::Base::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SCLF(Simd::Sse41::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SCLF(Simd::Avx2::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SCLF(Simd::Avx512bw::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SCLF(Simd::Neon::SynetScaleLayerForward), FUNC_SCLF(SimdSynetScaleLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

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
        SimdSynetCompatibilityType cP = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaUse);
        SimdSynetCompatibilityType cN = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);

#ifdef NDEBUG
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(2, 3, 30007, s, d, f, cN, 1, 1), f1, f2);
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(1, 255, 1005, s, d, f, cP, 1, 0), f1, f2);        
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(1, 65, 1603, s, d, f, cN, 0, 1), f1, f2);
#else
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(2, 3, 3007, s, d, f, cN, 1, 1), f1, f2);
        result = result && SynetScale8iForwardAutoTest(e, Scale8iParam(1, 25, 1005, s, d, f, cP, 1, 0), f1, f2);
#endif

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

        if (TestBase())
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Base::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Sse41::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Avx2::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetScale8iForwardAutoTest(FUNC_S8I(Simd::Avx512bw::SynetScale8iInit), FUNC_S8I(SimdSynetScale8iInit));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncS16b
        {
            typedef void* (*FuncPtr)(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias);

            FuncPtr func;
            String desc;

            FuncS16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t s, SimdTensorDataType st, SimdTensorDataType dt, SimdTensorFormatType f, SimdBool n, SimdBool b)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(s) + "-" + ToChar(st) + ToChar(dt) + 
                    (f == SimdTensorFormatNhwc ? "1-" : "0-") + (n ? "1" : "0") + (b ? "1" : "0") + "]";
            }

            void Call(void* context, const uint8_t* src, const float * norm, const float* bias, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetScale16bForward(context, src, norm, bias, dst);
            }
        };
    }

#define FUNC_S16B(function) FuncS16b(function, #function)

    bool SynetScale16bAutoTest(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias, FuncS16b f1, FuncS16b f2)
    {
        bool result = true;

        f1.Update(channels, spatial, srcType, dstType, format, norm, bias);
        f2.Update(channels, spatial, srcType, dstType, format, norm, bias);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Shape shape = ToShape(channels, spatial, format);
        Tensor32f src32f(shape), dst32f1(shape), dst32f2(shape), norm32f(Shp(channels)), bias32f(Shp(channels));
        Tensor16u src16b(shape), dst16b1(shape), dst16b2(shape);

        srand(0);
        FillRandom(src32f.Data(), src32f.Size(), -1.0, 1.0f);
        FillRandom(norm32f.Data(), norm32f.Size(), -1.0, 1.0f);
        FillRandom(bias32f.Data(), bias32f.Size(), -1.0, 1.0f);

        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16b.Data());

        Fill(dst32f1, 1.0f);
        Fill(dst32f2, 2.0f);

        Fill(dst16b1.Data(), dst16b1.Size(), uint16_t(1));
        Fill(dst16b2.Data(), dst16b2.Size(), uint16_t(2));

        const uint8_t* src = srcType == SimdTensorData32f ? (uint8_t*)src32f.Data() : (uint8_t*)src16b.Data();
        const float* pn = norm ? norm32f.Data() : NULL;
        const float* pb = bias ? bias32f.Data() : NULL;
        uint8_t* dst1 = dstType == SimdTensorData32f ? (uint8_t*)dst32f1.Data() : (uint8_t*)dst16b1.Data();
        uint8_t* dst2 = dstType == SimdTensorData32f ? (uint8_t*)dst32f2.Data() : (uint8_t*)dst16b2.Data();

        void* context1 = f1.func(channels, spatial, srcType, dstType, format, norm, bias);
        void* context2 = f2.func(channels, spatial, srcType, dstType, format, norm, bias);

        if (context1 == NULL)
            return true;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, pn, pb, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, pn, pb, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        float eps = EPS;
        if (dstType == SimdTensorData16b)
        {
            eps = eps * 7.8f;
            SimdBFloat16ToFloat32(dst16b1.Data(), dst16b1.Size(), dst32f1.Data());
            SimdBFloat16ToFloat32(dst16b2.Data(), dst16b2.Size(), dst32f2.Data());
        }
        result = result && Compare(dst32f1, dst32f2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetScale16bAutoTest(const FuncS16b& f1, const FuncS16b& f2)
    {
        bool result = true;

        const SimdTensorFormatType nchw = SimdTensorFormatNchw, nhwc = SimdTensorFormatNhwc;
        const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
        const SimdBool t = SimdTrue, f = SimdFalse;

#if 1
        result = result && SynetScale16bAutoTest(224, 144, f32, f32, nhwc, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, f32, b16, nhwc, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, b16, f32, nhwc, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, b16, b16, nhwc, t, t, f1, f2);

        result = result && SynetScale16bAutoTest(224, 144, f32, f32, nchw, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, f32, b16, nchw, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, b16, f32, nchw, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(224, 144, b16, b16, nchw, t, t, f1, f2);

        result = result && SynetScale16bAutoTest(333, 555, b16, b16, nhwc, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(333, 443, f32, b16, nchw, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(333, 443, b16, b16, nchw, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(333, 443, f32, b16, nhwc, t, t, f1, f2);
        result = result && SynetScale16bAutoTest(333, 225, b16, f32, nhwc, t, f, f1, f2);
        result = result && SynetScale16bAutoTest(333, 225, b16, f32, nhwc, f, t, f1, f2);
#endif

        return result;
    }

    bool SynetScale16bAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetScale16bAutoTest(FUNC_S16B(Simd::Base::SynetScale16bInit), FUNC_S16B(SimdSynetScale16bInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetScale16bAutoTest(FUNC_S16B(Simd::Sse41::SynetScale16bInit), FUNC_S16B(SimdSynetScale16bInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetScale16bAutoTest(FUNC_S16B(Simd::Avx2::SynetScale16bInit), FUNC_S16B(SimdSynetScale16bInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetScale16bAutoTest(FUNC_S16B(Simd::Avx512bw::SynetScale16bInit), FUNC_S16B(SimdSynetScale16bInit));
#endif

        return result;
    }
#endif
}
