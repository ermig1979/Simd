/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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

#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncAB
        {
            typedef void(*FuncPtr)(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncAB(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & bias, size_t channels, size_t spatial, SimdTensorFormatType format, const Tensor32f & dstSrc, Tensor32f & dstDst) const
            {
                Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(desc);
                func(bias.Data(), channels, spatial, dstDst.Data(), format);
            }
        };
    }

#define FUNC_AB(function) FuncAB(function, #function)

    bool SynetAddBiasAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncAB f1, FuncAB f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f bias( ToShape(channels, format) );
        Tensor32f dstSrc(ToShape(channels, spatial, format));
        Tensor32f dstDst1(ToShape(channels, spatial, format));
        Tensor32f dstDst2(ToShape(channels, spatial, format));

        TEST_ALIGN(SIMD_ALIGN);

        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        FillRandom(dstSrc.Data(), dstSrc.Size(), -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bias, channels, spatial, format, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bias, channels, spatial, format, dstSrc, dstDst2));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetAddBiasAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            result = result && SynetAddBiasAutoTest(H, W, format, f1, f2);
            result = result && SynetAddBiasAutoTest(H - O, W + O, format, f1, f2);
        }

        return result;
    }

    bool SynetAddBiasAutoTest()
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sse41::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx512bw::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Neon::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncA8I
        {
            typedef void(*FuncPtr)(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
                uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncA8I(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t b, size_t c, size_t s, SimdTensorFormatType f, SimdSynetCompatibilityType comp)
            {
                desc = desc + "[" + ToString(b) + "x" + ToString(c) + "x" + ToString(s) +
                    (f == SimdTensorFormatNhwc ? "-1" : "-0") + (Simd::Base::Narrowed(comp) ? "-n" : "-p") + "]";
            }

            void Call(const Tensor8u& aData, const Tensor32f & aScale, const Tensor32f& aShift,
                const Tensor8u& bData, const Tensor32f& bScale, const Tensor32f& bShift, 
                Tensor8u& cData, const Tensor32f& cScale, const Tensor32f& cShift, SimdSynetCompatibilityType comp) const
            {
                TEST_PERFORMANCE_TEST(desc);
                size_t channels = aData.Format() == SimdTensorFormatNhwc ? aData.Axis(2) : aData.Axis(1);
                size_t spatial = aData.Format() == SimdTensorFormatNhwc ? aData.Axis(1) : aData.Axis(2);
                func(aData.Data(), aScale.Data(), aShift.Data(), bData.Data(), bScale.Data(), bShift.Data(), cData.Data(), cScale.Data(), cShift.Data(),
                    aData.Axis(0), channels, spatial, aData.Format(), comp);
            }
        };
    }

#define FUNC_A8I(function) FuncA8I(function, #function)

    bool SynetAdd8iAutoTest(size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, int negative, SimdSynetCompatibilityType compatibility, FuncA8I f1, FuncA8I f2)
    {
        bool result = true;

        f1.Update(batch, channels, spatial, format, compatibility);
        f2.Update(batch, channels, spatial, format, compatibility);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Shape shape = (format == SimdTensorFormatNhwc ? Shape({ batch, spatial, channels }) : Shape({ batch, channels, spatial }));
        Tensor32f aMin({ channels }), aMax({ channels }), bMin({ channels }), bMax({ channels }), cMin({ channels }), cMax({ channels });
        Tensor32f aScale({ channels }), aShift({ channels }), bScale({ channels }), bShift({ channels }), cScale({ channels }), cShift({ channels });
        Tensor32f a(shape, format), b(shape, format), c(shape, format);
        Tensor8u aData(shape, format), bData(shape, format), cData1(shape, format), cData2(shape, format);

        FillRandom(a, aMin.Data(), aMax.Data(), channels, negative);
        SetSrc32fTo8u(a, aMin.Data(), aMax.Data(), channels, negative, compatibility, aScale.Data(), aShift.Data(), aData);

        FillRandom(b, bMin.Data(), bMax.Data(), channels, negative);
        SetSrc32fTo8u(b, bMin.Data(), bMax.Data(), channels, negative, compatibility, bScale.Data(), bShift.Data(), bData);

        for (size_t i = 0; i < a.Size(); ++i)
            c.Data()[i] = a.Data()[i] + b.Data()[i];
        SetDstStat(channels, negative, compatibility, c, cMin.Data(), cMax.Data(), cScale.Data(), cShift.Data());

        Fill(cData1, uint8_t(1));
        Fill(cData2, uint8_t(2));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(aData, aScale, aShift, bData, bScale, bShift, cData1, cScale, cShift, compatibility));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(aData, aScale, aShift, bData, bScale, bShift, cData2, cScale, cShift, compatibility));

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = (Simd::Base::FmaAvoid(compatibility) ? 0 : 1);
#else
        int differenceMax = 1;
#endif

        result = result && Compare(cData1, cData2, differenceMax, true, 64);

        return result;
    }

    bool SynetAdd8iAutoTest(const FuncA8I& f1, const FuncA8I& f2)
    {
        bool result = true;

        const SimdTensorFormatType nchw = SimdTensorFormatNchw, nhwc = SimdTensorFormatNhwc;
        SimdSynetCompatibilityType cP = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaUse);
        SimdSynetCompatibilityType cN = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaUse);

#ifdef NDEBUG
        result = result && SynetAdd8iAutoTest(1, 255, 1000, nchw, 1, cN, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 255, 1005, nchw, 0, cP, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 256, 1005, nhwc, 1, cN, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 255, 1005, nhwc, 0, cP, f1, f2);
#endif

        result = result && SynetAdd8iAutoTest(2, 65, 1603, nhwc, 0, cP, f1, f2);
        result = result && SynetAdd8iAutoTest(2, 64, 1603, nhwc, 1, cN, f1, f2);
        result = result && SynetAdd8iAutoTest(2, 65, 1603, nchw, 0, cP, f1, f2);
        result = result && SynetAdd8iAutoTest(2, 65, 1600, nchw, 1, cN, f1, f2);

        return result;
    }

    bool SynetAdd8iAutoTest()
    {
        bool result = true;

        result = result && SynetAdd8iAutoTest(FUNC_A8I(Simd::Base::SynetAdd8i), FUNC_A8I(SimdSynetAdd8i));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetAdd8iAutoTest(FUNC_A8I(Simd::Sse41::SynetAdd8i), FUNC_A8I(SimdSynetAdd8i));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetAdd8iAutoTest(FUNC_A8I(Simd::Avx2::SynetAdd8i), FUNC_A8I(SimdSynetAdd8i));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetAdd8iAutoTest(FUNC_A8I(Simd::Avx512bw::SynetAdd8i), FUNC_A8I(SimdSynetAdd8i));
#endif

        return result;
    }
#endif
}
