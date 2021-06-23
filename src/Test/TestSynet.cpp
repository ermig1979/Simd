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

    bool SynetAddBiasAutoTest(int mask, const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if(SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetAddBiasAutoTest(H, W, format, f1, f2);
                result = result && SynetAddBiasAutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetAddBiasAutoTest()
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(TFM_ANY, FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetAddBiasAutoTest(TFM_128, FUNC_AB(Simd::Sse2::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetAddBiasAutoTest(TFM_256, FUNC_AB(Simd::Avx::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetAddBiasAutoTest(TFM_512, FUNC_AB(Simd::Avx512f::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetAddBiasAutoTest(TFM_128, FUNC_AB(Simd::Neon::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif

        return result;
    }

    //-------------------------------------------------------------------------

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
        SimdSynetCompatibilityType cP = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iPrecise | SimdSynetCompatibilityFmaAvoid);
        SimdSynetCompatibilityType cN = (SimdSynetCompatibilityType)(SimdSynetCompatibility8iNarrowed | SimdSynetCompatibilityFmaAvoid);

        result = result && SynetAdd8iAutoTest(1, 255, 10000, nchw, 1, cN, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 255, 10005, nchw, 0, cP, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 256, 10005, nhwc, 1, cN, f1, f2);
        result = result && SynetAdd8iAutoTest(1, 255, 10005, nhwc, 0, cP, f1, f2);

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

    //-------------------------------------------------------------------------

    SIMD_INLINE String ToString(SimdSynetEltwiseOperationType type)
    {
        switch (type)
        {
        case SimdSynetEltwiseOperationProduct:
            return "[Pro]";
        case SimdSynetEltwiseOperationSum:
            return "[Sum]";
        case SimdSynetEltwiseOperationMax:
            return "[Max]";
        case SimdSynetEltwiseOperationMin:
            return "[Min]";
        }
        assert(0);
        return "[U]";
    }

    namespace
    {
        struct FuncELF
        {
            typedef void(*FuncPtr)(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst);

            FuncPtr func;
            String desc;

            FuncELF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncELF(const FuncELF & f, SimdSynetEltwiseOperationType type, size_t count) : func(f.func), desc(f.desc + ToString(type) + "[" + ToString(count) + "]") {}

            void Call(FloatPtrs src, const View & weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.data(), (float*)weight.data, count, size, type, (float*)dst.data);
            }
        };
    }

#define FUNC_ELF(function) FuncELF(function, #function)
#define ARGS_ELF(count, type, f1, f2) count, type, FuncELF(f1, type, count), FuncELF(f2, type, count)

    bool SynetEltwiseLayerForwardAutoTest(size_t size, size_t count, SimdSynetEltwiseOperationType type, const FuncELF & f1, const FuncELF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(size, count, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(src, -1.0, 1.0);
        FloatPtrs psrc(count);
        for (size_t i = 0; i < count; ++i)
            psrc[i] = src.Row<float>(i);
        View weight(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FillRandom32f(weight, -1.0, 1.0);
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(psrc, weight, count, size, type, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(psrc, weight, count, size, type, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetEltwiseLayerForwardAutoTest(const FuncELF & f1, const FuncELF & f2)
    {
        bool result = true;

        for (SimdSynetEltwiseOperationType type = SimdSynetEltwiseOperationProduct; type <= SimdSynetEltwiseOperationMin; type = (SimdSynetEltwiseOperationType)((size_t)type + 1))
        {
            for (size_t count = 2; count <= 3; ++count)
            {
                result = result && SynetEltwiseLayerForwardAutoTest(H*W, ARGS_ELF(count, type, f1, f2));
                result = result && SynetEltwiseLayerForwardAutoTest(H*W + O, ARGS_ELF(count, type, f1, f2));
            }
        }

        return result;
    }

    bool SynetEltwiseLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Base::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Sse2::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx2::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx512f::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Neon::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncLLCC
        {
            typedef void(*FuncPtr)(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncLLCC(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, size_t half, size_t channels, size_t spatial, const float * k, Tensor32f & dst, SimdTensorFormatType format) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), half, channels, spatial, k, dst.Data(), format);
            }
        };
    }

#define FUNC_LLCC(function) FuncLLCC(function, #function)

    bool SynetLrnLayerCrossChannelsAutoTest(size_t half, size_t channels, size_t spatial, SimdTensorFormatType format, FuncLLCC f1, FuncLLCC f2)
    {
        bool result = true;

        if (format >= SimdTensorFormatNchw4c)
            return result;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        float k[3] = { 1.00, 0.10, -0.75 };

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, half, channels, spatial, k, dst1, format));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, half, channels, spatial, k, dst2, format));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetLrnLayerCrossChannelsAutoTest(int mask, const FuncLLCC & f1, const FuncLLCC & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetLrnLayerCrossChannelsAutoTest(2, H, W, format, f1, f2);
                result = result && SynetLrnLayerCrossChannelsAutoTest(2, H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetLrnLayerCrossChannelsAutoTest()
    {
        bool result = true;

        result = result && SynetLrnLayerCrossChannelsAutoTest(TFM_ANY, FUNC_LLCC(Simd::Base::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(TFM_128, FUNC_LLCC(Simd::Sse2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(TFM_256, FUNC_LLCC(Simd::Avx2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(TFM_512, FUNC_LLCC(Simd::Avx512f::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(TFM_128, FUNC_LLCC(Simd::Neon::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncShLF
        {
            typedef void(*FuncPtr)(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type);

            FuncPtr func;
            String desc;

            FuncShLF(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format, int type)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(type) + "]";
            }

            void Call(const Tensor32f & src0, const Tensor32f & src1, size_t channels0, size_t channels1, size_t spatial, SimdTensorFormatType format, int type, Tensor32f & dst0, Tensor32f & dst1) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src0.Data(), src1.Data(), channels0, channels1, spatial, dst0.Data(), dst1.Data(), format, type);
            }
        };
    }

#define FUNC_SHLF(function) FuncShLF(function, #function)

    bool SynetShuffleLayerForwardAutoTest(size_t channels0, size_t channels1, size_t spatial, SimdTensorFormatType format, int type, FuncShLF f1, FuncShLF f2)
    {
        bool result = true;

        if (channels0 & 1)
            channels0++;
        if (channels1 & 1)
            channels1++;

        f1.Update(format, type);
        f2.Update(format, type);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels0 << " + " << channels1 << ", " << spatial << "].");

        size_t channels = (channels0 + channels1) / 2;
        Tensor32f src0(ToShape(type ? channels : channels0, spatial, format));
        Tensor32f src1(ToShape(type ? channels : channels1, spatial, format));
        Tensor32f dst10(ToShape(type ? channels0 : channels, spatial, format));
        Tensor32f dst11(ToShape(type ? channels1 : channels, spatial, format));
        Tensor32f dst20(ToShape(type ? channels0 : channels, spatial, format));
        Tensor32f dst21(ToShape(type ? channels1 : channels, spatial, format));

        FillRandom(src0.Data(), src0.Size(), -10.0, 10.0);
        FillRandom(src1.Data(), src1.Size(), -10.0, 10.0);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src0, src1, channels0, channels1, spatial, format, type, dst10, dst11));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src0, src1, channels0, channels1, spatial, format, type, dst20, dst21));

        result = result && Compare(dst10, dst20, EPS, true, 32, DifferenceBoth, "dst0");
        result = result && Compare(dst11, dst21, EPS, true, 32, DifferenceBoth, "dst1");

        return result;
    }

    bool SynetShuffleLayerForwardAutoTest(const FuncShLF & f1, const FuncShLF& f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            for (int type = 0; type <= 1; type++)
            {
                result = result && SynetShuffleLayerForwardAutoTest(H * 7 / 16, H * 9 / 16, W, format, type, f1, f2);
                result = result && SynetShuffleLayerForwardAutoTest(H * 7 / 16 + O, H * 9 / 16 - O, W + O, format, type, f1, f2);
            }
        }

        return result;
    }

    bool SynetShuffleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Base::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Sse2::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx512f::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Neon::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncSM
        {
            typedef void(*FuncPtr)(const float * src, size_t outer, size_t count, size_t inner, float * dst);

            FuncPtr func;
            String desc;

            FuncSM(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t outer, size_t count, size_t inner)
            {
                desc = desc + "[" + ToString(outer) + "-" + ToString(count) + "-" + ToString(inner) + "]";
            }

            void Call(const Tensor32f & src, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Axis(0), src.Axis(1), src.Axis(2), dst.Data());
            }
        };
    }

#define FUNC_SM(function) FuncSM(function, #function)

    bool SynetSoftmaxLayerForwardAutoTest(size_t outer, size_t count, size_t inner, FuncSM f1, FuncSM f2)
    {
        bool result = true;

        f1.Update(outer, count, inner);
        f2.Update(outer, count, inner);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Tensor32f src({ outer, count, inner });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f dst1({ outer, count, inner });
        Tensor32f dst2({ outer, count, inner });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetSoftmaxLayerForwardAutoTest(const FuncSM & f1, const FuncSM & f2)
    {
        bool result = true;

        result = result && SynetSoftmaxLayerForwardAutoTest(13175, 2, 1, f1, f2);
        result = result && SynetSoftmaxLayerForwardAutoTest(21824, 2, 1, f1, f2);
        result = result && SynetSoftmaxLayerForwardAutoTest(50, 10, 100, f1, f2);
        result = result && SynetSoftmaxLayerForwardAutoTest(4856, 3, 1, f1, f2);
        result = result && SynetSoftmaxLayerForwardAutoTest(6666, 3, 1, f1, f2);

        return result;
    }

    bool SynetSoftmaxLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetSoftmaxLayerForwardAutoTest(FUNC_SM(Simd::Base::SynetSoftmaxLayerForward), FUNC_SM(SimdSynetSoftmaxLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetSoftmaxLayerForwardAutoTest(FUNC_SM(Simd::Sse2::SynetSoftmaxLayerForward), FUNC_SM(SimdSynetSoftmaxLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetSoftmaxLayerForwardAutoTest(FUNC_SM(Simd::Avx2::SynetSoftmaxLayerForward), FUNC_SM(SimdSynetSoftmaxLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetSoftmaxLayerForwardAutoTest(FUNC_SM(Simd::Avx512f::SynetSoftmaxLayerForward), FUNC_SM(SimdSynetSoftmaxLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetSoftmaxLayerForwardAutoTest(FUNC_SM(Simd::Neon::SynetSoftmaxLayerForward), FUNC_SM(SimdSynetSoftmaxLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------

    SIMD_INLINE String ToString(SimdSynetUnaryOperation32fType type)
    {
        switch (type)
        {
        case SimdSynetUnaryOperation32fAbs:
            return "Abs";
        case SimdSynetUnaryOperation32fExp:
            return "Exp";
        case SimdSynetUnaryOperation32fLog:
            return "Log";
        case SimdSynetUnaryOperation32fNeg:
            return "Neg";
        case SimdSynetUnaryOperation32fRsqrt:
            return "Rsqrt";
        case SimdSynetUnaryOperation32fSqrt:
            return "Sqrt";
        case SimdSynetUnaryOperation32fTanh:
            return "Tanh";
        case SimdSynetUnaryOperation32fZero:
            return "Zero";
        }
        assert(0);
        return "???";
    }

    namespace
    {
        struct FuncUO
        {
            typedef void(*FuncPtr)(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst);

            FuncPtr func;
            String desc;

            FuncUO(const FuncPtr & f, const String& d) : func(f), desc(d) {}

            void Update(SimdSynetUnaryOperation32fType type)
            {
                desc = desc + "[" + ToString(type) + "]";
            }

            void Call(const Tensor32f& src, SimdSynetUnaryOperation32fType type, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Size(), type, dst.Data());
            }
        };
    }

#define FUNC_UO(function) FuncUO(function, #function)

    bool SynetUnaryOperation32fLayerForwardAutoTest(size_t size, SimdSynetUnaryOperation32fType type, FuncUO f1, FuncUO f2)
    {
        bool result = true;

        f1.Update(type);
        f2.Update(type);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Tensor32f src({ size });
        float lo = -10.0, hi = 10.0f;
        if (type == SimdSynetUnaryOperation32fLog)
            lo = 0.000000001f;
        FillRandom(src.Data(), src.Size(), lo, hi);

        Tensor32f dst1({ size });
        Tensor32f dst2({ size });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, type, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, type, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetUnaryOperation32fLayerForwardAutoTest(const FuncUO& f1, const FuncUO& f2)
    {
        bool result = true;

        for (int type = (int)SimdSynetUnaryOperation32fAbs; type <= (int)SimdSynetUnaryOperation32fZero; type++)
        {
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(H*W, (SimdSynetUnaryOperation32fType)type, f1, f2);
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(H*W + O, (SimdSynetUnaryOperation32fType)type, f1, f2);
        }

        return result;
    }

    bool SynetUnaryOperation32fLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Base::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Sse2::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Avx2::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Avx512f::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Neon::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool SynetEltwiseLayerForwardDataTest(bool create, size_t size, size_t count, SimdSynetEltwiseOperationType type, const FuncELF & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << size << "].");
        View src(size, count, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        FloatPtrs psrc(count);
        for (size_t i = 0; i < count; ++i)
            psrc[i] = src.Row<float>(i);
        View weight(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -1.0, 1.0);
            FillRandom32f(weight, -1.0, 1.0);

            TEST_SAVE(src);
            TEST_SAVE(weight);

            f.Call(psrc, weight, count, size, type, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weight);

            TEST_LOAD(dst1);

            f.Call(psrc, weight, count, size, type, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetEltwiseLayerForwardDataTest(bool create)
    {
        bool result = true; 

        for (SimdSynetEltwiseOperationType type = SimdSynetEltwiseOperationProduct; type <= SimdSynetEltwiseOperationMin; type = (SimdSynetEltwiseOperationType)((size_t)type + 1))
            for (size_t count = 2; count <= 2; ++count)
                result = result && SynetEltwiseLayerForwardDataTest(create, DH*DW, count, type, FuncELF(FUNC_ELF(SimdSynetEltwiseLayerForward), type, count));
       
        return result;
    }
#endif
}
