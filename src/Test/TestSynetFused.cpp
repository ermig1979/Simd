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

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncFLF0
        {
            typedef void(*FuncPtr)(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF0(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & bias, const Tensor32f & scale, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), bias.Data(), scale.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF0(function) FuncFLF0(function, #function)

    bool SynetFusedLayerForward0AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF0 f1, FuncFLF0 f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f bias(ToShape(channels, format));
        Tensor32f scale(ToShape(channels, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias, scale, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias, scale, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward0AutoTest(int mask, const FuncFLF0 & f1, const FuncFLF0 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward0AutoTest(H, W, format, f1, f2);
                result = result && SynetFusedLayerForward0AutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward0AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward0AutoTest(TFM_ANY, FUNC_FLF0(Simd::Base::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward0AutoTest(TFM_128, FUNC_FLF0(Simd::Sse2::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward0AutoTest(TFM_256, FUNC_FLF0(Simd::Avx::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward0AutoTest(TFM_512, FUNC_FLF0(Simd::Avx512f::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward0AutoTest(TFM_128, FUNC_FLF0(Simd::Neon::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF1
        {
            typedef void(*FuncPtr)(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF1(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & bias0, const Tensor32f & scale1, const Tensor32f & bias1, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), bias0.Data(), scale1.Data(), bias1.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF1(function) FuncFLF1(function, #function)

    bool SynetFusedLayerForward1AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF1 f1, FuncFLF1 f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f bias0(ToShape(channels, format));
        Tensor32f scale1(ToShape(channels, format));
        Tensor32f bias1(ToShape(channels, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(bias0.Data(), bias0.Size(), -10.0, 10.0);
        FillRandom(scale1.Data(), scale1.Size(), -10.0, 10.0);
        FillRandom(bias1.Data(), bias1.Size(), -10.0, 10.0);
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias0, scale1, bias1, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias0, scale1, bias1, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward1AutoTest(int mask, const FuncFLF1 & f1, const FuncFLF1 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward1AutoTest(H, W, format, f1, f2);
                result = result && SynetFusedLayerForward1AutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward1AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward1AutoTest(TFM_ANY, FUNC_FLF1(Simd::Base::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward1AutoTest(TFM_128, FUNC_FLF1(Simd::Sse2::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward1AutoTest(TFM_256, FUNC_FLF1(Simd::Avx::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward1AutoTest(TFM_512, FUNC_FLF1(Simd::Avx512f::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward1AutoTest(TFM_128, FUNC_FLF1(Simd::Neon::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF2
        {
            typedef void(*FuncPtr)(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, const float * slope, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF2(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & scale, const Tensor32f & bias, size_t channels, size_t spatial, float slope, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), scale.Data(), bias.Data(), channels, spatial, &slope, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF2(function) FuncFLF2(function, #function)

    bool SynetFusedLayerForward2AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF2 f1, FuncFLF2 f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f scale(ToShape(channels, format));
        Tensor32f bias(ToShape(channels, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        const float slope = 0.1;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, channels, spatial, slope, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, channels, spatial, slope, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward2AutoTest(int mask, const FuncFLF2 & f1, const FuncFLF2 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward2AutoTest(H, W, format, f1, f2);
                result = result && SynetFusedLayerForward2AutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward2AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward2AutoTest(TFM_ANY, FUNC_FLF2(Simd::Base::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward2AutoTest(TFM_128, FUNC_FLF2(Simd::Sse2::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward2AutoTest(TFM_256, FUNC_FLF2(Simd::Avx::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward2AutoTest(TFM_512, FUNC_FLF2(Simd::Avx512f::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward2AutoTest(TFM_128, FUNC_FLF2(Simd::Neon::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF3
        {
            typedef void(*FuncPtr)(const float * src, const float * bias, const float * scale, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF3(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & bias, const Tensor32f & scale, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), bias.Data(), scale.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF3(function) FuncFLF3(function, #function)

    bool SynetFusedLayerForward3AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF3 f1, FuncFLF3 f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f bias(ToShape(channels, format));
        Tensor32f scale(ToShape(channels, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias, scale, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias, scale, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward3AutoTest(int mask, const FuncFLF3 & f1, const FuncFLF3 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward3AutoTest(H, W, format, f1, f2);
                result = result && SynetFusedLayerForward3AutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward3AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward3AutoTest(TFM_ANY, FUNC_FLF3(Simd::Base::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward3AutoTest(TFM_128, FUNC_FLF3(Simd::Sse2::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward3AutoTest(TFM_256, FUNC_FLF3(Simd::Avx::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward3AutoTest(TFM_512, FUNC_FLF3(Simd::Avx512f::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward3AutoTest(TFM_128, FUNC_FLF3(Simd::Neon::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF4
        {
            typedef void(*FuncPtr)(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF4(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & bias0, float scale1, float bias1, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), bias0.Data(), &scale1, &bias1, channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF4(function) FuncFLF4(function, #function)

    bool SynetFusedLayerForward4AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF4 f1, FuncFLF4 f2)
    {
        bool result = true;

        if (SimdAlign(channels, SimdSynetTensorAlignment(format)) != channels)
            return result;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f bias0(ToShape(channels, format));
        Tensor32f dst1(ToShape(2*channels, spatial, format));
        Tensor32f dst2(ToShape(2*channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(bias0.Data(), bias0.Size(), -10.0, 10.0);
        float scale1 = 1.5f, bias1 = 0.5f;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias0, scale1, bias1, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias0, scale1, bias1, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward4AutoTest(int mask, const FuncFLF4 & f1, const FuncFLF4 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward4AutoTest(W, H, format, f1, f2);
                result = result && SynetFusedLayerForward4AutoTest(W - O, H + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward4AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward4AutoTest(TFM_ANY, FUNC_FLF4(Simd::Base::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward4AutoTest(TFM_128, FUNC_FLF4(Simd::Sse2::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward4AutoTest(TFM_256, FUNC_FLF4(Simd::Avx::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward4AutoTest(TFM_512, FUNC_FLF4(Simd::Avx512f::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward4AutoTest(TFM_128, FUNC_FLF4(Simd::Neon::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF8
        {
            typedef void(*FuncPtr)(const float * src0, const float * src1, const float * src2, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF8(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src0, const Tensor32f & src1, const Tensor32f & src2, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src0.Data(), src1.Data(), src2.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_FLF8(function) FuncFLF8(function, #function)

    bool SynetFusedLayerForward8AutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncFLF8 f1, FuncFLF8 f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src0(ToShape(channels, spatial, format));
        Tensor32f src1(ToShape(channels, spatial, format));
        Tensor32f src2(ToShape(channels, format));
        Tensor32f dst1(ToShape(2 * channels, spatial, format));
        Tensor32f dst2(ToShape(2 * channels, spatial, format));

        FillRandom(src0.Data(), src0.Size(), -10.0, 10.0);
        FillRandom(src1.Data(), src1.Size(), -10.0, 10.0);
        FillRandom(src2.Data(), src2.Size(), -10.0, 10.0);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src0, src1, src2, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src0, src1, src2, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetFusedLayerForward8AutoTest(int mask, const FuncFLF8 & f1, const FuncFLF8 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward8AutoTest(H, W, format, f1, f2);
                result = result && SynetFusedLayerForward8AutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward8AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward8AutoTest(TFM_ANY, FUNC_FLF8(Simd::Base::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward8AutoTest(TFM_128, FUNC_FLF8(Simd::Sse2::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward8AutoTest(TFM_256, FUNC_FLF8(Simd::Avx::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward8AutoTest(TFM_512, FUNC_FLF8(Simd::Avx512f::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward8AutoTest(TFM_128, FUNC_FLF8(Simd::Neon::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));
#endif

        return result;
    }

    namespace
    {
        struct FuncFLF9
        {
            typedef void(*FuncPtr)(const float * src0, const float * src1, const float * scale, const float * bias, size_t channels0, size_t channels1, size_t spatial, float * dst0, float * dst1, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncFLF9(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format, SimdBool copy)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString((int)copy) + "]";
            }

            void Call(const Tensor32f & src0, const Tensor32f & src1, const Tensor32f & scale, const Tensor32f & bias, size_t channels0, size_t channels1, size_t spatial, SimdTensorFormatType format, Tensor32f & dst0, Tensor32f & dst1) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src0.Data(), src1.Data(), scale.Data(), bias.Data(), channels0, channels1, spatial, dst0.Data(), dst1.Data(), format);
            }
        };
    }

#define FUNC_FLF9(function) FuncFLF9(function, #function)

    bool SynetFusedLayerForward9AutoTest(size_t channels0, size_t channels1, size_t spatial, SimdTensorFormatType format, SimdBool copy, FuncFLF9 f1, FuncFLF9 f2)
    {
        bool result = true;

        if (SimdAlign(channels0, SimdSynetTensorAlignment(format)) != channels0)
            return result;

        f1.Update(format, copy);
        f2.Update(format, copy);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels0 << "+" << channels1 << ", " << spatial << "].");

        Tensor32f src0(ToShape(channels0, spatial, format));
        Tensor32f src1(ToShape(channels1, spatial, format));
        Tensor32f scale(ToShape(channels0 + channels1, format));
        Tensor32f bias(ToShape(channels0 + channels1, format));
        Tensor32f dst10(ToShape(channels0 + channels1, spatial, format));
        Tensor32f dst20(ToShape(channels0 + channels1, spatial, format));
        Tensor32f dst11, dst21;

        FillRandom(src0.Data(), src0.Size(), -10.0, 10.0);
        FillRandom(src1.Data(), src1.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        if (copy)
        {
            dst11.Reshape(ToShape(channels0 + channels1, spatial, format));
            dst21.Reshape(ToShape(channels0 + channels1, spatial, format));
        }
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src0, src1, scale, bias, channels0, channels1, spatial, format, dst10, dst11));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src0, src1, scale, bias, channels0, channels1, spatial, format, dst20, dst21));

        result = result && Compare(dst10, dst20, EPS, true, 32, DifferenceBoth, "dst0");
        if (copy)
            result = result && Compare(dst11, dst21, EPS, true, 32, DifferenceBoth, "dst1");

        return result;
    }

    bool SynetFusedLayerForward9AutoTest(int mask, const FuncFLF9 & f1, const FuncFLF9 & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetFusedLayerForward9AutoTest(W / 2, W / 2, H, format, SimdFalse, f1, f2);
                result = result && SynetFusedLayerForward9AutoTest(W / 2, W / 2, H, format, SimdTrue, f1, f2);
                result = result && SynetFusedLayerForward9AutoTest(W / 2 - O, W / 2 + O, H - O, format, SimdFalse, f1, f2);
                result = result && SynetFusedLayerForward9AutoTest(W / 2 + O, W / 2 - O, H - O, format, SimdTrue, f1, f2);
            }
        }

        return result;
    }

    bool SynetFusedLayerForward9AutoTest()
    {
        bool result = true;

        result = result && SynetFusedLayerForward9AutoTest(TFM_ANY, FUNC_FLF9(Simd::Base::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetFusedLayerForward9AutoTest(TFM_128, FUNC_FLF9(Simd::Sse2::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetFusedLayerForward9AutoTest(TFM_256, FUNC_FLF9(Simd::Avx::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetFusedLayerForward9AutoTest(TFM_512, FUNC_FLF9(Simd::Avx512f::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetFusedLayerForward9AutoTest(TFM_128, FUNC_FLF9(Simd::Neon::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));
#endif

        return result;
    }
#endif
}
