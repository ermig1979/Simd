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
#include "Test/TestOptions.h"

#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncScs16b
        {
            typedef void(*FuncPtr)(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum);

            FuncPtr func;
            String desc;

            FuncScs16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t s, SimdTensorFormatType f)
            {
                desc = desc + "[" + ToString(c) + "-" + ToString(s) + (f == SimdTensorFormatNhwc ? "-1" : "-0") + "]";
            }

            void Call(const Tensor16u& src, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f& sum) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), channels, spatial, format, sum.Data());
            }
        };
    }

#define FUNC_SCS16B(function) FuncScs16b(function, #function)

    bool SynetChannelSum16bAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncScs16b f1, FuncScs16b f2)
    {
        bool result = true;

        f1.Update(channels, spatial, format);
        f2.Update(channels, spatial, format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << "-" << spatial << (format == SimdTensorFormatNhwc ? "-1" : "-0") << "].");

        Tensor32f src32f(ToShape(channels, spatial, format));
        Tensor16u src16b(ToShape(channels, spatial, format));

        Tensor32f sum1(ToShape(channels, format));
        Tensor32f sum2(ToShape(channels, format));

        FillRandom(src32f.Data(), src32f.Size(), -1.0, 1.0);
        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16b.Data());
        Fill(sum1, 1.0f);
        Fill(sum2, 2.0f);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src16b, channels, spatial, format, sum1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src16b, channels, spatial, format, sum2));

        result = result && Compare(sum1, sum2, EPS, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetChannelSum16bAutoTest(const FuncScs16b& f1, const FuncScs16b& f2)
    {
        bool result = true;

        SimdTensorFormatType nchw = SimdTensorFormatNchw, nhwc = SimdTensorFormatNhwc;

        result = result && SynetChannelSum16bAutoTest(555, 333, nchw, f1, f2);
        result = result && SynetChannelSum16bAutoTest(555, 333, nhwc, f1, f2);
        result = result && SynetChannelSum16bAutoTest(512, 512, nhwc, f1, f2);
        result = result && SynetChannelSum16bAutoTest(512, 512, nchw, f1, f2);
        result = result && SynetChannelSum16bAutoTest(256, 256, nhwc, f1, f2);

        return result;
    }

    bool SynetChannelSum16bAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetChannelSum16bAutoTest(FUNC_SCS16B(Simd::Base::SynetChannelSum16b), FUNC_SCS16B(SimdSynetChannelSum16b));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetChannelSum16bAutoTest(FUNC_SCS16B(Simd::Sse41::SynetChannelSum16b), FUNC_SCS16B(SimdSynetChannelSum16b));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetChannelSum16bAutoTest(FUNC_SCS16B(Simd::Avx2::SynetChannelSum16b), FUNC_SCS16B(SimdSynetChannelSum16b));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetChannelSum16bAutoTest(FUNC_SCS16B(Simd::Avx512bw::SynetChannelSum16b), FUNC_SCS16B(SimdSynetChannelSum16b));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

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

    bool SynetEltwiseLayerForwardAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Base::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Sse41::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx2::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx512bw::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Neon::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

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

    bool SynetLrnLayerCrossChannelsAutoTest(const FuncLLCC & f1, const FuncLLCC & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            result = result && SynetLrnLayerCrossChannelsAutoTest(2, H, W, format, f1, f2);
            result = result && SynetLrnLayerCrossChannelsAutoTest(2, H - O, W + O, format, f1, f2);
        }

        return result;
    }

    bool SynetLrnLayerCrossChannelsAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Base::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Sse41::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx512bw::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Neon::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

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

    bool SynetShuffleLayerForwardAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Base::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Sse41::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx2::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx512bw::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Neon::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncTs2d32f
        {
            typedef void(*FuncPtr)(const float* src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* ver, const float* hor, float* dst);

            FuncPtr func;
            String desc;

            FuncTs2d32f(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f& src, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const Tensor32f& ver, const Tensor32f& hor, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), channels, height, width, format, ver.Data(), hor.Data(), dst.Data());
            }
        };
    }

#define FUNC_TS2D32F(function) FuncTs2d32f(function, #function)

    bool SynetTiledScale2D32fAutoTest(size_t channels, size_t height, size_t width, SimdTensorFormatType format, FuncTs2d32f f1, FuncTs2d32f f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << height << ", " << width << "].");

        Tensor32f src(ToShape(channels, height, width, format));
        Tensor32f ver(ToShape(channels, 1, width, format));
        Tensor32f hor(ToShape(channels, height, 1, format));
        Tensor32f dst1(ToShape(channels, height, width, format));
        Tensor32f dst2(ToShape(channels, height, width, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(ver.Data(), ver.Size(), -10.0, 10.0);
        FillRandom(hor.Data(), hor.Size(), -10.0, 10.0);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, channels, height, width, format, ver, hor, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, channels, height, width, format, ver, hor, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetTiledScale2D32fAutoTest(const FuncTs2d32f& f1, const FuncTs2d32f& f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            result = result && SynetTiledScale2D32fAutoTest(C, (int)sqrt(H), (int)sqrt(W), format, f1, f2);
            result = result && SynetTiledScale2D32fAutoTest(C - O, (int)sqrt(H) + O / 2, (int)sqrt(W) + O / 2, format, f1, f2);
        }

        return result;
    }

    bool SynetTiledScale2D32fAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetTiledScale2D32fAutoTest(FUNC_TS2D32F(Simd::Base::SynetTiledScale2D32f), FUNC_TS2D32F(SimdSynetTiledScale2D32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetTiledScale2D32fAutoTest(FUNC_TS2D32F(Simd::Sse41::SynetTiledScale2D32f), FUNC_TS2D32F(SimdSynetTiledScale2D32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetTiledScale2D32fAutoTest(FUNC_TS2D32F(Simd::Avx2::SynetTiledScale2D32f), FUNC_TS2D32F(SimdSynetTiledScale2D32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetTiledScale2D32fAutoTest(FUNC_TS2D32F(Simd::Avx512bw::SynetTiledScale2D32f), FUNC_TS2D32F(SimdSynetTiledScale2D32f));
#endif

        return result;
    }
#endif
}
