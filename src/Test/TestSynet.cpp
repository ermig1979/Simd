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

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Sse41::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx2::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Avx512bw::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
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

    bool SynetLrnLayerCrossChannelsAutoTest()
    {
        bool result = true;

        result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Base::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Sse41::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx2::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetLrnLayerCrossChannelsAutoTest(FUNC_LLCC(Simd::Avx512bw::SynetLrnLayerCrossChannels), FUNC_LLCC(SimdSynetLrnLayerCrossChannels));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
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

    bool SynetShuffleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Base::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Sse41::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Avx512bw::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetShuffleLayerForwardAutoTest(FUNC_SHLF(Simd::Neon::SynetShuffleLayerForward), FUNC_SHLF(SimdSynetShuffleLayerForward));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

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

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Sse41::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Avx2::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Avx512bw::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetUnaryOperation32fLayerForwardAutoTest(FUNC_UO(Simd::Neon::SynetUnaryOperation32fLayerForward), FUNC_UO(SimdSynetUnaryOperation32fLayerForward));
#endif 

        return result;
    }
#endif
}
