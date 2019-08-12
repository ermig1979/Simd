/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Test
{
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetAddBiasAutoTest(TFM_128, FUNC_AB(Simd::Sse::SynetAddBias), FUNC_AB(SimdSynetAddBias));
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

    namespace
    {
        struct FuncCT
        {
            typedef void(*FuncPtr)(size_t n, size_t c, size_t hw, const float * src, SimdTensorFormatType srcFormat, float * dst, SimdTensorFormatType dstFormat);

            FuncPtr func;
            String desc;

            FuncCT(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType src, SimdTensorFormatType dst)
            {
                desc = desc + "[" + ToString(n) + "x" + ToString(c) + "x" + ToString(h) + "x" + ToString(w) + ":" + ToString(src) + "->" + ToString(dst) + "]";
            }

            void Call(size_t n, size_t c, size_t h, size_t w, const Tensor32f & src, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(n, c, h*w, src.Data(), src.Format(), dst.Data(), dst.Format());
            }
        };
    }

#define FUNC_CT(function) FuncCT(function, #function)

    bool SynetConvertTensorAutoTest(size_t n, size_t c, size_t h, size_t w, SimdTensorFormatType srcFormat, SimdTensorFormatType dstFormat, FuncCT f1, FuncCT f2)
    {
        bool result = true;

        f1.Update(n, c, h, w, srcFormat, dstFormat);
        f2.Update(n, c, h, w, srcFormat, dstFormat);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor32f src1(ToShape(n, c, h, w, srcFormat), srcFormat);
        Tensor32f dst1(ToShape(n, c, h, w, dstFormat), dstFormat);
        Tensor32f dst2(ToShape(n, c, h, w, dstFormat), dstFormat);
        TEST_ALIGN(SIMD_ALIGN);

#if 1
        FillDebug(src1);
#else
        FillRandom(src1.Data(), src1.Size(), -10.0, 10.0);
#endif

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(n, c, h, w, src1, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(n, c, h, w, src1, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceBoth, "forward");
#if 0
        if (srcFormat == SimdTensorFormatNchw || srcFormat == SimdTensorFormatNhwc || srcFormat == SimdTensorFormatOiyx || srcFormat == SimdTensorFormatYxio)
        {
            Tensor32f src2(ToShape(n, c, h, w, srcFormat), srcFormat);
            f2.Call(n, c, h, w, dst2, src2);
            result = result && Compare(src1, src2, EPS, true, 64, DifferenceBoth, "backward");
        }
#endif
        return result;
    }

    bool SynetConvertImageAutoTest(int mask, const FuncCT & f1, const FuncCT & f2)
    {
        bool result = true;

        for (SimdTensorFormatType src = SimdTensorFormatNchw; src <= SimdTensorFormatNchw16c && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatNchw; dst <= SimdTensorFormatNchw16c && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatNchw4c && dst >= SimdTensorFormatNchw4c) || 
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetConvertTensorAutoTest(9, W / 15 + 0, W / 60, W / 30, src, dst, f1, f2);
                result = result && SynetConvertTensorAutoTest(9, W / 15 - 1, W / 58, W / 31, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetConvertImageAutoTest()
    {
        bool result = true;

        result = result && SynetConvertImageAutoTest(TFM_ANY, FUNC_CT(Simd::Base::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetConvertImageAutoTest(TFM_128, FUNC_CT(Simd::Sse::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetConvertImageAutoTest(TFM_256, FUNC_CT(Simd::Avx::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetConvertImageAutoTest(TFM_512, FUNC_CT(Simd::Avx512f::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvertImageAutoTest(TFM_128, FUNC_CT(Simd::Neon::SynetConvertImage), FUNC_CT(SimdSynetConvertImage));
#endif

        return result;
    }

    bool SynetConvertFilterAutoTest(int mask, const FuncCT & f1, const FuncCT & f2)
    {
        bool result = true;

        for (SimdTensorFormatType src = SimdTensorFormatOiyx; src <= SimdTensorFormatOyxi16o && result; src = (SimdTensorFormatType)((int)src + 1))
        {
            for (SimdTensorFormatType dst = SimdTensorFormatOiyx; dst <= SimdTensorFormatOyxi16o && result; dst = (SimdTensorFormatType)((int)dst + 1))
            {
                if (src == dst || (src >= SimdTensorFormatOyxi4o && dst >= SimdTensorFormatOyxi4o) ||
                    ((SimdSynetTensorAlignment(src)&mask) == 0) || ((SimdSynetTensorAlignment(dst)&mask) == 0))
                    continue;
                result = result && SynetConvertTensorAutoTest(W * 9 / 30 + 0, W * 7 / 30 + 0, 3, 3, src, dst, f1, f2);
                result = result && SynetConvertTensorAutoTest(W * 9 / 10 - 1, W * 7 / 10 + 1, 1, 1, src, dst, f1, f2);
            }
        }

        return result;
    }

    bool SynetConvertFilterAutoTest()
    {
        bool result = true;

        result = result && SynetConvertFilterAutoTest(TFM_ANY, FUNC_CT(Simd::Base::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_128, FUNC_CT(Simd::Sse::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 
        
#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_256, FUNC_CT(Simd::Avx::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 
        
#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_512, FUNC_CT(Simd::Avx512f::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetConvertFilterAutoTest(TFM_128, FUNC_CT(Simd::Neon::SynetConvertFilter), FUNC_CT(SimdSynetConvertFilter));
#endif

        return result;
    }

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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetEltwiseLayerForwardAutoTest(FUNC_ELF(Simd::Sse::SynetEltwiseLayerForward), FUNC_ELF(SimdSynetEltwiseLayerForward));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward0AutoTest(TFM_128, FUNC_FLF0(Simd::Sse::SynetFusedLayerForward0), FUNC_FLF0(SimdSynetFusedLayerForward0));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward1AutoTest(TFM_128, FUNC_FLF1(Simd::Sse::SynetFusedLayerForward1), FUNC_FLF1(SimdSynetFusedLayerForward1));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward2AutoTest(TFM_128, FUNC_FLF2(Simd::Sse::SynetFusedLayerForward2), FUNC_FLF2(SimdSynetFusedLayerForward2));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward3AutoTest(TFM_128, FUNC_FLF3(Simd::Sse::SynetFusedLayerForward3), FUNC_FLF3(SimdSynetFusedLayerForward3));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward4AutoTest(TFM_128, FUNC_FLF4(Simd::Sse::SynetFusedLayerForward4), FUNC_FLF4(SimdSynetFusedLayerForward4));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward8AutoTest(TFM_128, FUNC_FLF8(Simd::Sse::SynetFusedLayerForward8), FUNC_FLF8(SimdSynetFusedLayerForward8));
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

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetFusedLayerForward9AutoTest(TFM_128, FUNC_FLF9(Simd::Sse::SynetFusedLayerForward9), FUNC_FLF9(SimdSynetFusedLayerForward9));
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

    namespace
    {
        struct FuncIPLF
        {
            typedef void(*FuncPtr)(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncIPLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncIPLF(const FuncIPLF & f, bool bias) : func(f.func), desc(f.desc + (bias ? "[1]" : "[0]")) {}

            void Call(const View & src, const View & weight, const View & bias, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)weight.data, (float*)bias.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_IPLF(function) FuncIPLF(function, #function)
#define ARGS_IPLF(bias, f1, f2) bias, FuncIPLF(f1, bias), FuncIPLF(f2, bias)

    bool SynetInnerProductLayerForwardAutoTest(size_t count, size_t size, bool hasBias, const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View weight(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias;
        View dst1(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -1.0, 1.0);
        FillRandom32f(weight, -1.0, 1.0);
        if (hasBias)
        {
            bias.Recreate(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(bias, -1.0, 1.0);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, weight, bias, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, weight, bias, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest(const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(false, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(false, f1, f2));

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Base::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Sse::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx2::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx512f::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Neon::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

        return result;
    }

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

    namespace
    {
        struct ParamP
        {
            size_t srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dstH, dstW;
            SimdBool trans, ceil;

            ParamP(size_t sC, size_t sH, size_t sW, Size k, Size s, Size b, Size e, ::SimdBool t, ::SimdBool c)
                : srcC(sC), srcH(sH), srcW(sW), kernelY(k.y), kernelX(k.x), strideY(s.y), strideX(s.x)
                , padY(b.y), padX(b.x), trans(t), ceil(c)
            {
                if (ceil)
                {
                    dstH = (size_t)(::ceil((float)(srcH + b.y + e.y - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::ceil((float)(srcW + b.x + e.x - kernelX) / strideX)) + 1;
                }
                else
                {
                    dstH = (size_t)(::floor((float)(srcH + b.y + e.y - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::floor((float)(srcW + b.x + e.x - kernelX) / strideX)) + 1;
                }
            }
        };        
            
        struct FuncP
        {
            typedef void(*FuncPtr)(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
                size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdBool trans);

            FuncPtr func;
            String desc;

            FuncP(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const ParamP & p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
                ss << "-" << p.kernelY << "x" << p.kernelX;
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padY) << "-" << p.trans;
                ss << "]";
                desc = ss.str();
            }

            void Call(const ParamP & p, const Tensor32f & src, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), p.srcC, p.srcH, p.srcW, p.kernelY, p.kernelX, p.strideY, p.strideX, p.padY, p.padX, dst.Data(), p.dstH, p.dstW, p.trans);
            }
        };
    }

#define FUNC_P(function) FuncP(function, #function)

    bool SynetPoolingForwardAutoTest(const ParamP & p, FuncP f1, FuncP f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor32f src({ p.trans ? p.srcH : p.srcC, p.trans ? p.srcW : p.srcH, p.trans ? p.srcC : p.srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f dst1({ p.trans ? p.dstH : p.srcC, p.trans ? p.dstW : p.dstH, p.trans ? p.srcC : p.dstW });
        Tensor32f dst2({ p.trans ? p.dstH : p.srcC, p.trans ? p.dstW : p.dstH, p.trans ? p.srcC : p.dstW });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetPoolingForwardMaxAutoTest(::SimdBool t, ::SimdBool c, const FuncP & f1, const FuncP & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingForwardAutoTest(ParamP(10, 238, 133, _2, _2, _0, _0,  t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(32, 99, 99, _3, _1, _1, _1, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(28, 22, 22, _3, _2, _0, _1, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(64, 21, 21, _3, _2, _1, _1, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(48, 9, 9, _2, _2, _1, _1, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(64, 8, 8, _2, _2, _0, _0, t, c), f1, f2);
        result = result && SynetPoolingForwardAutoTest(ParamP(24, 56, 48, _2, _2, _0, _0, t, c), f1, f2);

        return result;
    }

    bool SynetPoolingForwardMaxAutoTest(const FuncP & f1, const FuncP & f2)
    {
        bool result = true;

        result = result && SynetPoolingForwardMaxAutoTest(::SimdFalse, ::SimdTrue, f1, f2);
        result = result && SynetPoolingForwardMaxAutoTest(::SimdTrue, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetPoolingForwardMaxAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Base::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Sse::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Avx::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Avx2::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Avx512f::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPoolingForwardMaxAutoTest(FUNC_P(Simd::Neon::SynetPoolingForwardMax), FUNC_P(SimdSynetPoolingForwardMax));
#endif 

        return result;
    }

    namespace
    {
        struct FuncPLF
        {
            typedef void(*FuncPtr)(const float * src, const float * slope, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncPLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format)
            {
                desc = desc + "[" + ToString(format) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & slope, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), slope.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_PLF(function) FuncPLF(function, #function)

    bool SynetPreluLayerForwardAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncPLF f1, FuncPLF f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f slope(ToShape(channels, format));
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), slope.Size(), -10.0, 10.0);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetPreluLayerForwardAutoTest(int mask, const FuncPLF & f1, const FuncPLF & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetPreluLayerForwardAutoTest(H, W, format, f1, f2);
                result = result && SynetPreluLayerForwardAutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetPreluLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetPreluLayerForwardAutoTest(TFM_ANY, FUNC_PLF(Simd::Base::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetPreluLayerForwardAutoTest(TFM_128, FUNC_PLF(Simd::Sse::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPreluLayerForwardAutoTest(TFM_256, FUNC_PLF(Simd::Avx::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetPreluLayerForwardAutoTest(TFM_512, FUNC_PLF(Simd::Avx512f::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPreluLayerForwardAutoTest(TFM_128, FUNC_PLF(Simd::Neon::SynetPreluLayerForward), FUNC_PLF(SimdSynetPreluLayerForward));
#endif

        return result;
    }

    namespace
    {
        struct FuncRR
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * lower, const float * upper, float * dst);

            FuncPtr func;
            String desc;

            FuncRR(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & src, float lower, float upper, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, src.width, &lower, &upper, (float*)dst.data);
            }
        };
    }

#define FUNC_RR(function) FuncRR(function, #function)

    bool SynetRestrictRangeAutoTest(size_t size, const FuncRR & f1, const FuncRR & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        const float lower = -1.0f, upper = 1.0f;
        FillRandom32f(src, 2.0f*lower, 2.0f*upper);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, lower, upper, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, lower, upper, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetRestrictRangeAutoTest(const FuncRR & f1, const FuncRR & f2)
    {
        bool result = true;

        result = result && SynetRestrictRangeAutoTest(H*W, f1, f2);
        result = result && SynetRestrictRangeAutoTest(H*W + O, f1, f2);

        return result;
    }

    bool SynetRestrictRangeAutoTest()
    {
        bool result = true;

        result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Base::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Sse::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Avx::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Avx512f::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetRestrictRangeAutoTest(FUNC_RR(Simd::Neon::SynetRestrictRange), FUNC_RR(SimdSynetRestrictRange));
#endif 

        return result;
    }

    namespace
    {
        struct FuncSLF
        {
            typedef void(*FuncPtr)(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncSLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(SimdTensorFormatType format, bool HasBias)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString<int>(HasBias) + "]";
            }

            void Call(const Tensor32f & src, const Tensor32f & scale, const Tensor32f & bias, size_t channels, size_t spatial, SimdTensorFormatType format, Tensor32f & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), scale.Data(), bias.Data(), channels, spatial, dst.Data(), format);
            }
        };
    }

#define FUNC_SLF(function) FuncSLF(function, #function)

    bool SynetScaleLayerForwardAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, bool hasBias, FuncSLF f1, FuncSLF f2)
    {
        bool result = true;

        f1.Update(format, hasBias);
        f2.Update(format, hasBias);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << channels << ", " << spatial << "].");

        Tensor32f src(ToShape(channels, spatial, format));
        Tensor32f scale(ToShape(channels, format));
        Tensor32f bias;
        Tensor32f dst1(ToShape(channels, spatial, format));
        Tensor32f dst2(ToShape(channels, spatial, format));

        FillRandom(src.Data(), src.Size(), -10.0, 10.0);
        FillRandom(scale.Data(), scale.Size(), -10.0, 10.0);
        if (hasBias)
        {
            bias.Reshape(ToShape(channels, format));
            FillRandom(bias.Data(), bias.Size(), -10.0, 10.0);
        }
        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, channels, spatial, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, channels, spatial, format, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest(int mask, const FuncSLF & f1, const FuncSLF & f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNchw16c && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            if (SimdSynetTensorAlignment(format)&mask)
            {
                result = result && SynetScaleLayerForwardAutoTest(H, W, format, true, f1, f2);
                result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, format, true, f1, f2);
                result = result && SynetScaleLayerForwardAutoTest(H, W, format, false, f1, f2);
                result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, format, false, f1, f2);
            }
        }
        //result = result && SynetScaleLayerForwardAutoTest(3, W*W, SimdTensorFormatNhwc, true, f1, f2);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(TFM_ANY, FUNC_SLF(Simd::Base::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_128, FUNC_SLF(Simd::Sse::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_256, FUNC_SLF(Simd::Avx::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_256, FUNC_SLF(Simd::Avx2::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_512, FUNC_SLF(Simd::Avx512f::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetScaleLayerForwardAutoTest(TFM_128, FUNC_SLF(Simd::Neon::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

        return result;
    }

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
        result = result && SynetSoftmaxLayerForwardAutoTest(100, 10, 100, f1, f2);

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
}
