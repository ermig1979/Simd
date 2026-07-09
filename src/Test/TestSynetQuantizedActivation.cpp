/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Test/TestRandom.h"
#include "Test/TestString.h"
#include "Test/TestOptions.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSqplf
        {
            typedef void (*FuncPtr)(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* slope, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncSqplf(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t s, SimdTensorFormatType f)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(s) + "-" + ToString(f) + "]";
            }

            void Call(const Tensor8u& src, float srcScale, int srcZero, size_t channels, size_t spatial, const float * slope, Tensor8u& dst, float dstScale, int dstZero, SimdTensorFormatType format) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), &srcScale, srcZero, channels, spatial, slope, dst.Data(), &dstScale, dstZero, format);
            }
        };
    }

#define FUNC_SQPLF(function) FuncSqplf(function, #function)

    bool SynetQuantizedPreluLayerForwardAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncSqplf f1, FuncSqplf f2)
    {
        bool result = true;

        f1.Update(channels, spatial, format);
        f2.Update(channels, spatial, format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor8u src(ToShape(channels, spatial, format));
        Tensor8u dst0(ToShape(channels, spatial, format));
        Tensor8u dst1(ToShape(channels, spatial, format));
        Tensor32f slope(Shp(channels));

        srand(0);
        FillRandom(src);
        FillRandom(slope.Data(), slope.Size(), 0.0f, 0.1f);

        float srcScale = 50.0f, dstScale = 70.0f;
        int32_t srcZero = 47, dstZero = 30;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcScale, srcZero, channels, spatial, slope.Data(), dst0, dstScale, dstZero, format));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcScale, srcZero, channels, spatial, slope.Data(), dst1, dstScale, dstZero, format));

        result = result && Compare(dst0, dst1, 0, true, 64);

        return result;
    }

    bool SynetQuantizedPreluLayerForwardAutoTest(const FuncSqplf & f1, const FuncSqplf& f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            for (int type = 0; type <= 1; type++)
            {
                result = result && SynetQuantizedPreluLayerForwardAutoTest(H, W, format, f1, f2);
                result = result && SynetQuantizedPreluLayerForwardAutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetQuantizedPreluLayerForwardAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetQuantizedPreluLayerForwardAutoTest(FUNC_SQPLF(Simd::Base::SynetQuantizedPreluLayerForward), FUNC_SQPLF(SimdSynetQuantizedPreluLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetQuantizedPreluLayerForwardAutoTest(FUNC_SQPLF(Simd::Sse41::SynetQuantizedPreluLayerForward), FUNC_SQPLF(SimdSynetQuantizedPreluLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetQuantizedPreluLayerForwardAutoTest(FUNC_SQPLF(Simd::Avx2::SynetQuantizedPreluLayerForward), FUNC_SQPLF(SimdSynetQuantizedPreluLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetQuantizedPreluLayerForwardAutoTest(FUNC_SQPLF(Simd::Avx512bw::SynetQuantizedPreluLayerForward), FUNC_SQPLF(SimdSynetQuantizedPreluLayerForward));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetQuantizedPreluLayerForwardAutoTest(FUNC_SQPLF(Simd::Neon::SynetQuantizedPreluLayerForward), FUNC_SQPLF(SimdSynetQuantizedPreluLayerForward));
#endif

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncSqhlf
        {
            typedef void (*FuncPtr)(const uint8_t* src, const float* srcScale, int srcZero, size_t size, const float* shift, const float* scale, uint8_t* dst, const float* dstScale, int dstZero);

            FuncPtr func;
            String desc;

            FuncSqhlf(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t s)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(s) + "]";
            }

            void Call(const Tensor8u& src, float srcScale, int srcZero, size_t size, const float* shift, const float* scale, Tensor8u& dst, float dstScale, int dstZero) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), &srcScale, srcZero, size, shift, scale, dst.Data(), &dstScale, dstZero);
            }
        };
    }

#define FUNC_SQHLF(function) FuncSqhlf(function, #function)

    bool SynetQuantizedHswishLayerForwardAutoTest(size_t channels, size_t spatial, FuncSqhlf f1, FuncSqhlf f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor8u src(ToShape(channels, spatial, SimdTensorFormatNhwc));
        Tensor8u dst0(ToShape(channels, spatial, SimdTensorFormatNhwc));
        Tensor8u dst1(ToShape(channels, spatial, SimdTensorFormatNhwc));

        srand(0);
        FillRandom(src);

        float shift = 3.0f, scale = 1.0f / 6.0f, srcScale = 50.0f, dstScale = 70.0f;
        int32_t srcZero = 47, dstZero = 30;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcScale, srcZero, channels * spatial, &shift, &scale, dst0, dstScale, dstZero));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcScale, srcZero, channels * spatial, &shift, &scale, dst1, dstScale, dstZero));

        result = result && Compare(dst0, dst1, 0, true, 64);

        return result;
    }

    bool SynetQuantizedHswishLayerForwardAutoTest(const FuncSqhlf& f1, const FuncSqhlf& f2)
    {
        bool result = true;

        result = result && SynetQuantizedHswishLayerForwardAutoTest(H, W, f1, f2);
        result = result && SynetQuantizedHswishLayerForwardAutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetQuantizedHswishLayerForwardAutoTest(const Options& options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetQuantizedHswishLayerForwardAutoTest(FUNC_SQHLF(Simd::Base::SynetQuantizedHswishLayerForward), FUNC_SQHLF(SimdSynetQuantizedHswishLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetQuantizedHswishLayerForwardAutoTest(FUNC_SQHLF(Simd::Sse41::SynetQuantizedHswishLayerForward), FUNC_SQHLF(SimdSynetQuantizedHswishLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestSse41(options))
            result = result && SynetQuantizedHswishLayerForwardAutoTest(FUNC_SQHLF(Simd::Avx2::SynetQuantizedHswishLayerForward), FUNC_SQHLF(SimdSynetQuantizedHswishLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestSse41(options))
            result = result && SynetQuantizedHswishLayerForwardAutoTest(FUNC_SQHLF(Simd::Avx512bw::SynetQuantizedHswishLayerForward), FUNC_SQHLF(SimdSynetQuantizedHswishLayerForward));
#endif 

        return result;
    }
#endif
}
