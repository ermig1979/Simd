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
#include "Test/TestRandom.h"
#include "Test/TestString.h"
#include "Test/TestOptions.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSqclf
        {
            typedef void (*FuncPtr)(const uint8_t* src, const float* srcScale, int srcZero, size_t channels, size_t spatial, const float* scale, const float* bias, uint8_t* dst, const float* dstScale, int dstZero, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncSqclf(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t c, size_t s, SimdTensorFormatType f)
            {
                desc = desc + "[" + ToString(c) + "x" + ToString(s) + "-" + ToString(f) + "]";
            }

            void Call(const Tensor8u& src, float srcScale, int srcZero, size_t channels, size_t spatial, const float * scale, const float* bias, Tensor8u& dst, float dstScale, int dstZero, SimdTensorFormatType format) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), &srcScale, srcZero, channels, spatial, scale, bias, dst.Data(), &dstScale, dstZero, format);
            }
        };
    }

#define FUNC_SQCLF(function) FuncSqclf(function, #function)

    bool SynetQuantizedScaleLayerForwardAutoTest(size_t channels, size_t spatial, SimdTensorFormatType format, FuncSqclf f1, FuncSqclf f2)
    {
        bool result = true;

        f1.Update(channels, spatial, format);
        f2.Update(channels, spatial, format);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor8u src(ToShape(channels, spatial, format));
        Tensor8u dst0(ToShape(channels, spatial, format));
        Tensor8u dst1(ToShape(channels, spatial, format));
        Tensor32f scale(Shp(channels)), bias(Shp(channels));

        FillRandom(src);
        FillRandom(scale.Data(), scale.Size(), 0.5f, 1.5f);
        FillRandom(bias.Data(), bias.Size(), -10.0f, 10.0f);

        float srcScale = 50.0f, dstScale = 70.0f;
        int32_t srcZero = 47, dstZero = 30;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcScale, srcZero, channels, spatial, scale.Data(), bias.Data(), dst0, dstScale, dstZero, format));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcScale, srcZero, channels, spatial, scale.Data(), bias.Data(), dst1, dstScale, dstZero, format));

        result = result && Compare(dst0, dst1, 0, true, 64);

        return result;
    }

    bool SynetQuantizedScaleLayerForwardAutoTest(const FuncSqclf & f1, const FuncSqclf& f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            for (int type = 0; type <= 1; type++)
            {
                result = result && SynetQuantizedScaleLayerForwardAutoTest(H, W, format, f1, f2);
                result = result && SynetQuantizedScaleLayerForwardAutoTest(H - O, W + O, format, f1, f2);
            }
        }

        return result;
    }

    bool SynetQuantizedScaleLayerForwardAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetQuantizedScaleLayerForwardAutoTest(FUNC_SQCLF(Simd::Base::SynetQuantizedScaleLayerForward), FUNC_SQCLF(SimdSynetQuantizedScaleLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetQuantizedScaleLayerForwardAutoTest(FUNC_SQCLF(Simd::Sse41::SynetQuantizedScaleLayerForward), FUNC_SQCLF(SimdSynetQuantizedScaleLayerForward));
#endif 

//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2(options))
//            result = result && SynetQuantizedScaleLayerForwardAutoTest(FUNC_SQCLF(Simd::Avx2::SynetQuantizedScaleLayerForward), FUNC_SQCLF(SimdSynetQuantizedScaleLayerForward));
//#endif 
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
//            result = result && SynetQuantizedScaleLayerForwardAutoTest(FUNC_SQCLF(Simd::Avx512bw::SynetQuantizedScaleLayerForward), FUNC_SQCLF(SimdSynetQuantizedScaleLayerForward));
//#endif 

        return result;
    }
#endif
}
