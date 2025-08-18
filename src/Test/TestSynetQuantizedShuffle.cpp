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

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSqslf
        {
            typedef void (*FuncPtr)(const uint8_t* src0, int bias0, const float* norm0, size_t srcC0, const uint8_t* src1, int bias1, const float* norm1, size_t srcC1,
                size_t spatial, uint8_t* dst0, uint8_t* dst1, const float* scale, int zero, SimdTensorFormatType format, int type);

            FuncPtr func;
            String desc;

            FuncSqslf(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t srcC0, size_t srcC1, size_t spatial, SimdTensorFormatType format, int type)
            {
                desc = desc + "[" + ToString(srcC0 + srcC1) + "x" + ToString(spatial) + "-" + ToString(format) + "-" + ToString(type) + "]";
            }

            void Call(const Tensor8u& src0, int bias0, float norm0, size_t srcC0, const Tensor8u& src1, int bias1, float norm1, size_t srcC1,
                size_t spatial, Tensor8u& dst0, Tensor8u& dst1, float scale, int zero, SimdTensorFormatType format, int type) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src0.Data(), bias0, &norm0, srcC0, src1.Data(), bias1, &norm1, srcC1, spatial, dst0.Data(), dst1.Data(), &scale, zero, format, type);
            }
        };
    }

#define FUNC_SQSLF(function) FuncSqslf(function, #function)

    bool SynetQuantizedShuffleLayerForwardAutoTest(size_t srcC0, size_t srcC1, size_t spatial, SimdTensorFormatType format, int type, FuncSqslf f1, FuncSqslf f2)
    {
        bool result = true;

        if (srcC0 & 1)
            srcC0++;
        if (srcC1 & 1)
            srcC1++;

        f1.Update(srcC0, srcC1, spatial, format, type);
        f2.Update(srcC0, srcC1, spatial, format, type);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        size_t dstC = (srcC0 + srcC1) / 2;
        Tensor8u src0(ToShape(type ? dstC : srcC0, spatial, format));
        Tensor8u src1(ToShape(type ? dstC : srcC1, spatial, format));
        Tensor8u dst10(ToShape(type ? srcC0 : dstC, spatial, format));
        Tensor8u dst11(ToShape(type ? srcC1 : dstC, spatial, format));
        Tensor8u dst20(ToShape(type ? srcC0 : dstC, spatial, format));
        Tensor8u dst21(ToShape(type ? srcC1 : dstC, spatial, format));

        FillRandom(src0);
        FillRandom(src1);

        int32_t bias0 = 47, bias1 = 30, zero = 3;
        float norm0 = 0.01f, norm1 = 0.02f, scale = 100.0f;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src0, bias0, norm0, srcC0, src1, bias1, norm1, srcC1, spatial, dst10, dst11, scale, zero, format, type));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src0, bias0, norm0, srcC0, src1, bias1, norm1, srcC1, spatial, dst20, dst21, scale, zero, format, type));

        result = result && Compare(dst10, dst20, 0, true, 64, "dst0");
        result = result && Compare(dst11, dst21, 0, true, 64, "dst1");

        return result;
    }

    bool SynetQuantizedShuffleLayerForwardAutoTest(const FuncSqslf & f1, const FuncSqslf& f2)
    {
        bool result = true;

        for (SimdTensorFormatType format = SimdTensorFormatNchw; format <= SimdTensorFormatNhwc && result; format = (SimdTensorFormatType)((int)format + 1))
        {
            for (int type = 0; type <= 1; type++)
            {
                result = result && SynetQuantizedShuffleLayerForwardAutoTest(H * 7 / 16, H * 9 / 16, W, format, type, f1, f2);
                result = result && SynetQuantizedShuffleLayerForwardAutoTest(H * 7 / 16 + O, H * 9 / 16 - O, W + O, format, type, f1, f2);
            }
        }

        return result;
    }

    bool SynetQuantizedShuffleLayerForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetQuantizedShuffleLayerForwardAutoTest(FUNC_SQSLF(Simd::Base::SynetQuantizedShuffleLayerForward), FUNC_SQSLF(SimdSynetQuantizedShuffleLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetQuantizedShuffleLayerForwardAutoTest(FUNC_SQSLF(Simd::Sse41::SynetQuantizedShuffleLayerForward), FUNC_SQSLF(SimdSynetQuantizedShuffleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetQuantizedShuffleLayerForwardAutoTest(FUNC_SQSLF(Simd::Avx2::SynetQuantizedShuffleLayerForward), FUNC_SQSLF(SimdSynetQuantizedShuffleLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetQuantizedShuffleLayerForwardAutoTest(FUNC_SQSLF(Simd::Avx512bw::SynetQuantizedShuffleLayerForward), FUNC_SQSLF(SimdSynetQuantizedShuffleLayerForward));
#endif 

        return result;
    }
#endif
}
