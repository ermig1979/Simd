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

#include "Simd/SimdSynet.h"

#include "Simd/SimdMath.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncDl
        {
            typedef void(*FuncPtr)(const uint8_t* src, size_t size, int32_t bias, const float *norm, float *dst);

            FuncPtr func;
            String desc;

            FuncDl(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Call(const Tensor8u& src, int32_t bias, float norm, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Size(), bias, &norm, dst.Data());
            }
        };
    }

#define FUNC_DL(function) FuncDl(function, #function)

    bool SynetDequantizeLinearAutoTest(size_t size, FuncDl f1, FuncDl f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Tensor8u src(ToShape(size));
        FillRandom(src);

        Tensor32f dst1(ToShape(size));
        Tensor32f dst2(ToShape(size));

        int32_t bias = 47;
        float norm = 0.01;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, bias, norm, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, bias, norm, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool SynetDequantizeLinearAutoTest(FuncDl f1, FuncDl f2)
    {
        bool result = true;

        result = result && SynetDequantizeLinearAutoTest(H * W, f1, f2);
        result = result && SynetDequantizeLinearAutoTest(H * W + O, f1, f2);

        return result;
    }

    bool SynetDequantizeLinearAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetDequantizeLinearAutoTest(FUNC_DL(Simd::Base::SynetDequantizeLinear), FUNC_DL(SimdSynetDequantizeLinear));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetDequantizeLinearAutoTest(FUNC_DL(Simd::Sse41::SynetDequantizeLinear), FUNC_DL(SimdSynetDequantizeLinear));
#endif 

//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2())
//            result = result && SynetDequantizeLinearAutoTest(FUNC_DL(Simd::Avx2::SynetDequantizeLinear), FUNC_DL(SimdSynetDequantizeLinear));
//#endif 
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw())
//            result = result && SynetDequantizeLinearAutoTest(FUNC_DL(Simd::Avx512bw::SynetDequantizeLinear), FUNC_DL(SimdSynetDequantizeLinear));
//#endif 

        return result;
    }
#endif
}
