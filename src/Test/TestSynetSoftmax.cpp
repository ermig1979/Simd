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
#include "Test/TestString.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSm32f
        {
            typedef void(*FuncPtr)(const float * src, size_t outer, size_t count, size_t inner, float * dst);

            FuncPtr func;
            String desc;

            FuncSm32f(const FuncPtr & f, const String & d) : func(f), desc(d) {}

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

#define FUNC_SM32F(function) FuncSm32f(function, #function)

    bool SynetSoftmax32fAutoTest(size_t outer, size_t count, size_t inner, FuncSm32f f1, FuncSm32f f2)
    {
        bool result = true;

        f1.Update(outer, count, inner);
        f2.Update(outer, count, inner);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Tensor32f src({ outer, count, inner });
        FillRandom(src.Data(), src.Size(), -1000.0, 0.0f);

        Tensor32f dst1(ToShape(outer, count, inner, SimdTensorFormatNchw), SimdTensorFormatNchw, 0.1f);
        Tensor32f dst2(ToShape(outer, count, inner, SimdTensorFormatNchw), SimdTensorFormatNchw, 0.2f);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetSoftmax32fAutoTest(const FuncSm32f& f1, const FuncSm32f& f2)
    {
        bool result = true;

        result = result && SynetSoftmax32fAutoTest(392, 49, 1, f1, f2);
        result = result && SynetSoftmax32fAutoTest(21825, 2, 1, f1, f2);
        result = result && SynetSoftmax32fAutoTest(50, 10, 100, f1, f2);
        result = result && SynetSoftmax32fAutoTest(13666, 3, 1, f1, f2);
        result = result && SynetSoftmax32fAutoTest(749, 49, 1, f1, f2);
        result = result && SynetSoftmax32fAutoTest(16 * 196, 196, 1, f1, f2);

        return result;
    }

    bool SynetSoftmax32fAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetSoftmax32fAutoTest(FUNC_SM32F(Simd::Base::SynetSoftmax32f), FUNC_SM32F(SimdSynetSoftmax32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetSoftmax32fAutoTest(FUNC_SM32F(Simd::Sse41::SynetSoftmax32f), FUNC_SM32F(SimdSynetSoftmax32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetSoftmax32fAutoTest(FUNC_SM32F(Simd::Avx2::SynetSoftmax32f), FUNC_SM32F(SimdSynetSoftmax32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetSoftmax32fAutoTest(FUNC_SM32F(Simd::Avx512bw::SynetSoftmax32f), FUNC_SM32F(SimdSynetSoftmax32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && SynetSoftmax32fAutoTest(FUNC_SM32F(Simd::Neon::SynetSoftmax32f), FUNC_SM32F(SimdSynetSoftmax32f));
#endif 

        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncSm16b
        {
            typedef void(*FuncPtr)(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst);

            FuncPtr func;
            String desc;

            FuncSm16b(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t outer, size_t count, size_t inner)
            {
                desc = desc + "[" + ToString(outer) + "-" + ToString(count) + "-" + ToString(inner) + "]";
            }

            void Call(const Tensor16u& src, Tensor16u& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Axis(0), src.Axis(1), src.Axis(2), dst.Data());
            }
        };
    }

#define FUNC_SM16B(function) FuncSm16b(function, #function)

    bool SynetSoftmax16bAutoTest(size_t outer, size_t count, size_t inner, FuncSm16b f1, FuncSm16b f2)
    {
        bool result = true;

        f1.Update(outer, count, inner);
        f2.Update(outer, count, inner);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        SimdTensorFormatType format = SimdTensorFormatNchw;
        Shape shape = ToShape(outer, count, inner, format);

        Tensor32f src32f(shape, format);
        Tensor16u src16b(shape, format);
        FillRandom(src32f.Data(), src32f.Size(), -10.0, 10.0);
        SimdFloat32ToBFloat16(src32f.Data(), src32f.Size(), src16b.Data());

        Tensor32f dst32f1(shape, format);
        Tensor32f dst32f2(shape, format);
        Tensor16u dst16b1(shape, format);
        Tensor16u dst16b2(shape, format);


        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src16b, dst16b1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src16b, dst16b2));

        SimdBFloat16ToFloat32(dst16b1.Data(), dst16b1.Size(), dst32f1.Data());
        SimdBFloat16ToFloat32(dst16b2.Data(), dst16b2.Size(), dst32f2.Data());

        result = result && Compare(dst32f1, dst32f2, EPS * 8.0f, true, 32, DifferenceBoth);

        return result;
    }

    bool SynetSoftmax16bAutoTest(const FuncSm16b& f1, const FuncSm16b& f2)
    {
        bool result = true;

        result = result && SynetSoftmax16bAutoTest(392, 49, 1, f1, f2);
        result = result && SynetSoftmax16bAutoTest(21825, 2, 1, f1, f2);
        result = result && SynetSoftmax16bAutoTest(50, 10, 100, f1, f2);
        result = result && SynetSoftmax16bAutoTest(13666, 3, 1, f1, f2);
        result = result && SynetSoftmax16bAutoTest(749, 49, 1, f1, f2);
        result = result && SynetSoftmax16bAutoTest(16 * 196, 196, 1, f1, f2);

        return result;
    }

    bool SynetSoftmax16bAutoTest(const Options& options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetSoftmax16bAutoTest(FUNC_SM16B(Simd::Base::SynetSoftmax16b), FUNC_SM16B(SimdSynetSoftmax16b));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetSoftmax16bAutoTest(FUNC_SM16B(Simd::Sse41::SynetSoftmax16b), FUNC_SM16B(SimdSynetSoftmax16b));
#endif 

//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable && TestAvx2(options))
//            result = result && SynetSoftmax16bAutoTest(FUNC_SM16B(Simd::Avx2::SynetSoftmax16b), FUNC_SM16B(SimdSynetSoftmax16b));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
//            result = result && SynetSoftmax16bAutoTest(FUNC_SM16B(Simd::Avx512bw::SynetSoftmax16b), FUNC_SM16B(SimdSynetSoftmax16b));
//#endif

        return result;
    }

#endif
}
