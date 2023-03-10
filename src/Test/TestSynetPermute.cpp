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

#include "Simd/SimdSynetPermute.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncSP
        {
            typedef void*(*FuncPtr)(const size_t* shape, const size_t* order, size_t count, SimdTensorDataType type);

            FuncPtr func;
            String desc;

            FuncSP(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(const Shape& src, const Shape& dst, SimdTensorDataType type)
            {
                std::stringstream ss;
                ss << desc << "[";
                for (size_t i = 0; i < src.size(); ++i)
                    ss << (i ? "x" : "") << src[i];
                ss << "->";
                for (size_t i = 0; i < dst.size(); ++i)
                    ss << (i ? "x" : "") << dst[i];
                ss << "-" << ToString(type) << "]";
                desc = ss.str();
            }

            void Call(void * context, const uint8_t * src, uint8_t * dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdSynetPermuteForward(context, src, dst);
            }
        };
    }

#define FUNC_SP(function) FuncSP(function, #function)

    template <class T > bool SynetPermuteAutoTest(const Shape & srcShape, const Shape& order, SimdTensorDataType type, FuncSP f1, FuncSP f2)
    {
        bool result = true;

        Shape dstShape;
        for (size_t i = 0; i < srcShape.size(); ++i)
            dstShape.push_back(srcShape[order[i]]);

        f1.Update(srcShape, dstShape, type);
        f2.Update(srcShape, dstShape, type);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor<T> src(srcShape);
        Tensor<T> dst1(dstShape);
        Tensor<T> dst2(dstShape);

        FillRandom((uint8_t*)src.Data(), src.Size() * sizeof(T), 0, 255);
        memset(dst1.Data(), 1, dst1.Size() * sizeof(T));
        memset(dst2.Data(), 2, dst2.Size() * sizeof(T));

        void* context1 = f1.func(srcShape.data(), order.data(), order.size(), type);
        void* context2 = f2.func(srcShape.data(), order.data(), order.size(), type);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, (uint8_t*)src.Data(), (uint8_t*)dst1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, (uint8_t*)src.Data(), (uint8_t*)dst2.Data()));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, 1, true, 64);

        return result;
    }

    bool SynetPermuteAutoTest(const Shape& srcShape, const Shape& order, const FuncSP& f1, const FuncSP& f2)
    {
        bool result = true;

        result = result && SynetPermuteAutoTest<uint32_t>(srcShape, order, SimdTensorData32f, f1, f2);
        result = result && SynetPermuteAutoTest<uint16_t>(srcShape, order, SimdTensorData16f, f1, f2);
        result = result && SynetPermuteAutoTest<uint8_t>(srcShape, order, SimdTensorData8u, f1, f2);

        return result;
    }

    bool SynetPermuteAutoTest(const FuncSP& f1, const FuncSP& f2)
    {
        bool result = true;

        result = result && SynetPermuteAutoTest(Shp(333, 444), Shp(1, 0), f1, f2);
        result = result && SynetPermuteAutoTest(Shp(33, 66, 99), Shp(0, 2, 1), f1, f2);
        result = result && SynetPermuteAutoTest(Shp(11, 19, 25, 33), Shp(0, 3, 1, 2), f1, f2);
        result = result && SynetPermuteAutoTest(Shp(11, 19, 25, 9, 5), Shp(0, 3, 1, 2, 4), f1, f2);

        return result;
    }

    bool SynetPermuteAutoTest()
    {
        bool result = true;

        result = result && SynetPermuteAutoTest(FUNC_SP(Simd::Base::SynetPermuteInit), FUNC_SP(SimdSynetPermuteInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetPermuteAutoTest(FUNC_SP(Simd::Sse41::SynetPermuteInit), FUNC_SP(SimdSynetPermuteInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPermuteAutoTest(FUNC_SP(Simd::Avx2::SynetPermuteInit), FUNC_SP(SimdSynetPermuteInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetPermuteAutoTest(FUNC_SP(Simd::Avx512bw::SynetPermuteInit), FUNC_SP(SimdSynetPermuteInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPermuteAutoTest(FUNC_SP(Simd::Neon::SynetPermuteInit), FUNC_SP(SimdSynetPermuteInit));
#endif 

        return result;
    }
#endif
}
