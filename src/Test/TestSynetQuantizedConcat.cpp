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
        struct FuncSqclf
        {
            typedef void (*FuncPtr)(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst);

            FuncPtr func;
            String desc;

            FuncSqclf(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t num, const Shape & size)
            {
                std::stringstream ss;
                ss << desc << "[" << size.size() << "-" << num << "x" << size[0];
                for (size_t i = 1; i < size.size(); ++i)
                    ss << "+" << size[i];
                ss << "]";
                desc = ss.str();
            }

            void Call(size_t count, const uint8_t** src, size_t num, const size_t* size, const int32_t* bias, const float* norm, const float* scale, int32_t zero, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(count, src, num, size, bias, norm, scale, zero, dst);
            }
        };
    }

#define FUNC_SQCLF(function) FuncSqclf(function, #function)

    bool SynetQuantizedConcatLayerForwardAutoTest(size_t num, const Shape& size, FuncSqclf f1, FuncSqclf f2)
    {
        bool result = true;

        f1.Update(num, size);
        f2.Update(num, size);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        std::vector<Tensor8u> src(size.size());
        Ints bias(size.size());
        Buffer32f norm(size.size());
        ConstUInt8Ptrs ptr(size.size());
        size_t dstSize = 0;
        int zero = 3;
        float scale = 100.0;
        for (size_t i = 0; i < size.size(); ++i)
        {
            src[i].Reshape(Shp(num, size[i]));
            FillRandom(src[i]);
            ptr[i] = src[i].Data();
            dstSize += size[i];
            bias[i] = 33 + i * 3;
            norm[i] = 0.01f + 0.01f * i;
        }
        Tensor8u dst1(Shp(num, dstSize));
        Tensor8u dst2(Shp(num, dstSize));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(size.size(), ptr.data(), num, size.data(), bias.data(), norm.data(), &scale, zero, dst1.Data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(size.size(), ptr.data(), num, size.data(), bias.data(), norm.data(), &scale, zero, dst2.Data()));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool SynetQuantizedConcatLayerForwardAutoTest(const FuncSqclf & f1, const FuncSqclf& f2)
    {
        bool result = true;

        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100), f1, f2);
        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100, 101), f1, f2);
        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100, 101, 102), f1, f2);
        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100, 101, 102, 103), f1, f2);
        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100, 101, 102, 103, 104), f1, f2);
        result = result && SynetQuantizedConcatLayerForwardAutoTest(999, Shp(100, 101, 102, 103, 104, 105), f1, f2);

        return result;
    }

    bool SynetQuantizedConcatLayerForwardAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && SynetQuantizedConcatLayerForwardAutoTest(FUNC_SQCLF(Simd::Base::SynetQuantizedConcatLayerForward), FUNC_SQCLF(SimdSynetQuantizedConcatLayerForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && SynetQuantizedConcatLayerForwardAutoTest(FUNC_SQCLF(Simd::Sse41::SynetQuantizedConcatLayerForward), FUNC_SQCLF(SimdSynetQuantizedConcatLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && SynetQuantizedConcatLayerForwardAutoTest(FUNC_SQCLF(Simd::Avx2::SynetQuantizedConcatLayerForward), FUNC_SQCLF(SimdSynetQuantizedConcatLayerForward));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && SynetQuantizedConcatLayerForwardAutoTest(FUNC_SQCLF(Simd::Avx512bw::SynetQuantizedConcatLayerForward), FUNC_SQCLF(SimdSynetQuantizedConcatLayerForward));
#endif 

        return result;
    }
#endif
}
