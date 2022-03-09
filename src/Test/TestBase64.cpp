/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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

namespace Test
{
    namespace
    {
        struct FuncD
        {
            typedef void(*FunkPtr)(const uint8_t* src, size_t srcSize, uint8_t* dst, size_t *dstSize);

            FunkPtr func;
            String description;

            FuncD(const FunkPtr& f, const String& d) : func(f), description(d) {}

            void Call(const Buffer8u& src, Buffer8u& dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                size_t dstSize;
                func(src.data(), src.size(), dst.data(), &dstSize);
                dst.resize(dstSize);
            }
        };
    }

#define FUNC_D(func) FuncD(func, #func)

    bool Base64DecodeAutoTest(size_t size, const FuncD& f1, const FuncD& f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size = " << size << ".");
        TEST_ALIGN(size);

        Buffer8u orig(size);
        srand(2);
        FillRandom(orig.data(), orig.size(), 0, 255);
        orig[0] = 'M', orig[1] = 'a', orig[2] = 'n';

        Buffer8u src((size + 2) / 3 * 4);
        SimdBase64Encode(orig.data(), orig.size(), src.data());

        size_t dstS = src.size() / 4 * 3;
        Buffer8u dst1(dstS), dst2(dstS);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1.data(), dst1.size(), dst2.data(), dst2.size(), 0, true, 32);

        result = result && Compare(dst1.data(), dst1.size(), orig.data(), orig.size(), 0, true, 32);

        return result;
    }

    bool Base64DecodeAutoTest(const FuncD& f1, const FuncD& f2)
    {
        bool result = true;

        result = result && Base64DecodeAutoTest(W * H + 0, f1, f2);
        result = result && Base64DecodeAutoTest(W * H + 1, f1, f2);
        result = result && Base64DecodeAutoTest(W * H + 2, f1, f2);

        return result;
    }

    bool Base64DecodeAutoTest()
    {
        bool result = true;

        result = result && Base64DecodeAutoTest(FUNC_D(Simd::Base::Base64Decode), FUNC_D(SimdBase64Decode));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Base64DecodeAutoTest(FUNC_D(Simd::Sse41::Base64Decode), FUNC_D(SimdBase64Decode));
#endif  

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Base64DecodeAutoTest(FUNC_D(Simd::Avx2::Base64Decode), FUNC_D(SimdBase64Decode));
#endif  

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Base64DecodeAutoTest(FUNC_D(Simd::Avx512bw::Base64Decode), FUNC_D(SimdBase64Decode));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Base64DecodeAutoTest(FUNC_D(Simd::Neon::Base64Decode), FUNC_D(SimdBase64Decode));
#endif 

        return result;
    }

    //---------------------------------------------------------------------------------------------

    namespace
    {
        struct FuncE
        {
            typedef void(*FunkPtr)(const uint8_t *src, size_t size, uint8_t* dst);

            FunkPtr func;
            String description;

            FuncE(const FunkPtr & f, const String & d) : func(f), description(d) {}

            void Call(const Buffer8u& src, Buffer8u& dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data(), src.size(), dst.data());
            }
        };
    }

#define FUNC_E(func) FuncE(func, #func)

    bool Base64EncodeAutoTest(size_t size, const FuncE & f1, const FuncE & f2)
    {
        bool result = true;

        Buffer8u src(size);
        TEST_ALIGN(size);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size = " << size << ".");

        FillRandom(src.data(), src.size(), 0, 255);
        src[0] = 'M', src[1] = 'a', src[2] = 'n';

        size_t dstS = (size + 2) / 3 * 4;
        Buffer8u dst1(dstS), dst2(dstS);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1.data(), dstS, dst2.data(), dstS, 0, true, 32);

        return result;
    }

    bool Base64EncodeAutoTest(const FuncE & f1, const FuncE & f2)
    {
        bool result = true;

        result = result && Base64EncodeAutoTest(W * H + 0, f1, f2);
        result = result && Base64EncodeAutoTest(W * H + 1, f1, f2);
        result = result && Base64EncodeAutoTest(W * H + 2, f1, f2);

        return result;
    }

    bool Base64EncodeAutoTest()
    {
        bool result = true;

        result = result && Base64EncodeAutoTest(FUNC_E(Simd::Base::Base64Encode), FUNC_E(SimdBase64Encode));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Base64EncodeAutoTest(FUNC_E(Simd::Sse41::Base64Encode), FUNC_E(SimdBase64Encode));
#endif  

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Base64EncodeAutoTest(FUNC_E(Simd::Avx2::Base64Encode), FUNC_E(SimdBase64Encode));
#endif  

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Base64EncodeAutoTest(FUNC_E(Simd::Avx512bw::Base64Encode), FUNC_E(SimdBase64Encode));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && Base64EncodeAutoTest(FUNC_E(Simd::Neon::Base64Encode), FUNC_E(SimdBase64Encode));
#endif 
        
        return result;
    }
}
