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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestRandom.h"
#include "Test/TestOptions.h"

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4996)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace Test
{
    namespace
    {
        struct FuncM
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width, src.height, (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }
#define FUNC_M(function) FuncM(function, #function)

    bool NeuralPoolingMaxAutoTest(const Size & srcSize, const Size & stride, const Size & pooling, const Size & pad, float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcSize.x << ", " << srcSize.y << "].");

        View src(srcSize.x, srcSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        FillRandom32f(src, -1, 1);

        Size dstSize((srcSize - pooling + 2 * stride + 2 * pad - Size(1, 1)) / stride);
        View dst1(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));
        View dst2(dstSize.x, dstSize.y, View::Float, NULL, TEST_ALIGN(srcSize.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = Compare(dst1, dst2, eps, true, 32);

        return result;
    }

    bool NeuralPoolingMaxAutoTest(const Size & stride, const Size & pooling, const Size & pad, float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && NeuralPoolingMaxAutoTest(Size(W, H), stride, pooling, pad, eps, f1, f2);
        result = result && NeuralPoolingMaxAutoTest(Size(W + O, H - O), stride, pooling, pad, eps, f1, f2);

        return result;
    }

    bool NeuralPooling2x2Max3x3AutoTest(const Options & options)
    {
        bool result = true;
        Size stride(2, 2), pooling(3, 3), pad(0, 0);

        if (TestBase(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse41::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx2::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512bw::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sve2::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling2x2Max3x3), FUNC_M(SimdNeuralPooling2x2Max3x3));
#endif

        return result;
    }
}

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

