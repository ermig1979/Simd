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
        struct FuncUW
        {
            typedef void(*FuncPtr)(const float * x, size_t size, const float * a, const float * b, float * d, float * w);

            FuncPtr func;
            String description;

            FuncUW(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & x, float a, float b, const View & d, const View & w, View & dDst, View & wDst) const
            {
                Simd::Copy(d, dDst);
                Simd::Copy(w, wDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)x.data, x.width, &a, &b, (float*)dDst.data, (float*)wDst.data);
            }
        };
    }
#define FUNC_UW(function) FuncUW(function, #function)

    bool NeuralUpdateWeightsAutoTest(int size, float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View x(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View w(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(x, -10.0f, 10.0f);
        FillRandom32f(d, -10.0f, 10.0f);
        FillRandom32f(w, -10.0f, 10.0f);

        float a = 2.0, b = 3.0;

        View dDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(x, a, b, d, w, dDst1, wDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(x, a, b, d, w, dDst2, wDst2));

        result = result && Compare(dDst1, dDst2, error, true, 32, relative, "d");
        result = result && Compare(wDst1, wDst2, error, true, 32, relative, "w");

        return result;
    }

    bool NeuralUpdateWeightsAutoTest(float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        result = result && NeuralUpdateWeightsAutoTest(W*H, error, relative, f1, f2);
        result = result && NeuralUpdateWeightsAutoTest(W*H + O, error, relative, f1, f2);
        result = result && NeuralUpdateWeightsAutoTest(W*H - O, error, relative, f1, f2);

        return result;
    }

    bool NeuralUpdateWeightsAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Base::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Sse41::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Avx2::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Avx512bw::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Sve2::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && NeuralUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Neon::NeuralUpdateWeights), FUNC_UW(SimdNeuralUpdateWeights));
#endif

        return result;
    }

    namespace
    {
        struct FuncAGU
        {
            typedef void(*FuncPtr)(const float * delta, size_t size, size_t batch, const float * alpha, const float * epsilon, float * gradient, float * weight);

            FuncPtr func;
            String description;

            FuncAGU(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & delta, size_t batch, float alpha, float epsilon, const View & gradientSrc, const View & weightSrc, View & gradientDst, View & weightDst) const
            {
                Simd::Copy(gradientSrc, gradientDst);
                Simd::Copy(weightSrc, weightDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)delta.data, delta.width, batch, &alpha, &epsilon, (float*)gradientDst.data, (float*)weightDst.data);
            }
        };
    }
#define FUNC_AGU(function) FuncAGU(function, #function)

    bool NeuralAdaptiveGradientUpdateAutoTest(int size, float error, bool relative, const FuncAGU & f1, const FuncAGU & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View delta(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(delta, -1.0f, 1.0f);
        FillRandom32f(gradientSrc, 0.0f, 0.0001f);
        FillRandom32f(weightSrc, -1.0f, 1.0f);

        const size_t batch = 2;
        const float alpha = 1.0f, epsilon = 0.0001f;

        View gradientDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View gradientDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View weightDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst1, weightDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(delta, batch, alpha, epsilon, gradientSrc, weightSrc, gradientDst2, weightDst2));

        result = result && Compare(gradientDst1, gradientDst2, error, true, 32, relative, "gradient");
        result = result && Compare(weightDst1, weightDst2, error, true, 32, relative, "weight");

        return result;
    }

    bool NeuralAdaptiveGradientUpdateAutoTest(float error, bool relative, const FuncAGU & f1, const FuncAGU & f2)
    {
        bool result = true;

        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H, error, relative, f1, f2);
        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H + O, error, relative, f1, f2);
        result = result && NeuralAdaptiveGradientUpdateAutoTest(W*H - O, error, relative, f1, f2);

        return result;
    }

    bool NeuralAdaptiveGradientUpdateAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Base::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Sse41::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Avx2::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Avx512bw::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Sve2::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && NeuralAdaptiveGradientUpdateAutoTest(EPS, false, FUNC_AGU(Simd::Neon::NeuralAdaptiveGradientUpdate), FUNC_AGU(SimdNeuralAdaptiveGradientUpdate));
#endif

        return result;
    }

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

    bool NeuralPooling1x1Max3x3AutoTest(const Options & options)
    {
        bool result = true;
        Size stride(1, 1), pooling(3, 3), pad(1, 1);

        if (TestBase(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse41::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx2::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512bw::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sve2::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling1x1Max3x3), FUNC_M(SimdNeuralPooling1x1Max3x3));
#endif

        return result;
    }

    bool NeuralPooling2x2Max2x2AutoTest(const Options & options)
    {
        bool result = true;
        Size stride(2, 2), pooling(2, 2), pad(0, 0);

        if (TestBase(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Base::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options) && W >= Simd::Sse41::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sse41::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options) && W >= Simd::Avx2::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx2::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options) && W >= Simd::Avx512bw::DF)
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Avx512bw::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

#ifdef SIMD_SVE2_ENABLE
        if (Simd::Sve2::Enable && TestSve2(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Sve2::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon(options))
            result = result && NeuralPoolingMaxAutoTest(stride, pooling, pad, EPS, FUNC_M(Simd::Neon::NeuralPooling2x2Max2x2), FUNC_M(SimdNeuralPooling2x2Max2x2));
#endif

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

