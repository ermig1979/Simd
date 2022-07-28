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

#ifdef TEST_PERFORMANCE_TEST_ENABLE
#define SIMD_CHECK_PERFORMANCE() TEST_PERFORMANCE_TEST_(SIMD_FUNCTION)
#endif

#include "Simd/SimdNeural.hpp"

namespace Test
{
     namespace
    {
        struct FuncC2
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncC2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Size & size, const float * weights, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), size.x, size.y, weights, (float*)dstDst.data, dstDst.stride / sizeof(float));
            }
        };
    }
#define FUNC_C2(function) FuncC2(function, #function)

    bool NeuralAddConvolutionAutoTest(const Size & size, float eps, const Size & core, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x << ", " << size.y << "].");

        Size s(size), d(size);
        if (forward)
            s += core - Size(1, 1);
        else
            d += core - Size(1, 1);

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(src, 0, 1);

        View weights(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(weights, -1, 1);

        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(dstSrc, -1000, 1000);

        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true, 32, false);

        return result;
    }

    bool NeuralAddConvolutionAutoTest(float eps, const Size & core, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(Size(W, H), eps, core, forward, f1, f2);
        result = result && NeuralAddConvolutionAutoTest(Size(W - O, H + O), eps, core, forward, f1, f2);
        result = result && NeuralAddConvolutionAutoTest(Size(W + O, H - O), eps, core, forward, f1, f2);

        return result;
    }

    bool NeuralAddConvolution2x2ForwardAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse41::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution2x2Forward), FUNC_C2(SimdNeuralAddConvolution2x2Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3ForwardAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse41::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution3x3Forward), FUNC_C2(SimdNeuralAddConvolution3x3Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4ForwardAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse41::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution4x4Forward), FUNC_C2(SimdNeuralAddConvolution4x4Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5ForwardAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Base::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Sse41::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx2::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, true, FUNC_C2(Simd::Neon::NeuralAddConvolution5x5Forward), FUNC_C2(SimdNeuralAddConvolution5x5Forward));
#endif

        return result;
    }

    bool NeuralAddConvolution2x2BackwardAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse41::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution2x2Backward), FUNC_C2(SimdNeuralAddConvolution2x2Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3BackwardAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse41::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution3x3Backward), FUNC_C2(SimdNeuralAddConvolution3x3Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4BackwardAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse41::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution4x4Backward), FUNC_C2(SimdNeuralAddConvolution4x4Backward));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5BackwardAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Base::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Sse41::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx2::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Avx512bw::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionAutoTest(EPS, core, false, FUNC_C2(Simd::Neon::NeuralAddConvolution5x5Backward), FUNC_C2(SimdNeuralAddConvolution5x5Backward));
#endif

        return result;
    }

    namespace
    {
        struct FuncCS
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, const float * dst, size_t dstStride, size_t width, size_t height, float * sums);

            FuncPtr func;
            String description;

            FuncCS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & dst, const View & sumsSrc, View & sumsDst) const
            {
                Simd::Copy(sumsSrc, sumsDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), (float*)dst.data, dst.stride / sizeof(float), dst.width, dst.height, (float*)sumsDst.data);
            }
        };
    }
#define FUNC_CS(function) FuncCS(function, #function)

    bool NeuralAddConvolutionSumAutoTest(const Size & size, float eps, const Size & core, const FuncCS & f1, const FuncCS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x << ", " << size.y << "].");

        View src(size.x + core.x - 1, size.y + core.y - 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(src, -1, 1);

        View dst(size.x, size.y, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(dst, -1, 1);

        View sumsSrc(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        FillRandom32f(sumsSrc, 3000, 3000);

        View sumsDst1(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst2(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst, sumsSrc, sumsDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst, sumsSrc, sumsDst2));

        result = Compare(sumsDst1, sumsDst2, eps, true, 32);

        return result;
    }

    bool NeuralAddConvolutionSumAutoTest(float eps, const Size & core, const FuncCS & f1, const FuncCS & f2)
    {
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(Size(W, H), eps, core, f1, f2);
        result = result && NeuralAddConvolutionSumAutoTest(Size(W - O, H + O), eps, core, f1, f2);
        result = result && NeuralAddConvolutionSumAutoTest(Size(W + O, H - O), eps, core, f1, f2);

        return result;
    }

    bool NeuralAddConvolution2x2SumAutoTest()
    {
        Size core(2, 2);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512bw::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution2x2Sum), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution3x3SumAutoTest()
    {
        Size core(3, 3);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512bw::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution3x3Sum), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution4x4SumAutoTest()
    {
        Size core(4, 4);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512bw::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution4x4Sum), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
#endif

        return result;
    }

    bool NeuralAddConvolution5x5SumAutoTest()
    {
        Size core(5, 5);
        bool result = true;

        result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Base::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Sse41::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx2::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Avx512bw::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralAddConvolutionSumAutoTest(EPS, core, FUNC_CS(Simd::Neon::NeuralAddConvolution5x5Sum), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
#endif

        return result;
    }

    typedef Simd::Neural::Index Index;
    typedef Simd::Neural::Vector Vector;

    namespace
    {
        struct FuncCF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                const float * weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
                void * buffer, size_t * size, float * dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add);

            FuncPtr func;
            String description;

            FuncCF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            static Index DstIndex(const Index & src, size_t dstDepth, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation)
            {
                size_t w = (src.width + 2 * pad.x - (dilation.x * (kernel.x - 1) + 1)) / stride.x + 1;
                size_t h = (src.height + 2 * pad.y - (dilation.y * (kernel.y - 1) + 1)) / stride.y + 1;
                return Index(w, h, dstDepth);
            }

            void Update(const Index & srcIndex, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, const Index & dstIndex, int add)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << srcIndex.depth << "x" << srcIndex.height << "x" << srcIndex.width;
#if 0
                ss << "-" << kernel.x << "x" << kernel.y << "-" << pad.x << "x" << pad.y;
                ss << "-" << stride.x << "x" << stride.y << "-" << dilation.x << "x" << dilation.y;
                ss << "-" << dstIndex.width << "x" << dstIndex.height << "x" << dstIndex.depth << "]-" << add;
#else
                ss << "-" << dstIndex.depth << "x" << kernel.y << "x" << kernel.x;
                ss << "-" << stride.x << "-" << pad.x << "-1]";
#endif
                description = ss.str();
            }

            void Call(const Vector & src, const Index & srcIndex, const Vector & weight, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation,
                Vector & buffer, const Vector & dstSrc, Vector & dstDst, const Index & dstIndex, int add) const
            {
                if (add)
                    memcpy(dstDst.data(), dstSrc.data(), dstDst.size() * sizeof(float));
                size_t size = buffer.size() * sizeof(float);
                TEST_PERFORMANCE_TEST(description);
                func(src.data(), srcIndex.width, srcIndex.height, srcIndex.depth,
                    weight.data(), kernel.x, kernel.y, pad.x, pad.y, stride.x, stride.y, dilation.x, dilation.y,
                    buffer.data(), &size, dstDst.data(), dstIndex.width, dstIndex.height, dstIndex.depth, add);
            }
        };
    }
#define FUNC_CF(function) FuncCF(function, #function)

    bool NeuralConvolutionForwardAutoTest(const Index & srcIndex, size_t dstDepth, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, int add, float eps, FuncCF f1, FuncCF f2)
    {
        bool result = true;

        Index dstIndex = FuncCF::DstIndex(srcIndex, dstDepth, kernel, pad, stride, dilation);

        f1.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);
        f2.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        Test::PerformanceMeasurerStorage::s_storage.Align(SIMD_ALIGN*(srcIndex.depth%SIMD_ALIGN == 0));

        Vector src(srcIndex.Volume());
        Vector weight(kernel.x*kernel.y*srcIndex.depth*dstIndex.depth);
        Vector dstSrc(dstIndex.Volume());
        Vector dstDst1(dstIndex.Volume());
        Vector dstDst2(dstIndex.Volume());
        Vector buffer(dstIndex.Area()*srcIndex.depth*kernel.x*kernel.y * 2 + dstIndex.Area() * 2);

        FillRandom(src, 0, 1);
        FillRandom(weight, -1, 1);
        const float level = 100;
        FillRandom(dstSrc, level, level);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst1, dstIndex, add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst2, dstIndex, add));

        result = Compare(dstDst1, dstDst2, eps, true, 32, false);

        return result;
    }

    bool NeuralConvolutionForwardAutoTest(float eps, const FuncCF & f1, const FuncCF & f2)
    {
        bool result = true;
        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _5(5, 5), _7(7, 7);

#ifdef NDEBUG
#if 1
        //result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 1), 8, _3, _0, _1, _1, 0, eps, f1, f2);
        //result = result && NeuralConvolutionForwardAutoTest(Index(14, 14, 8), 12, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(6, 6, 12), 24, _3, _0, _1, _1, 0, eps, f1, f2);
        //result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 24), 32, _1, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 1), 12, _5, _0, _1, _1, 0, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(320, 180, 3), 10, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(159, 89, 10), 16, _3, _0, _1, _1, 0, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(157, 87, 16), 32, _3, _0, _1, _1, 0, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 576), 160, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 160), 960, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 960), 160, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 960), 320, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 320), 1280, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(10, 10, 1280), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(5, 5, 256), 512, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(5, 5, 512), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(3, 3, 128), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(3, 3, 256), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 128), 256, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 256), 128, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(2, 2, 256), 64, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(1, 1, 64), 128, _1, _0, _1, _1, 1, eps, f1, f2);
#endif
#if 0
        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 48), 48, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 96), 96, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 192), 192, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 384), 384, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 768), 768, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 1536), 1536, _1, _0, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 3072), 3072, _1, _0, _1, _1, 1, eps, f1, f2);

        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 16), 16, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 32), 32, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 64), 64, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 128), 128, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 256), 256, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 512), 512, _3, _1, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 1024),1024,  _3, _1, _1, _1, 1, eps, f1, f2);

        result = result && NeuralConvolutionForwardAutoTest(Index(256, 256, 10), 10, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(128, 128, 20), 20, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(64, 64, 40), 40, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(32, 32, 80), 80,  _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(16, 16, 160), 160, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(8, 8, 320), 320, _5, _2, _1, _1, 1, eps, f1, f2);
        result = result && NeuralConvolutionForwardAutoTest(Index(4, 4, 640), 640, _5, _2, _1, _1, 1, eps, f1, f2);
#endif
#else
        result = result && NeuralConvolutionForwardAutoTest(Index(6, 6, 12), 24, _3, _0, _1, _1, 0, eps, f1, f2);
#endif        

        return result;
    }

    bool NeuralConvolutionForwardAutoTest()
    {
        bool result = true;

        result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Base::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Sse41::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx2::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Avx512bw::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && NeuralConvolutionForwardAutoTest(EPS, FUNC_CF(Simd::Neon::NeuralConvolutionForward), FUNC_CF(SimdNeuralConvolutionForward));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool NeuralAddConvolutionDataTest(bool create, const Size & size, float eps, const Size & core, bool forward, const FuncC2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size.x << ", " << size.y << "].");

        Size s(size), d(size);
        if (forward)
            s += core - Size(1, 1);
        else
            d += core - Size(1, 1);

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(size.x));
        View weights(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(size.x));

        if (create)
        {
            FillRandom32f(src, 0, 1);
            FillRandom32f(weights, -1, 1);
            FillRandom32f(dstSrc, 1000, 2000);

            TEST_SAVE(src);
            TEST_SAVE(weights);
            TEST_SAVE(dstSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weights);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, eps, true, 32, true);
        }

        return result;
    }

    bool NeuralAddConvolution2x2ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(2, 2), true, FUNC_C2(SimdNeuralAddConvolution2x2Forward));
    }

    bool NeuralAddConvolution3x3ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(3, 3), true, FUNC_C2(SimdNeuralAddConvolution3x3Forward));
    }

    bool NeuralAddConvolution4x4ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(4, 4), true, FUNC_C2(SimdNeuralAddConvolution4x4Forward));
    }

    bool NeuralAddConvolution5x5ForwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(5, 5), true, FUNC_C2(SimdNeuralAddConvolution5x5Forward));
    }

    bool NeuralAddConvolution2x2BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(2, 2), false, FUNC_C2(SimdNeuralAddConvolution2x2Backward));
    }

    bool NeuralAddConvolution3x3BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(3, 3), false, FUNC_C2(SimdNeuralAddConvolution3x3Backward));
    }

    bool NeuralAddConvolution4x4BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(4, 4), false, FUNC_C2(SimdNeuralAddConvolution4x4Backward));
    }

    bool NeuralAddConvolution5x5BackwardDataTest(bool create)
    {
        return NeuralAddConvolutionDataTest(create, Size(DW, DH), EPS, Size(5, 5), false, FUNC_C2(SimdNeuralAddConvolution5x5Backward));
    }

    bool NeuralAddConvolutionSumDataTest(bool create, const Size & size, float eps, const Size & core, const FuncCS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size.x << ", " << size.y << "].");

        View src(size.x + core.x - 1, size.y + core.y - 1, View::Float, NULL, TEST_ALIGN(size.x));
        View dst(size.x, size.y, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsSrc(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst1(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));
        View sumsDst2(core.x*core.y, 1, View::Float, NULL, TEST_ALIGN(size.x));

        if (create)
        {
            FillRandom32f(src, -1, 1);
            FillRandom32f(dst, -1, 1);
            FillRandom32f(sumsSrc, 2000, 3000);

            TEST_SAVE(src);
            TEST_SAVE(dst);
            TEST_SAVE(sumsSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst, sumsSrc, sumsDst1));

            TEST_SAVE(sumsDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dst);
            TEST_LOAD(sumsSrc);

            TEST_LOAD(sumsDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst, sumsSrc, sumsDst2));

            TEST_SAVE(sumsDst2);

            result = Compare(sumsDst1, sumsDst2, eps, true, 32);
        }

        return result;
    }

    bool NeuralAddConvolution2x2SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(2, 2), FUNC_CS(SimdNeuralAddConvolution2x2Sum));
    }

    bool NeuralAddConvolution3x3SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(3, 3), FUNC_CS(SimdNeuralAddConvolution3x3Sum));
    }

    bool NeuralAddConvolution4x4SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(4, 4), FUNC_CS(SimdNeuralAddConvolution4x4Sum));
    }

    bool NeuralAddConvolution5x5SumDataTest(bool create)
    {
        return NeuralAddConvolutionSumDataTest(create, Size(DW, DH), EPS, Size(5, 5), FUNC_CS(SimdNeuralAddConvolution5x5Sum));
    }

    bool NeuralConvolutionForwardDataTest(bool create, const Index & srcIndex, const Size & kernel, const Size & pad, const Size & stride, const Size & dilation, int add, float eps, FuncCF f)
    {
        bool result = true;

        Index dstIndex = FuncCF::DstIndex(srcIndex, srcIndex.depth, kernel, pad, stride, dilation);

        f.Update(srcIndex, kernel, pad, stride, dilation, dstIndex, add);

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " .");

        Test::PerformanceMeasurerStorage::s_storage.Align(SIMD_ALIGN);

        Vector src(srcIndex.Volume());
        Vector weight(kernel.x*kernel.y*srcIndex.depth*dstIndex.depth);
        Vector dstSrc(dstIndex.Volume());
        Vector dstDst1(dstIndex.Volume());
        Vector dstDst2(dstIndex.Volume());
        Vector buffer;

        if (create)
        {
            FillRandom(src, 0, 1);
            FillRandom(weight, -1, 1);
            FillRandom(dstSrc, -1000, 1000);

            TEST_SAVE(src);
            TEST_SAVE(weight);
            TEST_SAVE(dstSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst1, dstIndex, add));

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weight);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, srcIndex, weight, kernel, pad, stride, dilation, buffer, dstSrc, dstDst2, dstIndex, add));

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, eps, true, 32);
        }

        return result;
    }

    bool NeuralConvolutionForwardDataTest(bool create)
    {
        Size _1(1, 1), _3(3, 3);
        return NeuralConvolutionForwardDataTest(create, Index(64, 64, 4), _3, _1, _1, _1, 1, EPS, FUNC_CF(SimdNeuralConvolutionForward));
    }
}

