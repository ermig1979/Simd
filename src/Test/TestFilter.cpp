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
#include "Test/TestString.h"

#include "Simd/SimdGaussianBlur.h"

namespace Test
{
    namespace
    {
        struct FuncC
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, View::PixelSize(src.format), dst.data, dst.stride);
            }
        };
    }

#define ARGS_C(format, width, height, function1, function2) \
    format, width, height, \
    FuncC(function1.func, function1.description + ColorDescription(format)), \
    FuncC(function2.func, function2.description + ColorDescription(format))

#define FUNC_C(function) \
    FuncC(function, std::string(#function))

    bool ColorFilterAutoTest(View::Format format, int width, int height, const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool ColorFilterAutoTest(const FuncC & f1, const FuncC & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ColorFilterAutoTest(ARGS_C(format, W, H, f1, f2));
            result = result && ColorFilterAutoTest(ARGS_C(format, W + O, H - O, f1, f2));
        }

        return result;
    }

    bool MeanFilter3x3AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::MeanFilter3x3), FUNC_C(SimdMeanFilter3x3));
#endif

        return result;
    }

    bool MedianFilterRhomb3x3AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::MedianFilterRhomb3x3), FUNC_C(SimdMedianFilterRhomb3x3));
#endif 

        return result;
    }

    bool MedianFilterRhomb5x5AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 2 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 2 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 2 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 2 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 2 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::MedianFilterRhomb5x5), FUNC_C(SimdMedianFilterRhomb5x5));
#endif

        return result;
    }

    bool MedianFilterSquare3x3AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::MedianFilterSquare3x3), FUNC_C(SimdMedianFilterSquare3x3));
#endif

        return result;
    }

    bool MedianFilterSquare5x5AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 2 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 2 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 2 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 2 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 2 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::MedianFilterSquare5x5), FUNC_C(SimdMedianFilterSquare5x5));
#endif

        return result;
    }

    bool GaussianBlur3x3AutoTest()
    {
        bool result = true;

        result = result && ColorFilterAutoTest(FUNC_C(Simd::Base::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Sse41::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx2::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Avx512bw::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Vmx::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && ColorFilterAutoTest(FUNC_C(Simd::Neon::GaussianBlur3x3), FUNC_C(SimdGaussianBlur3x3));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncG
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncG(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
            }
        };
    }

#define FUNC_G(function) \
    FuncG(function, std::string(#function))

    bool GrayFilterAutoTest(int width, int height, View::Format format, const FuncG & f1, const FuncG & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(width, height, format, NULL, TEST_ALIGN(width));
        View d2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool GrayFilterAutoTest(View::Format format, const FuncG & f1, const FuncG & f2)
    {
        bool result = true;

        result = result && GrayFilterAutoTest(W, H, format, f1, f2);
        result = result && GrayFilterAutoTest(W + O, H - O, format, f1, f2);

        return result;
    }

    bool AbsGradientSaturatedSumAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Base::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Sse41::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Avx2::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Avx512bw::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Vmx::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Neon::AbsGradientSaturatedSum), FUNC_G(SimdAbsGradientSaturatedSum));
#endif

        return result;
    }

    bool LbpEstimateAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Base::LbpEstimate), FUNC_G(SimdLbpEstimate));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 2 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Sse41::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 2 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Avx2::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 2 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Avx512bw::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 2 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Vmx::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 2 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Neon::LbpEstimate), FUNC_G(SimdLbpEstimate));
#endif 

        return result;
    }

    bool NormalizeHistogramAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Base::NormalizeHistogram), FUNC_G(SimdNormalizeHistogram));

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Gray8, FUNC_G(Simd::Avx512bw::NormalizeHistogram), FUNC_G(SimdNormalizeHistogram));
#endif 

        return result;
    }

    bool SobelDxAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::SobelDx), FUNC_G(SimdSobelDx));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::SobelDx), FUNC_G(SimdSobelDx));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::SobelDx), FUNC_G(SimdSobelDx));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::SobelDx), FUNC_G(SimdSobelDx));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::SobelDx), FUNC_G(SimdSobelDx));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::SobelDx), FUNC_G(SimdSobelDx));
#endif

        return result;
    }

    bool SobelDxAbsAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::SobelDxAbs), FUNC_G(SimdSobelDxAbs));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::SobelDxAbs), FUNC_G(SimdSobelDxAbs));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::SobelDxAbs), FUNC_G(SimdSobelDxAbs));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::SobelDxAbs), FUNC_G(SimdSobelDxAbs));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::SobelDxAbs), FUNC_G(SimdSobelDxAbs));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::SobelDxAbs), FUNC_G(SimdSobelDxAbs));
#endif

        return result;
    }

    bool SobelDyAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::SobelDy), FUNC_G(SimdSobelDy));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::SobelDy), FUNC_G(SimdSobelDy));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::SobelDy), FUNC_G(SimdSobelDy));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::SobelDy), FUNC_G(SimdSobelDy));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::SobelDy), FUNC_G(SimdSobelDy));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::SobelDy), FUNC_G(SimdSobelDy));
#endif

        return result;
    }

    bool SobelDyAbsAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::SobelDyAbs), FUNC_G(SimdSobelDyAbs));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::SobelDyAbs), FUNC_G(SimdSobelDyAbs));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::SobelDyAbs), FUNC_G(SimdSobelDyAbs));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::SobelDyAbs), FUNC_G(SimdSobelDyAbs));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::SobelDyAbs), FUNC_G(SimdSobelDyAbs));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::SobelDyAbs), FUNC_G(SimdSobelDyAbs));
#endif

        return result;
    }

    bool ContourMetricsAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::ContourMetrics), FUNC_G(SimdContourMetrics));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::ContourMetrics), FUNC_G(SimdContourMetrics));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::ContourMetrics), FUNC_G(SimdContourMetrics));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::ContourMetrics), FUNC_G(SimdContourMetrics));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::ContourMetrics), FUNC_G(SimdContourMetrics));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::ContourMetrics), FUNC_G(SimdContourMetrics));
#endif

        return result;
    }

    bool LaplaceAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::Laplace), FUNC_G(SimdLaplace));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::Laplace), FUNC_G(SimdLaplace));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::Laplace), FUNC_G(SimdLaplace));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::Laplace), FUNC_G(SimdLaplace));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::Laplace), FUNC_G(SimdLaplace));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::Laplace), FUNC_G(SimdLaplace));
#endif 

        return result;
    }

    bool LaplaceAbsAutoTest()
    {
        bool result = true;

        result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Base::LaplaceAbs), FUNC_G(SimdLaplaceAbs));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W - 1 >= Simd::Sse41::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Sse41::LaplaceAbs), FUNC_G(SimdLaplaceAbs));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W - 1 >= Simd::Avx2::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx2::LaplaceAbs), FUNC_G(SimdLaplaceAbs));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W - 1 >= Simd::Avx512bw::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Avx512bw::LaplaceAbs), FUNC_G(SimdLaplaceAbs));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W - 1 >= Simd::Vmx::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Vmx::LaplaceAbs), FUNC_G(SimdLaplaceAbs));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W - 1 >= Simd::Neon::A)
            result = result && GrayFilterAutoTest(View::Int16, FUNC_G(Simd::Neon::LaplaceAbs), FUNC_G(SimdLaplaceAbs));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncGB
        {
            typedef void* (*FuncPtr)(size_t width, size_t height, size_t channels, const float* sigma, const float* epsilon);

            FuncPtr func;
            String description;

            FuncGB(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Update(size_t c, float s)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << ToString(s, 1, true) << "-" << c << "]";
                description = ss.str();
            }

            void Call(const View& src, float sigma, float epsilon,  View& dst) const
            {
                void* filter = NULL;
                filter = func(src.width, src.height, src.ChannelCount(), &sigma, &epsilon);
                {
                    TEST_PERFORMANCE_TEST(description);
                    SimdGaussianBlurRun(filter, src.data, src.stride, dst.data, dst.stride);
                }
                SimdRelease(filter);
            }
        };
    }

#define FUNC_GB(function) \
    FuncGB(function, std::string(#function))

//#define TEST_GAUSSIAN_BLUR_REAL_IMAGE

    bool GaussianBlurAutoTest(size_t width, size_t height, size_t channels, float sigma, FuncGB f1, FuncGB f2)
    {
        bool result = true;

        f1.Update(channels, sigma);
        f2.Update(channels, sigma);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View::Format format;
        switch (channels)
        {
        case 1: format = View::Gray8; break;
        case 2: format = View::Uv16; break;
        case 3: format = View::Bgr24; break;
        case 4: format = View::Bgra32; break;
        default:
            assert(0);
        }
        const float epsilon = 0.001f;

        View src(width, height, format, NULL, TEST_ALIGN(width));
#ifdef TEST_GAUSSIAN_BLUR_REAL_IMAGE
        FillPicture(src);
#else
        FillRandom(src);
#endif
        View dst1(width, height, format, NULL, TEST_ALIGN(width));
        View dst2(width, height, format, NULL, TEST_ALIGN(width));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, sigma, epsilon, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, sigma, epsilon, dst2));

        result = result && Compare(dst1, dst2, 1, true, 64);

#ifdef TEST_GAUSSIAN_BLUR_REAL_IMAGE
        if (format == View::Bgr24)
        {
            src.Save("src.ppm");
            dst1.Save(String("dst_") + ToString((double)sigma, 1) + ".ppm");
        }
#endif

        return result;
    }

    bool GaussianBlurAutoTest(int channels, float sigma, const FuncGB& f1, const FuncGB& f2)
    {
        bool result = true;

        result = result && GaussianBlurAutoTest(W, H, channels, sigma, f1, f2);
        result = result && GaussianBlurAutoTest(W + O, H - O, channels, sigma, f1, f2);

        return result;
    }

    bool GaussianBlurAutoTest(const FuncGB& f1, const FuncGB& f2)
    {
        bool result = true;

        //result = result && GaussianBlurAutoTest(12, 8, 1, 5.0f, f1, f2);

        for (int channels = 1; channels <= 4; channels++)
        {
            result = result && GaussianBlurAutoTest(channels, 0.5f, f1, f2);
            result = result && GaussianBlurAutoTest(channels, 1.0f, f1, f2);
            result = result && GaussianBlurAutoTest(channels, 3.0f, f1, f2);
        }

        return result;
    }

    bool GaussianBlurAutoTest()
    {
        bool result = true;

        result = result && GaussianBlurAutoTest(FUNC_GB(Simd::Base::GaussianBlurInit), FUNC_GB(SimdGaussianBlurInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && GaussianBlurAutoTest(FUNC_GB(Simd::Sse41::GaussianBlurInit), FUNC_GB(SimdGaussianBlurInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && GaussianBlurAutoTest(FUNC_GB(Simd::Avx2::GaussianBlurInit), FUNC_GB(SimdGaussianBlurInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && GaussianBlurAutoTest(FUNC_GB(Simd::Avx512bw::GaussianBlurInit), FUNC_GB(SimdGaussianBlurInit));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && GaussianBlurAutoTest(FUNC_GB(Simd::Neon::GaussianBlurInit), FUNC_GB(SimdGaussianBlurInit));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool ColorFilterDataTest(bool create, int width, int height, View::Format format, const FuncC & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, format, NULL, TEST_ALIGN(width));

        View dst1(width, height, format, NULL, TEST_ALIGN(width));
        View dst2(width, height, format, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool ColorFilterDataTest(bool create, int width, int height, const FuncC & f)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ColorFilterDataTest(create, width, height, format, FuncC(f.func, f.description + Data::Description(format)));
        }

        return result;
    }

    bool MeanFilter3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMeanFilter3x3));

        return result;
    }

    bool MedianFilterRhomb3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterRhomb3x3));

        return result;
    }

    bool MedianFilterRhomb5x5DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterRhomb5x5));

        return result;
    }

    bool MedianFilterSquare3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterSquare3x3));

        return result;
    }

    bool MedianFilterSquare5x5DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdMedianFilterSquare5x5));

        return result;
    }

    bool GaussianBlur3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ColorFilterDataTest(create, DW, DH, FUNC_C(SimdGaussianBlur3x3));

        return result;
    }

    //-----------------------------------------------------------------------

    bool GrayFilterDataTest(bool create, int width, int height, View::Format format, const FuncG & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        View dst1(width, height, format, NULL, TEST_ALIGN(width));
        View dst2(width, height, format, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 32, 0);
        }

        return result;
    }

    bool AbsGradientSaturatedSumDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Gray8, FUNC_G(SimdAbsGradientSaturatedSum));

        return result;
    }

    bool LbpEstimateDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Gray8, FUNC_G(SimdLbpEstimate));

        return result;
    }

    bool NormalizeHistogramDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Gray8, FUNC_G(SimdNormalizeHistogram));

        return result;
    }

    bool SobelDxDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdSobelDx));

        return result;
    }

    bool SobelDxAbsDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdSobelDxAbs));

        return result;
    }

    bool SobelDyDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdSobelDy));

        return result;
    }

    bool SobelDyAbsDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdSobelDyAbs));

        return result;
    }

    bool ContourMetricsDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdContourMetrics));

        return result;
    }

    bool LaplaceDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdLaplace));

        return result;
    }

    bool LaplaceAbsDataTest(bool create)
    {
        bool result = true;

        result = result && GrayFilterDataTest(create, DW, DH, View::Int16, FUNC_G(SimdLaplaceAbs));

        return result;
    }

    //-----------------------------------------------------------------------

    static void Print(const uint8_t* img, size_t rows, size_t cols, const char * desc)
    {
        std::cout << desc << ":" << std::endl;
        for (size_t row = 0; row < rows; row++)
        {
            for (size_t col = 0; col < cols; col++)
                std::cout << int(img[row * cols + col]) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    bool GaussianBlurSpecialTest()
    {
        const size_t rows = 8, cols = 12;
        uint8_t src[rows * cols], dst[rows * cols];
        for (size_t row = 0; row < rows; row++)
            for (size_t col = 0; col < cols; col++)
                src[row * cols + col] = uint8_t(row * cols + col);

        const float radius = 0.5f;
        void * blur = SimdGaussianBlurInit(cols, rows, 1, &radius, NULL);
        SimdGaussianBlurRun(blur, src, cols, dst, cols);
        SimdRelease(blur);

        Print(src, rows, cols, "src");
        Print(dst, rows, cols, "dst");

        return true;
    }
}
