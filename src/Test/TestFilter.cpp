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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestString.h"
#include "Test/TestFile.h"
#include "Test/TestRandom.h"

#include "Simd/SimdGaussianBlur.h"
#include "Simd/SimdRecursiveBilateralFilter.h"

namespace Test
{
    const bool NOISE_IMAGE = true;

    static bool GetTestImage(View& image, size_t width, size_t height, size_t channels, const String& desc1, const String& desc2)
    {
        bool result = true;
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
        if (REAL_IMAGE.empty())
        {
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " [" << width << ", " << height << "].");
            image.Recreate(width, height, format, NULL, TEST_ALIGN(width));
            if(NOISE_IMAGE)
            {
                FillRandom(image);
            }
            else
            {
                ::srand(0);
                CreateTestImage(image, 10, 10);
                //FillPicture(image);
            }
        }
        else
        {
            String path = ROOT_PATH + "/data/image/" + REAL_IMAGE;
            if (!FileExists(path))
            {
                TEST_LOG_SS(Error, "File '" << path << "' is not exist!");
                return false;
            }
            if (!image.Load(path, format))
            {
                TEST_LOG_SS(Error, "Can't load image from '" << path << "'!");
                return false;
            }
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " at " << REAL_IMAGE << " [" << image.width << "x" << image.height << "].");
        }
        TEST_ALIGN(SIMD_ALIGN);
        return result;
    }

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

    bool GaussianBlurAutoTest(size_t width, size_t height, size_t channels, float sigma, FuncGB f1, FuncGB f2)
    {
        bool result = true;

        f1.Update(channels, sigma);
        f2.Update(channels, sigma);

        View src;
        if (!GetTestImage(src, width, height, channels, f1.description, f2.description))
            return false;

        View dst1(src.width, src.height, src.format, NULL, TEST_ALIGN(width));
        View dst2(src.width, src.height, src.format, NULL, TEST_ALIGN(width));

        const float epsilon = 0.001f;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, sigma, epsilon, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, sigma, epsilon, dst2));

        result = result && Compare(dst1, dst2, 1, true, 64);

        if (src.format == View::Bgr24 && NOISE_IMAGE == false)
        {
            src.Save("src.ppm");
            dst1.Save(String("dst_") + ToString((double)sigma, 1, 1) + ".ppm");
        }

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

    //---------------------------------------------------------------------------------------------

    SIMD_INLINE String ToStr(SimdRecursiveBilateralFilterFlags flags)
    {
        std::stringstream ss;
        ss << (Simd::Precise(flags) ? "p" : "f");
        ss << (Simd::DiffType(flags) == Simd::RbfDiffAvg ? "a" : (Simd::DiffType(flags) == Simd::RbfDiffMax ? "m" : "s"));
        return ss.str();
    }

    namespace
    {
        struct FuncRBF
        {
            typedef void* (*FuncPtr)(size_t width, size_t height, size_t channels, const float* spatial, const float* range, SimdRecursiveBilateralFilterFlags flags);

            FuncPtr func;
            String description;

            FuncRBF(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Update(size_t c, float s, float r, SimdRecursiveBilateralFilterFlags f)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << ToStr(f) << "-" << c << "]";
                description = ss.str();
            }

            void Call(const View& src, float spatial, float range, SimdRecursiveBilateralFilterFlags flags, View& dst) const
            {
                void* filter = NULL;
                filter = func(src.width, src.height, src.ChannelCount(), &spatial, &range, flags);
                {
                    TEST_PERFORMANCE_TEST(description);
                    SimdRecursiveBilateralFilterRun(filter, src.data, src.stride, dst.data, dst.stride);
                }
                SimdRelease(filter);
            }
        };
    }

#define FUNC_RBF(function) \
    FuncRBF(function, std::string(#function))

    SIMD_INLINE bool SaveRbf(const View & view, const String& desc, size_t width, size_t height, size_t channels, float spatial, float range, SimdRecursiveBilateralFilterFlags flags)
    {
        std::stringstream ss;
        ss << MakePath("_out", desc) << "_" << view.width << "x" << view.height << "x" << View::ChannelCount(view.format);
        ss << "_" << ToString(spatial, 2, 1) << "_" << ToString(range, 2, 1) << "_" << ToStr(flags) << ".png";
        return CreatePathIfNotExist(ss.str(), true) && view.Save(ss.str(), SimdImageFilePng);
    }

    bool RecursiveBilateralFilterAutoTest(size_t width, size_t height, size_t channels, 
        float spatial, float range, SimdRecursiveBilateralFilterFlags flags, FuncRBF f1, FuncRBF f2)
    {
        bool result = true;

        f1.Update(channels, spatial, range, flags);
        f2.Update(channels, spatial, range, flags);

        View src;
        if (!GetTestImage(src, width, height, channels, f1.description, f2.description))
            return false;

        View dst1(src.width, src.height, src.format, NULL, TEST_ALIGN(width));
        View dst2(src.width, src.height, src.format, NULL, TEST_ALIGN(width));
        Simd::Fill(dst1, 0x01);
        Simd::Fill(dst2, 0x03);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, spatial, range, flags, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, spatial, range, flags, dst2));

        int maxDifference = 0;
        if (!Simd::FmaAvoid(flags) || width != W || width <= 128)
#if defined(_WIN32) 
            maxDifference = 2;
#else
            maxDifference = 1;
#endif

        result = result && Compare(dst1, dst2, maxDifference, true, 64);

        if (!REAL_IMAGE.empty() || NOISE_IMAGE == false || result == false)
        {
            SaveRbf(src, "src", width, height, channels, spatial, range, flags);
            SaveRbf(dst1, "dst1", width, height, channels, spatial, range, flags);
            SaveRbf(dst2, "dst2", width, height, channels, spatial, range, flags);
        }

        return result;
    }

    bool RecursiveBilateralFilterAutoTest(size_t channels, float spatial, float range, SimdRecursiveBilateralFilterFlags flags, const FuncRBF& f1, const FuncRBF& f2)
    {
        bool result = true;

        result = result && RecursiveBilateralFilterAutoTest(W, H, channels, spatial, range, flags, f1, f2);
        result = result && RecursiveBilateralFilterAutoTest(W + O, H - O, channels, spatial, range, flags, f1, f2);

        return result;
    }

    bool RecursiveBilateralFilterAutoTest(const FuncRBF& f1, const FuncRBF& f2)
    {
        bool result = true;

#if defined(_WIN32) 
        int fma = 0;
#else
        int fma = SimdRecursiveBilateralFilterFmaAvoid;
#endif

        int fa = SimdRecursiveBilateralFilterFast | SimdRecursiveBilateralFilterDiffAvg | fma;
        int fm = SimdRecursiveBilateralFilterFast | SimdRecursiveBilateralFilterDiffMax | fma;
        int fs = SimdRecursiveBilateralFilterFast | SimdRecursiveBilateralFilterDiffSum | fma;
        int pa = SimdRecursiveBilateralFilterPrecise | SimdRecursiveBilateralFilterDiffAvg | fma;

        for (int channels = 1; channels <= 4; channels++)
        {
            if (channels == 2 && (!REAL_IMAGE.empty() || NOISE_IMAGE == false))
                continue;
            result = result && RecursiveBilateralFilterAutoTest(channels, 0.12f, 0.09f, (SimdRecursiveBilateralFilterFlags)fa, f1, f2);
            result = result && RecursiveBilateralFilterAutoTest(channels, 0.12f, 0.09f, (SimdRecursiveBilateralFilterFlags)fm, f1, f2);
            //result = result && RecursiveBilateralFilterAutoTest(channels, 0.12f, 0.09f, (SimdRecursiveBilateralFilterFlags)fs, f1, f2);
            result = result && RecursiveBilateralFilterAutoTest(channels, 0.12f, 0.09f, (SimdRecursiveBilateralFilterFlags)pa, f1, f2);
        }

        return result;
    }

    bool RecursiveBilateralFilterAutoTest()
    {
        bool result = true;

        result = result && RecursiveBilateralFilterAutoTest(FUNC_RBF(Simd::Base::RecursiveBilateralFilterInit), FUNC_RBF(SimdRecursiveBilateralFilterInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && RecursiveBilateralFilterAutoTest(FUNC_RBF(Simd::Sse41::RecursiveBilateralFilterInit), FUNC_RBF(SimdRecursiveBilateralFilterInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && RecursiveBilateralFilterAutoTest(FUNC_RBF(Simd::Avx2::RecursiveBilateralFilterInit), FUNC_RBF(SimdRecursiveBilateralFilterInit));
#endif

        return result;
    }

    //---------------------------------------------------------------------------------------------

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
