/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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

namespace Test
{
    namespace
    {
        struct FuncRC
        {
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount);

            FuncPtr func;
            String description;

            FuncRC(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(View::Format f)
            {
                description = description + ColorDescription(f);
            }

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, src.ChannelCount());
            }
        };
    }

#define FUNC_RC(function) FuncRC(function, #function)

    bool ReduceColorAutoTest(int width, int height, View::Format format, FuncRC f1, FuncRC f2)
    {
        bool result = true;

        f1.Update(format);
        f2.Update(format);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        const int reducedWidth = (width + 1) / 2;
        const int reducedHeight = (height + 1) / 2;

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(reducedWidth, reducedHeight, format, NULL, TEST_ALIGN(reducedWidth));
        View d2(reducedWidth, reducedHeight, format, NULL, TEST_ALIGN(reducedWidth));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool ReduceColorAutoTest(const FuncRC & f1, const FuncRC & f2)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
        {
            result = result && ReduceColorAutoTest(W, H, format, f1, f2);
            result = result && ReduceColorAutoTest(W + O, H - O, format, f1, f2);
        }

        return result;
    }

    bool ReduceColor2x2AutoTest()
    {
        bool result = true;

        result = result && ReduceColorAutoTest(FUNC_RC(Simd::Base::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && ReduceColorAutoTest(FUNC_RC(Simd::Sse2::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ReduceColorAutoTest(FUNC_RC(Simd::Sse41::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ReduceColorAutoTest(FUNC_RC(Simd::Avx2::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ReduceColorAutoTest(FUNC_RC(Simd::Avx512bw::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ReduceColorAutoTest(FUNC_RC(Simd::Neon::ReduceColor2x2), FUNC_RC(SimdReduceColor2x2));
#endif 

        return result;
    }

    namespace
    {
        struct FuncRG1
        {
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride);

            FuncPtr func;
            String description;

            FuncRG1(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride);
            }
        };
    }

#define FUNC_RG1(function) FuncRG1(function, #function)

    namespace
    {
        struct FuncRG2
        {
            typedef void(*FuncPtr)(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
                uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, int correction);

            FuncPtr func;
            String description;
            bool correction;

            FuncRG2(const FuncPtr & f, const String & d, bool c) : func(f), description(d + (c ? "[1]" : "[0]")), correction(c) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.width, dst.height, dst.stride, correction);
            }
        };
    }

#define FUNC_RG2(function, correction) FuncRG2(function, #function, correction)

    template <class Func> bool ReduceGrayAutoTest(int width, int height, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        const int reducedWidth = (width + 1) / 2;
        const int reducedHeight = (height + 1) / 2;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(s);

        View d1(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));
        View d2(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    template <class Func> bool ReduceGrayAutoTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && ReduceGrayAutoTest(W, H, f1, f2);
        result = result && ReduceGrayAutoTest(W + E, H - E, f1, f2);

        return result;
    }

    bool ReduceGray2x2AutoTest()
    {
        bool result = true;

        result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Base::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Sse2::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Sse41::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Avx2::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Avx512bw::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Vmx::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Neon::ReduceGray2x2), FUNC_RG1(SimdReduceGray2x2));
#endif 

        return result;
    }

    bool ReduceGray3x3AutoTest()
    {
        bool result = true;

        result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Base::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
        result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Base::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Sse2::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Sse2::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));
        }
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx2::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx2::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));
        }
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx512bw::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx512bw::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));
        }
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Vmx::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Vmx::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));
        }
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Neon::ReduceGray3x3, true), FUNC_RG2(SimdReduceGray3x3, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Neon::ReduceGray3x3, false), FUNC_RG2(SimdReduceGray3x3, false));
        }
#endif 

        return result;
    }

    bool ReduceGray4x4AutoTest()
    {
        bool result = true;

        result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Base::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W > Simd::Sse2::A)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Sse2::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W > Simd::Sse41::A)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Sse41::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W > Simd::Avx2::DA)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Avx2::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W > Simd::Avx512bw::DA)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Avx512bw::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W > Simd::Vmx::DA)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Vmx::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W > Simd::Neon::DA)
            result = result && ReduceGrayAutoTest(FUNC_RG1(Simd::Neon::ReduceGray4x4), FUNC_RG1(SimdReduceGray4x4));
#endif 

        return result;
    }

    bool ReduceGray5x5AutoTest()
    {
        bool result = true;

        result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Base::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
        result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Base::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Sse2::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Sse2::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));
        }
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx2::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx2::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));
        }
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx512bw::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Avx512bw::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));
        }
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Vmx::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Vmx::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));
        }
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
        {
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Neon::ReduceGray5x5, true), FUNC_RG2(SimdReduceGray5x5, true));
            result = result && ReduceGrayAutoTest(FUNC_RG2(Simd::Neon::ReduceGray5x5, false), FUNC_RG2(SimdReduceGray5x5, false));
        }
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool ReduceColorDataTest(bool create, int width, int height, View::Format format, FuncRC f)
    {
        bool result = true;

        f.Update(format);

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        const int reducedWidth = (width + 1) / 2;
        const int reducedHeight = (height + 1) / 2;

        View s(width, height, format, NULL, TEST_ALIGN(width));
        View d1(reducedWidth, reducedHeight, format, NULL, TEST_ALIGN(reducedWidth));
        View d2(reducedWidth, reducedHeight, format, NULL, TEST_ALIGN(reducedWidth));


        if (create)
        {
            FillRandom(s);
            TEST_SAVE(s);

            f.Call(s, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(d1);

            f.Call(s, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool ReduceColor2x2DataTest(bool create)
    {
        bool result = true;

        for (View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            result = result && ReduceColorDataTest(create, DW, DH, format, FUNC_RC(SimdReduceColor2x2));

        return result;
    }

    template <class Func> bool ReduceGrayDataTest(bool create, int width, int height, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        const int reducedWidth = (width + 1) / 2;
        const int reducedHeight = (height + 1) / 2;

        View s(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d1(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));
        View d2(reducedWidth, reducedHeight, View::Gray8, NULL, TEST_ALIGN(reducedWidth));


        if (create)
        {
            FillRandom(s);
            TEST_SAVE(s);

            f.Call(s, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(d1);

            f.Call(s, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool ReduceGray2x2DataTest(bool create)
    {
        bool result = true;

        result = result && ReduceGrayDataTest(create, DW, DH, FUNC_RG1(SimdReduceGray2x2));

        return result;
    }

    bool ReduceGray3x3DataTest(bool create)
    {
        bool result = true;

        result = result && ReduceGrayDataTest(create, DW, DH, FUNC_RG2(SimdReduceGray3x3, true));

        return result;
    }

    bool ReduceGray4x4DataTest(bool create)
    {
        bool result = true;

        result = result && ReduceGrayDataTest(create, DW, DH, FUNC_RG1(SimdReduceGray4x4));

        return result;
    }

    bool ReduceGray5x5DataTest(bool create)
    {
        bool result = true;

        result = result && ReduceGrayDataTest(create, DW, DH, FUNC_RG2(SimdReduceGray5x5, true));

        return result;
    }
}
