/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar,
*               2014-2016 Antonenka Mikhail.
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
        struct FuncO
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t width, size_t height, size_t srcStride, uint8_t * dst, size_t dstStride);
            FuncPtr func;
            String description;

            FuncO(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, src.height, src.stride, dst.data, dst.stride);
            }
        };

        struct FuncN
        {
            typedef void(*FuncPtr)(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);
            FuncPtr func;
            String description;

            FuncN(const FuncPtr& f, const String& d) : func(f), description(d) {}

            void Call(const View& src, View& dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, dst.data, dst.stride);
            }
        };
    }

#define FUNC_O(func) FuncO(func, #func)

#define FUNC_N(func) FuncN(func, #func)

    template<class Func> bool AnyToAnyAutoTest(int width, int height, View::Format srcType, View::Format dstType, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "].");

        View src(width, height, srcType, NULL, TEST_ALIGN(width));
        FillRandom(src);
        //FillSequence(src);

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    template<class Func> bool AnyToAnyAutoTest(View::Format srcType, View::Format dstType, const Func & f1, const Func & f2)
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(W, H, srcType, dstType, f1, f2);
        result = result && AnyToAnyAutoTest(W + O, H - O, srcType, dstType, f1, f2);

        return result;
    }

    bool BgraToBgrAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Base::BgraToBgr), FUNC_O(SimdBgraToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Sse41::BgraToBgr), FUNC_O(SimdBgraToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::F)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Avx2::BgraToBgr), FUNC_O(SimdBgraToBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Avx512bw::BgraToBgr), FUNC_O(SimdBgraToBgr));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Vmx::BgraToBgr), FUNC_O(SimdBgraToBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Bgr24, FUNC_O(Simd::Neon::BgraToBgr), FUNC_O(SimdBgraToBgr));
#endif 

        return result;
    }

    bool BgraToGrayAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Base::BgraToGray), FUNC_O(SimdBgraToGray));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Sse2::BgraToGray), FUNC_O(SimdBgraToGray));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Avx2::BgraToGray), FUNC_O(SimdBgraToGray));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Avx512bw::BgraToGray), FUNC_O(SimdBgraToGray));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Vmx::BgraToGray), FUNC_O(SimdBgraToGray));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Gray8, FUNC_O(Simd::Neon::BgraToGray), FUNC_O(SimdBgraToGray));
#endif 

        return result;
    }

    bool BgraToRgbAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgb24, FUNC_O(Simd::Base::BgraToRgb), FUNC_O(SimdBgraToRgb));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgb24, FUNC_O(Simd::Sse41::BgraToRgb), FUNC_O(SimdBgraToRgb));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::F)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgb24, FUNC_O(Simd::Avx2::BgraToRgb), FUNC_O(SimdBgraToRgb));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgb24, FUNC_O(Simd::Avx512bw::BgraToRgb), FUNC_O(SimdBgraToRgb));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgb24, FUNC_O(Simd::Neon::BgraToRgb), FUNC_O(SimdBgraToRgb));
#endif

        return result;
    }

    bool BgraToRgbaAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgba32, FUNC_O(Simd::Base::BgraToRgba), FUNC_O(SimdBgraToRgba));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgba32, FUNC_O(Simd::Sse41::BgraToRgba), FUNC_O(SimdBgraToRgba));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgba32, FUNC_O(Simd::Avx2::BgraToRgba), FUNC_O(SimdBgraToRgba));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgba32, FUNC_O(Simd::Avx512bw::BgraToRgba), FUNC_O(SimdBgraToRgba));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgra32, View::Rgba32, FUNC_O(Simd::Neon::BgraToRgba), FUNC_O(SimdBgraToRgba));
#endif

        return result;
    }

    bool BgrToGrayAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Base::BgrToGray), FUNC_O(SimdBgrToGray));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Sse2::BgrToGray), FUNC_O(SimdBgrToGray));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Sse41::BgrToGray), FUNC_O(SimdBgrToGray));
#endif 

#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Avx2::BgrToGray), FUNC_O(SimdBgrToGray));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Avx512bw::BgrToGray), FUNC_O(SimdBgrToGray));
#endif 

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Vmx::BgrToGray), FUNC_O(SimdBgrToGray));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Gray8, FUNC_O(Simd::Neon::BgrToGray), FUNC_O(SimdBgrToGray));
#endif

        return result;
    }

    bool BgrToHslAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgr24, View::Hsl24, FUNC_O(Simd::Base::BgrToHsl), FUNC_O(SimdBgrToHsl));

        return result;
    }

    bool BgrToHsvAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgr24, View::Hsv24, FUNC_O(Simd::Base::BgrToHsv), FUNC_O(SimdBgrToHsv));

        return result;
    }

    bool BgrToRgbAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Bgr24, View::Rgb24, FUNC_O(Simd::Base::BgrToRgb), FUNC_O(SimdBgrToRgb));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Rgb24, FUNC_O(Simd::Sse41::BgrToRgb), FUNC_O(SimdBgrToRgb));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Rgb24, FUNC_O(Simd::Avx2::BgrToRgb), FUNC_O(SimdBgrToRgb));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Rgb24, FUNC_O(Simd::Avx512bw::BgrToRgb), FUNC_O(SimdBgrToRgb));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Bgr24, View::Rgb24, FUNC_O(Simd::Neon::BgrToRgb), FUNC_O(SimdBgrToRgb));
#endif

        return result;
    }

    bool GrayToBgrAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Base::GrayToBgr), FUNC_O(SimdGrayToBgr));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Sse41::GrayToBgr), FUNC_O(SimdGrayToBgr));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Avx2::GrayToBgr), FUNC_O(SimdGrayToBgr));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Avx512bw::GrayToBgr), FUNC_O(SimdGrayToBgr));
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable && W >= Simd::Vmx::A)
            result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Vmx::GrayToBgr), FUNC_O(SimdGrayToBgr));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Gray8, View::Bgr24, FUNC_O(Simd::Neon::GrayToBgr), FUNC_O(SimdGrayToBgr));
#endif 

        return result;
    }

    bool Int16ToGrayAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Int16, View::Gray8, FUNC_O(Simd::Base::Int16ToGray), FUNC_O(SimdInt16ToGray));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && AnyToAnyAutoTest(View::Int16, View::Gray8, FUNC_O(Simd::Sse2::Int16ToGray), FUNC_O(SimdInt16ToGray));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Int16, View::Gray8, FUNC_O(Simd::Avx2::Int16ToGray), FUNC_O(SimdInt16ToGray));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Int16, View::Gray8, FUNC_O(Simd::Avx512bw::Int16ToGray), FUNC_O(SimdInt16ToGray));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Int16, View::Gray8, FUNC_O(Simd::Neon::Int16ToGray), FUNC_O(SimdInt16ToGray));
#endif 

        return result;
    }

    bool RgbToGrayAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Rgb24, View::Gray8, FUNC_O(Simd::Base::RgbToGray), FUNC_O(SimdRgbToGray));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && AnyToAnyAutoTest(View::Rgb24, View::Gray8, FUNC_O(Simd::Sse41::RgbToGray), FUNC_O(SimdRgbToGray));
#endif 

#if defined(SIMD_AVX2_ENABLE) && !defined(SIMD_CLANG_AVX2_BGR_TO_BGRA_ERROR)
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Rgb24, View::Gray8, FUNC_O(Simd::Avx2::RgbToGray), FUNC_O(SimdRgbToGray));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Rgb24, View::Gray8, FUNC_O(Simd::Avx512bw::RgbToGray), FUNC_O(SimdRgbToGray));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Rgb24, View::Gray8, FUNC_O(Simd::Neon::RgbToGray), FUNC_O(SimdRgbToGray));
#endif

        return result;
    }

    bool RgbaToGrayAutoTest()
    {
        bool result = true;

        result = result && AnyToAnyAutoTest(View::Rgba32, View::Gray8, FUNC_O(Simd::Base::RgbaToGray), FUNC_O(SimdRgbaToGray));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A)
            result = result && AnyToAnyAutoTest(View::Rgba32, View::Gray8, FUNC_O(Simd::Sse2::RgbaToGray), FUNC_O(SimdRgbaToGray));
#endif 

#if defined(SIMD_AVX2_ENABLE)
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && AnyToAnyAutoTest(View::Rgba32, View::Gray8, FUNC_O(Simd::Avx2::RgbaToGray), FUNC_O(SimdRgbaToGray));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && AnyToAnyAutoTest(View::Rgba32, View::Gray8, FUNC_O(Simd::Avx512bw::RgbaToGray), FUNC_O(SimdRgbaToGray));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A)
            result = result && AnyToAnyAutoTest(View::Rgba32, View::Gray8, FUNC_O(Simd::Neon::RgbaToGray), FUNC_O(SimdRgbaToGray));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    template<class Func> bool AnyToAnyDataTest(bool create, int width, int height, View::Format srcType, View::Format dstType, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, srcType, NULL, TEST_ALIGN(width));

        View dst1(width, height, dstType, NULL, TEST_ALIGN(width));
        View dst2(width, height, dstType, NULL, TEST_ALIGN(width));

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

            result = result && Compare(dst1, dst2, 0, true, 64, 0);
        }

        return result;
    }

    bool BgraToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Bgra32, View::Bgr24, FUNC_O(SimdBgraToBgr));

        return result;
    }

    bool BgraToGrayDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Bgra32, View::Gray8, FUNC_O(SimdBgraToGray));

        return result;
    }

    bool BgrToGrayDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Bgr24, View::Gray8, FUNC_O(SimdBgrToGray));

        return result;
    }

    bool BgrToHslDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Bgr24, View::Hsl24, FUNC_O(SimdBgrToHsl));

        return result;
    }

    bool BgrToHsvDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Bgr24, View::Hsv24, FUNC_O(SimdBgrToHsv));

        return result;
    }

    bool GrayToBgrDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Gray8, View::Bgr24, FUNC_O(SimdGrayToBgr));

        return result;
    }

    bool Int16ToGrayDataTest(bool create)
    {
        bool result = true;

        result = result && AnyToAnyDataTest(create, DW, DH, View::Int16, View::Gray8, FUNC_O(SimdInt16ToGray));

        return result;
    }

    //-----------------------------------------------------------------------

    bool ConvertImageSpecialTest(size_t width, size_t height, View::Format srcFormat, View::Format dstFormat)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test image conversion from " << ToString(srcFormat) << " to " << ToString(dstFormat) << " pixel format.");

        View src(width, height, srcFormat, NULL, TEST_ALIGN(width));
        FillRandom(src);

        View dst(width, height, dstFormat, NULL, TEST_ALIGN(width));

        Simd::Convert(src, dst);

        if (srcFormat != dstFormat && dstFormat != View::Gray8 && 
            (srcFormat != View::Bgra32 || dstFormat == View::Rgba32) &&
            (srcFormat != View::Rgba32 || dstFormat == View::Bgra32))
        {
            View rev(width, height, srcFormat, NULL, TEST_ALIGN(width));

            Simd::Convert(dst, rev);

            result = result && Compare(src, rev, 0, true, 64, 0);
        }

        return result;
    }

    bool ConvertImageSpecialTest()
    {
        bool result = true;

        View::Format formats[5] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 };
        for (size_t s = 0; s < 5; ++s)
            for (size_t d = 0; d < 5; ++d)
                result = result && ConvertImageSpecialTest(W, H, formats[s], formats[d]);

        return result;
    }
}
