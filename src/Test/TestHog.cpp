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

namespace Test
{
    namespace
    {
        struct FuncHDH
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height,
                size_t cellX, size_t cellY, size_t quantization, float * histograms);

            FuncPtr func;
            String description;

            FuncHDH(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Point & cell, size_t quantization, float * histograms) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, cell.x, cell.y, quantization, histograms);
            }
        };
    }

#define FUNC_HDH(function) FuncHDH(function, #function)

    bool HogDirectionHistogramsAutoTest(const Point & cell, const Point & size, size_t quantization, const FuncHDH & f1, const FuncHDH & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size.x*cell.x << ", " << size.y*cell.y << "].");

        View s(int(size.x*cell.x), int(size.y*cell.y), View::Gray8, NULL, TEST_ALIGN(size.x*cell.x));
        FillRandom(s);

        const size_t _size = quantization*size.x*size.y;
        Buffer32f h1(_size, 0), h2(_size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, cell, quantization, h1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, cell, quantization, h2.data()));

        result = result && Compare(h1, h2, EPS, true, 32);

        return result;
    }

    bool HogDirectionHistogramsAutoTest(const FuncHDH & f1, const FuncHDH & f2)
    {
        bool result = true;

        const size_t C = 8;
        Point c(C, C), s(W / C, H / C);
        const size_t q = 18;

        result = result && HogDirectionHistogramsAutoTest(c, s, q, f1, f2);
        result = result && HogDirectionHistogramsAutoTest(c, s + Point(1, 1), q, f1, f2);

        return result;
    }

    bool HogDirectionHistogramsAutoTest()
    {
        bool result = true;

        result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Base::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable && W >= Simd::Sse2::A + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Sse2::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Sse41::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Avx2::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx512bw::HA + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Avx512bw::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_VSX_ENABLE
        if (Simd::Vsx::Enable && W >= Simd::Vsx::A + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Vsx::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A + 2)
            result = result && HogDirectionHistogramsAutoTest(FUNC_HDH(Simd::Neon::HogDirectionHistograms), FUNC_HDH(SimdHogDirectionHistograms));
#endif

        return result;
    }

    namespace
    {
        struct FuncHEF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, float * features);

            FuncPtr func;
            String description;

            FuncHEF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float * features) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, features);
            }
        };
    }

#define FUNC_HEF(function) FuncHEF(function, #function)

    bool HogExtractFeaturesAutoTest(int width, int height, const FuncHEF & f1, const FuncHEF & f2)
    {
        bool result = true;

        width = (int)Simd::AlignHi(std::max(16, width), 8);
        height = (int)Simd::AlignHi(std::max(16, height), 8);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        const size_t size = (width / 8)*(height / 8) * 31;
        Buffer32f features1(size, 0), features2(size, 0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, features1.data()));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, features2.data()));

        result = result && Compare(features1, features2, EPS, true, 64);

        return result;
    }

    bool HogExtractFeaturesAutoTest(const FuncHEF & f1, const FuncHEF & f2)
    {
        bool result = true;

        result = result && HogExtractFeaturesAutoTest(W, H, f1, f2);
        result = result && HogExtractFeaturesAutoTest(W + 8, H - 8, f1, f2);

        return result;
    }

    bool HogExtractFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Base::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A + 2)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Sse41::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::HA + 2)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Avx2::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx512bw::HA + 2)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Avx512bw::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::A + 2)
            result = result && HogExtractFeaturesAutoTest(FUNC_HEF(Simd::Neon::HogExtractFeatures), FUNC_HEF(SimdHogExtractFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHD
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHD(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const size_t width, size_t count, FloatPtrs dst, size_t dstStride) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / 4, width, src.height, count, dst.data(), dstStride);
            }
        };
    }

#define FUNC_HD(function) FuncHD(function, #function)

    bool HogDeinterleaveAutoTest(size_t width, size_t height, size_t count, const FuncHD & f1, const FuncHD & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << ", " << count << "].");

        View src(width*count, height, View::Float, NULL, TEST_ALIGN(width*count));
        for (size_t row = 0; row < height; ++row)
            for (size_t col = 0; col < width; ++col)
                for (size_t i = 0; i < count; ++i)
                    src.At<float>(count*col + i, row) = float(row * 100000 + col * 100 + i);

        View dst1(width, height*count, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width, height*count, View::Float, NULL, TEST_ALIGN(width));
        FloatPtrs d1(count), d2(count);
        for (size_t i = 0; i < count; ++i)
        {
            d1[i] = (float*)(dst1.data + dst1.stride*height*i);
            d2[i] = (float*)(dst2.data + dst2.stride*height*i);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, width, count, d1, dst1.stride / 4));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, width, count, d2, dst2.stride / 4));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogDeinterleaveAutoTest(const FuncHD & f1, const FuncHD & f2)
    {
        bool result = true;

        size_t w = Simd::AlignHi(W / 8, SIMD_ALIGN), h = H / 8, c = 31;

        result = result && HogDeinterleaveAutoTest(w, h, c, f1, f2);
        result = result && HogDeinterleaveAutoTest(w + 1, h - 1, c, f1, f2);

        return result;
    }

    bool HogDeinterleaveAutoTest()
    {
        bool result = true;

        result = result && HogDeinterleaveAutoTest(FUNC_HD(Simd::Base::HogDeinterleave), FUNC_HD(SimdHogDeinterleave));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && HogDeinterleaveAutoTest(FUNC_HD(Simd::Sse2::HogDeinterleave), FUNC_HD(SimdHogDeinterleave));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogDeinterleaveAutoTest(FUNC_HD(Simd::Avx2::HogDeinterleave), FUNC_HD(SimdHogDeinterleave));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogDeinterleaveAutoTest(FUNC_HD(Simd::Avx512bw::HogDeinterleave), FUNC_HD(SimdHogDeinterleave));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogDeinterleaveAutoTest(FUNC_HD(Simd::Neon::HogDeinterleave), FUNC_HD(SimdHogDeinterleave));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHSF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * rowFilter, size_t rowSize, const float * colFilter, size_t colSize, float * dst, size_t dstStride, int add);

            FuncPtr func;
            String description;

            FuncHSF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHSF(const FuncHSF & f, int add) : func(f.func), description(f.description + (add ? "[1]" : "[0]")) {}

            void Call(const View & src, const Buffer32f & row, const Buffer32f & col, const View & dstSrc, View & dstDst, int add) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / 4, src.width, src.height, row.data(), row.size(), col.data(), col.size(), (float*)dstDst.data, dstDst.stride / 4, add);
            }
        };
    }

#define FUNC_HSF(function) FuncHSF(function, #function)

    bool HogFilterSeparableAutoTest(int width, int height, int rowSize, int colSize, int add, const FuncHSF & f1, const FuncHSF & f2)
    {
        bool result = true;

        colSize = std::min(colSize, height - 1);
        rowSize = std::min(rowSize, width - 1);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(src, -10.0f, 10.0f);

        Buffer32f col(colSize), row(rowSize);
        FillRandom(col, -1.0f, 1.0f);
        FillRandom(row, -1.0f, 1.0f);

        View dstSrc(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(dstSrc, -10.0f, 10.0f);
        View dstDst1(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        View dstDst2(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, row, col, dstSrc, dstDst1, add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, row, col, dstSrc, dstDst2, add));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 64, false);

        return result;
    }

    bool HogFilterSeparableAutoTest(const FuncHSF & f1, const FuncHSF & f2)
    {
        bool result = true;

        int w = (int)Simd::AlignHi(W / 4, SIMD_ALIGN), h = H / 4;

        for (int add = 0; result && add < 2; ++add)
        {
            result = result && HogFilterSeparableAutoTest(w, h, 10, 10, add, FuncHSF(f1, add), FuncHSF(f2, add));
            result = result && HogFilterSeparableAutoTest(w + 1, h - 1, 11, 9, add, FuncHSF(f1, add), FuncHSF(f2, add));
        }

        return result;
    }

    bool HogFilterSeparableAutoTest()
    {
        bool result = true;

        result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Base::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Sse2::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Avx2::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Avx512bw::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogFilterSeparableAutoTest(FUNC_HSF(Simd::Neon::HogFilterSeparable), FUNC_HSF(SimdHogFilterSeparable));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool HogDirectionHistogramsDataTest(bool create, int width, int height, const FuncHDH & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const Point cell(8, 8);
        const size_t quantization = 18;
        const size_t size = quantization*width*height / cell.x / cell.y;
        Buffer32f h1(size, 0), h2(size, 0);

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, cell, quantization, h1.data());

            TEST_SAVE(h1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(h1);

            f.Call(src, cell, quantization, h2.data());

            TEST_SAVE(h2);

            result = result && Compare(h1, h2, EPS, true, 32);
        }

        return result;
    }

    bool HogDirectionHistogramsDataTest(bool create)
    {
        bool result = true;

        result = result && HogDirectionHistogramsDataTest(create, DW, DH, FUNC_HDH(SimdHogDirectionHistograms));

        return result;
    }

    bool HogExtractFeaturesDataTest(bool create, int width, int height, const FuncHEF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        const size_t size = 31 * (width / 8)*(height / 8);
        Buffer32f f1(size, 0), f2(size, 0);

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, f1.data());

            TEST_SAVE(f1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(f1);

            f.Call(src, f2.data());

            TEST_SAVE(f2);

            result = result && Compare(f1, f2, EPS, true, 64);
        }

        return result;
    }

    bool HogExtractFeaturesDataTest(bool create)
    {
        bool result = true;

        result = result && HogExtractFeaturesDataTest(create, DW, DH, FUNC_HEF(SimdHogExtractFeatures));

        return result;
    }

    bool HogDeinterleaveDataTest(bool create, int width, int height, int count, const FuncHD & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << ", " << count << "].");

        View src(width*count, height, View::Float, NULL, TEST_ALIGN(width*count));

        View dst1(width, height*count, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width, height*count, View::Float, NULL, TEST_ALIGN(width));
        FloatPtrs d1(count), d2(count);
        for (int i = 0; i < count; ++i)
        {
            d1[i] = (float*)(dst1.data + dst1.stride*height*i);
            d2[i] = (float*)(dst2.data + dst2.stride*height*i);
        }

        if (create)
        {
            for (int row = 0; row < height; ++row)
                for (int col = 0; col < width; ++col)
                    for (int i = 0; i < count; ++i)
                        src.At<float>(count*col + i, row) = float(row * 100000 + col * 100 + i);

            TEST_SAVE(src);

            f.Call(src, width, count, d1, dst1.stride / 4);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, width, count, d2, dst2.stride / 4);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool HogDeinterleaveDataTest(bool create)
    {
        bool result = true;

        result = result && HogDeinterleaveDataTest(create, DW, DH, 9, FUNC_HD(SimdHogDeinterleave));

        return result;
    }

    bool HogFilterSeparableDataTest(bool create, int width, int height, int add, const FuncHSF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Float, NULL, TEST_ALIGN(width));

        const size_t colSize = 10, rowSize = 10;
        Buffer32f col(colSize), row(rowSize);

        View dstSrc(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        View dstDst1(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));
        View dstDst2(width - rowSize + 1, height - colSize + 1, View::Float, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom32f(src, -10.0f, 10.0f);
            FillRandom(col, -1.0f, 1.0f);
            FillRandom(row, -1.0f, 1.0f);
            FillRandom32f(dstSrc, -10.0f, 10.0f);

            TEST_SAVE(src);
            TEST_SAVE(col);
            TEST_SAVE(row);
            TEST_SAVE(dstSrc);

            f.Call(src, col, row, dstSrc, dstDst1, add);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(col);
            TEST_LOAD(row);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, col, row, dstSrc, dstDst2, add);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, EPS, true, 64, false);
        }

        return result;
    }

    bool HogFilterSeparableDataTest(bool create)
    {
        bool result = true;

        result = result && HogFilterSeparableDataTest(create, DW, DH, true, FUNC_HSF(SimdHogFilterSeparable));

        return result;
    }
}
