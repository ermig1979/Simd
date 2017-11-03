/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
    void FillCircle(View & view)
    {
        assert(view.format == View::Gray8);
        Point c = view.Size() / 2;
        ptrdiff_t r2 = Simd::Square(Simd::Min(view.width, view.height) / 4);
        for (size_t y = 0; y < view.height; ++y)
        {
            ptrdiff_t y2 = Simd::Square(y - c.y);
            uint8_t * data = view.data + view.stride*y;
            for (size_t x = 0; x < view.width; ++x)
            {
                ptrdiff_t x2 = Simd::Square(x - c.x);
                data[x] = x2 + y2 < r2 ? 255 : 0;
            }
        }
    }

    namespace
    {
        struct FuncHLEF
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t cell, float * features, size_t featuresStride);

            FuncPtr func;
            String description;

            FuncHLEF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLEF(const FuncHLEF & f, size_t c) : func(f.func), description(f.description + "[" + ToString(c) + "]") {}

            void Call(const View & src, size_t cell, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.stride, src.width, src.height, cell, (float*)dst.data, dst.stride/sizeof(float));
            }
        };
    }

#define FUNC_HLEF(function) FuncHLEF(function, #function)

#define ARGS_HLEF(cell, f1, f2) cell, FuncHLEF(f1, cell), FuncHLEF(f2, cell)

    bool HogLiteExtractFeaturesAutoTest(size_t width, size_t height, size_t size, size_t cell, const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(src);

        size_t dstX = width / cell - 2;
        size_t dstY = height / cell - 2;
        View dst1(dstX*size, dstY, View::Float, NULL, TEST_ALIGN(width));
        View dst2(dstX*size, dstY, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, cell, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, cell, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteExtractFeaturesAutoTest(const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(W, H, 16, ARGS_HLEF(4, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(W + O, H - O, 16, ARGS_HLEF(4, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(W, H, 16, ARGS_HLEF(8, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(W + O, H - O, 16, ARGS_HLEF(8, f1, f2));

        return result;
    }

    bool HogLiteExtractFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(FUNC_HLEF(Simd::Base::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(FUNC_HLEF(Simd::Sse41::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(FUNC_HLEF(Simd::Avx2::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(FUNC_HLEF(Simd::Avx512bw::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif
        return result;
    }

    namespace
    {
        struct FuncHLFF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHLFF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLFF(const FuncHLFF & f, size_t fis, size_t fes) : func(f.func), description(f.description + "[" + ToString(fis) + "x" + ToString(fes) + "]") {}

            void Call(const View & src, size_t featureSize, const View & filter, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / featureSize, src.height, featureSize,
                    (float*)filter.data, filter.width / featureSize, (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }

#define FUNC_HLFF(function) FuncHLFF(function, #function)

#define ARGS_HLFF(fis, fes, f1, f2) fis, fes, FuncHLFF(f1, fis, fes), FuncHLFF(f2, fis, fes)

    bool HogLiteFilterFeaturesAutoTest(size_t srcWidth, size_t srcHeight, size_t filterSize, size_t featureSize, const FuncHLFF & f1, const FuncHLFF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcWidth << ", " << srcHeight << "].");

        View filter(filterSize*featureSize, filterSize, View::Float, NULL, featureSize*sizeof(float));
        FillRandom32f(filter, 0.5f, 1.5f);

        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize*sizeof(float)));
        FillRandom32f(src, 0.5f, 1.5f);

        size_t dstWidth = srcWidth - filterSize + 1;
        size_t dstHeight = srcHeight - filterSize + 1;
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, featureSize, filter, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, featureSize, filter, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterFeaturesAutoTest(size_t filterSize, size_t featureSize, const FuncHLFF & f1, const FuncHLFF & f2)
    {
        bool result = true;

        result = result && HogLiteFilterFeaturesAutoTest(W / featureSize, H, filterSize, featureSize, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest((W + O)/ featureSize, H - O, filterSize, featureSize, f1, f2);

        return result;
    }

    bool HogLiteFilterFeaturesAutoTest(const FuncHLFF & f1, const FuncHLFF & f2)
    {
        bool result = true;

        result = result && HogLiteFilterFeaturesAutoTest(ARGS_HLFF(8, 16, f1, f2));
        result = result && HogLiteFilterFeaturesAutoTest(ARGS_HLFF(8, 8, f1, f2));

        return result;
    }

    bool HogLiteFilterFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Base::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Sse41::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Avx::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Avx2::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Avx512bw::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLRF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, 
                float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight);

            FuncPtr func;
            String description;

            FuncHLRF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLRF(const FuncHLRF & f, size_t fs) : func(f.func), description(f.description + "[" + ToString(fs) + "]") {}

            void Call(const View & src, size_t featureSize, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / featureSize, src.height, featureSize,
                   (float*)dst.data, dst.stride / sizeof(float), dst.width / featureSize, dst.height);
            }
        };
    }

#define FUNC_HLRF(function) FuncHLRF(function, #function)

#define ARGS_HLRF(fs, f1, f2) fs, FuncHLRF(f1, fs), FuncHLRF(f2, fs)

    bool HogLiteResizeFeaturesAutoTest(size_t srcWidth, size_t srcHeight, double k, size_t featureSize, const FuncHLRF & f1, const FuncHLRF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcWidth << ", " << srcHeight << "].");

        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        FillRandom32f(src, 0.5f, 1.5f);

        size_t dstWidth = size_t(srcWidth*k);
        size_t dstHeight = size_t(srcHeight*k);
        View dst1(dstWidth*featureSize, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth*featureSize, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, featureSize, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, featureSize, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteResizeFeaturesAutoTest(double k, size_t featureSize, const FuncHLRF & f1, const FuncHLRF & f2)
    {
        bool result = true;

        result = result && HogLiteResizeFeaturesAutoTest(W / featureSize, H, k, featureSize, f1, f2);
        result = result && HogLiteResizeFeaturesAutoTest((W + O) / featureSize, H - O, k, featureSize, f1, f2);

        return result;
    }

    bool HogLiteResizeFeaturesAutoTest(const FuncHLRF & f1, const FuncHLRF & f2)
    {
        bool result = true;

        result = result && HogLiteResizeFeaturesAutoTest(0.7, ARGS_HLRF(16, f1, f2));
        result = result && HogLiteResizeFeaturesAutoTest(0.7, ARGS_HLRF(8, f1, f2));

        return result;
    }

    bool HogLiteResizeFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Base::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Sse41::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Avx::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Avx2::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Avx512bw::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLCF
        {
            static const size_t SRC_FEATURE_SIZE = 16;
            static const size_t DST_FEATURE_SIZE = 8;

            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHLCF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const View & pca, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / SRC_FEATURE_SIZE, src.height, (float*)pca.data, (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }

#define FUNC_HLCF(function) FuncHLCF(function, #function)

    bool HogLiteCompressFeaturesAutoTest(size_t width, size_t height, const FuncHLCF & f1, const FuncHLCF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width*FuncHLCF::SRC_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));
        FillRandom32f(src, 0.5f, 1.5f);
        
        View pca(FuncHLCF::SRC_FEATURE_SIZE*FuncHLCF::SRC_FEATURE_SIZE, 1, View::Float, NULL, TEST_ALIGN(FuncHLCF::SRC_FEATURE_SIZE*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));
        FillRandom32f(pca, 0.5f, 1.5f);

        View dst1(width*FuncHLCF::DST_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));
        View dst2(width*FuncHLCF::DST_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, pca, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, pca, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteCompressFeaturesAutoTest(const FuncHLCF & f1, const FuncHLCF & f2)
    {
        bool result = true;

        result = result && HogLiteCompressFeaturesAutoTest(W / FuncHLCF::SRC_FEATURE_SIZE, H, f1, f2);
        result = result && HogLiteCompressFeaturesAutoTest((W + O) / FuncHLCF::SRC_FEATURE_SIZE, H - O, f1, f2);

        return result;
    }

    bool HogLiteCompressFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Base::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Sse41::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Avx::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Avx2::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Avx512bw::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLFS
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHLFS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLFS(const FuncHLFS & f, size_t hs, size_t vs, size_t fs) : func(f.func), description(f.description + "[" + ToString(hs) + "x" + ToString(vs) + "x" + ToString(fs) + "]") {}

            void Call(const View & src, size_t featureSize, const View & hFilter, const View & vFilter, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / featureSize, src.height, featureSize,
                    (float*)hFilter.data, hFilter.width / featureSize, (float*)vFilter.data, vFilter.width, (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }

#define FUNC_HLFS(function) FuncHLFS(function, #function)

#define ARGS_HLFS(hs, vs, fs, f1, f2) hs, vs, fs, FuncHLFS(f1, hs, vs, fs), FuncHLFS(f2, hs, vs, fs)

    bool HogLiteFilterSeparableAutoTest(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize, size_t featureSize, const FuncHLFS & f1, const FuncHLFS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcWidth << ", " << srcHeight << "].");

        View hFilter(hSize*featureSize, 1, View::Float, NULL, featureSize * sizeof(float));
        FillRandom32f(hFilter, 0.5f, 1.5f);

        View vFilter(vSize, 1, View::Float, NULL, sizeof(float));
        FillRandom32f(vFilter, 0.5f, 1.5f);

        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        FillRandom32f(src, 0.5f, 1.5f);

        size_t dstWidth = srcWidth - hSize + 1;
        size_t dstHeight = srcHeight - vSize + 1;
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, featureSize, hFilter, vFilter, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, featureSize, hFilter, vFilter, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterSeparableAutoTest(size_t hSize, size_t vSize, size_t featureSize, const FuncHLFS & f1, const FuncHLFS & f2)
    {
        bool result = true;

        result = result && HogLiteFilterSeparableAutoTest(W / featureSize, H, hSize, vSize, featureSize, f1, f2);
        result = result && HogLiteFilterSeparableAutoTest((W + O) / featureSize, H - O, hSize, vSize, featureSize, f1, f2);

        return result;
    }

    bool HogLiteFilterSeparableAutoTest(const FuncHLFS & f1, const FuncHLFS & f2)
    {
        bool result = true;

        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(8, 8, 16, f1, f2));
        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(8, 8, 8, f1, f2));

        return result;
    }

    bool HogLiteFilterSeparableAutoTest()
    {
        bool result = true;

        result = result && HogLiteFilterSeparableAutoTest(FUNC_HLFS(Simd::Base::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteFilterSeparableAutoTest(FUNC_HLFS(Simd::Sse41::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && HogLiteFilterSeparableAutoTest(FUNC_HLFS(Simd::Avx::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteFilterSeparableAutoTest(FUNC_HLFS(Simd::Avx2::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool HogLiteExtractFeaturesDataTest(bool create, size_t cell, size_t size, int width, int height, const FuncHLEF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        size_t dstX = width / cell - 2;
        size_t dstY = height / cell - 2;
        View dst1(dstX*size, dstY, View::Float, NULL, TEST_ALIGN(width));
        View dst2(dstX*size, dstY, View::Float, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom(src);

            TEST_SAVE(src);

            f.Call(src, cell, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, cell, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool HogLiteExtractFeaturesDataTest(bool create)
    {
        return HogLiteExtractFeaturesDataTest(create, 8, 16, DW, DH, FUNC_HLEF(SimdHogLiteExtractFeatures));
    }

    bool HogLiteFilterFeaturesDataTest(bool create, size_t srcWidth, size_t srcHeight, size_t filterSize, size_t featureSize, const FuncHLFF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcWidth << ", " << srcHeight << "].");

        View filter(filterSize*featureSize, filterSize, View::Float, NULL, featureSize * sizeof(float));
        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        size_t dstWidth = srcWidth - filterSize + 1;
        size_t dstHeight = srcHeight - filterSize + 1;
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        if (create)
        {
            FillRandom32f(filter);
            FillRandom32f(src);

            TEST_SAVE(filter);
            TEST_SAVE(src);

            f.Call(src, featureSize, filter, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(filter);
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, featureSize, filter, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterFeaturesDataTest(bool create)
    {
        return HogLiteFilterFeaturesDataTest(create, DW / 16, DH, 8, 16, FUNC_HLFF(SimdHogLiteFilterFeatures));
    }

    bool HogLiteResizeFeaturesDataTest(bool create, size_t srcWidth, size_t srcHeight, double k, size_t featureSize, const FuncHLRF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcWidth << ", " << srcHeight << "].");

        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        size_t dstWidth = size_t(srcWidth*k);
        size_t dstHeight = size_t(srcHeight*k);
        View dst1(dstWidth*featureSize, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth*featureSize, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        if (create)
        {
            FillRandom32f(src);

            TEST_SAVE(src);

            f.Call(src, featureSize, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, featureSize, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteResizeFeaturesDataTest(bool create)
    {
        return HogLiteResizeFeaturesDataTest(create, DW / 16, DH, 0.7, 16, FUNC_HLRF(SimdHogLiteResizeFeatures));
    }

    bool HogLiteCompressFeaturesDataTest(bool create, size_t width, size_t height, const FuncHLCF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width*FuncHLCF::SRC_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));
        View pca(FuncHLCF::SRC_FEATURE_SIZE*FuncHLCF::SRC_FEATURE_SIZE, 1, View::Float, NULL, TEST_ALIGN(FuncHLCF::SRC_FEATURE_SIZE*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));

        View dst1(width*FuncHLCF::DST_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));
        View dst2(width*FuncHLCF::DST_FEATURE_SIZE, height, View::Float, NULL, TEST_ALIGN(width*FuncHLCF::SRC_FEATURE_SIZE * sizeof(float)));

        if (create)
        {
            FillRandom32f(src);
            FillRandom32f(pca);

            TEST_SAVE(src);
            TEST_SAVE(pca);

            f.Call(src, pca, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(pca);

            TEST_LOAD(dst1);

            f.Call(src, pca, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteCompressFeaturesDataTest(bool create)
    {
        return HogLiteCompressFeaturesDataTest(create, DW / FuncHLCF::SRC_FEATURE_SIZE, DH, FUNC_HLCF(SimdHogLiteCompressFeatures));
    }

    bool HogLiteFilterSeparableDataTest(bool create, size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize, size_t featureSize, const FuncHLFS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcWidth << ", " << srcHeight << "].");

        View hFilter(hSize*featureSize, 1, View::Float, NULL, featureSize * sizeof(float));
        View vFilter(vSize, 1, View::Float, NULL, sizeof(float));
        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        size_t dstWidth = srcWidth - hSize + 1;
        size_t dstHeight = srcHeight - vSize + 1;
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        if (create)
        {
            FillRandom32f(hFilter, 0.5f, 1.5f);
            FillRandom32f(vFilter, 0.5f, 1.5f);
            FillRandom32f(src);

            TEST_SAVE(hFilter);
            TEST_SAVE(vFilter);
            TEST_SAVE(src);

            f.Call(src, featureSize, hFilter, vFilter, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(hFilter);
            TEST_LOAD(vFilter);
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, featureSize, hFilter, vFilter, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterSeparableDataTest(bool create)
    {
        return HogLiteFilterSeparableDataTest(create, DW / 16, DH, 8, 8, 16, FUNC_HLFS(SimdHogLiteFilterSeparable));
    }
}
