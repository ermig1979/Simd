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

    bool HogLiteExtractFeaturesAutoTest(size_t A, size_t width, size_t height, size_t size, size_t cell, const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        width = std::max(3 * cell, width);
        height = std::max(3 * cell, height);

        if ((width / cell - 1)*cell < A)
            return result;

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

    bool HogLiteExtractFeaturesAutoTest(size_t A, const FuncHLEF & f1, const FuncHLEF & f2)
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(A, W, H, 16, ARGS_HLEF(4, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(A, W + O, H - O, 16, ARGS_HLEF(4, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(A, W, H, 16, ARGS_HLEF(8, f1, f2));
        result = result && HogLiteExtractFeaturesAutoTest(A, W + O, H - O, 16, ARGS_HLEF(8, f1, f2));

        return result;
    }

    bool HogLiteExtractFeaturesAutoTest()
    {
        bool result = true;

        result = result && HogLiteExtractFeaturesAutoTest(1, FUNC_HLEF(Simd::Base::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(Simd::Sse41::A, FUNC_HLEF(Simd::Sse41::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(Simd::Avx2::A, FUNC_HLEF(Simd::Avx2::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(1, FUNC_HLEF(Simd::Avx512bw::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteExtractFeaturesAutoTest(Simd::Neon::A, FUNC_HLEF(Simd::Neon::HogLiteExtractFeatures), FUNC_HLEF(SimdHogLiteExtractFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLFF
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHLFF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t filterWidth, size_t filterHeight, size_t featureSize, int useMask)
            {
                std::stringstream ss;
                ss << description;
                ss << "[" << filterWidth << "-" << filterHeight << "-" << featureSize << "-" << useMask << "]";
                description = ss.str();
            }

            FuncHLFF(const FuncHLFF & f, size_t fis, size_t fes, int um) : func(f.func), description(f.description + "[" + ToString(fis) + "x" + ToString(fes) + "-" + ToString(um) + "]") {}

            void Call(const View & src, size_t featureSize, const View & filter, const View & mask, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / featureSize, src.height, featureSize,
                    (float*)filter.data, filter.width / featureSize, filter.height, (uint32_t*)mask.data, mask.stride / sizeof(uint32_t),
                    (float*)dst.data, dst.stride / sizeof(float));
            }
        };
    }

#define FUNC_HLFF(function) FuncHLFF(function, #function)

    void FillCorrelatedMask(View & mask, int range)
    {
        uint8_t * data = mask.data;
        size_t size = mask.DataSize();
        while (size)
        {
            size_t length = std::min<size_t>(Random(range)*mask.PixelSize(), size);
            memset(data, Random(2) ? 0xFF : 0x00, length);
            size -= length;
            data += length;
        }
    }

    bool HogLiteFilterFeaturesAutoTest(size_t srcWidth, size_t srcHeight, size_t filterWidth, size_t filterHeight, size_t featureSize, int useMask, FuncHLFF f1, FuncHLFF f2)
    {
        bool result = true;

        f1.Update(filterWidth, filterHeight, featureSize, useMask);
        f2.Update(filterWidth, filterHeight, featureSize, useMask);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcWidth << ", " << srcHeight << "].");

        View filter(filterWidth*featureSize, filterHeight, View::Float, NULL, featureSize*sizeof(float));
        FillRandom32f(filter, 0.5f, 1.5f);

        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize*sizeof(float)));
        FillRandom32f(src, 0.5f, 1.5f);

        size_t dstWidth = srcWidth - filterWidth + 1;
        size_t dstHeight = srcHeight - filterHeight + 1;
        View mask;
        if (useMask)
        {
            mask.Recreate(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(uint32_t)));
            FillCorrelatedMask(mask, 16);
        }
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, featureSize, filter, mask, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, featureSize, filter, mask, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterFeaturesAutoTest(size_t filterWidth, size_t filterHeight, size_t featureSize, int useMask, const FuncHLFF & f1, const FuncHLFF & f2)
    {
        bool result = true;

        result = result && HogLiteFilterFeaturesAutoTest(W / featureSize, H, filterWidth, filterHeight, featureSize, useMask, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest((W + O)/ featureSize, H - O, filterWidth, filterHeight, featureSize, useMask, f1, f2);

        return result;
    }

    bool HogLiteFilterFeaturesAutoTest(const FuncHLFF & f1, const FuncHLFF & f2)
    {
        bool result = true;

        result = result && HogLiteFilterFeaturesAutoTest(8, 8, 16, 1, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(8, 8, 8, 1, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(8, 8, 16, 0, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(8, 8, 8, 0, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(5, 7, 16, 0, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(9, 5, 8, 1, f1, f2);
        result = result && HogLiteFilterFeaturesAutoTest(5, 5, 8, 0, f1, f2);

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

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteFilterFeaturesAutoTest(FUNC_HLFF(Simd::Neon::HogLiteFilterFeatures), FUNC_HLFF(SimdHogLiteFilterFeatures));
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

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteResizeFeaturesAutoTest(FUNC_HLRF(Simd::Neon::HogLiteResizeFeatures), FUNC_HLRF(SimdHogLiteResizeFeatures));
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

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteCompressFeaturesAutoTest(FUNC_HLCF(Simd::Neon::HogLiteCompressFeatures), FUNC_HLCF(SimdHogLiteCompressFeatures));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLFS
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add);

            FuncPtr func;
            String description;

            FuncHLFS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLFS(const FuncHLFS & f, size_t hs, size_t vs, size_t fs, int add) 
                : func(f.func), description(f.description + "[" + ToString(hs) + "x" + ToString(vs) + "x" + ToString(fs) + "-" + ToString(add) + "]") {}

            void Call(const View & src, size_t featureSize, const View & hFilter, const View & vFilter, const View & dstSrc, View & dstDst, int add) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width / featureSize, src.height, featureSize,
                    (float*)hFilter.data, hFilter.width / featureSize, (float*)vFilter.data, vFilter.width, (float*)dstDst.data, dstDst.stride / sizeof(float), add);
            }
        };
    }

#define FUNC_HLFS(function) FuncHLFS(function, #function)

#define ARGS_HLFS(wm, hs, vs, fs, add, f1, f2) wm, hs, vs, fs, add, FuncHLFS(f1, hs, vs, fs, add), FuncHLFS(f2, hs, vs, fs, add)

    bool HogLiteFilterSeparableAutoTest(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize, size_t featureSize, int add, const FuncHLFS & f1, const FuncHLFS & f2)
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
        View dstSrc(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        FillRandom32f(dstSrc, 0.5f, 1.5f);
        View dstDst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dstDst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, featureSize, hFilter, vFilter, dstSrc, dstDst1, add));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, featureSize, hFilter, vFilter, dstSrc, dstDst2, add));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 64);

        return result;
    }

    bool HogLiteFilterSeparableAutoTest(size_t wMin, size_t hSize, size_t vSize, size_t featureSize, int add, const FuncHLFS & f1, const FuncHLFS & f2)
    {
        bool result = true;

        if (W / featureSize < hSize - 1 + wMin)
            return result;
        result = result && HogLiteFilterSeparableAutoTest(W / featureSize, H, hSize, vSize, featureSize, add, f1, f2);
        result = result && HogLiteFilterSeparableAutoTest((W + O) / featureSize, H - O, hSize, vSize, featureSize, add, f1, f2);

        return result;
    }

    bool HogLiteFilterSeparableAutoTest(size_t wMin, const FuncHLFS & f1, const FuncHLFS & f2)
    {
        bool result = true;

        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(wMin, 8, 8, 16, 1, f1, f2));
        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(wMin, 8, 8, 16, 0, f1, f2));
        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(wMin, 8, 8, 8, 1, f1, f2));
        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(wMin, 6, 6, 16, 1, f1, f2));
        result = result && HogLiteFilterSeparableAutoTest(ARGS_HLFS(wMin, 6, 6, 8, 1, f1, f2));

        return result;
    }

    bool HogLiteFilterSeparableAutoTest()
    {
        bool result = true;

        result = result && HogLiteFilterSeparableAutoTest(1, FUNC_HLFS(Simd::Base::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteFilterSeparableAutoTest(Simd::Sse41::F, FUNC_HLFS(Simd::Sse41::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && HogLiteFilterSeparableAutoTest(Simd::Avx::F, FUNC_HLFS(Simd::Avx::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteFilterSeparableAutoTest(Simd::Avx::F, FUNC_HLFS(Simd::Avx2::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteFilterSeparableAutoTest(1, FUNC_HLFS(Simd::Avx512bw::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteFilterSeparableAutoTest(Simd::Neon::F, FUNC_HLFS(Simd::Neon::HogLiteFilterSeparable), FUNC_HLFS(SimdHogLiteFilterSeparable));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLFM
        {
            typedef void(*FuncPtr)(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * value, size_t * col, size_t * row);

            FuncPtr func;
            String description;

            FuncHLFM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, View & value, View & row, View & col) const
            {
                TEST_PERFORMANCE_TEST(description);
                for(size_t i = 0; i < value.width; ++i)
                    func((float*)a.data + i, a.stride / sizeof(float), (float*)b.data + i, b.stride / sizeof(float), a.height, 
                        (float*)value.data + i, (size_t*)col.data + i, (size_t*)row.data + i);
            }
        };
    }

#define FUNC_HLFM(function) FuncHLFM(function, #function)

    bool HogLiteFindMax7x7AutoTest(size_t number, const FuncHLFM & f1, const FuncHLFM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << number << "].");

        View::Format format = sizeof(size_t) == 8 ? View::Int64 : View::Int32;

        View a(number + 6, 7, View::Float, NULL, TEST_ALIGN(number));
        FillRandom32f(a, 0.5f, 1.5f);

        View b(number + 6, 7, View::Float, NULL, TEST_ALIGN(number));
        FillRandom32f(b, 0.5f, 1.5f);

        View value1(number, 1, View::Float, NULL, TEST_ALIGN(number));
        View col1(number, 1, format, NULL, TEST_ALIGN(number));
        View row1(number, 1, format, NULL, TEST_ALIGN(number));

        View value2(number, 1, View::Float, NULL, TEST_ALIGN(number));
        View col2(number, 1, format, NULL, TEST_ALIGN(number));
        View row2(number, 1, format, NULL, TEST_ALIGN(number));


        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, value1, col1, row1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, value2, col2, row2));

        result = result && Compare(value1, value2, EPS, true, 64, "value");
        result = result && Compare(col1, col2, 0, true, 64, 0, "col");
        result = result && Compare(row1, row2, 0, true, 64, 0, "row");

        if (!result)
        {

        }

        return result;
    }

    bool HogLiteFindMax7x7AutoTest()
    {
        bool result = true;

        result = result && HogLiteFindMax7x7AutoTest(W, FUNC_HLFM(Simd::Base::HogLiteFindMax7x7), FUNC_HLFM(SimdHogLiteFindMax7x7));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteFindMax7x7AutoTest(W, FUNC_HLFM(Simd::Sse41::HogLiteFindMax7x7), FUNC_HLFM(SimdHogLiteFindMax7x7));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteFindMax7x7AutoTest(W, FUNC_HLFM(Simd::Avx2::HogLiteFindMax7x7), FUNC_HLFM(SimdHogLiteFindMax7x7));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteFindMax7x7AutoTest(W, FUNC_HLFM(Simd::Neon::HogLiteFindMax7x7), FUNC_HLFM(SimdHogLiteFindMax7x7));
#endif 

        return result;
    }

    namespace
    {
        struct FuncHLCM
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncHLCM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            FuncHLCM(const FuncHLCM & f, size_t si, size_t sc, float th)
                : func(f.func), description(f.description + "[" + ToString(si) + "x" + ToString(sc) + "-" + ToString((double)th, 1, true) + "]") {}


            void Call(const View & src, float threshold, size_t scale, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride / sizeof(float), src.width, src.height, &threshold, scale, size, (uint32_t*)dst.data, dst.stride / sizeof(uint32_t));
            }
        };
    }

#define FUNC_HLCM(function) FuncHLCM(function, #function)

#define ARGS_HLCM(si, sc, th, f1, f2) si, sc, th, FuncHLCM(f1, si, sc, th), FuncHLCM(f2, si, sc, th)

    bool HogLiteCreateMaskAutoTest(size_t srcWidth, size_t srcHeight, size_t size, size_t scale, float threshold, const FuncHLCM & f1, const FuncHLCM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcWidth << ", " << srcHeight << "].");

        View src(srcWidth, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*sizeof(float)));
        FillRandom32f(src, 0.0f, 1.0f);

        size_t dstWidth = srcWidth*scale + size - scale;
        size_t dstHeight = srcHeight*scale + size - scale;
        View dst1(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth * sizeof(float)));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, threshold, scale, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, threshold, scale, size, dst2));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool HogLiteCreateMaskAutoTest(const FuncHLCM & f1, const FuncHLCM & f2)
    {
        bool result = true;

        result = result && HogLiteCreateMaskAutoTest(W, H, ARGS_HLCM(7, 1, 0.9f, f1, f2));
        result = result && HogLiteCreateMaskAutoTest(W, H, ARGS_HLCM(7, 1, 0.5f, f1, f2));
        result = result && HogLiteCreateMaskAutoTest(W, H, ARGS_HLCM(7, 2, 0.9f, f1, f2));
        result = result && HogLiteCreateMaskAutoTest(W, H, ARGS_HLCM(7, 2, 0.5f, f1, f2));
        result = result && HogLiteCreateMaskAutoTest(W + O, H - O, ARGS_HLCM(7, 2, 0.9f, f1, f2));

        return result;
    }

    bool HogLiteCreateMaskAutoTest()
    {
        bool result = true;

        result = result && HogLiteCreateMaskAutoTest(FUNC_HLCM(Simd::Base::HogLiteCreateMask), FUNC_HLCM(SimdHogLiteCreateMask));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && HogLiteCreateMaskAutoTest(FUNC_HLCM(Simd::Sse41::HogLiteCreateMask), FUNC_HLCM(SimdHogLiteCreateMask));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && HogLiteCreateMaskAutoTest(FUNC_HLCM(Simd::Avx2::HogLiteCreateMask), FUNC_HLCM(SimdHogLiteCreateMask));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && HogLiteCreateMaskAutoTest(FUNC_HLCM(Simd::Avx512bw::HogLiteCreateMask), FUNC_HLCM(SimdHogLiteCreateMask));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && HogLiteCreateMaskAutoTest(FUNC_HLCM(Simd::Neon::HogLiteCreateMask), FUNC_HLCM(SimdHogLiteCreateMask));
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
        View mask(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(uint32_t)));
        View dst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        if (create)
        {
            FillRandom32f(src);
            FillRandom32f(filter);
            FillCorrelatedMask(mask, 16);

            TEST_SAVE(src);
            TEST_SAVE(filter);
            TEST_SAVE(mask);

            f.Call(src, featureSize, filter, mask, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(filter);
            TEST_LOAD(mask);

            TEST_LOAD(dst1);

            f.Call(src, featureSize, filter, mask, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool HogLiteFilterFeaturesDataTest(bool create)
    {
        return HogLiteFilterFeaturesDataTest(create, DW, DH, 8, 16, FUNC_HLFF(SimdHogLiteFilterFeatures));
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

        return result;
    }

    bool HogLiteCompressFeaturesDataTest(bool create)
    {
        return HogLiteCompressFeaturesDataTest(create, DW / FuncHLCF::SRC_FEATURE_SIZE, DH, FUNC_HLCF(SimdHogLiteCompressFeatures));
    }

    bool HogLiteFilterSeparableDataTest(bool create, size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize, size_t featureSize, int add, const FuncHLFS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcWidth << ", " << srcHeight << "].");

        View hFilter(hSize*featureSize, 1, View::Float, NULL, featureSize * sizeof(float));
        View vFilter(vSize, 1, View::Float, NULL, sizeof(float));
        View src(srcWidth*featureSize, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        size_t dstWidth = srcWidth - hSize + 1;
        size_t dstHeight = srcHeight - vSize + 1;
        View dstSrc(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dstDst1(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));
        View dstDst2(dstWidth, dstHeight, View::Float, NULL, TEST_ALIGN(srcWidth*featureSize * sizeof(float)));

        if (create)
        {
            FillRandom32f(hFilter, 0.5f, 1.5f);
            FillRandom32f(vFilter, 0.5f, 1.5f);
            FillRandom32f(src);
            FillRandom32f(dstSrc);

            TEST_SAVE(hFilter);
            TEST_SAVE(vFilter);
            TEST_SAVE(src);
            TEST_SAVE(dstSrc);

            f.Call(src, featureSize, hFilter, vFilter, dstSrc, dstDst1, add);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(hFilter);
            TEST_LOAD(vFilter);
            TEST_LOAD(src);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, featureSize, hFilter, vFilter, dstSrc, dstDst2, add);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, EPS, true, 64);
        }

        return result;
    }

    bool HogLiteFilterSeparableDataTest(bool create)
    {
        return HogLiteFilterSeparableDataTest(create, DW, DH, 8, 8, 16, 1, FUNC_HLFS(SimdHogLiteFilterSeparable));
    }

    bool HogLiteFindMax7x7DataTest(bool create, size_t number, const FuncHLFM & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << number << "].");

        View a(number + 6, 7, View::Float, NULL, TEST_ALIGN(number));
        View b(number + 6, 7, View::Float, NULL, TEST_ALIGN(number));

        View value1(number, 1, View::Float, NULL, TEST_ALIGN(number));
        View col1(number, 1, View::Int64, NULL, TEST_ALIGN(number));
        View row1(number, 1, View::Int64, NULL, TEST_ALIGN(number));

        View value2(number, 1, View::Float, NULL, TEST_ALIGN(number));
        View col2(number, 1, View::Int64, NULL, TEST_ALIGN(number));
        View row2(number, 1, View::Int64, NULL, TEST_ALIGN(number));

        if (create)
        {
            FillRandom32f(a, 0.5f, 1.5f);
            FillRandom32f(b, 0.5f, 1.5f);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, value1, col1, row1);

            TEST_SAVE(value1);
            TEST_SAVE(col1);
            TEST_SAVE(row1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(value1);
            TEST_LOAD(col1);
            TEST_LOAD(row1);

            f.Call(a, b, value2, col2, row2);

            TEST_SAVE(value2);
            TEST_SAVE(col2);
            TEST_SAVE(row2);

            result = result && Compare(value1, value2, EPS, true, 64, "value");
            result = result && Compare(col1, col2, 0, true, 64, 0, "col");
            result = result && Compare(row1, row2, 0, true, 64, 0, "row");
        }

        return result;
    }

    bool HogLiteFindMax7x7DataTest(bool create)
    {
        return HogLiteFindMax7x7DataTest(create, DW, FUNC_HLFM(SimdHogLiteFindMax7x7));
    }

    bool HogLiteCreateMaskDataTest(bool create, size_t srcWidth, size_t srcHeight, size_t size, size_t scale, float threshold, const FuncHLCM & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcWidth << ", " << srcHeight << "].");

        View src(srcWidth, srcHeight, View::Float, NULL, TEST_ALIGN(srcWidth * sizeof(float)));

        size_t dstWidth = srcWidth*scale + size - scale;
        size_t dstHeight = srcHeight*scale + size - scale;
        View dst1(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth * sizeof(float)));
        View dst2(dstWidth, dstHeight, View::Int32, NULL, TEST_ALIGN(srcWidth * sizeof(float)));

        if (create)
        {
            FillRandom32f(src, 0.0f, 1.0f);

            TEST_SAVE(src);

            f.Call(src, threshold, scale, size, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, threshold, scale, size, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, 0, true, 64);
        }

        return result;
    }

    bool HogLiteCreateMaskDataTest(bool create)
    {
        return HogLiteCreateMaskDataTest(create, DW, DH, 7, 2, 0.5f, FUNC_HLCM(SimdHogLiteCreateMask));
    }
}
