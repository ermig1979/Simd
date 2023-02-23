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
#include "Test/TestRandom.h"
#include "Test/TestFile.h"

#include "Simd/SimdWarpAffine.h"

#include "Simd/SimdDrawing.hpp"

namespace Test
{
    namespace
    {
        struct FuncWA
        {
            typedef void*(*FuncPtr)(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS,
                size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);

            FuncPtr func;
            String description;

            FuncWA(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags)
            {
                std::stringstream ss;
                ss << description << "[" << channels;
                ss << "-" << ((flags & SimdWarpAffineChannelMask) == SimdWarpAffineChannelByte ? "b" : "?");
                ss << "-" << ((flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpNearest ? "nr" : "bl");
                ss << "-" << ((flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant ? "c" : "t") << "-{ ";
                for(int i = 0; i < 6; ++i)
                    ss << std::setprecision(1) << std::fixed << mat[i] << " ";
                ss << "}:" << srcW << "x" << srcH << "->" << dstW << "x" << dstH << "]";
                description = ss.str();
            }

            void Call(const View & src, View & dst, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, const View & buf) const
            {
                void * context = NULL;
                context = func(src.width, src.height, src.stride, dst.width, dst.height, dst.stride, channels, mat, flags, border);
                if (context)
                {
                    if ((flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpBilinear && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderTransparent)
                        Simd::Copy(buf, dst);
                    {
                        TEST_PERFORMANCE_TEST(description);
                        SimdWarpAffineRun(context, src.data, dst.data);
                    }
                    SimdRelease(context);
                }
            }
        };
    }

#define FUNC_WA(function) \
    FuncWA(function, std::string(#function))

#define TEST_WARP_AFFINE_REAL_IMAGE

    bool SaveImage(const View& image, const String& name)
    {
        View bgr;
        if (image.format != View::Bgr24)
        {
            bgr.Recreate(image.Size(), View::Bgr24);
            if (image.format == View::Uv16)
            {
                for (size_t row = 0; row < image.height; ++row)
                {
                    const uint8_t* src = image.Row<uint8_t>(row);
                    uint8_t* dst = bgr.Row<uint8_t>(row);
                    for (size_t col = 0; col < image.width; ++col, src += 2, dst += 3)
                    {
                        dst[0] = src[0];
                        dst[1] = src[1];
                        dst[2] = 0;
                    }
                }
            }
            else
                Simd::Convert(image, bgr);
        }
        else
            bgr = image;
        const String dir = "_out";
        String path = MakePath(dir, name + ".png");
        return CreatePathIfNotExist(dir, false) && bgr.Save(path, SimdImageFilePng, 0);
    }

    bool WarpAffineAutoTest(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float * mat, SimdWarpAffineFlags flags, FuncWA f1, FuncWA f2)
    {
        bool result = true;

        f1.Update(srcW, srcH, dstW, dstH, channels, mat, flags);
        f2.Update(srcW, srcH, dstW, dstH, channels, mat, flags);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        View::Format format;
        if ((SimdWarpAffineChannelMask & flags) == SimdWarpAffineChannelByte)
        {
            switch (channels)
            {
            case 1: format = View::Gray8; break;
            case 2: format = View::Uv16; break;
            case 3: format = View::Bgr24; break;
            case 4: format = View::Bgra32; break;
            default:
                assert(0);
            }
        }
        else
            assert(0);

        View src(srcW, srcH, format, NULL, TEST_ALIGN(srcW));
        
        if ((SimdWarpAffineChannelMask & flags) == SimdWarpAffineChannelByte)
        {
#ifdef TEST_WARP_AFFINE_REAL_IMAGE
            ::srand(0);
            FillPicture(src);
#else
            FillRandom(src);
#endif
        }

        View buf(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        View dst1(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        View dst2(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        Simd::Fill(dst1, 0x33);
        if((flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant)
            Simd::Fill(dst2, 0x99);
        else
            Simd::Fill(dst2, 0x33);
        Simd::Copy(dst1, buf);
        uint8_t border[4] = { 11, 33, 55, 77 };

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, channels, mat, flags, border, buf));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, channels, mat, flags, border, buf));

#if !((defined(WIN32) && defined(SIMD_X86_ENABLE) && defined(_DEBUG)) || (defined(__clang__) && !defined(NDEBUG)))
        result = result && Compare(dst1, dst2, 0, true, 64);
#endif

#if defined(TEST_WARP_AFFINE_REAL_IMAGE)
        if (!result)
        {
            SaveImage(src, String("src"));
            SaveImage(dst1, String("dst1"));
            SaveImage(dst2, String("dst2"));
        }
#endif

        return result;
    }

    inline float* Mat(Buffer32f & buf, float m00, float m01, float m02, float m10, float m11, float m12)
    {
        buf = Buffer32f({ m00, m01, m02, m10, m11, m12 });
        return buf.data();
    }

    bool WarpAffineAutoTest(int channels, SimdWarpAffineFlags flags, const FuncWA & f1, const FuncWA & f2)
    {
        bool result = true;

        Buffer32f mat;

        //result = result && WarpAffineAutoTest(W, H, W, H, channels, Mat(mat, 0.6f, -0.4f, 0.0f, 0.4f, 0.6f, 0.0f), flags, f1, f2);
        //result = result && WarpAffineAutoTest(W, H, W, H, channels, Mat(mat, 0.7f, -0.7f, float(W / 4), 0.7f, 0.7f, float(-W / 4)), flags, f1, f2);
        result = result && WarpAffineAutoTest(W, H, W, H, channels, Mat(mat, 0.9f, -0.4f, float(W / 6), 0.4f, 0.9f, float(-W / 6)), flags, f1, f2);

        return result;
    }

    bool WarpAffineAutoTest(const FuncWA & f1, const FuncWA & f2)
    {
        bool result = true;

        std::vector<SimdWarpAffineFlags> channel = { SimdWarpAffineChannelByte };
        std::vector<SimdWarpAffineFlags> interp = { SimdWarpAffineInterpNearest, SimdWarpAffineInterpBilinear };
        std::vector<SimdWarpAffineFlags> border = { SimdWarpAffineBorderConstant, SimdWarpAffineBorderTransparent };
        for (size_t c = 0; c < channel.size(); ++c)
        {
            for (size_t i = 0; i < interp.size(); ++i)
            {
                for (size_t b = 0; b < border.size(); ++b)
                {
                    SimdWarpAffineFlags flags = (SimdWarpAffineFlags)(channel[c] | interp[i] | border[b]);
                    result = result && WarpAffineAutoTest(1, flags, f1, f2);
                    result = result && WarpAffineAutoTest(2, flags, f1, f2);
                    result = result && WarpAffineAutoTest(3, flags, f1, f2);
                    result = result && WarpAffineAutoTest(4, flags, f1, f2);
                }
            }
        }

        return result;
    }

    bool WarpAffineAutoTest()
    {
        bool result = true;

        result = result && WarpAffineAutoTest(FUNC_WA(Simd::Base::WarpAffineInit), FUNC_WA(SimdWarpAffineInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && WarpAffineAutoTest(FUNC_WA(Simd::Sse41::WarpAffineInit), FUNC_WA(SimdWarpAffineInit));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && WarpAffineAutoTest(FUNC_WA(Simd::Avx2::WarpAffineInit), FUNC_WA(SimdWarpAffineInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && WarpAffineAutoTest(FUNC_WA(Simd::Avx512bw::WarpAffineInit), FUNC_WA(SimdWarpAffineInit));
#endif

//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable)
//            result = result && ResizerAutoTest(FUNC_RS(Simd::Neon::ResizerInit), FUNC_RS(SimdResizerInit));
//#endif 

        return result;
    }
}

//-------------------------------------------------------------------------------------------------

#ifdef SIMD_OPENCV_ENABLE
#include <opencv2/core/core.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Test
{
    void WarpAffineSimd(const View& src, View& dst, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, const View& buf)
    {
        if ((flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpBilinear && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderTransparent)
            Simd::Copy(buf, dst);
        TEST_PERFORMANCE_TEST("WarpAffineSimd");
        Simd::WarpAffine(src, mat, dst, flags, border);
    }

    void WarpAffineOpenCv(const View& src, View& dst, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, const View& buf)
    {
        cv::Mat cSrc = src, cDst = dst;
        cv::Mat cMat(2, 3, CV_32FC1);
        for (int i = 0; i < 6; ++i)
            ((float*)cMat.data)[i] = mat[i];
        int cFlags = (flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpNearest ?
            cv::INTER_NEAREST : cv::INTER_LINEAR;
        int borderMode = (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant ?
            cv::BORDER_CONSTANT : cv::BORDER_TRANSPARENT;
        cv::Scalar_<float> cBorder;
        for (int i = 0; i < 4; ++i)
            cBorder[i] = border[i];
        if ((flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpBilinear && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderTransparent)
            Simd::Copy(buf, dst);
        TEST_PERFORMANCE_TEST("WarpAffineOpenCV");
        cv::warpAffine(cSrc, cDst, cMat, dst.Size(), cFlags, borderMode, cBorder);
    }

    bool WarpAffineOpenCvSpecialTest(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags)
    {
        bool result = true;

        View::Format format;
        switch (channels)
        {
        case 3: format = View::Bgr24; break;
        default:
            assert(0);
        }

        View src(srcW, srcH, format, NULL, TEST_ALIGN(srcW));
        ::srand(0);
        FillPicture(src);
        Simd::DrawRectangle(src, Rect(0, 0, srcW - 1, srcH - 1), Simd::Pixel::Bgr24(255, 255, 0), 1);

        View buf(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        View dst1(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        View dst2(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        Simd::Fill(buf, 0x77);
        Simd::Fill(dst1, 0x77);
        Simd::Fill(dst2, 0x77);

        uint8_t border[4] = { 11, 33, 55, 77 };

        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
        cv::setNumThreads((int)SimdGetThreadNumber());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(WarpAffineSimd(src, dst1, channels, mat, flags, border, buf));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(WarpAffineOpenCv(src, dst2, channels, mat, flags, border, buf));

        result = result && Compare(dst1, dst2, 0, true, 64);

#if defined(TEST_WARP_AFFINE_REAL_IMAGE)
        if (!result)
        {
            SaveImage(src, String("_src"));
            SaveImage(dst1, String("dst_simd"));
            SaveImage(dst2, String("dst_ocv"));
        }
#endif

        return result;
    }

    bool WarpAffineOpenCvSpecialTest()
    {
        bool result = true;

        std::vector<SimdWarpAffineFlags> channel = { SimdWarpAffineChannelByte };
        std::vector<SimdWarpAffineFlags> interp = { SimdWarpAffineInterpNearest, SimdWarpAffineInterpBilinear };
        std::vector<SimdWarpAffineFlags> border = { SimdWarpAffineBorderConstant, SimdWarpAffineBorderTransparent };
        SimdWarpAffineFlags flags = (SimdWarpAffineFlags)(channel[0] | interp[1] | border[0]);
        Buffer32f mat;

        //result = result && WarpAffineOpenCvSpecialTest(W, H, W, H, 3, Mat(mat, 0.7f, -0.7f, float(W / 4), 0.7f, 0.7f, float(-W / 4)), flags);
        //result = result && WarpAffineOpenCvSpecialTest(W, H, W, H, 3, Mat(mat, 0.7f, -0.7f, 0.0f, 0.7f, 0.7f, 0.0f), flags);
        //result = result && WarpAffineOpenCvSpecialTest(W, H, W, H, 3, Mat(mat, 0.6f, -0.4f, 0.0f, 0.4f, 0.6f, 0.0f), flags);
        //result = result && WarpAffineOpenCvSpecialTest(W, H, W, H, 3, Mat(mat, 0.9f, -0.4f, float(W / 6), 0.4f, 0.9f, float(-W / 6)), flags);
        //result = result && WarpAffineOpenCvSpecialTest(W, H, W, H, 3, Mat(mat, 0.9f, 0.4f, float(-W / 6), -0.4f, 0.9f, float(W / 6)), flags);
        result = result && WarpAffineOpenCvSpecialTest(W, H, int(W * 1.3), int(H * 1.2), 3, Mat(mat, 0.9f, 0.413f, float(+W / 12), -0.413f, 0.9f, float(W / 6)), flags);
        //result = result && WarpAffineOpenCvSpecialTest(W, H, int(W * 1.3), H, 3, Mat(mat, 0.9f, 0.1f, float(+W / 12), -0.1f, 0.9f, float(W / 6)), flags);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, PerformanceMeasurerStorage::s_storage.ConsoleReport(false, true));
        PerformanceMeasurerStorage::s_storage.Clear();
#endif

        return result;
    }
}
#endif