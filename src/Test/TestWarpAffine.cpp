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
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdWarpAffine.h"

namespace Test
{
    //String ToString(SimdWarpAffineFlags flags)
    //{
    //    switch (method)
    //    {
    //    case SimdResizeMethodNearest: return "NrO";
    //    case SimdResizeMethodNearestPytorch: return "NrP";
    //    case SimdResizeMethodBilinear: return "BlO";
    //    case SimdResizeMethodBilinearCaffe: return "BlC";
    //    case SimdResizeMethodBilinearPytorch: return "BlP";
    //    case SimdResizeMethodBicubic: return "BcO";
    //    case SimdResizeMethodArea: return "ArO";
    //    case SimdResizeMethodAreaFast: return "ArF";
    //    default: assert(0); return "";
    //    }
    //}

    namespace
    {
        struct FuncWA
        {
            typedef void*(*FuncPtr)(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);

            FuncPtr func;
            String description;

            FuncWA(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags)
            {
                std::stringstream ss;
                ss << description << "[";
                ss << channels << ":" << srcW << "x" << srcH << "->" << dstW << "x" << dstH;
                ss << "]";
                description = ss.str();
            }

            void Call(const View & src, View & dst, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border) const
            {
                void * context = NULL;
                context = func(src.width, src.height, dst.width, dst.height, channels, mat, flags, border);
                if (context)
                {
                    {
                        TEST_PERFORMANCE_TEST(description);
                        SimdWarpAffineRun(context, src.data, src.stride, dst.data, dst.stride);
                    }
                    SimdRelease(context);
                }
            }
        };
    }

#define FUNC_WA(function) \
    FuncWA(function, std::string(#function))

//#define TEST_WARP_AFFINE_REAL_IMAGE

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

        View dst1(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        View dst2(dstW, dstH, format, NULL, TEST_ALIGN(dstW));
        Simd::Fill(dst1, 0x01);
        Simd::Fill(dst2, 0x02);

        uint8_t border[4] = { 1, 2, 3, 4 };

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, channels, mat, flags, border));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, channels, mat, flags, border));

        if (format == View::Float)
            result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);
        else if(format == View::Int16)
            result = result && Compare(dst1, dst2, 1, true, 64);
        else
            result = result && Compare(dst1, dst2, 0, true, 64);

#if defined(TEST_WARP_AFFINE_REAL_IMAGE) && 0
        if (format == View::Bgr24)
        {
            src.Save(String("src.png"), SimdImageFilePng);
            dst1.Save(String("dst.png"), SimdImageFilePng);
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

        result = result && WarpAffineAutoTest(W, H, W, H, channels, Mat(mat, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f), flags, f1, f2);

        return result;
    }

    bool WarpAffineAutoTest(const FuncWA & f1, const FuncWA & f2)
    {
        bool result = true;

        std::vector<SimdWarpAffineFlags> channel = { SimdWarpAffineChannelByte };
        std::vector<SimdWarpAffineFlags> interp = { SimdWarpAffineInterpNearest, SimdWarpAffineInterpBilinear };
        for (size_t c = 0; c < channel.size(); ++c)
        {
            for (size_t i = 0; i < interp.size(); ++i)
            {
                SimdWarpAffineFlags flags = (SimdWarpAffineFlags)(channel[c] | interp[i]);
                result = result && WarpAffineAutoTest(1, flags, f1, f2);
                result = result && WarpAffineAutoTest(2, flags, f1, f2);
                result = result && WarpAffineAutoTest(3, flags, f1, f2);
                result = result && WarpAffineAutoTest(4, flags, f1, f2);
            }
        }

        return result;
    }

    bool WarpAffineAutoTest()
    {
        bool result = true;

        result = result && WarpAffineAutoTest(FUNC_WA(Simd::Base::WarpAffineInit), FUNC_WA(SimdWarpAffineInit));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable)
//            result = result && ResizerAutoTest(FUNC_RS(Simd::Sse41::ResizerInit), FUNC_RS(SimdResizerInit));
//#endif
//
//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable)
//            result = result && ResizerAutoTest(FUNC_RS(Simd::Avx2::ResizerInit), FUNC_RS(SimdResizerInit));
//#endif
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable)
//            result = result && ResizerAutoTest(FUNC_RS(Simd::Avx512bw::ResizerInit), FUNC_RS(SimdResizerInit));
//#endif
//
//#ifdef SIMD_NEON_ENABLE
//        if (Simd::Neon::Enable)
//            result = result && ResizerAutoTest(FUNC_RS(Simd::Neon::ResizerInit), FUNC_RS(SimdResizerInit));
//#endif 

        return result;
    }
}
