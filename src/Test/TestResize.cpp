/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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

#include "Simd/SimdResizer.h"

namespace Test
{
    String ToString(SimdResizeMethodType method)
    {
        switch (method)
        {
        case SimdResizeMethodNearest: return "NrO";
        case SimdResizeMethodNearestPytorch: return "NrP";
        case SimdResizeMethodBilinear: return "BlO";
        case SimdResizeMethodBilinearCaffe: return "BlC";
        case SimdResizeMethodBilinearPytorch: return "BlP";
        case SimdResizeMethodBicubic: return "BcO";
        case SimdResizeMethodArea: return "ArO";
        case SimdResizeMethodAreaFast: return "ArF";
        default: assert(0); return "";
        }
    }

    String ToString(SimdResizeChannelType type)
    {
        switch (type)
        {
        case SimdResizeChannelByte:  return "int8";
        case SimdResizeChannelShort:  return "int16";
        case SimdResizeChannelFloat:  return "fp32";
        case SimdResizeChannelBf16:  return "bf16";
        default: assert(0); return "";
        }
    }

    namespace
    {
        struct FuncRS
        {
            typedef void*(*FuncPtr)(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);

            FuncPtr func;
            String description;

            FuncRS(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(SimdResizeMethodType method, SimdResizeChannelType type, size_t channels, size_t srcW, size_t srcH, size_t dstW, size_t dstH)
            {
                std::stringstream ss;
                ss << description << "[" << channels << ":" << srcW << "x" << srcH << "->" << dstW << "x" << dstH;
                ss << ":" << ToString(method) << "-" << ToString(type) << "]";
                description = ss.str();
            }

            void Call(const View & src, View & dst, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method) const
            {
                void * resizer = NULL;
                if (src.format == View::Float || src.format == View::Int16)
                    resizer = func(src.width / channels, src.height, dst.width / channels, dst.height, channels, type, method);
                else
                    resizer = func(src.width, src.height, dst.width, dst.height, channels, type, method);
                if (resizer)
                {
                    {
                        TEST_PERFORMANCE_TEST(description);
                        SimdResizerRun(resizer, src.data, src.stride, dst.data, dst.stride);
                    }
                    SimdRelease(resizer);
                }
            }
        };
    }

#define FUNC_RS(function) \
    FuncRS(function, std::string(#function))

//#define TEST_RESIZE_REAL_IMAGE

    bool ResizerAutoTest(SimdResizeMethodType method, SimdResizeChannelType type, size_t channels, size_t srcW, size_t srcH, size_t dstW, size_t dstH, FuncRS f1, FuncRS f2)
    {
        bool result = true;

        f1.Update(method, type, channels, srcW, srcH, dstW, dstH);
        f2.Update(method, type, channels, srcW, srcH, dstW, dstH);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << ".");

        View::Format format;
        if (type == SimdResizeChannelFloat)
        {
            format = View::Float;
            srcW *= channels;
            dstW *= channels;
        }
        else if (type == SimdResizeChannelShort || type == SimdResizeChannelBf16)
        {
            format = View::Int16;
            srcW *= channels;
            dstW *= channels;
        }
        else if (type == SimdResizeChannelByte)
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

        View src(srcW, srcH, format);
        if (type == SimdResizeChannelFloat)
            FillRandom32f(src);
        else if (type == SimdResizeChannelShort)
            FillRandom16u(src);
        else if (type == SimdResizeChannelBf16)
        {
            View src32f(srcW, srcH, View::Float);
            FillRandom32f(src32f, 0.0f, 10.0f);
            for (size_t row = 0; row < srcH; row++)
                SimdFloat32ToBFloat16(src32f.Row<float>(row), srcW, src.Row<uint16_t>(row));
        }
        else
        {
#ifdef TEST_RESIZE_REAL_IMAGE
            ::srand(0);
            FillPicture(src);
#else
            FillRandom(src);
#endif
        }

        View dst1(dstW, dstH, format);
        View dst2(dstW, dstH, format);
        if (format == View::Int16)
        {
            Simd::FillPixel(dst1, uint16_t(0x0001));
            Simd::FillPixel(dst2, uint16_t(0x0002));
        }
        else
        {
            Simd::Fill(dst1, 0x01);
            Simd::Fill(dst2, 0x02);
        }

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, channels, type, method));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, channels, type, method));

        if (type == SimdResizeChannelFloat)
            result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);
        else if (type == SimdResizeChannelBf16)
        {
            View dst32f1(dstW, dstH, View::Float), dst32f2(dstW, dstH, View::Float);
            for (size_t row = 0; row < dstH; row++)
            {
                SimdBFloat16ToFloat32(dst1.Row<uint16_t>(row), dstW, dst32f1.Row<float>(row));
                SimdBFloat16ToFloat32(dst2.Row<uint16_t>(row), dstW, dst32f2.Row<float>(row));
            }
            result = result && Compare(dst32f1, dst32f2, EPS*8.0f, true, 64, DifferenceAbsolute);
        }
        else if(type == SimdResizeChannelShort)
            result = result && Compare(dst1, dst2, 1, true, 64);
        else
            result = result && Compare(dst1, dst2, 0, true, 64);

#if defined(TEST_RESIZE_REAL_IMAGE) && 0
        String suffix = ToString(method) + "_" + ToString(method == SimdResizeMethodBicubic ? SIMD_RESIZER_BICUBIC_BITS : 4);
        if (format == View::Bgr24)
        {
            src.Save(String("src_") + suffix + ".ppm");
            dst1.Save(String("dst_") + suffix + ".ppm");
        }
        if (format == View::Gray8)
        {
            src.Save(String("src_") + suffix + ".pgm", SimdImageFilePgmTxt);
            dst1.Save(String("dst_") + suffix + ".pgm", SimdImageFilePgmTxt);
        }
#endif

        return result;
    }

    bool ResizerAutoTest(SimdResizeMethodType method, SimdResizeChannelType type, int channels, int width, int height, double k, FuncRS f1, FuncRS f2)
    {
        return ResizerAutoTest(method, type, channels, int(width*k), int(height*k), width, height, f1, f2);
    }

    bool ResizerAutoTest(SimdResizeMethodType method, SimdResizeChannelType type, int channels, const FuncRS & f1, const FuncRS & f2)
    {
        bool result = true;

        result = result && ResizerAutoTest(method, type, channels, 124, 93, 319, 239, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 249, 187, 319, 239, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 499, 374, 319, 239, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 999, 749, 319, 239, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 1999, 1499, 319, 239, f1, f2);

#if 0
        result = result && ResizerAutoTest(method, type, channels, 1919, 1081, 299, 168, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 1920, 1080, 299, 168, f1, f2);

        result = result && ResizerAutoTest(method, type, channels, 1920, 1080, 480, 270, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 1920, 1080, 240, 135, f1, f2);

        result = result && ResizerAutoTest(method, type, channels, 960, 540, 480, 270, f1, f2);
        result = result && ResizerAutoTest(method, type, channels, 960, 540, 240, 135, f1, f2);
#endif

        return result;
    }

    bool ResizerAutoTest(const FuncRS & f1, const FuncRS & f2)
    {
        bool result = true;

        //result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelFloat, 64, f1, f2);
        //result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelFloat, 16, f1, f2);
        //result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelFloat, 10, f1, f2);
        //result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelFloat, 3, f1, f2);

        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 64, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 16, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 10, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 4, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 3, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 2, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelBf16, 1, f1, f2);

        return result;

        result = result && ResizerAutoTest(SimdResizeMethodNearest, SimdResizeChannelBf16, 1, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodNearest, SimdResizeChannelBf16, 3, f1, f2);
        result = result && ResizerAutoTest(SimdResizeMethodNearest, SimdResizeChannelBf16, 8, f1, f2);

        //result = result && ResizerAutoTest(SimdResizeMethodAreaFast, SimdResizeChannelByte, 3, 530, 404, 96, 96, f1, f2);
        //result = result && ResizerAutoTest(SimdResizeMethodBilinear, SimdResizeChannelByte, 4, 100, 1, 200, 10, f1, f2);
        //result = result && ResizerAutoTest(SimdResizeMethodBicubic, SimdResizeChannelByte, 4, 100, 2, 200, 10, f1, f2);

#if !defined(__aarch64__) || 1  
        std::vector<SimdResizeMethodType> methods = { SimdResizeMethodNearest, SimdResizeMethodBilinear, SimdResizeMethodBicubic, SimdResizeMethodArea, SimdResizeMethodAreaFast };
        for (size_t m = 0; m < methods.size(); ++m)
        {
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelByte, 1, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelByte, 2, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelByte, 3, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelByte, 4, f1, f2);
            if (methods[m] == SimdResizeMethodBicubic || methods[m] == SimdResizeMethodArea || methods[m] == SimdResizeMethodAreaFast)
                continue;
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelShort, 1, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelShort, 2, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelShort, 3, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelShort, 4, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelFloat, 1, f1, f2);
            result = result && ResizerAutoTest(methods[m], SimdResizeChannelFloat, 3, f1, f2);
        }
#endif

        return result;
    }

    bool ResizerAutoTest()
    {
        bool result = true;

        if (TestBase())
            result = result && ResizerAutoTest(FUNC_RS(Simd::Base::ResizerInit), FUNC_RS(SimdResizerInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41())
            result = result && ResizerAutoTest(FUNC_RS(Simd::Sse41::ResizerInit), FUNC_RS(SimdResizerInit));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2())
            result = result && ResizerAutoTest(FUNC_RS(Simd::Avx2::ResizerInit), FUNC_RS(SimdResizerInit));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw())
            result = result && ResizerAutoTest(FUNC_RS(Simd::Avx512bw::ResizerInit), FUNC_RS(SimdResizerInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && TestNeon())
            result = result && ResizerAutoTest(FUNC_RS(Simd::Neon::ResizerInit), FUNC_RS(SimdResizerInit));
#endif 

        return result;
    }

    //---------------------------------------------------------------------------------------------

    bool ResizeYuv420pSpecialTest(SimdResizeMethodType method)
    {
        bool result = true;

        TEST_LOG_SS(Info, "ResizeYuv420pSpecialTest for " << ToString(method) << " .");

        View bgraSrc(W, H, View::Bgra32);
#ifdef TEST_RESIZE_REAL_IMAGE
        ::srand(0);
        FillPicture(bgraSrc);
#else
        FillRandom(bgraSrc);
#endif
        View yuvSrc(W, H * 3 / 2, View::Gray8, NULL, 1);
        View ySrc(W, H, W, View::Gray8, yuvSrc.data);
        View uSrc(W / 2, H / 2, W / 2, View::Gray8, yuvSrc.data + H * W);
        View vSrc(W / 2, H / 2, W / 2, View::Gray8, yuvSrc.data + H * W * 5 / 4);
        Simd::BgraToYuv420p(bgraSrc, ySrc, uSrc, vSrc);

        View yuvDst(W, H * 3 / 2, View::Gray8, NULL, 1);
        View yDst(W, H, W, View::Gray8, yuvDst.data);
        View uDst(W / 2, H / 2, W / 2, View::Gray8, yuvDst.data + H * W);
        View vDst(W / 2, H / 2, W / 2, View::Gray8, yuvDst.data + H * W * 5 / 4);
        Simd::Fill(yuvDst, 0);

        Rect ySrcRect(0, 0, W, H), uvSrcRect = ySrcRect / 2;
        Rect yDstRect(0, 0, W / 2, H / 2), uvDstRect = yDstRect / 2;

#if 0
        void* yResizer = SimdResizerInit(ySrcRect.Width(), ySrcRect.Height(), 
            yDstRect.Width(), yDstRect.Height(), 1, SimdResizeChannelByte, method);
        void* uvResizer = SimdResizerInit(uvSrcRect.Width(), uvSrcRect.Height(),
            uvDstRect.Width(), uvDstRect.Height(), 1, SimdResizeChannelByte, method);

        SimdResizerRun(yResizer, ySrc.Region(ySrcRect).data, ySrc.stride, yDst.Region(yDstRect).data, yDst.stride);
        SimdResizerRun(uvResizer, uSrc.Region(uvSrcRect).data, uSrc.stride, uDst.Region(uvDstRect).data, uDst.stride);
        SimdResizerRun(uvResizer, vSrc.Region(uvSrcRect).data, vSrc.stride, vDst.Region(uvDstRect).data, vDst.stride);

        SimdRelease(yResizer);
        SimdRelease(uvResizer);
#else
        uint8_t* syuv = yuvSrc.data, * tyuv = yuvDst.data;
        size_t sw = W, tw = W, sh = H, th = H;
        Rect sr = ySrcRect, tr = yDstRect;

        void* rcy, * rcu, * rcv;
        rcy = SimdResizerInit(sr.Width(), sr.Height(), tr.Width(), tr.Height(), 1, SimdResizeChannelByte, method);
        if (rcy) 
        {
            SimdResizerRun(rcy, syuv + sr.Left() + sr.Top() * sw, sw, tyuv + tr.Left() + tr.Top() * tw, tw);
            SimdRelease(rcy);
        }
        rcu = SimdResizerInit(sr.Width() / 2, sr.Height() / 2, tr.Width() / 2, tr.Height() / 2, 1, SimdResizeChannelByte, method);
        if (rcu)
        {
            SimdResizerRun(rcu, syuv + sw * sh + sr.Left() / 2 + sr.Top() * sw / 2, sw / 2, tyuv + tw * th + tr.Left() / 2 + tr.Top() / 2 * tw / 2, tw / 2);
            SimdRelease(rcu);
        }
        rcv = SimdResizerInit(sr.Width() / 2, sr.Height() / 2, tr.Width() / 2, tr.Height() / 2, 1, SimdResizeChannelByte, method);
        if (rcv) 
        {
            SimdResizerRun(rcv, syuv + sw * sh * 5 / 4 + sr.Left() / 2 + sr.Top() * sw / 2, sw / 2, tyuv + tw * th * 5 / 4 + tr.Left() / 2 + tr.Top() / 2 * tw / 2, tw / 2);
            SimdRelease(rcv);
        }
#endif

        View bgraDst(W, H, View::Bgra32);
        Simd::Yuv420pToBgra(yDst, uDst, vDst, bgraDst);

        bgraSrc.Save("src.jpg", SimdImageFileJpeg, 85);
        bgraDst.Region(yDstRect).Save(String("dst_") + ToString(method) + ".jpg", SimdImageFileJpeg, 85);

        return result;
    }

    bool ResizeYuv420pSpecialTest()
    {
        bool result = true;

        result = result && ResizeYuv420pSpecialTest(SimdResizeMethodNearest);

        result = result && ResizeYuv420pSpecialTest(SimdResizeMethodBilinear);

        result = result && ResizeYuv420pSpecialTest(SimdResizeMethodBicubic);

        result = result && ResizeYuv420pSpecialTest(SimdResizeMethodArea);

        result = result && ResizeYuv420pSpecialTest(SimdResizeMethodAreaFast);

        return result;
    }
}
