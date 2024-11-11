/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageLoadJpeg.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdInterleave.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
#define jpeg__div4(x) ((uint8_t) ((x) >> 2))
#define jpeg__div16(x) ((uint8_t) ((x) >> 4))

        static uint8_t* JpegResampleRowHv2(uint8_t* out, const uint8_t* in_near, const uint8_t* in_far, int w, int)
        {
            int i, t0, t1;
            if (w == 1)
            {
                out[0] = out[1] = jpeg__div4(3 * in_near[0] + in_far[0] + 2);
                return out;
            }
            t1 = 3 * in_near[0] + in_far[0];
            for (i = 0; i < ((w - 1) & ~7); i += 8)
            {
                __m128i zero = _mm_setzero_si128();
                __m128i farb = _mm_loadl_epi64((__m128i*) (in_far + i));
                __m128i nearb = _mm_loadl_epi64((__m128i*) (in_near + i));
                __m128i farw = _mm_unpacklo_epi8(farb, zero);
                __m128i nearw = _mm_unpacklo_epi8(nearb, zero);
                __m128i diff = _mm_sub_epi16(farw, nearw);
                __m128i nears = _mm_slli_epi16(nearw, 2);
                __m128i curr = _mm_add_epi16(nears, diff);
                __m128i prv0 = _mm_slli_si128(curr, 2);
                __m128i nxt0 = _mm_srli_si128(curr, 2);
                __m128i prev = _mm_insert_epi16(prv0, t1, 0);
                __m128i next = _mm_insert_epi16(nxt0, 3 * in_near[i + 8] + in_far[i + 8], 7);
                __m128i bias = _mm_set1_epi16(8);
                __m128i curs = _mm_slli_epi16(curr, 2);
                __m128i prvd = _mm_sub_epi16(prev, curr);
                __m128i nxtd = _mm_sub_epi16(next, curr);
                __m128i curb = _mm_add_epi16(curs, bias);
                __m128i even = _mm_add_epi16(prvd, curb);
                __m128i odd = _mm_add_epi16(nxtd, curb);
                __m128i int0 = _mm_unpacklo_epi16(even, odd);
                __m128i int1 = _mm_unpackhi_epi16(even, odd);
                __m128i de0 = _mm_srli_epi16(int0, 4);
                __m128i de1 = _mm_srli_epi16(int1, 4);
                __m128i outv = _mm_packus_epi16(de0, de1);
                _mm_storeu_si128((__m128i*) (out + i * 2), outv);
                t1 = 3 * in_near[i + 7] + in_far[i + 7];
            }
            t0 = t1;
            t1 = 3 * in_near[i] + in_far[i];
            out[i * 2] = jpeg__div16(3 * t1 + t0 + 8);
            for (++i; i < w; ++i) 
            {
                t0 = t1;
                t1 = 3 * in_near[i] + in_far[i];
                out[i * 2 - 1] = jpeg__div16(3 * t0 + t1 + 8);
                out[i * 2] = jpeg__div16(3 * t1 + t0 + 8);
            }
            out[w * 2 - 1] = jpeg__div4(t1 + 2);
            return out;
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void YuvToBgr(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* bgr)
        {
            __m128i y8 = _mm_loadu_si128((__m128i*)y);
            __m128i u8 = _mm_loadu_si128((__m128i*)u);
            __m128i v8 = _mm_loadu_si128((__m128i*)v);
            __m128i blue = YuvToBlue<Base::Trect871>(y8, u8);
            __m128i green = YuvToGreen<Base::Trect871>(y8, u8, v8);
            __m128i red = YuvToRed<Base::Trect871>(y8, v8);
            _mm_storeu_si128((__m128i*)bgr + 0, InterleaveBgr<0>(blue, green, red));
            _mm_storeu_si128((__m128i*)bgr + 1, InterleaveBgr<1>(blue, green, red));
            _mm_storeu_si128((__m128i*)bgr + 2, InterleaveBgr<2>(blue, green, red));
        }

        void JpegYuv420pToBgr(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2, widthA = AlignLo(width, A);
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < widthA; col += A)
                    YuvToBgr(y + col, bu + col, bv + col, bgr + col * 3);
                for (size_t col = widthA; col < width; ++col)
                    Base::YuvToBgr<Base::Trect871>(y[col], bu[col], bv[col], bgr + col * 3);
                y += yStride;
                bgr += bgrStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        //void JpegYuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType)
        //{
        //    size_t hL = height - 1, w2 = (width + 1) / 2;
        //    Array8u buf(width * 2 + 6);
        //    uint8_t* bu = buf.data, * bv = buf.data + width + 3;
        //    for (size_t row = 0; row < height; row += 1)
        //    {
        //        int odd = row & 1;
        //        JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
        //        JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
        //        for (size_t col = 0; col < width; ++col)
        //            YuvToRgb<Trect871>(y[col], bu[col], bv[col], rgb + col * 3);
        //        y += yStride;
        //        rgb += rgbStride;
        //        if (odd)
        //        {
        //            u += uStride;
        //            v += vStride;
        //        }
        //    }
        //}

        //void JpegYuv420pToBgra(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType)
        //{
        //    size_t hL = height - 1, w2 = (width + 1) / 2;
        //    Array8u buf(width * 2 + 6);
        //    uint8_t* bu = buf.data, * bv = buf.data + width + 3;
        //    for (size_t row = 0; row < height; row += 1)
        //    {
        //        int odd = row & 1;
        //        JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
        //        JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
        //        for (size_t col = 0; col < width; ++col)
        //            YuvToBgra<Trect871>(y[col], bu[col], bv[col], alpha, bgra + col * 4);
        //        y += yStride;
        //        bgra += bgraStride;
        //        if (odd)
        //        {
        //            u += uStride;
        //            v += vStride;
        //        }
        //    }
        //}

        //void JpegYuv420pToRgba(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType)
        //{
        //    size_t hL = height - 1, w2 = (width + 1) / 2;
        //    Array8u buf(width * 2 + 6);
        //    uint8_t* bu = buf.data, * bv = buf.data + width + 3;
        //    for (size_t row = 0; row < height; row += 1)
        //    {
        //        int odd = row & 1;
        //        JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
        //        JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
        //        for (size_t col = 0; col < width; ++col)
        //            YuvToRgba<Trect871>(y[col], bu[col], bv[col], alpha, rgba + col * 4);
        //        y += yStride;
        //        rgba += rgbaStride;
        //        if (odd)
        //        {
        //            u += uStride;
        //            v += vStride;
        //        }
        //    }
        //}

        //-------------------------------------------------------------------------------------------------

        ImageJpegLoader::ImageJpegLoader(const ImageLoaderParam& param)
            : Base::ImageJpegLoader(param)
        {
            //_context->idctBlock = JpegIdctBlock;
            _context->resampleRowHv2 = JpegResampleRowHv2;
            //_context->yuvToRgbRow = JpegYuvToRgbRow;
            if (_param.format == SimdPixelFormatGray8)
                _context->rgbaToAny = Sse41::RgbaToGray;
            if (_param.format == SimdPixelFormatBgr24)
            {
                _context->yuv444pToBgr = Sse41::Yuv444pToBgrV2;
                _context->yuv420pToBgr = Sse41::JpegYuv420pToBgr;
                _context->rgbaToAny = Sse41::BgraToRgb;
            }
            if (_param.format == SimdPixelFormatBgra32)
            {
                _context->yuv444pToBgra = Sse41::Yuv444pToBgraV2;
                //_context->yuv420pToBgra = Base::JpegYuv420pToBgra;
                _context->rgbaToAny = Sse41::BgraToRgba;
            }
            if (_param.format == SimdPixelFormatRgb24)
            {
                _context->yuv444pToBgr = Sse41::Yuv444pToRgbV2;
            //    _context->yuv420pToBgr = Base::JpegYuv420pToRgb;
                _context->rgbaToAny = Sse41::BgraToBgr;
            }
            if (_param.format == SimdPixelFormatRgba32)
            {
                _context->yuv444pToBgra = Sse41::Yuv444pToRgbaV2;
                //_context->yuv420pToBgra = Base::JpegYuv420pToRgba;
            }
        }
    }
#endif
}
