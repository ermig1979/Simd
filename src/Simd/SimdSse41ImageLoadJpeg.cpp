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
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        static void JpegIdctBlock(const int16_t* src, uint8_t* dst, int stride)
        {
#define jpeg__f2f(x)  ((int) (((x) * 4096 + 0.5)))

            __m128i row0, row1, row2, row3, row4, row5, row6, row7;
            __m128i tmp;

#define dct_rot(out0,out1, x,y,c0,c1) \
      __m128i c0##lo = _mm_unpacklo_epi16((x),(y)); \
      __m128i c0##hi = _mm_unpackhi_epi16((x),(y)); \
      __m128i out0##_l = _mm_madd_epi16(c0##lo, c0); \
      __m128i out0##_h = _mm_madd_epi16(c0##hi, c0); \
      __m128i out1##_l = _mm_madd_epi16(c0##lo, c1); \
      __m128i out1##_h = _mm_madd_epi16(c0##hi, c1)

#define dct_widen(out, in) \
      __m128i out##_l = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), (in)), 4); \
      __m128i out##_h = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), (in)), 4)

#define dct_wadd(out, a, b) \
      __m128i out##_l = _mm_add_epi32(a##_l, b##_l); \
      __m128i out##_h = _mm_add_epi32(a##_h, b##_h)

#define dct_wsub(out, a, b) \
      __m128i out##_l = _mm_sub_epi32(a##_l, b##_l); \
      __m128i out##_h = _mm_sub_epi32(a##_h, b##_h)

#define dct_bfly32o(out0, out1, a,b,bias,s) \
      { \
         __m128i abiased_l = _mm_add_epi32(a##_l, bias); \
         __m128i abiased_h = _mm_add_epi32(a##_h, bias); \
         dct_wadd(sum, abiased, b); \
         dct_wsub(dif, abiased, b); \
         out0 = _mm_packs_epi32(_mm_srai_epi32(sum_l, s), _mm_srai_epi32(sum_h, s)); \
         out1 = _mm_packs_epi32(_mm_srai_epi32(dif_l, s), _mm_srai_epi32(dif_h, s)); \
      }

#define dct_interleave8(a, b) \
      tmp = a; \
      a = _mm_unpacklo_epi8(a, b); \
      b = _mm_unpackhi_epi8(tmp, b)

#define dct_interleave16(a, b) \
      tmp = a; \
      a = _mm_unpacklo_epi16(a, b); \
      b = _mm_unpackhi_epi16(tmp, b)

#define dct_pass(bias,shift) \
      { \
         dct_rot(t2e,t3e, row2,row6, rot0_0,rot0_1); \
         __m128i sum04 = _mm_add_epi16(row0, row4); \
         __m128i dif04 = _mm_sub_epi16(row0, row4); \
         dct_widen(t0e, sum04); \
         dct_widen(t1e, dif04); \
         dct_wadd(x0, t0e, t3e); \
         dct_wsub(x3, t0e, t3e); \
         dct_wadd(x1, t1e, t2e); \
         dct_wsub(x2, t1e, t2e); \
         dct_rot(y0o,y2o, row7,row3, rot2_0,rot2_1); \
         dct_rot(y1o,y3o, row5,row1, rot3_0,rot3_1); \
         __m128i sum17 = _mm_add_epi16(row1, row7); \
         __m128i sum35 = _mm_add_epi16(row3, row5); \
         dct_rot(y4o,y5o, sum17,sum35, rot1_0,rot1_1); \
         dct_wadd(x4, y0o, y4o); \
         dct_wadd(x5, y1o, y5o); \
         dct_wadd(x6, y2o, y5o); \
         dct_wadd(x7, y3o, y4o); \
         dct_bfly32o(row0,row7, x0,x7,bias,shift); \
         dct_bfly32o(row1,row6, x1,x6,bias,shift); \
         dct_bfly32o(row2,row5, x2,x5,bias,shift); \
         dct_bfly32o(row3,row4, x3,x4,bias,shift); \
      }

            const __m128i rot0_0 = SetInt16(Base::JpegIdctK00, Base::JpegIdctK00 + Base::JpegIdctK01);
            const __m128i rot0_1 = SetInt16(Base::JpegIdctK00 + Base::JpegIdctK02, Base::JpegIdctK00);
            const __m128i rot1_0 = SetInt16(Base::JpegIdctK03 + Base::JpegIdctK08, Base::JpegIdctK03);
            const __m128i rot1_1 = SetInt16(Base::JpegIdctK03, Base::JpegIdctK03 + Base::JpegIdctK09);
            const __m128i rot2_0 = SetInt16(Base::JpegIdctK10 + Base::JpegIdctK04, Base::JpegIdctK10);
            const __m128i rot2_1 = SetInt16(Base::JpegIdctK10, Base::JpegIdctK10 + Base::JpegIdctK06);
            const __m128i rot3_0 = SetInt16(Base::JpegIdctK11 + Base::JpegIdctK05, Base::JpegIdctK11);
            const __m128i rot3_1 = SetInt16(Base::JpegIdctK11, Base::JpegIdctK11 + Base::JpegIdctK07);
            const __m128i bias_0 = _mm_set1_epi32(512);
            const __m128i bias_1 = _mm_set1_epi32(65536 + (128 << 17));
            row0 = _mm_load_si128((const __m128i*) (src + 0 * 8));
            row1 = _mm_load_si128((const __m128i*) (src + 1 * 8));
            row2 = _mm_load_si128((const __m128i*) (src + 2 * 8));
            row3 = _mm_load_si128((const __m128i*) (src + 3 * 8));
            row4 = _mm_load_si128((const __m128i*) (src + 4 * 8));
            row5 = _mm_load_si128((const __m128i*) (src + 5 * 8));
            row6 = _mm_load_si128((const __m128i*) (src + 6 * 8));
            row7 = _mm_load_si128((const __m128i*) (src + 7 * 8));
            dct_pass(bias_0, 10);

            {
                dct_interleave16(row0, row4);
                dct_interleave16(row1, row5);
                dct_interleave16(row2, row6);
                dct_interleave16(row3, row7);

                dct_interleave16(row0, row2);
                dct_interleave16(row1, row3);
                dct_interleave16(row4, row6);
                dct_interleave16(row5, row7);

                dct_interleave16(row0, row1);
                dct_interleave16(row2, row3);
                dct_interleave16(row4, row5);
                dct_interleave16(row6, row7);
            }

            dct_pass(bias_1, 17);
            {
                __m128i p0 = _mm_packus_epi16(row0, row1);
                __m128i p1 = _mm_packus_epi16(row2, row3);
                __m128i p2 = _mm_packus_epi16(row4, row5);
                __m128i p3 = _mm_packus_epi16(row6, row7);

                dct_interleave8(p0, p2);
                dct_interleave8(p1, p3);

                dct_interleave8(p0, p1);
                dct_interleave8(p2, p3);

                dct_interleave8(p0, p2);
                dct_interleave8(p1, p3);

                StoreHalf<0>((__m128i*)(dst + 0 * stride), p0);
                StoreHalf<1>((__m128i*)(dst + 1 * stride), p0);
                StoreHalf<0>((__m128i*)(dst + 2 * stride), p2);
                StoreHalf<1>((__m128i*)(dst + 3 * stride), p2);
                StoreHalf<0>((__m128i*)(dst + 4 * stride), p1);
                StoreHalf<1>((__m128i*)(dst + 5 * stride), p1);
                StoreHalf<0>((__m128i*)(dst + 6 * stride), p3);
                StoreHalf<1>((__m128i*)(dst + 7 * stride), p3);
            }

#undef dct_const
#undef dct_rot
#undef dct_widen
#undef dct_wadd
#undef dct_wsub
#undef dct_bfly32o
#undef dct_interleave8
#undef dct_interleave16
#undef dct_pass
        }

        //-------------------------------------------------------------------------------------------------

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
            size_t hL = height - 1, w2 = (width + 1) / 2, wA = AlignLo(width, A);
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < wA; col += A)
                    YuvToBgr(y + col, bu + col, bv + col, bgr + col * 3);
                for (size_t col = wA; col < width; ++col)
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

        SIMD_INLINE void YuvToRgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* rgb)
        {
            __m128i y8 = _mm_loadu_si128((__m128i*)y);
            __m128i u8 = _mm_loadu_si128((__m128i*)u);
            __m128i v8 = _mm_loadu_si128((__m128i*)v);
            __m128i blue = YuvToBlue<Base::Trect871>(y8, u8);
            __m128i green = YuvToGreen<Base::Trect871>(y8, u8, v8);
            __m128i red = YuvToRed<Base::Trect871>(y8, v8);
            _mm_storeu_si128((__m128i*)rgb + 0, InterleaveBgr<0>(red, green, blue));
            _mm_storeu_si128((__m128i*)rgb + 1, InterleaveBgr<1>(red, green, blue));
            _mm_storeu_si128((__m128i*)rgb + 2, InterleaveBgr<2>(red, green, blue));
        }

        void JpegYuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2, wA = AlignLo(width, A);
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < wA; col += A)
                    YuvToRgb(y + col, bu + col, bv + col, rgb + col * 3);
                for (size_t col = wA; col < width; ++col)
                    Base::YuvToRgb<Base::Trect871>(y[col], bu[col], bv[col], rgb + col * 3);
                y += yStride;
                rgb += rgbStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void YuvToBgra16(__m128i y16, __m128i u16, __m128i v16, const __m128i& a_0, __m128i* bgra)
        {
            const __m128i b16 = YuvToBlue16<Base::Trect871>(y16, u16);
            const __m128i g16 = YuvToGreen16<Base::Trect871>(y16, u16, v16);
            const __m128i r16 = YuvToRed16<Base::Trect871>(y16, v16);
            const __m128i bg8 = _mm_or_si128(b16, _mm_slli_si128(g16, 1));
            const __m128i ra8 = _mm_or_si128(r16, a_0);
            _mm_storeu_si128(bgra + 0, _mm_unpacklo_epi16(bg8, ra8));
            _mm_storeu_si128(bgra + 1, _mm_unpackhi_epi16(bg8, ra8));
        }

        SIMD_INLINE void YuvToBgra(const uint8_t* y, const uint8_t* u, const uint8_t* v, const __m128i& a_0, uint8_t* bgra)
        {
            __m128i y8 = _mm_loadu_si128((__m128i*)y);
            __m128i u8 = _mm_loadu_si128((__m128i*)u);
            __m128i v8 = _mm_loadu_si128((__m128i*)v);
            YuvToBgra16(UnpackY<Base::Trect871, 0>(y8), UnpackUV<Base::Trect871, 0>(u8), UnpackUV<Base::Trect871, 0>(v8), a_0, (__m128i*)bgra + 0);
            YuvToBgra16(UnpackY<Base::Trect871, 1>(y8), UnpackUV<Base::Trect871, 1>(u8), UnpackUV<Base::Trect871, 1>(v8), a_0, (__m128i*)bgra + 2);
        }

        void JpegYuv420pToBgra(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* bgra, size_t bgraStride, uint8_t alpha, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2, wA = AlignLo(width, A);
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            __m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < wA; col += A)
                    YuvToBgra(y + col, bu + col, bv + col, a_0, bgra + col * 4);
                for (size_t col = wA; col < width; ++col)
                    Base::YuvToBgra<Base::Trect871>(y[col], bu[col], bv[col], alpha, bgra + col * 4);
                y += yStride;
                bgra += bgraStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void YuvToRgba16(__m128i y16, __m128i u16, __m128i v16, const __m128i& a_0, __m128i* rgba)
        {
            const __m128i b16 = YuvToBlue16<Base::Trect871>(y16, u16);
            const __m128i g16 = YuvToGreen16<Base::Trect871>(y16, u16, v16);
            const __m128i r16 = YuvToRed16<Base::Trect871>(y16, v16);
            const __m128i rg8 = _mm_or_si128(r16, _mm_slli_si128(g16, 1));
            const __m128i ba8 = _mm_or_si128(b16, a_0);
            _mm_storeu_si128(rgba + 0, _mm_unpacklo_epi16(rg8, ba8));
            _mm_storeu_si128(rgba + 1, _mm_unpackhi_epi16(rg8, ba8));
        }

        SIMD_INLINE void YuvToRgba(const uint8_t* y, const uint8_t* u, const uint8_t* v, const __m128i& a_0, uint8_t* rgba)
        {
            __m128i y8 = _mm_loadu_si128((__m128i*)y);
            __m128i u8 = _mm_loadu_si128((__m128i*)u);
            __m128i v8 = _mm_loadu_si128((__m128i*)v);
            YuvToRgba16(UnpackY<Base::Trect871, 0>(y8), UnpackUV<Base::Trect871, 0>(u8), UnpackUV<Base::Trect871, 0>(v8), a_0, (__m128i*)rgba + 0);
            YuvToRgba16(UnpackY<Base::Trect871, 1>(y8), UnpackUV<Base::Trect871, 1>(u8), UnpackUV<Base::Trect871, 1>(v8), a_0, (__m128i*)rgba + 2);
        }

        void JpegYuv420pToRgba(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, uint8_t* rgba, size_t rgbaStride, uint8_t alpha, SimdYuvType)
        {
            size_t hL = height - 1, w2 = (width + 1) / 2, wA = AlignLo(width, A);
            Array8u buf(width * 2 + 6);
            uint8_t* bu = buf.data, * bv = buf.data + width + 3;
            __m128i a_0 = _mm_slli_si128(_mm_set1_epi16(alpha), 1);
            for (size_t row = 0; row < height; row += 1)
            {
                int odd = row & 1;
                JpegResampleRowHv2(bu, u, odd ? (row == hL ? u : u + uStride) : (row == 0 ? u : u - uStride), (int)w2, 0);
                JpegResampleRowHv2(bv, v, odd ? (row == hL ? v : v + uStride) : (row == 0 ? v : v - uStride), (int)w2, 0);
                for (size_t col = 0; col < wA; col += A)
                    YuvToRgba(y + col, bu + col, bv + col, a_0, rgba + col * 4);
                for (size_t col = wA; col < width; ++col)
                    Base::YuvToRgba<Base::Trect871>(y[col], bu[col], bv[col], alpha, rgba + col * 4);
                y += yStride;
                rgba += rgbaStride;
                if (odd)
                {
                    u += uStride;
                    v += vStride;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        ImageJpegLoader::ImageJpegLoader(const ImageLoaderParam& param)
            : Base::ImageJpegLoader(param)
        {
            _context->idctBlock = JpegIdctBlock;
            _context->resampleRowHv2 = JpegResampleRowHv2;
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
                _context->yuv420pToBgra = Sse41::JpegYuv420pToBgra;
                _context->rgbaToAny = Sse41::BgraToRgba;
            }
            if (_param.format == SimdPixelFormatRgb24)
            {
                _context->yuv444pToBgr = Sse41::Yuv444pToRgbV2;
                _context->yuv420pToBgr = Sse41::JpegYuv420pToRgb;
                _context->rgbaToAny = Sse41::BgraToBgr;
            }
            if (_param.format == SimdPixelFormatRgba32)
            {
                _context->yuv444pToBgra = Sse41::Yuv444pToRgbaV2;
                _context->yuv420pToBgra = Sse41::JpegYuv420pToRgba;
            }
        }
    }
#endif
}
