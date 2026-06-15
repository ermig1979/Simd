/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdCast.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint8_t AlphaBlending(const svuint8_t& src, const svuint8_t& dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1)
        {
            svuint16_t lo = svmlalb_u16(svmlalb_u16(_1, dst, ialpha), src, alpha);
            svuint16_t hi = svmlalt_u16(svmlalt_u16(_1, dst, ialpha), src, alpha);
            lo = svaddwt_u16(lo, To8u(lo));
            hi = svaddwt_u16(hi, To8u(hi));
            return svshrnt_n_u16(svshrnb_n_u16(lo, 8), hi, 8);
        }

        template<size_t channelCount> void MakeAlphaBlending(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask);

        template<> SIMD_INLINE void MakeAlphaBlending<1>(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svst1_u8(mask, dst, AlphaBlending(svld1_u8(mask, src), svld1_u8(mask, dst), alpha, ialpha, _1));
        }

        template<> SIMD_INLINE void MakeAlphaBlending<2>(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x2_t _src = svld2_u8(mask, src);
            svuint8x2_t _dst = svld2_u8(mask, dst);
            svst2_u8(mask, dst, svcreate2_u8(
                AlphaBlending(svget2(_src, 0), svget2(_dst, 0), alpha, ialpha, _1),
                AlphaBlending(svget2(_src, 1), svget2(_dst, 1), alpha, ialpha, _1)));
        }

        template<> SIMD_INLINE void MakeAlphaBlending<3>(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x3_t _src = svld3_u8(mask, src);
            svuint8x3_t _dst = svld3_u8(mask, dst);
            svst3_u8(mask, dst, svcreate3_u8(
                AlphaBlending(svget3(_src, 0), svget3(_dst, 0), alpha, ialpha, _1),
                AlphaBlending(svget3(_src, 1), svget3(_dst, 1), alpha, ialpha, _1),
                AlphaBlending(svget3(_src, 2), svget3(_dst, 2), alpha, ialpha, _1)));
        }

        template<> SIMD_INLINE void MakeAlphaBlending<4>(const uint8_t* src, uint8_t* dst, const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x4_t _src = svld4_u8(mask, src);
            svuint8x4_t _dst = svld4_u8(mask, dst);
            svst4_u8(mask, dst, svcreate4_u8(
                AlphaBlending(svget4(_src, 0), svget4(_dst, 0), alpha, ialpha, _1),
                AlphaBlending(svget4(_src, 1), svget4(_dst, 1), alpha, ialpha, _1),
                AlphaBlending(svget4(_src, 2), svget4(_dst, 2), alpha, ialpha, _1),
                AlphaBlending(svget4(_src, 3), svget4(_dst, 3), alpha, ialpha, _1)));
        }

        template<size_t channelCount> SIMD_INLINE void MakeAlphaBlending(const uint8_t* src, uint8_t* dst, const uint8_t* alpha, const svuint16_t& _1, const svuint8_t & _255, const svbool_t& mask)
        {
            svuint8_t _alpha = svld1_u8(mask, alpha);
            svuint8_t ialpha = svsub_u8_x(mask, _255, _alpha);
            MakeAlphaBlending<channelCount>(src, dst, _alpha, ialpha, _1, mask);
        }

        template<size_t channelCount> void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* alpha, size_t alphaStride, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint16_t _1 = svdup_n_u16(1);
            svuint8_t _255 = svdup_n_u8(255);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * channelCount)
                    MakeAlphaBlending<channelCount>(src + offset, dst + offset, alpha + col, _1, _255, body);
                if (widthA < width)
                    MakeAlphaBlending<channelCount>(src + offset, dst + offset, alpha + col, _1, _255, tail);
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaBlending(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t* alpha, size_t alphaStride, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlending<1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t AlphaBlending2x(const svuint8_t& src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const svuint8_t& src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, const svuint8_t& dst, const svuint16_t& _1)
        {
            return AlphaBlending(src1, AlphaBlending(src0, dst, alpha0, ialpha0, _1), alpha1, ialpha1, _1);
        }

        template<size_t channelCount> void MakeAlphaBlending2x(const uint8_t* src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const uint8_t* src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask);

        template<> SIMD_INLINE void MakeAlphaBlending2x<1>(const uint8_t* src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const uint8_t* src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svst1_u8(mask, dst, AlphaBlending2x(svld1_u8(mask, src0), alpha0, ialpha0, svld1_u8(mask, src1), alpha1, ialpha1, svld1_u8(mask, dst), _1));
        }

        template<> SIMD_INLINE void MakeAlphaBlending2x<2>(const uint8_t* src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const uint8_t* src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x2_t _src0 = svld2_u8(mask, src0);
            svuint8x2_t _src1 = svld2_u8(mask, src1);
            svuint8x2_t _dst = svld2_u8(mask, dst);
            svst2_u8(mask, dst, svcreate2_u8(
                AlphaBlending2x(svget2(_src0, 0), alpha0, ialpha0, svget2(_src1, 0), alpha1, ialpha1, svget2(_dst, 0), _1),
                AlphaBlending2x(svget2(_src0, 1), alpha0, ialpha0, svget2(_src1, 1), alpha1, ialpha1, svget2(_dst, 1), _1)));
        }

        template<> SIMD_INLINE void MakeAlphaBlending2x<3>(const uint8_t* src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const uint8_t* src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x3_t _src0 = svld3_u8(mask, src0);
            svuint8x3_t _src1 = svld3_u8(mask, src1);
            svuint8x3_t _dst = svld3_u8(mask, dst);
            svst3_u8(mask, dst, svcreate3_u8(
                AlphaBlending2x(svget3(_src0, 0), alpha0, ialpha0, svget3(_src1, 0), alpha1, ialpha1, svget3(_dst, 0), _1),
                AlphaBlending2x(svget3(_src0, 1), alpha0, ialpha0, svget3(_src1, 1), alpha1, ialpha1, svget3(_dst, 1), _1),
                AlphaBlending2x(svget3(_src0, 2), alpha0, ialpha0, svget3(_src1, 2), alpha1, ialpha1, svget3(_dst, 2), _1)));
        }

        template<> SIMD_INLINE void MakeAlphaBlending2x<4>(const uint8_t* src0, const svuint8_t& alpha0, const svuint8_t& ialpha0,
            const uint8_t* src1, const svuint8_t& alpha1, const svuint8_t& ialpha1, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x4_t _src0 = svld4_u8(mask, src0);
            svuint8x4_t _src1 = svld4_u8(mask, src1);
            svuint8x4_t _dst = svld4_u8(mask, dst);
            svst4_u8(mask, dst, svcreate4_u8(
                AlphaBlending2x(svget4(_src0, 0), alpha0, ialpha0, svget4(_src1, 0), alpha1, ialpha1, svget4(_dst, 0), _1),
                AlphaBlending2x(svget4(_src0, 1), alpha0, ialpha0, svget4(_src1, 1), alpha1, ialpha1, svget4(_dst, 1), _1),
                AlphaBlending2x(svget4(_src0, 2), alpha0, ialpha0, svget4(_src1, 2), alpha1, ialpha1, svget4(_dst, 2), _1),
                AlphaBlending2x(svget4(_src0, 3), alpha0, ialpha0, svget4(_src1, 3), alpha1, ialpha1, svget4(_dst, 3), _1)));
        }

        template<size_t channelCount> SIMD_INLINE void MakeAlphaBlending2x(const uint8_t* src0, const uint8_t* alpha0,
            const uint8_t* src1, const uint8_t* alpha1, uint8_t* dst, const svuint16_t& _1, const svuint8_t& _255, const svbool_t& mask)
        {
            svuint8_t _alpha0 = svld1_u8(mask, alpha0);
            svuint8_t _alpha1 = svld1_u8(mask, alpha1);
            svuint8_t ialpha0 = svsub_u8_x(mask, _255, _alpha0);
            svuint8_t ialpha1 = svsub_u8_x(mask, _255, _alpha1);
            MakeAlphaBlending2x<channelCount>(src0, _alpha0, ialpha0, src1, _alpha1, ialpha1, dst, _1, mask);
        }

        template<size_t channelCount> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            svuint16_t _1 = svdup_n_u16(1);
            svuint8_t _255 = svdup_n_u8(255);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * channelCount)
                    MakeAlphaBlending2x<channelCount>(src0 + offset, alpha0 + col, src1 + offset, alpha1 + col, dst + offset, _1, _255, body);
                if (widthA < width)
                    MakeAlphaBlending2x<channelCount>(src0 + offset, alpha0 + col, src1 + offset, alpha1 + col, dst + offset, _1, _255, tail);
                src0 += src0Stride;
                alpha0 += alpha0Stride;
                src1 += src1Stride;
                alpha1 += alpha1Stride;
                dst += dstStride;
            }
        }

        void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlending2x<1>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 2: AlphaBlending2x<2>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 3: AlphaBlending2x<3>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 4: AlphaBlending2x<4>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}
