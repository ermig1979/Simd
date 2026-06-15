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
#include "Simd/SimdYuvToBgr.h"

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

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void AlphaBlending(const svuint8_t& src, uint8_t* dst, const svuint8_t& alpha,
            const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svst1_u8(mask, dst, AlphaBlending(src, svld1_u8(mask, dst), alpha, ialpha, _1));
        }

        template <class T> SIMD_INLINE void AlphaBlendingBgraToYuv420p(const uint8_t* bgra0, size_t bgraStride,
            uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, const svuint16_t& _1,
            const svuint8_t& _255, const svbool_t& maskY, const svbool_t& maskUv)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            svuint8x4_t bgra00 = svld4_u8(maskY, bgra0);
            svuint8x4_t bgra10 = svld4_u8(maskY, bgra1);

            svuint8_t a0 = svget4(bgra00, 3);
            AlphaBlending(BgrToY8<T>(svget4(bgra00, 0), svget4(bgra00, 1), svget4(bgra00, 2)),
                y0, a0, svsub_u8_x(maskY, _255, a0), _1, maskY);

            svuint8_t a1 = svget4(bgra10, 3);
            AlphaBlending(BgrToY8<T>(svget4(bgra10, 0), svget4(bgra10, 1), svget4(bgra10, 2)),
                y1, a1, svsub_u8_x(maskY, _255, a1), _1, maskY);

            svuint16_t blue = AverageUv(svget4(bgra00, 0), svget4(bgra10, 0), maskY);
            svuint16_t green = AverageUv(svget4(bgra00, 1), svget4(bgra10, 1), maskY);
            svuint16_t red = AverageUv(svget4(bgra00, 2), svget4(bgra10, 2), maskY);
            svuint8_t alpha = PackSeqI16ToU8(To16i(AverageUv(a0, a1, maskY)), svundef_s16());
            svuint8_t ialpha = svsub_u8_x(maskUv, _255, alpha);

            AlphaBlending(PackSeqI16ToU8(BgrToU16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()), u, alpha, ialpha, _1, maskUv);
            AlphaBlending(PackSeqI16ToU8(BgrToV16<T>(To16i(blue), To16i(green), To16i(red)), svundef_s16()), v, alpha, ialpha, _1, maskUv);
        }

        template <class T> void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            size_t A = svlen(svuint8_t()), HA = A / 2, A4 = A * 4;
            size_t widthA = AlignLo(width, A), tail = width - widthA;
            const svbool_t bodyY = svptrue_b8();
            const svbool_t bodyUv = svwhilelt_b8(size_t(0), HA);
            const svbool_t tailY = svwhilelt_b8(size_t(0), tail);
            const svbool_t tailUv = svwhilelt_b8(size_t(0), tail / 2);
            const svuint16_t _1 = svdup_n_u16(1);
            const svuint8_t _255 = svdup_n_u8(255);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t colBgra = 0, colY = 0, colUv = 0;
                for (; colY < widthA; colBgra += A4, colY += A, colUv += HA)
                    AlphaBlendingBgraToYuv420p<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, _1, _255, bodyY, bodyUv);
                if (colY < width)
                    AlphaBlendingBgraToYuv420p<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUv, v + colUv, _1, _255, tailY, tailUv);
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: AlphaBlendingBgraToYuv420p<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: AlphaBlendingBgraToYuv420p<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: AlphaBlendingBgraToYuv420p<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: AlphaBlendingBgraToYuv420p<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void MakeAlphaBlendingUniform(const uint8_t* src, uint8_t* dst,
            const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
        {
            svst1_u8(mask, dst, AlphaBlending(svld1_u8(mask, src), svld1_u8(mask, dst), alpha, ialpha, _1));
        }

        void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            size_t size = width * channelCount;
            size_t A = svlen(svuint8_t()), sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(sizeA, size);
            svuint16_t _1 = svdup_n_u16(1);
            svuint8_t _alpha = svdup_n_u8(alpha);
            svuint8_t ialpha = svdup_n_u8(255 - alpha);
            for (size_t row = 0; row < height; ++row)
            {
                size_t offset = 0;
                for (; offset < sizeA; offset += A)
                    MakeAlphaBlendingUniform(src + offset, dst + offset, _alpha, ialpha, _1, body);
                if (sizeA < size)
                    MakeAlphaBlendingUniform(src + offset, dst + offset, _alpha, ialpha, _1, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<size_t channelCount> struct AlphaFiller;

        template<> struct AlphaFiller<1>
        {
            static SIMD_INLINE void Run(uint8_t* dst, const svuint8_t& c0, const svuint8_t& c1, const svuint8_t& c2, const svuint8_t& c3,
                const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
            {
                svst1_u8(mask, dst, AlphaBlending(c0, svld1_u8(mask, dst), alpha, ialpha, _1));
            }
        };

        template<> struct AlphaFiller<2>
        {
            static SIMD_INLINE void Run(uint8_t* dst, const svuint8_t& c0, const svuint8_t& c1, const svuint8_t& c2, const svuint8_t& c3,
                const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
            {
                svuint8x2_t _dst = svld2_u8(mask, dst);
                svst2_u8(mask, dst, svcreate2_u8(
                    AlphaBlending(c0, svget2(_dst, 0), alpha, ialpha, _1),
                    AlphaBlending(c1, svget2(_dst, 1), alpha, ialpha, _1)));
            }
        };

        template<> struct AlphaFiller<3>
        {
            static SIMD_INLINE void Run(uint8_t* dst, const svuint8_t& c0, const svuint8_t& c1, const svuint8_t& c2, const svuint8_t& c3,
                const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
            {
                svuint8x3_t _dst = svld3_u8(mask, dst);
                svst3_u8(mask, dst, svcreate3_u8(
                    AlphaBlending(c0, svget3(_dst, 0), alpha, ialpha, _1),
                    AlphaBlending(c1, svget3(_dst, 1), alpha, ialpha, _1),
                    AlphaBlending(c2, svget3(_dst, 2), alpha, ialpha, _1)));
            }
        };

        template<> struct AlphaFiller<4>
        {
            static SIMD_INLINE void Run(uint8_t* dst, const svuint8_t& c0, const svuint8_t& c1, const svuint8_t& c2, const svuint8_t& c3,
                const svuint8_t& alpha, const svuint8_t& ialpha, const svuint16_t& _1, const svbool_t& mask)
            {
                svuint8x4_t _dst = svld4_u8(mask, dst);
                svst4_u8(mask, dst, svcreate4_u8(
                    AlphaBlending(c0, svget4(_dst, 0), alpha, ialpha, _1),
                    AlphaBlending(c1, svget4(_dst, 1), alpha, ialpha, _1),
                    AlphaBlending(c2, svget4(_dst, 2), alpha, ialpha, _1),
                    AlphaBlending(c3, svget4(_dst, 3), alpha, ialpha, _1)));
            }
        };

        template<size_t channelCount> void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height,
            const svuint8_t& c0, const svuint8_t& c1, const svuint8_t& c2, const svuint8_t& c3, const uint8_t* alpha, size_t alphaStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint16_t _1 = svdup_n_u16(1);
            const svuint8_t _255 = svdup_n_u8(255);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * channelCount)
                {
                    svuint8_t _alpha = svld1_u8(body, alpha + col);
                    AlphaFiller<channelCount>::Run(dst + offset, c0, c1, c2, c3, _alpha, svsub_u8_x(body, _255, _alpha), _1, body);
                }
                if (widthA < width)
                {
                    svuint8_t _alpha = svld1_u8(tail, alpha + col);
                    AlphaFiller<channelCount>::Run(dst + offset, c0, c1, c2, c3, _alpha, svsub_u8_x(tail, _255, _alpha), _1, tail);
                }
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaFilling(uint8_t* dst, size_t dstStride, size_t width, size_t height, const uint8_t* channel, size_t channelCount, const uint8_t* alpha, size_t alphaStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            svuint8_t c0 = svdup_n_u8(channel[0]);
            svuint8_t c1 = c0, c2 = c0, c3 = c0;
            switch (channelCount)
            {
            case 1:
                AlphaFilling<1>(dst, dstStride, width, height, c0, c1, c2, c3, alpha, alphaStride);
                break;
            case 2:
                c1 = svdup_n_u8(channel[1]);
                AlphaFilling<2>(dst, dstStride, width, height, c0, c1, c2, c3, alpha, alphaStride);
                break;
            case 3:
                c1 = svdup_n_u8(channel[1]);
                c2 = svdup_n_u8(channel[2]);
                AlphaFilling<3>(dst, dstStride, width, height, c0, c1, c2, c3, alpha, alphaStride);
                break;
            case 4:
                c1 = svdup_n_u8(channel[1]);
                c2 = svdup_n_u8(channel[2]);
                c3 = svdup_n_u8(channel[3]);
                AlphaFilling<4>(dst, dstStride, width, height, c0, c1, c2, c3, alpha, alphaStride);
                break;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint8_t AlphaPremultiply(const svuint8_t& value, const svuint8_t& alpha, const svuint16_t& _1)
        {
            svuint16_t lo = svmlalb_u16(_1, value, alpha);
            svuint16_t hi = svmlalt_u16(_1, value, alpha);
            lo = svaddwt_u16(lo, To8u(lo));
            hi = svaddwt_u16(hi, To8u(hi));
            return svshrnt_n_u16(svshrnb_n_u16(lo, 8), hi, 8);
        }

        template<bool argb> void AlphaPremultiply(const uint8_t* src, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask);

        template<> SIMD_INLINE void AlphaPremultiply<false>(const uint8_t* src, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x4_t bgra = svld4_u8(mask, src);
            svuint8_t alpha = svget4(bgra, 3);
            svst4_u8(mask, dst, svcreate4_u8(
                AlphaPremultiply(svget4(bgra, 0), alpha, _1),
                AlphaPremultiply(svget4(bgra, 1), alpha, _1),
                AlphaPremultiply(svget4(bgra, 2), alpha, _1),
                alpha));
        }

        template<> SIMD_INLINE void AlphaPremultiply<true>(const uint8_t* src, uint8_t* dst, const svuint16_t& _1, const svbool_t& mask)
        {
            svuint8x4_t argb = svld4_u8(mask, src);
            svuint8_t alpha = svget4(argb, 0);
            svst4_u8(mask, dst, svcreate4_u8(
                alpha,
                AlphaPremultiply(svget4(argb, 1), alpha, _1),
                AlphaPremultiply(svget4(argb, 2), alpha, _1),
                AlphaPremultiply(svget4(argb, 3), alpha, _1)));
        }

        template<bool argb> void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint16_t _1 = svdup_n_u16(1);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * 4)
                    AlphaPremultiply<argb>(src + offset, dst + offset, _1, body);
                if (widthA < width)
                    AlphaPremultiply<argb>(src + offset, dst + offset, _1, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaPremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaPremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE svuint32_t AlphaUnpremultiply32(const svuint32_t& value, const svuint32_t& alpha,
            const svfloat32_t& _0, const svfloat32_t& _255)
        {
            const svbool_t mask = svptrue_b32();
            svfloat32_t scale = svdiv_f32_x(mask, _255, svcvt_f32_u32_x(mask, svmax_n_u32_x(mask, alpha, 1)));
            scale = svsel_f32(svcmpeq_n_u32(mask, alpha, 0), _0, scale);
            return svcvt_u32_f32_x(mask, svmin_f32_x(mask, svmul_f32_x(mask, svcvt_f32_u32_x(mask, value), scale), _255));
        }

        SIMD_INLINE svuint8_t AlphaUnpremultiply(const svuint8_t& value, const svuint8_t& alpha,
            const svfloat32_t& _0, const svfloat32_t& _255)
        {
            svuint16_t valueLo = svmovlb_u16(value);
            svuint16_t valueHi = svmovlt_u16(value);
            svuint16_t alphaLo = svmovlb_u16(alpha);
            svuint16_t alphaHi = svmovlt_u16(alpha);

            svuint16_t lo = svqxtnt_u32(svqxtnb_u32(AlphaUnpremultiply32(svmovlb_u32(valueLo), svmovlb_u32(alphaLo), _0, _255)),
                AlphaUnpremultiply32(svmovlt_u32(valueLo), svmovlt_u32(alphaLo), _0, _255));
            svuint16_t hi = svqxtnt_u32(svqxtnb_u32(AlphaUnpremultiply32(svmovlb_u32(valueHi), svmovlb_u32(alphaHi), _0, _255)),
                AlphaUnpremultiply32(svmovlt_u32(valueHi), svmovlt_u32(alphaHi), _0, _255));
            return svqxtnt_u16(svqxtnb_u16(lo), hi);
        }

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, uint8_t* dst,
            const svfloat32_t& _0, const svfloat32_t& _255, const svbool_t& mask);

        template<> SIMD_INLINE void AlphaUnpremultiply<false>(const uint8_t* src, uint8_t* dst,
            const svfloat32_t& _0, const svfloat32_t& _255, const svbool_t& mask)
        {
            svuint8x4_t bgra = svld4_u8(mask, src);
            svuint8_t alpha = svget4(bgra, 3);
            svst4_u8(mask, dst, svcreate4_u8(
                AlphaUnpremultiply(svget4(bgra, 0), alpha, _0, _255),
                AlphaUnpremultiply(svget4(bgra, 1), alpha, _0, _255),
                AlphaUnpremultiply(svget4(bgra, 2), alpha, _0, _255),
                alpha));
        }

        template<> SIMD_INLINE void AlphaUnpremultiply<true>(const uint8_t* src, uint8_t* dst,
            const svfloat32_t& _0, const svfloat32_t& _255, const svbool_t& mask)
        {
            svuint8x4_t argb = svld4_u8(mask, src);
            svuint8_t alpha = svget4(argb, 0);
            svst4_u8(mask, dst, svcreate4_u8(
                alpha,
                AlphaUnpremultiply(svget4(argb, 1), alpha, _0, _255),
                AlphaUnpremultiply(svget4(argb, 2), alpha, _0, _255),
                AlphaUnpremultiply(svget4(argb, 3), alpha, _0, _255)));
        }

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride)
        {
            size_t A = svlen(svuint8_t()), widthA = AlignLo(width, A);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svfloat32_t _0 = svdup_n_f32(0.0f);
            const svfloat32_t _255 = svdup_n_f32(255.00001f);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0, offset = 0;
                for (; col < widthA; col += A, offset += A * 4)
                    AlphaUnpremultiply<argb>(src + offset, dst + offset, _0, _255, body);
                if (widthA < width)
                    AlphaUnpremultiply<argb>(src + offset, dst + offset, _0, _255, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaUnpremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaUnpremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
#endif
}
