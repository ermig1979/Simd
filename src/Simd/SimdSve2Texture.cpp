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
#include "Simd/SimdMemory.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint8_t TextureBoostedSaturatedGradient(const svuint8_t& a, const svuint8_t& b,
            const svuint8_t& saturation, const svuint8_t& boost, const svbool_t& mask)
        {
            svuint8_t positive = svmin_u8_x(mask, svqsub_u8_x(mask, b, a), saturation);
            svuint8_t negative = svmin_u8_x(mask, svqsub_u8_x(mask, a, b), saturation);
            return svmul_u8_x(mask, svsub_u8_x(mask, svadd_u8_x(mask, saturation, positive), negative), boost);
        }

        SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t* src, size_t stride, uint8_t* dx, uint8_t* dy,
            const svuint8_t& saturation, const svuint8_t& boost, const svbool_t& mask)
        {
            svuint8_t s10 = svld1_u8(mask, src - 1);
            svuint8_t s12 = svld1_u8(mask, src + 1);
            svuint8_t s01 = svld1_u8(mask, src - stride);
            svuint8_t s21 = svld1_u8(mask, src + stride);
            svst1_u8(mask, dx, TextureBoostedSaturatedGradient(s10, s12, saturation, boost, mask));
            svst1_u8(mask, dy, TextureBoostedSaturatedGradient(s01, s21, saturation, boost, mask));
        }

        void TextureBoostedSaturatedGradient(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t* dx, size_t dxStride, uint8_t* dy, size_t dyStride)
        {
            assert(int(2) * saturation * boost <= 0xFF);

            size_t A = svcntb();
            svuint8_t _saturation = svdup_n_u8(saturation);
            svuint8_t _boost = svdup_n_u8(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                dx[0] = 0;
                dy[0] = 0;
                for (size_t col = 1; col < width - 1; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, width - 1);
                    TextureBoostedSaturatedGradient(src + col, srcStride, dx + col, dy + col, _saturation, _boost, mask);
                }
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        SIMD_INLINE void TextureBoostedUv(const uint8_t* src, uint8_t* dst, const svuint8_t& min,
            const svuint8_t& max, const svuint8_t& boost, const svbool_t& mask)
        {
            svuint8_t value = svld1_u8(mask, src);
            value = svmin_u8_x(mask, svmax_u8_x(mask, value, min), max);
            svst1_u8(mask, dst, svmul_u8_x(mask, svsub_u8_x(mask, value, min), boost));
        }

        void TextureBoostedUv(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t* dst, size_t dstStride)
        {
            assert(boost < 0x80);

            size_t A = svcntb();
            int min = 128 - (128 / boost);
            int max = 255 - min;

            svuint8_t _min = svdup_n_u8(min);
            svuint8_t _max = svdup_n_u8(max);
            svuint8_t _boost = svdup_n_u8(boost);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, width);
                    TextureBoostedUv(src + col, dst + col, _min, _max, _boost, mask);
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE void TextureGetDifferenceSum(const uint8_t* src, const uint8_t* lo, const uint8_t* hi,
            const svbool_t& mask, const svuint8_t& zero, svint32_t& sum)
        {
            svuint8_t _src = svsel_u8(mask, svld1_u8(mask, src), zero);
            svuint8_t avg = svsel_u8(mask, svrhadd_u8_x(mask, svld1_u8(mask, lo), svld1_u8(mask, hi)), zero);

            svint16_t diffLo = svsub_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlb_u16(_src)), svreinterpret_s16_u16(svmovlb_u16(avg)));
            svint16_t diffHi = svsub_s16_x(svptrue_b16(), svreinterpret_s16_u16(svmovlt_u16(_src)), svreinterpret_s16_u16(svmovlt_u16(avg)));
            sum = svadd_s32_x(svptrue_b32(), sum, svmovlb_s32(diffLo));
            sum = svadd_s32_x(svptrue_b32(), sum, svmovlt_s32(diffLo));
            sum = svadd_s32_x(svptrue_b32(), sum, svmovlb_s32(diffHi));
            sum = svadd_s32_x(svptrue_b32(), sum, svmovlt_s32(diffHi));
        }

        void TextureGetDifferenceSum(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* lo, size_t loStride, const uint8_t* hi, size_t hiStride, int64_t* sum)
        {
            const size_t A = svcntb();
            const size_t widthA = AlignLo(width, A);
            const size_t blockSize = A << 12;
            const size_t blockCount = (widthA + blockSize - 1) / blockSize;
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            const svuint8_t zero = svdup_n_u8(0);

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t block = 0; block < blockCount; ++block)
                {
                    svint32_t blockSum = svdup_n_s32(0);
                    for (size_t col = block * blockSize, end = Min(col + blockSize, widthA); col < end; col += A)
                        TextureGetDifferenceSum(src + col, lo + col, hi + col, body, zero, blockSum);
                    *sum += svaddv_s32(svptrue_b32(), blockSum);
                }
                if (widthA < width)
                {
                    svint32_t tailSum = svdup_n_s32(0);
                    TextureGetDifferenceSum(src + widthA, lo + widthA, hi + widthA, tail, zero, tailSum);
                    *sum += svaddv_s32(svptrue_b32(), tailSum);
                }
                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        SIMD_INLINE void TexturePerformCompensation(const uint8_t* src, uint8_t* dst, const svuint8_t& shift, bool add, const svbool_t& mask)
        {
            svuint8_t value = svld1_u8(mask, src);
            svst1_u8(mask, dst, add ? svqadd_u8(value, shift) : svqsub_u8(value, shift));
        }

        void TexturePerformCompensation(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t* dst, size_t dstStride)
        {
            assert(shift > -0xFF && shift < 0xFF);

            if (shift == 0)
            {
                if (src != dst)
                {
                    for (size_t row = 0; row < height; ++row)
                    {
                        memcpy(dst, src, width);
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                return;
            }

            const size_t A = svcntb();
            const bool add = shift > 0;
            const svuint8_t _shift = svdup_n_u8(add ? shift : -shift);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; col += A)
                {
                    svbool_t mask = svwhilelt_b8(col, width);
                    TexturePerformCompensation(src + col, dst + col, _shift, add, mask);
                }
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif
}
