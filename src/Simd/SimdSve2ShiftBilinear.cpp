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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint8_t Interpolate(svuint8_t s[2][2], svuint8_t k[2][2])
        {
            svuint16_t lo = svmlalb_u16(svdup_n_u16(Base::BILINEAR_ROUND_TERM), s[0][0], k[0][0]);
            svuint16_t hi = svmlalt_u16(svdup_n_u16(Base::BILINEAR_ROUND_TERM), s[0][0], k[0][0]);
            lo = svmlalb_u16(lo, s[0][1], k[0][1]);
            hi = svmlalt_u16(hi, s[0][1], k[0][1]);
            lo = svmlalb_u16(lo, s[1][0], k[1][0]);
            hi = svmlalt_u16(hi, s[1][0], k[1][0]);
            lo = svmlalb_u16(lo, s[1][1], k[1][1]);
            hi = svmlalt_u16(hi, s[1][1], k[1][1]);
            return svshrnt_n_u16(svshrnb_n_u16(lo, Base::BILINEAR_SHIFT), hi, Base::BILINEAR_SHIFT);
        }

        SIMD_INLINE svuint8_t Interpolate(svuint8_t s[2], svuint8_t k[2])
        {
            svuint16_t lo = svmlalb_u16(svdup_n_u16(Base::LINEAR_ROUND_TERM), s[0], k[0]);
            svuint16_t hi = svmlalt_u16(svdup_n_u16(Base::LINEAR_ROUND_TERM), s[0], k[0]);
            lo = svmlalb_u16(lo, s[1], k[1]);
            hi = svmlalt_u16(hi, s[1], k[1]);
            return svshrnt_n_u16(svshrnb_n_u16(lo, Base::LINEAR_SHIFT), hi, Base::LINEAR_SHIFT);
        }

        SIMD_INLINE void LoadBlock(const uint8_t* src, size_t dx, size_t dy, svuint8_t s[2][2], const svbool_t& mask)
        {
            s[0][0] = svld1_u8(mask, src);
            s[0][1] = svld1_u8(mask, src + dx);
            s[1][0] = svld1_u8(mask, src + dy);
            s[1][1] = svld1_u8(mask, src + dy + dx);
        }

        SIMD_INLINE void LoadBlock(const uint8_t* src, size_t dr, svuint8_t s[2], const svbool_t& mask)
        {
            s[0] = svld1_u8(mask, src);
            s[1] = svld1_u8(mask, src + dr);
        }

        void ShiftBilinear(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            int fDx, int fDy, uint8_t* dst, size_t dstStride)
        {
            const size_t size = width * channelCount;
            const size_t A = svcntb();
            if (fDy)
            {
                if (fDx)
                {
                    svuint8_t k[2][2], s[2][2];
                    k[0][0] = svdup_n_u8((Base::FRACTION_RANGE - fDx) * (Base::FRACTION_RANGE - fDy));
                    k[0][1] = svdup_n_u8(fDx * (Base::FRACTION_RANGE - fDy));
                    k[1][0] = svdup_n_u8((Base::FRACTION_RANGE - fDx) * fDy);
                    k[1][1] = svdup_n_u8(fDx * fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            LoadBlock(src + col, channelCount, srcStride, s, mask);
                            svst1_u8(mask, dst + col, Interpolate(s, k));
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    svuint8_t k[2], s[2];
                    k[0] = svdup_n_u8(Base::FRACTION_RANGE - fDy);
                    k[1] = svdup_n_u8(fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            LoadBlock(src + col, srcStride, s, mask);
                            svst1_u8(mask, dst + col, Interpolate(s, k));
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
            }
            else
            {
                if (fDx)
                {
                    svuint8_t k[2], s[2];
                    k[0] = svdup_n_u8(Base::FRACTION_RANGE - fDx);
                    k[1] = svdup_n_u8(fDx);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            LoadBlock(src + col, channelCount, s, mask);
                            svst1_u8(mask, dst + col, Interpolate(s, k));
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    for (size_t row = 0; row < height; ++row)
                    {
                        memcpy(dst, src, size);
                        src += srcStride;
                        dst += dstStride;
                    }
                }
            }
        }

        void ShiftBilinear(
            const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t* bkg, size_t bkgStride, const double* shiftX, const double* shiftY,
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t* dst, size_t dstStride)
        {
            int fDx, fDy;
            Base::CommonShiftAction(src, srcStride, width, height, channelCount, bkg, bkgStride, shiftX, shiftY,
                cropLeft, cropTop, cropRight, cropBottom, dst, dstStride, fDx, fDy);

            if (*shiftX + svcntb() < cropRight - cropLeft)
                Sve2::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
            else
                Base::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
        }
    }
#endif
}

