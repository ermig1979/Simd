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
        SIMD_INLINE svuint8_t Interpolate(const svuint8_t& s00, const svuint8_t& s01, const svuint8_t& s10, const svuint8_t& s11,
            const svuint8_t& k00, const svuint8_t& k01, const svuint8_t& k10, const svuint8_t& k11)
        {
            svuint16_t lo = svmlalb_u16(svdup_n_u16(Base::BILINEAR_ROUND_TERM), s00, k00);
            svuint16_t hi = svmlalt_u16(svdup_n_u16(Base::BILINEAR_ROUND_TERM), s00, k00);
            lo = svmlalb_u16(lo, s01, k01);
            hi = svmlalt_u16(hi, s01, k01);
            lo = svmlalb_u16(lo, s10, k10);
            hi = svmlalt_u16(hi, s10, k10);
            lo = svmlalb_u16(lo, s11, k11);
            hi = svmlalt_u16(hi, s11, k11);
            return svshrnt_n_u16(svshrnb_n_u16(lo, Base::BILINEAR_SHIFT), hi, Base::BILINEAR_SHIFT);
        }

        SIMD_INLINE svuint8_t Interpolate(const svuint8_t& s0, const svuint8_t& s1, const svuint8_t& k0, const svuint8_t& k1)
        {
            svuint16_t lo = svmlalb_u16(svdup_n_u16(Base::LINEAR_ROUND_TERM), s0, k0);
            svuint16_t hi = svmlalt_u16(svdup_n_u16(Base::LINEAR_ROUND_TERM), s0, k0);
            lo = svmlalb_u16(lo, s1, k1);
            hi = svmlalt_u16(hi, s1, k1);
            return svshrnt_n_u16(svshrnb_n_u16(lo, Base::LINEAR_SHIFT), hi, Base::LINEAR_SHIFT);
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
                    svuint8_t k00 = svdup_n_u8((Base::FRACTION_RANGE - fDx) * (Base::FRACTION_RANGE - fDy));
                    svuint8_t k01 = svdup_n_u8(fDx * (Base::FRACTION_RANGE - fDy));
                    svuint8_t k10 = svdup_n_u8((Base::FRACTION_RANGE - fDx) * fDy);
                    svuint8_t k11 = svdup_n_u8(fDx * fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            svuint8_t s00 = svld1_u8(mask, src + col);
                            svuint8_t s01 = svld1_u8(mask, src + col + channelCount);
                            svuint8_t s10 = svld1_u8(mask, src + col + srcStride);
                            svuint8_t s11 = svld1_u8(mask, src + col + srcStride + channelCount);
                            svst1_u8(mask, dst + col, Interpolate(s00, s01, s10, s11, k00, k01, k10, k11));
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    svuint8_t k0 = svdup_n_u8(Base::FRACTION_RANGE - fDy);
                    svuint8_t k1 = svdup_n_u8(fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            svuint8_t s0 = svld1_u8(mask, src + col);
                            svuint8_t s1 = svld1_u8(mask, src + col + srcStride);
                            svst1_u8(mask, dst + col, Interpolate(s0, s1, k0, k1));
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
                    svuint8_t k0 = svdup_n_u8(Base::FRACTION_RANGE - fDx);
                    svuint8_t k1 = svdup_n_u8(fDx);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < size; col += A)
                        {
                            svbool_t mask = svwhilelt_b8(col, size);
                            svuint8_t s0 = svld1_u8(mask, src + col);
                            svuint8_t s1 = svld1_u8(mask, src + col + channelCount);
                            svst1_u8(mask, dst + col, Interpolate(s0, s1, k0, k1));
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

