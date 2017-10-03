/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        const v128_u16 K16_LINEAR_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::LINEAR_ROUND_TERM);
        const v128_u16 K16_BILINEAR_ROUND_TERM = SIMD_VEC_SET1_EPI16(Base::BILINEAR_ROUND_TERM);
        const v128_u16 K16_BILINEAR_SHIFT = SIMD_VEC_SET1_EPI16(Base::BILINEAR_SHIFT);

        SIMD_INLINE v128_u16 Interpolate(v128_u16 s[2][2], v128_u16 k[2][2])
        {
            v128_u16 r = vec_mladd(s[0][0], k[0][0], K16_BILINEAR_ROUND_TERM);
            r = vec_mladd(s[0][1], k[0][1], r);
            r = vec_mladd(s[1][0], k[1][0], r);
            r = vec_mladd(s[1][1], k[1][1], r);
            return vec_sr(r, K16_BILINEAR_SHIFT);
        }

        SIMD_INLINE v128_u8 Interpolate(v128_u16 s[2][2][2], v128_u16 k[2][2])
        {
            return vec_packs(Interpolate(s[0], k), Interpolate(s[1], k));
        }

        SIMD_INLINE v128_u16 Interpolate(v128_u16 s[2], v128_u16 k[2])
        {
            v128_u16 r = vec_mladd(s[0], k[0], K16_BILINEAR_ROUND_TERM);
            r = vec_mladd(s[1], k[1], r);
            return vec_sr(r, K16_BILINEAR_SHIFT);
        }

        SIMD_INLINE v128_u8 Interpolate(v128_u16 s[2][2], v128_u16 k[2])
        {
            return vec_packs(Interpolate(s[0], k), Interpolate(s[1], k));
        }

        SIMD_INLINE void LoadBlock(const uint8_t * src, v128_u16 & lo, v128_u16 & hi)
        {
            const v128_u8 t = Load<false>(src);
            lo = UnpackLoU8(t);
            hi = UnpackHiU8(t);
        }

        SIMD_INLINE void LoadBlock(const uint8_t *src, size_t dx, size_t dy, v128_u16 s[2][2][2])
        {
            LoadBlock(src, s[0][0][0], s[1][0][0]);
            LoadBlock(src + dx, s[0][0][1], s[1][0][1]);
            LoadBlock(src + dy, s[0][1][0], s[1][1][0]);
            LoadBlock(src + dy + dx, s[0][1][1], s[1][1][1]);
        }

        SIMD_INLINE void LoadBlock(const uint8_t *src, size_t dr, v128_u16 s[2][2])
        {
            LoadBlock(src, s[0][0], s[1][0]);
            LoadBlock(src + dr, s[0][1], s[1][1]);
        }

        void ShiftBilinear(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            int fDx, int fDy, uint8_t *dst, size_t dstStride)
        {
            size_t size = width*channelCount;
            size_t alignedSize = AlignLo(size, A);

            if (fDy)
            {
                if (fDx)
                {
                    v128_u16 k[2][2], s[2][2][2];
                    k[0][0] = SetU16((Base::FRACTION_RANGE - fDx)*(Base::FRACTION_RANGE - fDy));
                    k[0][1] = SetU16(fDx*(Base::FRACTION_RANGE - fDy));
                    k[1][0] = SetU16((Base::FRACTION_RANGE - fDx)*fDy);
                    k[1][1] = SetU16(fDx*fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < alignedSize; col += A)
                        {
                            LoadBlock(src + col, channelCount, srcStride, s);
                            Store<false>(dst + col, Interpolate(s, k));
                        }
                        if (size != alignedSize)
                        {
                            LoadBlock(src + size - A, channelCount, srcStride, s);
                            Store<false>(dst + size - A, Interpolate(s, k));
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    v128_u16 k[2], s[2][2];
                    k[0] = SetU16(Base::FRACTION_RANGE - fDy);
                    k[1] = SetU16(fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < alignedSize; col += A)
                        {
                            LoadBlock(src + col, srcStride, s);
                            Store<false>(dst + col, Interpolate(s, k));
                        }
                        if (size != alignedSize)
                        {
                            LoadBlock(src + size - A, srcStride, s);
                            Store<false>(dst + size - A, Interpolate(s, k));
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
                    v128_u16 k[2], s[2][2];
                    k[0] = SetU16(Base::FRACTION_RANGE - fDx);
                    k[1] = SetU16(fDx);
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < alignedSize; col += A)
                        {
                            LoadBlock(src + col, channelCount, s);
                            Store<false>(dst + col, Interpolate(s, k));
                        }
                        if (size != alignedSize)
                        {
                            LoadBlock(src + size - A, channelCount, s);
                            Store<false>(dst + size - A, Interpolate(s, k));
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
            const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride)
        {
            int fDx, fDy;
            Base::CommonShiftAction(src, srcStride, width, height, channelCount, bkg, bkgStride, shiftX, shiftY,
                cropLeft, cropTop, cropRight, cropBottom, dst, dstStride, fDx, fDy);

            if (*shiftX + A < cropRight - cropLeft)
                Vmx::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
            else
                Base::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
        }

    }
#endif// SIMD_VMX_ENABLE
}
