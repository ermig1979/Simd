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
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        const __m512i K16_LINEAR_ROUND_TERM = SIMD_MM512_SET1_EPI16(Base::LINEAR_ROUND_TERM);
        const __m512i K16_BILINEAR_ROUND_TERM = SIMD_MM512_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        const int BILINEAR_SHIFT_EVEN = Base::BILINEAR_SHIFT - 1;
        const int BILINEAR_ROUND_TERM_EVEN = 1 << (BILINEAR_SHIFT_EVEN - 1);
        const __m512i K16_BILINEAR_ROUND_TERM_EVEN = SIMD_MM512_SET1_EPI16(BILINEAR_ROUND_TERM_EVEN);

        SIMD_INLINE __m512i Interpolate(__m512i s[2][2], __m512i k[2][2])
        {
            __m512i sum0 = _mm512_add_epi16(_mm512_mullo_epi16(s[0][0], k[0][0]), _mm512_mullo_epi16(s[0][1], k[0][1]));
            __m512i sum1 = _mm512_add_epi16(_mm512_mullo_epi16(s[1][0], k[1][0]), _mm512_mullo_epi16(s[1][1], k[1][1]));
            return _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(sum0, sum1), K16_BILINEAR_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool mask> SIMD_INLINE void LoadBlock(const uint8_t * src, __m512i & lo, __m512i & hi, __mmask64 tail = -1)
        {
            const __m512i _src = Load<false, mask>(src, tail);
            lo = UnpackU8<0>(_src);
            hi = UnpackU8<1>(_src);
        }

        template<bool mask> SIMD_INLINE void Interpolate(const uint8_t * src, size_t dx, size_t dy, __m512i k[2][2], uint8_t * dst, __mmask64 tail = -1)
        {
            __m512i s[2][2][2];
            LoadBlock<mask>(src, s[0][0][0], s[1][0][0], tail);
            LoadBlock<mask>(src + dx, s[0][0][1], s[1][0][1], tail);
            LoadBlock<mask>(src + dy, s[0][1][0], s[1][1][0], tail);
            LoadBlock<mask>(src + dy + dx, s[0][1][1], s[1][1][1], tail);
            Store<false, mask>(dst, _mm512_packus_epi16(Interpolate(s[0], k), Interpolate(s[1], k)), tail);
        }

        template<bool mask> SIMD_INLINE void Interpolate(const uint8_t * src, size_t dx, size_t dy, __m512i k[2], uint8_t * dst, __mmask64 tail = -1)
        {
            const __m512i s00 = Load<false, mask>(src, tail);
            const __m512i s01 = Load<false, mask>(src + dx, tail);
            const __m512i s10 = Load<false, mask>(src + dy, tail);
            const __m512i s11 = Load<false, mask>(src + dy + dx, tail);
            __m512i lo = _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<0>(s00, s01), k[0]),
                _mm512_maddubs_epi16(UnpackU8<0>(s10, s11), k[1])), K16_BILINEAR_ROUND_TERM_EVEN), BILINEAR_SHIFT_EVEN);
            __m512i hi = _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<1>(s00, s01), k[0]),
                _mm512_maddubs_epi16(UnpackU8<1>(s10, s11), k[1])), K16_BILINEAR_ROUND_TERM_EVEN), BILINEAR_SHIFT_EVEN);
            Store<false, mask>(dst, _mm512_packus_epi16(lo, hi), tail);
        }

        template<bool mask> SIMD_INLINE void Interpolate(const uint8_t * src, size_t dr, const __m512i & k, uint8_t * dst, __mmask64 tail = -1)
        {
            const __m512i s0 = Load<false, mask>(src, tail);
            const __m512i s1 = Load<false, mask>(src + dr, tail);
            __m512i lo = _mm512_srli_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<0>(s0, s1), k), K16_LINEAR_ROUND_TERM), Base::LINEAR_SHIFT);
            __m512i hi = _mm512_srli_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<1>(s0, s1), k), K16_LINEAR_ROUND_TERM), Base::LINEAR_SHIFT);
            Store<false, mask>(dst, _mm512_packus_epi16(lo, hi), tail);
        }

        void ShiftBilinear(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            int fDx, int fDy, uint8_t *dst, size_t dstStride)
        {
            size_t size = width*channelCount;
            size_t alignedSize = AlignLo(size, A);
            __mmask64 tailMask = TailMask64(size - alignedSize);

            if (fDy)
            {
                if (fDx)
                {
                    if (fDx & fDy & 1)
                    {
                        __m512i k[2][2];
                        k[0][0] = _mm512_set1_epi16((Base::FRACTION_RANGE - fDx)*(Base::FRACTION_RANGE - fDy));
                        k[0][1] = _mm512_set1_epi16(fDx*(Base::FRACTION_RANGE - fDy));
                        k[1][0] = _mm512_set1_epi16((Base::FRACTION_RANGE - fDx)*fDy);
                        k[1][1] = _mm512_set1_epi16(fDx*fDy);
                        for (size_t row = 0; row < height; ++row)
                        {
                            size_t col = 0;
                            for (; col < alignedSize; col += A)
                                Interpolate<false>(src + col, channelCount, srcStride, k, dst + col);
                            if (col < size)
                                Interpolate<true>(src + col, channelCount, srcStride, k, dst + col, tailMask);
                            src += srcStride;
                            dst += dstStride;
                        }
                    }
                    else
                    {
                        __m512i k[2];
                        k[0] = SetInt8((Base::FRACTION_RANGE - fDx)*(Base::FRACTION_RANGE - fDy) / 2, fDx*(Base::FRACTION_RANGE - fDy) / 2);
                        k[1] = SetInt8((Base::FRACTION_RANGE - fDx)*fDy / 2, fDx*fDy / 2);
                        for (size_t row = 0; row < height; ++row)
                        {
                            size_t col = 0;
                            for (; col < alignedSize; col += A)
                                Interpolate<false>(src + col, channelCount, srcStride, k, dst + col);
                            if (col < size)
                                Interpolate<true>(src + col, channelCount, srcStride, k, dst + col, tailMask);
                            src += srcStride;
                            dst += dstStride;
                        }
                    }
                }
                else
                {
                    __m512i k = SetInt8(Base::FRACTION_RANGE - fDy, fDy);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < alignedSize; col += A)
                            Interpolate<false>(src + col, srcStride, k, dst + col);
                        if (col < size)
                            Interpolate<true>(src + col, srcStride, k, dst + col, tailMask);
                        src += srcStride;
                        dst += dstStride;
                    }
                }
            }
            else
            {
                if (fDx)
                {
                    __m512i k = SetInt8(Base::FRACTION_RANGE - fDx, fDx);
                    for (size_t row = 0; row < height; ++row)
                    {
                        size_t col = 0;
                        for (; col < alignedSize; col += A)
                            Interpolate<false>(src + col, channelCount, k, dst + col);
                        if (col < size)
                            Interpolate<true>(src + col, channelCount, k, dst + col, tailMask);
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
                Avx512bw::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
            else
                Base::ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
        }
    }
#endif//SIMD_AVX512bw_ENABLE
}

