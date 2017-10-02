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
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void Binarization(const uint8_t * src,
            const __m512i & value, const __m512i & positive, const __m512i & negative, uint8_t * dst, __mmask64 m = -1)
        {
            __mmask64 mm = Compare8u<compareType>(Load<align, mask>(src, m), value);
            Store<align, mask>(dst, _mm512_mask_blend_epi8(mm, negative, positive), m);
        }

        template <bool align, SimdCompareType compareType> SIMD_INLINE void Binarization4(const uint8_t * src,
            const __m512i & value, const __m512i & positive, const __m512i & negative, uint8_t * dst)
        {
            Store<align>(dst + 0 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 0 * A), value), negative, positive));
            Store<align>(dst + 1 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 1 * A), value), negative, positive));
            Store<align>(dst + 2 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 2 * A), value), negative, positive));
            Store<align>(dst + 3 * A, _mm512_mask_blend_epi8(Compare8u<compareType>(Load<align>(src + 3 * A), value), negative, positive));
        }

        template <bool align, SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i _value = _mm512_set1_epi8(value);
            __m512i _positive = _mm512_set1_epi8(positive);
            __m512i _negative = _mm512_set1_epi8(negative);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    Binarization4<align, compareType>(src + col, _value, _positive, _negative, dst + col);
                for (; col < alignedWidth; col += A)
                    Binarization<align, false, compareType>(src + col, _value, _positive, _negative, dst + col);
                if (col < width)
                    Binarization<align, true, compareType>(src + col, _value, _positive, _negative, dst + col, tailMask);
                src += srcStride;
                dst += dstStride;
            }
        }

        template <SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Binarization<true, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            else
                Binarization<false, compareType>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
        }

        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return Binarization<SimdCompareEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareNotEqual:
                return Binarization<SimdCompareNotEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreater:
                return Binarization<SimdCompareGreater>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return Binarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesser:
                return Binarization<SimdCompareLesser>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return Binarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t edge)
                {
                    size_t size = sizeof(uint8_t)*(width + 2 * edge) + sizeof(uint32_t)*(2 * width + 2 * edge);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    s = (uint8_t*)_p + edge;
                    s0a0 = (uint32_t*)(s + width + edge) + edge;
                    sum = (uint32_t*)(s0a0 + width + edge);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint8_t * s;
                uint32_t * s0a0;
                uint32_t * sum;
            private:
                void *_p;
            };
        }

        template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void AddRows(const uint8_t * src, uint8_t * sum, const __m512i & value, __mmask64 tail = -1)
        {
            __mmask64 inc = Compare8u<compareType>(Load<align, mask>(src, tail), value);
            __m512i _sum = Load<true, mask>(sum, tail);
            _sum = _mm512_mask_add_epi8(_sum, inc, _sum, K8_01);
            Store<true, mask>(sum, _sum, tail);
        }

        template <bool align, bool mask, SimdCompareType compareType> SIMD_INLINE void SubRows(const uint8_t * src, uint8_t * sum, const __m512i & value, __mmask64 tail = -1)
        {
            __mmask64 dec = Compare8u<compareType>(Load<align, mask>(src, tail), value);
            __m512i _sum = Load<true, mask>(sum, tail);
            _sum = _mm512_mask_sub_epi8(_sum, dec, _sum, K8_01);
            Store<true, mask>(sum, _sum, tail);
        }

        template <bool mask> SIMD_INLINE void Unpack(const uint8_t * sum, const __m512i & area, uint32_t * s0a0, const __mmask16 * tailMasks)
        {
            const __m512i _sum = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<true>(sum)));
            const __m512i saLo = _mm512_unpacklo_epi8(_sum, area);
            const __m512i saHi = _mm512_unpackhi_epi8(_sum, area);
            Store<true, mask>(s0a0 + 0 * F, _mm512_unpacklo_epi8(saLo, K_ZERO), tailMasks[0]);
            Store<true, mask>(s0a0 + 1 * F, _mm512_unpackhi_epi8(saLo, K_ZERO), tailMasks[1]);
            Store<true, mask>(s0a0 + 2 * F, _mm512_unpacklo_epi8(saHi, K_ZERO), tailMasks[2]);
            Store<true, mask>(s0a0 + 3 * F, _mm512_unpackhi_epi8(saHi, K_ZERO), tailMasks[3]);
        }

        template <bool align, bool mask> SIMD_INLINE void Binarization(const uint32_t * sum, const __m512i & ff_threshold, const __m512i & positive, const __m512i & negative, uint8_t * dst, __mmask64 tail = -1)
        {
            union Mask
            {
                __mmask16 m16[4];
                __mmask64 m64[1];
            } mm;
            mm.m16[0] = _mm512_cmpgt_epi32_mask(_mm512_madd_epi16((Load<true>(sum + 0 * F)), ff_threshold), K_ZERO);
            mm.m16[1] = _mm512_cmpgt_epi32_mask(_mm512_madd_epi16((Load<true>(sum + 1 * F)), ff_threshold), K_ZERO);
            mm.m16[2] = _mm512_cmpgt_epi32_mask(_mm512_madd_epi16((Load<true>(sum + 2 * F)), ff_threshold), K_ZERO);
            mm.m16[3] = _mm512_cmpgt_epi32_mask(_mm512_madd_epi16((Load<true>(sum + 3 * F)), ff_threshold), K_ZERO);
            Store<align, mask>(dst, _mm512_mask_blend_epi8(mm.m64[0], negative, positive), tail);
        }

        template <bool align, SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x7F);

            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __mmask16 tailMasks[4];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask16(width - alignedWidth - F*c);

            const __m512i ff_threshold = SetInt16(0xFF, -threshold);
            const __m512i _value = _mm512_set1_epi8(value);
            const __m512i _positive = _mm512_set1_epi8(positive);
            const __m512i _negative = _mm512_set1_epi8(negative);

            Buffer buffer(AlignHi(width, A), AlignHi(neighborhood + 1, A));
            uint8_t area = 0;
            size_t col = 0;

            for (size_t row = 0; row < neighborhood; ++row)
            {
                area++;
                const uint8_t * s = src + row*srcStride;
                for (col = 0; col < alignedWidth; col += A)
                    AddRows<align, false, compareType>(s + col, buffer.s + col, _value);
                if (col < width)
                    AddRows<align, true, compareType>(s + col, buffer.s + col, _value, tailMask);
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    area++;
                    const uint8_t * s = src + (row + neighborhood)*srcStride;
                    for (col = 0; col < alignedWidth; col += A)
                        AddRows<align, false, compareType>(s + col, buffer.s + col, _value);
                    if (col < width)
                        AddRows<align, true, compareType>(s + col, buffer.s + col, _value, tailMask);
                }
                if (row > neighborhood)
                {
                    area--;
                    const uint8_t * s = src + (row - neighborhood - 1)*srcStride;
                    for (col = 0; col < alignedWidth; col += A)
                        SubRows<align, false, compareType>(s + col, buffer.s + col, _value);
                    if (col < width)
                        SubRows<align, true, compareType>(s + col, buffer.s + col, _value, tailMask);
                }

                __m512i _area = _mm512_set1_epi8(area);
                for (col = 0; col < alignedWidth; col += A)
                    Unpack<false>(buffer.s + col, _area, buffer.s0a0 + col, tailMasks);
                if (col < width)
                    Unpack<true>(buffer.s + col, _area, buffer.s0a0 + col, tailMasks);

                uint32_t sum = 0;
                for (col = 0; col < neighborhood; ++col)
                {
                    sum += buffer.s0a0[col];
                }
                for (col = 0; col < width; ++col)
                {
                    sum += buffer.s0a0[col + neighborhood];
                    sum -= buffer.s0a0[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                }

                for (col = 0; col < alignedWidth; col += A)
                    Binarization<align, false>(buffer.sum + col, ff_threshold, _positive, _negative, dst + col);
                if (col < width)
                    Binarization<align, true>(buffer.sum + col, ff_threshold, _positive, _negative, dst + col, tailMask);

                dst += dstStride;
            }
        }

        template <SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AveragingBinarization<true, compareType>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            else
                AveragingBinarization<false, compareType>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
        }

        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
            uint8_t * dst, size_t dstStride, SimdCompareType compareType)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return AveragingBinarization<SimdCompareEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareNotEqual:
                return AveragingBinarization<SimdCompareNotEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreater:
                return AveragingBinarization<SimdCompareGreater>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return AveragingBinarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesser:
                return AveragingBinarization<SimdCompareLesser>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return AveragingBinarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}
