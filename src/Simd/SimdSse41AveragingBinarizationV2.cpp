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
#include "Simd/SimdStore.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdMath.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t edge)
                {
                    _width = AlignHi(width, A);
                    _edge = AlignHi(edge, A);
                    _size = _width + 2 * _edge;
                    size_t size = sizeof(int32_t) * (2 * _size + 2 * _width);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    int32_t* data = (int32_t*)_p;
                    rs = data + _edge;
                    ra = data + _size + _edge;
                    sum = data + 2 * _size;
                    area = sum + _width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                int32_t* rs;
                int32_t* ra;
                int32_t* sum;
                int32_t* area;
            private:
                size_t _width;
                size_t _edge;
                size_t _size;
                void* _p;
            };
        }

        SIMD_INLINE void Add16i(const __m128i& value, int32_t* dst)
        {
            const __m128i lo = _mm_unpacklo_epi8(value, K_ZERO);
            const __m128i hi = _mm_unpackhi_epi8(value, K_ZERO);
            Store<true>((__m128i*)dst + 0, _mm_add_epi32(Load<true>((__m128i*)dst + 0), _mm_unpacklo_epi16(lo, K_ZERO)));
            Store<true>((__m128i*)dst + 1, _mm_add_epi32(Load<true>((__m128i*)dst + 1), _mm_unpackhi_epi16(lo, K_ZERO)));
            Store<true>((__m128i*)dst + 2, _mm_add_epi32(Load<true>((__m128i*)dst + 2), _mm_unpacklo_epi16(hi, K_ZERO)));
            Store<true>((__m128i*)dst + 3, _mm_add_epi32(Load<true>((__m128i*)dst + 3), _mm_unpackhi_epi16(hi, K_ZERO)));
        }

        SIMD_INLINE void Sub16i(const __m128i& value, int32_t* dst)
        {
            const __m128i lo = _mm_unpacklo_epi8(value, K_ZERO);
            const __m128i hi = _mm_unpackhi_epi8(value, K_ZERO);
            Store<true>((__m128i*)dst + 0, _mm_sub_epi32(Load<true>((__m128i*)dst + 0), _mm_unpacklo_epi16(lo, K_ZERO)));
            Store<true>((__m128i*)dst + 1, _mm_sub_epi32(Load<true>((__m128i*)dst + 1), _mm_unpackhi_epi16(lo, K_ZERO)));
            Store<true>((__m128i*)dst + 2, _mm_sub_epi32(Load<true>((__m128i*)dst + 2), _mm_unpacklo_epi16(hi, K_ZERO)));
            Store<true>((__m128i*)dst + 3, _mm_sub_epi32(Load<true>((__m128i*)dst + 3), _mm_unpackhi_epi16(hi, K_ZERO)));
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void AddRow(const uint8_t* src, int32_t* rs, int32_t* ra, const __m128i& valueMask, const __m128i& countMask)
        {
            const __m128i value = _mm_and_si128(Load<srcAlign>((__m128i*)src), valueMask);
            const __m128i count = countMask;
            if (dstAlign)
            {
                Add16i(value, rs);
                Add16i(count, ra);
            }
            else
            {
                const __m128i lo = _mm_unpacklo_epi8(value, K_ZERO);
                const __m128i hi = _mm_unpackhi_epi8(value, K_ZERO);
                Store<false>((__m128i*)rs + 0, _mm_add_epi32(Load<false>((__m128i*)rs + 0), _mm_unpacklo_epi16(lo, K_ZERO)));
                Store<false>((__m128i*)rs + 1, _mm_add_epi32(Load<false>((__m128i*)rs + 1), _mm_unpackhi_epi16(lo, K_ZERO)));
                Store<false>((__m128i*)rs + 2, _mm_add_epi32(Load<false>((__m128i*)rs + 2), _mm_unpacklo_epi16(hi, K_ZERO)));
                Store<false>((__m128i*)rs + 3, _mm_add_epi32(Load<false>((__m128i*)rs + 3), _mm_unpackhi_epi16(hi, K_ZERO)));

                const __m128i clo = _mm_unpacklo_epi8(count, K_ZERO);
                const __m128i chi = _mm_unpackhi_epi8(count, K_ZERO);
                Store<false>((__m128i*)ra + 0, _mm_add_epi32(Load<false>((__m128i*)ra + 0), _mm_unpacklo_epi16(clo, K_ZERO)));
                Store<false>((__m128i*)ra + 1, _mm_add_epi32(Load<false>((__m128i*)ra + 1), _mm_unpackhi_epi16(clo, K_ZERO)));
                Store<false>((__m128i*)ra + 2, _mm_add_epi32(Load<false>((__m128i*)ra + 2), _mm_unpacklo_epi16(chi, K_ZERO)));
                Store<false>((__m128i*)ra + 3, _mm_add_epi32(Load<false>((__m128i*)ra + 3), _mm_unpackhi_epi16(chi, K_ZERO)));
            }
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void SubRow(const uint8_t* src, int32_t* rs, int32_t* ra, const __m128i& valueMask, const __m128i& countMask)
        {
            const __m128i value = _mm_and_si128(Load<srcAlign>((__m128i*)src), valueMask);
            const __m128i count = countMask;
            if (dstAlign)
            {
                Sub16i(value, rs);
                Sub16i(count, ra);
            }
            else
            {
                const __m128i lo = _mm_unpacklo_epi8(value, K_ZERO);
                const __m128i hi = _mm_unpackhi_epi8(value, K_ZERO);
                Store<false>((__m128i*)rs + 0, _mm_sub_epi32(Load<false>((__m128i*)rs + 0), _mm_unpacklo_epi16(lo, K_ZERO)));
                Store<false>((__m128i*)rs + 1, _mm_sub_epi32(Load<false>((__m128i*)rs + 1), _mm_unpackhi_epi16(lo, K_ZERO)));
                Store<false>((__m128i*)rs + 2, _mm_sub_epi32(Load<false>((__m128i*)rs + 2), _mm_unpacklo_epi16(hi, K_ZERO)));
                Store<false>((__m128i*)rs + 3, _mm_sub_epi32(Load<false>((__m128i*)rs + 3), _mm_unpackhi_epi16(hi, K_ZERO)));

                const __m128i clo = _mm_unpacklo_epi8(count, K_ZERO);
                const __m128i chi = _mm_unpackhi_epi8(count, K_ZERO);
                Store<false>((__m128i*)ra + 0, _mm_sub_epi32(Load<false>((__m128i*)ra + 0), _mm_unpacklo_epi16(clo, K_ZERO)));
                Store<false>((__m128i*)ra + 1, _mm_sub_epi32(Load<false>((__m128i*)ra + 1), _mm_unpackhi_epi16(clo, K_ZERO)));
                Store<false>((__m128i*)ra + 2, _mm_sub_epi32(Load<false>((__m128i*)ra + 2), _mm_unpacklo_epi16(chi, K_ZERO)));
                Store<false>((__m128i*)ra + 3, _mm_sub_epi32(Load<false>((__m128i*)ra + 3), _mm_unpackhi_epi16(chi, K_ZERO)));
            }
        }

        template <bool srcAlign, bool bufAlign> SIMD_INLINE __m128i Compare(const uint8_t* src, const int32_t* sum, const int32_t* area, const __m128i& shift)
        {
            const __m128i s8 = Load<srcAlign>((__m128i*)src);
            const __m128i lo = _mm_unpacklo_epi8(s8, K_ZERO);
            const __m128i hi = _mm_unpackhi_epi8(s8, K_ZERO);
            const __m128i v0 = _mm_add_epi32(_mm_unpacklo_epi16(lo, K_ZERO), shift);
            const __m128i v1 = _mm_add_epi32(_mm_unpackhi_epi16(lo, K_ZERO), shift);
            const __m128i v2 = _mm_add_epi32(_mm_unpacklo_epi16(hi, K_ZERO), shift);
            const __m128i v3 = _mm_add_epi32(_mm_unpackhi_epi16(hi, K_ZERO), shift);
            const __m128i m0 = _mm_cmpgt_epi32(_mm_mullo_epi32(v0, Load<bufAlign>((__m128i*)area + 0)), Load<bufAlign>((__m128i*)sum + 0));
            const __m128i m1 = _mm_cmpgt_epi32(_mm_mullo_epi32(v1, Load<bufAlign>((__m128i*)area + 1)), Load<bufAlign>((__m128i*)sum + 1));
            const __m128i m2 = _mm_cmpgt_epi32(_mm_mullo_epi32(v2, Load<bufAlign>((__m128i*)area + 2)), Load<bufAlign>((__m128i*)sum + 2));
            const __m128i m3 = _mm_cmpgt_epi32(_mm_mullo_epi32(v3, Load<bufAlign>((__m128i*)area + 3)), Load<bufAlign>((__m128i*)sum + 3));
            return _mm_packs_epi16(_mm_packs_epi32(m0, m1), _mm_packs_epi32(m2, m3));
        }

        template <bool align> void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            const size_t alignedWidth = AlignLo(width, A);
            const __m128i tailValueMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);
            const __m128i tailCountMask = ShiftLeft(K8_01, A - width + alignedWidth);
            const __m128i _positive = _mm_set1_epi8(positive);
            const __m128i _negative = _mm_set1_epi8(negative);
            const __m128i _shift = _mm_set1_epi32(shift);

            Buffer buffer(width, neighborhood + 1);

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K_INV_ZERO, K8_01);
                if (alignedWidth != width)
                    AddRow<false, false>(ps + width - A, buffer.rs + width - A, buffer.ra + width - A, tailValueMask, tailCountMask);
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t* ps = src + (row + neighborhood) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        AddRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K_INV_ZERO, K8_01);
                    if (alignedWidth != width)
                        AddRow<false, false>(ps + width - A, buffer.rs + width - A, buffer.ra + width - A, tailValueMask, tailCountMask);
                }

                if (row > neighborhood)
                {
                    const uint8_t* ps = src + (row - neighborhood - 1) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        SubRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K_INV_ZERO, K8_01);
                    if (alignedWidth != width)
                        SubRow<false, false>(ps + width - A, buffer.rs + width - A, buffer.ra + width - A, tailValueMask, tailCountMask);
                }

                int32_t sum = 0, area = 0;
                for (size_t col = 0; col < neighborhood; ++col)
                {
                    sum += buffer.rs[col];
                    area += buffer.ra[col];
                }
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < width; ++col)
                {
                    sum += buffer.rs[col + neighborhood] - buffer.rs[col - neighborhood - 1];
                    area += buffer.ra[col + neighborhood] - buffer.ra[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                    buffer.area[col] = area;
                }

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m128i mask = Compare<align, true>(ps + col, buffer.sum + col, buffer.area + col, _shift);
                    Store<align>((__m128i*)(dst + col), Combine(mask, _positive, _negative));
                }
                if (alignedWidth != width)
                {
                    const __m128i mask = Compare<false, false>(ps + width - A, buffer.sum + width - A, buffer.area + width - A, _shift);
                    Store<false>((__m128i*)(dst + width - A), Combine(mask, _positive, _negative));
                }
                dst += dstStride;
            }
        }

        void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                AveragingBinarizationV2<true>(src, srcStride, width, height, neighborhood, shift, positive, negative, dst, dstStride);
            else
                AveragingBinarizationV2<false>(src, srcStride, width, height, neighborhood, shift, positive, negative, dst, dstStride);
        }
    }
#endif
}
