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

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
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

        template <bool align> SIMD_INLINE void Add32i(const __m256i& value, int32_t* dst)
        {
            const __m128i lo = _mm256_castsi256_si128(value);
            const __m128i hi = _mm256_extracti128_si256(value, 1);
            Store<align>((__m256i*)dst + 0, _mm256_add_epi32(Load<align>((__m256i*)dst + 0), _mm256_cvtepu8_epi32(lo)));
            Store<align>((__m256i*)dst + 1, _mm256_add_epi32(Load<align>((__m256i*)dst + 1), _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))));
            Store<align>((__m256i*)dst + 2, _mm256_add_epi32(Load<align>((__m256i*)dst + 2), _mm256_cvtepu8_epi32(hi)));
            Store<align>((__m256i*)dst + 3, _mm256_add_epi32(Load<align>((__m256i*)dst + 3), _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))));
        }

        template <bool align> SIMD_INLINE void Sub32i(const __m256i& value, int32_t* dst)
        {
            const __m128i lo = _mm256_castsi256_si128(value);
            const __m128i hi = _mm256_extracti128_si256(value, 1);
            Store<align>((__m256i*)dst + 0, _mm256_sub_epi32(Load<align>((__m256i*)dst + 0), _mm256_cvtepu8_epi32(lo)));
            Store<align>((__m256i*)dst + 1, _mm256_sub_epi32(Load<align>((__m256i*)dst + 1), _mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8))));
            Store<align>((__m256i*)dst + 2, _mm256_sub_epi32(Load<align>((__m256i*)dst + 2), _mm256_cvtepu8_epi32(hi)));
            Store<align>((__m256i*)dst + 3, _mm256_sub_epi32(Load<align>((__m256i*)dst + 3), _mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8))));
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void AddRow(const uint8_t* src, int32_t* rs, int32_t* ra, const __m256i& valueMask, const __m256i& countMask)
        {
            Add32i<dstAlign>(_mm256_and_si256(Load<srcAlign>((__m256i*)src), valueMask), rs);
            Add32i<dstAlign>(countMask, ra);
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void SubRow(const uint8_t* src, int32_t* rs, int32_t* ra, const __m256i& valueMask, const __m256i& countMask)
        {
            Sub32i<dstAlign>(_mm256_and_si256(Load<srcAlign>((__m256i*)src), valueMask), rs);
            Sub32i<dstAlign>(countMask, ra);
        }

        template <bool srcAlign, bool bufAlign> SIMD_INLINE __m256i Compare(const uint8_t* src, const int32_t* sum, const int32_t* area, const __m256i& shift)
        {
            const __m256i s8 = Load<srcAlign>((__m256i*)src);
            const __m128i lo = _mm256_castsi256_si128(s8);
            const __m128i hi = _mm256_extracti128_si256(s8, 1);
            const __m256i v0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(lo), shift);
            const __m256i v1 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)), shift);
            const __m256i v2 = _mm256_add_epi32(_mm256_cvtepu8_epi32(hi), shift);
            const __m256i v3 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)), shift);
            const __m256i m0 = _mm256_cmpgt_epi32(_mm256_mullo_epi32(v0, Load<bufAlign>((__m256i*)area + 0)), Load<bufAlign>((__m256i*)sum + 0));
            const __m256i m1 = _mm256_cmpgt_epi32(_mm256_mullo_epi32(v1, Load<bufAlign>((__m256i*)area + 1)), Load<bufAlign>((__m256i*)sum + 1));
            const __m256i m2 = _mm256_cmpgt_epi32(_mm256_mullo_epi32(v2, Load<bufAlign>((__m256i*)area + 2)), Load<bufAlign>((__m256i*)sum + 2));
            const __m256i m3 = _mm256_cmpgt_epi32(_mm256_mullo_epi32(v3, Load<bufAlign>((__m256i*)area + 3)), Load<bufAlign>((__m256i*)sum + 3));
            return PackI16ToI8(PackI32ToI16(m0, m1), PackI32ToI16(m2, m3));
        }

        template <bool align> void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            const size_t alignedWidth = AlignLo(width, A);
            const __m256i tailValueMask = SetMask<uint8_t>(0, A - width + alignedWidth, 0xFF);
            const __m256i tailCountMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            const __m256i _positive = _mm256_set1_epi8(positive);
            const __m256i _negative = _mm256_set1_epi8(negative);
            const __m256i _shift = _mm256_set1_epi32(shift);

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
                    const __m256i mask = Compare<align, true>(ps + col, buffer.sum + col, buffer.area + col, _shift);
                    Store<align>((__m256i*)(dst + col), _mm256_blendv_epi8(_negative, _positive, mask));
                }
                if (alignedWidth != width)
                {
                    const __m256i mask = Compare<false, false>(ps + width - A, buffer.sum + width - A, buffer.area + width - A, _shift);
                    Store<false>((__m256i*)(dst + width - A), _mm256_blendv_epi8(_negative, _positive, mask));
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
