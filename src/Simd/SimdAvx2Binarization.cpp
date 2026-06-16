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
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align, SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

            __m256i _value = _mm256_set1_epi8(value);
            __m256i _positive = _mm256_set1_epi8(positive);
            __m256i _negative = _mm256_set1_epi8(negative);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m256i mask = Compare8u<compareType>(Load<align>((__m256i*)(src + col)), _value);
                    Store<align>((__m256i*)(dst + col), _mm256_blendv_epi8(_negative, _positive, mask));
                }
                if (alignedWidth != width)
                {
                    const __m256i mask = Compare8u<compareType>(Load<false>((__m256i*)(src + width - A)), _value);
                    Store<false>((__m256i*)(dst + width - A), _mm256_blendv_epi8(_negative, _positive, mask));
                }
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
                    size_t size = sizeof(uint16_t)*(width + 2 * edge) + sizeof(uint32_t)*(2 * width + 2 * edge);
                    _p = Allocate(size);
                    memset(_p, 0, size);
                    sa = (uint16_t*)_p + edge;
                    s0a0 = (uint32_t*)(sa + width + edge) + edge;
                    sum = (uint32_t*)(s0a0 + width + edge);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sa;
                uint32_t * s0a0;
                uint32_t * sum;
            private:
                void *_p;
            };
        }

        template <bool srcAlign, bool dstAlign, SimdCompareType compareType>
        SIMD_INLINE void AddRows(const uint8_t * src, uint16_t * sa, const __m256i & value, const __m256i & mask)
        {
            const __m256i inc = _mm256_permute4x64_epi64(_mm256_and_si256(Compare8u<compareType>(Load<srcAlign>((__m256i*)src), value), mask), 0xD8);
            Store<dstAlign>((__m256i*)sa + 0, _mm256_add_epi8(Load<dstAlign>((__m256i*)sa + 0), _mm256_unpacklo_epi8(inc, _mm256_permute4x64_epi64(mask, 0xD8))));
            Store<dstAlign>((__m256i*)sa + 1, _mm256_add_epi8(Load<dstAlign>((__m256i*)sa + 1), _mm256_unpackhi_epi8(inc, _mm256_permute4x64_epi64(mask, 0xD8))));
        }

        template <bool srcAlign, bool dstAlign, SimdCompareType compareType>
        SIMD_INLINE void SubRows(const uint8_t * src, uint16_t * sa, const __m256i & value, const __m256i & mask)
        {
            const __m256i dec = _mm256_permute4x64_epi64(_mm256_and_si256(Compare8u<compareType>(Load<srcAlign>((__m256i*)src), value), mask), 0xD8);
            Store<dstAlign>((__m256i*)sa + 0, _mm256_sub_epi8(Load<dstAlign>((__m256i*)sa + 0), _mm256_unpacklo_epi8(dec, _mm256_permute4x64_epi64(mask, 0xD8))));
            Store<dstAlign>((__m256i*)sa + 1, _mm256_sub_epi8(Load<dstAlign>((__m256i*)sa + 1), _mm256_unpackhi_epi8(dec, _mm256_permute4x64_epi64(mask, 0xD8))));
        }

        template <bool align>
        SIMD_INLINE __m256i CompareSum(const uint32_t * sum, const __m256i & ff_threshold)
        {
            const __m256i mask0 = _mm256_cmpgt_epi32(_mm256_madd_epi16(Load<align>((__m256i*)sum + 0), ff_threshold), K_ZERO);
            const __m256i mask1 = _mm256_cmpgt_epi32(_mm256_madd_epi16(Load<align>((__m256i*)sum + 1), ff_threshold), K_ZERO);
            const __m256i mask2 = _mm256_cmpgt_epi32(_mm256_madd_epi16(Load<align>((__m256i*)sum + 2), ff_threshold), K_ZERO);
            const __m256i mask3 = _mm256_cmpgt_epi32(_mm256_madd_epi16(Load<align>((__m256i*)sum + 3), ff_threshold), K_ZERO);
            return PackI16ToI8(PackI32ToI16(mask0, mask1), PackI32ToI16(mask2, mask3));
        }

        template <bool align, SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x7F);

            const size_t alignedWidth = AlignLo(width, A);

            const __m256i tailMask = SetMask<uint8_t>(0, A - width + alignedWidth, 1);
            const __m256i ff_threshold = SetInt16(0xFF, -threshold);
            const __m256i _value = _mm256_set1_epi8(value);
            const __m256i _positive = _mm256_set1_epi8(positive);
            const __m256i _negative = _mm256_set1_epi8(negative);

            Buffer buffer(AlignHi(width, A), AlignHi(neighborhood + 1, A));

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t * s = src + row*srcStride;
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                if (alignedWidth != width)
                    AddRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t * s = src + (row + neighborhood)*srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        AddRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                    if (alignedWidth != width)
                        AddRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
                }
                if (row > neighborhood)
                {
                    const uint8_t * s = src + (row - neighborhood - 1)*srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        SubRows<align, true, compareType>(s + col, buffer.sa + col, _value, K8_01);
                    if (alignedWidth != width)
                        SubRows<false, false, compareType>(s + width - A, buffer.sa + width - A, _value, tailMask);
                }

                for (size_t col = 0; col < width; col += HA)
                {
                    const __m256i sa = LoadPermuted<true>((__m256i*)(buffer.sa + col));
                    Store<true>((__m256i*)(buffer.s0a0 + col) + 0, _mm256_unpacklo_epi8(sa, K_ZERO));
                    Store<true>((__m256i*)(buffer.s0a0 + col) + 1, _mm256_unpackhi_epi8(sa, K_ZERO));
                }

                uint32_t sum = 0;
                for (size_t col = 0; col < neighborhood; ++col)
                {
                    sum += buffer.s0a0[col];
                }
                for (size_t col = 0; col < width; ++col)
                {
                    sum += buffer.s0a0[col + neighborhood];
                    sum -= buffer.s0a0[col - neighborhood - 1];
                    buffer.sum[col] = sum;
                }

                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const __m256i mask = CompareSum<true>(buffer.sum + col, ff_threshold);
                    Store<align>((__m256i*)(dst + col), _mm256_blendv_epi8(_negative, _positive, mask));
                }
                if (alignedWidth != width)
                {
                    const __m256i mask = CompareSum<false>(buffer.sum + width - A, ff_threshold);
                    Store<false>((__m256i*)(dst + width - A), _mm256_blendv_epi8(_negative, _positive, mask));
                }

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

        //-----------------------------------------------------------------------------------------

        namespace
        {
            struct BufferV2
            {
                BufferV2(size_t width, size_t edge)
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

                ~BufferV2()
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

            BufferV2 buffer(width, neighborhood + 1);

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
#endif// SIMD_AVX2_ENABLE
}
