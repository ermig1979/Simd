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
#include "Simd/SimdCompare.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align, SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

            uint8x16_t _value = vdupq_n_u8(value);
            uint8x16_t _positive = vdupq_n_u8(positive);
            uint8x16_t _negative = vdupq_n_u8(negative);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    const uint8x16_t mask = Compare8u<compareType>(Load<align>(src + col), _value);
                    Store<align>(dst + col, vbslq_u8(mask, _positive, _negative));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t mask = Compare8u<compareType>(Load<false>(src + width - A), _value);
                    Store<false>(dst + width - A, vbslq_u8(mask, _positive, _negative));
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
        SIMD_INLINE void AddRows(const uint8_t * src, uint16_t * sa, const uint8x16_t & value, const uint8x16_t & mask)
        {
            const uint8x16_t inc = vandq_u8(Compare8u<compareType>(Load<srcAlign>(src), value), mask);
            uint8x16x2_t _sa = Load2<dstAlign>((uint8_t*)sa);
            _sa.val[0] = vaddq_u8(_sa.val[0], inc);
            _sa.val[1] = vaddq_u8(_sa.val[1], mask);
            Store2<dstAlign>((uint8_t*)sa, _sa);
        }

        template <bool srcAlign, bool dstAlign, SimdCompareType compareType>
        SIMD_INLINE void SubRows(const uint8_t * src, uint16_t * sa, const uint8x16_t & value, const uint8x16_t & mask)
        {
            const uint8x16_t dec = vandq_u8(Compare8u<compareType>(Load<srcAlign>(src), value), mask);
            uint8x16x2_t _sa = Load2<dstAlign>((uint8_t*)sa);
            _sa.val[0] = vsubq_u8(_sa.val[0], dec);
            _sa.val[1] = vsubq_u8(_sa.val[1], mask);
            Store2<dstAlign>((uint8_t*)sa, _sa);
        }

        SIMD_INLINE uint32x4_t CompareSum(const uint32x4_t & sum, const uint32x4_t & area, const uint32x4_t & threshold)
        {
            return vcgtq_u32(vmulq_u32(sum, K32_000000FF), vmulq_u32(area, threshold));
        }

        template <bool align>
        SIMD_INLINE uint16x8_t CompareSum(const uint16_t * sum, const uint32x4_t & threshold)
        {
            uint16x8x2_t _sum = Load2<align>(sum);
            uint32x4_t lo = CompareSum(UnpackU16<0>(_sum.val[0]), UnpackU16<0>(_sum.val[1]), threshold);
            uint32x4_t hi = CompareSum(UnpackU16<1>(_sum.val[0]), UnpackU16<1>(_sum.val[1]), threshold);
            return PackU32(lo, hi);
        }

        template <bool align>
        SIMD_INLINE uint8x16_t CompareSum(const uint32_t * sum, const uint32x4_t & threshold)
        {
            uint16x8_t lo = CompareSum<align>((uint16_t*)sum + 0, threshold);
            uint16x8_t hi = CompareSum<align>((uint16_t*)sum + A, threshold);
            return PackU16(lo, hi);
        }

        template <bool align, SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x7F);

            const size_t alignedWidth = AlignLo(width, A);

            const uint8x16_t  tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            uint8x16_t _value = vdupq_n_u8(value);
            uint8x16_t _positive = vdupq_n_u8(positive);
            uint8x16_t _negative = vdupq_n_u8(negative);
            uint32x4_t _threshold = vdupq_n_u32(threshold);

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
                    const uint8x16_t sa = Load<true>((uint8_t*)(buffer.sa + col));
                    Store<true>((uint16_t*)(buffer.s0a0 + col + 0), UnpackU8<0>(sa));
                    Store<true>((uint16_t*)(buffer.s0a0 + col + 4), UnpackU8<1>(sa));
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
                    const uint8x16_t mask = CompareSum<true>(buffer.sum + col, _threshold);
                    Store<align>(dst + col, vbslq_u8(mask, _positive, _negative));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t mask = CompareSum<false>(buffer.sum + width - A, _threshold);
                    Store<false>(dst + width - A, vbslq_u8(mask, _positive, _negative));
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
    }
#endif// SIMD_NEON_ENABLE
}
