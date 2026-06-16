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

        template <bool align> SIMD_INLINE void Add16i(const uint8x16_t& value, int32_t* dst)
        {
            const uint16x8_t lo = vmovl_u8(vget_low_u8(value));
            const uint16x8_t hi = vmovl_u8(vget_high_u8(value));
            Store<align>(dst + 0 * F, vaddq_s32(Load<align>(dst + 0 * F), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo)))));
            Store<align>(dst + 1 * F, vaddq_s32(Load<align>(dst + 1 * F), vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo)))));
            Store<align>(dst + 2 * F, vaddq_s32(Load<align>(dst + 2 * F), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi)))));
            Store<align>(dst + 3 * F, vaddq_s32(Load<align>(dst + 3 * F), vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi)))));
        }

        template <bool align> SIMD_INLINE void Sub16i(const uint8x16_t& value, int32_t* dst)
        {
            const uint16x8_t lo = vmovl_u8(vget_low_u8(value));
            const uint16x8_t hi = vmovl_u8(vget_high_u8(value));
            Store<align>(dst + 0 * F, vsubq_s32(Load<align>(dst + 0 * F), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo)))));
            Store<align>(dst + 1 * F, vsubq_s32(Load<align>(dst + 1 * F), vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo)))));
            Store<align>(dst + 2 * F, vsubq_s32(Load<align>(dst + 2 * F), vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi)))));
            Store<align>(dst + 3 * F, vsubq_s32(Load<align>(dst + 3 * F), vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi)))));
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void AddRow(const uint8_t* src, int32_t* rs, int32_t* ra, const uint8x16_t& valueMask, const uint8x16_t& countMask)
        {
            Add16i<dstAlign>(vandq_u8(Load<srcAlign>(src), valueMask), rs);
            Add16i<dstAlign>(countMask, ra);
        }

        template <bool srcAlign, bool dstAlign> SIMD_INLINE void SubRow(const uint8_t* src, int32_t* rs, int32_t* ra, const uint8x16_t& valueMask, const uint8x16_t& countMask)
        {
            Sub16i<dstAlign>(vandq_u8(Load<srcAlign>(src), valueMask), rs);
            Sub16i<dstAlign>(countMask, ra);
        }

        SIMD_INLINE uint8x16_t PackMask(const uint32x4_t& m0, const uint32x4_t& m1, const uint32x4_t& m2, const uint32x4_t& m3)
        {
            return vcombine_u8(vmovn_u16(vcombine_u16(vmovn_u32(m0), vmovn_u32(m1))), vmovn_u16(vcombine_u16(vmovn_u32(m2), vmovn_u32(m3))));
        }

        template <bool srcAlign, bool bufAlign> SIMD_INLINE uint8x16_t Compare(const uint8_t* src, const int32_t* sum, const int32_t* area, const int32x4_t& shift)
        {
            const uint8x16_t s8 = Load<srcAlign>(src);
            const uint16x8_t lo = vmovl_u8(vget_low_u8(s8));
            const uint16x8_t hi = vmovl_u8(vget_high_u8(s8));
            const int32x4_t v0 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo))), shift);
            const int32x4_t v1 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo))), shift);
            const int32x4_t v2 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi))), shift);
            const int32x4_t v3 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi))), shift);
            const uint32x4_t m0 = vcgtq_s32(vmulq_s32(v0, Load<bufAlign>(area + 0 * F)), Load<bufAlign>(sum + 0 * F));
            const uint32x4_t m1 = vcgtq_s32(vmulq_s32(v1, Load<bufAlign>(area + 1 * F)), Load<bufAlign>(sum + 1 * F));
            const uint32x4_t m2 = vcgtq_s32(vmulq_s32(v2, Load<bufAlign>(area + 2 * F)), Load<bufAlign>(sum + 2 * F));
            const uint32x4_t m3 = vcgtq_s32(vmulq_s32(v3, Load<bufAlign>(area + 3 * F)), Load<bufAlign>(sum + 3 * F));
            return PackMask(m0, m1, m2, m3);
        }

        template <bool align> void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            const size_t alignedWidth = AlignLo(width, A);
            const uint8x16_t tailValueMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            const uint8x16_t tailCountMask = ShiftLeft(K8_01, A - width + alignedWidth);
            const uint8x16_t _positive = vdupq_n_u8(positive);
            const uint8x16_t _negative = vdupq_n_u8(negative);
            const int32x4_t _shift = vdupq_n_s32(shift);

            BufferV2 buffer(width, neighborhood + 1);

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < alignedWidth; col += A)
                    AddRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K8_FF, K8_01);
                if (alignedWidth != width)
                    AddRow<false, false>(ps + width - A, buffer.rs + width - A, buffer.ra + width - A, tailValueMask, tailCountMask);
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t* ps = src + (row + neighborhood) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        AddRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K8_FF, K8_01);
                    if (alignedWidth != width)
                        AddRow<false, false>(ps + width - A, buffer.rs + width - A, buffer.ra + width - A, tailValueMask, tailCountMask);
                }

                if (row > neighborhood)
                {
                    const uint8_t* ps = src + (row - neighborhood - 1) * srcStride;
                    for (size_t col = 0; col < alignedWidth; col += A)
                        SubRow<align, true>(ps + col, buffer.rs + col, buffer.ra + col, K8_FF, K8_01);
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
                    const uint8x16_t mask = Compare<align, true>(ps + col, buffer.sum + col, buffer.area + col, _shift);
                    Store<align>(dst + col, vbslq_u8(mask, _positive, _negative));
                }
                if (alignedWidth != width)
                {
                    const uint8x16_t mask = Compare<false, false>(ps + width - A, buffer.sum + width - A, buffer.area + width - A, _shift);
                    Store<false>(dst + width - A, vbslq_u8(mask, _positive, _negative));
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
#endif// SIMD_NEON_ENABLE
}
