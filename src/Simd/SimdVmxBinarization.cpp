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
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <bool align, bool first, SimdCompareType compareType>
        SIMD_INLINE void Binarization(const Loader<align> & src, const v128_u8 & value, const v128_u8 & positive, const v128_u8 & negative, Storer<align> & dst)
        {
            const v128_u8 mask = Compare8u<compareType>(Load<align, first>(src), value);
            Store<align, first>(dst, vec_sel(negative, positive, mask));
        }

        template <bool align, SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, A);

            const v128_u8 _value = SetU8(value);
            const v128_u8 _positive = SetU8(positive);
            const v128_u8 _negative = SetU8(negative);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _src(src);
                Storer<align> _dst(dst);
                Binarization<align, true, compareType>(_src, _value, _positive, _negative, _dst);
                for (size_t col = A; col < alignedWidth; col += A)
                    Binarization<align, false, compareType>(_src, _value, _positive, _negative, _dst);
                Flush(_dst);

                if (alignedWidth != width)
                {
                    Loader<false> _src(src + width - A);
                    Storer<false> _dst(dst + width - A);
                    Binarization<false, true, compareType>(_src, _value, _positive, _negative, _dst);
                    Flush(_dst);

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

        template <bool srcAlign, bool dstAlign, bool first, SimdCompareType compareType>
        SIMD_INLINE void AddRows(const Loader<srcAlign> & src, const v128_u8 & value, const v128_u8 & mask, const Loader<dstAlign> & saSrc, Storer<dstAlign> & saDst)
        {
            const v128_u8 inc = vec_and(Compare8u<compareType>(Load<srcAlign, first>(src), value), mask);
            Store<dstAlign, first>(saDst, vec_add(Load<dstAlign, first>(saSrc), (v128_u8)UnpackLoU8(mask, inc)));
            Store<dstAlign, false>(saDst, vec_add(Load<dstAlign, false>(saSrc), (v128_u8)UnpackHiU8(mask, inc)));
        }

        template <bool srcAlign, bool dstAlign, bool first, SimdCompareType compareType>
        SIMD_INLINE void SubRows(const Loader<srcAlign> & src, const v128_u8 & value, const v128_u8 & mask, const Loader<dstAlign> & saSrc, Storer<dstAlign> & saDst)
        {
            const v128_u8 dec = vec_and(Compare8u<compareType>(Load<srcAlign, first>(src), value), mask);
            Store<dstAlign, first>(saDst, vec_sub(Load<dstAlign, first>(saSrc), (v128_u8)UnpackLoU8(mask, dec)));
            Store<dstAlign, false>(saDst, vec_sub(Load<dstAlign, false>(saSrc), (v128_u8)UnpackHiU8(mask, dec)));
        }

        template <bool srcAlign, bool dstAlign, bool first>
        SIMD_INLINE void CompareSum(const Loader<srcAlign> & sum, const v128_s16 & ff_threshold, const v128_u8 & positive, const v128_u8 & negative, Storer<dstAlign> & dst)
        {
            const v128_u32 mask0 = (v128_u32)vec_cmpgt(vec_msum((v128_s16)Load<srcAlign, first>(sum), ff_threshold, (v128_s32)K32_00000000), (v128_s32)K32_00000000);
            const v128_u32 mask1 = (v128_u32)vec_cmpgt(vec_msum((v128_s16)Load<srcAlign, false>(sum), ff_threshold, (v128_s32)K32_00000000), (v128_s32)K32_00000000);
            const v128_u32 mask2 = (v128_u32)vec_cmpgt(vec_msum((v128_s16)Load<srcAlign, false>(sum), ff_threshold, (v128_s32)K32_00000000), (v128_s32)K32_00000000);
            const v128_u32 mask3 = (v128_u32)vec_cmpgt(vec_msum((v128_s16)Load<srcAlign, false>(sum), ff_threshold, (v128_s32)K32_00000000), (v128_s32)K32_00000000);
            const v128_u8 mask = vec_pack(vec_pack(mask0, mask1), vec_pack(mask2, mask3));
            Store<dstAlign, first>(dst, vec_sel(negative, positive, mask));
        }

        template <bool align, SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x7F);

            const size_t alignedWidth = AlignLo(width, A);

            const v128_u8 tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            const v128_s16 ff_threshold = SetI16(0xFF, -threshold);
            const v128_u8 _value = SetU8(value);
            const v128_u8 _positive = SetU8(positive);
            const v128_u8 _negative = SetU8(negative);

            Buffer buffer(AlignHi(width, A), AlignHi(neighborhood + 1, A));

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t * s = src + row*srcStride;
                Loader<align> _src(s);
                Loader<true> saSrc(buffer.sa);
                Storer<true> saDst(buffer.sa);
                AddRows<align, true, true, compareType>(_src, _value, K8_01, saSrc, saDst);
                for (size_t col = A; col < alignedWidth; col += A)
                    AddRows<align, true, false, compareType>(_src, _value, K8_01, saSrc, saDst);
                Flush(saDst);
                if (alignedWidth != width)
                {
                    Loader<false> _src(s + width - A), saSrc(buffer.sa + width - A);
                    Storer<false> saDst(buffer.sa + width - A);
                    AddRows<false, false, true, compareType>(_src, _value, tailMask, saSrc, saDst);
                    Flush(saDst);
                }
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t * s = src + (row + neighborhood)*srcStride;
                    Loader<align> _src(s);
                    Loader<true> saSrc(buffer.sa);
                    Storer<true> saDst(buffer.sa);
                    AddRows<align, true, true, compareType>(_src, _value, K8_01, saSrc, saDst);
                    for (size_t col = A; col < alignedWidth; col += A)
                        AddRows<align, true, false, compareType>(_src, _value, K8_01, saSrc, saDst);
                    Flush(saDst);
                    if (alignedWidth != width)
                    {
                        Loader<false> _src(s + width - A), saSrc(buffer.sa + width - A);
                        Storer<false> saDst(buffer.sa + width - A);
                        AddRows<false, false, true, compareType>(_src, _value, tailMask, saSrc, saDst);
                        Flush(saDst);
                    }
                }
                if (row > neighborhood)
                {
                    const uint8_t * s = src + (row - neighborhood - 1)*srcStride;
                    Loader<align> _src(s);
                    Loader<true> saSrc(buffer.sa);
                    Storer<true> saDst(buffer.sa);
                    SubRows<align, true, true, compareType>(_src, _value, K8_01, saSrc, saDst);
                    for (size_t col = A; col < alignedWidth; col += A)
                        SubRows<align, true, false, compareType>(_src, _value, K8_01, saSrc, saDst);
                    Flush(saDst);
                    if (alignedWidth != width)
                    {
                        Loader<false> _src(s + width - A), saSrc(buffer.sa + width - A);
                        Storer<false> saDst(buffer.sa + width - A);
                        SubRows<false, false, true, compareType>(_src, _value, tailMask, saSrc, saDst);
                        Flush(saDst);
                    }
                }

                for (size_t col = 0; col < width; col += HA)
                {
                    const v128_u8 sa = (v128_u8)Load<true>(buffer.sa + col);
                    Store<true>(buffer.s0a0 + col + 0, (v128_u32)UnpackLoU8(sa));
                    Store<true>(buffer.s0a0 + col + 4, (v128_u32)UnpackHiU8(sa));
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

                Loader<true> _sum(buffer.sum);
                Storer<align> _dst(dst);
                CompareSum<true, align, true>(_sum, ff_threshold, _positive, _negative, _dst);
                for (size_t col = A; col < alignedWidth; col += A)
                    CompareSum<true, align, false>(_sum, ff_threshold, _positive, _negative, _dst);
                Flush(_dst);
                if (alignedWidth != width)
                {
                    Loader<false> _sum(buffer.sum + width - A);
                    Storer<false> _dst(dst + width - A);
                    CompareSum<false, false, true>(_sum, ff_threshold, _positive, _negative, _dst);
                    Flush(_dst);
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
#endif// SIMD_VMX_ENABLE
}
