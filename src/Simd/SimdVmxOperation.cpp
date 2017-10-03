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

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        template <SimdOperationBinary8uType type> SIMD_INLINE v128_u8 OperationBinary8u(const v128_u8 & a, const v128_u8 & b);

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uAverage>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_avg(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uAnd>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_and(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uOr>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_or(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uMaximum>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_max(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uMinimum>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_min(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_subs(a, b);
        }

        template <> SIMD_INLINE v128_u8 OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(const v128_u8 & a, const v128_u8 & b)
        {
            return vec_adds(a, b);
        }

        template <SimdOperationBinary8uType type, bool align, bool first>
        SIMD_INLINE void OperationBinary8u(const Loader<align> & a, const Loader<align> & b, Storer<align> & dst)
        {
            Store<align, first>(dst, OperationBinary8u<type>(Load<align, first>(a), Load<align, first>(b)));
        }

        template <SimdOperationBinary8uType type, bool align>
        SIMD_INLINE void OperationBinary8u(const uint8_t * a, const uint8_t * b, size_t offset, uint8_t * dst)
        {
            v128_u8 _a = Load<align>(a + offset);
            v128_u8 _b = Load<align>(b + offset);
            Store<align>(dst + offset, OperationBinary8u<type>(_a, _b));
        }

        template <bool align, SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(width*channelCount >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t size = channelCount*width;
            size_t alignedSize = Simd::AlignLo(size, A);
            size_t fullAlignedSize = Simd::AlignLo(size, QA);
            for (size_t row = 0; row < height; ++row)
            {
                if (align)
                {
                    size_t offset = 0;
                    for (; offset < fullAlignedSize; offset += QA)
                    {
                        OperationBinary8u<type, true>(a, b, offset, dst);
                        OperationBinary8u<type, true>(a, b, offset + A, dst);
                        OperationBinary8u<type, true>(a, b, offset + 2 * A, dst);
                        OperationBinary8u<type, true>(a, b, offset + 3 * A, dst);
                    }
                    for (; offset < alignedSize; offset += A)
                        OperationBinary8u<type, true>(a, b, offset, dst);
                }
                else
                {
                    Loader<align> _a(a), _b(b);
                    Storer<align> _dst(dst);
                    OperationBinary8u<type, align, true>(_a, _b, _dst);
                    for (size_t offset = A; offset < alignedSize; offset += A)
                        OperationBinary8u<type, align, false>(_a, _b, _dst);
                    Flush(_dst);
                }
                if (alignedSize != size)
                    OperationBinary8u<type, false>(a, b, size - A, dst);

                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <SimdOperationBinary8uType type> void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary8u<true, type>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            else
                OperationBinary8u<false, type>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
        }

        void OperationBinary8u(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type)
        {
            switch (type)
            {
            case SimdOperationBinary8uAverage:
                return OperationBinary8u<SimdOperationBinary8uAverage>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uAnd:
                return OperationBinary8u<SimdOperationBinary8uAnd>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uOr:
                return OperationBinary8u<SimdOperationBinary8uOr>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMaximum:
                return OperationBinary8u<SimdOperationBinary8uMaximum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uMinimum:
                return OperationBinary8u<SimdOperationBinary8uMinimum>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedSubtraction:
                return OperationBinary8u<SimdOperationBinary8uSaturatedSubtraction>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            case SimdOperationBinary8uSaturatedAddition:
                return OperationBinary8u<SimdOperationBinary8uSaturatedAddition>(a, aStride, b, bStride, width, height, channelCount, dst, dstStride);
            default:
                assert(0);
            }
        }

        template <SimdOperationBinary16iType type> SIMD_INLINE v128_s16 OperationBinary16i(const v128_s16 & a, const v128_s16 & b);

        template <> SIMD_INLINE v128_s16 OperationBinary16i<SimdOperationBinary16iAddition>(const v128_s16 & a, const v128_s16 & b)
        {
            return vec_add(a, b);
        }

        template <> SIMD_INLINE v128_s16 OperationBinary16i<SimdOperationBinary16iSubtraction>(const v128_s16 & a, const v128_s16 & b)
        {
            return vec_sub(a, b);
        }

        template <SimdOperationBinary16iType type, bool align, bool first>
        SIMD_INLINE void OperationBinary16i(const Loader<align> & a, const Loader<align> & b, Storer<align> & dst)
        {
            Store<align, first>(dst, OperationBinary16i<type>((v128_s16)Load<align, first>(a), (v128_s16)Load<align, first>(b)));
        }

        template <bool align, SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(width >= HA);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride));

            size_t alignedWidth = Simd::AlignLo(width, HA);
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> _a(a), _b(b);
                Storer<align> _dst(dst);
                OperationBinary16i<type, align, true>(_a, _b, _dst);
                for (size_t col = HA; col < alignedWidth; col += HA)
                    OperationBinary16i<type, align, false>(_a, _b, _dst);
                Flush(_dst);

                if (alignedWidth != width)
                {
                    size_t offset = 2 * width - A;
                    Loader<false> _a(a + offset), _b(b + offset);
                    Storer<false> _dst(dst + offset);
                    OperationBinary16i<type, false, true>(_a, _b, _dst);
                    Flush(_dst);
                }

                a += aStride;
                b += bStride;
                dst += dstStride;
            }
        }

        template <SimdOperationBinary16iType type> void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride) && Aligned(dst) && Aligned(dstStride))
                OperationBinary16i<true, type>(a, aStride, b, bStride, width, height, dst, dstStride);
            else
                OperationBinary16i<false, type>(a, aStride, b, bStride, width, height, dst, dstStride);
        }

        void OperationBinary16i(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
            size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type)
        {
            switch (type)
            {
            case SimdOperationBinary16iAddition:
                return OperationBinary16i<SimdOperationBinary16iAddition>(a, aStride, b, bStride, width, height, dst, dstStride);
            case SimdOperationBinary16iSubtraction:
                return OperationBinary16i<SimdOperationBinary16iSubtraction>(a, aStride, b, bStride, width, height, dst, dstStride);
            default:
                assert(0);
            }
        }

        SIMD_INLINE v128_u8 VectorProduct(const v128_u16 & vertical, const v128_u8 & horizontal)
        {
            v128_u16 lo = DivideBy255(vec_mladd(vertical, UnpackU8<0>(horizontal), K16_0000));
            v128_u16 hi = DivideBy255(vec_mladd(vertical, UnpackU8<1>(horizontal), K16_0000));
            return vec_pack(lo, hi);
        }

        template <bool align> SIMD_INLINE void VectorProduct(const v128_u16 & vertical, const uint8_t * horizontal, size_t offset, uint8_t * dst)
        {
            Store<align>(dst + offset, VectorProduct(vertical, Load<align>(horizontal + offset)));
        }

        template <bool align> void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(horizontal) && Aligned(dst) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            for (size_t row = 0; row < height; ++row)
            {
                v128_u16 _vertical = SetU16(vertical[row]);
                if (align)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += QA)
                    {
                        VectorProduct<true>(_vertical, horizontal, col, dst);
                        VectorProduct<true>(_vertical, horizontal, col + A, dst);
                        VectorProduct<true>(_vertical, horizontal, col + 2 * A, dst);
                        VectorProduct<true>(_vertical, horizontal, col + 3 * A, dst);
                    }
                    for (; col < alignedWidth; col += A)
                        VectorProduct<true>(_vertical, horizontal, col, dst);
                }
                else
                {
                    Storer<align> _dst(dst);
                    _dst.First(VectorProduct(_vertical, Load<align>(horizontal)));
                    for (size_t col = A; col < alignedWidth; col += A)
                        _dst.Next(VectorProduct(_vertical, Load<align>(horizontal + col)));
                    Flush(_dst);
                }
                if (alignedWidth != width)
                    VectorProduct<false>(_vertical, horizontal, width - A, dst);
                dst += stride;
            }
        }

        void VectorProduct(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height)
        {
            if (Aligned(horizontal) && Aligned(dst) && Aligned(stride))
                VectorProduct<true>(vertical, horizontal, dst, stride, width, height);
            else
                VectorProduct<false>(vertical, horizontal, dst, stride, width, height);
        }
    }
#endif// SIMD_VMX_ENABLE
}
